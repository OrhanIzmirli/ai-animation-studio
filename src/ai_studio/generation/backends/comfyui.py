"""Talking to a local ComfyUI: configuration, HTTP client, server lifecycle.

Shared plumbing for adapters that run a model through ComfyUI. Nothing here
knows which model or workflow is used. ComfyUI is a separate program with its
own Python environment: adapters only speak HTTP to it, so it adds no
dependency to this project.

Configuration is a small local YAML file, never a path in the source tree:

    configs/comfyui.local.yaml        (git-ignored; copy comfyui.example.yaml)
    or the file named by $AI_STUDIO_COMFYUI_CONFIG

Standard library + PyYAML only.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ai_studio.generation.backends.base import BackendNotInstalledError
from ai_studio.paths import resolve_path

CONFIG_ENV_VAR = "AI_STUDIO_COMFYUI_CONFIG"
DEFAULT_CONFIG_PATH = "configs/comfyui.local.yaml"
EXAMPLE_CONFIG_PATH = "configs/comfyui.example.yaml"
HOST = "127.0.0.1"  # the server is never exposed beyond this machine
# `models` keys. text_encoder and vae are required; each backend needs its own
# GGUF: "unet" (Turbo) or "unet_base" (base model).
REQUIRED_MODEL_KINDS = ("text_encoder", "vae")
UNET_KINDS = ("unet", "unet_base")
MODEL_KINDS = UNET_KINDS + REQUIRED_MODEL_KINDS
# Where ComfyUI looks for each kind of model file, relative to <comfyui_dir>/models.
MODEL_FOLDERS = {
    "unet": ("diffusion_models", "unet"),
    "unet_base": ("diffusion_models", "unet"),
    "text_encoder": ("text_encoders", "clip"),
    "vae": ("vae",),
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, kw_only=True)
class ComfyUIConfig:
    comfyui_dir: Path
    models: dict[str, str]  # kind -> file name, see MODEL_KINDS
    python: Path | None = None  # None: <comfyui_dir>/venv python
    port: int = 8188
    startup_timeout_seconds: float = 300.0
    generation_timeout_seconds: float = 3600.0
    # Optional pins: kind -> expected sha256. A file that does not match is refused.
    sha256: dict[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()  # extra arguments for ComfyUI's main.py

    def __post_init__(self) -> None:
        if (
            not isinstance(self.models, dict)
            or not set(REQUIRED_MODEL_KINDS) <= set(self.models) <= set(MODEL_KINDS)
            or not set(self.models) & set(UNET_KINDS)
        ):
            raise ValueError(
                f"models must name {REQUIRED_MODEL_KINDS} and at least one of {UNET_KINDS}"
            )
        for kind, name in self.models.items():
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError(f"models.{kind} must be a file name without a directory, got {name!r}")
        if not isinstance(self.port, int) or isinstance(self.port, bool) or not 0 < self.port < 65536:
            raise ValueError("port must be an int between 1 and 65535")
        for name in ("startup_timeout_seconds", "generation_timeout_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive number")
        if not isinstance(self.sha256, dict) or not set(self.sha256) <= set(MODEL_KINDS):
            raise ValueError(f"sha256 keys must be among {MODEL_KINDS}")
        for kind, digest in self.sha256.items():
            if not isinstance(digest, str) or not _SHA256_RE.match(digest):
                raise ValueError(f"sha256.{kind} must be a lowercase hex SHA-256")

    @property
    def python_executable(self) -> Path:
        if self.python is not None:
            return self.python
        venv = self.comfyui_dir / "venv"
        return venv / "Scripts" / "python.exe" if sys.platform == "win32" else venv / "bin" / "python"

    def model_path(self, kind: str) -> Path | None:
        """The model file on disk, or None if it is in none of ComfyUI's folders."""
        for folder in MODEL_FOLDERS[kind]:
            path = self.comfyui_dir / "models" / folder / self.models[kind]
            if path.is_file():
                return path
        return None


def config_path(root: str | Path | None = None) -> Path:
    override = os.environ.get(CONFIG_ENV_VAR)
    return Path(override).expanduser().resolve() if override else resolve_path(DEFAULT_CONFIG_PATH, root)


def load_config(path: str | Path | None = None) -> ComfyUIConfig:
    """Read the local ComfyUI configuration. A missing or invalid file is a
    setup problem: BackendNotInstalledError."""
    path = Path(path) if path is not None else config_path()
    if not path.is_file():
        raise BackendNotInstalledError(
            f"ComfyUI is not configured: {path} not found. Copy {EXAMPLE_CONFIG_PATH} to "
            f"{DEFAULT_CONFIG_PATH} and set comfyui_dir (or point ${CONFIG_ENV_VAR} at a config file)."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("expected a mapping at the top level")
        unknown = set(data) - {
            "comfyui_dir", "python", "port", "startup_timeout_seconds", "generation_timeout_seconds",
            "models", "sha256", "extra_args",
        }
        if unknown:
            raise ValueError(f"unknown keys {sorted(unknown)}")
        if not data.get("comfyui_dir"):
            raise ValueError("comfyui_dir is required")
        fields: dict[str, Any] = {
            key: data[key] for key in ("port", "startup_timeout_seconds", "generation_timeout_seconds")
            if data.get(key) is not None
        }
        return ComfyUIConfig(
            comfyui_dir=Path(str(data["comfyui_dir"])).expanduser(),
            python=Path(str(data["python"])).expanduser() if data.get("python") else None,
            models=data.get("models") or {},
            sha256={k: str(v) for k, v in (data.get("sha256") or {}).items() if v},
            extra_args=tuple(str(arg) for arg in data.get("extra_args") or ()),
            **fields,
        )
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise BackendNotInstalledError(f"invalid ComfyUI configuration {path}: {exc}") from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(directory: Path) -> str | None:
    """HEAD commit of a git checkout, or None (not a checkout, no git)."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", commit) else None


class ComfyUIError(RuntimeError):
    """ComfyUI answered with an error, or a prompt failed."""


class ComfyUIClient:
    """The few HTTP calls an adapter needs."""

    def __init__(self, port: int, timeout_seconds: float = 120.0) -> None:
        self.base_url = f"http://{HOST}:{port}"
        self.timeout_seconds = timeout_seconds

    def _open(self, request: urllib.request.Request | str) -> bytes:
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read()

    def get_json(self, path: str) -> Any:
        return json.loads(self._open(self.base_url + path))

    def is_up(self) -> bool:
        try:
            self.get_json("/system_stats")
            return True
        except (OSError, ValueError):  # URLError, connection refused, timeouts, bad JSON
            return False

    def system_stats(self) -> dict[str, Any]:
        return self.get_json("/system_stats")

    def node_info(self, node_class: str) -> dict[str, Any] | None:
        """/object_info entry for one node class; None if this ComfyUI does not have it."""
        return self.get_json(f"/object_info/{urllib.parse.quote(node_class)}").get(node_class)

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            self.base_url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            body = self._open(request)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:2000]
            raise ComfyUIError(f"POST {path} failed with HTTP {exc.code}: {detail}") from exc
        return json.loads(body) if body else None

    def upload_image(self, path: Path, name: str) -> str:
        """Upload a PNG into ComfyUI's input folder; returns the name to use in LoadImage."""
        boundary = uuid.uuid4().hex
        body = b"".join([
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            "Content-Type: image/png\r\n\r\n".encode(), path.read_bytes(), b"\r\n",
            f'--{boundary}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode(),
            f"--{boundary}--\r\n".encode(),
        ])
        request = urllib.request.Request(
            self.base_url + "/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            return json.loads(self._open(request))["name"]
        except urllib.error.HTTPError as exc:
            raise ComfyUIError(f"uploading {name} failed with HTTP {exc.code}") from exc

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        return self.post_json("/prompt", {"prompt": workflow, "client_id": "ai_studio"})["prompt_id"]

    def interrupt(self) -> None:
        try:
            self.post_json("/interrupt", {})
        except (OSError, ComfyUIError, ValueError):
            pass  # best effort: the caller is already reporting a timeout

    def wait_for_prompt(self, prompt_id: str, timeout_seconds: float, poll_seconds: float = 1.0) -> dict[str, Any]:
        """Poll /history until the prompt has finished; returns its outputs.
        TimeoutError after timeout_seconds (the prompt is then interrupted);
        ComfyUIError if ComfyUI reports that the prompt failed."""
        deadline = time.monotonic() + timeout_seconds
        while True:
            entry = self.get_json(f"/history/{prompt_id}").get(prompt_id)
            status = (entry or {}).get("status") or {}
            if entry is not None and status.get("completed") is not None:
                if status.get("status_str") != "success":
                    errors = [m[1] for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    brief = "; ".join(
                        f"{e.get('node_type')}: {e.get('exception_type')}: {e.get('exception_message')}"
                        for e in errors
                    ) or json.dumps(status)[:1000]
                    raise ComfyUIError(f"ComfyUI prompt failed: {brief.strip()}")
                return entry.get("outputs") or {}
            if time.monotonic() >= deadline:
                self.interrupt()
                raise TimeoutError(f"ComfyUI did not finish the prompt within {timeout_seconds:g} s")
            time.sleep(poll_seconds)

    def view(self, image: dict[str, Any]) -> bytes:
        """Bytes of one output image, as listed in a prompt's outputs."""
        query = urllib.parse.urlencode({key: image.get(key, "") for key in ("filename", "subfolder", "type")})
        return self._open(self.base_url + "/view?" + query)


def _terminate_tree(process: subprocess.Popen) -> None:
    """Stop a process and its children (a venv python on Windows is a launcher
    whose child is the real interpreter)."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=60, check=False,
        )
    else:
        import signal

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except OSError:
            process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


class ComfyUIServer:
    """A ComfyUI on HOST:port. `ensure_running()` reuses one that already
    answers there, otherwise starts one headless and waits for it. `close()`
    stops the server only if this object started it."""

    def __init__(self, config: ComfyUIConfig) -> None:
        self.config = config
        self.client = ComfyUIClient(config.port)
        self.process: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self.started_by_us = False

    def ensure_running(self) -> None:
        if self.client.is_up():
            return
        config = self.config
        main = config.comfyui_dir / "main.py"
        if not main.is_file():
            raise BackendNotInstalledError(f"ComfyUI not found: {main} does not exist")
        python = config.python_executable
        if not python.is_file():
            raise BackendNotInstalledError(f"ComfyUI's python not found: {python}")

        log = tempfile.NamedTemporaryFile(prefix="ai_studio_comfyui_", suffix=".log", delete=False)
        self.log_path = Path(log.name)
        command = [
            str(python), "-u", "main.py", "--listen", HOST, "--port", str(config.port),
            "--disable-auto-launch", *config.extra_args,
        ]
        self.process = subprocess.Popen(
            command, cwd=config.comfyui_dir, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=sys.platform != "win32",
        )
        log.close()
        self.started_by_us = True
        atexit.register(self.close)  # safety net if the caller forgets close()

        deadline = time.monotonic() + config.startup_timeout_seconds
        while not self.client.is_up():
            if self.process.poll() is not None:
                code, tail = self.process.returncode, self.log_tail()
                self.close()
                raise BackendNotInstalledError(f"ComfyUI exited with code {code} while starting:\n{tail}")
            if time.monotonic() >= deadline:
                self.close()
                raise BackendNotInstalledError(
                    f"ComfyUI did not answer on {self.client.base_url} within "
                    f"{config.startup_timeout_seconds:g} s"
                )
            time.sleep(0.5)

    def log_tail(self, lines: int = 20) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        text = self.log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(re.split(r"[\r\n]+", text.strip())[-lines:])

    def close(self) -> None:
        if self.process is not None:
            _terminate_tree(self.process)
            self.process = None
        self.started_by_us = False
