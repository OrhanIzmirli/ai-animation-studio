"""THROWAWAY SPIKE (Task 6a) - not production code, not a backend adapter.

Can this laptop (RTX 4060 Laptop 8 GB, 16 GB RAM) generate the
comic_rooftop_01 shot locally with Wan 2.2 TI2V-5B GGUF in ComfyUI?

    python spikes/comfyui_wan_spike.py --variant turbo
    python spikes/comfyui_wan_spike.py --variant base

Starts ComfyUI headless on 127.0.0.1 and drives it only through its HTTP API:
GET /system_stats, GET /object_info, POST /upload/image, POST /prompt,
GET /history/<id>, GET /view. The workflow is built in API format and checked
against /object_info before it is queued, so a node or input that does not
exist in the installed ComfyUI fails here, not inside the server.

Two prompts are queued so that the stages can be timed separately:
    A  loaders -> text encode -> start-frame latent -> KSampler -> SaveLatent
    B  the same graph + VAEDecode -> SaveImage   (A's nodes come from the cache)
Inside A, sampling time is taken from the sampler's progress lines in the
server log; the rest of A is counted as load.

Results go to spikes/output/<name>/ (git-ignored): frames/, out.mp4,
comfyui.log, result.json. See spikes/LOCAL_COMFYUI_SPIKE.md.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ai_studio.benchmarking import load_definition  # noqa: E402
from ai_studio.generation.backends.wan_ti2v import encode_frames, prepare_start_frame  # noqa: E402
from ai_studio.quality.store import measure_video  # noqa: E402

DEFAULT_COMFY = Path(r"C:\Users\rhn23\ai-tools\ComfyUI")
DEFINITION = REPO / "benchmarks/definitions/comic_rooftop_01.yaml"
SMOKE_METRICS = REPO / "benchmarks/metrics/comic_rooftop_01/20261004T145422Z-a264ad6e/metrics_v1.json"
VARIANTS = {
    # steps / cfg from the task; sampler, scheduler and shift: see the report
    "turbo": {"unet": "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf", "steps": 4, "cfg": 1.0},
    "base": {"unet": "Wan2.2-TI2V-5B-Q8_0.gguf", "steps": 20, "cfg": 5.0},
}
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan2.2_vae.safetensors"
PROGRESS_RE = re.compile(r"(\d+)/(\d+) \[")


# --- HTTP helpers (standard library only) ---

class Api:
    def __init__(self, port: int) -> None:
        self.base = f"http://127.0.0.1:{port}"

    def get(self, path: str, raw: bool = False):
        with urllib.request.urlopen(self.base + path, timeout=120) as response:
            data = response.read()
        return data if raw else json.loads(data)

    def post_json(self, path: str, payload: dict):
        request = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"POST {path} -> {exc.code}: {exc.read().decode(errors='replace')}") from exc

    def upload_image(self, path: Path, name: str) -> str:
        boundary = uuid.uuid4().hex
        body = b"".join([
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n".encode(), path.read_bytes(), b"\r\n",
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
            f"--{boundary}--\r\n".encode(),
        ])
        request = urllib.request.Request(
            self.base + "/upload/image", data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())["name"]


# --- server process, log capture, resource monitor ---

class Server:
    def __init__(self, comfy: Path, port: int, log_path: Path, extra_args: list[str]) -> None:
        python = comfy / "venv" / "Scripts" / "python.exe"
        self.command = [str(python), "-u", "main.py", "--listen", "127.0.0.1", "--port", str(port), *extra_args]
        self.lines: list[tuple[float, str]] = []  # (time, text), split on \r and \n
        self.log_path = log_path
        self.process = subprocess.Popen(
            self.command, cwd=comfy, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        buffer = b""
        with open(self.log_path, "w", encoding="utf-8") as log:
            while chunk := self.process.stdout.read1(4096):
                buffer += chunk
                parts = re.split(rb"[\r\n]+", buffer)
                buffer = parts.pop()
                for part in parts:
                    text = part.decode(errors="replace").strip()
                    if text:
                        now = time.time()
                        self.lines.append((now, text))
                        log.write(f"{now:.2f} {text}\n")
                        log.flush()

    def stop(self) -> None:
        for child in psutil.Process(self.process.pid).children(recursive=True):
            child.kill()
        self.process.kill()
        self.process.wait(timeout=30)


class Monitor:
    """Samples GPU memory (nvidia-smi), system RAM, swap and the server
    process once per second and keeps the peaks per named stage."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stage = "idle"
        self.peaks: dict[str, dict[str, float]] = {}
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def gpu_used_mib() -> float | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            return float(out.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None

    def sample(self) -> dict:
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        sample = {
            "t": time.time(), "stage": self.stage, "vram_used_mib": self.gpu_used_mib(),
            "ram_used_gb": (memory.total - memory.available) / 2**30,
            "ram_available_gb": memory.available / 2**30,
            "swap_used_gb": swap.used / 2**30,
        }
        try:
            # venv python.exe is a launcher: the real interpreter is its child.
            root = psutil.Process(self.pid)
            infos = [p.memory_info() for p in [root, *root.children(recursive=True)]]
            sample["server_rss_gb"] = sum(i.rss for i in infos) / 2**30
            sample["server_private_gb"] = sum(getattr(i, "private", i.vms) for i in infos) / 2**30
        except psutil.Error:
            pass
        return sample

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self.sample()
            self.samples.append(sample)
            peak = self.peaks.setdefault(sample["stage"], {})
            for key, value in sample.items():
                if key in ("t", "stage") or value is None:
                    continue
                if key == "ram_available_gb":
                    peak["ram_available_min_gb"] = min(peak.get("ram_available_min_gb", value), value)
                else:
                    peak[key + "_peak"] = max(peak.get(key + "_peak", value), value)
            self._stop.wait(1.0)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


# --- workflow ---

def build_workflow(args, image_name: str, prompt: str, negative: str, prefix: str) -> tuple[dict, dict]:
    """(stage A, stage B) in ComfyUI API format. Node class names and input
    names are checked against /object_info by check_workflow()."""
    common = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": args.unet}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TEXT_ENCODER, "type": "wan", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"shift": args.shift, "model": ["1", 0]}},
        "8": {"class_type": "Wan22ImageToVideoLatent", "inputs": {
            "vae": ["3", 0], "width": args.width, "height": args.height, "length": args.frames,
            "batch_size": 1, "start_image": ["4", 0],
        }},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "seed": args.seed, "steps": args.steps, "cfg": args.cfg,
            "sampler_name": args.sampler, "scheduler": args.scheduler,
            "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["8", 0], "denoise": 1.0,
        }},
    }
    stage_a = {**common, "10": {"class_type": "SaveLatent", "inputs": {
        "samples": ["9", 0], "filename_prefix": f"{prefix}/latent",
    }}}
    decode = (
        {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["9", 0], "vae": ["3", 0], "tile_size": args.tile_size, "overlap": 64,
            "temporal_size": args.temporal_size, "temporal_overlap": 8,
        }} if args.tiled_decode else
        {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}}
    )
    stage_b = {**common, "11": decode, "12": {"class_type": "SaveImage", "inputs": {
        "images": ["11", 0], "filename_prefix": f"{prefix}/frame",
    }}}
    return stage_a, stage_b


def check_workflow(api: Api, workflow: dict) -> dict:
    """Verify every node class and input against the running server. Returns
    {class: python module} for the report."""
    seen = {}
    for node_id, node in workflow.items():
        cls = node["class_type"]
        try:
            info = api.get(f"/object_info/{urllib.parse.quote(cls)}")[cls]
        except (KeyError, urllib.error.HTTPError):
            raise RuntimeError(f"node class {cls!r} does not exist in this ComfyUI") from None
        required = info["input"].get("required", {})
        optional = info["input"].get("optional", {})
        given = set(node["inputs"])
        missing, unknown = set(required) - given, given - set(required) - set(optional)
        if missing or unknown:
            raise RuntimeError(f"node {node_id} ({cls}): missing inputs {sorted(missing)}, unknown {sorted(unknown)}")
        for name, value in node["inputs"].items():
            spec = {**required, **optional}[name]
            choices = spec[0] if isinstance(spec[0], list) else (spec[1] or {}).get("options") if len(spec) > 1 else None
            if isinstance(choices, list) and not isinstance(value, list) and value not in choices:
                raise RuntimeError(f"node {node_id} ({cls}).{name}: {value!r} not in {choices[:12]}")
        seen[cls] = info.get("python_module")
    return seen


def run_prompt(api: Api, workflow: dict, server: Server, timeout: float) -> dict:
    started = time.time()
    prompt_id = api.post_json("/prompt", {"prompt": workflow, "client_id": "spike"})["prompt_id"]
    while True:
        if server.process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited with code {server.process.returncode} during the prompt")
        history = api.get(f"/history/{prompt_id}")
        if prompt_id in history and history[prompt_id].get("status", {}).get("completed") is not None:
            entry = history[prompt_id]
            status = entry["status"]
            if status.get("status_str") != "success":
                errors = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
                brief = [{k: e.get(k) for k in ("node_type", "exception_type", "exception_message")} for e in errors]
                raise RuntimeError(f"prompt failed: {json.dumps(brief)[:2000]}")
            return {"prompt_id": prompt_id, "started": started, "finished": time.time(), "outputs": entry["outputs"]}
        if time.time() - started > timeout:
            raise RuntimeError(f"prompt did not finish within {timeout:.0f} s")
        time.sleep(1.0)


def sampling_window(server: Server, start: float, end: float, steps: int) -> tuple[float, float] | None:
    """(first, last) timestamps of the sampler's progress lines 0/steps .. steps/steps."""
    first = last = None
    for stamp, text in server.lines:
        if not start <= stamp <= end:
            continue
        match = PROGRESS_RE.search(text)
        if match and int(match.group(2)) == steps:
            if int(match.group(1)) == 0 and first is None:
                first = stamp
            if int(match.group(1)) == steps:
                last = stamp
    return (first, last) if first is not None and last is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--comfy", type=Path, default=DEFAULT_COMFY)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg", type=float)
    parser.add_argument("--sampler", default="uni_pc")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--tiled-decode", action="store_true", help="VAEDecodeTiled instead of VAEDecode")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--temporal-size", type=int, default=64, help="frames per temporal tile (VAEDecodeTiled)")
    parser.add_argument("--port", type=int, default=8199)
    parser.add_argument("--name", default=None, help="output folder name under spikes/output/")
    parser.add_argument("--timeout", type=float, default=3 * 3600)
    parser.add_argument("--comfy-arg", action="append", default=[], help="extra argument for ComfyUI main.py")
    args = parser.parse_args()
    variant = VARIANTS[args.variant]
    args.unet = variant["unet"]
    args.steps = args.steps or variant["steps"]
    args.cfg = variant["cfg"] if args.cfg is None else args.cfg

    name = args.name or f"{args.variant}_{args.width}x{args.height}x{args.frames}_s{args.steps}"
    out_dir = REPO / "spikes" / "output" / name
    if out_dir.exists():
        print(f"{out_dir} already exists; choose another --name")
        return 2
    (out_dir / "frames").mkdir(parents=True)

    definition = load_definition(DEFINITION)
    work = Path(tempfile.mkdtemp(prefix="wan_spike_"))
    # Same preprocessing as the Wan adapter; the original frame is only read.
    _, conditioning = prepare_start_frame(REPO / definition.start_frame, work, args.width, args.height)
    derived = Path(conditioning.derived_file)
    print("start frame:", conditioning.preprocessing, "->", derived)

    result: dict = {
        "name": name, "variant": args.variant, "settings": {
            key: getattr(args, key) for key in
            ("unet", "width", "height", "frames", "seed", "steps", "cfg", "sampler", "scheduler", "shift", "tiled_decode", "tile_size", "temporal_size")
        },
        "text_encoder": TEXT_ENCODER, "vae": VAE, "start_frame_preprocessing": conditioning.preprocessing,
        "status": "failed",
    }
    api = Api(args.port)
    baseline = Monitor.gpu_used_mib()
    memory = psutil.virtual_memory()
    result["baseline"] = {
        "vram_used_mib": baseline, "ram_used_gb": (memory.total - memory.available) / 2**30,
        "swap_used_gb": psutil.swap_memory().used / 2**30, "on_ac_power": getattr(psutil.sensors_battery(), "power_plugged", None),
    }
    server = Server(args.comfy, args.port, out_dir / "comfyui.log", args.comfy_arg)
    monitor = Monitor(server.process.pid)
    try:
        boot = time.time()
        while True:
            try:
                result["system_stats"] = api.get("/system_stats")
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if server.process.poll() is not None:
                    raise RuntimeError("ComfyUI did not start; see comfyui.log") from None
                if time.time() - boot > 600:
                    raise RuntimeError("ComfyUI did not answer within 600 s") from None
                time.sleep(1.0)
        result["server_start_seconds"] = round(time.time() - boot, 1)
        print(f"ComfyUI up after {result['server_start_seconds']} s")

        image_name = api.upload_image(derived, f"{name}_start_frame.png")
        stage_a, stage_b = build_workflow(args, image_name, definition.prompt, definition.negative_prompt, f"spike_{name}")
        result["node_modules"] = check_workflow(api, stage_b) | check_workflow(api, stage_a)
        (out_dir / "workflow_stage_a.json").write_text(json.dumps(stage_a, indent=2), encoding="utf-8")
        (out_dir / "workflow_stage_b.json").write_text(json.dumps(stage_b, indent=2), encoding="utf-8")

        monitor.stage = "A_load_and_sampling"
        a = run_prompt(api, stage_a, server, args.timeout)
        monitor.stage = "B_decode_and_save"
        b = run_prompt(api, stage_b, server, args.timeout)
        monitor.stage = "after"

        window = sampling_window(server, a["started"], a["finished"], args.steps)
        a_total = a["finished"] - a["started"]
        sampling = (window[1] - window[0]) if window else None
        result["timing_seconds"] = {
            "stage_a_total": round(a_total, 1),
            "sampling": None if sampling is None else round(sampling, 1),
            "load_and_encode": None if sampling is None else round(a_total - sampling, 1),
            "seconds_per_step": None if sampling is None else round(sampling / args.steps, 1),
            "stage_b_decode_and_save_png": round(b["finished"] - b["started"], 1),
            "total": round(b["finished"] - a["started"], 1),
        }

        images = [image for output in b["outputs"].values() for image in output.get("images", [])]
        frame_paths = []
        for index, image in enumerate(images):
            query = urllib.parse.urlencode({k: image[k] for k in ("filename", "subfolder", "type")})
            path = out_dir / "frames" / f"frame_{index:04d}.png"
            path.write_bytes(api.get("/view?" + query, raw=True))
            frame_paths.append(path)
        result["frames_fetched"] = len(frame_paths)
        result["status"] = "success"
    except Exception as exc:  # the spike records failures instead of hiding them
        result["error"] = f"{type(exc).__name__}: {exc}"
        print("FAILED:", result["error"])
    finally:
        monitor.stop()
        server.stop()
        log_text = "\n".join(text for _, text in server.lines)
        result["peaks"] = monitor.peaks
        result["log_flags"] = {
            "out_of_memory": bool(re.search(r"out of memory|OutOfMemory|OOM", log_text, re.I)),
            "tiled_vae_fallback": "tiled" in log_text.lower() and "retrying" in log_text.lower(),
            "lowvram_or_partial_load": sorted(set(re.findall(r"loaded (?:partially|completely)[^\n]*", log_text)))[:6],
        }
        (out_dir / "samples.json").write_text(json.dumps(monitor.samples), encoding="utf-8")

    if result["status"] == "success":
        import numpy as np
        from PIL import Image

        class LazyFrames:  # encode_frames only needs len() and indexing/iteration
            def __len__(self):
                return len(frame_paths)

            def __getitem__(self, index):
                with Image.open(frame_paths[index]) as image:
                    return np.asarray(image.convert("RGB"))

            def __iter__(self):
                return (self[i] for i in range(len(self)))

        video = out_dir / "out.mp4"
        result["encoder"] = encode_frames(LazyFrames(), video)  # the Wan adapter's encoder settings
        record = measure_video(video, benchmark_id="spike", run_id=name, reference_image=derived)
        result["metrics"] = record.metrics
        result["video_sha256"] = record.video_sha256
        smoke = json.loads(SMOKE_METRICS.read_text(encoding="utf-8"))["metrics"]
        print(f"\n{'metric':28} {'this run':>12} {'L4 smoke run':>14}")
        for key, value in record.metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key:28} {value:12.3f} {smoke.get(key, float('nan')):14.3f}")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result.get(k) for k in ("status", "error", "timing_seconds", "peaks", "log_flags")}, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
