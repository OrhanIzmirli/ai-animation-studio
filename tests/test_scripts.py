"""The helper scripts in scripts/: bundling, collecting and importing runs,
and the Camber launchers. No torch, conda or GPU needed."""

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_bundle = load_script("make_bundle")
collect_run = load_script("collect_run")
import_run = load_script("import_run")
camber_run = load_script("camber_run")
camber_setup = load_script("camber_setup")


@pytest.mark.parametrize(
    "name", ["make_bundle", "collect_run", "import_run", "camber_run", "camber_setup"],
)
def test_helper_scripts_use_only_the_standard_library(name):
    # They run with a base python that has neither the project nor its dependencies.
    tree = ast.parse((SCRIPTS / f"{name}.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= sys.stdlib_module_names, imported - sys.stdlib_module_names


# --- make_bundle ---

FRAME = "benchmarks/references/hero/frame.png"
FRAME_BYTES = b"\x89PNG fake canonical frame"
FRAME_SHA = hashlib.sha256(FRAME_BYTES).hexdigest()


def git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "core.autocrlf=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    """A small git repository shaped like the project."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    root = tmp_path / "repo"
    files = {
        ".gitignore": ".venv/\noutputs/*\n!outputs/.gitkeep\ndist/\n__pycache__/\nbenchmarks/references/**\n",
        "pyproject.toml": "[project]\nname = 'x'\n",
        "src/pkg/module.py": "VALUE = 1\n",
        "outputs/.gitkeep": "",
        "benchmarks/results/b/run-0.json": "{}\n",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8", newline="\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    # git-ignored files that exist in a real working copy
    for name, data in {
        FRAME: FRAME_BYTES,
        ".venv/lib/site.py": b"venv",
        "outputs/b/run-0/out.mp4": b"video",
        "dist/old.zip": b"old",
        "src/pkg/__pycache__/module.cpython-311.pyc": b"cache",
    }.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    return root


def bundle(repo, **kwargs):
    return make_bundle.make_bundle(repo, start_frame=FRAME, start_frame_sha256=FRAME_SHA, **kwargs)


def test_bundle_contains_tracked_files_git_and_the_start_frame(repo):
    path = bundle(repo)
    assert path == repo / "dist" / "ai-animation-studio-bundle.zip"
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert archive.read(f"ai-animation-studio/{FRAME}") == FRAME_BYTES
        assert archive.read("ai-animation-studio/src/pkg/module.py") == b"VALUE = 1\n"

    assert all(name.startswith("ai-animation-studio/") for name in names)
    relative = {name.removeprefix("ai-animation-studio/") for name in names}
    assert {".gitignore", "pyproject.toml", "src/pkg/module.py", "outputs/.gitkeep",
            "benchmarks/results/b/run-0.json", FRAME} <= relative
    assert ".git/HEAD" in relative and any(name.startswith(".git/objects/") for name in relative)
    # ignored / generated content stays out
    assert not any(name.startswith((".venv/", "dist/")) or "__pycache__" in name for name in relative)
    assert [name for name in relative if name.startswith("outputs/")] == ["outputs/.gitkeep"]


def test_unpacked_bundle_is_a_clean_checkout_of_the_same_commit(repo, tmp_path):
    path = bundle(repo)
    target = tmp_path / "unpacked"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    unpacked = target / "ai-animation-studio"

    def output(directory, *args):
        return subprocess.run(["git", *args], cwd=directory, capture_output=True, text=True, check=True).stdout.strip()

    assert output(unpacked, "rev-parse", "HEAD") == output(repo, "rev-parse", "HEAD")
    assert output(unpacked, "status", "--porcelain") == ""


def test_dirty_tree_is_refused_unless_allowed(repo):
    (repo / "src/pkg/module.py").write_text("VALUE = 2\n", encoding="utf-8", newline="\n")
    with pytest.raises(make_bundle.BundleError, match="uncommitted"):
        bundle(repo)
    assert not (repo / "dist" / "ai-animation-studio-bundle.zip").exists()

    with zipfile.ZipFile(bundle(repo, allow_dirty=True)) as archive:
        assert archive.read("ai-animation-studio/src/pkg/module.py") == b"VALUE = 2\n"


def test_untracked_file_counts_as_dirty_and_is_never_packed(repo):
    (repo / "notes.txt").write_text("scratch", encoding="utf-8")
    with pytest.raises(make_bundle.BundleError, match="notes.txt"):
        bundle(repo)
    with zipfile.ZipFile(bundle(repo, allow_dirty=True)) as archive:
        assert "ai-animation-studio/notes.txt" not in archive.namelist()


def test_wrong_or_missing_start_frame_is_refused(repo):
    (repo / FRAME).write_bytes(b"a different image")
    with pytest.raises(make_bundle.BundleError, match="not the canonical file"):
        bundle(repo)
    (repo / FRAME).unlink()
    with pytest.raises(make_bundle.BundleError, match="start frame not found"):
        bundle(repo)


def test_bundle_pins_the_canonical_start_frame():
    assert make_bundle.START_FRAME == "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
    assert make_bundle.START_FRAME_SHA256 == "3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb"


# --- collect_run / import_run ---

def write_run(root, benchmark, run_id, timestamp="2026-10-04T10:00:00+00:00", with_output=True):
    record = root / "benchmarks" / "results" / benchmark / f"{run_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"run_id": run_id, "timestamp": timestamp}), encoding="utf-8")
    if with_output:
        output_dir = root / "outputs" / benchmark / run_id
        (output_dir / "frames").mkdir(parents=True)
        (output_dir / "out.mp4").write_bytes(b"video " + run_id.encode())
        (output_dir / "start_frame_1280x704.png").write_bytes(b"png")
        (output_dir / "frames" / "0001.png").write_bytes(b"frame")


def tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file() and "dist" not in path.relative_to(root).parts
    }


def test_collect_then_import_round_trip(tmp_path):
    cloud, local = tmp_path / "cloud", tmp_path / "local"
    local.mkdir()
    write_run(cloud, "bench", "run-1")

    archive = collect_run.collect_run(cloud, "bench", "run-1")
    assert archive == cloud / "dist" / "run-bench-run-1.zip"
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == [
            "benchmarks/results/bench/run-1.json",
            "outputs/bench/run-1/frames/0001.png",
            "outputs/bench/run-1/out.mp4",
            "outputs/bench/run-1/start_frame_1280x704.png",
        ]

    written = import_run.import_run(archive, local)
    assert sorted(written) == sorted(tree(cloud))
    assert tree(local) == tree(cloud)


def test_collect_defaults_to_the_newest_run(tmp_path):
    write_run(tmp_path, "bench", "zzz-old", timestamp="2026-10-01T10:00:00+00:00")
    write_run(tmp_path, "bench", "aaa-new", timestamp="2026-10-04T10:00:00+00:00")
    write_run(tmp_path, "other", "zzz-other", timestamp="2026-10-05T10:00:00+00:00")
    assert collect_run.collect_run(tmp_path, "bench").name == "run-bench-aaa-new.zip"


def test_collect_failed_run_without_output_directory(tmp_path):
    write_run(tmp_path, "bench", "failed", with_output=False)
    with zipfile.ZipFile(collect_run.collect_run(tmp_path, "bench", "failed")) as z:
        assert z.namelist() == ["benchmarks/results/bench/failed.json"]


@pytest.mark.parametrize(
    "benchmark, run_id, match",
    [("bench", "missing", "no record"), ("nothing", None, "no runs recorded"),
     ("../bench", "run-1", "not a valid id"), ("bench", "../run-1", "not a valid id")],
)
def test_collect_errors(tmp_path, benchmark, run_id, match):
    write_run(tmp_path, "bench", "run-1")
    with pytest.raises(collect_run.CollectError, match=match):
        collect_run.collect_run(tmp_path, benchmark, run_id)


def test_import_refuses_to_overwrite_and_writes_nothing(tmp_path):
    cloud, local = tmp_path / "cloud", tmp_path / "local"
    write_run(cloud, "bench", "run-1")
    archive = collect_run.collect_run(cloud, "bench", "run-1")
    existing = local / "outputs" / "bench" / "run-1" / "out.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already here")

    with pytest.raises(import_run.ImportRunError, match="refusing to overwrite") as exc_info:
        import_run.import_run(archive, local)
    assert "outputs/bench/run-1/out.mp4" in str(exc_info.value)
    assert tree(local) == {"outputs/bench/run-1/out.mp4": b"already here"}  # untouched, nothing added

    import_run.import_run(archive, tmp_path / "fresh")  # the same zip imports fine elsewhere
    with pytest.raises(import_run.ImportRunError, match="refusing to overwrite"):
        import_run.import_run(archive, tmp_path / "fresh")  # but never twice


@pytest.mark.parametrize(
    "name",
    [
        "../evil.txt",
        "outputs/../../evil.txt",
        "outputs/bench/../../../evil.txt",
        "/etc/evil.txt",
        "C:/evil.txt",
        "outputs\\..\\..\\evil.txt",
        "outputs/./x.txt",
        "src/ai_studio/cli.py",       # inside the project, but not a run file
        "benchmarks/definitions/x.yaml",
    ],
)
def test_import_rejects_escaping_and_unexpected_paths(tmp_path, name):
    local = tmp_path / "local"
    local.mkdir()
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("benchmarks/results/bench/run-1.json", "{}")
        z.writestr(zipfile.ZipInfo(name), "evil")

    with pytest.raises(import_run.ImportRunError, match="unsafe path|unexpected path|escapes"):
        import_run.import_run(archive, local)
    assert tree(tmp_path) == {"evil.zip": archive.read_bytes()}  # nothing written anywhere


def test_import_rejects_non_zip(tmp_path):
    bad = tmp_path / "not.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(import_run.ImportRunError, match="cannot open"):
        import_run.import_run(bad, tmp_path)


# --- camber_run ---

def run_args(*argv):
    return camber_run.build_parser().parse_args(["--seed", "42", *argv])


def test_camber_run_defaults():
    command = camber_run.build_command(run_args(), "RUN", Path("/home/jovyan/ai-animation-studio"))
    assert command[0] == str(camber_run.ENV_PYTHON)
    assert Path(command[2]).name == "run_benchmark.py"
    assert command[3:] == [
        "--benchmark", "comic_rooftop_01", "--backend", "wan-ti2v-5b", "--seed", "42",
        "--width", "1280", "--height", "704", "--run-id", "RUN", "--param", "cpu_offload=true",
    ]


def test_camber_run_extra_params_and_cpu_offload_override():
    command = camber_run.build_command(
        run_args("--param", "vae_tiling=true", "--notes", "retry after OOM"), "RUN", Path("/p"),
    )
    assert command[-6:] == ["--param", "cpu_offload=true", "--param", "vae_tiling=true", "--notes", "retry after OOM"]

    command = camber_run.build_command(run_args("--param", "cpu_offload=false"), "RUN", Path("/p"))
    assert [c for c in command if c.startswith("cpu_offload")] == ["cpu_offload=false"]


def test_camber_run_environment(tmp_path):
    env = camber_run.run_environment(tmp_path)
    assert env["AI_STUDIO_ROOT"] == str(tmp_path)
    assert Path(env["HF_HOME"]) == Path("/home/jovyan/hf")
    assert env["HF_HUB_OFFLINE"] == "1"


def test_camber_run_id_format():
    import re

    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", camber_run.new_run_id())


def never_start(*args, **kwargs):
    raise AssertionError("the benchmark must not be started")


def test_camber_run_refuses_without_nvidia_smi(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(camber_run.shutil, "which", lambda name: None)
    monkeypatch.setattr(camber_run, "stream", never_start)
    assert camber_run.main(["--seed", "42"], project_root=tmp_path) == 2
    assert "start the GPU Xsmall node" in capsys.readouterr().out


def test_camber_run_refuses_when_nvidia_smi_lists_no_gpu(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(camber_run.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        camber_run.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 9, stdout="", stderr="No devices were found"),
    )
    monkeypatch.setattr(camber_run, "stream", never_start)
    assert camber_run.main(["--seed", "42"], project_root=tmp_path) == 2
    assert "start the GPU Xsmall node" in capsys.readouterr().out


def gpu_present(monkeypatch):
    monkeypatch.setattr(camber_run.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        camber_run.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="Tesla L4, 23034 MiB\n", stderr=""),
    )


def test_camber_run_refuses_without_the_env(monkeypatch, tmp_path, capsys):
    gpu_present(monkeypatch)
    monkeypatch.setattr(camber_run, "ENV_PYTHON", tmp_path / "missing" / "python")
    monkeypatch.setattr(camber_run, "stream", never_start)
    assert camber_run.main(["--seed", "42"], project_root=tmp_path) == 2
    assert "camber_setup.py" in capsys.readouterr().out


def test_camber_run_runs_the_cli_then_collects(monkeypatch, tmp_path, capsys):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPTS / "collect_run.py", project / "scripts" / "collect_run.py")
    env_python = tmp_path / "env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.write_bytes(b"")
    hf_home = tmp_path / "hf"
    hf_home.mkdir()
    gpu_present(monkeypatch)
    monkeypatch.setattr(camber_run, "ENV_PYTHON", env_python)
    monkeypatch.setattr(camber_run, "HF_HOME", hf_home)
    calls = []

    def fake_stream(command, *, cwd=None, env=None):
        calls.append((command, cwd, env))
        run_id = command[command.index("--run-id") + 1]
        write_run(project, "comic_rooftop_01", run_id)  # what the real CLI would leave behind
        return 1  # a recorded failed run

    monkeypatch.setattr(camber_run, "stream", fake_stream)
    code = camber_run.main(["--seed", "7", "--param", "vae_tiling=true"], project_root=project)
    out = capsys.readouterr().out

    assert code == 1  # the CLI's exit code is passed through
    (command, cwd, env), = calls
    assert command[0] == str(env_python) and cwd == project
    assert env["HF_HUB_OFFLINE"] == "1" and env["AI_STUDIO_ROOT"] == str(project)
    assert "Tesla L4" in out
    (archive,) = (project / "dist").glob("run-comic_rooftop_01-*.zip")
    assert str(archive) in out
    with zipfile.ZipFile(archive) as z:
        assert any(name.endswith("/out.mp4") for name in z.namelist())


def test_camber_run_reports_when_nothing_was_recorded(monkeypatch, tmp_path, capsys):
    env_python = tmp_path / "python"
    env_python.write_bytes(b"")
    gpu_present(monkeypatch)
    monkeypatch.setattr(camber_run, "ENV_PYTHON", env_python)
    monkeypatch.setattr(camber_run, "HF_HOME", tmp_path)
    monkeypatch.setattr(camber_run, "stream", lambda command, **kwargs: 2)
    assert camber_run.main(["--seed", "7"], project_root=tmp_path) == 2
    assert "nothing was recorded" in capsys.readouterr().out
    assert not (tmp_path / "dist").exists()


# --- camber_setup ---

@pytest.fixture
def setup_sandbox(monkeypatch, tmp_path):
    """camber_setup pointed at temp directories, with commands recorded
    instead of run."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    (project / "requirements").mkdir(parents=True)
    (project / "requirements" / "camber-wan.txt").write_text("pyyaml==6.0.3\n", encoding="utf-8")
    env_dir = home / "envs" / "ai-studio"
    monkeypatch.setattr(camber_setup, "HOME_DIR", home)
    monkeypatch.setattr(camber_setup, "ENVS_DIR", home / "envs")
    monkeypatch.setattr(camber_setup, "ENV_DIR", env_dir)
    monkeypatch.setattr(camber_setup, "ENV_PYTHON", env_dir / "bin" / "python")
    monkeypatch.setattr(camber_setup, "HF_HOME", home / "hf")
    monkeypatch.setattr(camber_setup, "disk_use_gb", lambda path: 1.0)
    box = argparse.Namespace(project=project, env_python=env_dir / "bin" / "python", calls=[], fail_on=None)

    def fake_stream(command, *, cwd=None, env=None):
        box.calls.append((command, env))
        if box.fail_on and box.fail_on in command:
            return 1
        if "create" in command:
            box.env_python.parent.mkdir(parents=True)
            box.env_python.write_bytes(b"")
        return 0

    monkeypatch.setattr(camber_setup, "stream", fake_stream)
    return box


def test_setup_creates_env_and_installs_in_order(setup_sandbox, capsys):
    camber_setup.setup(setup_sandbox.project)
    commands = [command for command, _ in setup_sandbox.calls]
    python = str(setup_sandbox.env_python)

    assert len(commands) == 4
    create = commands[0]
    assert create[1:5] == ["create", "--yes", "--prefix", str(setup_sandbox.env_python.parents[1])]
    assert "python=3.11" in create and "conda-forge" in create
    assert commands[1] == [python, "-m", "pip", "install", "--requirement",
                           str(setup_sandbox.project / "requirements" / "camber-wan.txt")]
    assert commands[2] == [python, "-m", "pip", "install", "--editable", ".", "--no-deps"]
    assert commands[3][:2] == [python, "-c"]
    assert "import torch, torchvision, diffusers, transformers, ai_studio" in commands[3][2]
    for _, env in setup_sandbox.calls:  # caches never land on the persistent disk
        assert env["CONDA_PKGS_DIRS"] == "/tmp/conda-pkgs" and env["PIP_CACHE_DIR"] == "/tmp/pip-cache"


def test_setup_is_idempotent(setup_sandbox):
    camber_setup.setup(setup_sandbox.project)
    setup_sandbox.calls.clear()
    camber_setup.setup(setup_sandbox.project)
    commands = [command for command, _ in setup_sandbox.calls]
    assert len(commands) == 3 and not any("create" in command for command in commands)


def test_setup_removes_an_incomplete_env_before_creating(setup_sandbox):
    leftover = setup_sandbox.env_python.parents[1] / "conda-meta"
    leftover.mkdir(parents=True)
    camber_setup.setup(setup_sandbox.project)
    assert not leftover.exists() and setup_sandbox.env_python.exists()


def test_setup_stops_at_the_first_failing_step(setup_sandbox):
    setup_sandbox.fail_on = "--requirement"
    with pytest.raises(camber_setup.SetupError, match="exited with code 1"):
        camber_setup.setup(setup_sandbox.project)
    assert len(setup_sandbox.calls) == 2  # create, failed install; nothing after


def test_setup_fails_without_the_requirements_file(setup_sandbox):
    (setup_sandbox.project / "requirements" / "camber-wan.txt").unlink()
    with pytest.raises(camber_setup.SetupError, match="not found"):
        camber_setup.setup(setup_sandbox.project)
    assert setup_sandbox.calls == []


def test_setup_warns_when_the_quota_is_nearly_full(setup_sandbox, monkeypatch, capsys):
    monkeypatch.setattr(camber_setup, "disk_use_gb", lambda path: 46.5)
    camber_setup.report_disk_use()
    assert "WARNING: more than 45 GB used" in capsys.readouterr().out


# --- requirements file ---

def test_camber_requirements_are_exact_pins():
    lines = [
        line.strip()
        for line in (SCRIPTS.parent / "requirements" / "camber-wan.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines[0] == "--extra-index-url https://download.pytorch.org/whl/cu126"
    pins = dict(line.split("==") for line in lines[1:])
    assert pins["diffusers"] == "0.35.2"
    assert pins["torch"].endswith("+cu126") and pins["torchvision"].endswith("+cu126")
    assert {"transformers", "accelerate", "ftfy", "pillow", "numpy", "huggingface_hub", "pyyaml"} <= set(pins)
    assert "imageio" not in pins
