import sys

import pytest

from ai_studio.benchmarking import environment
from ai_studio.benchmarking.environment import (
    ENVIRONMENT_KEYS,
    collect_environment,
    parse_gpu_query,
    parse_reported_cuda_version,
)

CLASSIC_HEADER = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
"""

NEW_HEADER = """\
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 617.14                 KMD Version: 617.14        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
"""


def test_collects_all_keys():
    env = collect_environment()
    assert set(env) == set(ENVIRONMENT_KEYS)
    assert env["operating_system"]
    assert env["python_version"].startswith("3.11")


def test_works_without_pytorch_gpu_nvidia_smi_or_git(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "torch", None)  # makes "import torch" fail
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)  # no nvidia-smi, no git

    env = collect_environment(repo_dir=tmp_path)

    assert set(env) == set(ENVIRONMENT_KEYS)
    assert env["python_version"]
    for key in ENVIRONMENT_KEYS:
        if key not in ("operating_system", "python_version", "env_vars"):
            assert env[key] is None, key


# --- whose torch ---

@pytest.mark.parametrize("fails", [False, True])
def test_generic_torch_column_comes_from_the_backend_not_the_orchestrator(monkeypatch, project, request_factory, fails):
    import dataclasses

    from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRunError, BenchmarkRunner
    from conftest import FailingBackend, FakeBackend, fake_probe

    class OrchestratorTorch:  # a CPU build in the process that only drives the run
        __version__ = "2.13.0+cpu"
        version = type("v", (), {"cuda": None})
        cuda = type("c", (), {"is_available": staticmethod(lambda: False)})

    class OutOfProcessBackend(FakeBackend):
        def generate(self, request, output_dir):
            result = super().generate(request, output_dir)
            return dataclasses.replace(result, environment={"pytorch_version": "2.11.0+cu128", "cuda_version": "12.8"})

    monkeypatch.setitem(sys.modules, "torch", OrchestratorTorch)
    runner = BenchmarkRunner(BenchmarkLogger(project / "results"), "outputs", root=project, media_probe=fake_probe)
    if fails:
        with pytest.raises(BenchmarkRunError):
            runner.run(request_factory(), FailingBackend("generate", RuntimeError("boom")), run_id="r")
    else:
        runner.run(request_factory(), OutOfProcessBackend(), run_id="r")
    environment_ = runner.logger.load("object_replacement_01", "r").environment

    assert environment_["orchestrator_pytorch_version"] == "2.13.0+cpu"
    if fails:  # nothing ran the model: unknown, not "the orchestrator's CPU build"
        assert environment_["pytorch_version"] is None and environment_["cuda_version"] is None
    else:
        assert environment_["pytorch_version"] == "2.11.0+cu128" and environment_["cuda_version"] == "12.8"


def test_records_written_before_the_split_still_load():
    import json
    from pathlib import Path

    from ai_studio.benchmarking import BenchmarkResult

    root = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "comic_rooftop_01"
    for path in sorted(root.glob("*.json")):
        record = BenchmarkResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        assert "pytorch_version" in record.environment  # old meaning: the orchestrator's torch


# --- git status: code vs data ---

DATA_ONLY = """\
?? benchmarks/results/comic_rooftop_01/20261004T210526Z-92e20c65.json
?? benchmarks/metrics/comic_rooftop_01/20261004T210526Z-92e20c65/
?? benchmarks/evaluations/b/r/alice.r001.json
?? benchmarks/comic_evaluations/b/r/alice.r001.json
 M benchmarks/results/b/old.json
?? outputs/b/r/
"""


def test_changes_only_in_data_paths_are_not_dirty():
    info = environment.classify_git_status(DATA_ONLY)
    assert info["git_dirty"] is False
    assert info["git_untracked_data_only"] is True
    assert info["git_status"] == DATA_ONLY.splitlines()  # nothing hidden


def test_clean_tree():
    assert environment.classify_git_status("") == {
        "git_dirty": False, "git_untracked_data_only": False, "git_status": [],
    }


@pytest.mark.parametrize(
    "line",
    [
        " M src/ai_studio/cli.py",
        "M  src/ai_studio/cli.py",
        "?? scripts/new_tool.py",
        "?? notes.txt",
        " M benchmarks/definitions/comic_rooftop_01.yaml",  # a definition is not run data
        "?? benchmarks/references/comic_hero_01/extra.png",
        " D configs/quality.yaml",
        "R  outputs/b/r/out.mp4 -> src/out.mp4",  # one side outside the data paths
        "?? benchmarks/results_backup/x.json",  # only the exact data folders count
        "?? outputs.txt",
    ],
)
def test_any_code_change_is_dirty_even_next_to_data_changes(line):
    info = environment.classify_git_status(DATA_ONLY + line + "\n")
    assert info["git_dirty"] is True and info["git_untracked_data_only"] is False
    assert line in info["git_status"]


def test_rename_inside_data_paths_and_quoted_paths_are_data():
    info = environment.classify_git_status(
        'R  benchmarks/results/b/a.json -> benchmarks/results/b/c.json\n?? "outputs/b/run with space/"\n'
    )
    assert info["git_dirty"] is False and info["git_untracked_data_only"] is True


def test_git_status_is_capped():
    many = "".join(f"?? outputs/b/run-{i}/\n" for i in range(500))
    info = environment.classify_git_status(many)
    assert len(info["git_status"]) == environment.MAX_GIT_STATUS_LINES and info["git_dirty"] is False


def test_git_info_on_a_real_repository(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    if environment.shutil.which("git") is None:
        pytest.skip("git not installed")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "-m", "c")
    assert environment._git_info(tmp_path)["git_dirty"] is False

    (tmp_path / "benchmarks" / "results" / "b").mkdir(parents=True)
    (tmp_path / "benchmarks" / "results" / "b" / "r.json").write_text("{}", encoding="utf-8")
    (tmp_path / "outputs" / "b" / "r").mkdir(parents=True)
    (tmp_path / "outputs" / "b" / "r" / "out.mp4").write_bytes(b"v")
    info = environment._git_info(tmp_path)
    assert info["git_dirty"] is False and info["git_untracked_data_only"] is True
    assert len(info["git_commit"]) == 40 and info["git_status"]

    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    info = environment._git_info(tmp_path)
    assert info["git_dirty"] is True and info["git_untracked_data_only"] is False
    assert " M code.py" in info["git_status"]  # the leading status column survives


# --- env_vars ---

def test_env_vars_records_only_the_allowlist(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("HF_TOKEN", "secret-must-not-be-recorded")

    env = collect_environment()

    assert environment.RECORDED_ENV_VARS == ("PYTORCH_CUDA_ALLOC_CONF", "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE")
    assert env["env_vars"] == {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": None,
        "HF_HUB_OFFLINE": "1",
    }
    assert "secret-must-not-be-recorded" not in repr(env)


def test_env_vars_are_none_when_unset(monkeypatch):
    for name in environment.RECORDED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert collect_environment()["env_vars"] == dict.fromkeys(environment.RECORDED_ENV_VARS)


@pytest.mark.parametrize("fails", [False, True])
def test_env_vars_land_in_success_and_failed_run_records(monkeypatch, project, request_factory, fails):
    from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRunError, BenchmarkRunner
    from conftest import FailingBackend, FakeBackend, fake_probe

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    runner = BenchmarkRunner(  # default environment collector: the real one
        BenchmarkLogger(project / "results"), "outputs", root=project, media_probe=fake_probe,
    )
    if fails:
        with pytest.raises(BenchmarkRunError):
            runner.run(request_factory(), FailingBackend("generate", MemoryError("CUDA out of memory")), run_id="r")
    else:
        runner.run(request_factory(), FakeBackend(), run_id="r")

    saved = runner.logger.load("object_replacement_01", "r")
    assert saved.status == ("failed" if fails else "success")
    assert saved.environment["env_vars"] == {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": None,
        "HF_HUB_OFFLINE": None,
    }


def test_nvidia_smi_unavailable(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    assert environment._nvidia_smi_info() == {}


def test_nvidia_smi_failing_returns_empty(monkeypatch):
    def failing_run(*args, **kwargs):
        raise environment.subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(environment.subprocess, "run", failing_run)
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/" + name)
    assert environment._nvidia_smi_info() == {}
    assert environment._git_info(environment._default_repo_dir()) == {}


def test_nvidia_smi_full_capture(monkeypatch):
    monkeypatch.setattr(environment.shutil, "which", lambda name: name)

    def fake_run(args, cwd=None):
        if any(a.startswith("--query-gpu") for a in args):
            return "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 617.14"
        return NEW_HEADER

    monkeypatch.setattr(environment, "_run", fake_run)
    monkeypatch.setitem(sys.modules, "torch", None)

    env = collect_environment()
    assert env["gpu_model"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert env["vram_gb"] == 8.0
    assert env["driver_version"] == "617.14"
    assert env["reported_cuda_version"] == "13.4"
    assert env["gpu_count"] == 1
    assert env["cuda_version"] is None  # torch-built CUDA: unknown without torch


def test_parse_gpu_query_single_and_multi_gpu():
    assert parse_gpu_query("NVIDIA RTX 4090, 24564, 560.94") == {
        "gpu_model": "NVIDIA RTX 4090", "vram_gb": 23.99, "driver_version": "560.94", "gpu_count": 1,
    }
    multi = parse_gpu_query("NVIDIA H100 80GB HBM3, 81559, 550.54\nNVIDIA H100 80GB HBM3, 81559, 550.54\n")
    assert multi["gpu_count"] == 2 and multi["gpu_model"] == "NVIDIA H100 80GB HBM3"


def test_parse_gpu_query_garbage():
    assert parse_gpu_query("") == {}
    assert parse_gpu_query("No devices were found") == {}
    assert parse_gpu_query("name, notanumber, 1.0") == {}


def test_parse_reported_cuda_version_both_header_formats():
    assert parse_reported_cuda_version(CLASSIC_HEADER) == "12.4"
    assert parse_reported_cuda_version(NEW_HEADER) == "13.4"
    assert parse_reported_cuda_version("no version here") is None


def test_torch_without_cuda(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        __version__ = "9.9.9+cpu"
        version = type("v", (), {"cuda": None})
        cuda = FakeCuda

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(environment, "_nvidia_smi_info", lambda: {})

    env = collect_environment()
    # This process's torch is the ORCHESTRATOR's. Which torch ran the model is
    # for the backend to say, so the generic column makes no claim.
    assert env["orchestrator_pytorch_version"] == "9.9.9+cpu"
    assert env["orchestrator_cuda_version"] is None
    assert env["pytorch_version"] is None and env["cuda_version"] is None
    assert env["gpu_model"] is None


def test_broken_torch_import_does_not_raise(monkeypatch):
    def broken_import(name):
        raise OSError("CUDA DLL failed to load")

    monkeypatch.setattr(environment.importlib, "import_module", broken_import)
    assert environment._torch_info() == {}
