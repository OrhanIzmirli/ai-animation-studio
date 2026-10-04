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
    assert env["pytorch_version"] == "9.9.9+cpu"
    assert env["cuda_version"] is None
    assert env["gpu_model"] is None


def test_broken_torch_import_does_not_raise(monkeypatch):
    def broken_import(name):
        raise OSError("CUDA DLL failed to load")

    monkeypatch.setattr(environment.importlib, "import_module", broken_import)
    assert environment._torch_info() == {}
