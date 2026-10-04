"""Backend registry and the benchmark command line."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ai_studio import cli, paths
from ai_studio.cli import main, parse_param, parse_params, run_benchmark
from ai_studio.generation.backends import (
    BACKENDS,
    BackendNotInstalledError,
    JoyAIVideoEditBackend,
    KiwiEditBackend,
    VideoBackend,
    WanEditCtrlBackend,
    WanTI2V5BBackend,
    get_backend,
)
from ai_studio.generation.backends import registry
from conftest import FakeGenerationBackend, fake_probe

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFINITION = """\
benchmark_id: {benchmark_id}
task_type: video_generation
status: {status}
prompt: a rooftop at sunset
duration_seconds: 5
fps: 24
duration_tolerance_seconds: 0.1
"""


# --- registry ---

def test_registry_lists_every_backend_by_its_backend_name():
    assert BACKENDS == {
        "wan-ti2v-5b": WanTI2V5BBackend,
        "joyai-video-edit": JoyAIVideoEditBackend,
        "kiwi-edit": KiwiEditBackend,
        "wan2.2-vace-fun-a14b-editctrl": WanEditCtrlBackend,
    }
    for name, backend_class in BACKENDS.items():
        assert issubclass(backend_class, VideoBackend)
        assert backend_class().metadata().backend_name == name


def test_get_backend_returns_the_class():
    assert get_backend("wan-ti2v-5b") is WanTI2V5BBackend


def test_unknown_backend_error_lists_valid_names():
    with pytest.raises(KeyError) as exc_info:
        get_backend("wan")
    message = str(exc_info.value)
    assert "unknown backend 'wan'" in message
    for name in BACKENDS:
        assert name in message


def test_runner_does_not_know_the_registry():
    import inspect

    import ai_studio.benchmarking.runner as runner_module

    assert "registry" not in inspect.getsource(runner_module)


# --- --param parsing ---

@pytest.mark.parametrize(
    "text, expected",
    [
        ("cpu_offload=true", ("cpu_offload", True)),
        ("vae_tiling=false", ("vae_tiling", False)),
        ("num_inference_steps=50", ("num_inference_steps", 50)),
        ("guidance_scale=4.5", ("guidance_scale", 4.5)),
        ("revision=b8fff7315c768468a5333511427288870b2e9635", ("revision", "b8fff7315c768468a5333511427288870b2e9635")),
        ("revision=main", ("revision", "main")),
        ('name="quoted"', ("name", "quoted")),
        ("empty=", ("empty", "")),
        ("expr=a=b", ("expr", "a=b")),
        ("nothing=null", ("nothing", None)),
        ("sizes=[1, 2]", ("sizes", [1, 2])),
        ("flag=True", ("flag", "True")),  # not JSON: stays a string, the backend rejects it
    ],
)
def test_parse_param(text, expected):
    assert parse_param(text) == expected


@pytest.mark.parametrize("bad", ["cpu_offload", "=true", " =1", ""])
def test_parse_param_rejects_malformed(bad):
    with pytest.raises(ValueError, match="key=value"):
        parse_param(bad)


def test_parse_params_rejects_duplicates():
    assert parse_params(["a=1", "b=x"]) == {"a": 1, "b": "x"}
    with pytest.raises(ValueError, match="more than once"):
        parse_params(["a=1", "a=2"])


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--benchmark", "b", "--backend", "x", "--seed", "1", "--width", "1280"],  # no height
        ["--benchmark", "b", "--backend", "x", "--seed", "one", "--width", "1280", "--height", "704"],
        ["--benchmark", "b", "--backend", "x", "--seed", "1", "--width", "1280", "--height", "704", "--param", "oops"],
    ],
)
def test_bad_arguments_exit_with_code_2(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2


# --- end to end with a fake backend ---

class FailingGenerationBackend(FakeGenerationBackend):
    def __init__(self):
        super().__init__(fail_in="generate", exc=MemoryError("CUDA out of memory"))


class NotInstalledGenerationBackend(FakeGenerationBackend):
    def prepare(self):
        raise BackendNotInstalledError("fake backend is not installed")


@pytest.fixture
def cli_project(project, monkeypatch):
    """Project root with one ready and one draft definition, fake backends
    registered, and a fake media probe."""
    definitions = project / "benchmarks" / "definitions"
    definitions.mkdir(parents=True)
    (definitions / "cli_01.yaml").write_text(
        DEFINITION.format(benchmark_id="cli_01", status="ready"), encoding="utf-8",
    )
    (definitions / "draft_01.yaml").write_text(
        DEFINITION.format(benchmark_id="draft_01", status="draft"), encoding="utf-8",
    )
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    monkeypatch.setitem(registry.BACKENDS, "fake-gen", FakeGenerationBackend)
    monkeypatch.setitem(registry.BACKENDS, "fake-failing", FailingGenerationBackend)
    monkeypatch.setitem(registry.BACKENDS, "fake-missing", NotInstalledGenerationBackend)
    monkeypatch.setattr(cli, "ffprobe_video", fake_probe)
    return project


def argv(benchmark="cli_01", backend="fake-gen", *extra):
    return ["--benchmark", benchmark, "--backend", backend, "--seed", "42",
            "--width", "1280", "--height", "704", *extra]


def written_files(project):
    return sorted(
        path.relative_to(project).as_posix()
        for folder in ("benchmarks/results", "outputs") if (project / folder).exists()
        for path in (project / folder).rglob("*") if path.is_file()
    )


def test_successful_run_exits_0_and_reports(cli_project, capsys):
    code = main(argv("cli_01", "fake-gen", "--run-id", "run-1", "--notes", "first try",
                     "--param", "steps=30", "--param", "label=baseline"))
    out = capsys.readouterr().out

    assert code == 0
    assert "status: success" in out
    assert "record: benchmarks/results/cli_01/run-1.json" in out
    assert "output video: outputs/cli_01/run-1/gen.mp4" in out
    assert "measured: 24 fps, 120 frames, 5.000 s, 1280x704" in out
    assert "total time:" in out

    record = json.loads((cli_project / "benchmarks/results/cli_01/run-1.json").read_text(encoding="utf-8"))
    assert record["status"] == "success" and record["seed"] == 42
    assert record["resolution"] == {"width": 1280, "height": 704}
    assert record["model_parameters"] == {"steps": 30, "label": "baseline", "sampler": "fake"}
    assert record["notes"] == "first try"
    assert (cli_project / "outputs/cli_01/run-1/gen.mp4").is_file()


def test_run_benchmark_returns_written_project_relative_paths(cli_project):
    lines = []
    code, written = run_benchmark(
        benchmark="cli_01", backend="fake-gen", seed=1, width=1280, height=704,
        run_id="run-2", out=lines.append,
    )
    assert code == 0
    assert written == ["benchmarks/results/cli_01/run-2.json", "outputs/cli_01/run-2"]
    assert all((cli_project / path).exists() for path in written)
    assert any(line.startswith("status: success") for line in lines)


def test_generated_run_id_is_used_when_none_is_given(cli_project):
    code, written = run_benchmark(
        benchmark="cli_01", backend="fake-gen", seed=1, width=1280, height=704, out=lambda line: None,
    )
    assert code == 0 and len(written) == 2
    assert Path(written[0]).stem == Path(written[1]).name


def test_failed_run_exits_1_and_is_recorded(cli_project, capsys):
    code = main(argv("cli_01", "fake-failing", "--run-id", "boom"))
    out = capsys.readouterr().out

    assert code == 1
    assert "status: failed (stage: generation)" in out
    assert "error: MemoryError: CUDA out of memory" in out
    assert "record: benchmarks/results/cli_01/boom.json" in out
    record = json.loads((cli_project / "benchmarks/results/cli_01/boom.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed" and record["failure_stage"] == "generation"


def test_draft_benchmark_exits_2_and_writes_nothing(cli_project, capsys):
    code = main(argv("draft_01"))
    out = capsys.readouterr().out
    assert code == 2
    assert "setup problem, nothing recorded" in out and "draft" in out
    assert written_files(cli_project) == []


@pytest.mark.parametrize(
    "benchmark, backend, expected",
    [
        ("cli_01", "fake-missing", "BackendNotInstalledError"),
        ("cli_01", "no-such-backend", "unknown backend 'no-such-backend'; valid names:"),
        ("no_such_benchmark", "fake-gen", "no benchmark definition benchmarks/definitions/no_such_benchmark.yaml"),
        ("../cli_01", "fake-gen", "must be usable as a file name"),
        ("cli_01", "kiwi-edit", "UnsupportedCapabilityError"),  # editing backend, generation benchmark
    ],
)
def test_setup_problems_exit_2_and_write_nothing(cli_project, capsys, benchmark, backend, expected):
    code = main(argv(benchmark, backend))
    out = capsys.readouterr().out
    assert code == 2
    assert "setup problem, nothing recorded" in out and expected in out
    assert written_files(cli_project) == []


def test_existing_run_id_exits_2_and_keeps_the_first_run(cli_project, capsys):
    assert main(argv("cli_01", "fake-gen", "--run-id", "same")) == 0
    before = written_files(cli_project)
    assert main(argv("cli_01", "fake-gen", "--run-id", "same")) == 2
    assert "already exists" in capsys.readouterr().out
    assert written_files(cli_project) == before


def test_wrapper_script_runs_the_cli(cli_project):
    # A real subprocess: the wrapper must work without the package installed.
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_benchmark.py"), *argv("draft_01", "wan-ti2v-5b")],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": ""},
    )
    assert completed.returncode == 2, completed.stderr
    assert "draft" in completed.stdout
