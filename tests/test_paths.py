import os
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import (
    BenchmarkLogger,
    BenchmarkRunner,
    Resolution,
    load_definition,
    load_definitions,
)
from conftest import FAKE_ENVIRONMENT, FakeBackend

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_project_root_is_the_repository():
    assert paths.project_root() == REPO_ROOT
    assert (paths.project_root() / "pyproject.toml").is_file()


def test_project_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path))
    assert paths.project_root() == tmp_path.resolve()


def test_project_root_env_override_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError):
        paths.project_root()


def test_resolve_path_ignores_cwd(monkeypatch, tmp_path):
    expected = REPO_ROOT / "benchmarks" / "definitions"
    monkeypatch.chdir(tmp_path)
    assert paths.resolve_path("benchmarks/definitions") == expected
    assert paths.definitions_dir() == expected


def test_absolute_paths_are_kept(tmp_path):
    assert paths.resolve_path(tmp_path / "x.mp4") == (tmp_path / "x.mp4").resolve()


def test_to_project_relative(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert paths.to_project_relative(root / "benchmarks" / "a.mp4", root) == "benchmarks/a.mp4"
    assert paths.to_project_relative("benchmarks/a.mp4", root) == "benchmarks/a.mp4"
    outside = tmp_path / "elsewhere" / "b.mp4"
    assert paths.to_project_relative(outside, root) == outside.resolve().as_posix()
    assert "\\" not in paths.to_project_relative(outside, root)
    assert paths.to_project_relative("outputs/x/../y/out.mp4", root) == "outputs/y/out.mp4"


@pytest.mark.skipif(os.name != "nt", reason="backslash is a path separator only on Windows")
def test_windows_backslash_paths_normalize(tmp_path):
    root = tmp_path / "root"
    windows_style = str(root / "outputs" / "b" / "run" / "out.mp4").replace("/", "\\")
    assert paths.to_project_relative(windows_style, root) == "outputs/b/run/out.mp4"
    mixed = str(root) + r"\outputs/b\run/out.mp4"
    assert paths.to_project_relative(mixed, root) == "outputs/b/run/out.mp4"


@pytest.mark.parametrize("cwd", ["repo", "tmp", "subdir"])
def test_repository_definition_loads_from_any_cwd(monkeypatch, tmp_path, cwd):
    target = {"repo": REPO_ROOT, "tmp": tmp_path, "subdir": REPO_ROOT / "src" / "ai_studio"}[cwd]
    monkeypatch.chdir(target)
    definition = load_definition("benchmarks/definitions/object_replacement_01.yaml")
    assert definition.benchmark_id == "object_replacement_01"
    assert "object_replacement_01" in load_definitions()


def make_project(root: Path) -> None:
    (root / "benchmarks" / "inputs").mkdir(parents=True)
    (root / "benchmarks" / "references").mkdir(parents=True)
    (root / "benchmarks" / "definitions").mkdir(parents=True)
    (root / "benchmarks" / "inputs" / "clip.mp4").write_bytes(b"video")
    (root / "benchmarks" / "references" / "teapot.png").write_bytes(b"image")
    (root / "benchmarks" / "definitions" / "swap.yaml").write_text(
        "benchmark_id: swap_01\n"
        "source_video: benchmarks/inputs/clip.mp4\n"
        "reference_images: [benchmarks/references/teapot.png]\n"
        "prompt: replace the mug with the teapot\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("cwd", ["root", "outside", "nested"])
def test_same_definition_runs_identically_from_any_cwd(monkeypatch, tmp_path, cwd):
    root = tmp_path / "project"
    make_project(root)
    (tmp_path / "outside").mkdir()
    target = {"root": root, "outside": tmp_path / "outside", "nested": root / "benchmarks" / "inputs"}[cwd]
    monkeypatch.chdir(target)
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(root))

    definition = load_definitions()["swap_01"]
    request = definition.to_request(seed=1, output_resolution=Resolution(width=64, height=64))
    backend = FakeBackend()
    runner = BenchmarkRunner(
        BenchmarkLogger("benchmarks/results"), "outputs",
        environment_collector=lambda: dict(FAKE_ENVIRONMENT),
    )
    record = runner.run(request, backend, run_id="run-1")

    # Records are identical regardless of cwd: portable, project-relative paths.
    assert record.source_video == "benchmarks/inputs/clip.mp4"
    assert record.reference_assets == ["benchmarks/references/teapot.png"]
    assert set(record.input_checksums) == {"benchmarks/inputs/clip.mp4", "benchmarks/references/teapot.png"}
    # Files landed under the project root, not under the cwd.
    assert (root / "benchmarks" / "results" / "swap_01" / "run-1.json").is_file()
    assert (root / "outputs" / "swap_01" / "run-1" / "out.mp4").is_file()
