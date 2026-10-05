"""Protection of recorded run data: read-only run files, and clean-up code
that refuses to delete anything under outputs/ or benchmarks/."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

from ai_studio import paths, protection
from ai_studio.benchmarking import BenchmarkRunError
from ai_studio.generation.backends.comfyui import ComfyUIConfig
from ai_studio.generation.backends.wan_comfyui import OUTPUT_PREFIX, WanComfyUIBackend
from ai_studio.protection import (
    ProtectedPathError,
    assert_deletable,
    is_protected,
    is_read_only,
    make_read_only,
    protect_run_files,
    safe_rmdir,
    safe_rmtree,
    safe_unlink,
)
from conftest import FailingBackend, FakeBackend
from test_start_frame import FRAME, RES, ImageToVideoBackend, frame, write_definition  # noqa: F401 (fixture)

BENCH = "object_replacement_01"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def touch(path, data=b"data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# --- which paths are protected ---

@pytest.mark.parametrize("relative", [
    "outputs", "benchmarks", "outputs/b/run-1/out.mp4", "benchmarks/results/b/run-1.json",
    "outputs/trailers/trailer_01", "src/../outputs/b", "benchmarks/inputs/../references/frame.png",
])
def test_everything_under_outputs_and_benchmarks_is_protected(project, relative):
    assert is_protected(project / relative, project)


@pytest.mark.parametrize("relative", [
    ".", "src/module.py", "outputs_old/file", "benchmarks2", "outputs/../tmp/file", "docs/outputs/file",
])
def test_paths_outside_the_data_folders_are_not_protected(project, relative):
    assert not is_protected(project / relative, project)


def test_another_projects_folders_are_not_protected(project, tmp_path):
    assert not is_protected(tmp_path / "elsewhere" / "outputs" / "x.mp4", project)


def test_root_defaults_to_the_project_root(project, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    assert is_protected(project / "outputs" / "x.mp4")
    assert not is_protected(project / "tmp" / "x.mp4")


# --- clean-up helpers ---

def test_safe_unlink_refuses_run_data_and_leaves_it_in_place(project):
    video = touch(project / "outputs" / "b" / "run-1" / "out.mp4")
    record = touch(project / "benchmarks" / "results" / "b" / "run-1.json")
    for path in (video, record):
        with pytest.raises(ProtectedPathError, match="clean-up never touches"):
            safe_unlink(path, project)
        assert path.read_bytes() == b"data"


def test_safe_rmtree_and_rmdir_refuse_run_data(project):
    video = touch(project / "outputs" / "b" / "run-1" / "out.mp4")
    empty = project / "benchmarks" / "empty"
    empty.mkdir(parents=True)
    for target in (video.parent, project / "outputs", project / "benchmarks"):
        with pytest.raises(ProtectedPathError):
            safe_rmtree(target, project)
        # ignore_errors silences deletion problems, never the refusal
        with pytest.raises(ProtectedPathError):
            safe_rmtree(target, project, ignore_errors=True)
    with pytest.raises(ProtectedPathError):
        safe_rmdir(empty, project)
    assert video.is_file() and empty.is_dir()


def test_refusal_sees_through_dot_dot(project):
    video = touch(project / "outputs" / "b" / "out.mp4")
    (project / "tmp").mkdir()
    with pytest.raises(ProtectedPathError):
        safe_unlink(project / "tmp" / ".." / "outputs" / "b" / "out.mp4", project)
    assert video.is_file()


def test_safe_helpers_delete_outside_the_data_folders(project):
    file = touch(project / "tmp" / "work" / "frame.png")
    tree = touch(project / "tmp" / "tree" / "a" / "b.txt").parents[1]
    empty = project / "tmp" / "empty"
    empty.mkdir()
    safe_unlink(file, project)
    safe_rmtree(tree, project)
    safe_rmdir(empty, project)
    assert not file.exists() and not tree.exists() and not empty.exists()


def test_protected_path_error_is_a_permission_error():
    # best-effort clean-up code that catches OSError keeps working
    assert issubclass(ProtectedPathError, PermissionError) and issubclass(ProtectedPathError, OSError)


def test_unknown_project_root_means_refusal(tmp_path, monkeypatch):
    def no_root():
        raise RuntimeError("cannot locate the project root")

    monkeypatch.setattr(protection, "project_root", no_root)
    file = touch(tmp_path / "anything.tmp")
    with pytest.raises(ProtectedPathError, match="project root is unknown"):
        safe_unlink(file)
    with pytest.raises(ProtectedPathError):
        assert_deletable(file)
    assert file.is_file()


# --- read-only files ---

def test_make_read_only(tmp_path):
    file = touch(tmp_path / "out.mp4")
    assert not is_read_only(file)
    make_read_only(file)
    assert is_read_only(file)
    assert file.read_bytes() == b"data"  # still readable
    if os.name == "nt" or os.geteuid() != 0:  # root ignores permission bits
        with pytest.raises(PermissionError):
            file.write_bytes(b"changed")
    if os.name == "nt":  # on Windows a read-only file cannot be deleted either
        with pytest.raises(PermissionError):
            file.unlink()
    assert file.read_bytes() == b"data"


# --- the runner protects what it recorded ---

def test_successful_run_marks_its_video_read_only(runner, project, request_factory):
    record = runner.run(request_factory(), FakeBackend(), run_id="run-1")
    video = project / record.output_video
    assert is_read_only(video)
    # inputs are not the run's files: they stay as they were
    assert not is_read_only(project / "benchmarks" / "inputs" / "source.mp4")
    # the record names the same, unchanged file
    assert runner.logger.load(BENCH, "run-1").output_video == record.output_video


def test_successful_run_marks_its_derived_start_frame_read_only(runner, project, frame):  # noqa: F811
    definition = write_definition(project, f"fps: 24\nstart_frame: {FRAME}\n")
    record = runner.run_definition(definition, ImageToVideoBackend(), seed=1, output_resolution=RES, run_id="r1")
    assert is_read_only(project / record.output_video)
    assert is_read_only(project / record.start_frame["derived_file"])
    assert not is_read_only(frame)  # the original start frame is an input


def test_failed_run_protects_nothing(runner, project, request_factory):
    with pytest.raises(BenchmarkRunError) as info:
        runner.run(request_factory(), FailingBackend("generate", RuntimeError("boom")), run_id="run-f")
    assert info.value.record.status == "failed"
    assert protect_run_files(info.value.record, project) == []


def test_protect_run_files_skips_missing_files(runner, project, request_factory):
    record = runner.run(request_factory(), FakeBackend(), run_id="run-1")
    video = project / record.output_video
    os.chmod(video, 0o666)
    assert protect_run_files(record, project) == [video.resolve()]
    assert is_read_only(video)
    os.chmod(video, 0o666)
    video.rename(video.with_name("moved.mp4"))
    assert protect_run_files(record, project) == []


# --- clean-up call sites ---

def comfy_config(comfy_dir):
    return ComfyUIConfig(
        comfyui_dir=comfy_dir, models={"unet": "u.gguf", "text_encoder": "t.safetensors", "vae": "v.safetensors"},
        python=Path(sys.executable), port=1,
        startup_timeout_seconds=1, generation_timeout_seconds=1, sha256={},
    )


def comfy_leftovers(comfy_dir):
    frames = touch(comfy_dir / "output" / OUTPUT_PREFIX / "key" / "frame_0001.png")
    uploaded = touch(comfy_dir / "input" / "ai_studio_key.png")
    return frames, uploaded


def test_comfyui_clean_up_removes_its_working_copies(project, tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    comfy_dir = tmp_path / "ComfyUI"
    frames, uploaded = comfy_leftovers(comfy_dir)
    WanComfyUIBackend._clean_up(comfy_config(comfy_dir), "ai_studio_key.png", "key")
    assert not frames.parent.exists() and not uploaded.exists()


@pytest.mark.parametrize("folder", ["outputs", "benchmarks"])
def test_comfyui_clean_up_never_deletes_under_the_data_folders(project, monkeypatch, folder):
    # A ComfyUI directory misconfigured to live inside the project's data.
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    comfy_dir = project / folder / "ComfyUI"
    frames, uploaded = comfy_leftovers(comfy_dir)
    WanComfyUIBackend._clean_up(comfy_config(comfy_dir), "ai_studio_key.png", "key")  # best effort: no error
    assert frames.is_file() and uploaded.is_file()


def load_camber_setup():
    spec = importlib.util.spec_from_file_location("script_camber_setup_protection", SCRIPTS / "camber_setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("folder", ["outputs", "benchmarks"])
def test_camber_setup_refuses_to_remove_an_env_inside_the_data_folders(tmp_path, monkeypatch, folder):
    camber_setup = load_camber_setup()
    project = tmp_path / "project"
    touch(project / "requirements" / "camber-wan.txt", b"pyyaml\n")
    env_dir = project / folder / "envs" / "ai-studio"
    leftover = touch(env_dir / "half-made.txt")
    monkeypatch.setattr(camber_setup, "ENV_DIR", env_dir)
    monkeypatch.setattr(camber_setup, "ENVS_DIR", env_dir.parent)
    monkeypatch.setattr(camber_setup, "ENV_PYTHON", env_dir / "bin" / "python")
    monkeypatch.setattr(camber_setup, "stream", lambda *args, **kwargs: pytest.fail("no command may run"))
    with pytest.raises(camber_setup.SetupError, match="refusing to remove"):
        camber_setup.setup(project)
    assert leftover.is_file()
