"""The local ComfyUI backend for Wan 2.2 TI2V-5B Turbo. No ComfyUI, GPU or
model needed: a fake ComfyUI HTTP server stands in (tests/fake_comfyui.py)."""

import hashlib
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest

from ai_studio import paths
from ai_studio.benchmarking import BenchmarkLogger, BenchmarkRunError, BenchmarkRunner, Resolution, load_definition
from ai_studio.benchmarking.runner import sha256_file
from ai_studio.generation.backends import (
    BACKENDS,
    BackendNotInstalledError,
    VideoGenerationBackend,
    WanTI2V5BBackend,
    WanTurboComfyUIBackend,
)
from ai_studio.generation.backends import comfyui as comfyui_module
from ai_studio.generation.backends import registry
from ai_studio.generation.backends import wan_turbo_comfyui as turbo
from ai_studio.generation.backends.comfyui import ComfyUIConfig, ComfyUIServer, load_config
from ai_studio.generation.backends.wan_turbo_comfyui import build_workflow, temporal_tile_for
from ai_studio.generation.capabilities import Capability
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.system_memory import LowestAvailableRam, available_ram_gb, total_ram_gb
from conftest import FAKE_ENVIRONMENT
from fake_comfyui import TEXT_ENCODER, UNET, VAE, FakeComfyUI

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

REPO_ROOT = Path(__file__).resolve().parents[1]
LANDSCAPE = Resolution(width=1280, height=704)
FRAME = "benchmarks/references/comic_hero_01/rooftop_frame.png"
MODELS = {"unet": UNET, "text_encoder": TEXT_ENCODER, "vae": VAE}
MODEL_BYTES = {"unet": b"fake turbo gguf", "text_encoder": b"fake umt5", "vae": b"fake vae"}
MODEL_SHA = {kind: hashlib.sha256(data).hexdigest() for kind, data in MODEL_BYTES.items()}

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def comfy_dir(tmp_path):
    """A directory shaped like a ComfyUI install, with tiny model files and
    the fake server as main.py."""
    root = tmp_path / "ComfyUI"
    for kind, folder in (("unet", "diffusion_models"), ("text_encoder", "text_encoders"), ("vae", "vae")):
        (root / "models" / folder).mkdir(parents=True)
        (root / "models" / folder / MODELS[kind]).write_bytes(MODEL_BYTES[kind])
    shutil.copy(Path(__file__).parent / "fake_comfyui.py", root / "main.py")
    return root


@pytest.fixture
def fake():
    server = FakeComfyUI().start()
    yield server
    server.stop()


def make_config(comfy_dir, port, **overrides):
    fields = dict(
        comfyui_dir=comfy_dir, models=dict(MODELS), python=Path(sys.executable), port=port,
        startup_timeout_seconds=30, generation_timeout_seconds=30, sha256=dict(MODEL_SHA),
    )
    fields.update(overrides)
    return ComfyUIConfig(**fields)


def make_request(**overrides):
    fields = dict(
        prompt="a young man sits on a rooftop ledge at sunset", negative_prompt="photorealistic",
        start_frame="frame.png", duration_seconds=5, fps=24, output_resolution=LANDSCAPE, seed=42,
    )
    fields.update(overrides)
    return VideoGenerationRequest(**fields)


# --- identity and registry ---

def test_backend_is_registered_under_its_own_name_and_is_not_the_base_model():
    assert BACKENDS["wan-ti2v-5b-turbo-comfyui"] is WanTurboComfyUIBackend
    assert issubclass(WanTurboComfyUIBackend, VideoGenerationBackend)
    assert WanTurboComfyUIBackend.capabilities == {Capability.VIDEO_GENERATION}
    assert WanTurboComfyUIBackend.supports_start_frame is True

    turbo_meta = WanTurboComfyUIBackend(make_config(Path("nowhere"), 1)).metadata()
    base_meta = WanTI2V5BBackend().metadata()
    assert turbo_meta.backend_name == "wan-ti2v-5b-turbo-comfyui"
    assert turbo_meta.model_name == "Wan2.2-TI2V-5B-Turbo (quanhaol) GGUF Q8_0"
    assert turbo_meta.model_version == MODEL_SHA["unet"]  # the pinned GGUF sha256
    assert turbo_meta.model_name != base_meta.model_name
    assert turbo_meta.backend_name != base_meta.backend_name


def test_metadata_works_without_any_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv(comfyui_module.CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))
    meta = WanTurboComfyUIBackend().metadata()
    assert meta.model_version == "unverified" and meta.workflow_version == "i2v-turbo-tiled-v1"


def test_module_needs_no_heavy_dependency():
    import subprocess

    code = (
        "import sys\n"
        "import ai_studio.generation.backends.wan_turbo_comfyui\n"
        "bad = [m for m in ('torch', 'PIL', 'numpy', 'psutil') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    import os
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")})


def test_no_machine_specific_path_in_the_source():
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "Users\\" not in text and "/Users/" not in text and "ai-tools" not in text, path


# --- temporal tile: computed, never a setting ---

@pytest.mark.parametrize("frames", [1, 5, 13, 49, 97, 117, 120, 121, 122, 124, 125])
def test_temporal_tile_always_covers_the_whole_clip(frames):
    tile = temporal_tile_for(frames)
    assert tile > frames and tile % 4 == 0
    assert tile // 4 >= (frames - 1) // 4 + 1  # latent frames per tile >= latent frames of the clip


def test_temporal_tile_for_the_benchmark_matches_the_spike():
    assert temporal_tile_for(121) == 128  # spike run (d)


@pytest.mark.parametrize("key", ["temporal_tile", "temporal_size", "cfg", "guidance_scale", "vae_tiling"])
def test_temporal_tile_and_cfg_are_not_parameters(key):
    assert key not in turbo.ALLOWED_BACKEND_PARAMETERS
    with pytest.raises(ValueError, match="unknown backend_parameters"):
        WanTurboComfyUIBackend(make_config(Path("x"), 1)).validate_inputs(make_request(backend_parameters={key: 64}))


# --- workflow ---

def workflow(**settings):
    return build_workflow(
        models=MODELS, image_name="start.png", prompt="PROMPT", negative_prompt="NEG", width=1280, height=704,
        num_frames=121, seed=42, settings={**turbo.DEFAULTS, **settings}, output_prefix="ai_studio_runs/abc",
    )


def test_default_workflow_is_spike_run_d():
    nodes = workflow()
    by_class = {node["class_type"]: node["inputs"] for node in nodes.values()}
    assert set(by_class) == set(turbo.REQUIRED_NODES)  # no LatentMultiply by default
    assert by_class["UnetLoaderGGUF"] == {"unet_name": UNET}
    assert by_class["CLIPLoader"]["clip_name"] == TEXT_ENCODER and by_class["CLIPLoader"]["type"] == "wan"
    assert by_class["VAELoader"] == {"vae_name": VAE}
    assert by_class["ModelSamplingSD3"]["shift"] == 8.0
    assert by_class["Wan22ImageToVideoLatent"] | {"vae": 0, "start_image": 0} == {
        "vae": 0, "start_image": 0, "width": 1280, "height": 704, "length": 121, "batch_size": 1,
    }
    sampler = by_class["KSampler"]
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (42, 4, 1.0)
    assert (sampler["sampler_name"], sampler["scheduler"], sampler["denoise"]) == ("uni_pc", "simple", 1.0)
    decode = by_class["VAEDecodeTiled"]
    assert (decode["tile_size"], decode["overlap"]) == (512, 64)
    assert (decode["temporal_size"], decode["temporal_overlap"]) == (128, 8)
    assert decode["samples"] == ["9", 0]  # straight from the sampler
    assert by_class["SaveImage"]["filename_prefix"] == "ai_studio_runs/abc/frame"
    texts = [n["inputs"]["text"] for n in nodes.values() if n["class_type"] == "CLIPTextEncode"]
    assert texts == ["PROMPT", "NEG"]  # passed unchanged


def test_latent_scale_inserts_a_multiply_node_before_the_decode():
    nodes = workflow(latent_scale=0.8)
    multiply = next(n for n in nodes.values() if n["class_type"] == "LatentMultiply")
    assert multiply["inputs"] == {"samples": ["9", 0], "multiplier": 0.8}
    decode = next(n for n in nodes.values() if n["class_type"] == "VAEDecodeTiled")
    multiply_id = next(i for i, n in nodes.items() if n["class_type"] == "LatentMultiply")
    assert decode["inputs"]["samples"] == [multiply_id, 0]


# --- validation ---

def test_valid_requests_accepted():
    backend = WanTurboComfyUIBackend(make_config(Path("x"), 1))
    backend.validate_inputs(make_request())
    backend.validate_inputs(make_request(fps=None, output_resolution=Resolution(width=704, height=1280), backend_parameters={
        "num_inference_steps": 6, "shift": 5, "sampler": "euler", "scheduler": "beta",
        "spatial_tile": 256, "latent_scale": 0.8,
    }))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"start_frame": None}, "start_frame is required"),
        ({"fps": 16}, "fps must be 24"),
        ({"output_resolution": Resolution(width=1280, height=720)}, "output_resolution must be"),
        ({"duration_seconds": 6}, "not verified"),
        ({"seed": -1}, "seed must be >= 0"),
        ({"backend_parameters": {"num_inference_steps": 0}}, "num_inference_steps"),
        ({"backend_parameters": {"shift": 0}}, "shift"),
        ({"backend_parameters": {"sampler": ""}}, "sampler"),
        ({"backend_parameters": {"scheduler": 3}}, "scheduler"),
        ({"backend_parameters": {"spatial_tile": 500}}, "spatial_tile"),
        ({"backend_parameters": {"spatial_tile": 32}}, "spatial_tile"),
        ({"backend_parameters": {"latent_scale": 0}}, "latent_scale"),
        ({"backend_parameters": {"latent_scale": "0.8"}}, "latent_scale"),
        ({"backend_parameters": {"cpu_offload": True}}, "unknown backend_parameters"),
    ],
)
def test_invalid_requests_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        WanTurboComfyUIBackend(make_config(Path("x"), 1)).validate_inputs(make_request(**overrides))


# --- configuration ---

def test_example_config_is_valid_and_has_no_personal_path(monkeypatch):
    example = REPO_ROOT / "configs" / "comfyui.example.yaml"
    config = load_config(example)
    assert config.models == MODELS and config.port == 8188
    assert set(config.sha256) == {"unet", "text_encoder", "vae"}
    assert config.sha256["unet"] == "b4d7d1888d5ec2c29046cc443239faad99614cd3d356c61d8192fcac5f4c98fc"
    assert "rhn23" not in example.read_text(encoding="utf-8")
    assert config.python_executable.parent.name in ("Scripts", "bin")


def test_local_config_is_git_ignored():
    assert "configs/*.local.yaml" in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")


def test_config_is_found_through_the_environment_variable(monkeypatch, tmp_path, comfy_dir):
    path = tmp_path / "my.yaml"
    path.write_text(
        f"comfyui_dir: {comfy_dir.as_posix()}\nport: 9000\n"
        f"models: {{unet: {UNET}, text_encoder: {TEXT_ENCODER}, vae: {VAE}}}\n", encoding="utf-8",
    )
    monkeypatch.setenv(comfyui_module.CONFIG_ENV_VAR, str(path))
    config = load_config()
    assert config.comfyui_dir == comfy_dir and config.port == 9000 and config.sha256 == {}
    assert config.model_path("unet") == comfy_dir / "models" / "diffusion_models" / UNET


@pytest.mark.parametrize(
    "text, match",
    [
        (None, "not configured"),
        ("- a list\n", "expected a mapping"),
        ("port: 8188\n", "comfyui_dir is required"),
        ("comfyui_dir: x\nmodels: {unet: a.gguf}\n", "models must name exactly"),
        ("comfyui_dir: x\nmodels: {unet: sub/a.gguf, text_encoder: t, vae: v}\n", "file name without a directory"),
        ("comfyui_dir: x\nmodels: {unet: a, text_encoder: t, vae: v}\nport: 0\n", "port"),
        ("comfyui_dir: x\nmodels: {unet: a, text_encoder: t, vae: v}\nsha256: {unet: abc}\n", "sha256"),
        ("comfyui_dir: x\nmodels: {unet: a, text_encoder: t, vae: v}\ncomfy_dir: y\n", "unknown keys"),
    ],
)
def test_missing_or_invalid_config_is_a_setup_problem(tmp_path, text, match):
    path = tmp_path / "c.yaml"
    if text is not None:
        path.write_text(text, encoding="utf-8")
    with pytest.raises(BackendNotInstalledError, match=match):
        load_config(path)


# --- prepare(): not installed ---

def test_missing_comfyui_directory(tmp_path):
    with pytest.raises(BackendNotInstalledError, match="is not a directory"):
        WanTurboComfyUIBackend(make_config(tmp_path / "nope", free_port())).prepare()


def test_missing_model_file(comfy_dir, fake):
    (comfy_dir / "models" / "text_encoders" / TEXT_ENCODER).unlink()
    with pytest.raises(BackendNotInstalledError, match=f"text_encoder: {TEXT_ENCODER}"):
        WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)).prepare()


def test_wrong_model_file_is_refused(comfy_dir, fake):
    (comfy_dir / "models" / "diffusion_models" / UNET).write_bytes(b"some other model")
    with pytest.raises(BackendNotInstalledError, match="not the expected unet file"):
        WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)).prepare()


@pytest.mark.parametrize("node", ["UnetLoaderGGUF", "Wan22ImageToVideoLatent", "VAEDecodeTiled"])
def test_missing_node_class(comfy_dir, node):
    server = FakeComfyUI(missing_nodes=[node]).start()
    try:
        with pytest.raises(BackendNotInstalledError, match=node):
            WanTurboComfyUIBackend(make_config(comfy_dir, server.port)).prepare()
    finally:
        server.stop()


def test_missing_optional_node_matters_only_when_it_is_used(comfy_dir):
    server = FakeComfyUI(missing_nodes=["LatentMultiply"]).start()
    try:
        plain = WanTurboComfyUIBackend(make_config(comfy_dir, server.port))
        plain.validate_inputs(make_request())
        plain.prepare()  # fine: the default workflow does not need it

        scaled = WanTurboComfyUIBackend(make_config(comfy_dir, server.port))
        scaled.validate_inputs(make_request(backend_parameters={"latent_scale": 0.8}))
        with pytest.raises(BackendNotInstalledError, match="LatentMultiply"):
            scaled.prepare()
    finally:
        server.stop()


def test_server_that_does_not_list_the_model(comfy_dir, fake):
    config = make_config(comfy_dir, fake.port, models={**MODELS, "vae": "another_vae.safetensors"}, sha256={})
    (comfy_dir / "models" / "vae" / "another_vae.safetensors").write_bytes(b"x")
    with pytest.raises(BackendNotInstalledError, match="does not list 'another_vae.safetensors'"):
        WanTurboComfyUIBackend(config).prepare()


def test_incompatible_node_inputs(comfy_dir, fake, monkeypatch):
    import fake_comfyui

    changed = {**fake_comfyui.NODES["VAEDecodeTiled"]["required"]}
    del changed["temporal_size"]
    monkeypatch.setitem(fake_comfyui.NODES, "VAEDecodeTiled", {"required": changed})
    with pytest.raises(BackendNotInstalledError, match="unknown inputs \\['temporal_size'\\]"):
        WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)).prepare()


# --- lifecycle: start, reuse, stop ---

def test_prepare_reuses_a_running_server_and_does_not_stop_it(comfy_dir, fake):
    with WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)) as backend:
        backend.prepare()
        assert backend._server.started_by_us is False and backend._server.process is None
        assert backend._environment["comfyui_server_reused"] is True
    assert ComfyUIServer(make_config(comfy_dir, fake.port)).client.is_up()  # still running


def test_prepare_starts_comfyui_and_close_stops_it(comfy_dir):
    port = free_port()
    backend = WanTurboComfyUIBackend(make_config(comfy_dir, port))
    probe = ComfyUIServer(make_config(comfy_dir, port)).client
    assert not probe.is_up()

    backend.prepare()
    try:
        assert probe.is_up()
        server = backend._server
        assert server.started_by_us and server.process.poll() is None
        assert "--listen" in server.process.args and "127.0.0.1" in server.process.args
        assert backend._environment["comfyui_server_reused"] is False
        assert backend._environment["comfyui_torch_version"] == "2.11.0+cu128"
        assert backend._environment["comfyui_torch_cuda_build"] == "cu128"
        assert backend._environment["model_sha256"] == MODEL_SHA
        assert backend.metadata().model_version == MODEL_SHA["unet"]
        process = server.process
    finally:
        backend.close()
    assert process.poll() is not None and not probe.is_up()
    backend.close()  # idempotent


def test_context_manager_stops_the_server_it_started(comfy_dir):
    port = free_port()
    with WanTurboComfyUIBackend(make_config(comfy_dir, port)) as backend:
        backend.prepare()
        process = backend._server.process
    assert process.poll() is not None


def test_comfyui_that_exits_on_start_is_not_installed(comfy_dir):
    config = make_config(comfy_dir, free_port(), extra_args=("--exit-immediately",))
    with pytest.raises(BackendNotInstalledError, match="exited with code") as exc_info:
        WanTurboComfyUIBackend(config).prepare()
    assert "refusing to start" in str(exc_info.value)  # the server's own output is shown


def test_started_server_missing_a_node_is_stopped_again(comfy_dir):
    port = free_port()
    config = make_config(comfy_dir, port, extra_args=("--missing-node", "UnetLoaderGGUF"))
    with pytest.raises(BackendNotInstalledError, match="ComfyUI-GGUF custom node"):
        WanTurboComfyUIBackend(config).prepare()
    assert not ComfyUIServer(config).client.is_up()  # not left running


def test_missing_python_or_main(comfy_dir):
    with pytest.raises(BackendNotInstalledError, match="python not found"):
        WanTurboComfyUIBackend(make_config(comfy_dir, free_port(), python=comfy_dir / "no-python")).prepare()
    (comfy_dir / "main.py").unlink()
    with pytest.raises(BackendNotInstalledError, match="main.py does not exist"):
        WanTurboComfyUIBackend(make_config(comfy_dir, free_port())).prepare()


# --- system memory helper ---

def test_available_ram_is_readable_on_this_platform():
    available, total = available_ram_gb(), total_ram_gb()
    assert available is not None and total is not None and 0 < available <= total
    with LowestAvailableRam(interval_seconds=0.01) as ram:
        pass
    assert 0 < ram.lowest_gb <= total


# --- end to end through the real BenchmarkRunner ---

DEFINITION = f"""\
benchmark_id: comic_rooftop_like
task_type: video_generation
status: ready
start_frame: {FRAME}
prompt: Graphic novel style animated shot. A young man sits on a rooftop ledge at late sunset.
negative_prompt: photorealistic, blurry
duration_seconds: 5
fps: 24
duration_tolerance_seconds: 0.1
camera_motion: slow cinematic push-in
"""


@pytest.fixture
def rooftop(project):
    frame = project / FRAME
    frame.parent.mkdir(parents=True, exist_ok=True)
    gradient = np.linspace(0, 255, 1672, dtype=np.uint8)
    Image.fromarray(np.stack([np.tile(gradient, (941, 1))] * 3, axis=-1), "RGB").save(frame)
    path = project / "benchmarks" / "definitions" / "comic_rooftop_like.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFINITION, encoding="utf-8")
    return load_definition(path), frame


def make_runner(project):
    return BenchmarkRunner(  # default media_probe: the real ffprobe
        BenchmarkLogger(project / "benchmarks" / "results"), "outputs",
        environment_collector=lambda: dict(FAKE_ENVIRONMENT), root=project,
    )


@needs_ffmpeg
def test_end_to_end_success_record(project, rooftop, comfy_dir, fake):
    definition, frame = rooftop
    before = sha256_file(frame)
    runner = make_runner(project)

    with WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)) as backend:
        record = runner.run_definition(definition, backend, seed=42, output_resolution=LANDSCAPE, run_id="run-1")

    assert record.status == "success"
    assert record.backend == "wan-ti2v-5b-turbo-comfyui"
    assert record.model == "Wan2.2-TI2V-5B-Turbo (quanhaol) GGUF Q8_0"
    assert record.model_version == MODEL_SHA["unet"]
    assert record.workflow_version == "i2v-turbo-tiled-v1"
    assert record.output_video == "outputs/comic_rooftop_like/run-1/out.mp4"

    # measured by the real ffprobe
    assert record.output_media["frame_count"] == 121 and record.frame_count == 121
    assert record.output_media["fps"] == 24 and record.fps == 24
    assert (record.output_media["width"], record.output_media["height"]) == (1280, 704)
    assert record.output_media["codec"] == "h264"

    assert record.start_frame["preprocessing"] == [
        "resize 1672x941 -> 1280x720 (lanczos, scale to cover)",
        "center-crop 1280x720 -> 1280x704 (8 px top, 8 px bottom)",
    ]
    assert (record.start_frame["effective_width"], record.start_frame["effective_height"]) == (1280, 704)
    assert record.start_frame["derived_file"] == "outputs/comic_rooftop_like/run-1/start_frame_1280x704.png"
    assert (project / record.start_frame["derived_file"]).is_file()
    assert sha256_file(frame) == before

    assert record.inference_steps == 4 and record.guidance == 1.0
    parameters = record.model_parameters
    assert parameters["temporal_tile"] == 128 and parameters["spatial_tile"] == 512
    assert (parameters["sampler"], parameters["scheduler"], parameters["shift"]) == ("uni_pc", "simple", 8.0)
    assert parameters["latent_scale"] == 1.0 and parameters["cfg"] == 1.0 and parameters["num_frames"] == 121
    assert (parameters["unet"], parameters["text_encoder"], parameters["vae"]) == (UNET, TEXT_ENCODER, VAE)
    assert parameters["encoder"]["crf"] == 12 and parameters["vae_decode"] == "VAEDecodeTiled"

    environment = record.environment
    assert environment["model_sha256"] == MODEL_SHA
    assert environment["comfyui_torch_version"] == "2.11.0+cu128"
    assert environment["comfyui_torch_cuda_build"] == "cu128"
    assert "comfyui_commit" in environment and "comfyui_gguf_commit" in environment
    assert environment["ram_available_before_gb"] > 0
    assert 0 < environment["ram_available_min_gb"] <= environment["ram_total_gb"]
    assert environment["comfyui_prompt_seconds"] >= 0

    assert "negative_prompt has no effect: this model runs at CFG 1" in record.notes
    assert "camera_motion not consumed separately" in record.notes
    assert "reused a ComfyUI that was already running" in record.notes

    # what ComfyUI was asked to do
    (queued,) = fake.prompts.values()
    sampler = next(n["inputs"] for n in queued.values() if n["class_type"] == "KSampler")
    assert (sampler["seed"], sampler["steps"], sampler["cfg"]) == (42, 4, 1.0)
    decode = next(n["inputs"] for n in queued.values() if n["class_type"] == "VAEDecodeTiled")
    assert decode["temporal_size"] == 128
    positive = queued["5"]["inputs"]["text"]
    assert positive == definition.prompt
    assert len(fake.uploads) == 1 and fake.uploads[0].startswith("ai_studio_") and fake.views >= 121


def test_low_ram_adds_a_warning_but_does_not_refuse(project, rooftop, comfy_dir, fake, monkeypatch):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")
    definition, _ = rooftop
    fake.frames = 121
    monkeypatch.setattr(turbo, "available_ram_gb", lambda: 3.3)
    with WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)) as backend:
        record = make_runner(project).run_definition(definition, backend, seed=1, output_resolution=LANDSCAPE)
    assert record.status == "success"
    assert "only 3.3 GB RAM was available before loading (below 8 GB)" in record.notes
    assert record.environment["ram_available_before_gb"] == 3.3


def test_missing_node_through_the_runner_records_nothing(project, rooftop, comfy_dir):
    definition, _ = rooftop
    server = FakeComfyUI(missing_nodes=["UnetLoaderGGUF"]).start()
    runner = make_runner(project)
    try:
        with pytest.raises(BackendNotInstalledError, match="UnetLoaderGGUF"):
            runner.run_definition(
                definition, WanTurboComfyUIBackend(make_config(comfy_dir, server.port)),
                seed=1, output_resolution=LANDSCAPE,
            )
    finally:
        server.stop()
    assert runner.logger.list_runs("comic_rooftop_like") == []
    assert server.prompts == {}


def test_timeout_is_a_recorded_failure(project, rooftop, comfy_dir):
    definition, _ = rooftop
    server = FakeComfyUI(never_finish=True).start()
    runner = make_runner(project)
    config = make_config(comfy_dir, server.port, generation_timeout_seconds=1.5)
    try:
        with pytest.raises(BenchmarkRunError, match="did not finish the prompt within 1.5 s") as exc_info:
            runner.run_definition(definition, WanTurboComfyUIBackend(config), seed=1,
                                  output_resolution=LANDSCAPE, run_id="slow")
    finally:
        server.stop()
    record = exc_info.value.record
    assert exc_info.value.stage == "generation"
    assert record.status == "failed" and record.failure_stage == "generation"
    assert record.error_type == "TimeoutError"
    assert record.backend == "wan-ti2v-5b-turbo-comfyui"
    assert server.interrupted == 1  # ComfyUI was told to stop
    assert runner.logger.load("comic_rooftop_like", "slow") == record


def test_comfyui_error_is_a_recorded_failure(project, rooftop, comfy_dir):
    definition, _ = rooftop
    server = FakeComfyUI(fail_with="CUDA out of memory").start()
    try:
        with pytest.raises(BenchmarkRunError, match="CUDA out of memory") as exc_info:
            make_runner(project).run_definition(
                definition, WanTurboComfyUIBackend(make_config(comfy_dir, server.port)),
                seed=1, output_resolution=LANDSCAPE,
            )
    finally:
        server.stop()
    assert exc_info.value.record.failure_stage == "generation"
    assert "torch.OutOfMemoryError" in exc_info.value.record.error_message


def test_unavailable_sampler_is_a_recorded_failure(project, rooftop, comfy_dir, fake):
    definition, _ = rooftop
    with pytest.raises(BenchmarkRunError, match="sampler 'dpm_fast' is not available") as exc_info:
        make_runner(project).run_definition(
            definition, WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)), seed=1,
            output_resolution=LANDSCAPE, backend_parameters={"sampler": "dpm_fast"},
        )
    assert exc_info.value.record.failure_stage == "generation" and fake.prompts == {}


@needs_ffmpeg
def test_wrong_frame_count_from_comfyui_fails_verification(project, rooftop, comfy_dir):
    definition, _ = rooftop
    server = FakeComfyUI(frames=97).start()
    try:
        with pytest.raises(BenchmarkRunError, match="97 frames") as exc_info:
            make_runner(project).run_definition(
                definition, WanTurboComfyUIBackend(make_config(comfy_dir, server.port)),
                seed=1, output_resolution=LANDSCAPE,
            )
    finally:
        server.stop()
    assert exc_info.value.stage == "verification"
    assert "asked ComfyUI for 121 frames, it returned 97" in exc_info.value.record.notes


# --- CLI: lifecycle and automatic metrics ---

@needs_ffmpeg
def test_cli_run_stores_metrics_and_stops_the_server(project, rooftop, comfy_dir, monkeypatch, capsys):
    from ai_studio.cli import main

    port = free_port()
    created = []

    def factory():
        created.append(WanTurboComfyUIBackend(make_config(comfy_dir, port)))
        return created[-1]

    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    monkeypatch.setitem(registry.BACKENDS, "wan-ti2v-5b-turbo-comfyui", factory)
    code = main(["--benchmark", "comic_rooftop_like", "--backend", "wan-ti2v-5b-turbo-comfyui",
                 "--seed", "42", "--width", "1280", "--height", "704", "--run-id", "cli-1"])
    out = capsys.readouterr().out

    assert code == 0, out
    assert "status: success" in out and "measured: 24 fps, 121 frames" in out
    assert "metrics: benchmarks/metrics/comic_rooftop_like/cli-1/metrics_v1.json" in out
    metrics = json.loads(
        (project / "benchmarks/metrics/comic_rooftop_like/cli-1/metrics_v1.json").read_text(encoding="utf-8")
    )
    assert metrics["metrics"]["frame_count"] == 121
    assert metrics["metrics"]["start_frame_psnr"] is not None  # measured against the derived start frame
    assert metrics["analysis_size"] == {"width": 320, "height": 176}

    # the CLI closed the backend: the ComfyUI it started is gone
    assert created[0]._server is None
    assert not ComfyUIServer(make_config(comfy_dir, port)).client.is_up()
    # the server's log was kept with the run
    assert "fake ComfyUI listening" in (project / "outputs/comic_rooftop_like/cli-1/comfyui.log").read_text()


@needs_ffmpeg
def test_cli_no_metrics_flag(project, rooftop, comfy_dir, fake, monkeypatch, capsys):
    from ai_studio.cli import main

    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    monkeypatch.setitem(
        registry.BACKENDS, "wan-ti2v-5b-turbo-comfyui",
        lambda: WanTurboComfyUIBackend(make_config(comfy_dir, fake.port)),
    )
    code = main(["--benchmark", "comic_rooftop_like", "--backend", "wan-ti2v-5b-turbo-comfyui",
                 "--seed", "42", "--width", "1280", "--height", "704", "--run-id", "cli-2", "--no-metrics"])
    assert code == 0 and "metrics" not in capsys.readouterr().out
    assert not (project / "benchmarks" / "metrics").exists()


def test_cli_without_configuration_exits_2(project, rooftop, monkeypatch, capsys, tmp_path):
    from ai_studio.cli import main

    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(project))
    monkeypatch.setenv(comfyui_module.CONFIG_ENV_VAR, str(tmp_path / "missing.yaml"))
    code = main(["--benchmark", "comic_rooftop_like", "--backend", "wan-ti2v-5b-turbo-comfyui",
                 "--seed", "42", "--width", "1280", "--height", "704"])
    out = capsys.readouterr().out
    assert code == 2 and "ComfyUI is not configured" in out and "comfyui.example.yaml" in out
    assert not (project / "benchmarks" / "results").exists()
