"""Wan 2.2 TI2V-5B adapter. No torch, diffusers or weights needed: the
pipeline is injected."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_studio.benchmarking import (
    BenchmarkLogger,
    BenchmarkRunError,
    BenchmarkRunner,
    Resolution,
    load_definition,
)
from ai_studio.benchmarking.runner import sha256_file
from ai_studio.generation.backends import (
    BackendNotInstalledError,
    VideoGenerationBackend,
    WanTI2V5BBackend,
)
from ai_studio.generation.backends import wan_ti2v
from ai_studio.generation.backends.wan_ti2v import (
    WAN_TI2V_5B_REVISION,
    encode_frames,
    frame_count_for,
    frame_to_uint8,
    prepare_start_frame,
    preprocess_start_frame,
)
from ai_studio.generation.capabilities import Capability
from ai_studio.generation.request import VideoGenerationRequest
from conftest import FAKE_ENVIRONMENT, fake_probe

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

LANDSCAPE = Resolution(width=1280, height=704)
PORTRAIT = Resolution(width=704, height=1280)
FRAME = "benchmarks/references/comic_hero_01/rooftop_frame.png"
HEAVY_MODULES = ("torch", "diffusers", "transformers", "huggingface_hub")

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def make_request(**overrides):
    fields = dict(
        prompt="a young man sits on a rooftop ledge at sunset, graphic novel style",
        negative_prompt="photorealistic",
        start_frame="frame.png",
        duration_seconds=5,
        fps=24,
        output_resolution=LANDSCAPE,
        seed=7,
    )
    fields.update(overrides)
    return VideoGenerationRequest(**fields)


class FakePipeline:
    """Stands in for the diffusers pipeline: returns `frame_count` frames
    (None: as many as asked for) of blocky noise, float32 in [0, 1]."""

    def __init__(self, frame_count=None):
        self.frame_count = frame_count
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        rng = np.random.default_rng(0)
        height, width = kwargs["height"], kwargs["width"]
        # A few distinct frames, repeated: keeps memory and encode time small.
        distinct = [
            np.kron(rng.random((height // 32, width // 32, 3), dtype=np.float32),
                    np.ones((32, 32, 1), dtype=np.float32))
            for _ in range(4)
        ]
        count = self.frame_count or kwargs["num_frames"]
        return SimpleNamespace(frames=[[distinct[i % 4] for i in range(count)]])


def write_start_frame(path, size=(1672, 941), mode="RGB"):
    path.parent.mkdir(parents=True, exist_ok=True)
    gradient = np.linspace(0, 255, size[0], dtype=np.uint8)
    pixels = np.stack([np.tile(gradient, (size[1], 1))] * 3, axis=-1)
    Image.fromarray(pixels, "RGB").convert(mode).save(path)
    return path


# --- frame count ---

@pytest.mark.parametrize(
    "duration, expected",
    [(5, 121), (5.0417, 121), (4, 97), (3, 73), (2, 49), (1, 25), (0.5, 13), (4.9, 117), (0.1, 5)],
)
def test_frame_count_is_nearest_4n_plus_1(duration, expected):
    frames = frame_count_for(duration)
    assert frames == expected and frames % 4 == 1


def test_five_seconds_is_inside_the_benchmark_tolerance():
    assert abs(frame_count_for(5) / 24 - 5) < 0.1


def test_frame_count_rounds_half_up_not_to_even():
    # (frames_wanted - 1) / 4 == 2.5 exactly -> n = 3
    assert frame_count_for(11 / 24) == 13


# --- contract ---

def test_declares_generation_with_start_frame_support():
    assert issubclass(WanTI2V5BBackend, VideoGenerationBackend)
    assert WanTI2V5BBackend.capabilities == {Capability.VIDEO_GENERATION}
    assert WanTI2V5BBackend.supports_start_frame is True


def test_module_import_pulls_in_no_heavy_dependency():
    code = (
        "import sys\n"
        "import ai_studio.generation.backends.wan_ti2v\n"
        "bad = [m for m in ('torch', 'diffusers', 'transformers', 'huggingface_hub', 'PIL', 'numpy')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": str(src)})


def test_metadata_works_without_torch(monkeypatch):
    for name in HEAVY_MODULES:
        monkeypatch.setitem(sys.modules, name, None)  # importing it raises ImportError
    meta = WanTI2V5BBackend().metadata()
    assert meta.backend_name == "wan-ti2v-5b"
    assert meta.backend_version == "upstream-diffusers-0.1"
    assert meta.model_name == "Wan2.2-TI2V-5B"
    assert meta.model_version == WAN_TI2V_5B_REVISION == "b8fff7315c768468a5333511427288870b2e9635"
    assert meta.workflow_version == "i2v-v1"


def test_metadata_reports_pinned_revision():
    assert WanTI2V5BBackend(revision="abc123").metadata().model_version == "abc123"
    assert WanTI2V5BBackend(revision=None).metadata().model_version == "unpinned"


@pytest.mark.parametrize("missing", ["torch", "diffusers"])
def test_prepare_without_dependencies_raises_not_installed(monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(BackendNotInstalledError, match=r'pip install -e "\.\[wan\]"'):
        WanTI2V5BBackend().prepare()


def test_prepare_without_cuda_raises_not_installed(monkeypatch):
    def module(name, **attrs):
        fake = type(sys)(name)
        vars(fake).update(attrs)
        return fake

    monkeypatch.setitem(sys.modules, "torch", module(
        "torch", cuda=SimpleNamespace(is_available=lambda: False),
    ))
    monkeypatch.setitem(sys.modules, "diffusers", module(
        "diffusers", AutoencoderKLWan=object, WanImageToVideoPipeline=object,
    ))
    monkeypatch.setitem(sys.modules, "transformers", module("transformers"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", module("huggingface_hub", snapshot_download=None))
    with pytest.raises(BackendNotInstalledError, match="CUDA"):
        WanTI2V5BBackend().prepare()


def test_injected_pipeline_skips_torch_entirely(monkeypatch):
    for name in HEAVY_MODULES:
        monkeypatch.setitem(sys.modules, name, None)
    pipeline = FakePipeline()
    backend = WanTI2V5BBackend(pipeline_factory=lambda: pipeline)
    backend.prepare()
    assert backend._pipe is pipeline


def test_generate_before_prepare_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="prepare"):
        WanTI2V5BBackend().generate(make_request(), tmp_path)


# --- validation ---

def test_valid_request_accepted():
    WanTI2V5BBackend().validate_inputs(make_request())
    WanTI2V5BBackend().validate_inputs(make_request(fps=None, output_resolution=PORTRAIT))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"start_frame": None}, "start_frame is required"),
        ({"fps": 16}, "fps must be 24"),
        ({"fps": 23.976}, "fps must be 24"),
        ({"fps": 30}, "fps must be 24"),
        ({"output_resolution": Resolution(width=1280, height=720)}, "output_resolution must be"),
        ({"output_resolution": Resolution(width=704, height=704)}, "output_resolution must be"),
        ({"output_resolution": Resolution(width=832, height=480)}, "output_resolution must be"),
        ({"duration_seconds": 5.2}, "125 frames"),
        ({"duration_seconds": 10}, "not verified"),
        ({"backend_parameters": {"steps": 30}}, "unknown backend_parameters"),
        ({"backend_parameters": {"num_frames": 81}}, "unknown backend_parameters"),
        ({"backend_parameters": {"num_inference_steps": 0}}, "num_inference_steps"),
        ({"backend_parameters": {"num_inference_steps": 30.5}}, "num_inference_steps"),
        ({"backend_parameters": {"guidance_scale": "5"}}, "guidance_scale"),
        ({"backend_parameters": {"guidance_scale": -1}}, "guidance_scale"),
        ({"backend_parameters": {"revision": ""}}, "revision"),
        ({"backend_parameters": {"cpu_offload": "yes"}}, "cpu_offload"),
        ({"backend_parameters": {"vae_tiling": 1}}, "vae_tiling"),
    ],
)
def test_invalid_requests_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        WanTI2V5BBackend().validate_inputs(make_request(**overrides))


def test_all_allowed_backend_parameters_accepted():
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request(backend_parameters={
        "num_inference_steps": 30, "guidance_scale": 4, "revision": "abc123", "cpu_offload": True,
        "vae_tiling": True,
    }))
    assert backend.metadata().model_version == "abc123"


# --- vae_tiling ---

class TilingPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.tiling_calls = 0
        self.vae = SimpleNamespace(enable_tiling=self._enable_tiling)

    def _enable_tiling(self):
        self.tiling_calls += 1


@pytest.mark.parametrize("requested, expected_calls", [({"vae_tiling": True}, 1), ({"vae_tiling": False}, 0), ({}, 0)])
def test_vae_tiling_is_enabled_only_on_request(requested, expected_calls):
    pipeline = TilingPipeline()
    backend = WanTI2V5BBackend(pipeline_factory=lambda: pipeline)
    backend.validate_inputs(make_request(backend_parameters=requested))
    backend.prepare()
    backend.prepare()  # same settings: not loaded or tiled twice
    assert pipeline.tiling_calls == expected_calls


def test_changing_vae_tiling_reloads_the_pipeline():
    created = []

    def factory():
        created.append(TilingPipeline())
        return created[-1]

    backend = WanTI2V5BBackend(pipeline_factory=factory)
    backend.validate_inputs(make_request())
    backend.prepare()
    backend.validate_inputs(make_request(backend_parameters={"vae_tiling": True}))
    backend.prepare()
    assert [p.tiling_calls for p in created] == [0, 1]


# --- loading the real pipeline (torch / diffusers / huggingface_hub mocked) ---

class FakeLoadedPipeline:
    def __init__(self, expand_timesteps=True):
        self.config = SimpleNamespace(expand_timesteps=expand_timesteps)
        self.events = []
        self.vae = SimpleNamespace(enable_tiling=lambda: self.events.append("vae.enable_tiling"))

    def enable_model_cpu_offload(self):
        self.events.append("enable_model_cpu_offload")

    def to(self, device):
        self.events.append(f"to({device})")


@pytest.fixture
def fake_stack(monkeypatch, tmp_path):
    """Stand-ins for torch, diffusers, transformers and huggingface_hub that
    record how the adapter loads the model."""
    log = SimpleNamespace(snapshot=[], vae=[], pipeline=[], pipe=FakeLoadedPipeline())
    snapshot_dir = tmp_path / "hub" / "snapshots" / "0123abc"
    snapshot_dir.mkdir(parents=True)

    def module(name, **attrs):
        fake = type(sys)(name)
        vars(fake).update(attrs)
        return fake

    def snapshot_download(repo, **kwargs):
        log.snapshot.append((repo, kwargs))
        return str(snapshot_dir)

    class AutoencoderKLWan:
        @staticmethod
        def from_pretrained(path, **kwargs):
            log.vae.append((Path(path), kwargs))
            return "the-vae"

    class WanImageToVideoPipeline:
        @staticmethod
        def from_pretrained(path, **kwargs):
            log.pipeline.append((Path(path), kwargs))
            return log.pipe

    monkeypatch.setitem(sys.modules, "torch", module(
        "torch", __version__="2.8.0+cu126", float32="float32", bfloat16="bfloat16",  # noqa: E501
        cuda=SimpleNamespace(is_available=lambda: True), version=SimpleNamespace(cuda="12.6"),
        Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: ("generator", device, seed)),
    ))
    monkeypatch.setitem(sys.modules, "diffusers", module(
        "diffusers", __version__="0.36.0",
        AutoencoderKLWan=AutoencoderKLWan, WanImageToVideoPipeline=WanImageToVideoPipeline,
    ))
    monkeypatch.setitem(sys.modules, "transformers", module("transformers", __version__="4.56.2"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", module("huggingface_hub", snapshot_download=snapshot_download))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    log.snapshot_dir = snapshot_dir
    return log


def test_prepare_loads_the_pinned_revision_as_documented(fake_stack):
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request())
    backend.prepare()

    assert fake_stack.snapshot == [(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", {"revision": WAN_TI2V_5B_REVISION, "local_files_only": False},
    )]
    assert fake_stack.vae == [(fake_stack.snapshot_dir, {"subfolder": "vae", "torch_dtype": "float32"})]
    assert fake_stack.pipeline == [(fake_stack.snapshot_dir, {"vae": "the-vae", "torch_dtype": "bfloat16"})]
    assert fake_stack.pipe.events == ["to(cuda)"]
    assert backend.metadata().model_version == "0123abc"  # the resolved snapshot commit
    assert backend._make_generator(7) == ("generator", "cuda", 7)
    assert backend._environment == {
        "pytorch_version": "2.8.0+cu126", "cuda_version": "12.6",  # the torch that ran the model
        "torch_version": "2.8.0+cu126", "diffusers_version": "0.36.0",
        "transformers_version": "4.56.2", "torch_cuda_version": "12.6",
    }


def test_prepare_with_cpu_offload_and_vae_tiling(fake_stack):
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request(backend_parameters={"cpu_offload": True, "vae_tiling": True}))
    backend.prepare()
    assert fake_stack.pipe.events == ["vae.enable_tiling", "enable_model_cpu_offload"]


@pytest.mark.parametrize("value, offline", [("1", True), ("true", True), ("0", False), ("", False)])
def test_hf_hub_offline_resolves_from_the_local_cache_only(fake_stack, monkeypatch, value, offline):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request())
    backend.prepare()
    assert fake_stack.snapshot[0][1]["local_files_only"] is offline


def test_offline_with_missing_weights_explains_itself(fake_stack, monkeypatch):
    def not_cached(repo, **kwargs):
        raise OSError("snapshot folder not found")

    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", not_cached)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="HF_HUB_OFFLINE is set"):
        WanTI2V5BBackend().prepare()


@pytest.mark.parametrize("version", ["0.35.2", "0.35.0", "0.35.0.dev0"])
def test_vae_tiling_on_old_diffusers_is_refused_before_loading(fake_stack, monkeypatch, version):
    monkeypatch.setattr(sys.modules["diffusers"], "__version__", version)
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request(backend_parameters={"vae_tiling": True}))
    with pytest.raises(BackendNotInstalledError, match="diffusers <0.36 tiles the Wan 2.2 VAE without patch_size; upgrade"):
        backend.prepare()
    assert fake_stack.snapshot == [] and fake_stack.pipeline == []  # nothing was loaded


@pytest.mark.parametrize("version", ["0.36.0", "0.36.0.dev0", "0.37.1", "1.0.0"])
def test_vae_tiling_on_new_enough_diffusers_loads(fake_stack, monkeypatch, version):
    monkeypatch.setattr(sys.modules["diffusers"], "__version__", version)
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request(backend_parameters={"vae_tiling": True}))
    backend.prepare()
    assert fake_stack.pipe.events == ["vae.enable_tiling", "to(cuda)"]


def test_old_diffusers_without_tiling_still_loads(fake_stack, monkeypatch):
    monkeypatch.setattr(sys.modules["diffusers"], "__version__", "0.35.2")
    backend = WanTI2V5BBackend()
    backend.validate_inputs(make_request())
    backend.prepare()
    assert fake_stack.pipe.events == ["to(cuda)"]


def test_version_tuple():
    assert wan_ti2v.version_tuple("0.35.2") == (0, 35, 2)
    assert wan_ti2v.version_tuple("0.36.0.dev0") == (0, 36, 0)
    assert wan_ti2v.version_tuple("0.36.0") >= wan_ti2v.MIN_DIFFUSERS_FOR_VAE_TILING > wan_ti2v.version_tuple("0.35.2")


def test_wrong_weights_are_rejected(fake_stack):
    fake_stack.pipe.config.expand_timesteps = False
    with pytest.raises(RuntimeError, match="wrong weights"):
        WanTI2V5BBackend().prepare()


# --- start-frame preprocessing ---

def test_canonical_size_is_scaled_to_cover_then_center_cropped(tmp_path):
    source = write_start_frame(tmp_path / "frame.png")
    with Image.open(source) as original:
        image, steps = preprocess_start_frame(original, 1280, 704)
    assert image.size == (1280, 704) and image.mode == "RGB"
    assert steps == [
        "resize 1672x941 -> 1280x720 (lanczos, scale to cover)",
        "center-crop 1280x720 -> 1280x704 (8 px top, 8 px bottom)",
    ]


def test_prepare_start_frame_writes_derived_copy_and_leaves_original_alone(tmp_path):
    source = write_start_frame(tmp_path / "in" / "frame.png")
    before = sha256_file(source)
    output_dir = tmp_path / "run"

    image, conditioning = prepare_start_frame(source, output_dir, 1280, 704)

    assert sha256_file(source) == before
    assert conditioning.preprocessing == [
        "resize 1672x941 -> 1280x720 (lanczos, scale to cover)",
        "center-crop 1280x720 -> 1280x704 (8 px top, 8 px bottom)",
    ]
    assert (conditioning.effective_width, conditioning.effective_height) == (1280, 704)
    derived = Path(conditioning.derived_file)
    assert derived == output_dir / "start_frame_1280x704.png"
    with Image.open(derived) as saved:
        assert saved.size == (1280, 704) and saved.format == "PNG"
        assert saved.tobytes() == image.tobytes()


def test_portrait_target_crops_the_sides(tmp_path):
    source = write_start_frame(tmp_path / "frame.png")
    with Image.open(source) as original:
        image, steps = preprocess_start_frame(original, 704, 1280)
    assert image.size == (704, 1280)
    assert steps == [
        "resize 1672x941 -> 2274x1280 (lanczos, scale to cover)",
        "center-crop 2274x1280 -> 704x1280 (785 px left, 785 px right)",
    ]


def test_crop_is_centered(tmp_path):
    # Left half black, right half white, twice as wide as needed: the centre
    # crop keeps the boundary in the middle.
    pixels = np.zeros((64, 256, 3), dtype=np.uint8)
    pixels[:, 128:] = 255
    image, steps = preprocess_start_frame(Image.fromarray(pixels), 128, 64)
    result = np.asarray(image)
    assert steps == ["center-crop 256x64 -> 128x64 (64 px left, 64 px right)"]
    assert result[:, :64].max() == 0 and result[:, 64:].min() == 255


def test_exact_size_needs_no_preprocessing(tmp_path):
    source = write_start_frame(tmp_path / "frame.png", size=(1280, 704))
    with Image.open(source) as original:
        image, steps = preprocess_start_frame(original, 1280, 704)
        assert steps == [] and image.tobytes() == original.tobytes()


def test_non_rgb_image_is_converted_and_reported(tmp_path):
    source = write_start_frame(tmp_path / "frame.png", size=(1280, 704), mode="RGBA")
    with Image.open(source) as original:
        image, steps = preprocess_start_frame(original, 1280, 704)
    assert image.mode == "RGB" and steps == ["convert RGBA -> RGB"]


# --- frame conversion and encoding ---

def test_float_frames_are_clipped_and_rounded():
    frame = np.array([[[-0.5, 0.0, 0.5], [1.0, 1.7, 0.999], [0.002, 0.25, float("nan")]]], dtype=np.float32)
    assert frame_to_uint8(frame).tolist() == [[[0, 0, 128], [255, 255, 255], [1, 64, 0]]]
    assert frame_to_uint8(frame).dtype == np.uint8


def test_uint8_frames_pass_through_unchanged():
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert frame_to_uint8(frame) is frame


@pytest.mark.parametrize("bad", [np.zeros((4, 4)), np.zeros((4, 4, 4)), np.zeros((4, 4, 3), dtype=np.int32)])
def test_invalid_frames_rejected(bad):
    with pytest.raises(ValueError):
        frame_to_uint8(bad)


def test_missing_ffmpeg_is_a_runtime_error(monkeypatch, tmp_path):
    monkeypatch.setattr(wan_ti2v.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg not found"):
        encode_frames([np.zeros((32, 32, 3), dtype=np.float32)], tmp_path / "out.mp4")


def test_no_frames_is_a_runtime_error(tmp_path):
    with pytest.raises(RuntimeError, match="no frames"):
        encode_frames([], tmp_path / "out.mp4")


@needs_ffmpeg
def test_encode_frames_writes_h264_yuv420p_at_24_fps(tmp_path):
    from ai_studio.media_probe import ffprobe_video

    frames = np.random.default_rng(1).random((9, 48, 64, 3), dtype=np.float32)
    output = tmp_path / "out.mp4"
    settings = encode_frames(frames, output)
    assert settings == {
        "codec": "libx264", "pix_fmt": "yuv420p", "crf": 12, "preset": "slow", "fps": 24, "audio": False,
    }
    info = ffprobe_video(output)
    assert (info.width, info.height, info.fps, info.frame_count, info.codec) == (64, 48, 24, 9, "h264")
    streams = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,pix_fmt", "-of", "csv=p=0", str(output)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert streams == ["video,yuv420p"]  # one stream: no audio


@needs_ffmpeg
def test_ffmpeg_failure_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        encode_frames([np.zeros((32, 32, 3), dtype=np.uint8)], tmp_path / "missing_dir" / "out.mp4")


# --- end to end through the real BenchmarkRunner ---

DEFINITION = f"""\
benchmark_id: comic_rooftop_like
task_type: video_generation
status: ready
start_frame: {FRAME}
prompt: >-
  Graphic novel style animated shot. A young man sits on a rooftop ledge at
  late sunset. Slow cinematic camera push-in.
negative_prompt: photorealistic, blurry
duration_seconds: 5
fps: 24
duration_tolerance_seconds: 0.1
camera_motion: slow cinematic push-in
subject_motion:
  - remains mostly stationary
visual_direction:
  - graphic novel aesthetic
"""


@pytest.fixture
def rooftop(project):
    """Project with a comic_rooftop_01-like definition and a 1672x941 start frame."""
    frame = write_start_frame(project / FRAME)
    path = project / "benchmarks" / "definitions" / "comic_rooftop_like.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFINITION, encoding="utf-8")
    return load_definition(path), frame


def make_runner(project, **kwargs):
    return BenchmarkRunner(
        BenchmarkLogger(project / "benchmarks" / "results"), "outputs",
        environment_collector=lambda: dict(FAKE_ENVIRONMENT), root=project, **kwargs,
    )


@needs_ffmpeg
def test_end_to_end_with_real_encoding_and_real_ffprobe(project, rooftop):
    definition, frame = rooftop
    before = sha256_file(frame)
    pipeline = FakePipeline()
    backend = WanTI2V5BBackend(pipeline_factory=lambda: pipeline)
    runner = make_runner(project)  # default media_probe: the real ffprobe

    record = runner.run_definition(
        definition, backend, seed=1234, output_resolution=LANDSCAPE, run_id="run-1",
        backend_parameters={"num_inference_steps": 30},
    )

    assert record.status == "success"
    assert record.backend == "wan-ti2v-5b" and record.model == "Wan2.2-TI2V-5B"
    assert record.workflow_version == "i2v-v1"
    assert record.output_video == "outputs/comic_rooftop_like/run-1/out.mp4"

    # Measured by ffprobe, not reported by the adapter.
    assert record.output_media["probe"].startswith("ffprobe ")
    assert record.output_media["frame_count"] == 121 and record.frame_count == 121
    assert record.output_media["fps"] == 24 and record.fps == 24
    assert (record.output_media["width"], record.output_media["height"]) == (1280, 704)
    assert record.output_media["codec"] == "h264"
    assert record.output_media["duration_seconds"] == 5.041667

    assert record.start_frame == {
        "path": FRAME,
        "sha256": before,
        "original_width": 1672,
        "original_height": 941,
        "preprocessing": [
            "resize 1672x941 -> 1280x720 (lanczos, scale to cover)",
            "center-crop 1280x720 -> 1280x704 (8 px top, 8 px bottom)",
        ],
        "effective_width": 1280,
        "effective_height": 704,
        "derived_file": "outputs/comic_rooftop_like/run-1/start_frame_1280x704.png",
    }
    run_dir = project / "outputs" / "comic_rooftop_like" / "run-1"
    assert (project / record.start_frame["derived_file"]).parent == run_dir
    assert sha256_file(frame) == before  # the original is never touched

    assert record.inference_steps == 30 and record.guidance == 5.0
    assert record.model_parameters == {
        "model_repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "revision": WAN_TI2V_5B_REVISION,
        "pipeline": "WanImageToVideoPipeline",
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "num_frames": 121,
        "width": 1280,
        "height": 704,
        "fps": 24,
        "cpu_offload": False,
        "vae_tiling": False,
        "torch_dtype": "bfloat16",
        "vae_dtype": "float32",
        "encoder": {
            "codec": "libx264", "pix_fmt": "yuv420p", "crf": 12, "preset": "slow", "fps": 24, "audio": False,
        },
    }
    assert "ffmpeg_version" in record.environment
    assert "camera_motion, subject_motion, visual_direction not consumed separately" in record.notes
    assert "measured resolution" not in record.notes

    # What the pipeline was actually asked for.
    (call,) = pipeline.calls
    assert call["prompt"] == definition.prompt and call["negative_prompt"] == definition.negative_prompt
    assert (call["width"], call["height"], call["num_frames"]) == (1280, 704, 121)
    assert call["num_inference_steps"] == 30 and call["guidance_scale"] == 5.0
    assert call["output_type"] == "np"
    assert call["image"].size == (1280, 704)


@needs_ffmpeg
def test_wrong_frame_count_fails_verification(project, rooftop):
    definition, _ = rooftop
    backend = WanTI2V5BBackend(pipeline_factory=lambda: FakePipeline(frame_count=97))
    runner = make_runner(project)

    with pytest.raises(BenchmarkRunError, match="97 frames") as exc_info:
        runner.run_definition(definition, backend, seed=1, output_resolution=LANDSCAPE, run_id="short")
    assert exc_info.value.stage == "verification"
    record = exc_info.value.record
    assert record.status == "failed" and record.failure_stage == "verification"
    assert "4.042s" in record.error_message
    assert record.output_media is None
    assert "asked the pipeline for 121 frames, it returned 97" in record.notes


def test_invalid_request_is_recorded_as_a_failed_run(project, rooftop):
    definition, _ = rooftop
    backend = WanTI2V5BBackend(pipeline_factory=FakePipeline)
    runner = make_runner(project, media_probe=fake_probe)

    with pytest.raises(BenchmarkRunError, match="output_resolution must be") as exc_info:
        runner.run_definition(
            definition, backend, seed=1, output_resolution=Resolution(width=1280, height=720), run_id="bad",
        )
    record = exc_info.value.record
    assert record.failure_stage == "validation" and record.error_type == "ValueError"
    assert record.backend == "wan-ti2v-5b"
    assert not (project / "outputs" / "comic_rooftop_like" / "bad").exists()


def test_not_installed_backend_records_nothing(project, rooftop, monkeypatch):
    definition, _ = rooftop
    monkeypatch.setitem(sys.modules, "diffusers", None)
    runner = make_runner(project, media_probe=fake_probe)
    with pytest.raises(BackendNotInstalledError):
        runner.run_definition(definition, WanTI2V5BBackend(), seed=1, output_resolution=LANDSCAPE)
    assert runner.logger.list_runs("comic_rooftop_like") == []
