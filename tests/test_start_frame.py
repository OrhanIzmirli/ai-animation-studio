"""Image-to-video benchmark inputs: pinned fps and an explicit start_frame."""

import dataclasses
import json
import struct
import zlib
from pathlib import Path

import pytest

from ai_studio.benchmarking import BenchmarkRunError, Resolution, load_definition
from ai_studio.benchmarking.runner import sha256_file
from ai_studio.generation.capabilities import UnsupportedCapabilityError
from ai_studio.generation.request import VideoGenerationRequest
from ai_studio.generation.result import StartFrameConditioning
from ai_studio.media import image_dimensions
from conftest import FakeBackend, FakeGenerationBackend

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
CANONICAL_SHA = "3a3e76750196317ee5142d7d6d00d2c68af58a993ee54ca8cc9fa665dbf1e0fb"
FRAME = "benchmarks/references/frame.png"
RES = Resolution(width=64, height=36)


def png_bytes(width, height):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + b"\x80\x20\x20" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class ImageToVideoBackend(FakeGenerationBackend):
    """Declares start-frame conditioning and reports how it used the frame."""

    supports_start_frame = True

    def __init__(self, *, write_derived=True, report=True, derived_name="start_frame_32x18.png", **kwargs):
        super().__init__(**kwargs)
        self.write_derived, self.report, self.derived_name = write_derived, report, derived_name

    def generate(self, request, output_dir):
        result = super().generate(request, output_dir)
        if not (request.start_frame and self.report):
            return result
        derived = None
        if self.write_derived:
            derived = Path(output_dir) / self.derived_name
            derived.parent.mkdir(parents=True, exist_ok=True)
            derived.write_bytes(png_bytes(32, 18))  # a derived copy, never the original
        conditioning = StartFrameConditioning(
            preprocessing=["resize 40x22 -> 32x18 (lanczos)"],
            effective_width=32, effective_height=18,
            derived_file=str(derived) if derived else None,
        )
        return dataclasses.replace(result, start_frame_conditioning=conditioning)


@pytest.fixture
def frame(project, input_files):  # after input_files, which creates benchmarks/references/
    path = project / FRAME
    path.write_bytes(png_bytes(40, 22))
    return path


def write_definition(project, extra=""):
    path = project / "benchmarks" / "definitions" / "i2v.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "benchmark_id: i2v_01\ntask_type: video_generation\nstatus: ready\n"
        "prompt: a rooftop at sunset\nduration_seconds: 5\n" + extra,
        encoding="utf-8",
    )
    return load_definition(path)


# --- fps: definition -> request -> record ---

def test_fps_survives_definition_request_record(runner, project, frame):
    definition = write_definition(project, f"fps: 24\nstart_frame: {FRAME}\n")
    assert definition.fps == 24
    request = definition.to_request(seed=1, output_resolution=RES)
    assert request.fps == 24

    record = runner.run_definition(definition, ImageToVideoBackend(), seed=1, output_resolution=RES, run_id="r1")
    data = json.loads(runner.logger.path_for("i2v_01", "r1").read_text(encoding="utf-8"))
    assert data["task_parameters"]["requested_fps"] == 24
    assert data["fps"] == 24  # the fake backend honours the requested fps
    assert record.task_parameters["requested_fps"] == 24


def test_pinned_fps_cannot_be_overridden_at_run_time(project):
    definition = write_definition(project, "fps: 24\n")
    with pytest.raises(ValueError, match="pinned fps"):
        definition.to_request(seed=1, output_resolution=RES, fps=30)
    assert definition.to_request(seed=1, output_resolution=RES, fps=24).fps == 24


def test_invalid_fps_rejected(project):
    for bad in ("0", "-24", "fast"):
        with pytest.raises(ValueError):
            write_definition(project, f"fps: {bad}\n")


# --- start_frame: definition -> request -> record ---

def test_start_frame_survives_definition_request_record(runner, project, frame):
    definition = write_definition(project, f"fps: 24\nstart_frame: {FRAME}\n")
    request = definition.to_request(seed=1, output_resolution=RES)
    assert request.start_frame == FRAME and request.reference_images == []

    record = runner.run_definition(definition, ImageToVideoBackend(), seed=1, output_resolution=RES, run_id="r1")
    saved = runner.logger.load("i2v_01", "r1")
    assert saved == record
    assert record.start_frame["path"] == FRAME
    assert record.start_frame["original_width"] == 40 and record.start_frame["original_height"] == 22


def test_start_frame_checksum_recorded(runner, project, frame):
    definition = write_definition(project, f"start_frame: {FRAME}\n")
    record = runner.run_definition(definition, ImageToVideoBackend(), seed=1, output_resolution=RES)
    assert record.start_frame["sha256"] == sha256_file(frame)
    assert record.input_checksums[FRAME] == sha256_file(frame)


def test_backend_receives_absolute_start_frame(runner, project, frame):
    backend = ImageToVideoBackend()
    runner.run_definition(write_definition(project, f"start_frame: {FRAME}\n"), backend, seed=1, output_resolution=RES)
    assert Path(backend.received.start_frame).is_absolute()


# --- start_frame and reference_images are separate ---

def test_start_frame_and_references_are_recorded_separately(runner, project, frame, input_files):
    _, reference = input_files  # benchmarks/references/ref.png
    definition = write_definition(
        project, f"start_frame: {FRAME}\nreference_images: [benchmarks/references/ref.png]\n"
    )
    backend = ImageToVideoBackend()
    record = runner.run_definition(definition, backend, seed=1, output_resolution=RES)

    assert record.reference_assets == ["benchmarks/references/ref.png"]
    assert record.start_frame["path"] == FRAME
    assert FRAME not in record.reference_assets
    assert Path(backend.received.start_frame).name == "frame.png"
    assert [Path(p).name for p in backend.received.reference_images] == ["ref.png"]


def test_same_image_cannot_be_start_frame_and_reference():
    with pytest.raises(ValueError, match="not interchangeable"):
        VideoGenerationRequest(
            prompt="p", duration_seconds=5, output_resolution=RES, seed=1,
            start_frame="a.png", reference_images=["a.png"],
        )


def test_definition_rejects_duplicated_start_frame(project):
    with pytest.raises(ValueError, match="reference_images"):
        write_definition(project, f"start_frame: {FRAME}\nreference_images: [{FRAME}]\n")


# --- no silent downgrade ---

def test_backend_without_start_frame_support_is_refused(runner, project, frame):
    definition = write_definition(project, f"start_frame: {FRAME}\n")
    backend = FakeGenerationBackend()  # generation-capable, but no start-frame conditioning
    assert not backend.supports_start_frame

    with pytest.raises(UnsupportedCapabilityError, match="start-frame"):
        runner.run_definition(definition, backend, seed=1, output_resolution=RES)
    assert backend.calls == []
    assert runner.logger.list_runs("i2v_01") == []


def test_backend_that_ignores_start_frame_fails_verification(runner, project, frame):
    definition = write_definition(project, f"start_frame: {FRAME}\n")
    backend = ImageToVideoBackend(report=False)  # claims support, silently does not use it

    with pytest.raises(BenchmarkRunError, match="verification") as exc_info:
        runner.run_definition(definition, backend, seed=1, output_resolution=RES)
    record = exc_info.value.record
    assert record.status == "failed" and "silent downgrade" in record.error_message
    assert record.start_frame["sha256"] == sha256_file(frame)  # still pinned in the failure record
    assert record.start_frame["preprocessing"] is None


def test_conditioning_reported_without_start_frame_fails(runner, project):
    class Confused(ImageToVideoBackend):
        def generate(self, request, output_dir):
            result = super().generate(request, output_dir)
            return dataclasses.replace(result, start_frame_conditioning=StartFrameConditioning(
                preprocessing=[], effective_width=8, effective_height=8,
            ))

    with pytest.raises(BenchmarkRunError, match="verification"):
        runner.run_definition(write_definition(project), Confused(), seed=1, output_resolution=RES)


def test_editing_backend_cannot_take_start_frame_requests(runner, project, frame):
    with pytest.raises(UnsupportedCapabilityError):
        runner.run_definition(write_definition(project, f"start_frame: {FRAME}\n"), FakeBackend(),
                              seed=1, output_resolution=RES)


# --- preprocessing metadata ---

def test_preprocessing_metadata_recorded(runner, project, frame):
    record = runner.run_definition(
        write_definition(project, f"start_frame: {FRAME}\n"), ImageToVideoBackend(),
        seed=1, output_resolution=RES, run_id="r1",
    )
    assert record.start_frame == {
        "path": FRAME,
        "sha256": sha256_file(frame),
        "original_width": 40,
        "original_height": 22,
        "preprocessing": ["resize 40x22 -> 32x18 (lanczos)"],
        "effective_width": 32,
        "effective_height": 18,
        "derived_file": "outputs/i2v_01/r1/start_frame_32x18.png",
    }
    assert (project / record.start_frame["derived_file"]).is_file()


def test_no_preprocessing_is_recorded_as_empty_list(runner, project, frame):
    class AsIs(ImageToVideoBackend):
        def generate(self, request, output_dir):
            result = super().generate(request, output_dir)
            return dataclasses.replace(result, start_frame_conditioning=StartFrameConditioning(
                preprocessing=[], effective_width=40, effective_height=22,
            ))

    record = runner.run_definition(write_definition(project, f"start_frame: {FRAME}\n"), AsIs(write_derived=False),
                                   seed=1, output_resolution=RES)
    assert record.start_frame["preprocessing"] == []
    assert record.start_frame["derived_file"] is None


def test_derived_file_outside_run_dir_rejected(runner, project, frame):
    backend = ImageToVideoBackend(derived_name="../../stray.png")
    with pytest.raises(BenchmarkRunError, match="verification"):
        runner.run_definition(write_definition(project, f"start_frame: {FRAME}\n"), backend,
                              seed=1, output_resolution=RES)


def test_invalid_conditioning_reports_rejected():
    for bad in (
        {"preprocessing": "resize", "effective_width": 8, "effective_height": 8},
        {"preprocessing": [""], "effective_width": 8, "effective_height": 8},
        {"preprocessing": [], "effective_width": 0, "effective_height": 8},
        {"preprocessing": [], "effective_width": 8, "effective_height": 8.0},
    ):
        with pytest.raises(ValueError):
            StartFrameConditioning(**bad)


# --- canonical image stays byte-identical ---

def test_start_frame_bytes_unchanged_by_preprocessing_backend(runner, project, frame):
    before = frame.read_bytes()
    runner.run_definition(write_definition(project, f"start_frame: {FRAME}\n"), ImageToVideoBackend(),
                          seed=1, output_resolution=RES)
    assert frame.read_bytes() == before


def test_backend_modifying_start_frame_is_caught(runner, project, frame):
    class Overwrites(ImageToVideoBackend):
        def generate(self, request, output_dir):
            Path(request.start_frame).write_bytes(png_bytes(32, 18))  # resizes in place: forbidden
            return super().generate(request, output_dir)

    with pytest.raises(BenchmarkRunError, match="modified an input"):
        runner.run_definition(write_definition(project, f"start_frame: {FRAME}\n"), Overwrites(),
                              seed=1, output_resolution=RES)


@pytest.mark.skipif(not CANONICAL.is_file(), reason="canonical benchmark media is git-ignored")
def test_canonical_rooftop_frame_is_byte_identical():
    assert sha256_file(CANONICAL) == CANONICAL_SHA
    assert image_dimensions(CANONICAL) == Resolution(width=1672, height=941)


# --- the repository benchmark ---

def test_comic_rooftop_01_pins_image_to_video_inputs():
    definition = load_definition(REPO_ROOT / "benchmarks/definitions/comic_rooftop_01.yaml")
    assert definition.status == "ready"
    assert definition.duration_seconds == 5 and definition.fps == 24
    assert definition.start_frame == "benchmarks/references/comic_hero_01/rooftop_frame_v001.png"
    assert definition.start_frame not in definition.reference_images
    assert definition.reference_images == []
    prompt = definition.prompt.lower()
    for visible in ("curly dark hair", "dark hooded streetwear", "dark-red accents",
                    "dark trousers", "light sneakers", "dark backpack"):
        assert visible in prompt, visible
    for stale in ("short messy", "charcoal", "light-grey t-shirt"):
        assert stale not in prompt, stale


# --- compatibility ---

def test_older_generation_definitions_still_load_and_run(runner, project):
    # Written before fps/start_frame existed: no fps, no start_frame.
    definition = write_definition(project)
    assert definition.fps is None and definition.start_frame is None
    record = runner.run_definition(definition, FakeGenerationBackend(), seed=1, output_resolution=RES, fps=16)
    assert record.task_parameters["requested_fps"] == 16
    assert record.start_frame is None


def test_editing_definitions_reject_generation_only_fields(project):
    path = project / "edit.yaml"
    for extra in ("fps: 24", f"start_frame: {FRAME}"):
        path.write_text(f"benchmark_id: e\ntask_type: video_editing\n{extra}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="do not apply"):
            load_definition(path)


# --- image_dimensions ---

def test_image_dimensions_png_and_jpeg(tmp_path):
    png = tmp_path / "a.png"
    png.write_bytes(png_bytes(7, 5))
    assert image_dimensions(png) == Resolution(width=7, height=5)

    jpeg = tmp_path / "a.jpg"  # SOI, APP0 segment, SOF0 with 11x13
    jpeg.write_bytes(
        b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
        + b"\xff\xc0" + struct.pack(">HBHH", 11, 8, 13, 11) + b"\x03" + b"\x00" * 9
    )
    assert image_dimensions(jpeg) == Resolution(width=11, height=13)

    other = tmp_path / "a.bin"
    other.write_bytes(b"not an image")
    assert image_dimensions(other) is None
