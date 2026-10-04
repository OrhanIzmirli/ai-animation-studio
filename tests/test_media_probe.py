import shutil
import subprocess

import pytest

from ai_studio import media_probe
from ai_studio.media_probe import OUTPUT_MEDIA_KEYS, ProbeUnavailableError, VideoInfo, ffprobe_video


def info(**overrides):
    fields = dict(probe="ffprobe 6.1.1", width=1280, height=704, fps=24.0, frame_count=121, codec="h264")
    fields.update(overrides)
    return VideoInfo(**fields)


# --- VideoInfo ---

def test_duration_is_frames_over_fps():
    assert info().duration_seconds == pytest.approx(121 / 24)


def test_to_record_has_exactly_the_output_media_keys():
    record = info().to_record()
    assert tuple(record) == OUTPUT_MEDIA_KEYS
    assert record == {
        "probe": "ffprobe 6.1.1", "width": 1280, "height": 704, "fps": 24.0,
        "frame_count": 121, "duration_seconds": 5.041667, "codec": "h264",
    }


def test_codec_is_optional():
    assert info(codec=None).to_record()["codec"] is None


@pytest.mark.parametrize(
    "bad",
    [
        {"probe": ""},
        {"width": 0},
        {"height": -1},
        {"width": 12.0},
        {"frame_count": 0},
        {"frame_count": True},
        {"fps": 0},
        {"fps": -24},
        {"fps": "24"},
        {"codec": ""},
    ],
)
def test_invalid_video_info_rejected(bad):
    with pytest.raises(ValueError):
        info(**bad)


def test_video_info_is_frozen():
    with pytest.raises(AttributeError):
        info().fps = 30


# --- ffprobe_video without a real ffprobe ---

def test_missing_ffprobe_raises_probe_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(media_probe.shutil, "which", lambda name: None)
    with pytest.raises(ProbeUnavailableError, match="ffprobe"):
        ffprobe_video(tmp_path / "out.mp4")


def test_probe_unavailable_is_a_runtime_error():
    assert issubclass(ProbeUnavailableError, RuntimeError)


@pytest.mark.parametrize(
    "value, expected",
    [("24/1", 24.0), ("24000/1001", 24000 / 1001), ("25", 25.0), ("0/0", None), ("0/1", None), ("n/a", None)],
)
def test_parse_rate(value, expected):
    assert media_probe._parse_rate(value) == expected


# --- real ffprobe ---

@pytest.fixture
def tiny_mp4(tmp_path):
    """12 frames at 24 fps, 64x48, H.264 / yuv420p."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not installed")
    path = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x48:rate=24",
            "-frames:v", "12", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True, timeout=120,
    )
    return path


def test_ffprobe_measures_a_real_video(tiny_mp4):
    measured = ffprobe_video(tiny_mp4)
    assert (measured.width, measured.height) == (64, 48)
    assert measured.fps == 24
    assert measured.frame_count == 12
    assert measured.duration_seconds == 0.5
    assert measured.codec == "h264"
    assert measured.probe.startswith("ffprobe ") and measured.probe != "ffprobe unknown"


def test_ffprobe_broken_file_is_runtime_error_not_unavailable(tmp_path):
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not installed")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    with pytest.raises(RuntimeError) as exc_info:
        ffprobe_video(broken)
    assert not isinstance(exc_info.value, ProbeUnavailableError)
