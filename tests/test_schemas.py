import pytest

from ai_studio.benchmarking import BenchmarkRequest, Resolution
from ai_studio.generation.result import GenerationResult


# --- BenchmarkRequest ---

def test_request_minimal_is_valid():
    request = BenchmarkRequest(
        benchmark_id="b1",
        source_video="in.mp4",
        prompt="edit",
        output_resolution=Resolution(width=64, height=64),
        seed=0,
    )
    assert request.reference_images == []
    assert request.backend_parameters == {}
    assert request.negative_prompt is None


def test_request_supports_multiple_references(request_factory):
    request = request_factory(reference_images=["a.png", "b.png", "c.png"])
    assert request.input_paths[1:] == ["a.png", "b.png", "c.png"]


def test_request_keeps_arbitrary_backend_parameters(request_factory):
    params = {"sampler": "x", "nested": {"lora": [{"name": "l", "weight": 0.5}]}}
    assert request_factory(backend_parameters=params).backend_parameters == params


@pytest.mark.parametrize(
    "overrides",
    [
        {"benchmark_id": ""},
        {"benchmark_id": "../x"},
        {"source_video": ""},
        {"prompt": "   "},
        {"seed": "1"},
        {"seed": False},
        {"seed": 1.5},
        {"output_resolution": (1280, 720)},
        {"reference_images": "single.png"},
        {"reference_images": [""]},
        {"negative_prompt": 5},
        {"backend_parameters": None},
        {"backend_parameters": {"not_json": object()}},
    ],
)
def test_request_rejects_invalid(request_factory, overrides):
    with pytest.raises(ValueError):
        request_factory(**overrides)


# --- GenerationResult ---

def make_result(**overrides):
    fields = dict(
        output_video="out.mp4",
        backend_name="b",
        backend_version="1",
        model_name="m",
        model_version="1",
    )
    fields.update(overrides)
    return GenerationResult(**fields)


def test_generation_result_minimal_is_valid():
    result = make_result()
    assert result.parameters == {} and result.environment == {} and result.warnings == []
    assert result.fps is None and result.frame_count is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"output_video": ""},
        {"backend_name": ""},
        {"model_version": None},
        {"parameters": []},
        {"parameters": {"x": object()}},
        {"environment": None},
        {"warnings": "one warning"},
        {"warnings": [1]},
        {"generation_time_seconds": -1},
        {"fps": 0},
        {"frame_count": 0},
    ],
)
def test_generation_result_rejects_invalid(overrides):
    with pytest.raises(ValueError):
        make_result(**overrides)
