import inspect

import pytest

from ai_studio.generation.backends import (
    BackendMetadata,
    BackendNotInstalledError,
    JoyAIVideoEditBackend,
    KiwiEditBackend,
    VideoEditingBackend,
    WanEditCtrlBackend,
)
from conftest import FakeBackend

PLACEHOLDERS = [JoyAIVideoEditBackend, KiwiEditBackend, WanEditCtrlBackend]


def test_interface_is_abstract():
    with pytest.raises(TypeError):
        VideoEditingBackend()


def test_interface_exposes_the_contract():
    for method in ("metadata", "validate_inputs", "prepare", "generate"):
        assert method in VideoEditingBackend.__abstractmethods__


def test_incomplete_backend_cannot_be_instantiated():
    class Incomplete(VideoEditingBackend):
        def metadata(self):
            ...

    with pytest.raises(TypeError):
        Incomplete()


def test_fake_backend_satisfies_contract(request_factory, tmp_path):
    backend = FakeBackend()
    request = request_factory()
    backend.validate_inputs(request)
    backend.prepare()
    result = backend.generate(request, tmp_path / "out")
    assert isinstance(backend.metadata(), BackendMetadata)
    assert result.output_video.endswith("out.mp4")


def test_interface_source_mentions_no_specific_model():
    import ai_studio.generation.backends.base as base

    source = inspect.getsource(base).lower()
    for name in ("joyai", "kiwi", "wan", "editctrl", "vace"):
        assert name not in source


def test_metadata_rejects_empty_values():
    with pytest.raises(ValueError):
        BackendMetadata(
            backend_name="", backend_version="1", model_name="m",
            model_version="1", workflow_version="w",
        )


@pytest.mark.parametrize("backend_cls", PLACEHOLDERS)
def test_placeholder_metadata_works_without_install(backend_cls):
    backend = backend_cls()
    assert isinstance(backend, VideoEditingBackend)
    meta = backend.metadata()
    assert meta.backend_name and meta.model_name
    assert meta.backend_version == "placeholder"


@pytest.mark.parametrize("backend_cls", PLACEHOLDERS)
def test_placeholder_generate_raises_not_installed(backend_cls, request_factory, tmp_path):
    backend = backend_cls()
    with pytest.raises(BackendNotInstalledError, match="not installed") as exc:
        backend.generate(request_factory(), tmp_path)
    assert backend.metadata().backend_name in str(exc.value)


@pytest.mark.parametrize("backend_cls", PLACEHOLDERS)
def test_placeholder_prepare_raises_not_installed(backend_cls):
    with pytest.raises(BackendNotInstalledError):
        backend_cls().prepare()


def test_placeholder_backend_names_are_unique():
    names = [cls().metadata().backend_name for cls in PLACEHOLDERS]
    assert len(set(names)) == len(names)
