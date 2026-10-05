"""Backend registry: backend_name -> backend class.

The only place that maps a name to a model adapter. The runner never looks
here; launchers (the CLI) resolve a name to a class and hand the runner an
instance.
"""

from __future__ import annotations

from ai_studio.generation.backends.base import Backend, VideoBackend
from ai_studio.generation.backends.fake_image import FakeImageBackend
from ai_studio.generation.backends.joyai import JoyAIVideoEditBackend
from ai_studio.generation.backends.kiwi import KiwiEditBackend
from ai_studio.generation.backends.wan_editctrl import WanEditCtrlBackend
from ai_studio.generation.backends.wan_ti2v import WanTI2V5BBackend
from ai_studio.generation.backends.wan_comfyui import WanBaseComfyUIBackend, WanTurboComfyUIBackend

# Keys are each backend's metadata().backend_name.
BACKENDS: dict[str, type[VideoBackend]] = {
    "wan-ti2v-5b": WanTI2V5BBackend,
    "wan-ti2v-5b-turbo-comfyui": WanTurboComfyUIBackend,
    "wan-ti2v-5b-comfyui": WanBaseComfyUIBackend,
    "joyai-video-edit": JoyAIVideoEditBackend,
    "kiwi-edit": KiwiEditBackend,
    "wan2.2-vace-fun-a14b-editctrl": WanEditCtrlBackend,
}


# Still-image backends, kept apart from the video ones. Only the model-free
# fake is registered until a real image backend is chosen.
IMAGE_BACKENDS: dict[str, type[Backend]] = {
    "fake-image": FakeImageBackend,
}


def all_backends() -> dict[str, type[Backend]]:
    """Every registered backend, video and image, by name."""
    return {**BACKENDS, **IMAGE_BACKENDS}


def get_backend(name: str) -> type[Backend]:
    """The backend class (video or image) registered under `name`."""
    backends = all_backends()
    try:
        return backends[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; valid names: {', '.join(sorted(backends))}") from None
