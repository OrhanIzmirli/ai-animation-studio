"""Backend registry: backend_name -> backend class.

The only place that maps a name to a model adapter. The runner never looks
here; launchers (the CLI) resolve a name to a class and hand the runner an
instance.
"""

from __future__ import annotations

from ai_studio.generation.backends.base import VideoBackend
from ai_studio.generation.backends.joyai import JoyAIVideoEditBackend
from ai_studio.generation.backends.kiwi import KiwiEditBackend
from ai_studio.generation.backends.wan_editctrl import WanEditCtrlBackend
from ai_studio.generation.backends.wan_ti2v import WanTI2V5BBackend
from ai_studio.generation.backends.wan_turbo_comfyui import WanTurboComfyUIBackend

# Keys are each backend's metadata().backend_name.
BACKENDS: dict[str, type[VideoBackend]] = {
    "wan-ti2v-5b": WanTI2V5BBackend,
    "wan-ti2v-5b-turbo-comfyui": WanTurboComfyUIBackend,
    "joyai-video-edit": JoyAIVideoEditBackend,
    "kiwi-edit": KiwiEditBackend,
    "wan2.2-vace-fun-a14b-editctrl": WanEditCtrlBackend,
}


def get_backend(name: str) -> type[VideoBackend]:
    """The backend class registered under `name`."""
    try:
        return BACKENDS[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; valid names: {', '.join(sorted(BACKENDS))}") from None
