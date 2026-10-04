"""Replaceable video backends, one contract per capability."""

from ai_studio.generation.backends.base import (
    BackendMetadata,
    BackendNotInstalledError,
    NotInstalledBackend,
    VideoBackend,
    VideoEditingBackend,
    VideoGenerationBackend,
)
from ai_studio.generation.backends.joyai import JoyAIVideoEditBackend
from ai_studio.generation.backends.kiwi import KiwiEditBackend
from ai_studio.generation.backends.registry import BACKENDS, get_backend
from ai_studio.generation.backends.wan_editctrl import WanEditCtrlBackend
from ai_studio.generation.backends.wan_ti2v import WanTI2V5BBackend
from ai_studio.generation.backends.wan_turbo_comfyui import WanTurboComfyUIBackend

__all__ = [
    "BACKENDS",
    "BackendMetadata",
    "BackendNotInstalledError",
    "JoyAIVideoEditBackend",
    "KiwiEditBackend",
    "NotInstalledBackend",
    "VideoBackend",
    "VideoEditingBackend",
    "VideoGenerationBackend",
    "WanEditCtrlBackend",
    "WanTI2V5BBackend",
    "WanTurboComfyUIBackend",
    "get_backend",
]
