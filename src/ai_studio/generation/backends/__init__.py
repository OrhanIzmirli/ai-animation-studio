"""Replaceable video and image backends, one contract per capability."""

from ai_studio.generation.backends.base import (
    Backend,
    BackendMetadata,
    BackendNotInstalledError,
    ImageEditingBackend,
    ImageGenerationBackend,
    NotInstalledBackend,
    VideoBackend,
    VideoEditingBackend,
    VideoGenerationBackend,
)
from ai_studio.generation.backends.joyai import JoyAIVideoEditBackend
from ai_studio.generation.backends.kiwi import KiwiEditBackend
from ai_studio.generation.backends.fake_image import FakeImageBackend
from ai_studio.generation.backends.registry import BACKENDS, IMAGE_BACKENDS, all_backends, get_backend
from ai_studio.generation.backends.wan_editctrl import WanEditCtrlBackend
from ai_studio.generation.backends.wan_ti2v import WanTI2V5BBackend
from ai_studio.generation.backends.wan_comfyui import WanBaseComfyUIBackend, WanTurboComfyUIBackend

__all__ = [
    "BACKENDS",
    "IMAGE_BACKENDS",
    "Backend",
    "BackendMetadata",
    "BackendNotInstalledError",
    "FakeImageBackend",
    "ImageEditingBackend",
    "ImageGenerationBackend",
    "JoyAIVideoEditBackend",
    "KiwiEditBackend",
    "NotInstalledBackend",
    "VideoBackend",
    "VideoEditingBackend",
    "VideoGenerationBackend",
    "WanEditCtrlBackend",
    "WanBaseComfyUIBackend",
    "WanTI2V5BBackend",
    "WanTurboComfyUIBackend",
    "all_backends",
    "get_backend",
]
