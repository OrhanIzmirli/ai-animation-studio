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
from ai_studio.generation.backends.wan_editctrl import WanEditCtrlBackend

__all__ = [
    "BackendMetadata",
    "BackendNotInstalledError",
    "JoyAIVideoEditBackend",
    "KiwiEditBackend",
    "NotInstalledBackend",
    "VideoBackend",
    "VideoEditingBackend",
    "VideoGenerationBackend",
    "WanEditCtrlBackend",
]
