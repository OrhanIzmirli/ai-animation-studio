"""Replaceable video-editing backends."""

from ai_studio.generation.backends.base import (
    BackendMetadata,
    BackendNotInstalledError,
    NotInstalledBackend,
    VideoEditingBackend,
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
    "VideoEditingBackend",
    "WanEditCtrlBackend",
]
