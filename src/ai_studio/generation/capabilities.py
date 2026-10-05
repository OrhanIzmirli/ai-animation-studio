"""What a backend can do. Models are selected by capability and by measured
benchmark performance; no model is forced into a contract it does not
naturally support.

A capability's value doubles as the benchmark `task_type`.
Future: VIDEO_REPAIR ("video_repair") - video/frames/region -> repaired video.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    VIDEO_GENERATION = "video_generation"  # text / image / references -> new video
    VIDEO_EDITING = "video_editing"  # source video / references / instruction -> edited video
    IMAGE_GENERATION = "image_generation"  # text / references -> new still image(s)
    IMAGE_EDITING = "image_editing"  # source image / references / instruction -> edited image(s)


VIDEO_CAPABILITIES = frozenset({Capability.VIDEO_GENERATION, Capability.VIDEO_EDITING})
IMAGE_CAPABILITIES = frozenset({Capability.IMAGE_GENERATION, Capability.IMAGE_EDITING})


class UnsupportedCapabilityError(ValueError):
    """The backend does not declare the capability the request needs."""
