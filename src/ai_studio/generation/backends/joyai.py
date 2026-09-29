"""JoyAI-Video-Edit adapter. Placeholder: not installed."""

from ai_studio.generation.backends.base import BackendMetadata, NotInstalledBackend


class JoyAIVideoEditBackend(NotInstalledBackend):
    _metadata = BackendMetadata(
        backend_name="joyai-video-edit",
        backend_version="placeholder",
        model_name="JoyAI-Video-Edit",
        model_version="unknown",
        workflow_version="none",
    )
