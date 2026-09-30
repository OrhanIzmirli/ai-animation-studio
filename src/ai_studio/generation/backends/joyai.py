"""JoyAI-Video-Edit adapter. Placeholder: not installed."""

from ai_studio.generation.backends.base import BackendMetadata, NotInstalledBackend, VideoEditingBackend


class JoyAIVideoEditBackend(NotInstalledBackend, VideoEditingBackend):
    """Declared as VIDEO_EDITING only. Other capabilities are not assumed;
    they must be verified separately before being declared."""

    _metadata = BackendMetadata(
        backend_name="joyai-video-edit",
        backend_version="placeholder",
        model_name="JoyAI-Video-Edit",
        model_version="unknown",
        workflow_version="none",
    )
