"""Wan 2.2 VACE-Fun A14B + EditCtrl adapter. Placeholder: not installed."""

from ai_studio.generation.backends.base import BackendMetadata, NotInstalledBackend, VideoEditingBackend


class WanEditCtrlBackend(NotInstalledBackend, VideoEditingBackend):
    """Declared as VIDEO_EDITING only. Other capabilities are not assumed;
    they must be verified separately before being declared."""

    _metadata = BackendMetadata(
        backend_name="wan2.2-vace-fun-a14b-editctrl",
        backend_version="placeholder",
        model_name="Wan2.2-VACE-Fun-A14B + EditCtrl",
        model_version="unknown",
        workflow_version="none",
    )
