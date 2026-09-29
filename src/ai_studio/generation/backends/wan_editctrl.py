"""Wan 2.2 VACE-Fun A14B + EditCtrl adapter. Placeholder: not installed."""

from ai_studio.generation.backends.base import BackendMetadata, NotInstalledBackend


class WanEditCtrlBackend(NotInstalledBackend):
    _metadata = BackendMetadata(
        backend_name="wan2.2-vace-fun-a14b-editctrl",
        backend_version="placeholder",
        model_name="Wan2.2-VACE-Fun-A14B + EditCtrl",
        model_version="unknown",
        workflow_version="none",
    )
