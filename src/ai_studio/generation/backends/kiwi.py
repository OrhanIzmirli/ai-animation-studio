"""Kiwi-Edit adapter. Placeholder: not installed."""

from ai_studio.generation.backends.base import BackendMetadata, NotInstalledBackend


class KiwiEditBackend(NotInstalledBackend):
    _metadata = BackendMetadata(
        backend_name="kiwi-edit",
        backend_version="placeholder",
        model_name="Kiwi-Edit",
        model_version="unknown",
        workflow_version="none",
    )
