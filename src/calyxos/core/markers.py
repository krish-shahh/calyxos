"""Markers for stored state in Talos."""

from typing import Any, TypeVar

T = TypeVar("T")


class StoredMarker:
    """Marker class to indicate that an attribute is stored state."""

    def __init__(self, default: Any = None) -> None:
        self.default = default

    def __repr__(self) -> str:
        return f"Stored(default={self.default!r})"


# Singleton instance for type checking and use in annotations
Stored = StoredMarker()
