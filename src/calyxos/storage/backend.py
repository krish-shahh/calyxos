"""Abstract storage backend protocol."""

from typing import Any, Protocol


class StorageBackend(Protocol):
    """
    Protocol for storage backends.

    A storage backend is responsible for persisting and restoring the stored
    nodes of a calyxos-managed object.  Only stored nodes are persisted; derived
    values are always recomputed on demand.

    Keys are user-supplied strings (e.g. ``"portfolio-1"``, ``"pipeline-main"``)
    that remain stable across process restarts.
    """

    def save(self, key: str, stored_values: dict[str, Any]) -> None:
        """
        Save stored values for an object.

        Args:
            key: A stable, user-supplied string identifier.
            stored_values: Dict mapping method_name -> value.
        """
        ...

    def load(self, key: str) -> dict[str, Any] | None:
        """
        Load stored values for an object.

        Args:
            key: The same string identifier used when saving.

        Returns:
            Dict mapping method_name -> value, or None if not found.
        """
        ...

    def delete(self, key: str) -> None:
        """Delete an object's stored values."""
        ...

    def exists(self, key: str) -> bool:
        """Check if stored values exist for an object."""
        ...
