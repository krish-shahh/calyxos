"""Abstract storage backend protocol."""

from typing import Any, Protocol


class StorageBackend(Protocol):
    """
    Protocol for storage backends.

    A storage backend is responsible for persisting and restoring the stored
    nodes of a Talos-managed object. Only stored nodes are persisted; derived
    values are always recomputed on demand.
    """

    def save(self, object_id: int, stored_values: dict[str, Any]) -> None:
        """
        Save stored values for an object.

        Args:
            object_id: The Python id() of the object
            stored_values: Dict mapping method_name -> value
        """
        ...

    def load(self, object_id: int) -> dict[str, Any] | None:
        """
        Load stored values for an object.

        Args:
            object_id: The Python id() of the object

        Returns:
            Dict mapping method_name -> value, or None if object not found
        """
        ...

    def delete(self, object_id: int) -> None:
        """Delete an object's stored values."""
        ...

    def exists(self, object_id: int) -> bool:
        """Check if stored values exist for an object."""
        ...
