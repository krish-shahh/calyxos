"""Persistence utilities for calyxos objects."""

from typing import Any, TypeVar

from calyxos.core.decorator import get_graph
from calyxos.storage.backend import StorageBackend

T = TypeVar("T")

# Global mapping of object id -> loaded stored values
_loaded_values: dict[int, dict[str, Any]] = {}


def save_object(obj: Any, backend: StorageBackend, key: str) -> None:
    """Save the stored state of a calyxos object to the backend.

    Only stored nodes are persisted; derived values are recomputed on load.

    Args:
        obj: The calyxos-managed object to save.
        backend: The storage backend to use.
        key: A stable string identifier (e.g. ``"pipeline-main"``).
             Must be the same key used later with :func:`load_object`.
    """
    graph = get_graph(obj)
    stored_values: dict[str, Any] = {}

    for node in graph.get_stored_nodes():
        stored_values[node.method_name] = node.value

    backend.save(key, stored_values)


def load_object(obj: T, backend: StorageBackend, key: str) -> T:
    """Load the stored state of a calyxos object from the backend.

    Restores stored nodes and rebuilds derived values lazily.

    Args:
        obj: The calyxos-managed object to load into.
        backend: The storage backend to use.
        key: The same string identifier used when saving.

    Returns:
        The object with restored stored state.
    """
    stored_values = backend.load(key)

    if stored_values is not None:
        # Cache loaded values keyed by object's runtime id so the stored
        # wrapper can pick them up on first access.
        _loaded_values[id(obj)] = stored_values

    return obj


def get_loaded_stored_value(obj_id: int, method_name: str) -> Any | None:
    """Get a loaded stored value for an object, if it was loaded."""
    if obj_id not in _loaded_values:
        return None
    return _loaded_values[obj_id].get(method_name)


def clear_loaded_values() -> None:
    """Clear all loaded values (for testing)."""
    _loaded_values.clear()
