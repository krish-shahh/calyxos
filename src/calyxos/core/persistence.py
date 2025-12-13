"""Persistence utilities for Talos objects."""

from typing import Any, TypeVar

from calyxos.core.decorator import get_graph
from calyxos.storage.backend import StorageBackend

T = TypeVar("T")

# Global mapping of object id to loaded stored values
_loaded_values: dict[int, dict[str, Any]] = {}


def save_object(obj: Any, backend: StorageBackend) -> None:
    """
    Save the stored state of a Talos object to the backend.

    Only stored nodes are persisted; derived values are recomputed on load.

    Args:
        obj: The Talos-managed object to save
        backend: The storage backend to use
    """
    graph = get_graph(obj)
    stored_values = {}

    for node in graph.get_stored_nodes():
        stored_values[node.method_name] = node.value

    backend.save(id(obj), stored_values)


def load_object(obj: T, backend: StorageBackend) -> T:
    """
    Load the stored state of a Talos object from the backend.

    Restores stored nodes and rebuilds derived values lazily.
    Stores loaded values in a global cache that stored nodes will check
    on their first access.

    Args:
        obj: The Talos-managed object to load into
        backend: The storage backend to use

    Returns:
        The object with restored stored state
    """
    # Load stored values from backend
    stored_values = backend.load(id(obj))

    if stored_values is not None:
        # Cache loaded values for this object
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
