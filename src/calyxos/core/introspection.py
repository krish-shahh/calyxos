"""Introspection utilities for Talos objects."""

from typing import Any

from talos.core.decorator import get_graph
from talos.graph.node import NodeType


def enable_dir(obj: Any) -> None:
    """
    Enable enhanced dir() support for a Talos-managed object.

    This patches the object's __dir__ method to show all available
    Talos-managed methods (@fn and @stored decorators).

    Usage:
        from talos import fn, stored, enable_dir

        class MyModel:
            @fn
            def compute_something(self) -> int:
                return 42

        obj = MyModel()
        enable_dir(obj)
        dir(obj)  # Now shows compute_something
    """
    original_dir = obj.__dir__ if hasattr(obj, "__dir__") else lambda: object.__dir__(obj)

    def talos_dir() -> list[str]:
        """Enhanced dir() that includes Talos-managed methods."""
        # Get the original dir() listing
        items = set(original_dir())

        # Add all Talos-managed methods from the graph
        graph = get_graph(obj)
        for node in graph.get_all_nodes():
            items.add(node.method_name)

        return sorted(items)

    # Bind the new __dir__ method to the object
    # Note: we're dynamically setting __dir__, which is technically allowed
    setattr(obj, "__dir__", talos_dir)


def get_talos_methods(obj: Any) -> dict[str, dict[str, Any]]:
    """
    Get detailed information about all Talos-managed methods on an object.

    Returns a dict mapping method names to info dicts containing:
    - type: "stored" or "derived"
    - is_valid: whether the cached value is current
    - value: the cached value (if computed)
    - compute_count: how many times this node has been recomputed
    """
    graph = get_graph(obj)
    result = {}

    for node in graph.get_all_nodes():
        result[node.method_name] = {
            "type": node.node_type.value,
            "is_valid": node.is_valid,
            "value": node.value,
            "compute_count": node.compute_count,
        }

    return result


def list_stored_methods(obj: Any) -> list[str]:
    """Get list of all @stored methods on a Talos object."""
    graph = get_graph(obj)
    return sorted(
        node.method_name
        for node in graph.get_all_nodes()
        if node.node_type == NodeType.STORED
    )


def list_computed_methods(obj: Any) -> list[str]:
    """Get list of all @fn (computed) methods on a Talos object."""
    graph = get_graph(obj)
    return sorted(
        node.method_name
        for node in graph.get_all_nodes()
        if node.node_type == NodeType.DERIVED
    )
