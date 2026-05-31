"""Introspection utilities for calyxos objects."""

from __future__ import annotations

from typing import Any

from calyxos.core.decorator import get_graph
from calyxos.core.flags import NodeFlag
from calyxos.graph.node import NodeType


def enable_dir(obj: Any) -> None:
    """
    Enable enhanced dir() support for a calyxos-managed object.

    This patches the object's __dir__ method to show all available
    calyxos-managed methods (@fn and @stored decorators).

    Usage:
        from calyxos import fn, stored, enable_dir

        class MyModel:
            @fn
            def compute_something(self) -> int:
                return 42

        obj = MyModel()
        enable_dir(obj)
        dir(obj)  # Now shows compute_something
    """
    original_dir = obj.__dir__ if hasattr(obj, "__dir__") else lambda: object.__dir__(obj)

    def calyxos_dir() -> list[str]:
        """Enhanced dir() that includes calyxos-managed methods."""
        # Get the original dir() listing
        items = set(original_dir())

        # Add all calyxos-managed methods from the graph
        graph = get_graph(obj)
        for node in graph.get_all_nodes():
            items.add(node.method_name)

        return sorted(items)

    # Bind the new __dir__ method to the object
    # Note: we're dynamically setting __dir__, which is technically allowed
    obj.__dir__ = calyxos_dir


def get_calyxos_methods(obj: Any) -> dict[str, dict[str, Any]]:
    """
    Get detailed information about all calyxos-managed methods on an object.

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
    """Get list of all @stored methods on a calyxos object."""
    graph = get_graph(obj)
    return sorted(
        node.method_name
        for node in graph.get_all_nodes()
        if node.node_type == NodeType.STORED
    )


def list_computed_methods(obj: Any) -> list[str]:
    """Get list of all @fn (computed) methods on a calyxos object."""
    graph = get_graph(obj)
    return sorted(
        node.method_name
        for node in graph.get_all_nodes()
        if node.node_type == NodeType.DERIVED
    )


def is_overridden(obj: Any, method_name: str) -> bool:
    """Check if a node is currently overridden in a context or layer."""
    graph = get_graph(obj)
    nd = graph._find_node_by_name(method_name)
    if nd is None:
        return False
    has_override, _ = graph._get_active_override(nd.key())
    return has_override


def is_set(obj: Any, method_name: str) -> bool:
    """Check if a node has been explicitly set (has CAN_SET or STORED flag and a value)."""
    graph = get_graph(obj)
    nd = graph._find_node_by_name(method_name)
    if nd is None:
        return False
    return nd.has_flag(NodeFlag.CAN_SET) or nd.has_flag(NodeFlag.STORED)


def get_node_flags(obj: Any, method_name: str) -> NodeFlag:
    """Get the flags of a node."""
    graph = get_graph(obj)
    nd = graph._find_node_by_name(method_name)
    if nd is None:
        return NodeFlag.NONE
    return nd.flags
