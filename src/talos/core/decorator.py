"""Decorators for Talos functions and stored state."""

import functools
import hashlib
from collections.abc import Callable
from typing import Any, TypeVar, cast

from talos.graph.graph import ComputationGraph
from talos.graph.node import NodeType
from talos.tracking.context import get_current_frame, record_node_access

F = TypeVar("F", bound=Callable[..., Any])

# Global registry of computation graphs keyed by object id
_graphs: dict[int, ComputationGraph] = {}


def get_graph(obj: Any) -> ComputationGraph:
    """Get or create the computation graph for an object."""
    # Check if object has a custom id override (for testing/persistence)
    if hasattr(obj, "_talos_override_id"):
        obj_id = obj._talos_override_id
    else:
        obj_id = id(obj)

    if obj_id not in _graphs:
        _graphs[obj_id] = ComputationGraph(obj_id)
    return _graphs[obj_id]


def _compute_args_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Compute a stable hash for function arguments."""
    # Skip 'self' parameter
    try:
        items = []
        for arg in args[1:]:  # Skip self
            items.append(repr(arg).encode())
        for k, v in sorted(kwargs.items()):
            items.append(f"{k}={v!r}".encode())
        content = b"|".join(items)
        return int(hashlib.md5(content).hexdigest(), 16)
    except Exception:
        # Fallback: use object ids for unhashable objects
        parts = []
        for arg in args[1:]:
            try:
                hash(arg)
                parts.append(str(hash(arg)))
            except TypeError:
                parts.append(str(id(arg)))
        for k, v in sorted(kwargs.items()):
            try:
                hash(v)
                parts.append(f"{k}={hash(v)}")
            except TypeError:
                parts.append(f"{k}={id(v)}")
        content = "|".join(parts).encode()
        return int(hashlib.md5(content).hexdigest(), 16)


def fn(func: F) -> F:
    """
    Decorator that converts a method into a memoized, dependency-aware node.

    The decorated method will:
    - Be evaluated lazily and cached
    - Record runtime dependencies when called
    - Be recomputed only when dependencies change
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Get the computation graph for this object
        graph = get_graph(self)

        # Use the same ID as the graph (with override support)
        if hasattr(self, "_talos_override_id"):
            obj_id = self._talos_override_id
        else:
            obj_id = id(self)

        # Compute hash of arguments
        args_hash = _compute_args_hash((self,) + args, kwargs)

        # Get or create the node
        node = graph.get_or_create_node(
            method_name=func.__name__,
            args_hash=args_hash,
            node_type=NodeType.DERIVED,
            compute_fn=lambda: func(self, *args, **kwargs),
        )

        # Record this node access in the current frame (for dependency tracking)
        current_frame = get_current_frame()
        if current_frame is not None:
            record_node_access(obj_id, func.__name__, args_hash)

        # Evaluate the node (with caching and dependency resolution)
        return graph.evaluate_node(node)

    return cast(F, wrapper)


def stored(func: F) -> F:
    """
    Decorator that marks a method/property as stored state.

    Stored nodes are persisted via the storage backend. When a stored value
    changes, invalidation is propagated to downstream dependents.

    The initial value is determined by calling the decorated function once,
    or restored from a loaded object's cache.
    """

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Delayed import to avoid circular dependency
        from talos.core.persistence import get_loaded_stored_value

        # Get the computation graph for this object
        graph = get_graph(self)
        # Use the same ID as the graph (with override support)
        if hasattr(self, "_talos_override_id"):
            obj_id = self._talos_override_id
        else:
            obj_id = id(self)

        # Compute hash of arguments
        args_hash = _compute_args_hash((self,) + args, kwargs)

        # Check if node already exists
        existing_node = graph.get_node(func.__name__, args_hash)
        if existing_node is None:
            # First access: check for loaded value first
            loaded_value = get_loaded_stored_value(obj_id, func.__name__)
            if loaded_value is not None:
                initial_value = loaded_value
            else:
                # No loaded value, compute initial value
                initial_value = func(self, *args, **kwargs)

            # For stored nodes, the compute_fn just returns the cached value
            # It's not used to compute from dependencies like derived nodes
            node = graph.get_or_create_node(
                method_name=func.__name__,
                args_hash=args_hash,
                node_type=NodeType.STORED,
                compute_fn=lambda: node.value,  # Use node.value directly
            )
            node.value = initial_value
            node.is_valid = True
        else:
            node = existing_node

        # Record this node access in the current frame
        current_frame = get_current_frame()
        if current_frame is not None:
            record_node_access(obj_id, func.__name__, args_hash)

        # Always return the cached value for stored nodes
        # (They are only invalidated by explicit set_stored calls)
        return node.value

    return cast(F, wrapper)


def set_stored(obj: Any, method_name: str, value: Any) -> None:
    """
    Set a stored value and propagate invalidation.

    This is the mechanism for modifying stored state in Talos.
    """
    graph = get_graph(obj)

    # Find the stored node with this method name and update it
    stored_node = None
    for node in graph.get_all_nodes():
        if node.method_name == method_name and node.node_type == NodeType.STORED:
            stored_node = node
            break

    # If node doesn't exist yet, create it
    if stored_node is None:
        stored_node = graph.get_or_create_node(
            method_name=method_name,
            args_hash=0,  # Stored values typically don't have args
            node_type=NodeType.STORED,
            compute_fn=lambda: value,
        )

    # Update the cached value
    stored_node.value = value
    stored_node.is_valid = True
    stored_node.compute_count += 1

    # Invalidate all downstream dependents
    graph.invalidate_node(
        method_name=method_name,
        args_hash=stored_node.args_hash,
        reason="stored value modified",
    )


def clear_graph(obj: Any) -> None:
    """Clear the computation graph for an object (for testing/reset)."""
    obj_id = id(obj)
    _graphs.pop(obj_id, None)
