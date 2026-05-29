"""Decorators for calyxos reactive computation nodes."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from typing import Any, TypeVar, cast, overload

from calyxos.core.flags import NodeFlag
from calyxos.graph.graph import ComputationGraph
from calyxos.graph.node import NodeType
from calyxos.tracking.context import get_current_frame, record_node_access

F = TypeVar("F", bound=Callable[..., Any])

# Global registry of computation graphs keyed by object id
_graphs: dict[int, ComputationGraph] = {}


def get_graph(obj: Any) -> ComputationGraph:
    """Get or create the computation graph for an object."""
    from calyxos.graph.registry import CrossObjectRegistry

    # Check if object has a custom id override (for testing/persistence)
    if hasattr(obj, "_calyxos_override_id"):
        obj_id = obj._calyxos_override_id
    else:
        obj_id = id(obj)

    if obj_id not in _graphs:
        graph = ComputationGraph(obj_id)
        _graphs[obj_id] = graph
        # Register in cross-object registry
        CrossObjectRegistry.get().register_graph(obj_id, graph)
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


# ---------------------------------------------------------------------------
# Unified @node decorator
# ---------------------------------------------------------------------------


def node(
    *flags: NodeFlag,
    get_changes: Callable[..., Any] | None = None,
) -> Callable[[F], F]:
    """Unified decorator for creating reactive computation nodes.

    Args:
        *flags: ``NodeFlag`` values controlling node behaviour.
            No flags = pure computed node (equivalent to ``@fn``).
            ``NodeFlag.CAN_SET`` = value settable via ``set_value()``.
            ``NodeFlag.CAN_OVERRIDE`` = value overridable in a context/layer.
            ``NodeFlag.STORED`` = persistent node (implies CAN_SET).
        get_changes: Optional reverse-propagation callback.  Signature:
            ``(self, desired_value) -> list[NodeChange]``.

    Usage::

        @node()
        def computed(self) -> int: ...

        @node(NodeFlag.CAN_SET)
        def settable(self) -> int: ...

        @node(NodeFlag.STORED)
        def persistent(self) -> float: ...
    """
    combined = NodeFlag.NONE
    for f in flags:
        combined |= f

    # STORED implies CAN_SET
    if NodeFlag.STORED in combined:
        combined |= NodeFlag.CAN_SET

    is_stored = NodeFlag.STORED in combined

    def decorator(func: F) -> F:
        if is_stored:
            return _make_stored_wrapper(func, combined, get_changes)
        return _make_computed_wrapper(func, combined, get_changes)

    return decorator


def _make_computed_wrapper(
    func: F,
    flags: NodeFlag,
    get_changes_fn: Callable[..., Any] | None,
) -> F:
    """Build the wrapper for a computed (derived) node."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        graph = get_graph(self)

        if hasattr(self, "_calyxos_override_id"):
            obj_id = self._calyxos_override_id
        else:
            obj_id = id(self)

        args_hash = _compute_args_hash((self,) + args, kwargs)

        nd = graph.get_or_create_node(
            method_name=func.__name__,
            args_hash=args_hash,
            node_type=NodeType.DERIVED,
            compute_fn=lambda: func(self, *args, **kwargs),
            flags=flags,
            get_changes_fn=get_changes_fn,
        )

        current_frame = get_current_frame()
        if current_frame is not None:
            record_node_access(obj_id, func.__name__, args_hash)

        return graph.evaluate_node(nd)

    # Attach metadata so introspection can discover flags
    wrapper._calyxos_flags = flags  # type: ignore[attr-defined]
    wrapper._calyxos_node = True  # type: ignore[attr-defined]
    return cast(F, wrapper)


def _make_stored_wrapper(
    func: F,
    flags: NodeFlag,
    get_changes_fn: Callable[..., Any] | None,
) -> F:
    """Build the wrapper for a stored (persistent/settable) node."""

    @functools.wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        from calyxos.core.persistence import get_loaded_stored_value

        graph = get_graph(self)

        if hasattr(self, "_calyxos_override_id"):
            obj_id = self._calyxos_override_id
        else:
            obj_id = id(self)

        args_hash = _compute_args_hash((self,) + args, kwargs)

        existing_node = graph.get_node(func.__name__, args_hash)
        if existing_node is None:
            loaded_value = get_loaded_stored_value(obj_id, func.__name__)
            if loaded_value is not None:
                initial_value = loaded_value
            else:
                initial_value = func(self, *args, **kwargs)

            nd = graph.get_or_create_node(
                method_name=func.__name__,
                args_hash=args_hash,
                node_type=NodeType.STORED,
                compute_fn=lambda: nd.value,
                flags=flags,
                get_changes_fn=get_changes_fn,
            )
            nd.value = initial_value
            nd.is_valid = True
        else:
            nd = existing_node

        current_frame = get_current_frame()
        if current_frame is not None:
            record_node_access(obj_id, func.__name__, args_hash)

        return nd.value

    wrapper._calyxos_flags = flags  # type: ignore[attr-defined]
    wrapper._calyxos_node = True  # type: ignore[attr-defined]
    return cast(F, wrapper)


# ---------------------------------------------------------------------------
# Backward-compatible @fn and @stored decorators
# ---------------------------------------------------------------------------


def fn(func: F) -> F:
    """Decorator that converts a method into a memoized, dependency-aware node.

    Equivalent to ``@node()`` (pure computed, no flags).
    """
    return node()(func)


def stored(func: F) -> F:
    """Decorator that marks a method/property as stored state.

    Equivalent to ``@node(NodeFlag.STORED)``.
    """
    return node(NodeFlag.STORED)(func)


# ---------------------------------------------------------------------------
# Value mutation
# ---------------------------------------------------------------------------


def set_stored(obj: Any, method_name: str, value: Any) -> None:
    """Set a stored value and propagate invalidation.

    This is the legacy mechanism for modifying stored state.
    Prefer ``set_value()`` for new code.
    """
    set_value(obj, method_name, value)


def set_value(
    obj: Any,
    method_name: str,
    value: Any,
    args_hash: int | None = None,
    _depth: int = 0,
) -> None:
    """Set a node's value and propagate invalidation.

    Works for nodes with ``CAN_SET`` or ``STORED`` flags.
    If the node has a ``get_changes`` callback, the value is propagated
    upstream via reverse propagation instead of being set directly.

    Raises:
        ValueError: If the node does not have the CAN_SET or STORED flag.
        RuntimeError: If reverse propagation exceeds the recursion limit.
    """
    if _depth > 10:
        raise RuntimeError(
            f"Reverse propagation recursion limit exceeded at {method_name}"
        )

    graph = get_graph(obj)

    # Find the node
    target_node = None
    if args_hash is not None:
        target_node = graph.get_node(method_name, args_hash)
    else:
        # Search by method name (stored values typically don't have args)
        for nd in graph.get_all_nodes():
            if nd.method_name == method_name:
                target_node = nd
                break

    # If node doesn't exist yet, create it as a stored node
    if target_node is None:
        target_node = graph.get_or_create_node(
            method_name=method_name,
            args_hash=args_hash or 0,
            node_type=NodeType.STORED,
            compute_fn=lambda: value,
            flags=NodeFlag.STORED,
        )

    # Check for reverse propagation
    if target_node.get_changes_fn is not None:
        from calyxos.core.reverse import NodeChange

        changes = target_node.get_changes_fn(obj, value)
        for change in changes:
            if not isinstance(change, NodeChange):
                raise TypeError(
                    f"get_changes must return NodeChange instances, got {type(change)}"
                )
            set_value(
                change.target_obj,
                change.target_method,
                change.value,
                args_hash=change.args_hash if change.args_hash != 0 else None,
                _depth=_depth + 1,
            )
        return

    # Validate flags allow setting
    if not target_node.has_flag(NodeFlag.CAN_SET) and not target_node.has_flag(
        NodeFlag.STORED
    ):
        # For backward compat: nodes created as NodeType.STORED without flags
        # are still settable
        if target_node.node_type != NodeType.STORED:
            raise ValueError(
                f"Node '{method_name}' does not have CAN_SET or STORED flag. "
                f"Use @node(NodeFlag.CAN_SET) or @node(NodeFlag.STORED) to allow setting."
            )

    # Value-equality guard: skip if the new value matches the existing cached value
    if target_node.is_valid:
        try:
            if target_node.value is value or target_node.value == value:
                return
        except Exception:
            pass

    # Update the value
    target_node.value = value
    target_node.is_valid = True
    target_node.compute_count += 1

    # Invalidate dependents (this also marks the node itself invalid as a
    # side-effect of the BFS start, so we re-assert validity afterwards)
    graph.invalidate_node(
        method_name=method_name,
        args_hash=target_node.args_hash,
        reason="stored value modified",
    )

    # Re-assert the node's own validity — invalidate_node marks the starting
    # node invalid, but we just explicitly set its value.
    target_node.value = value
    target_node.is_valid = True


# ---------------------------------------------------------------------------
# @map_node: per-element fan-out over a collection
# ---------------------------------------------------------------------------


def map_node(
    source: str,
    *flags: NodeFlag,
) -> Callable[[F], F]:
    """Decorator that maps a function over each element of a collection node.

    Creates per-element nodes in the dependency graph so that when the
    source collection changes, only the affected elements are recomputed.

    Args:
        source: Name of the method that returns the source collection.
        *flags: Optional ``NodeFlag`` values forwarded to the per-element nodes.

    Usage::

        class Pipeline:
            @node(NodeFlag.STORED)
            def documents(self) -> list[str]:
                return ["a", "b", "c"]

            @map_node("documents")
            def upper(self, doc: str) -> str:
                return doc.upper()

        p = Pipeline()
        p.upper()  # ["A", "B", "C"]
    """
    combined = NodeFlag.NONE
    for f in flags:
        combined |= f

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            graph = get_graph(self)

            if hasattr(self, "_calyxos_override_id"):
                obj_id = self._calyxos_override_id
            else:
                obj_id = id(self)

            orch_hash = _compute_args_hash((self,) + args, kwargs)

            def compute_orchestrated() -> list[Any]:
                # Access the source collection (records dependency)
                collection = getattr(self, source)()

                results: list[Any] = []
                for element in collection:
                    elem_hash = _compute_args_hash((self, element), {})

                    elem_nd = graph.get_or_create_node(
                        method_name=f"_map_{func.__name__}",
                        args_hash=elem_hash,
                        node_type=NodeType.DERIVED,
                        compute_fn=lambda e=element: func(self, e),
                        flags=combined,
                    )

                    # Record element-node access in the orchestrator's frame
                    record_node_access(obj_id, f"_map_{func.__name__}", elem_hash)

                    results.append(graph.evaluate_node(elem_nd))

                return results

            orch_nd = graph.get_or_create_node(
                method_name=func.__name__,
                args_hash=orch_hash,
                node_type=NodeType.DERIVED,
                compute_fn=compute_orchestrated,
                flags=combined,
            )

            # Record orchestrator access in the parent frame (if any)
            current_frame = get_current_frame()
            if current_frame is not None:
                record_node_access(obj_id, func.__name__, orch_hash)

            return graph.evaluate_node(orch_nd)

        wrapper._calyxos_flags = combined  # type: ignore[attr-defined]
        wrapper._calyxos_node = True  # type: ignore[attr-defined]
        wrapper._calyxos_map = True  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorator


def clear_graph(obj: Any) -> None:
    """Clear the computation graph for an object (for testing/reset)."""
    obj_id = id(obj)
    _graphs.pop(obj_id, None)
