"""Context system for temporary node overrides with automatic reversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from calyxos.core.flags import NodeFlag

if TYPE_CHECKING:
    from calyxos.graph.graph import ComputationGraph

NodeKey = tuple[int, str, int]


@dataclass
class ContextFrame:
    """One level of context nesting.

    Stores overrides applied in this scope and the original node state so
    it can be restored when the context exits.
    """

    # Overrides set during this context: node_key -> override_value
    overrides: dict[NodeKey, Any] = field(default_factory=dict)

    # Saved (value, is_valid) for each node BEFORE it was overridden in this
    # frame, so we can restore on exit.  Only the first override per node
    # within a single frame is saved — subsequent overrides of the same node
    # overwrite the override but don't update the saved state.
    saved_state: dict[NodeKey, tuple[Any, bool]] = field(default_factory=dict)


class GraphContext:
    """Context manager for temporary node overrides.

    Any overrides set inside the block are automatically reverted when the
    block exits.  Contexts are nestable: an inner context can override a node
    already overridden in an outer context, and exiting the inner context
    restores the outer context's override — not the original value.

    Usage::

        graph = get_graph(obj)
        with graph.context() as ctx:
            ctx.override(obj, "spot_price", 105.0)
            print(obj.option_price())   # uses spot_price=105.0
        # spot_price reverts to its pre-context value
    """

    def __init__(self, graph: ComputationGraph) -> None:
        self._graph = graph
        self._active = False

    def __enter__(self) -> GraphContext:
        self._graph._push_context_frame()
        self._active = True
        return self

    def __exit__(self, *exc: object) -> bool:
        self._active = False
        self._graph._pop_context_frame()
        return False

    def override(
        self,
        obj: Any,
        method_name: str,
        value: Any,
        args_hash: int = 0,
    ) -> None:
        """Override a node's value within this context.

        The node must have the ``CAN_OVERRIDE`` or ``CAN_SET`` flag.

        Args:
            obj: The object owning the node (used to resolve its graph).
            method_name: Name of the node method.
            value: The override value.
            args_hash: Argument hash for parameterised nodes (default 0).

        Raises:
            RuntimeError: If called outside the ``with`` block (context not active).
        """
        if not self._active:
            raise RuntimeError(
                "Cannot call override() on an inactive context. "
                "Use it inside the 'with graph.context() as ctx:' block."
            )
        self._graph._set_context_override(method_name, args_hash, value)
