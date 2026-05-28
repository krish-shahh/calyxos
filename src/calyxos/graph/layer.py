"""Layer system for persistent computation state across re-entries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calyxos.graph.graph import ComputationGraph

NodeKey = tuple[int, str, int]


@dataclass
class Layer:
    """A named computation layer that preserves cached state across entries.

    Unlike contexts, layers preserve computed state when you exit them and
    restore it when you re-enter.  This is useful for sensitivity analysis
    where you bump an input, compute downstream results, exit, then later
    re-enter and the bumped computation is still cached.

    Usage::

        L = graph.layer("bump_spot")
        with L:
            set_value(obj, "spot", 105.0)
            result = obj.option_price()   # computes and caches in layer
        # exit: snapshot saved, graph restored to pre-layer state

        with L:
            result = obj.option_price()   # cached, no recompute
    """

    name: str
    _graph: ComputationGraph

    # Overrides explicitly set within this layer
    _overrides: dict[NodeKey, Any] = field(default_factory=dict)

    # Snapshot of all node states when the layer was last exited
    _snapshot: dict[NodeKey, tuple[Any, bool]] = field(default_factory=dict)

    # Snapshot of node states before the layer was first entered
    _base_snapshot: dict[NodeKey, tuple[Any, bool]] = field(default_factory=dict)

    _active: bool = False

    def __enter__(self) -> Layer:
        if self._active:
            raise RuntimeError(f"Layer '{self.name}' is already active")
        self._active = True
        self._graph._enter_layer(self)
        return self

    def __exit__(self, *exc: object) -> bool:
        self._graph._exit_layer(self)
        self._active = False
        return False

    def set(self, method_name: str, value: Any, args_hash: int = 0) -> None:
        """Override a node's value within this layer.

        The override persists across re-entries.
        """
        nd = self._graph._find_node_by_name(method_name, args_hash)
        if nd is None:
            raise KeyError(
                f"Node '{method_name}' does not exist. "
                "Access the node at least once before setting in a layer."
            )
        key = nd.key()
        self._overrides[key] = value
        if self._active:
            nd.value = value
            nd.is_valid = True
            # Invalidate dependents
            for parent_key in list(nd.parents):
                parent_nd = self._graph.nodes.get(parent_key)
                if parent_nd is not None:
                    self._graph.invalidate_node(
                        parent_nd.method_name,
                        parent_nd.args_hash,
                        reason="layer override applied",
                    )
