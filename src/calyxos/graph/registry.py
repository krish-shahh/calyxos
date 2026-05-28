"""Cross-object dependency registry using weak references."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from calyxos.graph.graph import ComputationGraph

NodeKey = tuple[int, str, int]


class CrossObjectRegistry:
    """Global registry for cross-object dependency edges.

    When a node on Object A depends on a node on Object B (i.e. A's method
    calls B's method during evaluation), the edge is registered here.  When
    B's node is invalidated, this registry is consulted to propagate the
    invalidation to A's dependent nodes.

    Uses weak references to ``ComputationGraph`` instances so that objects
    can be garbage collected normally.
    """

    _instance: CrossObjectRegistry | None = None

    def __init__(self) -> None:
        # Maps object_id -> weakref to its ComputationGraph
        self._graph_refs: dict[int, weakref.ref[Any]] = {}

        # Maps child_key -> set of parent_keys in OTHER graphs that depend on it
        self._cross_edges: dict[NodeKey, set[NodeKey]] = {}

    @classmethod
    def get(cls) -> CrossObjectRegistry:
        """Get the singleton registry instance."""
        if cls._instance is None:
            cls._instance = CrossObjectRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def register_graph(self, object_id: int, graph: ComputationGraph) -> None:
        """Register a graph so it can be looked up by object_id."""
        self._graph_refs[object_id] = weakref.ref(graph)

    def get_graph(self, object_id: int) -> ComputationGraph | None:
        """Get a graph by object_id, or None if it was garbage collected."""
        ref = self._graph_refs.get(object_id)
        if ref is not None:
            return ref()
        return None

    def add_cross_edge(self, child_key: NodeKey, parent_key: NodeKey) -> None:
        """Record that parent_key (in another graph) depends on child_key."""
        if child_key not in self._cross_edges:
            self._cross_edges[child_key] = set()
        self._cross_edges[child_key].add(parent_key)

    def get_cross_parents(self, child_key: NodeKey) -> set[NodeKey]:
        """Get nodes in OTHER graphs that depend on child_key."""
        return self._cross_edges.get(child_key, set()).copy()

    def cleanup_dead_refs(self) -> None:
        """Remove entries for garbage-collected objects."""
        dead_ids = {oid for oid, ref in self._graph_refs.items() if ref() is None}
        for oid in dead_ids:
            del self._graph_refs[oid]

        # Remove cross edges where either side belongs to a dead object
        for child_key in list(self._cross_edges):
            if child_key[0] in dead_ids:
                del self._cross_edges[child_key]
            else:
                self._cross_edges[child_key] = {
                    p for p in self._cross_edges[child_key] if p[0] not in dead_ids
                }
                if not self._cross_edges[child_key]:
                    del self._cross_edges[child_key]
