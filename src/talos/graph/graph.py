"""Instance-scoped computation graph."""

import threading
from collections.abc import Callable
from typing import Any

from talos.graph.node import Node, NodeType
from talos.tracking.context import (
    pop_frame,
    push_frame,
)


class ComputationGraph:
    """Manages the computation graph for a single object instance."""

    def __init__(self, object_id: int) -> None:
        self.object_id = object_id
        self.nodes: dict[tuple[int, str, int], Node] = {}
        self._lock = threading.RLock()

    def get_or_create_node(
        self,
        method_name: str,
        args_hash: int,
        node_type: NodeType,
        compute_fn: Callable[[], Any],
    ) -> Node:
        """Get an existing node or create a new one."""
        key = (self.object_id, method_name, args_hash)

        with self._lock:
            if key in self.nodes:
                return self.nodes[key]

            node = Node(
                object_id=self.object_id,
                method_name=method_name,
                args_hash=args_hash,
                node_type=node_type,
                compute_fn=compute_fn,
            )
            self.nodes[key] = node
            return node

    def evaluate_node(
        self, node: Node, recursion_guard: set[tuple[int, str, int]] | None = None
    ) -> Any:
        """
        Evaluate a node, with bottom-up dependency resolution.

        Returns the cached value if valid; otherwise recomputes from dependencies.
        Records parent->child relationships during evaluation.
        """
        if recursion_guard is None:
            recursion_guard = set()

        key = node.key()
        if key in recursion_guard:
            raise RuntimeError(f"Cycle detected in computation graph at {key}")

        recursion_guard.add(key)

        # Return cached value if valid
        if node.is_valid:
            return node.value

        # Push frame to track dependencies
        frame = push_frame(node.object_id, node.method_name, node.args_hash)

        try:
            # Evaluate the compute function with tracking enabled
            result = node.compute_fn()

            # Extract dependencies from the frame
            with self._lock:
                node.children = frame.accessed_nodes.copy()

                # Update parent pointers in child nodes
                for child_key in frame.accessed_nodes:
                    child_node = self.nodes.get(child_key)
                    if child_node is not None:
                        child_node.parents.add(key)

                # Cache the value
                node.value = result
                node.is_valid = True
                node.compute_count += 1

            return result
        finally:
            pop_frame()
            recursion_guard.discard(key)

    def invalidate_node(
        self, method_name: str, args_hash: int, reason: str = "stored value changed"
    ) -> None:
        """
        Invalidate a node and propagate invalidation to downstream dependents.

        Does not eagerly recompute; marks nodes as dirty for lazy recomputation.
        """
        key = (self.object_id, method_name, args_hash)
        node = self.nodes.get(key)

        if node is None:
            return

        with self._lock:
            # Invalidate this node
            node.is_valid = False
            node.last_recompute_reason = reason

            # BFS to invalidate all downstream dependents (nodes that depend on this one)
            queue = list(node.parents)  # nodes that depend on this one
            visited = {key}

            while queue:
                parent_key = queue.pop(0)
                if parent_key in visited:
                    continue
                visited.add(parent_key)

                parent_node = self.nodes.get(parent_key)
                if parent_node is None:
                    continue

                if parent_node.is_valid:
                    parent_node.is_valid = False
                    parent_node.last_recompute_reason = reason
                    queue.extend(parent_node.parents)

    def get_node(self, method_name: str, args_hash: int) -> Node | None:
        """Get a node by method name and args hash, if it exists."""
        key = (self.object_id, method_name, args_hash)
        return self.nodes.get(key)

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes in the graph."""
        with self._lock:
            return list(self.nodes.values())

    def get_stored_nodes(self) -> list[Node]:
        """Get all stored nodes in the graph."""
        with self._lock:
            return [n for n in self.nodes.values() if n.node_type == NodeType.STORED]

    def get_invalid_nodes(self) -> list[Node]:
        """Get all invalid (dirty) nodes in the graph."""
        with self._lock:
            return [n for n in self.nodes.values() if not n.is_valid]
