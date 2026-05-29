"""Instance-scoped computation graph."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from calyxos.core.flags import NodeFlag
from calyxos.graph.node import Node, NodeType
from calyxos.tracking.context import (
    pop_frame,
    push_frame,
)

if TYPE_CHECKING:
    from calyxos.graph.context import ContextFrame
    from calyxos.graph.layer import Layer

NodeKey = tuple[int, str, int]


class ComputationGraph:
    """Manages the computation graph for a single object instance."""

    def __init__(self, object_id: int) -> None:
        self.object_id = object_id
        self.nodes: dict[NodeKey, Node] = {}
        self._lock = threading.RLock()

        # Context override stack (Phase 2)
        self._context_stack: list[ContextFrame] = []

        # Active layer stack (Phase 3)
        self._layer_stack: list[Layer] = []

    # ------------------------------------------------------------------
    # Node CRUD
    # ------------------------------------------------------------------

    def get_or_create_node(
        self,
        method_name: str,
        args_hash: int,
        node_type: NodeType,
        compute_fn: Callable[[], Any],
        flags: NodeFlag = NodeFlag.NONE,
        get_changes_fn: Callable[..., Any] | None = None,
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
                flags=flags,
                get_changes_fn=get_changes_fn,
            )
            self.nodes[key] = node
            return node

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_node(
        self, node: Node, recursion_guard: set[NodeKey] | None = None
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

        # Check context/layer overrides first
        has_override, override_value = self._get_active_override(key)
        if has_override:
            return override_value

        # Return cached value if valid
        if node.is_valid:
            return node.value

        # Early cutoff: if this node has been computed before, verify whether
        # any input actually changed before running the (potentially expensive)
        # compute function.  Walk the recorded children, evaluate any that are
        # dirty, and check their _value_changed flag.
        if node.compute_count > 0 and node.children:
            if not self._any_child_changed(node, recursion_guard):
                node.is_valid = True
                node._value_changed = False
                return node.value

        # Save old value for change detection after recomputation
        old_value = node.value
        had_old = node.compute_count > 0

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
                        # Same-graph dependency
                        child_node.parents.add(key)
                    elif child_key[0] != self.object_id:
                        # Cross-object dependency: register in global registry
                        from calyxos.graph.registry import CrossObjectRegistry

                        registry = CrossObjectRegistry.get()
                        remote_graph = registry.get_graph(child_key[0])
                        if remote_graph is not None:
                            remote_node = remote_graph.nodes.get(child_key)
                            if remote_node is not None:
                                remote_node.parents.add(key)
                        registry.add_cross_edge(child_key, key)

                # Cache the value
                node.value = result
                node.is_valid = True
                node.compute_count += 1

                # Track whether the value actually changed (for early cutoff
                # propagation to this node's parents)
                if had_old:
                    try:
                        node._value_changed = bool(result != old_value)
                    except Exception:
                        node._value_changed = result is not old_value
                else:
                    node._value_changed = False

            return result
        finally:
            pop_frame()
            recursion_guard.discard(key)

    # ------------------------------------------------------------------
    # Early cutoff helpers
    # ------------------------------------------------------------------

    def _any_child_changed(
        self, node: Node, recursion_guard: set[NodeKey] | None
    ) -> bool:
        """Check whether any of *node*'s recorded children changed value.

        For each dirty child, evaluate it first (which recursively applies
        early cutoff).  Then inspect ``_value_changed``.  Returns ``True``
        as soon as any child is found to have changed, so the caller knows
        it must recompute.
        """
        for child_key in node.children:
            child = self.nodes.get(child_key)

            # Cross-object child — look up in the registry
            if child is None and child_key[0] != self.object_id:
                from calyxos.graph.registry import CrossObjectRegistry

                registry = CrossObjectRegistry.get()
                remote_graph = registry.get_graph(child_key[0])
                if remote_graph is not None:
                    child = remote_graph.nodes.get(child_key)

            if child is None:
                return True  # Conservative: treat missing nodes as changed

            # If the child is dirty, evaluate it (may itself early-cutoff)
            if not child.is_valid:
                if child_key[0] == self.object_id:
                    self.evaluate_node(child, recursion_guard)
                else:
                    from calyxos.graph.registry import CrossObjectRegistry

                    registry = CrossObjectRegistry.get()
                    remote_graph = registry.get_graph(child_key[0])
                    if remote_graph is not None:
                        remote_graph.evaluate_node(child, recursion_guard)
                    else:
                        return True

            # After evaluation the child is valid.  Did its value change?
            if child._value_changed:
                return True

        return False

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_node(
        self,
        method_name: str,
        args_hash: int,
        reason: str = "stored value changed",
        _visited: set[NodeKey] | None = None,
    ) -> None:
        """
        Invalidate a node and propagate invalidation to downstream dependents.

        Does not eagerly recompute; marks nodes as dirty for lazy recomputation.
        Propagates across object boundaries via the CrossObjectRegistry.
        """
        key = (self.object_id, method_name, args_hash)
        node = self.nodes.get(key)

        if node is None:
            return

        # Shared visited set across cross-object invalidation calls
        if _visited is None:
            _visited = set()

        if key in _visited:
            return
        _visited.add(key)

        with self._lock:
            # Invalidate this node
            node.is_valid = False
            node._value_changed = True
            node.last_recompute_reason = reason

            # BFS to invalidate all downstream dependents (nodes that depend on this one)
            queue = list(node.parents)

            while queue:
                parent_key = queue.pop(0)
                if parent_key in _visited:
                    continue

                parent_node = self.nodes.get(parent_key)
                if parent_node is None:
                    # Cross-object parent — don't add to _visited here;
                    # let the cross-object propagation code below handle it
                    continue

                _visited.add(parent_key)

                if parent_node.is_valid:
                    parent_node.is_valid = False
                    parent_node._value_changed = True
                    parent_node.last_recompute_reason = reason
                    queue.extend(parent_node.parents)

        # Propagate invalidation to cross-object dependents
        from calyxos.graph.registry import CrossObjectRegistry

        registry = CrossObjectRegistry.get()
        cross_parents = registry.get_cross_parents(key)
        for cross_key in cross_parents:
            if cross_key in _visited:
                continue
            remote_graph = registry.get_graph(cross_key[0])
            if remote_graph is not None:
                remote_graph.invalidate_node(
                    method_name=cross_key[1],
                    args_hash=cross_key[2],
                    reason=reason,
                    _visited=_visited,
                )

    # ------------------------------------------------------------------
    # Context system (Phase 2)
    # ------------------------------------------------------------------

    def context(self) -> "GraphContext":
        """Create a new override context.

        Usage::

            with graph.context() as ctx:
                ctx.override(obj, "spot", 105.0)
                print(obj.option_price())
            # overrides reverted
        """
        from calyxos.graph.context import GraphContext

        return GraphContext(self)

    def _push_context_frame(self) -> None:
        from calyxos.graph.context import ContextFrame

        self._context_stack.append(ContextFrame())

    def _pop_context_frame(self) -> None:
        if not self._context_stack:
            return

        frame = self._context_stack.pop()

        with self._lock:
            # Restore saved node state for every override in this frame
            for key, (saved_value, saved_valid) in frame.saved_state.items():
                nd = self.nodes.get(key)
                if nd is not None:
                    nd.value = saved_value
                    nd.is_valid = saved_valid
                    nd._value_changed = True

            # Invalidate dependents of every overridden node so they pick up
            # the restored values on next access
            for key in frame.overrides:
                nd = self.nodes.get(key)
                if nd is not None:
                    for parent_key in list(nd.parents):
                        parent_nd = self.nodes.get(parent_key)
                        if parent_nd is not None and parent_nd.is_valid:
                            self.invalidate_node(
                                parent_nd.method_name,
                                parent_nd.args_hash,
                                reason="context override reverted",
                            )

    def _find_node_by_name(self, method_name: str, args_hash: int = 0) -> Node | None:
        """Find a node by method name, with fallback to name-only search.

        If the exact (object_id, method_name, args_hash) key exists, return it.
        Otherwise search by method_name alone and return the first match.
        """
        key = (self.object_id, method_name, args_hash)
        nd = self.nodes.get(key)
        if nd is not None:
            return nd
        # Fallback: search by name (handles the case where args_hash doesn't
        # match because _compute_args_hash produces a non-zero hash even for
        # no-arg methods)
        for nd in self.nodes.values():
            if nd.method_name == method_name:
                return nd
        return None

    def _set_context_override(
        self, method_name: str, args_hash: int, value: Any
    ) -> None:
        if not self._context_stack:
            raise RuntimeError("Cannot set override outside a context block")

        nd = self._find_node_by_name(method_name, args_hash)
        if nd is None:
            raise KeyError(
                f"Node '{method_name}' (args_hash={args_hash}) does not exist. "
                "Access the node at least once before overriding."
            )
        key = nd.key()

        # Validate flag
        if not (nd.has_flag(NodeFlag.CAN_OVERRIDE) or nd.has_flag(NodeFlag.CAN_SET)):
            raise ValueError(
                f"Node '{method_name}' cannot be overridden. "
                "Use @node(NodeFlag.CAN_OVERRIDE) or @node(NodeFlag.CAN_SET)."
            )

        frame = self._context_stack[-1]

        with self._lock:
            # Save original state only on the first override of this node
            # within this frame
            if key not in frame.saved_state:
                frame.saved_state[key] = (nd.value, nd.is_valid)

            frame.overrides[key] = value
            nd.value = value
            nd.is_valid = True
            nd._value_changed = True

            # Invalidate dependents so they recompute with the override
            for parent_key in list(nd.parents):
                parent_nd = self.nodes.get(parent_key)
                if parent_nd is not None:
                    self.invalidate_node(
                        parent_nd.method_name,
                        parent_nd.args_hash,
                        reason="context override applied",
                    )

    def _get_active_override(self, key: NodeKey) -> tuple[bool, Any]:
        """Check for an active context or layer override (most recent first)."""
        # Check context stack (top-down = most recent first)
        for frame in reversed(self._context_stack):
            if key in frame.overrides:
                return True, frame.overrides[key]

        # Check layer stack
        for layer in reversed(self._layer_stack):
            if key in layer._overrides:
                return True, layer._overrides[key]

        return False, None

    @property
    def in_context(self) -> bool:
        """True if inside a context block."""
        return len(self._context_stack) > 0

    # ------------------------------------------------------------------
    # Layer system (Phase 3)
    # ------------------------------------------------------------------

    def layer(self, name: str = "") -> "Layer":
        """Create a new computation layer.

        Usage::

            L = graph.layer("bump_spot")
            with L:
                set_value(obj, "spot", 105.0)
                result = obj.option_price()
            # state preserved in L

            with L:
                result = obj.option_price()  # cached, no recompute
        """
        from calyxos.graph.layer import Layer

        return Layer(name=name, _graph=self)

    def _enter_layer(self, layer: "Layer") -> None:
        with self._lock:
            # Save entire graph state before applying layer
            if not layer._base_snapshot:
                # First entry: snapshot the current graph state
                layer._base_snapshot = {
                    key: (nd.value, nd.is_valid)
                    for key, nd in self.nodes.items()
                }

            # If the layer has a previous snapshot (re-entry), restore it
            if layer._snapshot:
                for key, (val, valid) in layer._snapshot.items():
                    nd = self.nodes.get(key)
                    if nd is not None:
                        nd.value = val
                        nd.is_valid = valid
                        nd._value_changed = True

            # Apply layer overrides
            for key, val in layer._overrides.items():
                nd = self.nodes.get(key)
                if nd is not None:
                    nd.value = val
                    nd.is_valid = True
                    nd._value_changed = True

            self._layer_stack.append(layer)

    def _exit_layer(self, layer: "Layer") -> None:
        with self._lock:
            # Save the current state as the layer's snapshot
            layer._snapshot = {
                key: (nd.value, nd.is_valid)
                for key, nd in self.nodes.items()
            }

            # Remove from stack
            if layer in self._layer_stack:
                self._layer_stack.remove(layer)

            # Restore the base snapshot
            if layer._base_snapshot:
                for key, (val, valid) in layer._base_snapshot.items():
                    nd = self.nodes.get(key)
                    if nd is not None:
                        nd.value = val
                        nd.is_valid = valid
                        nd._value_changed = True

            # Invalidate dependents of overridden nodes
            for key in layer._overrides:
                nd = self.nodes.get(key)
                if nd is not None:
                    for parent_key in list(nd.parents):
                        parent_nd = self.nodes.get(parent_key)
                        if parent_nd is not None:
                            self.invalidate_node(
                                parent_nd.method_name,
                                parent_nd.args_hash,
                                reason="layer exited",
                            )

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

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
