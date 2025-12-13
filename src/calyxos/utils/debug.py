"""Debug utilities for inspecting and tracing Talos graphs."""

from typing import Any

from calyxos.core.decorator import get_graph
from calyxos.graph.node import NodeType


class GraphDebugger:
    """Utilities for introspecting and debugging computation graphs."""

    def __init__(self, obj: Any) -> None:
        """Initialize debugger for an object."""
        self.obj = obj
        self.graph = get_graph(obj)

    def print_graph(self) -> None:
        """Print a human-readable representation of the graph."""
        print(f"\nComputation Graph for {self.obj.__class__.__name__} (id={id(self.obj)})")
        print("=" * 70)

        nodes = self.graph.get_all_nodes()
        if not nodes:
            print("  (empty)")
            return

        # Group nodes by type
        stored_nodes = [n for n in nodes if n.node_type == NodeType.STORED]
        derived_nodes = [n for n in nodes if n.node_type == NodeType.DERIVED]

        if stored_nodes:
            print("\nSTORED NODES:")
            for node in stored_nodes:
                status = "valid" if node.is_valid else "INVALID"
                print(
                    f"  {node.method_name} (args_hash={node.args_hash}): "
                    f"{status}, value={node.value!r}, "
                    f"compute_count={node.compute_count}"
                )

        if derived_nodes:
            print("\nDERIVED NODES:")
            for node in derived_nodes:
                status = "valid" if node.is_valid else "INVALID"
                print(
                    f"  {node.method_name} (args_hash={node.args_hash}): "
                    f"{status}, value={node.value!r}, "
                    f"compute_count={node.compute_count}"
                )
                if node.last_recompute_reason:
                    print(f"    reason: {node.last_recompute_reason}")

        print("\nDEPENDENCY EDGES:")
        for node in nodes:
            if node.children:
                print(f"  {node.method_name} depends on:")
                for child_key in node.children:
                    child_node = self.graph.nodes.get(child_key)
                    if child_node:
                        print(f"    - {child_node.method_name}")

        invalid_count = sum(1 for n in nodes if not n.is_valid)
        print(f"\nSUMMARY: {len(nodes)} total, {invalid_count} invalid, "
              f"{len(stored_nodes)} stored, {len(derived_nodes)} derived")

    def get_recompute_trace(self, method_name: str) -> list[tuple[str, str]]:
        """
        Get trace of why a method needs recomputation.

        Returns list of (node_name, reason) tuples showing the recomputation chain.
        """
        trace: list[tuple[str, str]] = []

        def walk(node_name: str, depth: int = 0) -> None:
            node = next(
                (n for n in self.graph.get_all_nodes() if n.method_name == node_name), None
            )
            if node is None:
                return

            if not node.is_valid:
                reason = node.last_recompute_reason or "unknown"
                trace.append(("  " * depth + node_name, reason))

                for child_key in node.children:
                    child_node = self.graph.nodes.get(child_key)
                    if child_node and child_node.method_name != node_name:
                        walk(child_node.method_name, depth + 1)

        walk(method_name)
        return trace

    def get_node_info(self, method_name: str) -> dict[str, Any]:
        """Get detailed information about a node."""
        node = next(
            (n for n in self.graph.get_all_nodes() if n.method_name == method_name), None
        )

        if node is None:
            return {}

        parent_names = []
        for k in node.parents:
            parent_node = self.graph.nodes.get(k)
            if parent_node is not None:
                parent_names.append(parent_node.method_name)

        child_names = []
        for k in node.children:
            child_node = self.graph.nodes.get(k)
            if child_node is not None:
                child_names.append(child_node.method_name)

        return {
            "method_name": node.method_name,
            "type": node.node_type.value,
            "is_valid": node.is_valid,
            "value": node.value,
            "compute_count": node.compute_count,
            "last_recompute_reason": node.last_recompute_reason,
            "parents": parent_names,
            "children": child_names,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        nodes = self.graph.get_all_nodes()
        stored = [n for n in nodes if n.node_type == NodeType.STORED]
        derived = [n for n in nodes if n.node_type == NodeType.DERIVED]
        invalid = [n for n in nodes if not n.is_valid]

        total_computes = sum(n.compute_count for n in nodes)

        return {
            "total_nodes": len(nodes),
            "stored_nodes": len(stored),
            "derived_nodes": len(derived),
            "invalid_nodes": len(invalid),
            "total_computes": total_computes,
            "avg_computes_per_node": (
                total_computes / len(nodes) if nodes else 0
            ),
        }
