"""Node representation in the computation graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from calyxos.core.flags import NodeFlag


class NodeType(Enum):
    """Type of computation node."""

    STORED = "stored"
    DERIVED = "derived"


@dataclass
class Node:
    """Represents a single computation node in the graph."""

    object_id: int
    method_name: str
    args_hash: int
    node_type: NodeType
    compute_fn: Callable[[], Any]

    # Node flags (CAN_SET, CAN_OVERRIDE, STORED)
    flags: NodeFlag = NodeFlag.NONE

    # Cached value
    value: Any = None
    is_valid: bool = False

    # Dependency tracking
    parents: set[tuple[int, str, int]] = field(default_factory=set)
    children: set[tuple[int, str, int]] = field(default_factory=set)

    # For debug/trace
    compute_count: int = 0
    last_recompute_reason: str | None = None

    # Reverse propagation callback (set via @node(get_changes=...))
    get_changes_fn: Callable[..., Any] | None = None

    def has_flag(self, flag: NodeFlag) -> bool:
        """Check if this node has the given flag."""
        return bool(self.flags & flag)

    def __hash__(self) -> int:
        """Hash based on node identity."""
        return hash((self.object_id, self.method_name, self.args_hash))

    def __eq__(self, other: Any) -> bool:
        """Equality based on node identity."""
        if not isinstance(other, Node):
            return False
        return (
            self.object_id == other.object_id
            and self.method_name == other.method_name
            and self.args_hash == other.args_hash
        )

    def __repr__(self) -> str:
        return (
            f"Node(obj_id={self.object_id}, method={self.method_name}, "
            f"args_hash={self.args_hash}, type={self.node_type.value}, "
            f"valid={self.is_valid}, parents={len(self.parents)}, "
            f"children={len(self.children)})"
        )

    def key(self) -> tuple[int, str, int]:
        """Return the unique key for this node."""
        return (self.object_id, self.method_name, self.args_hash)
