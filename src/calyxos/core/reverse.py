"""Reverse propagation support for bidirectional binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NodeChange:
    """Describes a value change to propagate upstream via reverse propagation.

    When a node with a ``get_changes`` callback has ``set_value()`` called on it,
    the callback returns a list of ``NodeChange`` instances indicating which
    upstream nodes should actually be modified.
    """

    target_obj: Any
    target_method: str
    value: Any
    args_hash: int = 0
