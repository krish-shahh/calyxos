"""Context manager to temporarily disable dependency tracking."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from calyxos.tracking.context import _evaluation_stack


@contextmanager
def disconnect() -> Iterator[None]:
    """Temporarily disable dependency tracking.

    Nodes accessed inside this block still evaluate and return their values
    normally, but no parent-child edges are recorded.  This is useful for
    reading node values for side-effects (logging, pretty-printing) without
    creating spurious dependencies.

    Usage::

        with disconnect():
            print(f"Current spot: {obj.spot()}")  # no dependency created
    """
    saved_stack = _evaluation_stack.get()
    _evaluation_stack.set([])
    try:
        yield
    finally:
        _evaluation_stack.set(saved_stack)
