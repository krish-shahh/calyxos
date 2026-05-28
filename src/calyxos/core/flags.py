"""Node flags for controlling node behavior in the computation graph."""

from enum import Flag, auto


class NodeFlag(Flag):
    """Flags that control how a node behaves in the computation graph.

    Flags can be combined: ``NodeFlag.CAN_SET | NodeFlag.CAN_OVERRIDE``.

    - ``CAN_SET``: The node's value can be explicitly set via ``set_value()``.
    - ``CAN_OVERRIDE``: The node's value can be temporarily overridden in a context or layer.
    - ``STORED``: The node is persistent (implies CAN_SET). Values are saved via storage backends.
    """

    NONE = 0
    CAN_SET = auto()
    CAN_OVERRIDE = auto()
    STORED = auto()


# Convenience aliases for the public API
CanSet = NodeFlag.CAN_SET
CanOverride = NodeFlag.CAN_OVERRIDE
Stored = NodeFlag.STORED
