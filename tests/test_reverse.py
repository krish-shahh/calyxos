"""Tests for reverse propagation (get_changes / bidirectional binding)."""

import pytest

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag
from calyxos.core.reverse import NodeChange


class TestReverseBasics:
    """Test basic reverse propagation via get_changes."""

    def test_basic_reverse_propagation(self) -> None:
        """Setting TwoX to 6 sets X to 3 via get_changes."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def x(self) -> float:
                return 1.0

            @node(
                NodeFlag.CAN_SET,
                get_changes=lambda self, val: [NodeChange(self, "x", val / 2)],
            )
            def two_x(self) -> float:
                return self.x() * 2

        m = Model()
        assert m.two_x() == 2.0

        # Setting two_x should reverse-propagate to x
        set_value(m, "two_x", 6.0)

        assert m.x() == 3.0
        assert m.two_x() == 6.0

    def test_reverse_propagation_updates_downstream(self) -> None:
        """Reverse propagation causes downstream nodes to recompute correctly."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def base(self) -> float:
                return 10.0

            @node(
                NodeFlag.CAN_SET,
                get_changes=lambda self, val: [NodeChange(self, "base", val - 5)],
            )
            def plus_five(self) -> float:
                return self.base() + 5

            @node()
            def output(self) -> float:
                return self.plus_five() * 2

        m = Model()
        assert m.output() == 30.0  # (10+5)*2

        # Set plus_five to 25 -> base should become 20
        set_value(m, "plus_five", 25.0)

        assert m.base() == 20.0
        assert m.plus_five() == 25.0
        assert m.output() == 50.0  # 25*2


class TestReverseChaining:
    """Test chained reverse propagation."""

    def test_chained_reverse(self) -> None:
        """Reverse propagation chains through multiple nodes."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def a(self) -> float:
                return 1.0

            @node(
                NodeFlag.CAN_SET,
                get_changes=lambda self, val: [NodeChange(self, "a", val / 2)],
            )
            def b(self) -> float:
                return self.a() * 2

            @node(
                NodeFlag.CAN_SET,
                get_changes=lambda self, val: [NodeChange(self, "b", val / 3)],
            )
            def c(self) -> float:
                return self.b() * 3

        m = Model()
        assert m.c() == 6.0  # 1*2*3

        # Set c to 60 -> b should become 20 -> a should become 10
        set_value(m, "c", 60.0)

        assert m.a() == 10.0
        assert m.b() == 20.0
        assert m.c() == 60.0


class TestReverseEdgeCases:
    """Test error handling and edge cases."""

    def test_recursion_limit(self) -> None:
        """Infinite reverse propagation is caught by recursion limit."""

        class Model:
            @node(
                NodeFlag.CAN_SET,
                get_changes=lambda self, val: [NodeChange(self, "a", val)],
            )
            def a(self) -> float:
                return 1.0

        m = Model()
        m.a()

        with pytest.raises(RuntimeError, match="recursion limit"):
            set_value(m, "a", 99.0)

    def test_get_changes_returns_wrong_type(self) -> None:
        """get_changes returning non-NodeChange raises TypeError."""

        class Model:
            @node(
                NodeFlag.CAN_SET,
                get_changes=lambda self, val: ["not a NodeChange"],
            )
            def x(self) -> float:
                return 1.0

        m = Model()
        m.x()

        with pytest.raises(TypeError, match="NodeChange"):
            set_value(m, "x", 5.0)

    def test_reverse_with_no_get_changes_sets_directly(self) -> None:
        """Nodes without get_changes are set directly (normal behavior)."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def x(self) -> float:
                return 1.0

        m = Model()
        assert m.x() == 1.0

        set_value(m, "x", 42.0)
        assert m.x() == 42.0
