"""Tests for the disconnect() context manager."""

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag
from calyxos.tracking.disconnect import disconnect


class TestDisconnectBasics:
    """Test that disconnect() suppresses dependency tracking."""

    def test_disconnect_returns_correct_value(self) -> None:
        """Nodes inside disconnect still return their computed values."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 42

        m = Model()
        with disconnect():
            assert m.x() == 42

    def test_disconnect_no_dependency_created(self) -> None:
        """Accessing a node inside disconnect does not create a dependency edge."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @node()
            def y(self) -> int:
                # Normally, calling x() here would create a dependency x -> y
                # But we wrap it in disconnect to suppress that
                with disconnect():
                    _ = self.x()
                return 99

        m = Model()
        assert m.y() == 99

        # Changing x should NOT invalidate y because the dep was disconnected
        graph = get_graph(m)
        y_node = graph.get_node("y", m.y.__wrapped__.__code__.co_varnames[0] if False else 0)

        # Find the y node
        y_nodes = [n for n in graph.get_all_nodes() if n.method_name == "y"]
        assert len(y_nodes) == 1
        y_nd = y_nodes[0]

        # y should have no children (no recorded dependencies)
        assert len(y_nd.children) == 0

    def test_disconnect_does_not_affect_outside(self) -> None:
        """Dependencies tracked outside disconnect still work normally."""

        class Model:
            def __init__(self) -> None:
                self.y_count = 0

            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @node()
            def y(self) -> int:
                self.y_count += 1
                return self.x() * 2

        m = Model()
        assert m.y() == 20
        assert m.y_count == 1

        # This should still invalidate y (normal dep tracking)
        set_value(m, "x", 5)
        assert m.y() == 10
        assert m.y_count == 2

    def test_disconnect_nesting(self) -> None:
        """Nested disconnect blocks work correctly."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

        m = Model()

        with disconnect():
            assert m.x() == 10
            with disconnect():
                assert m.x() == 10
            assert m.x() == 10

    def test_disconnect_restores_stack(self) -> None:
        """The evaluation stack is properly restored after disconnect exits."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @node()
            def y(self) -> int:
                # Read x inside disconnect — should NOT be a dependency
                with disconnect():
                    _ = self.x()
                # Read x normally — SHOULD be a dependency
                return self.x() * 2

        m = Model()
        assert m.y() == 20

        # x -> y dependency should exist (from the call outside disconnect)
        y_nodes = [n for n in get_graph(m).get_all_nodes() if n.method_name == "y"]
        assert len(y_nodes) == 1
        assert len(y_nodes[0].children) == 1  # x is a child of y

        # Changing x should invalidate y
        set_value(m, "x", 5)
        assert m.y() == 10
