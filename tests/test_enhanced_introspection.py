"""Tests for enhanced graph introspection."""

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag
from calyxos.core.introspection import get_node_flags, is_overridden, is_set
from calyxos.utils.debug import GraphDebugger


class TestEnhancedIntrospection:
    """Test new introspection functions."""

    def test_is_overridden_in_context(self) -> None:
        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()
        assert not is_overridden(m, "x")

        graph = get_graph(m)
        with graph.context() as ctx:
            ctx.override(m, "x", 99)
            assert is_overridden(m, "x")

        assert not is_overridden(m, "x")

    def test_is_set(self) -> None:
        class Model:
            @node(NodeFlag.CAN_SET)
            def x(self) -> int:
                return 1

            @node()
            def y(self) -> int:
                return 2

        m = Model()
        m.x()
        m.y()

        assert is_set(m, "x")
        assert not is_set(m, "y")

    def test_get_node_flags(self) -> None:
        class Model:
            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

            @node(NodeFlag.STORED)
            def y(self) -> int:
                return 2

        m = Model()
        m.x()
        m.y()

        flags_x = get_node_flags(m, "x")
        assert NodeFlag.CAN_SET in flags_x
        assert NodeFlag.CAN_OVERRIDE in flags_x

        flags_y = get_node_flags(m, "y")
        assert NodeFlag.STORED in flags_y
        assert NodeFlag.CAN_SET in flags_y  # STORED implies CAN_SET


class TestGraphDebuggerEnhanced:
    """Test enhanced GraphDebugger methods."""

    def test_dump_dependency_tree(self) -> None:
        class Model:
            @node(NodeFlag.STORED)
            def a(self) -> int:
                return 1

            @node(NodeFlag.STORED)
            def b(self) -> int:
                return 2

            @node()
            def c(self) -> int:
                return self.a() + self.b()

            @node()
            def d(self) -> int:
                return self.c() * 2

        m = Model()
        m.d()

        dbg = GraphDebugger(m)
        tree = dbg.dump_dependency_tree("d")

        assert "d:" in tree
        assert "c:" in tree
        assert "a:" in tree
        assert "b:" in tree

    def test_list_computing_nodes(self) -> None:
        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

            @node()
            def y(self) -> int:
                return self.x() * 2

            @node()
            def z(self) -> int:
                return self.y() + 1

        m = Model()
        m.z()

        dbg = GraphDebugger(m)
        nodes = dbg.list_computing_nodes("z")

        assert "z" in nodes
        assert "y" in nodes
        assert "x" in nodes

    def test_get_node_status(self) -> None:
        class Model:
            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 42

        m = Model()
        m.x()

        dbg = GraphDebugger(m)
        status = dbg.get_node_status("x")

        assert status["method_name"] == "x"
        assert status["is_valid"] is True
        assert status["is_overridden"] is False
        assert status["current_value"] == 42
        assert NodeFlag.CAN_SET in status["flags"]

    def test_get_node_status_with_override(self) -> None:
        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()
        graph = get_graph(m)
        dbg = GraphDebugger(m)

        with graph.context() as ctx:
            ctx.override(m, "x", 999)
            status = dbg.get_node_status("x")
            assert status["is_overridden"] is True
            assert status["override_value"] == 999

    def test_get_node_status_nonexistent(self) -> None:
        class Model:
            @node()
            def x(self) -> int:
                return 1

        m = Model()
        dbg = GraphDebugger(m)
        assert dbg.get_node_status("nonexistent") == {}
