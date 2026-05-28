"""Tests for the context system (temporary overrides)."""

import pytest

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag


class TestContextBasics:
    """Test basic context override and revert behavior."""

    def test_override_and_revert(self) -> None:
        """Overriding a node in a context reverts on exit."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def spot(self) -> float:
                return 100.0

        m = Model()
        assert m.spot() == 100.0

        graph = get_graph(m)
        with graph.context() as ctx:
            ctx.override(m, "spot", 150.0)
            assert m.spot() == 150.0

        # Reverted
        assert m.spot() == 100.0

    def test_dependents_recompute_in_context(self) -> None:
        """Dependents of an overridden node recompute with the override."""

        class Model:
            def __init__(self) -> None:
                self.double_count = 0

            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 10

            @node()
            def double_x(self) -> int:
                self.double_count += 1
                return self.x() * 2

        m = Model()
        assert m.double_x() == 20
        assert m.double_count == 1

        graph = get_graph(m)
        with graph.context() as ctx:
            ctx.override(m, "x", 50)
            assert m.double_x() == 100
            assert m.double_count == 2

        # Reverted — double_x should recompute with original x
        assert m.double_x() == 20
        assert m.double_count == 3

    def test_dependents_revert_after_context(self) -> None:
        """After context exit, dependents return original values."""

        class Model:
            @node(NodeFlag.STORED)
            def base(self) -> float:
                return 1.0

            @node(NodeFlag.CAN_OVERRIDE)
            def rate(self) -> float:
                return 0.05

            @node()
            def result(self) -> float:
                return self.base() * (1 + self.rate())

        m = Model()
        assert m.result() == 1.05

        graph = get_graph(m)
        with graph.context() as ctx:
            ctx.override(m, "rate", 0.10)
            assert m.result() == 1.10

        assert m.result() == 1.05


class TestNestedContexts:
    """Test that contexts can be nested correctly."""

    def test_inner_overrides_outer(self) -> None:
        """Inner context override shadows the outer context."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        assert m.x() == 1  # Ensure node exists
        graph = get_graph(m)

        with graph.context() as outer:
            outer.override(m, "x", 10)
            assert m.x() == 10

            with graph.context() as inner:
                inner.override(m, "x", 100)
                assert m.x() == 100

            # Inner exited — outer override restored
            assert m.x() == 10

        # Outer exited — original restored
        assert m.x() == 1

    def test_nested_with_different_nodes(self) -> None:
        """Inner and outer contexts can override different nodes."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def a(self) -> int:
                return 1

            @node(NodeFlag.CAN_OVERRIDE)
            def b(self) -> int:
                return 2

            @node()
            def total(self) -> int:
                return self.a() + self.b()

        m = Model()
        assert m.total() == 3

        graph = get_graph(m)
        with graph.context() as outer:
            outer.override(m, "a", 10)
            assert m.total() == 12

            with graph.context() as inner:
                inner.override(m, "b", 20)
                assert m.total() == 30  # a=10 from outer, b=20 from inner

            # Inner exited — b reverts to 2
            assert m.total() == 12

        # Outer exited — both revert
        assert m.total() == 3


class TestContextErrors:
    """Test error handling in contexts."""

    def test_override_without_flag_raises(self) -> None:
        """Overriding a node without CAN_OVERRIDE raises ValueError."""

        class Model:
            @node()
            def x(self) -> int:
                return 1

        m = Model()
        m.x()  # Ensure node exists

        graph = get_graph(m)
        with graph.context() as ctx:
            with pytest.raises(ValueError, match="cannot be overridden"):
                ctx.override(m, "x", 99)

    def test_override_nonexistent_node_raises(self) -> None:
        """Overriding a node that hasn't been accessed raises KeyError."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        # Don't call m.x() — node doesn't exist yet

        graph = get_graph(m)
        with graph.context() as ctx:
            with pytest.raises(KeyError, match="does not exist"):
                ctx.override(m, "x", 99)

    def test_exception_inside_context_still_reverts(self) -> None:
        """If an exception is raised inside a context, overrides still revert."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()

        graph = get_graph(m)
        try:
            with graph.context() as ctx:
                ctx.override(m, "x", 999)
                assert m.x() == 999
                raise ValueError("boom")
        except ValueError:
            pass

        # Should be reverted
        assert m.x() == 1

    def test_can_set_also_allows_override(self) -> None:
        """CAN_SET flag also permits overriding in a context."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()

        graph = get_graph(m)
        with graph.context() as ctx:
            ctx.override(m, "x", 42)
            assert m.x() == 42

        assert m.x() == 1
