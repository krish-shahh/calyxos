"""Tests for the unified @node decorator and NodeFlag system."""

import pytest

from calyxos import fn, stored
from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag
from calyxos.graph.node import NodeType


class TestNodeDecoratorBasics:
    """Test that @node() behaves like @fn and @node(STORED) like @stored."""

    def test_node_no_flags_is_computed(self) -> None:
        """@node() creates a pure computed node, equivalent to @fn."""

        class Counter:
            def __init__(self) -> None:
                self.call_count = 0

            @node()
            def value(self) -> int:
                self.call_count += 1
                return 42

        c = Counter()
        assert c.value() == 42
        assert c.call_count == 1

        # Second call should be cached
        assert c.value() == 42
        assert c.call_count == 1

    def test_node_stored_flag(self) -> None:
        """@node(NodeFlag.STORED) creates a stored node, equivalent to @stored."""

        class Account:
            @node(NodeFlag.STORED)
            def balance(self) -> float:
                return 100.0

        acct = Account()
        assert acct.balance() == 100.0

        graph = get_graph(acct)
        stored_nodes = graph.get_stored_nodes()
        assert len(stored_nodes) == 1
        assert stored_nodes[0].method_name == "balance"
        assert stored_nodes[0].node_type == NodeType.STORED

    def test_node_stored_implies_can_set(self) -> None:
        """STORED flag implies CAN_SET."""

        class Account:
            @node(NodeFlag.STORED)
            def balance(self) -> float:
                return 100.0

        acct = Account()
        acct.balance()

        graph = get_graph(acct)
        nd = graph.get_stored_nodes()[0]
        assert nd.has_flag(NodeFlag.CAN_SET)
        assert nd.has_flag(NodeFlag.STORED)

    def test_node_can_set_allows_set_value(self) -> None:
        """CAN_SET flag allows set_value() to modify the node."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def threshold(self) -> float:
                return 0.5

        m = Model()
        assert m.threshold() == 0.5

        set_value(m, "threshold", 0.8)
        assert m.threshold() == 0.8

    def test_node_stored_allows_set_value(self) -> None:
        """STORED flag allows set_value()."""

        class Model:
            @node(NodeFlag.STORED)
            def learning_rate(self) -> float:
                return 0.01

        m = Model()
        assert m.learning_rate() == 0.01

        set_value(m, "learning_rate", 0.001)
        assert m.learning_rate() == 0.001

    def test_set_value_rejects_pure_computed(self) -> None:
        """set_value() raises ValueError for pure computed nodes without CAN_SET."""

        class Model:
            @node()
            def output(self) -> int:
                return 42

        m = Model()
        m.output()  # Create the node

        with pytest.raises(ValueError, match="CAN_SET"):
            set_value(m, "output", 99)

    def test_node_can_override_flag(self) -> None:
        """CAN_OVERRIDE flag is stored on the node."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def rate(self) -> float:
                return 0.05

        m = Model()
        m.rate()

        graph = get_graph(m)
        nodes = graph.get_all_nodes()
        assert len(nodes) == 1
        assert nodes[0].has_flag(NodeFlag.CAN_OVERRIDE)

    def test_combined_flags(self) -> None:
        """Multiple flags can be combined."""

        class Model:
            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def rate(self) -> float:
                return 0.05

        m = Model()
        m.rate()

        graph = get_graph(m)
        nd = graph.get_all_nodes()[0]
        assert nd.has_flag(NodeFlag.CAN_SET)
        assert nd.has_flag(NodeFlag.CAN_OVERRIDE)
        assert not nd.has_flag(NodeFlag.STORED)


class TestNodeMemoization:
    """Test that @node preserves memoization and argument handling."""

    def test_node_with_args(self) -> None:
        """@node() with arguments creates separate cached instances."""

        class Pricer:
            def __init__(self) -> None:
                self.call_count = 0

            @node()
            def price(self, currency: str) -> float:
                self.call_count += 1
                return {"USD": 100.0, "EUR": 85.0}.get(currency, 0.0)

        p = Pricer()
        assert p.price("USD") == 100.0
        assert p.call_count == 1

        assert p.price("EUR") == 85.0
        assert p.call_count == 2

        # Cached
        assert p.price("USD") == 100.0
        assert p.call_count == 2

    def test_node_with_kwargs(self) -> None:
        """@node() handles keyword arguments correctly."""

        class Calc:
            def __init__(self) -> None:
                self.call_count = 0

            @node()
            def compute(self, a: int = 1, b: int = 2) -> int:
                self.call_count += 1
                return a + b

        c = Calc()
        assert c.compute(a=10, b=20) == 30
        assert c.call_count == 1

        assert c.compute(a=10, b=20) == 30
        assert c.call_count == 1


class TestNodeDependencyTracking:
    """Test that @node tracks dependencies and invalidates correctly."""

    def test_invalidation_propagates(self) -> None:
        """Changing a stored node invalidates dependent computed nodes."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @node()
            def double_x(self) -> int:
                return self.x() * 2

        m = Model()
        assert m.double_x() == 20

        set_value(m, "x", 5)
        assert m.double_x() == 10

    def test_selective_invalidation(self) -> None:
        """Only affected nodes are invalidated."""

        class Model:
            def __init__(self) -> None:
                self.a_count = 0
                self.b_count = 0

            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

            @node(NodeFlag.STORED)
            def y(self) -> int:
                return 2

            @node()
            def a(self) -> int:
                self.a_count += 1
                return self.x() * 10

            @node()
            def b(self) -> int:
                self.b_count += 1
                return self.y() * 10

        m = Model()
        assert m.a() == 10
        assert m.b() == 20
        assert m.a_count == 1
        assert m.b_count == 1

        # Changing x should only invalidate a, not b
        set_value(m, "x", 5)
        assert m.a() == 50
        assert m.a_count == 2

        # b should still be cached
        assert m.b() == 20
        assert m.b_count == 1


class TestBackwardCompatibility:
    """Verify that @fn and @stored still work exactly as before."""

    def test_fn_still_works(self) -> None:
        class Counter:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def value(self) -> int:
                self.call_count += 1
                return 42

        c = Counter()
        assert c.value() == 42
        assert c.call_count == 1
        assert c.value() == 42
        assert c.call_count == 1

    def test_stored_still_works(self) -> None:
        from calyxos.core.decorator import set_stored

        class Account:
            @stored
            def balance(self) -> float:
                return 100.0

        acct = Account()
        assert acct.balance() == 100.0

        set_stored(acct, "balance", 200.0)
        assert acct.balance() == 200.0

    def test_fn_and_node_interop(self) -> None:
        """@fn and @node() can coexist on the same class."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @fn
            def double_x(self) -> int:
                return self.x() * 2

        m = Model()
        assert m.double_x() == 20

        set_value(m, "x", 5)
        assert m.double_x() == 10
