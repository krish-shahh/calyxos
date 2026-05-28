"""Tests for cross-object dependency tracking."""

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag
from calyxos.graph.registry import CrossObjectRegistry


class TestCrossObjectBasics:
    """Test cross-object dependency tracking and invalidation."""

    def setup_method(self) -> None:
        CrossObjectRegistry.reset()

    def test_cross_object_dependency(self) -> None:
        """Object A depends on Object B's node. Changing B invalidates A."""

        class Market:
            @node(NodeFlag.STORED)
            def spot(self) -> float:
                return 100.0

        class Option:
            def __init__(self, market: Market) -> None:
                self.market = market
                self.compute_count = 0

            @node()
            def price(self) -> float:
                self.compute_count += 1
                return self.market.spot() * 1.1

        mkt = Market()
        opt = Option(mkt)

        # First evaluation: establishes cross-object dependency
        price = opt.price()
        assert price > 0
        assert opt.compute_count == 1

        # Cached on second access
        _ = opt.price()
        assert opt.compute_count == 1

        # Change market's spot — should invalidate option's price
        set_value(mkt, "spot", 200.0)
        new_price = opt.price()
        assert opt.compute_count == 2
        assert new_price > price

    def test_cross_object_chain(self) -> None:
        """A -> B -> C dependency chain across objects."""

        class Source:
            @node(NodeFlag.STORED)
            def value(self) -> int:
                return 10

        class Middle:
            def __init__(self, source: "Source") -> None:
                self.source = source

            @node()
            def doubled(self) -> int:
                return self.source.value() * 2

        class Consumer:
            def __init__(self, middle: "Middle") -> None:
                self.middle = middle
                self.count = 0

            @node()
            def result(self) -> int:
                self.count += 1
                return self.middle.doubled() + 1

        src = Source()
        mid = Middle(src)
        con = Consumer(mid)

        assert con.result() == 21  # 10*2+1
        assert con.count == 1

        # Change source — should propagate through middle to consumer
        set_value(src, "value", 50)
        assert con.result() == 101  # 50*2+1
        assert con.count == 2

    def test_cross_object_selective_invalidation(self) -> None:
        """Only nodes depending on the changed cross-object node invalidate."""

        class DataA:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

        class DataB:
            @node(NodeFlag.STORED)
            def y(self) -> int:
                return 2

        class Consumer:
            def __init__(self, a: "DataA", b: "DataB") -> None:
                self.a = a
                self.b = b
                self.from_a_count = 0
                self.from_b_count = 0

            @node()
            def from_a(self) -> int:
                self.from_a_count += 1
                return self.a.x() * 10

            @node()
            def from_b(self) -> int:
                self.from_b_count += 1
                return self.b.y() * 10

        a = DataA()
        b = DataB()
        c = Consumer(a, b)

        assert c.from_a() == 10
        assert c.from_b() == 20
        assert c.from_a_count == 1
        assert c.from_b_count == 1

        # Change a's x — only from_a should invalidate
        set_value(a, "x", 5)
        assert c.from_a() == 50
        assert c.from_a_count == 2
        assert c.from_b() == 20
        assert c.from_b_count == 1  # Not recomputed

    def test_registry_cleanup(self) -> None:
        """Registry tracks graphs with weak references."""
        registry = CrossObjectRegistry.get()

        class Ephemeral:
            @node(NodeFlag.STORED)
            def val(self) -> int:
                return 42

        obj = Ephemeral()
        _ = obj.val()
        obj_id = id(obj)

        assert registry.get_graph(obj_id) is not None

        # The graph is held by _graphs dict in decorator.py, so it won't be
        # GC'd until we clear it
        from calyxos.core.decorator import _graphs
        del _graphs[obj_id]

        # Now cleanup should detect it
        registry.cleanup_dead_refs()
