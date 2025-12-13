"""Tests for the @fn and @stored decorators."""

from calyxos import fn, stored
from calyxos.core.decorator import get_graph, set_stored


class TestDecoratorBasics:
    """Test basic decorator functionality."""

    def test_fn_decorator_memoization(self) -> None:
        """Test that @fn memoizes results."""

        class Counter:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def get_value(self) -> int:
                self.call_count += 1
                return 42

        counter = Counter()
        assert counter.get_value() == 42
        assert counter.call_count == 1

        # Second call should use cached value
        assert counter.get_value() == 42
        assert counter.call_count == 1

    def test_fn_decorator_with_args(self) -> None:
        """Test that @fn handles arguments correctly."""

        class Multiplier:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def multiply(self, a: int, b: int) -> int:
                self.call_count += 1
                return a * b

        m = Multiplier()
        assert m.multiply(3, 4) == 12
        assert m.call_count == 1

        # Same args should use cache
        assert m.multiply(3, 4) == 12
        assert m.call_count == 1

        # Different args should recompute
        assert m.multiply(2, 5) == 10
        assert m.call_count == 2

    def test_stored_decorator_basic(self) -> None:
        """Test that @stored marks nodes as stored."""

        class Account:
            @stored
            def balance(self) -> float:
                return 100.0

        account = Account()
        graph = get_graph(account)

        assert account.balance() == 100.0

        stored_nodes = graph.get_stored_nodes()
        assert len(stored_nodes) == 1
        assert stored_nodes[0].method_name == "balance"

    def test_set_stored_modifies_value(self) -> None:
        """Test that set_stored modifies stored values."""

        class Account:
            @stored
            def balance(self) -> float:
                return 100.0

        account = Account()

        assert account.balance() == 100.0

        set_stored(account, "balance", 200.0)

        assert account.balance() == 200.0

    def test_multiple_instances_independent_graphs(self) -> None:
        """Test that different instances have independent graphs."""

        class Counter:
            def __init__(self, initial: int) -> None:
                self.initial = initial

            @fn
            def get_value(self) -> int:
                return self.initial

        c1 = Counter(10)
        c2 = Counter(20)

        assert c1.get_value() == 10
        assert c2.get_value() == 20

        graph1 = get_graph(c1)
        graph2 = get_graph(c2)

        assert graph1 is not graph2
        assert id(c1) == graph1.object_id
        assert id(c2) == graph2.object_id

    def test_fn_with_kwargs(self) -> None:
        """Test @fn with keyword arguments."""

        class Calc:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def compute(self, a: int = 1, b: int = 2) -> int:
                self.call_count += 1
                return a + b

        calc = Calc()

        assert calc.compute(a=10, b=20) == 30
        assert calc.call_count == 1

        # Same kwargs
        assert calc.compute(a=10, b=20) == 30
        assert calc.call_count == 1

        # Different kwargs
        assert calc.compute(a=5, b=5) == 10
        assert calc.call_count == 2

    def teardown_method(self) -> None:
        """Clean up after each test."""
        # Note: In a real implementation, you'd have a better cleanup mechanism
        pass


class TestDecoratorEdgeCases:
    """Test edge cases and error conditions."""

    def test_fn_with_unhashable_args(self) -> None:
        """Test @fn with unhashable argument types."""

        class Processor:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def process(self, data: list[int]) -> int:
                self.call_count += 1
                return sum(data)

        proc = Processor()

        # Should work even with unhashable args (list)
        assert proc.process([1, 2, 3]) == 6
        assert proc.call_count == 1

    def test_fn_with_none_args(self) -> None:
        """Test @fn with None arguments."""

        class Handler:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def handle(self, value: int | None = None) -> str:
                self.call_count += 1
                return "none" if value is None else str(value)

        handler = Handler()

        assert handler.handle(None) == "none"
        assert handler.handle(None) == "none"
        assert handler.call_count == 1

        assert handler.handle(42) == "42"
        assert handler.call_count == 2
