"""Tests for debug utilities and introspection."""

import pytest

from talos import GraphDebugger, fn, stored
from talos.core.decorator import get_graph, set_stored


class TestGraphDebugger:
    """Test graph introspection and debugging utilities."""

    def test_print_graph_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing an empty graph."""

        class Empty:
            pass

        obj = Empty()
        debugger = GraphDebugger(obj)
        debugger.print_graph()

        captured = capsys.readouterr()
        assert "(empty)" in captured.out

    def test_print_graph_with_nodes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test printing a graph with nodes."""

        class Model:
            @stored
            def input_value(self) -> int:
                return 10

            @fn
            def output_value(self) -> int:
                return self.input_value() * 2

        model = Model()
        _ = model.output_value()

        debugger = GraphDebugger(model)
        debugger.print_graph()

        captured = capsys.readouterr()
        assert "STORED NODES:" in captured.out
        assert "DERIVED NODES:" in captured.out
        assert "input_value" in captured.out
        assert "output_value" in captured.out

    def test_get_node_info(self) -> None:
        """Test getting detailed node information."""

        class Model:
            @fn
            def compute(self) -> int:
                return 42

        model = Model()
        _ = model.compute()

        debugger = GraphDebugger(model)
        info = debugger.get_node_info("compute")

        assert info["method_name"] == "compute"
        assert info["type"] == "derived"
        assert info["is_valid"] is True
        assert info["value"] == 42
        assert info["compute_count"] == 1

    def test_get_stats(self) -> None:
        """Test getting graph statistics."""

        class Model:
            @stored
            def x(self) -> int:
                return 1

            @fn
            def y(self) -> int:
                return self.x() + 1

            @fn
            def z(self) -> int:
                return self.y() + 1

        model = Model()
        _ = model.z()

        debugger = GraphDebugger(model)
        stats = debugger.get_stats()

        assert stats["total_nodes"] == 3
        assert stats["stored_nodes"] == 1
        assert stats["derived_nodes"] == 2
        assert stats["invalid_nodes"] == 0

    def test_get_recompute_trace(self) -> None:
        """Test getting recomputation trace."""

        class Model:
            @stored
            def base(self) -> int:
                return 1

            @fn
            def level1(self) -> int:
                return self.base() + 1

            @fn
            def level2(self) -> int:
                return self.level1() + 1

        model = Model()
        _ = model.level2()

        # Invalidate and check trace
        set_stored(model, "base", 10)

        debugger = GraphDebugger(model)
        trace = debugger.get_recompute_trace("level2")

        # Should have invalidation chain
        assert len(trace) > 0


class TestNodeTracking:
    """Test that nodes track compute count and other metadata."""

    def test_compute_count(self) -> None:
        """Test that compute count is tracked correctly."""

        class Counter:
            def __init__(self) -> None:
                self.call_count = 0

            @fn
            def base_value(self) -> int:
                self.call_count += 1
                return 10

            @fn
            def get_value(self) -> int:
                return self.base_value() + 32

        counter = Counter()
        graph = get_graph(counter)

        # Initial call
        _ = counter.get_value()
        get_value_node = next(n for n in graph.get_all_nodes() if n.method_name == "get_value")
        assert get_value_node.compute_count == 1

        # Cached call
        _ = counter.get_value()
        assert get_value_node.compute_count == 1

        # After invalidation and recompute - invalidate base_value
        base_node = next(n for n in graph.get_all_nodes() if n.method_name == "base_value")
        graph.invalidate_node("base_value", base_node.args_hash)

        _ = counter.get_value()
        assert get_value_node.compute_count == 2

    def test_recompute_reason_tracked(self) -> None:
        """Test that recomputation reasons are tracked."""

        class Model:
            @stored
            def input_value(self) -> int:
                return 10

            @fn
            def output(self) -> int:
                return self.input_value() * 2

        model = Model()
        _ = model.output()

        set_stored(model, "input_value", 20)

        graph = get_graph(model)
        output_node = next(n for n in graph.get_all_nodes() if n.method_name == "output")

        assert output_node.last_recompute_reason is not None
        assert "stored value" in output_node.last_recompute_reason.lower()
