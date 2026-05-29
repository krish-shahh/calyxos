"""Tests for value-equality guard and early cutoff optimizations."""

from calyxos import fn, node, set_value, stored
from calyxos.core.decorator import get_graph, set_stored
from calyxos.core.flags import NodeFlag
from calyxos.graph.registry import CrossObjectRegistry


class TestValueEqualityGuard:
    """Test that set_value() short-circuits when the value hasn't changed."""

    def test_same_value_skips_invalidation(self) -> None:
        """Setting the same value should not invalidate dependents."""

        class Model:
            @stored
            def x(self) -> int:
                return 10

            @fn
            def double(self) -> int:
                return self.x() * 2

        m = Model()
        graph = get_graph(m)

        assert m.double() == 20
        double_node = next(n for n in graph.get_all_nodes() if n.method_name == "double")
        assert double_node.is_valid

        # Set the same value — dependents should remain valid
        set_value(m, "x", 10)
        assert double_node.is_valid

    def test_different_value_invalidates(self) -> None:
        """Setting a different value should still invalidate dependents."""

        class Model:
            @stored
            def x(self) -> int:
                return 10

            @fn
            def double(self) -> int:
                return self.x() * 2

        m = Model()
        graph = get_graph(m)

        assert m.double() == 20
        double_node = next(n for n in graph.get_all_nodes() if n.method_name == "double")

        set_value(m, "x", 15)
        assert not double_node.is_valid
        assert m.double() == 30

    def test_equality_guard_prevents_recomputation(self) -> None:
        """Verify compute_count doesn't increase when value is unchanged."""

        class Model:
            @stored
            def x(self) -> int:
                return 10

            @fn
            def expensive(self) -> int:
                return self.x() ** 2

        m = Model()
        graph = get_graph(m)

        assert m.expensive() == 100
        exp_node = next(n for n in graph.get_all_nodes() if n.method_name == "expensive")
        count_before = exp_node.compute_count

        # Set same value multiple times
        set_value(m, "x", 10)
        set_value(m, "x", 10)
        set_value(m, "x", 10)

        # expensive was never invalidated, so accessing it should not recompute
        assert m.expensive() == 100
        assert exp_node.compute_count == count_before

    def test_equality_guard_with_identity_check(self) -> None:
        """Same object identity should short-circuit even without __eq__."""

        class NoEq:
            def __eq__(self, other):
                raise TypeError("no equality")

        obj = NoEq()

        class Model:
            @node(NodeFlag.STORED)
            def data(self):
                return None

        m = Model()
        # Access the node first so the wrapper creates it with the right args_hash
        assert m.data() is None

        graph = get_graph(m)
        set_value(m, "data", obj)
        assert m.data() is obj

        data_node = next(n for n in graph.get_all_nodes() if n.method_name == "data")
        assert data_node.is_valid

        # Same identity — should short-circuit despite broken __eq__
        set_value(m, "data", obj)
        assert data_node.is_valid

    def test_equality_guard_with_broken_eq(self) -> None:
        """When __eq__ raises, different objects should still propagate."""

        class NoEq:
            def __init__(self, val):
                self.val = val

            def __eq__(self, other):
                raise TypeError("no equality")

        class Model:
            @node(NodeFlag.STORED)
            def data(self):
                return None

            @fn
            def derived(self):
                return self.data()

        m = Model()
        # Access node through wrapper first so it exists with the right key
        m.data()
        set_value(m, "data", NoEq(1))
        m.derived()

        graph = get_graph(m)
        derived_node = next(n for n in graph.get_all_nodes() if n.method_name == "derived")

        # Different object with broken __eq__ — should NOT short-circuit
        set_value(m, "data", NoEq(2))
        assert not derived_node.is_valid


class TestEarlyCutoff:
    """Test that early cutoff stops unnecessary recomputation cascades."""

    def test_intermediate_unchanged_stops_cascade(self) -> None:
        """When an intermediate node recomputes to the same value, downstream
        nodes should not recompute."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def raw(self) -> int:
                return 5

            @fn
            def clamped(self) -> int:
                # Clamp to [0, 10]
                return max(0, min(10, self.raw()))

            @fn
            def final(self) -> int:
                return self.clamped() * 100

        m = Model()
        graph = get_graph(m)

        assert m.final() == 500
        final_node = next(n for n in graph.get_all_nodes() if n.method_name == "final")
        count_before = final_node.compute_count

        # Change raw from 5 to 7 — clamped changes (5→7), final recomputes
        set_value(m, "raw", 7)
        assert m.final() == 700
        assert final_node.compute_count == count_before + 1

        count_before = final_node.compute_count

        # Change raw from 7 to 8 — clamped changes (7→8), final recomputes
        set_value(m, "raw", 8)
        assert m.final() == 800
        assert final_node.compute_count == count_before + 1

        count_before = final_node.compute_count

        # Change raw from 8 to 15 — clamped stays 10 (clamped), final should NOT recompute
        set_value(m, "raw", 15)
        assert m.final() == 1000  # Wait, clamped(8) = 8, clamped(15) = 10
        # Actually clamped DID change (8 → 10), so final should recompute
        assert final_node.compute_count == count_before + 1

        count_before = final_node.compute_count

        # NOW change raw from 15 to 20 — clamped stays 10, final should NOT recompute
        set_value(m, "raw", 20)
        assert m.final() == 1000
        assert final_node.compute_count == count_before  # No recomputation!

    def test_deep_chain_cutoff(self) -> None:
        """Early cutoff should propagate through a deep chain."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def input_val(self) -> int:
                return 0

            @fn
            def sign(self) -> str:
                return "pos" if self.input_val() >= 0 else "neg"

            @fn
            def label(self) -> str:
                return f"value is {self.sign()}"

            @fn
            def report(self) -> str:
                return f"Report: {self.label()}"

        m = Model()
        graph = get_graph(m)

        assert m.report() == "Report: value is pos"
        report_node = next(n for n in graph.get_all_nodes() if n.method_name == "report")
        label_node = next(n for n in graph.get_all_nodes() if n.method_name == "label")
        count_report = report_node.compute_count
        count_label = label_node.compute_count

        # Change from 0 to 5 — sign stays "pos", nothing downstream recomputes
        set_value(m, "input_val", 5)
        assert m.report() == "Report: value is pos"
        assert report_node.compute_count == count_report  # Cutoff at sign
        assert label_node.compute_count == count_label

    def test_cutoff_with_diamond_dependency(self) -> None:
        """Early cutoff in diamond pattern: A→B, A→C, B+C→D."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def a(self) -> int:
                return 10

            @fn
            def b(self) -> int:
                # Always returns 1 regardless of a
                _ = self.a()
                return 1

            @fn
            def c(self) -> int:
                return self.a() * 2

            @fn
            def d(self) -> int:
                return self.b() + self.c()

        m = Model()
        graph = get_graph(m)

        assert m.d() == 1 + 20  # 21
        d_node = next(n for n in graph.get_all_nodes() if n.method_name == "d")
        count_d = d_node.compute_count

        # Change a from 10 to 20: b still returns 1, c changes 20→40
        # Since c changed, d must recompute
        set_value(m, "a", 20)
        assert m.d() == 1 + 40  # 41
        assert d_node.compute_count == count_d + 1

    def test_cutoff_all_branches_unchanged(self) -> None:
        """When ALL branches of a diamond produce unchanged values, the join
        node should not recompute."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def a(self) -> int:
                return 10

            @fn
            def b(self) -> int:
                _ = self.a()
                return 42  # Constant

            @fn
            def c(self) -> int:
                _ = self.a()
                return 99  # Constant

            @fn
            def d(self) -> int:
                return self.b() + self.c()

        m = Model()
        graph = get_graph(m)

        assert m.d() == 42 + 99
        d_node = next(n for n in graph.get_all_nodes() if n.method_name == "d")
        count_d = d_node.compute_count

        # Both b and c are constant — d should not recompute
        set_value(m, "a", 999)
        assert m.d() == 42 + 99
        assert d_node.compute_count == count_d

    def test_cutoff_cross_object(self) -> None:
        """Early cutoff should work across object boundaries."""

        class Source:
            @node(NodeFlag.STORED)
            def val(self) -> int:
                return 5

        class Middle:
            def __init__(self, src: Source):
                self.src = src

            @fn
            def clamped(self) -> int:
                return max(0, min(10, self.src.val()))

        class Consumer:
            def __init__(self, mid: Middle):
                self.mid = mid

            @fn
            def output(self) -> int:
                return self.mid.clamped() * 10

        CrossObjectRegistry.reset()
        s = Source()
        mid = Middle(s)
        c = Consumer(mid)

        assert c.output() == 50
        output_node = next(
            n for n in get_graph(c).get_all_nodes() if n.method_name == "output"
        )
        count_before = output_node.compute_count

        # Change val to 15 — clamped goes from 5 to 10, output must recompute
        set_value(s, "val", 15)
        assert c.output() == 100
        assert output_node.compute_count == count_before + 1

        count_before = output_node.compute_count

        # Change val to 20 — clamped stays 10, output should NOT recompute
        set_value(s, "val", 20)
        assert c.output() == 100
        assert output_node.compute_count == count_before

    def setup_method(self) -> None:
        CrossObjectRegistry.reset()

    def teardown_method(self) -> None:
        CrossObjectRegistry.reset()
