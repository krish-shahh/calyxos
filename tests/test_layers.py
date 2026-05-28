"""Tests for the layer system (persistent computation state)."""

import pytest

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag


class TestLayerBasics:
    """Test basic layer create/enter/exit behavior."""

    def test_layer_preserves_computation(self) -> None:
        """Computed values inside a layer are cached for re-entry."""

        class Model:
            def __init__(self) -> None:
                self.compute_count = 0

            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def spot(self) -> float:
                return 100.0

            @node()
            def price(self) -> int:
                self.compute_count += 1
                return int(self.spot()) * 2

        m = Model()
        assert m.price() == 200
        assert m.compute_count == 1

        graph = get_graph(m)
        layer = graph.layer("bump")

        with layer:
            set_value(m, "spot", 105.0)
            assert m.price() == 210
            assert m.compute_count == 2

        # After exit, original state restored (from base snapshot, no recompute)
        assert m.price() == 200
        assert m.compute_count == 2  # Restored from snapshot, not recomputed

        # Re-enter: cached layer state, no recompute
        with layer:
            assert m.price() == 210
            assert m.compute_count == 2  # Still NOT recomputed

    def test_layer_exit_restores_original(self) -> None:
        """Exiting a layer restores the pre-layer state."""

        class Model:
            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 10

        m = Model()
        assert m.x() == 10

        graph = get_graph(m)
        layer = graph.layer("test")

        with layer:
            set_value(m, "x", 99)
            assert m.x() == 99

        assert m.x() == 10

    def test_layer_with_set_method(self) -> None:
        """Layer.set() applies overrides that persist across re-entries."""

        class Model:
            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 10

            @node()
            def double_x(self) -> int:
                return self.x() * 2

        m = Model()
        assert m.double_x() == 20

        graph = get_graph(m)
        layer = graph.layer("test")

        with layer:
            layer.set("x", 50)
            assert m.double_x() == 100

        # Original restored
        assert m.double_x() == 20

        # Re-enter: layer override still applies
        with layer:
            assert m.double_x() == 100


class TestLayerEdgeCases:
    """Test edge cases for layers."""

    def test_double_enter_raises(self) -> None:
        """Entering a layer that is already active raises RuntimeError."""

        class Model:
            @node(NodeFlag.CAN_SET)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()

        graph = get_graph(m)
        layer = graph.layer("test")

        with layer:
            with pytest.raises(RuntimeError, match="already active"):
                with layer:
                    pass

    def test_two_independent_layers(self) -> None:
        """Two layers maintain independent state."""

        class Model:
            def __init__(self) -> None:
                self.count = 0

            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 10

            @node()
            def result(self) -> int:
                self.count += 1
                return self.x() * 2

        m = Model()
        assert m.result() == 20

        graph = get_graph(m)
        layer_a = graph.layer("bump_up")
        layer_b = graph.layer("bump_down")

        # Layer A: bump x up
        with layer_a:
            set_value(m, "x", 50)
            assert m.result() == 100

        # Layer B: bump x down
        with layer_b:
            set_value(m, "x", 1)
            assert m.result() == 2

        # Re-enter each layer: cached
        count_before = m.count
        with layer_a:
            assert m.result() == 100
            assert m.count == count_before  # No recompute

        with layer_b:
            assert m.result() == 2

    def test_context_inside_layer(self) -> None:
        """A context can be nested inside a layer."""

        class Model:
            @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 10

            @node()
            def result(self) -> int:
                return self.x() * 3

        m = Model()
        assert m.result() == 30

        graph = get_graph(m)
        layer = graph.layer("test")

        with layer:
            set_value(m, "x", 20)
            assert m.result() == 60

            with graph.context() as ctx:
                ctx.override(m, "x", 100)
                assert m.result() == 300

            # Context exited, layer's value restored
            assert m.result() == 60

        # Layer exited, original restored
        assert m.result() == 30
