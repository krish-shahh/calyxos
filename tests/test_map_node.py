"""Tests for the @map_node decorator."""

from calyxos import fn, node, set_value, stored
from calyxos.core.decorator import get_graph, map_node
from calyxos.core.flags import NodeFlag
from calyxos.graph.registry import CrossObjectRegistry


class TestMapNodeBasic:
    """Basic map_node behaviour."""

    def test_simple_map(self) -> None:
        """map_node should apply the function to each element."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[str]:
                return ["a", "b", "c"]

            @map_node("items")
            def upper(self, item: str) -> str:
                return item.upper()

        p = Pipeline()
        assert p.upper() == ["A", "B", "C"]

    def test_map_with_numeric_transform(self) -> None:
        """map_node with numeric transformation."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def numbers(self) -> list[int]:
                return [1, 2, 3, 4, 5]

            @map_node("numbers")
            def squared(self, n: int) -> int:
                return n ** 2

        p = Pipeline()
        assert p.squared() == [1, 4, 9, 16, 25]

    def test_map_empty_collection(self) -> None:
        """map_node on an empty collection returns an empty list."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[str]:
                return []

            @map_node("items")
            def process(self, item: str) -> str:
                return item.upper()

        p = Pipeline()
        assert p.process() == []

    def test_map_single_element(self) -> None:
        """map_node with a single element."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[int]:
                return [42]

            @map_node("items")
            def doubled(self, n: int) -> int:
                return n * 2

        p = Pipeline()
        assert p.doubled() == [84]


class TestMapNodeCaching:
    """Test per-element caching behaviour."""

    def test_unchanged_elements_not_recomputed(self) -> None:
        """When the source collection changes, unchanged elements should
        reuse cached results."""

        compute_counts: dict[str, int] = {}

        class Pipeline:
            @node(NodeFlag.STORED)
            def docs(self) -> list[str]:
                return ["a", "b", "c"]

            @map_node("docs")
            def process(self, doc: str) -> str:
                compute_counts[doc] = compute_counts.get(doc, 0) + 1
                return doc.upper()

        p = Pipeline()
        assert p.process() == ["A", "B", "C"]
        assert compute_counts == {"a": 1, "b": 1, "c": 1}

        # Change collection: remove "b", add "d"
        set_value(p, "docs", ["a", "c", "d"])
        result = p.process()
        assert result == ["A", "C", "D"]

        # "a" and "c" should NOT have been recomputed
        assert compute_counts["a"] == 1
        assert compute_counts["c"] == 1
        # "d" is new — computed once
        assert compute_counts["d"] == 1

    def test_memoization_on_identical_reset(self) -> None:
        """Setting the same collection should not trigger any recomputation."""

        call_count = 0

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[int]:
                return [1, 2, 3]

            @map_node("items")
            def processed(self, n: int) -> int:
                nonlocal call_count
                call_count += 1
                return n * 10

        p = Pipeline()
        assert p.processed() == [10, 20, 30]
        assert call_count == 3

        # Set same collection — equality guard on set_value prevents invalidation
        set_value(p, "items", [1, 2, 3])
        assert p.processed() == [10, 20, 30]
        assert call_count == 3  # No recomputation


class TestMapNodeDependencies:
    """Test that map_node elements track their own dependencies."""

    def test_element_depends_on_shared_config(self) -> None:
        """Each element node can depend on other nodes."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def multiplier(self) -> int:
                return 2

            @node(NodeFlag.STORED)
            def numbers(self) -> list[int]:
                return [1, 2, 3]

            @map_node("numbers")
            def scaled(self, n: int) -> int:
                return n * self.multiplier()

        p = Pipeline()
        assert p.scaled() == [2, 4, 6]

        # Changing multiplier should invalidate all element nodes
        set_value(p, "multiplier", 10)
        assert p.scaled() == [10, 20, 30]

    def test_element_dependency_change_does_not_affect_others(self) -> None:
        """Changing a dependency that only affects some elements should
        leave others cached (via early cutoff on the element nodes)."""

        compute_counts: dict[int, int] = {}

        class Pipeline:
            @node(NodeFlag.STORED)
            def threshold(self) -> int:
                return 5

            @node(NodeFlag.STORED)
            def numbers(self) -> list[int]:
                return [1, 3, 7, 9]

            @map_node("numbers")
            def clamped(self, n: int) -> int:
                compute_counts[n] = compute_counts.get(n, 0) + 1
                return min(n, self.threshold())

        p = Pipeline()
        assert p.clamped() == [1, 3, 5, 5]
        assert all(v == 1 for v in compute_counts.values())

        # Raise threshold from 5 to 8
        # All elements share the multiplier dependency, so all element nodes
        # are invalidated and recomputed. But values for 1 and 3 don't change.
        set_value(p, "threshold", 8)
        result = p.clamped()
        assert result == [1, 3, 7, 8]


class TestMapNodeIntegration:
    """Integration tests combining map_node with other features."""

    def test_map_node_with_downstream_consumer(self) -> None:
        """A regular node can depend on a map_node's output."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def values(self) -> list[int]:
                return [1, 2, 3]

            @map_node("values")
            def doubled(self, n: int) -> int:
                return n * 2

            @fn
            def total(self) -> int:
                return sum(self.doubled())

        p = Pipeline()
        assert p.total() == 12  # 2 + 4 + 6

        set_value(p, "values", [10, 20])
        assert p.total() == 60  # 20 + 40

    def test_duplicate_elements(self) -> None:
        """Duplicate elements in the collection should share the same
        per-element node and return the same result."""

        call_count = 0

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[str]:
                return ["x", "x", "y"]

            @map_node("items")
            def process(self, item: str) -> str:
                nonlocal call_count
                call_count += 1
                return item.upper()

        p = Pipeline()
        result = p.process()
        assert result == ["X", "X", "Y"]
        # "x" should only be computed once (same element hash)
        assert call_count == 2  # "x" once, "y" once

    def test_map_node_metadata(self) -> None:
        """map_node should set the expected metadata attributes."""

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[int]:
                return [1]

            @map_node("items")
            def process(self, n: int) -> int:
                return n

        assert hasattr(Pipeline.process, "_calyxos_node")
        assert Pipeline.process._calyxos_node is True
        assert hasattr(Pipeline.process, "_calyxos_map")
        assert Pipeline.process._calyxos_map is True

    def setup_method(self) -> None:
        CrossObjectRegistry.reset()

    def teardown_method(self) -> None:
        CrossObjectRegistry.reset()
