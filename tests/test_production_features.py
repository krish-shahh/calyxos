"""Tests for production features: error handling, TTL, profiler, executor, map GC, async."""

import asyncio
import time

from calyxos import fn, node, set_value, stored, get_graph, map_node
from calyxos.core.decorator import clear_graph
from calyxos.core.flags import NodeFlag
from calyxos.graph.registry import CrossObjectRegistry
from calyxos.utils.profiler import Profiler
from calyxos.utils.distributed import DistributedExecutor


# ======================================================================
# Error handling
# ======================================================================


class TestErrorHandling:
    """Test that compute_fn exceptions are captured and re-raised cleanly."""

    def test_error_is_captured(self) -> None:
        class Model:
            @node()
            def broken(self) -> int:
                raise ValueError("boom")

        m = Model()
        graph = get_graph(m)

        try:
            m.broken()
        except ValueError:
            pass

        broken_node = next(n for n in graph.get_all_nodes() if n.method_name == "broken")
        assert broken_node.error is not None
        assert isinstance(broken_node.error, ValueError)

    def test_error_re_raises_on_access(self) -> None:
        class Model:
            @node()
            def broken(self) -> int:
                raise RuntimeError("fail")

        m = Model()
        try:
            m.broken()
        except RuntimeError:
            pass

        # Second access should re-raise the stored error
        try:
            m.broken()
            assert False, "Should have raised"
        except RuntimeError as e:
            assert str(e) == "fail"

    def test_retry_clears_error(self) -> None:
        call_count = 0

        class Model:
            @node()
            def flaky(self) -> int:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("transient")
                return 42

        m = Model()
        graph = get_graph(m)

        try:
            m.flaky()
        except RuntimeError:
            pass

        # Retry
        graph.retry_errors()
        assert m.flaky() == 42
        assert call_count == 2

    def test_get_error_nodes(self) -> None:
        class Model:
            @node()
            def ok(self) -> int:
                return 1

            @node()
            def broken(self) -> int:
                raise ValueError("bad")

        m = Model()
        graph = get_graph(m)
        m.ok()
        try:
            m.broken()
        except ValueError:
            pass

        errors = graph.get_error_nodes()
        assert len(errors) == 1
        assert errors[0].method_name == "broken"


# ======================================================================
# TTL / cache eviction
# ======================================================================


class TestTTL:
    """Test time-to-live cache expiry."""

    def test_ttl_expires_value(self) -> None:
        call_count = 0

        class Model:
            @node(ttl=0.05)  # 50ms TTL
            def data(self) -> int:
                nonlocal call_count
                call_count += 1
                return call_count

        m = Model()
        assert m.data() == 1
        assert m.data() == 1  # cached

        time.sleep(0.06)  # wait for expiry

        assert m.data() == 2  # recomputed

    def test_evict_expired(self) -> None:
        class Model:
            @node(ttl=0.01)
            def fast(self) -> int:
                return 1

        m = Model()
        graph = get_graph(m)
        m.fast()
        time.sleep(0.02)
        count = graph.evict_expired()
        assert count == 1


# ======================================================================
# Profiler auto-instrumentation
# ======================================================================


class TestProfiler:
    """Test automatic profiler instrumentation."""

    def test_enable_records_timing(self) -> None:
        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @node()
            def doubled(self) -> int:
                return self.x() * 2

        m = Model()
        prof = Profiler(m)
        prof.enable()

        m.doubled()

        prof.disable()

        profile = prof.get_profile("doubled")
        assert profile is not None
        assert profile.compute_count == 1
        assert profile.total_time > 0

    def test_enable_records_cache_hits(self) -> None:
        class Model:
            @node()
            def val(self) -> int:
                return 42

        m = Model()
        prof = Profiler(m)
        prof.enable()

        m.val()  # compute
        m.val()  # cache hit

        prof.disable()

        profile = prof.get_profile("val")
        assert profile is not None
        assert profile.compute_count == 1
        assert profile.cached_hits == 1


# ======================================================================
# Distributed executor
# ======================================================================


class TestDistributedExecutor:
    """Test real parallel execution."""

    def test_execute_returns_results(self) -> None:
        class Model:
            @node(NodeFlag.STORED)
            def a(self) -> int:
                return 1

            @node()
            def b(self) -> int:
                return self.a() + 1

            @node()
            def c(self) -> int:
                return self.b() + 1

        m = Model()
        # Evaluate once to build the graph
        m.c()

        executor = DistributedExecutor(m, workers=2)
        results = executor.execute()

        assert "a" in results
        assert "b" in results
        assert "c" in results
        assert results["c"] == 3


# ======================================================================
# Map node GC
# ======================================================================


class TestMapNodeGC:
    """Test that stale per-element nodes are garbage collected."""

    def test_removed_elements_are_gc_ed(self) -> None:
        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[str]:
                return ["a", "b", "c"]

            @map_node("items")
            def upper(self, item: str) -> str:
                return item.upper()

        p = Pipeline()
        graph = get_graph(p)

        assert p.upper() == ["A", "B", "C"]

        # 3 element nodes + 1 orchestrator + 1 source = 5
        map_nodes_before = [n for n in graph.get_all_nodes() if n.method_name.startswith("_map_")]
        assert len(map_nodes_before) == 3

        # Remove "b" and "c"
        set_value(p, "items", ["a"])
        assert p.upper() == ["A"]

        # "b" and "c" element nodes should have been GC'd
        map_nodes_after = [n for n in graph.get_all_nodes() if n.method_name.startswith("_map_")]
        assert len(map_nodes_after) == 1

    def test_gc_orphan_map_nodes_manual(self) -> None:
        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[int]:
                return [1, 2, 3]

            @map_node("items")
            def squared(self, n: int) -> int:
                return n ** 2

        p = Pipeline()
        graph = get_graph(p)
        p.squared()

        # Manual GC should find nothing (all nodes are referenced)
        assert graph.gc_orphan_map_nodes() == 0


# ======================================================================
# Async evaluation
# ======================================================================


class TestAsyncEvaluation:
    """Test async graph evaluation."""

    def test_async_fn_basic(self) -> None:
        from calyxos import async_fn

        class Model:
            @async_fn
            async def compute(self) -> int:
                await asyncio.sleep(0.001)
                return 42

        m = Model()
        result = asyncio.run(m.compute())
        assert result == 42

    def test_async_fn_caches(self) -> None:
        from calyxos import async_fn

        call_count = 0

        class Model:
            @async_fn
            async def compute(self) -> int:
                nonlocal call_count
                call_count += 1
                return 99

        m = Model()
        r1 = asyncio.run(m.compute())
        r2 = asyncio.run(m.compute())
        assert r1 == r2 == 99
        assert call_count == 1

    def test_async_map_node(self) -> None:
        from calyxos import async_map_node

        class Pipeline:
            @node(NodeFlag.STORED)
            def items(self) -> list[int]:
                return [1, 2, 3]

            @async_map_node("items")
            async def doubled(self, n: int) -> int:
                await asyncio.sleep(0.001)
                return n * 2

        p = Pipeline()
        result = asyncio.run(p.doubled())
        assert result == [2, 4, 6]

    def setup_method(self) -> None:
        CrossObjectRegistry.reset()

    def teardown_method(self) -> None:
        CrossObjectRegistry.reset()
