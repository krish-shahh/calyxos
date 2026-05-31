"""Performance profiling and optimization hints for calyxos graphs."""

import time
from dataclasses import dataclass
from typing import Any

from calyxos.core.decorator import get_graph


@dataclass
class NodeProfile:
    """Performance metrics for a single node."""

    method_name: str
    node_type: str
    compute_count: int
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    cached_hits: int = 0

    @property
    def avg_time(self) -> float:
        if self.compute_count == 0:
            return 0.0
        return self.total_time / self.compute_count

    @property
    def cache_hit_rate(self) -> float:
        total_accesses = self.compute_count + self.cached_hits
        if total_accesses == 0:
            return 0.0
        return self.cached_hits / total_accesses


class Profiler:
    """Profile execution time and cache effectiveness of calyxos nodes.

    Use :meth:`enable` for automatic instrumentation — the profiler hooks
    into ``evaluate_node`` so every computation and cache hit is recorded
    without manual ``start_timer`` / ``stop_timer`` calls::

        prof = Profiler(obj)
        prof.enable()          # hook into the graph
        obj.expensive()        # automatically timed
        prof.print_profile_report()
        prof.disable()         # unhook
    """

    def __init__(self, obj: Any) -> None:
        self.obj = obj
        self.graph = get_graph(obj)
        self.profiles: dict[str, NodeProfile] = {}
        self._active_timers: dict[str, float] = {}
        self._enabled = False

    # ------------------------------------------------------------------
    # Automatic instrumentation
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Attach this profiler to the graph so ``evaluate_node`` records
        timing and cache hits automatically."""
        if self._enabled:
            return
        self.graph._profiler = self
        self._enabled = True

    def disable(self) -> None:
        """Detach the profiler from the graph."""
        if not self._enabled:
            return
        if self.graph._profiler is self:
            self.graph._profiler = None
        self._enabled = False

    # ------------------------------------------------------------------
    # Manual instrumentation (still available)
    # ------------------------------------------------------------------

    def start_timer(self, method_name: str) -> None:
        """Start timing a method."""
        self._active_timers[method_name] = time.perf_counter()

    def stop_timer(self, method_name: str) -> None:
        """Stop timing a method and record the result."""
        if method_name not in self._active_timers:
            return

        elapsed = time.perf_counter() - self._active_timers.pop(method_name)

        if method_name not in self.profiles:
            node = next(
                (n for n in self.graph.get_all_nodes() if n.method_name == method_name),
                None,
            )
            if node is None:
                return

            self.profiles[method_name] = NodeProfile(
                method_name=method_name,
                node_type=node.node_type.value,
                compute_count=0,
            )

        profile = self.profiles[method_name]
        profile.total_time += elapsed
        profile.min_time = min(profile.min_time, elapsed)
        profile.max_time = max(profile.max_time, elapsed)
        profile.compute_count += 1

    def record_cache_hit(self, method_name: str) -> None:
        """Record that a cached value was used."""
        if method_name not in self.profiles:
            node = next(
                (n for n in self.graph.get_all_nodes() if n.method_name == method_name),
                None,
            )
            if node is None:
                return

            self.profiles[method_name] = NodeProfile(
                method_name=method_name,
                node_type=node.node_type.value,
                compute_count=0,
            )

        self.profiles[method_name].cached_hits += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_profile(self, method_name: str) -> NodeProfile | None:
        return self.profiles.get(method_name)

    def get_all_profiles(self) -> dict[str, NodeProfile]:
        return dict(self.profiles)

    def get_optimization_hints(self) -> list[str]:
        hints: list[str] = []

        for method_name, profile in self.profiles.items():
            if profile.compute_count > 0 and profile.avg_time > 0.1:
                hints.append(
                    f"  {method_name}: High avg time ({profile.avg_time:.3f}s). "
                    f"Consider parallelizing or using external caching."
                )

            if profile.node_type == "derived" and profile.cache_hit_rate < 0.5:
                hints.append(
                    f"  {method_name}: Low cache hit rate ({profile.cache_hit_rate:.1%}). "
                    f"Dependencies change frequently."
                )

            if (
                profile.node_type == "stored"
                and profile.cached_hits > 0
                and profile.compute_count == 0
            ):
                hints.append(
                    f"  {method_name}: Well-cached stored node. "
                    f"Cache efficiency: {profile.cache_hit_rate:.1%}"
                )

        return hints

    def print_profile_report(self) -> None:
        print("\n" + "=" * 80)
        print("CALYXOS PERFORMANCE PROFILE")
        print("=" * 80)

        if not self.profiles:
            print("(No profiling data recorded)")
            return

        sorted_profiles = sorted(
            self.profiles.items(), key=lambda x: x[1].avg_time, reverse=True
        )

        print(f"\n{'Method':<20} {'Type':<10} {'Computes':<10} {'Avg Time':<12} {'Cache Hit':<12}")
        print("-" * 80)

        for method_name, profile in sorted_profiles:
            print(
                f"{method_name:<20} {profile.node_type:<10} {profile.compute_count:<10} "
                f"{profile.avg_time*1000:>8.2f}ms  {profile.cache_hit_rate:>10.1%}"
            )

        print("\n" + "=" * 80)
        print("OPTIMIZATION HINTS")
        print("=" * 80)
        hints = self.get_optimization_hints()
        if hints:
            for hint in hints:
                print(hint)
        else:
            print("  No optimization recommendations.")
        print("=" * 80)
