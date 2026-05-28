"""CalyxOS What-If Analysis — Contexts

Demonstrates:
- graph.context() for temporary overrides
- Nested contexts
- Automatic reversion on exit
- disconnect() for side-effect-free reads
"""

from calyxos import NodeFlag, get_graph, node, set_value
from calyxos.tracking.disconnect import disconnect


class OptionPricer:
    """A simplified option pricing model with overridable inputs."""

    @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
    def spot(self) -> float:
        return 100.0

    @node(NodeFlag.CAN_OVERRIDE)
    def volatility(self) -> float:
        return 0.20

    @node(NodeFlag.CAN_OVERRIDE)
    def rate(self) -> float:
        return 0.05

    @node()
    def call_price(self) -> float:
        """Simplified call price = spot * vol * sqrt(T) + rate adjustment."""
        return self.spot() * self.volatility() * 1.0 + self.rate() * self.spot()

    @node()
    def put_price(self) -> float:
        """Simplified put price."""
        return self.call_price() * 0.8


def demo_contexts():
    print("=" * 60)
    print("WHAT-IF ANALYSIS WITH CONTEXTS")
    print("=" * 60)

    pricer = OptionPricer()
    graph = get_graph(pricer)

    print(f"Base case:")
    print(f"  Spot={pricer.spot()}, Vol={pricer.volatility()}, Rate={pricer.rate()}")
    print(f"  Call={pricer.call_price():.2f}, Put={pricer.put_price():.2f}")

    # ── Scenario 1: Market stress ─────────────────────────────────────────
    print(f"\n--- Scenario 1: Market stress (spot drops to 80) ---")
    with graph.context() as ctx:
        ctx.override(pricer, "spot", 80.0)
        print(f"  Spot={pricer.spot()}, Call={pricer.call_price():.2f}, Put={pricer.put_price():.2f}")

    # Automatically reverted
    print(f"\nAfter context exit (reverted):")
    print(f"  Spot={pricer.spot()}, Call={pricer.call_price():.2f}")

    # ── Scenario 2: Nested what-if ────────────────────────────────────────
    print(f"\n--- Scenario 2: Nested what-if ---")
    with graph.context() as outer:
        outer.override(pricer, "spot", 120.0)
        print(f"  Outer: Spot={pricer.spot()}, Call={pricer.call_price():.2f}")

        with graph.context() as inner:
            inner.override(pricer, "volatility", 0.40)
            print(f"  Inner: Vol={pricer.volatility()}, Call={pricer.call_price():.2f}")

        print(f"  Back to outer: Vol={pricer.volatility()}, Call={pricer.call_price():.2f}")

    print(f"  Fully reverted: Spot={pricer.spot()}, Vol={pricer.volatility()}")


def demo_disconnect():
    print(f"\n{'=' * 60}")
    print("DISCONNECT — SIDE-EFFECT-FREE READS")
    print("=" * 60)

    class Logger:
        @node(NodeFlag.STORED)
        def data(self) -> int:
            return 42

        @node()
        def processed(self) -> int:
            # We want to log data without creating a dependency
            with disconnect():
                print(f"    [log] Current data value: {self.data()}")
            return 99  # Independent of data

    logger = Logger()
    print(f"  processed={logger.processed()}")

    # Changing data should NOT invalidate processed
    set_value(logger, "data", 100)
    print(f"  After changing data, processed={logger.processed()} (still cached)")


if __name__ == "__main__":
    demo_contexts()
    demo_disconnect()
