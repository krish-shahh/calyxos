"""CalyxOS Reactive Basics — Getting Started

Demonstrates:
- @node decorator with flags (CAN_SET, STORED, CAN_OVERRIDE)
- Automatic dependency tracking and lazy invalidation
- Memoization with arguments
- set_value() for explicit mutation
"""

from calyxos import NodeFlag, get_graph, node, set_value
from calyxos.utils.debug import GraphDebugger


# ── 1. Basic reactive nodes ──────────────────────────────────────────────────

class Portfolio:
    """A simple portfolio that recomputes totals when positions change."""

    @node(NodeFlag.STORED)
    def num_shares(self) -> int:
        return 100

    @node(NodeFlag.CAN_SET)
    def price_per_share(self) -> float:
        return 50.0

    @node()
    def market_value(self) -> float:
        """Automatically depends on num_shares and price_per_share."""
        return self.num_shares() * self.price_per_share()

    @node()
    def tax_estimate(self) -> float:
        """Depends on market_value — two levels deep."""
        return self.market_value() * 0.15


def demo_basics():
    print("=" * 60)
    print("1. BASIC REACTIVE NODES")
    print("=" * 60)

    p = Portfolio()

    # First access computes and caches
    print(f"Market value: ${p.market_value():,.2f}")
    print(f"Tax estimate: ${p.tax_estimate():,.2f}")

    # Changing price automatically invalidates market_value and tax_estimate
    print("\n--- Setting price to $75 ---")
    set_value(p, "price_per_share", 75.0)

    print(f"Market value: ${p.market_value():,.2f}")  # Recomputed
    print(f"Tax estimate: ${p.tax_estimate():,.2f}")   # Recomputed

    # Dump the dependency tree
    dbg = GraphDebugger(p)
    print(f"\nDependency tree for tax_estimate:")
    print(dbg.dump_dependency_tree("tax_estimate"))


# ── 2. Argument memoization ──────────────────────────────────────────────────

class Pricer:
    """Each unique argument combo gets its own cached node."""

    def __init__(self):
        self.call_count = 0

    @node()
    def price(self, currency: str) -> float:
        self.call_count += 1
        rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 110.0}
        return 100.0 * rates.get(currency, 1.0)


def demo_memoization():
    print("\n" + "=" * 60)
    print("2. ARGUMENT MEMOIZATION")
    print("=" * 60)

    pricer = Pricer()

    print(f"USD price: {pricer.price('USD')} (calls: {pricer.call_count})")
    print(f"EUR price: {pricer.price('EUR')} (calls: {pricer.call_count})")
    print(f"USD again:  {pricer.price('USD')} (calls: {pricer.call_count})")  # Cached!
    print(f"EUR again:  {pricer.price('EUR')} (calls: {pricer.call_count})")  # Cached!


# ── 3. Selective invalidation ────────────────────────────────────────────────

class Model:
    """Only nodes that depend on a changed input recompute."""

    def __init__(self):
        self.a_count = 0
        self.b_count = 0

    @node(NodeFlag.STORED)
    def input_x(self) -> int:
        return 10

    @node(NodeFlag.STORED)
    def input_y(self) -> int:
        return 20

    @node()
    def from_x(self) -> int:
        self.a_count += 1
        return self.input_x() * 2

    @node()
    def from_y(self) -> int:
        self.b_count += 1
        return self.input_y() * 3


def demo_selective():
    print("\n" + "=" * 60)
    print("3. SELECTIVE INVALIDATION")
    print("=" * 60)

    m = Model()
    print(f"from_x={m.from_x()}, from_y={m.from_y()}")
    print(f"  (a_count={m.a_count}, b_count={m.b_count})")

    print("\n--- Changing input_x only ---")
    set_value(m, "input_x", 50)

    print(f"from_x={m.from_x()}, from_y={m.from_y()}")
    print(f"  (a_count={m.a_count}, b_count={m.b_count})")
    print("  ^ Notice b_count didn't increase — from_y wasn't recomputed!")


if __name__ == "__main__":
    demo_basics()
    demo_memoization()
    demo_selective()
