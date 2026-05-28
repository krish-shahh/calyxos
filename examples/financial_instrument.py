"""calyxos Financial Instrument — Cross-Object Dependencies & Reverse Propagation

Demonstrates:
- Cross-object dependencies (Instrument depends on Market)
- Reverse propagation (set a derived value, upstream adjusts automatically)
- Full dependency graph introspection
"""

from calyxos import NodeChange, NodeFlag, get_graph, node, set_value
from calyxos.utils.debug import GraphDebugger


# ── Market data source (shared across instruments) ────────────────────────────

class Market:
    """Global market data that instruments depend on."""

    @node(NodeFlag.STORED)
    def spot(self) -> float:
        return 100.0

    @node(NodeFlag.STORED)
    def risk_free_rate(self) -> float:
        return 0.05


# ── Forward contract ─────────────────────────────────────────────────────────

class Forward:
    """A forward contract whose price depends on market spot and rate.

    Demonstrates cross-object dependencies and reverse propagation.
    """

    def __init__(self, market: Market, maturity: float = 1.0) -> None:
        self.market = market
        self.maturity = maturity

    @node()
    def forward_price(self) -> float:
        """F = S * e^(r*T) (simplified as S * (1 + r*T))."""
        return self.market.spot() * (1 + self.market.risk_free_rate() * self.maturity)

    @node(
        NodeFlag.CAN_SET,
        get_changes=lambda self, val: [
            NodeChange(
                self.market,
                "spot",
                val / (1 + self.market.risk_free_rate() * self.maturity),
            )
        ],
    )
    def implied_spot(self) -> float:
        """The spot implied by the forward price.

        Setting this value reverse-propagates to Market.spot.
        """
        return self.market.spot()


def demo_cross_object():
    print("=" * 60)
    print("CROSS-OBJECT DEPENDENCIES")
    print("=" * 60)

    mkt = Market()
    fwd = Forward(mkt, maturity=1.0)

    print(f"Market spot: {mkt.spot()}")
    print(f"Forward price: {fwd.forward_price():.2f}")

    # Change market spot — forward should auto-recompute
    print(f"\n--- Market spot moves to 110 ---")
    set_value(mkt, "spot", 110.0)
    print(f"Forward price: {fwd.forward_price():.2f}")

    # Change risk-free rate
    print(f"\n--- Rate changes to 0.03 ---")
    set_value(mkt, "risk_free_rate", 0.03)
    print(f"Forward price: {fwd.forward_price():.2f}")


def demo_reverse_propagation():
    print(f"\n{'=' * 60}")
    print("REVERSE PROPAGATION (BIDIRECTIONAL BINDING)")
    print("=" * 60)

    mkt = Market()
    fwd = Forward(mkt, maturity=1.0)

    print(f"Market spot: {mkt.spot()}")
    print(f"Forward price: {fwd.forward_price():.2f}")
    print(f"Implied spot: {fwd.implied_spot():.2f}")

    # Set the implied spot — this reverse-propagates to Market.spot
    print(f"\n--- Setting implied_spot to 120 ---")
    set_value(fwd, "implied_spot", 120.0)

    print(f"Market spot (adjusted): {mkt.spot():.2f}")
    print(f"Implied spot: {fwd.implied_spot():.2f}")
    print(f"Forward price: {fwd.forward_price():.2f}")


def demo_introspection():
    print(f"\n{'=' * 60}")
    print("GRAPH INTROSPECTION")
    print("=" * 60)

    mkt = Market()
    fwd = Forward(mkt)

    # Trigger all evaluations
    _ = fwd.forward_price()
    _ = fwd.implied_spot()

    dbg = GraphDebugger(fwd)
    print("\nForward's dependency tree for forward_price:")
    print(dbg.dump_dependency_tree("forward_price"))

    print(f"\nAll nodes involved in computing forward_price:")
    for name in dbg.list_computing_nodes("forward_price"):
        print(f"  - {name}")

    print(f"\nNode status for forward_price:")
    status = dbg.get_node_status("forward_price")
    for k, v in status.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo_cross_object()
    demo_reverse_propagation()
    demo_introspection()
