"""calyxos Graph Visualization — Graphviz

Demonstrates rendering computation graphs to PNG/SVG using graphviz.
Requires: pip install graphviz (and the Graphviz system package).
"""

from calyxos import NodeChange, NodeFlag, get_graph, node, set_value
from calyxos.utils.debug import GraphDebugger


# ── 1. Simple dependency graph ────────────────────────────────────────────────

class Portfolio:
    @node(NodeFlag.STORED)
    def num_shares(self) -> int:
        return 100

    @node(NodeFlag.CAN_SET)
    def price_per_share(self) -> float:
        return 50.0

    @node()
    def market_value(self) -> float:
        return self.num_shares() * self.price_per_share()

    @node()
    def tax_estimate(self) -> float:
        return self.market_value() * 0.15

    @node()
    def net_value(self) -> float:
        return self.market_value() - self.tax_estimate()


def demo_simple():
    print("1. Rendering simple dependency graph...")
    p = Portfolio()
    _ = p.net_value()  # trigger evaluation

    dbg = GraphDebugger(p)
    path = dbg.render("portfolio_graph", directory="docs", fmt="png")
    print(f"   Saved to: {path}")

    # Also render SVG
    path_svg = dbg.render("portfolio_graph", directory="docs", fmt="svg")
    print(f"   Saved to: {path_svg}")


# ── 2. Graph with invalid nodes ──────────────────────────────────────────────

def demo_invalid():
    print("\n2. Rendering graph with invalid (dirty) nodes...")
    p = Portfolio()
    _ = p.net_value()

    # Invalidate by changing price
    set_value(p, "price_per_share", 75.0)
    # Don't access net_value — leaves it invalid

    dbg = GraphDebugger(p)
    path = dbg.render(
        "portfolio_invalid",
        directory="docs",
        fmt="png",
        title="Portfolio after price change (invalid nodes in red)",
        show_counts=True,
    )
    print(f"   Saved to: {path}")


# ── 3. Graph with context overrides ──────────────────────────────────────────

class RiskModel:
    @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
    def spot(self) -> float:
        return 100.0

    @node(NodeFlag.CAN_OVERRIDE)
    def rate(self) -> float:
        return 0.05

    @node()
    def pv(self) -> float:
        return self.spot() / (1 + self.rate())

    @node()
    def risk(self) -> float:
        return self.pv() * self.rate()


def demo_overrides():
    print("\n3. Rendering graph with active context overrides (gold nodes)...")
    model = RiskModel()
    _ = model.risk()

    graph = get_graph(model)
    dbg = GraphDebugger(model)

    with graph.context() as ctx:
        ctx.override(model, "spot", 120.0)
        _ = model.risk()  # recompute with override

        path = dbg.render(
            "risk_overridden",
            directory="docs",
            fmt="png",
            title="Risk model with spot overridden to 120",
        )
        print(f"   Saved to: {path}")


# ── 4. Cross-object dependency graph ─────────────────────────────────────────

class Market:
    @node(NodeFlag.STORED)
    def spot(self) -> float:
        return 100.0

class Forward:
    def __init__(self, market: Market) -> None:
        self.market = market

    @node()
    def price(self) -> float:
        return self.market.spot() * 1.05


def demo_cross_object():
    print("\n4. Rendering cross-object dependency graph...")
    mkt = Market()
    fwd = Forward(mkt)
    _ = fwd.price()

    dbg = GraphDebugger(fwd)
    path = dbg.render(
        "cross_object",
        directory="docs",
        fmt="png",
        title="Forward depends on Market.spot (dashed = cross-object)",
    )
    print(f"   Saved to: {path}")


# ── 5. Interactive HTML visualization ─────────────────────────────────────────

def demo_interactive():
    """Open an interactive graph in the browser. No extra deps needed."""
    print("\n5. Opening interactive graph in browser...")

    p = Portfolio()
    _ = p.net_value()

    # Invalidate some nodes so you can see red borders
    set_value(p, "price_per_share", 75.0)

    dbg = GraphDebugger(p)

    # Option A: open_in_browser() writes a temp file and opens it
    path = dbg.open_in_browser(title="Portfolio — interactive")
    print(f"   Opened: {path}")

    # Option B: serve() starts a local HTTP server (uncomment to try)
    # dbg.serve(title="Portfolio — served", port=8787)


# ── 6. Get the Digraph object directly (graphviz) ────────────────────────────

def demo_programmatic():
    print("\n6. Programmatic access to graphviz.Digraph...")
    p = Portfolio()
    _ = p.net_value()

    dbg = GraphDebugger(p)
    dot = dbg.to_graphviz(rankdir="LR", show_values=False)

    # Access the DOT source
    print(f"   DOT source length: {len(dot.source)} chars")
    print(f"   First 200 chars:\n   {dot.source[:200]}...")


if __name__ == "__main__":
    demo_simple()
    demo_invalid()
    demo_overrides()
    demo_cross_object()
    demo_interactive()
    demo_programmatic()
    print("\nAll visualizations saved to docs/")
