"""calyxos Sensitivity Analysis — Layers

Demonstrates:
- Layers for persistent computation state
- Re-entry without recomputation
- Comparing base vs bumped scenarios
"""

from calyxos import NodeFlag, get_graph, node, set_value


class RiskModel:
    """A risk model where we bump inputs and measure output sensitivity."""

    def __init__(self) -> None:
        self.compute_count = 0

    @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
    def spot(self) -> float:
        return 100.0

    @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
    def rate(self) -> float:
        return 0.05

    @node()
    def pv(self) -> float:
        """Present value — expensive computation."""
        self.compute_count += 1
        return self.spot() / (1 + self.rate())


def demo_layers():
    print("=" * 60)
    print("SENSITIVITY ANALYSIS WITH LAYERS")
    print("=" * 60)

    model = RiskModel()
    graph = get_graph(model)

    # Compute base case
    base_pv = model.pv()
    print(f"Base PV: {base_pv:.4f}  (computes: {model.compute_count})")

    # ── Bump spot up by 1 ─────────────────────────────────────────────────
    bump_up = graph.layer("spot_up")
    with bump_up:
        set_value(model, "spot", 101.0)
        up_pv = model.pv()
        print(f"Spot+1 PV: {up_pv:.4f}  (computes: {model.compute_count})")

    # ── Bump spot down by 1 ───────────────────────────────────────────────
    bump_down = graph.layer("spot_down")
    with bump_down:
        set_value(model, "spot", 99.0)
        down_pv = model.pv()
        print(f"Spot-1 PV: {down_pv:.4f}  (computes: {model.compute_count})")

    # ── Compute delta (finite difference) ─────────────────────────────────
    delta = (up_pv - down_pv) / 2.0
    print(f"\nDelta (dPV/dSpot): {delta:.4f}")

    # ── Re-enter layers — no recomputation! ───────────────────────────────
    count_before = model.compute_count
    with bump_up:
        assert model.pv() == up_pv  # Cached!
    with bump_down:
        assert model.pv() == down_pv  # Cached!

    print(f"\nRe-entered both layers: computes={model.compute_count} (unchanged from {count_before})")
    print("  No recomputation needed — layer snapshots preserved!")


def demo_rate_sensitivity():
    print(f"\n{'=' * 60}")
    print("RATE SENSITIVITY LADDER")
    print("=" * 60)

    model = RiskModel()
    graph = get_graph(model)
    base_pv = model.pv()

    print(f"{'Rate':>8}  {'PV':>10}  {'Change':>10}")
    print(f"{'----':>8}  {'----':>10}  {'------':>10}")
    print(f"{'0.0500':>8}  {base_pv:>10.4f}  {'(base)':>10}")

    layers = {}
    for bp in [-100, -50, -25, 25, 50, 100]:
        rate = 0.05 + bp / 10000
        name = f"rate_{bp:+d}bp"
        layer = graph.layer(name)
        layers[bp] = layer

        with layer:
            set_value(model, "rate", rate)
            pv = model.pv()
            change = pv - base_pv
            print(f"{rate:>8.4f}  {pv:>10.4f}  {change:>+10.4f}")

    # All layers are cached — re-access is free
    print(f"\nTotal computations: {model.compute_count}")
    print(f"  ({1 + len(layers)} scenarios computed, each only once)")


if __name__ == "__main__":
    demo_layers()
    demo_rate_sensitivity()
