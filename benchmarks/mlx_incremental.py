#!/usr/bin/env python3
"""Go / no-go benchmark: calyxos incremental MLX vs full rebuild.

Pipeline (simplified transformer block):

    tokens  ──►  embed  ──►  ln1  ──►  attn  ──►  ln2  ──►  ffn  ──►  proj
      │                       │
    W_embed                 W_q,W_k,W_v                    W_up,W_down  W_proj

Scenario
--------
1.  Build the full pipeline and evaluate once (cold start – both paths equal).
2.  Mutate **W_up** (the FFN up-projection weight, mid-graph).
3.  Re-evaluate the final output.
    - *Full rebuild*: reconstruct every stage from scratch and ``mx.eval()``.
    - *Calyxos incremental*: only ``ffn`` and ``proj`` recompute; ``embed``,
      ``ln1``, ``attn``, ``ln2`` are cache hits.

We report wall-clock latency and (when available) peak Metal memory.

Usage::

    python benchmarks/mlx_incremental.py [--dim 512] [--seq 256] [--heads 8] [--iters 20]
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time

try:
    import mlx.core as mx
except ImportError:
    print("ERROR: mlx is not installed.  pip install mlx", file=sys.stderr)
    sys.exit(1)

# Ensure the local src/ is importable when running from repo root
sys.path.insert(0, "src")

from calyxos.ml.mlx_graph import MLXGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_mem_mb() -> float | None:
    """Return peak Metal memory in MiB, or None if unavailable."""
    try:
        return mx.get_peak_memory() / (1024 * 1024)
    except AttributeError:
        try:
            return mx.metal.get_peak_memory() / (1024 * 1024)
        except Exception:
            return None


def _reset_peak_mem() -> None:
    try:
        mx.reset_peak_memory()
    except AttributeError:
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Full-rebuild baseline (no caching, rebuild everything every call)
# ---------------------------------------------------------------------------

def full_rebuild(
    tokens: mx.array,
    W_embed: mx.array,
    W_q: mx.array,
    W_k: mx.array,
    W_v: mx.array,
    W_up: mx.array,
    W_down: mx.array,
    W_proj: mx.array,
) -> mx.array:
    """Execute the full pipeline from scratch and return the evaluated result."""
    # embed
    x = tokens @ W_embed

    # layer norm 1
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x_ln1 = (x - mean) * mx.rsqrt(var + 1e-5)

    # self-attention (single-head simplified)
    Q = x_ln1 @ W_q
    K = x_ln1 @ W_k
    V = x_ln1 @ W_v
    d_k = Q.shape[-1]
    scores = (Q @ mx.transpose(K, (0, 2, 1))) * (d_k ** -0.5) if Q.ndim == 3 else (Q @ K.T) * (d_k ** -0.5)
    attn_out = mx.softmax(scores, axis=-1) @ V
    x_attn = x + attn_out  # residual

    # layer norm 2
    mean2 = mx.mean(x_attn, axis=-1, keepdims=True)
    var2 = mx.var(x_attn, axis=-1, keepdims=True)
    x_ln2 = (x_attn - mean2) * mx.rsqrt(var2 + 1e-5)

    # FFN (up -> gelu -> down)
    up = mx.maximum(x_ln2 @ W_up, 0)  # ReLU for simplicity
    down = up @ W_down
    x_ffn = x_attn + down  # residual

    # output projection
    out = x_ffn @ W_proj
    mx.eval(out)
    return out


# ---------------------------------------------------------------------------
# Calyxos incremental pipeline
# ---------------------------------------------------------------------------

def build_calyxos_pipeline(
    dim: int,
    ffn_mult: int = 4,
) -> tuple[MLXGraph, dict[str, object], MLXGraph]:
    """Build an MLXGraph mirroring the full_rebuild pipeline.

    Returns (graph, vars_dict, last_node).
    """
    g = MLXGraph()

    # -- input vars --
    v_tokens = g.var("tokens", mx.zeros((1,)))       # placeholder
    v_W_embed = g.var("W_embed", mx.zeros((1,)))
    v_W_q = g.var("W_q", mx.zeros((1,)))
    v_W_k = g.var("W_k", mx.zeros((1,)))
    v_W_v = g.var("W_v", mx.zeros((1,)))
    v_W_up = g.var("W_up", mx.zeros((1,)))
    v_W_down = g.var("W_down", mx.zeros((1,)))
    v_W_proj = g.var("W_proj", mx.zeros((1,)))

    # -- computation nodes --
    n_embed = g.node(
        "embed",
        lambda tok, we: tok @ we,
        [v_tokens, v_W_embed],
    )

    def _ln(x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + 1e-5)

    n_ln1 = g.node("ln1", lambda x: _ln(x), [n_embed])

    def _attn(x_ln: mx.array, wq: mx.array, wk: mx.array, wv: mx.array) -> mx.array:
        Q = x_ln @ wq
        K = x_ln @ wk
        V = x_ln @ wv
        d_k = Q.shape[-1]
        if Q.ndim == 3:
            scores = (Q @ mx.transpose(K, (0, 2, 1))) * (d_k ** -0.5)
        else:
            scores = (Q @ K.T) * (d_k ** -0.5)
        return mx.softmax(scores, axis=-1) @ V

    n_attn = g.node("attn", _attn, [n_ln1, v_W_q, v_W_k, v_W_v])

    # residual around attention: need embed output too
    n_attn_res = g.node(
        "attn_res",
        lambda embed, attn: embed + attn,
        [n_embed, n_attn],
    )

    n_ln2 = g.node("ln2", lambda x: _ln(x), [n_attn_res])

    def _ffn(x_ln: mx.array, w_up: mx.array, w_down: mx.array) -> mx.array:
        return mx.maximum(x_ln @ w_up, 0) @ w_down

    n_ffn = g.node("ffn", _ffn, [n_ln2, v_W_up, v_W_down])

    n_ffn_res = g.node(
        "ffn_res",
        lambda x_attn, ffn: x_attn + ffn,
        [n_attn_res, n_ffn],
    )

    n_proj = g.node(
        "proj",
        lambda x, wp: x @ wp,
        [n_ffn_res, v_W_proj],
    )

    vars_dict = {
        "tokens": v_tokens,
        "W_embed": v_W_embed,
        "W_q": v_W_q,
        "W_k": v_W_k,
        "W_v": v_W_v,
        "W_up": v_W_up,
        "W_down": v_W_down,
        "W_proj": v_W_proj,
    }
    return g, vars_dict, n_proj


# ---------------------------------------------------------------------------
# Benchmark harness
# ---------------------------------------------------------------------------

def run_benchmark(
    dim: int = 512,
    seq_len: int = 256,
    n_iters: int = 20,
) -> dict:
    ffn_dim = dim * 4

    # Shared weight tensors (same across both paths)
    def make_weights() -> dict[str, mx.array]:
        return {
            "tokens": mx.random.normal((seq_len, dim)),
            "W_embed": mx.random.normal((dim, dim)) * 0.02,
            "W_q": mx.random.normal((dim, dim)) * 0.02,
            "W_k": mx.random.normal((dim, dim)) * 0.02,
            "W_v": mx.random.normal((dim, dim)) * 0.02,
            "W_up": mx.random.normal((dim, ffn_dim)) * 0.02,
            "W_down": mx.random.normal((ffn_dim, dim)) * 0.02,
            "W_proj": mx.random.normal((dim, dim)) * 0.02,
        }

    w = make_weights()
    # Materialise so the benchmark doesn't include init cost
    mx.eval(*w.values())

    # ---- warm-up both paths once ----
    _ = full_rebuild(**w)

    g, gvars, out_node = build_calyxos_pipeline(dim)
    for name, arr in w.items():
        gvars[name].set(arr)
    _ = g.eval(out_node)

    # ---- benchmark loop: mutate W_up, then re-evaluate ----
    full_times: list[float] = []
    full_peak_mem: list[float] = []
    incr_times: list[float] = []
    incr_peak_mem: list[float] = []
    incr_stale_counts: list[int] = []

    for i in range(n_iters):
        # Generate a new W_up each iteration
        new_W_up = mx.random.normal((dim, ffn_dim)) * 0.02
        mx.eval(new_W_up)

        # -- full rebuild --
        gc.collect()
        _reset_peak_mem()
        w_copy = dict(w)
        w_copy["W_up"] = new_W_up
        t0 = time.perf_counter()
        _ = full_rebuild(**w_copy)
        full_dt = time.perf_counter() - t0
        full_times.append(full_dt)
        pm = _peak_mem_mb()
        if pm is not None:
            full_peak_mem.append(pm)

        # -- calyxos incremental --
        gc.collect()
        _reset_peak_mem()
        gvars["W_up"].set(new_W_up)  # only this changes
        stale_before = g.stale_nodes()
        incr_stale_counts.append(len(stale_before))
        t0 = time.perf_counter()
        _ = g.eval(out_node)
        incr_dt = time.perf_counter() - t0
        incr_times.append(incr_dt)
        pm = _peak_mem_mb()
        if pm is not None:
            incr_peak_mem.append(pm)

    results = {
        "config": {"dim": dim, "seq_len": seq_len, "ffn_dim": ffn_dim, "iters": n_iters},
        "full_rebuild": {
            "mean_ms": statistics.mean(full_times) * 1000,
            "median_ms": statistics.median(full_times) * 1000,
            "stdev_ms": statistics.stdev(full_times) * 1000 if len(full_times) > 1 else 0,
            "peak_mem_mb": max(full_peak_mem) if full_peak_mem else None,
        },
        "calyxos_incremental": {
            "mean_ms": statistics.mean(incr_times) * 1000,
            "median_ms": statistics.median(incr_times) * 1000,
            "stdev_ms": statistics.stdev(incr_times) * 1000 if len(incr_times) > 1 else 0,
            "peak_mem_mb": max(incr_peak_mem) if incr_peak_mem else None,
            "avg_stale_nodes": statistics.mean(incr_stale_counts),
            "total_nodes": len(g._nodes),
        },
    }

    # Compute speedup
    if results["calyxos_incremental"]["mean_ms"] > 0:
        results["speedup_x"] = round(
            results["full_rebuild"]["mean_ms"]
            / results["calyxos_incremental"]["mean_ms"],
            2,
        )
    else:
        results["speedup_x"] = float("inf")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark: calyxos incremental MLX vs full rebuild"
    )
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--seq", type=int, default=256, help="Sequence length")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    args = parser.parse_args()

    print(f"MLX incremental benchmark")
    print(f"  dim={args.dim}  seq={args.seq}  iters={args.iters}")
    print(f"  Pipeline: tokens -> embed -> ln1 -> attn -> ln2 -> ffn -> proj")
    print(f"  Mutation: W_up (FFN up-projection, mid-graph)")
    print()

    results = run_benchmark(dim=args.dim, seq_len=args.seq, n_iters=args.iters)

    cfg = results["config"]
    fb = results["full_rebuild"]
    ci = results["calyxos_incremental"]

    print(f"--- Full Rebuild (all {8} stages) ---")
    print(f"  mean:   {fb['mean_ms']:8.2f} ms")
    print(f"  median: {fb['median_ms']:8.2f} ms")
    print(f"  stdev:  {fb['stdev_ms']:8.2f} ms")
    if fb["peak_mem_mb"] is not None:
        print(f"  peak mem: {fb['peak_mem_mb']:.1f} MiB")

    print()
    print(f"--- Calyxos Incremental ({ci['avg_stale_nodes']:.0f}/{ci['total_nodes']} nodes recomputed) ---")
    print(f"  mean:   {ci['mean_ms']:8.2f} ms")
    print(f"  median: {ci['median_ms']:8.2f} ms")
    print(f"  stdev:  {ci['stdev_ms']:8.2f} ms")
    if ci["peak_mem_mb"] is not None:
        print(f"  peak mem: {ci['peak_mem_mb']:.1f} MiB")

    print()
    print(f"Speedup: {results['speedup_x']}x")

    # Go / no-go verdict
    print()
    if results["speedup_x"] >= 1.2:
        print("VERDICT: GO - incremental path shows meaningful speedup")
    elif results["speedup_x"] >= 1.0:
        print("VERDICT: MARGINAL - incremental path is faster but gains are small")
    else:
        print("VERDICT: NO-GO - incremental overhead exceeds rebuild cost at this scale")


if __name__ == "__main__":
    main()
