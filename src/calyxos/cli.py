"""calyxos CLI entry point.

Usage:
    calyxos                      show version and help
    calyxos demo                 run benchmark + drop into the TUI inspector
    calyxos demo --no-inspect    run benchmark only (no TUI)
    calyxos mlx-demo             run MLX benchmark + drop into the MLX TUI
    calyxos mlx-demo --no-inspect  run MLX benchmark only (no TUI)
"""

from __future__ import annotations

import sys


def _check_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _print_help() -> None:
    from calyxos import __version__

    if _check_rich():
        from rich.console import Console
        from rich.panel import Panel

        c = Console()
        c.print()
        c.print(Panel(
            f"[bold]calyxos[/] v{__version__}\n"
            "[dim]reactive computation engine for python[/]",
            border_style="blue", padding=(1, 2),
        ))
        c.print()
        c.print("  [bold cyan]calyxos demo[/]              run benchmark + TUI inspector")
        c.print("  [bold cyan]calyxos demo --no-inspect[/]  run benchmark only")
        c.print("  [bold cyan]calyxos mlx-demo[/]          MLX incremental benchmark + TUI")
        c.print("  [bold cyan]calyxos mlx-demo --no-inspect[/]  MLX benchmark only")
        c.print()
        c.print("  [dim]in your code:[/]")
        c.print("    [green]from calyxos import inspect[/]")
        c.print("    [green]inspect(my_object)[/]           [dim]# drop into the TUI[/]")
        c.print()
    else:
        print(f"\ncalyxos v{__version__}")
        print("reactive computation engine for python\n")
        print("  calyxos demo              run benchmark + TUI inspector")
        print("  calyxos demo --no-inspect  run benchmark only")
        print("  calyxos mlx-demo          MLX incremental benchmark + TUI")
        print("  calyxos mlx-demo --no-inspect  MLX benchmark only")
        print()
        print("  in your code:")
        print("    from calyxos import inspect")
        print("    inspect(my_object)")
        print()
        print(f"  note: install 'rich' for the TUI: pip install calyxos[tui]")
        print()


def _run_demo(args: list[str]) -> None:
    if not _check_rich():
        print("error: the TUI requires 'rich'. install it with:")
        print("  pip install calyxos[tui]")
        sys.exit(1)

    # Import here so the demo module doesn't need to be loaded at startup
    import math
    import time

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    from calyxos import NodeFlag, get_graph, node, set_value

    console = Console(width=76)

    # ── Inline demo (self-contained, no external file needed) ───────

    WORK_MS = 8

    def expensive(val: float) -> float:
        end = time.perf_counter() + WORK_MS / 1000
        while time.perf_counter() < end:
            pass
        return val

    class DemoPipeline:
        """8 independent streams converging to one output."""

        def __init__(self) -> None:
            self.compute_count = 0

        def _track(self, val: float) -> float:
            self.compute_count += 1
            return expensive(val)

        @node(NodeFlag.CAN_SET)
        def input_a(self) -> float: return 10.0
        @node(NodeFlag.CAN_SET)
        def input_b(self) -> float: return 20.0
        @node(NodeFlag.CAN_SET)
        def input_c(self) -> float: return 30.0
        @node(NodeFlag.CAN_SET)
        def input_d(self) -> float: return 40.0
        @node(NodeFlag.CAN_SET)
        def input_e(self) -> float: return 50.0
        @node(NodeFlag.CAN_SET)
        def input_f(self) -> float: return 60.0
        @node(NodeFlag.CAN_SET)
        def input_g(self) -> float: return 70.0
        @node(NodeFlag.CAN_SET)
        def input_h(self) -> float: return 80.0

        @node()
        def a1(self) -> float: return self._track(self.input_a() ** 2)
        @node()
        def a2(self) -> float: return self._track(math.log1p(self.a1()))
        @node()
        def a3(self) -> float: return self._track(math.sqrt(self.a2() + 1))
        @node()
        def b1(self) -> float: return self._track(self.input_b() ** 2)
        @node()
        def b2(self) -> float: return self._track(math.log1p(self.b1()))
        @node()
        def b3(self) -> float: return self._track(math.sqrt(self.b2() + 1))
        @node()
        def c1(self) -> float: return self._track(self.input_c() ** 2)
        @node()
        def c2(self) -> float: return self._track(math.log1p(self.c1()))
        @node()
        def c3(self) -> float: return self._track(math.sqrt(self.c2() + 1))
        @node()
        def d1(self) -> float: return self._track(self.input_d() ** 2)
        @node()
        def d2(self) -> float: return self._track(math.log1p(self.d1()))
        @node()
        def d3(self) -> float: return self._track(math.sqrt(self.d2() + 1))
        @node()
        def e1(self) -> float: return self._track(self.input_e() ** 2)
        @node()
        def e2(self) -> float: return self._track(math.log1p(self.e1()))
        @node()
        def e3(self) -> float: return self._track(math.sqrt(self.e2() + 1))
        @node()
        def f1(self) -> float: return self._track(self.input_f() ** 2)
        @node()
        def f2(self) -> float: return self._track(math.log1p(self.f1()))
        @node()
        def f3(self) -> float: return self._track(math.sqrt(self.f2() + 1))
        @node()
        def g1(self) -> float: return self._track(self.input_g() ** 2)
        @node()
        def g2(self) -> float: return self._track(math.log1p(self.g1()))
        @node()
        def g3(self) -> float: return self._track(math.sqrt(self.g2() + 1))
        @node()
        def h1(self) -> float: return self._track(self.input_h() ** 2)
        @node()
        def h2(self) -> float: return self._track(math.log1p(self.h1()))
        @node()
        def h3(self) -> float: return self._track(math.sqrt(self.h2() + 1))

        @node()
        def output(self) -> float:
            total = (self.a3() + self.b3() + self.c3() + self.d3()
                     + self.e3() + self.f3() + self.g3() + self.h3())
            return self._track(total)

    class NaivePipeline:
        def __init__(self) -> None:
            self.compute_count = 0
            self.inputs = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

        def _track(self, val: float) -> float:
            self.compute_count += 1
            return expensive(val)

        def compute_all(self) -> float:
            streams = []
            for inp in self.inputs:
                v1 = self._track(inp ** 2)
                v2 = self._track(math.log1p(v1))
                v3 = self._track(math.sqrt(v2 + 1))
                streams.append(v3)
            return self._track(sum(streams))

    TOTAL = 25
    AFFECTED = 4

    console.print()
    console.print(Panel(
        "[bold white]calyxos benchmark[/]\n"
        "[dim]selective recomputation vs recompute-from-scratch[/]",
        border_style="blue", padding=(1, 2),
    ))
    console.print()
    console.print(f"[bold cyan]graph:[/]   8 input streams x 3 transforms -> 1 output")
    console.print(f"[bold cyan]nodes:[/]   {TOTAL} derived nodes, ~{WORK_MS}ms each")
    console.print(f"[bold cyan]change:[/]  1 input out of 8\n")

    # Phase 1
    console.rule("[bold]phase 1 — first computation", style="dim")
    console.print()

    pipe = DemoPipeline()
    t0 = time.perf_counter()
    pipe.output()
    t_cx_init = (time.perf_counter() - t0) * 1000

    naive = NaivePipeline()
    t0 = time.perf_counter()
    naive.compute_all()
    t_nv_init = (time.perf_counter() - t0) * 1000

    t1 = Table(box=box.SIMPLE_HEAVY, show_edge=False, pad_edge=False)
    t1.add_column("", style="bold", min_width=12)
    t1.add_column("nodes", justify="center", min_width=8)
    t1.add_column("time", justify="center", min_width=10)
    t1.add_row("naive", f"{naive.compute_count}", f"{t_nv_init:.0f}ms")
    t1.add_row("calyxos", f"{pipe.compute_count}", f"{t_cx_init:.0f}ms")
    console.print(t1)
    console.print("[dim]  same cost on cold start — both compute all 25 nodes.\n[/]")

    # Phase 2
    console.rule("[bold]phase 2 — change ONE input  (input_a: 10 -> 15)", style="dim")
    console.print()

    console.print(Panel(
        "[dim]  A   B   C   D   E   F   G   H       <- 8 inputs[/]\n"
        "[dim]  |   |   |   |   |   |   |   |[/]\n"
        "[bold green]  a1[/][dim]  b1  c1  d1  e1  f1  g1  h1      <- transform 1[/]\n"
        "[bold green]  a2[/][dim]  b2  c2  d2  e2  f2  g2  h2      <- transform 2[/]\n"
        "[bold green]  a3[/][dim]  b3  c3  d3  e3  f3  g3  h3      <- transform 3[/]\n"
        "[dim]  \\   |   |   |   |   |   |   /[/]\n"
        "[dim]             [/][bold green]output[/][dim]                     <- final[/]",
        title="[bold green]green[/] = recomputed",
        border_style="green", padding=(0, 2),
    ))
    console.print()

    naive.inputs[0] = 15.0
    naive.compute_count = 0
    t0 = time.perf_counter()
    naive.compute_all()
    t_nv_upd = (time.perf_counter() - t0) * 1000
    nv_count = naive.compute_count

    pipe.compute_count = 0
    t0 = time.perf_counter()
    set_value(pipe, "input_a", 15.0)
    pipe.output()
    t_cx_upd = (time.perf_counter() - t0) * 1000
    cx_count = pipe.compute_count

    speedup = t_nv_upd / t_cx_upd if t_cx_upd > 0 else float("inf")

    t2 = Table(box=box.HEAVY_HEAD, show_edge=False, pad_edge=False)
    t2.add_column("", style="bold", min_width=12)
    t2.add_column("recomputed", justify="center", min_width=14)
    t2.add_column("skipped", justify="center", min_width=10)
    t2.add_column("time", justify="center", min_width=10)
    t2.add_column("speedup", justify="center", min_width=12)
    t2.add_row("naive", f"[red]{nv_count} / {TOTAL}[/]", f"[red]0[/]", f"{t_nv_upd:.0f}ms", "")
    t2.add_row(
        "[green]calyxos[/]",
        f"[green]{cx_count} / {TOTAL}[/]",
        f"[green]{TOTAL - cx_count}[/]",
        f"[green]{t_cx_upd:.0f}ms[/]",
        f"[bold green]{speedup:.1f}x faster[/]",
    )
    console.print(t2)

    # Phase 3
    console.print()
    console.rule("[bold]phase 3 — sensitivity analysis  (10 scenarios)", style="dim")
    console.print()

    N_SCENARIOS = 10
    bumps = [10.0 + i for i in range(1, N_SCENARIOS + 1)]

    set_value(pipe, "input_a", 10.0)
    pipe.output()

    naive.inputs[0] = 10.0
    naive.compute_count = 0
    t0 = time.perf_counter()
    for b in bumps:
        naive.inputs[0] = b
        naive.compute_all()
    t_nv_scen = (time.perf_counter() - t0) * 1000

    graph = get_graph(pipe)
    pipe.compute_count = 0
    layers = {}
    t0 = time.perf_counter()
    for b in bumps:
        lyr = graph.layer(f"bump_{b}")
        layers[b] = lyr
        with lyr:
            set_value(pipe, "input_a", b)
            pipe.output()
    t_cx_scen = (time.perf_counter() - t0) * 1000
    cx_scen_count = pipe.compute_count

    pipe.compute_count = 0
    t0 = time.perf_counter()
    for lyr in layers.values():
        with lyr:
            pipe.output()
    t_reentry = (time.perf_counter() - t0) * 1000

    scen_speedup = t_nv_scen / t_cx_scen if t_cx_scen > 0 else float("inf")
    reentry_speedup = t_nv_scen / t_reentry if t_reentry > 0 else float("inf")

    t3 = Table(box=box.HEAVY_HEAD, show_edge=False, pad_edge=False)
    t3.add_column("", style="bold", min_width=22)
    t3.add_column("nodes", justify="center", min_width=8)
    t3.add_column("time", justify="center", min_width=12)
    t3.add_column("vs naive", justify="center", min_width=12)
    t3.add_row("naive (10 reruns)", f"[red]250[/]", f"{t_nv_scen:.0f}ms", "")
    t3.add_row("[green]calyxos (10 layers)[/]", f"[green]{cx_scen_count}[/]", f"[green]{t_cx_scen:.0f}ms[/]", f"[green]{scen_speedup:.1f}x[/]")
    re_str = f"{t_reentry:.1f}ms" if t_reentry >= 0.1 else f"{t_reentry * 1000:.0f}us"
    t3.add_row(
        "[bold green]calyxos re-entry[/]",
        "[bold green]0[/]",
        f"[bold green]{re_str}[/]",
        f"[bold green]{reentry_speedup:.0f}x[/]" if t_reentry >= 0.01 else "[bold green]inf[/]",
    )
    console.print(t3)

    # Summary
    console.print()
    console.print(Panel(
        f"[bold white]what just happened?[/]\n\n"
        f"  [bold white]1.[/] built a 25-node computation graph\n"
        f"  [bold white]2.[/] changed [bold yellow]1[/] of 8 inputs\n"
        f"  [bold white]3.[/] naive recomputed [bold red]all {TOTAL}[/] nodes\n"
        f"  [bold white]4.[/] calyxos recomputed [bold green]only {AFFECTED}[/] — skipped {TOTAL - AFFECTED}\n"
        f"  [bold white]5.[/] ran 10 what-if scenarios with [bold green]layer caching[/]\n"
        f"  [bold white]6.[/] re-entering any scenario: [bold green]0 recomputes[/]\n\n"
        "  [dim]real pipelines have 100s-1000s of nodes.[/]\n"
        "  [dim]each skipped node saves real wall-clock time.[/]\n\n"
        "  [bold cyan]pip install calyxos[/]       "
        "[dim]github.com/krish-shahh/calyxos[/]",
        title="[bold]results",
        border_style="cyan", padding=(1, 2),
    ))
    console.print()

    # Drop into TUI unless --no-inspect
    if "--no-inspect" not in args:
        from calyxos.tui import inspect as tui_inspect
        tui_inspect(pipe)


def _run_mlx_demo(args: list[str]) -> None:
    if not _check_rich():
        print("error: the TUI requires 'rich'. install it with:")
        print("  pip install calyxos[tui]")
        sys.exit(1)

    try:
        import mlx.core as mx
    except ImportError:
        print("error: mlx is required for the mlx demo. install it with:")
        print("  pip install calyxos[mlx]")
        sys.exit(1)

    import gc
    import time

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box

    from calyxos.ml.mlx_graph import MLXGraph

    console = Console(width=76)

    DIM = 512
    FFN_DIM = DIM * 4
    SEQ = 256
    N_ITERS = 10

    console.print()
    console.print(Panel(
        "[bold white]calyxos mlx benchmark[/]\n"
        "[dim]incremental tensor recomputation on apple silicon[/]",
        border_style="magenta", padding=(1, 2),
    ))
    console.print()
    console.print(f"[bold magenta]pipeline:[/]  tokens -> embed -> ln1 -> attn -> ln2 -> ffn -> proj")
    console.print(f"[bold magenta]config:[/]    dim={DIM}  seq={SEQ}  ffn={FFN_DIM}")
    console.print(f"[bold magenta]mutation:[/]  W_up (FFN weight, mid-graph)")
    console.print()

    # -- build graph --
    g = MLXGraph()
    v_tokens = g.var("tokens", mx.random.normal((SEQ, DIM)))
    v_W_embed = g.var("W_embed", mx.random.normal((DIM, DIM)) * 0.02)
    v_W_q = g.var("W_q", mx.random.normal((DIM, DIM)) * 0.02)
    v_W_k = g.var("W_k", mx.random.normal((DIM, DIM)) * 0.02)
    v_W_v = g.var("W_v", mx.random.normal((DIM, DIM)) * 0.02)
    v_W_up = g.var("W_up", mx.random.normal((DIM, FFN_DIM)) * 0.02)
    v_W_down = g.var("W_down", mx.random.normal((FFN_DIM, DIM)) * 0.02)
    v_W_proj = g.var("W_proj", mx.random.normal((DIM, DIM)) * 0.02)

    def _ln(x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + 1e-5)

    def _attn(x_ln: mx.array, wq: mx.array, wk: mx.array, wv: mx.array) -> mx.array:
        Q, K, V = x_ln @ wq, x_ln @ wk, x_ln @ wv
        scores = (Q @ K.T) * (Q.shape[-1] ** -0.5)
        return mx.softmax(scores, axis=-1) @ V

    n_embed = g.node("embed", lambda t, w: t @ w, [v_tokens, v_W_embed])
    n_ln1 = g.node("ln1", lambda x: _ln(x), [n_embed])
    n_attn = g.node("attn", _attn, [n_ln1, v_W_q, v_W_k, v_W_v])
    n_attn_res = g.node("attn_res", lambda e, a: e + a, [n_embed, n_attn])
    n_ln2 = g.node("ln2", lambda x: _ln(x), [n_attn_res])
    n_ffn = g.node("ffn", lambda x, wu, wd: mx.maximum(x @ wu, 0) @ wd, [n_ln2, v_W_up, v_W_down])
    n_ffn_res = g.node("ffn_res", lambda x, f: x + f, [n_attn_res, n_ffn])
    n_proj = g.node("proj", lambda x, w: x @ w, [n_ffn_res, v_W_proj])

    # materialise weights + cold-start
    mx.eval(*[v.value for v in g._vars.values()])
    g.eval(n_proj)

    # -- full rebuild function --
    def full_rebuild() -> mx.array:
        t = v_tokens.value
        x = t @ v_W_embed.value
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_ln = (x - mean) * mx.rsqrt(var + 1e-5)
        Q, K, V = x_ln @ v_W_q.value, x_ln @ v_W_k.value, x_ln @ v_W_v.value
        scores = (Q @ K.T) * (Q.shape[-1] ** -0.5)
        attn = mx.softmax(scores, axis=-1) @ V
        x2 = x + attn
        mean2 = mx.mean(x2, axis=-1, keepdims=True)
        var2 = mx.var(x2, axis=-1, keepdims=True)
        x_ln2 = (x2 - mean2) * mx.rsqrt(var2 + 1e-5)
        ffn = mx.maximum(x_ln2 @ v_W_up.value, 0) @ v_W_down.value
        x3 = x2 + ffn
        out = x3 @ v_W_proj.value
        mx.eval(out)
        return out

    # -- benchmark --
    console.rule("[bold]benchmark — mutate W_up, measure rebuild vs incremental", style="dim")
    console.print()

    full_times = []
    incr_times = []

    for _ in range(N_ITERS):
        new_W_up = mx.random.normal((DIM, FFN_DIM)) * 0.02
        mx.eval(new_W_up)

        v_W_up.set(new_W_up)
        gc.collect()
        t0 = time.perf_counter()
        full_rebuild()
        full_times.append((time.perf_counter() - t0) * 1000)

        gc.collect()
        t0 = time.perf_counter()
        g.eval(n_proj)
        incr_times.append((time.perf_counter() - t0) * 1000)

    import statistics
    full_med = statistics.median(full_times)
    incr_med = statistics.median(incr_times)
    speedup = full_med / incr_med if incr_med > 0 else float("inf")

    stale = g.stale_nodes()
    # After eval, nothing is stale — show what WOULD be stale
    v_W_up.set(mx.random.normal((DIM, FFN_DIM)) * 0.02)
    stale_after_set = g.stale_nodes()
    stale_names = [n.name for n in stale_after_set]

    t = Table(box=box.HEAVY_HEAD, show_edge=False, pad_edge=False)
    t.add_column("", style="bold", min_width=18)
    t.add_column("recomputed", justify="center", min_width=14)
    t.add_column("median", justify="center", min_width=10)
    t.add_column("speedup", justify="center", min_width=12)
    t.add_row("full rebuild", f"[red]8 / 8[/]", f"{full_med:.2f}ms", "")
    t.add_row(
        "[green]calyxos incr[/]",
        f"[green]{len(stale_names)} / 8[/]",
        f"[green]{incr_med:.2f}ms[/]",
        f"[bold green]{speedup:.1f}x faster[/]",
    )
    console.print(t)
    console.print()

    console.print(Panel(
        f"[bold white]what happened?[/]\n\n"
        f"  [bold white]1.[/] built an 8-node transformer pipeline as an MLXGraph\n"
        f"  [bold white]2.[/] changed [bold yellow]W_up[/] (FFN weight, mid-graph)\n"
        f"  [bold white]3.[/] full rebuild recomputed [bold red]all 8[/] stages\n"
        f"  [bold white]4.[/] calyxos recomputed [bold green]only {len(stale_names)}[/]: {', '.join(stale_names)}\n"
        f"  [bold white]5.[/] arrays stayed lazy — single [bold cyan]mx.eval()[/] at the boundary\n"
        f"  [bold white]6.[/] no tensor hashing — version counters only\n",
        title="[bold]results",
        border_style="magenta", padding=(1, 2),
    ))
    console.print()

    # Drop into TUI unless --no-inspect
    if "--no-inspect" not in args:
        from calyxos.tui import inspect_mlx
        inspect_mlx(g)


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    if args[0] in ("-v", "--version"):
        from calyxos import __version__
        print(f"calyxos {__version__}")
        return

    if args[0] == "demo":
        _run_demo(args[1:])
        return

    if args[0] == "mlx-demo":
        _run_mlx_demo(args[1:])
        return

    # Unknown command
    print(f"unknown command: {args[0]}")
    _print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
