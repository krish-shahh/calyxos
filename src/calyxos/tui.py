"""calyxos interactive graph inspector.

Usage::

    from calyxos.tui import inspect, inspect_mlx
    inspect(my_object)          # core calyxos objects
    inspect_mlx(mlx_graph)      # MLXGraph objects
"""

from __future__ import annotations

import readline  # noqa: F401 — enables arrow-key history in input()
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

from calyxos.core.decorator import get_graph, set_value
from calyxos.core.flags import NodeFlag
from calyxos.graph.node import Node, NodeType

console = Console()

# ── Helpers ─────────────────────────────────────────────────────────────


def _flag_str(nd: Node) -> str:
    parts = []
    if nd.has_flag(NodeFlag.STORED):
        parts.append("STORED")
    elif nd.has_flag(NodeFlag.CAN_SET):
        parts.append("CAN_SET")
    if nd.has_flag(NodeFlag.CAN_OVERRIDE):
        parts.append("CAN_OVERRIDE")
    return ", ".join(parts) if parts else "-"


def _val_str(val: Any, max_len: int = 40) -> str:
    r = repr(val)
    return r if len(r) <= max_len else r[: max_len - 3] + "..."


def _status_str(nd: Node) -> str:
    if nd.is_valid:
        return "[green]valid[/]"
    return "[red]invalid[/]"


def _type_str(nd: Node) -> str:
    if nd.node_type == NodeType.STORED:
        return "[yellow]stored[/]"
    return "[cyan]derived[/]"


def _resolve_names(graph, keys: set) -> list[str]:
    names = []
    for k in keys:
        child = graph.nodes.get(k)
        if child is not None:
            names.append(child.method_name)
        else:
            names.append(f"<{k[1]}@{k[0]}>")
    return sorted(names)


# ── Commands ────────────────────────────────────────────────────────────


def cmd_graph(obj: Any, _args: str) -> None:
    """Show all nodes in the computation graph."""
    graph = get_graph(obj)
    nodes = graph.get_all_nodes()

    if not nodes:
        console.print("[dim]  (empty graph — access some nodes first)[/]")
        return

    stored = sorted(
        [n for n in nodes if n.node_type == NodeType.STORED],
        key=lambda n: n.method_name,
    )
    derived = sorted(
        [n for n in nodes if n.node_type == NodeType.DERIVED],
        key=lambda n: n.method_name,
    )

    t = Table(
        box=box.ROUNDED,
        title=f"[bold]{obj.__class__.__name__}[/] — {len(nodes)} nodes",
        title_style="",
        show_lines=False,
        pad_edge=False,
    )
    t.add_column("node", style="bold")
    t.add_column("type", justify="center")
    t.add_column("status", justify="center")
    t.add_column("value", max_width=36)
    t.add_column("computes", justify="right")
    t.add_column("flags")

    for nd in stored + derived:
        t.add_row(
            nd.method_name,
            _type_str(nd),
            _status_str(nd),
            _val_str(nd.value),
            str(nd.compute_count),
            _flag_str(nd),
        )

    console.print(t)


def cmd_node(obj: Any, args: str) -> None:
    """Show detailed info for a single node."""
    name = args.strip()
    if not name:
        console.print("[red]usage: node <name>[/]")
        return

    graph = get_graph(obj)
    nd = graph._find_node_by_name(name)
    if nd is None:
        console.print(f"[red]node '{name}' not found[/]")
        return

    parents = _resolve_names(graph, nd.parents)
    children = _resolve_names(graph, nd.children)

    rows = [
        ("name", nd.method_name),
        ("type", nd.node_type.value),
        ("status", "valid" if nd.is_valid else "INVALID"),
        ("value", _val_str(nd.value, 60)),
        ("flags", _flag_str(nd)),
        ("computes", str(nd.compute_count)),
        ("reason", nd.last_recompute_reason or "-"),
        ("depends on", ", ".join(children) if children else "(none)"),
        ("depended by", ", ".join(parents) if parents else "(none)"),
    ]

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="dim", min_width=12)
    t.add_column()
    for k, v in rows:
        t.add_row(k, str(v))

    console.print(Panel(t, title=f"[bold]{name}[/]", border_style="cyan", padding=(0, 1)))


def _node_badge(nd: Node) -> str:
    """Render a single node as a colored badge string."""
    if nd.node_type == NodeType.STORED or nd.has_flag(NodeFlag.CAN_SET):
        icon = "[yellow]■[/]"
        name_style = "bold yellow"
    else:
        icon = "[cyan]◆[/]"
        name_style = "bold cyan"

    if nd.is_valid:
        status = "[green]●[/]"
    else:
        status = "[bold red]✗[/]"
        name_style = "bold red"

    val = _val_str(nd.value, 18)
    return f"{icon} [{name_style}]{nd.method_name}[/] {status} [dim]{val}[/]"


def cmd_tree(obj: Any, args: str) -> None:
    """Show dependency tree for a node."""
    name = args.strip()
    if not name:
        console.print("[red]usage: tree <name>[/]")
        return

    graph = get_graph(obj)
    nd = graph._find_node_by_name(name)
    if nd is None:
        console.print(f"[red]node '{name}' not found[/]")
        return

    guide = "green" if nd.is_valid else "red"
    rich_tree = Tree(_node_badge(nd), guide_style=guide)
    visited: set[str] = {nd.method_name}

    def _build(parent_tree: Tree, node: Node) -> None:
        for child_key in sorted(node.children, key=lambda k: k[1]):
            child_nd = graph.nodes.get(child_key)
            if child_nd is None:
                parent_tree.add(f"[dim]? {child_key[1]} (external)[/]")
                continue
            if child_nd.method_name in visited:
                parent_tree.add(f"{_node_badge(child_nd)} [dim](ref)[/]")
                continue
            visited.add(child_nd.method_name)
            child_guide = "green" if child_nd.is_valid else "red"
            branch = parent_tree.add(_node_badge(child_nd), guide_style=child_guide)
            _build(branch, child_nd)

    _build(rich_tree, nd)
    console.print(Panel(
        rich_tree,
        title=f"[bold]tree:[/] {name}   [yellow]■[/] input  [cyan]◆[/] derived  [green]●[/] valid  [red]✗[/] dirty",
        border_style="dim",
        padding=(0, 1),
    ))


def _compute_layers(graph: Any) -> list[list[Node]]:
    """Topological sort into layers (inputs first, outputs last)."""
    nodes = graph.get_all_nodes()
    if not nodes:
        return []

    # Build name -> node map and adjacency
    by_name: dict[str, Node] = {}
    for nd in nodes:
        by_name[nd.method_name] = nd

    # in-degree based on children (deps). Nodes with no children = layer 0.
    in_deg: dict[str, int] = {}
    dependents: dict[str, list[str]] = {}
    for nd in nodes:
        child_names = []
        for ck in nd.children:
            cn = graph.nodes.get(ck)
            if cn is not None:
                child_names.append(cn.method_name)
        in_deg[nd.method_name] = len(child_names)
        for c in child_names:
            dependents.setdefault(c, []).append(nd.method_name)

    # BFS layering
    layers: list[list[Node]] = []
    ready = [n for n in by_name if in_deg.get(n, 0) == 0]
    visited: set[str] = set()

    while ready:
        layer = sorted(ready)
        layers.append([by_name[n] for n in layer])
        visited.update(layer)
        next_ready = []
        for n in layer:
            for dep in dependents.get(n, []):
                in_deg[dep] -= 1
                if in_deg[dep] == 0 and dep not in visited:
                    next_ready.append(dep)
        ready = next_ready

    return layers


def cmd_flow(obj: Any, _args: str) -> None:
    """Show the computation graph as a layered flow diagram."""
    graph = get_graph(obj)
    layers = _compute_layers(graph)

    if not layers:
        console.print("[dim]  (empty graph)[/]")
        return

    lines: list[str] = []

    for i, layer in enumerate(layers):
        # Build badges for this layer
        badges = []
        for nd in layer:
            if nd.node_type == NodeType.STORED or nd.has_flag(NodeFlag.CAN_SET):
                color = "yellow"
            elif not nd.is_valid:
                color = "red"
            else:
                color = "cyan"

            val = _val_str(nd.value, 12)
            if nd.is_valid:
                status_dot = "[green]●[/]"
            else:
                status_dot = "[bold red]✗[/]"
                color = "red"
            badges.append(f"[{color}]{nd.method_name}[/]{status_dot}[dim]={val}[/]")

        # Layer label
        if i == 0:
            label = "inputs"
        elif i == len(layers) - 1:
            label = "output"
        else:
            label = f"layer {i}"

        row = "  ".join(badges)
        lines.append(f"  [dim]{label:>8}[/]  {row}")

        # Arrow between layers
        if i < len(layers) - 1:
            lines.append(f"  [dim]{'':>8}  {'│':>1}[/]")
            lines.append(f"  [dim]{'':>8}  {'▼':>1}[/]")

    content = "\n".join(lines)
    console.print(Panel(
        content,
        title=f"[bold]{obj.__class__.__name__}[/]  [yellow]■[/] input  [cyan]◆[/] derived  [green]●[/] valid  [red]✗[/] dirty",
        border_style="blue",
        padding=(1, 1),
    ))


def cmd_set(obj: Any, args: str) -> None:
    """Set a node value and show what was invalidated."""
    parts = args.strip().split(maxsplit=1)
    if len(parts) != 2:
        console.print("[red]usage: set <name> <value>[/]")
        return

    name, raw_val = parts

    try:
        val = eval(raw_val)  # noqa: S307 — intentional for interactive use
    except Exception as e:
        console.print(f"[red]bad value: {e}[/]")
        return

    graph = get_graph(obj)

    # snapshot validity before
    before = {nd.method_name: nd.is_valid for nd in graph.get_all_nodes()}

    try:
        set_value(obj, name, val)
    except (ValueError, KeyError) as e:
        console.print(f"[red]{e}[/]")
        return

    # find what was invalidated
    after = {nd.method_name: nd.is_valid for nd in graph.get_all_nodes()}
    invalidated = [n for n in after if before.get(n, True) and not after[n]]

    console.print(f"[green]set {name} = {_val_str(val)}[/]")
    if invalidated:
        console.print(f"[yellow]invalidated:[/] {', '.join(sorted(invalidated))}")
    else:
        console.print("[dim]no downstream nodes invalidated[/]")


def cmd_eval(obj: Any, args: str) -> None:
    """Evaluate a node and show its value."""
    name = args.strip()
    if not name:
        console.print("[red]usage: eval <name>[/]")
        return

    method = getattr(obj, name, None)
    if method is None or not callable(method):
        console.print(f"[red]'{name}' is not a method on {obj.__class__.__name__}[/]")
        return

    try:
        result = method()
    except Exception as e:
        console.print(f"[red]error: {e}[/]")
        return

    console.print(f"[green]{name}() = {_val_str(result, 60)}[/]")


def cmd_stats(obj: Any, _args: str) -> None:
    """Show graph statistics."""
    graph = get_graph(obj)
    nodes = graph.get_all_nodes()
    stored = sum(1 for n in nodes if n.node_type == NodeType.STORED)
    derived = sum(1 for n in nodes if n.node_type == NodeType.DERIVED)
    invalid = sum(1 for n in nodes if not n.is_valid)
    total_computes = sum(n.compute_count for n in nodes)

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="dim", min_width=16)
    t.add_column()
    t.add_row("total nodes", str(len(nodes)))
    t.add_row("stored", str(stored))
    t.add_row("derived", str(derived))
    t.add_row("invalid", f"[red]{invalid}[/]" if invalid else "0")
    t.add_row("total computes", str(total_computes))

    console.print(Panel(t, title="[bold]stats[/]", border_style="dim", padding=(0, 1)))


def cmd_invalid(obj: Any, _args: str) -> None:
    """List all invalid (dirty) nodes."""
    graph = get_graph(obj)
    invalid = [n for n in graph.get_all_nodes() if not n.is_valid]

    if not invalid:
        console.print("[green]all nodes valid[/]")
        return

    for nd in sorted(invalid, key=lambda n: n.method_name):
        reason = nd.last_recompute_reason or "unknown"
        console.print(f"  [red]{nd.method_name}[/]  [dim]reason: {reason}[/]")


def cmd_help(_obj: Any, _args: str) -> None:
    """Show available commands."""
    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="bold cyan", min_width=22)
    t.add_column(style="dim")
    t.add_row("graph", "show all nodes")
    t.add_row("flow", "layered DAG view")
    t.add_row("node <name>", "inspect a node")
    t.add_row("tree <name>", "dependency tree from a node")
    t.add_row("set <name> <value>", "set value, show invalidation")
    t.add_row("eval <name>", "evaluate a node")
    t.add_row("stats", "graph statistics")
    t.add_row("invalid", "list dirty nodes")
    t.add_row("help", "this message")
    t.add_row("quit / q / exit", "exit inspector")
    console.print(Panel(t, title="[bold]commands[/]", border_style="dim", padding=(0, 1)))


COMMANDS = {
    "graph": cmd_graph,
    "flow": cmd_flow,
    "node": cmd_node,
    "tree": cmd_tree,
    "set": cmd_set,
    "eval": cmd_eval,
    "stats": cmd_stats,
    "invalid": cmd_invalid,
    "help": cmd_help,
}


# ── Main entry point ───────────────────────────────────────────────────


def inspect(obj: Any) -> None:
    """Drop into an interactive graph inspector for a calyxos object.

    Args:
        obj: Any object that has calyxos-decorated methods with an
             existing computation graph (i.e., some nodes have been
             evaluated at least once).
    """
    graph = get_graph(obj)
    n_nodes = len(graph.get_all_nodes())

    console.print()
    console.print(Panel(
        f"[bold]{obj.__class__.__name__}[/] — {n_nodes} nodes\n"
        "[dim]type 'help' for commands, 'quit' to exit[/]",
        title="[bold]calyxos inspector[/]",
        border_style="blue",
        padding=(0, 2),
    ))
    console.print()

    # show graph overview on entry
    cmd_graph(obj, "")
    console.print()

    while True:
        try:
            raw = console.input("[bold blue]calyxos>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        raw = raw.strip()
        if not raw:
            continue

        if raw in ("quit", "q", "exit"):
            break

        parts = raw.split(maxsplit=1)
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        handler = COMMANDS.get(cmd_name)
        if handler is None:
            console.print(f"[red]unknown command: {cmd_name}[/]  (type 'help')")
            continue

        handler(obj, cmd_args)
        console.print()


# ══════════════════════════════════════════════════════════════════════
#  MLX Graph Inspector
# ══════════════════════════════════════════════════════════════════════


def _mx_shape_str(arr: Any) -> str:
    """Format an mx.array as 'shape dtype', e.g. '(512, 2048) float32'."""
    try:
        return f"{list(arr.shape)} {arr.dtype}"
    except Exception:
        return repr(arr)[:40]


def _mlx_node_badge(name: str, kind: str, stale: bool) -> str:
    if kind == "var":
        icon = "[yellow]■[/]"
        name_style = "bold yellow"
    else:
        icon = "[cyan]◆[/]"
        name_style = "bold cyan"

    if stale:
        status = "[bold red]✗[/]"
        name_style = "bold red"
    else:
        status = "[green]●[/]"

    return f"{icon} [{name_style}]{name}[/] {status}"


def mlx_cmd_graph(g: Any, _args: str) -> None:
    from calyxos.ml.mlx_graph import MLXNode, MLXVar

    t = Table(
        box=box.ROUNDED,
        title=f"[bold]MLXGraph[/] — {len(g._vars)} vars, {len(g._nodes)} nodes",
        title_style="",
        show_lines=False,
        pad_edge=False,
    )
    t.add_column("name", style="bold")
    t.add_column("type", justify="center")
    t.add_column("status", justify="center")
    t.add_column("shape", max_width=28)
    t.add_column("version", justify="right")

    for v in g._vars.values():
        t.add_row(
            v.name,
            "[yellow]var[/]",
            "[green]—[/]",
            _mx_shape_str(v.value),
            str(v.version),
        )

    from calyxos.ml.mlx_graph import _UNSET

    for name in g._topo_order:
        nd = g._nodes[name]
        stale = nd.is_stale
        has_value = nd._cached is not _UNSET
        t.add_row(
            nd.name,
            "[cyan]node[/]",
            "[bold red]stale[/]" if stale else "[green]valid[/]",
            _mx_shape_str(nd._cached) if has_value else "[dim]—[/]",
            str(nd.version),
        )

    console.print(t)


def mlx_cmd_flow(g: Any, _args: str) -> None:
    from calyxos.ml.mlx_graph import MLXNode, MLXVar

    # Build adjacency for topological layering
    all_names: list[str] = list(g._vars.keys()) + list(g._topo_order)
    in_deg: dict[str, int] = {n: 0 for n in all_names}
    children_of: dict[str, list[str]] = {n: [] for n in all_names}

    for nd in g._nodes.values():
        for inp in nd._inputs:
            children_of[inp.name].append(nd.name)
            in_deg[nd.name] += 1

    # BFS layering
    layers: list[list[str]] = []
    ready = [n for n in all_names if in_deg[n] == 0]
    visited: set[str] = set()

    while ready:
        layer = sorted(ready)
        layers.append(layer)
        visited.update(layer)
        next_ready = []
        for n in layer:
            for dep in children_of.get(n, []):
                in_deg[dep] -= 1
                if in_deg[dep] == 0 and dep not in visited:
                    next_ready.append(dep)
        ready = next_ready

    lines: list[str] = []
    for i, layer in enumerate(layers):
        badges = []
        for name in layer:
            if name in g._vars:
                badges.append(_mlx_node_badge(name, "var", False))
            else:
                nd = g._nodes[name]
                badges.append(_mlx_node_badge(name, "node", nd.is_stale))

        if i == 0:
            label = "inputs"
        elif i == len(layers) - 1:
            label = "output"
        else:
            label = f"layer {i}"

        row = "  ".join(badges)
        lines.append(f"  [dim]{label:>8}[/]  {row}")

        if i < len(layers) - 1:
            lines.append(f"  [dim]{'':>8}  {'│':>1}[/]")
            lines.append(f"  [dim]{'':>8}  {'▼':>1}[/]")

    content = "\n".join(lines)
    console.print(Panel(
        content,
        title="[bold]MLXGraph[/]  [yellow]■[/] var  [cyan]◆[/] node  [green]●[/] valid  [red]✗[/] stale",
        border_style="blue",
        padding=(1, 1),
    ))


def mlx_cmd_node(g: Any, args: str) -> None:
    from calyxos.ml.mlx_graph import MLXNode, MLXVar

    name = args.strip()
    if not name:
        console.print("[red]usage: node <name>[/]")
        return

    if name in g._vars:
        v = g._vars[name]
        rows = [
            ("name", v.name),
            ("type", "var (input)"),
            ("version", str(v.version)),
            ("shape", _mx_shape_str(v.value)),
        ]
        # find dependents
        deps = [nd.name for nd in g._nodes.values() if v in nd._inputs]
        rows.append(("used by", ", ".join(sorted(deps)) if deps else "(none)"))

        t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        t.add_column(style="dim", min_width=12)
        t.add_column()
        for k, val in rows:
            t.add_row(k, val)
        console.print(Panel(t, title=f"[bold yellow]{name}[/]", border_style="yellow", padding=(0, 1)))
        return

    if name in g._nodes:
        nd = g._nodes[name]
        inputs = [inp.name for inp in nd._inputs]
        # find dependents
        deps = [n2.name for n2 in g._nodes.values() if nd in n2._inputs]
        stale = nd.is_stale

        rows = [
            ("name", nd.name),
            ("type", "node (compute)"),
            ("status", "[bold red]stale[/]" if stale else "[green]valid[/]"),
            ("version", str(nd.version)),
            ("inputs", ", ".join(inputs) if inputs else "(none)"),
            ("used by", ", ".join(sorted(deps)) if deps else "(none)"),
        ]
        if nd._cached is not None and not stale:
            rows.append(("shape", _mx_shape_str(nd._cached)))

        t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        t.add_column(style="dim", min_width=12)
        t.add_column()
        for k, val in rows:
            t.add_row(k, val)
        console.print(Panel(t, title=f"[bold cyan]{name}[/]", border_style="cyan", padding=(0, 1)))
        return

    console.print(f"[red]'{name}' not found in graph[/]")


def mlx_cmd_set(g: Any, args: str) -> None:
    import mlx.core as mx

    parts = args.strip().split()
    if len(parts) < 2:
        console.print("[red]usage: set <var> random <shape...>  or  set <var> zeros <shape...>[/]")
        return

    name = parts[0]
    if name not in g._vars:
        console.print(f"[red]'{name}' is not a var (only vars can be set)[/]")
        return

    fill = parts[1]
    try:
        shape = tuple(int(x) for x in parts[2:])
    except ValueError:
        # Fall back: use current shape
        shape = tuple(g._vars[name].value.shape)

    if not shape:
        shape = tuple(g._vars[name].value.shape)

    if fill == "random":
        new_val = mx.random.normal(shape)
    elif fill == "zeros":
        new_val = mx.zeros(shape)
    elif fill == "ones":
        new_val = mx.ones(shape)
    else:
        console.print(f"[red]unknown fill: {fill} (use random, zeros, ones)[/]")
        return

    g._vars[name].set(new_val)

    stale = g.stale_nodes()
    console.print(f"[green]set {name}[/] -> {_mx_shape_str(new_val)}")
    if stale:
        console.print(f"[yellow]stale:[/] {', '.join(n.name for n in stale)}")
    else:
        console.print("[dim]no downstream nodes affected[/]")


def mlx_cmd_eval(g: Any, args: str) -> None:
    import time

    name = args.strip()
    if not name:
        # Evaluate all stale nodes
        stale = g.stale_nodes()
        if not stale:
            console.print("[green]all nodes valid, nothing to evaluate[/]")
            return
        targets = stale
    else:
        if name not in g._nodes:
            console.print(f"[red]'{name}' is not a compute node[/]")
            return
        targets = [g._nodes[name]]

    stale_before = g.stale_nodes()
    t0 = time.perf_counter()
    g.eval(*targets)
    elapsed = (time.perf_counter() - t0) * 1000

    stale_after = g.stale_nodes()
    recomputed = len(stale_before) - len(stale_after)

    for nd in targets:
        console.print(f"  [green]{nd.name}[/] -> {_mx_shape_str(nd._cached)}")

    console.print(f"[dim]{recomputed} node(s) recomputed in {elapsed:.2f}ms[/]")


def mlx_cmd_stale(g: Any, _args: str) -> None:
    stale = g.stale_nodes()
    if not stale:
        console.print("[green]all nodes valid[/]")
        return

    for nd in stale:
        console.print(f"  [red]{nd.name}[/]  [dim]v={nd.version}[/]")

    console.print(f"[dim]{len(stale)}/{len(g._nodes)} stale[/]")


def mlx_cmd_stats(g: Any, _args: str) -> None:
    stale = g.stale_nodes()
    total_versions = sum(nd.version for nd in g._nodes.values())

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="dim", min_width=16)
    t.add_column()
    t.add_row("vars", str(len(g._vars)))
    t.add_row("compute nodes", str(len(g._nodes)))
    t.add_row("stale", f"[red]{len(stale)}[/]" if stale else "0")
    t.add_row("total recomputes", str(total_versions))

    console.print(Panel(t, title="[bold]stats[/]", border_style="dim", padding=(0, 1)))


def mlx_cmd_tree(g: Any, args: str) -> None:
    """Show dependency tree for an MLX node."""
    from calyxos.ml.mlx_graph import MLXVar, MLXNode

    name = args.strip()
    if not name:
        console.print("[red]usage: tree <name>[/]")
        return

    target = g._nodes.get(name)
    if target is None:
        console.print(f"[red]node '{name}' not found[/]")
        return

    def _badge(item: Any) -> str:
        if isinstance(item, MLXVar):
            return f"[yellow]■[/] [bold yellow]{item.name}[/] [green]●[/] [dim]{_mx_shape_str(item.value)}[/]"
        stale = item.is_stale
        status = "[bold red]✗[/]" if stale else "[green]●[/]"
        name_style = "bold red" if stale else "bold cyan"
        shape = _mx_shape_str(item._cached) if item._cached is not _UNSET else "?"
        return f"[cyan]◆[/] [{name_style}]{item.name}[/] {status} [dim]{shape}[/]"

    # Need the _UNSET sentinel
    from calyxos.ml.mlx_graph import _UNSET

    guide = "red" if target.is_stale else "green"
    rich_tree = Tree(_badge(target), guide_style=guide)
    visited: set[str] = {target.name}

    def _build(parent_tree: Tree, node: Any) -> None:
        if not isinstance(node, MLXNode):
            return
        for inp in node._inputs:
            if inp.name in visited:
                parent_tree.add(f"{_badge(inp)} [dim](ref)[/]")
                continue
            visited.add(inp.name)
            child_guide = "red" if (isinstance(inp, MLXNode) and inp.is_stale) else "green"
            branch = parent_tree.add(_badge(inp), guide_style=child_guide)
            _build(branch, inp)

    _build(rich_tree, target)
    console.print(Panel(
        rich_tree,
        title=f"[bold]tree:[/] {name}   [yellow]■[/] var  [cyan]◆[/] node  [green]●[/] fresh  [red]✗[/] stale",
        border_style="dim",
        padding=(0, 1),
    ))


def mlx_cmd_help(_g: Any, _args: str) -> None:
    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style="bold cyan", min_width=30)
    t.add_column(style="dim")
    t.add_row("graph", "show all vars and nodes")
    t.add_row("flow", "layered DAG view")
    t.add_row("tree <name>", "dependency tree from a node")
    t.add_row("node <name>", "inspect a var or node")
    t.add_row("set <var> random [shape...]", "set var, show staleness cascade")
    t.add_row("eval [name]", "evaluate node(s) with mx.eval()")
    t.add_row("stale", "list stale nodes")
    t.add_row("stats", "graph statistics")
    t.add_row("help", "this message")
    t.add_row("quit / q / exit", "exit inspector")
    console.print(Panel(t, title="[bold]commands[/]", border_style="dim", padding=(0, 1)))


MLX_COMMANDS = {
    "graph": mlx_cmd_graph,
    "flow": mlx_cmd_flow,
    "tree": mlx_cmd_tree,
    "node": mlx_cmd_node,
    "set": mlx_cmd_set,
    "eval": mlx_cmd_eval,
    "stale": mlx_cmd_stale,
    "stats": mlx_cmd_stats,
    "help": mlx_cmd_help,
}


def inspect_mlx(g: Any) -> None:
    """Drop into an interactive inspector for an MLXGraph.

    Args:
        g: An ``MLXGraph`` instance with vars and nodes registered.
    """
    console.print()
    console.print(Panel(
        f"[bold]MLXGraph[/] — {len(g._vars)} vars, {len(g._nodes)} nodes\n"
        "[dim]type 'help' for commands, 'quit' to exit[/]",
        title="[bold]calyxos mlx inspector[/]",
        border_style="magenta",
        padding=(0, 2),
    ))
    console.print()

    mlx_cmd_graph(g, "")
    console.print()

    while True:
        try:
            raw = console.input("[bold magenta]mlx>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        raw = raw.strip()
        if not raw:
            continue

        if raw in ("quit", "q", "exit"):
            break

        parts = raw.split(maxsplit=1)
        cmd_name = parts[0].lower()
        cmd_args = parts[1] if len(parts) > 1 else ""

        handler = MLX_COMMANDS.get(cmd_name)
        if handler is None:
            console.print(f"[red]unknown command: {cmd_name}[/]  (type 'help')")
            continue

        handler(g, cmd_args)
        console.print()
