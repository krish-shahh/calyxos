"""Debug utilities for inspecting and tracing CalyxOS graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from calyxos.core.decorator import get_graph
from calyxos.core.flags import NodeFlag
from calyxos.graph.node import NodeType


class GraphDebugger:
    """Utilities for introspecting and debugging computation graphs."""

    def __init__(self, obj: Any) -> None:
        """Initialize debugger for an object."""
        self.obj = obj
        self.graph = get_graph(obj)

    def print_graph(self) -> None:
        """Print a human-readable representation of the graph."""
        print(f"\nComputation Graph for {self.obj.__class__.__name__} (id={id(self.obj)})")
        print("=" * 70)

        nodes = self.graph.get_all_nodes()
        if not nodes:
            print("  (empty)")
            return

        # Group nodes by type
        stored_nodes = [n for n in nodes if n.node_type == NodeType.STORED]
        derived_nodes = [n for n in nodes if n.node_type == NodeType.DERIVED]

        if stored_nodes:
            print("\nSTORED NODES:")
            for node in stored_nodes:
                status = "valid" if node.is_valid else "INVALID"
                print(
                    f"  {node.method_name} (args_hash={node.args_hash}): "
                    f"{status}, value={node.value!r}, "
                    f"compute_count={node.compute_count}"
                )

        if derived_nodes:
            print("\nDERIVED NODES:")
            for node in derived_nodes:
                status = "valid" if node.is_valid else "INVALID"
                print(
                    f"  {node.method_name} (args_hash={node.args_hash}): "
                    f"{status}, value={node.value!r}, "
                    f"compute_count={node.compute_count}"
                )
                if node.last_recompute_reason:
                    print(f"    reason: {node.last_recompute_reason}")

        print("\nDEPENDENCY EDGES:")
        for node in nodes:
            if node.children:
                print(f"  {node.method_name} depends on:")
                for child_key in node.children:
                    child_node = self.graph.nodes.get(child_key)
                    if child_node:
                        print(f"    - {child_node.method_name}")

        invalid_count = sum(1 for n in nodes if not n.is_valid)
        print(f"\nSUMMARY: {len(nodes)} total, {invalid_count} invalid, "
              f"{len(stored_nodes)} stored, {len(derived_nodes)} derived")

    def get_recompute_trace(self, method_name: str) -> list[tuple[str, str]]:
        """
        Get trace of why a method needs recomputation.

        Returns list of (node_name, reason) tuples showing the recomputation chain.
        """
        trace: list[tuple[str, str]] = []

        def walk(node_name: str, depth: int = 0) -> None:
            node = next(
                (n for n in self.graph.get_all_nodes() if n.method_name == node_name), None
            )
            if node is None:
                return

            if not node.is_valid:
                reason = node.last_recompute_reason or "unknown"
                trace.append(("  " * depth + node_name, reason))

                for child_key in node.children:
                    child_node = self.graph.nodes.get(child_key)
                    if child_node and child_node.method_name != node_name:
                        walk(child_node.method_name, depth + 1)

        walk(method_name)
        return trace

    def get_node_info(self, method_name: str) -> dict[str, Any]:
        """Get detailed information about a node."""
        node = next(
            (n for n in self.graph.get_all_nodes() if n.method_name == method_name), None
        )

        if node is None:
            return {}

        parent_names = []
        for k in node.parents:
            parent_node = self.graph.nodes.get(k)
            if parent_node is not None:
                parent_names.append(parent_node.method_name)

        child_names = []
        for k in node.children:
            child_node = self.graph.nodes.get(k)
            if child_node is not None:
                child_names.append(child_node.method_name)

        return {
            "method_name": node.method_name,
            "type": node.node_type.value,
            "is_valid": node.is_valid,
            "value": node.value,
            "compute_count": node.compute_count,
            "last_recompute_reason": node.last_recompute_reason,
            "parents": parent_names,
            "children": child_names,
        }

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the graph."""
        nodes = self.graph.get_all_nodes()
        stored = [n for n in nodes if n.node_type == NodeType.STORED]
        derived = [n for n in nodes if n.node_type == NodeType.DERIVED]
        invalid = [n for n in nodes if not n.is_valid]

        total_computes = sum(n.compute_count for n in nodes)

        return {
            "total_nodes": len(nodes),
            "stored_nodes": len(stored),
            "derived_nodes": len(derived),
            "invalid_nodes": len(invalid),
            "total_computes": total_computes,
            "avg_computes_per_node": (
                total_computes / len(nodes) if nodes else 0
            ),
        }

    def dump_dependency_tree(self, method_name: str, max_depth: int = 10) -> str:
        """Return a text tree of all nodes involved in computing *method_name*.

        Each line shows the node name, its validity, and flags.
        """
        lines: list[str] = []

        def _walk(name: str, depth: int, visited: set[str]) -> None:
            if depth > max_depth or name in visited:
                return
            visited.add(name)

            nd = next(
                (n for n in self.graph.get_all_nodes() if n.method_name == name),
                None,
            )
            if nd is None:
                lines.append("  " * depth + f"{name} (not found)")
                return

            status = "valid" if nd.is_valid else "INVALID"
            flag_parts = []
            if nd.has_flag(NodeFlag.CAN_SET):
                flag_parts.append("CAN_SET")
            if nd.has_flag(NodeFlag.CAN_OVERRIDE):
                flag_parts.append("CAN_OVERRIDE")
            if nd.has_flag(NodeFlag.STORED):
                flag_parts.append("STORED")
            flags_str = f" [{','.join(flag_parts)}]" if flag_parts else ""

            has_override, _ = self.graph._get_active_override(nd.key())
            override_str = " (OVERRIDDEN)" if has_override else ""

            lines.append(
                f"{'  ' * depth}{name}: {status}, value={nd.value!r}"
                f"{flags_str}{override_str}"
            )

            for child_key in nd.children:
                child_nd = self.graph.nodes.get(child_key)
                child_name = child_nd.method_name if child_nd else f"<{child_key}>"
                _walk(child_name, depth + 1, visited)

        _walk(method_name, 0, set())
        return "\n".join(lines)

    def list_computing_nodes(self, method_name: str) -> list[str]:
        """List all nodes transitively required to compute *method_name*."""
        result: list[str] = []
        visited: set[str] = set()

        def _walk(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            result.append(name)

            nd = next(
                (n for n in self.graph.get_all_nodes() if n.method_name == name),
                None,
            )
            if nd is None:
                return

            for child_key in nd.children:
                child_nd = self.graph.nodes.get(child_key)
                if child_nd is not None:
                    _walk(child_nd.method_name)

        _walk(method_name)
        return result

    def get_node_status(self, method_name: str) -> dict[str, Any]:
        """Return detailed status of a node including flags and override state."""
        nd = self.graph._find_node_by_name(method_name)
        if nd is None:
            return {}

        has_override, override_value = self.graph._get_active_override(nd.key())

        return {
            "method_name": nd.method_name,
            "is_valid": nd.is_valid,
            "is_overridden": has_override,
            "override_value": override_value if has_override else None,
            "has_set_value": nd.has_flag(NodeFlag.CAN_SET) or nd.has_flag(NodeFlag.STORED),
            "flags": nd.flags,
            "current_value": nd.value,
            "compute_count": nd.compute_count,
            "node_type": nd.node_type.value,
        }

    # ------------------------------------------------------------------
    # Graphviz visualization
    # ------------------------------------------------------------------

    def to_graphviz(
        self,
        title: str | None = None,
        show_values: bool = True,
        show_counts: bool = False,
        rankdir: str = "BT",
    ) -> Any:
        """Build a ``graphviz.Digraph`` of the computation graph.

        Requires the ``graphviz`` Python package (``pip install graphviz``).

        Node colors:
        - **Green** — stored/input nodes (``STORED`` or ``CAN_SET``)
        - **Blue** — pure computed / derived nodes
        - **Red border** — invalid (dirty) nodes
        - **Gold fill** — currently overridden in a context/layer

        Edge style:
        - **Solid** — dependency within the same object
        - **Dashed gray** — cross-object dependency

        Args:
            title: Graph title. Defaults to the class name.
            show_values: Show cached values inside node labels.
            show_counts: Show compute counts inside node labels.
            rankdir: Graphviz rank direction (``"BT"`` = bottom-to-top,
                ``"TB"`` = top-to-bottom, ``"LR"`` = left-to-right).

        Returns:
            A ``graphviz.Digraph`` instance.  Call ``.render()`` to write
            to a file, or ``.pipe(format="png")`` for raw bytes.

        Raises:
            ImportError: If the ``graphviz`` package is not installed.
        """
        try:
            import graphviz  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "graphviz package is required for visualization. "
                "Install it with: pip install graphviz"
            ) from exc

        label = title or f"{self.obj.__class__.__name__} computation graph"
        dot = graphviz.Digraph(
            name="calyxos_graph",
            comment=label,
            graph_attr={
                "rankdir": rankdir,
                "label": label,
                "labelloc": "t",
                "fontname": "Helvetica",
                "fontsize": "16",
                "bgcolor": "#fafafa",
                "pad": "0.4",
                "nodesep": "0.6",
                "ranksep": "0.8",
            },
            node_attr={
                "fontname": "Helvetica",
                "fontsize": "11",
                "style": "filled",
                "shape": "record",
            },
            edge_attr={
                "fontname": "Helvetica",
                "fontsize": "9",
                "color": "#555555",
                "arrowsize": "0.7",
            },
        )

        nodes = self.graph.get_all_nodes()
        node_ids: dict[tuple[int, str, int], str] = {}

        for nd in nodes:
            nid = f"n_{nd.method_name}_{nd.args_hash}"
            node_ids[nd.key()] = nid

            # --- determine appearance ---
            has_override, _ = self.graph._get_active_override(nd.key())

            if has_override:
                fillcolor = "#fff3cd"  # gold — overridden
                fontcolor = "#856404"
            elif nd.node_type == NodeType.STORED or nd.has_flag(NodeFlag.CAN_SET):
                fillcolor = "#d4edda"  # green — input/stored
                fontcolor = "#155724"
            else:
                fillcolor = "#d1ecf1"  # blue — derived
                fontcolor = "#0c5460"

            border_color = "#dc3545" if not nd.is_valid else "#888888"
            penwidth = "2.5" if not nd.is_valid else "1.0"

            # --- build label ---
            flag_parts: list[str] = []
            if nd.has_flag(NodeFlag.STORED):
                flag_parts.append("STORED")
            elif nd.has_flag(NodeFlag.CAN_SET):
                flag_parts.append("CAN_SET")
            if nd.has_flag(NodeFlag.CAN_OVERRIDE):
                flag_parts.append("CAN_OVERRIDE")

            header = nd.method_name
            if flag_parts:
                header += f"  [{', '.join(flag_parts)}]"

            rows = [header]

            status = "OVERRIDDEN" if has_override else ("valid" if nd.is_valid else "INVALID")
            rows.append(f"status: {status}")

            if show_values and nd.value is not None:
                val_repr = repr(nd.value)
                if len(val_repr) > 40:
                    val_repr = val_repr[:37] + "..."
                rows.append(f"value: {val_repr}")

            if show_counts:
                rows.append(f"computes: {nd.compute_count}")

            label_str = "\\n".join(rows)

            dot.node(
                nid,
                label=label_str,
                fillcolor=fillcolor,
                fontcolor=fontcolor,
                color=border_color,
                penwidth=penwidth,
            )

        # --- edges (child -> parent = "depends on" direction) ---
        for nd in nodes:
            parent_id = node_ids.get(nd.key())
            if parent_id is None:
                continue
            for child_key in nd.children:
                child_id = node_ids.get(child_key)
                if child_id is not None:
                    # same-object edge
                    dot.edge(child_id, parent_id)
                else:
                    # cross-object edge — draw a placeholder node
                    cross_label = f"{child_key[1]}"
                    cross_id = f"cross_{child_key[0]}_{child_key[1]}_{child_key[2]}"
                    if cross_id not in node_ids:
                        node_ids[cross_id] = cross_id  # type: ignore[assignment]
                        dot.node(
                            cross_id,
                            label=f"{cross_label}\\n(external object)",
                            shape="ellipse",
                            style="filled,dashed",
                            fillcolor="#f8f9fa",
                            fontcolor="#6c757d",
                            color="#adb5bd",
                        )
                    dot.edge(cross_id, parent_id, style="dashed", color="#adb5bd")

        return dot

    def render(
        self,
        filename: str = "calyxos_graph",
        directory: str = ".",
        fmt: str = "png",
        view: bool = False,
        **kwargs: Any,
    ) -> str:
        """Render the computation graph to an image file.

        Convenience wrapper around :meth:`to_graphviz` + ``Digraph.render``.

        Args:
            filename: Output filename (without extension).
            directory: Output directory.
            fmt: Output format (``"png"``, ``"svg"``, ``"pdf"``).
            view: Open the rendered file in the default viewer.
            **kwargs: Forwarded to :meth:`to_graphviz`.

        Returns:
            The path to the rendered file.
        """
        dot = self.to_graphviz(**kwargs)
        path = dot.render(
            filename=filename,
            directory=directory,
            format=fmt,
            cleanup=True,
            view=view,
        )
        return path

    def _repr_svg_(self) -> str:
        """Jupyter notebook integration — render inline SVG."""
        dot = self.to_graphviz()
        return dot.pipe(format="svg").decode("utf-8")

    # ------------------------------------------------------------------
    # Interactive HTML visualization (vis.js)
    # ------------------------------------------------------------------

    def to_html(self, title: str | None = None) -> str:
        """Generate a self-contained interactive HTML page using vis.js.

        The HTML loads vis.js from CDN and renders the computation graph
        with draggable nodes, zoom/pan, and hover tooltips showing node
        details (value, flags, validity, compute count).

        No Python dependencies required beyond the stdlib.

        Args:
            title: Page title.  Defaults to the class name.

        Returns:
            A complete HTML string.
        """
        import html as html_mod
        import json

        label = title or f"{self.obj.__class__.__name__} computation graph"
        nodes_list = self.graph.get_all_nodes()

        vis_nodes: list[dict[str, Any]] = []
        vis_edges: list[dict[str, Any]] = []
        seen_cross: set[str] = set()

        for nd in nodes_list:
            nid = f"{nd.method_name}_{nd.args_hash}"
            has_override, override_val = self.graph._get_active_override(nd.key())

            # Color scheme
            if has_override:
                bg = "#fff3cd"
                border = "#ffc107"
                font_color = "#856404"
            elif nd.node_type == NodeType.STORED or nd.has_flag(NodeFlag.CAN_SET):
                bg = "#d4edda"
                border = "#28a745"
                font_color = "#155724"
            else:
                bg = "#d1ecf1"
                border = "#17a2b8"
                font_color = "#0c5460"

            if not nd.is_valid:
                border = "#dc3545"

            # Flags for label
            flag_parts: list[str] = []
            if nd.has_flag(NodeFlag.STORED):
                flag_parts.append("STORED")
            elif nd.has_flag(NodeFlag.CAN_SET):
                flag_parts.append("CAN_SET")
            if nd.has_flag(NodeFlag.CAN_OVERRIDE):
                flag_parts.append("CAN_OVERRIDE")
            flags_str = f"  [{', '.join(flag_parts)}]" if flag_parts else ""

            status = "OVERRIDDEN" if has_override else ("valid" if nd.is_valid else "INVALID")

            # Tooltip
            val_repr = repr(nd.value)
            if len(val_repr) > 80:
                val_repr = val_repr[:77] + "..."
            tooltip_lines = [
                f"Node: {nd.method_name}",
                f"Type: {nd.node_type.value}",
                f"Status: {status}",
                f"Value: {val_repr}",
                f"Flags: {nd.flags!r}",
                f"Compute count: {nd.compute_count}",
            ]
            if nd.last_recompute_reason:
                tooltip_lines.append(f"Last reason: {nd.last_recompute_reason}")
            if has_override:
                tooltip_lines.append(f"Override value: {override_val!r}")

            tooltip = html_mod.escape("\n".join(tooltip_lines))

            vis_nodes.append({
                "id": nid,
                "label": f"{nd.method_name}{flags_str}\n{status}",
                "title": tooltip,
                "color": {
                    "background": bg,
                    "border": border,
                    "highlight": {"background": bg, "border": "#333"},
                },
                "font": {"color": font_color, "size": 14, "face": "monospace"},
                "shape": "box",
                "borderWidth": 3 if not nd.is_valid else 1,
                "borderWidthSelected": 3,
            })

            # Edges
            for child_key in nd.children:
                child_nd = self.graph.nodes.get(child_key)
                if child_nd is not None:
                    child_id = f"{child_nd.method_name}_{child_nd.args_hash}"
                    vis_edges.append({
                        "from": child_id,
                        "to": nid,
                        "arrows": "to",
                        "color": {"color": "#555", "highlight": "#333"},
                    })
                else:
                    # Cross-object
                    cross_id = f"cross_{child_key[0]}_{child_key[1]}_{child_key[2]}"
                    if cross_id not in seen_cross:
                        seen_cross.add(cross_id)
                        vis_nodes.append({
                            "id": cross_id,
                            "label": f"{child_key[1]}\n(external)",
                            "title": f"Cross-object node: {child_key[1]}\nObject ID: {child_key[0]}",
                            "color": {
                                "background": "#f8f9fa",
                                "border": "#adb5bd",
                            },
                            "font": {"color": "#6c757d", "size": 12, "face": "monospace"},
                            "shape": "ellipse",
                            "borderWidth": 1,
                            "shapeProperties": {"borderDashes": [5, 5]},
                        })
                    vis_edges.append({
                        "from": cross_id,
                        "to": nid,
                        "arrows": "to",
                        "dashes": True,
                        "color": {"color": "#adb5bd"},
                    })

        nodes_json = json.dumps(vis_nodes, indent=2)
        edges_json = json.dumps(vis_edges, indent=2)
        title_escaped = html_mod.escape(label)

        return _VIS_HTML_TEMPLATE.format(
            title=title_escaped,
            nodes_json=nodes_json,
            edges_json=edges_json,
        )

    def open_in_browser(self, title: str | None = None) -> str:
        """Render the interactive graph and open it in the default browser.

        Writes a temporary HTML file and opens it.  The file persists so
        the browser tab keeps working after this method returns.

        Args:
            title: Page title.

        Returns:
            The path to the HTML file.
        """
        import tempfile
        import webbrowser

        html_content = self.to_html(title=title)
        fd, path = tempfile.mkstemp(suffix=".html", prefix="calyxos_graph_")
        with open(fd, "w") as f:
            f.write(html_content)
        webbrowser.open(f"file://{path}")
        return path

    def serve(
        self,
        title: str | None = None,
        port: int = 0,
    ) -> str:
        """Start a local HTTP server and open the interactive graph.

        The server runs in a background thread and serves a single HTML
        page.  Press Ctrl-C in the terminal to stop it, or just close
        the browser tab — the server shuts down automatically after
        serving the page.

        Args:
            title: Page title.
            port: Port number (0 = pick a free port automatically).

        Returns:
            The URL the graph is served at.
        """
        import http.server
        import threading
        import webbrowser

        html_bytes = self.to_html(title=title).encode("utf-8")

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass  # silence request logs

        server = http.server.HTTPServer(("127.0.0.1", port), _Handler)
        actual_port = server.server_address[1]
        url = f"http://127.0.0.1:{actual_port}"

        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        webbrowser.open(url)
        print(f"Serving graph at {url}  (Ctrl-C to stop)")
        return url


# ---------------------------------------------------------------------------
# vis.js HTML template (outside the class to keep it readable)
# ---------------------------------------------------------------------------

_VIS_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, sans-serif;
    background: #fafafa;
  }}
  #header {{
    padding: 16px 24px;
    background: #fff;
    border-bottom: 1px solid #e0e0e0;
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  #header h1 {{
    font-size: 18px;
    font-weight: 600;
    color: #1a1a1a;
  }}
  #header .legend {{
    display: flex;
    gap: 12px;
    font-size: 12px;
    color: #666;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #ccc;
  }}
  #graph {{
    width: 100vw;
    height: calc(100vh - 57px);
  }}
</style>
</head>
<body>
<div id="header">
  <h1>{title}</h1>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#d4edda;border-color:#28a745"></div>Stored / Input</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#d1ecf1;border-color:#17a2b8"></div>Computed</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#fff3cd;border-color:#ffc107"></div>Overridden</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#fff;border-color:#dc3545;border-width:2px"></div>Invalid</div>
  </div>
</div>
<div id="graph"></div>
<script>
var nodes = new vis.DataSet({nodes_json});
var edges = new vis.DataSet({edges_json});
var container = document.getElementById("graph");
var data = {{ nodes: nodes, edges: edges }};
var options = {{
  layout: {{
    hierarchical: {{
      enabled: true,
      direction: "DU",
      sortMethod: "directed",
      levelSeparation: 100,
      nodeSpacing: 160,
    }},
  }},
  physics: {{
    enabled: false,
  }},
  interaction: {{
    hover: true,
    tooltipDelay: 100,
    zoomView: true,
    dragView: true,
    dragNodes: true,
    navigationButtons: true,
    keyboard: {{ enabled: true }},
  }},
  edges: {{
    smooth: {{ type: "cubicBezier", forceDirection: "vertical" }},
    width: 1.5,
  }},
  nodes: {{
    margin: {{ top: 10, bottom: 10, left: 14, right: 14 }},
    shadow: {{ enabled: true, size: 4, x: 2, y: 2, color: "rgba(0,0,0,0.1)" }},
  }},
}};
var network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""
