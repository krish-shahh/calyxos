"""Tests for graphviz visualization."""

import os
import tempfile

import pytest

from calyxos.core.decorator import get_graph, node, set_value
from calyxos.core.flags import NodeFlag
from calyxos.utils.debug import GraphDebugger


class TestGraphvizVisualization:
    """Test graphviz rendering of computation graphs."""

    def test_to_graphviz_returns_digraph(self) -> None:
        """to_graphviz() returns a graphviz.Digraph object."""
        graphviz = pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

            @node()
            def y(self) -> int:
                return self.x() * 2

        m = Model()
        m.y()

        dbg = GraphDebugger(m)
        dot = dbg.to_graphviz()

        assert isinstance(dot, graphviz.Digraph)
        assert "x" in dot.source
        assert "y" in dot.source

    def test_to_graphviz_empty_graph(self) -> None:
        """to_graphviz() works on an empty graph (no nodes accessed yet)."""
        pytest.importorskip("graphviz")

        class Model:
            @node()
            def x(self) -> int:
                return 1

        m = Model()
        dbg = GraphDebugger(m)
        dot = dbg.to_graphviz()
        assert dot is not None

    def test_to_graphviz_shows_stored_flag(self) -> None:
        """STORED nodes show their flag in the label."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def data(self) -> int:
                return 42

        m = Model()
        m.data()

        dbg = GraphDebugger(m)
        dot = dbg.to_graphviz()
        assert "STORED" in dot.source

    def test_to_graphviz_invalid_nodes(self) -> None:
        """Invalid nodes are rendered with INVALID status."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

            @node()
            def y(self) -> int:
                return self.x() * 2

        m = Model()
        m.y()
        set_value(m, "x", 5)  # Invalidates y

        dbg = GraphDebugger(m)
        dot = dbg.to_graphviz()
        assert "INVALID" in dot.source

    def test_to_graphviz_overridden_nodes(self) -> None:
        """Overridden nodes show OVERRIDDEN status."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()
        graph = get_graph(m)

        with graph.context() as ctx:
            ctx.override(m, "x", 99)
            dbg = GraphDebugger(m)
            dot = dbg.to_graphviz()
            assert "OVERRIDDEN" in dot.source

    def test_render_creates_file(self) -> None:
        """render() writes an image file to disk."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

            @node()
            def y(self) -> int:
                return self.x() + 1

        m = Model()
        m.y()

        dbg = GraphDebugger(m)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = dbg.render(
                filename="test_graph",
                directory=tmpdir,
                fmt="png",
            )
            assert os.path.exists(path)
            assert path.endswith(".png")

    def test_render_svg_creates_file(self) -> None:
        """render() can produce SVG output."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def a(self) -> int:
                return 1

        m = Model()
        m.a()

        dbg = GraphDebugger(m)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = dbg.render(
                filename="test_svg",
                directory=tmpdir,
                fmt="svg",
            )
            assert os.path.exists(path)
            assert path.endswith(".svg")

    def test_to_graphviz_custom_options(self) -> None:
        """to_graphviz accepts customization options."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()

        dbg = GraphDebugger(m)
        dot = dbg.to_graphviz(
            title="Custom Title",
            show_values=False,
            show_counts=True,
            rankdir="LR",
        )
        assert "Custom Title" in dot.source
        assert "computes:" in dot.source
        assert "LR" in dot.source

    def test_to_graphviz_cross_object(self) -> None:
        """Cross-object dependencies render as dashed edges to placeholder nodes."""
        pytest.importorskip("graphviz")

        class Source:
            @node(NodeFlag.STORED)
            def val(self) -> int:
                return 10

        class Consumer:
            def __init__(self, src: Source):
                self.src = src

            @node()
            def result(self) -> int:
                return self.src.val() * 2

        s = Source()
        c = Consumer(s)
        c.result()

        dbg = GraphDebugger(c)
        dot = dbg.to_graphviz()
        assert "external object" in dot.source
        assert "dashed" in dot.source

    def test_repr_svg(self) -> None:
        """_repr_svg_ returns SVG string for Jupyter integration."""
        pytest.importorskip("graphviz")

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()

        dbg = GraphDebugger(m)
        svg = dbg._repr_svg_()
        assert "<svg" in svg
        assert "</svg>" in svg


class TestInteractiveHtml:
    """Test interactive HTML visualization (vis.js)."""

    def test_to_html_returns_valid_html(self) -> None:
        """to_html() returns a complete HTML document."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 10

            @node()
            def y(self) -> int:
                return self.x() * 2

        m = Model()
        m.y()

        dbg = GraphDebugger(m)
        html = dbg.to_html()

        assert "<!DOCTYPE html>" in html
        assert "vis-network" in html
        assert "x_" in html  # node id
        assert "y_" in html
        assert "</html>" in html

    def test_to_html_contains_node_data(self) -> None:
        """HTML includes node labels and status."""

        class Model:
            @node(NodeFlag.STORED)
            def base(self) -> float:
                return 42.0

            @node()
            def derived(self) -> float:
                return self.base() + 1

        m = Model()
        m.derived()

        html = GraphDebugger(m).to_html()
        assert "base" in html
        assert "derived" in html
        assert "STORED" in html

    def test_to_html_shows_invalid(self) -> None:
        """Invalid nodes appear in the HTML output."""

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

            @node()
            def y(self) -> int:
                return self.x() * 2

        m = Model()
        m.y()
        set_value(m, "x", 5)  # invalidates y

        html = GraphDebugger(m).to_html()
        assert "INVALID" in html
        assert "#dc3545" in html  # red border color

    def test_to_html_shows_overridden(self) -> None:
        """Overridden nodes appear with gold color."""

        class Model:
            @node(NodeFlag.CAN_OVERRIDE)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()
        graph = get_graph(m)

        with graph.context() as ctx:
            ctx.override(m, "x", 99)
            html = GraphDebugger(m).to_html()
            assert "OVERRIDDEN" in html
            assert "#fff3cd" in html  # gold bg

    def test_to_html_cross_object(self) -> None:
        """Cross-object dependencies appear as dashed edges."""

        class Source:
            @node(NodeFlag.STORED)
            def val(self) -> int:
                return 10

        class Consumer:
            def __init__(self, src: Source):
                self.src = src

            @node()
            def result(self) -> int:
                return self.src.val() * 2

        s = Source()
        c = Consumer(s)
        c.result()

        html = GraphDebugger(c).to_html()
        assert "external" in html
        assert '"dashes": true' in html or '"dashes":true' in html

    def test_to_html_custom_title(self) -> None:
        """Custom title appears in the HTML."""

        class Model:
            @node()
            def x(self) -> int:
                return 1

        m = Model()
        m.x()

        html = GraphDebugger(m).to_html(title="My Custom Graph")
        assert "My Custom Graph" in html

    def test_to_html_empty_graph(self) -> None:
        """to_html works on an empty graph."""

        class Model:
            @node()
            def x(self) -> int:
                return 1

        m = Model()
        html = GraphDebugger(m).to_html()
        assert "<!DOCTYPE html>" in html

    def test_open_in_browser_creates_file(self) -> None:
        """open_in_browser() creates an HTML file on disk."""
        import unittest.mock

        class Model:
            @node(NodeFlag.STORED)
            def x(self) -> int:
                return 1

        m = Model()
        m.x()
        dbg = GraphDebugger(m)

        # Mock webbrowser.open so we don't actually open a browser in tests
        with unittest.mock.patch("webbrowser.open"):
            path = dbg.open_in_browser()

        assert os.path.exists(path)
        assert path.endswith(".html")

        with open(path) as f:
            content = f.read()
        assert "vis-network" in content

        os.unlink(path)
