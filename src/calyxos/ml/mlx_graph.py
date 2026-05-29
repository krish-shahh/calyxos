"""MLX execution backend for calyxos reactive graphs.

Provides version-tracked tensor variables and lazy computation nodes that
integrate with MLX's graph-based evaluation model.  Key invariants:

1. **Version-based invalidation** – each ``MLXVar`` carries a monotonic version
   counter.  Downstream nodes compare stored input-version snapshots against
   current versions to decide staleness.  We *never* hash or compare
   ``mx.array`` contents, which would force premature materialization.

2. **Lazy through the graph** – ``MLXNode.value`` always returns an unevaluated
   ``mx.array`` (the result of composing MLX ops).  The only place
   ``mx.eval()`` is called is ``MLXGraph.eval()``, preserving MLX's kernel
   fusion across the entire recomputed subgraph.

3. **Incremental recomputation** – when an input changes only the transitive
   dependents are recomputed; unchanged subtrees return their cached lazy
   arrays directly.

Requires ``mlx`` (``pip install calyxos[mlx]``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

try:
    import mlx.core as mx
except ImportError as _exc:
    raise ImportError(
        "MLX is required for the mlx_graph module. "
        "Install it with: pip install calyxos[mlx]"
    ) from _exc


# ---------------------------------------------------------------------------
# MLXVar – a versioned input tensor
# ---------------------------------------------------------------------------


class MLXVar:
    """A named, versioned input variable holding an ``mx.array``.

    Every call to :meth:`set` increments the version counter.  Downstream
    :class:`MLXNode` instances use this version – *not* the array contents –
    to decide whether they are stale.

    Parameters
    ----------
    name:
        Human-readable identifier (must be unique within an :class:`MLXGraph`).
    value:
        Initial ``mx.array``.  Stored as-is (lazy – no ``mx.eval``).
    """

    __slots__ = ("name", "_value", "_version")

    def __init__(self, name: str, value: mx.array) -> None:
        self.name = name
        self._value = value
        self._version: int = 0

    # -- public API ----------------------------------------------------------

    @property
    def value(self) -> mx.array:
        """Current (possibly lazy) array."""
        return self._value

    @property
    def version(self) -> int:
        """Monotonically increasing mutation counter."""
        return self._version

    def set(self, value: mx.array) -> None:
        """Replace the held array and bump the version."""
        self._value = value
        self._version += 1

    def __repr__(self) -> str:
        return f"MLXVar({self.name!r}, v={self._version})"


# ---------------------------------------------------------------------------
# MLXNode – a lazy computation node
# ---------------------------------------------------------------------------

# Sentinel indicating "never computed"
_UNSET: Any = object()


class MLXNode:
    """A computation node that caches a lazy ``mx.array``.

    The node stores the version of every input (``MLXVar`` or upstream
    ``MLXNode``) that was current when it last computed.  On access it
    compares those stored versions against the live versions; if any
    differ the node recomputes.  The recomputation itself returns a *lazy*
    ``mx.array`` – no ``mx.eval()`` is called.

    Parameters
    ----------
    name:
        Unique identifier within the graph.
    fn:
        A callable ``(*mx.array) -> mx.array`` that performs the
        computation.  Receives the *values* of its inputs in the same
        order as *inputs*.
    inputs:
        Ordered sequence of ``MLXVar`` / ``MLXNode`` this node depends on.
    """

    __slots__ = (
        "name",
        "_fn",
        "_inputs",
        "_cached",
        "_input_versions",
        "_version",
    )

    def __init__(
        self,
        name: str,
        fn: Callable[..., mx.array],
        inputs: list[MLXVar | MLXNode],
    ) -> None:
        self.name = name
        self._fn = fn
        self._inputs: list[MLXVar | MLXNode] = list(inputs)
        self._cached: mx.array | Any = _UNSET
        self._input_versions: list[int] = []  # snapshot at last compute
        self._version: int = 0  # bumped on every recompute

    # -- public API ----------------------------------------------------------

    @property
    def version(self) -> int:
        """Recomputation counter (used by downstream nodes for staleness)."""
        return self._version

    @property
    def is_stale(self) -> bool:
        """``True`` if any input has been updated since last compute.

        Checks both direct version changes (an ``MLXVar`` was ``.set()``)
        and *transitive* staleness (an upstream ``MLXNode``'s own inputs
        changed even though it hasn't recomputed yet).
        """
        if self._cached is _UNSET:
            return True
        if len(self._input_versions) != len(self._inputs):
            return True
        for stored, inp in zip(self._input_versions, self._inputs):
            if inp.version != stored:
                return True
            # An upstream node may be transitively stale even though its
            # version hasn't bumped yet (it hasn't recomputed).
            if isinstance(inp, MLXNode) and inp.is_stale:
                return True
        return False

    @property
    def value(self) -> mx.array:
        """Return the cached lazy array, recomputing first if stale.

        **No ``mx.eval()`` is called here.**  The returned array is a
        lazy computation graph that MLX will fuse when eventually
        evaluated.
        """
        if self.is_stale:
            self._recompute()
        return self._cached

    # -- internals -----------------------------------------------------------

    def _recompute(self) -> None:
        """Run ``fn`` on current input values and cache the lazy result."""
        input_values = [inp.value for inp in self._inputs]
        self._cached = self._fn(*input_values)
        self._input_versions = [inp.version for inp in self._inputs]
        self._version += 1

    def __repr__(self) -> str:
        stale = self.is_stale
        return f"MLXNode({self.name!r}, stale={stale}, v={self._version})"


# ---------------------------------------------------------------------------
# MLXGraph – the DAG orchestrator
# ---------------------------------------------------------------------------


class MLXGraph:
    """A directed acyclic graph of :class:`MLXVar` and :class:`MLXNode`.

    The graph is the single entry-point for:

    * registering variables and computation nodes,
    * querying staleness,
    * materializing results via a single ``mx.eval()`` call.

    Example
    -------
    ::

        g = MLXGraph()
        x = g.var("x", mx.ones((4, 4)))
        w = g.var("w", mx.ones((4, 4)))
        y = g.node("y", lambda x, w: x @ w, [x, w])
        z = g.node("z", lambda y: mx.softmax(y, axis=-1), [y])

        result = g.eval(z)          # single mx.eval()
        x.set(mx.zeros((4, 4)))     # mutate one input
        result2 = g.eval(z)         # only y, z recompute; w untouched
    """

    def __init__(self) -> None:
        self._vars: dict[str, MLXVar] = {}
        self._nodes: dict[str, MLXNode] = {}
        # Insertion-ordered list for topological traversal
        self._topo_order: list[str] = []

    # -- construction --------------------------------------------------------

    def var(self, name: str, value: mx.array) -> MLXVar:
        """Create and register a new input variable."""
        if name in self._vars or name in self._nodes:
            raise ValueError(f"Duplicate name in graph: {name!r}")
        v = MLXVar(name, value)
        self._vars[name] = v
        return v

    def node(
        self,
        name: str,
        fn: Callable[..., mx.array],
        inputs: list[MLXVar | MLXNode],
    ) -> MLXNode:
        """Create and register a new computation node.

        *inputs* must already be registered in this graph.  Nodes must be
        added in dependency order (a node's inputs must exist before it).
        """
        if name in self._vars or name in self._nodes:
            raise ValueError(f"Duplicate name in graph: {name!r}")
        # Validate inputs belong to this graph
        for inp in inputs:
            if isinstance(inp, MLXVar) and inp.name not in self._vars:
                raise ValueError(f"Input var {inp.name!r} not registered")
            if isinstance(inp, MLXNode) and inp.name not in self._nodes:
                raise ValueError(f"Input node {inp.name!r} not registered")
        n = MLXNode(name, fn, inputs)
        self._nodes[name] = n
        self._topo_order.append(name)
        return n

    # -- query ---------------------------------------------------------------

    def stale_nodes(self) -> list[MLXNode]:
        """Return the list of nodes that need recomputation, in topo order."""
        return [
            self._nodes[name]
            for name in self._topo_order
            if self._nodes[name].is_stale
        ]

    def summary(self) -> dict[str, Any]:
        """Return a diagnostic summary of the graph state."""
        stale = self.stale_nodes()
        return {
            "vars": len(self._vars),
            "nodes": len(self._nodes),
            "stale": len(stale),
            "stale_names": [n.name for n in stale],
        }

    # -- evaluation ----------------------------------------------------------

    def eval(self, *targets: MLXNode) -> list[mx.array] | mx.array:
        """Materialize one or more node outputs with a single ``mx.eval()``.

        This is the **only** place ``mx.eval()`` is called.  All upstream
        nodes that are stale will lazily recompute first (building the MLX
        computation graph), then a single ``mx.eval()`` fuses and executes
        the entire subgraph on the GPU/ANE.

        Returns a single ``mx.array`` if one target is given, otherwise a
        list.
        """
        # Touch .value on each target (triggers lazy recompute cascade)
        arrays = [t.value for t in targets]
        # Single mx.eval materialises everything
        mx.eval(*arrays)
        if len(arrays) == 1:
            return arrays[0]
        return arrays

    def eval_timed(
        self, *targets: MLXNode
    ) -> tuple[list[mx.array] | mx.array, float]:
        """Like :meth:`eval` but also returns wall-clock seconds."""
        t0 = time.perf_counter()
        result = self.eval(*targets)
        elapsed = time.perf_counter() - t0
        return result, elapsed
