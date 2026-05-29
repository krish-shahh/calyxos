# calyxos

[![PyPI version](https://img.shields.io/pypi/v/calyxos.svg)](https://pypi.org/project/calyxos/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/calyxos.svg)](https://pypi.org/project/calyxos/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**A reactive dependency graph computation engine for Python.** calyxos turns ordinary methods into memoized, dependency-aware nodes that automatically cache results, track dependencies at runtime, and selectively recompute only what changed. Early cutoff stops recomputation cascades when intermediate results haven't actually changed, and `@map_node` fans out computation per-element so a single document change doesn't rebuild an entire pipeline. Inspired by Jane Street's [Incremental](https://github.com/janestreet/incremental) library, built for Python's object model.

```python
from calyxos import node, NodeFlag, set_value, get_graph

class Portfolio:
    @node(NodeFlag.STORED)
    def spot(self) -> float:
        return 100.0

    @node()
    def market_value(self) -> float:
        return self.spot() * 1000  # dependency tracked automatically

    @node()
    def tax(self) -> float:
        return self.market_value() * 0.15

p = Portfolio()
print(p.tax())          # 15000.0 — computed once, cached

set_value(p, "spot", 120.0)
print(p.tax())          # 18000.0 — only affected nodes recomputed
```

## Installation

```bash
pip install calyxos
```

From source:

```bash
git clone https://github.com/krish-shahh/calyxos.git
cd calyxos
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+. Zero runtime dependencies (stdlib only).

Optional extras:

```bash
pip install calyxos[tui]    # interactive TUI inspector (rich)
pip install calyxos[mlx]    # MLX tensor backend (Apple Silicon)
pip install calyxos[viz]    # Graphviz visualization
```

## TUI Inspector

calyxos ships with a built-in terminal UI for exploring computation graphs interactively.

**Run the demos** (benchmark + inspector):

```bash
calyxos demo          # core reactive graph benchmark + TUI
calyxos mlx-demo      # MLX tensor backend benchmark + TUI
```

**Inspect your own objects** in code:

```python
from calyxos import node, NodeFlag, set_value, inspect

class MyModel:
    @node(NodeFlag.CAN_SET)
    def x(self) -> float: return 10.0

    @node()
    def result(self) -> float: return self.x() ** 2

m = MyModel()
m.result()       # compute the graph
inspect(m)       # drop into the TUI
```

**Commands** inside the TUI:

| Command | What it does |
|---------|-------------|
| `graph` | Show all nodes with status, values, flags |
| `flow` | Layered DAG view of the full graph |
| `node <name>` | Inspect a single node (deps, dependents, flags, TTL, error) |
| `tree <name>` | Dependency tree from a node |
| `set <name> <value>` | Set a value, shows which nodes were invalidated |
| `eval <name>` | Evaluate a node |
| `stats` | Graph statistics (includes error count) |
| `invalid` | List all dirty nodes |
| `errors` | List all errored nodes |
| `retry` | Clear error state for recomputation |
| `gc` | Remove orphaned per-element map nodes |
| `quit` | Exit |

## MLX Backend (Apple Silicon)

calyxos includes an execution backend for [MLX](https://github.com/ml-explore/mlx) that brings incremental recomputation to tensor workloads on Apple Silicon. When you change one weight in an 8-stage transformer pipeline, only the affected stages rerun. Everything else returns cached lazy arrays. A single `mx.eval()` call fuses the result.

```bash
pip install calyxos[mlx]
```

```python
from calyxos.ml.mlx_graph import MLXGraph
import mlx.core as mx

g = MLXGraph()

# register inputs
x = g.var("x", mx.ones((4, 4)))
w = g.var("w", mx.random.normal((4, 4)))

# register computation nodes
h = g.node("h", lambda: g["x"].value @ g["w"].value, inputs=["x", "w"])
out = g.node("out", lambda: mx.relu(g["h"].value), inputs=["h"])

# evaluate (single mx.eval call, fused on GPU/ANE)
g.eval("out")
print(out.value)

# mutate one input — only downstream nodes recompute
g["w"].set(mx.random.normal((4, 4)))
print(g.stale_nodes())   # ['h', 'out'] — x is unchanged
g.eval("out")             # only h and out rerun
```

**Run the MLX demo** (simplified transformer benchmark + TUI):

```bash
calyxos mlx-demo
```

The MLX TUI inspector has its own commands tailored for tensor graphs (`set <var> random`, `eval`, `stale`, `flow`).

**Benchmark results** (dim=512, seq=256, Apple Silicon):
- ~1.5x speedup when mutating a single mid-graph weight
- ~30% less peak Metal memory vs full rebuild

The backend uses version-based staleness detection (no array hashing), preserves MLX's lazy evaluation semantics, and never calls `mx.eval()` until you ask for it.

## Core Concepts

### The `@node` Decorator

The unified `@node` decorator is the primary API. Flags control behavior:

| Flag | Meaning |
|------|---------|
| *(none)* | Pure computed node. Cached, recomputed when deps change. |
| `CAN_SET` | Value can be explicitly set via `set_value()`. |
| `CAN_OVERRIDE` | Value can be temporarily overridden in a context or layer. |
| `STORED` | Persistent node (implies `CAN_SET`). Saved via storage backends. |

```python
from calyxos import node, NodeFlag

class Model:
    @node(NodeFlag.STORED)
    def learning_rate(self) -> float:
        return 0.01                          # persistent input

    @node(NodeFlag.CAN_SET, NodeFlag.CAN_OVERRIDE)
    def temperature(self) -> float:
        return 1.0                           # settable + overridable

    @node()
    def output(self) -> float:
        return self.learning_rate() * self.temperature()  # pure computed
```

The legacy `@fn` and `@stored` decorators still work. `@fn` = `@node()`, `@stored` = `@node(NodeFlag.STORED)`.

### Dependency Tracking

Dependencies are captured at runtime by recording which nodes are accessed during evaluation. No static analysis or declarations needed.

```python
class Pipeline:
    @node(NodeFlag.STORED)
    def raw_data(self) -> list:
        return [1, 2, 3]

    @node()
    def processed(self) -> list:
        return [x * 2 for x in self.raw_data()]  # dep recorded automatically

    @node()
    def summary(self) -> float:
        return sum(self.processed()) / len(self.processed())
```

### Per-Element Mapping with `@map_node`

`@map_node(source)` applies a function independently to each element of a collection, creating per-element nodes in the dependency graph. When the collection changes, only new or modified elements are recomputed — unchanged elements return their cached result.

```python
from calyxos import node, NodeFlag, set_value, map_node

class Pipeline:
    @node(NodeFlag.STORED)
    def documents(self) -> list[str]:
        return ["doc_a", "doc_b", "doc_c"]

    @map_node("documents")
    def processed(self, doc: str) -> str:
        return doc.upper()  # expensive per-document transform

    @node()
    def report(self) -> str:
        return ", ".join(self.processed())

p = Pipeline()
print(p.report())  # "DOC_A, DOC_B, DOC_C"

# Add a document — only "doc_d" is computed, the rest are cached
set_value(p, "documents", ["doc_a", "doc_b", "doc_c", "doc_d"])
print(p.report())  # "DOC_A, DOC_B, DOC_C, DOC_D"
```

Per-element nodes can depend on other nodes too. If a shared config changes, all element nodes are invalidated, but early cutoff skips elements whose result didn't actually change:

```python
class Pipeline:
    @node(NodeFlag.STORED)
    def multiplier(self) -> int:
        return 2

    @node(NodeFlag.STORED)
    def numbers(self) -> list[int]:
        return [1, 2, 3]

    @map_node("numbers")
    def scaled(self, n: int) -> int:
        return n * self.multiplier()  # each element depends on multiplier

p = Pipeline()
print(p.scaled())  # [2, 4, 6]

set_value(p, "multiplier", 10)
print(p.scaled())  # [10, 20, 30]
```

### Lazy Invalidation & Early Cutoff

When a node's value changes, calyxos marks all transitive dependents as invalid but does **not** recompute them eagerly. Recomputation happens lazily on next access.

```python
set_value(pipeline, "raw_data", [10, 20, 30])
# processed and summary are now invalid, but NOT recomputed yet

pipeline.summary()  # triggers recomputation of processed, then summary
```

**Value-equality guard:** `set_value()` short-circuits when the new value matches the existing cached value (`is` identity check, then `==`). Re-setting the same value is a no-op — no invalidation cascade, no wasted work.

```python
set_value(p, "spot", 100.0)  # spot is already 100.0
# Nothing happens — dependents remain valid and cached
```

**Early cutoff:** When a dirty node is recomputed and produces the same result as before, calyxos stops the cascade. Downstream nodes that depend on the unchanged intermediate are re-validated without running their compute functions.

```python
class Model:
    @node(NodeFlag.CAN_SET)
    def raw_input(self) -> int:
        return 5

    @node()
    def clamped(self) -> int:
        return max(0, min(10, self.raw_input()))  # clamps to [0, 10]

    @node()
    def expensive_report(self) -> str:
        return f"value={self.clamped()}"  # only reruns if clamped changes

m = Model()
m.expensive_report()  # "value=5"

set_value(m, "raw_input", 15)
m.expensive_report()  # "value=10" — clamped changed (5→10), report recomputes

set_value(m, "raw_input", 20)
m.expensive_report()  # "value=10" — clamped unchanged (still 10), report skipped
```

## What-If Analysis with Contexts

Contexts let you temporarily override node values for scenario analysis. Overrides revert automatically on exit. Contexts are nestable.

```python
model = Model()
graph = get_graph(model)

# Base case
print(model.output())  # 0.01

# Scenario: what if temperature is 2.0?
with graph.context() as ctx:
    ctx.override(model, "temperature", 2.0)
    print(model.output())  # 0.02 — dependents recompute

# Automatically reverted
print(model.output())  # 0.01

# Nested scenarios
with graph.context() as outer:
    outer.override(model, "temperature", 2.0)
    with graph.context() as inner:
        inner.override(model, "temperature", 5.0)
        print(model.output())  # 0.05
    print(model.output())  # 0.02 — inner reverted, outer still active
print(model.output())  # 0.01 — both reverted
```

## Sensitivity Analysis with Layers

Layers are like contexts but **preserve computed state** after exit. Re-entering a layer restores the cached computation without rerunning anything. This is ideal for sensitivity analysis where you bump an input, compute results, exit, then re-enter later.

```python
graph = get_graph(model)
layer = graph.layer("bump_temp")

with layer:
    set_value(model, "temperature", 2.0)
    result = model.output()  # computed once

# Base case restored
print(model.output())  # 0.01

# Re-enter: cached, no recomputation
with layer:
    print(model.output())  # 0.02 — instant, from snapshot
```

## Cross-Object Dependencies

Nodes on different objects can depend on each other. calyxos tracks these cross-object edges and propagates invalidation across object boundaries.

```python
class Market:
    @node(NodeFlag.STORED)
    def spot(self) -> float:
        return 100.0

class Instrument:
    def __init__(self, market: Market):
        self.market = market

    @node()
    def price(self) -> float:
        return self.market.spot() * 1.05  # cross-object dependency

mkt = Market()
inst = Instrument(mkt)
print(inst.price())  # 105.0

set_value(mkt, "spot", 200.0)
print(inst.price())  # 210.0 — automatically invalidated and recomputed
```

Internally, objects are tracked via weak references so they can be garbage collected normally.

## Reverse Propagation

Nodes can define a `get_changes` callback for bidirectional binding. Setting a derived node's value propagates upstream to modify the appropriate input.

```python
from calyxos import NodeChange

class Model:
    @node(NodeFlag.CAN_SET)
    def x(self) -> float:
        return 1.0

    @node(
        NodeFlag.CAN_SET,
        get_changes=lambda self, val: [NodeChange(self, "x", val / 2)]
    )
    def two_x(self) -> float:
        return self.x() * 2

m = Model()
print(m.two_x())  # 2.0

set_value(m, "two_x", 10.0)  # reverse-propagates: x = 5.0
print(m.x())      # 5.0
print(m.two_x())  # 10.0
```

## Disconnect

The `disconnect()` context manager suppresses dependency tracking. Useful for reading node values for logging or debugging without creating spurious edges.

```python
from calyxos import disconnect

class Logger:
    @node(NodeFlag.STORED)
    def data(self) -> int:
        return 42

    @node()
    def result(self) -> int:
        with disconnect():
            print(f"[log] data={self.data()}")  # no dependency created
        return 99  # independent of data
```

## Error Handling & Retry

If `compute_fn()` raises an exception, calyxos captures it on the node. Subsequent accesses re-raise the stored error until the node is retried or its inputs change.

```python
graph = get_graph(model)

try:
    model.flaky_fetch()
except ConnectionError:
    pass

# Inspect error state
print(graph.get_error_nodes())  # [Node(method=flaky_fetch, ...)]

# Retry all errored nodes (clears error, marks dirty)
graph.retry_errors()
model.flaky_fetch()  # recomputes
```

## TTL & Cache Eviction

Nodes can have a time-to-live. After the TTL expires, the cached value is treated as stale and recomputed on next access.

```python
@node(ttl=3600)  # 1 hour
def embedding(self) -> list[float]:
    return call_embedding_api(self.text())

# Force-expire all TTL nodes
graph.evict_expired()
```

## Automatic Profiling

The `Profiler` hooks into `evaluate_node` automatically — no manual `start_timer` / `stop_timer` calls.

```python
from calyxos import Profiler

prof = Profiler(model)
prof.enable()               # attach to the graph

model.expensive_pipeline()  # automatically timed

prof.print_profile_report() # table + optimization hints
prof.disable()              # detach
```

## Parallel Execution

`DistributedExecutor` evaluates independent nodes concurrently using a thread pool.

```python
from calyxos import DistributedExecutor

executor = DistributedExecutor(model, workers=4)
results = executor.execute()  # evaluates all nodes, parallelising within stages
```

## Async Evaluation

`@async_fn` and `@async_map_node` use `asyncio.gather` to evaluate independent branches concurrently — ideal for pipelines that call external APIs.

```python
from calyxos import async_fn, async_map_node

class Pipeline:
    @node(NodeFlag.STORED)
    def urls(self) -> list[str]:
        return ["http://a.com/api", "http://b.com/api"]

    @async_map_node("urls")
    async def fetched(self, url: str) -> dict:
        async with aiohttp.ClientSession() as s:
            async with s.get(url) as r:
                return await r.json()

p = Pipeline()
results = asyncio.run(p.fetched())  # all URLs fetched concurrently
```

## Storage & Persistence

Only `@node(NodeFlag.STORED)` values are persisted. Derived values recompute from inputs on load, guaranteeing determinism. Storage keys are user-supplied strings that remain stable across process restarts.

```python
from calyxos import SQLiteStorage, JSONStorage
from calyxos.core.persistence import save_object, load_object

backend = SQLiteStorage("data.db")

# Save with a stable key
save_object(model, backend, key="pipeline-main")

# In a different process / after restart
model2 = MyPipeline()
load_object(model2, backend, key="pipeline-main")
# Stored values restored, derived values recompute lazily
```

Implement the `StorageBackend` protocol for custom backends.

## Graph Visualization

Render computation graphs as images using [Graphviz](https://graphviz.org/). Install the optional dependency with `pip install calyxos[viz]`.

```python
from calyxos import GraphDebugger

dbg = GraphDebugger(model)

# Render to file
dbg.render("my_graph", directory=".", fmt="png")

# Get a graphviz.Digraph for programmatic use
dot = dbg.to_graphviz(show_values=True, show_counts=True, rankdir="BT")
dot.render("output", format="svg")
```

Node colors indicate state at a glance:

| Color | Meaning |
|-------|---------|
| Green | Stored / settable input node |
| Blue | Pure computed / derived node |
| Red border | Invalid (dirty) node |
| Gold | Currently overridden in a context or layer |

Dashed edges indicate cross-object dependencies.

![Portfolio graph](docs/portfolio_graph.png)
![Invalid nodes after mutation](docs/portfolio_invalid.png)

Jupyter notebooks get inline SVG rendering automatically via `_repr_svg_`.

## Graph Introspection

```python
from calyxos import GraphDebugger, is_overridden, get_node_flags

dbg = GraphDebugger(model)

# Dependency tree (text)
print(dbg.dump_dependency_tree("output"))

# All nodes involved in a computation
print(dbg.list_computing_nodes("output"))

# Node status (validity, flags, override state)
print(dbg.get_node_status("temperature"))

# Check override state
print(is_overridden(model, "temperature"))

# Get flags
print(get_node_flags(model, "temperature"))
```

## Architecture

calyxos is organized into four layers:

![calyxos Architecture](docs/architecture.png)

D2 sources live in `docs/` — re-render with `d2 docs/architecture.d2 docs/architecture.png`.

```
src/calyxos/
├── core/                    # Decorators, flags, reverse propagation
│   ├── decorator.py         # @node, @fn, @stored, set_value, get_graph
│   ├── flags.py             # NodeFlag enum (CAN_SET, CAN_OVERRIDE, STORED)
│   ├── reverse.py           # NodeChange for bidirectional binding
│   ├── introspection.py     # is_overridden, get_node_flags, etc.
│   └── persistence.py       # save/load utilities
├── graph/                   # Computation graph engine
│   ├── graph.py             # ComputationGraph (evaluation, invalidation)
│   ├── node.py              # Node dataclass (value, flags, edges)
│   ├── context.py           # GraphContext for temporary overrides
│   ├── layer.py             # Layer for persistent computation snapshots
│   └── registry.py          # CrossObjectRegistry (weak-ref tracking)
├── tracking/                # Runtime dependency tracking
│   ├── context.py           # EvaluationFrame stack (contextvars)
│   └── disconnect.py        # disconnect() context manager
├── storage/                 # Pluggable persistence backends
│   ├── backend.py           # StorageBackend protocol
│   ├── sqlite.py            # SQLiteStorage
│   └── json_storage.py      # JSONStorage
├── ml/                      # ML extensions
│   ├── mlx_graph.py         # MLX execution backend (MLXGraph, MLXVar, MLXNode)
│   └── tensor_memoization.py # Tensor memoization utilities
└── utils/                   # Debugging, profiling, analysis
    ├── debug.py             # GraphDebugger
    ├── profiler.py          # Performance profiling
    ├── distributed.py       # Parallelization analysis
    └── gradient_tracking.py # Autodiff integration
```

### Key Design Decisions

- **Runtime tracking over static analysis**: Dependencies are discovered by recording node accesses during execution, not by parsing AST. This handles conditional deps, loops, and polymorphism correctly.
- **Lazy invalidation with early cutoff**: Changing a value marks dependents dirty but doesn't recompute. On access, dirty nodes verify whether their inputs actually changed before running `compute_fn`. If no input changed, the cached value is re-validated without recomputation, and the cutoff propagates upward.
- **Value-equality guard**: `set_value()` compares new values against cached values and short-circuits when they match, preventing invalidation cascades from no-op mutations.
- **Per-element fan-out with GC**: `@map_node` creates independent sub-nodes for each collection element. When elements are removed, their orphaned nodes are garbage collected automatically.
- **Error capture and retry**: If `compute_fn` raises, the exception is stored on the node. Subsequent accesses re-raise it cleanly. `graph.retry_errors()` clears error state for recomputation.
- **TTL-based expiry**: Nodes can have a time-to-live. Expired values are treated as stale and recomputed on next access.
- **Async-native**: `@async_fn` and `@async_map_node` use `asyncio.gather` to evaluate independent branches concurrently.
- **Instance-scoped graphs**: Each object has its own computation graph. Cross-object edges use weak references.
- **Zero dependencies**: Core uses only Python stdlib (threading, contextvars, hashlib, dataclasses).

## Examples

```bash
# Getting started with @node, flags, caching
python examples/reactive_basics.py

# Contexts for what-if scenario analysis
python examples/what_if_analysis.py

# Layers for sensitivity analysis (bump-and-recompute)
python examples/sensitivity_analysis.py

# Cross-object deps, reverse propagation, introspection
python examples/financial_instrument.py

# Graphviz visualization (requires: pip install graphviz)
python examples/graph_visualization.py

# MLX incremental benchmark (requires: pip install calyxos[mlx])
python benchmarks/mlx_incremental.py
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Type checking
mypy src/calyxos/

# Linting
ruff check src/calyxos/
```

The test suite includes 145 tests covering:

- Core memoization and argument handling
- Dependency tracking (conditional, diamond, cross-object)
- Invalidation propagation (selective, lazy, cross-graph)
- Value-equality guard and early cutoff optimization
- `@map_node` per-element caching, dependency tracking, and GC
- Error handling (capture, re-raise, retry)
- TTL / cache eviction
- Automatic profiler instrumentation
- Parallel execution with `DistributedExecutor.execute()`
- Async evaluation (`@async_fn`, `@async_map_node`)
- Contexts (override, revert, nesting, exception safety)
- Layers (snapshot, restore, re-entry, independence)
- Reverse propagation (basic, chained, recursion limit)
- Storage with stable string keys (SQLite, JSON, cross-process roundtrip)
- Enhanced introspection (tree dumps, node status, flags)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `pytest`, `mypy`, and `ruff` pass
5. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

calyxos is inspired by:

- [Jane Street Incremental](https://github.com/janestreet/incremental) — incremental computation for OCaml
- [Salsa](https://github.com/salsa-rs/salsa) — incremental computation for Rust
- [MobX](https://mobx.js.org/) — reactive state management for JavaScript
- Computational spreadsheets — the original reactive dependency graphs
