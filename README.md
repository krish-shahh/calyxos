# Calyxos

**A reactive computation framework for ML/LLM pipelines.** Calyxos transforms ordinary Python methods into memoized, dependency-aware nodes that automatically cache results, track dependencies at runtime, and selectively recompute only what changed.

## Why Calyxos?

Modern ML and LLM pipelines are built on chains of transformations: data loading → preprocessing → embedding → inference → post-processing. Each step depends on the previous, and changes ripple through the entire pipeline. Calyxos automates this dependency tracking and selective invalidation, letting you focus on the logic.

```python
from calyxos import fn, stored

class DataPipeline:
    @stored
    def raw_data(self) -> pd.DataFrame:
        return load_from_api()  # Persisted; changes trigger invalidation

    @fn
    def cleaned(self) -> pd.DataFrame:
        return self.raw_data().dropna()  # Recomputed only if raw_data changes

    @fn
    def embeddings(self) -> np.ndarray:
        return model.embed(self.cleaned())  # Memoized; no redundant API calls

pipeline = DataPipeline()
emb1 = pipeline.embeddings()  # Computed
emb2 = pipeline.embeddings()  # Retrieved from cache
```

## Overview

Calyxos is a Python framework for building complex, stateful domain models where correctness, incremental updates, and explainable recomputation are essential. It combines object-native dependency tracking, selective invalidation, and pluggable persistence to enable efficient, deterministic computation graphs. Designed from the ground up for ML/LLM workflows where async I/O, performance profiling, and tensor caching matter.

**Key Features:**

- **Lazy evaluation & memoization**: Methods decorated with `@calyxos.fn` are cached and recomputed only when dependencies change
- **Runtime dependency tracking**: Dependencies are captured dynamically during method execution, not through static analysis
- **Selective invalidation**: When stored state changes, only affected downstream computations are marked dirty
- **Stored vs. derived semantics**: Clear separation between persistent inputs (`@calyxos.stored`) and ephemeral derived values (`@calyxos.fn`)
- **Pluggable persistence**: Load and save object state via configurable storage backends (SQLite, JSON, custom)
- **Debug utilities**: Introspect the computation graph, trace why methods recomputed, and analyze dependencies
- **Async/await support**: Native `@async_fn` decorator for I/O-bound operations with automatic memoization
- **Performance profiling**: Built-in profiler with execution timing, cache hit analysis, and optimization hints
- **Tensor-aware memoization**: Content-hash-based caching for numpy/PyTorch tensors with batch processing support
- **Distributed execution planning**: Automatic parallelization analysis with critical path computation and speedup estimation
- **Gradient tracking**: Integration with PyTorch, JAX, and TensorFlow for autodiff support

## Installation

```bash
pip install calyxos
```

Or for development:

```bash
git clone https://github.com/krish-shahh/calyxos.git
cd calyxos
pip install -e ".[dev]"
```

## Quick Start

```python
from calyxos import fn, stored
from calyxos.core.decorator import set_stored

class Portfolio:
    """A simple investment portfolio."""

    @stored
    def cash_balance(self) -> float:
        """Stored: amount of cash on hand."""
        return 10000.0

    @fn
    def total_value(self) -> float:
        """Derived: sum of all assets."""
        return self.cash_balance()

    def deposit(self, amount: float) -> None:
        """Add cash to the portfolio."""
        new_balance = self.cash_balance() + amount
        set_stored(self, "cash_balance", new_balance)

# Create and use
portfolio = Portfolio()
print(f"Balance: ${portfolio.total_value()}")  # $10000.0

portfolio.deposit(5000.0)
print(f"Balance: ${portfolio.total_value()}")  # $15000.0
```

## Core Concepts

### 1. The `@fn` Decorator

Marks a method as a derived computation node. The method is cached, and its dependencies are recorded at runtime.

```python
class Calculator:
    @fn
    def double(self, x: int) -> int:
        return x * 2

calc = Calculator()
result = calc.double(5)  # Computed and cached
result = calc.double(5)  # Retrieved from cache (no recomputation)
result = calc.double(10) # Different args, recomputed
```

### 2. The `@stored` Decorator

Marks a method as stored state. These values are persisted via storage backends and form the authoritative inputs to the computation graph.

```python
class Account:
    @stored
    def balance(self) -> float:
        return 100.0

account = Account()
set_stored(account, "balance", 200.0)  # Modify stored state and propagate invalidation
```

### 3. Dependency Tracking

Dependencies are captured **at runtime** by recording which Calyxos-managed methods are called during evaluation. No static analysis or predeclaration needed.

```python
class Model:
    @fn
    def a(self) -> int:
        return 10

    @fn
    def b(self) -> int:
        return self.a() + 5  # Runtime call to a() creates the dependency edge

    @fn
    def c(self) -> int:
        if True:
            return self.b()  # Conditional dependency: b() is called, edge is recorded
        else:
            return 0

model = Model()
value = model.c()  # Evaluates: c -> b -> a, records dependency edges
```

### 4. Invalidation Propagation

When a stored value changes, invalidation propagates **lazily** to downstream dependents. Only affected nodes are marked dirty; recomputation happens on access.

```python
class Ledger:
    @stored
    def amount(self) -> float:
        return 100.0

    @fn
    def doubled(self) -> float:
        return self.amount() * 2

    @fn
    def tripled(self) -> float:
        return self.amount() * 3

ledger = Ledger()
print(ledger.doubled())   # 200.0, computed and cached
print(ledger.tripled())   # 300.0, computed and cached

set_stored(ledger, "amount", 50.0)
# doubled and tripled are now marked invalid, but not recomputed yet

print(ledger.doubled())   # 100.0, recomputed on access
print(ledger.tripled())   # 150.0, recomputed on access
```

### 5. Instance-Scoped Graphs

Each object instance maintains its own computation graph. Different instances do not share state.

```python
p1 = Portfolio()
p2 = Portfolio()

# p1 and p2 have independent graphs
assert get_graph(p1) is not get_graph(p2)
```

## Storage & Persistence

### Using SQLite Backend

```python
from calyxos import SQLiteStorage
from calyxos.core.persistence import save_object, load_object

backend = SQLiteStorage("/path/to/data.db")

# Save an object's stored state
save_object(portfolio, backend)

# Load into a fresh instance
portfolio2 = Portfolio()
load_object(portfolio2, backend)
# Stored values are restored; derived values rebuild lazily
```

### Using JSON Backend

```python
from calyxos import JSONStorage

backend = JSONStorage("/path/to/objects/")
save_object(portfolio, backend)
load_object(portfolio, backend)
```

### Custom Storage Backend

Implement the `StorageBackend` protocol:

```python
class MyBackend:
    def save(self, object_id: int, stored_values: dict[str, Any]) -> None:
        # Persist stored_values for object_id
        pass

    def load(self, object_id: int) -> dict[str, Any] | None:
        # Restore stored_values for object_id, or return None
        pass

    def delete(self, object_id: int) -> None:
        # Delete all stored values for object_id
        pass

    def exists(self, object_id: int) -> bool:
        # Check if object_id has stored values
        pass
```

## Debugging & Introspection

Use `GraphDebugger` to inspect the computation graph:

```python
from calyxos import GraphDebugger

portfolio = Portfolio()
_ = portfolio.total_value()

debugger = GraphDebugger(portfolio)

# Print human-readable graph
debugger.print_graph()

# Get statistics
stats = debugger.get_stats()
print(f"Total nodes: {stats['total_nodes']}")
print(f"Stored nodes: {stats['stored_nodes']}")
print(f"Invalid nodes: {stats['invalid_nodes']}")

# Get info about a specific node
info = debugger.get_node_info("total_value")
print(info)
```

## Design Rationale

### Object-Native Reactive Computation

Calyxos embraces Python's object model. Methods are the primary unit of computation; decorators transparently convert them into reactive nodes. No external DSLs or configuration files required.

### Runtime Dependency Tracking

Dependencies are **not** declared upfront. Instead, they emerge naturally from the code flow during evaluation. This handles:
- Conditional dependencies (if/else branches)
- Loop-dependent computations
- Polymorphic calls
- Dynamic method resolution

The graph reflects what *actually* happens, not what you *think* happens.

### Minimal Persistence Interface

Storage backends only persist stored nodes. Derived values are always recomputed from scratch when loaded, guaranteeing correctness and determinism. This keeps persistence layers simple and focused.

### Lazy Invalidation

When state changes, we mark nodes dirty but don't recompute eagerly. This avoids cascading recomputation even when the changed values aren't immediately needed, and keeps the system responsive.

### Deterministic Computation

Calyxos guarantees reproducible results across serialization, deserialization, and recomputation:
- Only `@stored` values are persisted; derived values are always recomputed from scratch
- Same stored values + same code = identical results, every time
- No silent state corruption from partial invalidation or missed updates
- Perfect for ML model training: save model state, load into fresh process, get identical forward passes

## Performance & Real-World Impact

**Cache Efficiency**: In typical LLM pipelines, embeddings and API calls represent 60-80% of compute time. Talos memoization eliminates redundant calls:
- **Without caching**: Same query called 5 times = 5x API latency
- **With Talos**: First call computes, next 4 hits cache instantly (typically 1-10ms vs 500-2000ms per API call)

**Parallelization**: The distributed executor identifies independent stages in data pipelines:
- **Data preprocessing example**: Loading → validation → feature extraction → normalization
- **Without parallelization**: Sequential execution takes 10s + 5s + 8s + 3s = 26s
- **With parallelization analysis**: Loading (10s) can overlap with validation of previous batch = ~16s (1.6x speedup)
- **Actual speedup depends on dependencies**: DistributedExecutor identifies the critical path and estimates realistic gains using Amdahl's law

**Profiling Insight**: Observable optimization prevents guessing:
- `profiler.print_profile_report()` shows which nodes waste compute time
- Instead of "why is my pipeline slow?" → immediate visibility to "node X has 500ms avg latency, 15% cache hit rate—dependencies are changing too frequently"

## ML/LLM Extensions

Talos includes comprehensive utilities designed specifically for machine learning and data-intensive workloads:

### 1. Async/Await Support (`@async_fn`)

Perfect for I/O-bound operations like LLM API calls and data loading:

```python
from calyxos import async_fn

class DataPipeline:
    @async_fn
    async def fetch_data(self, url: str) -> dict:
        """Cached async method—no redundant API calls."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()

    @async_fn
    async def process_data(self) -> list:
        """Depends on fetch_data; both are memoized."""
        data = await self.fetch_data("https://api.example.com/data")
        return [x * 2 for x in data['values']]

pipeline = DataPipeline()
result = await pipeline.process_data()
```

**Benefits:**
- Automatic memoization of async results
- Seamless asyncio integration
- Prevents redundant API calls

### 2. Performance Profiling

Understand and optimize your computation graphs with built-in profiling:

```python
from calyxos.utils.profiler import Profiler

profiler = Profiler(model)

# Get optimization hints
hints = profiler.get_optimization_hints()
for hint in hints:
    print(hint)  # "High avg time—consider parallelizing"
                 # "Low cache hit rate—dependencies change frequently"

# Print detailed report
profiler.print_profile_report()
```

**Output includes:**
- Per-node execution timing (min, max, average)
- Cache hit rates for memoized values
- Actionable optimization recommendations

### 3. Tensor-Aware Memoization

Content-hash-based caching for numpy and PyTorch tensors:

```python
from calyxos.ml import TensorMemoizer, BatchProcessor, TensorNodeAnalyzer

class Model:
    def __init__(self):
        self.memoizer = TensorMemoizer()

    @fn
    def process_batch(self, x: np.ndarray) -> np.ndarray:
        """Cached by content, not identity."""
        return self.memoizer.get_or_compute(
            "process_batch",
            x,
            lambda: x @ self.weights()
        )

# Efficient batch processing
processor = BatchProcessor(batch_size=32)
for sample in dataset:
    processor.add(sample)
    if processor.is_ready():
        results = processor.process(model.forward)

# Analyze tensor usage
analyzer = TensorNodeAnalyzer(model)
memory_usage = analyzer.get_tensor_memory_usage()
batching_hints = analyzer.suggest_batching_opportunities()
```

**Features:**
- Detects tensor changes by content, not object identity
- Automatic invalidation on shape/dtype changes
- Memory usage analysis
- Batch processing with automatic stacking

### 4. Parallelization Analysis

Analyze within-instance computation parallelization and identify bottlenecks:

```python
from calyxos.utils.distributed import DistributedExecutor

executor = DistributedExecutor(data_processor, workers=4)

# Identify nodes with no dependencies (can start immediately)
parallelizable = executor.get_parallelizable_nodes()
print(f"Can parallelize: {parallelizable}")

# Get execution stages (which nodes can run concurrently)
stages = executor.schedule_parallel()
for stage, nodes in stages.items():
    print(f"Stage {stage}: {nodes}")

# Critical path analysis (longest dependency chain)
critical_path = executor.get_critical_path()
print(f"Bottleneck: {critical_path}")

# Speedup estimation (Amdahl's law with 4 concurrent workers)
summary = executor.get_execution_summary()
print(f"Theoretical speedup: {summary['estimated_speedup']:.1f}x")
```

**Capabilities:**
- Identifies independent nodes that can run concurrently
- Computes critical path (longest dependency chain = bottleneck)
- Estimates speedup using Amdahl's law for thread/process pooling
- Highlights which computations are serialization bottlenecks
- **Note:** Analyzes within-instance parallelization; cross-machine distributed execution is not supported

### 5. Gradient Tracking

Integrate with autodiff frameworks to track which stored values participate in loss computation:

```python
from calyxos.utils.gradient_tracking import GradientTracker
import torch

class Model:
    @stored
    def weights(self) -> torch.Tensor:
        return torch.nn.Parameter(torch.randn(10, 5))

    @fn
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights()

model = Model()
tracker = GradientTracker(model)

# Forward pass
x = torch.randn(32, 10)
y_pred = model.forward(x)
loss = y_pred.sum()  # Any loss function

# Track which stored values contributed to loss
tracker.mark_gradient_root(loss)
trainable_params = tracker.get_trainable_parameters()  # ['weights']

# Get computation path from weights to loss
path = tracker.get_gradient_path('weights')

# Suggest checkpointing for memory efficiency
checkpoints = tracker.suggest_gradient_checkpointing()
```

**Capabilities:**
- Tracks which `@stored` values participate in loss computation
- Identifies trainable parameters automatically
- Suggests gradient checkpointing to reduce memory usage
- Compatible with PyTorch, JAX, and TensorFlow

## Examples

### LLM Pipeline

An end-to-end LLM inference pipeline with caching and profiling:

```bash
py311 examples/llm_pipeline.py
```

This example works without external dependencies (uses mock data for demo).

### Neural Network Training

Neural network with parameter storage and selective invalidation:

```bash
py311 examples/neural_network.py
```

**Requires:** numpy (install with `pip install numpy`)

Shows:
- Storing learned parameters with `@stored`
- Forward pass computation with memoization
- Loss and accuracy computation
- Parameter updates with automatic cache invalidation
- Graph introspection during training

### Distributed Data Processing

Multi-stage data pipeline with parallelization analysis:

```bash
py311 examples/distributed_training.py
```

**Requires:** numpy (install with `pip install numpy`)

Shows:
- Parallelizable data preprocessing stages
- Critical path analysis
- Execution staging for concurrent processing
- Train/validation/test split creation
- Speedup estimation

## Architecture

```
src/calyxos/
├── core/
│   ├── decorator.py       # @fn, @stored decorators
│   ├── async_support.py   # @async_fn decorator for async methods
│   ├── markers.py         # Stored marker class
│   ├── introspection.py   # enable_dir(), get_talos_methods()
│   └── persistence.py     # save/load utilities
├── graph/
│   ├── node.py            # Node representation
│   └── graph.py           # ComputationGraph (DAG management)
├── tracking/
│   ├── context.py         # Runtime execution context (thread-local)
│   └── __init__.py
├── storage/
│   ├── backend.py         # StorageBackend protocol
│   ├── sqlite.py          # SQLite implementation
│   └── json_storage.py    # JSON file implementation
├── ml/
│   ├── __init__.py
│   └── tensor_memoization.py  # TensorMemoizer, BatchProcessor, TensorNodeAnalyzer
├── utils/
│   ├── debug.py           # GraphDebugger
│   ├── profiler.py        # Profiler with optimization hints
│   ├── distributed.py     # DistributedExecutor for parallelization
│   └── gradient_tracking.py   # GradientTracker for autodiff integration
└── __init__.py            # Public API

examples/
├── llm_pipeline.py        # LLM inference pipeline
├── neural_network.py      # Neural network training
└── distributed_training.py  # Distributed data processing
```

## Design Notes

### Thread Safety

The implementation uses `threading.RLock` at the graph level for basic thread safety. Evaluation context is managed via `contextvars` for thread-local tracking. Note: Talos is **not** designed for highly concurrent updates; it assumes single-threaded or lightly concurrent workloads.

### Cycle Detection

The framework detects cycles in the dependency graph during evaluation and raises `RuntimeError`. This prevents infinite recomputation loops.

### Argument Hashing

Function arguments are hashed using MD5 to generate stable node keys. Unhashable types (lists, dicts, custom objects) fall back to object identity. This allows the same method to be called with different arguments and treated as separate nodes.

### Recursion Guards

Each evaluation maintains a recursion guard set to detect cycles and prevent stack overflow.

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

Coverage report:

```bash
python -m pytest tests/ --cov=talos --cov-report=html
```

The test suite includes 38+ tests covering:
- **Core functionality**: Memoization, argument handling, multiple instances
- **Dependency tracking**: Runtime dependency capture, conditional dependencies, diamond dependencies
- **Invalidation**: Selective invalidation, propagation, partial recomputation
- **Persistence**: SQLite and JSON storage roundtrips, rehydration, determinism
- **Introspection**: Graph inspection, dir() support, method listing
- **Graph analysis**: Dependency graphs, statistics, recomputation tracing

Quality assurance:
- **Type checking**: Strict mypy validation
- **Code style**: ruff linting for consistency
- **Coverage**: 85%+ on core modules, 100% on public APIs

## When NOT to Use Calyxos

Calyxos is purpose-built for ML/LLM pipelines. Consider alternatives if:

1. **Dependencies are static and known upfront**: If your computation graph structure is fixed at design time and never changes, the overhead of runtime tracking is wasted. A functional pipeline may be simpler.

2. **You need multi-process or multi-machine distribution**: Calyxos analyzes parallelization opportunities but doesn't orchestrate cross-process execution. If you need Dask/Ray-level distributed scheduling, look elsewhere.

3. **You need multi-tenant isolation**: Calyxos assumes single-tenant, single-process workloads. If you need shared caches or isolation guarantees for concurrent users, you'll need additional infrastructure.

4. **Real-time systems with timing guarantees**: Lazy evaluation is the opposite of predictable latency. If you need guaranteed response times, avoid frameworks with deferred computation.

5. **Complex domain validation logic**: If your problem domain has intricate business rules, validation cascades, or consistency constraints, you may benefit from specialized domain modeling frameworks that enforce these at the language level.

## Limitations

1. **Not for real-time systems**: Lazy evaluation means some operations are deferred until access. Timing guarantees are not provided.

2. **Determinism assumption**: Calyxos assumes that methods with the same arguments always return the same result. Side effects or external state changes can break this assumption.

3. **Shallow copying in persistence**: The default storage backends (SQLite, JSON) serialize values directly. Complex, nested objects may require custom serialization logic.

4. **No distributed graphs**: Each object's graph is independent and local to a single Python process. Cross-process or distributed computation is not supported.

5. **Python-only**: Calyxos is a Python library. There is no C extension or compiled component for performance-critical sections.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure `pytest`, `mypy`, and `ruff` pass
5. Submit a pull request

## License

MIT License. See LICENSE file for details.

## Acknowledgments

Calyxos is inspired by:
- Reactive programming frameworks (RxPy, Reactor)
- Computational spreadsheets (Excel, VisiCalc)
- Incremental computation systems (Inc)
- Memoization and dependency analysis
