# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-05-28

### Added

#### ReactiveGraph Engine
- **Unified `@node` decorator** with `NodeFlag` enum (`CAN_SET`, `CAN_OVERRIDE`, `STORED`) — replaces the need to choose between `@fn`/`@stored` upfront
- **Context system** (`graph.context()`) — scoped, nestable overrides for what-if analysis with automatic reversion on exit
- **Layer system** (`graph.layer()`) — persistent computation snapshots that survive exit and restore on re-entry, ideal for sensitivity analysis
- **Cross-object dependencies** — nodes on different objects can depend on each other with automatic invalidation propagation across object boundaries; uses weak references for GC safety
- **Reverse propagation** (`get_changes` / `NodeChange`) — bidirectional binding where setting a derived node's value propagates upstream to modify the appropriate input
- **`disconnect()` context manager** — suppresses dependency tracking for side-effect reads (logging, debugging) without creating spurious edges
- **`set_value()` function** — generalized value mutation that validates flags, handles reverse propagation, and works with `CAN_SET` nodes (not just `STORED`)

#### Enhanced Introspection
- `dump_dependency_tree()` — text tree of all nodes involved in computing a given node
- `list_computing_nodes()` — flat list of transitive dependencies
- `get_node_status()` — detailed dict with validity, override state, flags, and current value
- `is_overridden()`, `is_set()`, `get_node_flags()` — quick-check introspection functions

#### New Examples
- `reactive_basics.py` — getting started with `@node`, flags, memoization, selective invalidation
- `what_if_analysis.py` — contexts for scenario analysis, nested contexts, disconnect
- `sensitivity_analysis.py` — layers for bump-and-recompute workflows, rate sensitivity ladder
- `financial_instrument.py` — cross-object deps, reverse propagation, full graph introspection

#### Documentation
- Complete README rewrite with quickstart, core concepts, and API coverage for all new features
- D2 architecture diagrams (`docs/architecture.d2`, `docs/invalidation-flow.d2`, `docs/context-layer.d2`)

### Changed
- `@fn` and `@stored` are now thin wrappers around `@node()` and `@node(NodeFlag.STORED)` respectively — fully backward compatible
- `set_stored()` now delegates to `set_value()` internally
- `Node` dataclass gains `flags`, `get_changes_fn`, and `has_flag()` method
- `ComputationGraph.invalidate_node()` now propagates across object boundaries via `CrossObjectRegistry`
- Test suite expanded from 38 to 91 tests
- Version bumped to 0.2.0

### Fixed
- Invalidation BFS no longer incorrectly marks cross-object parent keys as visited (which prevented cross-graph propagation)

## [0.1.0] - 2024-12-13

### Added

#### Core Features
- `@fn` decorator for lazy evaluation and memoization of computed methods
- `@stored` decorator for persistent state with automatic invalidation
- `@async_fn` decorator for async methods with automatic memoization
- Runtime dependency tracking via contextvars (no static analysis needed)
- Selective invalidation with lazy recomputation
- Instance-scoped computation graphs with cycle detection
- Pluggable persistence with SQLite and JSON backends

#### Storage Backends
- SQLiteStorage for persistent object state
- JSONStorage for file-based persistence
- StorageBackend protocol for custom implementations

#### ML/LLM Extensions
- Async/await support for I/O-bound operations
- Performance profiling with optimization hints (Profiler class)
- Tensor-aware memoization (TensorMemoizer, BatchProcessor)
- Distributed execution planning with critical path analysis (DistributedExecutor)
- Gradient tracking for autodiff frameworks (GradientTracker)
- TensorNodeAnalyzer for identifying tensor operations

#### Developer Tools
- GraphDebugger for computation graph introspection
- Graph statistics and recomputation tracing
- Introspection utilities (enable_dir, get_calyxos_methods, list_stored_methods)
- Comprehensive docstrings and type hints (mypy strict mode)

#### Documentation
- Comprehensive README with examples and trade-offs
- 3 ML-focused examples:
  - LLM inference pipeline with async caching
  - Neural network training with selective invalidation
  - Distributed data processing with parallelization analysis

### Quality Assurance
- 38 comprehensive test cases covering core functionality
- 85%+ test coverage on core modules
- 100% type hint coverage with mypy strict configuration
- ruff linting for code consistency
- Zero production dependencies

### First Release Features
- Minimal, focused feature set optimized for ML/LLM pipelines
- Clean, production-ready codebase
- Honest documentation about capabilities and limitations
- Clear positioning for open source adoption

## Future Roadmap (Post-v0.1.0)

### Planned Enhancements
- Advanced tensor batching strategies
- Complete autodiff framework hooks (PyTorch backward, JAX vjp, TensorFlow GradientTape)
- Actual distributed worker pool execution
- Hardware-aware scheduling (GPU/CPU placement)
- Integration with Apache Spark / Dask
- Real-time profiling dashboards

### Possible Additions
- C++ extensions for performance-critical sections
- Caching policy customization (LRU, TTL, etc.)
- Streaming tensor support for large datasets
- Database-backed distributed graphs
- REST API for remote execution

---

[0.2.0]: https://github.com/krish-shahh/calyxos/releases/tag/v0.2.0
[0.1.0]: https://github.com/krish-shahh/calyxos/releases/tag/v0.1.0
