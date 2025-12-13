# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/krish-shahh/calyxos/releases/tag/v0.1.0
