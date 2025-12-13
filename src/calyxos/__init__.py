"""
Talos: Reactive object computation framework with memoized, dependency-aware nodes.

A framework that transforms methods on domain objects into memoized, dependency-aware
nodes within an instance-scoped computation graph. Methods decorated with @talos.fn are
evaluated lazily and cached, with Talos recording runtime execution dependencies to
construct a directed acyclic graph that reflects actual method calls.
"""

from talos.core.async_support import async_fn
from talos.core.decorator import clear_graph, fn, stored
from talos.core.introspection import (
    enable_dir,
    get_talos_methods,
    list_computed_methods,
    list_stored_methods,
)
from talos.core.markers import Stored
from talos.graph.graph import ComputationGraph
from talos.graph.node import Node
from talos.ml.tensor_memoization import (
    BatchProcessor,
    TensorMemoizer,
    TensorNodeAnalyzer,
)
from talos.storage.backend import StorageBackend
from talos.storage.json_storage import JSONStorage
from talos.storage.sqlite import SQLiteStorage
from talos.utils.debug import GraphDebugger
from talos.utils.distributed import DistributedExecutor, NodeExecutionPlan
from talos.utils.gradient_tracking import GradientTracker, enable_autograd_tracking
from talos.utils.profiler import Profiler

__version__ = "0.1.0"

__all__ = [
    # Core decorators
    "fn",
    "stored",
    "async_fn",
    # Graph and storage
    "ComputationGraph",
    "Node",
    "StorageBackend",
    "SQLiteStorage",
    "JSONStorage",
    "Stored",
    # Introspection
    "enable_dir",
    "get_talos_methods",
    "list_computed_methods",
    "list_stored_methods",
    # ML/Tensor utilities
    "TensorMemoizer",
    "BatchProcessor",
    "TensorNodeAnalyzer",
    # Profiling and optimization
    "Profiler",
    "GradientTracker",
    "enable_autograd_tracking",
    # Distributed execution
    "DistributedExecutor",
    "NodeExecutionPlan",
    # Debugging
    "GraphDebugger",
    "clear_graph",
]
