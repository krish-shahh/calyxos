"""
Talos: Reactive object computation framework with memoized, dependency-aware nodes.

A framework that transforms methods on domain objects into memoized, dependency-aware
nodes within an instance-scoped computation graph. Methods decorated with @calyxos.fn are
evaluated lazily and cached, with Talos recording runtime execution dependencies to
construct a directed acyclic graph that reflects actual method calls.
"""

from calyxos.core.async_support import async_fn
from calyxos.core.decorator import clear_graph, fn, stored
from calyxos.core.introspection import (
    enable_dir,
    get_calyxos_methods,
    list_computed_methods,
    list_stored_methods,
)
from calyxos.core.markers import Stored
from calyxos.graph.graph import ComputationGraph
from calyxos.graph.node import Node
from calyxos.ml.tensor_memoization import (
    BatchProcessor,
    TensorMemoizer,
    TensorNodeAnalyzer,
)
from calyxos.storage.backend import StorageBackend
from calyxos.storage.json_storage import JSONStorage
from calyxos.storage.sqlite import SQLiteStorage
from calyxos.utils.debug import GraphDebugger
from calyxos.utils.distributed import DistributedExecutor, NodeExecutionPlan
from calyxos.utils.gradient_tracking import GradientTracker, enable_autograd_tracking
from calyxos.utils.profiler import Profiler

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
    "get_calyxos_methods",
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
