"""
calyxos: Reactive object computation framework with memoized, dependency-aware nodes.

A framework that transforms methods on domain objects into memoized, dependency-aware
nodes within an instance-scoped computation graph. Methods decorated with @calyxos.fn are
evaluated lazily and cached, with calyxos recording runtime execution dependencies to
construct a directed acyclic graph that reflects actual method calls.
"""

from calyxos.core.async_support import async_fn
from calyxos.core.decorator import clear_graph, fn, get_graph, node, set_stored, set_value, stored
from calyxos.core.flags import CanOverride, CanSet, NodeFlag
from calyxos.core.flags import Stored as StoredFlag
from calyxos.core.introspection import (
    enable_dir,
    get_calyxos_methods,
    get_node_flags,
    is_overridden,
    is_set,
    list_computed_methods,
    list_stored_methods,
)
from calyxos.core.markers import Stored
from calyxos.core.reverse import NodeChange
from calyxos.tracking.disconnect import disconnect
from calyxos.graph.graph import ComputationGraph
from calyxos.graph.layer import Layer
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

__version__ = "0.2.1"

__all__ = [
    # Core decorators
    "fn",
    "stored",
    "async_fn",
    "node",
    # Node flags
    "NodeFlag",
    "CanSet",
    "CanOverride",
    "StoredFlag",
    # Value mutation
    "set_stored",
    "set_value",
    "get_graph",
    # Graph and storage
    "ComputationGraph",
    "Node",
    "Layer",
    "StorageBackend",
    "SQLiteStorage",
    "JSONStorage",
    "Stored",
    # Introspection
    "enable_dir",
    "get_calyxos_methods",
    "list_computed_methods",
    "list_stored_methods",
    "is_overridden",
    "is_set",
    "get_node_flags",
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
    # Tracking utilities
    "disconnect",
    # Reverse propagation
    "NodeChange",
    # Debugging
    "GraphDebugger",
    "clear_graph",
]
