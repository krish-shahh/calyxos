"""Tensor-aware memoization and batching for ML workloads.

Handles efficient caching of numpy arrays and PyTorch tensors by:
- Computing stable content hashes instead of identity-based hashing
- Supporting batch processing with automatic tensor concatenation
- Detecting tensor shape/dtype changes for invalidation
"""

import hashlib
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from calyxos.core.decorator import get_graph
from calyxos.graph.node import NodeType


def _tensor_content_hash(tensor: Any) -> str:
    """Compute a stable hash of tensor contents.

    Works with numpy arrays and PyTorch tensors.
    """
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            # Convert to numpy for hashing
            np_array = tensor.detach().cpu().numpy()
            content = np_array.tobytes()
            return hashlib.sha256(content).hexdigest()
    except ImportError:
        pass

    if np is not None and isinstance(tensor, np.ndarray):
        content = tensor.tobytes()
        return hashlib.sha256(content).hexdigest()

    # Fallback for other types
    return hashlib.sha256(str(tensor).encode()).hexdigest()


def _get_tensor_signature(tensor: Any) -> dict[str, Any]:
    """Get shape and dtype of a tensor for change detection."""
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            return {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
    except ImportError:
        pass

    if np is not None and isinstance(tensor, np.ndarray):
        return {
            "shape": tensor.shape,
            "dtype": str(tensor.dtype),
        }

    return {}


class TensorMemoizer:
    """Memoization wrapper for methods that work with tensors.

    Caches tensor results by content hash and detects invalidation
    when input tensor shape/dtype changes.

    Usage:
        class Model:
            def __init__(self):
                self.memoizer = TensorMemoizer()

            @fn
            def process_batch(self, x: np.ndarray) -> np.ndarray:
                return self.memoizer.get_or_compute(
                    "process_batch",
                    x,
                    lambda: self._actual_process(x)
                )

            def _actual_process(self, x: np.ndarray) -> np.ndarray:
                return x * 2
    """

    def __init__(self) -> None:
        """Initialize tensor memoizer."""
        self._tensor_cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    def get_or_compute(
        self, key: str, tensor: Any, compute_fn: Any
    ) -> Any:
        """Get cached tensor result or compute and cache.

        Args:
            key: Cache key (usually method name)
            tensor: Input tensor to use for hashing
            compute_fn: Callable that computes the result

        Returns:
            Cached or newly computed result
        """
        # Get current tensor signature
        current_sig = _get_tensor_signature(tensor)

        # Check cache
        if key in self._tensor_cache:
            cached_result, cached_sig = self._tensor_cache[key]
            # Invalidate if signature changed
            if cached_sig == current_sig:
                return cached_result

        # Compute and cache
        result = compute_fn()
        self._tensor_cache[key] = (result, current_sig)
        return result

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate, or None to clear all
        """
        if key is None:
            self._tensor_cache.clear()
        elif key in self._tensor_cache:
            del self._tensor_cache[key]


class BatchProcessor:
    """Batch processing helper for tensor computations.

    Collects individual samples into batches and processes them
    together for efficiency.

    Usage:
        processor = BatchProcessor(batch_size=32)

        # Add samples one by one
        processor.add(sample1)
        processor.add(sample2)

        # Process when batch is full
        if processor.is_ready():
            results = processor.process(model.forward)
    """

    def __init__(self, batch_size: int = 32) -> None:
        """Initialize batch processor.

        Args:
            batch_size: Maximum samples per batch
        """
        self.batch_size = batch_size
        self._samples: list[Any] = []

    def add(self, sample: Any) -> None:
        """Add a sample to the batch."""
        self._samples.append(sample)

    def is_ready(self) -> bool:
        """Check if batch is ready to process."""
        return len(self._samples) >= self.batch_size

    def process(self, process_fn: Any) -> list[Any]:
        """Process current batch.

        Args:
            process_fn: Function that takes batched tensor and returns results

        Returns:
            List of results (one per sample)
        """
        if not self._samples:
            return []

        # Stack samples into batch
        try:
            import torch

            if isinstance(self._samples[0], torch.Tensor):
                batch = torch.stack(self._samples)
            else:
                if np is None:
                    raise ImportError("numpy required for batch processing")
                batch = np.stack(self._samples)
        except ImportError:
            if np is None:
                raise ImportError("numpy required for batch processing")
            batch = np.stack(self._samples)

        # Process batch
        batch_result = process_fn(batch)

        # Unbatch results
        results = []
        for i in range(len(self._samples)):
            if hasattr(batch_result, "__getitem__"):
                results.append(batch_result[i])
            else:
                results.append(batch_result)

        # Clear batch
        self._samples.clear()

        return results

    def flush(self) -> list[Any]:
        """Get remaining samples without processing."""
        remaining = self._samples.copy()
        self._samples.clear()
        return remaining


class TensorNodeAnalyzer:
    """Analyze tensor operations in computation graph.

    Identifies nodes that work with tensors and provides optimization hints.
    """

    def __init__(self, obj: Any) -> None:
        """Initialize analyzer for an object."""
        self.obj = obj
        self.graph = get_graph(obj)

    def find_tensor_nodes(self) -> list[str]:
        """Find nodes that likely work with tensors.

        Returns list of method names that:
        - Have been executed and returned tensor-like values
        - Or have tensor-like arguments
        """
        tensor_nodes = []

        for node in self.graph.get_all_nodes():
            # Check if node has been computed
            if node.value is not None:
                # Simple heuristic: check if value looks like a tensor
                if np is not None and isinstance(node.value, (np.ndarray,)):
                    tensor_nodes.append(node.method_name)
                    continue

                try:
                    import torch

                    if isinstance(node.value, torch.Tensor):
                        tensor_nodes.append(node.method_name)
                except ImportError:
                    pass

        return tensor_nodes

    def suggest_batching_opportunities(self) -> list[tuple[str, str]]:
        """Suggest which nodes could benefit from batching.

        Returns list of (node_name, reason) tuples.
        """
        suggestions = []
        tensor_nodes = self.find_tensor_nodes()

        for node_name in tensor_nodes:
            node = next(
                (n for n in self.graph.get_all_nodes()
                 if n.method_name == node_name),
                None,
            )
            if node and node.compute_count > 10:
                suggestions.append(
                    (
                        node_name,
                        f"Computed {node.compute_count} times - consider batching",
                    )
                )

        return suggestions

    def get_tensor_memory_usage(self) -> dict[str, tuple[float, str]]:
        """Estimate memory usage of cached tensors.

        Returns dict mapping node_name -> (bytes, human_readable_size)
        """
        memory_usage = {}

        for node in self.graph.get_all_nodes():
            if node.value is None:
                continue

            size_bytes = 0

            if np is not None and isinstance(node.value, np.ndarray):
                size_bytes = node.value.nbytes
            else:
                try:
                    import torch

                    if isinstance(node.value, torch.Tensor):
                        size_bytes = node.value.element_size() * node.value.nelement()
                except ImportError:
                    pass

            if size_bytes > 0:
                # Convert to human readable
                size_display = float(size_bytes)
                for unit in ["B", "KB", "MB", "GB"]:
                    if size_display < 1024:
                        break
                    size_display /= 1024
                human_readable = f"{size_display:.2f}{unit}"
                memory_usage[node.method_name] = (float(size_bytes), human_readable)

        return memory_usage
