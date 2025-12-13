"""Automatic gradient flow tracking for ML/autodiff support."""

from typing import Any

from calyxos.core.decorator import get_graph


class GradientTracker:
    """
    Track gradient flow through the Talos computation graph.

    Enables integration with PyTorch, JAX, and other autodiff libraries
    by recording which stored values participate in loss computation.

    Usage:
        import torch
        from calyxos import fn, stored
        from calyxos.utils.gradient_tracking import GradientTracker

        class Model:
            def __init__(self):
                self.params = torch.nn.Parameter(torch.randn(10))

            @stored
            def weights(self) -> torch.Tensor:
                return self.params

            @fn
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weights()

        model = Model()
        tracker = GradientTracker(model)

        # Forward pass
        x = torch.randn(5, 10)
        output = model.forward(x)
        loss = output.sum()

        # Track which stored values contributed to the loss
        tracker.mark_gradient_root(loss)
        params_in_loss = tracker.get_gradient_roots()
    """

    def __init__(self, obj: Any) -> None:
        """Initialize gradient tracker for an object."""
        self.obj = obj
        self.graph = get_graph(obj)
        self._gradient_roots: set[str] = set()
        self._gradient_paths: dict[str, set[str]] = {}

    def mark_gradient_root(self, value: Any) -> None:
        """
        Mark a value as a gradient root (e.g., loss tensor in ML).

        For PyTorch: mark the loss tensor to track which parameters it depends on.
        For JAX: mark the output to enable reverse-mode differentiation.
        """
        # In a real implementation, this would trace backward through the graph
        # and record which stored values participated in computing this value
        pass

    def get_gradient_roots(self) -> list[str]:
        """Get stored node names that participate in gradient computation."""
        return sorted(self._gradient_roots)

    def get_gradient_path(self, stored_name: str) -> list[str]:
        """Get the computation path from a stored value to the loss."""
        return sorted(self._gradient_paths.get(stored_name, set()))

    def suggest_gradient_checkpointing(self) -> list[tuple[str, float]]:
        """
        Suggest which nodes to use gradient checkpointing on.

        Returns list of (method_name, memory_savings_estimate) tuples.
        Useful for reducing memory during backpropagation in large models.
        """
        suggestions = []

        for node in self.graph.get_all_nodes():
            # Suggest checkpointing for expensive derived nodes with many dependents
            if node.node_type.value == "derived" and len(node.parents) > 2:
                # Estimate: memory savings ≈ node size * num_parents
                estimated_savings = 1.0 * len(node.parents)
                suggestions.append((node.method_name, estimated_savings))

        return sorted(suggestions, key=lambda x: x[1], reverse=True)

    def get_trainable_parameters(self) -> list[str]:
        """Get list of stored nodes that should be trainable parameters."""
        return [
            node.method_name
            for node in self.graph.get_stored_nodes()
            if node.method_name in self._gradient_roots
        ]


def enable_autograd_tracking(obj: Any, framework: str = "pytorch") -> GradientTracker:
    """
    Enable automatic differentiation tracking for an object.

    Args:
        obj: Talos-managed object
        framework: "pytorch", "jax", or "tensorflow"

    Returns:
        GradientTracker instance
    """
    tracker = GradientTracker(obj)

    if framework == "pytorch":
        _enable_pytorch_tracking(obj, tracker)
    elif framework == "jax":
        _enable_jax_tracking(obj, tracker)
    elif framework == "tensorflow":
        _enable_tensorflow_tracking(obj, tracker)

    return tracker


def _enable_pytorch_tracking(obj: Any, tracker: GradientTracker) -> None:
    """Hook into PyTorch's autograd graph."""
    # Placeholder: In a full implementation, this would register backward hooks
    # on tensors to track gradient flow through Talos nodes
    pass


def _enable_jax_tracking(obj: Any, tracker: GradientTracker) -> None:
    """Hook into JAX's transformation machinery."""
    # Placeholder: In a full implementation, this would integrate with
    # jax.grad, jax.value_and_grad, and custom vjp/jvp rules
    pass


def _enable_tensorflow_tracking(obj: Any, tracker: GradientTracker) -> None:
    """Hook into TensorFlow's GradientTape."""
    # Placeholder: In a full implementation, this would register watched variables
    # and tape callbacks to track gradient flow
    pass
