"""Neural network training example with Talos.

Demonstrates:
- Storing model parameters and configuration
- Computing derived values (forward pass, loss, metrics)
- Selective invalidation on parameter updates
- Gradient tracking for efficient backpropagation
"""

from typing import Any

import numpy as np

from talos import fn, stored


class SimpleNeuralNetwork:
    """Simple feedforward neural network with Talos integration.

    Shows how to structure an ML model for automatic dependency tracking
    and selective recomputation.

    Example usage:
        model = SimpleNeuralNetwork(input_dim=10, hidden_dim=20, output_dim=2)

        # Training loop
        for epoch in range(10):
            # Generate or load batch
            X = np.random.randn(32, 10)
            y = np.random.randint(0, 2, 32)

            # Forward pass (memoized)
            logits = model.forward(X)

            # Compute loss (memoized, depends on logits + targets)
            loss = model.compute_loss(logits, y)

            # Backward pass (would integrate with autograd)
            gradients = model.compute_gradients(logits, y)

            # Update parameters (invalidates forward/loss/gradients)
            model.update_parameters(gradients, learning_rate=0.001)
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 20,
        output_dim: int = 2,
        seed: int = 42,
    ) -> None:
        """Initialize neural network.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output class dimension
            seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seed = seed

        # Initialize parameters
        np.random.seed(seed)
        self._w1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self._w2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self._b1 = np.zeros((1, hidden_dim))
        self._b2 = np.zeros((1, output_dim))

    @stored
    def w1(self) -> np.ndarray:
        """First layer weights (stored for persistence)."""
        return self._w1.copy()

    @stored
    def w2(self) -> np.ndarray:
        """Second layer weights (stored for persistence)."""
        return self._w2.copy()

    @stored
    def b1(self) -> np.ndarray:
        """First layer biases (stored for persistence)."""
        return self._b1.copy()

    @stored
    def b2(self) -> np.ndarray:
        """Second layer biases (stored for persistence)."""
        return self._b2.copy()

    @fn
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network.

        Args:
            x: Input batch of shape (batch_size, input_dim)

        Returns:
            Output logits of shape (batch_size, output_dim)
        """
        # Hidden layer with ReLU
        w1 = self.w1()
        b1 = self.b1()
        z1 = x @ w1 + b1
        h = np.maximum(z1, 0)  # ReLU

        # Output layer (logits)
        w2 = self.w2()
        b2 = self.b2()
        logits = h @ w2 + b2

        return logits

    @fn
    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Apply softmax to logits.

        Args:
            logits: Raw network outputs

        Returns:
            Probability distribution
        """
        # Numerical stability: subtract max before exp
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    @fn
    def compute_loss(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute cross-entropy loss.

        Args:
            logits: Model output logits
            targets: Target class indices (0 to num_classes-1)

        Returns:
            Scalar loss value
        """
        probs = self.softmax(logits)
        batch_size = logits.shape[0]

        # Cross-entropy loss
        correct_logprobs = -np.log(probs[np.arange(batch_size), targets])
        loss = np.mean(correct_logprobs)

        return float(loss)

    @fn
    def compute_accuracy(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute classification accuracy.

        Args:
            logits: Model output logits
            targets: Target class indices

        Returns:
            Accuracy as fraction (0-1)
        """
        predictions = np.argmax(logits, axis=1)
        correct = np.sum(predictions == targets)
        accuracy = correct / len(targets)
        return float(accuracy)

    @fn
    def compute_gradients(
        self, logits: np.ndarray, targets: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute gradients via backpropagation.

        In a real implementation, would use autodiff framework.
        Here we compute numerical gradients for demo.

        Args:
            logits: Model output logits
            targets: Target class indices

        Returns:
            Dictionary of gradients for each parameter
        """
        # Simple numerical gradient computation (for demo)
        eps = 1e-4
        gradients = {}

        for param_name in ["w1", "w2", "b1", "b2"]:
            param = getattr(self, param_name)()
            grad = np.zeros_like(param)

            # Compute numerical gradients (slow but simple)
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    param[i, j] += eps
                    setattr(self, f"_{param_name}", param.copy())
                    loss_plus = self.compute_loss(logits, targets)

                    param[i, j] -= 2 * eps
                    setattr(self, f"_{param_name}", param.copy())
                    loss_minus = self.compute_loss(logits, targets)

                    grad[i, j] = (loss_plus - loss_minus) / (2 * eps)
                    param[i, j] += eps

            gradients[param_name] = grad

        return gradients

    def update_parameters(
        self, gradients: dict[str, np.ndarray], learning_rate: float = 0.01
    ) -> None:
        """Update parameters using gradients.

        This invalidates all derived values (forward, loss, metrics).

        Args:
            gradients: Dictionary of gradients for each parameter
            learning_rate: Learning rate for parameter update
        """
        from talos.core.decorator import get_graph

        # Update parameters directly
        for param_name, grad in gradients.items():
            param = getattr(self, f"_{param_name}")
            setattr(self, f"_{param_name}", param - learning_rate * grad)

        # Invalidate all derived nodes
        graph = get_graph(self)
        for node in graph.get_all_nodes():
            if node.method_name not in ["w1", "w2", "b1", "b2"]:
                graph.invalidate_node(node.method_name, node.args_hash)


class ConvolutionalNetwork:
    """Simple CNN example (simplified for Talos integration).

    Demonstrates handling of higher-dimensional data and learned filters.
    """

    def __init__(
        self,
        num_filters: int = 32,
        filter_size: int = 3,
        output_classes: int = 10,
    ) -> None:
        """Initialize CNN.

        Args:
            num_filters: Number of convolutional filters
            filter_size: Size of filter kernel
            output_classes: Number of output classes
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.output_classes = output_classes

        # Initialize filters (simplified)
        np.random.seed(42)
        self._filters = np.random.randn(
            num_filters, filter_size, filter_size, 3
        ) * 0.01

    @stored
    def filters(self) -> np.ndarray:
        """Convolutional filters (stored for persistence)."""
        return self._filters.copy()

    @fn
    def convolve(self, x: np.ndarray) -> np.ndarray:
        """Apply convolution (simplified implementation).

        In practice, would use efficient convolution operations.

        Args:
            x: Input image batch

        Returns:
            Convolved feature maps
        """
        # This is simplified - real implementation would use
        # proper convolution with padding, stride, etc.
        filters = self.filters()
        # Stub: return average across spatial dimensions
        return x.mean(axis=(1, 2), keepdims=True)

    @fn
    def pool(self, x: np.ndarray) -> np.ndarray:
        """Apply max pooling (simplified).

        Args:
            x: Input feature maps

        Returns:
            Pooled feature maps
        """
        return x  # Stub for demo

    @fn
    def classify(self, pooled: np.ndarray) -> np.ndarray:
        """Classify from pooled features.

        Args:
            pooled: Pooled feature maps

        Returns:
            Class logits
        """
        # Stub: return random logits for demo
        return np.random.randn(pooled.shape[0], self.output_classes)


if __name__ == "__main__":
    # Demo usage
    print("Neural Network Example")
    print("=" * 60)

    model = SimpleNeuralNetwork(input_dim=10, hidden_dim=20, output_dim=2)

    # Example 1: Forward pass
    print("\n1. Forward pass (will be cached)...")
    X_batch = np.random.randn(32, 10)
    logits = model.forward(X_batch)
    print(f"   Input shape: {X_batch.shape}")
    print(f"   Output shape: {logits.shape}")

    # Example 2: Compute loss
    print("\n2. Computing loss...")
    y_batch = np.random.randint(0, 2, 32)
    loss = model.compute_loss(logits, y_batch)
    print(f"   Loss: {loss:.4f}")

    # Example 3: Compute accuracy
    print("\n3. Computing accuracy...")
    accuracy = model.compute_accuracy(logits, y_batch)
    print(f"   Accuracy: {accuracy:.1%}")

    # Example 4: Using cached values
    print("\n4. Demonstrating caching...")
    logits2 = model.forward(X_batch)
    print(f"   Second call uses cache: same logits = {np.allclose(logits, logits2)}")

    # Example 5: Parameter update invalidates cache
    print("\n5. Updating parameters (invalidates cache)...")
    from talos.core.decorator import get_graph

    graph = get_graph(model)
    before_update = graph.get_or_create_node(
        "forward", 0, None, lambda: None
    ).is_valid

    # Simple parameter update
    model._w1 = model._w1 * 0.99  # Small update
    model.update_parameters(
        {"w1": np.zeros_like(model.w1()), "w2": np.zeros_like(model.w2()),
         "b1": np.zeros_like(model.b1()), "b2": np.zeros_like(model.b2())},
        learning_rate=0.01,
    )

    # Forward pass now recomputes with new parameters
    logits3 = model.forward(X_batch)
    print(f"   After parameter update, forward pass recomputes")
    print(f"   New logits differ: {not np.allclose(logits, logits3)}")

    # Example 6: Graph inspection
    print("\n6. Graph inspection...")
    from talos import list_computed_methods, list_stored_methods

    stored_methods = list_stored_methods(model)
    computed_methods = list_computed_methods(model)
    print(f"   Stored parameters: {stored_methods}")
    print(f"   Computed values: {computed_methods}")

    print("\n✓ Neural network demo complete!")
