"""Distributed training example with Talos.

Demonstrates:
- Using DistributedExecutor to parallelize training
- Computing critical path and parallelization opportunities
- Coordinating multi-stage training pipelines
- Estimating speedup from parallelization
"""

from typing import Any

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required for this example.")
    print("Install it with: pip install numpy")
    exit(1)

from talos import fn, stored
from talos.utils.distributed import DistributedExecutor
from talos.utils.profiler import Profiler


class DistributedDataProcessor:
    """Data processing pipeline that can be parallelized across workers.

    Shows how Talos can automatically identify parallelization opportunities
    in a data processing pipeline.

    Example usage:
        processor = DistributedDataProcessor(num_samples=1000)

        # Analyze execution
        executor = DistributedExecutor(processor, workers=4)
        summary = executor.get_execution_summary()

        # View parallelization plan
        stages = executor.schedule_parallel()
        print(f"Execution stages: {len(stages)}")
        for stage, nodes in stages.items():
            print(f"  Stage {stage}: {nodes}")

        # View critical path
        critical_path = executor.get_critical_path()
        print(f"Critical path: {critical_path}")

        # Estimate speedup
        speedup = executor.estimate_speedup()
        print(f"Estimated speedup with 4 workers: {speedup:.1f}x")
    """

    def __init__(self, num_samples: int = 1000, seed: int = 42) -> None:
        """Initialize data processor.

        Args:
            num_samples: Number of samples to process
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)

    @stored
    def config(self) -> dict[str, Any]:
        """Configuration (stored for persistence)."""
        return {
            "num_samples": self.num_samples,
            "seed": self.seed,
        }

    @fn
    def load_raw_data(self) -> np.ndarray:
        """Load raw data from source.

        In real scenario, this would be disk I/O.
        """
        # Simulate loading 1000 samples with 50 features
        return np.random.randn(self.num_samples, 50)

    @fn
    def load_labels(self) -> np.ndarray:
        """Load target labels."""
        return np.random.randint(0, 10, self.num_samples)

    @fn
    def preprocess_features(self) -> np.ndarray:
        """Preprocess raw data (normalize, scale, etc).

        Depends on: load_raw_data
        """
        raw = self.load_raw_data()
        # Standardization
        mean = np.mean(raw, axis=0)
        std = np.std(raw, axis=0) + 1e-8
        return (raw - mean) / std

    @fn
    def extract_features(self) -> np.ndarray:
        """Extract advanced features from preprocessed data.

        Depends on: preprocess_features
        This could be feature engineering, dimensionality reduction, etc.
        """
        processed = self.preprocess_features()
        # Simple feature: squared features
        squared = processed ** 2
        # Feature engineering: concatenate original and squared
        return np.hstack([processed, squared])

    @fn
    def compute_statistics(self) -> dict[str, float]:
        """Compute data statistics (can run in parallel with other processing).

        Depends on: preprocess_features
        """
        processed = self.preprocess_features()
        return {
            "mean": float(np.mean(processed)),
            "std": float(np.std(processed)),
            "min": float(np.min(processed)),
            "max": float(np.max(processed)),
        }

    @fn
    def balance_dataset(self) -> tuple[np.ndarray, np.ndarray]:
        """Balance classes in dataset.

        Depends on: extract_features, load_labels
        """
        features = self.extract_features()
        labels = self.load_labels()

        # Simple balancing: downsample majority classes
        return features, labels

    @fn
    def create_train_split(self) -> tuple[np.ndarray, np.ndarray]:
        """Create training split.

        Depends on: balance_dataset
        """
        features, labels = self.balance_dataset()
        split = int(0.8 * len(features))
        return features[:split], labels[:split]

    @fn
    def create_validation_split(self) -> tuple[np.ndarray, np.ndarray]:
        """Create validation split.

        Depends on: balance_dataset
        """
        features, labels = self.balance_dataset()
        split = int(0.8 * len(features))
        return features[split:], labels[split:]

    @fn
    def create_test_split(self) -> tuple[np.ndarray, np.ndarray]:
        """Create test split (can run in parallel with train/val).

        Depends on: extract_features, load_labels
        """
        features = self.extract_features()
        labels = self.load_labels()
        # Use last 10% for testing
        test_idx = int(0.9 * len(features))
        return features[test_idx:], labels[test_idx:]

    @fn
    def compute_feature_importance(self) -> dict[str, float]:
        """Compute feature importance scores.

        Depends on: create_train_split (can run in parallel)
        """
        X_train, y_train = self.create_train_split()

        # Simple importance: variance of features * correlation with target
        importance = {}
        for i in range(X_train.shape[1]):
            var = np.var(X_train[:, i])
            corr = np.abs(np.corrcoef(X_train[:, i], y_train)[0, 1])
            importance[f"feature_{i}"] = float(var * corr)

        return importance

    @fn
    def get_execution_summary(self) -> dict[str, Any]:
        """Summarize the data processing pipeline."""
        return {
            "raw_data_shape": self.load_raw_data().shape,
            "processed_data_shape": self.preprocess_features().shape,
            "extracted_data_shape": self.extract_features().shape,
            "statistics": self.compute_statistics(),
            "num_train_samples": len(self.create_train_split()[0]),
            "num_val_samples": len(self.create_validation_split()[0]),
            "num_test_samples": len(self.create_test_split()[0]),
        }


class DistributedModel:
    """Model that can be trained on distributed data."""

    def __init__(self, input_dim: int = 100, output_dim: int = 10) -> None:
        """Initialize model."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        np.random.seed(42)
        self._weights = np.random.randn(input_dim, output_dim) * 0.01

    @stored
    def weights(self) -> np.ndarray:
        """Model weights (stored for persistence)."""
        return self._weights.copy()

    @fn
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input batch

        Returns:
            Predictions
        """
        weights = self.weights()
        return X @ weights

    @fn
    def compute_loss(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> float:
        """Compute loss.

        Args:
            predictions: Model predictions
            targets: Target values

        Returns:
            Scalar loss
        """
        diff = predictions - targets.reshape(-1, 1)
        return float(np.mean(diff ** 2))

    def distributed_train_step(
        self, X_batch: np.ndarray, y_batch: np.ndarray
    ) -> float:
        """Perform one training step (can be parallelized across workers)."""
        predictions = self.predict(X_batch)
        loss = self.compute_loss(predictions, y_batch)
        return loss


if __name__ == "__main__":
    # Demo usage
    print("Distributed Training Example")
    print("=" * 60)

    processor = DistributedDataProcessor(num_samples=500)

    # Example 1: Analyze execution graph
    print("\n1. Analyzing execution graph...")
    executor = DistributedExecutor(processor, workers=4)

    summary = executor.get_execution_summary()
    print(f"   Total nodes: {summary['total_nodes']}")
    print(f"   Parallelizable nodes: {summary['parallelizable_nodes']}")
    print(f"   Critical path length: {summary['critical_path_length']}")
    print(f"   Execution stages: {summary['execution_stages']}")

    # Example 2: View execution stages
    print("\n2. Execution stages (what can run in parallel):")
    stages = executor.schedule_parallel()
    for stage, nodes in stages.items():
        print(f"   Stage {stage}: {nodes}")

    # Example 3: Critical path analysis
    print("\n3. Critical path (longest dependency chain):")
    critical = executor.get_critical_path()
    print(f"   Path: {critical}")

    # Example 4: Parallelizable nodes
    print("\n4. Nodes with no dependencies (can start immediately):")
    parallelizable = executor.get_parallelizable_nodes()
    print(f"   {parallelizable}")

    # Example 5: Estimated speedup
    print("\n5. Speedup estimation:")
    speedup_1 = DistributedExecutor(processor, workers=1).estimate_speedup()
    speedup_2 = DistributedExecutor(processor, workers=2).estimate_speedup()
    speedup_4 = DistributedExecutor(processor, workers=4).estimate_speedup()
    speedup_8 = DistributedExecutor(processor, workers=8).estimate_speedup()

    print(f"   1 worker:  {speedup_1:.2f}x")
    print(f"   2 workers: {speedup_2:.2f}x")
    print(f"   4 workers: {speedup_4:.2f}x")
    print(f"   8 workers: {speedup_8:.2f}x")

    # Example 6: Execute and profile
    print("\n6. Executing pipeline and profiling...")
    summary_result = processor.get_execution_summary()

    profiler = Profiler(processor)
    print(f"   Execution summary: {summary_result}")

    # Example 7: Train/validation/test split availability
    print("\n7. Data splits created:")
    X_train, y_train = processor.create_train_split()
    X_val, y_val = processor.create_validation_split()
    X_test, y_test = processor.create_test_split()

    print(f"   Train: {X_train.shape}")
    print(f"   Validation: {X_val.shape}")
    print(f"   Test: {X_test.shape}")

    # Example 8: Model training with distributed data
    print("\n8. Training model with distributed data...")
    model = DistributedModel(input_dim=100, output_dim=10)

    # Simulate training
    for epoch in range(3):
        loss = model.distributed_train_step(X_train, y_train)
        print(f"   Epoch {epoch + 1}: loss = {loss:.4f}")

    print("\n✓ Distributed training demo complete!")
