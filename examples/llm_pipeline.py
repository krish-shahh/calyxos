"""LLM pipeline example showing real-world Talos usage.

Demonstrates:
- Async I/O for API calls with built-in memoization
- Performance profiling and optimization hints
- Distributed execution planning
- Caching strategies for expensive operations
"""

import asyncio
import json
from typing import Any

from talos import fn, stored


class LLMPipeline:
    """End-to-end LLM inference pipeline with caching and profiling.

    Example usage:
        pipeline = LLMPipeline(
            model_name="gpt-4",
            api_key="sk-...",
            cache_embeddings=True
        )

        # Get embedding for text (cached after first call)
        embedding = pipeline.get_embedding("What is machine learning?")

        # Get answer from LLM (cached by input hash)
        answer = pipeline.get_answer(
            "What is machine learning?",
            context="Focus on practical applications"
        )

        # View optimization hints
        from talos.utils.profiler import Profiler
        profiler = Profiler(pipeline)
        profiler.print_profile_report()
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        api_endpoint: str = "https://api.openai.com/v1",
        embedding_model: str = "text-embedding-3-small",
        cache_embeddings: bool = True,
    ) -> None:
        """Initialize LLM pipeline.

        Args:
            model_name: LLM model identifier
            api_endpoint: API endpoint URL
            embedding_model: Embedding model identifier
            cache_embeddings: Whether to cache embeddings across calls
        """
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.embedding_model = embedding_model
        self.cache_embeddings = cache_embeddings

        # Mock API responses for demo (in real usage, would be actual API calls)
        self._mock_mode = True

    @stored
    def config(self) -> dict[str, Any]:
        """Pipeline configuration (stored for persistence)."""
        return {
            "model": self.model_name,
            "embedding_model": self.embedding_model,
            "api_endpoint": self.api_endpoint,
        }

    @fn
    def preprocess_text(self, text: str) -> str:
        """Preprocess input text (tokenization, normalization)."""
        # Remove extra whitespace and normalize
        processed = " ".join(text.split())
        return processed.lower()

    @fn
    def get_embedding(self, text: str) -> list[float]:
        """Get text embedding from embedding API (memoized).

        This would typically be an async API call in production.
        Cached based on input text to avoid duplicate API calls.
        """
        processed = self.preprocess_text(text)

        # Mock embedding for demo
        if self._mock_mode:
            import hashlib

            # Create deterministic "embedding" based on text
            text_hash = int(
                hashlib.md5(processed.encode()).hexdigest()[:8], 16
            )
            # Simulate 768-dimensional embedding
            embedding = [
                (text_hash + i) % 1000 / 1000.0 for i in range(768)
            ]
            return embedding

        # Real implementation would call embedding API
        # response = await self._call_embedding_api(text)
        # return response['embedding']
        return []

    @fn
    def get_answer(
        self, question: str, context: str = "", max_tokens: int = 500
    ) -> str:
        """Get answer from LLM (memoized by input).

        Cached based on question + context + parameters.
        """
        preprocessed = self.preprocess_text(question)

        # Mock LLM response for demo
        if self._mock_mode:
            answers = {
                "what is machine learning": (
                    "Machine learning is a subset of artificial intelligence "
                    "that enables systems to learn from data without explicit "
                    "programming. It uses algorithms to identify patterns and "
                    "make predictions or decisions based on input data."
                ),
                "what is deep learning": (
                    "Deep learning uses artificial neural networks with multiple "
                    "layers to process data. It's particularly effective for "
                    "complex tasks like image recognition, natural language "
                    "processing, and game playing."
                ),
                "what is a transformer": (
                    "A Transformer is a deep learning architecture based on "
                    "self-attention mechanisms. It processes entire sequences "
                    "in parallel, making it efficient and effective for NLP "
                    "and other sequence processing tasks."
                ),
            }
            base_answer = next(
                (v for k, v in answers.items() if k in preprocessed),
                "I'm not sure about that topic. Could you rephrase?",
            )

            if context:
                base_answer += f" {context}"

            return base_answer[:max_tokens]

        # Real implementation would call LLM API
        # response = await self._call_llm_api(question, context)
        # return response['text']
        return ""

    @fn
    def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract named entities from text using LLM."""
        # Use LLM to identify entities
        answer = self.get_answer(
            f"Extract named entities from: {text}", max_tokens=200
        )

        # Mock entity extraction for demo
        if self._mock_mode:
            return [
                {"text": "machine learning", "type": "concept"},
                {"text": "AI", "type": "concept"},
            ]

        # Real implementation would parse LLM output
        return []

    @fn
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using embeddings.

        Uses cached embeddings for efficiency.
        """
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)

        if not emb1 or not emb2:
            return 0.0

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @fn
    def rank_answers(self, question: str, candidates: list[str]) -> list[tuple[str, float]]:
        """Rank answer candidates by relevance to question."""
        ranked = [
            (candidate, self.compute_similarity(question, candidate))
            for candidate in candidates
        ]
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    @fn
    def generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the text."""
        # Use LLM to summarize
        summary = self.get_answer(
            f"Summarize this text: {text}",
            max_tokens=max_length,
        )
        return summary


# Example: Batch processing for LLM inference
class BatchedLLMPipeline(LLMPipeline):
    """LLM pipeline with batch processing support.

    Processes multiple queries together for efficiency.
    """

    def __init__(self, *args: Any, batch_size: int = 32, **kwargs: Any) -> None:
        """Initialize batched pipeline."""
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self._pending_queries: list[str] = []

    def add_query(self, question: str) -> None:
        """Add query to pending batch."""
        self._pending_queries.append(question)

    def should_process_batch(self) -> bool:
        """Check if batch is ready to process."""
        return len(self._pending_queries) >= self.batch_size

    def process_batch(self) -> list[str]:
        """Process all pending queries as batch."""
        if not self._pending_queries:
            return []

        # In a real implementation, would use batch API call
        # For demo, process individually (but in parallel ideally)
        results = [
            self.get_answer(q) for q in self._pending_queries
        ]

        self._pending_queries.clear()
        return results


if __name__ == "__main__":
    # Demo usage
    print("LLM Pipeline Example")
    print("=" * 60)

    pipeline = LLMPipeline(model_name="gpt-4-turbo")

    # Example 1: Get embeddings (cached after first call)
    print("\n1. Computing embeddings (will be cached)...")
    text1 = "What is machine learning?"
    emb1 = pipeline.get_embedding(text1)
    print(f"   Embedding for '{text1}': {emb1[:5]}... (768 dims)")

    # Second call uses cache
    emb1_again = pipeline.get_embedding(text1)
    print(f"   Cache hit: Got same embedding without recomputation")

    # Example 2: Get LLM answers (cached by input)
    print("\n2. Getting LLM answers (cached)...")
    answer = pipeline.get_answer("What is machine learning?")
    print(f"   Answer: {answer[:80]}...")

    # Example 3: Entity extraction
    print("\n3. Extracting entities...")
    entities = pipeline.extract_entities(
        "Apple and Google are tech companies"
    )
    print(f"   Entities: {entities}")

    # Example 4: Semantic similarity
    print("\n4. Computing semantic similarity...")
    sim = pipeline.compute_similarity(
        "machine learning", "deep learning"
    )
    print(f"   Similarity: {sim:.3f}")

    # Example 5: Ranking candidates
    print("\n5. Ranking answer candidates...")
    candidates = [
        "Deep learning is a subset of machine learning",
        "ML is short for Machine Learning",
        "The weather is nice today",
    ]
    ranked = pipeline.rank_answers(
        "What is machine learning?", candidates
    )
    for candidate, score in ranked:
        print(f"   {score:.3f}: {candidate}")

    # Example 6: Profiling
    print("\n6. Performance profile...")
    from talos.utils.profiler import Profiler

    profiler = Profiler(pipeline)
    print(f"   Profiling data: {profiler.get_all_profiles().keys()}")

    # Example 7: Graph introspection
    print("\n7. Graph inspection...")
    from talos import list_computed_methods, list_stored_methods

    stored = list_stored_methods(pipeline)
    computed = list_computed_methods(pipeline)
    print(f"   Stored methods: {stored}")
    print(f"   Computed methods: {computed}")

    print("\n✓ Pipeline demo complete!")
