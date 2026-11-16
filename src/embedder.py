"""BGE embeddings without a RAG framework.

This module handles neural inference only. The model is downloaded into the
Hugging Face cache on its first use; subsequent runs can use the local cache.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
MODEL_MAX_TOKENS = 512
# BGE's official instruction is for short search queries, NOT corpus passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """Encode passages and search queries with the same BGE model.

    A model can be injected for unit testing without downloading weights.
    """

    def __init__(self, *, model=None, device: str = "cpu") -> None:
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is missing. Run: python -m pip install sentence-transformers"
                ) from exc
            model = SentenceTransformer(MODEL_NAME, device=None if device == "auto" else device)
        self.model = model
        self.model_name = MODEL_NAME
        self.dimension = int(model.get_sentence_embedding_dimension())
        if self.dimension != EMBEDDING_DIM:
            raise ValueError(f"Unexpected embedding dimension {self.dimension}; expected {EMBEDDING_DIM}")
        self.max_tokens = min(int(model.max_seq_length), MODEL_MAX_TOKENS)
        if self.max_tokens < 2:
            raise ValueError("Invalid model token limit")
        self.tokenizer = model.tokenizer
        if self.tokenizer is None:
            raise ValueError("Model does not expose a tokenizer for token checks")

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence) or not texts:
            raise ValueError("Expected a nonempty sequence of texts")
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("All texts must be nonempty strings")
        return list(texts)

    def token_lengths(self, texts: Sequence[str]) -> list[int]:
        """Count real model tokens, including special tokens, without truncation."""
        items = self._validate_texts(texts)
        result = self.tokenizer(
            items, add_special_tokens=True, padding=False, truncation=False,
            return_attention_mask=False,
        )
        sequences = result["input_ids"]
        if len(sequences) != len(items):
            raise ValueError("Tokenizer returned a different number of sequences")
        return [len(ids) for ids in sequences]

    def encode_passages(self, texts: Sequence[str], *, batch_size: int = 16) -> np.ndarray:
        """One normalized float32 vector per passage, with no silent truncation."""
        items = self._validate_texts(texts)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        counts = self.token_lengths(items)
        for i, count in enumerate(counts):
            if count > self.max_tokens:
                raise ValueError(
                    f"Text {i + 1} has {count} tokens, exceeding model limit {self.max_tokens}. "
                    "Do not silently truncate medical text. Re-chunk with a smaller --chunk-size "
                    "(for example 800) and regenerate embeddings."
                )
        vectors = self.model.encode(
            items, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(items), self.dimension):
            raise ValueError(f"Invalid embedding shape: {vectors.shape}")
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding contains NaN or infinity")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=0.02, rtol=0):
            raise ValueError("Embedding vectors must be L2-normalized")
        return vectors

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a short search question with the recommended BGE query prefix."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be nonempty text")
        return self.encode_passages([QUERY_INSTRUCTION + query.strip()])[0]
