from __future__ import annotations
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        # BGE-small requires this prefix for queries (not for documents)
        prefixed = f"Represent this sentence for searching: {query}"
        vec = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vec[0], dtype=np.float32)
