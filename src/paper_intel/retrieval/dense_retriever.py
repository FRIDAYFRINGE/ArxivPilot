from __future__ import annotations

from paper_intel.index.embedder import Embedder
from paper_intel.index.vector_store import VectorStore


class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        paper_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        vec = self.embedder.embed_query(query)
        return self.store.search(vec, top_k=top_k, filter_paper_ids=paper_ids)
