from __future__ import annotations

from paper_intel.index.bm25_index import BM25Index


class SparseRetriever:
    def __init__(self, bm25_index: BM25Index):
        self.index = bm25_index

    def retrieve(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        return self.index.search(query, top_k=top_k)
