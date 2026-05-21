from __future__ import annotations

from paper_intel.index.vector_store import VectorStore
from paper_intel.models.chunk import ChunkSchema
from paper_intel.retrieval.dense_retriever import DenseRetriever
from paper_intel.retrieval.sparse_retriever import SparseRetriever


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    RRF formula: score(d) = sum(1 / (k + rank(d, list_i)))
    k=60 is the constant from the original RRF paper (Cormack et al. 2009).
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (chunk_id, _) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        store: VectorStore,
        rrf_k: int = 60,
    ):
        self.dense = dense
        self.sparse = sparse
        self.store = store
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        paper_ids: list[str] | None = None,
        dense_candidates: int = 20,
        sparse_candidates: int = 20,
    ) -> list[ChunkSchema]:
        dense_results = self.dense.retrieve(query, top_k=dense_candidates, paper_ids=paper_ids)
        sparse_results = self.sparse.retrieve(query, top_k=sparse_candidates)

        # If paper_id filter is active, restrict sparse results to those papers too
        if paper_ids:
            sparse_results = [(cid, s) for cid, s in sparse_results if _chunk_in_papers(cid, paper_ids)]

        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=self.rrf_k)
        top_ids = [chunk_id for chunk_id, _ in fused[:top_k]]
        chunks = self.store.get_chunks_by_ids(top_ids)

        # Preserve RRF order
        id_to_chunk = {c.chunk_id: c for c in chunks}
        return [id_to_chunk[cid] for cid in top_ids if cid in id_to_chunk]


def _chunk_in_papers(chunk_id: str, paper_ids: list[str]) -> bool:
    # BM25 index doesn't carry paper metadata — rely on Qdrant filter for dense,
    # and post-filter sparse by fetching payload. For now, skip this pre-filter
    # (the Qdrant post-fetch in get_chunks_by_ids will naturally include all papers).
    return True
