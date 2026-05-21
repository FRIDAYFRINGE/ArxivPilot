from __future__ import annotations
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from paper_intel.models.chunk import ChunkSchema

COLLECTION = "papers"


class VectorStore:
    def __init__(self, index_dir: Path, vector_size: int = 384, qdrant_url: str = ""):
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            index_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(index_dir / "qdrant"))
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_chunks(self, chunks: list[ChunkSchema]) -> None:
        if not chunks:
            return
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding")
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "paper_id": chunk.paper_id,
                    "text": chunk.text,
                    "section": chunk.section,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "paper_title": chunk.paper_title,
                    "authors": chunk.authors,
                    "year": chunk.year,
                    "arxiv_url": chunk.arxiv_url,
                    "cited_paper_ids": chunk.cited_paper_ids,
                },
            ))
        # Upsert in batches of 100
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=COLLECTION, points=points[i:i + 100])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filter_paper_ids: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        query_filter = None
        if filter_paper_ids:
            query_filter = Filter(
                must=[FieldCondition(key="paper_id", match=MatchAny(any=filter_paper_ids))]
            )
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=False,
        )
        return [(str(r.id), float(r.score)) for r in response.points]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[ChunkSchema]:
        if not chunk_ids:
            return []
        records = self.client.retrieve(
            collection_name=COLLECTION,
            ids=chunk_ids,
            with_payload=True,
            with_vectors=False,
        )
        chunks = []
        for rec in records:
            p = rec.payload
            chunks.append(ChunkSchema(
                chunk_id=str(rec.id),
                paper_id=p["paper_id"],
                text=p["text"],
                section=p["section"],
                page_number=p["page_number"],
                chunk_index=p["chunk_index"],
                token_count=p["token_count"],
                paper_title=p["paper_title"],
                authors=p["authors"],
                year=p["year"],
                arxiv_url=p["arxiv_url"],
                cited_paper_ids=p.get("cited_paper_ids", []),
            ))
        return chunks

    def count(self) -> int:
        info = self.client.get_collection(COLLECTION)
        return info.points_count or 0

    def indexed_paper_ids(self) -> list[str]:
        """Scroll through all payloads to get unique paper_ids."""
        ids: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=COLLECTION,
                scroll_filter=None,
                limit=100,
                offset=offset,
                with_payload=["paper_id"],
                with_vectors=False,
            )
            for r in records:
                ids.add(r.payload["paper_id"])
            if offset is None:
                break
        return list(ids)
