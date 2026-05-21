from __future__ import annotations
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from paper_intel.models.chunk import ChunkSchema


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[ChunkSchema]) -> None:
        corpus = [_tokenize(c.text) for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def add_chunks(self, chunks: list[ChunkSchema]) -> None:
        """Rebuild index with additional chunks appended."""
        if self.bm25 is None:
            self.build(chunks)
            return
        # BM25Okapi doesn't support incremental updates — rebuild
        existing_texts = list(self.bm25.corpus)
        new_texts = [_tokenize(c.text) for c in chunks]
        self.chunk_ids.extend(c.chunk_id for c in chunks)
        self.bm25 = BM25Okapi(existing_texts + new_texts)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in indexed[:top_k]:
            if score > 0:
                results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]

    def __len__(self) -> int:
        return len(self.chunk_ids)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()
