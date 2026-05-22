from __future__ import annotations
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from paper_intel.models.chunk import ChunkSchema


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []
        self._corpus: list[list[str]] = []  # keep raw tokens so we can rebuild incrementally

    def build(self, chunks: list[ChunkSchema]) -> None:
        self._corpus = [_tokenize(c.text) for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.bm25 = BM25Okapi(self._corpus)

    def add_chunks(self, chunks: list[ChunkSchema]) -> None:
        """Rebuild index with additional chunks appended."""
        new_tokens = [_tokenize(c.text) for c in chunks]
        self._corpus.extend(new_tokens)
        self.chunk_ids.extend(c.chunk_id for c in chunks)
        self.bm25 = BM25Okapi(self._corpus)

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
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "corpus": self._corpus,
            }, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunk_ids = data["chunk_ids"]
        self._corpus = data.get("corpus", [])
        if self._corpus:
            self.bm25 = BM25Okapi(self._corpus)

    def __len__(self) -> int:
        return len(self.chunk_ids)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()
