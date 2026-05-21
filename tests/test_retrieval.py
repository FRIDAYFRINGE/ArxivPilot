import pytest
from paper_intel.retrieval.hybrid import reciprocal_rank_fusion
from paper_intel.index.bm25_index import BM25Index
from tests.conftest import make_chunk


class TestRRF:
    def test_chunks_in_both_lists_rank_higher(self):
        list_a = [("c1", 1.0), ("c2", 0.9), ("c3", 0.8)]
        list_b = [("c2", 1.0), ("c3", 0.9), ("c4", 0.8)]
        fused = reciprocal_rank_fusion([list_a, list_b])
        top_ids = [f[0] for f in fused[:2]]
        assert "c2" in top_ids
        assert "c3" in top_ids

    def test_single_list_passthrough(self):
        ranked = [("a", 1.0), ("b", 0.5)]
        fused = reciprocal_rank_fusion([ranked])
        assert fused[0][0] == "a"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([[], []]) == []

    def test_scores_are_positive(self):
        fused = reciprocal_rank_fusion([[("x", 1.0)], [("x", 0.5)]])
        assert all(score > 0 for _, score in fused)

    def test_k60_constant(self):
        # With k=60, first rank contributes 1/61 ≈ 0.0164
        fused = reciprocal_rank_fusion([[("a", 1.0)]])
        assert abs(fused[0][1] - 1 / 61) < 1e-6


class TestBM25Index:
    def test_build_and_search(self):
        chunks = [
            make_chunk("attention mechanism transformer self-attention", chunk_index=0),
            make_chunk("convolutional neural network image classification", chunk_index=1),
            make_chunk("recurrent neural network language model", chunk_index=2),
        ]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("attention transformer", top_k=3)
        assert len(results) > 0
        top_id = results[0][0]
        assert top_id == chunks[0].chunk_id

    def test_search_returns_nonzero_scores(self):
        # BM25Okapi IDF = log((N-df+0.5)/(df+0.5)); need N≥3, df=1 for positive score
        chunks = [
            make_chunk("the quick brown fox jumps", chunk_index=0),
            make_chunk("the lazy dog sleeps all day", chunk_index=1),
            make_chunk("neural networks learn representations", chunk_index=2),
        ]
        idx = BM25Index()
        idx.build(chunks)
        results = idx.search("fox", top_k=5)
        assert len(results) > 0
        assert results[0][1] > 0

    def test_empty_index_returns_empty(self):
        idx = BM25Index()
        assert idx.search("anything") == []

    def test_save_load(self, tmp_path):
        chunks = [
            make_chunk("gradient descent optimization learning rate", chunk_index=0),
            make_chunk("attention mechanism self-attention transformer", chunk_index=1),
            make_chunk("convolutional neural network image recognition", chunk_index=2),
        ]
        idx = BM25Index()
        idx.build(chunks)
        path = tmp_path / "bm25.pkl"
        idx.save(path)

        idx2 = BM25Index()
        idx2.load(path)
        results = idx2.search("gradient", top_k=3)
        assert len(results) > 0
