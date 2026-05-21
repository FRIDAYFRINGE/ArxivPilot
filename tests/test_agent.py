import pytest
from unittest.mock import MagicMock, patch
from paper_intel.agent.tool_executor import FinalAnswer, ToolExecutor
from paper_intel.graph.citation_graph import CitationGraph
from tests.conftest import make_chunk


def _make_executor(chunks=None):
    if chunks is None:
        chunks = [make_chunk()]

    retriever = MagicMock()
    retriever.retrieve.return_value = chunks
    retriever.store.indexed_paper_ids.return_value = [c.paper_id for c in chunks]

    graph = CitationGraph()
    contradiction_checker = MagicMock()
    contradiction_checker.check.return_value = MagicMock(
        verdict="AGREE", explanation="Same claim.", source_a="a", source_b="b",
        claim_a="x", claim_b="x", confidence=0.9
    )
    return ToolExecutor(retriever, graph, contradiction_checker)


class TestToolExecutor:
    def test_hybrid_search_returns_formatted_text(self):
        chunks = [make_chunk("attention mechanism self-attention")]
        executor = _make_executor(chunks)
        result, cache = executor.execute("hybrid_search", {"query": "attention"}, {})
        assert isinstance(result, str)
        assert "[CHUNK" in result
        assert chunks[0].chunk_id in cache

    def test_hybrid_search_empty_result(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        retriever.store.indexed_paper_ids.return_value = []
        graph = CitationGraph()
        checker = MagicMock()
        executor = ToolExecutor(retriever, graph, checker)
        result, cache = executor.execute("hybrid_search", {"query": "nothing"}, {})
        assert "No relevant" in result
        assert len(cache) == 0

    def test_finalize_answer_returns_final_answer(self):
        executor = _make_executor()
        result, _ = executor.execute(
            "finalize_answer",
            {"answer": "The answer is X.", "cited_chunk_ids": [], "contradiction_flags": []},
            {},
        )
        assert isinstance(result, FinalAnswer)
        assert result.answer == "The answer is X."

    def test_check_contradiction_returns_string(self):
        executor = _make_executor()
        result, _ = executor.execute(
            "check_contradiction",
            {"claim_a": "A", "source_a": "p1", "claim_b": "B", "source_b": "p2"},
            {},
        )
        assert isinstance(result, str)
        assert "AGREE" in result or "DISAGREE" in result or "UNCERTAIN" in result

    def test_unknown_tool_returns_error(self):
        executor = _make_executor()
        result, _ = executor.execute("unknown_tool", {}, {})
        assert "Unknown tool" in result
