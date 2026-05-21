import json
import pytest
from unittest.mock import MagicMock, patch
from paper_intel.reasoning.contradiction import ContradictionChecker
from paper_intel.reasoning.hallucination import HallucinationEvaluator
from paper_intel.graph.citation_graph import CitationGraph
from tests.conftest import make_chunk, make_paper


def _mock_client(response_text: str) -> MagicMock:
    """Creates a mock Groq client that returns the given text."""
    message = MagicMock()
    message.content = response_text
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


class TestContradictionChecker:
    def test_disagree_verdict(self):
        resp = json.dumps({
            "verdict": "DISAGREE",
            "explanation": "The claims are incompatible.",
            "confidence": 0.9,
        })
        client = _mock_client(resp)
        checker = ContradictionChecker(client, "groq-test")
        result = checker.check("A uses 4GB", "paper_a", "A uses 8GB", "paper_b")
        assert result.verdict == "DISAGREE"
        assert result.confidence == 0.9

    def test_agree_verdict(self):
        resp = json.dumps({
            "verdict": "AGREE",
            "explanation": "Both say the same thing.",
            "confidence": 0.95,
        })
        client = _mock_client(resp)
        checker = ContradictionChecker(client, "groq-test")
        result = checker.check("X achieves 90%", "p1", "X reaches 90% accuracy", "p2")
        assert result.verdict == "AGREE"

    def test_invalid_json_returns_uncertain(self):
        client = _mock_client("not json at all")
        checker = ContradictionChecker(client, "groq-test")
        result = checker.check("A", "p1", "B", "p2")
        assert result.verdict == "UNCERTAIN"


class TestHallucinationEvaluator:
    def test_pass_verdict_all_supported(self):
        # decompose returns 2 facts; verify returns both as supported
        decompose_resp = "Fact one about attention.\nFact two about transformers."
        verify_resp = json.dumps({
            "supported": True,
            "supporting_chunk_id": "abc",
            "confidence": 0.9,
            "rationale": "Directly stated.",
        })

        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            msg = MagicMock()
            msg.content = decompose_resp if call_count[0] == 1 else verify_resp
            msg.tool_calls = None
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        client.chat.completions.create.side_effect = side_effect

        evaluator = HallucinationEvaluator(client, "groq-test")
        report = evaluator.evaluate("Some answer text.", [make_chunk()])
        assert report.verdict == "PASS"
        assert report.support_ratio == 1.0

    def test_fail_verdict_unsupported_facts(self):
        decompose_resp = "Fact one.\nFact two.\nFact three."
        verify_resp = json.dumps({
            "supported": False,
            "supporting_chunk_id": None,
            "confidence": 0.2,
            "rationale": "Not found.",
        })

        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            msg = MagicMock()
            msg.content = decompose_resp if call_count[0] == 1 else verify_resp
            msg.tool_calls = None
            choice = MagicMock()
            choice.message = msg
            resp = MagicMock()
            resp.choices = [choice]
            return resp

        client.chat.completions.create.side_effect = side_effect

        evaluator = HallucinationEvaluator(client, "groq-test", pass_threshold=0.9, warn_threshold=0.7)
        report = evaluator.evaluate("An answer with unsupported claims.", [make_chunk()])
        assert report.verdict == "FAIL"
        assert report.support_ratio == 0.0
        assert len(report.hallucinated_facts) == 3


class TestCitationGraph:
    def test_add_paper_creates_node(self):
        graph = CitationGraph()
        meta = make_paper("1706.03762", references=["1409.0473"])
        graph.add_paper(meta)
        assert graph.has_paper("1706.03762")

    def test_edges_created_for_references(self):
        graph = CitationGraph()
        meta = make_paper("1706.03762", references=["1409.0473", "1508.04025"])
        graph.add_paper(meta)
        cites = graph.get_neighbors("1706.03762", direction="cites", depth=1)
        assert "1409.0473" in cites
        assert "1508.04025" in cites

    def test_cited_by_direction(self):
        graph = CitationGraph()
        meta = make_paper("1706.03762", references=["1409.0473"])
        graph.add_paper(meta)
        cited_by = graph.get_neighbors("1409.0473", direction="cited_by", depth=1)
        assert "1706.03762" in cited_by

    def test_bfs_depth_2(self):
        graph = CitationGraph()
        graph.add_paper(make_paper("A", references=["B"]))
        graph.add_paper(make_paper("B", references=["C"]))
        neighbors = graph.get_neighbors("A", direction="cites", depth=2)
        assert "B" in neighbors
        assert "C" in neighbors

    def test_save_load(self, tmp_path):
        graph = CitationGraph()
        graph.add_paper(make_paper("X", references=["Y"]))
        path = tmp_path / "graph.pkl"
        graph.save(path)

        graph2 = CitationGraph()
        graph2.load(path)
        assert graph2.has_paper("X")
        assert "Y" in graph2.get_neighbors("X", direction="cites")
