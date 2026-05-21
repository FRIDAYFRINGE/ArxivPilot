from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from paper_intel.models.chunk import ChunkSchema
from paper_intel.prompts.templates import format_chunks_for_agent

if TYPE_CHECKING:
    from paper_intel.graph.citation_graph import CitationGraph
    from paper_intel.reasoning.contradiction import ContradictionChecker
    from paper_intel.retrieval.hybrid import HybridRetriever


@dataclass
class FinalAnswer:
    answer: str
    cited_chunk_ids: list[str]
    contradiction_flags: list[str] = field(default_factory=list)


class ToolExecutor:
    def __init__(
        self,
        retriever: "HybridRetriever",
        graph: "CitationGraph",
        contradiction_checker: "ContradictionChecker",
    ):
        self.retriever = retriever
        self.graph = graph
        self.contradiction_checker = contradiction_checker

    def execute(
        self,
        tool_name: str,
        tool_input: dict,
        chunk_cache: dict[str, ChunkSchema],
    ) -> tuple[str | FinalAnswer, dict[str, ChunkSchema]]:
        """
        Returns (result, updated_cache).
        result is either a string (tool output for Claude) or FinalAnswer (terminates loop).
        """
        if tool_name == "hybrid_search":
            return self._hybrid_search(tool_input, chunk_cache)
        elif tool_name == "expand_citations":
            return self._expand_citations(tool_input, chunk_cache)
        elif tool_name == "check_contradiction":
            result = self._check_contradiction(tool_input)
            return result, chunk_cache
        elif tool_name == "finalize_answer":
            fa = FinalAnswer(
                answer=tool_input["answer"],
                cited_chunk_ids=tool_input.get("cited_chunk_ids", []),
                contradiction_flags=tool_input.get("contradiction_flags", []),
            )
            return fa, chunk_cache
        else:
            return f"Unknown tool: {tool_name}", chunk_cache

    def _hybrid_search(
        self, inputs: dict, cache: dict[str, ChunkSchema]
    ) -> tuple[str, dict[str, ChunkSchema]]:
        query = inputs["query"]
        top_k = inputs.get("top_k", 8)
        paper_ids = inputs.get("paper_ids") or None

        chunks = self.retriever.retrieve(query, top_k=top_k, paper_ids=paper_ids)
        if not chunks:
            return "No relevant chunks found for this query.", cache

        for c in chunks:
            cache[c.chunk_id] = c

        return format_chunks_for_agent(chunks), cache

    def _expand_citations(
        self, inputs: dict, cache: dict[str, ChunkSchema]
    ) -> tuple[str, dict[str, ChunkSchema]]:
        paper_id = inputs["paper_id"]
        query = inputs["query"]
        direction = inputs.get("direction", "both")
        depth = inputs.get("depth", 1)

        indexed_ids = set(self.retriever.store.indexed_paper_ids())
        neighbor_ids = self.graph.indexed_neighbors(paper_id, indexed_ids, direction, depth)

        if not neighbor_ids:
            return (
                f"No indexed neighbors found for paper {paper_id} "
                f"(direction={direction}, depth={depth}).",
                cache,
            )

        chunks = self.retriever.retrieve(query, top_k=8, paper_ids=neighbor_ids)
        if not chunks:
            return f"No relevant chunks found in {len(neighbor_ids)} neighbor papers.", cache

        for c in chunks:
            cache[c.chunk_id] = c

        header = f"Retrieved from {len(neighbor_ids)} papers neighboring {paper_id}:\n\n"
        return header + format_chunks_for_agent(chunks), cache

    def _check_contradiction(self, inputs: dict) -> str:
        result = self.contradiction_checker.check(
            claim_a=inputs["claim_a"],
            source_a=inputs["source_a"],
            claim_b=inputs["claim_b"],
            source_b=inputs["source_b"],
        )
        return (
            f"CONTRADICTION CHECK RESULT\n"
            f"Verdict: {result.verdict}\n"
            f"Confidence: {result.confidence:.2f}\n"
            f"Explanation: {result.explanation}\n"
            f"Claim A ({result.source_a}): {result.claim_a}\n"
            f"Claim B ({result.source_b}): {result.claim_b}"
        )
