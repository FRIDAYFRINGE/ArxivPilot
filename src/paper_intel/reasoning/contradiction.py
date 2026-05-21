from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Literal

from paper_intel.models.chunk import ChunkSchema
from paper_intel.prompts.system_prompts import CONTRADICTION_SYSTEM_PROMPT
from paper_intel.prompts.templates import contradiction_user_prompt


@dataclass
class ContradictionResult:
    verdict: Literal["AGREE", "DISAGREE", "UNCERTAIN"]
    explanation: str
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    confidence: float


class ContradictionChecker:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def check(
        self,
        claim_a: str,
        source_a: str,
        claim_b: str,
        source_b: str,
    ) -> ContradictionResult:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": CONTRADICTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": contradiction_user_prompt(claim_a, source_a, claim_b, source_b),
                },
            ],
        )
        text = _extract_text(response)
        data = _parse_json(text)

        return ContradictionResult(
            verdict=data.get("verdict", "UNCERTAIN"),
            explanation=data.get("explanation", "Unable to parse response."),
            claim_a=claim_a,
            source_a=source_a,
            claim_b=claim_b,
            source_b=source_b,
            confidence=float(data.get("confidence", 0.5)),
        )

    def scan_chunks_for_contradictions(
        self,
        chunks: list[ChunkSchema],
        topic: str,
    ) -> list[ContradictionResult]:
        """
        Scan chunks from different papers for contradictions on a topic.
        Uses embedding similarity as a pre-filter to avoid O(N^2) LLM calls.
        """
        import numpy as np

        by_paper: dict[str, list[ChunkSchema]] = {}
        for c in chunks:
            by_paper.setdefault(c.paper_id, []).append(c)

        paper_ids = list(by_paper.keys())
        if len(paper_ids) < 2:
            return []

        results: list[ContradictionResult] = []

        for i in range(len(paper_ids)):
            for j in range(i + 1, len(paper_ids)):
                pid_a, pid_b = paper_ids[i], paper_ids[j]
                chunks_a = by_paper[pid_a]
                chunks_b = by_paper[pid_b]

                best_pair = _most_similar_pair(chunks_a, chunks_b)
                if best_pair is None:
                    continue

                chunk_a, chunk_b, sim = best_pair
                if sim < 0.70:
                    continue

                result = self.check(
                    claim_a=chunk_a.text[:400],
                    source_a=pid_a,
                    claim_b=chunk_b.text[:400],
                    source_b=pid_b,
                )
                if result.verdict == "DISAGREE":
                    results.append(result)

        return results


def _most_similar_pair(
    chunks_a: list[ChunkSchema],
    chunks_b: list[ChunkSchema],
) -> tuple[ChunkSchema, ChunkSchema, float] | None:
    import numpy as np

    a_with_emb = [c for c in chunks_a if c.embedding]
    b_with_emb = [c for c in chunks_b if c.embedding]
    if not a_with_emb or not b_with_emb:
        return None

    vecs_a = np.array([c.embedding for c in a_with_emb])
    vecs_b = np.array([c.embedding for c in b_with_emb])
    sims = vecs_a @ vecs_b.T

    idx = np.unravel_index(np.argmax(sims), sims.shape)
    return a_with_emb[idx[0]], b_with_emb[idx[1]], float(sims[idx])


def _extract_text(response) -> str:
    return (response.choices[0].message.content or "").strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}
