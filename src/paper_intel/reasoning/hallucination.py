from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Literal

from paper_intel.models.chunk import ChunkSchema
from paper_intel.prompts.system_prompts import HALLUCINATION_SYSTEM_PROMPT
from paper_intel.prompts.templates import fact_decompose_user_prompt, fact_verify_user_prompt


@dataclass
class AtomicFact:
    fact_text: str
    supported: bool
    supporting_chunk_id: str | None
    confidence: float
    rationale: str = ""


@dataclass
class HallucinationReport:
    answer: str
    atomic_facts: list[AtomicFact]
    support_ratio: float
    hallucinated_facts: list[AtomicFact]
    verdict: Literal["PASS", "WARN", "FAIL"]

    PASS_THRESHOLD: float = 0.90
    WARN_THRESHOLD: float = 0.70


class HallucinationEvaluator:
    def __init__(
        self,
        client,
        model: str,
        pass_threshold: float = 0.90,
        warn_threshold: float = 0.70,
    ):
        self.client = client
        self.model = model
        self.pass_threshold = pass_threshold
        self.warn_threshold = warn_threshold

    def decompose_to_facts(self, answer: str) -> list[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "Extract atomic facts from an answer. Output one fact per line, nothing else.",
                },
                {"role": "user", "content": fact_decompose_user_prompt(answer)},
            ],
        )
        text = _extract_text(response)
        facts = [line.strip() for line in text.split("\n") if line.strip()]
        return facts

    def verify_fact(self, fact: str, chunks: list[ChunkSchema]) -> AtomicFact:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {"role": "system", "content": HALLUCINATION_SYSTEM_PROMPT},
                {"role": "user", "content": fact_verify_user_prompt(fact, chunks)},
            ],
        )
        text = _extract_text(response)
        data = _parse_json(text)

        return AtomicFact(
            fact_text=fact,
            supported=bool(data.get("supported", False)),
            supporting_chunk_id=data.get("supporting_chunk_id"),
            confidence=float(data.get("confidence", 0.5)),
            rationale=data.get("rationale", ""),
        )

    def evaluate(
        self,
        answer: str,
        retrieved_chunks: list[ChunkSchema],
    ) -> HallucinationReport:
        facts = self.decompose_to_facts(answer)
        if not facts:
            return HallucinationReport(
                answer=answer,
                atomic_facts=[],
                support_ratio=1.0,
                hallucinated_facts=[],
                verdict="PASS",
            )

        atomic_facts: list[AtomicFact] = []
        for fact in facts:
            af = self.verify_fact(fact, retrieved_chunks)
            atomic_facts.append(af)

        supported_count = sum(1 for af in atomic_facts if af.supported)
        support_ratio = supported_count / len(atomic_facts)
        hallucinated = [af for af in atomic_facts if not af.supported]

        if support_ratio >= self.pass_threshold:
            verdict: Literal["PASS", "WARN", "FAIL"] = "PASS"
        elif support_ratio >= self.warn_threshold:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        return HallucinationReport(
            answer=answer,
            atomic_facts=atomic_facts,
            support_ratio=support_ratio,
            hallucinated_facts=hallucinated,
            verdict=verdict,
        )


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
