from __future__ import annotations
import json
from dataclasses import dataclass, field

from paper_intel.prompts.system_prompts import DECOMPOSE_SYSTEM_PROMPT
from paper_intel.prompts.templates import decompose_user_prompt, aggregate_answers_prompt


@dataclass
class SubQuestion:
    index: int
    question: str
    depends_on: list[int] = field(default_factory=list)


@dataclass
class DecompositionResult:
    original_question: str
    sub_questions: list[SubQuestion]
    needs_decomposition: bool
    reasoning: str


class QueryDecomposer:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def decompose(self, question: str) -> DecompositionResult:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": decompose_user_prompt(question)},
            ],
        )
        text = _extract_text(response)
        data = _parse_json(text)

        sub_questions = [
            SubQuestion(
                index=sq["index"],
                question=sq["question"],
                depends_on=sq.get("depends_on", []),
            )
            for sq in data.get("sub_questions", [{"index": 0, "question": question, "depends_on": []}])
        ]
        return DecompositionResult(
            original_question=question,
            sub_questions=sub_questions,
            needs_decomposition=data.get("needs_decomposition", False),
            reasoning=data.get("reasoning", ""),
        )

    def aggregate_answers(
        self,
        original_question: str,
        sub_answers: list[tuple[str, str]],
    ) -> str:
        if len(sub_answers) == 1:
            return sub_answers[0][1]

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {
                    "role": "system",
                    "content": "You are a technical research synthesizer. Combine sub-answers into a coherent, complete response.",
                },
                {"role": "user", "content": aggregate_answers_prompt(original_question, sub_answers)},
            ],
        )
        return _extract_text(response)

    def execution_order(self, sub_questions: list[SubQuestion]) -> list[list[SubQuestion]]:
        """Returns sub-questions grouped into waves by dependency order."""
        waves: list[list[SubQuestion]] = []
        remaining = list(sub_questions)
        completed: set[int] = set()

        while remaining:
            wave = [sq for sq in remaining if all(d in completed for d in sq.depends_on)]
            if not wave:
                waves.append(remaining)
                break
            waves.append(wave)
            for sq in wave:
                completed.add(sq.index)
            remaining = [sq for sq in remaining if sq not in wave]

        return waves


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
