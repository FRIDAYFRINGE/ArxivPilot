from paper_intel.models.chunk import ChunkSchema


def format_chunks_for_agent(chunks: list[ChunkSchema]) -> str:
    parts = []
    for chunk in chunks:
        # Cap text at ~800 chars (~200 tokens) to stay within tight TPM budgets
        text = chunk.text[:800] + ("..." if len(chunk.text) > 800 else "")
        parts.append(
            f"[CHUNK {chunk.chunk_id}]\n"
            f"Paper: {chunk.paper_title} ({chunk.year}) arxiv:{chunk.paper_id}\n"
            f"Section: {chunk.section}\n"
            f"{text}"
        )
    return "\n---\n".join(parts)


def contradiction_user_prompt(
    claim_a: str, source_a: str, claim_b: str, source_b: str
) -> str:
    return (
        f'Claim A (from paper {source_a}):\n"{claim_a}"\n\n'
        f'Claim B (from paper {source_b}):\n"{claim_b}"\n\n'
        'Output:\n'
        '{\n'
        '  "verdict": "AGREE" | "DISAGREE" | "UNCERTAIN",\n'
        '  "explanation": "One sentence.",\n'
        '  "confidence": <float 0.0-1.0>\n'
        '}'
    )


def decompose_user_prompt(question: str) -> str:
    return (
        f"Question: {question}\n\n"
        "Decide if this question needs decomposition. Output:\n"
        "{\n"
        '  "needs_decomposition": true | false,\n'
        '  "reasoning": "Why or why not.",\n'
        '  "sub_questions": [\n'
        '    {"index": 0, "question": "...", "depends_on": []},\n'
        '    {"index": 1, "question": "...", "depends_on": [0]}\n'
        "  ]\n"
        "}\n"
        'If needs_decomposition is false, sub_questions should contain exactly one item '
        "with the original question."
    )


def fact_decompose_user_prompt(answer: str) -> str:
    return (
        "Decompose this answer into atomic, self-contained factual claims.\n"
        "Rules:\n"
        "- Each claim must be a single, specific assertion.\n"
        "- Remove hedging language.\n"
        "- Do not include meta-claims about the answer.\n"
        "- Output one claim per line, no numbering, no bullets.\n\n"
        f"Answer:\n{answer}"
    )


def fact_verify_user_prompt(fact: str, chunks: list[ChunkSchema]) -> str:
    passages = format_chunks_for_agent(chunks)
    return (
        f"Fact to verify: {fact}\n\n"
        f"Passages:\n{passages}\n\n"
        "Output ONLY valid JSON:\n"
        "{\n"
        '  "supported": true | false,\n'
        '  "supporting_chunk_id": "<chunk_id or null>",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "rationale": "One sentence."\n'
        "}"
    )


def aggregate_answers_prompt(original_question: str, sub_qa: list[tuple[str, str]]) -> str:
    pairs = "\n\n".join(
        f"Sub-question {i + 1}: {q}\nAnswer: {a}" for i, (q, a) in enumerate(sub_qa)
    )
    return (
        f"Original question: {original_question}\n\n"
        f"Sub-question answers:\n{pairs}\n\n"
        "Synthesize a coherent, complete answer to the original question. "
        "Preserve all citation markers from the sub-answers."
    )
