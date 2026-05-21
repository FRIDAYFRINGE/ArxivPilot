from __future__ import annotations
import uuid

from paper_intel.models.chunk import ChunkSchema
from paper_intel.models.paper import PaperMetadata


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def chunk_paper(
    pages: list[dict],
    metadata: PaperMetadata,
    chunk_size_tokens: int = 350,
    overlap_tokens: int = 70,
) -> list[ChunkSchema]:
    """
    Sentence-aware sliding window chunker.
    Groups text blocks by section, then splits into overlapping chunks.
    Each chunk carries the section name and page number of its first sentence.
    """
    # Group consecutive blocks into section buckets
    sections = _group_by_section(pages)
    chunks: list[ChunkSchema] = []
    chunk_index = 0

    for section_name, blocks in sections:
        sentences = _split_to_sentences(blocks)
        if not sentences:
            continue

        start = 0
        while start < len(sentences):
            # Accumulate sentences until chunk_size_tokens exceeded
            accumulated: list[tuple[str, int]] = []  # (sentence, page_number)
            token_count = 0
            i = start
            while i < len(sentences) and token_count < chunk_size_tokens:
                s_text, s_page = sentences[i]
                token_count += estimate_tokens(s_text)
                accumulated.append((s_text, s_page))
                i += 1

            if not accumulated:
                break

            chunk_text = " ".join(s for s, _ in accumulated)
            chunk_page = accumulated[0][1]

            chunks.append(ChunkSchema(
                chunk_id=str(uuid.uuid4()),
                paper_id=metadata.paper_id,
                text=chunk_text,
                section=section_name,
                page_number=chunk_page,
                chunk_index=chunk_index,
                token_count=estimate_tokens(chunk_text),
                paper_title=metadata.title,
                authors=metadata.authors,
                year=metadata.year,
                arxiv_url=metadata.arxiv_url,
                cited_paper_ids=list(metadata.references),
            ))
            chunk_index += 1

            # Advance by (chunk - overlap): back up overlap_tokens worth of sentences
            backed = 0
            back_i = len(accumulated) - 1
            while back_i >= 0 and backed < overlap_tokens:
                backed += estimate_tokens(accumulated[back_i][0])
                back_i -= 1
            overlap_count = len(accumulated) - 1 - back_i
            start += max(1, len(accumulated) - overlap_count)

    return chunks


def _group_by_section(pages: list[dict]) -> list[tuple[str, list[dict]]]:
    """Returns [(section_name, [block_dicts])] preserving order."""
    groups: list[tuple[str, list[dict]]] = []
    current_section = None
    current_blocks: list[dict] = []

    for block in pages:
        section = block["section_header"]
        if section != current_section:
            if current_blocks and current_section is not None:
                groups.append((current_section, current_blocks))
            current_section = section
            current_blocks = [block]
        else:
            current_blocks.append(block)

    if current_blocks and current_section is not None:
        groups.append((current_section, current_blocks))

    return groups


def _split_to_sentences(blocks: list[dict]) -> list[tuple[str, int]]:
    """
    Splits blocks into (sentence, page_number) pairs.
    Splits on '. ', '? ', '! ' — avoids splitting abbreviations crudely
    by requiring the sentence fragment to be at least 20 chars.
    """
    sentences: list[tuple[str, int]] = []
    for block in blocks:
        page = block["page_number"]
        text = block["raw_text"].strip()
        if not text:
            continue
        # naive sentence split on ". " with min length guard
        raw_sentences = text.replace("? ", ". ").replace("! ", ". ").split(". ")
        for s in raw_sentences:
            s = s.strip()
            if len(s) >= 15:
                sentences.append((s, page))
    return sentences
