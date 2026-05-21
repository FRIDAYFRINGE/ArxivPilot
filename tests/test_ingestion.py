import pytest
from paper_intel.ingestion.chunker import chunk_paper, estimate_tokens
from paper_intel.models.paper import PaperMetadata


def _sample_pages(n_blocks: int = 20) -> list[dict]:
    sentences = [
        "The attention mechanism allows models to focus on relevant parts of the input sequence.",
        "Self-attention computes relationships between all positions in a sequence.",
        "Multi-head attention enables attending to different representation subspaces.",
        "The Transformer eliminates recurrence entirely and uses only attention mechanisms.",
        "Positional encoding injects sequence order information into the model.",
    ]
    pages = []
    for i in range(n_blocks):
        pages.append({
            "page_number": i // 4,
            "section_header": "Introduction" if i < 10 else "Methods",
            "raw_text": sentences[i % len(sentences)],
        })
    return pages


def _sample_metadata() -> PaperMetadata:
    return PaperMetadata(
        paper_id="1706.03762",
        title="Attention Is All You Need",
        authors=["Vaswani, A."],
        abstract="Abstract text",
        year=2017,
        arxiv_url="https://arxiv.org/abs/1706.03762",
    )


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world") == int(2 * 1.3)

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestChunkPaper:
    def test_produces_chunks(self):
        pages = _sample_pages(30)
        meta = _sample_metadata()
        chunks = chunk_paper(pages, meta, chunk_size_tokens=50, overlap_tokens=10)
        assert len(chunks) > 0

    def test_chunk_metadata(self):
        pages = _sample_pages(30)
        meta = _sample_metadata()
        chunks = chunk_paper(pages, meta, chunk_size_tokens=50, overlap_tokens=10)
        for c in chunks:
            assert c.paper_id == "1706.03762"
            assert c.paper_title == "Attention Is All You Need"
            assert c.year == 2017
            assert c.chunk_id != ""

    def test_sections_preserved(self):
        pages = _sample_pages(30)
        meta = _sample_metadata()
        chunks = chunk_paper(pages, meta)
        sections = {c.section for c in chunks}
        assert "Introduction" in sections or "Methods" in sections

    def test_overlap_exists(self):
        pages = _sample_pages(40)
        meta = _sample_metadata()
        chunks = chunk_paper(pages, meta, chunk_size_tokens=30, overlap_tokens=15)
        if len(chunks) > 1:
            # Consecutive chunks should share at least some common words
            words_0 = set(chunks[0].text.lower().split())
            words_1 = set(chunks[1].text.lower().split())
            # There should be SOME overlap (common words)
            assert len(words_0 & words_1) > 0
