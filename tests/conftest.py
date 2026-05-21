import uuid
import pytest
from paper_intel.models.chunk import ChunkSchema
from paper_intel.models.paper import PaperMetadata


def make_chunk(
    text: str = "The attention mechanism computes a weighted sum of values.",
    paper_id: str = "1706.03762",
    section: str = "Abstract",
    chunk_index: int = 0,
) -> ChunkSchema:
    return ChunkSchema(
        chunk_id=str(uuid.uuid4()),
        paper_id=paper_id,
        text=text,
        section=section,
        page_number=0,
        chunk_index=chunk_index,
        token_count=len(text.split()),
        paper_title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        year=2017,
        arxiv_url="https://arxiv.org/abs/1706.03762",
    )


def make_paper(
    paper_id: str = "1706.03762",
    references: list[str] | None = None,
) -> PaperMetadata:
    return PaperMetadata(
        paper_id=paper_id,
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        abstract="We propose the Transformer architecture based solely on attention mechanisms.",
        year=2017,
        arxiv_url="https://arxiv.org/abs/1706.03762",
        references=references or [],
    )


@pytest.fixture
def sample_chunk():
    return make_chunk()


@pytest.fixture
def sample_chunks():
    return [
        make_chunk(
            text="Attention mechanisms allow models to focus on relevant input tokens.",
            chunk_index=0,
        ),
        make_chunk(
            text="The Transformer relies entirely on self-attention to compute representations.",
            chunk_index=1,
        ),
        make_chunk(
            text="Multi-head attention projects queries, keys, and values h times.",
            chunk_index=2,
        ),
    ]


@pytest.fixture
def sample_paper():
    return make_paper()
