from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ChunkSchema:
    chunk_id: str
    paper_id: str
    text: str
    section: str
    page_number: int
    chunk_index: int
    token_count: int
    paper_title: str
    authors: list[str]
    year: int
    arxiv_url: str
    embedding: list[float] | None = None
    bm25_doc_id: int | None = None
    cited_paper_ids: list[str] = field(default_factory=list)

    def citation_label(self) -> str:
        first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        return f"{first_author} et al. ({self.year})"

    def source_label(self) -> str:
        return f"{self.paper_title} | {self.section} | p.{self.page_number + 1}"
