from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    arxiv_url: str
    pdf_path: str = ""
    references: list[str] = field(default_factory=list)

    def short_label(self) -> str:
        first_author = self.authors[0].split()[-1] if self.authors else "Unknown"
        return f"{first_author} et al. ({self.year})"


@dataclass
class CitationEdge:
    source_paper_id: str
    target_paper_id: str
    context: str = ""
