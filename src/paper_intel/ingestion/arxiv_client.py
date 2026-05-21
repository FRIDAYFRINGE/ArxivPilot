from __future__ import annotations
import time
from pathlib import Path

import arxiv

from paper_intel.models.paper import PaperMetadata

# arxiv enforces ~3 req/sec; we stay well under that and back off on 429/503
_INTER_REQUEST_DELAY = 1.0  # seconds between metadata fetches (arxiv allows ~3 req/sec)


def _with_retry(fn, retries: int = 4):
    """Call fn(), retrying with exponential backoff on rate-limit errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
            print(f"  arxiv error ({e}), retrying in {wait}s...", flush=True)
            time.sleep(wait)


def fetch_paper_metadata(arxiv_id: str) -> PaperMetadata:
    def _fetch():
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(client.results(search))
        if not results:
            raise ValueError(f"No paper found for arxiv ID: {arxiv_id}")
        r = results[0]
        return PaperMetadata(
            paper_id=arxiv_id,
            title=r.title,
            authors=[str(a) for a in r.authors[:3]],
            abstract=r.summary,
            year=r.published.year,
            arxiv_url=r.entry_id,
        )
    return _with_retry(_fetch)


def download_pdf(metadata: PaperMetadata, dest_dir: Path) -> Path:
    import requests

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{metadata.paper_id}.pdf"
    if dest.exists():
        metadata.pdf_path = str(dest)
        return dest

    def _download():
        url = f"https://arxiv.org/pdf/{metadata.paper_id}"
        resp = requests.get(url, timeout=120, headers={"User-Agent": "paper-intel/0.1"})
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        metadata.pdf_path = str(dest)
        return dest

    return _with_retry(_download)


def search_papers(query: str, max_results: int = 100) -> list[PaperMetadata]:
    """
    Search arxiv by query string.
    Respects rate limits — use small max_results for interactive calls,
    larger values for bulk background ingestion.
    """
    def _search():
        client = arxiv.Client(num_retries=3, page_size=min(max_results, 100))
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for r in client.results(search):
            arxiv_id = r.entry_id.split("/")[-1].split("v")[0]
            papers.append(PaperMetadata(
                paper_id=arxiv_id,
                title=r.title,
                authors=[str(a) for a in r.authors[:3]],
                abstract=r.summary,
                year=r.published.year,
                arxiv_url=r.entry_id,
            ))
            time.sleep(_INTER_REQUEST_DELAY)
        return papers

    return _with_retry(_search)
