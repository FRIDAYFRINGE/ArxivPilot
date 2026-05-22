from __future__ import annotations
import re
from pathlib import Path


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Returns list of {page_number, section_header, raw_text} dicts.
    Section headers are detected by font size: if a block's max font size
    is > 1.1x the document body font size, it's treated as a header.
    """
    import fitz  # pymupdf

    doc = fitz.open(str(pdf_path))
    body_font_size = _estimate_body_font_size(doc)

    pages = []
    current_section = "Introduction"

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # skip non-text blocks
                continue
            text, max_font = _extract_block_text_and_font(block)
            if not text.strip():
                continue
            if _is_section_header(text, max_font, body_font_size):
                current_section = text.strip()
            else:
                pages.append({
                    "page_number": page_num,
                    "section_header": current_section,
                    "raw_text": text.strip(),
                })

    doc.close()
    return pages


def extract_references(pdf_path: Path) -> list[str]:
    """
    Scans the last 20% of pages for arxiv IDs (pattern: NNNN.NNNNN).
    Returns deduplicated list of matched IDs.
    """
    import fitz

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    start_page = max(0, int(total_pages * 0.80))

    full_text = ""
    for page in doc:
        if page.number >= start_page:
            full_text += page.get_text()
    doc.close()

    # Match arxiv IDs: old format (hep-th/9901001) and new (1706.03762, 2305.10601)
    pattern = r"\b(\d{4}\.\d{4,5})\b"
    matches = re.findall(pattern, full_text)
    return list(dict.fromkeys(matches))  # deduplicate, preserve order


def _estimate_body_font_size(doc) -> float:
    font_sizes: list[float] = []
    for page in list(doc)[:5]:  # sample first 5 pages
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span.get("size", 0))
    if not font_sizes:
        return 10.0
    # Body font = most common size (mode-like: median of lower half)
    font_sizes.sort()
    return font_sizes[len(font_sizes) // 2]


def _extract_block_text_and_font(block: dict) -> tuple[str, float]:
    lines = block.get("lines", [])
    text_parts = []
    max_font = 0.0
    for line in lines:
        for span in line.get("spans", []):
            text_parts.append(span.get("text", ""))
            max_font = max(max_font, span.get("size", 0))
    return " ".join(text_parts), max_font


def _is_section_header(text: str, font_size: float, body_font_size: float) -> bool:
    if font_size <= body_font_size * 1.1:
        return False
    if len(text.strip()) > 120:  # headers are short
        return False
    return True
