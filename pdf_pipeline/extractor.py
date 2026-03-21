"""
pdf_pipeline/extractor.py
==========================
Top-level orchestrator for hybrid PDF extraction.

Architecture decision:
  No single library works reliably across all bank PDFs.
  pdfplumber is best on clean text-layer PDFs (most bank docs).
  PyMuPDF is faster and handles some edge cases pdfplumber chokes on.
  Camelot extracts tables with far higher accuracy than any OCR approach.
  OCR (PaddleOCR → Tesseract) is reserved for genuinely scanned pages only.

  This cascade is intentional: we try the cheapest/most accurate method first
  and only escalate when the previous step produces insufficient text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Minimum characters on a page before we consider it "text-layer present"
_MIN_TEXT_CHARS = 50
# Fraction of pages that must have text for the whole PDF to be "text-layer"
_TEXT_LAYER_PAGE_FRACTION = 0.5


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: list[dict]          # list of {headers, rows, raw_text}
    extraction_method: str      # "pdfplumber" | "pymupdf" | "ocr_paddle" | "ocr_tesseract"
    confidence: float = 1.0


@dataclass
class ExtractedDocument:
    source_path: str
    bank: str
    pages: list[PageContent]
    is_scanned: bool
    total_chars: int = 0
    extraction_method: str = ""

    def __post_init__(self):
        self.total_chars = sum(len(p.text) for p in self.pages)


def is_scanned_pdf(pdf_path: Path) -> bool:
    """
    Detect whether a PDF is scanned (image-only) or has a text layer.

    Strategy: extract text from a sample of pages with pdfplumber.
    If fewer than _TEXT_LAYER_PAGE_FRACTION of pages have >= _MIN_TEXT_CHARS,
    treat the whole document as scanned.
    """
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            n = len(pdf.pages)
            sample = pdf.pages[:min(n, 5)]   # check first 5 pages
            text_pages = sum(
                1 for p in sample
                if len((p.extract_text() or "").strip()) >= _MIN_TEXT_CHARS
            )
            is_scanned = (text_pages / max(len(sample), 1)) < _TEXT_LAYER_PAGE_FRACTION
            logger.info(
                "[Extractor] %s — %d/%d sample pages have text → %s",
                pdf_path.name, text_pages, len(sample),
                "SCANNED" if is_scanned else "TEXT-LAYER",
            )
            return is_scanned
    except Exception as e:
        logger.warning("[Extractor] is_scanned_pdf failed: %s — assuming text-layer", e)
        return False


def extract_pdf(pdf_path: str | Path, bank: str = "Unknown") -> ExtractedDocument:
    """
    Main entry point. Routes to the right extraction strategy automatically.

    Returns an ExtractedDocument with per-page content ready for chunking.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("[Extractor] Starting: %s (%.1f MB)", pdf_path.name,
                pdf_path.stat().st_size / 1_048_576)

    scanned = is_scanned_pdf(pdf_path)

    if not scanned:
        pages = _extract_text_pdf(pdf_path)
        method = "text_layer"
    else:
        pages = _extract_scanned_pdf(pdf_path)
        method = "ocr"

    # For any page still below threshold after primary extraction, try OCR escalation
    pages = _escalate_empty_pages(pdf_path, pages)

    doc = ExtractedDocument(
        source_path=str(pdf_path),
        bank=bank,
        pages=pages,
        is_scanned=scanned,
        extraction_method=method,
    )
    logger.info(
        "[Extractor] Done: %s — %d pages  %d chars  method=%s",
        pdf_path.name, len(pages), doc.total_chars, method,
    )
    return doc


# ---------------------------------------------------------------------------
# Text-layer path
# ---------------------------------------------------------------------------

def _extract_text_pdf(pdf_path: Path) -> list[PageContent]:
    """
    Extract text + tables from a text-layer PDF.

    pdfplumber is primary: it gives reliable character-level text with
    layout preservation.  PyMuPDF is the fallback for pages pdfplumber
    fails on (rare, usually malformed pages).
    """
    pages: list[PageContent] = []

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = (page.extract_text() or "").strip()

                if len(text) < _MIN_TEXT_CHARS:
                    # pdfplumber got nothing — try PyMuPDF on this page
                    fallback = _pymupdf_single_page(pdf_path, i - 1)
                    if len(fallback) >= _MIN_TEXT_CHARS:
                        text = fallback
                        method = "pymupdf"
                    else:
                        method = "pdfplumber_empty"
                else:
                    method = "pdfplumber"

                from pdf_pipeline.table_extractor import extract_tables_from_page
                tables = extract_tables_from_page(pdf_path, i)

                pages.append(PageContent(
                    page_number=i,
                    text=text,
                    tables=tables,
                    extraction_method=method,
                ))
    except Exception as e:
        logger.warning("[Extractor] pdfplumber failed for %s: %s — using PyMuPDF", pdf_path.name, e)
        pages = _pymupdf_all_pages(pdf_path)

    return pages


def _pymupdf_single_page(pdf_path: Path, page_index: int) -> str:
    """Extract text from a single page using PyMuPDF."""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            return doc[page_index].get_text("text").strip()
        finally:
            doc.close()
    except Exception as e:
        logger.debug("[Extractor] PyMuPDF single page failed p%d: %s", page_index + 1, e)
        return ""


def _pymupdf_all_pages(pdf_path: Path) -> list[PageContent]:
    """Full PyMuPDF extraction — used when pdfplumber fails entirely."""
    pages = []
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        for i in range(len(doc)):
            text = doc[i].get_text("text").strip()
            pages.append(PageContent(
                page_number=i + 1,
                text=text,
                tables=[],
                extraction_method="pymupdf",
            ))
        doc.close()
    except Exception as e:
        logger.error("[Extractor] PyMuPDF all-pages failed: %s", e)
    return pages


# ---------------------------------------------------------------------------
# Scanned path
# ---------------------------------------------------------------------------

def _extract_scanned_pdf(pdf_path: Path) -> list[PageContent]:
    """
    Extract text from a scanned PDF using OCR.

    PaddleOCR is primary (better accuracy on printed documents).
    Tesseract is fallback (universally available, no deep learning required).
    """
    from pdf_pipeline.ocr import render_pdf_pages, paddle_ocr_page, tesseract_ocr_page

    pages: list[PageContent] = []
    rendered = render_pdf_pages(pdf_path, dpi=300)

    for page_num, img_bytes in rendered:
        # Try PaddleOCR first
        text, method = paddle_ocr_page(img_bytes), "ocr_paddle"
        if len(text.strip()) < _MIN_TEXT_CHARS:
            text = tesseract_ocr_page(img_bytes)
            method = "ocr_tesseract"

        if len(text.strip()) < _MIN_TEXT_CHARS:
            logger.warning("[Extractor] p%d — all OCR methods returned < %d chars",
                           page_num, _MIN_TEXT_CHARS)
            method = "ocr_failed"

        pages.append(PageContent(
            page_number=page_num,
            text=text.strip(),
            tables=[],
            extraction_method=method,
        ))

    return pages


# ---------------------------------------------------------------------------
# Escalation for empty pages in text PDFs
# ---------------------------------------------------------------------------

def _escalate_empty_pages(pdf_path: Path, pages: list[PageContent]) -> list[PageContent]:
    """
    For any page that ended up with < _MIN_TEXT_CHARS after primary extraction,
    attempt OCR as a last resort.  This handles hybrid PDFs that are mostly
    text-layer but contain a few scanned pages (e.g. signed signature pages).
    """
    empty_indices = [
        i for i, p in enumerate(pages)
        if len(p.text) < _MIN_TEXT_CHARS and p.extraction_method != "ocr_failed"
    ]
    if not empty_indices:
        return pages

    logger.info("[Extractor] Escalating %d empty page(s) to OCR", len(empty_indices))
    from pdf_pipeline.ocr import render_pdf_pages, paddle_ocr_page, tesseract_ocr_page

    rendered_map = {
        page_num: img
        for page_num, img in render_pdf_pages(pdf_path, dpi=300, page_indices=[
            pages[i].page_number - 1 for i in empty_indices
        ])
    }

    for i in empty_indices:
        page = pages[i]
        img_bytes = rendered_map.get(page.page_number)
        if img_bytes is None:
            continue
        text = paddle_ocr_page(img_bytes)
        if len(text.strip()) < _MIN_TEXT_CHARS:
            text = tesseract_ocr_page(img_bytes)
            method = "ocr_tesseract_escalated"
        else:
            method = "ocr_paddle_escalated"
        pages[i] = PageContent(
            page_number=page.page_number,
            text=text.strip(),
            tables=page.tables,
            extraction_method=method,
        )

    return pages
