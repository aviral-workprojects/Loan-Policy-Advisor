"""
pdf_pipeline.py  (v3 — multimodal, page-level, resilient)
===========================================================
Production-grade PDF intelligence pipeline.

Architecture:
    PDF file
        → file-size guard (>5MB → fallback immediately)
        → split into single-page PDFs (via pdfplumber)
        → per-page multi-model cascade:
            Step 1: nemoretriever-page-elements-v3  (structure)
            Step 2: nemotron-ocr-v1                 (OCR fallback for failed/image pages)
            Step 3: nemotron-table-structure-v1     (table enhancement when tables present)
        → merge per-page results
        → section-aware / table-aware chunking
        → attach metadata (bank, field_hint, section_title, model_source)
        → cache to processed_data/<stem>.json
        → List[StructuredChunk]

Safe fallback:
    Any stage failure → pdfplumber plain-text extraction.
    System never fully fails.

Key behaviours:
    500 errors → retried (up to max_retries)
    422 errors → not retried (bad request format)
    401/403    → raises immediately (auth error — user must fix key)
    PDF > 5MB  → skips NVIDIA, uses pdfplumber
    Page fails → OCR attempted before giving up on that page

Usage:
    from pdf_pipeline import process_pdf, PDFPipeline, test_single_pdf

    chunks = process_pdf("data/hdfc_pdfs/hdfc_personal_loan.pdf")
    for c in chunks:
        print(c["bank"], c["doc_type"], c["model_source"], c["text"][:80])
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field as dc_field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _cfg(key: str, default: Any) -> Any:
    try:
        import config as c
        return getattr(c, key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

_PAGE_ELEMENTS_URL   = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v3"
_OCR_URL             = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1"
_TABLE_STRUCTURE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1"

# File size above which NVIDIA API is skipped entirely (bytes)
_MAX_PDF_BYTES = 5 * 1024 * 1024   # 5 MB

# NVIDIA base64 image payload size limit (~180 KB encoded).
# DPI is stepped down if the encoded image exceeds this.
_MAX_B64_BYTES  = 180_000
_DPI_HIGH       = 150   # first attempt
_DPI_LOW        = 72    # retry if image too large — 72 DPI stays safely under NVIDIA limit

# HTTP status codes that warrant a retry
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# HTTP status codes that are fatal — no point retrying
_FATAL_STATUSES = {400, 401, 403, 422}


# ---------------------------------------------------------------------------
# StructuredChunk — output data structure
# ---------------------------------------------------------------------------

@dataclass
class StructuredChunk:
    """
    Single knowledge unit produced by the PDF pipeline.
    Fully compatible with DocChunk in retrieval.py.
    model_source records which model (or fallback) produced this chunk.
    """
    text:             str
    source:           str            # original filename
    bank:             str
    doc_type:         str            # "rule"|"table"|"paragraph"|"regulatory"|"pdf"|"ocr"
    chunk_id:         int   = 0
    similarity_score: float = 0.0    # populated at retrieval time
    page_number:      int   = 0
    section_title:    str   = ""
    element_type:     str   = "text" # "text"|"table"|"title"|"figure_caption"|"ocr"
    field_hint:       str   = ""     # "monthly_income"|"credit_score"|…
    structured_data:  dict  = dc_field(default_factory=dict)
    content_hash:     str   = ""
    model_source:     str   = ""     # "page_elements"|"ocr"|"table_structure"|"pdfplumber"

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]

    def to_doc_chunk(self):
        from services.retrieval import DocChunk
        return DocChunk(
            text=self.text, source=self.source, bank=self.bank,
            doc_type=self.doc_type, chunk_id=self.chunk_id,
            similarity_score=self.similarity_score,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Bank / field detection
# ---------------------------------------------------------------------------

_BANK_KEYWORDS: dict[str, list[str]] = {
    "Axis":        ["axis bank", "axis personal", "axisbank"],
    "HDFC":        ["hdfc bank", "hdfc personal", "hdfcbank"],
    "ICICI":       ["icici bank", "icici personal", "icicidirect"],
    "SBI":         ["state bank", "sbi", "xpress credit", "sbi personal"],
    "RBI":         ["reserve bank", "rbi", "rbi guidelines", "master circular"],
    "Paisabazaar": ["paisabazaar"],
    "BankBazaar":  ["bankbazaar"],
}

_BANK_PATH_MAP = {
    "hdfc": "HDFC", "sbi": "SBI", "icici": "ICICI",
    "axis": "Axis", "rbi": "RBI",
}

_FIELD_PATTERNS: list[tuple[str, str]] = [
    (r"income|salary|earning",                "monthly_income"),
    (r"cibil|credit\s*score|credit\s*rating", "credit_score"),
    (r"\bage\b|years\s*old|minimum.*age",     "age"),
    (r"dti|debt.to.income|emi.*income",       "dti_ratio"),
    (r"employment|salaried|self.employed",    "employment_type"),
    (r"experience|work.*month|tenure",        "work_experience_months"),
]

def detect_bank(filename: str, content_sample: str = "") -> str:
    text = (filename + " " + content_sample[:500]).lower()
    for bank, kws in _BANK_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return bank
    for key, bank in _BANK_PATH_MAP.items():
        if key in filename.lower():
            return bank
    return "Unknown"

def detect_field_hint(text: str) -> str:
    text_lower = text.lower()
    for pattern, f in _FIELD_PATTERNS:
        if re.search(pattern, text_lower):
            return f
    return ""


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}.*?(Ltd|Limited|Bank)\b", "", text)
    return text.strip()

def _split_into_sentences(text: str, max_words: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current, count = [], [], 0
    for s in sentences:
        wc = len(s.split())
        if count + wc > max_words and current:
            chunks.append(" ".join(current))
            current, count = [s], wc
        else:
            current.append(s)
            count += wc
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Table → text
# ---------------------------------------------------------------------------

def table_to_text(table_data: dict | list | str) -> tuple[str, dict]:
    if isinstance(table_data, str):
        return _clean_text(table_data), {"raw": table_data}

    if isinstance(table_data, list):
        rows, structured_rows = [], []
        for row in table_data:
            if isinstance(row, (list, tuple)):
                rows.append(" | ".join(str(c).strip() for c in row if str(c).strip()))
                structured_rows.append(list(row))
            elif isinstance(row, dict):
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
                structured_rows.append(row)
            else:
                rows.append(str(row))
        return _clean_text("\n".join(rows)), {"rows": structured_rows}

    if isinstance(table_data, dict):
        headers = table_data.get("headers", table_data.get("columns", []))
        rows    = table_data.get("rows", table_data.get("data", []))
        lines   = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
        for row in rows:
            if isinstance(row, (list, tuple)):
                if headers and len(headers) == len(row):
                    lines.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, row)))
                else:
                    lines.append(" | ".join(str(c) for c in row))
            elif isinstance(row, dict):
                lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return _clean_text("\n".join(lines)), table_data

    return str(table_data), {"raw": str(table_data)}


# ---------------------------------------------------------------------------
# Page splitting (pdfplumber)
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path, dpi: int = _DPI_HIGH) -> list[tuple[int, bytes]]:
    """
    Convert a PDF to a list of (page_number, png_bytes) tuples using PyMuPDF.

    DPI controls resolution:
        150 dpi  → good quality, ~100-160 KB per page encoded
        100 dpi  → compressed, ~50-90 KB per page encoded (fits NVIDIA limit more safely)

    Falls back to pypdf page-splitting if PyMuPDF is not installed,
    returning single-page PDF bytes instead of PNGs.  The caller
    (_nvidia_post) always expects PNG, so the fallback raises ImportError
    clearly rather than silently sending the wrong format.

    Raises:
        ImportError  if neither PyMuPDF nor a fallback is available.
    """
    try:
        import fitz   # PyMuPDF
        doc    = fitz.open(str(pdf_path))
        images: list[tuple[int, bytes]] = []
        matrix = fitz.Matrix(dpi / 72, dpi / 72)   # 72 DPI is the PDF default

        for page_index in range(len(doc)):
            page    = doc[page_index]
            pixmap  = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")

            logger.debug(
                "[pdf_to_images] p%d  dpi=%d  size=%.1f KB",
                page_index + 1, dpi, len(png_bytes) / 1024,
            )
            images.append((page_index + 1, png_bytes))

        doc.close()
        logger.info(
            "[pdf_to_images] %s → %d page image(s) at %d DPI",
            pdf_path.name, len(images), dpi,
        )
        return images

    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for NVIDIA CV API calls. "
            "Install it with:  pip install pymupdf"
        )


def _image_to_b64(png_bytes: bytes) -> str:
    """Encode PNG bytes as a base64 string."""
    return base64.b64encode(png_bytes).decode("utf-8")


def _compress_image(
    pdf_path: Path,
    page_index: int,
    dpi: int = _DPI_HIGH,
) -> tuple[bytes, int, bool]:
    """
    Render one PDF page to PNG at the given DPI.
    If the base64 size exceeds _MAX_B64_BYTES, retry at _DPI_LOW.

    Returns:
        (png_bytes, dpi_used, was_compressed)
    """
    import fitz
    doc = fitz.open(str(pdf_path))
    try:
        page   = doc[page_index]
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix    = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")
    finally:
        doc.close()

    b64_len = len(_image_to_b64(png_bytes))

    if b64_len < _MAX_B64_BYTES:
        return png_bytes, dpi, False

    # Too large — retry at lower DPI
    logger.info(
        "[PDFPipeline] p%d  image %.0f KB > limit %.0f KB — compressing to %d DPI",
        page_index + 1, b64_len / 1024, _MAX_B64_BYTES / 1024, _DPI_LOW,
    )
    doc2   = fitz.open(str(pdf_path))
    try:
        page2  = doc2[page_index]
        matrix2 = fitz.Matrix(_DPI_LOW / 72, _DPI_LOW / 72)
        pix2   = page2.get_pixmap(matrix=matrix2, alpha=False)
        png_bytes2 = pix2.tobytes("png")
    finally:
        doc2.close()

    return png_bytes2, _DPI_LOW, True


# kept for backward-compat (used by old tests that import _split_pdf_pages)
def _split_pdf_pages(pdf_path: Path) -> list[bytes]:
    """Legacy alias — returns single-page PDF bytes. Use pdf_to_images() instead."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        pages: list[bytes] = []
        for page in reader.pages:
            writer = pypdf.PdfWriter()
            writer.add_page(page)
            buf = io.BytesIO()
            writer.write(buf)
            pages.append(buf.getvalue())
        return pages
    except ImportError:
        return [pdf_path.read_bytes()]


# ---------------------------------------------------------------------------
# Low-level NVIDIA API caller (shared by all three endpoints)
# ---------------------------------------------------------------------------

def _nvidia_post(
    url:         str,
    api_key:     str,
    image_bytes: bytes,        # PNG bytes — NOT pdf bytes, NOT a file path
    filename:    str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
    timeout:     int   = 30,
) -> dict | list:
    """
    POST a PNG image to any NVIDIA CV endpoint using the correct format:
        JSON payload: {"input": [{"type": "image_url",
                                  "url":  "data:image/png;base64,<b64>"}]}

    This is the format all NVIDIA multimodal CV models expect.
    Sending PDFs directly (multipart/form-data) causes HTTP 500 errors.

    Retry policy:
        Retried:     timeout, 429, 500, 502, 503, 504
        Not retried: 400, 401, 403, 422

    Returns:
        Parsed JSON response dict or list.
    Raises:
        requests.exceptions.HTTPError for fatal status codes.
        RuntimeError if all retries exhausted.
    """
    image_b64 = _image_to_b64(image_bytes)
    b64_len   = len(image_b64)

    if b64_len >= _MAX_B64_BYTES:
        # Caller should have pre-compressed; warn but proceed anyway
        logger.warning(
            "[NVIDIA API] Image b64 size %d exceeds limit %d for %s — "
            "API may reject; consider lower DPI",
            b64_len, _MAX_B64_BYTES, filename,
        )

    logger.debug(
        "[NVIDIA API] %s  image_b64_len=%d (%.1f KB)",
        url.split("/")[-1], b64_len, b64_len / 1024,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    payload = {
        "input": [
            {
                "type": "image_url",
                "url":  f"data:image/png;base64,{image_b64}",
            }
        ]
    }

    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # ── Detailed pre-raise logging ────────────────────────────────
            if resp.status_code in _FATAL_STATUSES:
                logger.error(
                    "[NVIDIA API] HTTP %d %s  url=%s  body=%s",
                    resp.status_code, resp.reason, url.split("/")[-1], resp.text[:400],
                )
                if resp.status_code in (401, 403):
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code} — check NVIDIA_API_KEY", response=resp
                    )
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp
                )

            if resp.status_code in _RETRYABLE_STATUSES:
                wait = retry_delay * attempt
                logger.warning(
                    "[NVIDIA API] HTTP %d on attempt %d/%d for %s — retrying in %.1fs  body=%s",
                    resp.status_code, attempt, max_retries,
                    url.split("/")[-1], wait, resp.text[:200],
                )
                last_err = requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code}", response=resp
                )
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            wait = retry_delay * attempt
            last_err = TimeoutError(f"Timed out on attempt {attempt}")
            logger.warning(
                "[NVIDIA API] Timeout on attempt %d/%d for %s — retrying in %.1fs",
                attempt, max_retries, url.split("/")[-1], wait,
            )
            if attempt < max_retries:
                time.sleep(wait)

        except requests.exceptions.HTTPError:
            raise

        except Exception as e:
            last_err = e
            logger.warning(
                "[NVIDIA API] Unexpected error on attempt %d/%d: %s",
                attempt, max_retries, e,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    logger.error(
        "[NVIDIA API] Final failure after %d retries for %s: %s",
        max_retries, url.split("/")[-1], last_err,
    )
    raise RuntimeError(
        f"NVIDIA API ({url.split('/')[-1]}) failed after {max_retries} attempts: {last_err}"
    )


# ---------------------------------------------------------------------------
# Step 1 — Page Elements
# ---------------------------------------------------------------------------

def _call_page_elements(
    page_bytes: bytes,
    filename:   str,
    api_key:    str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """
    Call nemoretriever-page-elements-v3 on a single-page PDF.
    Returns normalised element list: [{type, content, page, bbox}, …]
    Raises on non-retryable errors, RuntimeError on exhausted retries.
    """
    data = _nvidia_post(
        _PAGE_ELEMENTS_URL, api_key, page_bytes, filename,
        max_retries=max_retries, retry_delay=retry_delay,
    )

    # Normalise response shape.
    # NVIDIA NIM returns {"data": [...], "usage": {...}}
    # Some endpoints return {"pages": [...]}, {"elements": [...]}, or a bare list.
    if isinstance(data, list):
        raw_pages = data
    elif isinstance(data, dict):
        if "data" in data:
            # Primary NIM format: {"data": [<element>, ...], "usage": {...}}
            raw_pages = [{"page": 1, "elements": data["data"]}]
        elif "pages" in data:
            raw_pages = data["pages"]
        elif "elements" in data:
            raw_pages = [{"page": 1, "elements": data["elements"]}]
        else:
            logger.warning(
                "[PageElements] Unexpected response shape — keys=%s",
                list(data.keys()),
            )
            raw_pages = []
    else:
        logger.warning("[PageElements] Unexpected response type: %s", type(data))
        raw_pages = []

    elements: list[dict] = []
    for page_data in raw_pages:
        pn = page_data.get("page", 1)
        for elem in page_data.get("elements", []):
            # NIM uses "label" for element type and "text" for content;
            # older / other shapes use "type" and "content"
            elements.append({
                "type":    elem.get("label") or elem.get("type", "text"),
                "content": elem.get("text")  or elem.get("content", ""),
                "page":    pn,
                "bbox":    elem.get("bbox", []),
            })
    return elements


# ---------------------------------------------------------------------------
# Step 2 — OCR fallback
# ---------------------------------------------------------------------------

def _call_ocr(
    page_bytes:  bytes,
    filename:    str,
    api_key:     str,
    max_retries: int   = 2,
    retry_delay: float = 1.5,
) -> list[dict]:
    """
    Call nemotron-ocr-v1 on a page that page-elements failed on.
    Returns elements in the same normalised format as _call_page_elements.
    """
    data = _nvidia_post(
        _OCR_URL, api_key, page_bytes, filename,
        max_retries=max_retries, retry_delay=retry_delay,
    )

    # OCR response shape varies — normalise to our element format.
    # NIM primary format: {"data": [{"text": "...", ...}, ...], "usage": {...}}
    elements: list[dict] = []
    if isinstance(data, dict):
        if "data" in data:
            # Primary NIM format
            for item in data["data"]:
                t = item.get("text", item.get("content", "")) if isinstance(item, dict) else str(item)
                if str(t).strip():
                    elements.append({"type": "text", "content": _clean_text(str(t)), "page": 1, "bbox": []})
        else:
            text = data.get("text", data.get("content", data.get("result", "")))
            if isinstance(text, str) and text.strip():
                elements.append({"type": "text", "content": _clean_text(text), "page": 1, "bbox": []})
            elif isinstance(text, list):
                for item in text:
                    t = item.get("text", item.get("content", str(item))) if isinstance(item, dict) else str(item)
                    if t.strip():
                        elements.append({"type": "text", "content": _clean_text(t), "page": 1, "bbox": []})
    elif isinstance(data, list):
        for item in data:
            t = item.get("text", item.get("content", "")) if isinstance(item, dict) else str(item)
            if str(t).strip():
                elements.append({"type": "text", "content": _clean_text(str(t)), "page": 1, "bbox": []})
    elif isinstance(data, str) and data.strip():
        elements.append({"type": "text", "content": _clean_text(data), "page": 1, "bbox": []})

    return elements


# ---------------------------------------------------------------------------
# Step 3 — Table structure enhancement
# ---------------------------------------------------------------------------

def _call_table_structure(
    page_bytes:  bytes,
    filename:    str,
    api_key:     str,
    max_retries: int   = 2,
    retry_delay: float = 1.5,
) -> list[dict]:
    """
    Call nemotron-table-structure-v1 on a page that contains tables.
    Returns enhanced table elements merged into our element format.
    """
    data = _nvidia_post(
        _TABLE_STRUCTURE_URL, api_key, page_bytes, filename,
        max_retries=max_retries, retry_delay=retry_delay,
    )

    elements: list[dict] = []
    tables = []
    if isinstance(data, dict):
        if "data" in data:
            # Primary NIM format: {"data": [<table>, ...], "usage": {...}}
            tables = data["data"]
        else:
            tables = data.get("tables", data.get("results", [data] if "cells" in data or "rows" in data else []))
    elif isinstance(data, list):
        tables = data

    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        readable, structured = table_to_text(tbl)
        if readable.strip():
            elements.append({
                "type":            "table",
                "content":         tbl,
                "content_text":    readable,
                "page":            tbl.get("page", 1),
                "bbox":            tbl.get("bbox", []),
            })

    return elements


# ---------------------------------------------------------------------------
# Per-page processor (multi-model cascade)
# ---------------------------------------------------------------------------

@dataclass
class _PageResult:
    page_number: int
    elements:    list[dict]
    model_used:  str     # "page_elements" | "ocr" | "table_structure" | "pdfplumber"
    success:     bool

def _process_single_page(
    page_bytes:   bytes,
    page_number:  int,
    source_name:  str,
    api_key:      str,
    max_retries:  int,
    retry_delay:  float,
) -> _PageResult:
    """
    Multi-model cascade for a single page.

    Step 1: Try page_elements  → if OK, optionally enhance tables with table_structure
    Step 2: If step 1 fails    → try OCR
    Step 3: If step 2 fails    → return empty (caller falls back to pdfplumber for that page)
    """
    fname = f"page_{page_number}_{source_name}"

    # ── Step 1: Page Elements (layout + structure) ───────────────────────
    pe_elements: list[dict] = []
    pe_ok = False
    try:
        pe_elements = _call_page_elements(
            page_bytes, fname, api_key,
            max_retries=max_retries, retry_delay=retry_delay,
        )
        # Annotate page number; empty list is a valid response (blank page),
        # not treated as a failure — only an API exception counts as failure.
        for e in pe_elements:
            e["page"] = page_number
        pe_ok = True

        # Enhance tables if detected
        has_tables = any(e.get("type") == "table" for e in pe_elements)
        if has_tables:
            try:
                table_elements = _call_table_structure(
                    page_bytes, fname, api_key,
                    max_retries=1, retry_delay=retry_delay,
                )
                if table_elements:
                    non_table = [e for e in pe_elements if e.get("type") != "table"]
                    for te in table_elements:
                        te["page"] = page_number
                    pe_elements = non_table + table_elements
                    logger.info(
                        "[PDFPipeline] p%d table_structure enhanced %d table(s)",
                        page_number, len(table_elements),
                    )
            except Exception as te:
                logger.debug("[PDFPipeline] p%d table_structure skipped: %s", page_number, te)

        type_counts = {e.get("type","?"): 0 for e in pe_elements}
        for e in pe_elements:
            type_counts[e.get("type","?")] += 1
        logger.info("[PDFPipeline] p%d  page_elements OK  types=%s", page_number, type_counts)

    except Exception as e1:
        logger.warning(
            "[PDFPipeline] p%d  page_elements FAILED (%s) — will rely on OCR",
            page_number, e1,
        )

    # ── Step 2: OCR — always runs to capture text page_elements may miss ──
    # Runs regardless of whether page_elements succeeded, because
    # nemoretriever-page-elements focuses on layout while nemotron-ocr
    # specialises in reading text — their outputs are complementary.
    ocr_elements: list[dict] = []
    try:
        ocr_elements = _call_ocr(
            page_bytes, fname, api_key,
            max_retries=max(1, max_retries - 1), retry_delay=retry_delay,
        )
        for e in ocr_elements:
            e["page"]         = page_number
            e["element_type"] = "ocr"
        if ocr_elements:
            logger.info(
                "[PDFPipeline] p%d  OCR OK  elements=%d", page_number, len(ocr_elements),
            )
    except Exception as e2:
        logger.warning("[PDFPipeline] p%d  OCR FAILED (%s)", page_number, e2)

    # ── Merge: page_elements provides structure; OCR provides text ────────
    # page_elements returns structural labels reliably but text content is
    # often empty ("text": "") for scanned/stylised PDFs.
    # We must check actual content length, not just element type presence.
    if pe_ok or ocr_elements:
        def _has_meaningful_text(elems: list[dict], min_len: int = 20) -> bool:
            """True if any element has real readable content (not just a label)."""
            for e in elems:
                t = str(e.get("content", "") or "").strip()
                if len(t) >= min_len:
                    return True
            return False

        has_real_text = _has_meaningful_text(pe_elements)
        merged = list(pe_elements)

        if ocr_elements and not has_real_text:
            # page_elements gave us structural labels but no readable body text
            # (empty "text" field) — inject OCR output to fill the gap
            merged.extend(ocr_elements)
            logger.info(
                "[PDFPipeline] p%d  OCR injected (page_elements had no readable text)",
                page_number,
            )
        elif ocr_elements and not pe_ok:
            # page_elements API call failed entirely — use OCR as sole source
            merged = ocr_elements

        # Gate: only return success if at least one element has real text.
        # Must check all possible text fields — NIM uses different field names
        # across model versions ("content", "text", "content_text").
        def _has_any_content(elems: list[dict]) -> bool:
            for e in elems:
                t = (
                    str(e.get("content",       "") or "")
                    or str(e.get("text",        "") or "")
                    or str(e.get("content_text","") or "")
                ).strip()
                if len(t) > 10:
                    return True
            return False

        has_any_content = _has_any_content(merged)
        if merged and has_any_content:
            model_tag = (
                "page_elements+ocr" if (pe_ok and ocr_elements and not has_real_text)
                else ("ocr" if not pe_ok else "page_elements")
            )
            return _PageResult(page_number, merged, model_tag, True)

    # ── All NVIDIA models failed → signal pdfplumber gap-fill ─────────────
    logger.warning(
        "[PDFPipeline] p%d  all models failed — page will use pdfplumber", page_number,
    )
    return _PageResult(page_number, [], "pdfplumber", False)


# ---------------------------------------------------------------------------
# Full-PDF NVIDIA pipeline
# ---------------------------------------------------------------------------

def _call_page_elements_api(
    pdf_path:    Path,
    api_key:     str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """
    Public entry point for the multi-model page-level pipeline.

    Returns normalised list of page dicts: [{"page": N, "elements": [...], "model": …}]

    Behaviour:
        - File > 5MB → raises FileSizeError immediately (caller falls back)
        - Splits PDF into per-page bytes (pypdf)
        - Runs each page through the 3-step cascade in sequence
        - Collects results; failed pages marked for pdfplumber recovery
    """
    # ── File size guard ───────────────────────────────────────────────────
    size_bytes = pdf_path.stat().st_size
    if size_bytes > _MAX_PDF_BYTES:
        raise FileSizeError(
            f"{pdf_path.name} is {size_bytes/1024/1024:.1f} MB "
            f"(limit {_MAX_PDF_BYTES/1024/1024:.0f} MB) — using pdfplumber"
        )

    # ── Convert PDF pages to PNG images (required by NVIDIA CV models) ────
    try:
        page_images = pdf_to_images(pdf_path, dpi=_DPI_HIGH)
    except ImportError as ie:
        # PyMuPDF not installed — fall back immediately
        logger.warning("[PDFPipeline] %s — using pdfplumber", ie)
        raise FileSizeError("PyMuPDF not available") from ie

    logger.info(
        "[PDFPipeline] Processing %s — %d page(s), %.1f MB",
        pdf_path.name, len(page_images), size_bytes / 1024 / 1024,
    )

    all_pages: list[dict] = []
    failed_page_indices: list[int] = []

    for page_num, png_bytes in page_images:
        i = page_num - 1

        # Check size and compress if needed — avoids silent 500 from oversized image
        b64_len = len(_image_to_b64(png_bytes))
        compressed = False
        if b64_len >= _MAX_B64_BYTES:
            logger.info(
                "[PDFPipeline] p%d  %.0f KB > limit %.0f KB — recompressing at %d DPI",
                page_num, b64_len / 1024, _MAX_B64_BYTES / 1024, _DPI_LOW,
            )
            try:
                png_bytes, _, compressed = _compress_image(pdf_path, i, dpi=_DPI_HIGH)
            except Exception as ce:
                logger.warning("[PDFPipeline] p%d compression failed: %s", page_num, ce)

        if compressed:
            logger.info(
                "[PDFPipeline] p%d  compressed to %d DPI  new_size=%.0f KB",
                page_num, _DPI_LOW, len(_image_to_b64(png_bytes)) / 1024,
            )

        result = _process_single_page(
            png_bytes, page_num, pdf_path.name,
            api_key, max_retries, retry_delay,
        )
        if result.success and result.elements:
            all_pages.append({
                "page":     page_num,
                "elements": result.elements,
                "model":    result.model_used,
            })
        else:
            failed_page_indices.append(i)

    # ── Log summary ────────────────────────────────────────────────────────
    model_counts: dict[str, int] = {}
    for p in all_pages:
        m = p.get("model", "unknown")
        model_counts[m] = model_counts.get(m, 0) + 1

    logger.info(
        "[PDFPipeline] %s complete — %d/%d pages succeeded  models=%s  fallback_pages=%d",
        pdf_path.name, len(all_pages), len(page_images),
        model_counts, len(failed_page_indices),
    )

    return all_pages, failed_page_indices


class FileSizeError(Exception):
    """Raised when a PDF exceeds the configured size limit."""


# ---------------------------------------------------------------------------
# Element → StructuredChunk conversion
# ---------------------------------------------------------------------------

def _chunk_elements(
    elements:          list[dict],
    bank:              str,
    source:            str,
    model_source:      str = "page_elements",
    section_max_words: int = 400,
    table_max_words:   int = 300,
) -> list[StructuredChunk]:
    chunks:          list[StructuredChunk] = []
    chunk_id:        int  = 0
    current_section: str  = ""
    buffer_words:    list[str] = []
    buffer_page:     int  = 0

    def flush_buffer():
        nonlocal buffer_words, chunk_id, buffer_page
        if not buffer_words:
            return
        text = _clean_text(" ".join(buffer_words))
        if not text:
            buffer_words = []
            return
        chunks.append(StructuredChunk(
            text=text, source=source, bank=bank,
            doc_type="rule" if detect_field_hint(text) else "paragraph",
            chunk_id=chunk_id, page_number=buffer_page,
            section_title=current_section, element_type="text",
            field_hint=detect_field_hint(text), model_source=model_source,
        ))
        chunk_id += 1
        buffer_words = []

    for elem in elements:
        etype   = elem.get("element_type") or elem.get("type", "text")
        # table elements may have pre-rendered text from table_structure step
        content = (
            elem.get("content_text")
            or elem.get("content")
            or elem.get("text")          # NIM / OCR sometimes stores text here
            or elem.get("value")
            or elem.get("description")
            or ""
        )
        page    = elem.get("page", 0)
        src     = elem.get("element_type", model_source) if elem.get("element_type") == "ocr" else model_source

        if etype in ("title", "heading"):
            flush_buffer()
            current_section = _clean_text(str(content)) if content else ""
            if current_section:
                chunks.append(StructuredChunk(
                    text=current_section, source=source, bank=bank,
                    doc_type="paragraph", chunk_id=chunk_id,
                    page_number=page, element_type="title", model_source=src,
                ))
                chunk_id += 1

        elif etype == "table":
            flush_buffer()
            # content_text already rendered by table_structure step if present
            if isinstance(content, str) and content.strip():
                readable = content
                structured = {"rendered": content}
            else:
                readable, structured = table_to_text(elem.get("content", content))
            if not readable.strip():
                continue
            for sub in _split_into_sentences(readable, table_max_words):
                if not sub.strip():
                    continue
                chunks.append(StructuredChunk(
                    text=sub, source=source, bank=bank,
                    doc_type="table", chunk_id=chunk_id,
                    page_number=page, section_title=current_section,
                    element_type="table", field_hint=detect_field_hint(sub),
                    structured_data=structured, model_source=src,
                ))
                chunk_id += 1

        elif etype == "figure_caption":
            flush_buffer()
            text = _clean_text(str(content))
            if text:
                chunks.append(StructuredChunk(
                    text=text, source=source, bank=bank,
                    doc_type="paragraph", chunk_id=chunk_id,
                    page_number=page, section_title=current_section,
                    element_type="figure_caption", model_source=src,
                ))
                chunk_id += 1

        else:   # text, list_item, ocr
            if content is None:
                continue
            text = _clean_text(str(content))
            if not text:
                # content field is empty — try other string fields in the element
                # (handles cases where NVIDIA returns text in an unexpected field)
                for fallback_key in ("text", "value", "description", "body"):
                    alt = str(elem.get(fallback_key, "") or "").strip()
                    if alt:
                        text = _clean_text(alt)
                        break
            if not text:
                continue
            words = text.split()
            if not buffer_page:
                buffer_page = page
            if len(buffer_words) + len(words) > section_max_words:
                flush_buffer()
                buffer_page = page
            buffer_words.extend(words)

    flush_buffer()
    return chunks


# ---------------------------------------------------------------------------
# pdfplumber fallback (per-page capable)
# ---------------------------------------------------------------------------

def _fallback_extract(
    pdf_path:          Path,
    bank:              str,
    failed_page_indices: list[int] | None = None,
) -> list[StructuredChunk]:
    """
    Plain text extraction using pdfplumber.
    If failed_page_indices is provided, only those pages are extracted
    (used to fill in gaps after partial NVIDIA success).
    If None, the full document is extracted.
    """
    max_words     = _cfg("PDF_SECTION_CHUNK_MAX", 512)
    chunk_overlap = _cfg("CHUNK_OVERLAP", 64)

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = pdf.pages
            if failed_page_indices is not None:
                pages_to_read = [pdf.pages[i] for i in failed_page_indices if i < len(pdf.pages)]
            else:
                pages_to_read = list(pdf.pages)
            full_text = "\n".join(p.extract_text() or "" for p in pages_to_read)
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            if failed_page_indices is not None:
                full_text = "\n".join(
                    reader.pages[i].extract_text() or ""
                    for i in failed_page_indices if i < len(reader.pages)
                )
            else:
                full_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            logger.error("[PDFPipeline] No PDF lib available. Install pdfplumber or pypdf.")
            return []

    words = full_text.split()
    if not words:
        return []

    chunks:   list[StructuredChunk] = []
    chunk_id: int = 0
    start = 0
    while start < len(words):
        end  = min(start + max_words, len(words))
        text = _clean_text(" ".join(words[start:end]))
        if len(text.split()) >= 5:
            chunks.append(StructuredChunk(
                text=text, source=pdf_path.name, bank=bank,
                doc_type="pdf", chunk_id=chunk_id,
                field_hint=detect_field_hint(text),
                model_source="pdfplumber",
            ))
            chunk_id += 1
        if end == len(words):
            break
        start += max_words - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# PDFPipeline — orchestrator
# ---------------------------------------------------------------------------

class PDFPipeline:
    """
    Orchestrates PDF → StructuredChunk list.

    USE_NVIDIA_PDF=true:
        1. File size check  (> 5MB → fallback)
        2. Split into pages
        3. Per-page cascade: page_elements → OCR → table_structure
        4. pdfplumber for any failed pages (partial recovery)
        5. Merge all chunks + re-number

    USE_NVIDIA_PDF=false:
        → pdfplumber only (fast, no API cost)
    """

    def __init__(
        self,
        api_key:       str  | None = None,
        use_nvidia:    bool | None = None,
        processed_dir: Path | None = None,
    ):
        self.api_key       = api_key    or _cfg("NVIDIA_API_KEY", "")
        self.use_nvidia    = use_nvidia if use_nvidia is not None else _cfg("USE_NVIDIA_PDF", False)
        self.processed_dir = processed_dir or Path(_cfg("PROCESSED_DATA_DIR", "processed_data"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._max_retries = int(_cfg("PDF_API_MAX_RETRIES", 3))
        self._retry_delay = float(_cfg("PDF_API_RETRY_DELAY", 2.0))
        self._section_max = int(_cfg("PDF_SECTION_CHUNK_MAX", 400))
        self._table_max   = int(_cfg("PDF_TABLE_CHUNK_MAX",   300))

    def _file_hash(self, p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    def _cache_path(self, p: Path) -> Path:
        return self.processed_dir / (p.stem + ".json")

    def _load_cache(self, p: Path) -> list[StructuredChunk] | None:
        cache = self._cache_path(p)
        if not cache.exists():
            return None
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("file_hash") != self._file_hash(p):
                return None
            return [StructuredChunk(**c) for c in data["chunks"]]
        except Exception as e:
            logger.warning("[PDFPipeline] Cache load error: %s", e)
            return None

    def _save_cache(self, p: Path, chunks: list[StructuredChunk]) -> None:
        data = {
            "file_hash":   self._file_hash(p),
            "source":      p.name,
            "bank":        chunks[0].bank if chunks else "Unknown",
            "chunk_count": len(chunks),
            "chunks":      [c.to_dict() for c in chunks],
        }
        self._cache_path(p).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("[PDFPipeline] Cached %d chunks → %s", len(chunks), p.stem + ".json")

    def process(self, pdf_path: str | Path) -> list[StructuredChunk]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        cached = self._load_cache(pdf_path)
        if cached:
            logger.info("[PDFPipeline] Cache hit: %s (%d chunks)", pdf_path.name, len(cached))
            return cached

        bank   = detect_bank(pdf_path.name)
        chunks = self._process_nvidia(pdf_path, bank) if (self.use_nvidia and self.api_key) else []

        if not chunks:
            if self.use_nvidia and self.api_key:
                logger.info("[PDFPipeline] NVIDIA produced 0 chunks — full pdfplumber fallback")
            chunks = _fallback_extract(pdf_path, bank)

        for i, c in enumerate(chunks):
            c.chunk_id = i

        if chunks:
            self._save_cache(pdf_path, chunks)
        else:
            logger.warning("[PDFPipeline] No chunks extracted from %s", pdf_path.name)

        return chunks

    def _process_nvidia(self, pdf_path: Path, bank: str) -> list[StructuredChunk]:
        """
        Run the multi-model page-level pipeline.
        Returns all chunks (including pdfplumber gap-fill for failed pages).
        Falls back entirely to pdfplumber on FileSizeError or auth errors.
        """
        try:
            raw_pages, failed_indices = _call_page_elements_api(
                pdf_path, self.api_key,
                max_retries=self._max_retries,
                retry_delay=self._retry_delay,
            )
        except FileSizeError as e:
            logger.info("[PDFPipeline] %s — using pdfplumber", e)
            return []
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (401, 403):
                raise RuntimeError(
                    f"NVIDIA API auth failed (HTTP {status}). Check NVIDIA_API_KEY."
                ) from e
            logger.warning("[PDFPipeline] HTTP %d — pdfplumber fallback: %s", status, e)
            return []
        except Exception as e:
            logger.warning("[PDFPipeline] NVIDIA pipeline error — pdfplumber fallback: %s", e)
            return []

        # ── Convert API results to chunks ─────────────────────────────────
        all_chunks: list[StructuredChunk] = []

        for page_dict in raw_pages:
            elements   = page_dict.get("elements", [])
            model_used = page_dict.get("model", "page_elements")

            # Debug: check all possible text fields (NIM varies field names by model version)
            sample_texts = []
            for _e in elements:
                _t = (
                    str(_e.get("content",       "") or "")
                    or str(_e.get("text",        "") or "")
                    or str(_e.get("content_text","") or "")
                ).strip()
                if _t:
                    sample_texts.append(_t)
            sample = sample_texts[0][:120] if sample_texts else "NO TEXT FOUND IN ANY FIELD"
            logger.info(
                "[DEBUG] p%d  elements=%d  model=%s  has_text=%s  sample=%s",
                page_dict.get("page", "?"),
                len(elements),
                model_used,
                bool(sample_texts),
                sample,
            )

            # Improve bank detection from actual page content
            if bank == "Unknown":
                sample = " ".join(
                    str(e.get("content", ""))[:150]
                    for e in elements[:8]
                    if e.get("type") in ("text", "title")
                )
                bank = detect_bank(pdf_path.name, sample)

            page_chunks = _chunk_elements(
                elements, bank, pdf_path.name,
                model_source=model_used,
                section_max_words=self._section_max,
                table_max_words=self._table_max,
            )
            all_chunks.extend(page_chunks)

        # ── Gap-fill: pdfplumber for pages that NVIDIA couldn't process ───
        if failed_indices:
            logger.info(
                "[PDFPipeline] Gap-fill: pdfplumber for %d failed page(s): %s",
                len(failed_indices), [i + 1 for i in failed_indices],
            )
            gap_chunks = _fallback_extract(pdf_path, bank, failed_page_indices=failed_indices)
            all_chunks.extend(gap_chunks)

        # Summary
        model_counts: dict[str, int] = {}
        for c in all_chunks:
            model_counts[c.model_source] = model_counts.get(c.model_source, 0) + 1
        logger.info(
            "[PDFPipeline] %s — %d total chunks  model_sources=%s",
            pdf_path.name, len(all_chunks), model_counts,
        )
        return all_chunks



# ---------------------------------------------------------------------------
# Backward-compat alias (used by tests and external code)
# ---------------------------------------------------------------------------

def _normalise_elements(raw_pages: list[dict]) -> list[dict]:
    """
    Flatten a list of page dicts into a single element list.
    Kept for backward compatibility — new code uses _chunk_elements directly.
    """
    elements: list[dict] = []
    for page_data in raw_pages:
        pn = page_data.get("page", 0)
        for elem in page_data.get("elements", []):
            elements.append({
                "type":    elem.get("type", "text"),
                "content": elem.get("content", ""),
                "page":    pn,
                "bbox":    elem.get("bbox", []),
            })
    return elements

# ---------------------------------------------------------------------------
# Public convenience functions
# ---------------------------------------------------------------------------

def process_pdf(file_path: str) -> list[dict]:
    """Process a single PDF and return list of chunk dicts."""
    return [c.to_dict() for c in PDFPipeline().process(file_path)]


def test_single_pdf(file_path: str, api_key: str | None = None) -> None:
    """
    Smoke-test a single PDF through the full pipeline.

    python -c "from pdf_pipeline import test_single_pdf; \
               test_single_pdf('data/hdfc_pdfs/hdfc_personal_loan.pdf')"
    """
    import os
    key = api_key or os.getenv("NVIDIA_API_KEY", "")

    print(f"[test_single_pdf] File:         {file_path}")
    print(f"[test_single_pdf] API key set:  {bool(key)}")
    print(f"[test_single_pdf] USE_NVIDIA:   {os.getenv('USE_NVIDIA_PDF', 'false')}")
    size_mb = Path(file_path).stat().st_size / 1024 / 1024 if Path(file_path).exists() else 0
    print(f"[test_single_pdf] File size:    {size_mb:.2f} MB (limit 5 MB)")
    print()

    if key and size_mb <= 5:
        print("[test_single_pdf] Converting page 1 to PNG (PyMuPDF)…")
        try:
            images = pdf_to_images(Path(file_path), dpi=_DPI_HIGH)
            print(f"[test_single_pdf] {len(images)} page(s) converted")

            p1_num, p1_png = images[0]
            b64_len = len(_image_to_b64(p1_png))
            print(f"[test_single_pdf] p1 PNG size: {len(p1_png)/1024:.1f} KB  "
                  f"b64: {b64_len/1024:.1f} KB  limit: {_MAX_B64_BYTES/1024:.0f} KB  "
                  f"ok={'YES' if b64_len < _MAX_B64_BYTES else 'COMPRESSED'}")

            result = _process_single_page(
                p1_png, p1_num, Path(file_path).name,
                key, max_retries=1, retry_delay=1.0,
            )
            if result.success:
                type_counts: dict[str, int] = {}
                for e in result.elements:
                    t = e.get("type", "?")
                    type_counts[t] = type_counts.get(t, 0) + 1
                print(f"[test_single_pdf] p1 SUCCESS  model={result.model_used}  types={type_counts}")
            else:
                print("[test_single_pdf] p1 FAILED — both page_elements and OCR exhausted")
        except ImportError:
            print("[test_single_pdf] PyMuPDF not installed. Run: pip install pymupdf")
        except Exception as e:
            print(f"[test_single_pdf] Error during page-level test: {e}")
    else:
        if size_mb > 5:
            print(f"[test_single_pdf] Skipping NVIDIA test — file too large ({size_mb:.1f} MB)")
        else:
            print("[test_single_pdf] Skipping NVIDIA test — NVIDIA_API_KEY not set")

    print()
    print("[test_single_pdf] Running full pipeline…")
    pipeline = PDFPipeline(api_key=key or None)
    chunks   = pipeline.process(file_path)
    print(f"[test_single_pdf] Total chunks: {len(chunks)}")
    if chunks:
        type_counts2: dict[str, int] = {}
        model_counts: dict[str, int] = {}
        for c in chunks:
            type_counts2[c.doc_type]     = type_counts2.get(c.doc_type, 0) + 1
            model_counts[c.model_source] = model_counts.get(c.model_source, 0) + 1
        print(f"[test_single_pdf] Chunk types:    {type_counts2}")
        print(f"[test_single_pdf] Model sources:  {model_counts}")
        print(f"[test_single_pdf] Bank:           {chunks[0].bank}")
        print(f"[test_single_pdf] Sample:")
        print(f"  [{chunks[0].doc_type}|{chunks[0].model_source}] {chunks[0].text[:120]}")
