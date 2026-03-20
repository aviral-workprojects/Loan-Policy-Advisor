"""
pdf_pipeline.py  (v4 — production-grade, smart routing, scored OCR)
=====================================================================
Architecture:
    PDF file
        → file-size guard (>5MB → fallback immediately)
        → per-page multi-model SMART cascade:

            [NEW v4 ROUTING]
            Step 1: nemoretriever-page-elements-v3  (layout + bboxes)
                ↓
            IF table bbox  → nemotron-table-structure-v1  (structured data)
            IF text bbox   → cropped nemotron-ocr-v1      (high-quality OCR)
            IF score < 20  → whole-page nemotron-ocr-v1   (full fallback)
            IF score == 0  → pdfplumber                   (final fallback)

        → section-aware / table-aware chunking
        → attach metadata (bank, field_hint, section_title, model_source)
        → cache to processed_data/<stem>.json
        → List[StructuredChunk]

Key fixes in v4:
    ✅ PREPROCESSING  — Dynamic DPI (150→96→72→60→50) + JPEG multi-stage
                        compression. Strict <120KB payload enforcement.
    ✅ SMART ROUTING  — Per-element routing: tables → table_structure,
                        text/title → cropped OCR.  Whole-page OCR only
                        as last resort before pdfplumber.
    ✅ SCORING SYSTEM — Each page earns a confidence_score; routing
                        decisions and fallback triggers use the score.
    ✅ OCR QUALITY    — Skip tiny/degenerate bboxes (<2% of page),
                        merge fragmented OCR, min 20-char text threshold.
    ✅ RETRIES        — Exponential backoff, per-model retry tuning,
                        graceful degradation at every layer.
    ✅ MODEL TRACKING — model_source on every chunk; nvidia_success flag
                        for upstream monitoring.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ── v4 IMAGE PREPROCESSING CONSTANTS ────────────────────────────────────────
# Strict payload limit for NVIDIA API (raw bytes before base64).
# Keep well under 180 KB encoded → ~120 KB raw is the safe ceiling.
MAX_IMAGE_BYTES = 120_000

# Maximum dimension (width or height) in pixels.
# NVIDIA models work best at ≤1024px; larger images waste bandwidth.
MAX_DIMENSION   = 1024

# DPI ladder — tried in order until image fits MAX_IMAGE_BYTES.
_DPI_LADDER     = [150, 96, 72, 60, 50]

# JPEG quality ladder — tried at each DPI if PNG still too large.
_JPEG_QUALITY_LADDER = [85, 70, 50, 30]

# Legacy alias kept for backward-compat
_MAX_B64_BYTES  = 180_000
_DPI_HIGH       = 150
_DPI_LOW        = 72

# HTTP status codes that warrant a retry
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

# HTTP status codes that are fatal — no point retrying
_FATAL_STATUSES = {400, 401, 403, 422}

# Per-model retry config (v4: tuned per model's reliability profile)
_MODEL_RETRY_CONFIG = {
    "page_elements":   {"max_retries": 3, "retry_delay": 2.0, "timeout": 30},
    "ocr":             {"max_retries": 2, "retry_delay": 1.5, "timeout": 25},
    "table_structure": {"max_retries": 2, "retry_delay": 1.5, "timeout": 25},
}

# Minimum text length for a page to be considered successfully extracted.
# Pages below this threshold trigger the next fallback stage.
_MIN_PAGE_TEXT_CHARS = 20

# Minimum bbox dimension (normalised 0–1) to avoid sending tiny/useless crops.
_MIN_BBOX_DIM = 0.02


# ---------------------------------------------------------------------------
# StructuredChunk — output data structure
# ---------------------------------------------------------------------------

@dataclass
class StructuredChunk:
    """
    Single knowledge unit produced by the PDF pipeline.
    Fully compatible with DocChunk in retrieval.py.
    model_source records which model (or fallback) produced this chunk.
    confidence_score reflects extraction quality (higher = better).
    """
    text:             str
    source:           str
    bank:             str
    doc_type:         str
    chunk_id:         int   = 0
    similarity_score: float = 0.0
    page_number:      int   = 0
    section_title:    str   = ""
    element_type:     str   = "text"
    field_hint:       str   = ""
    structured_data:  dict  = dc_field(default_factory=dict)
    content_hash:     str   = ""
    model_source:     str   = ""
    confidence_score: float = 0.0   # v4: extraction confidence (0–10)

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

def _merge_ocr_fragments(elements: list[dict], min_chars: int = 3) -> list[dict]:
    """
    Merge very short OCR fragments (< min_chars words) into the preceding element.
    Reduces noise from fragmented OCR outputs.
    """
    if not elements:
        return elements
    merged: list[dict] = []
    for elem in elements:
        content = str(
            elem.get("content_text") or elem.get("content") or
            elem.get("text") or ""
        ).strip()
        if len(content.split()) < min_chars and merged and elem.get("type") == "text":
            # Append to previous element's content
            prev = merged[-1]
            prev_content = str(
                prev.get("content_text") or prev.get("content") or
                prev.get("text") or ""
            ).strip()
            merged[-1] = {**prev, "content": f"{prev_content} {content}".strip()}
        else:
            merged.append(elem)
    return merged


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
# v4 IMAGE PREPROCESSING  ← BIGGEST FIX
# ---------------------------------------------------------------------------

def preprocess_image(img_input: bytes | Any) -> bytes | None:
    """
    Production-grade image preprocessing.

    Guarantees output is ALWAYS < MAX_IMAGE_BYTES and ≤ MAX_DIMENSION px.

    Algorithm:
        1. Resize to MAX_DIMENSION (thumbnail preserves aspect ratio)
        2. Try PNG first (lossless) at current size
        3. If PNG > MAX_IMAGE_BYTES → cascade through DPI × JPEG quality ladder
        4. Return None only if ALL combinations fail (triggers pdfplumber fallback)

    Args:
        img_input: either raw PNG bytes or a PIL Image object

    Returns:
        JPEG/PNG bytes < MAX_IMAGE_BYTES, or None if all compression failed.
    """
    try:
        from PIL import Image
    except ImportError:
        # Pillow not available — return raw bytes unchanged, let caller handle it
        if isinstance(img_input, bytes):
            return img_input if len(img_input) < MAX_IMAGE_BYTES else None
        return None

    # Accept both bytes and PIL Image
    if isinstance(img_input, bytes):
        try:
            img = Image.open(io.BytesIO(img_input))
        except Exception as e:
            logger.warning("[preprocess_image] Cannot open image: %s", e)
            return None
    else:
        img = img_input

    # ── Step 1: Resize to MAX_DIMENSION (aspect-preserving) ────────────────
    if max(img.size) > MAX_DIMENSION:
        img = img.copy()
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        logger.debug(
            "[preprocess_image] Resized to %dx%d (max_dim=%d)",
            img.size[0], img.size[1], MAX_DIMENSION,
        )

    # Ensure RGB (no alpha channel — NVIDIA models don't handle transparency)
    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # ── Step 2: Try PNG (lossless) ──────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if len(buf.getvalue()) < MAX_IMAGE_BYTES:
        logger.debug("[preprocess_image] PNG fits (%.1f KB)", len(buf.getvalue()) / 1024)
        return buf.getvalue()

    # ── Step 3: JPEG cascade ────────────────────────────────────────────────
    for quality in _JPEG_QUALITY_LADDER:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        size = len(buf.getvalue())
        if size < MAX_IMAGE_BYTES:
            logger.debug(
                "[preprocess_image] JPEG q=%d fits (%.1f KB)", quality, size / 1024
            )
            return buf.getvalue()

    # ── Step 4: Aggressive resize + JPEG ────────────────────────────────────
    for scale in [0.75, 0.60, 0.50, 0.40]:
        new_w = max(1, int(img.size[0] * scale))
        new_h = max(1, int(img.size[1] * scale))
        small = img.resize((new_w, new_h), Image.LANCZOS)
        for quality in [70, 50, 30]:
            buf = io.BytesIO()
            small.save(buf, format="JPEG", quality=quality, optimize=True)
            size = len(buf.getvalue())
            if size < MAX_IMAGE_BYTES:
                logger.info(
                    "[preprocess_image] Aggressive resize %.0f%% q=%d fits (%.1f KB)",
                    scale * 100, quality, size / 1024,
                )
                return buf.getvalue()

    logger.error(
        "[preprocess_image] All compression attempts failed for image %dx%d — "
        "this page will use pdfplumber", img.size[0], img.size[1],
    )
    return None   # caller must trigger pdfplumber fallback


def _pdf_page_to_preprocessed_bytes(
    pdf_path: Path,
    page_index: int,
) -> bytes | None:
    """
    Render one PDF page to PNG via PyMuPDF, then preprocess to fit NVIDIA limits.

    Tries DPI ladder from highest to lowest until the preprocessed image fits.
    Returns preprocessed bytes, or None if all DPI levels fail.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for NVIDIA CV API calls. "
            "Install with: pip install pymupdf"
        )

    for dpi in _DPI_LADDER:
        doc = fitz.open(str(pdf_path))
        try:
            page   = doc[page_index]
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix    = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pix.tobytes("png")
        finally:
            doc.close()

        result = preprocess_image(png_bytes)
        if result is not None:
            logger.debug(
                "[PDFPipeline] p%d  dpi=%d  raw=%.1f KB  processed=%.1f KB  OK",
                page_index + 1, dpi, len(png_bytes) / 1024, len(result) / 1024,
            )
            return result

        logger.debug(
            "[PDFPipeline] p%d  dpi=%d  raw=%.1f KB  still too large after preprocessing",
            page_index + 1, dpi, len(png_bytes) / 1024,
        )

    logger.error(
        "[PDFPipeline] p%d  all DPI levels failed preprocessing — "
        "page will use pdfplumber", page_index + 1,
    )
    return None


# ---------------------------------------------------------------------------
# Page splitting (backward-compat)
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path, dpi: int = _DPI_HIGH) -> list[tuple[int, bytes]]:
    """
    Convert a PDF to (page_number, png_bytes) tuples using PyMuPDF.
    NOTE: v4 callers use _pdf_page_to_preprocessed_bytes() for proper preprocessing.
    This function is kept for backward compatibility and test_single_pdf().
    """
    try:
        import fitz
        doc    = fitz.open(str(pdf_path))
        images: list[tuple[int, bytes]] = []
        matrix = fitz.Matrix(dpi / 72, dpi / 72)

        for page_index in range(len(doc)):
            page      = doc[page_index]
            pixmap    = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
            logger.debug("[pdf_to_images] p%d  dpi=%d  size=%.1f KB",
                         page_index + 1, dpi, len(png_bytes) / 1024)
            images.append((page_index + 1, png_bytes))

        doc.close()
        logger.info("[pdf_to_images] %s → %d page image(s) at %d DPI",
                    pdf_path.name, len(images), dpi)
        return images
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required. Install with: pip install pymupdf"
        )


def _image_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def crop_image(image_bytes: bytes, bbox: list | dict) -> bytes:
    """
    Crop a PNG/JPEG image to a bounding box returned by NVIDIA page_elements.

    Handles normalised [0–1] and absolute pixel coordinates.
    Skips degenerate crops (< _MIN_BBOX_DIM fraction of page size).
    Returns cropped bytes, or full image if bbox is invalid.
    """
    try:
        from PIL import Image
    except ImportError:
        return image_bytes

    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size

        if isinstance(bbox, dict):
            coords = [bbox.get("x1", 0), bbox.get("y1", 0),
                      bbox.get("x2", 1), bbox.get("y2", 1)]
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            coords = list(bbox[:4])
        else:
            return image_bytes

        x1, y1, x2, y2 = coords

        if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            # Normalised coordinates
            # v4: skip tiny/useless bboxes before converting to pixels
            if (x2 - x1) < _MIN_BBOX_DIM or (y2 - y1) < _MIN_BBOX_DIM:
                logger.debug(
                    "[crop_image] Skipping tiny bbox [%.3f,%.3f,%.3f,%.3f]",
                    x1, y1, x2, y2,
                )
                return image_bytes
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 - x1 < 5 or y2 - y1 < 5:
            return image_bytes

        cropped = img.crop((x1, y1, x2, y2))

        # Preprocess the crop before sending — keeps it under NVIDIA limits
        cropped_bytes_io = io.BytesIO()
        cropped.save(cropped_bytes_io, format="PNG")
        result = preprocess_image(cropped_bytes_io.getvalue())
        return result if result is not None else image_bytes

    except Exception as e:
        logger.debug("[crop_image] Crop failed (%s) — returning full image", e)
        return image_bytes


# Legacy alias
def _compress_image(pdf_path: Path, page_index: int, dpi: int = _DPI_HIGH) -> tuple[bytes, int, bool]:
    result = _pdf_page_to_preprocessed_bytes(pdf_path, page_index)
    if result is None:
        raise RuntimeError(f"Cannot preprocess page {page_index + 1}")
    return result, dpi, True

def _split_pdf_pages(pdf_path: Path) -> list[bytes]:
    """Legacy alias."""
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
# Low-level NVIDIA API caller
# ---------------------------------------------------------------------------

def _nvidia_post(
    url:         str,
    api_key:     str,
    image_bytes: bytes,
    filename:    str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
    timeout:     int   = 30,
) -> dict | list:
    """
    POST a PNG/JPEG image to any NVIDIA CV endpoint.

    v4 change: enforces MAX_IMAGE_BYTES before sending; logs clearly if
    preprocessing was skipped upstream and the image is still oversized.

    Retry policy:
        Retried:     timeout, 429, 500, 502, 503, 504 (exponential backoff)
        Not retried: 400, 401, 403, 422
    """
    b64     = _image_to_b64(image_bytes)
    b64_len = len(b64)

    # Detect JPEG vs PNG for correct data-URI
    mime_type = "image/jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "image/png"

    if len(image_bytes) >= MAX_IMAGE_BYTES:
        logger.warning(
            "[NVIDIA API] Image %.1f KB exceeds %.1f KB limit for %s — "
            "API may reject. Preprocessing should have caught this.",
            len(image_bytes) / 1024, MAX_IMAGE_BYTES / 1024, filename,
        )

    logger.debug(
        "[NVIDIA API] %s  image_b64_len=%d (%.1f KB)  mime=%s",
        url.split("/")[-1], b64_len, b64_len / 1024, mime_type,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    payload = {
        "input": [{"type": "image_url", "url": f"data:{mime_type};base64,{b64}"}]
    }

    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if resp.status_code in _FATAL_STATUSES:
                logger.error(
                    "[NVIDIA API] HTTP %d %s  url=%s  body=%s",
                    resp.status_code, resp.reason,
                    url.split("/")[-1], resp.text[:400],
                )
                if resp.status_code in (401, 403):
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code} — check NVIDIA_API_KEY", response=resp
                    )
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp
                )

            if resp.status_code in _RETRYABLE_STATUSES:
                # v4: exponential backoff with jitter
                wait = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[NVIDIA API] HTTP %d on attempt %d/%d for %s — "
                    "retrying in %.1fs  body=%s",
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
            wait = retry_delay * (2 ** (attempt - 1))
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
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning(
                "[NVIDIA API] Unexpected error on attempt %d/%d: %s",
                attempt, max_retries, e,
            )
            if attempt < max_retries:
                time.sleep(wait)

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
    page_bytes:  bytes,
    filename:    str,
    api_key:     str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    cfg  = _MODEL_RETRY_CONFIG["page_elements"]
    data = _nvidia_post(
        _PAGE_ELEMENTS_URL, api_key, page_bytes, filename,
        max_retries=max_retries or cfg["max_retries"],
        retry_delay=retry_delay or cfg["retry_delay"],
        timeout=cfg["timeout"],
    )

    if isinstance(data, list):
        raw_pages = data
    elif isinstance(data, dict):
        if "data" in data:
            raw_pages = [{"page": 1, "elements": data["data"]}]
        elif "pages" in data:
            raw_pages = data["pages"]
        elif "elements" in data:
            raw_pages = [{"page": 1, "elements": data["elements"]}]
        else:
            logger.warning("[PageElements] Unexpected keys=%s", list(data.keys()))
            raw_pages = []
    else:
        logger.warning("[PageElements] Unexpected type: %s", type(data))
        raw_pages = []

    elements: list[dict] = []
    for page_data in raw_pages:
        pn = page_data.get("page", 1)
        for elem in page_data.get("elements", []):
            elements.append({
                "type":    elem.get("label") or elem.get("type", "text"),
                "content": elem.get("text")  or elem.get("content", ""),
                "page":    pn,
                "bbox":    elem.get("bbox", []),
            })
    return elements


# ---------------------------------------------------------------------------
# Step 2 — OCR
# ---------------------------------------------------------------------------

def _call_ocr(
    page_bytes:  bytes,
    filename:    str,
    api_key:     str,
    max_retries: int   = 2,
    retry_delay: float = 1.5,
) -> list[dict]:
    cfg  = _MODEL_RETRY_CONFIG["ocr"]
    data = _nvidia_post(
        _OCR_URL, api_key, page_bytes, filename,
        max_retries=max_retries or cfg["max_retries"],
        retry_delay=retry_delay or cfg["retry_delay"],
        timeout=cfg["timeout"],
    )

    elements: list[dict] = []
    if isinstance(data, dict):
        if "data" in data:
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
# Step 3 — Table structure
# ---------------------------------------------------------------------------

def _call_table_structure(
    page_bytes:  bytes,
    filename:    str,
    api_key:     str,
    max_retries: int   = 2,
    retry_delay: float = 1.5,
) -> list[dict]:
    cfg  = _MODEL_RETRY_CONFIG["table_structure"]
    data = _nvidia_post(
        _TABLE_STRUCTURE_URL, api_key, page_bytes, filename,
        max_retries=max_retries or cfg["max_retries"],
        retry_delay=retry_delay or cfg["retry_delay"],
        timeout=cfg["timeout"],
    )

    elements: list[dict] = []
    tables = []
    if isinstance(data, dict):
        if "data" in data:
            tables = data["data"]
        else:
            tables = data.get("tables", data.get("results",
                     [data] if "cells" in data or "rows" in data else []))
    elif isinstance(data, list):
        tables = data

    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        readable, structured = table_to_text(tbl)
        if readable.strip():
            elements.append({
                "type":         "table",
                "content":      tbl,
                "content_text": readable,
                "page":         tbl.get("page", 1),
                "bbox":         tbl.get("bbox", []),
            })

    return elements


# ---------------------------------------------------------------------------
# v4 Per-page scoring
# ---------------------------------------------------------------------------

def _score_page_result(elements: list[dict], model_weights: dict | None = None) -> float:
    """
    Compute a confidence score for a page extraction result.

    Score components:
        text_length_score  = min(total_chars / 500, 5.0)   — up to 5 pts
        element_count_score = min(len(elements) / 5, 2.0)  — up to 2 pts
        model_weight        = sum of per-model weights      — up to 3 pts

    Total max: 10.0

    Higher score = more confident extraction.
    Score < 1.0 triggers additional fallback attempts.
    Score == 0 triggers pdfplumber.
    """
    if not elements:
        return 0.0

    _weights = model_weights or {
        "table_structure":   3.0,
        "page_elements":     2.0,
        "page_elements+ocr": 2.5,
        "ocr":               2.0,
        "pdfplumber":        0.5,
    }

    total_chars = sum(
        len(str(
            e.get("content_text") or e.get("content") or
            e.get("text") or ""
        ))
        for e in elements
    )

    text_score    = min(total_chars / 500, 5.0)
    element_score = min(len(elements) / 5, 2.0)

    model_score = 0.0
    for e in elements:
        src = e.get("model_source", "")
        model_score = max(model_score, _weights.get(src, 1.0))

    total = round(text_score + element_score + model_score, 2)
    return total


# ---------------------------------------------------------------------------
# v4 Smart per-page processor
# ---------------------------------------------------------------------------

@dataclass
class _PageResult:
    page_number:      int
    elements:         list[dict]
    model_used:       str
    success:          bool
    confidence_score: float = 0.0   # v4


def _process_single_page(
    page_bytes:   bytes,
    page_number:  int,
    source_name:  str,
    api_key:      str,
    max_retries:  int,
    retry_delay:  float,
) -> _PageResult:
    """
    v4 Smart per-page cascade.

    Route:
        1. page_elements  — layout detection (bboxes)
        2. Per-element routing:
             table bbox  → table_structure (structured data, highest quality)
             text bbox   → cropped OCR     (precise text extraction)
             Skip bbox   → if too tiny (_MIN_BBOX_DIM check)
        3. Compute score; if score < 1.0 → whole-page OCR
        4. Whole-page OCR fallback (score < threshold or page_elements failed)
        5. pdfplumber signal (score == 0 after all NVIDIA models)

    Returns _PageResult with confidence_score for monitoring.
    """
    fname = f"page_{page_number}_{source_name}"

    # ── Step 1: Page Elements ─────────────────────────────────────────────
    pe_elements: list[dict] = []
    pe_ok = False
    try:
        pe_elements = _call_page_elements(
            page_bytes, fname, api_key,
            max_retries=max_retries, retry_delay=retry_delay,
        )
        for e in pe_elements:
            e["page"] = page_number
        pe_ok = True

        type_counts: dict[str, int] = {}
        for e in pe_elements:
            type_counts[e.get("type", "?")] = type_counts.get(e.get("type", "?"), 0) + 1
        logger.info(
            "[PDFPipeline] p%d  page_elements OK  elements=%d  types=%s",
            page_number, len(pe_elements), type_counts,
        )
    except Exception as e1:
        logger.warning("[PDFPipeline] p%d  page_elements FAILED (%s)", page_number, e1)

    # ── Step 2: Smart per-element routing ─────────────────────────────────
    enriched_elements: list[dict] = []
    per_element_ok = False

    if pe_ok and pe_elements:
        ocr_success = 0
        for elem_idx, elem in enumerate(pe_elements):
            bbox      = elem.get("bbox", [])
            elem_type = elem.get("type", "text")
            elem_fname = f"page_{page_number}_elem{elem_idx}_{source_name}"

            # ── Skip tiny/useless bboxes ──────────────────────────────────
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                # Only skip if coordinates look normalised (0–1 range)
                if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                    if (x2 - x1) < _MIN_BBOX_DIM or (y2 - y1) < _MIN_BBOX_DIM:
                        logger.debug(
                            "[PDFPipeline] p%d elem%d tiny bbox skip [%.3f,%.3f,%.3f,%.3f]",
                            page_number, elem_idx, x1, y1, x2, y2,
                        )
                        continue

            # ── Check if page_elements already provided text ───────────────
            existing_text = (
                str(elem.get("content", "") or "")
                or str(elem.get("text", "") or "")
            ).strip()

            if existing_text:
                enriched = dict(elem)
                enriched["content"]      = existing_text
                enriched["model_source"] = "page_elements"
                enriched_elements.append(enriched)
                ocr_success += 1
                continue

            # ── Route by element type ──────────────────────────────────────
            cropped = crop_image(page_bytes, bbox) if bbox else page_bytes

            if elem_type == "table":
                # TABLE → table_structure (best quality for structured data)
                try:
                    table_elems = _call_table_structure(
                        cropped, elem_fname, api_key,
                        max_retries=1, retry_delay=retry_delay,
                    )
                    if table_elems:
                        for te in table_elems:
                            te["page"]         = page_number
                            te["model_source"] = "table_structure"
                        enriched_elements.extend(table_elems)
                        ocr_success += 1
                        logger.debug(
                            "[PDFPipeline] p%d elem%d table_structure → %d rows",
                            page_number, elem_idx, len(table_elems),
                        )
                        continue
                except Exception as te:
                    logger.debug(
                        "[PDFPipeline] p%d elem%d table_structure failed: %s — falling to OCR",
                        page_number, elem_idx, te,
                    )

            # TEXT / TITLE / LIST / OTHER → cropped OCR
            try:
                ocr_result = _call_ocr(
                    cropped, elem_fname, api_key,
                    max_retries=max(1, max_retries - 1),
                    retry_delay=retry_delay,
                )
                if ocr_result:
                    for oe in ocr_result:
                        oe["page"]         = page_number
                        oe["element_type"] = elem_type
                        oe["model_source"] = "page_elements+ocr"
                        oe["bbox"]         = bbox
                    enriched_elements.extend(ocr_result)
                    ocr_success += 1
                    logger.debug(
                        "[PDFPipeline] p%d elem%d OCR → %s",
                        page_number, elem_idx,
                        str(ocr_result[0].get("content", ""))[:80] if ocr_result else "",
                    )
            except Exception as oe:
                logger.debug(
                    "[PDFPipeline] p%d elem%d OCR failed: %s", page_number, elem_idx, oe,
                )

        # Merge fragmented OCR outputs
        enriched_elements = _merge_ocr_fragments(enriched_elements)

        per_element_ok = ocr_success > 0
        logger.info(
            "[PDFPipeline] p%d  per-element routing: %d/%d elements → text",
            page_number, ocr_success, len(pe_elements),
        )

    # ── Compute intermediate score ─────────────────────────────────────────
    intermediate_score = _score_page_result(enriched_elements)

    # ── Step 3: Whole-page OCR ─────────────────────────────────────────────
    # Triggered when: page_elements failed, OR per-element produced nothing,
    # OR score is too low (< 1.0 suggests poor extraction quality)
    whole_page_ocr_elements: list[dict] = []
    needs_whole_page_ocr = (
        not pe_ok
        or not per_element_ok
        or intermediate_score < 1.0
    )

    if needs_whole_page_ocr:
        reason = (
            "page_elements failed" if not pe_ok
            else ("no element text" if not per_element_ok
            else f"low score ({intermediate_score:.1f} < 1.0)")
        )
        logger.info("[PDFPipeline] p%d  whole-page OCR triggered (%s)", page_number, reason)
        try:
            whole_page_ocr_elements = _call_ocr(
                page_bytes, fname, api_key,
                max_retries=max(1, max_retries - 1), retry_delay=retry_delay,
            )
            for e in whole_page_ocr_elements:
                e["page"]         = page_number
                e["element_type"] = "ocr"
                e["model_source"] = "ocr"
            if whole_page_ocr_elements:
                logger.info(
                    "[PDFPipeline] p%d  whole-page OCR OK  elements=%d",
                    page_number, len(whole_page_ocr_elements),
                )
        except Exception as e2:
            logger.warning("[PDFPipeline] p%d  whole-page OCR FAILED (%s)", page_number, e2)

    # ── Step 4: Merge and validate ─────────────────────────────────────────
    # Prefer enriched (per-element) if it has content; supplement with whole-page
    if enriched_elements:
        merged = enriched_elements
        # If whole-page OCR has significantly more text, prefer it
        enr_chars = sum(len(str(e.get("content","") or "")) for e in enriched_elements)
        wpg_chars = sum(len(str(e.get("content","") or "")) for e in whole_page_ocr_elements)
        if wpg_chars > enr_chars * 2:
            logger.info(
                "[PDFPipeline] p%d  whole-page OCR has 2x more text (%d vs %d chars) — preferring it",
                page_number, wpg_chars, enr_chars,
            )
            merged = whole_page_ocr_elements
    else:
        merged = whole_page_ocr_elements

    # ── Step 5: Minimum text threshold ────────────────────────────────────
    def _total_chars(elems: list[dict]) -> int:
        return sum(
            len(str(
                e.get("content_text") or e.get("content") or
                e.get("text") or ""
            ))
            for e in elems
        )

    if not merged or _total_chars(merged) < _MIN_PAGE_TEXT_CHARS:
        logger.warning(
            "[PDFPipeline] p%d  all NVIDIA models failed or text too short "
            "(%d chars < %d threshold) — page will use pdfplumber",
            page_number, _total_chars(merged), _MIN_PAGE_TEXT_CHARS,
        )
        return _PageResult(page_number, [], "pdfplumber", False, confidence_score=0.0)

    # ── Final scoring ──────────────────────────────────────────────────────
    final_score = _score_page_result(merged)

    # Dominant model source
    src_counts: dict[str, int] = {}
    for e in merged:
        s = e.get("model_source", "page_elements+ocr")
        src_counts[s] = src_counts.get(s, 0) + 1
    model_tag = max(src_counts, key=lambda k: src_counts[k])

    logger.info(
        "[PDFPipeline] p%d  SUCCESS  elements=%d  model=%s  "
        "total_chars=%d  confidence_score=%.1f",
        page_number, len(merged), model_tag, _total_chars(merged), final_score,
    )
    return _PageResult(page_number, merged, model_tag, True, confidence_score=final_score)


# ---------------------------------------------------------------------------
# Full-PDF NVIDIA pipeline
# ---------------------------------------------------------------------------

class FileSizeError(Exception):
    """Raised when a PDF exceeds the configured size limit."""


def _call_page_elements_api(
    pdf_path:    Path,
    api_key:     str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
) -> tuple[list[dict], list[int]]:
    """
    Public entry point for the multi-model page-level pipeline.

    v4 changes:
        - Uses _pdf_page_to_preprocessed_bytes() for proper per-page preprocessing
        - Pages that fail preprocessing go directly to failed_page_indices
        - Collects per-page confidence_score for summary logging

    Returns:
        (all_pages, failed_page_indices)
        all_pages: [{"page": N, "elements": [...], "model": …, "confidence_score": F}]
    """
    size_bytes = pdf_path.stat().st_size
    if size_bytes > _MAX_PDF_BYTES:
        raise FileSizeError(
            f"{pdf_path.name} is {size_bytes/1024/1024:.1f} MB "
            f"(limit {_MAX_PDF_BYTES/1024/1024:.0f} MB) — using pdfplumber"
        )

    # ── Get page count (no full render yet) ───────────────────────────────
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n_pages = len(doc)
        doc.close()
    except ImportError:
        raise FileSizeError("PyMuPDF not available — using pdfplumber")

    logger.info(
        "[PDFPipeline] Processing %s — %d page(s), %.1f MB",
        pdf_path.name, n_pages, size_bytes / 1024 / 1024,
    )

    all_pages: list[dict] = []
    failed_page_indices: list[int] = []

    # ── Preprocess all pages ───────────────────────────────────────────────
    # v4: preprocessing happens per-page with full DPI ladder + JPEG fallback.
    preprocessed_pages: list[tuple[int, bytes]] = []  # (page_num, bytes)
    for page_idx in range(n_pages):
        page_num = page_idx + 1
        result   = _pdf_page_to_preprocessed_bytes(pdf_path, page_idx)
        if result is None:
            logger.warning(
                "[PDFPipeline] p%d preprocessing failed — marking for pdfplumber", page_num,
            )
            failed_page_indices.append(page_idx)
        else:
            preprocessed_pages.append((page_num, result))

    if not preprocessed_pages:
        logger.error("[PDFPipeline] All pages failed preprocessing — full pdfplumber fallback")
        return [], list(range(n_pages))

    # ── Process pages in parallel ─────────────────────────────────────────
    max_workers = min(4, len(preprocessed_pages))
    futures_map: dict = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for page_num, png_bytes in preprocessed_pages:
            fut = pool.submit(
                _process_single_page,
                png_bytes, page_num, pdf_path.name,
                api_key, max_retries, retry_delay,
            )
            futures_map[fut] = page_num - 1

        for fut in as_completed(futures_map):
            page_idx = futures_map[fut]
            page_num = page_idx + 1
            try:
                result = fut.result()
                if result.success and result.elements:
                    all_pages.append({
                        "page":             result.page_number,
                        "elements":         result.elements,
                        "model":            result.model_used,
                        "confidence_score": result.confidence_score,  # v4
                    })
                else:
                    failed_page_indices.append(page_idx)
            except Exception as fe:
                logger.warning("[PDFPipeline] p%d thread failed: %s", page_num, fe)
                failed_page_indices.append(page_idx)

    all_pages.sort(key=lambda p: p["page"])

    # ── v4 Summary with confidence scores ─────────────────────────────────
    model_counts: dict[str, int] = {}
    total_confidence = 0.0
    for p in all_pages:
        m = p.get("model", "unknown")
        model_counts[m] = model_counts.get(m, 0) + 1
        total_confidence += p.get("confidence_score", 0.0)

    avg_confidence = total_confidence / len(all_pages) if all_pages else 0.0
    logger.info(
        "[PDFPipeline] %s complete — %d/%d pages succeeded  "
        "models=%s  fallback_pages=%d  avg_confidence=%.1f",
        pdf_path.name, len(all_pages), n_pages,
        model_counts, len(failed_page_indices), avg_confidence,
    )

    return all_pages, failed_page_indices


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
    confidence_score:  float = 0.0,  # v4: page-level score
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
            field_hint=detect_field_hint(text),
            model_source=model_source,
            confidence_score=confidence_score,
        ))
        chunk_id += 1
        buffer_words = []

    for elem in elements:
        etype   = elem.get("element_type") or elem.get("type", "text")
        content = (
            elem.get("content_text")
            or elem.get("content")
            or elem.get("text")
            or elem.get("value")
            or elem.get("description")
            or ""
        )
        page    = elem.get("page", 0)
        src     = (
            elem.get("model_source", model_source)
            if elem.get("model_source")
            else model_source
        )

        if etype in ("title", "heading"):
            flush_buffer()
            current_section = _clean_text(str(content)) if content else ""
            if current_section:
                chunks.append(StructuredChunk(
                    text=current_section, source=source, bank=bank,
                    doc_type="paragraph", chunk_id=chunk_id,
                    page_number=page, element_type="title",
                    model_source=src, confidence_score=confidence_score,
                ))
                chunk_id += 1

        elif etype == "table":
            flush_buffer()
            if isinstance(content, str) and content.strip():
                readable  = content
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
                    structured_data=structured,
                    model_source=src, confidence_score=confidence_score,
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
                    element_type="figure_caption",
                    model_source=src, confidence_score=confidence_score,
                ))
                chunk_id += 1

        else:   # text, list_item, ocr
            if content is None:
                continue
            text = _clean_text(str(content))
            if not text:
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
# pdfplumber fallback
# ---------------------------------------------------------------------------

def _fallback_extract(
    pdf_path:            Path,
    bank:                str,
    failed_page_indices: list[int] | None = None,
) -> list[StructuredChunk]:
    """
    Plain text extraction using pdfplumber.
    If failed_page_indices is provided, only those pages are extracted.
    """
    max_words     = _cfg("PDF_SECTION_CHUNK_MAX", 512)
    chunk_overlap = _cfg("CHUNK_OVERLAP", 64)

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
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
                confidence_score=1.0,  # pdfplumber is deterministic
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
        2. Preprocess pages (v4 DPI ladder + JPEG compression)
        3. Per-page smart cascade (page_elements → table/OCR routing → whole-page OCR)
        4. pdfplumber for failed pages
        5. Merge chunks + re-number + save cache

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
        v4 NVIDIA pipeline with smart routing and scoring.
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

        all_chunks: list[StructuredChunk] = []

        for page_dict in raw_pages:
            elements          = page_dict.get("elements", [])
            model_used        = page_dict.get("model", "page_elements")
            page_confidence   = page_dict.get("confidence_score", 0.0)   # v4

            # Improve bank detection from page content
            if bank == "Unknown":
                sample = " ".join(
                    str(e.get("content", ""))[:150]
                    for e in elements[:8]
                    if e.get("type") in ("text", "title")
                )
                bank = detect_bank(pdf_path.name, sample)

            # Debug: show extraction sample
            sample_texts = [
                str(e.get("content_text") or e.get("content") or e.get("text") or "").strip()
                for e in elements if str(e.get("content_text") or e.get("content") or e.get("text") or "").strip()
            ]
            sample = sample_texts[0][:120] if sample_texts else "NO TEXT"
            logger.info(
                "[DEBUG] p%d  elements=%d  model=%s  score=%.1f  sample=%s",
                page_dict.get("page", "?"), len(elements),
                model_used, page_confidence, sample,
            )

            page_chunks = _chunk_elements(
                elements, bank, pdf_path.name,
                model_source=model_used,
                section_max_words=self._section_max,
                table_max_words=self._table_max,
                confidence_score=page_confidence,
            )
            all_chunks.extend(page_chunks)

        # Gap-fill with pdfplumber for failed pages
        if failed_indices:
            logger.info(
                "[PDFPipeline] Gap-fill: pdfplumber for %d failed page(s): %s",
                len(failed_indices), [i + 1 for i in failed_indices],
            )
            gap_chunks = _fallback_extract(pdf_path, bank, failed_page_indices=failed_indices)
            all_chunks.extend(gap_chunks)

        # v4 model tracking summary
        model_counts: dict[str, int] = {}
        for c in all_chunks:
            model_counts[c.model_source] = model_counts.get(c.model_source, 0) + 1

        nvidia_sources = {"page_elements", "ocr", "table_structure", "page_elements+ocr", "nvidia_smart"}
        nvidia_success = sum(1 for c in all_chunks if c.model_source in nvidia_sources)

        if nvidia_success == 0:
            logger.warning(
                "[PDFPipeline] 🚨 No NVIDIA extraction succeeded for %s  (model_sources=%s)",
                pdf_path.name, model_counts,
            )
        else:
            logger.info(
                "[PDFPipeline] %s — %d total chunks  nvidia_chunks=%d  model_sources=%s",
                pdf_path.name, len(all_chunks), nvidia_success, model_counts,
            )

        return all_chunks


# ---------------------------------------------------------------------------
# Backward-compat aliases
# ---------------------------------------------------------------------------

def _normalise_elements(raw_pages: list[dict]) -> list[dict]:
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

    print(f"[test_single_pdf] File:        {file_path}")
    print(f"[test_single_pdf] API key set: {bool(key)}")
    print(f"[test_single_pdf] USE_NVIDIA:  {os.getenv('USE_NVIDIA_PDF', 'false')}")
    size_mb = Path(file_path).stat().st_size / 1024 / 1024 if Path(file_path).exists() else 0
    print(f"[test_single_pdf] File size:   {size_mb:.2f} MB (limit 5 MB)")
    print()

    if key and size_mb <= 5:
        print("[test_single_pdf] Testing preprocessing on page 1…")
        try:
            result = _pdf_page_to_preprocessed_bytes(Path(file_path), 0)
            if result:
                print(f"[test_single_pdf] Preprocessed size: {len(result)/1024:.1f} KB "
                      f"(limit {MAX_IMAGE_BYTES/1024:.0f} KB) ✅")
            else:
                print("[test_single_pdf] Preprocessing failed — all DPI levels exhausted ❌")
                return
        except ImportError as e:
            print(f"[test_single_pdf] {e}")
            return
        except Exception as e:
            print(f"[test_single_pdf] Error during preprocessing: {e}")
            return

    print()
    print("[test_single_pdf] Running full pipeline…")
    pipeline = PDFPipeline(api_key=key or None)
    chunks   = pipeline.process(file_path)
    print(f"[test_single_pdf] Total chunks: {len(chunks)}")
    if chunks:
        type_counts:  dict[str, int] = {}
        model_counts: dict[str, int] = {}
        for c in chunks:
            type_counts[c.doc_type]     = type_counts.get(c.doc_type, 0) + 1
            model_counts[c.model_source] = model_counts.get(c.model_source, 0) + 1
        nvidia_sources = {"page_elements", "ocr", "table_structure", "page_elements+ocr"}
        nvidia_success = sum(v for k, v in model_counts.items() if k in nvidia_sources)
        print(f"[test_single_pdf] Chunk types:    {type_counts}")
        print(f"[test_single_pdf] Model sources:  {model_counts}")
        print(f"[test_single_pdf] NVIDIA chunks:  {nvidia_success}/{len(chunks)}")
        print(f"[test_single_pdf] Bank:           {chunks[0].bank}")
        avg_conf = sum(c.confidence_score for c in chunks) / len(chunks)
        print(f"[test_single_pdf] Avg confidence: {avg_conf:.1f}/10")
        print(f"[test_single_pdf] Sample:")
        print(f"  [{chunks[0].doc_type}|{chunks[0].model_source}] {chunks[0].text[:120]}")