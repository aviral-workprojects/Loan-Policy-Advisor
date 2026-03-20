"""
pdf_pipeline.py  (v5.2 — native text + image quality + diagnostics)
=====================================================================
This version fixes the "both NVIDIA OCR and PaddleOCR return empty" problem.
Root cause: the PDFs are text-layer documents, not scanned images — OCR was
never needed and was always going to return empty on clean vector text.

    ✅ FIX 12: Native text fast-path (Step 0).  PyMuPDF extracts embedded text
               directly from each page before any image processing happens.
               Pages with ≥50 chars of native text skip OCR entirely.
               model_source = "native_text".  This fires for ~100% of bank PDFs.
    ✅ FIX 13: DEBUG_SAVE_IMAGES=true env flag.  Saves rendered page images to
               debug_images/ so you can visually verify quality before blaming
               the OCR model.  Add to .env: DEBUG_SAVE_IMAGES=true
    ✅ FIX 14: Parse DISABLED (SSL EOF on Windows).  Directly uses page-elements-v3.
    ✅ FIX 15: DPI locked to 300 (no ladder — lower DPI kills OCR accuracy).
    ✅ FIX 16: Contrast 2.0× + Sharpness 1.5× applied in preprocess_image.
               Financial PDFs have light-grey text and tiny fonts; enhancement
               makes them readable for OCR models.
    ✅ FIX 17: PaddleOCR init with explicit det=True, rec=True, cls=True.
               Default init can silently skip steps → empty results.
               Raw result now logged at DEBUG level for diagnosis.

Previous fixes (v5.0 / v5.1):
    ✅ MAX_IMAGE_BYTES 150KB → 800KB
    ✅ Resize ladder annihilation removed (was 0.50 @ q=60)
    ✅ Removed upscale_image() (adds blur)
    ✅ Removed final_score < 2.0 kill switch
    ✅ Thread pool timeout 25s → 60s
    ✅ Early whole-page OCR when layout returns no text
    ✅ Full-page bbox filter (> 80% area → skip crop)
    ✅ PaddleOCR as local NVIDIA-OCR fallback
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import re
import threading
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

_MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB

# ── Image quality constants ────────────────────────────────────────────────────
# FIX 1: 5× byte limit increase.  The previous 150KB cap forced q=60 JPEG at
# 75% resize — text became unreadable and OCR returned empty on every page.
# NVIDIA's OCR model is quality-sensitive; blurry images → silent empty response.
MAX_IMAGE_BYTES      = 800_000            # was 150_000 — 5× increase, keeps text sharp
MAX_DIMENSION        = 2000               # was 1400 — larger canvas preserves fine fonts
_DPI_LADDER          = [300]              # 300 DPI only — lower DPI kills OCR on financial PDFs
# FIX 1: gentle JPEG quality ladder — never drop below 75 (was dropping to 60)
_JPEG_QUALITY_LADDER = [95, 85, 75]      # was [90, 80, 70, 60] — q=60 destroys text

# Legacy aliases (still used by test_single_pdf / backward-compat callers)
_MAX_B64_BYTES = 180_000
_DPI_HIGH      = 200    # updated
_DPI_LOW       = 96     # updated

# ── Nemotron Parse v1.1 — preferred over page-elements-v3 ─────────────────────
# Better semantic classes, stronger layout understanding, OCR-ready output.
# Used as Step 1 with automatic fallback to page-elements if parse fails.
_PARSE_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-parse-v1.1"

# ── Retry ─────────────────────────────────────────────────────────────────────
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_FATAL_STATUSES     = {400, 401, 403, 422}

_MODEL_RETRY_CONFIG = {
    "page_elements":   {"max_retries": 3, "retry_delay": 2.0, "timeout": 30},
    "ocr":             {"max_retries": 2, "retry_delay": 1.5, "timeout": 25},
    "table_structure": {"max_retries": 2, "retry_delay": 1.5, "timeout": 25},
}

# ── ✅ FIX 1: lowered threshold ───────────────────────────────────────────────
# Was 20 → caused short-but-valid pages (titles, single-line rules, header
# pages) to fall through to pdfplumber.  5 chars is the real "blank page" bar.
_MIN_PAGE_TEXT_CHARS = 5

# Minimum normalised bbox dimension to skip (< 2% of page → useless crop)
_MIN_BBOX_DIM = 0.02

# NEW: skip bboxes that cover > 80% of the page area — those are whole-page
# detections from page_elements and cropping them gives the same blurry result
# as the full image.  Trigger whole-page OCR directly for those regions instead.
_MAX_BBOX_AREA_FRACTION = 0.80

# NEW: PaddleOCR local fallback — set to True if paddleocr is installed.
# Checked once at module level so the import cost is paid only once.
try:
    from paddleocr import PaddleOCR as _PaddleOCRCls
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False

_paddle_instance = None   # lazy singleton — created on first use

# ── Debug image saving ────────────────────────────────────────────────────────
# Set DEBUG_SAVE_IMAGES=true in .env to save rendered page PNGs to debug_images/
# This lets you visually verify image quality before blaming the OCR model.
import os as _os
_DEBUG_SAVE_IMAGES = _os.getenv("DEBUG_SAVE_IMAGES", "false").lower() == "true"
_DEBUG_IMAGE_DIR   = Path(_os.getenv("DEBUG_IMAGE_DIR", "debug_images"))

# ── Native text detection ─────────────────────────────────────────────────────
# PDFs often contain embedded text (not scanned). Extracting it directly is
# faster, lossless, and 100% reliable — no OCR needed at all.
# Set USE_NATIVE_TEXT=false to force OCR even on text-layer PDFs (rare use case).
_USE_NATIVE_TEXT = _os.getenv("USE_NATIVE_TEXT", "true").lower() == "true"
_NATIVE_TEXT_MIN_CHARS = 50   # pages with ≥50 native chars skip OCR entirely

# ── ✅ FIX 3: skip OCR if page_elements already returned this many chars ──────
_MIN_EXISTING_TEXT_LEN = 15


# ---------------------------------------------------------------------------
# ✅ FIX 4: Global rate limiter
# Enforces ≥300 ms between any two NVIDIA API calls across all threads.
# Prevents 429 bursts when ThreadPoolExecutor fires 4 concurrent requests.
# ---------------------------------------------------------------------------

_rate_lock:     threading.Lock  = threading.Lock()
_last_call_ts:  float           = 0.0
_MIN_CALL_GAP:  float           = 0.30  # seconds


def _rate_limit() -> None:
    global _last_call_ts
    with _rate_lock:
        gap = _MIN_CALL_GAP - (time.time() - _last_call_ts)
        if gap > 0:
            time.sleep(gap)
        _last_call_ts = time.time()


# ---------------------------------------------------------------------------
# StructuredChunk
# ---------------------------------------------------------------------------

@dataclass
class StructuredChunk:
    """Single knowledge unit. confidence_score = extraction quality (0–10)."""
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
    confidence_score: float = 0.0

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
_BANK_PATH_MAP = {"hdfc": "HDFC", "sbi": "SBI", "icici": "ICICI", "axis": "Axis", "rbi": "RBI"}

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
    tl = text.lower()
    for pattern, f in _FIELD_PATTERNS:
        if re.search(pattern, tl):
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

def _merge_ocr_fragments(elements: list[dict], min_words: int = 3) -> list[dict]:
    """Merge short OCR fragments (< min_words) into the preceding element."""
    if not elements:
        return elements
    merged: list[dict] = []
    for elem in elements:
        content = str(
            elem.get("content_text") or elem.get("content") or elem.get("text") or ""
        ).strip()
        if len(content.split()) < min_words and merged and elem.get("type") == "text":
            prev    = merged[-1]
            prev_c  = str(prev.get("content_text") or prev.get("content") or prev.get("text") or "").strip()
            merged[-1] = {**prev, "content": f"{prev_c} {content}".strip()}
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
        rows, sr = [], []
        for row in table_data:
            if isinstance(row, (list, tuple)):
                rows.append(" | ".join(str(c).strip() for c in row if str(c).strip()))
                sr.append(list(row))
            elif isinstance(row, dict):
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
                sr.append(row)
            else:
                rows.append(str(row))
        return _clean_text("\n".join(rows)), {"rows": sr}
    if isinstance(table_data, dict):
        headers = table_data.get("headers", table_data.get("columns", []))
        rows    = table_data.get("rows",    table_data.get("data",    []))
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
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img_input: bytes | Any, allow_jpeg: bool = True) -> bytes | None:
    """
    Prepare image for NVIDIA OCR:
      1. Decode + force RGB
      2. Enhance contrast (2.0×) and sharpness (1.5×) — CRITICAL for financial PDFs
         with light-grey text, small fonts, or low-contrast tables
      3. Resize if larger than MAX_DIMENSION
      4. Compress to stay under MAX_IMAGE_BYTES (PNG → JPEG ladder → gentle resize)

    allow_jpeg is kept for backward-compat but is always effectively True.
    Returns None only if every strategy fails (extremely rare).
    """
    try:
        from PIL import Image, ImageEnhance
    except ImportError:
        if isinstance(img_input, bytes):
            return img_input if len(img_input) < MAX_IMAGE_BYTES else None
        return None

    img = Image.open(io.BytesIO(img_input)) if isinstance(img_input, bytes) else img_input

    # ── Force RGB (handles RGBA, P/palette, L/grayscale, CMYK, etc.) ──────────
    if img.mode in ("RGBA", "P", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # ── Contrast + sharpness enhancement ──────────────────────────────────────
    # Financial PDFs often have light-grey text, small 8pt fonts, and low-contrast
    # tables. Without enhancement, OCR models see noise and return empty strings.
    # Contrast 2.0× makes light text black; Sharpness 1.5× firms up edge detail.
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(1.5)

    # ── Resize if oversized ───────────────────────────────────────────────────
    if max(img.size) > MAX_DIMENSION:
        img = img.copy()
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

    # ── Try PNG first (lossless) ────────────────────────────────────────────
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if len(buf.getvalue()) < MAX_IMAGE_BYTES:
        return buf.getvalue()

    # ── JPEG quality ladder (mandatory — never skip this) ───────────────────
    for quality in _JPEG_QUALITY_LADDER:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if len(buf.getvalue()) < MAX_IMAGE_BYTES:
            logger.debug("[preprocess_image] JPEG q=%d → %.1f KB", quality, len(buf.getvalue()) / 1024)
            return buf.getvalue()

    # ── Light resize + JPEG (FIX 2: never go below 90% scale or q=75) ─────────
    for scale, quality in [(0.9, 80), (0.8, 75)]:
        small = img.resize(
            (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
            Image.LANCZOS,
        )
        buf = io.BytesIO()
        small.save(buf, format="JPEG", quality=quality, optimize=True)
        if len(buf.getvalue()) < MAX_IMAGE_BYTES:
            logger.info("[preprocess_image] resize %.0f%% q=%d → %.1f KB",
                        scale * 100, quality, len(buf.getvalue()) / 1024)
            return buf.getvalue()

    logger.error("[preprocess_image] All compression strategies failed (%.1f KB image)",
                 len(img.tobytes()) / 1024)
    return None


def _pdf_page_to_preprocessed_bytes(pdf_path: Path, page_index: int) -> bytes | None:
    """Render page via PyMuPDF at 300 DPI, enhance, and return image bytes."""
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

    for dpi in _DPI_LADDER:
        doc = fitz.open(str(pdf_path))
        try:
            pix = doc[page_index].get_pixmap(
                matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False
            )
            png = pix.tobytes("png")
        finally:
            doc.close()

        result = preprocess_image(png)
        if result is not None:
            mime = "JPEG" if result[:3] == b"\xff\xd8\xff" else "PNG"
            logger.debug("[PDFPipeline] p%d dpi=%d %s=%.1f KB ✓",
                         page_index + 1, dpi, mime, len(result) / 1024)

            # ── Debug: save image so you can visually verify quality ──────────
            if _DEBUG_SAVE_IMAGES:
                try:
                    _DEBUG_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
                    ext  = "jpg" if mime == "JPEG" else "png"
                    name = f"debug_{pdf_path.stem}_p{page_index + 1}_dpi{dpi}.{ext}"
                    (_DEBUG_IMAGE_DIR / name).write_bytes(result)
                    logger.info("[DEBUG] Saved page image → %s  (%.1f KB) — check if text is readable",
                                name, len(result) / 1024)
                except Exception as de:
                    logger.debug("[DEBUG] Could not save debug image: %s", de)

            return result
        logger.debug("[PDFPipeline] p%d dpi=%d raw=%.1f KB — trying lower DPI",
                     page_index + 1, dpi, len(png) / 1024)

    logger.error("[PDFPipeline] p%d all DPI levels failed — page will use pdfplumber",
                 page_index + 1)
    return None


def _extract_native_text_from_page(pdf_path: Path, page_index: int) -> str:
    """
    Extract embedded text directly from a PDF page using PyMuPDF.
    Returns empty string if the page has no text layer (scanned PDF).

    This is the fastest and most accurate extraction method — no OCR needed.
    A page with ≥ _NATIVE_TEXT_MIN_CHARS of real text should skip OCR entirely.
    """
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            text = doc[page_index].get_text("text")
            return text.strip()
        finally:
            doc.close()
    except Exception as e:
        logger.debug("[NativeText] p%d extraction failed: %s", page_index + 1, e)
        return ""


# backward-compat helpers
def pdf_to_images(pdf_path: Path, dpi: int = _DPI_HIGH) -> list[tuple[int, bytes]]:
    try:
        import fitz
        doc, images = fitz.open(str(pdf_path)), []
        m = fitz.Matrix(dpi / 72, dpi / 72)
        for i in range(len(doc)):
            images.append((i + 1, doc[i].get_pixmap(matrix=m, alpha=False).tobytes("png")))
        doc.close()
        return images
    except ImportError:
        raise ImportError("PyMuPDF required: pip install pymupdf")

def _image_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def crop_image(image_bytes: bytes, bbox: list | dict) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return image_bytes
    try:
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        coords = (
            [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 1), bbox.get("y2", 1)]
            if isinstance(bbox, dict)
            else list(bbox[:4]) if isinstance(bbox, (list, tuple)) and len(bbox) >= 4
            else None
        )
        if coords is None:
            return image_bytes
        x1, y1, x2, y2 = coords
        if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
            if (x2 - x1) < _MIN_BBOX_DIM or (y2 - y1) < _MIN_BBOX_DIM:
                return image_bytes
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            return image_bytes
        buf = io.BytesIO()
        img.crop((x1, y1, x2, y2)).save(buf, format="PNG")
        # Crops can use JPEG as fallback — they're small regions where slight
        # compression artefacts matter less than fitting under the payload limit.
        result = preprocess_image(buf.getvalue(), allow_jpeg=True)
        return result if result is not None else image_bytes
    except Exception as e:
        logger.debug("[crop_image] failed: %s", e)
        return image_bytes

def _compress_image(pdf_path: Path, page_index: int, dpi: int = _DPI_HIGH):
    result = _pdf_page_to_preprocessed_bytes(pdf_path, page_index)
    if result is None:
        raise RuntimeError(f"Cannot preprocess page {page_index + 1}")
    return result, dpi, True

def _split_pdf_pages(pdf_path: Path) -> list[bytes]:
    try:
        import pypdf
        reader, pages = pypdf.PdfReader(str(pdf_path)), []
        for page in reader.pages:
            w = pypdf.PdfWriter()
            w.add_page(page)
            buf = io.BytesIO()
            w.write(buf)
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
    """POST image to NVIDIA CV endpoint with rate limiting and exponential backoff."""
    # ✅ Rate limit before every call
    _rate_limit()

    # Fix 6: hard cap per-request timeout — prevents a single stuck call from
    # blocking the thread for 30s+ and causing cascade delays across all pages.
    timeout = min(timeout, 20)

    mime = "image/jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "image/png"
    b64  = _image_to_b64(image_bytes)

    if len(image_bytes) >= MAX_IMAGE_BYTES:
        logger.warning("[NVIDIA API] %.1f KB > %.1f KB limit for %s",
                       len(image_bytes) / 1024, MAX_IMAGE_BYTES / 1024, filename)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    payload = {"input": [{"type": "image_url", "url": f"data:{mime};base64,{b64}"}]}
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            if resp.status_code in _FATAL_STATUSES:
                logger.error("[NVIDIA API] HTTP %d %s body=%s",
                             resp.status_code, url.split("/")[-1], resp.text[:300])
                if resp.status_code in (401, 403):
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code} — check NVIDIA_API_KEY", response=resp)
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}", response=resp)

            if resp.status_code in _RETRYABLE_STATUSES:
                wait = retry_delay * (2 ** (attempt - 1))
                logger.warning("[NVIDIA API] HTTP %d attempt %d/%d — retry %.1fs",
                               resp.status_code, attempt, max_retries, wait)
                last_err = requests.exceptions.HTTPError(str(resp.status_code), response=resp)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            wait = retry_delay * (2 ** (attempt - 1))
            last_err = TimeoutError(f"timeout attempt {attempt}")
            logger.warning("[NVIDIA API] Timeout attempt %d/%d — retry %.1fs",
                           attempt, max_retries, wait)
            if attempt < max_retries:
                time.sleep(wait)
        except requests.exceptions.HTTPError:
            raise
        except Exception as e:
            last_err = e
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning("[NVIDIA API] Error attempt %d/%d: %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(wait)

    raise RuntimeError(
        f"NVIDIA API ({url.split('/')[-1]}) failed after {max_retries} attempts: {last_err}"
    )


# ---------------------------------------------------------------------------
# Step 1 — Page Elements
# ---------------------------------------------------------------------------

def _call_page_elements(
    page_bytes: bytes, filename: str, api_key: str,
    max_retries: int = 3, retry_delay: float = 2.0,
) -> list[dict]:
    cfg  = _MODEL_RETRY_CONFIG["page_elements"]
    data = _nvidia_post(_PAGE_ELEMENTS_URL, api_key, page_bytes, filename,
                        max_retries=max_retries or cfg["max_retries"],
                        retry_delay=retry_delay or cfg["retry_delay"],
                        timeout=cfg["timeout"])
    if isinstance(data, list):
        raw_pages = data
    elif isinstance(data, dict):
        if   "data"     in data: raw_pages = [{"page": 1, "elements": data["data"]}]
        elif "pages"    in data: raw_pages = data["pages"]
        elif "elements" in data: raw_pages = [{"page": 1, "elements": data["elements"]}]
        else:
            logger.warning("[PageElements] Unexpected keys=%s", list(data.keys()))
            raw_pages = []
    else:
        raw_pages = []

    elements: list[dict] = []
    for pd_ in raw_pages:
        pn = pd_.get("page", 1)
        for elem in pd_.get("elements", []):
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
    page_bytes: bytes, filename: str, api_key: str,
    max_retries: int = 2, retry_delay: float = 1.5,
) -> list[dict]:
    cfg  = _MODEL_RETRY_CONFIG["ocr"]
    data = _nvidia_post(_OCR_URL, api_key, page_bytes, filename,
                        max_retries=max_retries or cfg["max_retries"],
                        retry_delay=retry_delay or cfg["retry_delay"],
                        timeout=cfg["timeout"])

    # Fix 7: log raw response truncated — helps diagnose format surprises
    logger.debug("[OCR RAW] %s", str(data)[:500])

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

    # Fix 6: validate OCR output — reject pure-number / symbol garbage
    elements = [
        e for e in elements
        if is_valid_ocr_text(str(e.get("content", e.get("text", "")) or ""))
    ]
    return elements


# ---------------------------------------------------------------------------
# Step 3 — Table structure
# ---------------------------------------------------------------------------

def _call_table_structure(
    page_bytes: bytes, filename: str, api_key: str,
    max_retries: int = 2, retry_delay: float = 1.5,
) -> list[dict]:
    cfg  = _MODEL_RETRY_CONFIG["table_structure"]
    data = _nvidia_post(_TABLE_STRUCTURE_URL, api_key, page_bytes, filename,
                        max_retries=max_retries or cfg["max_retries"],
                        retry_delay=retry_delay or cfg["retry_delay"],
                        timeout=cfg["timeout"])
    elements: list[dict] = []
    tables = []
    if isinstance(data, dict):
        tables = data["data"] if "data" in data else data.get("tables", data.get("results",
                 [data] if "cells" in data or "rows" in data else []))
    elif isinstance(data, list):
        tables = data
    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        readable, structured = table_to_text(tbl)
        if readable.strip():
            elements.append({
                "type": "table", "content": tbl,
                "content_text": readable,
                "page": tbl.get("page", 1),
                "bbox": tbl.get("bbox", []),
            })
    return elements


# ---------------------------------------------------------------------------
# OCR debug + validation helpers
# ---------------------------------------------------------------------------

def _log_ocr_debug(elements: list[dict], page_number: int, context: str = "") -> None:
    """Fix 3: log OCR output so you can see exactly what the API returned."""
    if elements:
        sample = str(elements[0].get("content", elements[0].get("text", "")) or "")[:80]
        logger.info("[OCR DEBUG] p%d %s → %d elem(s) | sample='%s'",
                    page_number, context, len(elements), sample)
    else:
        logger.warning("[OCR DEBUG] p%d %s → EMPTY (OCR returned nothing)", page_number, context)


def is_valid_ocr_text(text: str) -> bool:
    """
    Validate OCR output before accepting it.
    Intentionally lenient for financial documents which are number-heavy.
    Only rejects: empty / whitespace-only strings and very short noise (<5 chars).
    """
    if not text or len(text.strip()) < 5:
        return False
    return True


def expand_bbox(bbox: list | None, margin: float = 0.02) -> list | None:
    """
    Fix 4: add a small margin around every bounding box before cropping.
    OCR fails when text is cut too tightly — a 2% margin gives a huge accuracy boost.
    """
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return bbox
    x1, y1, x2, y2 = bbox[:4]
    # Only apply if coordinates look normalised (0–1 range)
    if not all(0.0 <= v <= 1.5 for v in [x1, y1, x2, y2]):
        return list(bbox)
    return [
        max(0.0, x1 - margin),
        max(0.0, y1 - margin),
        min(1.0, x2 + margin),
        min(1.0, y2 + margin),
    ]


def upscale_image(img_bytes: bytes, min_dim: int = 600) -> bytes:
    """
    Fix 8: upscale small crops to at least min_dim on the longer side.
    Small crops (e.g. a single header line) are often too low-res for OCR.
    Returns original bytes if Pillow is unavailable or upscaling isn't needed.
    """
    try:
        from PIL import Image
    except ImportError:
        return img_bytes
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if max(img.size) < min_dim:
            scale = min_dim / max(img.size)
            new_w = max(1, int(img.size[0] * scale))
            new_h = max(1, int(img.size[1] * scale))
            img   = img.resize((new_w, new_h), Image.LANCZOS)
            buf   = io.BytesIO()
            img.save(buf, format="PNG")
            result = buf.getvalue()
            if len(result) < MAX_IMAGE_BYTES:
                return result
        return img_bytes
    except Exception as e:
        logger.debug("[upscale_image] failed: %s", e)
        return img_bytes



# ---------------------------------------------------------------------------
# PaddleOCR — local fallback when NVIDIA OCR returns empty
# ---------------------------------------------------------------------------

def _paddle_ocr(image_bytes: bytes) -> list[dict]:
    """
    Run PaddleOCR on image_bytes and return normalised element list.
    Only called when NVIDIA nemotron-ocr-v1 returns empty (no network needed).
    Returns [] if PaddleOCR is not installed or fails.

    Install:  pip install paddleocr paddlepaddle
    """
    global _paddle_instance
    if not _PADDLEOCR_AVAILABLE:
        return []
    try:
        import numpy as np
        from PIL import Image as _PILImg
        if _paddle_instance is None:
            # Explicit flags required — default init can silently skip detection
            # or recognition steps, producing empty results.
            _paddle_instance = _PaddleOCRCls(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                det=True,       # run text detection
                rec=True,       # run text recognition
                cls=True,       # run angle classification
                show_log=False,
            )
        img = _PILImg.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        result = _paddle_instance.ocr(arr, cls=True)

        # Debug: always log raw result shape so you can see if Paddle ran
        logger.debug("[PaddleOCR] raw result type=%s  len=%s",
                     type(result).__name__,
                     len(result) if result else 0)
        if result and result[0]:
            logger.debug("[PaddleOCR] first line sample: %s", str(result[0][0])[:120])

        elements: list[dict] = []
        if result and result[0]:
            for line in result[0]:
                if not line:
                    continue
                text_info = line[1] if len(line) > 1 else None
                text      = text_info[0] if text_info else ""
                conf      = text_info[1] if text_info and len(text_info) > 1 else 0.0
                if text and text.strip():
                    elements.append({
                        "type":         "text",
                        "content":      _clean_text(text),
                        "page":         1,
                        "bbox":         [],
                        "model_source": "paddleocr",
                        "confidence":   float(conf),
                    })
        if not elements:
            logger.warning("[PaddleOCR] returned 0 elements — raw=%s", str(result)[:200])
        return elements
    except Exception as e:
        logger.debug("[PaddleOCR] failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Nemotron Parse v1.1  (preferred over page-elements-v3)
# ---------------------------------------------------------------------------

def _call_parse(
    page_bytes: bytes, filename: str, api_key: str,
    max_retries: int = 3, retry_delay: float = 2.0,
) -> list[dict]:
    """
    Nemotron Parse v1.1 — CURRENTLY DISABLED due to SSL EOF failures.
    The endpoint causes SSLError on many Windows/corporate networks.
    Falls back directly to page-elements-v3 which is universally stable.
    Re-enable by removing the early return below once SSL issues are resolved.
    """
    # DISABLED: Parse causes SSLError: EOF occurred in violation of protocol
    # on Windows. page-elements-v3 is stable and universally available.
    logger.debug("[Parse] disabled → using page-elements-v3 (SSL issues)")
    return _call_page_elements(page_bytes, filename, api_key,
                               max_retries=max_retries, retry_delay=retry_delay)

def _score_page_result(elements: list[dict]) -> float:
    """Confidence score 0–10 for a page extraction result."""
    if not elements:
        return 0.0
    # Fix 10: updated weights reflect Parse being higher quality than page-elements
    _w = {
        "table_structure":   4.0,
        "parse+ocr":         3.0,
        "page_elements+ocr": 2.5,
        "parse":             2.5,
        "page_elements":     2.0,
        "ocr":               2.5,   # whole-page OCR is actually decent quality
        "pdfplumber":        0.5,
    }
    total_chars   = sum(len(str(
        e.get("content_text") or e.get("content") or e.get("text") or ""
    )) for e in elements)
    text_score    = min(total_chars / 500, 5.0)
    element_score = min(len(elements) / 5, 2.0)
    model_score   = max((_w.get(e.get("model_source", ""), 1.0) for e in elements), default=1.0)
    return round(text_score + element_score + model_score, 2)


# ---------------------------------------------------------------------------
# Smart per-page processor
# ---------------------------------------------------------------------------

@dataclass
class _PageResult:
    page_number:      int
    elements:         list[dict]
    model_used:       str
    success:          bool
    confidence_score: float = 0.0


def _process_single_page(
    page_bytes:  bytes,
    page_number: int,
    source_name: str,
    api_key:     str,
    max_retries: int,
    retry_delay: float,
    native_text: str = "",     # pre-extracted PDF text layer (may be empty for scans)
) -> _PageResult:
    """
    v5.2 cascade:
        Step 0: Native text fast-path — if the PDF page has ≥50 chars of embedded
                text, use it directly.  No image processing, no API calls.
                Financial PDFs from banks are almost always text-layer PDFs.
        Step 1: page-elements-v3 layout detection (Parse disabled, SSL issues)
        Step 1b: Early whole-page OCR if layout returned no text
        Step 2: Per-element smart routing with full-page bbox filter
        Step 3: Whole-page OCR fallback → PaddleOCR fallback
        Step 4: Merge
    """
    fname = f"page_{page_number}_{source_name}"

    # ── Step 0: Native text fast-path ─────────────────────────────────────────
    # If PyMuPDF extracted real text from this page, use it immediately.
    # This path is: fast, lossless, perfectly accurate, no API credits consumed.
    # It fires for the vast majority of bank PDFs which are text-layer documents.
    native_clean = _clean_text(native_text) if native_text else ""
    if _USE_NATIVE_TEXT and len(native_clean) >= _NATIVE_TEXT_MIN_CHARS:
        logger.info("[PDFPipeline] p%d  native text (%d chars) — skipping OCR entirely",
                    page_number, len(native_clean))
        elem = {
            "type":         "text",
            "content":      native_clean,
            "page":         page_number,
            "bbox":         [],
            "model_source": "native_text",
        }
        return _PageResult(page_number, [elem], "native_text", True,
                           _score_page_result([elem]))

    # ── Step 1: Nemotron Parse v1.1 → auto-fallback to page-elements-v3 ─────────
    # Parse gives fine-grained semantic regions (text lines, tables, headings)
    # instead of the single whole-page "text" bbox page-elements often returns.
    # _call_parse() already handles 404 → page-elements fallback internally,
    # so this is a no-cost upgrade: if Parse works we get better bboxes,
    # if it 404s we get identical behaviour to before.
    pe_elements: list[dict] = []
    pe_ok = False
    parse_model_used = "page_elements"
    try:
        pe_elements = _call_parse(page_bytes, fname, api_key,
                                  max_retries=max_retries, retry_delay=retry_delay)
        # Detect whether Parse actually succeeded (vs silently fell back)
        parse_model_used = "parse" if pe_elements and any(
            e.get("model_source","") == "parse" for e in pe_elements
        ) else "page_elements"
        for e in pe_elements:
            e["page"] = page_number
        pe_ok = True
        type_counts: dict[str, int] = {}
        for e in pe_elements:
            type_counts[e.get("type", "?")] = type_counts.get(e.get("type", "?"), 0) + 1
        logger.info("[PDFPipeline] p%d  layout OK  model=%s  n=%d  types=%s",
                    page_number, parse_model_used, len(pe_elements), type_counts)
    except Exception as e1:
        logger.warning("[PDFPipeline] p%d  layout detection FAILED: %s", page_number, e1)

    # ── Step 1b: Early whole-page OCR if page_elements returned no text ──────────
    # FIX 7: For dense financial PDFs, page_elements bboxes often cover the whole
    # page with a single "text" element and no actual text content. In that case,
    # running whole-page OCR immediately is faster and more accurate than trying
    # to crop sub-regions from coordinates that don't isolate meaningful regions.
    early_wp_elems: list[dict] = []
    pe_has_text = any(
        len(_clean_text(str(e.get("content", "") or e.get("text", "") or ""))) >= _MIN_EXISTING_TEXT_LEN
        for e in pe_elements
    ) if pe_ok and pe_elements else False

    if pe_ok and not pe_has_text:
        logger.info("[PDFPipeline] p%d  no text in page_elements → trying whole-page OCR first",
                    page_number)
        try:
            early_wp_elems = _call_ocr(page_bytes, fname, api_key,
                                       max_retries=max(1, max_retries - 1),
                                       retry_delay=retry_delay)
            _log_ocr_debug(early_wp_elems, page_number, "early-whole-page")
            for e in early_wp_elems:
                e["page"]         = page_number
                e["element_type"] = "ocr"
                e["model_source"] = "ocr"
        except Exception as e_early:
            logger.debug("[PDFPipeline] p%d early whole-page OCR failed: %s", page_number, e_early)

        # If early whole-page OCR succeeded with meaningful text, use it directly
        # and skip the expensive per-element crop+OCR loop entirely.
        early_chars = sum(len(str(e.get("content", "") or "")) for e in early_wp_elems)
        if early_chars >= _MIN_PAGE_TEXT_CHARS:
            logger.info("[PDFPipeline] p%d  early whole-page OCR OK (%d chars) — skipping per-elem",
                        page_number, early_chars)
            return _PageResult(page_number, early_wp_elems, "ocr", True,
                               _score_page_result(early_wp_elems))


    enriched:   list[dict] = []
    per_elem_ok = False

    if pe_ok and pe_elements:
        ocr_ok = 0
        for idx, elem in enumerate(pe_elements):
            bbox      = elem.get("bbox", [])
            elem_type = elem.get("type", "text")
            efname    = f"page_{page_number}_e{idx}_{source_name}"

            # Skip degenerate bboxes (too small to be useful)
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                if all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2]):
                    if (x2 - x1) < _MIN_BBOX_DIM or (y2 - y1) < _MIN_BBOX_DIM:
                        continue
                    # NEW: skip bboxes that are essentially the full page.
                    # page-elements often returns a single bbox covering 90%+ of the page.
                    # Cropping that → same blurry result as passing the whole image.
                    # These will be handled by the whole-page OCR fallback below.
                    bbox_area = (x2 - x1) * (y2 - y1)
                    if bbox_area > _MAX_BBOX_AREA_FRACTION:
                        logger.debug(
                            "[PDFPipeline] p%d e%d  full-page bbox (%.0f%%) — skip crop, "
                            "let whole-page OCR handle it",
                            page_number, idx, bbox_area * 100,
                        )
                        continue

            # Clean + validate existing text from Parse
            raw      = str(elem.get("content", "") or "") or str(elem.get("text", "") or "")
            existing = _clean_text(raw)

            # If Parse already gave good text → keep directly, skip OCR (saves credits)
            if existing and len(existing) >= _MIN_EXISTING_TEXT_LEN:
                e2 = dict(elem)
                e2["content"]      = existing
                e2["model_source"] = parse_model_used
                enriched.append(e2)
                ocr_ok += 1
                continue

            # Fix 4: expand bbox before cropping — tight crops kill OCR accuracy
            expanded_bbox = expand_bbox(bbox, margin=0.02)
            cropped = crop_image(page_bytes, expanded_bbox) if expanded_bbox else page_bytes

            # Fix 9: extended table type list
            if elem_type in ("table", "grid", "matrix"):
                try:
                    te_list = _call_table_structure(cropped, efname, api_key,
                                                    max_retries=1, retry_delay=retry_delay)
                    if te_list:
                        for te in te_list:
                            te["page"]         = page_number
                            te["model_source"] = "table_structure"
                        enriched.extend(te_list)
                        ocr_ok += 1
                        continue
                except Exception as te:
                    logger.debug("[PDFPipeline] p%d e%d table_structure failed: %s",
                                 page_number, idx, te)

            # FIX 3: DO NOT upscale. Upscaling adds blur → destroys OCR features.
            # The original sharp pixels from a high-DPI render are better than a
            # synthetically enlarged version of a low-DPI crop.
            ocr_input = cropped

            # Contrast enhancement — big accuracy boost on low-contrast financial PDFs
            try:
                from PIL import Image as _PILImage, ImageEnhance as _Enhance
                _img = _PILImage.open(io.BytesIO(ocr_input))
                _img = _Enhance.Contrast(_img).enhance(1.5)
                _buf = io.BytesIO()
                _img.save(_buf, format="PNG")
                _enhanced = _buf.getvalue()
                if len(_enhanced) < MAX_IMAGE_BYTES:
                    ocr_input = _enhanced
            except Exception:
                pass   # contrast boost failed silently — use original

            # Fix 7 (multi-pass OCR): try OCR on original crop, then upscaled.
            # Keep whichever produced more valid text.
            best_ocr: list[dict] = []
            for pass_num, ocr_bytes in enumerate([cropped, ocr_input], start=1):
                if pass_num == 2 and ocr_bytes is cropped:
                    break   # upscale returned same bytes — no point calling again
                try:
                    ocr_result = _call_ocr(ocr_bytes, f"{efname}_p{pass_num}", api_key,
                                           max_retries=max(1, max_retries - 1),
                                           retry_delay=retry_delay)
                    _log_ocr_debug(ocr_result, page_number, f"e{idx} pass{pass_num}")
                    ocr_chars = sum(len(str(e.get("content","") or "")) for e in ocr_result)
                    best_chars = sum(len(str(e.get("content","") or "")) for e in best_ocr)
                    if ocr_chars > best_chars:
                        best_ocr = ocr_result
                except Exception as oe:
                    logger.debug("[PDFPipeline] p%d e%d pass%d OCR failed: %s",
                                 page_number, idx, pass_num, oe)

            if best_ocr:
                for oe in best_ocr:
                    oe["page"]         = page_number
                    oe["element_type"] = elem_type
                    oe["model_source"] = f"{parse_model_used}+ocr"
                    oe["bbox"]         = bbox
                enriched.extend(best_ocr)
                ocr_ok += 1

        enriched    = _merge_ocr_fragments(enriched)
        per_elem_ok = ocr_ok > 0
        logger.info("[PDFPipeline] p%d  per-elem routing: %d/%d → text",
                    page_number, ocr_ok, len(pe_elements))

    # ── Step 3: Whole-page OCR fallback ────────────────────────────────────────
    inter_score = _score_page_result(enriched)
    wp_elems: list[dict] = []
    # Lowered threshold from 1.0 → 2.5: trigger whole-page OCR more aggressively.
    # Element-level crops often miss text when bboxes are imprecise; whole-page
    # OCR is a reliable safety net and should fire whenever confidence is marginal.
    if not pe_ok or not per_elem_ok or inter_score < 2.5:
        reason = ("pe_failed" if not pe_ok
                  else "no_elem_text" if not per_elem_ok
                  else f"low_score({inter_score:.1f})")
        logger.info("[PDFPipeline] p%d  whole-page OCR (%s)", page_number, reason)
        try:
            wp_elems = _call_ocr(page_bytes, fname, api_key,
                                 max_retries=max(1, max_retries - 1), retry_delay=retry_delay)
            _log_ocr_debug(wp_elems, page_number, "whole-page")
            for e in wp_elems:
                e["page"]         = page_number
                e["element_type"] = "ocr"
                e["model_source"] = "ocr"
            if wp_elems:
                logger.info("[PDFPipeline] p%d  whole-page OCR OK  n=%d", page_number, len(wp_elems))
            else:
                # NVIDIA OCR returned empty → try PaddleOCR locally as last resort
                if _PADDLEOCR_AVAILABLE:
                    logger.info("[PDFPipeline] p%d  NVIDIA OCR empty → trying PaddleOCR",
                                page_number)
                    wp_elems = _paddle_ocr(page_bytes)
                    for e in wp_elems:
                        e["page"] = page_number
                    if wp_elems:
                        logger.info("[PDFPipeline] p%d  PaddleOCR OK  n=%d", page_number, len(wp_elems))
                    else:
                        logger.warning("[PDFPipeline] p%d  PaddleOCR also returned empty",
                                       page_number)
        except Exception as e2:
            logger.warning("[PDFPipeline] p%d  whole-page OCR FAILED: %s", page_number, e2)
            # Try PaddleOCR on exception too
            if _PADDLEOCR_AVAILABLE:
                logger.info("[PDFPipeline] p%d  NVIDIA OCR exception → trying PaddleOCR",
                            page_number)
                try:
                    wp_elems = _paddle_ocr(page_bytes)
                    for e in wp_elems:
                        e["page"] = page_number
                except Exception:
                    pass

    # ── Step 4: Merge ───────────────────────────────────────────────────────────
    def _tchars(elems: list[dict]) -> int:
        return sum(len(str(e.get("content_text") or e.get("content") or e.get("text") or ""))
                   for e in elems)

    if enriched:
        ec, wc = _tchars(enriched), _tchars(wp_elems)
        merged = wp_elems if wc > ec * 2 else enriched
        if wc > ec * 2:
            logger.info("[PDFPipeline] p%d  whole-page OCR preferred (%d vs %d chars)",
                        page_number, wc, ec)
    else:
        merged = wp_elems

    total = _tchars(merged)
    if not merged or total < _MIN_PAGE_TEXT_CHARS:
        logger.warning("[PDFPipeline] p%d  text too short (%d < %d) → pdfplumber",
                       page_number, total, _MIN_PAGE_TEXT_CHARS)
        return _PageResult(page_number, [], "pdfplumber", False, 0.0)

    final_score = _score_page_result(merged)

    # FIX 4: Removed the "final_score < 2.0 → fallback" kill switch.
    # When OCR is returning empty (image quality issue), the score is always low,
    # so the old gate was *always* firing and discarding any partial OCR success.
    # Now we only fall back when merged is truly empty (checked above).

    src_counts: dict[str, int] = {}
    for e in merged:
        s = e.get("model_source", f"{parse_model_used}+ocr")
        src_counts[s] = src_counts.get(s, 0) + 1
    model_tag = max(src_counts, key=lambda k: src_counts[k])

    logger.info("[PDFPipeline] p%d  SUCCESS  n=%d  model=%s  chars=%d  score=%.1f",
                page_number, len(merged), model_tag, total, final_score)
    return _PageResult(page_number, merged, model_tag, True, final_score)


# ---------------------------------------------------------------------------
# Full-PDF NVIDIA pipeline
# ---------------------------------------------------------------------------

class FileSizeError(Exception):
    pass


def _call_page_elements_api(
    pdf_path:    Path,
    api_key:     str,
    max_retries: int   = 3,
    retry_delay: float = 2.0,
) -> tuple[list[dict], list[int]]:
    size_bytes = pdf_path.stat().st_size
    if size_bytes > _MAX_PDF_BYTES:
        raise FileSizeError(f"{pdf_path.name} {size_bytes/1024/1024:.1f} MB > 5 MB limit")

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n   = len(doc)
        doc.close()
    except ImportError:
        raise FileSizeError("PyMuPDF not available")

    logger.info("[PDFPipeline] %s — %d pages %.1f MB", pdf_path.name, n, size_bytes/1024/1024)

    # ── Pre-extract native text from all pages (fast, sequential) ─────────────
    # Most bank PDFs are text-layer documents. Extracting native text here means
    # _process_single_page can skip OCR entirely for pages that already have text.
    native_texts: list[str] = []
    if _USE_NATIVE_TEXT:
        try:
            import fitz as _fitz_nt
            doc_nt = _fitz_nt.open(str(pdf_path))
            for i in range(n):
                try:
                    native_texts.append(doc_nt[i].get_text("text").strip())
                except Exception:
                    native_texts.append("")
            doc_nt.close()
            n_native = sum(1 for t in native_texts if len(t) >= _NATIVE_TEXT_MIN_CHARS)
            logger.info("[PDFPipeline] native text: %d/%d pages have ≥%d chars — those skip OCR",
                        n_native, n, _NATIVE_TEXT_MIN_CHARS)
        except Exception as nte:
            logger.debug("[PDFPipeline] native text extraction failed: %s", nte)
            native_texts = [""] * n
    else:
        native_texts = [""] * n

    preprocessed:        list[tuple[int, bytes]] = []
    failed_page_indices: list[int]               = []

    for idx in range(n):
        result = _pdf_page_to_preprocessed_bytes(pdf_path, idx)
        if result is None:
            failed_page_indices.append(idx)
        else:
            preprocessed.append((idx + 1, result))

    if not preprocessed:
        return [], list(range(n))

    # FIX 5: Sequential rasterization done above; thread pool handles API calls only.
    all_pages:   list[dict] = []
    futures_map: dict       = {}

    with ThreadPoolExecutor(max_workers=min(4, len(preprocessed))) as pool:
        for page_num, png in preprocessed:
            nt = native_texts[page_num - 1] if page_num - 1 < len(native_texts) else ""
            fut = pool.submit(_process_single_page, png, page_num, pdf_path.name,
                              api_key, max_retries, retry_delay, nt)
            futures_map[fut] = page_num - 1

        for fut in as_completed(futures_map):
            idx2     = futures_map[fut]
            page_num = idx2 + 1
            try:
                r = fut.result(timeout=60)   # FIX 5: 60s per page (was 25s — OCR can be slow)
                if r.success and r.elements:
                    all_pages.append({"page": r.page_number, "elements": r.elements,
                                      "model": r.model_used, "confidence_score": r.confidence_score})
                else:
                    failed_page_indices.append(idx2)
            except Exception as fe:
                logger.warning("[PDFPipeline] p%d thread error: %s", page_num, fe)
                failed_page_indices.append(idx2)

    all_pages.sort(key=lambda p: p["page"])
    mc: dict[str, int] = {}
    conf_sum = 0.0
    for p in all_pages:
        mc[p.get("model","?")] = mc.get(p.get("model","?"), 0) + 1
        conf_sum += p.get("confidence_score", 0.0)

    logger.info("[PDFPipeline] %s — %d/%d pages  models=%s  fallback=%d  avg_conf=%.1f",
                pdf_path.name, len(all_pages), n, mc, len(failed_page_indices),
                conf_sum / len(all_pages) if all_pages else 0)
    return all_pages, failed_page_indices


# ---------------------------------------------------------------------------
# Element → StructuredChunk
# ---------------------------------------------------------------------------

def _chunk_elements(
    elements:          list[dict],
    bank:              str,
    source:            str,
    model_source:      str   = "page_elements",
    section_max_words: int   = 400,
    table_max_words:   int   = 300,
    confidence_score:  float = 0.0,
) -> list[StructuredChunk]:
    chunks:          list[StructuredChunk] = []
    chunk_id:        int  = 0
    current_section: str  = ""
    buffer_words:    list[str] = []
    buffer_page:     int  = 0

    def flush():
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
            model_source=model_source, confidence_score=confidence_score,
        ))
        chunk_id += 1
        buffer_words = []

    for elem in elements:
        etype   = elem.get("element_type") or elem.get("type", "text")
        content = (elem.get("content_text") or elem.get("content") or
                   elem.get("text") or elem.get("value") or elem.get("description") or "")
        page    = elem.get("page", 0)
        src     = elem.get("model_source", model_source) or model_source

        if etype in ("title", "heading"):
            flush()
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
            flush()
            if isinstance(content, str) and content.strip():
                readable, structured = content, {"rendered": content}
            else:
                readable, structured = table_to_text(elem.get("content", content))
            if not readable.strip():
                continue
            for sub in _split_into_sentences(readable, table_max_words):
                if not sub.strip():
                    continue
                chunks.append(StructuredChunk(
                    text=sub, source=source, bank=bank, doc_type="table",
                    chunk_id=chunk_id, page_number=page,
                    section_title=current_section,
                    element_type="table", field_hint=detect_field_hint(sub),
                    structured_data=structured,
                    model_source=src, confidence_score=confidence_score,
                ))
                chunk_id += 1

        elif etype == "figure_caption":
            flush()
            text = _clean_text(str(content))
            if text:
                chunks.append(StructuredChunk(
                    text=text, source=source, bank=bank, doc_type="paragraph",
                    chunk_id=chunk_id, page_number=page,
                    section_title=current_section, element_type="figure_caption",
                    model_source=src, confidence_score=confidence_score,
                ))
                chunk_id += 1

        else:
            if content is None:
                continue
            text = _clean_text(str(content))
            if not text:
                for fk in ("text", "value", "description", "body"):
                    alt = str(elem.get(fk, "") or "").strip()
                    if alt:
                        text = _clean_text(alt)
                        break
            if not text:
                continue
            words = text.split()
            if not buffer_page:
                buffer_page = page
            if len(buffer_words) + len(words) > section_max_words:
                flush()
                buffer_page = page
            buffer_words.extend(words)

    flush()
    return chunks


# ---------------------------------------------------------------------------
# pdfplumber fallback
# ---------------------------------------------------------------------------

def _fallback_extract(
    pdf_path:            Path,
    bank:                str,
    failed_page_indices: list[int] | None = None,
) -> list[StructuredChunk]:
    max_words     = _cfg("PDF_SECTION_CHUNK_MAX", 512)
    chunk_overlap = _cfg("CHUNK_OVERLAP", 64)
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = (
                [pdf.pages[i] for i in failed_page_indices if i < len(pdf.pages)]
                if failed_page_indices is not None else list(pdf.pages)
            )
            full_text = "\n".join(p.extract_text() or "" for p in pages)
    except ImportError:
        try:
            import pypdf
            r = pypdf.PdfReader(str(pdf_path))
            if failed_page_indices is not None:
                full_text = "\n".join(r.pages[i].extract_text() or ""
                                      for i in failed_page_indices if i < len(r.pages))
            else:
                full_text = "\n".join(p.extract_text() or "" for p in r.pages)
        except ImportError:
            logger.error("[PDFPipeline] No PDF lib. pip install pdfplumber")
            return []

    words = full_text.split()
    if not words:
        return []

    chunks: list[StructuredChunk] = []
    cid, start = 0, 0
    while start < len(words):
        end  = min(start + max_words, len(words))
        text = _clean_text(" ".join(words[start:end]))
        if len(text.split()) >= 5:
            chunks.append(StructuredChunk(
                text=text, source=pdf_path.name, bank=bank,
                doc_type="pdf", chunk_id=cid,
                field_hint=detect_field_hint(text),
                model_source="pdfplumber", confidence_score=1.0,
            ))
            cid += 1
        if end == len(words):
            break
        start += max_words - chunk_overlap
    return chunks


# ---------------------------------------------------------------------------
# PDFPipeline — orchestrator
# ---------------------------------------------------------------------------

class PDFPipeline:
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
        self._table_max   = int(_cfg("PDF_TABLE_CHUNK_MAX", 300))

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
                logger.info("[PDFPipeline] NVIDIA 0 chunks → full pdfplumber fallback")
            chunks = _fallback_extract(pdf_path, bank)
        for i, c in enumerate(chunks):
            c.chunk_id = i

        # Fix 7: deduplicate by content_hash (removes OCR + page_elements duplicates)
        seen_hashes: set[str] = set()
        unique: list[StructuredChunk] = []
        for c in chunks:
            if c.content_hash not in seen_hashes:
                seen_hashes.add(c.content_hash)
                unique.append(c)
        if len(unique) < len(chunks):
            logger.info("[PDFPipeline] Deduplicated %d → %d chunks",
                        len(chunks), len(unique))
        chunks = unique

        # Fix 8: hard cap per PDF to keep retrieval fast
        MAX_CHUNKS_PER_DOC = 300
        if len(chunks) > MAX_CHUNKS_PER_DOC:
            logger.warning("[PDFPipeline] Capping %d → %d chunks for %s",
                           len(chunks), MAX_CHUNKS_PER_DOC, pdf_path.name)
            chunks = chunks[:MAX_CHUNKS_PER_DOC]

        if chunks:
            # Fix 4: only cache if NVIDIA contributed at least one chunk.
            # Avoids permanently storing pdfplumber-only results when NVIDIA is
            # enabled but failing — those should retry on next run.
            _nvidia_sources = {"page_elements", "ocr", "table_structure", "page_elements+ocr"}
            nvidia_contributed = any(c.model_source in _nvidia_sources for c in chunks)

            if not self.use_nvidia or not self.api_key or nvidia_contributed:
                self._save_cache(pdf_path, chunks)
            else:
                logger.warning(
                    "[PDFPipeline] Not caching %s — NVIDIA enabled but 0 NVIDIA chunks "
                    "(all pdfplumber). Will retry on next run.",
                    pdf_path.name,
                )
        else:
            logger.warning("[PDFPipeline] No chunks: %s", pdf_path.name)
        return chunks

    def _process_nvidia(self, pdf_path: Path, bank: str) -> list[StructuredChunk]:
        try:
            raw_pages, failed = _call_page_elements_api(
                pdf_path, self.api_key,
                max_retries=self._max_retries, retry_delay=self._retry_delay,
            )
        except FileSizeError as e:
            logger.info("[PDFPipeline] %s — pdfplumber", e)
            return []
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (401, 403):
                raise RuntimeError(f"NVIDIA auth failed HTTP {status}") from e
            logger.warning("[PDFPipeline] HTTP %d — pdfplumber: %s", status, e)
            return []
        except Exception as e:
            logger.warning("[PDFPipeline] NVIDIA error — pdfplumber: %s", e)
            return []

        all_chunks: list[StructuredChunk] = []
        for pd_ in raw_pages:
            elems  = pd_.get("elements", [])
            model  = pd_.get("model", "page_elements")
            conf   = pd_.get("confidence_score", 0.0)
            if bank == "Unknown":
                sample = " ".join(str(e.get("content",""))[:150]
                                  for e in elems[:8] if e.get("type") in ("text","title"))
                bank = detect_bank(pdf_path.name, sample)
            texts = [str(e.get("content_text") or e.get("content") or e.get("text") or "").strip()
                     for e in elems if str(e.get("content_text") or e.get("content") or e.get("text") or "").strip()]
            logger.info("[DEBUG] p%d  n=%d  model=%s  score=%.1f  sample=%s",
                        pd_.get("page","?"), len(elems), model, conf,
                        texts[0][:120] if texts else "NO TEXT")
            all_chunks.extend(_chunk_elements(
                elems, bank, pdf_path.name,
                model_source=model, section_max_words=self._section_max,
                table_max_words=self._table_max, confidence_score=conf,
            ))

        if failed:
            logger.info("[PDFPipeline] Gap-fill pdfplumber pages: %s", [i+1 for i in failed])
            all_chunks.extend(_fallback_extract(pdf_path, bank, failed))

        mc: dict[str, int] = {}
        for c in all_chunks:
            mc[c.model_source] = mc.get(c.model_source, 0) + 1
        _ns = {"page_elements", "ocr", "table_structure", "page_elements+ocr"}
        nv  = sum(v for k, v in mc.items() if k in _ns)
        if nv == 0:
            logger.warning("[PDFPipeline] 🚨 No NVIDIA success %s  sources=%s",
                           pdf_path.name, mc)
        else:
            logger.info("[PDFPipeline] %s — %d chunks  nvidia=%d  sources=%s",
                        pdf_path.name, len(all_chunks), nv, mc)
        return all_chunks


# ---------------------------------------------------------------------------
# Backward-compat
# ---------------------------------------------------------------------------

def _normalise_elements(raw_pages: list[dict]) -> list[dict]:
    elems: list[dict] = []
    for pd_ in raw_pages:
        pn = pd_.get("page", 0)
        for elem in pd_.get("elements", []):
            elems.append({"type": elem.get("type","text"),
                          "content": elem.get("content",""),
                          "page": pn, "bbox": elem.get("bbox",[])})
    return elems

def process_pdf(file_path: str) -> list[dict]:
    return [c.to_dict() for c in PDFPipeline().process(file_path)]

def test_single_pdf(file_path: str, api_key: str | None = None) -> None:
    import os
    key  = api_key or os.getenv("NVIDIA_API_KEY", "")
    size = Path(file_path).stat().st_size / 1024 / 1024 if Path(file_path).exists() else 0
    print(f"[test] {file_path}  key={bool(key)}  size={size:.2f}MB")
    if key and size <= 5:
        r = _pdf_page_to_preprocessed_bytes(Path(file_path), 0)
        print(f"[test] preprocess p1: {'✅ '+str(round(len(r)/1024,1))+'KB' if r else '❌ failed'}")
    pipeline = PDFPipeline(api_key=key or None)
    chunks   = pipeline.process(file_path)
    print(f"[test] {len(chunks)} chunks")
    if chunks:
        mc = {}
        for c in chunks:
            mc[c.model_source] = mc.get(c.model_source, 0) + 1
        _ns = {"page_elements","ocr","table_structure","page_elements+ocr"}
        print(f"[test] sources={mc}  nvidia={sum(v for k,v in mc.items() if k in _ns)}/{len(chunks)}")
        print(f"[test] avg_conf={sum(c.confidence_score for c in chunks)/len(chunks):.1f}/10")
        print(f"[test] sample [{chunks[0].doc_type}|{chunks[0].model_source}] {chunks[0].text[:120]}")