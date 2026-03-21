"""
pdf_pipeline/ocr.py
====================
OCR layer for scanned PDFs.

Design:
  PaddleOCR  → primary  (neural, high accuracy on printed text, no internet needed)
  Tesseract  → fallback (ubiquitous, simple, works everywhere)

Neither is called on text-layer PDFs — the extractor routes those through
pdfplumber/PyMuPDF which is faster and perfectly accurate.

Install:
  pip install paddleocr paddlepaddle   # PaddleOCR
  pip install pytesseract              # Tesseract Python wrapper
  # Also install the Tesseract binary:
  # Windows: https://github.com/UB-Mannheim/tesseract/wiki
  # Ubuntu:  apt install tesseract-ocr
  # macOS:   brew install tesseract
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PaddleOCR singleton — lazy init, expensive to construct
# ---------------------------------------------------------------------------

_paddle_instance = None
_paddle_available = False

try:
    from paddleocr import PaddleOCR as _PaddleOCRClass
    _paddle_available = True
except ImportError:
    pass


def _get_paddle():
    global _paddle_instance
    if not _paddle_available:
        return None
    if _paddle_instance is None:
        logger.info("[OCR] Initialising PaddleOCR (first call)…")
        # PaddleOCR 2.x constructor only takes: use_angle_cls, lang, use_gpu, show_log
        # det/rec/cls are NOT constructor args — they are passed to .ocr() call instead.
        # Passing them to the constructor causes: "Unknown argument: det"
        _paddle_instance = _PaddleOCRClass(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
        logger.info("[OCR] PaddleOCR ready")
    return _paddle_instance


# ---------------------------------------------------------------------------
# Tesseract availability
# ---------------------------------------------------------------------------

_tesseract_available = False
try:
    import pytesseract as _pytesseract
    _pytesseract.get_tesseract_version()
    _tesseract_available = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render_pdf_pages(
    pdf_path: Path,
    dpi: int = 300,
    page_indices: list[int] | None = None,
) -> list[tuple[int, bytes]]:
    """
    Render PDF pages to PNG images using PyMuPDF at the specified DPI.

    Args:
        pdf_path:     path to PDF
        dpi:          render resolution — 300 is the sweet spot for OCR accuracy
        page_indices: 0-based list of pages to render; None = all pages

    Returns:
        List of (1-based page_number, png_bytes) tuples.

    300 DPI is chosen deliberately:
      - 200 DPI: small fonts become illegible for OCR
      - 300 DPI: industry standard for document OCR
      - 400 DPI: marginally better but 78% more memory/bandwidth
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF required for OCR rendering: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    matrix = fitz.Matrix(dpi / 72, dpi / 72)

    indices = page_indices if page_indices is not None else list(range(len(doc)))
    results: list[tuple[int, bytes]] = []

    for idx in indices:
        if idx < 0 or idx >= len(doc):
            continue
        try:
            pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
            results.append((idx + 1, pix.tobytes("png")))
        except Exception as e:
            logger.warning("[OCR] render p%d failed: %s", idx + 1, e)

    doc.close()
    return results


# ---------------------------------------------------------------------------
# PaddleOCR
# ---------------------------------------------------------------------------

def paddle_ocr_page(img_bytes: bytes) -> str:
    """
    Run PaddleOCR on a single page image.
    Returns extracted text as a plain string, empty string on failure.
    """
    paddle = _get_paddle()
    if paddle is None:
        logger.debug("[OCR] PaddleOCR not available")
        return ""

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr = np.array(img)

        result = paddle.ocr(arr, det=True, rec=True, cls=True)
        if not result or not result[0]:
            logger.debug("[OCR] PaddleOCR returned empty result")
            return ""

        lines = []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
            conf = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
            if text.strip() and conf > 0.5:   # filter low-confidence garbage
                lines.append(text.strip())

        result_text = " ".join(lines)
        logger.debug("[OCR] PaddleOCR: %d chars", len(result_text))
        return result_text

    except Exception as e:
        logger.warning("[OCR] PaddleOCR failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------

def tesseract_ocr_page(img_bytes: bytes) -> str:
    """
    Run Tesseract OCR on a single page image.
    Returns extracted text, empty string on failure or if not installed.
    """
    if not _tesseract_available:
        logger.debug("[OCR] Tesseract not available")
        return ""

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # PSM 3: fully automatic page segmentation, best for multi-column docs
        # OEM 3: default engine (LSTM + legacy for best compatibility)
        config = "--psm 3 --oem 3"
        text = _pytesseract.image_to_string(img, config=config)
        result = text.strip()
        logger.debug("[OCR] Tesseract: %d chars", len(result))
        return result

    except Exception as e:
        logger.warning("[OCR] Tesseract failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Availability report — called at startup for observability
# ---------------------------------------------------------------------------

def ocr_availability() -> dict:
    return {
        "paddleocr":  _paddle_available,
        "tesseract":  _tesseract_available,
        "any_ocr":    _paddle_available or _tesseract_available,
    }