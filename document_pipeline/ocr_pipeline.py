"""
document_pipeline/ocr_pipeline.py
====================================
OCR pipeline for scanned PDFs and image uploads.

Stack (in order of preference):
  PaddleOCR  → primary  (neural, excellent on printed text, local)
  Tesseract  → fallback (universally available, proven on bank docs)

NVIDIA OCR models (nemotron-ocr-v1) are intentionally NOT used here.
The NVIDIA CV stack has known SSL/endpoint issues on Windows (diagnosed
during development) and the local PaddleOCR + Tesseract stack is more
reliable for production use on the current hardware.

Usage:
    pipeline = OCRPipeline()
    text, tables, method = pipeline.process_pdf(pdf_bytes)
    text, tables, method = pipeline.process_image(image_bytes)
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DPI = 300   # 300 DPI is the industry standard for OCR accuracy


# ---------------------------------------------------------------------------
# PaddleOCR singleton
# ---------------------------------------------------------------------------

_paddle = None
_paddle_available = False

try:
    from paddleocr import PaddleOCR as _PaddleOCRClass
    _paddle_available = True
except ImportError:
    pass


def _get_paddle():
    global _paddle
    if not _paddle_available:
        return None
    if _paddle is None:
        logger.info("[OCRPipeline] Initialising PaddleOCR…")
        # det/rec/cls go on .ocr() call, NOT the constructor (PaddleOCR 2.x)
        _paddle = _PaddleOCRClass(use_angle_cls=True, lang="en",
                                   use_gpu=False, show_log=False)
        logger.info("[OCRPipeline] PaddleOCR ready")
    return _paddle


# ---------------------------------------------------------------------------
# Tesseract path auto-detection (Windows venv PATH isolation fix)
# ---------------------------------------------------------------------------

_tesseract_available = False
try:
    import pytesseract as _pytesseract
    _tess_path = (
        os.getenv("TESSERACT_PATH")
        or shutil.which("tesseract")
        or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        or r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    )
    if _tess_path:
        _pytesseract.pytesseract.tesseract_cmd = _tess_path
    _pytesseract.get_tesseract_version()
    _tesseract_available = True
    logger.info("[OCRPipeline] Tesseract available")
except Exception:
    pass


# ---------------------------------------------------------------------------
# OCRPipeline
# ---------------------------------------------------------------------------

class OCRPipeline:

    def process_pdf(self, pdf_bytes: bytes) -> tuple[str, list[dict], str]:
        """
        Render a scanned PDF to images then OCR each page.

        Returns:
            (raw_text, tables, method)
        """
        images = self._render_pdf(pdf_bytes)
        if not images:
            logger.warning("[OCRPipeline] No pages rendered from PDF")
            return "", [], "ocr_failed"

        all_text, method = self._ocr_images(images)
        return all_text, [], method   # tables from scanned docs are handled via text

    def process_image(self, image_bytes: bytes) -> tuple[str, list[dict], str]:
        """
        OCR a single image (PNG, JPG, etc.).

        Returns:
            (raw_text, tables, method)
        """
        text, method = self._ocr_single_image(image_bytes)
        return text, [], method

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render_pdf(self, pdf_bytes: bytes) -> list[bytes]:
        """Render each PDF page to a PNG image at 300 DPI using PyMuPDF."""
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            matrix = fitz.Matrix(_DPI / 72, _DPI / 72)
            images = []
            for i in range(len(doc)):
                try:
                    pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
                    images.append(pix.tobytes("png"))
                except Exception as e:
                    logger.warning("[OCRPipeline] Page %d render failed: %s", i + 1, e)
            doc.close()
            logger.info("[OCRPipeline] Rendered %d pages at %d DPI", len(images), _DPI)
            return images
        except ImportError:
            logger.error("[OCRPipeline] PyMuPDF not installed (pip install pymupdf)")
            return []
        except Exception as e:
            logger.error("[OCRPipeline] PDF render failed: %s", e)
            return []

    # ── OCR orchestration ─────────────────────────────────────────────────────

    def _ocr_images(self, images: list[bytes]) -> tuple[str, str]:
        """OCR a list of page images, returning combined text."""
        page_texts = []
        for i, img_bytes in enumerate(images, start=1):
            text, _ = self._ocr_single_image(img_bytes)
            if text.strip():
                page_texts.append(text)
            logger.debug("[OCRPipeline] p%d: %d chars", i, len(text))

        combined = "\n\n".join(page_texts)
        method = "ocr_paddle" if _paddle_available else ("ocr_tesseract" if _tesseract_available else "ocr_failed")
        return combined, method

    def _ocr_single_image(self, image_bytes: bytes) -> tuple[str, str]:
        """
        Run OCR on a single image.
        PaddleOCR → Tesseract fallback → empty string (never raises).
        """
        # Try PaddleOCR first
        if _paddle_available:
            try:
                text = self._paddle_ocr(image_bytes)
                if text.strip():
                    return text, "ocr_paddle"
                logger.debug("[OCRPipeline] PaddleOCR returned empty — trying Tesseract")
            except Exception as e:
                logger.warning("[OCRPipeline] PaddleOCR failed: %s", e)

        # Tesseract fallback
        if _tesseract_available:
            try:
                text = self._tesseract_ocr(image_bytes)
                if text.strip():
                    return text, "ocr_tesseract"
            except Exception as e:
                logger.warning("[OCRPipeline] Tesseract failed: %s", e)

        logger.warning("[OCRPipeline] All OCR methods failed — returning empty")
        return "", "ocr_failed"

    # ── PaddleOCR ─────────────────────────────────────────────────────────────

    def _paddle_ocr(self, image_bytes: bytes) -> str:
        from PIL import Image
        paddle = _get_paddle()
        if paddle is None:
            return ""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        result = paddle.ocr(arr, det=True, rec=True, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            if not line or len(line) < 2:
                continue
            text_info = line[1]
            text = text_info[0] if isinstance(text_info, (list, tuple)) else str(text_info)
            conf = float(text_info[1]) if isinstance(text_info, (list, tuple)) and len(text_info) > 1 else 1.0
            if text.strip() and conf > 0.4:
                lines.append(text.strip())
        return " ".join(lines)

    # ── Tesseract ─────────────────────────────────────────────────────────────

    def _tesseract_ocr(self, image_bytes: bytes) -> str:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # PSM 3: auto page segmentation; OEM 3: LSTM + legacy
        text = _pytesseract.image_to_string(img, config="--psm 3 --oem 3")
        return text.strip()
