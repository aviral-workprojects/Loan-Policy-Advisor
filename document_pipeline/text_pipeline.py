"""
document_pipeline/text_pipeline.py
=====================================
Text-layer PDF extraction pipeline.

Used when a PDF has selectable/copyable text (digitally generated).
This covers ~95% of bank statements, salary slips, and ITR documents.

Stack:
  pdfplumber  → primary  (best character accuracy, table extraction)
  PyMuPDF     → fallback (page-level, when pdfplumber fails)
  Camelot     → table extraction (lattice + stream modes)
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MIN_TEXT_CHARS_PER_PAGE = 50
_TEXT_LAYER_THRESHOLD    = 0.5    # fraction of sample pages that must have text


class TextPipeline:

    # ── Type detection ───────────────────────────────────────────────────────

    @staticmethod
    def is_text_layer(pdf_bytes: bytes) -> bool:
        """
        Return True if the PDF has a meaningful text layer.
        Samples up to 5 pages; if ≥50% have ≥50 chars, it's text-layer.
        """
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                sample = pdf.pages[:5]
                text_pages = sum(
                    1 for p in sample
                    if len((p.extract_text() or "").strip()) >= _MIN_TEXT_CHARS_PER_PAGE
                )
                result = (text_pages / max(len(sample), 1)) >= _TEXT_LAYER_THRESHOLD
                logger.info("[TextPipeline] text_layer=%s  (%d/%d sample pages have text)",
                            result, text_pages, len(sample))
                return result
        except Exception as e:
            logger.warning("[TextPipeline] is_text_layer check failed: %s — assuming text", e)
            return True

    # ── Main extraction ──────────────────────────────────────────────────────

    def process(self, pdf_bytes: bytes) -> tuple[str, list[dict], str]:
        """
        Extract text and tables from a text-layer PDF.

        Returns:
            (raw_text: str, tables: list[dict], method: str)
        """
        text, method = self._extract_text(pdf_bytes)
        tables       = self._extract_tables(pdf_bytes)

        # Append table natural language to text so extractor can parse table data
        if tables:
            nl_tables = "\n".join(t.get("raw_text", "") for t in tables if t.get("raw_text"))
            if nl_tables:
                text = text + "\n\n[TABLE DATA]\n" + nl_tables

        return text, tables, method

    # ── Text extraction ──────────────────────────────────────────────────────

    def _extract_text(self, pdf_bytes: bytes) -> tuple[str, str]:
        # Try pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                pages_text = []
                empty_pages = 0
                for page in pdf.pages:
                    t = (page.extract_text() or "").strip()
                    if len(t) >= _MIN_TEXT_CHARS_PER_PAGE:
                        pages_text.append(t)
                    else:
                        empty_pages += 1

                if pages_text:
                    combined = "\n\n".join(pages_text)
                    logger.info("[TextPipeline] pdfplumber: %d chars  %d empty pages",
                                len(combined), empty_pages)
                    # Fill empty pages with PyMuPDF fallback
                    if empty_pages > 0:
                        combined = self._fill_empty_pages_pymupdf(pdf_bytes, combined)
                    return combined, "pdfplumber"

        except Exception as e:
            logger.warning("[TextPipeline] pdfplumber failed: %s", e)

        # Full PyMuPDF fallback
        return self._extract_pymupdf(pdf_bytes)

    def _extract_pymupdf(self, pdf_bytes: bytes) -> tuple[str, str]:
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = [doc[i].get_text("text").strip() for i in range(len(doc))]
            doc.close()
            text = "\n\n".join(p for p in pages_text if p)
            logger.info("[TextPipeline] PyMuPDF: %d chars", len(text))
            return text, "pymupdf"
        except Exception as e:
            logger.error("[TextPipeline] PyMuPDF failed: %s", e)
            return "", "failed"

    def _fill_empty_pages_pymupdf(self, pdf_bytes: bytes, existing_text: str) -> str:
        """Try PyMuPDF on all pages and append anything not already in text."""
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extras = []
            for i in range(len(doc)):
                t = doc[i].get_text("text").strip()
                if t and t not in existing_text:
                    extras.append(t)
            doc.close()
            if extras:
                return existing_text + "\n\n" + "\n\n".join(extras)
        except Exception:
            pass
        return existing_text

    # ── Table extraction ─────────────────────────────────────────────────────

    def _extract_tables(self, pdf_bytes: bytes) -> list[dict]:
        """
        Extract tables from a PDF using Camelot (lattice → stream) then
        pdfplumber as a last resort.
        """
        # Write to temp file (Camelot requires a file path, not bytes)
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp.close()

        tables = []
        try:
            tables = self._camelot_tables(tmp.name)
            if not tables:
                tables = self._pdfplumber_tables(pdf_bytes)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

        logger.info("[TextPipeline] Extracted %d tables", len(tables))
        return tables

    def _camelot_tables(self, pdf_path: str) -> list[dict]:
        try:
            import camelot
            results = []
            for flavor in ("lattice", "stream"):
                try:
                    tbls = camelot.read_pdf(pdf_path, pages="all", flavor=flavor,
                                            suppress_stdout=True)
                    for tbl in tbls:
                        if tbl.accuracy < 40 or tbl.df.empty:
                            continue
                        results.append(self._df_to_dict(tbl.df, flavor))
                    if results:
                        return results
                except Exception as e:
                    logger.debug("[TextPipeline] Camelot %s failed: %s", flavor, e)
            return results
        except ImportError:
            return []

    def _pdfplumber_tables(self, pdf_bytes: bytes) -> list[dict]:
        try:
            import pdfplumber
            results = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    for raw in (page.extract_tables() or []):
                        if not raw or len(raw) < 2:
                            continue
                        headers = [str(c or "").strip() for c in raw[0]]
                        rows    = [[str(c or "").strip() for c in row] for row in raw[1:]]
                        results.append({
                            "headers":  headers,
                            "rows":     rows,
                            "raw_text": self._table_to_nl(headers, rows),
                            "method":   "pdfplumber",
                        })
            return results
        except Exception as e:
            logger.debug("[TextPipeline] pdfplumber tables failed: %s", e)
            return []

    @staticmethod
    def _df_to_dict(df, method: str) -> dict:
        import re as _re
        headers = [str(c).strip() for c in df.columns.tolist()]
        if all(h.isdigit() or h.startswith("Unnamed") for h in headers):
            headers = [str(c).strip() for c in df.iloc[0].tolist()]
            rows    = df.iloc[1:].values.tolist()
        else:
            rows = df.values.tolist()
        headers = [h for h in headers if h]
        rows    = [[str(c).strip() for c in row] for row in rows if any(str(c).strip() for c in row)]
        nl = TextPipeline._table_to_nl(headers, rows)
        return {"headers": headers, "rows": rows, "raw_text": nl, "method": f"camelot_{method}"}

    @staticmethod
    def _table_to_nl(headers: list[str], rows: list[list[str]]) -> str:
        """Convert table to natural language for the entity extractor."""
        if not rows:
            return ""
        parts = []
        is_kv = len(headers) == 2 and all(len(r) == 2 for r in rows)
        if is_kv:
            for row in rows:
                k, v = row[0].strip(), row[1].strip()
                if k and v:
                    parts.append(f"{k}: {v}.")
        else:
            for row in rows:
                cells = [f"{h}: {v}" for h, v in zip(headers, row) if h.strip() and v.strip()]
                if cells:
                    parts.append(" | ".join(cells) + ".")
        return " ".join(parts)
