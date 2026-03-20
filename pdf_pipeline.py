"""
pdf_pipeline.py
================
Production-grade PDF intelligence pipeline using NVIDIA Page Elements API.

Architecture:
    PDF file
        → Base64 encode
        → NVIDIA nemoretriever-page-elements-v3 API
        → Structured elements (text, table, title, figure_caption)
        → Normalize + deduplicate
        → Section-aware / table-aware chunking
        → Metadata attachment (bank, type, field_hint)
        → List[StructuredChunk]

Fallback:
    If NVIDIA API unavailable or USE_NVIDIA_PDF=false
        → pdfplumber basic text extraction
        → word-level chunking (existing behaviour)

Usage:
    from pdf_pipeline import process_pdf, PDFPipeline

    # Process one file
    chunks = process_pdf("data/hdfc_pdfs/hdfc_personal_loan.pdf")

    # Or use the pipeline object for batch processing
    pipeline = PDFPipeline()
    chunks = pipeline.process("data/sbi_pdfs/sbi_xpress_credit.pdf")
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (read at import time so module works standalone too)
# ---------------------------------------------------------------------------

def _cfg(key: str, default: Any) -> Any:
    """Lazy config read — avoids circular import if used before dotenv loads."""
    try:
        import config as c
        return getattr(c, key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Output data structure
# ---------------------------------------------------------------------------

@dataclass
class StructuredChunk:
    """
    Single knowledge unit produced by the PDF pipeline.

    Compatible with DocChunk in retrieval.py — both have the same
    fields that FAISS indexing needs (text, source, bank, doc_type).
    Extra fields here are stored in FAISS metadata for filtered retrieval.
    """
    text:            str
    source:          str           # original filename
    bank:            str           # "Axis" | "HDFC" | "ICICI" | "SBI" | "RBI" | …
    doc_type:        str           # "rule" | "table" | "paragraph" | "regulatory"
    chunk_id:        int  = 0
    similarity_score: float = 0.0  # populated at retrieval time
    page_number:     int  = 0
    section_title:   str  = ""     # section heading this chunk belongs to
    element_type:    str  = "text" # "text" | "table" | "title" | "figure_caption"
    field_hint:      str  = ""     # e.g. "income" | "cibil" | "age" (for rule chunks)
    structured_data: dict = field(default_factory=dict)  # raw table JSON if element_type=table
    content_hash:    str  = ""     # md5 of text — used for dedup / cache check

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]

    def to_doc_chunk(self):
        """Convert to DocChunk for FAISS indexing (backward compat)."""
        from services.retrieval import DocChunk
        return DocChunk(
            text=self.text,
            source=self.source,
            bank=self.bank,
            doc_type=self.doc_type,
            chunk_id=self.chunk_id,
            similarity_score=self.similarity_score,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Bank detection
# ---------------------------------------------------------------------------

_BANK_KEYWORDS: dict[str, list[str]] = {
    "Axis":  ["axis bank", "axis personal", "axisbank"],
    "HDFC":  ["hdfc bank", "hdfc personal", "hdfcbank"],
    "ICICI": ["icici bank", "icici personal", "icicidirect"],
    "SBI":   ["state bank", "sbi", "xpress credit", "sbi personal"],
    "RBI":   ["reserve bank", "rbi", "rbi guidelines", "master circular"],
    "Paisabazaar": ["paisabazaar"],
    "BankBazaar":  ["bankbazaar"],
}

def detect_bank(filename: str, content_sample: str = "") -> str:
    """
    Infer bank from filename and first ~500 chars of content.
    Returns canonical bank name or "Unknown".
    """
    text = (filename + " " + content_sample[:500]).lower()
    for bank, keywords in _BANK_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return bank

    # Path-based fallback (e.g. data/hdfc_pdfs/ → HDFC)
    path_lower = filename.lower()
    path_map = {
        "hdfc": "HDFC", "sbi": "SBI", "icici": "ICICI",
        "axis": "Axis", "rbi": "RBI",
    }
    for key, bank in path_map.items():
        if key in path_lower:
            return bank

    return "Unknown"


# ---------------------------------------------------------------------------
# Field hint detection
# ---------------------------------------------------------------------------

_FIELD_PATTERNS: list[tuple[str, str]] = [
    (r"income|salary|earning",                "monthly_income"),
    (r"cibil|credit\s*score|credit\s*rating", "credit_score"),
    (r"\bage\b|years\s*old|minimum.*age",     "age"),
    (r"dti|debt.to.income|emi.*income",       "dti_ratio"),
    (r"employment|salaried|self.employed",    "employment_type"),
    (r"experience|work.*month|tenure",        "work_experience_months"),
]

def detect_field_hint(text: str) -> str:
    """Return the most likely financial field this chunk is about."""
    text_lower = text.lower()
    for pattern, field in _FIELD_PATTERNS:
        if re.search(pattern, text_lower):
            return field
    return ""


# ---------------------------------------------------------------------------
# Text cleaning utilities
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalise whitespace, remove page artifacts."""
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common PDF artifacts
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"©\s*\d{4}.*?(Ltd|Limited|Bank)\b", "", text)
    return text.strip()


def _split_into_sentences(text: str, max_words: int) -> list[str]:
    """Split long text at sentence boundaries, respecting max_words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0
    for sent in sentences:
        word_count = len(sent.split())
        if current_count + word_count > max_words and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_count = word_count
        else:
            current.append(sent)
            current_count += word_count
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Table → text conversion
# ---------------------------------------------------------------------------

def table_to_text(table_data: dict | list | str) -> tuple[str, dict]:
    """
    Convert a table element from NVIDIA Page Elements into readable text
    plus a structured_data dict for storage.

    Returns: (readable_text, structured_dict)
    """
    if isinstance(table_data, str):
        return _clean_text(table_data), {"raw": table_data}

    if isinstance(table_data, list):
        # List of rows
        rows = []
        structured = {"rows": []}
        for row in table_data:
            if isinstance(row, (list, tuple)):
                row_text = " | ".join(str(cell).strip() for cell in row if str(cell).strip())
                rows.append(row_text)
                structured["rows"].append(list(row))
            elif isinstance(row, dict):
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items())
                rows.append(row_text)
                structured["rows"].append(row)
            else:
                rows.append(str(row))
        return _clean_text("\n".join(rows)), structured

    if isinstance(table_data, dict):
        # Common formats: {"headers": [...], "rows": [[...]]}
        headers = table_data.get("headers", table_data.get("columns", []))
        rows    = table_data.get("rows", table_data.get("data", []))

        lines: list[str] = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
        for row in rows:
            if isinstance(row, (list, tuple)):
                if headers and len(headers) == len(row):
                    lines.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, row)))
                else:
                    lines.append(" | ".join(str(cell) for cell in row))
            elif isinstance(row, dict):
                lines.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
        return _clean_text("\n".join(lines)), table_data

    return str(table_data), {"raw": str(table_data)}


# ---------------------------------------------------------------------------
# NVIDIA Page Elements API
# ---------------------------------------------------------------------------

_PAGE_ELEMENTS_URL = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v3"

def _call_page_elements_api(
    pdf_path: Path,
    api_key:  str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> list[dict]:
    """
    Send a PDF to the NVIDIA nemoretriever-page-elements-v3 API.

    REQUEST FORMAT (critical):
        multipart/form-data with the PDF file as "file" field.
        Do NOT use json= or base64 body — the endpoint is a CV model
        that expects a raw file upload, identical to an HTML <input type="file">.

    Retry policy:
        Retries on: timeout, 429 (rate limit), 5xx (server errors)
        Does NOT retry: 422 (bad request format), 401/403 (auth), 400

    Returns:
        Normalised list of page dicts: [{"page": N, "elements": [...]}]

    Raises:
        requests.exceptions.HTTPError for non-retryable HTTP errors.
        RuntimeError if all retries are exhausted.

    Response shapes handled:
        {"pages": [...]}                  → standard multi-page
        [{"page": N, "elements": [...]}]  → list directly
        {"elements": [...]}               → single-page
    """
    # Only Authorization + Accept — do NOT set Content-Type manually.
    # requests sets it automatically to multipart/form-data with the
    # correct boundary when files= is used.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("[PDFPipeline] Attempt %d/%d — %s", attempt, max_retries, pdf_path.name)

            with open(pdf_path, "rb") as fh:
                resp = requests.post(
                    _PAGE_ELEMENTS_URL,
                    headers=headers,
                    files={"file": (pdf_path.name, fh, "application/pdf")},
                    timeout=60,
                )

            # ── Detailed error logging BEFORE raise_for_status ────────────
            if resp.status_code == 422:
                logger.error(
                    "[PDFPipeline] 422 Unprocessable Entity — "
                    "request format rejected by NVIDIA API. "
                    "Response body: %s", resp.text[:500],
                )
                # 422 means the server understood the request but rejected
                # its content — retrying with same payload won't help.
                raise requests.exceptions.HTTPError(
                    f"422 Unprocessable Entity: {resp.text[:200]}",
                    response=resp,
                )

            if resp.status_code == 401:
                logger.error("[PDFPipeline] 401 Unauthorized — check NVIDIA_API_KEY")
                raise requests.exceptions.HTTPError("401 Unauthorized", response=resp)

            if resp.status_code == 400:
                logger.error(
                    "[PDFPipeline] 400 Bad Request — %s", resp.text[:300]
                )
                raise requests.exceptions.HTTPError(
                    f"400 Bad Request: {resp.text[:200]}", response=resp
                )

            resp.raise_for_status()   # catches any other 4xx/5xx

            data = resp.json()

            # ── Normalise response shape ──────────────────────────────────
            if isinstance(data, list):
                pages = data
            elif "pages" in data:
                pages = data["pages"]
            elif "elements" in data:
                pages = [{"page": 1, "elements": data["elements"]}]
            else:
                # Unknown shape — log and return as-is for best-effort parsing
                logger.warning(
                    "[PDFPipeline] Unexpected response shape: %s",
                    list(data.keys()) if isinstance(data, dict) else type(data),
                )
                pages = data

            # ── Log success ───────────────────────────────────────────────
            total_elements = sum(len(p.get("elements", [])) for p in pages) if isinstance(pages, list) else 0
            type_counts: dict[str, int] = {}
            if isinstance(pages, list):
                for p in pages:
                    for e in p.get("elements", []):
                        t = e.get("type", "unknown")
                        type_counts[t] = type_counts.get(t, 0) + 1

            logger.info(
                "[PDFPipeline] NVIDIA extraction SUCCESS — "
                "%d pages, %d elements, types=%s",
                len(pages) if isinstance(pages, list) else "?",
                total_elements,
                type_counts,
            )
            return pages

        except requests.exceptions.Timeout:
            last_err = TimeoutError(f"Timed out on attempt {attempt}")
            logger.warning("[PDFPipeline] Timeout on attempt %d", attempt)

        except requests.exceptions.HTTPError as e:
            last_err = e
            status = e.response.status_code if e.response is not None else 0
            # Only retry on rate limit (429) or server errors (5xx)
            if status in (429, 502, 503, 504):
                wait = retry_delay * attempt
                logger.warning(
                    "[PDFPipeline] HTTP %d on attempt %d — retrying in %.1fs",
                    status, attempt, wait,
                )
                time.sleep(wait)
            else:
                # 400, 401, 403, 422 etc. — retrying won't help
                logger.error(
                    "[PDFPipeline] Non-retryable HTTP %d — aborting retries: %s",
                    status, e,
                )
                raise

        except Exception as e:
            last_err = e
            logger.warning(
                "[PDFPipeline] Unexpected error on attempt %d: %s",
                attempt, e,
            )

        if attempt < max_retries:
            time.sleep(retry_delay)

    raise RuntimeError(
        f"NVIDIA Page Elements API failed after {max_retries} attempts: {last_err}"
    )


# ---------------------------------------------------------------------------
# Element normalizer
# ---------------------------------------------------------------------------

def _normalise_elements(raw_pages: list[dict]) -> list[dict]:
    """
    Convert the raw NVIDIA API response into a flat, consistent element list.
    Each element: {type, content, page, bbox}
    """
    elements: list[dict] = []
    for page_data in raw_pages:
        page_num = page_data.get("page", 0)
        for elem in page_data.get("elements", []):
            elements.append({
                "type":    elem.get("type", "text"),
                "content": elem.get("content", ""),
                "page":    page_num,
                "bbox":    elem.get("bbox", []),
            })
    return elements


# ---------------------------------------------------------------------------
# Section-aware chunker
# ---------------------------------------------------------------------------

def _chunk_elements(
    elements: list[dict],
    bank:     str,
    source:   str,
    section_max_words: int = 400,
    table_max_words:   int = 300,
) -> list[StructuredChunk]:
    """
    Convert a flat element list into StructuredChunks.

    Strategy:
    - title elements → update current section heading
    - text/list_item → accumulate into section buffer, flush at max_words
    - table → convert to text immediately (one chunk per table)
    - figure_caption → short chunk on its own
    """
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
        if len(text.split()) < 5:   # skip noise-level fragments
            buffer_words = []
            return
        chunks.append(StructuredChunk(
            text=text,
            source=source,
            bank=bank,
            doc_type="rule" if detect_field_hint(text) else "paragraph",
            chunk_id=chunk_id,
            page_number=buffer_page,
            section_title=current_section,
            element_type="text",
            field_hint=detect_field_hint(text),
        ))
        chunk_id += 1
        buffer_words = []

    for elem in elements:
        etype   = elem.get("type", "text")
        content = elem.get("content", "")
        page    = elem.get("page", 0)

        # ── Title / section heading ────────────────────────────────────────
        if etype in ("title", "heading"):
            flush_buffer()
            current_section = _clean_text(str(content)) if content else ""
            # Also store the title itself as a tiny chunk (helps retrieval)
            if current_section and len(current_section.split()) >= 2:
                chunks.append(StructuredChunk(
                    text=current_section,
                    source=source, bank=bank,
                    doc_type="paragraph", chunk_id=chunk_id,
                    page_number=page, section_title="",
                    element_type="title",
                ))
                chunk_id += 1

        # ── Table ─────────────────────────────────────────────────────────
        elif etype == "table":
            flush_buffer()
            readable, structured = table_to_text(content)
            if not readable.strip():
                continue

            # Split large tables into sub-chunks
            sub_texts = _split_into_sentences(readable, table_max_words)
            for sub in sub_texts:
                if len(sub.split()) < 3:
                    continue
                chunks.append(StructuredChunk(
                    text=sub,
                    source=source, bank=bank,
                    doc_type="table", chunk_id=chunk_id,
                    page_number=page, section_title=current_section,
                    element_type="table",
                    field_hint=detect_field_hint(sub),
                    structured_data=structured,
                ))
                chunk_id += 1

        # ── Figure caption ────────────────────────────────────────────────
        elif etype == "figure_caption":
            flush_buffer()
            text = _clean_text(str(content))
            if text and len(text.split()) >= 4:
                chunks.append(StructuredChunk(
                    text=text,
                    source=source, bank=bank,
                    doc_type="paragraph", chunk_id=chunk_id,
                    page_number=page, section_title=current_section,
                    element_type="figure_caption",
                ))
                chunk_id += 1

        # ── Regular text / list_item ──────────────────────────────────────
        else:
            text = _clean_text(str(content))
            if not text:
                continue
            words = text.split()
            if not buffer_page:
                buffer_page = page

            # Flush if adding this text would exceed section_max_words
            if len(buffer_words) + len(words) > section_max_words:
                flush_buffer()
                buffer_page = page

            buffer_words.extend(words)

    flush_buffer()  # final flush
    return chunks


# ---------------------------------------------------------------------------
# Fallback: basic pdfplumber extraction
# ---------------------------------------------------------------------------

def _fallback_extract(pdf_path: Path, bank: str) -> list[StructuredChunk]:
    """
    Plain text extraction using pdfplumber (no NVIDIA API).
    Used when USE_NVIDIA_PDF=false or when the API call fails.
    Produces word-level overlapping chunks (same as the old retrieval.py behaviour).
    """
    max_words   = _cfg("PDF_SECTION_CHUNK_MAX", 512)
    chunk_overlap = _cfg("CHUNK_OVERLAP", 64)

    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
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
        text = " ".join(words[start:end])
        text = _clean_text(text)
        if len(text.split()) >= 5:
            chunks.append(StructuredChunk(
                text=text,
                source=pdf_path.name,
                bank=bank,
                doc_type="pdf",
                chunk_id=chunk_id,
                field_hint=detect_field_hint(text),
            ))
            chunk_id += 1
        if end == len(words):
            break
        start += max_words - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PDFPipeline:
    """
    Orchestrates PDF → structured chunks.

    Behaviour:
        USE_NVIDIA_PDF=true  → NVIDIA Page Elements API → structured chunks
        USE_NVIDIA_PDF=false → pdfplumber fallback → word-level chunks

    Caching:
        Processed results are saved to processed_data/<stem>.json.
        On subsequent runs, if the file hash matches, the cache is reused
        (avoids expensive API calls for unchanged PDFs).
    """

    def __init__(
        self,
        api_key:    str  | None = None,
        use_nvidia: bool | None = None,
        processed_dir: Path | None = None,
    ):
        self.api_key       = api_key       or _cfg("NVIDIA_API_KEY", "")
        self.use_nvidia    = use_nvidia    if use_nvidia is not None else _cfg("USE_NVIDIA_PDF", False)
        self.processed_dir = processed_dir or Path(_cfg("PROCESSED_DATA_DIR", "processed_data"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._max_retries  = int(_cfg("PDF_API_MAX_RETRIES", 3))
        self._retry_delay  = float(_cfg("PDF_API_RETRY_DELAY", 2.0))
        self._section_max  = int(_cfg("PDF_SECTION_CHUNK_MAX", 400))
        self._table_max    = int(_cfg("PDF_TABLE_CHUNK_MAX",   300))

    # ── Cache helpers ──────────────────────────────────────────────────────

    def _file_hash(self, pdf_path: Path) -> str:
        return hashlib.md5(pdf_path.read_bytes()).hexdigest()

    def _cache_path(self, pdf_path: Path) -> Path:
        return self.processed_dir / (pdf_path.stem + ".json")

    def _load_cache(self, pdf_path: Path) -> list[StructuredChunk] | None:
        cache = self._cache_path(pdf_path)
        if not cache.exists():
            return None
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("file_hash") != self._file_hash(pdf_path):
                return None   # file changed — invalidate
            return [StructuredChunk(**c) for c in data["chunks"]]
        except Exception as e:
            logger.warning("[PDFPipeline] Cache load failed: %s", e)
            return None

    def _save_cache(self, pdf_path: Path, chunks: list[StructuredChunk]) -> None:
        cache = self._cache_path(pdf_path)
        data  = {
            "file_hash": self._file_hash(pdf_path),
            "source":    pdf_path.name,
            "bank":      chunks[0].bank if chunks else "Unknown",
            "chunk_count": len(chunks),
            "chunks":    [c.to_dict() for c in chunks],
        }
        cache.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("[PDFPipeline] Saved %d chunks → %s", len(chunks), cache.name)

    # ── Main entry point ───────────────────────────────────────────────────

    def process(self, pdf_path: str | Path) -> list[StructuredChunk]:
        """
        Process a single PDF file → list of StructuredChunk objects.

        Checks cache first. Falls back to pdfplumber if NVIDIA API unavailable.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # ── Cache check ───────────────────────────────────────────────────
        cached = self._load_cache(pdf_path)
        if cached:
            logger.info("[PDFPipeline] Cache hit: %s (%d chunks)", pdf_path.name, len(cached))
            return cached

        # ── Detect bank from filename ──────────────────────────────────────
        bank = detect_bank(pdf_path.name)

        # ── Route to NVIDIA or fallback ────────────────────────────────────
        if self.use_nvidia and self.api_key:
            chunks = self._process_nvidia(pdf_path, bank)
        else:
            if self.use_nvidia and not self.api_key:
                logger.warning(
                    "[PDFPipeline] USE_NVIDIA_PDF=true but NVIDIA_API_KEY not set — using fallback"
                )
            chunks = _fallback_extract(pdf_path, bank)

        if not chunks:
            logger.warning("[PDFPipeline] No chunks extracted from %s", pdf_path.name)
            return []

        # Re-number chunk IDs sequentially
        for i, c in enumerate(chunks):
            c.chunk_id = i

        self._save_cache(pdf_path, chunks)
        return chunks

    def _process_nvidia(self, pdf_path: Path, bank: str) -> list[StructuredChunk]:
        """
        Call NVIDIA Page Elements API and convert response to structured chunks.
        Falls back to pdfplumber only on real failures (not request format errors,
        which are surfaced immediately so the caller can fix config).
        """
        try:
            raw_pages = _call_page_elements_api(
                pdf_path,                        # pass Path, not bytes
                api_key=self.api_key,
                max_retries=self._max_retries,
                retry_delay=self._retry_delay,
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (401, 403):
                # Auth failures — don't fall back silently, raise so the user knows
                raise RuntimeError(
                    f"NVIDIA API authentication failed (HTTP {status}). "
                    f"Check NVIDIA_API_KEY in your .env file."
                ) from e
            # 422 / 400 / other client errors — fall back with clear warning
            logger.warning(
                "[PDFPipeline] NVIDIA API request error (HTTP %d) for %s: %s "
                "— falling back to pdfplumber. Check pdf_pipeline.py if this persists.",
                status, pdf_path.name, e,
            )
            return _fallback_extract(pdf_path, bank)
        except Exception as e:
            logger.warning(
                "[PDFPipeline] NVIDIA API failed for %s: %s — falling back to pdfplumber",
                pdf_path.name, e,
            )
            return _fallback_extract(pdf_path, bank)

        elements = _normalise_elements(raw_pages)
        logger.info("[PDFPipeline] %s: %d elements across %d pages",
                    pdf_path.name, len(elements), len(raw_pages))

        # Detect bank from content if filename alone wasn't enough
        text_sample = " ".join(
            str(e.get("content", ""))[:200]
            for e in elements[:10]
            if e.get("type") in ("text", "title")
        )
        if bank == "Unknown":
            bank = detect_bank(pdf_path.name, text_sample)

        return _chunk_elements(
            elements, bank, pdf_path.name,
            section_max_words=self._section_max,
            table_max_words=self._table_max,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def process_pdf(file_path: str) -> list[dict]:
    """
    Public API — process a single PDF and return list of chunk dicts.

    Example:
        chunks = process_pdf("data/hdfc_pdfs/hdfc_personal_loan.pdf")
        for c in chunks:
            print(c["bank"], c["doc_type"], c["text"][:80])
    """
    pipeline = PDFPipeline()
    chunks   = pipeline.process(file_path)
    return [c.to_dict() for c in chunks]


def test_single_pdf(file_path: str, api_key: str | None = None) -> None:
    """
    Quick smoke-test: process one PDF and print a summary.
    Run from project root:
        python -c "from pdf_pipeline import test_single_pdf; test_single_pdf('data/hdfc_pdfs/hdfc_personal_loan.pdf')"

    Args:
        file_path : path to any .pdf file
        api_key   : NVIDIA API key (reads NVIDIA_API_KEY from .env if not supplied)
    """
    import os
    from pathlib import Path

    key = api_key or os.getenv("NVIDIA_API_KEY", "")

    print(f"[test_single_pdf] File:        {file_path}")
    print(f"[test_single_pdf] API key set: {bool(key)}")
    print(f"[test_single_pdf] USE_NVIDIA:  {os.getenv('USE_NVIDIA_PDF', 'false')}")
    print()

    # Direct API test (bypasses cache and pipeline logic)
    if key:
        print("[test_single_pdf] Testing raw NVIDIA API call…")
        try:
            pages = _call_page_elements_api(
                Path(file_path),
                api_key=key,
                max_retries=1,    # single attempt for quick smoke test
                retry_delay=1.0,
            )
            type_counts: dict[str, int] = {}
            for p in pages:
                for e in p.get("elements", []):
                    t = e.get("type", "?")
                    type_counts[t] = type_counts.get(t, 0) + 1
            print(f"[test_single_pdf] SUCCESS — pages={len(pages)}, elements={type_counts}")

            # Show first 3 elements
            print("[test_single_pdf] First elements:")
            for p in pages[:2]:
                for e in p.get("elements", [])[:3]:
                    preview = str(e.get("content", ""))[:80]
                    print(f"  [{e.get('type')}] {preview}")

        except Exception as e:
            print(f"[test_single_pdf] FAILED: {e}")
    else:
        print("[test_single_pdf] Skipping raw API test — NVIDIA_API_KEY not set")

    print()
    print("[test_single_pdf] Running full pipeline (with fallback)…")
    pipeline = PDFPipeline(api_key=key or None)
    chunks = pipeline.process(file_path)
    print(f"[test_single_pdf] Total chunks: {len(chunks)}")
    if chunks:
        type_counts2: dict[str, int] = {}
        for c in chunks:
            type_counts2[c.doc_type] = type_counts2.get(c.doc_type, 0) + 1
        print(f"[test_single_pdf] Chunk types: {type_counts2}")
        print(f"[test_single_pdf] Bank:        {chunks[0].bank}")
        print(f"[test_single_pdf] Sample chunk:")
        print(f"  [{chunks[0].doc_type}] {chunks[0].text[:120]}")
