"""
pdf_pipeline/chunker.py
========================
Convert ExtractedDocument pages into retrieval-ready DocChunks.

Design decisions:
  - Target 300–500 words per chunk (sweet spot for embedding models)
  - 75-word overlap to prevent context loss at chunk boundaries
  - Tables become their own chunks (natural language version)
  - Section titles are tracked and inherited by child chunks
  - field_hint tagging enables query-aware boosting in retrieval
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field

from pdf_pipeline.extractor import ExtractedDocument, PageContent

logger = logging.getLogger(__name__)

CHUNK_WORDS   = 400     # target words per chunk
OVERLAP_WORDS = 75      # words of overlap between consecutive chunks
MIN_CHUNK_WORDS = 20    # discard chunks shorter than this


@dataclass
class DocumentChunk:
    text: str
    source: str
    bank: str
    doc_type: str
    chunk_id: int = 0
    page_number: int = 0
    section_title: str = ""
    element_type: str = "text"   # "text" | "table" | "title"
    field_hint: str = ""
    extraction_method: str = ""
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.md5(self.text.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Field hint patterns — used for query-aware boosting in retrieval
# ---------------------------------------------------------------------------

_FIELD_PATTERNS: list[tuple[str, str]] = [
    (r"income|salary|earning|ctc|lpa",                    "monthly_income"),
    (r"cibil|credit\s*score|credit\s*rating",             "credit_score"),
    (r"\bage\b|years\s*old|minimum\s*age|maximum\s*age",  "age"),
    (r"dti|debt.to.income|emi.*income|obligations",       "dti_ratio"),
    (r"employment|salaried|self.employed|professional",   "employment_type"),
    (r"experience|work.*month|tenure|service",            "work_experience_months"),
    (r"interest\s*rate|roi|rate\s*of\s*interest",         "interest_rate"),
    (r"loan\s*amount|principal|disburs",                  "loan_amount"),
    (r"processing\s*fee|charges|fees",                    "fees"),
    (r"repayment|tenure|emi|instalment",                  "tenure"),
]

def detect_field_hint(text: str) -> str:
    tl = text.lower()
    for pattern, field_name in _FIELD_PATTERNS:
        if re.search(pattern, tl):
            return field_name
    return ""


_SECTION_PATTERNS = re.compile(
    r"^(eligibility|criteria|features?|benefits?|fees?|charges?|"
    r"interest rate|documentation|documents\s*required|terms?\s*and\s*conditions?|"
    r"who\s*can\s*apply|about|overview|summary)\b",
    re.IGNORECASE | re.MULTILINE,
)

def _detect_section_title(text: str) -> str:
    """Try to extract a section header from text."""
    lines = text.strip().split("\n")
    first_line = lines[0].strip()
    if len(first_line) < 80 and re.search(_SECTION_PATTERNS, first_line):
        return first_line
    return ""


# ---------------------------------------------------------------------------
# Main chunking function
# ---------------------------------------------------------------------------

def chunk_document(doc: ExtractedDocument) -> list[DocumentChunk]:
    """
    Convert an ExtractedDocument into a flat list of DocumentChunks.

    Processing order per page:
      1. Page text → sentence-aware word-window chunking with overlap
      2. Tables from that page → each table becomes its own chunk
    """
    chunks: list[DocumentChunk] = []
    chunk_id = 0
    current_section = ""

    for page in doc.pages:
        # Update section title if detectable from page start
        detected = _detect_section_title(page.text)
        if detected:
            current_section = detected

        # Text chunks
        text_chunks = _split_text(page.text, CHUNK_WORDS, OVERLAP_WORDS)
        for tc in text_chunks:
            if len(tc.split()) < MIN_CHUNK_WORDS:
                continue
            chunks.append(DocumentChunk(
                text=tc,
                source=doc.source_path,
                bank=doc.bank,
                doc_type=_classify_doc_type(tc, doc.source_path),
                chunk_id=chunk_id,
                page_number=page.page_number,
                section_title=current_section,
                element_type="text",
                field_hint=detect_field_hint(tc),
                extraction_method=page.extraction_method,
            ))
            chunk_id += 1

        # Table chunks — each table becomes one chunk (its natural language form)
        for table in page.tables:
            table_text = table.get("raw_text", "").strip()
            if len(table_text.split()) < MIN_CHUNK_WORDS:
                continue
            # Prepend section context so the table chunk is self-contained
            if current_section:
                table_text = f"{current_section}: {table_text}"
            chunks.append(DocumentChunk(
                text=table_text,
                source=doc.source_path,
                bank=doc.bank,
                doc_type="table",
                chunk_id=chunk_id,
                page_number=page.page_number,
                section_title=current_section,
                element_type="table",
                field_hint=detect_field_hint(table_text),
                extraction_method=table.get("method", "table"),
            ))
            chunk_id += 1

    # Deduplicate by content hash
    seen: set[str] = set()
    unique: list[DocumentChunk] = []
    for c in chunks:
        if c.content_hash not in seen:
            seen.add(c.content_hash)
            unique.append(c)

    if len(unique) < len(chunks):
        logger.info("[Chunker] Deduplicated %d → %d chunks", len(chunks), len(unique))

    logger.info(
        "[Chunker] %s — %d chunks  bank=%s",
        doc.source_path, len(unique), doc.bank,
    )
    return unique


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def _split_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    """
    Split text into word-window chunks with sentence-boundary awareness.

    We split on sentence boundaries where possible to avoid cutting mid-sentence.
    The overlap carries the last `overlap_words` words of the previous chunk
    forward, preserving cross-sentence context.
    """
    if not text.strip():
        return []

    # Split into sentences first
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks: list[str] = []
    current_words: list[str] = []
    current_count = 0

    for sentence in sentences:
        s_words = sentence.split()
        if current_count + len(s_words) > chunk_words and current_words:
            chunks.append(" ".join(current_words))
            # Keep overlap
            overlap = current_words[-overlap_words:] if len(current_words) > overlap_words else current_words[:]
            current_words = overlap + s_words
            current_count = len(current_words)
        else:
            current_words.extend(s_words)
            current_count += len(s_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ---------------------------------------------------------------------------
# Document type classification
# ---------------------------------------------------------------------------

_DOC_TYPE_PATTERNS = {
    "eligibility":  re.compile(r"eligib|criteria|qualify|minimum|required|age|income|salary", re.I),
    "interest_rate": re.compile(r"interest\s*rate|roi|% p\.?a|per\s*annum|bps", re.I),
    "table":        re.compile(r"\|"),
    "regulatory":   re.compile(r"rbi|reserve\s*bank|circular|guideline|regulation|compliance", re.I),
    "comparison":   re.compile(r"vs\.?\s|compared?\s*to|better\s*than|paisabazaar|bankbazaar", re.I),
    "fees":         re.compile(r"processing\s*fee|charge|penalty|foreclosure|prepayment", re.I),
}

def _classify_doc_type(text: str, source: str) -> str:
    source_lower = source.lower()
    if "paisabazaar" in source_lower or "bankbazaar" in source_lower:
        return "comparison"
    if "rbi" in source_lower:
        return "regulatory"
    for doc_type, pattern in _DOC_TYPE_PATTERNS.items():
        if pattern.search(text):
            return doc_type
    return "pdf"
