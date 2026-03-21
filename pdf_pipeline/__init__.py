"""
pdf_pipeline — Hybrid document intelligence for the Loan Advisor system.

Public API:
  from pdf_pipeline import extract_pdf, chunk_document, RetrievalService
  from pdf_pipeline import understand_query, EligibilityEngine
"""

from pdf_pipeline.extractor        import extract_pdf, is_scanned_pdf, ExtractedDocument
from pdf_pipeline.chunker          import chunk_document, DocumentChunk
from pdf_pipeline.retriever        import RetrievalService, DocChunk
from pdf_pipeline.query_understanding import understand_query, QuerySignals
from pdf_pipeline.eligibility_engine  import EligibilityEngine, EligibilityReport
from pdf_pipeline.table_extractor  import table_availability
from pdf_pipeline.ocr              import ocr_availability
from pdf_pipeline.embeddings       import get_embedder

__all__ = [
    "extract_pdf", "is_scanned_pdf", "ExtractedDocument",
    "chunk_document", "DocumentChunk",
    "RetrievalService", "DocChunk",
    "understand_query", "QuerySignals",
    "EligibilityEngine", "EligibilityReport",
    "table_availability", "ocr_availability",
    "get_embedder",
]
