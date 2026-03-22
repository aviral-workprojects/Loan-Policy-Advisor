"""
document_pipeline — Upload & Intelligent Processing System
===========================================================
Adds document upload capability to the Loan Advisor.

Users upload a PDF or image; the system extracts structured financial
signals (income, CIBIL, age, employment type), merges them with the
existing knowledge base, and returns a full eligibility decision.

Public API:
    from document_pipeline import DocumentProcessor
    result = DocumentProcessor().process(file_bytes, filename, query)
"""

from document_pipeline.router    import DocumentProcessor
from document_pipeline.extractor import ExtractedProfile

__all__ = ["DocumentProcessor", "ExtractedProfile"]
