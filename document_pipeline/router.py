"""
document_pipeline/router.py
============================
Top-level orchestrator for the document upload pipeline.

Flow:
  file bytes
    → validate (size, type)
    → detect: text-layer PDF  OR  scanned/image
    → TEXT PIPELINE   (pdfplumber → PyMuPDF → Camelot)
      OR
      OCR PIPELINE    (PaddleOCR → Tesseract fallback)
    → entity extractor  (regex + heuristic NLP)
    → data fusion       (doc data > user query > KB defaults)
    → rule engine       (reuse existing YAML rules)
    → LLM explanation   (reuse existing LLM service)
    → DocumentResult

Design principles:
  - Never fails completely (always falls back)
  - Document data has highest trust over query text
  - NVIDIA CV models are NOT used (SSL issues on Windows, per prior debugging)
  - PaddleOCR → Tesseract is the reliable local OCR stack
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_MAX_UPLOAD_MB   = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024
_UPLOAD_DIR      = Path(os.getenv("UPLOAD_DIR", "uploads/user_docs"))
_PROCESSED_DIR   = _UPLOAD_DIR / "processed"
_ENABLE_UPLOAD   = os.getenv("ENABLE_DOCUMENT_UPLOAD", "true").lower() == "true"

_ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
_ALLOWED_MIMETYPES  = {
    "application/pdf", "image/png", "image/jpeg",
    "image/tiff", "image/bmp", "image/webp",
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DocumentResult:
    """Full response returned by POST /analyze-document."""
    decision:          str
    summary:           str
    detailed_explanation: str
    recommendations:   list[str]
    extracted_data:    dict           # raw extracted profile fields
    eligible_banks:    list[str]
    rejected_banks:    list[str]
    banks_compared:    list[dict]
    rule_results:      list[dict]
    confidence:        float
    sources_cited:     list[str]
    reasoning_context: dict
    validation_issues: list[dict]
    best_bank:         str | None = None
    best_bank_reason:  str | None = None
    processing_method: str = ""       # "text_layer" | "ocr_paddle" | "ocr_tesseract"
    latency_ms:        float = 0.0
    file_hash:         str = ""


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """
    Entry point for the document upload pipeline.

    Usage:
        processor = DocumentProcessor(pipeline)
        result = processor.process(file_bytes, filename, optional_query)
    """

    def __init__(self, pipeline=None):
        """
        Args:
            pipeline: LoanAdvisorPipeline instance (injected from API).
                      If None, a new instance is created (useful for testing).
        """
        self._pipeline = pipeline
        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public entry point ──────────────────────────────────────────────────

    def process(
        self,
        file_bytes: bytes,
        filename:   str,
        query:      str = "",
    ) -> DocumentResult:
        """
        Process an uploaded document and return a loan eligibility result.

        Args:
            file_bytes: raw file content from the upload
            filename:   original filename (used to detect extension)
            query:      optional natural language query from the user
                        (merged with extracted document data)

        Returns:
            DocumentResult with full eligibility decision
        """
        if not _ENABLE_UPLOAD:
            raise RuntimeError("Document upload is disabled. Set ENABLE_DOCUMENT_UPLOAD=true.")

        t0 = time.perf_counter()

        # ── 1. Validate ────────────────────────────────────────────────────
        self._validate(file_bytes, filename)

        file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        logger.info("[DocPipeline] Processing: %s  %.1f KB  hash=%s",
                    filename, len(file_bytes) / 1024, file_hash)

        # ── 2. Check cache ─────────────────────────────────────────────────
        cached = self._load_cached_result(file_hash)
        if cached:
            logger.info("[DocPipeline] Cache hit: %s", file_hash)
            cached["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return DocumentResult(**cached)

        # ── 3. Route → extract text ────────────────────────────────────────
        ext = Path(filename).suffix.lower()
        is_image = ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

        if is_image:
            from document_pipeline.ocr_pipeline import OCRPipeline
            raw_text, tables, method = OCRPipeline().process_image(file_bytes)
        else:
            # PDF: detect type then route
            from document_pipeline.text_pipeline import TextPipeline
            from document_pipeline.ocr_pipeline  import OCRPipeline

            if TextPipeline.is_text_layer(file_bytes):
                logger.info("[DocPipeline] Routing → TEXT pipeline")
                raw_text, tables, method = TextPipeline().process(file_bytes)
            else:
                logger.info("[DocPipeline] Routing → OCR pipeline")
                raw_text, tables, method = OCRPipeline().process_pdf(file_bytes)

        logger.info("[DocPipeline] Extracted %d chars  %d tables  method=%s",
                    len(raw_text), len(tables), method)

        # ── 4. Extract structured profile from text ────────────────────────
        from document_pipeline.extractor import EntityExtractor
        extracted = EntityExtractor().extract(raw_text, tables)
        logger.info("[DocPipeline] Extracted profile: %s", extracted.to_profile())

        # ── 5. Fuse with user query ────────────────────────────────────────
        from document_pipeline.fusion import DataFusion
        merged_profile, merged_query = DataFusion().fuse(extracted, query)
        logger.info("[DocPipeline] Merged profile: %s", merged_profile)

        # ── 6. Run through existing pipeline ──────────────────────────────
        pipeline = self._get_pipeline()
        advisor_response = self._run_pipeline(pipeline, merged_query, merged_profile, raw_text)

        # ── 7. Build result ────────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        result = DocumentResult(
            decision=advisor_response.decision,
            summary=advisor_response.summary,
            detailed_explanation=advisor_response.detailed_explanation,
            recommendations=advisor_response.recommendations,
            extracted_data=extracted.to_dict(),
            eligible_banks=advisor_response.reasoning_context.get("eligible_banks", []),
            rejected_banks=advisor_response.reasoning_context.get("rejected_banks", []),
            banks_compared=advisor_response.banks_compared,
            rule_results=advisor_response.rule_results,
            confidence=advisor_response.confidence,
            sources_cited=advisor_response.sources_cited,
            reasoning_context=advisor_response.reasoning_context,
            validation_issues=advisor_response.validation_issues,
            best_bank=advisor_response.best_bank,
            best_bank_reason=advisor_response.best_bank_reason,
            processing_method=method,
            latency_ms=latency_ms,
            file_hash=file_hash,
        )

        # ── 8. Save processed output ───────────────────────────────────────
        self._save_processed(file_hash, filename, raw_text, extracted, result)

        logger.info("[DocPipeline] Done: %s  decision=%s  %.0fms",
                    filename, result.decision, latency_ms)
        return result

    # ── Validation ──────────────────────────────────────────────────────────

    def _validate(self, file_bytes: bytes, filename: str) -> None:
        if len(file_bytes) == 0:
            raise ValueError("Uploaded file is empty.")
        if len(file_bytes) > _MAX_UPLOAD_BYTES:
            raise ValueError(
                f"File too large: {len(file_bytes)/1024/1024:.1f} MB "
                f"(max {_MAX_UPLOAD_MB} MB). Set MAX_UPLOAD_SIZE_MB in .env to increase."
            )
        ext = Path(filename).suffix.lower()
        if ext not in _ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
            )

    # ── Cache ────────────────────────────────────────────────────────────────

    def _load_cached_result(self, file_hash: str) -> dict | None:
        cache_path = _PROCESSED_DIR / f"{file_hash}_result.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_processed(
        self,
        file_hash: str,
        filename:  str,
        raw_text:  str,
        extracted,
        result:    DocumentResult,
    ) -> None:
        try:
            stem = f"{file_hash}"
            # Raw text
            (_PROCESSED_DIR / f"{stem}_raw_text.txt").write_text(raw_text, encoding="utf-8")
            # Structured extracted data
            (_PROCESSED_DIR / f"{stem}_structured.json").write_text(
                json.dumps({
                    "filename":  filename,
                    "file_hash": file_hash,
                    "extracted": extracted.to_dict(),
                    "method":    result.processing_method,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Full result (for cache)
            (_PROCESSED_DIR / f"{stem}_result.json").write_text(
                json.dumps(asdict(result), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.debug("[DocPipeline] Saved processed output for %s", file_hash)
        except Exception as e:
            logger.warning("[DocPipeline] Could not save processed output: %s", e)

    # ── Pipeline integration ─────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        # Lazy import — create one if not injected (testing / standalone use)
        from pipeline import LoanAdvisorPipeline
        self._pipeline = LoanAdvisorPipeline()
        return self._pipeline

    def _run_pipeline(self, pipeline, merged_query: str, merged_profile: dict, raw_text: str):
        """
        Inject the merged profile directly into the pipeline, bypassing LLM parse.
        This is the key integration point: document data has highest trust, so we
        override the profile rather than letting the LLM re-extract from a query.
        """
        import time as _time
        from dataclasses import dataclass, asdict
        from services.rule_engine import RuleEngine, BankResult
        from services.reasoning import (
            build_reasoning_context, compute_confidence,
            compute_final_score, validate_consistency,
        )
        from services.fusion import ContextFusion

        t0 = _time.perf_counter()

        # DTI guard
        if merged_profile.get("dti_ratio") and merged_profile["dti_ratio"] > 1.5:
            merged_profile["dti_ratio"] /= 100.0

        # Retrieve context (use raw_text as supplemental context for LLM)
        chunks = []
        try:
            if pipeline.retrieval._index is not None:
                chunks = pipeline.retrieval.retrieve(
                    query=merged_query or "personal loan eligibility",
                    top_k=5,
                )
        except Exception as re:
            logger.warning("[DocPipeline] Retrieval failed: %s", re)

        retrieval_scores = [c.similarity_score for c in chunks]

        # Rule engine
        bank_results: list[BankResult] = []
        if merged_profile:
            banks = pipeline.rule_engine.available_banks()
            for bank in banks:
                br = pipeline.rule_engine.evaluate(bank, merged_profile)
                bank_results.append(br)

        # Reasoning
        reasoning = build_reasoning_context(bank_results, merged_profile)
        from pipeline import _determine_decision, BANK_RATES
        decision = _determine_decision(bank_results, reasoning, merged_profile)

        confidence, conf_breakdown = compute_confidence(
            bank_results, reasoning, len(chunks), retrieval_scores
        )
        reasoning.confidence_breakdown = conf_breakdown

        issues = validate_consistency(bank_results, decision, reasoning)

        rule_results_dicts = [br.to_dict() for br in bank_results]
        banks_compared = sorted(
            [{
                "name":       br.bank,
                "eligible":   br.eligible,
                "rate":       BANK_RATES.get(br.bank.lower(), "N/A"),
                "score":      int(compute_final_score(br) * 100),
                "rule_score": round(br.rule_score * 100),
                "summary":    br.summary,
            } for br in bank_results],
            key=lambda x: x["score"], reverse=True,
        )

        # Build supplemental context: doc text + KB chunks
        fused_context = ContextFusion().fuse(chunks)
        if raw_text:
            fused_context = f"[Uploaded Document]\n{raw_text[:2000]}\n\n---\n\n{fused_context}"

        # LLM explanation
        if not bank_results and not chunks:
            explanation = {
                "summary": "Insufficient data. Please ensure the document contains income, age, and CIBIL information.",
                "detailed_explanation": "Could not extract enough profile data from the uploaded document.",
                "recommendations": [
                    "Ensure the document contains salary/income information.",
                    "Include CIBIL score or credit score details.",
                    "Provide age and employment type information.",
                ],
                "sources_cited": [],
            }
        else:
            explanation = pipeline.llm.explain(
                query=merged_query or "Analyze my loan eligibility from the uploaded document.",
                rule_results=rule_results_dicts,
                retrieved_context=fused_context,
                parsed_query={"intent": "eligibility", "entities": merged_profile},
                pre_decision=decision,
                decision_context=reasoning.to_dict(),
            )

        latency = (_time.perf_counter() - t0) * 1000

        from pipeline import AdvisorResponse
        return AdvisorResponse(
            decision=decision,
            summary=explanation.get("summary", ""),
            detailed_explanation=explanation.get("detailed_explanation", ""),
            recommendations=explanation.get("recommendations", []),
            banks_compared=banks_compared,
            rule_results=rule_results_dicts,
            confidence=confidence,
            sources_cited=explanation.get("sources_cited", []),
            reasoning_context=reasoning.to_dict(),
            validation_issues=[
                {"severity": i.severity, "bank": i.bank, "message": i.message}
                for i in issues
            ],
            best_bank=reasoning.best_bank,
            best_bank_reason=reasoning.best_bank_reason,
            almost_eligible_bank=reasoning.closest_bank,
            critical_failures=reasoning.critical_failures or [],
            latency_ms=round(latency, 1),
            from_cache=False,
        )
