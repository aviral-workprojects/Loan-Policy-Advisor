"""
FastAPI Application v4 — Loan Policy Advisor + Document Upload
==============================================================
Adds POST /analyze-document to the existing v3 API.
All existing endpoints (/query, /health, /banks, etc.) are unchanged.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import LoanAdvisorPipeline, AdvisorResponse

_pipeline: LoanAdvisorPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("[API] Initializing Loan Advisor Pipeline…")
    _pipeline = LoanAdvisorPipeline()
    print("[API] Pipeline ready.")
    yield


app = FastAPI(title="Loan Policy Advisor v4", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ---------------------------------------------------------------------------
# Existing models (unchanged)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:      str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(default="default_session", description="Unique ID for user session")


class BankComparison(BaseModel):
    name:     str
    eligible: bool
    rate:     str
    score:    float
    summary:  str = ""


class ValidationIssue(BaseModel):
    severity: str
    bank:     str = ""
    message:  str


class QueryResponse(BaseModel):
    decision:             str
    summary:              str
    detailed_explanation: str
    recommendations:      list[str]
    banks_compared:       list[BankComparison]
    rule_results:         list[dict]
    confidence:           float
    sources_cited:        list[str]
    reasoning_context:    dict
    validation_issues:    list[ValidationIssue]
    best_bank:            str | None = None
    best_bank_reason:     str | None = None
    almost_eligible_bank: str | None = None
    critical_failures:    list[dict] = []
    latency_ms:           float = 0.0
    from_cache:           bool = False


# ---------------------------------------------------------------------------
# Document upload models (new)
# ---------------------------------------------------------------------------

class DocumentResponse(BaseModel):
    decision:             str
    summary:              str
    detailed_explanation: str
    recommendations:      list[str]
    extracted_data:       dict        # raw extracted profile fields
    eligible_banks:       list[str]
    rejected_banks:       list[str]
    banks_compared:       list[dict]
    rule_results:         list[dict]
    confidence:           float
    sources_cited:        list[str]
    reasoning_context:    dict
    validation_issues:    list[dict]
    best_bank:            str | None = None
    best_bank_reason:     str | None = None
    processing_method:    str = ""
    latency_ms:           float = 0.0
    file_hash:            str = ""


# ---------------------------------------------------------------------------
# Existing endpoints (unchanged)
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result: AdvisorResponse = _pipeline.query(request.query, session_id=request.session_id)
        return QueryResponse(**result.__dict__)
    except Exception as e:
        import traceback
        print(f"[API ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "pipeline_ready":  _pipeline is not None,
        "version":         "4.0.0",
        "upload_enabled":  True,
    }


@app.get("/banks")
async def list_banks():
    if not _pipeline:
        raise HTTPException(503)
    return {"banks": sorted(_pipeline.rule_engine.available_banks())}


@app.get("/sources")
async def list_sources():
    if not _pipeline:
        raise HTTPException(503)
    from collections import defaultdict
    summary: dict = defaultdict(lambda: {"count": 0, "bank": "", "type": "", "sample": ""})
    for c in _pipeline.retrieval._chunks:
        k = c.source
        summary[k]["count"] += 1
        summary[k]["bank"]   = c.bank
        summary[k]["type"]   = c.doc_type
        if not summary[k]["sample"]:
            summary[k]["sample"] = c.text[:120].replace("\n", " ")
    return {"total_chunks": len(_pipeline.retrieval._chunks), "files": dict(summary)}


@app.get("/search-test")
async def search_test(q: str = "personal loan eligibility"):
    if not _pipeline:
        raise HTTPException(503)
    chunks = _pipeline.retrieval.retrieve(q, top_k=6)
    return {"query": q, "retrieved": [
        {"rank": i + 1, "source": c.source, "bank": c.bank, "type": c.doc_type,
         "preview": c.text[:180].replace("\n", " ")}
        for i, c in enumerate(chunks)
    ]}


@app.post("/rebuild-index")
async def rebuild():
    if not _pipeline:
        raise HTTPException(503)
    count = _pipeline.retrieval.load_documents()
    _pipeline.retrieval.build_index(save=True)
    return {"status": "rebuilt", "chunks_indexed": count}


@app.delete("/cache")
async def clear_cache():
    if not _pipeline:
        raise HTTPException(503)
    _pipeline.cache.clear()
    return {"status": "cache cleared"}


# ---------------------------------------------------------------------------
# NEW: Document Upload endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze-document", response_model=DocumentResponse)
async def analyze_document(
    file:       UploadFile = File(..., description="PDF, PNG, JPG, or TIFF document"),
    query:      str        = Form("", description="Optional natural language query to combine with document"),
    session_id: str        = Form("default_session", description="Unique ID for user session"),
):
    """
    Upload a document (salary slip, bank statement, ITR, CIBIL report, or scan)
    and receive a full loan eligibility analysis.

    The system:
      1. Detects document type (text-layer PDF vs scanned/image)
      2. Extracts structured financial signals (income, CIBIL, age, employment)
      3. Merges with knowledge base and applies YAML eligibility rules
      4. Returns decision + per-bank breakdown + recommendations

    Supported formats: PDF, PNG, JPG, JPEG, TIFF, BMP, WEBP
    Max size: configurable via MAX_UPLOAD_SIZE_MB in .env (default 10 MB)
    """
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Read file bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Process through document pipeline
    try:
        from document_pipeline.router import DocumentProcessor
        processor = DocumentProcessor(pipeline=_pipeline)
        result = processor.process(
            file_bytes=file_bytes,
            filename=file.filename or "upload.pdf",
            query=query.strip(),
            session_id=session_id,
        )
    except ValueError as e:
        # Validation errors (wrong type, too large, etc.)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        import traceback
        print(f"[API ERROR /analyze-document]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return DocumentResponse(
        decision=result.decision,
        summary=result.summary,
        detailed_explanation=result.detailed_explanation,
        recommendations=result.recommendations,
        extracted_data=result.extracted_data,
        eligible_banks=result.eligible_banks,
        rejected_banks=result.rejected_banks,
        banks_compared=result.banks_compared,
        rule_results=result.rule_results,
        confidence=result.confidence,
        sources_cited=result.sources_cited,
        reasoning_context=result.reasoning_context,
        validation_issues=result.validation_issues,
        best_bank=result.best_bank,
        best_bank_reason=result.best_bank_reason,
        processing_method=result.processing_method,
        latency_ms=result.latency_ms,
        file_hash=result.file_hash,
    )