"""
FastAPI Application v3 — Loan Policy Advisor
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="Loan Policy Advisor v3", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)

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
    latency_ms:           float
    from_cache:           bool

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result: AdvisorResponse = _pipeline.query(request.query)
        return QueryResponse(**result.__dict__)
    except Exception as e:
        import traceback
        print(f"[API ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None, "version": "3.0.0"}

@app.get("/banks")
async def list_banks():
    if not _pipeline:
        raise HTTPException(503)
    return {"banks": sorted(_pipeline.rule_engine.available_banks())}

@app.get("/sources")
async def list_sources():
    """Show all indexed document chunks grouped by source file."""
    if not _pipeline:
        raise HTTPException(503)
    from collections import defaultdict
    summary: dict = defaultdict(lambda: {"count":0,"bank":"","type":"","sample":""})
    for c in _pipeline.retrieval._chunks:
        k = c.source
        summary[k]["count"] += 1
        summary[k]["bank"]   = c.bank
        summary[k]["type"]   = c.doc_type
        if not summary[k]["sample"]:
            summary[k]["sample"] = c.text[:120].replace("\n"," ")
    return {"total_chunks": len(_pipeline.retrieval._chunks), "files": dict(summary)}

@app.get("/search-test")
async def search_test(q: str = "personal loan eligibility"):
    """Live retrieval test — verify PDF content is being found."""
    if not _pipeline:
        raise HTTPException(503)
    chunks = _pipeline.retrieval.retrieve(q, top_k=6)
    return {"query": q, "retrieved": [
        {"rank":i+1,"source":c.source,"bank":c.bank,"type":c.doc_type,
         "preview":c.text[:180].replace("\n"," ")}
        for i,c in enumerate(chunks)
    ]}

@app.post("/rebuild-index")
async def rebuild():
    if not _pipeline:
        raise HTTPException(503)
    count = _pipeline.retrieval.load_documents()
    _pipeline.retrieval.build_index(save=True)
    return {"status":"rebuilt","chunks_indexed":count}

@app.delete("/cache")
async def clear_cache():
    if not _pipeline:
        raise HTTPException(503)
    _pipeline.cache.clear()
    return {"status":"cache cleared"}
