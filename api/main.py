"""
FastAPI Application — Loan Policy Advisor API
==============================================
Endpoints:
  POST /query      — main advisory endpoint
  GET  /health     — health check
  GET  /banks      — list supported banks
  POST /rebuild    — rebuild FAISS index
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import LoanAdvisorPipeline, AdvisorResponse

# ---------------------------------------------------------------------------
# Global pipeline instance (initialized on startup)
# ---------------------------------------------------------------------------

_pipeline: LoanAdvisorPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("[API] Initializing Loan Advisor Pipeline…")
    _pipeline = LoanAdvisorPipeline()
    print("[API] Pipeline ready.")
    yield
    print("[API] Shutting down.")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Loan Policy Advisor",
    description="AI-powered loan eligibility and comparison engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000,
                       example="Am I eligible for an Axis Bank loan with 40k salary, age 28, CIBIL 750?")

class QueryResponse(BaseModel):
    decision: str
    summary: str
    detailed_explanation: str
    recommendations: list[str]
    banks_compared: list[str]
    rule_results: list[dict]
    confidence: float
    sources_cited: list[str]
    latency_ms: float
    from_cache: bool

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main advisory endpoint.

    Accepts a natural language query about loan eligibility or comparison.
    Returns structured decision with explanations.
    """
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result: AdvisorResponse = _pipeline.query(request.query)
        return QueryResponse(**result.__dict__)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[API ERROR]\n{tb}")          # prints full traceback to uvicorn terminal
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_ready": _pipeline is not None}


@app.get("/banks")
async def list_banks():
    """List all banks with available rule files."""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    banks = _pipeline.rule_engine.available_banks()
    return {"banks": sorted(banks)}


@app.post("/rebuild-index")
async def rebuild_index():
    """Re-scan data directory and rebuild FAISS index. Useful after adding new docs."""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    count = _pipeline.retrieval.load_documents()
    _pipeline.retrieval.build_index(save=True)
    return {"status": "rebuilt", "chunks_indexed": count}


@app.delete("/cache")
async def clear_cache():
    """Clear the semantic response cache."""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    _pipeline.cache.clear()
    return {"status": "cache cleared"}


@app.get("/sources")
async def list_sources():
    """
    Shows every document chunk loaded into FAISS — grouped by source file.
    Use this to verify your PDFs were indexed correctly.
    GET http://localhost:8000/sources
    """
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    chunks = _pipeline.retrieval._chunks
    from collections import defaultdict
    summary: dict = defaultdict(lambda: {"count": 0, "bank": "", "type": "", "sample": ""})

    for c in chunks:
        key = c.source
        summary[key]["count"]  += 1
        summary[key]["bank"]    = c.bank
        summary[key]["type"]    = c.doc_type
        if not summary[key]["sample"]:
            summary[key]["sample"] = c.text[:120].replace("\n", " ")

    return {
        "total_chunks": len(chunks),
        "files": dict(summary),
    }


@app.get("/search-test")
async def search_test(q: str = "personal loan eligibility"):
    """
    Run a live semantic search and show which chunks are retrieved + their sources.
    Use this to verify PDF content is being found.
    Example: GET http://localhost:8000/search-test?q=RBI+guidelines+personal+loan
    """
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    chunks = _pipeline.retrieval.retrieve(q, top_k=8)
    return {
        "query": q,
        "retrieved": [
            {
                "rank":    i + 1,
                "source":  c.source,
                "bank":    c.bank,
                "type":    c.doc_type,
                "preview": c.text[:200].replace("\n", " "),
            }
            for i, c in enumerate(chunks)
        ],
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)