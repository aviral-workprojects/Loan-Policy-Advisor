"""
services/reranker.py
====================
NVIDIA Nemotron Reranker — cross-encoder re-scoring layer.

Pipeline position:
    FAISS retrieval (top_k=10)
        └─► rerank_documents(query, docs)   ← THIS MODULE
                └─► sorted top-3 docs
                        └─► LLM explanation

Cross-encoder vs bi-encoder:
    Bi-encoder (FAISS)  : encodes query and doc SEPARATELY → fast, approximate
    Cross-encoder (here): encodes query+doc TOGETHER → slower, much more precise

Supports two backends (controlled by RERANKER_BACKEND env var):
    "nvidia_api" : NVIDIA API Catalog  — cloud, needs NVIDIA_API_KEY
    "nim"        : Local NIM container — on-prem, no key, needs Docker

NVIDIA API Catalog (free tier available):
    https://build.nvidia.com/nvidia/llama-nemotron-rerank-1b-v2

Local NIM:
    docker run --gpus all -p 8001:8000 \\
      nvcr.io/nim/nvidia/llama-nemotron-rerank-1b-v2:latest
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from config import (
    USE_RERANKER,
    RERANKER_MODEL,
    RERANKER_BACKEND,
    NVIDIA_API_KEY,
    NIM_BASE_URL,
    RERANKER_TOP_N,
    RERANKER_TIMEOUT,
    RERANKER_MIN_QUERY_WORDS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# NVIDIA API Catalog endpoint for reranking
_NVIDIA_API_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"

# NIM local endpoint pattern  (NIM exposes the same schema on /v1/ranking)
_NIM_API_URL_TEMPLATE = "{base}/v1/ranking"


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class RerankResult:
    """Wrapper returned by rerank_documents so callers always get typed output."""
    documents: list[Any]      # DocChunk objects, sorted best-first
    scores: list[float]       # matching relevance scores (0–1)
    reranked: bool            # True if reranker actually ran; False if skipped/fallback
    reason: str               # human-readable note on what happened


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_payload(query: str, passages: list[str]) -> dict:
    """
    Build the JSON body for the NVIDIA reranking endpoint.

    API schema:
        {
          "model": "nvidia/llama-nemotron-rerank-1b-v2",
          "query": { "text": "<query string>" },
          "passages": [ { "text": "<passage 1>" }, ... ],
          "truncate": "END"   # silently truncate passages that exceed context
        }
    """
    return {
        "model": RERANKER_MODEL,
        "query": {"text": query},
        "passages": [{"text": p} for p in passages],
        "truncate": "END",
    }


def _call_nvidia_api(payload: dict) -> list[dict]:
    """
    POST to NVIDIA API Catalog and return the raw rankings list.

    Returns:
        [ {"index": 2, "logit": 3.14}, {"index": 0, "logit": 1.05}, ... ]
        Sorted best-first by the API itself.
    """
    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is not set. "
            "Get a free key at https://build.nvidia.com and set it in .env"
        )

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    resp = requests.post(
        _NVIDIA_API_URL,
        json=payload,
        headers=headers,
        timeout=RERANKER_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # Response shape: { "rankings": [ {"index": N, "logit": F}, ... ] }
    return data.get("rankings", [])


def _call_nim_endpoint(payload: dict) -> list[dict]:
    """
    POST to a local NIM container.
    NIM exposes an identical REST schema at /v1/ranking.
    """
    url = _NIM_API_URL_TEMPLATE.format(base=NIM_BASE_URL.rstrip("/"))

    resp = requests.post(
        url,
        json=payload,
        timeout=RERANKER_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    return data.get("rankings", [])


def _normalize_logits(rankings: list[dict]) -> list[float]:
    """
    Convert raw logits → [0, 1] scores using min-max normalization.
    Keeps relative ordering while giving interpretable confidence values.
    """
    if not rankings:
        return []

    logits = [r.get("logit", 0.0) for r in rankings]
    lo, hi = min(logits), max(logits)

    if hi == lo:
        # All identical → assign uniform score
        return [1.0] * len(logits)

    return [(l - lo) / (hi - lo) for l in logits]


def _is_simple_query(query: str) -> bool:
    """
    Heuristic: skip reranking for very short / single-word queries.
    Short queries usually have one clear intent and don't benefit from
    the cross-encoder's deeper query-passage comparison.

    Threshold controlled by RERANKER_MIN_QUERY_WORDS in config.
    """
    word_count = len(query.strip().split())
    return word_count < RERANKER_MIN_QUERY_WORDS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_documents(
    query: str,
    documents: list[Any],          # list of DocChunk objects (or any obj with .text)
    top_n: int = RERANKER_TOP_N,
) -> RerankResult:
    """
    Re-score `documents` against `query` using the Nemotron cross-encoder.

    Args:
        query     : the user's natural-language question
        documents : list of DocChunk objects from FAISS retrieval
        top_n     : how many top documents to return after reranking

    Returns:
        RerankResult with .documents sorted best-first and .scores aligned

    Fallback behaviour:
        Any error (network, auth, timeout) → returns original documents unchanged
        so the pipeline never breaks due to the reranker.

    Example:
        chunks = retrieval.retrieve(query, top_k=10)
        result = rerank_documents(query, chunks, top_n=3)
        best_docs = result.documents   # top-3, properly ranked
    """

    # ── Guard: feature flag ──────────────────────────────────────────────────
    if not USE_RERANKER:
        logger.debug("[Reranker] Disabled via USE_RERANKER=false")
        return RerankResult(
            documents=documents[:top_n],
            scores=[1.0] * min(top_n, len(documents)),
            reranked=False,
            reason="Reranker disabled in config (USE_RERANKER=false)",
        )

    # ── Guard: not enough docs to be worth reranking ─────────────────────────
    if len(documents) <= top_n:
        logger.info(
            "[Reranker] Skipped — only %d docs retrieved (≤ top_n=%d), "
            "no reranking needed",
            len(documents), top_n,
        )
        return RerankResult(
            documents=documents,
            scores=[1.0] * len(documents),
            reranked=False,
            reason=f"Only {len(documents)} docs retrieved; reranking not needed",
        )

    # ── Guard: simple query heuristic ────────────────────────────────────────
    if _is_simple_query(query):
        logger.info(
            "[Reranker] Skipped — query too short (%d words < %d threshold): %r",
            len(query.split()), RERANKER_MIN_QUERY_WORDS, query,
        )
        return RerankResult(
            documents=documents[:top_n],
            scores=[1.0] * top_n,
            reranked=False,
            reason=f"Query too short ({len(query.split())} words); using FAISS order",
        )

    # ── Extract text from DocChunk objects ───────────────────────────────────
    # Works with our DocChunk dataclass and any object that has a .text attribute
    passages: list[str] = []
    for doc in documents:
        if hasattr(doc, "text"):
            passages.append(doc.text)
        elif isinstance(doc, dict):
            passages.append(doc.get("text", str(doc)))
        else:
            passages.append(str(doc))

    logger.info("[Reranker] Retrieved docs: %d", len(documents))

    # ── Build payload ─────────────────────────────────────────────────────────
    payload = _build_payload(query, passages)

    # ── Call reranker backend ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        if RERANKER_BACKEND == "nim":
            rankings = _call_nim_endpoint(payload)
            backend_used = f"NIM @ {NIM_BASE_URL}"
        else:
            rankings = _call_nvidia_api(payload)
            backend_used = "NVIDIA API Catalog"

    except requests.exceptions.Timeout:
        logger.warning(
            "[Reranker] Timeout after %ds — falling back to FAISS order",
            RERANKER_TIMEOUT,
        )
        return RerankResult(
            documents=documents[:top_n],
            scores=[1.0] * top_n,
            reranked=False,
            reason=f"Reranker timed out ({RERANKER_TIMEOUT}s); using FAISS order",
        )

    except requests.exceptions.HTTPError as e:
        logger.warning("[Reranker] HTTP error %s — falling back to FAISS order", e)
        return RerankResult(
            documents=documents[:top_n],
            scores=[1.0] * top_n,
            reranked=False,
            reason=f"Reranker HTTP error ({e}); using FAISS order",
        )

    except Exception as e:
        logger.warning("[Reranker] Unexpected error — falling back: %s", e)
        return RerankResult(
            documents=documents[:top_n],
            scores=[1.0] * top_n,
            reranked=False,
            reason=f"Reranker error ({type(e).__name__}); using FAISS order",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # ── Parse rankings ────────────────────────────────────────────────────────
    # rankings: [ {"index": 2, "logit": 3.14}, ... ] already sorted best-first
    if not rankings:
        logger.warning("[Reranker] Empty rankings returned — falling back")
        return RerankResult(
            documents=documents[:top_n],
            scores=[1.0] * top_n,
            reranked=False,
            reason="Reranker returned empty rankings; using FAISS order",
        )

    # Normalise logits to [0, 1] range for readability
    scores_normalised = _normalize_logits(rankings)

    # Rebuild document list in reranked order, sliced to top_n
    reranked_docs:   list[Any]   = []
    reranked_scores: list[float] = []

    for rank_pos, item in enumerate(rankings[:top_n]):
        original_idx = item.get("index", -1)
        if original_idx < 0 or original_idx >= len(documents):
            continue
        reranked_docs.append(documents[original_idx])
        reranked_scores.append(scores_normalised[rank_pos])

    # Safety net: if reranker returned fewer than top_n items, pad with remaining
    if len(reranked_docs) < top_n:
        seen_indices = {item.get("index") for item in rankings[:top_n]}
        for i, doc in enumerate(documents):
            if i not in seen_indices and len(reranked_docs) < top_n:
                reranked_docs.append(doc)
                reranked_scores.append(0.0)

    logger.info(
        "[Reranker] Reranked docs: %d  (from %d candidates, %.0fms, backend=%s)",
        len(reranked_docs), len(documents), latency_ms, backend_used,
    )

    return RerankResult(
        documents=reranked_docs,
        scores=reranked_scores,
        reranked=True,
        reason=f"Reranked {len(documents)} → {len(reranked_docs)} docs via {backend_used}",
    )
