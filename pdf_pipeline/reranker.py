"""
pdf_pipeline/reranker.py
=========================
NVIDIA Nemotron cross-encoder reranker.

Cross-encoder vs bi-encoder:
  Bi-encoder (FAISS): encodes query and document separately → fast, approximate
  Cross-encoder (here): encodes query+document together → slower, more precise

Used as the final stage after RRF fusion to pick the best 3–5 chunks from
the top-20 candidates. Any failure falls back to RRF order silently.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

_NVIDIA_RERANK_URL = (
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"
)
_RERANKER_MODEL    = "nvidia/llama-nemotron-rerank-1b-v2"
_TIMEOUT           = 10
_MIN_QUERY_WORDS   = 4


@dataclass
class RerankResult:
    documents: list[Any]
    scores:    list[float]
    reranked:  bool
    reason:    str


def rerank_documents(
    query:     str,
    documents: list[Any],    # DocChunk objects with .text attribute
    top_n:     int = 5,
) -> RerankResult:
    """
    Re-score documents against query using Nemotron cross-encoder.
    Falls back to original order on any error.
    """
    import os
    api_key = os.getenv("NVIDIA_API_KEY", "")

    if not api_key:
        return RerankResult(documents=documents[:top_n], scores=[1.0]*top_n,
                            reranked=False, reason="NVIDIA_API_KEY not set")

    if len(documents) <= top_n:
        return RerankResult(documents=documents, scores=[1.0]*len(documents),
                            reranked=False, reason="Not enough docs to rerank")

    if len(query.split()) < _MIN_QUERY_WORDS:
        return RerankResult(documents=documents[:top_n], scores=[1.0]*top_n,
                            reranked=False, reason="Query too short")

    passages = [d.text if hasattr(d, "text") else str(d) for d in documents]
    payload  = {
        "model":    _RERANKER_MODEL,
        "query":    {"text": query},
        "passages": [{"text": p} for p in passages],
        "truncate": "END",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(_NVIDIA_RERANK_URL, json=payload, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
    except requests.exceptions.Timeout:
        return RerankResult(documents=documents[:top_n], scores=[1.0]*top_n,
                            reranked=False, reason=f"Reranker timeout ({_TIMEOUT}s)")
    except Exception as e:
        logger.warning("[Reranker] Failed: %s — using RRF order", e)
        return RerankResult(documents=documents[:top_n], scores=[1.0]*top_n,
                            reranked=False, reason=f"Reranker error: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    if not rankings:
        return RerankResult(documents=documents[:top_n], scores=[1.0]*top_n,
                            reranked=False, reason="Empty rankings returned")

    # Normalise logits to [0, 1]
    logits = [r.get("logit", 0.0) for r in rankings]
    lo, hi = min(logits), max(logits)
    normed = [1.0] * len(logits) if hi == lo else [(l - lo) / (hi - lo) for l in logits]

    reranked_docs:   list[Any]   = []
    reranked_scores: list[float] = []

    for rank_pos, item in enumerate(rankings[:top_n]):
        orig_idx = item.get("index", -1)
        if 0 <= orig_idx < len(documents):
            reranked_docs.append(documents[orig_idx])
            reranked_scores.append(normed[rank_pos])

    # Pad if needed
    if len(reranked_docs) < top_n:
        seen = {item.get("index") for item in rankings[:top_n]}
        for i, doc in enumerate(documents):
            if i not in seen and len(reranked_docs) < top_n:
                reranked_docs.append(doc)
                reranked_scores.append(0.0)

    logger.info("[Reranker] %d → %d docs  %.0fms", len(documents), len(reranked_docs), latency_ms)
    return RerankResult(
        documents=reranked_docs,
        scores=reranked_scores,
        reranked=True,
        reason=f"Reranked {len(documents)} → {len(reranked_docs)} via Nemotron",
    )
