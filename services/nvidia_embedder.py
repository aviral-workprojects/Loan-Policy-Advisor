"""
services/nvidia_embedder.py
===========================
NVIDIA NemoRetriever Embedding via API Catalog.
Model: nvidia/llama-3_2-nemoretriever-300m-embed-v2

Drop-in replacement for sentence-transformers — same .encode() interface.
Used for BOTH document indexing (bootstrap.py) and query embedding (retrieval).

Activation:
    Set in .env:
        NVIDIA_EMBED_BACKEND=nvidia_api
        NVIDIA_API_KEY=nvapi-...

    Then re-run:  python bootstrap.py   (rebuilds index with new embeddings)
    New dim is 1024 — set EMBEDDING_DIM=1024 in .env before bootstrapping.

Falls back to local sentence-transformers if NVIDIA_EMBED_BACKEND=local (default).

Fixes (v2):
    ✅ Correct endpoint: /v1/embeddings  (was /v1/retrieval/.../embeddings → 404)
    ✅ Batch size: 16   (was 96 — too large, causes silent failures / rate limits)
    ✅ Retry logic: 3 attempts with exponential backoff on 429/5xx
"""

from __future__ import annotations
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ✅ FIXED: correct NVIDIA embeddings endpoint
# The old path /v1/retrieval/.../embeddings does not exist → 404
_NVIDIA_EMBED_URL = "https://ai.api.nvidia.com/v1/embeddings"

# ✅ FIXED: reduced batch size (96 → 16) to avoid rate-limit and silent failures
_NVIDIA_BATCH_SIZE = 16

# Retry config
_MAX_RETRIES  = 3
_RETRY_DELAY  = 2.0   # seconds (doubled on each attempt)
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class NvidiaEmbedder:
    """
    Wraps the NVIDIA NemoRetriever 300M embed model via API Catalog.

    Interface mirrors sentence_transformers.SentenceTransformer so
    existing retrieval code needs zero changes — just swap the embedder.

    Usage:
        embedder = NvidiaEmbedder(api_key="nvapi-...")
        vecs = embedder.encode(["text1", "text2"], normalize_embeddings=True)
        # Returns np.ndarray shape (N, 1024)
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required for NvidiaEmbedder")
        self._api_key = api_key
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

    def _post_with_retry(self, payload: dict, timeout: int = 30) -> dict:
        """
        POST to _NVIDIA_EMBED_URL with exponential-backoff retry.
        Retries on 429 and 5xx. Raises immediately on 4xx (except 429).
        """
        import requests

        last_err: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    _NVIDIA_EMBED_URL,
                    json=payload,
                    headers=self._headers,
                    timeout=timeout,
                )

                if resp.status_code in _RETRYABLE_STATUSES:
                    wait = _RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "[NvidiaEmbed] HTTP %d on attempt %d/%d — retrying in %.1fs",
                        resp.status_code, attempt, _MAX_RETRIES, wait,
                    )
                    last_err = requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}", response=resp
                    )
                    if attempt < _MAX_RETRIES:
                        time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                last_err = TimeoutError(f"Timeout on attempt {attempt}")
                logger.warning(
                    "[NvidiaEmbed] Timeout on attempt %d/%d — retrying in %.1fs",
                    attempt, _MAX_RETRIES, wait,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

            except requests.exceptions.HTTPError:
                raise   # 4xx errors are not retried

            except Exception as e:
                last_err = e
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "[NvidiaEmbed] Unexpected error on attempt %d/%d: %s",
                    attempt, _MAX_RETRIES, e,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"NVIDIA Embeddings API failed after {_MAX_RETRIES} attempts: {last_err}"
        )

    def encode(
        self,
        sentences: list[str],
        batch_size: int = _NVIDIA_BATCH_SIZE,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of strings. Returns float32 ndarray (N, 1024).
        Automatically batches; each batch is retried independently on failure.
        """
        all_vecs: list[list[float]] = []
        n_batches = (len(sentences) + batch_size - 1) // batch_size

        for i in range(0, len(sentences), batch_size):
            batch      = sentences[i : i + batch_size]
            batch_num  = i // batch_size + 1

            if show_progress_bar:
                print(f"[NvidiaEmbed] Embedding batch {batch_num}/{n_batches} ({len(batch)} items)…")

            # ✅ FIXED payload format — matches /v1/embeddings schema
            payload = {
                "model":            "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
                "input":            batch,
                "input_type":       "passage",   # "passage" for docs, "query" for queries
                "encoding_format":  "float",
                "truncate":         "END",
            }

            t0   = time.perf_counter()
            data = self._post_with_retry(payload)
            elapsed = (time.perf_counter() - t0) * 1000

            # Response shape: {"data": [{"embedding": [...], "index": N}, ...]}
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            batch_vecs = [e["embedding"] for e in embeddings]
            all_vecs.extend(batch_vecs)

            logger.debug(
                "[NvidiaEmbed] Batch %d/%d — %d items in %.0fms",
                batch_num, n_batches, len(batch), elapsed,
            )

        arr = np.array(all_vecs, dtype="float32")

        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr   = arr / (norms + 1e-10)

        return arr

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query string.
        Uses input_type="query" (slightly different from passage embedding).
        """
        payload = {
            "model":           "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
            "input":           [query],
            "input_type":      "query",
            "encoding_format": "float",
            "truncate":        "END",
        }
        data = self._post_with_retry(payload, timeout=15)
        vec  = np.array(data["data"][0]["embedding"], dtype="float32")
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.reshape(1, -1)   # (1, 1024) to match sentence-transformers shape


def get_embedder(
    backend: str | None = None,
    api_key: str | None = None,
):
    """
    Factory: returns either NvidiaEmbedder or sentence-transformers model
    based on NVIDIA_EMBED_BACKEND config.

    Usage in retrieval.py:
        from services.nvidia_embedder import get_embedder
        embedder = get_embedder()
        vecs = embedder.encode(texts, normalize_embeddings=True)
    """
    from config import NVIDIA_EMBED_BACKEND, NVIDIA_API_KEY, EMBEDDING_MODEL

    backend = backend or NVIDIA_EMBED_BACKEND
    api_key = api_key or NVIDIA_API_KEY

    if backend == "nvidia_api":
        logger.info("[Embedder] Using NVIDIA NemoRetriever 300M (API)")
        return NvidiaEmbedder(api_key=api_key)
    else:
        logger.info("[Embedder] Using local sentence-transformers: %s", EMBEDDING_MODEL)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)