"""
services/nvidia_embedder.py  (v2)
==================================
NVIDIA NemoRetriever Embedding via API Catalog.
Model: nvidia/llama-3_2-nemoretriever-300m-embed-v2

Fixes vs v1:
    ✅ Endpoint:   /v1/embeddings  (was /v1/retrieval/.../embeddings → 404)
    ✅ Batch size: 16              (was 96 → rate-limit / silent failures)
    ✅ Retry:      3 attempts, exponential backoff on 429 / 5xx
"""

from __future__ import annotations
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ✅ Correct endpoint
_NVIDIA_EMBED_URL = "https://ai.api.nvidia.com/v1/embeddings"

# ✅ Safe batch size (16 instead of 96)
_NVIDIA_BATCH_SIZE = 16

_MAX_RETRIES        = 3
_RETRY_DELAY        = 2.0
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class NvidiaEmbedder:
    """
    Wraps NVIDIA NemoRetriever 300M via API Catalog.
    Drop-in for sentence_transformers.SentenceTransformer (.encode() interface).
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
        """POST to /v1/embeddings with exponential-backoff retry."""
        import requests
        last_err: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    _NVIDIA_EMBED_URL, json=payload,
                    headers=self._headers, timeout=timeout,
                )
                if resp.status_code in _RETRYABLE_STATUSES:
                    wait = _RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning("[NvidiaEmbed] HTTP %d attempt %d/%d — retry %.1fs",
                                   resp.status_code, attempt, _MAX_RETRIES, wait)
                    last_err = requests.exceptions.HTTPError(str(resp.status_code), response=resp)
                    if attempt < _MAX_RETRIES:
                        time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                last_err = TimeoutError(f"timeout attempt {attempt}")
                logger.warning("[NvidiaEmbed] Timeout attempt %d/%d — retry %.1fs",
                               attempt, _MAX_RETRIES, wait)
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

            except requests.exceptions.HTTPError:
                raise

            except Exception as e:
                last_err = e
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("[NvidiaEmbed] Error attempt %d/%d: %s", attempt, _MAX_RETRIES, e)
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"NVIDIA Embeddings API failed after {_MAX_RETRIES} attempts: {last_err}"
        )

    def encode(
        self,
        sentences:            list[str],
        batch_size:           int  = _NVIDIA_BATCH_SIZE,
        show_progress_bar:    bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Embed strings. Returns float32 ndarray (N, 1024)."""
        all_vecs: list[list[float]] = []
        n_batches = (len(sentences) + batch_size - 1) // batch_size

        for i in range(0, len(sentences), batch_size):
            batch     = sentences[i : i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress_bar:
                print(f"[NvidiaEmbed] Batch {batch_num}/{n_batches} ({len(batch)} items)…")

            from config import NVIDIA_EMBED_MODEL
            payload = {
                "model":           NVIDIA_EMBED_MODEL,
                "input":           batch,
                "input_type":      "passage",
                "encoding_format": "float",
                "truncate":        "END",
            }

            t0   = time.perf_counter()
            data = self._post_with_retry(payload)
            elapsed = (time.perf_counter() - t0) * 1000

            embeddings = sorted(data["data"], key=lambda x: x["index"])
            all_vecs.extend(e["embedding"] for e in embeddings)

            logger.debug("[NvidiaEmbed] Batch %d/%d — %d items in %.0fms",
                         batch_num, n_batches, len(batch), elapsed)

        arr = np.array(all_vecs, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr   = arr / (norms + 1e-10)
        return arr

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query string (uses input_type='query')."""
        from config import NVIDIA_EMBED_MODEL
        payload = {
            "model":           NVIDIA_EMBED_MODEL,
            "input":           [query],
            "input_type":      "query",
            "encoding_format": "float",
            "truncate":        "END",
        }
        data = self._post_with_retry(payload, timeout=15)
        vec  = np.array(data["data"][0]["embedding"], dtype="float32")
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.reshape(1, -1)


def get_embedder(backend: str | None = None, api_key: str | None = None):
    from config import NVIDIA_EMBED_BACKEND, NVIDIA_API_KEY, EMBEDDING_MODEL
    backend = backend or NVIDIA_EMBED_BACKEND
    api_key = api_key or NVIDIA_API_KEY
    if backend == "nvidia_api":
        logger.info("[Embedder] Using NVIDIA NemoRetriever 300M")
        return NvidiaEmbedder(api_key=api_key)
    logger.info("[Embedder] Using local sentence-transformers: %s", EMBEDDING_MODEL)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)