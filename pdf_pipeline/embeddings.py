"""
pdf_pipeline/embeddings.py
===========================
Embedding layer with NVIDIA NemoRetriever as primary and
sentence-transformers as local fallback.

The NVIDIA llama-nemotron-embed-1b-v2 model produces 1024-dim embeddings
optimised for retrieval tasks. sentence-transformers/all-MiniLM-L6-v2
(384-dim) is the fallback when no API key is set.

Both implement the same .encode() / .encode_query() interface so the
retrieval layer is backend-agnostic.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_NVIDIA_EMBED_URL   = "https://integrate.api.nvidia.com/v1/embeddings"   # fixed: was ai.api.nvidia.com which 404s
_NVIDIA_EMBED_MODEL = "nvidia/llama-3_2-nemoretriever-300m-embed-v2"
_NVIDIA_BATCH_SIZE  = 16
_MAX_RETRIES        = 3
_RETRY_DELAY        = 2.0
_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class NvidiaEmbedder:
    """
    NVIDIA NemoRetriever embedding via API Catalog.
    Implements .encode() compatible with sentence-transformers.
    """

    def __init__(self, api_key: str, model: str = _NVIDIA_EMBED_MODEL):
        if not api_key:
            raise ValueError("NVIDIA_API_KEY required for NvidiaEmbedder")
        self._api_key = api_key
        self._model   = model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }

    def encode(
        self,
        sentences:            list[str],
        batch_size:           int  = _NVIDIA_BATCH_SIZE,
        normalize_embeddings: bool = True,
        show_progress_bar:    bool = False,
    ) -> np.ndarray:
        all_vecs: list[list[float]] = []
        n_batches = (len(sentences) + batch_size - 1) // batch_size

        for i in range(0, len(sentences), batch_size):
            batch    = sentences[i : i + batch_size]
            batch_no = i // batch_size + 1
            if show_progress_bar:
                print(f"[Embed] Batch {batch_no}/{n_batches}")

            data = self._post_with_retry({
                "model":           self._model,
                "input":           batch,
                "input_type":      "passage",
                "encoding_format": "float",
                "truncate":        "END",
            })
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            all_vecs.extend(e["embedding"] for e in embeddings)

        arr = np.array(all_vecs, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr   = arr / (norms + 1e-10)
        return arr

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Embed a single query with input_type='query' (different from passage)."""
        data = self._post_with_retry({
            "model":           self._model,
            "input":           [query],
            "input_type":      "query",
            "encoding_format": "float",
            "truncate":        "END",
        })
        vec = np.array(data["data"][0]["embedding"], dtype="float32")
        if normalize:
            vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.reshape(1, -1)

    def _post_with_retry(self, payload: dict) -> dict:
        import requests
        last_err: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    _NVIDIA_EMBED_URL, json=payload,
                    headers=self._headers, timeout=30,
                )

                # 404 means wrong endpoint — retrying is pointless, fail immediately
                if resp.status_code == 404:
                    raise RuntimeError(
                        f"NVIDIA Embeddings 404: endpoint not found. "
                        f"Check NVIDIA_EMBED_MODEL and API key permissions. "
                        f"URL: {_NVIDIA_EMBED_URL}"
                    )

                if resp.status_code in _RETRYABLE_STATUSES:
                    wait = _RETRY_DELAY * (2 ** (attempt - 1))
                    logger.warning("[Embed] HTTP %d — retry in %.1fs", resp.status_code, wait)
                    last_err = Exception(f"HTTP {resp.status_code}")
                    if attempt < _MAX_RETRIES:
                        time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp.json()

            except RuntimeError:
                raise   # re-raise our own errors immediately (404, etc.)
            except Exception as e:
                last_err = e
                wait = _RETRY_DELAY * (2 ** (attempt - 1))
                logger.warning("[Embed] Error attempt %d/%d: %s", attempt, _MAX_RETRIES, e)
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(f"NVIDIA Embeddings failed after {_MAX_RETRIES} attempts: {last_err}")


class LocalEmbedder:
    """
    sentence-transformers fallback embedder.
    Produces 384-dim embeddings, no API key required.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("[Embed] Loading local model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(
        self,
        sentences:            list[str],
        batch_size:           int  = 32,
        normalize_embeddings: bool = True,
        show_progress_bar:    bool = False,
    ) -> np.ndarray:
        return self._get_model().encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        return self.encode([query], normalize_embeddings=normalize)


def get_embedder(
    backend:  str | None = None,
    api_key:  str | None = None,
    model:    str | None = None,
) -> NvidiaEmbedder | LocalEmbedder:
    """
    Factory with automatic fallback.

    Priority:
      1. NVIDIA NemoRetriever (if NVIDIA_EMBED_BACKEND=nvidia_api and key is set)
         → If NVIDIA API fails a probe request, falls back to local automatically.
      2. Local sentence-transformers (always works, no API key needed)

    The fallback is silent in production — a WARNING is logged but the pipeline
    continues. This prevents a wrong API key or endpoint from taking down the
    entire bootstrap process.
    """
    import os
    backend = backend or os.getenv("NVIDIA_EMBED_BACKEND", "local")
    api_key = api_key or os.getenv("NVIDIA_API_KEY", "")

    local_model = model or os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    if backend == "nvidia_api" and api_key:
        embed_model = model or os.getenv("NVIDIA_EMBED_MODEL", _NVIDIA_EMBED_MODEL)
        logger.info("[Embed] Attempting NVIDIA NemoRetriever (%s)…", embed_model)

        # Probe the API with a single short string before committing to it.
        # If the probe fails, fall back to local immediately rather than
        # discovering the failure mid-way through 823 chunks.
        try:
            probe = NvidiaEmbedder(api_key=api_key, model=embed_model)
            probe._post_with_retry({
                "model":           embed_model,
                "input":           ["probe"],
                "input_type":      "passage",
                "encoding_format": "float",
                "truncate":        "END",
            })
            logger.info("[Embed] ✅ NVIDIA NemoRetriever probe OK — using cloud embeddings")
            return probe
        except Exception as e:
            logger.warning(
                "[Embed] ⚠️  NVIDIA probe failed (%s) — falling back to local sentence-transformers.\n"
                "       To silence this: set NVIDIA_EMBED_BACKEND=local in .env",
                e,
            )
            # Fall through to local

    logger.info("[Embed] Using local sentence-transformers: %s", local_model)
    return LocalEmbedder(model_name=local_model)