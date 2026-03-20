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
"""

from __future__ import annotations
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# NVIDIA API endpoint for text embeddings
_NVIDIA_EMBED_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nemoretriever-300m-embed-v2/embeddings"

# Max passages per single API call (NVIDIA limit)
_NVIDIA_BATCH_SIZE = 96


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

    def encode(
        self,
        sentences: list[str],
        batch_size: int = _NVIDIA_BATCH_SIZE,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of strings.  Returns float32 ndarray (N, 1024).
        Automatically batches if len(sentences) > batch_size.
        """
        import requests

        all_vecs: list[list[float]] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            if show_progress_bar:
                print(f"[NvidiaEmbed] Embedding batch {i//batch_size + 1} ({len(batch)} items)…")

            payload = {
                "model": "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
                "input": batch,
                "input_type": "passage",   # "passage" for docs, "query" for queries
                "encoding_format": "float",
                "truncate": "END",
            }

            t0 = time.perf_counter()
            resp = requests.post(
                _NVIDIA_EMBED_URL,
                json=payload,
                headers=self._headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - t0) * 1000

            # Response shape: {"data": [{"embedding": [...], "index": N}, ...]}
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            batch_vecs = [e["embedding"] for e in embeddings]
            all_vecs.extend(batch_vecs)

            logger.debug(
                "[NvidiaEmbed] Batch %d/%d — %d items in %.0fms",
                i // batch_size + 1,
                (len(sentences) - 1) // batch_size + 1,
                len(batch), elapsed,
            )

        arr = np.array(all_vecs, dtype="float32")

        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / (norms + 1e-10)

        return arr

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embed a single query string.
        Uses input_type="query" (slightly different from passage embedding).
        """
        import requests

        payload = {
            "model": "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
            "input": [query],
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "END",
        }
        resp = requests.post(
            _NVIDIA_EMBED_URL,
            json=payload,
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        vec = np.array(resp.json()["data"][0]["embedding"], dtype="float32")
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

    backend  = backend  or NVIDIA_EMBED_BACKEND
    api_key  = api_key  or NVIDIA_API_KEY

    if backend == "nvidia_api":
        logger.info("[Embedder] Using NVIDIA NemoRetriever 300M (API)")
        return NvidiaEmbedder(api_key=api_key)
    else:
        logger.info("[Embedder] Using local sentence-transformers: %s", EMBEDDING_MODEL)
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
