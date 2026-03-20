"""
Semantic Cache  (v2 — with TTL)
================================
FIX #9: Added TTL so stale financial data is never served.
"""

from __future__ import annotations
import hashlib
import time
import pickle
from pathlib import Path
from collections import OrderedDict

import numpy as np

from config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_ENTRIES, CACHE_DIR

CACHE_TTL_SECONDS = 60 * 60 * 6   # 6 hours — financial data changes; don't cache too long


class SemanticCache:

    def __init__(self):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._embedder = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cache_file = CACHE_DIR / "semantic_cache.pkl"
        self._load()

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            from config import EMBEDDING_MODEL
            self._embedder = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        emb = self._get_embedder().encode([text], normalize_embeddings=True)
        return np.array(emb[0], dtype="float32")

    def _is_expired(self, entry: dict) -> bool:
        return (time.time() - entry.get("ts", 0)) > CACHE_TTL_SECONDS

    def get(self, query: str) -> dict | None:
        if not self._store:
            return None

        q_emb = self._embed(query)
        best_score = -1.0
        best_key   = None

        for key, entry in self._store.items():
            if self._is_expired(entry):
                continue   # skip stale entries
            score = float(np.dot(q_emb, entry["embedding"]))
            if score > best_score:
                best_score = score
                best_key   = key

        if best_score >= CACHE_SIMILARITY_THRESHOLD and best_key:
            print(f"[Cache] Hit (score={best_score:.3f})")
            return self._store[best_key]["response"]

        return None

    def set(self, query: str, response: dict) -> None:
        key = hashlib.md5(query.encode()).hexdigest()
        emb = self._embed(query)
        self._store[key] = {
            "embedding": emb,
            "response":  response,
            "ts":        time.time(),
        }
        # Evict expired + oldest over limit
        expired = [k for k, v in self._store.items() if self._is_expired(v)]
        for k in expired:
            del self._store[k]
        while len(self._store) > CACHE_MAX_ENTRIES:
            self._store.popitem(last=False)
        self._save()

    def _save(self):
        try:
            with open(self._cache_file, "wb") as f:
                pickle.dump(self._store, f)
        except Exception as e:
            print(f"[Cache] Save error: {e}")

    def _load(self):
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "rb") as f:
                    self._store = pickle.load(f)
                # Purge expired on load
                expired = [k for k, v in self._store.items() if self._is_expired(v)]
                for k in expired:
                    del self._store[k]
                print(f"[Cache] Loaded {len(self._store)} valid entries (TTL={CACHE_TTL_SECONDS//3600}h)")
            except Exception:
                self._store = OrderedDict()

    def clear(self):
        self._store.clear()
        if self._cache_file.exists():
            self._cache_file.unlink()
        print("[Cache] Cleared")
