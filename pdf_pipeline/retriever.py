"""
pdf_pipeline/retriever.py
==========================
Hybrid retrieval: FAISS dense search + BM25 sparse search, fused via RRF.

Why hybrid matters for loan queries:
  FAISS alone misses: "CIBIL 750", "₹35,000", exact bank names, acronyms
  BM25 alone misses:  paraphrases ("monthly salary" vs "income"), synonyms
  Together:           near-perfect recall on both exact and semantic queries

Pipeline:
  retrieve(query)
    ├── FAISS top-20      (semantic similarity)
    ├── BM25  top-20      (keyword / exact-term)
    ├── RRF fusion        (merge + deduplicate, FAISS weighted 1.2×)
    ├── field_hint boost  (query-aware score bump for matching field hints)
    └── Reranker          (Nemotron cross-encoder, optional, top-5 final)
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pdf_pipeline.chunker import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DocChunk alias for backward-compat with existing retrieval.py callers
# ---------------------------------------------------------------------------

@dataclass
class DocChunk:
    text:             str
    source:           str
    bank:             str
    doc_type:         str
    chunk_id:         int   = 0
    similarity_score: float = 0.0
    field_hint:       str   = ""
    section_title:    str   = ""
    element_type:     str   = "text"
    page_number:      int   = 0

    @classmethod
    def from_document_chunk(cls, dc: DocumentChunk, score: float = 0.0) -> "DocChunk":
        return cls(
            text=dc.text, source=dc.source, bank=dc.bank,
            doc_type=dc.doc_type, chunk_id=dc.chunk_id,
            similarity_score=score, field_hint=dc.field_hint,
            section_title=dc.section_title, element_type=dc.element_type,
            page_number=dc.page_number,
        )


# ---------------------------------------------------------------------------
# BM25 (pure Python, no dependencies)
# ---------------------------------------------------------------------------

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._idf:       dict[str, float]       = {}
        self._tf_corpus: list[dict[str, float]]  = []
        self._avg_dl = 0.0
        self._n      = 0

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        # Allow hyphens so terms like "pre-approved", "co-applicant" stay intact
        return re.findall(r"[a-z0-9₹%.,\-]+", text.lower())

    def fit(self, corpus: list[str]) -> "BM25":
        self._n         = len(corpus)
        self._tf_corpus = []
        total_len       = 0
        df: dict[str, int] = defaultdict(int)

        for doc in corpus:
            tokens = self._tokenise(doc)
            total_len += len(tokens)
            tf: dict[str, float] = defaultdict(float)
            for t in tokens:
                tf[t] += 1.0
            self._tf_corpus.append(dict(tf))
            for t in set(tokens):
                df[t] += 1

        self._avg_dl = total_len / max(self._n, 1)
        for term, freq in df.items():
            self._idf[term] = math.log((self._n - freq + 0.5) / (freq + 0.5) + 1.0)
        return self

    def scores(self, query: str) -> np.ndarray:
        q_tokens = self._tokenise(query)
        result   = np.zeros(self._n, dtype="float32")
        for term in q_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, tf_doc in enumerate(self._tf_corpus):
                tf  = tf_doc.get(term, 0.0)
                dl  = sum(tf_doc.values())
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                result[i] += idf * (num / den)
        return result

    def top_k(self, query: str, k: int, filter_fn=None) -> list[tuple[int, float]]:
        raw = self.scores(query)
        if filter_fn:
            for i in range(self._n):
                if not filter_fn(i):
                    raw[i] = -1.0
        idx = np.argsort(raw)[::-1]
        out = []
        for i in idx:
            if raw[i] <= 0:
                break
            out.append((int(i), float(raw[i])))
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: list[list[int]],
    k:            int         = 60,
    weights:      list[float] | None = None,
) -> list[tuple[int, float]]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[int, float] = defaultdict(float)
    for rl, w in zip(ranked_lists, weights):
        for rank, idx in enumerate(rl, start=1):
            scores[idx] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Query field hint patterns
# ---------------------------------------------------------------------------

_QUERY_FIELD_PATTERNS = [
    (re.compile(r"\bcibil\b|credit.?score",                re.I), "credit_score"),
    (re.compile(r"\bincome\b|\bsalary\b|\bearning",        re.I), "monthly_income"),
    (re.compile(r"\bage\b|\byears.old\b",                  re.I), "age"),
    (re.compile(r"\bdti\b|debt.to.income",                 re.I), "dti_ratio"),
    (re.compile(r"\bemployment\b|\bsalaried\b",            re.I), "employment_type"),
    (re.compile(r"\bexperience\b|\btenure\b",              re.I), "work_experience_months"),
    (re.compile(r"\binterest\s*rate\b|\broi\b",            re.I), "interest_rate"),
    (re.compile(r"\bfee\b|\bcharge\b|\bpenalty\b|\bprocessing\b|\bforeclos\b", re.I), "fees"),
]

def _query_field_hints(query: str) -> list[str]:
    return [fld for pat, fld in _QUERY_FIELD_PATTERNS if pat.search(query)]


# ---------------------------------------------------------------------------
# Retrieval service
# ---------------------------------------------------------------------------

class RetrievalService:

    def __init__(self, index_dir: Path, embed_dim: int = 384):
        self._index_dir = index_dir
        self._embed_dim = embed_dim
        self._chunks:    list[DocChunk] = []
        self._index      = None
        self._bm25:      BM25 | None    = None
        self._embedder   = None

    def _get_embedder(self):
        if self._embedder is None:
            from pdf_pipeline.embeddings import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def load_chunks(self, chunks: list[DocChunk]) -> None:
        """Load pre-built DocChunks directly (used after extraction pipeline)."""
        self._chunks = chunks
        logger.info("[Retrieval] Loaded %d chunks", len(chunks))

    def build_index(self, save: bool = True) -> None:
        """Build FAISS dense index + BM25 sparse index from loaded chunks."""
        if not self._chunks:
            raise ValueError("No chunks loaded. Call load_chunks() first.")

        import faiss

        texts    = [c.text for c in self._chunks]
        embedder = self._get_embedder()

        logger.info("[Retrieval] Embedding %d chunks…", len(texts))
        vecs = embedder.encode(texts, batch_size=64, show_progress_bar=True)
        dim  = vecs.shape[1]

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs.astype("float32"))

        self._bm25 = BM25().fit(texts)

        logger.info("[Retrieval] FAISS index: %d vectors dim=%d", self._index.ntotal, dim)

        if save:
            self._index_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self._index_dir / "index.faiss"))
            with open(self._index_dir / "chunks.pkl", "wb") as f:
                pickle.dump(self._chunks, f)
            with open(self._index_dir / "bm25.pkl", "wb") as f:
                pickle.dump(self._bm25, f)
            logger.info("[Retrieval] Saved index to %s", self._index_dir)

    def load_index(self) -> bool:
        """Load previously saved FAISS index and chunks from disk."""
        faiss_path  = self._index_dir / "index.faiss"
        chunks_path = self._index_dir / "chunks.pkl"
        if not faiss_path.exists() or not chunks_path.exists():
            return False
        try:
            import faiss
            self._index = faiss.read_index(str(faiss_path))
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            bm25_path = self._index_dir / "bm25.pkl"
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    self._bm25 = pickle.load(f)
            logger.info("[Retrieval] Loaded index: %d chunks", len(self._chunks))
            return True
        except ModuleNotFoundError as e:
            # faiss-cpu not installed — give a clear actionable message
            logger.error(
                "[Retrieval] faiss-cpu not installed. "
                "Fix: pip install faiss-cpu   then re-run bootstrap."
            )
            return False
        except Exception as e:
            logger.error("[Retrieval] Failed to load index: %s", e)
            return False

    def ensure_ready(self) -> None:
        """
        Load the FAISS index if not already loaded.
        Logs a clear warning instead of crashing — the API can still start
        and serve health checks / rule-engine-only queries without retrieval.
        Retrieval calls will return empty results and trigger the web search
        fallback rather than bringing down the whole process.
        """
        if self._index is None:
            loaded = self.load_index()
            if not loaded:
                # Check whether it's a missing install vs missing index file
                try:
                    import faiss  # noqa: F401
                    logger.warning(
                        "[Retrieval] ⚠️  No FAISS index found at %s. "
                        "RAG retrieval will return empty results until you run: "
                        "python bootstrap.py",
                        self._index_dir,
                    )
                except ModuleNotFoundError:
                    logger.warning(
                        "[Retrieval] ⚠️  faiss-cpu not installed. "
                        "RAG retrieval disabled. Fix: pip install faiss-cpu"
                    )
                # Do NOT raise — let the API start so health checks and
                # rule-engine-only queries still work.

    def retrieve(
        self,
        query:     str,
        top_k:     int                 = 5,
        banks:     Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
        fetch_k:   int                 = 20,
    ) -> list[DocChunk]:
        """
        Hybrid retrieval: FAISS + BM25 + RRF + field_hint boost + optional rerank.
        Returns empty list if index is not loaded — API stays up, web-search
        fallback fires in pipeline.py so queries still get answered.

        Args:
            query:     natural language query
            top_k:     final number of chunks to return
            banks:     filter to specific banks
            doc_types: filter to specific document types
            fetch_k:   candidates per retrieval method before fusion
        """
        self.ensure_ready()
        if self._index is None:
            logger.debug("[Retrieval] Index not available — returning empty results")
            return []
        candidates = self._hybrid_search(query, fetch_k, banks, doc_types)

        import os
        if os.getenv("USE_RERANKER", "true").lower() == "true" and len(candidates) > top_k:
            try:
                from pdf_pipeline.reranker import rerank_documents
                result = rerank_documents(query, candidates, top_n=top_k)
                logger.info("[Retrieval] Reranked %d → %d (reranked=%s)",
                            len(candidates), len(result.documents), result.reranked)
                return result.documents
            except Exception as e:
                logger.warning("[Retrieval] Reranker failed, using RRF order: %s", e)

        return candidates[:top_k]

    def _hybrid_search(
        self,
        query:     str,
        fetch_k:   int,
        banks:     Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[DocChunk]:
        faiss_results = self._faiss_search(query, fetch_k, banks, doc_types)
        bm25_results  = self._bm25_search(query, fetch_k, banks, doc_types)

        faiss_ranked  = [idx for idx, _ in faiss_results]
        faiss_score   = {idx: sc for idx, sc in faiss_results}
        bm25_ranked   = [idx for idx, _ in bm25_results]

        logger.debug("[Retrieval] FAISS=%d  BM25=%d", len(faiss_ranked), len(bm25_ranked))

        fused = _rrf_fuse([faiss_ranked, bm25_ranked], k=60, weights=[1.2, 1.0])

        chunks: list[DocChunk] = []
        scores: list[float]    = []
        seen:   set[int]       = set()

        for idx, rrf_score in fused[:fetch_k]:
            if idx in seen or idx >= len(self._chunks):
                continue
            seen.add(idx)
            chunk = self._chunks[idx]
            chunk.similarity_score = faiss_score.get(idx, rrf_score)
            chunks.append(chunk)
            scores.append(rrf_score)

        # Field hint boost
        query_hints = _query_field_hints(query)
        if query_hints:
            for i, chunk in enumerate(chunks):
                if chunk.field_hint in query_hints:
                    scores[i] = min(scores[i] + 0.15, 1.0)

        # Re-sort
        paired = sorted(zip(chunks, scores), key=lambda x: -x[1])
        result = [c for c, _ in paired]
        for c, s in paired:
            c.similarity_score = s

        logger.info("[Retrieval] Hybrid → %d candidates (hints=%s)",
                    len(result), query_hints or "none")
        return result

    def _faiss_search(self, query, fetch_k, banks, doc_types):
        embedder = self._get_embedder()
        if hasattr(embedder, "encode_query"):
            q_vec = embedder.encode_query(query, normalize=True)
        else:
            q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32").reshape(1, -1)

        search_k = min(fetch_k * 10, len(self._chunks))
        scores, indices = self._index.search(q_vec, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            c = self._chunks[idx]
            if banks     and c.bank     not in banks:     continue
            if doc_types and c.doc_type not in doc_types: continue
            results.append((int(idx), float(score)))
            if len(results) >= fetch_k:
                break
        return results

    def _bm25_search(self, query, fetch_k, banks, doc_types):
        if self._bm25 is None:
            return []
        def _filter(i):
            c = self._chunks[i]
            if banks     and c.bank     not in banks:     return False
            if doc_types and c.doc_type not in doc_types: return False
            return True
        return self._bm25.top_k(query, k=fetch_k, filter_fn=_filter)