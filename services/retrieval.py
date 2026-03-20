"""
Retrieval Service — Hybrid RAG Layer (v2)
==========================================
Adds to v1:
    ✅ BM25 keyword search  — exact-term recall (acronyms, numbers, named entities)
    ✅ Hybrid fusion        — RRF (Reciprocal Rank Fusion) merges FAISS + BM25
    ✅ Query-aware reranking — field_hint boosting aligns retrieval with query intent
    ✅ Batch-safe embedding  — build_index() respects embedder's batch_size
    ✅ Dim-mismatch guard   — detects FAISS/embedding dim mismatch at load time

Architecture:
    retrieve(query)
        ├── FAISS dense search     (semantic similarity)
        ├── BM25 sparse search     (keyword / exact-term)
        ├── RRF fusion             (merge + deduplicate by rank)
        ├── Query-aware boost      (field_hint match → score bump)
        └── Reranker               (Nemotron cross-encoder, optional)

Why hybrid matters for loan queries:
    FAISS alone misses:  "CIBIL 750", "₹35,000", "Axis Bank", "DTI 40%"
    BM25 alone misses:   paraphrases, synonyms, context
    Together:            near-perfect recall on both exact and semantic queries
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    DATA_DIR, FAISS_INDEX_DIR, EMBEDDING_MODEL, EMBEDDING_DIM,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    USE_RERANKER, RERANKER_FETCH_K, RERANKER_TOP_N,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document Chunk
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

# ---------------------------------------------------------------------------
# Source mapping
# ---------------------------------------------------------------------------

_SOURCE_MAP = {
    "axis":        ("Axis",        "eligibility"),
    "icici":       ("ICICI",       "eligibility"),
    "hdfc_pdfs":   ("HDFC",        "pdf"),
    "sbi_pdfs":    ("SBI",         "pdf"),
    "paisabazaar": ("Paisabazaar", "comparison"),
    "bankbazaar":  ("BankBazaar",  "comparison"),
    "rbi":         ("RBI",         "regulatory"),
}

# ---------------------------------------------------------------------------
# Query-aware field hint extraction
# ---------------------------------------------------------------------------

_QUERY_FIELD_PATTERNS: list[tuple[str, str]] = [
    (r"\bcibil\b|credit.?score",                 "credit_score"),
    (r"\bincome\b|\bsalary\b|\bearning",          "monthly_income"),
    (r"\bage\b|\byears.old\b",                    "age"),
    (r"\bdti\b|debt.to.income|emi.*income",       "dti_ratio"),
    (r"\bemployment\b|\bsalaried\b|\bself.emp",   "employment_type"),
    (r"\bexperience\b|\btenure\b|\bmonths.work",  "work_experience_months"),
]

def _extract_query_field_hints(query: str) -> list[str]:
    """Return field names that the query is explicitly asking about."""
    ql    = query.lower()
    hints = []
    for pattern, field_name in _QUERY_FIELD_PATTERNS:
        if re.search(pattern, ql):
            hints.append(field_name)
    return hints


# ---------------------------------------------------------------------------
# BM25 — pure-Python, no dependencies
# ---------------------------------------------------------------------------

class BM25:
    """
    Okapi BM25 over a fixed corpus.

    k1=1.5, b=0.75 are standard defaults that work well on short financial chunks.
    Tokenises on non-alphanumeric boundaries (handles ₹35,000 and CIBIL correctly).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1    = k1
        self.b     = b
        self._idf:       dict[str, float]      = {}
        self._tf_corpus: list[dict[str, float]] = []
        self._avg_dl     = 0.0
        self._n          = 0

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        # Fix 6: include .,  so "₹35,000" → ["₹35,000"] and "7.5%" → ["7.5%"]
        # instead of splitting into ["35","000"] / ["7","5"] which breaks
        # exact-match recall for EMI, interest-rate, and income queries.
        return re.findall(r"[a-z0-9₹%.,]+", text.lower())

    def fit(self, corpus: list[str]) -> "BM25":
        self._n          = len(corpus)
        self._tf_corpus  = []
        self._idf        = {}
        total_len        = 0
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
            self._idf[term] = math.log(
                (self._n - freq + 0.5) / (freq + 0.5) + 1.0
            )
        return self

    def scores(self, query: str) -> np.ndarray:
        """Return BM25 score for each document in the corpus."""
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

    def top_k(
        self,
        query:     str,
        k:         int,
        filter_fn: Optional[callable] = None,
    ) -> list[tuple[int, float]]:
        """Return (index, score) pairs sorted best-first, up to k results."""
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
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    ranked_lists: list[list[int]],
    k:            int   = 60,
    weights:      list[float] | None = None,
) -> list[tuple[int, float]]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    RRF score(d) = Σ weight_i / (k + rank_i(d))

    Args:
        ranked_lists : each list is a sequence of chunk indices, best-first
        k            : RRF constant (60 is the standard; higher = flatter scores)
        weights      : per-list multiplier; defaults to equal weights

    Returns:
        List of (chunk_idx, rrf_score) sorted best-first.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[int, float] = defaultdict(float)
    for rl, w in zip(ranked_lists, weights):
        for rank, idx in enumerate(rl, start=1):
            scores[idx] += w / (k + rank)

    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# RetrievalService — Hybrid FAISS + BM25
# ---------------------------------------------------------------------------

class RetrievalService:
    """
    Hybrid retrieval: FAISS (dense) + BM25 (sparse) fused with RRF,
    optionally followed by Nemotron reranker.
    """

    def __init__(self):
        self._chunks:   list[DocChunk] = []
        self._index     = None          # FAISS index
        self._vectors   = None          # np.ndarray of embeddings
        self._embedder  = None
        self._bm25:     BM25 | None = None
        self._is_built  = False

    # ------------------------------------------------------------------ init

    def _get_embedder(self):
        if self._embedder is None:
            from services.nvidia_embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_faiss(self):
        try:
            import faiss
            return faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Run: pip install faiss-cpu")

    # ------------------------------------------------------------------ chunk

    @staticmethod
    def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        words = text.split()
        if not words:
            return []
        chunks, start = [], 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += size - overlap
        return chunks

    # ------------------------------------------------------------------ load

    def load_documents(self) -> int:
        self._chunks = []
        chunk_id     = 0
        processed_stems: set[str] = set()

        # Source 1: structured JSON from pdf_pipeline
        if PROCESSED_DATA_DIR.exists():
            for json_path in sorted(PROCESSED_DATA_DIR.glob("*.json")):
                try:
                    data       = json.loads(json_path.read_text(encoding="utf-8"))
                    raw_chunks = data.get("chunks", [])
                    if not raw_chunks:
                        continue
                    for rc in raw_chunks:
                        self._chunks.append(DocChunk(
                            text=rc.get("text", ""),
                            source=rc.get("source", json_path.stem),
                            bank=rc.get("bank", "Unknown"),
                            doc_type=rc.get("doc_type", "pdf"),
                            chunk_id=chunk_id,
                            field_hint=rc.get("field_hint", ""),
                            section_title=rc.get("section_title", ""),
                            element_type=rc.get("element_type", "text"),
                            page_number=rc.get("page_number", 0),
                        ))
                        chunk_id += 1
                    processed_stems.add(json_path.stem)
                    logger.info("[Retrieval] Loaded JSON: %s (%d chunks)",
                                json_path.name, len(raw_chunks))
                except Exception as e:
                    logger.warning("[Retrieval] Failed %s: %s", json_path.name, e)

        # Source 2: plain-text / unprocessed files
        for folder, (bank, doc_type) in _SOURCE_MAP.items():
            folder_path = DATA_DIR / folder
            if not folder_path.exists():
                continue
            for file_path in sorted(folder_path.glob("*")):
                if file_path.suffix not in (".txt", ".pdf"):
                    continue
                if file_path.stem in processed_stems:
                    continue
                text = self._read_file(file_path)
                if not text.strip():
                    continue
                for chunk_text in self._chunk_text(text):
                    self._chunks.append(DocChunk(
                        text=chunk_text, source=file_path.name,
                        bank=bank, doc_type=doc_type, chunk_id=chunk_id,
                    ))
                    chunk_id += 1

        logger.info("[Retrieval] Total chunks loaded: %d", len(self._chunks))
        return len(self._chunks)

    @staticmethod
    def _read_file(path: Path) -> str:
        if path.suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    return "\n".join(p.extract_text() or "" for p in pdf.pages)
            except ImportError:
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(path))
                    return "\n".join(p.extract_text() or "" for p in reader.pages)
                except ImportError:
                    return ""
        return ""

    # ------------------------------------------------------------------ build

    def build_index(self, save: bool = True) -> None:
        """Embed all chunks, build FAISS index, and fit BM25."""
        if not self._chunks:
            self.load_documents()

        faiss    = self._get_faiss()
        embedder = self._get_embedder()
        texts    = [c.text for c in self._chunks]

        # ── FAISS ────────────────────────────────────────────────────────────
        print(f"[Retrieval] Embedding {len(texts)} chunks…")

        # Use embedder's preferred batch size if it exposes one
        batch_size = getattr(embedder, "_batch_size", 16)
        # For NVIDIA embedder keep it at 16 to avoid rate-limit hits
        if hasattr(embedder, "_api_key"):  # NvidiaEmbedder
            batch_size = 16

        vectors = embedder.encode(texts, batch_size=batch_size, show_progress_bar=True)
        vectors = np.array(vectors, dtype="float32")
        # Fix 5: do NOT re-normalize — embedder already returns unit vectors
        # when normalize_embeddings=True (which is the default). Double-normalizing
        # a unit vector distorts the L2 norm away from 1.0 for non-unit inputs.

        # Dimension sanity check
        actual_dim = vectors.shape[1]
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                "[Retrieval] Embedding dim mismatch: vectors=%d  EMBEDDING_DIM=%d  "
                "— using actual dim. Update EMBEDDING_DIM in .env.",
                actual_dim, EMBEDDING_DIM,
            )

        index = faiss.IndexFlatIP(actual_dim)
        index.add(vectors)
        self._index   = index
        self._vectors = vectors
        self._is_built = True

        # ── BM25 ─────────────────────────────────────────────────────────────
        print(f"[Retrieval] Fitting BM25 on {len(texts)} chunks…")
        self._bm25 = BM25().fit(texts)

        if save:
            self._save(vectors)

    def _save(self, vectors: np.ndarray) -> None:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss = self._get_faiss()
        faiss.write_index(self._index, str(FAISS_INDEX_DIR / "index.faiss"))
        with open(FAISS_INDEX_DIR / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        np.save(FAISS_INDEX_DIR / "vectors.npy", vectors)
        # Save BM25 separately (lightweight)
        with open(FAISS_INDEX_DIR / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        print(f"[Retrieval] Index saved ({len(self._chunks)} chunks, BM25 included)")

    def load_index(self) -> bool:
        """Load pre-built FAISS index + BM25 from disk."""
        idx_path   = FAISS_INDEX_DIR / "index.faiss"
        chunk_path = FAISS_INDEX_DIR / "chunks.pkl"
        if not (idx_path.exists() and chunk_path.exists()):
            return False

        faiss = self._get_faiss()
        self._index = faiss.read_index(str(idx_path))

        # Dimension check
        stored_dim  = self._index.d
        if stored_dim != EMBEDDING_DIM:
            logger.warning(
                "[Retrieval] FAISS index dim=%d but EMBEDDING_DIM=%d. "
                "If you switched embedders, delete models/ and re-run bootstrap.py.",
                stored_dim, EMBEDDING_DIM,
            )

        with open(chunk_path, "rb") as f:
            self._chunks = pickle.load(f)
        try:
            self._vectors = np.load(FAISS_INDEX_DIR / "vectors.npy")
        except FileNotFoundError:
            self._vectors = None

        # BM25 (may not exist if index was built with old code → rebuild inline)
        bm25_path = FAISS_INDEX_DIR / "bm25.pkl"
        if bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    self._bm25 = pickle.load(f)
                logger.info("[Retrieval] Loaded BM25 index")
            except Exception as e:
                logger.warning("[Retrieval] BM25 load failed (%s) — rebuilding", e)
                self._bm25 = BM25().fit([c.text for c in self._chunks])
        else:
            logger.info("[Retrieval] BM25 not found — building now (one-time)")
            self._bm25 = BM25().fit([c.text for c in self._chunks])
            try:
                with open(bm25_path, "wb") as f:
                    pickle.dump(self._bm25, f)
            except Exception:
                pass

        self._is_built = True
        print(f"[Retrieval] Loaded index: {len(self._chunks)} chunks "
              f"(FAISS dim={stored_dim}, BM25={'yes' if self._bm25 else 'no'})")
        return True

    def ensure_ready(self) -> None:
        if not self._is_built:
            if not self.load_index():
                self.load_documents()
                self.build_index()

    # ------------------------------------------------------------------ search

    def _faiss_search(
        self,
        query:     str,
        fetch_k:   int,
        banks:     Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[tuple[int, float]]:
        """
        Dense FAISS search.
        Returns (chunk_index, score) pairs, best-first.
        """
        embedder = self._get_embedder()

        # Use encode_query if available (NVIDIA embedder uses input_type="query")
        if hasattr(embedder, "encode_query"):
            q_vec = embedder.encode_query(query, normalize=True)
        else:
            q_vec = embedder.encode([query], normalize_embeddings=True)

        q_vec = np.array(q_vec, dtype="float32").reshape(1, -1)

        search_k   = min(fetch_k * 10, len(self._chunks))
        scores, indices = self._index.search(q_vec, search_k)

        results: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if banks     and chunk.bank     not in banks:
                continue
            if doc_types and chunk.doc_type not in doc_types:
                continue
            results.append((int(idx), float(score)))
            if len(results) >= fetch_k:
                break
        return results

    def _bm25_search(
        self,
        query:     str,
        fetch_k:   int,
        banks:     Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[tuple[int, float]]:
        """
        Sparse BM25 keyword search.
        Returns (chunk_index, score) pairs, best-first.
        """
        if self._bm25 is None:
            return []

        def _filter(i: int) -> bool:
            c = self._chunks[i]
            if banks     and c.bank     not in banks:     return False
            if doc_types and c.doc_type not in doc_types: return False
            return True

        return self._bm25.top_k(query, k=fetch_k, filter_fn=_filter)

    def _query_aware_boost(
        self,
        chunks:      list[DocChunk],
        scores:      list[float],
        query:       str,
        boost:       float = 0.15,
    ) -> list[float]:
        """
        Boost scores for chunks whose field_hint matches a field the query asks about.

        Example: query = "What is the CIBIL requirement?" → boost chunks with
        field_hint="credit_score" by `boost` amount.

        This is query-aware reranking without any API call — pure heuristic.
        """
        query_fields = _extract_query_field_hints(query)
        if not query_fields:
            return scores

        boosted = list(scores)
        for i, chunk in enumerate(chunks):
            if chunk.field_hint and chunk.field_hint in query_fields:
                boosted[i] = min(boosted[i] + boost, 1.0)
                logger.debug(
                    "[Retrieval] field_hint boost: chunk %d (%s) +%.2f for query field %s",
                    chunk.chunk_id, chunk.field_hint, boost, query_fields,
                )
        return boosted

    def _hybrid_search(
        self,
        query:     str,
        fetch_k:   int,
        banks:     Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[DocChunk]:
        """
        Hybrid FAISS + BM25 search fused with Reciprocal Rank Fusion.

        Steps:
            1. FAISS retrieves `fetch_k` semantic candidates
            2. BM25 retrieves `fetch_k` keyword candidates
            3. RRF merges both ranked lists (FAISS weighted 1.2×, BM25 1.0×)
            4. Query-aware field_hint boost applied
            5. Returns top `fetch_k` chunks with fused scores
        """
        # Dense
        faiss_results = self._faiss_search(query, fetch_k, banks, doc_types)
        faiss_ranked  = [idx for idx, _ in faiss_results]
        faiss_score   = {idx: sc for idx, sc in faiss_results}

        # Sparse
        bm25_results  = self._bm25_search(query, fetch_k, banks, doc_types)
        bm25_ranked   = [idx for idx, _ in bm25_results]

        logger.info(
            "[Retrieval] FAISS=%d  BM25=%d  (before fusion)",
            len(faiss_ranked), len(bm25_ranked),
        )

        # RRF fusion — FAISS weighted slightly higher (semantic > keyword for this domain)
        fused = _rrf_fuse(
            [faiss_ranked, bm25_ranked],
            k=60,
            weights=[1.2, 1.0],
        )

        # Materialise chunks + apply query-aware boost
        chunks_out: list[DocChunk] = []
        scores_out: list[float]    = []
        seen: set[int]             = set()

        for idx, rrf_score in fused[:fetch_k]:
            if idx in seen or idx >= len(self._chunks):
                continue
            seen.add(idx)
            chunk = self._chunks[idx]
            # Use FAISS cosine score if available, else normalised RRF score
            raw_score = faiss_score.get(idx, rrf_score)
            chunk.similarity_score = float(raw_score)
            chunks_out.append(chunk)
            scores_out.append(float(rrf_score))

        # Query-aware boost
        scores_out = self._query_aware_boost(chunks_out, scores_out, query)

        # Re-sort by boosted score
        paired  = sorted(zip(chunks_out, scores_out), key=lambda x: -x[1])
        chunks_out = [c for c, _ in paired]
        scores_out = [s for _, s in paired]

        # Attach final score
        for chunk, sc in zip(chunks_out, scores_out):
            chunk.similarity_score = sc

        logger.info(
            "[Retrieval] Hybrid fusion → %d chunks  (field_hints: %s)",
            len(chunks_out), _extract_query_field_hints(query) or "none",
        )
        return chunks_out

    # ------------------------------------------------------------------ retrieve

    def retrieve(
        self,
        query:     str,
        top_k:     int                  = TOP_K,
        banks:     Optional[list[str]]  = None,
        doc_types: Optional[list[str]]  = None,
    ) -> list[DocChunk]:
        """
        Main retrieval entry point.

        Pipeline:
            1. Hybrid search  (FAISS + BM25 + RRF + field_hint boost)
            2. Optional Nemotron reranker

        Falls back to FAISS-only if BM25 unavailable or reranker fails.
        """
        self.ensure_ready()

        fetch_k    = max(RERANKER_FETCH_K, top_k) if USE_RERANKER else top_k
        candidates = self._hybrid_search(query, fetch_k, banks, doc_types)

        logger.info("[Retrieval] Hybrid candidates: %d", len(candidates))

        if USE_RERANKER and len(candidates) > top_k:
            from services.reranker import rerank_documents
            result = rerank_documents(query, candidates, top_n=RERANKER_TOP_N)
            logger.info(
                "[Retrieval] After rerank: %d  reranked=%s  reason=%s",
                len(result.documents), result.reranked, result.reason,
            )
            return result.documents

        return candidates[:top_k]

    # ------------------------------------------------------------------ convenience

    def retrieve_by_bank(self, query: str, bank: str, top_k: int = TOP_K) -> list[DocChunk]:
        return self.retrieve(query, top_k=top_k, banks=[bank])

    def retrieve_regulatory(self, query: str, top_k: int = TOP_K) -> list[DocChunk]:
        return self.retrieve(query, top_k=top_k, doc_types=["regulatory"])

    def retrieve_comparison(self, query: str, top_k: int = TOP_K) -> list[DocChunk]:
        return self.retrieve(query, top_k=top_k, doc_types=["comparison"])

    # backward-compat: pure FAISS path (used by search-test endpoint)
    def _faiss_search_compat(
        self,
        query:     str,
        fetch_k:   int,
        banks:     Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[DocChunk]:
        """Legacy FAISS-only search, kept for /search-test endpoint."""
        results  = self._faiss_search(query, fetch_k, banks, doc_types)
        chunks   = []
        for idx, score in results:
            chunk = self._chunks[idx]
            chunk.similarity_score = score
            chunks.append(chunk)
        return chunks