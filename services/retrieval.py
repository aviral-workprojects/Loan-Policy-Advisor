"""
Retrieval Service — RAG Layer
==============================
- Loads text/PDF documents from data/ directory
- Chunks them with overlap
- Embeds using sentence-transformers
- Stores in FAISS with metadata
- Supports filtered retrieval by bank/source/type
"""

from __future__ import annotations
import json
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass, field, asdict
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
    text: str
    source: str        # filename
    bank: str          # e.g. "Axis", "ICICI", "RBI", "Paisabazaar"
    doc_type: str      # "eligibility" | "pdf" | "table" | "rule" | "comparison" | "regulatory"
    chunk_id: int = 0
    similarity_score: float = 0.0   # FAISS cosine similarity (populated at retrieval time)
    field_hint:    str = ""          # e.g. "monthly_income", "credit_score" — from structured PDF
    section_title: str = ""          # section heading from NVIDIA Page Elements
    element_type:  str = "text"      # "text" | "table" | "title" | "rule" | "paragraph"
    page_number:   int = 0           # source page in original PDF

# ---------------------------------------------------------------------------
# Source Mapping
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
# Retrieval Service
# ---------------------------------------------------------------------------

class RetrievalService:
    """FAISS-backed semantic retrieval with metadata filtering."""

    def __init__(self):
        self._chunks: list[DocChunk] = []
        self._index = None          # faiss index (lazy-loaded)
        self._embedder = None       # sentence transformer (lazy-loaded)
        self._is_built = False

    # ------------------------------------------------------------------ init

    def _get_embedder(self):
        if self._embedder is None:
            # Factory: uses NVIDIA NemoRetriever API or local sentence-transformers
            # depending on NVIDIA_EMBED_BACKEND setting in .env
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
        """Split text into overlapping word-level chunks."""
        words = text.split()
        if not words:
            return []
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += size - overlap
        return chunks

    # ------------------------------------------------------------------ load

    def load_documents(self) -> int:
        """
        Load documents from two sources:
        1. processed_data/*.json  — structured chunks from pdf_pipeline.py (preferred)
        2. data/**/*.txt and *.pdf — plain text files (fallback / non-PDF sources)

        Processed JSON files take priority over raw PDF files of the same stem.
        """
        self._chunks = []
        chunk_id = 0

        # Track which sources were covered by processed JSON
        processed_stems: set[str] = set()

        # ── Source 1: Structured chunks from pdf_pipeline output ─────────
        if PROCESSED_DATA_DIR.exists():
            for json_path in sorted(PROCESSED_DATA_DIR.glob("*.json")):
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
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
                    logger.info("[Retrieval] Loaded structured JSON: %s (%d chunks)",
                                json_path.name, len(raw_chunks))
                except Exception as e:
                    logger.warning("[Retrieval] Failed to load %s: %s", json_path.name, e)

        # ── Source 2: Plain text / unprocessed files ──────────────────────
        for folder, (bank, doc_type) in _SOURCE_MAP.items():
            folder_path = DATA_DIR / folder
            if not folder_path.exists():
                continue

            for file_path in sorted(folder_path.glob("*")):
                if file_path.suffix not in (".txt", ".pdf"):
                    continue

                # Skip if a processed JSON already covers this PDF
                if file_path.stem in processed_stems:
                    logger.debug("[Retrieval] Skipping %s (covered by processed JSON)", file_path.name)
                    continue

                text = self._read_file(file_path)
                if not text.strip():
                    continue

                for chunk_text in self._chunk_text(text):
                    self._chunks.append(DocChunk(
                        text=chunk_text,
                        source=file_path.name,
                        bank=bank,
                        doc_type=doc_type,
                        chunk_id=chunk_id,
                    ))
                    chunk_id += 1

        logger.info("[Retrieval] Total chunks loaded: %d", len(self._chunks))
        return len(self._chunks)

    @staticmethod
    def _read_file(path: Path) -> str:
        if path.suffix == ".txt":
            return path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix == ".pdf":
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
                    return ""  # PDF libs not available
        return ""

    # ------------------------------------------------------------------ build

    def build_index(self, save: bool = True) -> None:
        """Embed all chunks and build FAISS index."""
        if not self._chunks:
            self.load_documents()

        faiss = self._get_faiss()
        embedder = self._get_embedder()

        texts = [c.text for c in self._chunks]
        print(f"[Retrieval] Embedding {len(texts)} chunks…")
        vectors = embedder.encode(texts, batch_size=64, show_progress_bar=True)
        vectors = np.array(vectors, dtype="float32")

        # L2-normalize for cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-10)

        index = faiss.IndexFlatIP(EMBEDDING_DIM)   # inner product = cosine after normalization
        index.add(vectors)
        self._index = index
        self._vectors = vectors
        self._is_built = True

        if save:
            self._save(vectors)

    def _save(self, vectors: np.ndarray) -> None:
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss = self._get_faiss()
        faiss.write_index(self._index, str(FAISS_INDEX_DIR / "index.faiss"))
        with open(FAISS_INDEX_DIR / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        np.save(FAISS_INDEX_DIR / "vectors.npy", vectors)
        print(f"[Retrieval] Index saved ({len(self._chunks)} chunks)")

    def load_index(self) -> bool:
        """Load pre-built FAISS index from disk."""
        idx_path = FAISS_INDEX_DIR / "index.faiss"
        chunk_path = FAISS_INDEX_DIR / "chunks.pkl"
        if not (idx_path.exists() and chunk_path.exists()):
            return False

        faiss = self._get_faiss()
        self._index = faiss.read_index(str(idx_path))
        with open(chunk_path, "rb") as f:
            self._chunks = pickle.load(f)
        try:
            self._vectors = np.load(FAISS_INDEX_DIR / "vectors.npy")
        except FileNotFoundError:
            self._vectors = None
        self._is_built = True
        print(f"[Retrieval] Loaded index with {len(self._chunks)} chunks")
        return True

    def ensure_ready(self) -> None:
        """Auto-build index if not already loaded."""
        if not self._is_built:
            if not self.load_index():
                self.load_documents()
                self.build_index()

    # ------------------------------------------------------------------ query

    def _faiss_search(
        self,
        query: str,
        fetch_k: int,
        banks: Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[DocChunk]:
        """
        Pure FAISS bi-encoder search.
        Returns up to fetch_k chunks after metadata filtering.
        Internal helper — callers should use retrieve() instead.
        """
        embedder = self._get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype="float32")

        # Cast a wider net in the index, then apply metadata filters
        search_k = min(fetch_k * 10, len(self._chunks))
        scores, indices = self._index.search(q_vec, search_k)

        results: list[tuple[float, DocChunk]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if banks and chunk.bank not in banks:
                continue
            if doc_types and chunk.doc_type not in doc_types:
                continue
            results.append((float(score), chunk))
            if len(results) >= fetch_k:
                break

        # Attach the similarity score to each chunk so pipeline can use it
        scored: list[DocChunk] = []
        for score, chunk in results:
            chunk.similarity_score = float(score)
            scored.append(chunk)
        return scored

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        banks: Optional[list[str]] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[DocChunk]:
        """
        Main retrieval entry point.

        When USE_RERANKER is True (default):
            1. FAISS fetches RERANKER_FETCH_K (=10) candidates  — broad recall
            2. Nemotron reranker rescores all candidates         — precision boost
            3. Returns RERANKER_TOP_N (=3) best docs

        When USE_RERANKER is False (or reranker fails):
            Falls back to plain FAISS top-k ordering.

        The reranker import is deferred to avoid circular imports and
        to allow the module to load even when the reranker is disabled.
        """
        self.ensure_ready()

        if USE_RERANKER:
            # ── Step 1: Broad FAISS retrieval (more candidates for reranker) ──
            fetch_k = max(RERANKER_FETCH_K, top_k)
            candidates = self._faiss_search(query, fetch_k, banks, doc_types)

            logger.info("[Retrieval] Retrieved docs (FAISS): %d", len(candidates))

            # ── Step 2: Reranker re-scores and trims to top_n ─────────────────
            from services.reranker import rerank_documents   # deferred import
            result = rerank_documents(query, candidates, top_n=RERANKER_TOP_N)

            logger.info(
                "[Retrieval] Reranked docs: %d  | reranked=%s | reason=%s",
                len(result.documents), result.reranked, result.reason,
            )
            return result.documents

        else:
            # ── Plain FAISS path (reranker disabled) ──────────────────────────
            candidates = self._faiss_search(query, top_k, banks, doc_types)
            logger.info("[Retrieval] Retrieved docs (FAISS only): %d", len(candidates))
            return candidates

    def retrieve_by_bank(self, query: str, bank: str, top_k: int = TOP_K) -> list[DocChunk]:
        return self.retrieve(query, top_k=top_k, banks=[bank])

    def retrieve_regulatory(self, query: str, top_k: int = TOP_K) -> list[DocChunk]:
        return self.retrieve(query, top_k=top_k, doc_types=["regulatory"])

    def retrieve_comparison(self, query: str, top_k: int = TOP_K) -> list[DocChunk]:
        return self.retrieve(query, top_k=top_k, doc_types=["comparison"])
