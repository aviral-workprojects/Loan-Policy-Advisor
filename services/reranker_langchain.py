"""
services/reranker_langchain.py  [OPTIONAL]
==========================================
LangChain-flavoured reranking using:
    NVIDIARerank                    — wraps Nemotron via langchain-nvidia-ai-endpoints
    ContextualCompressionRetriever  — chains FAISS retriever → reranker transparently

Install extra dep:
    pip install langchain-nvidia-ai-endpoints langchain-community

Usage (drop-in for pipeline.py):
    from services.reranker_langchain import build_langchain_retriever
    retriever = build_langchain_retriever(faiss_index, query_embedder)
    docs = retriever.invoke("Eligibility for Axis personal loan")
    # Returns LangChain Document objects instead of DocChunk objects.
    # Each doc.page_content = chunk text, doc.metadata = {bank, doc_type, source}

NOTE:
    This file is completely optional. The primary reranker (reranker.py) does NOT
    depend on LangChain and works standalone. Use this only if your broader stack
    already uses LangChain and you want the unified retriever abstraction.
"""

from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_langchain_retriever(
    faiss_index: Any,               # a LangChain FAISS vectorstore object
    top_k_fetch: int = 10,
    top_n_rerank: int = 3,
) -> Any:
    """
    Wrap a LangChain FAISS vectorstore with Nemotron reranker.

    Args:
        faiss_index   : LangChain FAISS object (not our custom RetrievalService)
        top_k_fetch   : candidates fetched from FAISS
        top_n_rerank  : docs returned after reranking

    Returns:
        ContextualCompressionRetriever — call .invoke(query) to get reranked docs

    Example:
        from langchain_community.vectorstores import FAISS as LangChainFAISS
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        lc_faiss = LangChainFAISS.load_local("models/", embeddings)

        retriever = build_langchain_retriever(lc_faiss, top_k_fetch=10, top_n_rerank=3)
        results = retriever.invoke("Axis Bank personal loan eligibility")
        for doc in results:
            print(doc.page_content[:100], doc.metadata)
    """
    try:
        from langchain.retrievers import ContextualCompressionRetriever
        from langchain_nvidia_ai_endpoints import NVIDIARerank
    except ImportError as e:
        raise ImportError(
            f"LangChain integration requires extra deps: {e}\n"
            "Run: pip install langchain-nvidia-ai-endpoints langchain-community"
        ) from e

    from config import NVIDIA_API_KEY, RERANKER_MODEL, RERANKER_TOP_N

    if not NVIDIA_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY is not set. "
            "Get a free key at https://build.nvidia.com"
        )

    # ── Base FAISS retriever (bi-encoder, broad recall) ───────────────────────
    base_retriever = faiss_index.as_retriever(
        search_kwargs={"k": top_k_fetch}
    )

    # ── Nemotron cross-encoder reranker ───────────────────────────────────────
    reranker = NVIDIARerank(
        model=RERANKER_MODEL,
        api_key=NVIDIA_API_KEY,
        top_n=top_n_rerank,
    )

    # ── Combine: FAISS → Reranker in one retriever object ────────────────────
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    logger.info(
        "[LangChain Reranker] Built ContextualCompressionRetriever "
        "(fetch_k=%d → rerank top_n=%d, model=%s)",
        top_k_fetch, top_n_rerank, RERANKER_MODEL,
    )

    return compression_retriever
