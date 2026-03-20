"""
tests/test_reranker.py
=======================
Tests for the Nemotron reranker service.

All tests run WITHOUT real API keys by mocking the HTTP call.
Run: python tests/test_reranker.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Minimal DocChunk stub (mirrors services/retrieval.DocChunk)
# ---------------------------------------------------------------------------

@dataclass
class FakeChunk:
    text: str
    source: str = "test.txt"
    bank: str = "Axis"
    doc_type: str = "eligibility"
    chunk_id: int = 0


# ---------------------------------------------------------------------------
# Helpers to build mock responses
# ---------------------------------------------------------------------------

def _mock_nvidia_response(rankings: list[dict]) -> MagicMock:
    """Build a fake requests.Response with the NVIDIA ranking payload."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"rankings": rankings}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _make_chunks(n: int) -> list[FakeChunk]:
    return [
        FakeChunk(
            text=f"Document {i}: loan eligibility text about banks and income requirements.",
            chunk_id=i,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRerankerGuards(unittest.TestCase):
    """Guard conditions that skip reranking — no HTTP calls made."""

    def test_disabled_via_flag(self):
        """USE_RERANKER=false → returns first top_n docs unchanged."""
        with patch("config.USE_RERANKER", False):
            from services import reranker as rmod
            import importlib; importlib.reload(rmod)
            chunks = _make_chunks(10)
            result = rmod.rerank_documents("test query", chunks, top_n=3)
            self.assertFalse(result.reranked)
            self.assertEqual(len(result.documents), 3)
            self.assertEqual(result.documents[0].chunk_id, 0)   # unchanged order

    def test_fewer_docs_than_top_n(self):
        """Only 2 docs, top_n=3 → skip reranking, return all 2."""
        with patch("config.USE_RERANKER", True):
            from services.reranker import rerank_documents
            chunks = _make_chunks(2)
            result = rerank_documents("detailed eligibility query for Axis loan", chunks, top_n=3)
            self.assertFalse(result.reranked)
            self.assertEqual(len(result.documents), 2)

    def test_exact_top_n_docs(self):
        """Exactly top_n docs → skip reranking."""
        with patch("config.USE_RERANKER", True):
            from services.reranker import rerank_documents
            chunks = _make_chunks(3)
            result = rerank_documents("detailed eligibility query for Axis bank", chunks, top_n=3)
            self.assertFalse(result.reranked)

    def test_simple_query_skipped(self):
        """Short query (< RERANKER_MIN_QUERY_WORDS) → skip reranking."""
        import services.reranker as rmod
        import importlib
        with patch("config.USE_RERANKER", True), patch("config.RERANKER_MIN_QUERY_WORDS", 4):
            importlib.reload(rmod)
            chunks = _make_chunks(10)
            result = rmod.rerank_documents("Axis loan", chunks, top_n=3)   # only 2 words
            self.assertFalse(result.reranked)
            self.assertIn("too short", result.reason)


class TestRerankerSuccess(unittest.TestCase):
    """Happy-path: reranker runs and reorders documents correctly."""

    def test_reorders_documents(self):
        """
        Simulate NVIDIA API returning:
            index=5 (score highest), index=2, index=8
        Verify those 3 chunks appear first in that order.
        """
        chunks = _make_chunks(10)

        mock_rankings = [
            {"index": 5, "logit": 4.20},   # best
            {"index": 2, "logit": 3.10},   # second
            {"index": 8, "logit": 1.50},   # third
        ]

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", "test-key"), \
             patch("requests.post", return_value=_mock_nvidia_response(mock_rankings)):

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "What is the eligibility criteria for Axis Bank personal loan?",
                chunks,
                top_n=3,
            )

        self.assertTrue(result.reranked)
        self.assertEqual(len(result.documents), 3)
        self.assertEqual(result.documents[0].chunk_id, 5)
        self.assertEqual(result.documents[1].chunk_id, 2)
        self.assertEqual(result.documents[2].chunk_id, 8)

    def test_scores_normalised_0_to_1(self):
        """Normalised scores must all be in [0, 1]."""
        chunks = _make_chunks(10)
        mock_rankings = [
            {"index": 3, "logit": 10.0},
            {"index": 1, "logit":  5.0},
            {"index": 7, "logit": -2.0},
        ]

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", "test-key"), \
             patch("requests.post", return_value=_mock_nvidia_response(mock_rankings)):

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "What documents are needed for HDFC personal loan application?",
                chunks,
                top_n=3,
            )

        for score in result.scores:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_metadata_preserved(self):
        """All DocChunk fields (bank, doc_type, etc.) must survive reranking."""
        chunks = [
            FakeChunk("Axis eligibility text", bank="Axis",   doc_type="eligibility", chunk_id=0),
            FakeChunk("RBI regulatory text",   bank="RBI",    doc_type="regulatory",  chunk_id=1),
            FakeChunk("ICICI comparison text", bank="ICICI",  doc_type="comparison",  chunk_id=2),
            FakeChunk("SBI loan details",      bank="SBI",    doc_type="pdf",         chunk_id=3),
            FakeChunk("HDFC loan details",     bank="HDFC",   doc_type="pdf",         chunk_id=4),
        ]
        mock_rankings = [
            {"index": 1, "logit": 5.0},
            {"index": 3, "logit": 4.0},
            {"index": 0, "logit": 3.0},
        ]

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", "test-key"), \
             patch("requests.post", return_value=_mock_nvidia_response(mock_rankings)):

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "What are the RBI guidelines for personal loans in India?",
                chunks,
                top_n=3,
            )

        self.assertEqual(result.documents[0].bank, "RBI")
        self.assertEqual(result.documents[0].doc_type, "regulatory")
        self.assertEqual(result.documents[1].bank, "SBI")
        self.assertEqual(result.documents[2].bank, "Axis")


class TestRerankerFallback(unittest.TestCase):
    """Error conditions — all must fall back gracefully, never raise."""

    def test_timeout_fallback(self):
        """Network timeout → fallback to FAISS order, no exception."""
        import requests as req_module
        chunks = _make_chunks(10)

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", "test-key"), \
             patch("requests.post", side_effect=req_module.exceptions.Timeout()):

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "What is the minimum salary for ICICI personal loan eligibility?",
                chunks,
                top_n=3,
            )

        self.assertFalse(result.reranked)
        self.assertEqual(len(result.documents), 3)
        self.assertIn("timed out", result.reason.lower())

    def test_http_401_fallback(self):
        """Invalid API key (401) → fallback gracefully."""
        import requests as req_module
        chunks = _make_chunks(10)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_module.exceptions.HTTPError("401 Unauthorized")

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", "bad-key"), \
             patch("requests.post", return_value=mock_resp):

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "Compare personal loans from HDFC and SBI for salaried employees",
                chunks,
                top_n=3,
            )

        self.assertFalse(result.reranked)
        self.assertEqual(len(result.documents), 3)

    def test_empty_rankings_fallback(self):
        """Empty rankings list from API → fallback."""
        chunks = _make_chunks(10)

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", "test-key"), \
             patch("requests.post", return_value=_mock_nvidia_response([])):

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "Detailed eligibility criteria for Axis Bank personal loan 2024",
                chunks,
                top_n=3,
            )

        self.assertFalse(result.reranked)

    def test_missing_api_key_fallback(self):
        """No NVIDIA_API_KEY set → should raise ValueError caught as fallback."""
        chunks = _make_chunks(10)

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nvidia_api"), \
             patch("config.NVIDIA_API_KEY", ""):   # empty key

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            # Should not raise — must return a fallback RerankResult
            result = rmod.rerank_documents(
                "What is the CIBIL score required for HDFC bank personal loan?",
                chunks,
                top_n=3,
            )

        self.assertFalse(result.reranked)
        self.assertEqual(len(result.documents), 3)


class TestNIMBackend(unittest.TestCase):
    """NIM local endpoint (same schema, different URL)."""

    def test_nim_endpoint_called(self):
        """With RERANKER_BACKEND=nim, should POST to NIM URL, not NVIDIA API."""
        chunks = _make_chunks(10)
        mock_rankings = [{"index": 0, "logit": 3.0}, {"index": 1, "logit": 2.0}, {"index": 2, "logit": 1.0}]

        with patch("config.USE_RERANKER", True), \
             patch("config.RERANKER_BACKEND", "nim"), \
             patch("config.NIM_BASE_URL", "http://localhost:8001"), \
             patch("requests.post", return_value=_mock_nvidia_response(mock_rankings)) as mock_post:

            from services import reranker as rmod
            import importlib; importlib.reload(rmod)

            result = rmod.rerank_documents(
                "Eligibility criteria for personal loan with low CIBIL score",
                chunks,
                top_n=3,
            )

        # Verify the URL used was the NIM endpoint
        called_url = mock_post.call_args[0][0]
        self.assertIn("localhost:8001", called_url)
        self.assertIn("/v1/ranking", called_url)
        self.assertTrue(result.reranked)


# ---------------------------------------------------------------------------
# Manual demo (no mocking — prints live output for visual inspection)
# ---------------------------------------------------------------------------

def _demo_reranker_flow():
    """
    Demonstrate the before/after reranking effect with synthetic docs.
    No API call — uses mock to simulate NVIDIA response.
    """
    print("\n" + "="*65)
    print("  RERANKER DEMO")
    print("="*65)

    query = "What is the eligibility criteria for Axis Bank personal loan?"

    # Simulate 10 FAISS-retrieved docs (mixed relevance)
    docs = [
        FakeChunk("SBI Xpress Credit requires age 21-58 and minimum ₹15,000 income.", bank="SBI"),
        FakeChunk("RBI guidelines mandate KYC compliance for all personal loans.", bank="RBI"),
        FakeChunk("Axis Bank personal loan requires CIBIL score 720 or above.", bank="Axis"),
        FakeChunk("HDFC offers SmartEMI on purchases with existing customers.", bank="HDFC"),
        FakeChunk("Axis Bank personal loan minimum monthly income requirement is ₹15,000.", bank="Axis"),
        FakeChunk("Processing fee at ICICI Bank is 0.5% to 2.5% plus taxes.", bank="ICICI"),
        FakeChunk("Axis Bank requires minimum 12 months work experience for eligibility.", bank="Axis"),
        FakeChunk("BankBazaar comparison: SBI has the lowest income threshold at ₹15,000.", bank="BankBazaar"),
        FakeChunk("Axis Bank disburses loans within 48 hours for eligible applicants.", bank="Axis"),
        FakeChunk("ICICI Bank requires minimum 24 months of work experience.", bank="ICICI"),
    ]

    print(f"\nQuery: {query!r}\n")
    print("Before reranking (FAISS order):")
    for i, d in enumerate(docs):
        print(f"  [{i}] [{d.bank}] {d.text[:60]}…")

    # Simulate reranker preferring Axis-specific eligibility chunks
    mock_rankings = [
        {"index": 2, "logit": 5.80},   # Axis CIBIL requirement
        {"index": 4, "logit": 5.65},   # Axis income requirement
        {"index": 6, "logit": 5.40},   # Axis work experience
        {"index": 8, "logit": 2.10},   # Axis disbursal (less relevant)
        {"index": 0, "logit": 1.50},
    ]

    with patch("config.USE_RERANKER", True), \
         patch("config.RERANKER_BACKEND", "nvidia_api"), \
         patch("config.NVIDIA_API_KEY", "demo-key"), \
         patch("requests.post", return_value=_mock_nvidia_response(mock_rankings)):

        from services import reranker as rmod
        import importlib; importlib.reload(rmod)
        result = rmod.rerank_documents(query, docs, top_n=3)

    print("\nAfter reranking (Nemotron top-3):")
    for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
        print(f"  [{i+1}] score={score:.2f} [{doc.bank}] {doc.text[:60]}…")

    print(f"\nStatus : reranked={result.reranked}")
    print(f"Reason : {result.reason}")
    print("="*65 + "\n")


if __name__ == "__main__":
    _demo_reranker_flow()
    print("\nRunning unit tests...\n")
    unittest.main(argv=[""], verbosity=2, exit=False)
