"""
Loan Policy Advisor — Pipeline v4
====================================
Replaces the NVIDIA CV pipeline entirely with a hybrid extraction stack.

Flow:
  Query → Cache
        → QueryUnderstanding (intent + profile extraction)
        → Route (MoE: skip RAG for simple queries)
        → Retrieve (FAISS + BM25 + RRF + Reranker)
        → EligibilityEngine (deterministic per-bank rules)
        → ReasoningLayer (structured context)
        → Confidence
        → Validation
        → LLM Explanation (grounds explanation in rule results only)
        → Cache → Response

What changed from v3:
  - NVIDIA CV models (page-elements, OCR, parse) removed entirely
  - pdf_pipeline package replaces pdf_pipeline.py monolith
  - QueryUnderstanding replaces bare LLM parse for entity extraction
  - Table extraction (Camelot/Tabula) added as first-class concern
  - Extraction is now: pdfplumber → PyMuPDF → PaddleOCR → Tesseract
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any

from config import MIN_DOCS_THRESHOLD, MOE_SIMPLE_QUERY_WORDS

from services.llm      import LLMService
from services.cache    import SemanticCache
from services.rule_engine import RuleEngine, BankResult
from services.fusion   import ContextFusion
from services.search   import FallbackSearch
from services.reasoning import (
    build_reasoning_context,
    compute_confidence,
    compute_final_score,
    validate_consistency,
    ReasoningContext,
)

# New modules
from pdf_pipeline.query_understanding import understand_query, QuerySignals
from pdf_pipeline.retriever import RetrievalService, DocChunk

import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdvisorResponse:
    decision:             str
    summary:              str
    detailed_explanation: str
    recommendations:      list[str]
    banks_compared:       list[dict]
    rule_results:         list[dict]
    confidence:           float
    sources_cited:        list[str]
    reasoning_context:    dict
    validation_issues:    list[dict]
    best_bank:            str | None = None
    best_bank_reason:     str | None = None
    almost_eligible_bank: str | None = None
    critical_failures:    list[dict] | None = None
    latency_ms:           float = 0.0
    from_cache:           bool  = False


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

BANK_RATES = {
    "axis":  "9.98% – 22%",
    "hdfc":  "9.98% – 24%",
    "icici": "9.99% – 16.5%",
    "sbi":   "10.00% – 15.1%",
}


def _determine_decision(
    bank_results: list[BankResult],
    reasoning:    ReasoningContext,
    profile:      dict,
) -> str:
    """
    Map rule engine output to a user-facing decision label.

    States:
      "Eligible"          — all evaluated banks passed
      "Partially Eligible" — at least one bank passed
      "Not Eligible"      — all banks failed (no missing critical fields)
      "Partial Profile"   — profile has SOME data but missing critical fields
                            (income/age/CIBIL); can show partial rule evaluation
      "Insufficient Data" — truly empty profile, no evaluation possible
    """
    if not profile:
        return "Insufficient Data"
    if not bank_results:
        return "Insufficient Data"

    n_eligible = len(reasoning.eligible_banks)
    n_total    = len(bank_results)

    # Count banks blocked by missing critical fields vs actually failed
    n_missing_critical = sum(
        1 for br in bank_results
        if any(e.status == "missing" and e.field in {"age", "monthly_income", "credit_score"}
               for e in br.evaluations)
    )
    n_hard_fail = sum(
        1 for br in bank_results
        if not br.eligible and any(e.status == "fail" for e in br.evaluations)
    )

    if n_eligible == n_total:
        return "Eligible"
    elif n_eligible > 0:
        return "Partially Eligible"
    elif n_hard_fail > 0 and n_missing_critical == 0:
        # All banks failed due to actual rule failures (not missing data)
        return "Not Eligible"
    elif n_missing_critical > 0 and n_hard_fail == 0:
        # Profile exists but missing critical fields — partial evaluation only
        return "Partial Profile"
    else:
        # Mix of missing fields and failures
        return "Not Eligible"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LoanAdvisorPipeline:

    def __init__(self):
        self.llm         = LLMService()
        self.retrieval   = RetrievalService(
            index_dir=_index_dir(),
            embed_dim=_embed_dim(),
        )
        self.rule_engine = RuleEngine()
        self.fusion      = ContextFusion()
        self.search      = FallbackSearch()
        self.cache       = SemanticCache()
        self.retrieval.ensure_ready()

    def query(self, user_query: str) -> AdvisorResponse:
        t0 = time.perf_counter()

        # ── 0. Cache ──────────────────────────────────────────────────────────
        cached = self.cache.get(user_query)
        if cached:
            cached["from_cache"] = True
            return AdvisorResponse(**cached)

        # ── 1. Query Understanding ────────────────────────────────────────────
        # Rule-based entity extraction + intent — fast, no API call needed.
        signals: QuerySignals = understand_query(user_query)
        profile  = signals.profile

        # Fall back to LLM parsing for complex queries where rule-based
        # extraction produced an empty profile (rare ambiguous cases)
        if not profile and len(user_query.split()) > 8:
            logger.info("[Pipeline] Rule-based extraction empty — falling back to LLM parse")
            parsed = self.llm.parse_query(user_query)
            profile = parsed.profile
            signals.banks = signals.banks or parsed.banks
            signals.intent = signals.intent if signals.intent != "general" else parsed.intent

        # DTI guard: sometimes returned as 40 instead of 0.40
        if profile.get("dti_ratio") and profile["dti_ratio"] > 1.5:
            profile["dti_ratio"] /= 100.0

        logger.info("[Pipeline] intent=%s  banks=%s  profile=%s",
                    signals.intent, signals.banks, profile)

        # ── 2. Route — MoE shortcut for simple eligibility queries ──────────
        word_count = len(user_query.split())
        skip_rag   = (
            word_count <= MOE_SIMPLE_QUERY_WORDS
            and signals.intent == "eligibility"
            and bool(profile)
        )

        # ── 3. Retrieve ───────────────────────────────────────────────────────
        chunks: list[DocChunk] = []
        web_context = ""

        if not skip_rag:
            # Use the reformulated query for better retrieval recall
            retrieval_query = signals.reformulated_query or user_query
            chunks = self.retrieval.retrieve(
                query=retrieval_query,
                top_k=8,
                banks=signals.banks if signals.banks else None,
                fetch_k=20,
            )
            if len(chunks) < MIN_DOCS_THRESHOLD:
                results = self.search.search(user_query)
                web_context = self.search.format_results(results)
                logger.info("[Pipeline] Fallback web search: %d results", len(results))
        else:
            logger.info("[Pipeline] MoE: simple query (%dw) — skipping RAG", word_count)

        retrieval_scores = [c.similarity_score for c in chunks]

        # ── 4. Rule Engine ────────────────────────────────────────────────────
        bank_results: list[BankResult] = []

        if signals.intent in ("eligibility", "comparison") and profile:
            banks_to_check = signals.banks or self.rule_engine.available_banks()
            for bank in banks_to_check:
                br = self.rule_engine.evaluate(bank, profile)
                bank_results.append(br)
                status = "✅" if br.eligible else "❌"
                logger.info("[RuleEngine] %s: %s  pass=%d fail=%d miss=%d score=%.2f",
                            br.bank, status, len(br.passed), len(br.failed),
                            len(br.missing), br.rule_score)

        # ── 5. Reasoning ──────────────────────────────────────────────────────
        reasoning = build_reasoning_context(bank_results, profile)

        # ── 6. Decision ───────────────────────────────────────────────────────
        decision = _determine_decision(bank_results, reasoning, profile)
        logger.info("[Pipeline] Decision: %s  eligible=%s", decision, reasoning.eligible_banks)

        # ── 7. Confidence ─────────────────────────────────────────────────────
        confidence, conf_breakdown = compute_confidence(
            bank_results, reasoning, len(chunks), retrieval_scores
        )

        # Confidence floor: when the rule engine ran on real profile data,
        # the decision is deterministic — confidence should reflect that.
        if bank_results and profile:
            n_hard_data = sum(1 for k in ("monthly_income", "credit_score", "age") if k in profile)
            if n_hard_data >= 2:
                confidence = max(confidence, 0.65)
            if n_hard_data == 3:
                confidence = max(confidence, 0.80)

        reasoning.confidence_breakdown = conf_breakdown

        # ── 8. Validation ─────────────────────────────────────────────────────
        issues = validate_consistency(bank_results, decision, reasoning)

        # ── 9. Structured data for response ───────────────────────────────────
        rule_results_dicts = [br.to_dict() for br in bank_results]
        banks_compared = sorted(
            [
                {
                    "name":       br.bank,
                    "eligible":   br.eligible,
                    "rate":       BANK_RATES.get(br.bank.lower(), "N/A"),
                    "score":      int(compute_final_score(br) * 100),
                    "rule_score": round(br.rule_score * 100),
                    "summary":    br.summary,
                }
                for br in bank_results
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        # ── 10. LLM Explanation ───────────────────────────────────────────────
        fused_context = self.fusion.fuse(chunks)
        if web_context:
            fused_context += "\n\n" + web_context

        if not bank_results and not chunks:
            # True no-data case: no profile AND no retrieved context
            # Still attempt to answer factual queries using fallback web search
            if signals.intent in ("interest_rate", "fees", "document", "comparison"):
                # Factual question — answer from knowledge base / web, don't refuse
                explanation = self.llm.explain(
                    query=user_query,
                    rule_results=[],
                    retrieved_context=web_context or "No specific context retrieved.",
                    parsed_query={"intent": signals.intent, "entities": signals.entities},
                    pre_decision=decision,
                    decision_context=reasoning.to_dict(),
                )
            else:
                explanation = {
                    "summary": "Please provide your profile details to check eligibility.",
                    "detailed_explanation": (
                        "To evaluate your loan eligibility I need: monthly income (e.g. ₹35,000/month), "
                        "CIBIL score (e.g. 720), and age. You can also upload a salary slip or bank statement."
                    ),
                    "recommendations": [
                        "Share your monthly income (e.g. 'I earn ₹40,000 per month').",
                        "Share your CIBIL score (e.g. 'my CIBIL is 730').",
                        "Share your age (e.g. 'I am 28 years old').",
                        "Or upload a salary slip / bank statement using the 'Analyze Document' panel.",
                    ],
                    "sources_cited": [],
                }
        else:
            explanation = self.llm.explain(
                query=user_query,
                rule_results=rule_results_dicts,
                retrieved_context=fused_context,
                parsed_query={"intent": signals.intent, "entities": signals.entities},
                pre_decision=decision,
                decision_context=reasoning.to_dict(),
            )

        # ── 11. Build response ────────────────────────────────────────────────
        latency = (time.perf_counter() - t0) * 1000

        response = AdvisorResponse(
            decision=decision,
            summary=explanation.get("summary", ""),
            detailed_explanation=explanation.get("detailed_explanation", ""),
            recommendations=explanation.get("recommendations", []),
            banks_compared=banks_compared,
            rule_results=rule_results_dicts,
            confidence=confidence,
            sources_cited=explanation.get("sources_cited", []),
            reasoning_context=reasoning.to_dict(),
            validation_issues=[
                {"severity": i.severity, "bank": i.bank, "message": i.message}
                for i in issues
            ],
            best_bank=reasoning.best_bank,
            best_bank_reason=reasoning.best_bank_reason,
            almost_eligible_bank=reasoning.closest_bank,
            critical_failures=reasoning.critical_failures or [],
            latency_ms=round(latency, 1),
            from_cache=False,
        )

        self.cache.set(user_query, asdict(response))
        return response


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _index_dir() -> "Path":
    from pathlib import Path
    try:
        from config import FAISS_INDEX_DIR
        return FAISS_INDEX_DIR
    except Exception:
        return Path("models")


def _embed_dim() -> int:
    try:
        from config import EMBEDDING_DIM
        return int(EMBEDDING_DIM)
    except Exception:
        return 384