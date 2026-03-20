"""
Loan Policy Advisor — Main Pipeline  (v2 — all 10 fixes applied)
=================================================================
Pipeline:
  Query → Cache → Parse → Route → Retrieve → Rule Engine
        → Correct Decision Logic → Context Fusion → LLM Explain → Cache
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Any

from config import MIN_DOCS_THRESHOLD
from services.llm import LLMService, ParsedQuery
from services.retrieval import RetrievalService
from services.rule_engine import RuleEngine, RuleResult
from services.router import Router, RetrievalPlan
from services.fusion import ContextFusion
from services.search import FallbackSearch
from services.cache import SemanticCache


# ---------------------------------------------------------------------------
# Response shape
# ---------------------------------------------------------------------------

@dataclass
class AdvisorResponse:
    decision: str           # Eligible | Partially Eligible | Not Eligible | Insufficient Data
    summary: str
    detailed_explanation: str
    recommendations: list[str]
    banks_compared: list[dict]   # FIX #10: structured dicts, not plain strings
    rule_results: list[dict]
    confidence: float
    sources_cited: list[str]
    latency_ms: float = 0.0
    from_cache: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_confidence(rule_results: list[dict]) -> float:
    """
    FIX #5: Real confidence = passed_rules / total_evaluated_rules.
    Much more meaningful than a random heuristic.
    """
    total = sum(len(r.get("passed", [])) + len(r.get("failed", [])) for r in rule_results)
    passed = sum(len(r.get("passed", [])) for r in rule_results)
    if total == 0:
        return 0.5
    return round(passed / total, 2)


def _determine_decision(rule_results: list[dict], profile: dict) -> tuple[str, list[dict]]:
    """
    FIX #1: Correct multi-bank decision logic.
    Returns (decision_string, structured_banks_compared_list).

    Logic:
      - No profile at all              → Insufficient Data
      - All banks eligible             → Eligible
      - Some banks eligible            → Partially Eligible
      - No banks eligible              → Not Eligible
    """
    if not profile:
        return "Insufficient Data", []

    eligible_banks = [r for r in rule_results if r.get("eligible")]
    total_banks    = len(rule_results)

    # FIX #10: return structured dicts for frontend comparison table
    BANK_RATES = {"axis": "10.49%", "hdfc": "10.50%", "icici": "10.65%", "sbi": "11.05%"}
    banks_compared = [
        {
            "name":     r["bank"],
            "eligible": r["eligible"],
            "rate":     BANK_RATES.get(r["bank"].lower(), "N/A"),
            "score":    round(
                len(r.get("passed", [])) /
                max(len(r.get("passed", [])) + len(r.get("failed", [])), 1) * 100
            ),
            "summary":  r.get("summary", ""),
        }
        for r in rule_results
    ]

    if total_banks == 0:
        return "Insufficient Data", []
    elif len(eligible_banks) == total_banks:
        return "Eligible", banks_compared
    elif len(eligible_banks) > 0:
        return "Partially Eligible", banks_compared
    else:
        return "Not Eligible", banks_compared


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LoanAdvisorPipeline:

    def __init__(self):
        self.llm         = LLMService()
        self.retrieval   = RetrievalService()
        self.rule_engine = RuleEngine()
        self.router      = Router()
        self.fusion      = ContextFusion()
        self.search      = FallbackSearch()
        self.cache       = SemanticCache()
        self.retrieval.ensure_ready()

    def query(self, user_query: str) -> AdvisorResponse:
        t0 = time.perf_counter()

        # ── 0. Cache check ────────────────────────────────────────────────────
        cached = self.cache.get(user_query)
        if cached:
            cached["from_cache"] = True
            return AdvisorResponse(**cached)

        # ── 1. Query Understanding ────────────────────────────────────────────
        parsed: ParsedQuery = self.llm.parse_query(user_query)
        print(f"[Pipeline] Parsed: {parsed}")
        print(f"[Pipeline] Profile extracted: {parsed.profile}")

        # ── 2. Route ──────────────────────────────────────────────────────────
        plan: RetrievalPlan = self.router.route(parsed)
        print(f"[Pipeline] Plan: banks={plan.banks}, types={plan.doc_types}")

        # ── 3. Retrieve ───────────────────────────────────────────────────────
        chunks = self.retrieval.retrieve(
            query=user_query,
            banks=plan.banks if not plan.use_hybrid else None,
            doc_types=plan.doc_types if not plan.use_hybrid else None,
            top_k=8,
        )

        # ── 4. Fallback search ────────────────────────────────────────────────
        web_context = ""
        if len(chunks) < MIN_DOCS_THRESHOLD:
            print("[Pipeline] Sparse retrieval — triggering fallback search")
            results = self.search.search(user_query)
            web_context = self.search.format_results(results)

        # ── 5. Rule Engine — deterministic, per-bank ──────────────────────────
        rule_results: list[dict] = []
        profile = parsed.profile

        if parsed.intent in ("eligibility", "comparison") and profile:
            banks_to_check = plan.banks or list(self.rule_engine.available_banks())
            for bank in banks_to_check:
                result: RuleResult = self.rule_engine.evaluate(bank, profile)
                rule_results.append({
                    "bank":     result.bank,
                    "eligible": result.eligible,
                    "summary":  result.summary,
                    "passed":   result.passed,
                    "failed":   result.failed,
                    "missing":  result.missing,
                })
                status = "✅" if result.eligible else "❌"
                print(f"[RuleEngine] {bank}: {status} | passed={len(result.passed)} failed={len(result.failed)} missing={len(result.missing)}")

        # ── 6. FIX #1 — Correct decision logic (not delegated to LLM) ─────────
        pre_decision, banks_compared = _determine_decision(rule_results, profile)
        print(f"[Pipeline] Pre-decision: {pre_decision}")

        # ── 7. FIX #5 — Real confidence score ────────────────────────────────
        confidence = _compute_confidence(rule_results) if rule_results else 0.5

        # ── 8. Context Fusion ─────────────────────────────────────────────────
        fused_context = self.fusion.fuse(chunks)
        if web_context:
            fused_context += "\n\n" + web_context

        # ── 9. LLM Explanation — context only, decision already made ─────────
        # FIX: pass pre_decision so LLM cannot override the rule engine verdict
        if not rule_results and not chunks:
            explanation = {
                "summary": "Please provide age, monthly income, and CIBIL score to evaluate eligibility.",
                "detailed_explanation": "Insufficient profile data to run eligibility checks.",
                "recommendations": [
                    "Provide your age, monthly income, and CIBIL score.",
                    "Specify which bank(s) you are interested in.",
                ],
                "sources_cited": [],
            }
        else:
            # Build rich decision context for LLM (FIX: architecture upgrade)
            eligible_banks   = [r["bank"] for r in rule_results if r["eligible"]]
            failed_banks     = [r["bank"] for r in rule_results if not r["eligible"]]
            key_failures     = {
                r["bank"]: r["failed"][:2]
                for r in rule_results if r["failed"]
            }

            explanation = self.llm.explain(
                query=user_query,
                rule_results=rule_results,
                retrieved_context=fused_context,
                parsed_query=parsed.raw,
                pre_decision=pre_decision,        # LLM must respect this
                decision_context={
                    "eligible_banks":   eligible_banks,
                    "failed_banks":     failed_banks,
                    "key_failures":     key_failures,
                    "confidence":       confidence,
                },
            )

        # ── 10. Build final response ───────────────────────────────────────────
        latency = (time.perf_counter() - t0) * 1000

        response = AdvisorResponse(
            decision=pre_decision,                                  # ALWAYS use rule engine decision
            summary=explanation.get("summary", ""),
            detailed_explanation=explanation.get("detailed_explanation", ""),
            recommendations=explanation.get("recommendations", []),
            banks_compared=banks_compared,                          # structured dicts
            rule_results=rule_results,
            confidence=confidence,                                  # real score
            sources_cited=explanation.get("sources_cited", []),
            latency_ms=round(latency, 1),
            from_cache=False,
        )

        self.cache.set(user_query, asdict(response))
        return response
