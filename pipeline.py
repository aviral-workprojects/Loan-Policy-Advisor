"""
Loan Policy Advisor — Pipeline v3
===================================
Fully deterministic, per-bank, logically consistent pipeline.

Flow:
  Query → Cache → Parse → Route → Retrieve
        → Rule Engine (per-bank, strict)
        → Reasoning Layer (structured context)
        → Confidence (calibrated)
        → Consistency Validation
        → LLM Explanation (grounded, no hallucination)
        → Cache → Response
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Any

from config import MIN_DOCS_THRESHOLD
from services.llm import LLMService, ParsedQuery
from services.retrieval import RetrievalService
from services.rule_engine import RuleEngine, BankResult
from services.router import Router, RetrievalPlan
from services.fusion import ContextFusion
from services.search import FallbackSearch
from services.cache import SemanticCache
from services.reasoning import (
    build_reasoning_context,
    compute_confidence,
    validate_consistency,
    ReasoningContext,
)

# ---------------------------------------------------------------------------
# Response
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
    reasoning_context:    dict         # full structured reasoning for debugging/UI
    validation_issues:    list[dict]   # consistency check results
    latency_ms:           float = 0.0
    from_cache:           bool  = False

# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _determine_decision(
    bank_results: list[BankResult],
    reasoning:    ReasoningContext,
    profile:      dict,
) -> str:
    if not profile:
        return "Insufficient Data"
    if not bank_results:
        return "Insufficient Data"

    n_eligible = len(reasoning.eligible_banks)
    n_total    = len(bank_results)

    if reasoning.missing_data_banks and n_eligible == 0:
        return "Insufficient Data"
    elif n_eligible == n_total:
        return "Eligible"
    elif n_eligible > 0:
        return "Partially Eligible"
    else:
        return "Not Eligible"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

BANK_RATES = {"axis": "10.49%", "hdfc": "10.50%", "icici": "10.65%", "sbi": "11.05%"}

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

        # ── 0. Cache ──────────────────────────────────────────────────────────
        cached = self.cache.get(user_query)
        if cached:
            cached["from_cache"] = True
            return AdvisorResponse(**cached)

        # ── 1. Parse ──────────────────────────────────────────────────────────
        parsed: ParsedQuery = self.llm.parse_query(user_query)
        profile = parsed.profile

        # DTI guard: LLM sometimes returns 40 instead of 0.40
        if "dti_ratio" in profile and profile["dti_ratio"] is not None:
            if profile["dti_ratio"] > 1.5:
                profile["dti_ratio"] = profile["dti_ratio"] / 100.0

        print(f"[Pipeline] intent={parsed.intent} banks={parsed.banks}")
        print(f"[Pipeline] profile={profile}")

        # ── 2. Route ──────────────────────────────────────────────────────────
        plan = self.router.route(parsed)

        # ── 3. Retrieve ───────────────────────────────────────────────────────
        chunks = self.retrieval.retrieve(
            query=user_query,
            banks=plan.banks if not plan.use_hybrid else None,
            doc_types=plan.doc_types if not plan.use_hybrid else None,
            top_k=8,
        )
        web_context = ""
        if len(chunks) < MIN_DOCS_THRESHOLD:
            results = self.search.search(user_query)
            web_context = self.search.format_results(results)

        # ── 4. Rule Engine — strict per-bank evaluation ───────────────────────
        bank_results: list[BankResult] = []

        if parsed.intent in ("eligibility", "comparison") and profile:
            banks_to_check = plan.banks or self.rule_engine.available_banks()
            for bank in banks_to_check:
                br = self.rule_engine.evaluate(bank, profile)
                bank_results.append(br)
                status = "✅" if br.eligible else "❌"
                print(
                    f"[RuleEngine] {br.bank}: {status} | "
                    f"pass={len(br.passed)} fail={len(br.failed)} "
                    f"miss={len(br.missing)} score={br.rule_score:.2f}"
                )

        # ── 5. Reasoning layer ────────────────────────────────────────────────
        reasoning = build_reasoning_context(bank_results, profile)

        # ── 6. Decision (deterministic) ───────────────────────────────────────
        decision = _determine_decision(bank_results, reasoning, profile)
        print(f"[Pipeline] Decision: {decision} | eligible={reasoning.eligible_banks}")

        # ── 7. Calibrated confidence ──────────────────────────────────────────
        confidence = compute_confidence(bank_results, reasoning, len(chunks))
        print(f"[Pipeline] Confidence: {confidence}")

        # ── 8. Consistency validation ─────────────────────────────────────────
        issues = validate_consistency(bank_results, decision, reasoning)
        for issue in issues:
            print(f"[Validation] {issue.severity.upper()}: {issue.message}")

        # ── 9. Build structured data for frontend ─────────────────────────────
        rule_results_dicts = [br.to_dict() for br in bank_results]

        banks_compared = sorted(
            [
                {
                    "name":     br.bank,
                    "eligible": br.eligible,
                    "rate":     BANK_RATES.get(br.bank.lower(), "N/A"),
                    "score":    round(br.rule_score * 100),
                    "summary":  br.summary,
                }
                for br in bank_results
            ],
            key=lambda x: x["score"],
            reverse=True,
        )

        # ── 10. LLM explanation — strictly grounded ───────────────────────────
        fused_context = self.fusion.fuse(chunks)
        if web_context:
            fused_context += "\n\n" + web_context

        if not bank_results and not chunks:
            explanation = {
                "summary": "Insufficient profile data. Please provide age, monthly income, and CIBIL score.",
                "detailed_explanation": "Cannot evaluate eligibility without core profile data.",
                "recommendations": [
                    "Provide your age, monthly income (e.g. ₹35,000/month), and CIBIL score.",
                    "Mention which bank you are interested in.",
                ],
                "sources_cited": [],
            }
        else:
            explanation = self.llm.explain(
                query=user_query,
                rule_results=rule_results_dicts,
                retrieved_context=fused_context,
                parsed_query=parsed.raw,
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
            latency_ms=round(latency, 1),
            from_cache=False,
        )

        self.cache.set(user_query, asdict(response))
        return response
