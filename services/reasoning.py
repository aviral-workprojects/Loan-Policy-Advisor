"""
Reasoning Layer
===============
Transforms per-bank BankResult objects into a structured reasoning context
that the LLM uses exclusively — no numbers invented, no logic inferred.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from services.rule_engine import BankResult

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ReasoningContext:
    eligible_banks:   list[str]
    rejected_banks:   list[str]
    missing_data_banks: list[str]
    best_bank:        str | None      # highest rule_score among eligible
    closest_bank:     str | None      # highest rule_score among rejected (almost passed)
    failure_summary:  dict[str, int]  # e.g. {"low_cibil": 2, "low_income": 1}
    top_failures:     list[dict]      # top 3 specific failure details
    data_completeness: float          # 0–1: how complete the profile was
    profile_fields_provided: list[str]
    profile_fields_missing:  list[str]
    critical_failures:   list[dict] | None = None  # failures on CIBIL/income/age only
    best_bank_reason:    str | None = None
    confidence_breakdown: dict | None = None

    def to_dict(self) -> dict:
        return {
            "eligible_banks":          self.eligible_banks,
            "rejected_banks":          self.rejected_banks,
            "missing_data_banks":      self.missing_data_banks,
            "best_bank":               self.best_bank,
            "closest_bank":            self.closest_bank,
            "failure_summary":         self.failure_summary,
            "top_failures":            self.top_failures,
            "data_completeness":       round(self.data_completeness, 2),
            "profile_fields_provided": self.profile_fields_provided,
            "profile_fields_missing":  self.profile_fields_missing,
            "critical_failures":        self.critical_failures,
            "best_bank_reason":         self.best_bank_reason,
            "confidence_breakdown":     self.confidence_breakdown,
        }

# ---------------------------------------------------------------------------
# Bank scoring — penalised rule score
# ---------------------------------------------------------------------------

def compute_final_score(br) -> float:
    """
    Penalised score for ranking banks.

    base            = rule_score (pass_count / evaluated_count)
    failure_penalty = 0.15 per failed rule
    missing_penalty = 0.05 per missing optional field
    cap             = 0.60 if not eligible (so rejected banks never outscore eligible ones)

    Returns a float 0.05–1.00.
    """
    base            = br.rule_score
    failure_penalty = len(br.failed)  * 0.15
    missing_penalty = len(br.missing) * 0.05
    score           = base - failure_penalty - missing_penalty

    if not br.eligible:
        score = min(score, 0.60)

    return round(max(score, 0.05), 2)


def explain_best_bank(best_bank: str | None, bank_results: list) -> str | None:
    """
    Plain-English sentence: why this bank is the best choice.
    Uses ONLY measured data — no hallucination.
    """
    if not best_bank:
        return None
    br = next((b for b in bank_results if b.bank == best_bank), None)
    if br is None:
        return None
    score_pct = round(compute_final_score(br) * 100)
    n_passed  = len(br.passed)
    n_failed  = len(br.failed)
    if n_failed == 0:
        return (
            f"{best_bank} is the recommended bank — it passed all {n_passed} eligibility "
            f"rules with a score of {score_pct}% and has the lowest entry requirements "
            f"among eligible options."
        )
    else:
        return (
            f"{best_bank} has the highest eligibility score ({score_pct}%) among evaluated "
            f"banks, passing {n_passed} rules with {n_failed} condition(s) to address."
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

# Priority order: lower = more impactful (shown first, weighted heavier)
_FAILURE_PRIORITY = {
    "credit_score":          1,
    "monthly_income":        2,
    "dti_ratio":             3,
    "work_experience_months":4,
    "employment_type":       5,
    "age":                   6,
}

# Maps rule field names to human categories for failure summary
_FAILURE_CATEGORIES = {
    "monthly_income":        "low_income",
    "credit_score":          "low_cibil",
    "age":                   "age_out_of_range",
    "dti_ratio":             "high_dti",
    "work_experience_months":"insufficient_experience",
    "employment_type":       "ineligible_employment_type",
}

_KNOWN_PROFILE_FIELDS = [
    "age", "monthly_income", "credit_score",
    "employment_type", "work_experience_months", "dti_ratio",
]

def build_reasoning_context(
    bank_results: list[BankResult],
    profile: dict[str, Any],
) -> ReasoningContext:
    """
    Build a fully structured reasoning context from per-bank rule results.
    No inference — only what the rule engine actually measured.
    """

    eligible_banks:     list[str] = []
    rejected_banks:     list[str] = []
    missing_data_banks: list[str] = []
    failure_summary:    dict[str, int] = {}
    all_failures:       list[dict] = []

    for r in bank_results:
        if r.eligible:
            eligible_banks.append(r.bank)
        elif any(e.status == "missing" and e.field in {"age", "monthly_income", "credit_score"}
                 for e in r.evaluations):
            missing_data_banks.append(r.bank)
        else:
            rejected_banks.append(r.bank)

        # Tally failure categories
        for e in r.evaluations:
            if e.status == "fail":
                cat = _FAILURE_CATEGORIES.get(e.field, e.field)
                failure_summary[cat] = failure_summary.get(cat, 0) + 1
                all_failures.append({
                    "bank":         r.bank,
                    "field":        e.field,
                    "rule":         e.message,
                    "actual":       e.actual_value,
                    "required":     e.expected,
                    "operator":     e.operator,
                    "reason":       e.reason,
                    "category":     _FAILURE_CATEGORIES.get(e.field, e.field),
                })

    # Best bank = highest rule_score among eligible
    eligible_results = [r for r in bank_results if r.eligible]
    best_bank = (
        max(eligible_results, key=lambda r: r.rule_score).bank
        if eligible_results else None
    )

    # Closest bank = highest rule_score among rejected (nearly qualified)
    rejected_results = [r for r in bank_results if not r.eligible and r.rule_score > 0]
    closest_bank = (
        max(rejected_results, key=lambda r: r.rule_score).bank
        if rejected_results else None
    )

    # Sort all failures by priority (most impactful first), then deduplicate by field
    _CRITICAL_FIELDS_SET = {"credit_score", "monthly_income", "age"}
    all_failures.sort(key=lambda f: _FAILURE_PRIORITY.get(f["field"], 99))
    seen_fields: set[str] = set()
    top_failures: list[dict] = []
    for f in all_failures:
        if f["field"] not in seen_fields:
            top_failures.append(f)
            seen_fields.add(f["field"])
        if len(top_failures) >= 5:   # surface up to 5 distinct failure types
            break

    # Critical failures = CIBIL / income / age only (lenders' hard gates)
    critical_failures = [
        f for f in all_failures
        if f["field"] in _CRITICAL_FIELDS_SET
    ]

    # Data completeness
    provided = [k for k in _KNOWN_PROFILE_FIELDS if profile.get(k) is not None]
    missing  = [k for k in _KNOWN_PROFILE_FIELDS if profile.get(k) is None]
    completeness = len(provided) / len(_KNOWN_PROFILE_FIELDS)

    # Best bank explanation
    best_bank_reason = explain_best_bank(best_bank, bank_results)

    return ReasoningContext(
        eligible_banks=eligible_banks,
        rejected_banks=rejected_banks,
        missing_data_banks=missing_data_banks,
        best_bank=best_bank,
        closest_bank=closest_bank,
        failure_summary=failure_summary,
        top_failures=top_failures,
        data_completeness=completeness,
        profile_fields_provided=provided,
        profile_fields_missing=missing,
        best_bank_reason=best_bank_reason,
        critical_failures=critical_failures,
    )


# ---------------------------------------------------------------------------
# Calibrated confidence
# ---------------------------------------------------------------------------

def compute_confidence(
    bank_results: list[BankResult],
    reasoning:    ReasoningContext,
    retrieval_chunk_count: int = 5,
    retrieval_scores: list[float] | None = None,
) -> tuple[float, dict]:
    """
    Two-path calibrated confidence:

    Rule Engine path  (data_completeness > 0):
        base 0.85 + avg_rule_score × 0.14  →  cap 0.99
        Reflects that deterministic rule evaluation is high-certainty.

    RAG-only path (no profile data provided):
        base 0.60 + avg_retrieval × 0.20   →  cap 0.84
        Reflects that pure retrieval answers are probabilistic.
        Drops to 0.10 if no chunks were retrieved at all.

    Both paths share the same critical-failure cap (≤ 0.60) and
    multi-bank-rejection penalty (−0.04 per rejected bank beyond 1).
    """
    if not bank_results:
        return 0.3, {"final": 0.3, "note": "no bank results"}

    avg_rule_score = sum(r.rule_score for r in bank_results) / len(bank_results)

    if reasoning.data_completeness > 0:
        # ── Rule engine path ─────────────────────────────────────────────────
        base = 0.85
        raw  = min(base + avg_rule_score * 0.14, 0.99)
        path = "rule_engine"
        retrieval_conf = 0.0   # not used on this path
    else:
        # ── RAG-only path ────────────────────────────────────────────────────
        base = 0.60
        if retrieval_scores and len(retrieval_scores) > 0:
            retrieval_conf = round(
                min(float(sum(retrieval_scores)) / len(retrieval_scores), 1.0), 3
            )
        else:
            retrieval_conf = min(retrieval_chunk_count / 8.0, 1.0)
        raw  = min(base + retrieval_conf * 0.20, 0.84)
        if retrieval_chunk_count == 0:
            raw = 0.10
        path = "rag_only"

    # ── Shared caps / penalties ───────────────────────────────────────────────
    critical_failed = any(
        e.status == "fail" and e.field in {"monthly_income", "credit_score", "age"}
        for r in bank_results
        for e in r.evaluations
    )
    if critical_failed:
        raw = min(raw, 0.60)

    n_failed = len(reasoning.rejected_banks)
    if n_failed > 1:
        raw -= (n_failed - 1) * 0.04

    final = round(max(0.05, min(raw, 0.99)), 2)
    breakdown = {
        "path":                     path,
        "base":                     base,
        "avg_rule_score":           round(avg_rule_score, 3),
        "data_completeness":        round(reasoning.data_completeness, 3),
        "retrieval_confidence":     round(retrieval_conf, 3) if path == "rag_only" else None,
        "critical_failure_cap":     critical_failed,
        "rule_engine_active":       path == "rule_engine",
        "final":                    final,
    }
    return final, breakdown


# ---------------------------------------------------------------------------
# Consistency validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    severity: str    # "error" | "warning"
    message:  str
    bank:     str = ""

def validate_consistency(
    bank_results: list[BankResult],
    decision:     str,
    reasoning:    ReasoningContext,
) -> list[ValidationIssue]:
    """
    Checks logical consistency of the rule engine output.
    Returns list of issues — empty = consistent.
    """
    issues: list[ValidationIssue] = []

    for r in bank_results:
        # A bank marked eligible must have zero failed rules
        if r.eligible and r.failed:
            issues.append(ValidationIssue(
                severity="error",
                bank=r.bank,
                message=f"{r.bank} marked eligible but has {len(r.failed)} failed rule(s): {r.failed[:2]}",
            ))

        # A bank marked not-eligible must have at least one failure or critical missing
        if not r.eligible and not r.failed and not any(
            e.status == "missing" and e.field in {"age","monthly_income","credit_score"}
            for e in r.evaluations
        ):
            issues.append(ValidationIssue(
                severity="warning",
                bank=r.bank,
                message=f"{r.bank} marked not-eligible but has no explicit failures — check rule logic.",
            ))

        # DTI sanity check: dti_ratio should be between 0 and 1 (not 40 instead of 0.40)
        dti_evals = [e for e in r.evaluations if e.field == "dti_ratio" and e.actual_value is not None]
        for e in dti_evals:
            if isinstance(e.actual_value, float) and e.actual_value > 1.5:
                issues.append(ValidationIssue(
                    severity="warning",
                    bank=r.bank,
                    message=f"DTI ratio {e.actual_value} looks like a percentage (0–100) instead of decimal (0–1). "
                            f"Parser may have returned '{e.actual_value}' instead of '{e.actual_value/100:.2f}'.",
                ))

    # Decision consistency
    n_eligible = len(reasoning.eligible_banks)
    n_total    = len(bank_results)
    if decision == "Eligible" and n_eligible < n_total:
        issues.append(ValidationIssue(
            severity="error", bank="",
            message=f"Decision is 'Eligible' but only {n_eligible}/{n_total} banks passed.",
        ))
    if decision == "Not Eligible" and n_eligible > 0:
        issues.append(ValidationIssue(
            severity="error", bank="",
            message=f"Decision is 'Not Eligible' but {n_eligible} bank(s) actually passed.",
        ))

    # Confidence sanity: if breakdown exists and final < 0.5, warn
    if reasoning.confidence_breakdown:
        final_conf = reasoning.confidence_breakdown.get("final", 1.0)
        if final_conf < 0.50:
            issues.append(ValidationIssue(
                severity="warning", bank="",
                message=f"Low confidence ({final_conf:.0%}) — response may be unreliable. "
                        f"Consider providing more profile data.",
            ))

    # Reasoning vs rule mismatch: best_bank should be in eligible_banks
    if reasoning.best_bank and reasoning.best_bank not in reasoning.eligible_banks:
        issues.append(ValidationIssue(
            severity="error", bank=reasoning.best_bank,
            message=f"best_bank '{reasoning.best_bank}' is not in eligible_banks "
                    f"{reasoning.eligible_banks} — reasoning mismatch.",
        ))

    # Critical failure count mismatch: if critical_failures exist but decision = "Eligible"
    if reasoning.critical_failures and decision == "Eligible":
        # Only a problem if ALL banks are marked eligible yet critical failures exist
        if len(reasoning.eligible_banks) == n_total:
            issues.append(ValidationIssue(
                severity="warning", bank="",
                message=f"Critical failures detected ({len(reasoning.critical_failures)}) "
                        f"but all banks are marked eligible — verify rule thresholds.",
            ))

    return issues