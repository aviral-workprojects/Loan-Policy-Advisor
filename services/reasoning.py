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
        }

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

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

    # Top 3 most impactful failures (unique by field)
    seen_fields: set[str] = set()
    top_failures: list[dict] = []
    for f in sorted(all_failures, key=lambda x: x["bank"]):
        if f["field"] not in seen_fields:
            top_failures.append(f)
            seen_fields.add(f["field"])
        if len(top_failures) >= 3:
            break

    # Data completeness
    provided = [k for k in _KNOWN_PROFILE_FIELDS if profile.get(k) is not None]
    missing  = [k for k in _KNOWN_PROFILE_FIELDS if profile.get(k) is None]
    completeness = len(provided) / len(_KNOWN_PROFILE_FIELDS)

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
    )


# ---------------------------------------------------------------------------
# Calibrated confidence
# ---------------------------------------------------------------------------

def compute_confidence(
    bank_results: list[BankResult],
    reasoning:    ReasoningContext,
    retrieval_chunk_count: int = 5,
) -> float:
    """
    Calibrated confidence score:
        0.5 × avg_rule_score
      + 0.3 × data_completeness
      + 0.2 × retrieval_confidence

    Caps:
      - Any critical rule (income/CIBIL/age) failed → cap ≤ 0.60
      - Multiple banks failed                       → reduce by 0.05 per failed bank beyond 1
    """
    if not bank_results:
        return 0.3

    avg_rule_score = sum(r.rule_score for r in bank_results) / len(bank_results)
    data_comp      = reasoning.data_completeness
    retrieval_conf = min(retrieval_chunk_count / 8.0, 1.0)  # 8 chunks = full confidence

    raw = 0.5 * avg_rule_score + 0.3 * data_comp + 0.2 * retrieval_conf

    # Check if any critical rule failed
    critical_failed = any(
        e.status == "fail" and e.field in {"monthly_income", "credit_score", "age"}
        for r in bank_results
        for e in r.evaluations
    )
    if critical_failed:
        raw = min(raw, 0.60)

    # Penalty for multiple failed banks
    n_failed = len(reasoning.rejected_banks)
    if n_failed > 1:
        raw -= (n_failed - 1) * 0.04

    return round(max(0.05, min(raw, 0.99)), 2)


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

    return issues
