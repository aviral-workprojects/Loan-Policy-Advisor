"""
pdf_pipeline/eligibility_engine.py
=====================================
Deterministic eligibility reasoning engine.

Key design principle: the LLM NEVER makes eligibility decisions.
All pass/fail logic is computed here from YAML rules. The LLM only
explains the decision that this engine already made.

This separation gives:
  - Reproducible, auditable decisions
  - No hallucination of eligibility outcomes
  - Fast rule evaluation (no API call for the core decision)
  - Clear, structured failures that the LLM can explain precisely
"""

from __future__ import annotations

import logging
import operator as op
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RuleEval:
    """Result of evaluating one rule against the applicant profile."""
    rule_id:      str
    field:        str
    passed:       bool
    actual:       Any
    expected:     Any
    operator:     str
    message:      str
    status:       str       # "pass" | "fail" | "missing" | "error"
    reason:       str       # plain-English one-liner

@dataclass
class BankEligibility:
    bank:         str
    loan_type:    str
    eligible:     bool
    evaluations:  list[RuleEval]
    rule_score:   float     # pass_count / evaluated_count
    summary:      str

    @property
    def passed_rules(self) -> list[str]:
        return [e.message for e in self.evaluations if e.status == "pass"]

    @property
    def failed_rules(self) -> list[str]:
        return [e.message for e in self.evaluations if e.status == "fail"]

    @property
    def missing_fields(self) -> list[str]:
        return [e.message for e in self.evaluations if e.status == "missing"]

    def to_dict(self) -> dict:
        return {
            "bank":        self.bank,
            "eligible":    self.eligible,
            "rule_score":  self.rule_score,
            "summary":     self.summary,
            "passed":      self.passed_rules,
            "failed":      self.failed_rules,
            "missing":     self.missing_fields,
            "evaluations": [
                {
                    "rule_id":  e.rule_id,
                    "field":    e.field,
                    "passed":   e.passed,
                    "actual":   e.actual,
                    "expected": e.expected,
                    "operator": e.operator,
                    "message":  e.message,
                    "status":   e.status,
                    "reason":   e.reason,
                }
                for e in self.evaluations
            ],
        }


@dataclass
class EligibilityReport:
    """Full multi-bank eligibility report."""
    eligible_banks:   list[str]
    rejected_banks:   list[str]
    partial_banks:    list[str]
    best_bank:        str | None
    best_bank_reason: str | None
    closest_bank:     str | None     # highest score among rejected
    decision:         str            # "Eligible" | "Partially Eligible" | "Not Eligible" | "Insufficient Data"
    critical_failures: list[dict]   # income/CIBIL/age failures only
    top_failures:     list[dict]
    profile_completeness: float      # 0–1
    bank_results:     list[BankEligibility]
    confidence:       float

    def to_dict(self) -> dict:
        return {
            "decision":           self.decision,
            "eligible_banks":     self.eligible_banks,
            "rejected_banks":     self.rejected_banks,
            "best_bank":          self.best_bank,
            "best_bank_reason":   self.best_bank_reason,
            "closest_bank":       self.closest_bank,
            "critical_failures":  self.critical_failures,
            "top_failures":       self.top_failures,
            "profile_completeness": round(self.profile_completeness, 2),
            "confidence":         self.confidence,
            "bank_results":       [r.to_dict() for r in self.bank_results],
        }


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

_OPS = {
    ">=":     op.ge,
    "<=":     op.le,
    ">":      op.gt,
    "<":      op.lt,
    "==":     op.eq,
    "!=":     op.ne,
    "in":     lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}

# Critical fields — missing them blocks eligibility entirely
_CRITICAL_FIELDS = {"age", "monthly_income", "credit_score"}

# Optional fields — use defaults if not provided
_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "work_experience_months": 12,
    "dti_ratio":              0.35,
    "employment_type":        "salaried",
}

_ALL_PROFILE_FIELDS = list(_CRITICAL_FIELDS) + list(_OPTIONAL_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EligibilityEngine:
    """
    Loads YAML rule files and evaluates applicant profiles against them.

    Rule file format (rules/<bank>.yaml):
      loan_type: personal
      logic: all     # all rules must pass; use "any" for lenient mode
      rules:
        - id: min_income
          field: monthly_income
          operator: ">="
          value: 25000
          message: "Monthly income must be at least ₹25,000"
    """

    def __init__(self, rules_dir: Path):
        self.rules_dir = rules_dir
        self._cache: dict[str, dict] = {}

    def available_banks(self) -> list[str]:
        return [p.stem.upper() for p in sorted(self.rules_dir.glob("*.yaml"))]

    def evaluate(self, bank: str, profile: dict[str, Any]) -> BankEligibility:
        """Evaluate one bank's rules against the provided profile."""
        data = self._load_rules(bank)
        if data is None:
            return BankEligibility(
                bank=bank, loan_type="unknown", eligible=False,
                evaluations=[], rule_score=0.0,
                summary=f"No rules found for {bank}",
            )

        rules     = data.get("rules", [])
        logic     = data.get("logic", "all")
        loan_type = data.get("loan_type", "personal")

        # Merge profile with optional defaults (never override critical fields)
        effective = {**profile}
        for fname, fval in _OPTIONAL_DEFAULTS.items():
            if fname not in effective or effective[fname] is None:
                effective[fname] = fval

        evaluations: list[RuleEval] = []

        for rule in rules:
            rule_id = rule.get("id", "unnamed")
            fld     = rule["field"]
            oper    = rule["operator"]
            thresh  = rule["value"]
            message = rule.get("message", f"{fld} {oper} {thresh}")
            actual  = effective.get(fld)

            if actual is None:
                evaluations.append(RuleEval(
                    rule_id=rule_id, field=fld, passed=False,
                    actual=None, expected=thresh, operator=oper,
                    message=message, status="missing",
                    reason=f"'{fld}' not provided in applicant profile.",
                ))
                continue

            # Type coercion
            try:
                if isinstance(thresh, (int, float)) and oper not in ("in", "not_in"):
                    actual = float(actual)
            except (ValueError, TypeError):
                evaluations.append(RuleEval(
                    rule_id=rule_id, field=fld, passed=False,
                    actual=actual, expected=thresh, operator=oper,
                    message=message, status="error",
                    reason=f"Cannot compare '{actual}' with threshold '{thresh}'.",
                ))
                continue

            eval_fn = _OPS.get(oper)
            if eval_fn is None:
                evaluations.append(RuleEval(
                    rule_id=rule_id, field=fld, passed=False,
                    actual=actual, expected=thresh, operator=oper,
                    message=message, status="error",
                    reason=f"Unknown operator '{oper}'.",
                ))
                continue

            try:
                passed = bool(eval_fn(actual, thresh))
            except Exception as e:
                evaluations.append(RuleEval(
                    rule_id=rule_id, field=fld, passed=False,
                    actual=actual, expected=thresh, operator=oper,
                    message=message, status="error",
                    reason=f"Evaluation error: {e}",
                ))
                continue

            status = "pass" if passed else "fail"
            reason = (
                f"{actual} {oper} {thresh} ✓"
                if passed
                else f"Applicant has {actual}, requirement is {oper} {thresh}"
            )
            evaluations.append(RuleEval(
                rule_id=rule_id, field=fld, passed=passed,
                actual=actual, expected=thresh, operator=oper,
                message=message, status=status, reason=reason,
            ))

        # Determine overall eligibility
        fail_count     = sum(1 for e in evaluations if e.status == "fail")
        pass_count     = sum(1 for e in evaluations if e.status == "pass")
        evaluated      = pass_count + fail_count
        rule_score     = round(pass_count / max(evaluated, 1), 3)

        critical_missing = [
            e for e in evaluations
            if e.status == "missing" and e.field in _CRITICAL_FIELDS
        ]

        eligible = (
            (fail_count == 0 and not critical_missing) if logic == "all"
            else (pass_count > 0)
        )

        if eligible:
            summary = f"✅ Eligible — all {pass_count} rules passed."
        elif critical_missing:
            fields = ", ".join(e.field for e in critical_missing)
            summary = f"⚠️ Cannot evaluate — missing: {fields}"
        else:
            top = [e.reason for e in evaluations if e.status == "fail"][:2]
            summary = f"❌ Not eligible — {fail_count} rule(s) failed: {'; '.join(top)}"

        return BankEligibility(
            bank=bank, loan_type=loan_type, eligible=eligible,
            evaluations=evaluations, rule_score=rule_score, summary=summary,
        )

    def evaluate_all(self, profile: dict[str, Any]) -> list[BankEligibility]:
        """Evaluate profile against all available banks."""
        return [self.evaluate(bank.lower(), profile) for bank in self.available_banks()]

    def build_report(self, profile: dict[str, Any], banks: list[str] | None = None) -> EligibilityReport:
        """
        Evaluate profile against specified banks (or all) and build a full report.
        """
        target_banks = [b.lower() for b in (banks or self.available_banks())]
        results = [self.evaluate(b, profile) for b in target_banks]

        eligible_banks = [r.bank for r in results if r.eligible]
        rejected_banks = [r.bank for r in results if not r.eligible and not r.missing_fields]
        partial_banks  = [r.bank for r in results if not r.eligible and r.missing_fields]

        best_result  = max((r for r in results if r.eligible),     key=lambda r: r.rule_score, default=None)
        closest      = max((r for r in results if not r.eligible), key=lambda r: r.rule_score, default=None)

        best_bank        = best_result.bank if best_result else None
        best_bank_reason = _explain_best_bank(best_result) if best_result else None
        closest_bank     = closest.bank if closest else None

        # Gather critical failures
        critical_failures = [
            {
                "bank":     e.bank,
                "field":    ev.field,
                "actual":   ev.actual,
                "required": ev.expected,
                "operator": ev.operator,
                "reason":   ev.reason,
            }
            for e in results
            for ev in e.evaluations
            if ev.status == "fail" and ev.field in _CRITICAL_FIELDS
        ]

        # Top unique failures by field priority
        _FIELD_PRIORITY = {
            "credit_score": 1, "monthly_income": 2, "dti_ratio": 3,
            "work_experience_months": 4, "employment_type": 5, "age": 6,
        }
        all_failures = sorted(
            [
                {
                    "bank":     r.bank,
                    "field":    ev.field,
                    "actual":   ev.actual,
                    "required": ev.expected,
                    "operator": ev.operator,
                    "reason":   ev.reason,
                }
                for r in results
                for ev in r.evaluations
                if ev.status == "fail"
            ],
            key=lambda f: _FIELD_PRIORITY.get(f["field"], 99),
        )
        seen_fields: set[str] = set()
        top_failures: list[dict] = []
        for f in all_failures:
            if f["field"] not in seen_fields:
                top_failures.append(f)
                seen_fields.add(f["field"])
            if len(top_failures) >= 5:
                break

        # Profile completeness
        provided = [k for k in _ALL_PROFILE_FIELDS if profile.get(k) is not None]
        completeness = len(provided) / len(_ALL_PROFILE_FIELDS)

        # Decision
        n_eligible = len(eligible_banks)
        n_total    = len(results)
        if not profile:
            decision = "Insufficient Data"
        elif partial_banks and n_eligible == 0:
            decision = "Insufficient Data"
        elif n_eligible == n_total:
            decision = "Eligible"
        elif n_eligible > 0:
            decision = "Partially Eligible"
        else:
            decision = "Not Eligible"

        # Confidence
        avg_score = sum(r.rule_score for r in results) / max(len(results), 1)
        confidence = round(
            min(0.5 * avg_score + 0.3 * completeness + 0.2, 0.99),
            2,
        )

        return EligibilityReport(
            eligible_banks=eligible_banks,
            rejected_banks=rejected_banks,
            partial_banks=partial_banks,
            best_bank=best_bank,
            best_bank_reason=best_bank_reason,
            closest_bank=closest_bank,
            decision=decision,
            critical_failures=critical_failures,
            top_failures=top_failures,
            profile_completeness=completeness,
            bank_results=results,
            confidence=confidence,
        )

    def _load_rules(self, bank: str) -> dict | None:
        key = bank.lower()
        if key not in self._cache:
            path = self.rules_dir / f"{key}.yaml"
            if not path.exists():
                logger.warning("[Eligibility] No rule file: %s", path)
                return None
            with open(path, encoding="utf-8") as f:
                self._cache[key] = yaml.safe_load(f)
        return self._cache[key]


def _explain_best_bank(result: BankEligibility) -> str:
    score_pct = round(result.rule_score * 100)
    n_passed  = len(result.passed_rules)
    n_failed  = len(result.failed_rules)
    if n_failed == 0:
        return (
            f"{result.bank} is recommended — passed all {n_passed} eligibility "
            f"rules ({score_pct}% score) and has the most favourable terms."
        )
    return (
        f"{result.bank} has the highest eligibility score ({score_pct}%) among "
        f"evaluated banks, with {n_passed} rules passed and {n_failed} to address."
    )
