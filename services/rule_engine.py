"""
Rule Engine v3 — Strict Per-Bank Deterministic Evaluator
=========================================================
Each rule returns a rich RuleEvaluation with actual vs expected values.
No silent failures. No auto-pass on None.
Critical fields (age, income, credit_score) → missing = block eligibility.
Optional fields (employment_type, work_experience, dti_ratio) → use defaults.
"""

from __future__ import annotations
import operator as op
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from config import RULES_DIR

# ---------------------------------------------------------------------------
# Rich per-rule result
# ---------------------------------------------------------------------------

@dataclass
class RuleEvaluation:
    rule_id:      str
    field:        str
    passed:       bool
    actual_value: Any
    expected:     Any
    operator:     str
    message:      str
    status:       str      # "pass" | "fail" | "missing" | "error"
    reason:       str      # human-readable one-liner

@dataclass
class BankResult:
    bank:         str
    loan_type:    str
    eligible:     bool
    evaluations:  list[RuleEvaluation]
    rule_score:   float    # passed_count / total_evaluated (0–1)
    summary:      str

    # Convenience accessors used by pipeline/reasoning
    @property
    def passed(self) -> list[str]:
        return [e.message for e in self.evaluations if e.status == "pass"]

    @property
    def failed(self) -> list[str]:
        return [e.message for e in self.evaluations if e.status == "fail"]

    @property
    def missing(self) -> list[str]:
        return [e.message for e in self.evaluations if e.status == "missing"]

    @property
    def failed_evaluations(self) -> list[RuleEvaluation]:
        return [e for e in self.evaluations if e.status == "fail"]

    def to_dict(self) -> dict:
        return {
            "bank":       self.bank,
            "eligible":   self.eligible,
            "rule_score": self.rule_score,
            "summary":    self.summary,
            "passed":     self.passed,
            "failed":     self.failed,
            "missing":    self.missing,
            "evaluations": [
                {
                    "rule_id":      e.rule_id,
                    "field":        e.field,
                    "passed":       e.passed,
                    "actual_value": e.actual_value,
                    "expected":     e.expected,
                    "operator":     e.operator,
                    "message":      e.message,
                    "status":       e.status,
                    "reason":       e.reason,
                }
                for e in self.evaluations
            ],
        }

# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

_OPS: dict[str, Any] = {
    ">=":     op.ge,
    "<=":     op.le,
    ">":      op.gt,
    "<":      op.lt,
    "==":     op.eq,
    "!=":     op.ne,
    "in":     lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
}

# Critical fields: if absent, eligibility is blocked (cannot guess these)
_CRITICAL_FIELDS = {"age", "monthly_income", "credit_score"}

# Optional fields: safe defaults applied when absent
_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "work_experience_months": 12,
    "dti_ratio":              0.35,
    "employment_type":        "salaried",
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RuleEngine:

    def __init__(self, rules_dir: Path = RULES_DIR):
        self.rules_dir = rules_dir
        self._yaml_cache: dict[str, dict] = {}

    def _load(self, bank: str) -> dict | None:
        key = bank.lower()
        if key not in self._yaml_cache:
            path = self.rules_dir / f"{key}.yaml"
            if not path.exists():
                return None
            with open(path, encoding="utf-8") as f:
                self._yaml_cache[key] = yaml.safe_load(f)
        return self._yaml_cache[key]

    def available_banks(self) -> list[str]:
        return [p.stem for p in sorted(self.rules_dir.glob("*.yaml"))]

    def evaluate(self, bank: str, profile: dict[str, Any]) -> BankResult:
        """
        Evaluate profile against ONE bank's rules.
        Returns BankResult with per-rule detail.
        """
        data = self._load(bank)
        if data is None:
            return BankResult(
                bank=bank, loan_type="unknown", eligible=False,
                evaluations=[], rule_score=0.0,
                summary=f"No rule file found for '{bank}'.",
            )

        rules     = data.get("rules", [])
        logic     = data.get("logic", "all")
        loan_type = data.get("loan_type", "personal")

        # Build effective profile: original + optional defaults (never for critical)
        effective: dict[str, Any] = {**profile}
        for fname, fval in _OPTIONAL_DEFAULTS.items():
            if fname not in effective or effective[fname] is None:
                effective[fname] = fval

        evaluations: list[RuleEvaluation] = []

        for rule in rules:
            rule_id = rule.get("id", "unnamed")
            field_  = rule["field"]
            oper    = rule["operator"]
            thresh  = rule["value"]
            message = rule.get("message", f"{field_} {oper} {thresh}")

            actual = effective.get(field_)

            # ── MISSING ──────────────────────────────────────────────────────
            if actual is None:
                status = "missing"
                evaluations.append(RuleEvaluation(
                    rule_id=rule_id, field=field_,
                    passed=False, actual_value=None, expected=thresh,
                    operator=oper, message=message, status=status,
                    reason=f"Field '{field_}' not provided.",
                ))
                continue

            # ── TYPE COERCION ─────────────────────────────────────────────────
            try:
                if isinstance(thresh, (int, float)) and oper not in ("in", "not_in"):
                    actual = float(actual)
            except (ValueError, TypeError):
                evaluations.append(RuleEvaluation(
                    rule_id=rule_id, field=field_,
                    passed=False, actual_value=actual, expected=thresh,
                    operator=oper, message=message, status="error",
                    reason=f"Cannot compare '{actual}' with threshold '{thresh}'.",
                ))
                continue

            # ── EVALUATE ──────────────────────────────────────────────────────
            eval_fn = _OPS.get(oper)
            if eval_fn is None:
                evaluations.append(RuleEvaluation(
                    rule_id=rule_id, field=field_,
                    passed=False, actual_value=actual, expected=thresh,
                    operator=oper, message=message, status="error",
                    reason=f"Unknown operator '{oper}'.",
                ))
                continue

            try:
                result = bool(eval_fn(actual, thresh))
            except Exception as e:
                evaluations.append(RuleEvaluation(
                    rule_id=rule_id, field=field_,
                    passed=False, actual_value=actual, expected=thresh,
                    operator=oper, message=message, status="error",
                    reason=f"Evaluation error: {e}",
                ))
                continue

            if result:
                reason = f"{actual} {oper} {thresh} ✓"
                status = "pass"
            else:
                reason = f"{actual} {oper} {thresh} ✗ — requires {thresh}, got {actual}"
                status = "fail"

            evaluations.append(RuleEvaluation(
                rule_id=rule_id, field=field_,
                passed=result, actual_value=actual, expected=thresh,
                operator=oper, message=message, status=status, reason=reason,
            ))

        # ── DECIDE ────────────────────────────────────────────────────────────
        fail_count    = sum(1 for e in evaluations if e.status == "fail")
        missing_count = sum(1 for e in evaluations if e.status == "missing")
        pass_count    = sum(1 for e in evaluations if e.status == "pass")
        evaluated     = pass_count + fail_count   # missing/error not in denominator

        rule_score = round(pass_count / max(evaluated, 1), 3)

        # Critical missing fields block eligibility
        critical_missing = [
            e for e in evaluations
            if e.status == "missing" and e.field in _CRITICAL_FIELDS
        ]

        if logic == "all":
            eligible = (fail_count == 0) and (len(critical_missing) == 0)
        else:
            eligible = (pass_count > 0)

        # ── SUMMARY ───────────────────────────────────────────────────────────
        if eligible:
            summary = f"✅ Eligible — all {pass_count} rules passed."
        elif critical_missing:
            fields = ", ".join(e.field for e in critical_missing)
            summary = f"⚠️ Cannot evaluate — missing critical fields: {fields}"
        else:
            top_fails = [e.reason for e in evaluations if e.status == "fail"][:2]
            summary = f"❌ Not eligible — {fail_count} rule(s) failed: {'; '.join(top_fails)}"

        return BankResult(
            bank=bank, loan_type=loan_type, eligible=eligible,
            evaluations=evaluations, rule_score=rule_score, summary=summary,
        )
