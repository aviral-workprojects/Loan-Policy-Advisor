"""
Rule Engine — Deterministic DSL Evaluator
==========================================
100% deterministic — no LLM involvement.
Reads YAML rules per bank, evaluates applicant profile.
"""

from __future__ import annotations
import operator as op
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from config import RULES_DIR

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RuleResult:
    eligible: bool
    passed: list[str]
    failed: list[str]
    missing: list[str]
    bank: str
    loan_type: str
    summary: str

# ---------------------------------------------------------------------------
# Operator Registry
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

# Optional fields — safe defaults applied when absent so the engine always
# reaches a decision rather than returning "Insufficient Data"
_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "work_experience_months": 12,    # assume minimum met
    "dti_ratio":              0.35,  # assume healthy DTI
    "employment_type":        "salaried",  # most common case
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RuleEngine:

    def __init__(self, rules_dir: Path = RULES_DIR):
        self.rules_dir = rules_dir
        self._cache: dict[str, dict] = {}

    def _load_rules(self, bank: str) -> dict | None:
        key = bank.lower()
        if key in self._cache:
            return self._cache[key]
        rule_file = self.rules_dir / f"{key}.yaml"
        if not rule_file.exists():
            return None
        with open(rule_file, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._cache[key] = data
        return data

    def available_banks(self) -> list[str]:
        return [p.stem for p in self.rules_dir.glob("*.yaml")]

    def evaluate(self, bank: str, profile: dict[str, Any]) -> RuleResult:
        """
        Evaluate profile against bank rules deterministically.

        Key design decisions:
        - Optional fields (employment_type, work_experience, dti_ratio) get
          safe defaults so partial profiles still produce a clear decision.
        - Missing CRITICAL fields (age, monthly_income, credit_score) are
          tracked but do NOT block — they are reported as missing so the LLM
          can flag them in recommendations.
        - eligible = True only when zero failed rules.
        """
        rules_data = self._load_rules(bank)
        if rules_data is None:
            return RuleResult(
                eligible=False, passed=[], failed=[], missing=[],
                bank=bank, loan_type="unknown",
                summary=f"No rules found for bank '{bank}'.",
            )

        rules     = rules_data.get("rules", [])
        logic     = rules_data.get("logic", "all")
        loan_type = rules_data.get("loan_type", "personal")

        # Apply optional defaults for missing fields
        effective = {**profile}
        for fname, fdefault in _OPTIONAL_DEFAULTS.items():
            if fname not in effective or effective[fname] is None:
                effective[fname] = fdefault

        passed:  list[str] = []
        failed:  list[str] = []
        missing: list[str] = []

        for rule in rules:
            field_   = rule["field"]
            oper     = rule["operator"]
            thresh   = rule["value"]
            message  = rule.get("message", f"{field_} {oper} {thresh}")

            user_val = effective.get(field_)

            # Still missing after defaults → report but do NOT fail
            if user_val is None:
                missing.append(message)
                continue

            # Type coercion: numeric thresholds need float comparison
            try:
                if isinstance(thresh, (int, float)):
                    user_val = float(user_val)
            except (ValueError, TypeError):
                failed.append(f"{message} [type error: got {user_val!r}]")
                continue

            eval_fn = _OPS.get(oper)
            if eval_fn is None:
                failed.append(f"{message} [unknown operator '{oper}']")
                continue

            # DEBUG: uncomment to trace evaluation in uvicorn terminal
            # print(f"[RuleDebug] {bank}.{field_}: {user_val!r} {oper} {thresh!r} → {eval_fn(user_val, thresh)}")

            try:
                result = eval_fn(user_val, thresh)
            except Exception as e:
                failed.append(f"{message} [eval error: {e}]")
                continue

            if result:
                passed.append(message)
            else:
                failed.append(message)

        # ── Decision: eligible = zero failures (missing fields are noted, not failed)
        if logic == "all":
            eligible = len(failed) == 0
        else:  # "any"
            eligible = len(passed) > 0

        # ── Summary
        if eligible:
            summary = f"✅ Eligible for {bank} {loan_type} loan."
        elif failed:
            summary = f"❌ Not eligible — failed: {'; '.join(failed[:2])}"
        else:
            summary = f"⚠️ Missing info for {bank}: {'; '.join(missing[:2])}"

        return RuleResult(
            eligible=eligible, passed=passed, failed=failed, missing=missing,
            bank=bank, loan_type=loan_type, summary=summary,
        )

    def evaluate_all(self, profile: dict[str, Any]) -> list[RuleResult]:
        return [self.evaluate(bank, profile) for bank in self.available_banks()]
