"""
Test Suite — Loan Policy Advisor
=================================
Run with: python -m pytest tests/ -v

Tests:
  - Rule engine (deterministic, no LLM)
  - Retrieval (FAISS)
  - Router logic
  - Context fusion
  - Full pipeline (integration — requires API keys)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from services.rule_engine import RuleEngine
from services.router import Router
from services.llm import ParsedQuery
from services.fusion import ContextFusion
from services.retrieval import DocChunk

# ---------------------------------------------------------------------------
# Rule Engine Tests
# ---------------------------------------------------------------------------

class TestRuleEngine:
    """Deterministic tests — no API keys required."""

    def setup_method(self):
        self.engine = RuleEngine()

    def test_axis_eligible(self):
        profile = {
            "age": 28,
            "monthly_income": 40000,
            "credit_score": 750,
            "employment_type": "salaried",
            "work_experience_months": 24,
        }
        result = self.engine.evaluate("axis", profile)
        assert result.eligible is True
        assert len(result.failed) == 0
        assert "Axis" in result.bank

    def test_axis_low_credit_score(self):
        profile = {
            "age": 28,
            "monthly_income": 40000,
            "credit_score": 680,    # below 720 threshold
            "employment_type": "salaried",
            "work_experience_months": 24,
        }
        result = self.engine.evaluate("axis", profile)
        assert result.eligible is False
        assert any("CIBIL" in f or "720" in f for f in result.failed)

    def test_axis_underage(self):
        profile = {
            "age": 19,              # below 21 threshold
            "monthly_income": 40000,
            "credit_score": 750,
            "employment_type": "salaried",
            "work_experience_months": 12,
        }
        result = self.engine.evaluate("axis", profile)
        assert result.eligible is False

    def test_axis_low_income(self):
        profile = {
            "age": 25,
            "monthly_income": 12000,    # below 15000 threshold
            "credit_score": 750,
            "employment_type": "salaried",
            "work_experience_months": 12,
        }
        result = self.engine.evaluate("axis", profile)
        assert result.eligible is False

    def test_missing_fields(self):
        profile = {
            "age": 28,
            "monthly_income": 40000,
            # credit_score missing
        }
        result = self.engine.evaluate("axis", profile)
        assert result.eligible is False
        assert len(result.missing) > 0

    def test_icici_higher_income_threshold(self):
        """ICICI requires ₹25,000 vs Axis's ₹15,000"""
        profile = {
            "age": 25,
            "monthly_income": 20000,   # passes Axis but not ICICI
            "credit_score": 750,
            "employment_type": "salaried",
            "work_experience_months": 24,
        }
        axis_result  = self.engine.evaluate("axis", profile)
        icici_result = self.engine.evaluate("icici", profile)
        assert axis_result.eligible is True
        assert icici_result.eligible is False

    def test_hdfc_high_cibil_requirement(self):
        """HDFC requires 750 CIBIL — stricter than SBI's 700"""
        profile = {
            "age": 30,
            "monthly_income": 35000,
            "credit_score": 730,    # passes SBI (700) but not HDFC (750)
            "employment_type": "salaried",
            "work_experience_months": 24,
            "dti_ratio": 0.3,
        }
        sbi_result  = self.engine.evaluate("sbi", profile)
        hdfc_result = self.engine.evaluate("hdfc", profile)
        assert sbi_result.eligible is True
        assert hdfc_result.eligible is False

    def test_unknown_bank(self):
        result = self.engine.evaluate("UnknownBank", {})
        assert result.eligible is False
        assert "No rules found" in result.summary

    def test_evaluate_all(self):
        profile = {
            "age": 30,
            "monthly_income": 50000,
            "credit_score": 760,
            "employment_type": "salaried",
            "work_experience_months": 36,
            "dti_ratio": 0.25,
        }
        results = self.engine.evaluate_all(profile)
        assert len(results) > 0
        # With strong profile, most banks should pass
        eligible_count = sum(1 for r in results if r.eligible)
        assert eligible_count >= 3

# ---------------------------------------------------------------------------
# Router Tests
# ---------------------------------------------------------------------------

class TestRouter:

    def setup_method(self):
        self.router = Router()

    def _make_parsed(self, intent="general", banks=None, entities=None):
        return ParsedQuery({
            "intent": intent,
            "banks": banks or [],
            "entities": entities or {},
        })

    def test_single_bank_eligibility(self):
        parsed = self._make_parsed(intent="eligibility", banks=["Axis"])
        plan = self.router.route(parsed)
        assert "Axis" in plan.banks
        assert plan.use_bank_specific is True

    def test_comparison_uses_aggregators(self):
        parsed = self._make_parsed(intent="comparison", banks=["Axis", "ICICI"])
        plan = self.router.route(parsed)
        assert plan.use_aggregators is True
        assert "Axis" in plan.banks
        assert "ICICI" in plan.banks

    def test_general_no_bank_uses_all(self):
        parsed = self._make_parsed(intent="general")
        plan = self.router.route(parsed)
        assert len(plan.banks) >= 4  # all known banks

    def test_eligibility_no_bank_checks_all(self):
        parsed = self._make_parsed(intent="eligibility", banks=[])
        plan = self.router.route(parsed)
        assert len(plan.banks) >= 4

# ---------------------------------------------------------------------------
# Context Fusion Tests
# ---------------------------------------------------------------------------

class TestContextFusion:

    def setup_method(self):
        self.fusion = ContextFusion()

    def _make_chunk(self, text, bank, doc_type="eligibility"):
        return DocChunk(text=text, source="test.txt", bank=bank, doc_type=doc_type)

    def test_priority_ordering(self):
        chunks = [
            self._make_chunk("Aggregator insight", "Paisabazaar", "comparison"),
            self._make_chunk("Bank rule text", "Axis", "eligibility"),
            self._make_chunk("RBI regulation", "RBI", "regulatory"),
        ]
        result = self.fusion.fuse(chunks)
        # Bank content should appear before aggregator
        axis_pos = result.find("Axis")
        paisabazaar_pos = result.find("Paisabazaar")
        assert axis_pos < paisabazaar_pos

    def test_deduplication(self):
        # Same prefix text → should appear only once
        text = "Minimum monthly income is ₹15,000 for Axis Bank personal loan applicants."
        chunks = [
            self._make_chunk(text, "Axis"),
            self._make_chunk(text, "Axis"),   # duplicate
        ]
        result = self.fusion.fuse(chunks)
        # Should appear exactly once
        assert result.count("Minimum monthly income") == 1

    def test_empty_chunks(self):
        result = self.fusion.fuse([])
        assert "No relevant context" in result

# ---------------------------------------------------------------------------
# ParsedQuery Profile Normalization Tests
# ---------------------------------------------------------------------------

class TestParsedQueryProfile:

    def test_salary_normalization(self):
        pq = ParsedQuery({"intent": "eligibility", "banks": [], "entities": {"salary": 40000}})
        assert pq.profile.get("monthly_income") == 40000.0

    def test_income_key(self):
        pq = ParsedQuery({"intent": "eligibility", "banks": [], "entities": {"monthly_income": 30000}})
        assert pq.profile.get("monthly_income") == 30000.0

    def test_credit_score_alias(self):
        pq = ParsedQuery({"intent": "eligibility", "banks": [], "entities": {"cibil": 750}})
        assert pq.profile.get("credit_score") == 750.0

# ---------------------------------------------------------------------------
# Integration: Example queries (manual / requires API key)
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "Am I eligible for an Axis Bank personal loan? I am 28 years old, earn ₹40,000 per month, and have a CIBIL score of 750. I am salaried with 2 years experience.",
    "Compare personal loans from HDFC and SBI for someone with 30k salary and 720 CIBIL.",
    "What documents do I need for an ICICI personal loan?",
    "I earn 12 lakhs per year, age 35, CIBIL 780, salaried with 5 years exp. Which bank gives best rate?",
    "My CIBIL score is 650. Can I still get a personal loan?",
]

if __name__ == "__main__":
    # Run rule engine tests without pytest
    print("=== Running Rule Engine Tests ===\n")
    engine = RuleEngine()

    test_cases = [
        {
            "name": "Strong applicant — all banks should pass",
            "profile": {
                "age": 32,
                "monthly_income": 60000,
                "credit_score": 780,
                "employment_type": "salaried",
                "work_experience_months": 48,
                "dti_ratio": 0.2,
            },
        },
        {
            "name": "Borderline — low CIBIL (700)",
            "profile": {
                "age": 26,
                "monthly_income": 25000,
                "credit_score": 700,
                "employment_type": "salaried",
                "work_experience_months": 18,
                "dti_ratio": 0.35,
            },
        },
        {
            "name": "Not eligible — low income (₹12,000)",
            "profile": {
                "age": 24,
                "monthly_income": 12000,
                "credit_score": 720,
                "employment_type": "salaried",
                "work_experience_months": 14,
            },
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {tc['name']}")
        print(f"Profile: {tc['profile']}")
        results = engine.evaluate_all(tc["profile"])
        for r in results:
            icon = "✅" if r.eligible else "❌"
            print(f"  {icon} {r.bank}: {r.summary}")
