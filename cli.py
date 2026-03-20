#!/usr/bin/env python3
"""
CLI Demo — Loan Policy Advisor
================================
Test the pipeline from the command line without starting the API.

Usage:
  python cli.py
  python cli.py "Am I eligible for Axis loan with 40k salary and 750 CIBIL?"
  python cli.py --rule-only   (test rule engine without LLM)
"""

import sys
import os
import json
import argparse
sys.path.insert(0, os.path.dirname(__file__))

# ── Rule-only mode (no API key needed) ───────────────────────────────────────

def run_rule_demo():
    from services.rule_engine import RuleEngine

    print("\n" + "="*65)
    print("  RULE ENGINE DEMO  (no LLM / API key required)")
    print("="*65)

    engine = RuleEngine()

    scenarios = [
        {
            "label": "✅ Strong Applicant",
            "profile": {
                "age": 30, "monthly_income": 55000, "credit_score": 760,
                "employment_type": "salaried", "work_experience_months": 36, "dti_ratio": 0.2,
            },
        },
        {
            "label": "⚠️  Borderline Applicant (low CIBIL)",
            "profile": {
                "age": 26, "monthly_income": 28000, "credit_score": 705,
                "employment_type": "salaried", "work_experience_months": 20, "dti_ratio": 0.38,
            },
        },
        {
            "label": "❌ Below Threshold Applicant",
            "profile": {
                "age": 24, "monthly_income": 12000, "credit_score": 650,
                "employment_type": "salaried", "work_experience_months": 8,
            },
        },
        {
            "label": "❓ Missing Fields",
            "profile": {
                "age": 28, "monthly_income": 40000,
                # credit_score and employment_type missing
            },
        },
    ]

    for scenario in scenarios:
        print(f"\n{'─'*65}")
        print(f"  {scenario['label']}")
        print(f"  Profile: {scenario['profile']}")
        print()
        results = engine.evaluate_all(scenario["profile"])
        for r in results:
            icon = "✅" if r.eligible else ("⚠️ " if r.missing else "❌")
            print(f"  {icon} {r.bank:<8} | {r.summary}")


# ── Full pipeline mode ────────────────────────────────────────────────────────

def run_full(query: str):
    from pipeline import LoanAdvisorPipeline

    print("\n" + "="*65)
    print("  LOAN POLICY ADVISOR")
    print("="*65)
    print(f"  Query: {query}\n")

    pipeline = LoanAdvisorPipeline()
    response = pipeline.query(query)

    print(f"  Decision:    {response.decision}")
    print(f"  Confidence:  {response.confidence:.0%}")
    print(f"  Latency:     {response.latency_ms:.0f}ms {'(cached)' if response.from_cache else ''}")
    print(f"\n  Summary:\n  {response.summary}")

    if response.rule_results:
        print(f"\n  Rule Engine Results:")
        for r in response.rule_results:
            icon = "✅" if r["eligible"] else "❌"
            print(f"    {icon} {r['bank']}: {r['summary'][:70]}")

    if response.detailed_explanation:
        print(f"\n  Explanation:\n")
        for line in response.detailed_explanation.split("\n"):
            print(f"  {line}")

    if response.recommendations:
        print(f"\n  Recommendations:")
        for rec in response.recommendations:
            print(f"    • {rec}")

    if response.sources_cited:
        print(f"\n  Sources: {', '.join(response.sources_cited)}")

    print(f"\n{'='*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Loan Policy Advisor CLI")
    parser.add_argument("query", nargs="?", help="Natural language loan query")
    parser.add_argument("--rule-only", action="store_true", help="Run rule engine demo only (no API key needed)")
    args = parser.parse_args()

    if args.rule_only or (not args.query and not os.getenv("ANTHROPIC_API_KEY")):
        run_rule_demo()
    else:
        query = args.query or (
            "Am I eligible for Axis Bank personal loan? "
            "I am 28, earn ₹40,000/month, CIBIL score 750, salaried with 2 years experience."
        )
        run_full(query)


if __name__ == "__main__":
    main()
