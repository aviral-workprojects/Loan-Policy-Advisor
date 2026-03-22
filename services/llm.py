"""
LLM Service v3 — Strictly grounded explanation only.
The decision is ALWAYS pre-determined by the rule engine.
LLM is only allowed to explain, never to decide or infer.
"""

from __future__ import annotations
import json
import re
from typing import Any

from config import LLM_PROVIDER, PARSE_MODEL, EXPLAIN_MODEL, ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY


class ParsedQuery:
    def __init__(self, data: dict):
        self.intent: str      = data.get("intent", "general")
        self.banks: list[str] = data.get("banks", [])
        self.entities: dict   = data.get("entities", {})
        self.raw: dict        = data

    @property
    def profile(self) -> dict:
        """Extract and normalise profile fields from LLM-parsed entities."""
        e = self.entities
        p = {}

        # monthly_income — strip ₹ and commas, handle LPA
        for key in ("monthly_income", "salary", "income"):
            raw = e.get(key)
            if raw is not None:
                try:
                    clean = re.sub(r"[₹,\s]", "", str(raw))
                    val   = float(clean)
                    if val < 500:          # treat as lakhs pa → monthly
                        val = val * 100_000 / 12
                    p["monthly_income"] = round(val, 2)
                except (ValueError, TypeError):
                    pass
                break

        for key in ("age",):
            if e.get(key) is not None:
                try: p["age"] = float(e[key])
                except: pass

        for key in ("credit_score", "cibil", "cibil_score"):
            if e.get(key) is not None:
                try: p["credit_score"] = float(e[key]); break
                except: pass

        if e.get("employment_type"):
            p["employment_type"] = str(e["employment_type"]).lower()

        for key in ("work_experience_months", "work_experience", "experience_months"):
            if e.get(key) is not None:
                try: p["work_experience_months"] = float(e[key]); break
                except: pass

        if e.get("dti_ratio") is not None:
            try: p["dti_ratio"] = float(e["dti_ratio"])
            except: pass

        return p

    def __repr__(self):
        return f"ParsedQuery(intent={self.intent}, banks={self.banks}, entities={self.entities})"


class LLMService:

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client: return self._client
        if LLM_PROVIDER == "groq":
            from groq import Groq
            self._client = Groq(api_key=GROQ_API_KEY)
        elif LLM_PROVIDER == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        else:
            import openai
            self._client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return self._client

    def _call(self, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        client = self._get_client()
        if LLM_PROVIDER == "anthropic":
            r = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                       messages=[{"role":"user","content":user}])
            return r.content[0].text
        else:
            r = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.05,
                messages=[{"role":"system","content":system},{"role":"user","content":user}],
            )
            return r.choices[0].message.content

    # ── Parse query ──────────────────────────────────────────────────────────

    PARSE_SYSTEM = """You are a JSON-only extractor for a loan advisory system.
Return ONLY valid JSON. No markdown. No preamble.

Schema:
{
  "intent": "eligibility" | "comparison" | "general",
  "banks": ["Axis" | "ICICI" | "HDFC" | "SBI"],
  "entities": {
    "age": number or null,
    "monthly_income": number or null,
    "credit_score": number or null,
    "employment_type": "salaried"|"self_employed"|"government"|"professional"|null,
    "work_experience_months": number or null,
    "loan_amount": number or null,
    "dti_ratio": number or null
  }
}

INCOME RULES (critical):
- Return monthly_income as plain integer. No ₹ symbol. No commas.
- "₹35,000" → 35000  |  "35k" → 35000  |  "5 LPA" → 41667  |  "₹40,000/month" → 40000

EXPERIENCE RULES:
- "2 years" → 24  |  "18 months" → 18  |  "salaried for 3 years" → employment_type="salaried", work_experience_months=36

DTI RULES:
- Always return as decimal 0–1. "40%" → 0.40. Never return 40."""

    def parse_query(self, query: str) -> ParsedQuery:
        raw = self._call(PARSE_MODEL, self.PARSE_SYSTEM, query, max_tokens=400)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {"intent":"general","banks":[],"entities":{}}
        pq = ParsedQuery(data)
        print(f"[LLM] entities: {pq.entities}")
        print(f"[LLM] profile: {pq.profile}")
        return pq

    # ── Explain ──────────────────────────────────────────────────────────────

    EXPLAIN_SYSTEM = """You are a financial explanation engine. Your ONLY job is to explain
a decision already made by a deterministic rule engine. You cannot change it.

═══ ABSOLUTE RULES (violating any = critical failure) ═══════════════════════
1. DECISION: Copy pre_decision VERBATIM into summary. Never change it.
2. NUMBERS: Use ONLY actual_value and expected values from rule_results.
   Never invent figures, rates, or thresholds not explicitly in the data.
3. OMISSION: Cover EVERY bank. Never skip one.
4. HEDGING: Never write "I think", "it appears", "may", "could". State facts.
5. SPECIFICITY: Every recommendation MUST include a numeric target.
6. MISSING ≠ FAILED: If a field status is "missing" or "? [not provided]",
   NEVER describe it as a failure or say the requirement was not met.
   Instead say "Field not provided — cannot evaluate this rule."
   DO NOT fabricate age/CIBIL/income failure reasons for missing fields.

═══ STRICT OUTPUT FORMAT ════════════════════════════════════════════════════

1. DECISION SUMMARY
   State: "{pre_decision}"
   List eligible banks and rejected banks by name.
   If decision is "Partial Profile": explain which fields are missing and which rules could be evaluated.

2. BANK-WISE OUTCOMES
   For each bank:
   [BANK NAME]: ELIGIBLE / NOT ELIGIBLE / PARTIAL PROFILE
   ✓ [passed rule with actual value]
   ✗ [failed rule: actual X, required Y — exact reason from rule data]
   ? [missing field name] — not provided, cannot evaluate

   CRITICAL: ? lines are MISSING DATA, not failures. Never convert them to ✗.

3. TOP FAILURE REASONS (only real failures — status="fail")
   Use decision_context.top_failures only. Format:
   • [field]: applicant has [actual], bank requires [required] [operator]

4. BEST BANK RECOMMENDATION
   State: decision_context.best_bank
   Reason: decision_context.best_bank_reason (copy exactly if provided)

5. IMPROVEMENT ACTIONS
   3-4 items. Each MUST have a numeric target. Example format:
   • Raise CIBIL from [actual] to [required] to qualify at [bank]
   • Increase monthly income from ₹[actual] to ₹[required] for [bank]
   • Provide [missing field] to complete eligibility check

═════════════════════════════════════════════════════════════════════════════"""

    def explain(
        self,
        query: str,
        rule_results: list[dict],
        retrieved_context: str,
        parsed_query: dict,
        pre_decision: str = "",
        decision_context: dict | None = None,
    ) -> dict[str, Any]:

        # Build concise per-bank failure strings for the prompt
        bank_summaries = []
        for r in rule_results:
            eligible_label = "ELIGIBLE" if r["eligible"] else "NOT ELIGIBLE"
            has_missing = bool(r.get("missing"))
            if has_missing and not r.get("failed"):
                eligible_label = "PARTIAL PROFILE (missing fields — cannot fully evaluate)"
            lines = [f"  {r['bank']}: {eligible_label} (score={r.get('rule_score',0):.0%})"]
            for f in r.get("failed", []):
                lines.append(f"    ✗ FAILED: {f}")
            for m in r.get("missing", []):
                lines.append(f"    ? MISSING (NOT a failure — just not provided): {m}")
            for p in r.get("passed", [])[:3]:
                lines.append(f"    ✓ PASSED: {p}")
            bank_summaries.append("\n".join(lines))

        user_prompt = f"""
FINAL DECISION (must appear verbatim in summary): {pre_decision}

USER QUERY: {query}

APPLICANT PROFILE (only these fields were provided — others are MISSING, not failed):
{json.dumps(parsed_query.get("entities",{}), indent=2)}

DECISION CONTEXT (use these exact values — do not infer):
{json.dumps(decision_context or {}, indent=2)}

HARD FAILURES (rules that actually FAILED due to provided data not meeting threshold):
{json.dumps((decision_context or {}).get("critical_failures", []), indent=2)}

PER-BANK RULE RESULTS:
IMPORTANT: "? MISSING" lines mean data was not provided — DO NOT treat them as failures.
Only "✗ FAILED" lines are actual eligibility failures.
{chr(10).join(bank_summaries)}

KNOWLEDGE CONTEXT (use for rate/policy info only):
{retrieved_context[:1200]}

Respond with JSON only — no markdown, no code blocks:
{{
  "summary": "State '{pre_decision}'. Name eligible banks and rejected banks. If Partial Profile, name which fields are missing.",
  "detailed_explanation": "Sections 1–4 from the strict format above, merged into clear prose. NEVER describe missing fields as failures.",
  "recommendations": [
    "3-4 items, each with a specific numeric target referencing actual vs required values, or asking user to provide specific missing field"
  ],
  "sources_cited": ["source file names from knowledge context"]
}}
"""
        raw = self._call(EXPLAIN_MODEL, self.EXPLAIN_SYSTEM, user_prompt, max_tokens=1200)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "summary": f"{pre_decision}.",
                "detailed_explanation": raw[:500],
                "recommendations": [],
                "sources_cited": [],
            }