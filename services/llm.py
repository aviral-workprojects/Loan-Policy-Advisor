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
a decision that has already been made by a deterministic rule engine.

ABSOLUTE CONSTRAINTS — violating any of these is a critical failure:
1. The "pre_decision" value IS the final decision. Copy it verbatim to your output.
   You CANNOT change it. You CANNOT say "Insufficient Data" if it says "Eligible".
2. Use ONLY numbers and facts present in rule_results and decision_context.
   DO NOT invent income figures, interest rates, or thresholds not shown.
3. For each bank, cite the EXACT rule that failed using the "reason" field provided.
4. Recommendations must reference specific fields the applicant can improve.
5. Never say "I think" or "it appears" — state facts from the data only."""

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
            lines = [f"  {r['bank']}: {'ELIGIBLE' if r['eligible'] else 'NOT ELIGIBLE'} (score={r.get('rule_score',0):.0%})"]
            for f in r.get("failed", []):
                lines.append(f"    ✗ {f}")
            for m in r.get("missing", []):
                lines.append(f"    ? {m} [not provided]")
            for p in r.get("passed", [])[:2]:
                lines.append(f"    ✓ {p}")
            bank_summaries.append("\n".join(lines))

        user_prompt = f"""
FINAL DECISION (copy verbatim to output): {pre_decision}

USER QUERY: {query}

APPLICANT PROFILE:
{json.dumps(parsed_query.get("entities",{}), indent=2)}

DECISION CONTEXT:
{json.dumps(decision_context or {}, indent=2)}

PER-BANK RULE RESULTS:
{chr(10).join(bank_summaries)}

KNOWLEDGE CONTEXT (use for rate/policy info only):
{retrieved_context[:1200]}

Respond with JSON only — no markdown:
{{
  "summary": "One sentence: state '{pre_decision}' and name eligible/rejected banks",
  "detailed_explanation": "Per-bank breakdown using exact rule failure reasons from the data",
  "recommendations": ["3-4 specific steps referencing actual fields and thresholds"],
  "sources_cited": ["source names from knowledge context"]
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
