"""
LLM Service  (v2)
==================
parse_query()  — small/fast model: extract intent + entities
explain()      — powerful model: generate explanation (never overrides rule engine)

Providers: groq | anthropic | openai
"""

from __future__ import annotations
import json
import re
from typing import Any

from config import (
    LLM_PROVIDER, PARSE_MODEL, EXPLAIN_MODEL,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY,
)


class ParsedQuery:
    def __init__(self, data: dict):
        self.intent: str        = data.get("intent", "general")
        self.banks: list[str]   = data.get("banks", [])
        self.entities: dict     = data.get("entities", {})
        self.raw: dict          = data

    @property
    def profile(self) -> dict:
        """
        Normalises entities → rule engine field names.
        KEY FIX: income parsing strips ₹ and commas before converting to float.
        """
        e = self.entities
        p = {}

        # monthly_income — strip all non-numeric chars before float conversion
        for key in ("monthly_income", "salary", "income"):
            raw = e.get(key)
            if raw is not None:
                try:
                    # Remove ₹, commas, spaces, then parse
                    clean = re.sub(r"[₹,\s]", "", str(raw))
                    val = float(clean)
                    # If suspiciously small (< 500), treat as lakhs pa → convert to monthly
                    if val < 500:
                        val = val * 100000 / 12
                    p["monthly_income"] = val
                except (ValueError, TypeError):
                    pass
                break

        if e.get("age") is not None:
            try:
                p["age"] = float(e["age"])
            except (ValueError, TypeError):
                pass

        for key in ("credit_score", "cibil", "cibil_score"):
            if e.get(key) is not None:
                try:
                    p["credit_score"] = float(e[key])
                except (ValueError, TypeError):
                    pass
                break

        if e.get("employment_type"):
            p["employment_type"] = str(e["employment_type"]).lower()

        for key in ("work_experience_months", "work_experience", "experience_months"):
            if e.get(key) is not None:
                try:
                    p["work_experience_months"] = float(e[key])
                except (ValueError, TypeError):
                    pass
                break

        if e.get("dti_ratio") is not None:
            try:
                p["dti_ratio"] = float(e["dti_ratio"])
            except (ValueError, TypeError):
                pass

        return p

    def __repr__(self):
        return f"ParsedQuery(intent={self.intent}, banks={self.banks}, entities={self.entities})"


class LLMService:

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
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
            response = client.messages.create(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return response.content[0].text
        else:
            response = client.chat.completions.create(
                model=model, max_tokens=max_tokens, temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
            )
            return response.choices[0].message.content

    # ── Parse ────────────────────────────────────────────────────────────────

    PARSE_SYSTEM = """You are a JSON-only parser for a loan advisory system.
Return ONLY valid JSON, no markdown, no preamble.

Schema:
{
  "intent": "eligibility" | "comparison" | "general",
  "banks": ["Axis" | "ICICI" | "HDFC" | "SBI"],
  "entities": {
    "age": number or null,
    "monthly_income": number or null,
    "credit_score": number or null,
    "employment_type": "salaried" | "self_employed" | "government" | "professional" | null,
    "work_experience_months": number or null,
    "loan_amount": number or null,
    "dti_ratio": number or null
  }
}

CRITICAL income rules:
- Return monthly_income as a plain integer with NO symbols or commas.
- "₹35,000/month" or "35,000 salary" → 35000
- "₹40,000" → 40000
- "5 LPA" or "5 lakhs per annum" → 41667
- "35k" → 35000
- NEVER include ₹ or commas in the number value.

Work experience: convert years to months. "2 years" → 24. "18 months" → 18.
Employment: "salaried for X years" → employment_type="salaried", work_experience_months=X*12
"""

    def parse_query(self, query: str) -> ParsedQuery:
        raw = self._call(PARSE_MODEL, self.PARSE_SYSTEM, query, max_tokens=512)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = {"intent": "general", "banks": [], "entities": {}}
        pq = ParsedQuery(data)
        print(f"[LLM] Parsed entities: {pq.entities}")
        print(f"[LLM] Normalised profile: {pq.profile}")
        return pq

    # ── Explain ──────────────────────────────────────────────────────────────

    EXPLAIN_SYSTEM = """You are a trustworthy loan policy advisor explaining a decision.

ABSOLUTE RULES:
1. The pre_decision field is the FINAL verdict from the rule engine. Use it EXACTLY.
   Never change it. Never say "Insufficient Data" if pre_decision says otherwise.
2. Explain WHY based only on the rule_results provided.
3. Never invent eligibility criteria.
4. Be specific: name the bank, name the failed rule, give the actual numbers.
5. Recommendations must be concrete and actionable.
"""

    def explain(
        self,
        query: str,
        rule_results: list[dict],
        retrieved_context: str,
        parsed_query: dict,
        pre_decision: str = "",
        decision_context: dict | None = None,
    ) -> dict[str, Any]:

        user_prompt = f"""
FINAL DECISION (use verbatim): {pre_decision}

USER QUERY: {query}

DECISION CONTEXT:
{json.dumps(decision_context or {}, indent=2)}

RULE ENGINE RESULTS:
{json.dumps(rule_results, indent=2)}

APPLICANT ENTITIES:
{json.dumps(parsed_query.get("entities", {}), indent=2)}

KNOWLEDGE CONTEXT:
{retrieved_context[:1800]}

Respond with JSON only:
{{
  "summary": "One sentence: state the decision and which banks passed/failed",
  "detailed_explanation": "2-3 sentences per bank: state what passed and what failed with actual values",
  "recommendations": ["3-4 specific actionable tips"],
  "sources_cited": ["source names from knowledge context"]
}}
"""

        raw = self._call(EXPLAIN_MODEL, self.EXPLAIN_SYSTEM, user_prompt, max_tokens=1500)
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "summary": f"{pre_decision}.",
                "detailed_explanation": raw,
                "recommendations": [],
                "sources_cited": [],
            }
