"""
pdf_pipeline/query_understanding.py
=====================================
Query Understanding Layer — classifies intent, extracts structured signals,
and reformulates queries for optimal retrieval.

This is one of the highest-leverage modules in the system. A well-understood
query drives better routing, better retrieval filters, and better rule engine
lookups — without needing a better LLM.

Design: rule-based first (fast, no API cost), LLM-assisted for complex cases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hinglish / informal query normalisation
# ---------------------------------------------------------------------------

# Lightweight token-level replacements applied BEFORE any regex extraction.
# Goal: make informal/Hinglish queries parseable — not produce perfect English.
_HINGLISH_TOKENS: dict[str, str] = {
    r"\bmil\s+jayega\b":       "will i get",
    r"\bmil\s+sakta\s+hai\b":  "can i get",
    r"\bmil\s+sakta\b":        "can i get",
    r"\bkya\s+loan\b":         "can i get loan",
    r"\bloan\s+milega\b":      "will i get loan",
    r"\bkya\b":                "",
    r"\bpe\b":                 "for",
    r"\bke\s+liye\b":          "for",
    r"\bjaldi\b":              "fast urgent",
    r"\bkaun\s*sa\b":          "which",
    r"\bkitna\b":              "how much",
    r"\bsaal\b":               "years",
    r"\brupaye\b":             "rupees",
    r"\bnahi\b":               "no",
    r"\bhoga\b":               "will be",
    r"\bbro\b":                "",
    r"\byaar\b":               "",
    r"\bdude\b":               "",
}


def _normalize_query(query: str) -> str:
    """
    Normalise informal / Hinglish queries before rule-based signal extraction.

    Only light normalisation — produces enough English for the regex patterns
    to fire.  The *original* query is preserved for the LLM explain step;
    this output is only used for intent classification and entity extraction.
    """
    q = query.strip()
    for pattern, replacement in _HINGLISH_TOKENS.items():
        q = re.sub(pattern, replacement, q, flags=re.I)
    return re.sub(r"\s{2,}", " ", q).strip()


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class QuerySignals:
    intent: str                     # "eligibility" | "interest_rate" | "comparison" | "general" | "fees"
    banks: list[str]                # detected bank names
    loan_type: str                  # "personal" | "home" | "car" | "business" | ""
    entities: dict[str, Any]        # raw extracted entities
    profile: dict[str, Any]         # normalised applicant profile
    reformulated_query: str         # cleaned/expanded query for retrieval
    confidence: float               # 0–1 confidence in intent classification

    def __repr__(self) -> str:
        return (
            f"QuerySignals(intent={self.intent!r}, banks={self.banks}, "
            f"loan_type={self.loan_type!r}, profile={self.profile})"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_BANKS = {
    "axis":    "Axis",
    "hdfc":    "HDFC",
    "icici":   "ICICI",
    "sbi":     "SBI",
    "state bank": "SBI",
}

LOAN_TYPES = {
    "personal loan": "personal",
    "home loan":     "home",
    "housing loan":  "home",
    "car loan":      "car",
    "auto loan":     "car",
    "business loan": "business",
    "education loan":"education",
    "gold loan":     "gold",
}

_INTENT_PATTERNS = {
    "eligibility": re.compile(
        r"\beligib|\bqualif|\bcan i\b|\bam i\b|\bwill i\b|"
        r"\bdo i qualify|\bapply\b|\bget a loan\b|\bget approved\b|\bapproved\b|"
        r"\bneed.*loan\b|\bwant.*loan\b|\bget.*loan\b|"
        r"\bwhat to do\b|\burgent.*loan\b|\bloan.*urgent\b",
        re.I,
    ),
    "interest_rate": re.compile(
        r"\binterest rate|\broi\b|\brate of interest|\b% p\.?a|\bemi\b|\binstalment\b|"
        r"\bminimum (?:salary|income|cibil|age|score)\b|"
        r"\bwhat (?:is|are) (?:the )?(?:minimum|maximum|requirement)",
        re.I,
    ),
    "comparison": re.compile(
        r"\bcompare|\bvs\.?\b|\bbetter\b|\bbest\b|\bwhich bank|\bhighest|\blowest\b|"
        r"\bfastest\b|\bsafest\b|\bcheapest\b|\beasiest\b|\bquickest\b",
        re.I,
    ),
    "fees": re.compile(
        r"\bfee|\bcharge|\bpenalty|\bprocessing fee|\bforeclos|\bprepay",
        re.I,
    ),
    "document": re.compile(
        r"\bdocument|\bkyc\b|\bpapers?\b|\bsubmit\b|\brequired.*doc|\bdoc.*required",
        re.I,
    ),
}

_INCOME_PATTERNS = [
    # "35000", "35,000", "₹35000", "35k", "35 thousand"
    (re.compile(r"₹?\s*(\d[\d,]*)\s*(?:per\s*month|/month|pm\b|monthly)", re.I), 1.0),
    (re.compile(r"₹?\s*(\d+)\s*k\b", re.I), 1000.0),
    (re.compile(r"(\d+(?:\.\d+)?)\s*lpa\b", re.I), None),  # LPA handled specially
    (re.compile(r"salary\s+(?:of\s+)?₹?\s*(\d[\d,]*)", re.I), 1.0),
    (re.compile(r"earn(?:ing|s)?\s+₹?\s*(\d[\d,]*)", re.I), 1.0),
    (re.compile(r"income\s+(?:of\s+)?₹?\s*(\d[\d,]*)", re.I), 1.0),
]

_CIBIL_PATTERN  = re.compile(r"\b(cibil|credit\s*score)\s*(?:of|is|:|=)?\s*(\d{3})\b", re.I)
_AGE_PATTERN    = re.compile(
    r"\b(\d{2})\s*years?\s*(?:old|of\s*age)?\b|"         # "28 years old", "28 years"
    r"(?:age|aged?)\s*(?:of\s*|is\s*|:)?\s*(\d{2})\b|"  # "age 23", "aged 30"
    r"(?:\bam|is)\s+(\d{2})\b",                           # "I am 28"
    re.I,
)
_TENURE_PATTERN = re.compile(r"(\d+)\s*years?\s*(?:of\s*)?(?:experience|service|work)", re.I)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def understand_query(query: str) -> QuerySignals:
    """
    Parse a natural language loan query into structured signals.

    Rule-based extraction is fast and requires no API call.
    Returns QuerySignals with intent, entities, normalised profile,
    and a reformulated query optimised for retrieval.
    """
    query = query.strip()
    # Normalise informal / Hinglish text before running any pattern matching.
    # The original query is preserved; `normalised` is used for extraction only.
    normalised = _normalize_query(query)

    banks        = _detect_banks(normalised)
    loan_type    = _detect_loan_type(normalised)
    intent, conf = _classify_intent(normalised, banks)
    entities     = _extract_entities(normalised)
    profile      = _normalise_profile(entities)
    reformulated = _reformulate_query(normalised, intent, banks, loan_type, profile)

    signals = QuerySignals(
        intent=intent,
        banks=banks,
        loan_type=loan_type,
        entities=entities,
        profile=profile,
        reformulated_query=reformulated,
        confidence=conf,
    )
    logger.info("[QueryUnderstanding] %s", signals)
    return signals


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _classify_intent(query: str, banks: list[str]) -> tuple[str, float]:
    """
    Classify query intent using pattern matching.
    Returns (intent_label, confidence_0_to_1).
    """
    scores: dict[str, int] = {}
    for intent, pattern in _INTENT_PATTERNS.items():
        if pattern.search(query):
            scores[intent] = scores.get(intent, 0) + 2

    ql = query.lower()

    # "Factual question" flag: narrow to true information-seeking patterns.
    # "need" is intentionally excluded — "I need a loan" is eligibility, not factual.
    is_factual_question = bool(re.search(
        r"\bwhat\s+(?:is|are|documents?)\b|\bhow\s+much\b|"
        r"\bminimum\b|\bmaximum\b|\brequired\b|\bwhat\s+do\s+i\s+need\b",
        ql,
    ))
    has_profile_data = any(w in ql for w in ["salary", "cibil", "income", "age", "score", "earn"])
    if has_profile_data and not is_factual_question:
        scores["eligibility"] = scores.get("eligibility", 0) + 1

    # Strong eligibility signal: user expresses a personal need / desire for a loan
    if re.search(r"\bneed.*loan\b|\bwant.*loan\b|\bget.*loan\b", ql):
        scores["eligibility"] = scores.get("eligibility", 0) + 2

    # Boost comparison if multiple banks mentioned
    if len(banks) > 1:
        scores["comparison"] = scores.get("comparison", 0) + 2

    if not scores:
        return "general", 0.5

    best = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = round(scores[best] / total, 2) if total > 0 else 0.5
    return best, confidence


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def _detect_banks(query: str) -> list[str]:
    ql = query.lower()
    found = []
    for keyword, bank_name in KNOWN_BANKS.items():
        if keyword in ql and bank_name not in found:
            found.append(bank_name)
    return found


def _detect_loan_type(query: str) -> str:
    ql = query.lower()
    for phrase, loan_type in LOAN_TYPES.items():
        if phrase in ql:
            return loan_type
    # Default: if no loan type mentioned, assume personal loan (most common)
    return "personal"


def _extract_entities(query: str) -> dict[str, Any]:
    entities: dict[str, Any] = {}

    # Income extraction
    for pattern, multiplier in _INCOME_PATTERNS:
        m = pattern.search(query)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
                if multiplier is None:
                    # LPA — convert to monthly
                    val = val * 100_000 / 12
                else:
                    val *= multiplier
                # Sanity check: monthly income should be > 5000 and < 10 crore
                if 5_000 < val < 10_00_00_000:
                    entities["monthly_income"] = round(val, 2)
                    break
            except ValueError:
                pass

    # CIBIL / credit score
    m = _CIBIL_PATTERN.search(query)
    if m:
        try:
            score = int(m.group(2))
            if 300 <= score <= 900:
                entities["credit_score"] = score
        except ValueError:
            pass
    # Also look for bare 3-digit number that looks like a credit score
    else:
        m = re.search(r"\b([4-9]\d{2})\b", query)
        if m and "cibil" in query.lower() or "score" in query.lower():
            try:
                score = int(m.group(1))
                if 300 <= score <= 900:
                    entities["credit_score"] = score
            except ValueError:
                pass

    # Age — extended pattern: "28 years old" | "age 23" | "I am 28"
    m = _AGE_PATTERN.search(query)
    if m:
        try:
            age = int(next(g for g in m.groups() if g is not None))
            if 18 <= age <= 75:
                entities["age"] = age
        except (ValueError, StopIteration):
            pass

    # Work experience
    m = _TENURE_PATTERN.search(query)
    if m:
        try:
            years = int(m.group(1))
            if 0 < years < 50:
                entities["work_experience_months"] = years * 12
        except ValueError:
            pass

    # Employment type — order matters: check negative/unemployed FIRST to avoid
    # "\bjob\b" matching "no job" and falsely returning "salaried".
    ql = query.lower()
    if re.search(r"\bno\s+job\b|\bunemployed\b|\bjobless\b|\bno\s+(?:fixed\s+)?income\b", ql):
        entities["employment_type"] = "unemployed"
    elif re.search(r"\bself.?emp|\bfreelance[rd]?\b|\bfreelancing\b|\bgig\s*worker\b|"
                   r"\bconsultant\b|\bown\s+business\b|\bproprietor\b|\bbusiness\s+owner\b", ql):
        entities["employment_type"] = "self_employed"
    elif re.search(r"\bgovernment\b|\bpsu\b|\bpublic\s+sector\b|\bdefence\b|\bdefense\b", ql):
        entities["employment_type"] = "government"
    elif re.search(r"\bprofessional\b|\bdoctor\b|\bca\b|\bchartered\b|\blawyer\b|\barchitect\b", ql):
        entities["employment_type"] = "professional"
    elif re.search(r"\bsalaried\b|\bemployed\b|\bservice\b|\bworking\b", ql):
        entities["employment_type"] = "salaried"
    # Note: bare "\bjob\b" intentionally removed — too ambiguous with "no job"

    # DTI ratio
    m = re.search(r"\bdti\s*(?:of|is|=)?\s*(\d+)%?", query, re.I)
    if m:
        try:
            dti = float(m.group(1))
            if dti > 1.5:
                dti /= 100.0   # convert percentage to decimal
            if 0 < dti < 1:
                entities["dti_ratio"] = round(dti, 3)
        except ValueError:
            pass

    return entities


def _normalise_profile(entities: dict[str, Any]) -> dict[str, Any]:
    """Clean and validate the extracted entities into a profile dict."""
    profile = {}
    for key in ("monthly_income", "credit_score", "age",
                "work_experience_months", "dti_ratio"):
        if entities.get(key) is not None:
            profile[key] = entities[key]
    if entities.get("employment_type"):
        profile["employment_type"] = entities["employment_type"]
    return profile


# ---------------------------------------------------------------------------
# Query reformulation
# ---------------------------------------------------------------------------

def _reformulate_query(
    original: str,
    intent:   str,
    banks:    list[str],
    loan_type: str,
    profile:  dict[str, Any],
) -> str:
    """
    Build an expanded query string that improves retrieval recall.

    We append structured signal keywords so the embedding model and BM25
    both benefit from explicit terms the user may not have used.
    """
    parts = [original.strip()]

    if intent == "eligibility":
        parts.append("eligibility criteria requirements")

    if loan_type and loan_type != "personal":
        parts.append(f"{loan_type} loan")

    if banks:
        parts.append(" ".join(banks))

    # Add profile signals as terms — helps BM25 exact-match
    if profile.get("monthly_income"):
        parts.append(f"salary income {profile['monthly_income']:.0f}")
    if profile.get("credit_score"):
        parts.append(f"CIBIL credit score {profile['credit_score']}")
    if profile.get("employment_type"):
        parts.append(profile["employment_type"])

    # Deduplicate tokens while preserving order
    seen_tokens: set[str] = set()
    result_tokens: list[str] = []
    for part in parts:
        for token in part.split():
            tl = token.lower()
            if tl not in seen_tokens:
                seen_tokens.add(tl)
                result_tokens.append(token)

    return " ".join(result_tokens)