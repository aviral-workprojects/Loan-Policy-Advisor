"""
scraper/normalizer.py
======================
Converts raw extracted strings into clean, typed numeric values.

Every field that touches money, rates, or dates goes through here before
validation. The goal is "one place to fix" when a bank changes their
formatting — not scattered regexes everywhere.
"""

from __future__ import annotations
import re
from typing import Any


# ---------------------------------------------------------------------------
# Currency → float (₹)
# ---------------------------------------------------------------------------

def parse_inr(text: str) -> float | None:
    """
    Parse an Indian Rupee string into a float (base unit: rupees).

    Examples:
      "₹10 lakh"      → 1_000_000.0
      "₹10,00,000"    → 1_000_000.0
      "40000"         → 40_000.0
      "40k"           → 40_000.0
      "₹40,000/month" → 40_000.0
      "50 thousand"   → 50_000.0
      "1.5 crore"     → 1_500_000.0
    """
    if not text:
        return None
    text = str(text).lower().strip()
    text = re.sub(r"[₹,\s/month/year]", " ", text).strip()

    # Crore
    m = re.search(r"([\d.]+)\s*cr(?:ore)?", text)
    if m:
        return float(m.group(1)) * 1_00_00_000

    # Lakh
    m = re.search(r"([\d.]+)\s*la(?:kh|c|cs)?", text)
    if m:
        return float(m.group(1)) * 1_00_000

    # Thousand / k
    m = re.search(r"([\d.]+)\s*(?:thousand|k)\b", text)
    if m:
        return float(m.group(1)) * 1_000

    # Plain number (strip commas from Indian formatting)
    m = re.search(r"[\d,]+(?:\.\d+)?", text.replace(" ", ""))
    if m:
        try:
            return float(m.group().replace(",", ""))
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Percentage → float
# ---------------------------------------------------------------------------

def parse_percent(text: str) -> float | None:
    """
    Parse a percentage string into a float (0–100 scale, NOT 0–1).

    Examples:
      "10.49% p.a."  → 10.49
      "8.75 %"       → 8.75
      "10-15%"       → None  (ambiguous range — use parse_rate_range instead)
    """
    if not text:
        return None
    text = str(text).strip()
    m = re.search(r"([\d]+(?:\.\d+)?)\s*%", text)
    if m:
        val = float(m.group(1))
        if 0 < val < 100:
            return val
    return None


def parse_rate_range(text: str) -> list[float]:
    """
    Parse a rate range string into [min, max] floats.

    Examples:
      "10.49% – 24%"  → [10.49, 24.0]
      "10.5 to 21%"   → [10.5, 21.0]
      "Starting 10.5%"→ [10.5, 10.5]
    """
    if not text:
        return []
    text = str(text)

    # Explicit range
    m = re.search(
        r"([\d]+(?:\.\d+)?)\s*%?\s*(?:to|–|—|-|upto|up\s*to)\s*([\d]+(?:\.\d+)?)\s*%",
        text, re.IGNORECASE,
    )
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        if 0 < lo < 100 and 0 < hi < 100:
            return [lo, hi]

    # Single value
    all_pcts = [float(x) for x in re.findall(r"[\d]+(?:\.\d+)?(?=\s*%)", text)]
    valid = [x for x in all_pcts if 0 < x < 100]
    if len(valid) >= 2:
        return [min(valid), max(valid)]
    if len(valid) == 1:
        return [valid[0], valid[0]]

    return []


# ---------------------------------------------------------------------------
# Tenure → months
# ---------------------------------------------------------------------------

def parse_tenure_months(text: str) -> int | None:
    """
    Convert tenure text to months.

    Examples:
      "5 years"    → 60
      "84 months"  → 84
      "7 yr"       → 84
    """
    if not text:
        return None
    text = str(text).lower().strip()

    m = re.search(r"([\d]+(?:\.\d+)?)\s*(?:year|yr)", text)
    if m:
        return int(float(m.group(1)) * 12)

    m = re.search(r"([\d]+)\s*(?:month|mo\b)", text)
    if m:
        return int(m.group(1))

    m = re.search(r"^\s*(\d+)\s*$", text)
    if m:
        val = int(m.group(1))
        if val <= 30:
            return val * 12   # assume years
        return val            # assume months

    return None


def parse_tenure_range(text: str) -> list[int]:
    """
    Parse a tenure range into [min_months, max_months].

    Examples:
      "1 to 5 years" → [12, 60]
      "12–84 months" → [12, 84]
    """
    if not text:
        return []
    text = str(text).lower()

    parts = re.split(r"\s*(?:to|–|—|-)\s*", text)
    if len(parts) == 2:
        lo = parse_tenure_months(parts[0])
        hi = parse_tenure_months(parts[1])
        if lo and hi:
            if lo > hi:
                lo, hi = hi, lo
            return [lo, hi]

    single = parse_tenure_months(text)
    if single:
        return [single, single]

    return []


# ---------------------------------------------------------------------------
# Age range
# ---------------------------------------------------------------------------

def parse_age_range(text: str) -> list[int]:
    """
    Parse age requirement text into [min, max].

    Examples:
      "21 to 60 years"  → [21, 60]
      "Minimum 21 years"→ [21, 65]   (65 = default max)
      "23-58"           → [23, 58]
    """
    if not text:
        return []
    text = str(text)

    m = re.search(r"(\d+)\s*(?:to|–|—|-)\s*(\d+)", text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if 18 <= lo <= hi <= 80:
            return [lo, hi]

    m = re.search(r"(?:min(?:imum)?|at\s*least)\s*(\d+)", text, re.I)
    if m:
        lo = int(m.group(1))
        if 18 <= lo <= 40:
            return [lo, 65]

    ages = [int(x) for x in re.findall(r"\b(\d{2})\b", text) if 18 <= int(x) <= 80]
    if len(ages) >= 2:
        return [min(ages), max(ages)]

    return []


# ---------------------------------------------------------------------------
# Income → monthly ₹
# ---------------------------------------------------------------------------

def parse_monthly_income(text: str) -> float | None:
    """
    Parse income text into monthly rupees.

    Examples:
      "₹25,000 per month" → 25000.0
      "3 LPA"             → 25000.0
      "₹30k monthly"      → 30000.0
      "4.5 lakh per annum"→ 37500.0
    """
    if not text:
        return None
    text_lower = str(text).lower()

    # LPA (lakhs per annum)
    m = re.search(r"([\d.]+)\s*lpa\b", text_lower)
    if m:
        return round(float(m.group(1)) * 1_00_000 / 12, 2)

    # per annum in lakhs
    m = re.search(r"([\d.]+)\s*lakh.*?(?:per\s*annum|pa\b|annually)", text_lower)
    if m:
        return round(float(m.group(1)) * 1_00_000 / 12, 2)

    # explicit monthly
    m = re.search(r"(?:₹|rs\.?)?\s*([\d,]+(?:\.\d+)?)\s*(?:k\b|thousand)?\s*(?:per\s*month|monthly|pm\b|/month)", text_lower)
    if m:
        raw = m.group(1).replace(",", "")
        val = float(raw)
        if "k" in text_lower[m.start():m.end()] or "thousand" in text_lower[m.start():m.end()]:
            val *= 1000
        return val

    # bare INR amount (assume monthly if small, annual if large)
    val = parse_inr(text)
    if val:
        if val > 5_00_000:      # likely annual
            return round(val / 12, 2)
        if val >= 5_000:        # likely monthly
            return val

    return None


# ---------------------------------------------------------------------------
# CIBIL / credit score
# ---------------------------------------------------------------------------

def parse_cibil(text: str) -> int | None:
    """Extract a CIBIL score (300–900) from text."""
    m = re.search(r"\b([4-9]\d{2})\b", str(text))
    if m:
        score = int(m.group(1))
        if 300 <= score <= 900:
            return score
    return None


# ---------------------------------------------------------------------------
# Employment type normalisation
# ---------------------------------------------------------------------------

_EMP_MAP = {
    "salaried":          "salaried",
    "salary":            "salaried",
    "employed":          "salaried",
    "self.employ":       "self_employed",
    "self employ":       "self_employed",
    "business":          "self_employed",
    "professional":      "professional",
    "doctor":            "professional",
    "ca\b":              "professional",
    "government":        "government",
    "govt":              "government",
    "psu":               "government",
    "public sector":     "government",
    "nri":               "nri",
}

def parse_employment_types(text: str) -> list[str]:
    """Extract employment type mentions and normalise to canonical values."""
    tl = text.lower()
    found = []
    for pattern, canonical in _EMP_MAP.items():
        if re.search(pattern, tl) and canonical not in found:
            found.append(canonical)
    return found
