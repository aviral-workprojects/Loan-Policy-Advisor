"""
scraper/validator.py
=====================
Validates and sanitises LoanRecord objects before they are written to disk.

Rules:
  - Type checking on every field
  - Realistic range bounds for financial data
  - min < max consistency for all ranges
  - No negative values
  - Reject records with no useful content
  - Log every rejection with reason
"""

from __future__ import annotations
import hashlib
import logging
from dataclasses import fields

from scraper.schema import LoanRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation bounds
# ---------------------------------------------------------------------------

BOUNDS = {
    "interest_rate":   (0.5, 50.0),     # % p.a.
    "processing_fee":  (0, 1_00_000),   # ₹
    "processing_fee_pct": (0, 10),      # %
    "foreclosure_fee": (0, 10),         # %
    "min_income":      (5_000, 5_00_00_000),  # ₹/month
    "age_range":       (18, 80),        # years
    "loan_amount":     (1_000, 10_00_00_000), # ₹
    "tenure":          (1, 360),        # months
    "min_cibil":       (300, 900),
    "min_experience":  (0, 360),        # months
    "confidence":      (0.0, 1.0),
}

REQUIRED_FIELDS = {"bank", "source_url", "content"}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


def validate(record: LoanRecord) -> tuple[bool, list[str]]:
    """
    Validate a LoanRecord.

    Returns:
        (is_valid: bool, issues: list[str])

    A record is valid if it has no CRITICAL issues.
    WARNING-level issues are logged but do not cause rejection.
    """
    issues:   list[str] = []
    critical: list[str] = []

    # ── Required fields ────────────────────────────────────────────────────
    for fname in REQUIRED_FIELDS:
        val = getattr(record, fname, None)
        if not val:
            critical.append(f"Missing required field: {fname}")

    if record.content and len(record.content.strip()) < 50:
        critical.append(f"Content too short ({len(record.content)} chars) — likely failed extraction")

    # ── Type checks ────────────────────────────────────────────────────────
    if record.interest_rate is not None and not isinstance(record.interest_rate, list):
        issues.append("interest_rate must be a list")
        record.interest_rate = []

    if record.loan_amount is not None and not isinstance(record.loan_amount, list):
        issues.append("loan_amount must be a list")
        record.loan_amount = []

    if record.age_range is not None and not isinstance(record.age_range, list):
        issues.append("age_range must be a list")
        record.age_range = []

    if record.tenure is not None and not isinstance(record.tenure, list):
        issues.append("tenure must be a list")
        record.tenure = []

    # ── Range bounds ────────────────────────────────────────────────────────
    def check_float(val: float | None, field: str) -> bool:
        if val is None:
            return True
        lo, hi = BOUNDS[field]
        if not (lo <= val <= hi):
            issues.append(f"{field}={val} outside [{lo}, {hi}]")
            return False
        return True

    def check_list_range(lst: list, field: str) -> bool:
        if not lst:
            return True
        lo, hi = BOUNDS[field]
        for v in lst:
            if v is not None and not (lo <= v <= hi):
                issues.append(f"{field} value {v} outside [{lo}, {hi}]")
                return False
        return True

    check_float(record.processing_fee,     "processing_fee")
    check_float(record.processing_fee_pct, "processing_fee_pct")
    check_float(record.foreclosure_fee,    "foreclosure_fee")
    check_float(record.min_income,         "min_income")
    check_float(record.confidence,         "confidence")

    if record.min_cibil is not None:
        lo, hi = BOUNDS["min_cibil"]
        if not (lo <= record.min_cibil <= hi):
            issues.append(f"min_cibil={record.min_cibil} outside [{lo},{hi}]")
            record.min_cibil = None

    check_list_range(record.interest_rate, "interest_rate")
    check_list_range(record.loan_amount,   "loan_amount")
    check_list_range(record.age_range,     "age_range")
    check_list_range(record.tenure,        "tenure")

    # ── min < max consistency ────────────────────────────────────────────────
    for list_field in ("interest_rate", "loan_amount", "age_range", "tenure"):
        lst = getattr(record, list_field, [])
        if lst and len(lst) == 2 and lst[0] > lst[1]:
            issues.append(f"{list_field}: min ({lst[0]}) > max ({lst[1]}) — swapping")
            setattr(record, list_field, [lst[1], lst[0]])

    # ── No negatives ────────────────────────────────────────────────────────
    for list_field in ("interest_rate", "loan_amount", "tenure"):
        lst = getattr(record, list_field, [])
        if any(v is not None and v < 0 for v in lst):
            issues.append(f"Negative value in {list_field}")
            setattr(record, list_field, [v for v in lst if v is None or v >= 0])

    # ── Content hash (deduplication key) ────────────────────────────────────
    if record.content and not record.content_hash:
        record.content_hash = hashlib.sha256(
            record.content.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]

    # ── Bank normalisation ────────────────────────────────────────────────
    bank_map = {
        "axis bank": "Axis", "axisbank": "Axis",
        "hdfc bank": "HDFC", "hdfcbank": "HDFC",
        "icici bank": "ICICI", "icicidirect": "ICICI",
        "state bank of india": "SBI", "sbi": "SBI",
        "paisabazaar": "Paisabazaar",
        "bankbazaar": "BankBazaar",
        "rbi": "RBI", "reserve bank": "RBI",
    }
    bl = record.bank.lower()
    for key, canonical in bank_map.items():
        if key in bl:
            record.bank = canonical
            break

    # ── Log ──────────────────────────────────────────────────────────────────
    all_issues = critical + issues
    if critical:
        for c in critical:
            logger.warning("[Validator] REJECT [%s] %s | %s", record.bank, c, record.source_url)
        return False, all_issues

    if issues:
        for w in issues:
            logger.debug("[Validator] WARN [%s] %s", record.bank, w)

    return True, all_issues


# ---------------------------------------------------------------------------
# Deduplication store
# ---------------------------------------------------------------------------

class DeduplicationStore:
    """
    Tracks content hashes and source URLs to prevent duplicate storage.
    Persists to a simple text file so dedup survives restarts.
    """

    def __init__(self, cache_file=None):
        from pathlib import Path
        self._hashes:    set[str] = set()
        self._urls:      set[str] = set()
        self._cache_file = Path(cache_file) if cache_file else None
        if self._cache_file and self._cache_file.exists():
            self._load()

    def is_duplicate(self, record: LoanRecord) -> bool:
        if record.content_hash and record.content_hash in self._hashes:
            logger.debug("[Dedup] Duplicate hash: %s", record.content_hash)
            return True
        if record.source_url and record.source_url in self._urls:
            logger.debug("[Dedup] Duplicate URL: %s", record.source_url)
            return True
        return False

    def register(self, record: LoanRecord) -> None:
        if record.content_hash:
            self._hashes.add(record.content_hash)
        if record.source_url:
            self._urls.add(record.source_url)
        if self._cache_file:
            self._save()

    def _load(self):
        try:
            for line in self._cache_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("h:"):
                    self._hashes.add(line[2:])
                elif line.startswith("u:"):
                    self._urls.add(line[2:])
        except Exception:
            pass

    def _save(self):
        try:
            lines = [f"h:{h}" for h in self._hashes] + [f"u:{u}" for u in self._urls]
            self._cache_file.write_text("\n".join(lines))
        except Exception:
            pass
