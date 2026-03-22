"""
document_pipeline/extractor.py
================================
Entity extraction from OCR/text output.

Extracts structured financial signals from raw text using:
  1. Regex patterns (primary — fast, interpretable, no API cost)
  2. Heuristic NLP (context-aware scoring)
  3. Optional LLM extraction (fallback for complex layouts)

Fields extracted:
  name, age, monthly_income, employment_type, credit_score,
  employer_name, pan_number, loan_amount_requested, dti_ratio,
  bank_statement_balance, account_number (masked)

Design:
  All patterns are tuned for Indian financial documents:
  salary slips, bank statements, ITR, Form 16, CIBIL reports.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ExtractedProfile dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExtractedProfile:
    """Structured data extracted from an uploaded document."""

    # Core eligibility fields (directly fed to rule engine)
    monthly_income:          float | None = None
    annual_income:           float | None = None
    credit_score:            int   | None = None
    age:                     int   | None = None
    employment_type:         str   | None = None   # "salaried"|"self_employed"|"government"|"professional"
    work_experience_months:  int   | None = None
    dti_ratio:               float | None = None

    # Contextual fields (used for LLM context and recommendations)
    name:                    str   | None = None
    employer_name:           str   | None = None
    pan_number:              str   | None = None   # masked: ABCDE1234F → ABCDE****F
    loan_amount_requested:   float | None = None
    bank_balance:            float | None = None
    account_number:          str   | None = None   # masked
    document_type:           str   = ""            # "salary_slip"|"bank_statement"|"itr"|"cibil_report"|"unknown"

    # Extraction metadata
    extraction_confidence:   float = 0.0
    raw_fields_found:        list[str] = field(default_factory=list)

    def to_profile(self) -> dict[str, Any]:
        """Return only the fields the rule engine uses."""
        profile = {}
        if self.monthly_income is not None:
            profile["monthly_income"] = self.monthly_income
        if self.credit_score is not None:
            profile["credit_score"] = self.credit_score
        if self.age is not None:
            profile["age"] = self.age
        if self.employment_type is not None:
            profile["employment_type"] = self.employment_type
        if self.work_experience_months is not None:
            profile["work_experience_months"] = self.work_experience_months
        if self.dti_ratio is not None:
            profile["dti_ratio"] = self.dti_ratio
        return profile

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:

    def extract(self, raw_text: str, tables: list[dict] | None = None) -> ExtractedProfile:
        """
        Extract all financial entities from raw text + tables.

        Args:
            raw_text: combined text from all pages
            tables:   list of table dicts with 'raw_text' key

        Returns:
            ExtractedProfile with all found fields populated
        """
        # Merge table natural language into text for unified processing
        combined = raw_text
        if tables:
            table_nl = "\n".join(t.get("raw_text", "") for t in tables if t.get("raw_text"))
            if table_nl:
                combined = combined + "\n" + table_nl

        profile = ExtractedProfile()
        profile.document_type = self._detect_doc_type(combined)
        logger.info("[Extractor] Document type detected: %s", profile.document_type)

        # Run all extractors
        self._extract_income(combined, profile)
        self._extract_cibil(combined, profile)
        self._extract_age(combined, profile)
        self._extract_employment(combined, profile)
        self._extract_experience(combined, profile)
        self._extract_name(combined, profile)
        self._extract_employer(combined, profile)
        self._extract_pan(combined, profile)
        self._extract_loan_amount(combined, profile)
        self._extract_bank_balance(combined, profile)
        self._extract_dti(combined, profile)

        # Compute confidence
        core_fields = ["monthly_income", "credit_score", "age", "employment_type"]
        found_core = sum(1 for f in core_fields if getattr(profile, f) is not None)
        profile.extraction_confidence = round(found_core / len(core_fields), 2)

        logger.info(
            "[Extractor] Found: income=%s  CIBIL=%s  age=%s  emp=%s  confidence=%.0f%%",
            profile.monthly_income, profile.credit_score,
            profile.age, profile.employment_type,
            profile.extraction_confidence * 100,
        )
        return profile

    # ── Document type detection ──────────────────────────────────────────────

    def _detect_doc_type(self, text: str) -> str:
        tl = text.lower()
        if re.search(r"cibil\s*score|credit\s*score|credit\s*report|equifax|experian", tl):
            return "cibil_report"
        if re.search(r"salary\s*slip|pay\s*slip|payslip|pay\s*stub|gross\s*salary|net\s*salary", tl):
            return "salary_slip"
        if re.search(r"bank\s*statement|account\s*statement|transaction|balance\s*forward", tl):
            return "bank_statement"
        if re.search(r"income\s*tax\s*return|itr|form\s*16|assessment\s*year|gross\s*total\s*income", tl):
            return "itr"
        if re.search(r"form\s*26as|tds\s*certificate", tl):
            return "form_26as"
        return "unknown"

    # ── Income extraction ────────────────────────────────────────────────────

    def _extract_income(self, text: str, profile: ExtractedProfile) -> None:
        """
        Extract monthly income. Handles salary slips (net pay), bank statements
        (credited salary), and ITR (annual income ÷ 12).
        """
        # Pattern priority order: most specific first
        patterns = [
            # Salary slip: net pay / take home
            (r"(?:net\s*(?:pay|salary|take.?home|amount\s*payable))\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)", 1.0),
            # Salary credited / salary received
            (r"salary\s*(?:credited|received|paid|transferred)\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)", 1.0),
            # Gross salary/pay
            (r"(?:gross\s*(?:pay|salary|earnings?))\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)", 0.85),
            # Monthly salary / income explicit
            (r"(?:monthly\s*(?:salary|income|earnings?))\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)", 1.0),
            # CTC (annual — divide by 12)
            (r"(?:ctc|cost\s*to\s*company|annual\s*(?:ctc|salary|income|package))\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)", 0.7, True),
            # LPA
            (r"([\d.]+)\s*lpa\b", 0.8, True),
            # Total income (ITR)
            (r"(?:total\s*income|gross\s*total\s*income)\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)", 0.7, True),
        ]

        best_income = None
        best_confidence = 0.0

        for item in patterns:
            pat, conf = item[0], item[1]
            is_annual = len(item) > 2 and item[2]

            for m in re.finditer(pat, text, re.I):
                raw = m.group(1).replace(",", "")
                try:
                    val = float(raw)
                except ValueError:
                    continue

                # LPA special case
                if "lpa" in pat:
                    val = val * 1_00_000 / 12

                # Annual → monthly
                if is_annual and "lpa" not in pat:
                    if val > 5_00_000:    # > 5L, clearly annual
                        val = val / 12
                    elif val > 50_000:    # ambiguous — keep as is (likely monthly)
                        pass

                # Sanity bounds: ₹3,000–₹1,00,00,000 monthly
                if not (3_000 <= val <= 1_00_00_000):
                    continue

                if conf > best_confidence:
                    best_confidence = conf
                    best_income = round(val, 2)
                    profile.raw_fields_found.append("monthly_income")

        if best_income:
            profile.monthly_income = best_income
            # Also store annual
            profile.annual_income = round(best_income * 12, 2)

    # ── CIBIL score ──────────────────────────────────────────────────────────

    def _extract_cibil(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:cibil|credit|equifax|experian)\s*score\s*[:\-]?\s*(\d{3})",
            r"score\s*[:\-]?\s*(\d{3})\s*(?:/\s*900)?",
            r"\b([7-9]\d{2})\b.*?(?:cibil|credit|score)",
            r"(?:cibil|credit|score)[^\n]{0,30}?(\d{3})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    score = int(m.group(1))
                    if 300 <= score <= 900:
                        profile.credit_score = score
                        profile.raw_fields_found.append("credit_score")
                        return
                except ValueError:
                    pass

    # ── Age ──────────────────────────────────────────────────────────────────

    def _extract_age(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:age|date\s*of\s*birth)[^\n]{0,20}?(\d{2})\s*years?",
            r"(\d{2})\s*years?\s*(?:old|of\s*age)",
            # DOB → compute age
            r"(?:dob|date\s*of\s*birth)\s*[:\-]?\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    if len(m.groups()) == 3:
                        # DOB pattern — compute age
                        import datetime
                        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        if year < 100:
                            year += 1900 if year > 30 else 2000
                        dob = datetime.date(year, month, day)
                        today = datetime.date.today()
                        age = (today - dob).days // 365
                    else:
                        age = int(m.group(1))
                    if 18 <= age <= 80:
                        profile.age = age
                        profile.raw_fields_found.append("age")
                        return
                except (ValueError, OverflowError):
                    pass

    # ── Employment type ───────────────────────────────────────────────────────

    def _extract_employment(self, text: str, profile: ExtractedProfile) -> None:
        tl = text.lower()
        # Government / PSU
        if re.search(r"\bgovernment\b|\bpsu\b|\bpublic\s*sector|\bcivil\s*servant|\bcentral\s*govt", tl):
            profile.employment_type = "government"
        # Self-employed / business
        elif re.search(r"self.?employ|\bpropriet|\bpartner\b|\bpartnership|\bllp\b|\bpvt\s*ltd\b|\bprivate\s*limited", tl):
            profile.employment_type = "self_employed"
        # Professional
        elif re.search(r"\bdoctor\b|\bca\b|\bchartered\s*accountant|\badvocate\b|\blawyer\b|\bengineer\b", tl):
            profile.employment_type = "professional"
        # Salaried (most common)
        elif re.search(r"\bsalaried\b|\bemployee\b|\bpayslip\b|\bpay\s*slip\b|\bcompany\b|\bemployer\b", tl):
            profile.employment_type = "salaried"

        if profile.employment_type:
            profile.raw_fields_found.append("employment_type")

    # ── Work experience ───────────────────────────────────────────────────────

    def _extract_experience(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:experience|tenure|service)\s*[:\-]?\s*(\d+)\s*years?(?:\s*(\d+)\s*months?)?",
            r"(\d+)\s*years?\s*(?:of\s*)?(?:experience|service|employment)",
            r"(?:joining|date\s*of\s*joining)\s*[:\-]?\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    if "joining" in pat.lower() and len(m.groups()) == 3:
                        import datetime
                        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        if year < 100:
                            year += 2000
                        joined = datetime.date(year, month, day)
                        months = (datetime.date.today() - joined).days // 30
                        if 0 < months < 600:
                            profile.work_experience_months = months
                            profile.raw_fields_found.append("work_experience_months")
                            return
                    else:
                        years  = int(m.group(1))
                        months = int(m.group(2)) if m.lastindex >= 2 and m.group(2) else 0
                        total  = years * 12 + months
                        if 0 < total < 600:
                            profile.work_experience_months = total
                            profile.raw_fields_found.append("work_experience_months")
                            return
                except (ValueError, OverflowError):
                    pass

    # ── Name ─────────────────────────────────────────────────────────────────

    def _extract_name(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:name\s*of\s*(?:employee|applicant|customer|borrower))\s*[:\-]?\s*([A-Z][A-Za-z\s]{2,40})",
            r"(?:employee\s*name|customer\s*name)\s*[:\-]?\s*([A-Z][A-Za-z\s]{2,40})",
            r"(?:dear|to)\s*(?:mr\.?|mrs\.?|ms\.?|dr\.?)?\s*([A-Z][A-Za-z\s]{2,30})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                name = m.group(1).strip()
                # Filter obvious non-names
                if len(name.split()) >= 2 and len(name) <= 50:
                    profile.name = name
                    profile.raw_fields_found.append("name")
                    return

    # ── Employer ──────────────────────────────────────────────────────────────

    def _extract_employer(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:employer|company|organisation|organization)\s*(?:name)?\s*[:\-]?\s*([A-Z][A-Za-z0-9\s&.,]{2,60})",
            r"(?:paid\s*by|issued\s*by|from)\s*[:\-]?\s*([A-Z][A-Za-z0-9\s&.,]{2,60})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                emp = m.group(1).strip().rstrip(".,")
                if 3 <= len(emp) <= 60:
                    profile.employer_name = emp
                    profile.raw_fields_found.append("employer_name")
                    return

    # ── PAN (masked) ──────────────────────────────────────────────────────────

    def _extract_pan(self, text: str, profile: ExtractedProfile) -> None:
        m = re.search(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text)
        if m:
            pan = m.group(1)
            # Mask middle digits for privacy
            profile.pan_number = pan[:5] + "****" + pan[-1]
            profile.raw_fields_found.append("pan_number")

    # ── Loan amount requested ─────────────────────────────────────────────────

    def _extract_loan_amount(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:loan\s*(?:amount|requested|applied))\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)",
            r"(?:borrow|disburse|sanction)\s*(?:amount)?\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if 1_000 <= val <= 10_00_00_000:
                        profile.loan_amount_requested = val
                        profile.raw_fields_found.append("loan_amount_requested")
                        return
                except ValueError:
                    pass

    # ── Bank balance ──────────────────────────────────────────────────────────

    def _extract_bank_balance(self, text: str, profile: ExtractedProfile) -> None:
        patterns = [
            r"(?:closing\s*balance|available\s*balance|current\s*balance)\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)",
            r"(?:balance\s*as\s*on|balance\s*forward)\s*[:\-]?\s*₹?\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                try:
                    val = float(m.group(1).replace(",", ""))
                    if val >= 0:
                        profile.bank_balance = val
                        profile.raw_fields_found.append("bank_balance")
                        return
                except ValueError:
                    pass

    # ── DTI ratio ─────────────────────────────────────────────────────────────

    def _extract_dti(self, text: str, profile: ExtractedProfile) -> None:
        m = re.search(
            r"(?:dti|debt.to.income|emi.to.income)\s*(?:ratio)?\s*[:\-]?\s*([\d.]+)\s*%?",
            text, re.I,
        )
        if m:
            try:
                val = float(m.group(1))
                if val > 1.5:
                    val /= 100.0    # was percentage
                if 0 < val < 1:
                    profile.dti_ratio = val
                    profile.raw_fields_found.append("dti_ratio")
            except ValueError:
                pass
