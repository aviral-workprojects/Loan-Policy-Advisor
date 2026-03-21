"""
scraper/parsers/sbi_parser.py
==============================
SBI personal loan parser.

SBI's website has inconsistent HTML. Interest rates are often
embedded in plain paragraphs rather than structured tables.
Uses aggressive fallback parsing and paragraph scanning.
"""

from __future__ import annotations
import re
from bs4 import BeautifulSoup

from scraper.schema import LoanRecord
from scraper.normalizer import parse_rate_range, parse_monthly_income, parse_age_range
from scraper.parsers.base_parser import BaseParser


class SBIParser(BaseParser):

    BANK_NAME = "SBI"

    def enrich(self, soup: BeautifulSoup, url: str, record: LoanRecord) -> None:
        # SBI rate info is often in <p> or <td> tags with plain text
        self._scan_paragraphs(soup, record)
        self._extract_eligibility(soup, record)

    def _scan_paragraphs(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        """Scan all paragraphs for financial data — SBI's main extraction path."""
        full_text = " ".join(self.text_of(p) for p in soup.find_all(["p", "td", "li"]))
        full_text = re.sub(r"\s+", " ", full_text)

        if not record.interest_rate:
            rate = parse_rate_range(full_text)
            if rate:
                record.interest_rate = rate

        if not record.min_income:
            # SBI-specific income patterns
            for pat in [
                r"(?:minimum\s*)?(?:net\s*monthly\s*)?income\s*(?:of)?\s*₹?\s*([\d,]+)",
                r"income\s*(?:should\s*be|of|:)\s*₹?\s*([\d,]+)\s*(?:per\s*month)?",
            ]:
                m = re.search(pat, full_text, re.I)
                if m:
                    inc = parse_monthly_income(m.group(1) + " per month")
                    if inc:
                        record.min_income = inc
                        break

        # SBI often has CIBIL requirements in text
        if not record.min_cibil:
            m = re.search(r"cibil\s*score\s*(?:of\s*|above\s*|minimum\s*)?(\d{3})", full_text, re.I)
            if m:
                score = int(m.group(1))
                if 300 <= score <= 900:
                    record.min_cibil = score

        # Capture key facts from rate/eligibility paragraphs
        for p in soup.find_all(["p", "li"]):
            text = self.text_of(p)
            if re.search(r"\d+\.?\d*\s*%|\binterest\b|\beligib\b|\bsalary\b", text, re.I):
                if 30 < len(text) < 500 and text not in record.key_facts:
                    record.key_facts.append(text)

    def _extract_eligibility(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        section = self.find_section(soup, "eligibility", "who can avail", "criteria")
        if not section:
            return
        text = self.text_of(section)
        for sib in section.find_next_siblings()[:8]:
            text += " " + self.text_of(sib)

        if not record.age_range:
            record.age_range = parse_age_range(text)

        if re.search(r"\bsalaried\b", text, re.I) and "salaried" not in record.employment_type:
            record.employment_type.append("salaried")
        if re.search(r"government|pensioner|psu", text, re.I) and "government" not in record.employment_type:
            record.employment_type.append("government")
        if re.search(r"self.?employ", text, re.I) and "self_employed" not in record.employment_type:
            record.employment_type.append("self_employed")
