"""
scraper/parsers/axis_parser.py
================================
Axis Bank personal loan parser.

Axis uses a well-structured site with clear section headings:
  "Eligibility", "Interest Rates & Charges", "Documentation"
Each section has predictable CSS class patterns.
"""

from __future__ import annotations
import re
from bs4 import BeautifulSoup

from scraper.schema import LoanRecord
from scraper.normalizer import parse_rate_range, parse_monthly_income, parse_age_range, parse_inr
from scraper.parsers.base_parser import BaseParser


class AxisParser(BaseParser):

    BANK_NAME = "Axis"

    def enrich(self, soup: BeautifulSoup, url: str, record: LoanRecord) -> None:
        self._extract_eligibility(soup, record)
        self._extract_charges(soup, record)
        self._extract_features(soup, record)

    def _extract_eligibility(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        section = self.find_section(soup, "eligibility", "who can apply")
        if not section:
            return

        # Walk siblings to gather eligibility content
        parts = []
        el = section
        for _ in range(10):
            parts.append(self.text_of(el))
            el = el.find_next_sibling()
            if not el:
                break
        text = " ".join(parts)

        # Income
        if not record.min_income:
            m = re.search(
                r"(?:net\s*monthly|monthly\s*(?:net|take.?home)?)\s*income\s*"
                r"(?:of|above|:)?\s*([\d,₹.\s]+(?:lakh|lpa|k)?)",
                text, re.I,
            )
            if m:
                inc = parse_monthly_income(m.group(1))
                if inc:
                    record.min_income = inc

        # Age
        if not record.age_range:
            record.age_range = parse_age_range(text)

        # Employment type — Axis explicitly lists "Salaried" / "Self-employed"
        if not record.employment_type:
            if re.search(r"\bsalaried\b", text, re.I):
                record.employment_type.append("salaried")
            if re.search(r"self.?employed\b|business\s*owner", text, re.I):
                record.employment_type.append("self_employed")
            if re.search(r"\bprofessional\b", text, re.I):
                record.employment_type.append("professional")

        # Work experience
        m = re.search(r"(?:minimum|at\s*least)\s*(\d+)\s*(?:year|month)", text, re.I)
        if m:
            val = int(m.group(1))
            if "year" in text[m.start():m.end()].lower():
                val *= 12
            if not record.__dict__.get("min_experience"):
                record.__dict__["min_experience"] = val

        record.key_facts.append(text[:400])

    def _extract_charges(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        section = self.find_section(soup, "interest rate", "charges", "fee")
        if not section:
            return
        text = self.text_of(section)
        for sib in section.find_next_siblings()[:5]:
            text += " " + self.text_of(sib)

        if not record.interest_rate:
            rate = parse_rate_range(text)
            if rate:
                record.interest_rate = rate

        m = re.search(r"processing\s*(?:fee|charge)\s*(?:of|:)?\s*([\d.]+)\s*%", text, re.I)
        if m and not record.processing_fee_pct:
            record.processing_fee_pct = float(m.group(1))

        m = re.search(r"foreclos\w*\s*(?:charge|fee)?\s*(?:of|:)?\s*([\d.]+)\s*%", text, re.I)
        if m:
            record.__dict__["foreclosure_fee"] = float(m.group(1))

    def _extract_features(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        for ul in soup.find_all("ul"):
            items = [self.text_of(li) for li in ul.find_all("li")]
            relevant = [i for i in items if 15 < len(i) < 250]
            if len(relevant) >= 3:
                for item in relevant:
                    if item not in record.features:
                        record.features.append(item)
