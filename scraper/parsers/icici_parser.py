"""
scraper/parsers/icici_parser.py
================================
ICICI Bank personal loan parser.

ICICI's site uses React-rendered content. Key patterns:
  - Interest rates in a "rates-table" or similar component
  - Eligibility in card/accordion sections
  - Fees in a tabular format
"""

from __future__ import annotations
import re
from bs4 import BeautifulSoup

from scraper.schema import LoanRecord
from scraper.extractor import ExtractionResult
from scraper.normalizer import parse_rate_range, parse_monthly_income, parse_inr, parse_age_range
from scraper.parsers.base_parser import BaseParser


class ICICIParser(BaseParser):

    BANK_NAME = "ICICI"

    def enrich(self, soup: BeautifulSoup, url: str, record: LoanRecord) -> None:
        # ── Interest rates ────────────────────────────────────────────────
        if not record.interest_rate:
            record.interest_rate = self._extract_rates(soup)

        # ── Eligibility section ───────────────────────────────────────────
        self._extract_eligibility(soup, record)

        # ── Features / benefits ───────────────────────────────────────────
        self._extract_features(soup, record)

        # ── Fees ─────────────────────────────────────────────────────────
        self._extract_fees(soup, record)

        # ── Page title → content ──────────────────────────────────────────
        title = soup.find("h1")
        if title and not record.content:
            record.content = self.text_of(title) + " " + record.content

    def _extract_rates(self, soup: BeautifulSoup) -> list[float]:
        # Look for rate containers — ICICI uses multiple class patterns
        rate_containers = soup.find_all(
            class_=re.compile(r"rate|interest|roi", re.I)
        )
        for el in rate_containers:
            text = self.text_of(el)
            rate = parse_rate_range(text)
            if rate:
                return rate

        # Fallback: scan all text for "X% to Y% p.a." patterns
        full_text = soup.get_text(" ")
        return parse_rate_range(full_text)

    def _extract_eligibility(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        # Find eligibility section
        elig_section = self.find_section(soup, "eligibility", "who can apply", "criteria")
        if not elig_section:
            return

        # Get all text in this section
        section_text = ""
        el = elig_section
        for _ in range(8):   # walk up to 8 siblings
            section_text += " " + self.text_of(el)
            el = el.find_next_sibling()
            if not el:
                break

        # Income
        if not record.min_income:
            m = re.search(r"income\s*(?:of|above|:)?\s*([\d,₹.\s]+(?:lpa|lakh|k)?)", section_text, re.I)
            if m:
                inc = parse_monthly_income(m.group(1))
                if inc:
                    record.min_income = inc

        # Age
        if not record.age_range:
            record.age_range = parse_age_range(section_text)

        # Employment
        if not record.employment_type:
            if re.search(r"salaried", section_text, re.I):
                record.employment_type.append("salaried")
            if re.search(r"self.employ", section_text, re.I):
                record.employment_type.append("self_employed")

    def _extract_features(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        # ICICI uses list items in feature sections
        for ul in soup.find_all("ul"):
            parent_text = self.text_of(ul.find_parent())
            if re.search(r"feature|benefit|highlight|advantage", parent_text, re.I):
                for li in ul.find_all("li"):
                    text = self.text_of(li)
                    if 15 < len(text) < 250 and text not in record.features:
                        record.features.append(text)

    def _extract_fees(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        fee_section = self.find_section(soup, "processing fee", "charges", "fees")
        if not fee_section:
            return
        text = self.text_of(fee_section)
        m = re.search(r"processing\s*fee\s*(?:of|:)?\s*([\d.]+)\s*%", text, re.I)
        if m and not record.processing_fee_pct:
            record.processing_fee_pct = float(m.group(1))
