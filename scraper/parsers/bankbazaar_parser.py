"""
scraper/parsers/bankbazaar_parser.py
======================================
BankBazaar aggregator parser.

BankBazaar pages have:
  - A header rate display ("Starting @ X%")
  - Sortable comparison tables
  - Lender-specific rows we can extract per bank
  - FAQ sections with useful eligibility text
"""

from __future__ import annotations
import re
from bs4 import BeautifulSoup

from scraper.schema import LoanRecord
from scraper.normalizer import parse_rate_range, parse_inr, parse_monthly_income, parse_cibil
from scraper.parsers.base_parser import BaseParser


class BankBazaarParser(BaseParser):

    BANK_NAME = "BankBazaar"

    def enrich(self, soup: BeautifulSoup, url: str, record: LoanRecord) -> None:
        self._extract_header_rate(soup, record)
        self._extract_comparison_table(soup, record)
        self._extract_faq(soup, record)

    def _extract_header_rate(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        """BankBazaar often shows 'Starting @ X%' in a prominent header."""
        for el in soup.find_all(class_=re.compile(r"rate|starting|interest", re.I)):
            text = self.text_of(el)
            if re.search(r"start|from|@", text, re.I):
                rate = parse_rate_range(text)
                if rate and not record.interest_rate:
                    record.interest_rate = rate
                    break

    def _extract_comparison_table(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        """Extract lender comparison table — similar to Paisabazaar."""
        for table in soup.find_all("table"):
            header_row = table.find("tr")
            if not header_row:
                continue
            headers = [self.text_of(th) for th in header_row.find_all(["th", "td"])]
            if not any(re.search(r"bank|lender|rate|interest", h, re.I) for h in headers):
                continue

            rows = table.find_all("tr")[1:]
            for tr in rows:
                cells = [self.text_of(td) for td in tr.find_all("td")]
                if not cells:
                    continue
                pairs = [f"{headers[i]}: {cells[i]}"
                         for i in range(min(len(headers), len(cells)))
                         if cells[i].strip()]
                if pairs:
                    nl = " | ".join(pairs)
                    if nl not in record.key_facts:
                        record.key_facts.append(nl)

            # Build content from table
            if not record.content:
                all_rows = [" | ".join(headers)]
                for tr in rows:
                    cells = [self.text_of(td) for td in tr.find_all("td")]
                    if any(c.strip() for c in cells):
                        all_rows.append(" | ".join(cells))
                record.content = "\n".join(all_rows)[:3000]

    def _extract_faq(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        """FAQ sections contain useful eligibility and feature information."""
        faq_section = soup.find(class_=re.compile(r"faq|question|accordion", re.I))
        if not faq_section:
            faq_section = self.find_section(soup, "frequently asked", "faq")
        if not faq_section:
            return

        for q in faq_section.find_all(class_=re.compile(r"question|q\b", re.I)):
            a_el = q.find_next_sibling()
            if a_el:
                qa_text = self.text_of(q) + " " + self.text_of(a_el)
                if 30 < len(qa_text) < 500 and qa_text not in record.features:
                    record.features.append(qa_text[:400])
