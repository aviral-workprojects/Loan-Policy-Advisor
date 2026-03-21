"""
scraper/parsers/paisabazaar_parser.py
=======================================
Paisabazaar aggregator parser.

Paisabazaar has structured comparison tables with multiple banks.
Each row is one bank's loan product. We extract these as separate
LoanRecords — one per bank row — plus an overall comparison record.
"""

from __future__ import annotations
import re
from bs4 import BeautifulSoup

from scraper.schema import LoanRecord
from scraper.normalizer import parse_rate_range, parse_monthly_income, parse_inr, parse_cibil
from scraper.parsers.base_parser import BaseParser


class PaisabazaarParser(BaseParser):

    BANK_NAME = "Paisabazaar"

    def enrich(self, soup: BeautifulSoup, url: str, record: LoanRecord) -> None:
        # Extract multi-bank comparison table rows as key facts
        self._extract_comparison_table(soup, record)
        self._extract_features(soup, record)

    def _extract_comparison_table(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        """
        Paisabazaar comparison tables typically have columns:
          Bank | Interest Rate | Processing Fee | Loan Amount | Tenure
        Convert each row to natural language and store as key_facts.
        """
        for table in soup.find_all("table"):
            rows = []
            headers = []
            for i, tr in enumerate(table.find_all("tr")):
                cells = [self.text_of(td) for td in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                if i == 0 or not headers:
                    headers = cells
                else:
                    rows.append(cells)

            if not headers or len(rows) < 2:
                continue

            # Check if it looks like a bank comparison table
            header_text = " ".join(headers).lower()
            if not re.search(r"bank|interest|rate|lender", header_text):
                continue

            for row in rows:
                if len(row) < 2:
                    continue
                # Build natural language sentence for this row
                pairs = []
                for j, header in enumerate(headers):
                    if j < len(row) and row[j].strip():
                        pairs.append(f"{header}: {row[j]}")
                if pairs:
                    nl = " | ".join(pairs)
                    record.key_facts.append(nl)
                    # Try to extract rate from this row
                    if not record.interest_rate:
                        rate = parse_rate_range(nl)
                        if rate:
                            record.interest_rate = rate

            # Store a consolidated natural language version
            nl_lines = [" | ".join(headers)]
            for row in rows:
                if any(c.strip() for c in row):
                    nl_lines.append(" | ".join(row))
            record.content = "\n".join(nl_lines)[:3000]

    def _extract_features(self, soup: BeautifulSoup, record: LoanRecord) -> None:
        for li in soup.find_all("li"):
            text = self.text_of(li)
            if re.search(r"\b(?:rate|eligib|income|salary|cibil|fee|tenure|amount)\b",
                         text, re.I) and 20 < len(text) < 300:
                if text not in record.features:
                    record.features.append(text)
