"""
scraper/parsers/base_parser.py
================================
Abstract base class that all site-specific parsers inherit.

Each parser receives pre-fetched HTML + the ExtractionResult and
enriches the LoanRecord using site-specific knowledge of that bank's
HTML structure. If site-specific logic fails, the generic extractor
result is used as a fallback (already populated before the parser runs).
"""

from __future__ import annotations
import logging
import re
from abc import ABC, abstractmethod

from bs4 import BeautifulSoup

from scraper.schema import LoanRecord
from scraper.extractor import ExtractionResult

logger = logging.getLogger(__name__)


class BaseParser(ABC):

    BANK_NAME: str = ""

    def parse(
        self,
        html:        str,
        url:         str,
        generic:     ExtractionResult,
    ) -> LoanRecord:
        """
        Parse HTML into a LoanRecord.

        Steps:
          1. Build a base record from the generic ExtractionResult
          2. Call site-specific enrich() to apply specialised logic
          3. Return the enriched record

        If enrich() raises, log and return the base record.
        """
        record = self._base_record(url, generic)
        try:
            soup = BeautifulSoup(html, "html.parser")
            self.enrich(soup, url, record)
        except Exception as e:
            logger.warning("[%s] Site-specific parsing failed: %s — using generic result",
                           self.BANK_NAME, e)
        return record

    @abstractmethod
    def enrich(self, soup: BeautifulSoup, url: str, record: LoanRecord) -> None:
        """Site-specific enrichment. Modifies record in place."""
        ...

    def _base_record(self, url: str, g: ExtractionResult) -> LoanRecord:
        """Build a LoanRecord from generic ExtractionResult."""
        content_parts = []
        if g.raw_text:
            content_parts.append(g.raw_text)
        if g.tables_text:
            content_parts.extend(g.tables_text[:5])
        if g.key_facts:
            content_parts.extend(g.key_facts[:5])

        return LoanRecord(
            bank=self.BANK_NAME,
            source_url=url,
            loan_type="personal",
            interest_rate=g.interest_rate or [],
            processing_fee=g.processing_fee,
            processing_fee_pct=g.processing_fee_pct,
            min_income=g.min_income,
            age_range=g.age_range or [],
            loan_amount=g.loan_amount or [],
            tenure=g.tenure or [],
            min_cibil=g.min_cibil,
            employment_type=g.employment_type or [],
            features=g.features[:20],
            key_facts=g.key_facts[:10],
            content=" ".join(content_parts)[:4000],
            extraction_method=",".join(g.layers_used),
            confidence=self._score(g),
        )

    @staticmethod
    def _score(g: ExtractionResult) -> float:
        """Rough confidence score based on how many fields were extracted."""
        filled = sum([
            bool(g.interest_rate),
            bool(g.min_income),
            bool(g.age_range),
            bool(g.loan_amount),
            bool(g.tenure),
            bool(g.min_cibil),
            bool(g.features),
            bool(g.raw_text and len(g.raw_text) > 200),
        ])
        return round(filled / 8, 2)

    # ── Utility helpers for subclasses ─────────────────────────────────────

    @staticmethod
    def text_of(el) -> str:
        """Safe text extraction from a BeautifulSoup element."""
        if el is None:
            return ""
        return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()

    @staticmethod
    def find_section(soup: BeautifulSoup, *keywords: str):
        """Find the first element whose text contains any of the keywords."""
        pattern = re.compile("|".join(keywords), re.I)
        return soup.find(lambda tag: tag.name in ("h2","h3","h4","div","section","p")
                         and pattern.search(tag.get_text()))
