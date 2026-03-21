"""
scraper/parser_router.py
==========================
Routes a URL to the correct site-specific parser.

If no specific parser matches, falls back to the generic BaseParser
which uses only the ExtractionResult from the waterfall extractor.
"""

from __future__ import annotations
import logging
from urllib.parse import urlparse

from scraper.schema import LoanRecord
from scraper.extractor import ExtractionResult
from scraper.parsers.base_parser import BaseParser
from scraper.parsers.icici_parser import ICICIParser
from scraper.parsers.axis_parser import AxisParser
from scraper.parsers.sbi_parser import SBIParser
from scraper.parsers.paisabazaar_parser import PaisabazaarParser
from scraper.parsers.bankbazaar_parser import BankBazaarParser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic fallback parser
# ---------------------------------------------------------------------------

class GenericParser(BaseParser):
    """Used when no site-specific parser matches."""
    BANK_NAME = "Unknown"

    def enrich(self, soup, url, record):
        # Generic parser does nothing extra — base record from ExtractionResult is enough
        record.bank = _guess_bank_from_url(url) or record.bank


# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

_PARSERS: list[tuple[str, type[BaseParser]]] = [
    ("icicibank.com",    ICICIParser),
    ("axisbank.com",     AxisParser),
    ("sbi.co.in",        SBIParser),
    ("onlinesbi.sbi",    SBIParser),
    ("paisabazaar.com",  PaisabazaarParser),
    ("bankbazaar.com",   BankBazaarParser),
]


def get_parser(url: str) -> BaseParser:
    """
    Return the appropriate parser for the given URL.
    Falls back to GenericParser if no domain match found.
    """
    domain = urlparse(url).netloc.lower()
    for pattern, parser_cls in _PARSERS:
        if pattern in domain:
            logger.debug("[Router] %s → %s", domain, parser_cls.__name__)
            return parser_cls()

    logger.debug("[Router] %s → GenericParser", domain)
    return GenericParser()


def route_and_parse(
    html:    str,
    url:     str,
    generic: ExtractionResult,
) -> LoanRecord:
    """
    Route URL to parser, run site-specific enrichment, return LoanRecord.

    If the specific parser fails entirely, GenericParser is used as the
    final fallback so we always produce a LoanRecord.
    """
    parser = get_parser(url)
    try:
        record = parser.parse(html, url, generic)
        record.bank = record.bank or _guess_bank_from_url(url) or "Unknown"
        return record
    except Exception as e:
        logger.error("[Router] %s crashed (%s) — falling back to GenericParser",
                     type(parser).__name__, e)
        fallback = GenericParser()
        return fallback.parse(html, url, generic)


def _guess_bank_from_url(url: str) -> str:
    url_lower = url.lower()
    for pattern, name in [
        ("icici",        "ICICI"),
        ("axis",         "Axis"),
        ("sbi",          "SBI"),
        ("hdfc",         "HDFC"),
        ("paisabazaar",  "Paisabazaar"),
        ("bankbazaar",   "BankBazaar"),
        ("rbi",          "RBI"),
    ]:
        if pattern in url_lower:
            return name
    return ""
