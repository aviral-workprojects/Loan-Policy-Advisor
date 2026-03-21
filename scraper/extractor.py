"""
scraper/extractor.py
=====================
Multi-layer waterfall extraction engine.

For each fetched HTML page, tries 7 extraction methods in order,
combining results rather than stopping at the first success.
The idea: different layers extract different things. Structured
parsing gets tables and headings; regex catches numbers buried in
paragraphs; metadata catches schema.org / JSON-LD structured data.

Layer order:
  1. JSON-LD / schema.org structured data
  2. HTML heading + section parsing
  3. Table extraction
  4. Regex financial data extraction
  5. Meta tags extraction
  6. Body text fallback
  7. Linked subpage discovery (returns new URLs to crawl)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from scraper.normalizer import (
    parse_rate_range, parse_monthly_income, parse_inr,
    parse_age_range, parse_tenure_range, parse_cibil,
    parse_employment_types, parse_percent,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """All data extracted from a single HTML page."""
    interest_rate:    list[float] = field(default_factory=list)
    processing_fee:   float | None = None
    processing_fee_pct: float | None = None
    min_income:       float | None = None
    age_range:        list[int]   = field(default_factory=list)
    loan_amount:      list[float] = field(default_factory=list)
    tenure:           list[int]   = field(default_factory=list)
    min_cibil:        int   | None = None
    employment_type:  list[str]   = field(default_factory=list)
    features:         list[str]   = field(default_factory=list)
    key_facts:        list[str]   = field(default_factory=list)
    raw_text:         str         = ""
    tables_text:      list[str]   = field(default_factory=list)
    linked_urls:      list[str]   = field(default_factory=list)
    layers_used:      list[str]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_all(html: str, base_url: str) -> ExtractionResult:
    """
    Run all extraction layers against an HTML page.
    Results are accumulated (not first-wins) so each layer adds what it finds.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = ExtractionResult()

    # Remove noise elements that add no content
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "iframe", "noscript", "aside"]):
        tag.decompose()

    _layer1_jsonld(soup, result)
    _layer2_sections(soup, result)
    _layer3_tables(soup, result)
    _layer4_regex(soup, result)
    _layer5_meta(soup, result)
    _layer6_body(soup, result)
    _layer7_links(soup, base_url, result)

    logger.debug("[Extractor] Layers used: %s | Rate: %s | Income: %s",
                 result.layers_used, result.interest_rate, result.min_income)
    return result


# ---------------------------------------------------------------------------
# Layer 1 — JSON-LD / Schema.org
# ---------------------------------------------------------------------------

def _layer1_jsonld(soup: BeautifulSoup, r: ExtractionResult) -> None:
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, list):
                for item in data:
                    _parse_jsonld_item(item, r)
            elif isinstance(data, dict):
                _parse_jsonld_item(data, r)
            r.layers_used.append("jsonld")
        except Exception:
            pass


def _parse_jsonld_item(data: dict, r: ExtractionResult) -> None:
    # Look for FAQPage, FinancialProduct, or any structured loan data
    text_parts = []
    for val in _flatten_values(data):
        if isinstance(val, str) and len(val) > 20:
            text_parts.append(val)

    combined = " ".join(text_parts)
    if combined:
        _apply_regex_to_text(combined, r)


def _flatten_values(obj, depth=0) -> list:
    if depth > 5:
        return []
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, (int, float)):
        return [str(obj)]
    if isinstance(obj, list):
        out = []
        for item in obj:
            out.extend(_flatten_values(item, depth+1))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(_flatten_values(v, depth+1))
        return out
    return []


# ---------------------------------------------------------------------------
# Layer 2 — HTML sections
# ---------------------------------------------------------------------------

_FINANCIAL_HEADINGS = re.compile(
    r"eligib|interest\s*rate|fee|charge|income|salary|cibil|credit\s*score|"
    r"loan\s*amount|tenure|age|apply|feature|benefit|documentation|who\s*can",
    re.I,
)

def _layer2_sections(soup: BeautifulSoup, r: ExtractionResult) -> None:
    sections_found = 0
    for heading in soup.find_all(["h1", "h2", "h3", "h4"]):
        heading_text = heading.get_text(" ", strip=True)
        if not _FINANCIAL_HEADINGS.search(heading_text):
            continue

        # Collect text from siblings until the next heading
        section_parts = [heading_text]
        for sibling in heading.find_next_siblings():
            if sibling.name in ("h1", "h2", "h3", "h4"):
                break
            text = sibling.get_text(" ", strip=True)
            if text:
                section_parts.append(text)
                # Extract bullet/list items as features
                for li in sibling.find_all("li"):
                    feat = li.get_text(" ", strip=True)
                    if 10 < len(feat) < 300 and feat not in r.features:
                        r.features.append(feat)

        section_text = " ".join(section_parts)
        if len(section_text) > 50:
            _apply_regex_to_text(section_text, r)
            r.key_facts.append(section_text[:400])
            sections_found += 1

    if sections_found:
        r.layers_used.append("sections")


# ---------------------------------------------------------------------------
# Layer 3 — Tables
# ---------------------------------------------------------------------------

def _layer3_tables(soup: BeautifulSoup, r: ExtractionResult) -> None:
    tables_found = 0
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c.strip()]
            if cells:
                rows.append(cells)

        if not rows:
            continue

        # Convert table to natural language
        nl_parts = []
        headers = rows[0] if len(rows) > 1 else []

        for i, row in enumerate(rows[1:] if headers else rows):
            if len(row) == len(headers) and headers:
                pairs = [f"{headers[j]}: {row[j]}" for j in range(len(row))]
                nl_parts.append(" | ".join(pairs))
            elif len(row) == 2:
                nl_parts.append(f"{row[0]}: {row[1]}")
            else:
                nl_parts.append(" | ".join(row))

        table_text = " ".join(nl_parts)
        if len(table_text) > 30:
            r.tables_text.append(table_text[:1000])
            _apply_regex_to_text(table_text, r)
            tables_found += 1

    if tables_found:
        r.layers_used.append("tables")


# ---------------------------------------------------------------------------
# Layer 4 — Regex extraction
# ---------------------------------------------------------------------------

def _layer4_regex(soup: BeautifulSoup, r: ExtractionResult) -> None:
    # Get all visible text
    full_text = soup.get_text(" ", strip=True)
    full_text = re.sub(r"\s+", " ", full_text)

    before = bool(r.interest_rate)
    _apply_regex_to_text(full_text, r)
    if not before and r.interest_rate:
        r.layers_used.append("regex")


def _apply_regex_to_text(text: str, r: ExtractionResult) -> None:
    """Apply all financial regex patterns to a text string."""

    # Interest rate
    if not r.interest_rate:
        rate = parse_rate_range(text)
        if rate:
            r.interest_rate = rate

    # Income
    if not r.min_income:
        income_patterns = [
            r"(?:min(?:imum)?\s*(?:monthly)?\s*(?:income|salary|earning))\s*(?:of|:)?\s*([\d,₹.\s]+(?:lakh|lpa|k|thousand)?(?:\s*per\s*month|\s*pm)?)",
            r"(?:income|salary)\s+(?:should\s+be|must\s+be|required|atleast)\s*([\d,₹.\s]+(?:lakh|lpa|k)?)",
            r"monthly\s+(?:income|salary)\s+(?:above|more\s+than|greater\s+than|at\s+least)\s*([\d,₹.\s]+)",
        ]
        for pat in income_patterns:
            m = re.search(pat, text, re.I)
            if m:
                inc = parse_monthly_income(m.group(1))
                if inc:
                    r.min_income = inc
                    break

    # Age range
    if not r.age_range:
        age_patterns = [
            r"age\s*(?:between|from|:)?\s*(\d+)\s*(?:to|–|-|and)\s*(\d+)\s*years",
            r"(?:minimum|min)?\s*age\s*(?:of|:)?\s*(\d+)\s*years?",
        ]
        for pat in age_patterns:
            m = re.search(pat, text, re.I)
            if m:
                if m.lastindex == 2:
                    r.age_range = parse_age_range(m.group(0))
                else:
                    r.age_range = parse_age_range(m.group(0))
                if r.age_range:
                    break

    # Loan amount
    if not r.loan_amount:
        m = re.search(
            r"(?:loan|borrow)\s*(?:up\s*to|upto|maximum|of|ranging\s*from)?\s*"
            r"₹?\s*([\d.,]+\s*(?:lakh|crore|k|thousand)?)\s*(?:to|–)?\s*"
            r"₹?\s*([\d.,]+\s*(?:lakh|crore|k|thousand)?)?",
            text, re.I,
        )
        if m:
            lo = parse_inr(m.group(1)) if m.group(1) else None
            hi = parse_inr(m.group(2)) if m.group(2) else lo
            if lo and hi and lo <= hi:
                r.loan_amount = [lo, hi]

    # Tenure
    if not r.tenure:
        m = re.search(
            r"tenure\s*(?:of|:)?\s*(?:up\s*to\s*)?([\d]+)\s*(?:to\s*([\d]+))?\s*(year|month)",
            text, re.I,
        )
        if m:
            tr = parse_tenure_range(m.group(0))
            if tr:
                r.tenure = tr

    # CIBIL
    if not r.min_cibil:
        m = re.search(
            r"(?:cibil|credit\s*score)\s*(?:of|above|minimum|min|:)?\s*(\d{3})\+?",
            text, re.I,
        )
        if m:
            score = parse_cibil(m.group(1))
            if score:
                r.min_cibil = score

    # Processing fee
    if not r.processing_fee_pct:
        m = re.search(
            r"processing\s*(?:fee|charge)\s*(?:of|:)?\s*([\d.]+)\s*%",
            text, re.I,
        )
        if m:
            pct = float(m.group(1))
            if 0 < pct < 10:
                r.processing_fee_pct = pct

    # Employment types
    if not r.employment_type:
        emp = parse_employment_types(text)
        if emp:
            r.employment_type = emp


# ---------------------------------------------------------------------------
# Layer 5 — Meta tags
# ---------------------------------------------------------------------------

def _layer5_meta(soup: BeautifulSoup, r: ExtractionResult) -> None:
    meta_text_parts = []
    for meta in soup.find_all("meta"):
        content = meta.get("content", "")
        name = meta.get("name", meta.get("property", "")).lower()
        if name in ("description", "og:description", "twitter:description") and content:
            meta_text_parts.append(content)
            r.key_facts.append(content[:300])

    if meta_text_parts:
        _apply_regex_to_text(" ".join(meta_text_parts), r)
        r.layers_used.append("meta")


# ---------------------------------------------------------------------------
# Layer 6 — Body text fallback
# ---------------------------------------------------------------------------

def _layer6_body(soup: BeautifulSoup, r: ExtractionResult) -> None:
    # Collect main content area text — prefer main/article/section over full body
    content_el = (
        soup.find("main") or
        soup.find("article") or
        soup.find(id=re.compile(r"content|main|body", re.I)) or
        soup.find(class_=re.compile(r"content|main|body|container", re.I)) or
        soup.body
    )

    if content_el:
        raw = content_el.get_text(" ", strip=True)
        raw = re.sub(r"\s+", " ", raw).strip()
        # Store first 3000 chars as main content
        r.raw_text = raw[:3000]
        r.layers_used.append("body")


# ---------------------------------------------------------------------------
# Layer 7 — Linked subpage discovery
# ---------------------------------------------------------------------------

_RELEVANT_LINK_PATTERNS = re.compile(
    r"eligib|interest.?rate|fee|charge|emi|apply|personal.?loan|"
    r"home.?loan|compare|calculator",
    re.I,
)

def _layer7_links(soup: BeautifulSoup, base_url: str, r: ExtractionResult) -> None:
    domain = urlparse(base_url).netloc
    found = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        # Only follow same-domain links
        if urlparse(full).netloc != domain:
            continue
        if _RELEVANT_LINK_PATTERNS.search(text) or _RELEVANT_LINK_PATTERNS.search(href):
            if full not in found:
                found.append(full)

    r.linked_urls = found[:15]   # cap to avoid crawl explosion
    if found:
        r.layers_used.append("links")
