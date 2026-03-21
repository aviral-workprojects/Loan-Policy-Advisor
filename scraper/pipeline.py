"""
scraper/pipeline.py
====================
Main scraping pipeline orchestrator.

Architecture:
  1. URL discovery — seed URLs + crawl discovered links (same domain only)
  2. Parallel fetch  — ThreadPoolExecutor per URL (polite, max 3 concurrent)
  3. Waterfall extract — 7 layers per fetched page (sequential per page)
  4. Parser routing  — site-specific enrichment
  5. Validation      — range checks, type checks, required fields
  6. Deduplication   — hash + URL level
  7. Storage         — JSON + TXT per record into data/ subfolders

Parallelism model:
  URLS are fetched in parallel (I/O bound)
  Extraction layers are sequential per page (CPU bound, correctness critical)
  ↳ This matches the spec: "DO NOT parallelize layers"
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from scraper.fetcher   import fetch_with_waterfall, FetchResult
from scraper.extractor import extract_all
from scraper.parser_router import route_and_parse
from scraper.validator import validate, DeduplicationStore
from scraper.storage   import StorageWriter
from scraper.schema    import LoanRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed URLs
# ---------------------------------------------------------------------------

SEED_URLS: list[tuple[str, str]] = [
    # (url, bank_hint)
    ("https://www.icicibank.com/personal-banking/loans/personal-loan",              "ICICI"),
    ("https://www.icicibank.com/personal-banking/loans/personal-loan/interest-rates","ICICI"),
    ("https://www.icicibank.com/personal-banking/loans/personal-loan/eligibility",  "ICICI"),
    ("https://www.axisbank.com/retail/loans/personal-loan",                         "Axis"),
    ("https://www.axisbank.com/retail/loans/personal-loan/interest-rates-charges",  "Axis"),
    ("https://www.axisbank.com/retail/loans/personal-loan/eligibility-documentation","Axis"),
    ("https://sbi.co.in/web/personal-banking/loans/personal-loans",                 "SBI"),
    ("https://sbi.co.in/web/personal-banking/loans/personal-loans/xpress-credit",   "SBI"),
    ("https://www.paisabazaar.com/personal-loan/",                                  "Paisabazaar"),
    ("https://www.paisabazaar.com/personal-loan/eligibility/",                      "Paisabazaar"),
    ("https://www.bankbazaar.com/personal-loan.html",                               "BankBazaar"),
    ("https://www.bankbazaar.com/personal-loan-interest-rate.html",                 "BankBazaar"),
    ("https://www.bankbazaar.com/personal-loan-eligibility.html",                   "BankBazaar"),
]

# Max additional URLs to crawl per seed (discovered via Layer 7 links)
MAX_CRAWL_DEPTH   = 1       # 0 = seeds only, 1 = seeds + one level of links
MAX_LINKS_PER_PAGE = 5      # max discovered links to follow per page
MAX_WORKERS       = 3       # concurrent fetch threads (polite)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScrapingPipeline:

    def __init__(
        self,
        data_dir:    Path,
        seed_urls:   list[tuple[str, str]] | None = None,
        max_depth:   int  = MAX_CRAWL_DEPTH,
        max_workers: int  = MAX_WORKERS,
        on_record:   Callable[[LoanRecord], None] | None = None,
    ):
        self.data_dir    = data_dir
        self.seed_urls   = seed_urls or SEED_URLS
        self.max_depth   = max_depth
        self.max_workers = max_workers
        self.on_record   = on_record   # optional callback per record

        dedup_cache = data_dir / ".scrape_dedup.txt"
        self.dedup   = DeduplicationStore(cache_file=dedup_cache)
        self.writer  = StorageWriter(data_dir)
        self.records: list[LoanRecord] = []

    def run(self) -> list[LoanRecord]:
        """Run the full pipeline and return all valid LoanRecords."""
        logger.info("[Pipeline] Starting scrape — %d seed URLs", len(self.seed_urls))
        t0 = time.perf_counter()

        # Build URL queue: [(url, bank_hint, depth)]
        queue: list[tuple[str, str, int]] = [
            (url, bank, 0) for url, bank in self.seed_urls
        ]
        visited: set[str] = set()

        while queue:
            batch = []
            next_queue: list[tuple[str, str, int]] = []

            for url, bank, depth in queue:
                if url in visited:
                    continue
                if self.dedup._urls and url in self.dedup._urls:
                    logger.debug("[Pipeline] Skip (dedup): %s", url)
                    continue
                visited.add(url)
                batch.append((url, bank, depth))

            if not batch:
                break

            logger.info("[Pipeline] Batch of %d URLs (depth=%s)", len(batch),
                        set(d for _, _, d in batch))

            # Parallel fetch
            fetch_results = self._parallel_fetch(batch)

            # Sequential waterfall extraction per page
            for (url, bank, depth), fetch_result in fetch_results:
                if not fetch_result.success and not fetch_result.html:
                    logger.warning("[Pipeline] Skip (no content): %s", url)
                    continue

                records, discovered_urls = self._process_page(
                    fetch_result, bank_hint=bank
                )

                for record in records:
                    self.records.append(record)
                    if self.on_record:
                        self.on_record(record)

                # Queue discovered links (one level deeper)
                if depth < self.max_depth:
                    for disc_url in discovered_urls[:MAX_LINKS_PER_PAGE]:
                        if disc_url not in visited:
                            next_queue.append((disc_url, bank, depth + 1))

            queue = next_queue

        elapsed = time.perf_counter() - t0
        logger.info("[Pipeline] Done — %d records in %.1fs", len(self.records), elapsed)

        # Write summary
        if self.records:
            self.writer.write_summary(self.records)

        return self.records

    def _parallel_fetch(
        self,
        batch: list[tuple[str, str, int]],
    ) -> list[tuple[tuple[str, str, int], FetchResult]]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {
                pool.submit(fetch_with_waterfall, url): (url, bank, depth)
                for url, bank, depth in batch
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    fetch_result = future.result(timeout=60)
                except Exception as e:
                    logger.warning("[Pipeline] Fetch exception for %s: %s", item[0], e)
                    fetch_result = FetchResult(url=item[0], success=False, error=str(e))
                results.append((item, fetch_result))
        return results

    def _process_page(
        self,
        fetch_result: FetchResult,
        bank_hint:    str = "",
    ) -> tuple[list[LoanRecord], list[str]]:
        """
        Run waterfall extraction + parser + validation on one page.
        Returns (valid_records, discovered_urls).
        """
        url  = fetch_result.url
        html = fetch_result.html or ""

        if not html:
            return [], []

        # ── Waterfall extraction (Layers 1–7, sequential) ──────────────────
        try:
            generic = extract_all(html, base_url=url)
        except Exception as e:
            logger.error("[Pipeline] Extraction failed for %s: %s", url, e)
            return [], []

        discovered_urls = generic.linked_urls

        # ── Parser routing ────────────────────────────────────────────────
        try:
            record = route_and_parse(html, url, generic)
        except Exception as e:
            logger.error("[Pipeline] Parser failed for %s: %s", url, e)
            return [], []

        # Apply bank hint if parser didn't set one
        if (not record.bank or record.bank == "Unknown") and bank_hint:
            record.bank = bank_hint

        # ── Validation ────────────────────────────────────────────────────
        is_valid, issues = validate(record)
        if not is_valid:
            logger.warning("[Pipeline] Invalid record from %s: %s", url, issues[:2])
            return [], discovered_urls

        # ── Deduplication ─────────────────────────────────────────────────
        if self.dedup.is_duplicate(record):
            return [], discovered_urls

        self.dedup.register(record)

        # ── Storage ───────────────────────────────────────────────────────
        try:
            self.writer.write(record)
        except Exception as e:
            logger.error("[Pipeline] Storage failed for %s: %s", url, e)

        return [record], discovered_urls
