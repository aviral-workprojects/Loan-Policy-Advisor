#!/usr/bin/env python3
"""
run_scraper.py
===============
CLI entry point for the loan data scraping pipeline.

Place this file at the project root (same level as bootstrap.py).
Scraped data is written directly to data/ subfolders.
After running, execute bootstrap.py to rebuild the FAISS index.

Usage:
    python run_scraper.py                      # scrape all sources
    python run_scraper.py --url https://...    # scrape a single URL
    python run_scraper.py --bank axis          # scrape only Axis URLs
    python run_scraper.py --depth 0            # seeds only (no link following)
    python run_scraper.py --dry-run            # fetch + extract but don't write
    python run_scraper.py --no-playwright      # skip JS rendering
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scraper")


def main():
    parser = argparse.ArgumentParser(
        description="Loan Advisor — Web Scraping Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url",          type=str,  help="Scrape a single URL")
    parser.add_argument("--bank",         type=str,  help="Filter to a specific bank (axis|icici|sbi|paisabazaar|bankbazaar)")
    parser.add_argument("--depth",        type=int,  default=1, help="Crawl depth (0=seeds only, 1=follow links)")
    parser.add_argument("--workers",      type=int,  default=3, help="Parallel fetch workers")
    parser.add_argument("--dry-run",      action="store_true",  help="Extract but don't write to disk")
    parser.add_argument("--data-dir",     type=str,  default=None, help="Override data directory")
    parser.add_argument("--no-playwright",action="store_true",  help="Disable Playwright JS rendering")
    args = parser.parse_args()

    # ── Resolve data dir ──────────────────────────────────────────────────
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        try:
            from config import DATA_DIR
            data_dir = DATA_DIR
        except ImportError:
            data_dir = Path(__file__).parent / "data"

    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Loan Advisor — Scraping Pipeline")
    print("=" * 65)
    print(f"  Data dir: {data_dir}")
    print(f"  Depth:    {args.depth}")
    print(f"  Workers:  {args.workers}")
    print(f"  Dry run:  {args.dry_run}")
    print()

    # ── Build URL list ────────────────────────────────────────────────────
    from scraper.pipeline import ScrapingPipeline, SEED_URLS

    if args.url:
        seed_urls = [(args.url, "")]
        print(f"  Single URL mode: {args.url}\n")
    elif args.bank:
        bank_lower = args.bank.lower()
        seed_urls = [(url, bank) for url, bank in SEED_URLS if bank_lower in bank.lower()]
        if not seed_urls:
            print(f"  ⚠️  No seed URLs found for bank: {args.bank}")
            print(f"  Available: axis, icici, sbi, paisabazaar, bankbazaar")
            return
        print(f"  Bank filter: {args.bank} → {len(seed_urls)} URLs\n")
    else:
        seed_urls = SEED_URLS
        print(f"  All sources: {len(seed_urls)} seed URLs\n")

    # ── Run pipeline ──────────────────────────────────────────────────────
    t0 = time.perf_counter()

    if args.dry_run:
        # Dry run: extract + validate but don't write
        _dry_run(seed_urls, args.depth)
    else:
        pipeline = ScrapingPipeline(
            data_dir=data_dir,
            seed_urls=seed_urls,
            max_depth=args.depth,
            max_workers=args.workers,
        )
        records = pipeline.run()
        elapsed = time.perf_counter() - t0

        print("\n" + "=" * 65)
        print(f"  ✅ Done  —  {len(records)} records  in  {elapsed:.1f}s")
        print("=" * 65)

        # Per-bank summary
        from collections import defaultdict
        by_bank = defaultdict(list)
        for r in records:
            by_bank[r.bank].append(r)

        print(f"\n  {'Bank':<16}  {'Records':>7}  {'Has Rate':>9}  {'Has Income':>10}  {'Avg Conf':>9}")
        print(f"  {'-'*16}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*9}")
        for bank, recs in sorted(by_bank.items()):
            has_rate   = sum(1 for r in recs if r.interest_rate)
            has_income = sum(1 for r in recs if r.min_income)
            avg_conf   = sum(r.confidence for r in recs) / len(recs)
            print(f"  {bank:<16}  {len(recs):>7}  {has_rate:>9}  {has_income:>10}  {avg_conf:>9.2f}")

        print(f"\n  Files written to: {data_dir}")
        print(f"\n  Next step: python bootstrap.py --force")
        print("  (rebuilds FAISS index with new scraped content)")


def _dry_run(seed_urls, depth):
    """Fetch + extract + validate without writing."""
    from scraper.fetcher import fetch_with_waterfall
    from scraper.extractor import extract_all
    from scraper.parser_router import route_and_parse
    from scraper.validator import validate

    print("  DRY RUN — no files will be written\n")

    for url, bank in seed_urls[:3]:   # limit to first 3 in dry run
        print(f"  → {url}")
        result = fetch_with_waterfall(url)
        if not result.success:
            print(f"    ❌ Fetch failed: {result.error}")
            continue

        generic = extract_all(result.html, url)
        record  = route_and_parse(result.html, url, generic)
        is_valid, issues = validate(record)

        status = "✅" if is_valid else "❌"
        print(f"    {status} bank={record.bank}  rate={record.interest_rate}  "
              f"income={record.min_income}  cibil={record.min_cibil}  "
              f"conf={record.confidence:.2f}  layers={record.extraction_method}")
        if issues:
            print(f"       issues: {issues[:2]}")
    print()


if __name__ == "__main__":
    main()
