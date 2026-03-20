#!/usr/bin/env python3
"""
scripts/process_all_pdfs.py
============================
Batch PDF processing script.

- Walks data/ directory for all .pdf files
- Runs each through PDFPipeline (NVIDIA or fallback)
- Saves structured JSON to processed_data/
- Skips files whose hash hasn't changed (idempotent)
- Prints a summary report

Usage:
    python scripts/process_all_pdfs.py

    # Force reprocess (ignores cache)
    python scripts/process_all_pdfs.py --force

    # Process a single file
    python scripts/process_all_pdfs.py --file data/hdfc_pdfs/hdfc_personal_loan.pdf

    # Dry run — show what would be processed
    python scripts/process_all_pdfs.py --dry-run
"""

import sys
import os
import argparse
import json
import logging
import time
from pathlib import Path

# Allow running from project root or scripts/ directory
_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_all(
    data_dir: Path,
    processed_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    Process every PDF in data_dir and its subdirectories.
    Returns a summary dict.
    """
    from pdf_pipeline import PDFPipeline

    pipeline = PDFPipeline(processed_dir=processed_dir)

    # Collect all PDFs
    pdf_files = sorted(data_dir.rglob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0}

    logger.info("Found %d PDF file(s)", len(pdf_files))

    processed_count = 0
    skipped_count   = 0
    failed_count    = 0
    total_chunks    = 0
    results         = []

    for pdf_path in pdf_files:
        logger.info("─" * 55)
        logger.info("Processing: %s", pdf_path.relative_to(data_dir))

        if dry_run:
            logger.info("[DRY RUN] Would process: %s", pdf_path.name)
            continue

        # Force: delete cache before processing
        if force:
            cache_file = processed_dir / (pdf_path.stem + ".json")
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Deleted cache: %s", cache_file.name)

        t0 = time.perf_counter()
        try:
            chunks = pipeline.process(str(pdf_path))
            elapsed = (time.perf_counter() - t0) * 1000

            if not chunks:
                logger.warning("  ⚠️  No chunks extracted from %s", pdf_path.name)
                failed_count += 1
                continue

            # Count element types
            type_counts: dict[str, int] = {}
            for c in chunks:
                type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1

            logger.info(
                "  ✅ %d chunks | %.0fms | bank=%s | types=%s",
                len(chunks), elapsed, chunks[0].bank, type_counts,
            )

            results.append({
                "file":        pdf_path.name,
                "bank":        chunks[0].bank,
                "chunks":      len(chunks),
                "type_counts": type_counts,
                "elapsed_ms":  round(elapsed),
            })

            processed_count += 1
            total_chunks    += len(chunks)

        except Exception as e:
            logger.error("  ❌ Failed: %s — %s", pdf_path.name, e, exc_info=True)
            failed_count += 1

    return {
        "processed":    processed_count,
        "skipped":      skipped_count,
        "failed":       failed_count,
        "total_chunks": total_chunks,
        "files":        results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch PDF processing for Loan Advisor")
    parser.add_argument("--force",   action="store_true", help="Reprocess even if cache exists")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be processed")
    parser.add_argument("--file",    type=str,            help="Process a single PDF file")
    parser.add_argument("--data-dir",    default=None,    help="Override data directory")
    parser.add_argument("--output-dir",  default=None,    help="Override processed_data directory")
    args = parser.parse_args()

    from config import DATA_DIR, PROCESSED_DATA_DIR
    data_dir      = Path(args.data_dir)      if args.data_dir      else DATA_DIR
    processed_dir = Path(args.output_dir)    if args.output_dir    else PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Loan Advisor — PDF Processing Pipeline")
    print("=" * 60)
    print(f"  Data dir:      {data_dir}")
    print(f"  Output dir:    {processed_dir}")
    print(f"  Force:         {args.force}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    if args.file:
        # Single file mode
        from pdf_pipeline import PDFPipeline
        pipeline = PDFPipeline(processed_dir=processed_dir)
        if args.force:
            cache = processed_dir / (Path(args.file).stem + ".json")
            if cache.exists():
                cache.unlink()

        chunks = pipeline.process(args.file)
        print(f"\n✅ Processed {args.file}")
        print(f"   Chunks: {len(chunks)}")
        if chunks:
            print(f"   Bank:   {chunks[0].bank}")
            print(f"   Types:  {set(c.doc_type for c in chunks)}")
            print(f"\nSample chunk:")
            print(f"  [{chunks[0].doc_type}] {chunks[0].text[:150]}…")
    else:
        # Batch mode
        summary = process_all(
            data_dir=data_dir,
            processed_dir=processed_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        print(f"  Processed:    {summary['processed']}")
        print(f"  Failed:       {summary['failed']}")
        print(f"  Total chunks: {summary['total_chunks']}")

        if summary["files"]:
            print("\n  Per-file breakdown:")
            for f in summary["files"]:
                print(f"    {f['file']:<35} {f['chunks']:>4} chunks  [{f['bank']}]")

        print()
        if summary["processed"] > 0:
            print("  ✅ Done. Run bootstrap.py to rebuild the FAISS index.")
        elif not args.dry_run:
            print("  ⚠️  No PDFs found. Add .pdf files to data/ subdirectories.")


if __name__ == "__main__":
    main()
