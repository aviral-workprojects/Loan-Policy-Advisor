#!/usr/bin/env python3
"""
scripts/process_all_pdfs.py  (v4)
==================================
Batch PDF processing script.

v4 additions:
    ✅ NVIDIA success signal — warns loudly when no NVIDIA model contributed
    ✅ Model tracking        — per-file breakdown of model_source counts
    ✅ Confidence scoring    — avg confidence score per file
    ✅ Empty-chunk guard     — failed_count++ immediately on empty result
    ✅ Consistent logging    — emoji-coded, matches pdf_pipeline.py v4 format

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

_script_dir   = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Model sources that count as "NVIDIA succeeded"
_NVIDIA_SOURCES = {
    "page_elements",
    "ocr",
    "table_structure",
    "page_elements+ocr",
    "nvidia_smart",
}


def _count_model_sources(chunks) -> dict[str, int]:
    """Count chunks per model_source."""
    counts: dict[str, int] = {}
    for c in chunks:
        src = getattr(c, "model_source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def _nvidia_success_count(chunks) -> int:
    """Number of chunks produced by any NVIDIA model."""
    return sum(
        1 for c in chunks
        if getattr(c, "model_source", "") in _NVIDIA_SOURCES
    )


def _avg_confidence(chunks) -> float:
    """Average confidence_score across all chunks (v4 field)."""
    scores = [getattr(c, "confidence_score", 0.0) for c in chunks]
    return sum(scores) / len(scores) if scores else 0.0


def process_all(
    data_dir:     Path,
    processed_dir: Path,
    force:    bool = False,
    dry_run:  bool = False,
) -> dict:
    """
    Process every PDF in data_dir and its subdirectories.
    Returns a summary dict.
    """
    from pdf_pipeline import PDFPipeline

    pipeline = PDFPipeline(processed_dir=processed_dir)

    pdf_files = sorted(data_dir.rglob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0, "files": []}

    logger.info("Found %d PDF file(s)", len(pdf_files))

    processed_count = 0
    skipped_count   = 0
    failed_count    = 0
    total_chunks    = 0
    results         = []

    for pdf_path in pdf_files:
        logger.info("─" * 60)
        logger.info("Processing: %s", pdf_path.relative_to(data_dir))

        if dry_run:
            logger.info("[DRY RUN] Would process: %s", pdf_path.name)
            continue

        if force:
            cache_file = processed_dir / (pdf_path.stem + ".json")
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Deleted cache: %s", cache_file.name)

        t0 = time.perf_counter()
        try:
            chunks = pipeline.process(str(pdf_path))
            elapsed = (time.perf_counter() - t0) * 1000

            # ── EMPTY CHUNK GUARD ─────────────────────────────────────────
            if not chunks:
                logger.warning(
                    "  ⚠️  No chunks extracted (empty result) — %s", pdf_path.name
                )
                failed_count += 1
                continue

            # ── MODEL TRACKING ────────────────────────────────────────────
            model_counts  = _count_model_sources(chunks)
            nvidia_success = _nvidia_success_count(chunks)
            avg_conf       = _avg_confidence(chunks)

            # Count doc types
            type_counts: dict[str, int] = {}
            for c in chunks:
                type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1

            # ── NVIDIA SUCCESS SIGNAL ─────────────────────────────────────
            if nvidia_success == 0:
                logger.warning(
                    "  🚨 No NVIDIA extraction succeeded for %s  (model_sources=%s)",
                    pdf_path.name, model_counts,
                )
            elif set(model_counts.keys()).issubset({"pdfplumber"}):
                logger.warning(
                    "  ⚠️  Fallback-only extraction (no NVIDIA success) — %s",
                    pdf_path.name,
                )
            else:
                nvidia_pct = round(nvidia_success / len(chunks) * 100)
                logger.info(
                    "  ✅ NVIDIA extracted %d/%d chunks (%d%%)",
                    nvidia_success, len(chunks), nvidia_pct,
                )

            logger.info(
                "  ✅ %d chunks | %.0fms | bank=%s | types=%s | "
                "models=%s | avg_conf=%.1f",
                len(chunks), elapsed,
                chunks[0].bank if chunks else "Unknown",
                type_counts, model_counts, avg_conf,
            )

            results.append({
                "file":             pdf_path.name,
                "bank":             chunks[0].bank if chunks else "Unknown",
                "chunks":           len(chunks),
                "type_counts":      type_counts,
                "model_counts":     model_counts,
                "nvidia_success":   nvidia_success,
                "avg_confidence":   round(avg_conf, 2),
                "elapsed_ms":       round(elapsed),
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
    parser.add_argument("--force",      action="store_true", help="Reprocess even if cache exists")
    parser.add_argument("--dry-run",    action="store_true", help="Show what would be processed")
    parser.add_argument("--file",       type=str,            help="Process a single PDF file")
    parser.add_argument("--data-dir",   default=None,        help="Override data directory")
    parser.add_argument("--output-dir", default=None,        help="Override processed_data directory")
    args = parser.parse_args()

    from config import DATA_DIR, PROCESSED_DATA_DIR
    data_dir      = Path(args.data_dir)   if args.data_dir   else DATA_DIR
    processed_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Loan Advisor — PDF Processing Pipeline  (v4)")
    print("=" * 65)
    print(f"  Data dir:      {data_dir}")
    print(f"  Output dir:    {processed_dir}")
    print(f"  Force:         {args.force}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    if args.file:
        # ── Single file mode ──────────────────────────────────────────────
        from pdf_pipeline import PDFPipeline
        pipeline = PDFPipeline(processed_dir=processed_dir)

        if args.force:
            cache = processed_dir / (Path(args.file).stem + ".json")
            if cache.exists():
                cache.unlink()

        t0     = time.perf_counter()
        chunks = pipeline.process(args.file)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n{'='*65}")
        print(f"  Single-file result: {Path(args.file).name}")
        print(f"{'='*65}")

        # ── EMPTY CHUNK GUARD ─────────────────────────────────────────────
        if not chunks:
            print("  ⚠️  No chunks extracted (empty result)")
            return

        # ── MODEL TRACKING ────────────────────────────────────────────────
        model_counts   = _count_model_sources(chunks)
        nvidia_success = _nvidia_success_count(chunks)
        avg_conf       = _avg_confidence(chunks)
        type_counts    = {}
        for c in chunks:
            type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1

        print(f"  Chunks:          {len(chunks)}")
        print(f"  Bank:            {chunks[0].bank}")
        print(f"  Doc types:       {type_counts}")
        print(f"  Model sources:   {model_counts}")
        print(f"  Avg confidence:  {avg_conf:.1f}/10")
        print(f"  Elapsed:         {elapsed:.0f}ms")

        # ── NVIDIA SUCCESS SIGNAL ─────────────────────────────────────────
        if nvidia_success == 0:
            print(f"\n  🚨 WARNING: No NVIDIA extraction succeeded")
            print(f"     All chunks came from: {set(model_counts.keys())}")
            print(f"     Check NVIDIA_API_KEY and USE_NVIDIA_PDF settings")
        else:
            nvidia_pct = round(nvidia_success / len(chunks) * 100)
            print(f"\n  ✅ NVIDIA: {nvidia_success}/{len(chunks)} chunks ({nvidia_pct}%)")

        print(f"\n  Sample chunk:")
        print(f"    [{chunks[0].doc_type}|{chunks[0].model_source}] "
              f"{chunks[0].text[:150]}…")

        if len(chunks) > 1:
            print(f"\n  Last chunk:")
            print(f"    [{chunks[-1].doc_type}|{chunks[-1].model_source}] "
                  f"{chunks[-1].text[:150]}…")

    else:
        # ── Batch mode ────────────────────────────────────────────────────
        summary = process_all(
            data_dir=data_dir,
            processed_dir=processed_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

        print("\n" + "=" * 65)
        print("  Summary")
        print("=" * 65)
        print(f"  Processed:    {summary['processed']}")
        print(f"  Failed:       {summary['failed']}")
        print(f"  Total chunks: {summary['total_chunks']}")

        if summary["files"]:
            print("\n  Per-file breakdown:")
            print(f"  {'File':<38} {'Chunks':>6}  {'Bank':<10}  {'NVIDIA%':>7}  {'Conf':>5}  Models")
            print(f"  {'-'*38}  {'-'*6}  {'-'*10}  {'-'*7}  {'-'*5}  ------")

            total_nvidia = 0
            total_all    = 0
            for f in summary["files"]:
                nvidia_ok  = f.get("nvidia_success", 0)
                total_all_f = f.get("chunks", 1)
                nvidia_pct = round(nvidia_ok / max(total_all_f, 1) * 100)
                models_str = ", ".join(
                    f"{k}:{v}" for k, v in sorted(
                        f.get("model_counts", {}).items(),
                        key=lambda x: -x[1],
                    )
                )
                conf_str = f"{f.get('avg_confidence', 0.0):.1f}"
                nvidia_icon = "✅" if nvidia_ok > 0 else "🚨"
                print(
                    f"  {f['file']:<38} {total_all_f:>6}  {f['bank']:<10}  "
                    f"{nvidia_icon}{nvidia_pct:>4}%  {conf_str:>5}  {models_str}"
                )
                total_nvidia += nvidia_ok
                total_all    += total_all_f

            if total_all > 0:
                overall_pct = round(total_nvidia / total_all * 100)
                print(f"\n  Overall NVIDIA success: {total_nvidia}/{total_all} chunks ({overall_pct}%)")

        logger.info("=" * 65)
        logger.info("SUMMARY:")
        logger.info("  Processed : %d", summary["processed"])
        logger.info("  Skipped   : %d", summary["skipped"])
        logger.info("  Failed    : %d", summary["failed"])
        logger.info("  Chunks    : %d", summary["total_chunks"])
        logger.info("=" * 65)

        print()
        if summary["processed"] > 0:
            print("  ✅ Done. Run bootstrap.py to rebuild the FAISS index.")
        elif not args.dry_run:
            print("  ⚠️  No PDFs found. Add .pdf files to data/ subdirectories.")


if __name__ == "__main__":
    main()