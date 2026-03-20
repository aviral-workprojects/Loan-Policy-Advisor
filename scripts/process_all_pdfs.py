#!/usr/bin/env python3
"""
scripts/process_all_pdfs.py  (v4.2)
=====================================
All previous features retained, plus:
    ✅ Fix 9: True NVIDIA observability
               — nvidia_success = non-pdfplumber chunks
               — explicit ratio logged on every file
               — global summary shows overall NVIDIA% across all files
               — 🚨 warning fires when nvidia_success == 0
               — ⚠️  warning fires when NVIDIA% < 50%

Usage:
    python scripts/process_all_pdfs.py
    python scripts/process_all_pdfs.py --force
    python scripts/process_all_pdfs.py --file data/hdfc_pdfs/hdfc_personal_loan.pdf
    python scripts/process_all_pdfs.py --dry-run
"""

import sys
import os
import argparse
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

# Fix 9: model sources that count as "NVIDIA succeeded"
_NVIDIA_SOURCES = {
    "page_elements",
    "ocr",
    "table_structure",
    "page_elements+ocr",
    "nvidia_smart",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_model_sources(chunks) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chunks:
        src = getattr(c, "model_source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return counts


def _nvidia_success_count(chunks) -> int:
    """Fix 9: count chunks produced by any NVIDIA model."""
    return sum(1 for c in chunks if getattr(c, "model_source", "") in _NVIDIA_SOURCES)


def _avg_confidence(chunks) -> float:
    scores = [getattr(c, "confidence_score", 0.0) for c in chunks]
    return sum(scores) / len(scores) if scores else 0.0


def _log_nvidia_signal(pdf_name: str, nvidia_ok: int, total: int, model_counts: dict) -> None:
    """Fix 9: emit clear NVIDIA success/failure signal."""
    if total == 0:
        return
    pct = round(nvidia_ok / total * 100)
    if nvidia_ok == 0:
        logger.warning(
            "  🚨 No NVIDIA extraction succeeded for %s  (model_sources=%s)",
            pdf_name, model_counts,
        )
    elif pct < 50:
        logger.warning(
            "  ⚠️  Low NVIDIA rate for %s: %d/%d chunks (%d%%)  (model_sources=%s)",
            pdf_name, nvidia_ok, total, pct, model_counts,
        )
    else:
        logger.info(
            "  ✅ NVIDIA: %d/%d chunks (%d%%)  models=%s",
            nvidia_ok, total, pct, model_counts,
        )


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_all(
    data_dir:     Path,
    processed_dir: Path,
    force:    bool = False,
    dry_run:  bool = False,
) -> dict:
    from pdf_pipeline import PDFPipeline

    pipeline  = PDFPipeline(processed_dir=processed_dir)
    pdf_files = sorted(data_dir.rglob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return {"processed": 0, "skipped": 0, "failed": 0, "total_chunks": 0,
                "total_nvidia": 0, "files": []}

    logger.info("Found %d PDF file(s)", len(pdf_files))

    processed_count = 0
    failed_count    = 0
    total_chunks    = 0
    total_nvidia    = 0   # Fix 9: global nvidia counter
    results         = []

    for pdf_path in pdf_files:
        logger.info("─" * 65)
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
            chunks  = pipeline.process(str(pdf_path))
            elapsed = (time.perf_counter() - t0) * 1000

            # ── Empty chunk guard ─────────────────────────────────────────
            if not chunks:
                logger.warning("  ⚠️  No chunks extracted — %s", pdf_path.name)
                failed_count += 1
                continue

            # ── Model tracking ────────────────────────────────────────────
            model_counts   = _count_model_sources(chunks)
            nvidia_ok      = _nvidia_success_count(chunks)   # Fix 9
            avg_conf       = _avg_confidence(chunks)

            type_counts: dict[str, int] = {}
            for c in chunks:
                type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1

            # Fix 9: emit the clear signal
            _log_nvidia_signal(pdf_path.name, nvidia_ok, len(chunks), model_counts)

            # Fix 9: detailed log line with nvidia_success/total
            logger.info(
                "  📄 %d chunks | %.0fms | bank=%s | types=%s | "
                "models=%s | nvidia=%d/%d | conf=%.1f",
                len(chunks), elapsed,
                chunks[0].bank if chunks else "Unknown",
                type_counts, model_counts,
                nvidia_ok, len(chunks),
                avg_conf,
            )

            results.append({
                "file":           pdf_path.name,
                "bank":           chunks[0].bank if chunks else "Unknown",
                "chunks":         len(chunks),
                "type_counts":    type_counts,
                "model_counts":   model_counts,
                "nvidia_success": nvidia_ok,
                "nvidia_pct":     round(nvidia_ok / len(chunks) * 100),
                "avg_confidence": round(avg_conf, 2),
                "elapsed_ms":     round(elapsed),
            })

            processed_count += 1
            total_chunks    += len(chunks)
            total_nvidia    += nvidia_ok   # Fix 9

        except Exception as e:
            logger.error("  ❌ Failed: %s — %s", pdf_path.name, e, exc_info=True)
            failed_count += 1

    return {
        "processed":    processed_count,
        "skipped":      0,
        "failed":       failed_count,
        "total_chunks": total_chunks,
        "total_nvidia": total_nvidia,   # Fix 9
        "files":        results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch PDF processing — Loan Advisor v4.2")
    parser.add_argument("--force",      action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--file",       type=str, default=None)
    parser.add_argument("--data-dir",   default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    from config import DATA_DIR, PROCESSED_DATA_DIR
    data_dir      = Path(args.data_dir)   if args.data_dir   else DATA_DIR
    processed_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Loan Advisor — PDF Processing Pipeline  (v4.2)")
    print("=" * 65)
    print(f"  Data dir:      {data_dir}")
    print(f"  Output dir:    {processed_dir}")
    print(f"  Force:         {args.force}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    if args.file:
        # ── Single-file mode ──────────────────────────────────────────────
        from pdf_pipeline import PDFPipeline
        pipeline = PDFPipeline(processed_dir=processed_dir)

        if args.force:
            cache = processed_dir / (Path(args.file).stem + ".json")
            if cache.exists():
                cache.unlink()

        t0      = time.perf_counter()
        chunks  = pipeline.process(args.file)
        elapsed = (time.perf_counter() - t0) * 1000

        print(f"\n{'='*65}")
        print(f"  {Path(args.file).name}")
        print(f"{'='*65}")

        if not chunks:
            print("  ⚠️  No chunks extracted")
            return

        model_counts  = _count_model_sources(chunks)
        nvidia_ok     = _nvidia_success_count(chunks)   # Fix 9
        avg_conf      = _avg_confidence(chunks)
        type_counts   = {}
        for c in chunks:
            type_counts[c.doc_type] = type_counts.get(c.doc_type, 0) + 1

        nvidia_pct = round(nvidia_ok / len(chunks) * 100)

        print(f"  Chunks:        {len(chunks)}")
        print(f"  Bank:          {chunks[0].bank}")
        print(f"  Doc types:     {type_counts}")
        print(f"  Model sources: {model_counts}")
        print(f"  Elapsed:       {elapsed:.0f}ms")
        print(f"  Avg conf:      {avg_conf:.1f}/10")

        # Fix 9: clear NVIDIA signal
        print()
        if nvidia_ok == 0:
            print(f"  🚨 NVIDIA: 0/{len(chunks)} chunks (0%) — all from pdfplumber")
            print(f"     Check: NVIDIA_API_KEY set? USE_NVIDIA_PDF=true?")
        elif nvidia_pct < 50:
            print(f"  ⚠️  NVIDIA: {nvidia_ok}/{len(chunks)} ({nvidia_pct}%) — below 50%")
        else:
            print(f"  ✅ NVIDIA: {nvidia_ok}/{len(chunks)} ({nvidia_pct}%)")

        print(f"\n  Sample:")
        print(f"    [{chunks[0].doc_type}|{chunks[0].model_source}] {chunks[0].text[:150]}")

    else:
        # ── Batch mode ────────────────────────────────────────────────────
        summary = process_all(
            data_dir=data_dir,
            processed_dir=processed_dir,
            force=args.force,
            dry_run=args.dry_run,
        )

        total    = summary["total_chunks"]
        nv_total = summary["total_nvidia"]   # Fix 9

        print("\n" + "=" * 65)
        print("  Summary")
        print("=" * 65)
        print(f"  Processed:    {summary['processed']}")
        print(f"  Failed:       {summary['failed']}")
        print(f"  Total chunks: {total}")

        # Fix 9: global NVIDIA success rate
        if total > 0:
            global_pct = round(nv_total / total * 100)
            flag = "✅" if global_pct >= 50 else ("⚠️ " if global_pct > 0 else "🚨")
            print(f"  NVIDIA total: {flag} {nv_total}/{total} ({global_pct}%)")

        if summary["files"]:
            print()
            print(f"  {'File':<36} {'Chunks':>6}  {'NVIDIA%':>7}  {'Conf':>5}  Models")
            print(f"  {'-'*36}  {'-'*6}  {'-'*7}  {'-'*5}  ------")

            for f in summary["files"]:
                nv_ok  = f.get("nvidia_success", 0)
                nv_pct = f.get("nvidia_pct", 0)
                flag   = "✅" if nv_pct >= 50 else ("⚠️ " if nv_ok > 0 else "🚨")
                models = ", ".join(
                    f"{k}:{v}" for k, v in
                    sorted(f.get("model_counts", {}).items(), key=lambda x: -x[1])
                )
                print(
                    f"  {f['file']:<36} {f['chunks']:>6}  "
                    f"{flag}{nv_pct:>4}%  {f.get('avg_confidence',0):>5.1f}  {models}"
                )

        print()
        if summary["processed"] > 0:
            print("  ✅ Done. Run: python bootstrap.py to rebuild FAISS + BM25 index.")
        elif not args.dry_run:
            print("  ⚠️  No PDFs found. Add .pdf files to data/ subdirectories.")

        # Logger summary
        logger.info("=" * 65)
        logger.info("SUMMARY  processed=%d  failed=%d  chunks=%d  nvidia=%d/%d",
                    summary["processed"], summary["failed"],
                    total, nv_total, total)
        logger.info("=" * 65)


if __name__ == "__main__":
    main()