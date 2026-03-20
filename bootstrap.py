#!/usr/bin/env python3
"""
Bootstrap Script v2
====================
Run this ONCE (and again after adding new documents) to:
  1. (Optional) Process PDFs via NVIDIA Page Elements API
  2. Load all documents (structured JSON + plain text)
  3. Build and save the FAISS index

Usage:
    # Standard bootstrap (plain text extraction for PDFs)
    python bootstrap.py

    # With NVIDIA PDF processing (USE_NVIDIA_PDF=true in .env)
    python bootstrap.py --process-pdfs

    # Force reprocess all PDFs
    python bootstrap.py --process-pdfs --force
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(__file__))

from config import DATA_DIR, PROCESSED_DATA_DIR

def main():
    parser = argparse.ArgumentParser(description="Loan Advisor bootstrap")
    parser.add_argument("--process-pdfs", action="store_true",
                        help="Run PDF pipeline before indexing (USE_NVIDIA_PDF controls NVIDIA vs fallback)")
    parser.add_argument("--force",        action="store_true",
                        help="Force reprocess PDFs even if cached")
    args = parser.parse_args()

    print("=" * 60)
    print("Loan Policy Advisor — Bootstrap v2")
    print("=" * 60)

    # ── Step 1: PDF processing (optional) ─────────────────────────────────
    if args.process_pdfs:
        from pdf_pipeline import PDFPipeline
        from config import USE_NVIDIA_PDF

        pdf_files = list(DATA_DIR.rglob("*.pdf"))
        if pdf_files:
            print(f"\n[1] Processing {len(pdf_files)} PDF file(s)…")
            mode = "NVIDIA Page Elements" if USE_NVIDIA_PDF else "pdfplumber fallback"
            print(f"    Mode: {mode}")

            pipeline = PDFPipeline(processed_dir=PROCESSED_DATA_DIR)
            total_chunks = 0
            for pdf_path in sorted(pdf_files):
                if args.force:
                    cache = PROCESSED_DATA_DIR / (pdf_path.stem + ".json")
                    if cache.exists():
                        cache.unlink()
                try:
                    chunks = pipeline.process(str(pdf_path))
                    print(f"    ✅ {pdf_path.name}: {len(chunks)} chunks [{chunks[0].bank if chunks else 'Unknown'}]")
                    total_chunks += len(chunks)
                except Exception as e:
                    print(f"    ❌ {pdf_path.name}: {e}")
            print(f"    Total: {total_chunks} structured chunks")
        else:
            print("\n[1] No PDF files found — skipping PDF processing")
    else:
        step = "[1]" if args.process_pdfs else "[1]"
        print(f"\n{step} PDF processing skipped (pass --process-pdfs to enable)")

    # ── Step 2: Load documents ─────────────────────────────────────────────
    from services.retrieval import RetrievalService

    print("\n[2] Loading documents (structured JSON + plain text)…")
    retrieval = RetrievalService()
    count     = retrieval.load_documents()

    n_folders = len(list(DATA_DIR.iterdir()))
    n_json    = len(list(PROCESSED_DATA_DIR.glob("*.json"))) if PROCESSED_DATA_DIR.exists() else 0
    print(f"    Loaded {count} chunks  ({n_json} structured JSON + plain text from {n_folders} folders)")

    if count == 0:
        print("\n⚠️  No documents found. Make sure data/ folders contain .txt or .pdf files.")
        return

    # ── Step 3: Build FAISS index ──────────────────────────────────────────
    print("\n[3] Building FAISS index…")
    retrieval.build_index(save=True)

    print("\n✅ Bootstrap complete!")
    print(f"   Index saved to models/index.faiss")
    print(f"   Total chunks: {count}")
    print("\nStart the API:")
    print("   uvicorn api.main:app --reload --port 8000")

if __name__ == "__main__":
    main()
