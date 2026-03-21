#!/usr/bin/env python3
"""
Bootstrap v3 — Hybrid Extraction Pipeline
==========================================
Processes all PDFs using the new hybrid stack, builds FAISS + BM25 index.

Usage:
    python bootstrap.py                   # standard run
    python bootstrap.py --force           # reprocess all PDFs
    python bootstrap.py --file path.pdf   # single file mode
    python bootstrap.py --check-deps      # verify optional dependencies
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def check_dependencies() -> None:
    """Print availability of all optional dependencies."""
    print("\n── Dependency Check ──────────────────────────────────")

    def check(name, import_str):
        try:
            exec(f"import {import_str}")
            print(f"  ✅  {name}")
        except ImportError:
            print(f"  ❌  {name}  (optional)")

    check("pdfplumber",     "pdfplumber")
    check("PyMuPDF (fitz)", "fitz")
    check("Camelot",        "camelot")
    check("Tabula",         "tabula")
    check("PaddleOCR",      "paddleocr")
    check("Tesseract",      "pytesseract")
    check("FAISS",          "faiss")
    check("sentence-transformers", "sentence_transformers")
    check("NumPy",          "numpy")

    from pdf_pipeline.ocr           import ocr_availability
    from pdf_pipeline.table_extractor import table_availability

    print("\n  OCR backends:   ", ocr_availability())
    print("  Table backends: ", table_availability())
    print()


def process_pdfs(data_dir: Path, processed_dir: Path, force: bool = False) -> list:
    """
    Process all PDFs in data_dir using the hybrid extraction stack.
    Returns list of DocumentChunk objects ready for indexing.
    """
    from pdf_pipeline.extractor import extract_pdf
    from pdf_pipeline.chunker   import chunk_document
    from pdf_pipeline.retriever import DocChunk

    pdf_files = sorted(data_dir.rglob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", data_dir)
        return []

    logger.info("Found %d PDF files", len(pdf_files))

    all_chunks: list[DocChunk] = []
    processed_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        cache_path = processed_dir / (pdf_path.stem + ".json")

        # Check cache
        if cache_path.exists() and not force:
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_chunks = [DocChunk(**c) for c in data["chunks"]]
                logger.info("[Cache] %-40s %d chunks", pdf_path.name, len(cached_chunks))
                all_chunks.extend(cached_chunks)
                continue
            except Exception as e:
                logger.warning("[Cache] Failed to load %s: %s — reprocessing", cache_path.name, e)

        # Detect bank from path
        bank = _detect_bank(pdf_path)

        t0 = time.perf_counter()
        try:
            doc    = extract_pdf(pdf_path, bank=bank)
            chunks = chunk_document(doc)
        except Exception as e:
            logger.error("❌  %s — %s", pdf_path.name, e)
            continue

        elapsed = (time.perf_counter() - t0) * 1000

        # Convert to DocChunk and record stats
        doc_chunks = [
            DocChunk(
                text=c.text,
                source=pdf_path.name,
                bank=c.bank,
                doc_type=c.doc_type,
                chunk_id=c.chunk_id,
                field_hint=c.field_hint,
                section_title=c.section_title,
                element_type=c.element_type,
                page_number=c.page_number,
            )
            for c in chunks
        ]

        # Save to cache
        try:
            cache_data = {
                "source":    pdf_path.name,
                "bank":      bank,
                "is_scanned": doc.is_scanned,
                "method":    doc.extraction_method,
                "chunks":    [
                    {
                        "text":           dc.text,
                        "source":         dc.source,
                        "bank":           dc.bank,
                        "doc_type":       dc.doc_type,
                        "chunk_id":       dc.chunk_id,
                        "field_hint":     dc.field_hint,
                        "section_title":  dc.section_title,
                        "element_type":   dc.element_type,
                        "page_number":    dc.page_number,
                        "similarity_score": 0.0,
                    }
                    for dc in doc_chunks
                ],
            }
            cache_path.write_text(
                json.dumps(cache_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning("[Cache] Failed to save %s: %s", cache_path.name, e)

        method_icon = "📄" if not doc.is_scanned else "🔍"
        logger.info(
            "%s  %-40s  %d chunks  %.0fms  bank=%-8s  method=%s",
            method_icon, pdf_path.name, len(doc_chunks), elapsed,
            bank, doc.extraction_method,
        )

        all_chunks.extend(doc_chunks)

    return all_chunks


def build_index(chunks, index_dir: Path) -> None:
    """
    Build FAISS + BM25 index from chunks and save to disk.

    Detects the actual embedding dimension from the embedder that was
    selected (NVIDIA=1024, local sentence-transformers=384). This prevents
    a dim mismatch crash when NVIDIA fails and the fallback kicks in.
    """
    from pdf_pipeline.retriever  import RetrievalService
    from pdf_pipeline.embeddings import get_embedder

    # Build embedder first so we know the real dim before creating the index
    embedder = get_embedder()

    # Probe dimension with a single test encode
    try:
        test_vec = embedder.encode(["test"], normalize_embeddings=False)
        actual_dim = test_vec.shape[1]
        logger.info("✅  Embedder ready — dim=%d", actual_dim)
    except Exception as e:
        logger.error("❌  Cannot determine embedding dimension: %s", e)
        raise

    svc = RetrievalService(index_dir=index_dir, embed_dim=actual_dim)
    svc._embedder = embedder   # inject the already-initialised embedder
    svc.load_chunks(chunks)
    svc.build_index(save=True)
    logger.info("✅  Index saved → %s  (%d chunks, dim=%d)", index_dir, len(chunks), actual_dim)


def _detect_bank(pdf_path: Path) -> str:
    _MAP = {"hdfc": "HDFC", "sbi": "SBI", "icici": "ICICI", "axis": "Axis", "rbi": "RBI"}
    name = (pdf_path.parent.name + " " + pdf_path.stem).lower()
    for key, bank in _MAP.items():
        if key in name:
            return bank
    return "Unknown"


# ---------------------------------------------------------------------------
# .txt file loader  (eligibility.txt, comparison.txt, guidelines.txt, etc.)
# ---------------------------------------------------------------------------

# Maps data subfolder name → (canonical bank, doc_type)
_FOLDER_BANK_MAP = {
    "axis":        ("Axis",        "eligibility"),
    "hdfc_pdfs":   ("HDFC",        "eligibility"),
    "icici":       ("ICICI",       "eligibility"),
    "sbi_pdfs":    ("SBI",         "eligibility"),
    "bankbazaar":  ("BankBazaar",  "comparison"),
    "paisabazaar": ("Paisabazaar", "comparison"),
    "rbi":         ("RBI",         "regulatory"),
    "misc":        ("Unknown",     "pdf"),
}

def load_text_files(data_dir: Path) -> list:
    """
    Load all .txt files from data/ subfolders and chunk them.

    These are the curated knowledge files (eligibility.txt, comparison.txt,
    guidelines.txt, sbi_xpress_credit.txt, etc.) that were in the repo before
    any scraping happened. They contain high-quality hand-curated information
    that must be indexed alongside PDFs.

    Returns list of DocChunk objects.
    """
    from pdf_pipeline.retriever import DocChunk
    from pdf_pipeline.chunker   import CHUNK_WORDS, OVERLAP_WORDS, _split_text, detect_field_hint

    chunks: list[DocChunk] = []
    chunk_id = 0
    txt_files_found = 0

    for folder_path in sorted(data_dir.iterdir()):
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name.lower()
        bank, doc_type = _FOLDER_BANK_MAP.get(folder_name, ("Unknown", "pdf"))

        for txt_path in sorted(folder_path.glob("*.txt")):
            try:
                text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not text:
                    continue

                txt_files_found += 1
                text_chunks = _split_text(text, CHUNK_WORDS, OVERLAP_WORDS)

                for tc in text_chunks:
                    if len(tc.split()) < 15:
                        continue
                    chunks.append(DocChunk(
                        text=tc,
                        source=txt_path.name,
                        bank=bank,
                        doc_type=doc_type,
                        chunk_id=chunk_id,
                        field_hint=detect_field_hint(tc),
                    ))
                    chunk_id += 1

                logger.info("📝  %-40s  %d chunks  bank=%s",
                            txt_path.name,
                            len([c for c in chunks if c.source == txt_path.name]),
                            bank)
            except Exception as e:
                logger.warning("[TXT] Failed to read %s: %s", txt_path.name, e)

    logger.info("[TXT] Loaded %d .txt files → %d chunks", txt_files_found, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Scraped JSON loader  (*_scraped.json written by run_scraper.py)
# ---------------------------------------------------------------------------

def load_scraped_json(data_dir: Path) -> list:
    """
    Load all *_scraped.json files written by run_scraper.py.

    Each scraped JSON contains a LoanRecord with structured financial data
    and a pre-built RAG text string (to_rag_text()). We index the RAG text
    directly — it is already chunked and natural-language formatted.

    Returns list of DocChunk objects.
    """
    from pdf_pipeline.retriever import DocChunk
    from pdf_pipeline.chunker   import detect_field_hint

    chunks: list[DocChunk] = []
    chunk_id = 0
    json_files_found = 0

    for folder_path in sorted(data_dir.iterdir()):
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name.lower()
        _, doc_type = _FOLDER_BANK_MAP.get(folder_name, ("Unknown", "pdf"))

        for json_path in sorted(folder_path.glob("*_scraped.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                bank    = data.get("bank", "Unknown") or "Unknown"
                content = data.get("content", "").strip()

                if not content or len(content) < 50:
                    continue

                json_files_found += 1

                # Use the pre-built RAG text if available (richer than raw content)
                # Fall back to content field
                rag_text = content

                # Also append key_facts and features if present
                extras = []
                for feat in data.get("features", [])[:5]:
                    if feat and len(feat) > 20:
                        extras.append(feat)
                for fact in data.get("key_facts", [])[:5]:
                    if fact and len(fact) > 20:
                        extras.append(fact)
                if extras:
                    rag_text = rag_text + " " + " ".join(extras)

                chunks.append(DocChunk(
                    text=rag_text[:2000],
                    source=json_path.name,
                    bank=bank,
                    doc_type="comparison" if bank in ("Paisabazaar", "BankBazaar") else "eligibility",
                    chunk_id=chunk_id,
                    field_hint=detect_field_hint(rag_text),
                ))
                chunk_id += 1

            except Exception as e:
                logger.warning("[ScrapedJSON] Failed to read %s: %s", json_path.name, e)

    logger.info("[ScrapedJSON] Loaded %d *_scraped.json files → %d chunks",
                json_files_found, len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Loan Advisor Bootstrap v3")
    parser.add_argument("--force",       action="store_true", help="Reprocess all PDFs")
    parser.add_argument("--file",        type=str,            help="Process single PDF")
    parser.add_argument("--check-deps",  action="store_true", help="Check optional dependencies")
    parser.add_argument("--data-dir",    default=None)
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    if args.check_deps:
        check_dependencies()
        return

    from config import DATA_DIR, PROCESSED_DATA_DIR, FAISS_INDEX_DIR

    data_dir      = Path(args.data_dir)   if args.data_dir   else DATA_DIR
    processed_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DATA_DIR

    print("=" * 65)
    print("  Loan Advisor — Bootstrap v3 (Hybrid Extraction)")
    print("=" * 65)
    print(f"  Data dir:    {data_dir}")
    print(f"  Output dir:  {processed_dir}")
    print(f"  Index dir:   {FAISS_INDEX_DIR}")
    print(f"  Force:       {args.force}")
    print()

    if args.file:
        # Single file mode
        from pdf_pipeline.extractor import extract_pdf
        from pdf_pipeline.chunker   import chunk_document

        pdf_path = Path(args.file)
        bank = _detect_bank(pdf_path)
        doc  = extract_pdf(pdf_path, bank=bank)
        chunks = chunk_document(doc)

        print(f"\n  File:     {pdf_path.name}")
        print(f"  Bank:     {bank}")
        print(f"  Scanned:  {doc.is_scanned}")
        print(f"  Method:   {doc.extraction_method}")
        print(f"  Pages:    {len(doc.pages)}")
        print(f"  Chunks:   {len(chunks)}")
        print(f"  Chars:    {doc.total_chars}")
        if chunks:
            print(f"\n  Sample chunk [p{chunks[0].page_number}|{chunks[0].doc_type}]:")
            print(f"  {chunks[0].text[:200]}")
        return

    # Batch mode
    print("[1] Extracting PDFs…")
    pdf_chunks = process_pdfs(data_dir, processed_dir, force=args.force)

    print("\n[1b] Loading .txt files…")
    txt_chunks = load_text_files(data_dir)

    print("\n[1c] Loading scraped JSON files (*_scraped.json)…")
    scraped_chunks = load_scraped_json(data_dir)

    # Merge all chunk sources and re-assign sequential IDs
    all_chunks = pdf_chunks + txt_chunks + scraped_chunks
    for i, c in enumerate(all_chunks):
        c.chunk_id = i

    if not all_chunks:
        print("\n⚠️  No content found. Add PDF or .txt files to the data/ directory.")
        return

    print(f"\n    PDF chunks:     {len(pdf_chunks)}")
    print(f"    TXT chunks:     {len(txt_chunks)}")
    print(f"    Scraped chunks: {len(scraped_chunks)}")
    print(f"    ─────────────────────────")
    print(f"    Total chunks:   {len(all_chunks)}")

    print("\n[2] Building FAISS + BM25 index…")
    build_index(all_chunks, FAISS_INDEX_DIR)

    print("\n✅  Bootstrap complete!")
    print("\nStart the API:")
    print("   uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()