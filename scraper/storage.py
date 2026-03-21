"""
scraper/storage.py
===================
Writes validated LoanRecord objects to disk in the correct data/ subfolder.

Output per record:
  data/<bank_folder>/<slug>_scraped.json   — full structured record
  data/<bank_folder>/<slug>_scraped.txt    — RAG-ready natural language

The .txt file is what the bootstrap pipeline picks up and indexes into FAISS.
JSON is for debugging, reprocessing, and rule engine updates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

from scraper.schema import LoanRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Folder mapping: canonical bank name → data subfolder
# ---------------------------------------------------------------------------

_BANK_FOLDER_MAP = {
    "Axis":        "axis",
    "HDFC":        "hdfc_pdfs",
    "ICICI":       "icici",
    "SBI":         "sbi_pdfs",
    "Paisabazaar": "paisabazaar",
    "BankBazaar":  "bankbazaar",
    "RBI":         "rbi",
    "Unknown":     "misc",
}


def _bank_folder(bank: str) -> str:
    return _BANK_FOLDER_MAP.get(bank, "misc")


def _slug(record: LoanRecord) -> str:
    """Generate a filesystem-safe filename slug for a record."""
    parts = [
        record.bank.lower().replace(" ", "_"),
        record.loan_type,
    ]
    # Add a short hash of the URL to make it unique
    url_hash = hashlib.md5(record.source_url.encode()).hexdigest()[:8]
    parts.append(url_hash)
    slug = "_".join(p for p in parts if p)
    return re.sub(r"[^\w\-]", "_", slug)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class StorageWriter:

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def write(self, record: LoanRecord) -> tuple[Path, Path]:
        """
        Write a LoanRecord to disk.

        Returns:
            (json_path, txt_path)
        """
        folder = self.data_dir / _bank_folder(record.bank)
        folder.mkdir(parents=True, exist_ok=True)

        slug = _slug(record)
        json_path = folder / f"{slug}_scraped.json"
        txt_path  = folder / f"{slug}_scraped.txt"

        # Write JSON
        data = record.to_dict()
        data["_written_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        json_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write TXT (RAG-ready)
        rag_text = record.to_rag_text()
        txt_path.write_text(rag_text, encoding="utf-8")

        logger.info("[Storage] ✓ %s  →  %s  (%d chars)",
                    record.bank, json_path.name, len(rag_text))
        return json_path, txt_path

    def write_batch(self, records: list[LoanRecord]) -> list[tuple[Path, Path]]:
        results = []
        for r in records:
            try:
                paths = self.write(r)
                results.append(paths)
            except Exception as e:
                logger.error("[Storage] Failed to write %s: %s", r.bank, e)
        return results

    def write_summary(self, records: list[LoanRecord]) -> Path:
        """Write a summary JSON of all scraped records for quick inspection."""
        summary = {
            "total": len(records),
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "banks": {},
        }
        for r in records:
            b = r.bank
            if b not in summary["banks"]:
                summary["banks"][b] = {
                    "count": 0,
                    "has_rate": 0,
                    "has_income": 0,
                    "has_cibil": 0,
                    "avg_confidence": 0.0,
                    "urls": [],
                }
            entry = summary["banks"][b]
            entry["count"] += 1
            entry["has_rate"]   += bool(r.interest_rate)
            entry["has_income"] += bool(r.min_income)
            entry["has_cibil"]  += bool(r.min_cibil)
            entry["avg_confidence"] = round(
                (entry["avg_confidence"] * (entry["count"] - 1) + r.confidence) / entry["count"], 2
            )
            entry["urls"].append(r.source_url)

        summary_path = self.data_dir / "scrape_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("[Storage] Summary → %s", summary_path)
        return summary_path
