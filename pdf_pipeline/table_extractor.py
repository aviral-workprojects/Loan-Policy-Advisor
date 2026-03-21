"""
pdf_pipeline/table_extractor.py
================================
Table extraction with structured → natural language conversion.

Why tables matter for RAG:
  Bank PDFs contain critical information in tables:
    - Eligibility criteria grids (salary vs CIBIL vs tenure)
    - Interest rate slabs
    - EMI calculation tables
    - Fee schedules

  Raw table cells are nearly useless for LLMs. Converting them to
  natural language sentences ("For salary ₹30,000 the interest rate
  is 8.5%") dramatically improves retrieval accuracy and answer quality.

Stack:
  Camelot  → primary  (lattice mode for bordered tables, stream for borderless)
  Tabula   → fallback (Java-based, different parsing approach)
  pdfplumber → last resort (simple, less accurate for complex tables)

Install:
  pip install camelot-py[cv]   # needs opencv-python + ghostscript
  pip install tabula-py        # needs Java runtime
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_camelot_available = False
_tabula_available = False

try:
    import camelot
    _camelot_available = True
except ImportError:
    pass

try:
    import tabula
    _tabula_available = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def extract_tables_from_page(pdf_path: Path, page_number: int) -> list[dict]:
    """
    Extract all tables from a single PDF page.

    Returns list of dicts, each with:
      {
        "headers":  ["Column A", "Column B", ...],
        "rows":     [["val1", "val2"], ...],
        "raw_text": "natural language version",
        "page":     page_number,
        "method":   "camelot_lattice" | "camelot_stream" | "tabula" | "pdfplumber",
      }
    """
    tables = _try_camelot(pdf_path, page_number)
    if not tables and _tabula_available:
        tables = _try_tabula(pdf_path, page_number)
    if not tables:
        tables = _try_pdfplumber(pdf_path, page_number)
    return tables


def extract_all_tables(pdf_path: Path) -> list[dict]:
    """Extract tables from all pages of a PDF."""
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            n_pages = len(pdf.pages)
    except Exception:
        n_pages = 50   # safe upper bound

    all_tables = []
    for page_num in range(1, n_pages + 1):
        tables = extract_tables_from_page(pdf_path, page_num)
        all_tables.extend(tables)

    logger.info("[Tables] %s — extracted %d tables total", pdf_path.name, len(all_tables))
    return all_tables


# ---------------------------------------------------------------------------
# Camelot (primary)
# ---------------------------------------------------------------------------

def _try_camelot(pdf_path: Path, page_number: int) -> list[dict]:
    if not _camelot_available:
        return []

    tables = []

    # Try lattice mode first (works best when table has visible borders)
    for flavor in ("lattice", "stream"):
        try:
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_number),
                flavor=flavor,
                suppress_stdout=True,
            )
            for tbl in camelot_tables:
                if tbl.accuracy < 50:   # too many parsing errors — skip
                    continue
                df = tbl.df
                if df.empty:
                    continue
                parsed = _parse_dataframe(df, page_number, f"camelot_{flavor}")
                if parsed:
                    tables.append(parsed)

            if tables:
                logger.debug("[Tables] Camelot %s found %d table(s) on p%d",
                             flavor, len(tables), page_number)
                return tables

        except Exception as e:
            logger.debug("[Tables] Camelot %s failed p%d: %s", flavor, page_number, e)

    return tables


# ---------------------------------------------------------------------------
# Tabula (fallback)
# ---------------------------------------------------------------------------

def _try_tabula(pdf_path: Path, page_number: int) -> list[dict]:
    try:
        dfs = tabula.read_pdf(
            str(pdf_path),
            pages=page_number,
            multiple_tables=True,
            silent=True,
        )
        tables = []
        for df in dfs:
            if df is None or df.empty:
                continue
            parsed = _parse_dataframe(df, page_number, "tabula")
            if parsed:
                tables.append(parsed)
        if tables:
            logger.debug("[Tables] Tabula found %d table(s) on p%d", len(tables), page_number)
        return tables
    except Exception as e:
        logger.debug("[Tables] Tabula failed p%d: %s", page_number, e)
        return []


# ---------------------------------------------------------------------------
# pdfplumber (last resort)
# ---------------------------------------------------------------------------

def _try_pdfplumber(pdf_path: Path, page_number: int) -> list[dict]:
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number > len(pdf.pages):
                return []
            page = pdf.pages[page_number - 1]
            raw_tables = page.extract_tables()
            if not raw_tables:
                return []
            results = []
            for raw in raw_tables:
                if not raw or len(raw) < 2:
                    continue
                headers = [str(c or "").strip() for c in raw[0]]
                rows = [[str(c or "").strip() for c in row] for row in raw[1:]]
                raw_text = _table_to_natural_language(headers, rows)
                if raw_text.strip():
                    results.append({
                        "headers":  headers,
                        "rows":     rows,
                        "raw_text": raw_text,
                        "page":     page_number,
                        "method":   "pdfplumber",
                    })
            return results
    except Exception as e:
        logger.debug("[Tables] pdfplumber table extraction failed p%d: %s", page_number, e)
        return []


# ---------------------------------------------------------------------------
# DataFrame → structured dict
# ---------------------------------------------------------------------------

def _parse_dataframe(df, page_number: int, method: str) -> dict | None:
    """Convert a pandas DataFrame to our standard table dict format."""
    try:
        # First row as headers if it looks like a header row
        raw_headers = [str(c).strip() for c in df.columns.tolist()]
        if all(h.isdigit() or h.startswith("Unnamed") for h in raw_headers):
            # No meaningful column names — use first data row as headers
            raw_headers = [str(c).strip() for c in df.iloc[0].tolist()]
            data_rows = df.iloc[1:].values.tolist()
        else:
            data_rows = df.values.tolist()

        headers = [_clean_cell(h) for h in raw_headers]
        rows = [[_clean_cell(str(c)) for c in row] for row in data_rows]

        # Filter fully empty rows
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            return None

        raw_text = _table_to_natural_language(headers, rows)
        if not raw_text.strip():
            return None

        return {
            "headers":  headers,
            "rows":     rows,
            "raw_text": raw_text,
            "page":     page_number,
            "method":   method,
        }
    except Exception as e:
        logger.debug("[Tables] DataFrame parse failed: %s", e)
        return None


def _clean_cell(text: str) -> str:
    """Normalise a table cell value."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("\n", " ").replace("\r", "")
    return text if text.lower() not in ("nan", "none", "-", "") else ""


# ---------------------------------------------------------------------------
# Table → natural language (CRITICAL for RAG quality)
# ---------------------------------------------------------------------------

def _table_to_natural_language(headers: list[str], rows: list[list[str]]) -> str:
    """
    Convert a table into natural language sentences.

    This is one of the highest-impact improvements for RAG accuracy.
    LLMs retrieve and reason far better over prose than raw table cells.

    Examples:
      headers: ["Salary Range", "Min CIBIL", "Max Loan"]
      row:     ["₹25,000–₹50,000", "700", "₹10 lakh"]
      → "For salary range ₹25,000–₹50,000: minimum CIBIL score is 700,
         maximum loan amount is ₹10 lakh."

      headers: ["Feature", "Value"]
      row:     ["Interest Rate", "10.49% per annum"]
      → "Interest Rate: 10.49% per annum."
    """
    if not rows:
        return ""

    lines = []

    # Detect key-value tables (2-column with Feature | Value pattern)
    is_kv = (
        len(headers) == 2
        and not any(_is_numeric(h) for h in headers)
        and all(len(r) == 2 for r in rows)
    )

    if is_kv:
        for row in rows:
            key, val = row[0].strip(), row[1].strip()
            if key and val:
                lines.append(f"{key}: {val}.")
    else:
        # Multi-column table: describe each row with column context
        valid_headers = [h for h in headers if h.strip()]
        for row in rows:
            if not any(c.strip() for c in row):
                continue
            parts = []
            for header, cell in zip(headers, row):
                header, cell = header.strip(), cell.strip()
                if cell and header:
                    parts.append(f"{header} is {cell}")
                elif cell:
                    parts.append(cell)
            if parts:
                lines.append("For this entry: " + ", ".join(parts) + ".")

    return " ".join(lines)


def _is_numeric(text: str) -> bool:
    try:
        float(re.sub(r"[₹%,\s]", "", text))
        return True
    except ValueError:
        return False


def table_availability() -> dict:
    return {
        "camelot":  _camelot_available,
        "tabula":   _tabula_available,
        "pdfplumber": True,   # always available (core dependency)
    }
