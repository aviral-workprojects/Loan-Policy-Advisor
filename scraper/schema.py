"""
scraper/schema.py
==================
Canonical data schema for all scraped loan information.

Every piece of data that enters the pipeline is coerced into this shape.
Downstream consumers (FAISS, BM25, rule engine) can rely on field names
being consistent regardless of which bank or scraper produced them.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import time


# ---------------------------------------------------------------------------
# Canonical loan record
# ---------------------------------------------------------------------------

@dataclass
class LoanRecord:
    # Source identity
    bank:             str         = ""          # "Axis" | "HDFC" | "ICICI" | "SBI" | ...
    loan_type:        str         = "personal"  # "personal" | "home" | "car" | "business"
    source_url:       str         = ""
    scraped_at:       float       = field(default_factory=time.time)

    # Financial terms
    interest_rate:    list[float] = field(default_factory=list)   # [min%, max%]
    processing_fee:   float | None = None       # absolute ₹ or % of loan
    processing_fee_pct: float | None = None     # % of loan amount
    foreclosure_fee:  float | None = None       # %
    late_fee:         float | None = None       # ₹/month or %

    # Eligibility
    min_income:       float | None = None       # monthly ₹
    max_income:       float | None = None       # monthly ₹ (if capped)
    age_range:        list[int]    = field(default_factory=list)    # [min, max]
    employment_type:  list[str]    = field(default_factory=list)    # ["salaried","self_employed",...]
    min_cibil:        int   | None = None
    min_experience:   int   | None = None       # months

    # Loan terms
    loan_amount:      list[float] = field(default_factory=list)    # [min ₹, max ₹]
    tenure:           list[int]   = field(default_factory=list)    # [min months, max months]

    # Raw content for RAG
    content:          str         = ""          # cleaned full text for embedding
    features:         list[str]   = field(default_factory=list)
    key_facts:        list[str]   = field(default_factory=list)

    # Metadata
    content_hash:     str         = ""
    extraction_method: str        = ""          # which layer succeeded
    confidence:       float       = 0.0         # 0-1 extraction confidence

    def to_dict(self) -> dict:
        return asdict(self)

    def to_rag_text(self) -> str:
        """
        Convert record to a natural-language paragraph optimised for RAG retrieval.
        This is what gets stored in the .txt files and embedded into FAISS.
        """
        parts: list[str] = []

        if self.bank and self.loan_type:
            parts.append(f"{self.bank} {self.loan_type} loan.")

        if self.interest_rate and len(self.interest_rate) == 2:
            parts.append(
                f"Interest rate: {self.interest_rate[0]:.2f}% to {self.interest_rate[1]:.2f}% per annum."
            )
        elif self.interest_rate and len(self.interest_rate) == 1:
            parts.append(f"Interest rate: {self.interest_rate[0]:.2f}% per annum.")

        if self.loan_amount and len(self.loan_amount) >= 2:
            parts.append(
                f"Loan amount: ₹{self.loan_amount[0]:,.0f} to ₹{self.loan_amount[1]:,.0f}."
            )

        if self.tenure and len(self.tenure) >= 2:
            parts.append(f"Tenure: {self.tenure[0]} to {self.tenure[1]} months.")

        if self.min_income:
            parts.append(f"Minimum monthly income: ₹{self.min_income:,.0f}.")

        if self.age_range and len(self.age_range) >= 2:
            parts.append(f"Age requirement: {self.age_range[0]} to {self.age_range[1]} years.")

        if self.employment_type:
            parts.append(f"Eligible employment types: {', '.join(self.employment_type)}.")

        if self.min_cibil:
            parts.append(f"Minimum CIBIL score: {self.min_cibil}.")

        if self.processing_fee is not None:
            parts.append(f"Processing fee: ₹{self.processing_fee:,.0f}.")
        elif self.processing_fee_pct is not None:
            parts.append(f"Processing fee: {self.processing_fee_pct:.2f}% of loan amount.")

        if self.min_experience:
            parts.append(f"Minimum work experience: {self.min_experience} months.")

        for feat in self.features[:5]:
            if feat.strip():
                parts.append(feat.strip().rstrip(".") + ".")

        if self.content and len(self.content) > 100:
            parts.append(self.content[:800])

        return " ".join(parts)
