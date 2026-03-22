"""
document_pipeline/fusion.py
=============================
Data fusion layer: merges extracted document data with user query.

Priority (highest to lowest):
  1. Document data  — extracted directly from uploaded file (most trusted)
  2. User query     — supplemental fields from natural language query
  3. Default values — optional defaults from config (lowest trust)

Rationale: A salary slip showing ₹45,000/month is more reliable than
a user typing "I earn around 40k". Document data wins.

The fusion also builds a coherent query string for RAG retrieval that
incorporates both document-extracted and query-provided signals.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from document_pipeline.extractor import ExtractedProfile

logger = logging.getLogger(__name__)


class DataFusion:

    def fuse(
        self,
        extracted: ExtractedProfile,
        user_query: str = "",
    ) -> tuple[dict[str, Any], str]:
        """
        Merge extracted document profile with user query.

        Returns:
            merged_profile: dict ready for the rule engine
            merged_query:   string for RAG retrieval
        """
        # Start with document-extracted data (highest trust)
        merged = extracted.to_profile()

        # Parse user query for any additional fields not found in document
        if user_query:
            query_profile = self._parse_query_profile(user_query)
            for key, val in query_profile.items():
                if key not in merged or merged[key] is None:
                    # Only fill gaps — never override document data
                    merged[key] = val
                    logger.debug("[Fusion] Gap-filled %s from query: %s", key, val)

        # Validate and normalise the merged profile
        merged = self._validate_merged(merged)

        # Build enriched query string for RAG retrieval
        merged_query = self._build_query(merged, user_query, extracted)

        logger.info("[Fusion] Final profile: %s", merged)
        return merged, merged_query

    # ── Query parsing ────────────────────────────────────────────────────────

    def _parse_query_profile(self, query: str) -> dict[str, Any]:
        """
        Extract profile fields from the user's natural language query.
        Uses the existing query_understanding module if available.
        """
        try:
            from pdf_pipeline.query_understanding import understand_query
            signals = understand_query(query)
            return signals.profile
        except Exception:
            # Manual fallback
            return self._manual_query_parse(query)

    def _manual_query_parse(self, query: str) -> dict[str, Any]:
        profile = {}
        # Income
        m = re.search(r"(?:earn|salary|income|pay)\s*(?:of|is|=)?\s*₹?\s*(\d[\d,]*)\s*(?:k|thousand|lakh|lpa)?", query, re.I)
        if m:
            from scraper.normalizer import parse_monthly_income
            inc = parse_monthly_income(m.group(0))
            if inc:
                profile["monthly_income"] = inc
        # CIBIL
        m = re.search(r"(?:cibil|credit|score)\s*(?:of|is|=)?\s*(\d{3})", query, re.I)
        if m:
            score = int(m.group(1))
            if 300 <= score <= 900:
                profile["credit_score"] = score
        # Age
        m = re.search(r"(\d{2})\s*years?\s*old|\bage\s*(\d{2})\b", query, re.I)
        if m:
            age = int(m.group(1) or m.group(2))
            if 18 <= age <= 80:
                profile["age"] = age
        return profile

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_merged(self, profile: dict[str, Any]) -> dict[str, Any]:
        """Sanity-check and clean the merged profile."""
        clean = {}

        if "monthly_income" in profile and profile["monthly_income"]:
            val = float(profile["monthly_income"])
            if 3_000 <= val <= 1_00_00_000:
                clean["monthly_income"] = round(val, 2)

        if "credit_score" in profile and profile["credit_score"]:
            val = int(profile["credit_score"])
            if 300 <= val <= 900:
                clean["credit_score"] = val

        if "age" in profile and profile["age"]:
            val = int(profile["age"])
            if 18 <= val <= 80:
                clean["age"] = val

        if "employment_type" in profile and profile["employment_type"]:
            clean["employment_type"] = str(profile["employment_type"]).lower()

        if "work_experience_months" in profile and profile["work_experience_months"]:
            val = int(profile["work_experience_months"])
            if 0 < val < 600:
                clean["work_experience_months"] = val

        if "dti_ratio" in profile and profile["dti_ratio"] is not None:
            val = float(profile["dti_ratio"])
            if val > 1.5:
                val /= 100.0
            if 0 < val < 1:
                clean["dti_ratio"] = round(val, 4)

        return clean

    # ── Query building ────────────────────────────────────────────────────────

    def _build_query(
        self,
        merged_profile: dict[str, Any],
        user_query:     str,
        extracted:      ExtractedProfile,
    ) -> str:
        """
        Build a rich query string for RAG retrieval that incorporates
        both document signals and user intent.
        """
        parts = []

        # Start with user's own words if provided
        if user_query.strip():
            parts.append(user_query.strip())

        # Add document type context
        if extracted.document_type and extracted.document_type != "unknown":
            parts.append(f"document: {extracted.document_type.replace('_', ' ')}")

        # Add key profile signals as retrieval terms
        if merged_profile.get("monthly_income"):
            parts.append(f"salary {merged_profile['monthly_income']:.0f} monthly income")

        if merged_profile.get("credit_score"):
            parts.append(f"CIBIL credit score {merged_profile['credit_score']}")

        if merged_profile.get("employment_type"):
            parts.append(merged_profile["employment_type"])

        if merged_profile.get("age"):
            parts.append(f"age {merged_profile['age']}")

        # Always include eligibility as the core intent
        parts.append("personal loan eligibility criteria requirements")

        # Deduplicate tokens while preserving order
        seen: set[str] = set()
        result_tokens: list[str] = []
        for part in parts:
            for tok in part.split():
                tl = tok.lower()
                if tl not in seen:
                    seen.add(tl)
                    result_tokens.append(tok)

        return " ".join(result_tokens)
