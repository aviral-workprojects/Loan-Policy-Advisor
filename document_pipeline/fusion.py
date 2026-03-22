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
        """
        Fallback parser for when query_understanding is unavailable.
        Handles informal queries, k-suffix income, "age 23" patterns,
        freelancer detection, and the "no job" ≠ salaried edge case.
        """
        profile: dict[str, Any] = {}
        ql = query.lower()

        # ── Income ──────────────────────────────────────────────────────────
        # Priority 1: contextual earn/salary/income + number + optional suffix
        m = re.search(
            r"(?:earn|salary|income|pay)\s*(?:of|is|=|:)?\s*"
            r"₹?\s*(\d[\d,]*)\s*(k|thousand|lakh|lpa)?",
            query, re.I,
        )
        if m:
            raw = m.group(1).replace(",", "")
            try:
                val = float(raw)
                suffix = (m.group(2) or "").lower()
                if suffix in ("k", "thousand"):
                    val *= 1_000
                elif suffix == "lakh":
                    val *= 1_00_000
                elif suffix == "lpa":
                    val = val * 1_00_000 / 12
                if 3_000 < val < 10_00_00_000:
                    profile["monthly_income"] = round(val, 2)
            except ValueError:
                pass

        # Priority 2: bare "30k" / "₹40,000" when no contextual keyword present
        if "monthly_income" not in profile:
            m2 = re.search(r"₹?\s*(\d[\d,]+)\s*(k\b|thousand\b)?", query, re.I)
            if m2:
                raw = m2.group(1).replace(",", "")
                try:
                    val = float(raw)
                    if m2.group(2):
                        val *= 1_000
                    if 5_000 <= val <= 5_00_000:   # plausible monthly income range
                        profile["monthly_income"] = round(val, 2)
                except ValueError:
                    pass

        # ── CIBIL ────────────────────────────────────────────────────────────
        m = re.search(r"(?:cibil|credit|score)\s*(?:of|is|=|:)?\s*(\d{3})", query, re.I)
        if not m:
            m = re.search(r"\b([4-9]\d{2})\b", query)
        if m:
            score = int(m.group(1))
            if 300 <= score <= 900:
                profile["credit_score"] = score

        # ── Age ──────────────────────────────────────────────────────────────
        m = re.search(
            r"(\d{2})\s*years?\s*(?:old|of\s*age)?\b|"
            r"(?:age|aged?)\s*(?:of\s*|is\s*|:)?\s*(\d{2})\b|"
            r"(?:\bam|is)\s+(\d{2})\b",
            query, re.I,
        )
        if m:
            try:
                age = int(next(g for g in m.groups() if g is not None))
                if 18 <= age <= 80:
                    profile["age"] = age
            except (ValueError, StopIteration):
                pass

        # ── Employment type — check negative/unemployed FIRST ────────────────
        # "\bjob\b" removed from salaried check to avoid "no job" → salaried
        if re.search(r"\bno\s+job\b|\bunemployed\b|\bjobless\b", ql):
            profile["employment_type"] = "unemployed"
        elif re.search(
            r"\bfreelance[rd]?\b|\bfreelancing\b|\bself.?emp\b|\bconsultant\b|"
            r"\bown\s+business\b|\bproprietor\b|\bgig\b",
            ql,
        ):
            profile["employment_type"] = "self_employed"
        elif re.search(r"\bgovernment\b|\bpsu\b|\bdefence\b|\bdefense\b", ql):
            profile["employment_type"] = "government"
        elif re.search(r"\bsalaried\b|\bworking\b|\bemployed\b|\bservice\b", ql):
            profile["employment_type"] = "salaried"

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