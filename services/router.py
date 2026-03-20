"""
Router Service — MoE-style Query Routing
==========================================
Determines which retrieval sources to query based on:
  - Detected banks
  - Query intent
  - Entity completeness

Returns a RetrievalPlan that drives the retrieval layer.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from services.llm import ParsedQuery

KNOWN_BANKS = {"Axis", "ICICI", "HDFC", "SBI"}

@dataclass
class RetrievalPlan:
    """Instructions for the retrieval layer."""
    banks: list[str]                    # specific banks to query
    use_aggregators: bool = True        # Paisabazaar / BankBazaar
    use_regulatory: bool = False        # RBI docs
    use_bank_specific: bool = True      # bank-specific docs
    use_hybrid: bool = False            # all sources
    trigger_fallback: bool = False      # web search fallback
    intent: str = "general"

    @property
    def doc_types(self) -> list[str]:
        types = []
        if self.use_bank_specific:
            types.extend(["eligibility", "pdf"])
        if self.use_aggregators:
            types.extend(["comparison"])
        if self.use_regulatory:
            types.append("regulatory")
        return types or ["eligibility", "comparison", "regulatory", "pdf"]


class Router:
    """
    Routing logic:
      - Bank detected  → bank-specific DB + optional aggregators
      - Comparison     → aggregators + all banks
      - Regulatory     → RBI docs
      - General        → hybrid (all sources)
    """

    def route(self, parsed: ParsedQuery) -> RetrievalPlan:
        intent = parsed.intent
        banks  = [b for b in parsed.banks if b in KNOWN_BANKS]

        # ---------------------------------------------------------------- comparison
        if intent == "comparison":
            return RetrievalPlan(
                banks=banks if banks else list(KNOWN_BANKS),
                use_aggregators=True,
                use_bank_specific=True,
                use_regulatory=False,
                use_hybrid=False,
                intent=intent,
            )

        # ---------------------------------------------------------------- eligibility
        if intent == "eligibility":
            if banks:
                priority_banks = banks
            else:
                # FIX #6: default to most lenient banks first for faster/cheaper response
                priority_banks = ["Axis", "SBI"]
            return RetrievalPlan(
                banks=priority_banks,
                use_aggregators=True,
                use_bank_specific=True,
                use_regulatory=False,
                intent=intent,
            )

        # ---------------------------------------------------------------- regulatory
        if intent == "regulatory" or self._contains_regulatory_keywords(parsed):
            return RetrievalPlan(
                banks=banks,
                use_regulatory=True,
                use_aggregators=False,
                use_bank_specific=False,
                intent="regulatory",
            )

        # ---------------------------------------------------------------- general / fallback
        return RetrievalPlan(
            banks=banks if banks else list(KNOWN_BANKS),
            use_aggregators=True,
            use_bank_specific=bool(banks),
            use_regulatory=False,
            use_hybrid=not bool(banks),
            intent=intent,
        )

    @staticmethod
    def _contains_regulatory_keywords(parsed: ParsedQuery) -> bool:
        regulatory_terms = {"rbi", "regulation", "compliance", "rule", "law", "guideline"}
        # We'd need the original query text here; this is a placeholder
        return False
