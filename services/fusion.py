"""
Context Fusion
==============
Merges retrieved chunks from multiple sources,
deduplicates, and prioritizes by source authority:
  1. Bank-specific rules (highest authority)
  2. RBI regulatory
  3. Aggregator comparison data
"""

from __future__ import annotations
from services.retrieval import DocChunk

# Priority order — lower number = higher priority
SOURCE_PRIORITY = {
    "Axis":        1,
    "ICICI":       1,
    "HDFC":        1,
    "SBI":         1,
    "RBI":         2,
    "Paisabazaar": 3,
    "BankBazaar":  3,
}

class ContextFusion:

    def fuse(
        self,
        chunks: list[DocChunk],
        max_tokens: int = 3000,
    ) -> str:
        """
        Merge chunks into a single context string.
        - Deduplicate by text similarity (simple prefix check)
        - Sort by source authority
        - Truncate to max_tokens (approx word count)
        """
        if not chunks:
            return "No relevant context retrieved."

        # Sort by priority
        sorted_chunks = sorted(
            chunks,
            key=lambda c: SOURCE_PRIORITY.get(c.bank, 10)
        )

        # Deduplicate: remove near-duplicate chunks
        seen_prefixes: set[str] = set()
        unique: list[DocChunk] = []
        for chunk in sorted_chunks:
            prefix = chunk.text[:80].strip().lower()
            if prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                unique.append(chunk)

        # Build context string with source labels
        parts: list[str] = []
        word_count = 0
        for chunk in unique:
            header = f"[Source: {chunk.bank} | {chunk.doc_type}]"
            block = f"{header}\n{chunk.text}"
            words = len(block.split())
            if word_count + words > max_tokens:
                break
            parts.append(block)
            word_count += words

        return "\n\n---\n\n".join(parts)
