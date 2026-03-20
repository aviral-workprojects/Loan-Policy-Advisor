"""
Fallback Search Service
========================
Called ONLY when retrieved docs < MIN_DOCS_THRESHOLD.
Supports Tavily and SERP API.
"""

from __future__ import annotations
import os
from config import TAVILY_API_KEY, SERP_API_KEY, FALLBACK_PROVIDER

class FallbackSearch:

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Returns list of {title, snippet, url} dicts from web search.
        Tries configured provider; returns empty list on failure.
        """
        if FALLBACK_PROVIDER == "tavily" and TAVILY_API_KEY:
            return self._tavily(query, max_results)
        elif SERP_API_KEY:
            return self._serp(query, max_results)
        else:
            print("[FallbackSearch] No API key configured, skipping web search.")
            return []

    def _tavily(self, query: str, max_results: int) -> list[dict]:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(
                query=query + " India personal loan",
                max_results=max_results,
                search_depth="basic",
            )
            return [
                {
                    "title":   r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "url":     r.get("url", ""),
                    "source":  "tavily",
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            print(f"[FallbackSearch] Tavily error: {e}")
            return []

    def _serp(self, query: str, max_results: int) -> list[dict]:
        try:
            import requests
            params = {
                "api_key": SERP_API_KEY,
                "q":       query + " India personal loan",
                "num":     max_results,
            }
            resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return [
                {
                    "title":   r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "url":     r.get("link", ""),
                    "source":  "serp",
                }
                for r in data.get("organic_results", [])[:max_results]
            ]
        except Exception as e:
            print(f"[FallbackSearch] SERP error: {e}")
            return []

    @staticmethod
    def format_results(results: list[dict]) -> str:
        """Convert search results to plain text context for LLM."""
        if not results:
            return ""
        lines = ["[Web Search Results]"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r['title']}")
            lines.append(f"   {r['snippet']}")
            lines.append(f"   Source: {r['url']}")
        return "\n".join(lines)
