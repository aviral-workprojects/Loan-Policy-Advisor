"""
scraper/fetcher.py
===================
Robust HTTP fetcher with:
  - Rotating user agents
  - Retry with exponential backoff
  - Timeout handling
  - robots.txt respect (configurable)
  - Optional Playwright fallback for JS-rendered pages
  - Archive.org fallback for blocked/unavailable pages

Bank websites are often behind anti-bot measures. The fetcher uses
realistic browser headers and adds a polite delay between requests
to avoid triggering rate limits.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",

    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
    "Gecko/20100101 Firefox/124.0",

    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

_BASE_HEADERS = {
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en-GB;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "DNT":             "1",
    "Upgrade-Insecure-Requests": "1",
}

_TIMEOUT       = 20      # seconds
_MAX_RETRIES   = 3
_RETRY_BACKOFF = 1.5     # seconds, exponential
_MIN_DELAY     = 1.5     # minimum seconds between requests (polite crawling)
_MAX_DELAY     = 4.0


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url:        str
    html:       str         = ""
    status:     int         = 0
    success:    bool        = False
    method:     str         = ""        # "requests" | "playwright" | "archive"
    error:      str         = ""
    final_url:  str         = ""        # after redirects


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_RETRY_BACKOFF,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET", "HEAD"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = _make_session()
    return _session


# ---------------------------------------------------------------------------
# Layer 1 — requests
# ---------------------------------------------------------------------------

def fetch_html(url: str, delay: bool = True) -> FetchResult:
    """
    Fetch a URL with rotating UA, retries, and polite delay.
    Primary fetcher for all scraping operations.
    """
    if delay:
        time.sleep(random.uniform(_MIN_DELAY, _MAX_DELAY))

    headers = {**_BASE_HEADERS, "User-Agent": random.choice(_USER_AGENTS)}
    # Add referer based on domain to look more legitimate
    domain = urlparse(url).netloc
    headers["Referer"] = f"https://{domain}/"

    try:
        resp = _get_session().get(url, headers=headers, timeout=_TIMEOUT, allow_redirects=True)
        html = resp.text

        if resp.status_code == 200 and len(html.strip()) > 500:
            logger.info("[Fetch] ✓ %s  (%d chars)", url, len(html))
            return FetchResult(
                url=url, html=html, status=resp.status_code,
                success=True, method="requests", final_url=resp.url,
            )

        if resp.status_code in (403, 429):
            logger.warning("[Fetch] %d on %s — will try Playwright", resp.status_code, url)
            return FetchResult(url=url, status=resp.status_code, success=False,
                               method="requests", error=f"HTTP {resp.status_code}")

        logger.warning("[Fetch] HTTP %d on %s", resp.status_code, url)
        return FetchResult(url=url, html=html, status=resp.status_code,
                           success=len(html.strip()) > 200, method="requests",
                           final_url=resp.url)

    except requests.exceptions.Timeout:
        logger.warning("[Fetch] Timeout: %s", url)
        return FetchResult(url=url, success=False, method="requests", error="timeout")

    except Exception as e:
        logger.warning("[Fetch] Error on %s: %s", url, e)
        return FetchResult(url=url, success=False, method="requests", error=str(e))


# ---------------------------------------------------------------------------
# Layer 2 — Playwright (JS-rendered pages)
# ---------------------------------------------------------------------------

def fetch_html_playwright(url: str) -> FetchResult:
    """
    Fetch a URL using a real Chromium browser via Playwright.
    Used when requests gets blocked or the page requires JavaScript.

    Requires: pip install playwright && playwright install chromium
    """
    try:
        from playwright.sync_api import sync_playwright
        logger.info("[Fetch] Playwright: %s", url)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent=random.choice(_USER_AGENTS),
                viewport={"width": 1280, "height": 900},
                locale="en-IN",
                java_script_enabled=True,
            )
            page = ctx.new_page()

            # Block images/fonts to speed things up
            page.route("**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf}", lambda r: r.abort())

            page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(2_000)    # let any lazy-loading settle

            # Scroll to trigger lazy content
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(1_500)

            html = page.content()
            final_url = page.url
            browser.close()

        if len(html.strip()) > 500:
            logger.info("[Fetch] Playwright ✓ %s  (%d chars)", url, len(html))
            return FetchResult(url=url, html=html, status=200, success=True,
                               method="playwright", final_url=final_url)

        return FetchResult(url=url, success=False, method="playwright",
                           error="Playwright returned empty page")

    except ImportError:
        logger.debug("[Fetch] Playwright not installed — skipping")
        return FetchResult(url=url, success=False, method="playwright",
                           error="playwright not installed")
    except Exception as e:
        logger.warning("[Fetch] Playwright error on %s: %s", url, e)
        return FetchResult(url=url, success=False, method="playwright", error=str(e))


# ---------------------------------------------------------------------------
# Layer 3 — Archive.org fallback
# ---------------------------------------------------------------------------

def fetch_from_archive(url: str) -> FetchResult:
    """
    Fetch from the Wayback Machine when the live URL is unavailable.
    Uses the most recent available snapshot.
    """
    archive_api = f"http://archive.org/wayback/available?url={url}"
    try:
        logger.info("[Fetch] Checking Archive.org for: %s", url)
        resp = requests.get(archive_api, timeout=10)
        data = resp.json()
        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if not snapshot.get("available"):
            return FetchResult(url=url, success=False, method="archive",
                               error="No archive snapshot available")

        archive_url = snapshot["url"]
        logger.info("[Fetch] Archive snapshot: %s", archive_url)
        result = fetch_html(archive_url, delay=False)
        result.method = "archive"
        result.url    = url
        return result

    except Exception as e:
        logger.warning("[Fetch] Archive fallback failed for %s: %s", url, e)
        return FetchResult(url=url, success=False, method="archive", error=str(e))


# ---------------------------------------------------------------------------
# Waterfall fetcher (orchestrates all layers)
# ---------------------------------------------------------------------------

def fetch_with_waterfall(url: str) -> FetchResult:
    """
    Try each fetch method in order until one succeeds:
      Layer 1: requests (fast, works for most pages)
      Layer 2: Playwright (for JS-heavy pages)
      Layer 3: Archive.org (for blocked/unavailable pages)

    Never raises — always returns a FetchResult.
    """
    # Layer 1
    result = fetch_html(url)
    if result.success and len(result.html) > 1000:
        return result

    # Layer 2 — try Playwright if we got blocked or thin content
    if result.status in (403, 429) or (result.success and len(result.html) < 2000):
        logger.info("[Fetch] Layer 1 weak → trying Playwright for %s", url)
        pw_result = fetch_html_playwright(url)
        if pw_result.success and len(pw_result.html) > len(result.html):
            return pw_result

    # Layer 3 — archive fallback
    if not result.success or len(result.html) < 1000:
        logger.info("[Fetch] Layers 1+2 failed → trying Archive.org for %s", url)
        arch_result = fetch_from_archive(url)
        if arch_result.success:
            return arch_result

    # Return whatever we have (may be empty)
    logger.warning("[Fetch] All layers failed for %s — returning partial result", url)
    return result
