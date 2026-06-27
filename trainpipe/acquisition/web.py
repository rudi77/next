"""Web research + acquisition primitives (Phase 22, stage 3).

Three concerns, each injectable so the whole stack runs offline in tests:

* :class:`SearchProvider` — a query → candidate URLs lookup. ``mock`` for
  tests, ``tavily`` for real use (httpx against the Tavily API).
* :class:`Extractor` — HTML → clean main text. Uses ``trafilatura`` when the
  ``[acquisition]`` extra is installed, else a built-in tag-stripper.
* :func:`make_fetch_gate` — a per-URL gate that enforces robots.txt and a
  license heuristic *before* anything is downloaded. The design invariant is
  "no fetch without a license/robots check"; a URL the gate blocks is recorded
  but never fetched.

The driver wires these together; the pure phase logic in ``runner.py`` only
sees the small callables they expose, so research/acquire are testable with
plain mocks and no network.
"""

from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from ..integrations.labelstudio import LabelStudioError, _validate_base_url
from ..synth.runner import _post_with_retry

logger = logging.getLogger(__name__)


def _url_is_safe(url: str) -> bool:
    """Reuse the Label Studio SSRF guard: reject non-http(s), missing host,
    cloud-metadata endpoints, and hosts resolving to private/loopback/IMDS
    ranges. Search results are attacker-influenced, so every outbound URL is
    validated before we fetch it."""
    try:
        _validate_base_url(url)
        return True
    except LabelStudioError:
        return False

# A function that fetches a URL's text body, or None on any failure. Injectable
# so tests never touch the network.
TextFetcher = Callable[[str], "str | None"]

_USER_AGENT = "trainpipe-acquisition/1.0"


@dataclass
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------


class SearchProvider(ABC):
    name: str

    @abstractmethod
    def search(self, query: str, *, max_results: int) -> list[SearchHit]: ...


class MockSearchProvider(SearchProvider):
    """Returns a fixed list of canned hits (echoed for every query)."""

    name = "mock"

    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits or []

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        return self._hits[:max_results]


class TavilySearchProvider(SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "TavilySearchProvider needs TAVILY_API_KEY env var or explicit "
                "api_key"
            )

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        with httpx.Client(
            base_url="https://api.tavily.com",
            timeout=httpx.Timeout(30.0, connect=10.0),
        ) as client:
            # Reuse synth's classified retry/backoff (429/5xx → retry,
            # 4xx → fatal) instead of a bare raise_for_status.
            resp = _post_with_retry(
                client,
                "/search",
                json_body={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                },
            )
            body = resp.json()
        return [
            SearchHit(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
            )
            for r in (body.get("results") or [])
            if r.get("url")
        ]


_SEARCH_PROVIDERS: dict[str, type[SearchProvider]] = {
    "mock": MockSearchProvider,
    "tavily": TavilySearchProvider,
}


def make_search_provider(name: str) -> SearchProvider:
    cls = _SEARCH_PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown search provider {name!r}; options: "
            f"{sorted(_SEARCH_PROVIDERS)}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
_WS_RE = re.compile(r"\s+")


class Extractor(ABC):
    @abstractmethod
    def extract(self, html: str, url: str) -> str: ...


class SimpleExtractor(Extractor):
    """Dependency-free fallback: drop script/style, strip tags, collapse
    whitespace. Crude but keeps the module working without ``trafilatura``."""

    def extract(self, html: str, url: str) -> str:
        without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
        text = _TAG_RE.sub(" ", without_scripts)
        return _WS_RE.sub(" ", text).strip()


class TrafilaturaExtractor(Extractor):
    def __init__(self) -> None:
        # Import here so the module loads without the [acquisition] extra;
        # make_extractor() falls back to SimpleExtractor on ImportError.
        import trafilatura

        self._traf = trafilatura

    def extract(self, html: str, url: str) -> str:
        out = self._traf.extract(html, url=url) or ""
        return out.strip()


def make_extractor() -> Extractor:
    """Prefer trafilatura when available, else the built-in stripper."""
    try:
        return TrafilaturaExtractor()
    except Exception:
        return SimpleExtractor()


# ---------------------------------------------------------------------------
# Fetch gate: robots.txt + license heuristic
# ---------------------------------------------------------------------------

# Domains/TLDs we treat as openly licensed for training use. Conservative on
# purpose; everything else is "unknown" and (in strict mode) skipped. Matched
# as host suffixes (anchored), not bare substrings, so the recorded license
# status can't be spoofed by an attacker-chosen hostname like
# ``evil-wikipedia.org.attacker.net``.
_OPEN_LICENSE_HINTS = (
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "gov",
    "europa.eu",
    "creativecommons.org",
    "gutenberg.org",
)


def _host_matches(host: str, hint: str) -> bool:
    """True if ``host`` equals ``hint`` or is a subdomain of it."""
    return host == hint or host.endswith("." + hint)


def _license_status(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if any(_host_matches(host, hint) for hint in _OPEN_LICENSE_HINTS):
        return "open"
    return "unknown"


def _default_robots_fetcher(robots_url: str) -> str | None:
    try:
        resp = httpx.get(
            robots_url,
            headers={"User-Agent": _USER_AGENT},
            timeout=httpx.Timeout(10.0, connect=5.0),
            follow_redirects=True,
        )
        if resp.is_success:
            return resp.text
    except httpx.HTTPError:
        pass
    return None


@dataclass
class GateDecision:
    allowed: bool
    license_status: str  # "open" | "unknown" — always the real license


def make_fetch_gate(
    *,
    strict_license: bool = False,
    robots_fetcher: TextFetcher | None = None,
    url_safety: Callable[[str], bool] | None = None,
) -> Callable[[str], GateDecision]:
    """Build a per-URL gate: license heuristic → SSRF safety → robots.txt.

    A URL is allowed only if all three pass. ``license_status`` always carries
    the real license verdict so the audit ledger is honest, regardless of
    which check did the rejecting. ``strict_license`` rejects "unknown"-license
    URLs (and short-circuits before the network checks). robots.txt is fetched
    and parsed once per host (memoized in the closure). ``robots_fetcher`` and
    ``url_safety`` (the SSRF check, which does DNS) are injectable so tests run
    offline. A missing/unreadable robots.txt is treated as allowed — robots is
    the politeness guard; license + SSRF are the real ones.
    """
    fetch_robots = robots_fetcher or _default_robots_fetcher
    is_safe = url_safety or _url_is_safe
    parsers: dict[str, RobotFileParser | None] = {}

    def _robots_ok(url: str) -> bool:
        parts = urlparse(url)
        if not parts.scheme or not parts.netloc:
            return False
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in parsers:
            text = fetch_robots(urljoin(origin, "/robots.txt"))
            if text:
                parser = RobotFileParser()
                parser.parse(text.splitlines())
                parsers[origin] = parser
            else:
                parsers[origin] = None  # no robots.txt → allow all
        parser = parsers[origin]
        return parser is None or parser.can_fetch(_USER_AGENT, url)

    def gate(url: str) -> GateDecision:
        status = _license_status(url)
        if strict_license and status != "open":
            return GateDecision(False, status)  # skip network checks
        if not is_safe(url):
            return GateDecision(False, status)
        if not _robots_ok(url):
            return GateDecision(False, status)
        return GateDecision(True, status)

    return gate


def make_text_fetcher(extractor: Extractor) -> TextFetcher:
    """A real (network) page-text fetcher: GET the URL, extract main text.

    Returns None on any HTTP/parse failure so the acquire loop skips the
    source rather than failing the run.
    """

    def fetch_text(url: str) -> str | None:
        if not _url_is_safe(url):
            return None
        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": _USER_AGENT},
                timeout=httpx.Timeout(20.0, connect=10.0),
                follow_redirects=True,
            )
            if not resp.is_success:
                return None
        except httpx.HTTPError:
            return None
        text = extractor.extract(resp.text, url)
        return text or None

    return fetch_text
