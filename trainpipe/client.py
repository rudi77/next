"""Shared HTTP client for the trainpipe REST API.

Both the MCP server (``trainpipe.mcp``) and the operative CLI
(``trainpipe`` subcommands) drive the same REST API. This module holds the
one copy of the auth-header wiring and the error-unwrap logic so the two
front-ends stay in capability parity and don't drift apart.

The client is intentionally tiny: it adds the ``X-API-Key`` header, sets a
sane timeout, and turns HTTP errors into a single ``APIError`` whose message
preserves the server's actionable detail body.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8080"


class MissingAPIKey(RuntimeError):
    """Raised when no API key is available to build a client."""


class APIError(RuntimeError):
    """An HTTP error from the REST API, with the response body preserved."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


def build_client(
    base_url: str | None = None,
    api_key: str | None = None,
) -> httpx.Client:
    """Build an httpx client targeting the REST API.

    ``base_url`` / ``api_key`` fall back to ``TRAINPIPE_BASE_URL`` /
    ``TRAINPIPE_API_KEY``. Raises :class:`MissingAPIKey` if no key is set.
    """
    base_url = base_url or os.environ.get("TRAINPIPE_BASE_URL", DEFAULT_BASE_URL)
    api_key = api_key or os.environ.get("TRAINPIPE_API_KEY")
    if not api_key:
        raise MissingAPIKey(
            "TRAINPIPE_API_KEY environment variable must be set "
            "(use the same value you put in trainpipe's .env)"
        )
    return httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=httpx.Timeout(30.0, connect=5.0),
    )


def unwrap(resp: httpx.Response) -> Any:
    """Raise :class:`APIError` on HTTP errors; else return JSON body or text."""
    if resp.is_error:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise APIError(resp.status_code, detail)
    ctype = resp.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        return resp.json()
    return resp.text
