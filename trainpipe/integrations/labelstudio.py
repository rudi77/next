"""Label Studio bridge (Phase 10).

Imports completed annotations from a Label Studio project, maps them into
one of our supported JSONL formats, and registers the result as a regular
trainpipe dataset (so it appears in ``ds:<id>`` refs and the UI).

We deliberately *do not* build an annotation UI of our own — Label Studio
already exists, is OSS, and is good. We only need a directional adapter:
LS → trainpipe.

Supported import shapes (auto-detected from the first ~10 tasks):

* ``conversation``  — generic chat-style ``[{"messages": [...]}]`` records.
  Triggered when tasks have ``data.text`` or ``data.prompt`` and at least
  one annotation result of type ``textarea`` / ``choices``.
* ``text_ner``      — span labels over a ``text`` field, exported as
  ``{"text": "...", "entities": [{"start", "end", "label"}]}``.
* ``doc_layout``    — image bbox annotations (LS RectangleLabels), exported
  as ``{"images": ["..."], "gold_boxes": [{"box": [...], "label": "..."}]}``.

If detection is ambiguous, callers can force ``import_kind`` explicitly.

The HTTP client is pluggable so the bridge is testable without a real
LS server: pass a callable that takes ``(method, path, **kwargs)`` and
returns a dict/list/bytes.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

ImportKind = str  # "conversation" | "text_ner" | "doc_layout"


class LabelStudioError(RuntimeError):
    """Raised when an LS API call fails or a task is malformed."""


# ---------------------------------------------------------------------------
# URL validation (SSRF defense)
# ---------------------------------------------------------------------------


def _is_blocked_address(addr: str) -> bool:
    """Reject loopback, link-local, private, multicast, and the IMDS IPs.

    The Label Studio host should be a public-ish endpoint reachable over
    the corporate network. If a caller targets ``localhost``, ``::1``,
    ``169.254.169.254`` (AWS/Azure IMDS), ``metadata.google.internal``,
    or any RFC1918 / loopback address, we refuse to issue the request —
    LS bridge is not a generic HTTP proxy.
    """
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_base_url(base_url: str) -> str:
    """Sanity-check ``base_url``; return a credential-free canonical form.

    Raises :class:`LabelStudioError` on disallowed shapes:
      * non-http(s) scheme,
      * missing host,
      * host that resolves to a blocked IP range,
      * literally a blocked metadata hostname (``metadata.google.internal``).

    Strips any ``user:pass@`` in the URL so logs / dataset descriptions
    never carry embedded credentials.
    """
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https"):
        raise LabelStudioError(
            f"unsupported scheme {parts.scheme!r}; expected http/https"
        )
    host = parts.hostname
    if not host:
        raise LabelStudioError("base_url is missing a host")
    if host.lower() in {
        "metadata.google.internal",
        "metadata",
        "instance-data",
    }:
        raise LabelStudioError(
            f"host {host!r} is a cloud metadata endpoint; refusing"
        )
    # Resolve every address the host points at; one block on any of them
    # is enough to reject (DNS rebinding defense at-rest is also helped
    # by httpx not retaining the dns result across redirects).
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise LabelStudioError(f"cannot resolve host {host!r}: {e}") from None
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        if _is_blocked_address(sockaddr[0]):
            raise LabelStudioError(
                f"host {host!r} resolves to a blocked address {sockaddr[0]!r}"
            )
    # Re-emit without userinfo.
    netloc = host
    if parts.port:
        netloc = f"{host}:{parts.port}"
    return urlunsplit(
        (parts.scheme, netloc, parts.path or "", parts.query, parts.fragment)
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


# Caller-injected transport: (method, path, **kwargs) -> json-decoded body.
# Used to make the mapper testable without a real LS instance.
LsTransport = Callable[..., Any]


_TOKEN_REDACT = "<redacted>"


def _sanitize_error_body(text: str, token: str) -> str:
    """Best-effort redaction so an LS error reflecting our headers back
    can't leak the token to the API caller."""
    if not token:
        return text
    return text.replace(token, _TOKEN_REDACT)


def _make_transport_factory(
    client: httpx.Client, token: str
) -> LsTransport:
    def _send(method: str, path: str, **kwargs: Any) -> Any:
        resp = client.request(method, path, **kwargs)
        if resp.is_error:
            try:
                detail: Any = resp.json()
                detail_repr = json.dumps(detail)
            except ValueError:
                detail_repr = resp.text
            # LS sometimes echoes the Authorization header in error
            # payloads via misconfigured proxies — scrub before we
            # surface it to the API caller (who may not be a token holder).
            detail_repr = _sanitize_error_body(detail_repr, token)
            raise LabelStudioError(
                f"LS HTTP {resp.status_code} on {method} {path}: "
                f"{detail_repr[:1024]}"
            )
        ct = resp.headers.get("content-type", "")
        return resp.json() if ct.startswith("application/json") else resp.text

    return _send


def fetch_completed_tasks(
    transport: LsTransport,
    project_id: int,
    *,
    since_iso: str | None = None,
    page_size: int = 200,
    max_tasks: int | None = None,
) -> list[dict[str, Any]]:
    """Pull completed tasks from a Label Studio project.

    ``since_iso`` filters server-side via LS's ``completed_at__gte`` query
    parameter; combine with caller-side caching for incremental import.

    Pagination follows LS's ``page`` / ``page_size`` scheme until the page
    comes back shorter than ``page_size`` OR ``max_tasks`` is hit.
    """
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "completed": "true",
        }
        if since_iso:
            params["completed_at__gte"] = since_iso
        body = transport(
            "GET", f"/api/projects/{project_id}/tasks/", params=params
        )
        items = body if isinstance(body, list) else body.get("tasks") or []
        out.extend(items)
        if max_tasks is not None and len(out) >= max_tasks:
            return out[:max_tasks]
        if len(items) < page_size:
            return out
        page += 1


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------


def detect_import_kind(tasks: list[dict[str, Any]]) -> ImportKind:
    """Sniff the most likely import shape from the first few tasks.

    Heuristic order: doc_layout (RectangleLabels) → text_ner (Labels with
    start/end) → conversation (everything else with a text-ish field).
    """
    sample = tasks[: min(10, len(tasks))]
    for task in sample:
        for ann in task.get("annotations", []) or []:
            for r in ann.get("result", []) or []:
                t = r.get("type")
                if t == "rectanglelabels":
                    return "doc_layout"
                if t == "labels":
                    val = r.get("value", {})
                    if "start" in val and "end" in val:
                        return "text_ner"
    return "conversation"


def map_tasks_to_jsonl(
    tasks: list[dict[str, Any]], kind: ImportKind
) -> list[dict[str, Any]]:
    """Convert LS tasks to JSONL records in our target format.

    Tasks with no completed annotations are skipped. Bad records are
    logged and skipped (not raised) so one corrupted annotation can't
    abort a 10k-task import.
    """
    out: list[dict[str, Any]] = []
    for task in tasks:
        annotations = task.get("annotations") or []
        # Pick the first non-skipped annotation. Multi-annotator
        # consolidation is out of scope for the import bridge.
        chosen = next(
            (a for a in annotations if not a.get("was_cancelled")),
            None,
        )
        if chosen is None:
            continue
        try:
            if kind == "doc_layout":
                rec = _map_doc_layout(task, chosen)
            elif kind == "text_ner":
                rec = _map_text_ner(task, chosen)
            else:
                rec = _map_conversation(task, chosen)
        except Exception as e:
            logger.warning(
                "ls import: skipping task=%s — %s",
                task.get("id"),
                e,
            )
            continue
        if rec is not None:
            out.append(rec)
    return out


def _map_conversation(
    task: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, Any] | None:
    data = task.get("data") or {}
    prompt: str | None = None
    for key in ("prompt", "text", "input", "question"):
        v = data.get(key)
        if isinstance(v, str) and v.strip():
            prompt = v
            break
    if prompt is None:
        raise ValueError("no prompt/text field on task")

    # The most common annotation shapes for free-text replies are
    # textarea (single string) and choices (list).
    response_parts: list[str] = []
    for r in annotation.get("result", []) or []:
        val = r.get("value", {}) or {}
        if "text" in val:
            txt = val["text"]
            if isinstance(txt, list):
                response_parts.extend(str(t) for t in txt)
            else:
                response_parts.append(str(txt))
        elif "choices" in val:
            chs = val["choices"]
            if isinstance(chs, list):
                response_parts.append(", ".join(str(c) for c in chs))
    response = "\n".join(p for p in response_parts if p)
    if not response:
        return None
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def _map_text_ner(
    task: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, Any] | None:
    data = task.get("data") or {}
    text = data.get("text")
    if not isinstance(text, str):
        raise ValueError("no string 'text' field on task")
    entities: list[dict[str, Any]] = []
    for r in annotation.get("result", []) or []:
        if r.get("type") != "labels":
            continue
        val = r.get("value", {}) or {}
        if "start" not in val or "end" not in val:
            continue
        labels = val.get("labels") or []
        label = str(labels[0]) if labels else ""
        entities.append(
            {
                "start": int(val["start"]),
                "end": int(val["end"]),
                "label": label,
            }
        )
    return {"text": text, "entities": entities}


def _map_doc_layout(
    task: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, Any] | None:
    data = task.get("data") or {}
    image = (
        data.get("image")
        or data.get("ocr")
        or data.get("img")
    )
    if not isinstance(image, str):
        raise ValueError("no image field on task")
    boxes: list[dict[str, Any]] = []
    for r in annotation.get("result", []) or []:
        if r.get("type") != "rectanglelabels":
            continue
        val = r.get("value", {}) or {}
        try:
            x = float(val["x"])
            y = float(val["y"])
            w = float(val["width"])
            h = float(val["height"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"malformed rect: {e}") from None
        labels = val.get("rectanglelabels") or val.get("labels") or []
        label = str(labels[0]) if labels else ""
        # LS reports rectangles as percentages of the original image
        # dimensions; the original_width/height are in result, not value.
        ow = r.get("original_width")
        oh = r.get("original_height")
        if isinstance(ow, int | float) and isinstance(oh, int | float):
            box = [
                x * ow / 100.0,
                y * oh / 100.0,
                (x + w) * ow / 100.0,
                (y + h) * oh / 100.0,
            ]
        else:
            # No original dims — keep as percentages.
            box = [x, y, x + w, y + h]
        boxes.append({"box": box, "label": label})
    return {"images": [image], "gold_boxes": boxes}


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def import_project(
    *,
    base_url: str,
    token: str,
    project_id: int,
    import_kind: ImportKind | None = None,
    since_iso: str | None = None,
    transport: LsTransport | None = None,
    max_tasks: int | None = None,
) -> tuple[ImportKind, list[dict[str, Any]]]:
    """Pull + map a project. Returns ``(detected_kind, records)``.

    ``transport`` is optional — tests inject a fake; production opens a
    short-lived httpx client (closed on exit so we don't leak sockets
    per import). ``import_kind`` overrides detection. SSRF guard runs
    before any network call when no transport is injected.
    """
    if transport is None:
        canonical = _validate_base_url(base_url)
        with httpx.Client(
            base_url=canonical.rstrip("/"),
            headers={"Authorization": f"Token {token}"},
            timeout=httpx.Timeout(30.0, connect=5.0),
            # No following of cross-origin redirects (DNS-rebinding /
            # second-stage SSRF defense).
            follow_redirects=False,
        ) as client:
            transport = _make_transport_factory(client, token)
            tasks = fetch_completed_tasks(
                transport, project_id, since_iso=since_iso, max_tasks=max_tasks
            )
    else:
        tasks = fetch_completed_tasks(
            transport, project_id, since_iso=since_iso, max_tasks=max_tasks
        )
    if not tasks:
        return import_kind or "conversation", []
    kind = import_kind or detect_import_kind(tasks)
    records = map_tasks_to_jsonl(tasks, kind)
    return kind, records


def strip_url_credentials(url: str) -> str:
    """Return ``url`` with any ``user:pass@`` removed. Safe for logs."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    netloc = host if not parts.port else f"{host}:{parts.port}"
    return urlunsplit(
        (parts.scheme, netloc, parts.path or "", parts.query, parts.fragment)
    )


def write_jsonl(records: list[dict[str, Any]], path: str) -> int:
    """Write records as JSONL. Returns line count for the caller."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n
