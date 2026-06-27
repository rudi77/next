"""Pure phase logic for agentic data acquisition — no DB, no asyncio.

Each function here maps to one phase of the design doc and is independently
testable. The :class:`AcquisitionDriver` orchestrates them and owns all the
persistence. We reuse ``synth.runner``'s provider abstraction (Anthropic /
OpenAI / Mock) rather than re-implementing LLM calls — the only difference
from ``synth`` is that acquisition has no source dataset to expand, so it
*generates* records from the spec instead of transforming source records.

Robustness note: both intake and synthesize ask the LLM for JSON and parse
it, but neither *depends* on the model returning well-formed JSON. When a
reply can't be parsed (the ``mock`` provider returns prose, a real model
rambles) we fall back to a deterministic value derived from the spec. That
keeps the MVP fully exercisable end-to-end with ``provider=mock`` and no
network — a real dataset comes out either way.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..api.schemas import AcquisitionSpec
from ..redaction.redactor import redact_record
from ..synth.runner import FatalHTTPError, SynthAborted, SynthProvider
from .web import GateDecision, SearchProvider, TextFetcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort: pull the first ``{...}`` block out of an LLM reply and
    parse it. Returns ``None`` if nothing parseable is found — callers fall
    back to a deterministic value rather than failing the run.

    The greedy ``\\{.*\\}`` (DOTALL) span covers the pure-JSON case too: on a
    reply that is *only* the object it matches the whole string, so there's
    no need for a separate fast path."""
    if not text:
        return None
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Phase 1 — Intake
# ---------------------------------------------------------------------------

_INTAKE_PROMPT = """\
You are planning a dataset to fine-tune an LLM. Turn the operator's brief
into a JSON object with exactly these keys:
  "domain": short domain label (string),
  "locales": list of BCP-47 locales the data should cover,
  "target_capabilities": list of things the model should be able to do,
  "out_of_scope": list of things the model must NOT do,
  "format": one of "sft", "dpo", "chat", "completion",
  "open_questions": list of clarifying questions you'd ask before starting
                    (empty list if the brief is already clear enough).
Reply with ONLY the JSON object, no prose.

Brief: {brief}
"""


def _fallback_domain(brief: str) -> str:
    """A serviceable domain label from the brief's leading words."""
    return " ".join(brief.strip().split()[:6]) or "general"


def _fallback_spec(brief: str, target_count: int) -> AcquisitionSpec:
    """Deterministic spec when the intake LLM gives us nothing parseable.

    Domain = the brief's leading words; no open questions (so the run flows
    straight through). Good enough to demo the pipeline with ``mock``."""
    return AcquisitionSpec(
        domain=_fallback_domain(brief),
        locales=[],
        target_capabilities=[],
        out_of_scope=[],
        format="sft",
        target_count=target_count,
        open_questions=[],
    )


def intake_spec(
    provider: SynthProvider,
    *,
    model: str,
    brief: str,
    target_count: int,
    max_tokens: int = 1024,
) -> AcquisitionSpec:
    """Phase 1: brief → structured :class:`AcquisitionSpec`.

    Always returns a spec; never raises on a bad/empty LLM reply (falls back
    to :func:`_fallback_spec`). ``target_count`` from the request wins over
    anything the model invents, so the operator stays in control of volume.
    """
    prompt = _INTAKE_PROMPT.format(brief=brief)
    try:
        raw = provider.generate(prompt, model=model, max_tokens=max_tokens)
    except Exception:  # network / provider error — degrade, don't crash intake
        logger.warning("acquisition intake: provider call failed, using fallback")
        return _fallback_spec(brief, target_count)

    obj = _extract_json_object(raw)
    if obj is None:
        return _fallback_spec(brief, target_count)

    def _str_list(key: str) -> list[str]:
        v = obj.get(key)
        return [str(x) for x in v] if isinstance(v, list) else []

    fmt = obj.get("format")
    if fmt not in ("sft", "dpo", "chat", "completion"):
        fmt = "sft"
    return AcquisitionSpec(
        domain=str(obj.get("domain") or _fallback_domain(brief)),
        locales=_str_list("locales"),
        target_capabilities=_str_list("target_capabilities"),
        out_of_scope=_str_list("out_of_scope"),
        format=fmt,
        target_count=target_count,
        open_questions=_str_list("open_questions"),
    )


# ---------------------------------------------------------------------------
# Phase 2 — Research (find candidate sources, gate them)
# ---------------------------------------------------------------------------


@dataclass
class SourceEval:
    """One candidate source after the robots/license gate ran. ``used`` is set
    by :func:`acquire_records` once the source has actually been fetched."""

    url: str
    title: str
    topic: str
    license_status: str
    allowed: bool
    used: bool = False


def _research_queries(spec: AcquisitionSpec) -> list[str]:
    """Derive web-search queries from the spec. One per capability (scoped to
    the domain + first locale), falling back to the bare domain."""
    locale = spec.locales[0] if spec.locales else ""
    if spec.target_capabilities:
        parts_list = [[spec.domain, cap, locale] for cap in spec.target_capabilities]
    else:
        parts_list = [[spec.domain, locale]]
    return [" ".join(p for p in parts if p) for parts in parts_list]


def research_sources(
    search_provider: SearchProvider,
    gate: Callable[[str], GateDecision],
    spec: AcquisitionSpec,
    *,
    max_sources: int,
) -> list[SourceEval]:
    """Phase 2: search per query, gate each unique URL, cap at ``max_sources``.

    Every candidate is returned (allowed or not) so the caller can persist the
    full audit trail; only ``allowed`` ones should be fetched downstream. Each
    query only asks for the remaining budget so a paid search API isn't queried
    for more URLs than we'll keep.
    """
    seen: set[str] = set()
    out: list[SourceEval] = []
    for query in _research_queries(spec):
        for hit in search_provider.search(query, max_results=max_sources - len(out)):
            if not hit.url or hit.url in seen:
                continue
            seen.add(hit.url)
            decision = gate(hit.url)
            out.append(
                SourceEval(
                    url=hit.url,
                    title=hit.title,
                    topic=query,
                    license_status=decision.license_status,
                    allowed=decision.allowed,
                )
            )
            if len(out) >= max_sources:
                return out
    return out


# ---------------------------------------------------------------------------
# Phase 3a — Acquire (fetch allowed sources, turn text into records)
# ---------------------------------------------------------------------------

_ACQUIRE_PROMPT = """\
Below is text extracted from a web page about "{domain}".
Write ONE realistic fine-tuning example grounded in this text.
The model should be able to: {capabilities}
The model must NOT: {out_of_scope}
Reply with ONLY a JSON object of the form
  {{"prompt": <the user request>, "completion": <the ideal answer>}}

PAGE TEXT (may be truncated):
{page_text}
"""

# Cap page text sent to the LLM so a long article can't blow up token use.
_MAX_PAGE_CHARS = 4000


def acquire_records(
    provider: SynthProvider,
    *,
    model: str,
    sources: list[SourceEval],
    spec: AcquisitionSpec,
    fetch_text: TextFetcher,
    records_per_source: int = 2,
    max_tokens: int = 1024,
    should_stop: Callable[[], bool] | None = None,
) -> list[dict[str, str]]:
    """Phase 3a: for each allowed source, fetch its text and ask the LLM to
    distil ``records_per_source`` grounded examples.

    Returns the records and marks each fetched source's ``used`` flag in place.
    Sources that fail to fetch/extract are skipped (not fatal). Parse failures
    fall back to a deterministic record so a flaky model still yields data.
    ``should_stop`` is polled per source.
    """
    records: list[dict[str, str]] = []
    for src in sources:
        if should_stop is not None and should_stop():
            break
        if not src.allowed:
            continue
        text = fetch_text(src.url)
        if not text:
            continue
        src.used = True
        prompt = _ACQUIRE_PROMPT.format(
            domain=spec.domain,
            capabilities=", ".join(spec.target_capabilities) or "general competence",
            out_of_scope=", ".join(spec.out_of_scope) or "(no explicit limits)",
            page_text=text[:_MAX_PAGE_CHARS],
        )
        for i in range(records_per_source):
            # Poll per call (not just per source) so a call budget bounds spend
            # tightly rather than overshooting by up to records_per_source.
            if should_stop is not None and should_stop():
                return records
            try:
                raw = provider.generate(prompt, model=model, max_tokens=max_tokens)
            except Exception as e:
                logger.warning("acquisition acquire: provider failed on %s: %s", src.url, e)
                break
            records.append(_parse_record(raw, spec, i))
    return records


# ---------------------------------------------------------------------------
# Phase 3b — Synthesize (from-scratch, no source dataset)
# ---------------------------------------------------------------------------

_SYNTH_PROMPT = """\
Generate ONE realistic training example for fine-tuning a model in the
domain "{domain}".
Locales: {locales}
The model should be able to: {capabilities}
The model must NOT: {out_of_scope}
{answers}
Reply with ONLY a JSON object of the form
  {{"prompt": <the user request>, "completion": <the ideal answer>}}
Make example #{index} distinct from the others.
"""


def _spec_synth_prompt(
    spec: AcquisitionSpec, index: int, answers: dict[str, str] | None
) -> str:
    answer_block = ""
    if answers:
        joined = "; ".join(f"{q} -> {a}" for q, a in answers.items())
        answer_block = f"Operator clarifications: {joined}\n"
    return _SYNTH_PROMPT.format(
        domain=spec.domain,
        locales=", ".join(spec.locales) or "any",
        capabilities=", ".join(spec.target_capabilities) or "general competence",
        out_of_scope=", ".join(spec.out_of_scope) or "(no explicit limits)",
        answers=answer_block,
        index=index + 1,
    )


def _fallback_record(spec: AcquisitionSpec, index: int) -> dict[str, str]:
    """Deterministic prompt/completion when a reply isn't parseable JSON."""
    return {
        "prompt": f"[{spec.domain}] Beispielaufgabe #{index + 1}",
        "completion": (
            f"Beispielantwort #{index + 1} für die Domäne {spec.domain}."
        ),
    }


def _parse_record(raw: str, spec: AcquisitionSpec, index: int) -> dict[str, str]:
    obj = _extract_json_object(raw)
    if (
        obj is not None
        and isinstance(obj.get("prompt"), str)
        and isinstance(obj.get("completion"), str)
        and obj["prompt"]
        and obj["completion"]
    ):
        return {"prompt": obj["prompt"], "completion": obj["completion"]}
    return _fallback_record(spec, index)


def synthesize_records(
    provider: SynthProvider,
    *,
    model: str,
    spec: AcquisitionSpec,
    answers: dict[str, str] | None = None,
    max_tokens: int = 1024,
    max_consecutive_failures: int = 5,
    should_stop=None,
) -> list[dict[str, str]]:
    """Phase 3b: generate ``spec.target_count`` records from the spec.

    Mirrors ``synth.generate_synthetic``'s failure model: a
    :class:`FatalHTTPError` (bad key / malformed request) or
    ``max_consecutive_failures`` provider errors in a row trip
    :class:`SynthAborted` so a wedged provider can't burn through the whole
    target in a tight loop. Parse failures are *not* errors — they fall back
    to a deterministic record. ``should_stop`` is an optional callable polled
    between records so the driver can cancel mid-phase.
    """
    records: list[dict[str, str]] = []
    consecutive_failures = 0
    for i in range(spec.target_count):
        if should_stop is not None and should_stop():
            break
        prompt = _spec_synth_prompt(spec, i, answers)
        try:
            raw = provider.generate(prompt, model=model, max_tokens=max_tokens)
        except FatalHTTPError as e:
            raise SynthAborted(
                f"provider returned fatal error: {e}. Aborting after "
                f"{len(records)} records."
            ) from None
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                "acquisition synth: provider call failed (%d/%d): %s",
                consecutive_failures,
                max_consecutive_failures,
                e,
            )
            if consecutive_failures >= max_consecutive_failures:
                raise SynthAborted(
                    f"{consecutive_failures} consecutive provider failures "
                    f"(last: {e}). Aborting after {len(records)} records."
                ) from None
            continue
        consecutive_failures = 0
        records.append(_parse_record(raw, spec, i))
    return records


# ---------------------------------------------------------------------------
# Phase 4 — Curate
# ---------------------------------------------------------------------------


@dataclass
class CurateStats:
    """What the curate phase did — grows a field per future filter rather than
    a positional tuple slot."""

    dropped: int = 0
    redaction: dict[str, int] = field(default_factory=dict)


def curate(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], CurateStats]:
    """Phase 4: mandatory PII-redaction, then exact-duplicate dedup.

    Returns ``(curated, stats)``. Redaction runs first and is non-optional —
    the design invariant is that no PII reaches a registered dataset. Redacting
    before dedup means records that differ only in redacted PII collapse
    together. Dedup key is the (prompt, completion) pair; order is preserved so
    the output is reproducible.
    """
    redaction_counts: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    curated: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        rec, hits = redact_record(rec)
        for entity, n in hits.items():
            if n:
                redaction_counts[entity] = redaction_counts.get(entity, 0) + n
        key = (
            str(rec.get("prompt", "")).strip(),
            str(rec.get("completion", "")).strip(),
        )
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        curated.append(rec)
    return curated, CurateStats(dropped=dropped, redaction=redaction_counts)


def write_records_jsonl(records: list[dict[str, Any]], path: Path) -> int:
    """Write records as JSONL. Returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    return len(records)
