"""Synthetic-data runner.

A run reads ``source_dataset_path``, sends each (or a sampled subset of)
source record to the teacher LLM with the user's ``instruction``, and
writes the model's reply as a new JSONL record. The output dataset is
registered with a provenance description so audit later can trace
``ds:<id>`` back to its source + teacher.
"""

from __future__ import annotations

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SynthProvider(ABC):
    """One synchronous LLM-call provider. Implementations should retry
    once on 429/5xx but otherwise stay simple."""

    name: str

    @abstractmethod
    def generate(self, prompt: str, *, model: str, max_tokens: int) -> str:
        ...


# ---------------------------------------------------------------------------
# Concrete providers
# ---------------------------------------------------------------------------


class AnthropicProvider(SynthProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "AnthropicProvider needs ANTHROPIC_API_KEY env var or "
                "explicit api_key"
            )

    def generate(self, prompt: str, *, model: str, max_tokens: int) -> str:
        with httpx.Client(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            resp = client.post(
                "/v1/messages",
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.is_error:
                raise RuntimeError(
                    f"anthropic HTTP {resp.status_code}: {resp.text[:512]}"
                )
            body = resp.json()
            # Concatenate text blocks.
            blocks = body.get("content") or []
            return "".join(
                b.get("text", "") for b in blocks if b.get("type") == "text"
            )


class OpenAIProvider(SynthProvider):
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OpenAIProvider needs OPENAI_API_KEY env var or explicit api_key"
            )
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "https://api.openai.com"
        )

    def generate(self, prompt: str, *, model: str, max_tokens: int) -> str:
        with httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.is_error:
                raise RuntimeError(
                    f"openai HTTP {resp.status_code}: {resp.text[:512]}"
                )
            body = resp.json()
            choices = body.get("choices") or []
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "") or ""


class MockProvider(SynthProvider):
    """For tests: returns the prompt with a prefix so we can assert flow."""

    name = "mock"

    def __init__(self, transform=None) -> None:
        self._transform = transform or (lambda p: f"synth({p})")

    def generate(self, prompt: str, *, model: str, max_tokens: int) -> str:
        return self._transform(prompt)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def _iter_source(
    path: Path, seed: int, sample_n: int | None = None
) -> Iterator[dict[str, Any]]:
    """Stream the source JSONL. If ``sample_n`` is set, randomly sample N
    records (with replacement) — that's how we expand 1k → 5k.
    """
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    if not records:
        return
    if sample_n is None:
        yield from records
        return
    rng = random.Random(seed)
    for _ in range(sample_n):
        yield rng.choice(records)


def _format_prompt(record: dict[str, Any], instruction: str) -> str:
    """Combine the user's instruction with the source record's content.

    The teacher LLM sees:
        INSTRUCTION
        <source record JSON>
    """
    return f"{instruction.strip()}\n\nSource record:\n{json.dumps(record, ensure_ascii=False)}"


def generate_synthetic(
    *,
    provider: SynthProvider,
    model: str,
    source_path: Path,
    instruction: str,
    target_count: int,
    out_path: Path,
    seed: int = 0,
    max_tokens: int = 1024,
) -> int:
    """Run the full synth job. Returns the number of records written.

    Failures on individual records are logged and skipped; one bad
    teacher reply doesn't abort the whole batch.
    """
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for source_record in _iter_source(source_path, seed, sample_n=target_count):
            prompt = _format_prompt(source_record, instruction)
            try:
                completion = provider.generate(
                    prompt, model=model, max_tokens=max_tokens
                )
            except Exception:
                logger.exception(
                    "synth: provider call failed; skipping record"
                )
                continue
            record = {
                "prompt": prompt,
                "completion": completion,
                "_source": source_record,
            }
            out.write(json.dumps(record, ensure_ascii=False))
            out.write("\n")
            written += 1
    return written


_PROVIDERS: dict[str, type[SynthProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "mock": MockProvider,
}


def make_provider(name: str) -> SynthProvider:
    cls = _PROVIDERS.get(name)
    if cls is None:
        raise ValueError(
            f"unknown synth provider {name!r}; options: {sorted(_PROVIDERS)}"
        )
    return cls()
