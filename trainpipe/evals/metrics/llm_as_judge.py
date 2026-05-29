"""LLM-as-judge metric.

A teacher LLM scores each prediction against the gold reference using a
YAML-style rubric. Anthropic and OpenAI providers are first-class
(architectural decision: direct SDK, provider env-configurable).

Config:

* ``provider`` (str, default ``"anthropic"``) — ``"anthropic"`` or ``"openai"``.
* ``model`` (str, required for production) — e.g. ``"claude-sonnet-4-6"``,
  ``"gpt-4o"``.
* ``rubric`` (dict, required) — has at minimum::

      criteria: "Score how well the prediction extracts X from Y"
      scale: {min: 1, max: 5}

  Optional fields: ``examples`` (list of {prediction, gold, score}),
  ``system_prompt`` (str override), ``output_field`` (str, default
  ``"score"``).

* ``gold_field`` (str, default ``"gold"``) — passed verbatim into the
  judge prompt as the reference answer.
* ``max_retries`` (int, default 2) — judge API failures retried this many
  times before falling back to 0.0.

The judge's output must be valid JSON containing the chosen score field.
The score is then normalized to ``[0, 1]`` using ``scale.min/max``.

Tests bypass the real SDK by passing a callable in the constructor via
``judge_callable`` (kw-only); production code never sets this.
"""

import json
import logging
import os
from collections.abc import Callable
from typing import Any

from .base import Metric, register

logger = logging.getLogger(__name__)


_DEFAULT_SYSTEM = (
    "You are a strict evaluator. Compare the candidate answer to the "
    "reference answer using the supplied rubric, then reply with a single "
    "JSON object containing the score under the requested field name."
)


def _render_prompt(
    rubric: dict[str, Any],
    prediction: str,
    sample: dict[str, Any],
    gold_field: str,
    output_field: str,
) -> str:
    parts = [
        f"Criteria: {rubric.get('criteria', '<no criteria provided>').strip()}",
        f"Scale: integer from {rubric['scale']['min']} (worst) to "
        f"{rubric['scale']['max']} (best).",
        "",
        f"Reference: {sample.get(gold_field, '<no gold provided>')}",
        f"Candidate: {prediction}",
    ]
    examples = rubric.get("examples") or []
    if examples:
        parts.append("")
        parts.append("Examples:")
        for ex in examples:
            parts.append(
                f"- candidate={ex.get('prediction')!r} "
                f"reference={ex.get('gold')!r} score={ex.get('score')}"
            )
    parts.append("")
    parts.append(
        f'Reply with JSON only, like {{"{output_field}": <integer>}}.'
    )
    return "\n".join(parts)


def _parse_score(reply: str, output_field: str) -> float:
    # Greedy: find first ``{`` … ``}`` block. Judges sometimes prefix with
    # whitespace or a leading comment despite instructions.
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"judge reply has no JSON object: {reply!r}")
    payload = json.loads(reply[start : end + 1])
    if output_field not in payload:
        raise ValueError(
            f"judge reply missing field '{output_field}': {payload!r}"
        )
    return float(payload[output_field])


@register
class LLMAsJudgeMetric(Metric):
    kind = "llm_as_judge"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        judge_callable: Callable[[str], str] | None = None,
    ) -> None:
        self._judge_override = judge_callable
        super().__init__(config)

    def _validate_config(self) -> None:
        self.provider: str = str(self.config.get("provider", "anthropic")).lower()
        if self.provider not in ("anthropic", "openai"):
            raise ValueError(
                f"llm_as_judge.provider must be 'anthropic' or 'openai', "
                f"got '{self.provider}'"
            )
        self.model: str = str(self.config.get("model", "")).strip()
        if self._judge_override is None and not self.model:
            raise ValueError("llm_as_judge.model is required in production config")

        rubric = self.config.get("rubric")
        if not isinstance(rubric, dict):
            raise ValueError("llm_as_judge.rubric must be a dict")
        scale = rubric.get("scale")
        if (
            not isinstance(scale, dict)
            or "min" not in scale
            or "max" not in scale
            or float(scale["max"]) <= float(scale["min"])
        ):
            raise ValueError(
                "llm_as_judge.rubric.scale must have numeric min<max"
            )
        self.rubric = rubric
        self.scale_min = float(scale["min"])
        self.scale_max = float(scale["max"])
        self.output_field: str = str(rubric.get("output_field", "score"))
        self.system_prompt: str = str(
            rubric.get("system_prompt", _DEFAULT_SYSTEM)
        )

        self.gold_field: str = self.config.get("gold_field", "gold")
        self.max_retries: int = int(self.config.get("max_retries", 2))

    def _call_judge(self, prompt: str) -> str:
        if self._judge_override is not None:
            return self._judge_override(prompt)
        if self.provider == "anthropic":
            return _call_anthropic(self.model, self.system_prompt, prompt)
        if self.provider == "openai":
            return _call_openai(self.model, self.system_prompt, prompt)
        raise RuntimeError(f"unsupported provider: {self.provider}")

    def score(self, prediction: str, sample: dict[str, Any]) -> float:
        prompt = _render_prompt(
            self.rubric, prediction, sample, self.gold_field, self.output_field
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                reply = self._call_judge(prompt)
                raw = _parse_score(reply, self.output_field)
                norm = (raw - self.scale_min) / (self.scale_max - self.scale_min)
                return max(0.0, min(1.0, norm))
            except Exception as e:
                last_err = e
                logger.warning(
                    "llm_as_judge attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    e,
                )
        logger.error("llm_as_judge giving up after retries: %s", last_err)
        return 0.0


def _call_anthropic(model: str, system: str, prompt: str) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "llm_as_judge with provider=anthropic requires the 'anthropic' SDK"
        ) from e
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY must be set to use llm_as_judge with provider=anthropic"
        )
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=256,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    # Concatenate text-typed content blocks (ignores tool_use blocks).
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


def _call_openai(model: str, system: str, prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError(
            "llm_as_judge with provider=openai requires the 'openai' SDK"
        ) from e
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY must be set to use llm_as_judge with provider=openai"
        )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=256,
    )
    return resp.choices[0].message.content or ""
