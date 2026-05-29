"""Inference backends for the eval runner.

Architecture: the runner doesn't care *how* predictions are produced — it
just needs ``predict(sample, params) -> str``. The concrete backend is
swapped per-deployment:

* :class:`TransformersInferenceBackend` — production default. Loads the
  base model + LoRA adapter via Hugging Face ``transformers`` and ``peft``
  and generates greedy/sampled text. Lazy-imports the heavy deps so the
  trainpipe API still boots without ``torch`` available.
* :class:`MockInferenceBackend` — for tests. Maps prompt-derived keys to
  canned responses; no model load, no network.

Adding a new backend (e.g. swift CLI, vLLM, sglang) means subclassing
:class:`InferenceBackend` and wiring it in via the dispatcher's
``backend_factory``.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..api.schemas import InferenceParams

logger = logging.getLogger(__name__)


PromptExtractor = Callable[[dict[str, Any]], str]


def default_prompt_extractor(sample: dict[str, Any]) -> str:
    """Pull a single prompt string out of a sample dict.

    Order of fallbacks: ``messages`` (chat) → ``prompt`` → ``query`` →
    ``input``. For ``messages``, the *last user message before any
    assistant reply* is used so the gold assistant response stays out of
    the input. Raises ``ValueError`` if no prompt field is found.
    """
    msgs = sample.get("messages")
    if isinstance(msgs, list) and msgs:
        user_msgs = [
            m for m in msgs if isinstance(m, dict) and m.get("role") == "user"
        ]
        if user_msgs:
            return str(user_msgs[-1].get("content", ""))
    for key in ("prompt", "query", "input"):
        if key in sample:
            return str(sample[key])
    raise ValueError(
        "sample has no prompt field (expected one of: messages, prompt, query, input)"
    )


class InferenceBackend(ABC):
    """Stateful: ``open`` loads the model, ``predict`` per sample, ``close`` frees."""

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def predict(
        self, sample: dict[str, Any], params: InferenceParams
    ) -> str: ...

    @abstractmethod
    async def close(self) -> None: ...


class MockInferenceBackend(InferenceBackend):
    """Returns canned responses for tests.

    Configure either with ``responses_by_key`` (lookup keyed by the
    extracted prompt) or ``response_fn`` (full programmatic control).
    A missing key falls back to ``default_response``.
    """

    def __init__(
        self,
        *,
        responses_by_key: dict[str, str] | None = None,
        response_fn: Callable[[dict[str, Any], InferenceParams], str] | None = None,
        default_response: str = "",
        prompt_extractor: PromptExtractor = default_prompt_extractor,
    ) -> None:
        self._by_key = dict(responses_by_key or {})
        self._fn = response_fn
        self._default = default_response
        self._extractor = prompt_extractor
        self.predict_calls: list[dict[str, Any]] = []
        self._opened = False
        self._closed = False

    async def open(self) -> None:
        self._opened = True

    async def predict(
        self, sample: dict[str, Any], params: InferenceParams
    ) -> str:
        self.predict_calls.append({"sample": sample, "params": params})
        if self._fn is not None:
            return self._fn(sample, params)
        try:
            key = self._extractor(sample)
        except ValueError:
            key = ""
        return self._by_key.get(key, self._default)

    async def close(self) -> None:
        self._closed = True


class TransformersInferenceBackend(InferenceBackend):
    """Loads base + LoRA adapter via HF transformers + peft.

    Heavy deps (``torch``, ``transformers``, ``peft``) are imported lazily
    inside :meth:`open` so that the rest of the trainpipe API can run on
    dev machines that don't have them.
    """

    def __init__(
        self,
        *,
        base_model: str,
        adapter_path: Path | None,
        gpu_indices: list[int],
        prompt_extractor: PromptExtractor = default_prompt_extractor,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.gpu_indices = gpu_indices
        self._extractor = prompt_extractor
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    async def open(self) -> None:
        import asyncio

        await asyncio.to_thread(self._open_sync)

    def _open_sync(self) -> None:
        import os

        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        if self.gpu_indices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in self.gpu_indices
            )
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            device_map="auto" if self._device == "cuda" else None,
        )

        if self.adapter_path is not None:
            from peft import PeftModel  # type: ignore[import-not-found]

            model = PeftModel.from_pretrained(model, str(self.adapter_path))

        self._model = model

    async def predict(
        self, sample: dict[str, Any], params: InferenceParams
    ) -> str:
        import asyncio

        return await asyncio.to_thread(self._predict_sync, sample, params)

    def _predict_sync(
        self, sample: dict[str, Any], params: InferenceParams
    ) -> str:
        import torch  # type: ignore[import-not-found]

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("backend not opened")

        prompt = self._extractor(sample)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        do_sample = params.temperature > 0
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=params.max_new_tokens,
                do_sample=do_sample,
                temperature=params.temperature if do_sample else 1.0,
                top_p=params.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0][prompt_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    async def close(self) -> None:
        import asyncio

        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]

            del self._model
            self._model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("error closing TransformersInferenceBackend")
