"""REST API for the inference playground (Phase 8).

Endpoints
---------

* ``POST /inferences`` — synchronous: load the backend (LRU-cached) and
  return the full prediction. ``{"model_ref", "prompt", "params"?}``.
* ``POST /inferences/stream`` — same payload, response is SSE with one
  ``token`` event per chunk and a final ``done`` event.
* ``POST /inferences/compare`` — same prompt against multiple ``model_refs``,
  returns a list of predictions side-by-side. Honors LRU caching so the
  same ref re-used across requests doesn't reload.

Model-ref syntax (see :mod:`trainpipe.inference.service`):

* ``base:<hf-id-or-path>``  — bare base model.
* ``exp:<experiment_id>``   — base + experiment-trained adapter.
* ``<name>@<alias>``        — registered alias (e.g. ``invoice@production``).
* ``<name>@<version:int>``  — explicit registered version.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette.sse import EventSourceResponse

from ...inference.service import InferenceService, UnknownModelRef
from ..auth import require_api_key
from ..deps import get_inference_service
from ..schemas import InferenceParams

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/inferences",
    tags=["inferences"],
    dependencies=[Depends(require_api_key)],
)


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_ref: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    params: InferenceParams = Field(default_factory=InferenceParams)


class InferenceCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_refs: list[str] = Field(..., min_length=1, max_length=8)
    prompt: str = Field(..., min_length=1)
    params: InferenceParams = Field(default_factory=InferenceParams)


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_ref: str
    base_model: str
    adapter_path: str | None
    prediction: str


class InferenceCompareResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    results: list[InferenceResponse]


async def _resolve_and_predict(
    service: InferenceService,
    raw_ref: str,
    prompt: str,
    params: InferenceParams,
) -> InferenceResponse:
    try:
        resolved = await service.resolve(raw_ref)
    except UnknownModelRef as e:
        raise HTTPException(
            422,
            {"error": "unknown_model_ref", "ref": e.raw, "reason": e.reason},
        ) from None
    backend = await service.get(resolved)
    try:
        prediction = await backend.predict({"prompt": prompt}, params)
    except Exception:
        # Predict raised on a cached backend — its state is suspect
        # (CUDA OOM, half-detokenized state, etc.). Evict so the next
        # request triggers a fresh load instead of reusing the broken one.
        logger.exception("predict failed for %s", raw_ref)
        await service.invalidate(resolved)
        raise
    return InferenceResponse(
        model_ref=raw_ref,
        base_model=resolved.base_model,
        adapter_path=resolved.adapter_path,
        prediction=prediction,
    )


@router.post("")
async def infer(
    request: InferenceRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> InferenceResponse:
    return await _resolve_and_predict(
        service, request.model_ref, request.prompt, request.params
    )


@router.post("/stream")
async def infer_stream(
    request: InferenceRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> EventSourceResponse:
    """Stream the prediction back via SSE.

    The backend interface doesn't expose token-level streaming yet, so we
    fall back to chunked delivery: predict the full string, then emit it
    in ``~64-char`` chunks. Token streaming is a future enhancement and
    will plug in here without changing the wire protocol (chunk size is
    not part of the contract).
    """
    try:
        resolved = await service.resolve(request.model_ref)
    except UnknownModelRef as e:
        raise HTTPException(
            422,
            {"error": "unknown_model_ref", "ref": e.raw, "reason": e.reason},
        ) from None

    async def event_source():
        try:
            backend = await service.get(resolved)
        except Exception as e:
            logger.exception("stream load failed")
            yield {"event": "error", "data": str(e)}
            return
        try:
            prediction = await backend.predict(
                {"prompt": request.prompt}, request.params
            )
        except Exception as e:
            logger.exception("stream predict failed")
            # Evict broken backend so the next request reloads.
            await service.invalidate(resolved)
            yield {"event": "error", "data": str(e)}
            return
        # ``~64-char`` chunks. Use len(prediction) as ground truth so the
        # consumer sees roughly streaming UX without forcing a tokenizer.
        chunk_size = 64
        for i in range(0, len(prediction), chunk_size):
            yield {"event": "token", "data": prediction[i : i + chunk_size]}
            await asyncio.sleep(0)  # let the event loop flush
        yield {
            "event": "done",
            "data": str(len(prediction)),
        }

    return EventSourceResponse(event_source())


@router.post("/compare")
async def infer_compare(
    request: InferenceCompareRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> InferenceCompareResponse:
    """Run the same prompt against multiple model refs.

    Refs are processed sequentially (not concurrently) — the LRU cache
    only holds ``max_loaded`` backends, and two concurrent loads against
    the same single-GPU box would compete for memory anyway. Concurrent
    runs would also evict each other from the cache mid-flight.
    """
    results: list[InferenceResponse] = []
    for raw in request.model_refs:
        results.append(
            await _resolve_and_predict(service, raw, request.prompt, request.params)
        )
    return InferenceCompareResponse(prompt=request.prompt, results=results)


@router.get("/cache")
async def inspect_cache(
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> dict[str, Any]:
    """Diagnostic: which backends are currently loaded (LRU order)."""
    return {
        "max_loaded": service.max_loaded,
        "loaded": [
            {"base_model": b, "adapter_path": a}
            for (b, a) in service.cache_keys()
        ],
    }
