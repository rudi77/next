"""REST API for synthetic data generation (Phase 14).

``POST /synth`` runs a teacher-LLM synthesis job to expand a source
dataset, writes the result as JSONL, and registers it as a new dataset
(with a provenance description so the lineage is auditable).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ...core import repository
from ...core.db import Database
from ...settings import settings
from ...synth.runner import generate_synthetic, make_provider
from ...training.dataset_formats import detect_and_validate_info
from ...training.dataset_refs import (
    MalformedDatasetRef,
    UnknownDatasetRef,
    resolve_single,
)
from ..auth import require_api_key
from ..deps import get_db
from ..schemas import Dataset

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/synth",
    tags=["synth"],
    dependencies=[Depends(require_api_key)],
)


class SynthRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic", "openai", "mock"]
    model: str = Field(..., min_length=1)
    source_dataset: str = Field(
        ...,
        min_length=1,
        description="Source dataset as ``ds:<id>`` ref or filesystem path",
    )
    instruction: str = Field(..., min_length=1)
    target_count: int = Field(..., ge=1, le=10000)
    seed: int = 0
    max_tokens: int = Field(1024, ge=1, le=32768)
    name: str = Field(..., min_length=1, description="Name for the output dataset")


@router.post("", status_code=201)
async def run_synth(
    request: SynthRequest,
    db: Annotated[Database, Depends(get_db)],
) -> Dataset:
    async with db.connect() as conn:
        try:
            source_path = await resolve_single(request.source_dataset, conn)
        except UnknownDatasetRef as e:
            raise HTTPException(
                422, {"error": "unknown_dataset_ref", "ref_id": e.ref_id}
            ) from None
        except MalformedDatasetRef as e:
            raise HTTPException(
                422, {"error": "malformed_dataset_ref", "value": e.raw}
            ) from None

    source_file = Path(source_path.split("#", 1)[0])
    if not source_file.is_file():
        raise HTTPException(
            422, {"error": "source_file_missing", "path": str(source_file)}
        )

    try:
        provider = make_provider(request.provider)
    except RuntimeError as e:
        raise HTTPException(
            422,
            {
                "error": "provider_unavailable",
                "provider": request.provider,
                "detail": str(e),
            },
        ) from None

    dataset_id = uuid.uuid4().hex
    target_dir = settings.datasets_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{request.name}.jsonl"

    # Heavy network calls — push to a thread so the asyncio loop stays
    # responsive to other requests during a long synth.
    def _do() -> int:
        return generate_synthetic(
            provider=provider,
            model=request.model,
            source_path=source_file,
            instruction=request.instruction,
            target_count=request.target_count,
            out_path=target_path,
            seed=request.seed,
            max_tokens=request.max_tokens,
        )

    try:
        written = await asyncio.to_thread(_do)
    except Exception as e:
        logger.exception("synth run failed")
        try:
            target_path.unlink(missing_ok=True)
            target_dir.rmdir()
        except OSError:
            pass
        raise HTTPException(
            500, {"error": "synth_failed", "detail": str(e)}
        ) from None

    if written == 0:
        try:
            target_path.unlink(missing_ok=True)
            target_dir.rmdir()
        except OSError:
            pass
        raise HTTPException(
            422,
            {
                "error": "no_records_generated",
                "detail": "every provider call failed; check teacher model and key",
            },
        )

    sha = hashlib.sha256()
    with target_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    size = target_path.stat().st_size

    try:
        info = detect_and_validate_info(target_path)
    except Exception as e:
        raise HTTPException(
            500, {"error": "output_invalid", "detail": str(e)}
        ) from None

    provenance = (
        f"synthesized via {request.provider}:{request.model} from "
        f"{source_path} with instruction={request.instruction[:80]!r}"
    )

    digest = sha.hexdigest()
    async with db.connect() as conn:
        existing = await repository.get_dataset_by_sha(conn, digest)
        if existing is not None:
            try:
                target_path.unlink(missing_ok=True)
                target_dir.rmdir()
            except OSError:
                pass
            return existing
        await repository.create_dataset(
            conn,
            name=request.name,
            path=str(target_path),
            fmt=info.format,
            size_bytes=size,
            sha256=digest,
            line_count=info.line_count,
            description=provenance,
            dataset_id=dataset_id,
            media_kinds=info.media_kinds,
        )
        rec = await repository.get_dataset(conn, dataset_id)
    assert rec is not None
    return rec
