"""REST API for the named model registry (Phase 7).

A "model" is a named, versioned, immutable pointer to one experiment's
adapter output dir. Aliases are mutable labels that move between versions
within a family (``invoice-extractor@production`` etc.).

Endpoints
---------

* ``POST   /models`` — register one experiment as a named model version.
  Auto-increments ``version`` within ``name`` if omitted. Optionally moves
  an alias to the freshly registered version in the same call.
* ``GET    /models`` — list, optional ``name``/``alias`` filter.
* ``GET    /models/{name}`` — all versions under one family.
* ``GET    /models/{name}/{alias_or_version}`` — resolve. Numeric segments
  are matched against ``version``; non-numeric against ``alias``.
* ``POST   /models/{name}/aliases/{alias}`` — assign or move alias.
  Body: ``{"model_id": "..."}`` OR ``{"version": N}``.
* ``DELETE /models/{name}/aliases/{alias}`` — drop alias mapping.
* ``DELETE /models/{id}`` — remove a version (CASCADEs alias mappings).
  409 if any alias still references it unless ``?force=true``.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

from ...core import repository
from ...core.db import Database
from ...settings import settings
from ..auth import require_api_key
from ..deps import get_db
from ..schemas import (
    ExperimentStatus,
    ModelRegisterRequest,
    RegisteredModel,
)

router = APIRouter(
    prefix="/models",
    tags=["models"],
    dependencies=[Depends(require_api_key)],
)


class AliasAssign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    version: int | None = Field(None, ge=1)


def _adapter_path_for(experiment_id: str, spec_output_dir: str | None) -> str:
    """Where the trained adapter lives on disk for ``experiment_id``."""
    if spec_output_dir:
        return spec_output_dir
    return str(settings.output_base_dir / experiment_id)


async def _eval_summary_for(conn, experiment_id: str) -> dict[str, Any] | None:
    """Pick the most recent completed eval run per suite and flatten its
    aggregate into ``{suite_id: {metric: mean}}`` for the model summary.

    Empty dict when no eval runs exist — callers treat that as "missing
    promotion evidence" which the UI surfaces as a warning.
    """
    runs = await repository.list_eval_runs(conn, experiment_id=experiment_id, limit=500)
    # newest first: list_eval_runs orders by created_at DESC.
    seen_suites: set[str] = set()
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        if run.suite_id in seen_suites:
            continue
        seen_suites.add(run.suite_id)
        if run.aggregate is None:
            continue
        out[run.suite_id] = {
            name: float(agg.mean) for name, agg in run.aggregate.items()
        }
    return out or None


@router.post("", status_code=201)
async def register_model(
    request: ModelRegisterRequest,
    db: Annotated[Database, Depends(get_db)],
) -> RegisteredModel:
    async with db.connect() as conn:
        exp = await repository.get_experiment(conn, request.experiment_id)
        if exp is None:
            raise HTTPException(
                422,
                {
                    "error": "unknown_experiment",
                    "experiment_id": request.experiment_id,
                },
            )
        if exp.status != ExperimentStatus.COMPLETED:
            raise HTTPException(
                422,
                {
                    "error": "experiment_not_completed",
                    "experiment_id": request.experiment_id,
                    "status": exp.status.value,
                },
            )

        adapter_path = _adapter_path_for(
            request.experiment_id, exp.spec.output_dir
        )
        eval_summary = await _eval_summary_for(conn, request.experiment_id)

        try:
            model_id, _version = await repository.register_model_atomic(
                conn,
                name=request.name,
                explicit_version=request.version,
                base_model=exp.spec.model,
                adapter_path=adapter_path,
                experiment_id=request.experiment_id,
                run_id=exp.mlflow_run_id,
                eval_summary=eval_summary,
                description=request.description,
                alias=request.alias,
            )
        except ValueError as e:
            if str(e) == "version_exists":
                raise HTTPException(
                    409,
                    {
                        "error": "version_exists",
                        "name": request.name,
                        "version": request.version,
                    },
                ) from None
            raise

        # Phase 15: record lineage so the GDPR-audit can answer
        # "which models trained on dataset X?". Best-effort — failure
        # here doesn't roll back the registration.
        try:
            spec_paths = [
                ref.split("#", 1)[0]
                for ref in list(exp.spec.dataset) + list(exp.spec.val_dataset)
            ]
            path_to_ds = await repository.datasets_by_paths(conn, spec_paths)
            ds_ids = list({did for did in path_to_ds.values()})
            if ds_ids:
                await repository.record_model_lineage(conn, model_id, ds_ids)
        except Exception:
            logger.exception(
                "lineage record failed for model=%s (non-fatal)", model_id
            )

        model = await repository.get_model(conn, model_id)
    assert model is not None
    return model


@router.get("")
async def list_models(
    db: Annotated[Database, Depends(get_db)],
    name: str | None = None,
    alias: str | None = None,
) -> list[RegisteredModel]:
    async with db.connect() as conn:
        return await repository.list_models(conn, name=name, alias=alias)


@router.get("/{name}")
async def list_models_by_name(
    name: str,
    db: Annotated[Database, Depends(get_db)],
) -> list[RegisteredModel]:
    async with db.connect() as conn:
        return await repository.list_models_by_name(conn, name)


class TrainedOnDataset(BaseModel):
    """One dataset that a model was trained on."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    version: int
    path: str
    line_count: int | None = None
    media_kinds: list[str] = Field(default_factory=list)


@router.get("/{model_id}/datasets")
async def list_trained_on_datasets(
    model_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> list[TrainedOnDataset]:
    """All datasets recorded as training inputs for ``model_id``.

    The lineage rows are populated at register-time by walking the
    experiment's ``spec.dataset`` / ``val_dataset`` and matching paths
    against the registered datasets — so this only includes registered
    datasets the experiment actually used. HF-id refs (anything that
    wasn't a registered upload) won't show up here.

    Declared BEFORE ``/{name}/{alias_or_version}`` so FastAPI matches
    the trailing literal ``/datasets`` segment first; otherwise a call
    like ``GET /models/<hex>/datasets`` would resolve to the alias
    lookup with ``alias_or_version = "datasets"``.
    """
    async with db.connect() as conn:
        model = await repository.get_model(conn, model_id)
        if model is None:
            raise HTTPException(404, "model not found")
        ds_ids = await repository.datasets_used_by_model(conn, model_id)
        if not ds_ids:
            return []
        # Fetch each dataset row; the repo doesn't have a bulk-by-id
        # helper today and the typical model has 1-5 datasets so the
        # extra round-trips are fine.
        out: list[TrainedOnDataset] = []
        for ds_id in ds_ids:
            rec = await repository.get_dataset(conn, ds_id)
            if rec is None:
                # CASCADE delete should have removed the lineage row;
                # belt-and-suspenders so a corrupt row doesn't 500 us.
                continue
            out.append(
                TrainedOnDataset(
                    id=rec.id,
                    name=rec.name,
                    version=rec.version,
                    path=rec.path,
                    line_count=rec.line_count,
                    media_kinds=rec.media_kinds,
                )
            )
    return out


@router.get("/{name}/{alias_or_version}")
async def resolve_model(
    name: str,
    alias_or_version: str,
    db: Annotated[Database, Depends(get_db)],
) -> RegisteredModel:
    async with db.connect() as conn:
        model: RegisteredModel | None
        if alias_or_version.isdigit():
            model = await repository.get_model_by_name_version(
                conn, name, int(alias_or_version)
            )
        else:
            model = await repository.resolve_model_alias(
                conn, name, alias_or_version
            )
    if model is None:
        raise HTTPException(
            404,
            {
                "error": "not_found",
                "name": name,
                "ref": alias_or_version,
            },
        )
    return model


@router.post("/{name}/aliases/{alias}")
async def assign_alias(
    name: str,
    alias: str,
    payload: AliasAssign,
    db: Annotated[Database, Depends(get_db)],
) -> RegisteredModel:
    if payload.model_id is None and payload.version is None:
        raise HTTPException(
            422,
            {"error": "missing_target", "hint": "pass model_id or version"},
        )
    async with db.connect() as conn:
        target: RegisteredModel | None
        if payload.model_id is not None:
            target = await repository.get_model(conn, payload.model_id)
        else:
            assert payload.version is not None
            target = await repository.get_model_by_name_version(
                conn, name, payload.version
            )
        if target is None:
            raise HTTPException(404, "model not found")
        if target.name != name:
            raise HTTPException(
                422,
                {
                    "error": "cross_family_alias",
                    "alias_family": name,
                    "model_family": target.name,
                },
            )
        await repository.set_model_alias(
            conn, name=name, alias=alias, model_id=target.id
        )
        refreshed = await repository.get_model(conn, target.id)
    assert refreshed is not None
    return refreshed


@router.delete("/{name}/aliases/{alias}")
async def remove_alias(
    name: str,
    alias: str,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, str]:
    async with db.connect() as conn:
        ok = await repository.delete_model_alias(conn, name, alias)
    return {"status": "deleted" if ok else "not_found"}


class QuantizeRequest(BaseModel):
    """Quantize an existing model and register the output as a new
    version under the same family (e.g. ``invoice-extractor`` v3 → v4)."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., min_length=1, description="awq or gptq")
    bits: int = Field(..., ge=2, le=16)
    description: str | None = None


# Injected for tests so the route can run without real GPU + swift.
_quantize_backend_override = None


def _set_quantize_backend(backend) -> None:
    global _quantize_backend_override
    _quantize_backend_override = backend


@router.post("/{model_id}/quantize", status_code=201)
async def quantize_model_route(
    model_id: str,
    request: QuantizeRequest,
    db: Annotated[Database, Depends(get_db)],
) -> RegisteredModel:
    import asyncio

    from ...quantization.runner import quantize_model

    if request.method not in ("awq", "gptq"):
        raise HTTPException(
            422,
            {"error": "unsupported_method", "method": request.method},
        )

    async with db.connect() as conn:
        parent = await repository.get_model(conn, model_id)
    if parent is None:
        raise HTTPException(404, "model not found")
    if parent.adapter_path is None:
        raise HTTPException(
            422,
            {
                "error": "no_adapter_path",
                "detail": "parent model has no on-disk adapter to quantize",
            },
        )

    out_dir = (
        settings.output_base_dir
        / "quantized"
        / model_id
        / f"{request.method}-{request.bits}bit"
    )

    def _do():
        return quantize_model(
            source_adapter_path=parent.adapter_path,
            out_dir=out_dir,
            method=request.method,  # type: ignore[arg-type]
            bits=request.bits,
            backend=_quantize_backend_override,
        )

    try:
        result = await asyncio.to_thread(_do)
    except Exception as e:
        logger.exception("quantize failed for model=%s", model_id)
        raise HTTPException(
            500, {"error": "quantize_failed", "detail": str(e)}
        ) from None

    summary = parent.eval_summary  # keep parent's evals as initial baseline
    desc = (
        request.description
        or f"quantized {request.method}:{request.bits}bit from model {parent.id}"
    )
    async with db.connect() as conn:
        new_id, version = await repository.register_model_atomic(
            conn,
            name=parent.name,
            explicit_version=None,
            base_model=parent.base_model,
            adapter_path=result.output_dir,
            experiment_id=parent.experiment_id,
            run_id=parent.run_id,
            eval_summary=summary,
            description=desc,
            alias=None,
        )
        new_model = await repository.get_model(conn, new_id)
    assert new_model is not None
    return new_model


@router.delete("/{model_id}", status_code=200)
async def delete_model(
    model_id: str,
    db: Annotated[Database, Depends(get_db)],
    force: bool = Query(False),
) -> dict[str, Any]:
    async with db.connect() as conn:
        model = await repository.get_model(conn, model_id)
        if model is None:
            raise HTTPException(404, "model not found")
        if model.aliases and not force:
            raise HTTPException(
                409,
                {
                    "error": "model_has_aliases",
                    "aliases": model.aliases,
                    "hint": "pass force=true to delete anyway",
                },
            )
        ok = await repository.delete_model(conn, model_id)
    return {"deleted": ok}
