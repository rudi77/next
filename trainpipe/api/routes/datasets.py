"""POST /datasets uploads a file, validates the format, stores it on disk,
and registers an entry in the SQLite ``datasets`` table. Use ``ds:<id>``
in an ExperimentSpec's ``dataset`` / ``val_dataset`` to reference the
registered file.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.responses import PlainTextResponse

from ...core import repository
from ...core.db import Database
from ...settings import settings
from ...training.dataset_formats import DatasetFormatError, detect_and_validate
from ..auth import require_api_key
from ..deps import get_db
from ..schemas import Dataset

router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
    dependencies=[Depends(require_api_key)],
)

_CHUNK = 8 * 1024 * 1024  # 8 MiB
_SAFE_NAME = "".join(chr(c) for c in range(0x20, 0x7F) if chr(c) not in '/\\:*?"<>|')


def _sanitize_filename(raw: str) -> str:
    cleaned = "".join(c if c in _SAFE_NAME else "_" for c in raw).strip("._")
    return cleaned or "dataset"


@router.post("", status_code=201)
async def upload_dataset(
    db: Annotated[Database, Depends(get_db)],
    response: Response,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
) -> Dataset:
    if not file.filename:
        raise HTTPException(422, "filename is required")
    safe_filename = _sanitize_filename(file.filename)
    dataset_id = uuid.uuid4().hex

    target_dir = settings.datasets_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_filename

    sha = hashlib.sha256()
    size = 0
    try:
        with target_path.open("wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                sha.update(chunk)
                size += len(chunk)
                if size > settings.max_dataset_upload_bytes:
                    raise HTTPException(
                        413,
                        f"upload exceeds limit of "
                        f"{settings.max_dataset_upload_bytes} bytes",
                    )
        try:
            fmt, line_count = detect_and_validate(target_path)
        except DatasetFormatError as e:
            raise HTTPException(
                422,
                {
                    "error": "invalid_dataset_format",
                    "detail": str(e),
                    "filename": safe_filename,
                },
            ) from None
    except HTTPException:
        # Clean up the partial/invalid file so we don't leak disk.
        try:
            target_path.unlink(missing_ok=True)
            target_dir.rmdir()
        except OSError:
            pass
        raise

    digest = sha.hexdigest()

    async with db.connect() as conn:
        existing = await repository.get_dataset_by_sha(conn, digest)
        if existing is not None:
            # Identical content is already registered — drop the freshly
            # written duplicate and return the existing record (200, not 201)
            # instead of cloning the file on disk.
            try:
                target_path.unlink(missing_ok=True)
                target_dir.rmdir()
            except OSError:
                pass
            response.status_code = 200
            return existing
        await repository.create_dataset(
            conn,
            name=name,
            path=str(target_path),
            fmt=fmt,
            size_bytes=size,
            sha256=digest,
            line_count=line_count,
            description=description,
            dataset_id=dataset_id,
        )
        rec = await repository.get_dataset(conn, dataset_id)
    assert rec is not None
    return rec


@router.get("")
async def list_datasets(
    db: Annotated[Database, Depends(get_db)],
) -> list[Dataset]:
    async with db.connect() as conn:
        return await repository.list_datasets(conn)


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    db: Annotated[Database, Depends(get_db)],
) -> Dataset:
    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
    if rec is None:
        raise HTTPException(404, "dataset not found")
    return rec


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    db: Annotated[Database, Depends(get_db)],
    n: int = Query(10, ge=1, le=1000),
) -> PlainTextResponse:
    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
    if rec is None:
        raise HTTPException(404, "dataset not found")
    path = Path(rec.path)
    if not path.exists():
        raise HTTPException(410, "dataset file missing on disk")
    if rec.format != "parquet":
        # Everything except parquet is UTF-8 text we can line-tail.
        lines: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                lines.append(line.rstrip("\n"))
        return PlainTextResponse("\n".join(lines))
    return PlainTextResponse(
        f"(format '{rec.format}' has no plain-text preview)"
    )


@router.delete("/{dataset_id}", status_code=200)
async def delete_dataset(
    dataset_id: str,
    db: Annotated[Database, Depends(get_db)],
    force: bool = Query(
        False,
        description="Delete even if queued/running experiments reference it.",
    ),
) -> dict[str, bool]:
    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
        if rec is None:
            raise HTTPException(404, "dataset not found")
        if not force:
            blockers = await repository.active_experiments_referencing_path(
                conn, rec.path
            )
            if blockers:
                raise HTTPException(
                    409,
                    {
                        "error": "dataset_in_use",
                        "detail": (
                            "dataset is referenced by active experiments; "
                            "pass force=true to delete anyway"
                        ),
                        "experiment_ids": blockers,
                    },
                )
        ok = await repository.delete_dataset(conn, dataset_id)
    if ok:
        path = Path(rec.path)
        try:
            path.unlink(missing_ok=True)
            if path.parent.is_dir() and not any(path.parent.iterdir()):
                path.parent.rmdir()
        except OSError:
            pass
    return {"deleted": ok}
