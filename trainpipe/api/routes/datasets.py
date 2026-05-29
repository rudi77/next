"""POST /datasets uploads a file, validates the format, stores it on disk,
and registers an entry in the SQLite ``datasets`` table. Use ``ds:<id>``
in an ExperimentSpec's ``dataset`` / ``val_dataset`` to reference the
registered file.
"""

import hashlib
import uuid
import zipfile
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
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from ...core import repository
from ...core.db import Database
from ...integrations import labelstudio as ls
from ...settings import settings
from ...training.dataset_formats import (
    DatasetFormatError,
    detect_and_validate_info,
)
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
            info = detect_and_validate_info(target_path)
        except DatasetFormatError as e:
            raise HTTPException(
                422,
                {
                    "error": "invalid_dataset_format",
                    "detail": str(e),
                    "filename": safe_filename,
                },
            ) from None
        fmt = info.format
        line_count = info.line_count
        media_kinds = info.media_kinds
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
            media_kinds=media_kinds,
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


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


def _safe_join(base: Path, rel: str) -> Path | None:
    """Resolve ``rel`` under ``base``, refusing traversal or symlink
    escape. Returns ``None`` on any attempt to read outside ``base``.

    Walks each path component checking ``is_symlink()`` BEFORE
    ``resolve()``; this catches a symlink inside ``base`` whose target
    is inside ``base`` too — those are still suspicious and we refuse
    them so an attacker can't use bundle uploads as a generic file pipe.
    """
    raw = Path(rel)
    if raw.is_absolute() or ".." in raw.parts:
        return None
    base_resolved = base.resolve()
    current = base_resolved
    for part in raw.parts:
        current = current / part
        if current.is_symlink():
            return None
    candidate = current.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        return None
    return candidate


@router.post("/bundle", status_code=201)
async def upload_bundle(
    db: Annotated[Database, Depends(get_db)],
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
) -> Dataset:
    """Upload a zip bundle: one .jsonl manifest + an ``images/`` directory.

    Used for image-JSONL training. Image paths in the JSONL are expected
    to be *relative to the bundle root* (e.g. ``images/doc-001.png``).
    Validation: at least one .jsonl is present, sample rows reference
    files that resolve inside the bundle, no zip-slip.
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(422, "expected a .zip file")
    safe_filename = _sanitize_filename(file.filename)
    dataset_id = uuid.uuid4().hex
    target_dir = settings.datasets_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / safe_filename

    sha = hashlib.sha256()
    size = 0
    try:
        with zip_path.open("wb") as out:
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
                        f"bundle exceeds limit of "
                        f"{settings.max_dataset_upload_bytes} bytes",
                    )

        try:
            extracted = target_dir / "bundle"
            extracted.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                # Defense against zip-slip and symlink escape:
                #   - reject absolute member names / drive letters
                #   - reject ``..`` segments
                #   - reject any entry whose Unix mode bits mark it as a
                #     symlink (`S_IFLNK == 0o120000`). ZipFile.extractall
                #     happily materializes those on POSIX, and then
                #     ``Path.resolve()`` in ``_safe_join`` would follow
                #     the link out of ``image_root``.
                for info in zf.infolist():
                    member = info.filename
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise HTTPException(
                            422,
                            {
                                "error": "unsafe_zip_path",
                                "name": member,
                            },
                        )
                    # external_attr high 16 bits = Unix mode on archives
                    # created by Info-ZIP / Python zipfile on POSIX.
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == 0o120000:
                        raise HTTPException(
                            422,
                            {
                                "error": "unsafe_zip_symlink",
                                "name": member,
                            },
                        )
                zf.extractall(extracted)
        except zipfile.BadZipFile:
            raise HTTPException(
                422, {"error": "invalid_zip", "filename": safe_filename}
            ) from None

        jsonls = sorted(extracted.rglob("*.jsonl"))
        if not jsonls:
            raise HTTPException(
                422, {"error": "no_jsonl_in_bundle", "hint": "zip must contain a *.jsonl manifest"},
            )
        if len(jsonls) > 1:
            raise HTTPException(
                422,
                {
                    "error": "multiple_jsonl_in_bundle",
                    "candidates": [str(p.relative_to(extracted)) for p in jsonls],
                },
            )
        manifest = jsonls[0]
        try:
            info = detect_and_validate_info(manifest)
        except DatasetFormatError as e:
            raise HTTPException(
                422,
                {
                    "error": "invalid_dataset_format",
                    "detail": str(e),
                    "filename": str(manifest.relative_to(extracted)),
                },
            ) from None
        if not info.media_kinds:
            raise HTTPException(
                422,
                {
                    "error": "bundle_without_media",
                    "hint": "bundle uploads are for image/video JSONL; "
                    "plain text JSONL should use POST /datasets",
                },
            )
    except HTTPException:
        # Clean up partial dir on any error.
        try:
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
        except OSError:
            pass
        raise

    digest = sha.hexdigest()
    async with db.connect() as conn:
        existing = await repository.get_dataset_by_sha(conn, digest)
        if existing is not None:
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)
            return existing
        await repository.create_dataset(
            conn,
            name=name,
            path=str(manifest),
            fmt=info.format,
            size_bytes=size,
            sha256=digest,
            line_count=info.line_count,
            description=description,
            dataset_id=dataset_id,
            media_kinds=info.media_kinds,
            image_root=str(extracted),
        )
        rec = await repository.get_dataset(conn, dataset_id)
    assert rec is not None
    return rec


@router.get("/{dataset_id}/media")
async def get_media(
    dataset_id: str,
    path: str,
    db: Annotated[Database, Depends(get_db)],
) -> FileResponse:
    """Serve a file inside the dataset's bundle (for image thumbnails in UI).

    ``path`` is a relative path under the bundle's ``image_root``. Path
    traversal (``..``, absolute paths) is rejected.
    """
    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
    if rec is None:
        raise HTTPException(404, "dataset not found")
    if not rec.image_root:
        raise HTTPException(404, "dataset has no media bundle")
    safe = _safe_join(Path(rec.image_root), path)
    if safe is None or not safe.is_file():
        raise HTTPException(404, "media file not found")
    if safe.suffix.lower() not in _IMAGE_EXTS:
        raise HTTPException(415, "unsupported media type")
    return FileResponse(str(safe))


class SplitRequest(BaseModel):
    """Split a JSONL dataset into train/val by ratio."""

    model_config = ConfigDict(extra="forbid")

    ratio: str = Field(
        "90:10",
        description="Two integer ratios separated by ':' summing to 100",
    )
    seed: int = 0
    train_name: str | None = None
    val_name: str | None = None


class MixSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    weight: float = Field(..., gt=0.0, le=1000.0)


class MixRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    sources: list[MixSource] = Field(..., min_length=2, max_length=16)
    target_count: int | None = Field(
        None, ge=1, description="Total samples in the mix; default = sum of source lines"
    )
    seed: int = 0


@router.post("/{dataset_id}/split", status_code=201)
async def split_dataset(
    dataset_id: str,
    request: SplitRequest,
    db: Annotated[Database, Depends(get_db)],
) -> dict[str, Dataset]:
    """Split a JSONL dataset into two new datasets by line ratio.

    The split is deterministic given ``seed`` (full shuffle then slice),
    so two callers can reproduce the same train/val partition.
    """
    parts = request.ratio.split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise HTTPException(
            422, {"error": "bad_ratio", "value": request.ratio}
        )
    a, b = int(parts[0]), int(parts[1])
    if a + b != 100 or a == 0 or b == 0:
        raise HTTPException(
            422, {"error": "bad_ratio", "detail": "ratios must sum to 100"}
        )

    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
    if rec is None:
        raise HTTPException(404, "dataset not found")
    if rec.format != "jsonl":
        raise HTTPException(
            422,
            {
                "error": "unsupported_format",
                "format": rec.format,
                "hint": "split currently only supports jsonl",
            },
        )

    import random as _random

    with open(rec.path, encoding="utf-8") as f:
        lines = [ln for ln in (l.strip() for l in f) if ln]
    if len(lines) < 2:
        raise HTTPException(
            422, {"error": "too_few_records_to_split", "count": len(lines)}
        )
    rng = _random.Random(request.seed)
    rng.shuffle(lines)
    cut = (len(lines) * a) // 100
    train_lines, val_lines = lines[:cut], lines[cut:]

    return {
        "train": await _persist_derived(
            db,
            parent=rec,
            name=request.train_name or f"{rec.name}-train",
            lines=train_lines,
            provenance=(
                f"split from ds:{rec.id} ratio={request.ratio} side=train "
                f"seed={request.seed}"
            ),
            lineage_role="split-of",
        ),
        "val": await _persist_derived(
            db,
            parent=rec,
            name=request.val_name or f"{rec.name}-val",
            lines=val_lines,
            provenance=(
                f"split from ds:{rec.id} ratio={request.ratio} side=val "
                f"seed={request.seed}"
            ),
            lineage_role="split-of",
        ),
    }


@router.post("/mixes", status_code=201)
async def create_mix(
    request: MixRequest,
    db: Annotated[Database, Depends(get_db)],
) -> Dataset:
    """Combine N source datasets weighted; output a new JSONL dataset."""
    import random as _random

    async with db.connect() as conn:
        recs: list[Dataset] = []
        for src in request.sources:
            rec = await repository.get_dataset(conn, src.dataset_id)
            if rec is None:
                raise HTTPException(
                    422,
                    {
                        "error": "unknown_dataset",
                        "dataset_id": src.dataset_id,
                    },
                )
            if rec.format != "jsonl":
                raise HTTPException(
                    422,
                    {
                        "error": "unsupported_format",
                        "dataset_id": src.dataset_id,
                        "format": rec.format,
                    },
                )
            recs.append(rec)

    rng = _random.Random(request.seed)
    weights = [s.weight for s in request.sources]
    weight_sum = sum(weights)

    # Buffer every source in memory — for the trainpipe use case
    # (datasets in the 1k–100k range) this is fine; a streaming
    # implementation can come later if someone tries to mix 10M-row
    # files.
    pools: list[list[str]] = []
    for rec in recs:
        with open(rec.path, encoding="utf-8") as f:
            pools.append([ln for ln in (l.strip() for l in f) if ln])

    if request.target_count is None:
        target = sum(len(p) for p in pools)
    else:
        target = request.target_count

    indices = list(range(len(request.sources)))
    normalized = [w / weight_sum for w in weights]

    out_lines: list[str] = []
    for _ in range(target):
        i = rng.choices(indices, weights=normalized, k=1)[0]
        pool = pools[i]
        if not pool:
            continue
        out_lines.append(rng.choice(pool))

    if not out_lines:
        raise HTTPException(
            422,
            {"error": "no_records_in_sources", "detail": "all source files were empty"},
        )

    all_parent_ids = [r.id for r in recs]
    parts_desc = ", ".join(
        f"ds:{s.dataset_id}*{s.weight}" for s in request.sources
    )
    provenance = (
        f"mix of [{parts_desc}] target_count={target} seed={request.seed}"
    )
    parent_fake = recs[0]  # only needed for image_root/version handling
    return await _persist_derived(
        db,
        parent=parent_fake,
        name=request.name,
        lines=out_lines,
        provenance=provenance,
        # All N source ids — the legacy derived_from column gets the
        # first parent for back-compat, and dataset_lineage gets all
        # of them so GDPR queries return correct results.
        parent_ids=all_parent_ids,
        lineage_role="mix-of",
    )


async def _persist_derived(
    db: Database,
    *,
    parent: Dataset,
    name: str,
    lines: list[str],
    provenance: str,
    parent_id: str | None = None,
    parent_ids: list[str] | None = None,
    lineage_role: str = "derived-from",
) -> Dataset:
    """Write ``lines`` as a new JSONL dataset, register, return.

    ``parent_ids``: when set, all of these are recorded in
    ``dataset_lineage`` for the new row. The 1:1 ``datasets.derived_from``
    column gets the first one (back-compat). Falls back to ``[parent_id
    or parent.id]`` when not specified — that covers split / redact /
    LS-import which only have one parent anyway.
    """
    new_id = uuid.uuid4().hex
    target_dir = settings.datasets_dir / new_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{name}.jsonl"
    with target_path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln)
            f.write("\n")

    sha = hashlib.sha256()
    with target_path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            sha.update(chunk)
    size = target_path.stat().st_size
    info = detect_and_validate_info(target_path)
    digest = sha.hexdigest()

    if parent_ids is None:
        parent_ids = [parent_id or parent.id]
    derived_from_legacy = parent_ids[0] if parent_ids else parent.id

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
            name=name,
            path=str(target_path),
            fmt=info.format,
            size_bytes=size,
            sha256=digest,
            line_count=info.line_count,
            description=provenance,
            dataset_id=new_id,
            media_kinds=info.media_kinds,
            version=parent.version + 1,
            derived_from=derived_from_legacy,
        )
        await repository.record_dataset_lineage(
            conn, new_id, parent_ids, role=lineage_role
        )
        rec = await repository.get_dataset(conn, new_id)
    assert rec is not None
    return rec


class RedactRequest(BaseModel):
    """Run PII redaction over a JSONL dataset and register the result.

    The redacted dataset is a *new* dataset with provenance pointing
    back at the source (the original is left alone — important for the
    audit trail).
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[str] = Field(
        default_factory=lambda: ["email", "phone", "iban", "credit_card"],
        description="Subset of email/phone/iban/credit_card/de_tax_id",
    )
    name: str | None = Field(
        None, description="Name for the redacted dataset; defaults to '<src> (redacted)'"
    )


@router.post("/{dataset_id}/redact", status_code=201)
async def redact_dataset(
    dataset_id: str,
    request: RedactRequest,
    db: Annotated[Database, Depends(get_db)],
) -> Dataset:
    from ...redaction.redactor import redact_jsonl

    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
    if rec is None:
        raise HTTPException(404, "dataset not found")
    if rec.format != "jsonl":
        raise HTTPException(
            422,
            {
                "error": "unsupported_format",
                "format": rec.format,
                "hint": "redact currently only supports jsonl",
            },
        )

    new_id = uuid.uuid4().hex
    target_dir = settings.datasets_dir / new_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{rec.name}-redacted.jsonl"

    rows, hit_counts = redact_jsonl(
        rec.path, str(target_path), entities=request.entities
    )

    if rows == 0:
        try:
            target_path.unlink(missing_ok=True)
            target_dir.rmdir()
        except OSError:
            pass
        raise HTTPException(
            422,
            {
                "error": "empty_source",
                "detail": "source jsonl had no records",
            },
        )

    sha = hashlib.sha256()
    with target_path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            sha.update(chunk)

    size = target_path.stat().st_size
    info = detect_and_validate_info(target_path)
    digest = sha.hexdigest()
    hit_summary = ", ".join(f"{k}={v}" for k, v in hit_counts.items() if v > 0)
    provenance = (
        f"redacted from ds:{rec.id} (entities={'+'.join(request.entities)}, "
        f"hits: {hit_summary or 'none'})"
    )
    name = request.name or f"{rec.name} (redacted)"

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
            name=name,
            path=str(target_path),
            fmt=info.format,
            size_bytes=size,
            sha256=digest,
            line_count=info.line_count,
            description=provenance,
            dataset_id=new_id,
            media_kinds=info.media_kinds,
            version=rec.version + 1,
            derived_from=rec.id,
        )
        await repository.record_dataset_lineage(
            conn, new_id, [rec.id], role="redacted-from"
        )
        out = await repository.get_dataset(conn, new_id)
    assert out is not None
    return out


@router.get("/{dataset_id}/models")
async def models_using_dataset(
    dataset_id: str,
    db: Annotated[Database, Depends(get_db)],
    recursive: bool = False,
) -> dict[str, list[str]]:
    """List the model ids that have ``dataset_id`` in their lineage —
    the GDPR ‘which models trained on this data?’ query.

    Pass ``recursive=true`` to also include models trained on datasets
    that *derive* from ``dataset_id`` (mixes, splits, redacted copies).
    Use this for the proper GDPR "forget" query — direct usage misses
    indirect chains.
    """
    async with db.connect() as conn:
        rec = await repository.get_dataset(conn, dataset_id)
        if rec is None:
            raise HTTPException(404, "dataset not found")
        if recursive:
            ids = await repository.models_using_dataset_recursive(
                conn, dataset_id
            )
        else:
            ids = await repository.models_using_dataset(conn, dataset_id)
    return {"model_ids": ids}


class LabelStudioImportRequest(BaseModel):
    """Pull completed annotations from a Label Studio project and
    register the result as a trainpipe dataset.

    Provenance is recorded in the dataset's description so a later audit
    can trace ``ds:<id>`` back to its LS project + import timestamp.
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(..., min_length=1)
    token: str = Field(..., min_length=1)
    project_id: int = Field(..., ge=1)
    name: str = Field(..., min_length=1)
    # Optional explicit shape; auto-detected from the first ~10 tasks.
    import_kind: str | None = Field(
        None,
        description="Force a shape: conversation | text_ner | doc_layout. "
        "Omit to auto-detect.",
    )
    # Pass the ``created_at`` of the last LS-imported dataset for the
    # same project to fetch only new annotations.
    since_iso: str | None = None
    max_tasks: int | None = Field(None, ge=1)


@router.post("/from-labelstudio", status_code=201)
async def import_labelstudio(
    request: LabelStudioImportRequest,
    db: Annotated[Database, Depends(get_db)],
) -> Dataset:
    try:
        kind, records = await _run_ls_import(request)
    except ls.LabelStudioError as e:
        raise HTTPException(
            502, {"error": "labelstudio_error", "detail": str(e)}
        ) from None

    if not records:
        raise HTTPException(
            422,
            {
                "error": "no_records",
                "detail": "no completed annotations matched the import "
                "(empty project, all skipped, or filter excluded all)",
                "kind": kind,
            },
        )

    dataset_id = uuid.uuid4().hex
    target_dir = settings.datasets_dir / dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{request.name}.jsonl"

    line_count = ls.write_jsonl(records, str(target_path))
    size = target_path.stat().st_size

    sha = hashlib.sha256()
    with target_path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            sha.update(chunk)
    digest = sha.hexdigest()

    try:
        info = detect_and_validate_info(target_path)
    except DatasetFormatError as e:
        try:
            target_path.unlink(missing_ok=True)
            target_dir.rmdir()
        except OSError:
            pass
        raise HTTPException(
            422, {"error": "mapper_output_invalid", "detail": str(e)}
        ) from None

    # Strip any user:pass@ baked into base_url so persisted provenance
    # never carries credentials. Token never goes into the description.
    safe_url = ls.strip_url_credentials(request.base_url)
    provenance = (
        f"imported from Label Studio project {request.project_id} "
        f"at {safe_url} (kind={kind})"
    )

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
            line_count=line_count,
            description=provenance,
            dataset_id=dataset_id,
            media_kinds=info.media_kinds,
        )
        rec = await repository.get_dataset(conn, dataset_id)
    assert rec is not None
    return rec


async def _run_ls_import(
    request: LabelStudioImportRequest,
) -> tuple[str, list[dict]]:
    """Wrap the sync httpx-based importer so it doesn't block the loop."""
    import asyncio

    def _do() -> tuple[str, list[dict]]:
        return ls.import_project(
            base_url=request.base_url,
            token=request.token,
            project_id=request.project_id,
            import_kind=request.import_kind,
            since_iso=request.since_iso,
            max_tasks=request.max_tasks,
        )

    return await asyncio.to_thread(_do)


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
            if rec.image_root:
                # Bundle uploads own the whole datasets/<id>/ directory tree
                # (extracted images + manifest). Remove it as a unit, but
                # only after verifying it really lives under the datasets
                # dir — defense against a corrupted ``image_root`` pointing
                # somewhere arbitrary on disk.
                import shutil
                bundle_root = Path(rec.image_root).parent
                datasets_root = settings.datasets_dir.resolve()
                try:
                    bundle_root.resolve().relative_to(datasets_root)
                except ValueError:
                    # image_root is outside datasets_dir — refuse to rmtree.
                    pass
                else:
                    shutil.rmtree(bundle_root, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
                if path.parent.is_dir() and not any(path.parent.iterdir()):
                    path.parent.rmdir()
        except OSError:
            pass
    return {"deleted": ok}
