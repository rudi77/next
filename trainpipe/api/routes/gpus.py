from typing import Annotated

from fastapi import APIRouter, Depends

from ...core.db import Database
from ...scheduler.gpu_pool import GpuPool
from ..auth import require_api_key
from ..deps import get_db, get_gpu_pool

router = APIRouter(
    prefix="/gpus",
    tags=["gpus"],
    dependencies=[Depends(require_api_key)],
)


@router.get("")
async def list_gpus(
    db: Annotated[Database, Depends(get_db)],
    gpu_pool: Annotated[GpuPool, Depends(get_gpu_pool)],
) -> dict:
    async with db.connect() as conn:
        leases = await gpu_pool.status(conn)
    free = [g["index"] for g in leases if g["experiment_id"] is None]
    return {
        "total": gpu_pool.total,
        "free": free,
        "leases": leases,
    }
