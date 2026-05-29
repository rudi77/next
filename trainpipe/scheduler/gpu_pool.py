"""GPU discovery and lease accounting backed by SQLite.

Detection uses pynvml; on hosts without NVIDIA drivers we degrade gracefully to
an empty pool so the API still boots (no experiment will ever leave 'queued').

Leases are persisted in the gpu_leases table so a process restart can recover
state and orphaned leases are released on boot.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    memory_total_mb: int


def detect_gpus(visible: list[int] | None = None) -> list[GpuInfo]:
    """Discover NVIDIA GPUs via pynvml. Returns ``[]`` if no driver/library."""
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("pynvml not importable; running with empty GPU pool")
        return []

    try:
        pynvml.nvmlInit()
    except Exception as e:
        logger.warning("NVML init failed (%s); running with empty GPU pool", e)
        return []

    try:
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GpuInfo] = []
        for i in range(count):
            if visible is not None and i not in visible:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode() if isinstance(name_raw, bytes) else str(name_raw)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuInfo(index=i, name=name, memory_total_mb=int(mem.total // (1024 * 1024)))
            )
        return gpus
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class GpuPool:
    """SQLite-backed GPU lease tracker.

    All mutations go through an asyncio lock so concurrent dispatch attempts
    don't double-allocate.
    """

    def __init__(self, gpus: list[GpuInfo]) -> None:
        self.gpus = {g.index: g for g in gpus}
        self._lock = asyncio.Lock()

    @property
    def indices(self) -> list[int]:
        return sorted(self.gpus.keys())

    @property
    def total(self) -> int:
        return len(self.gpus)

    async def sync_leases(self, conn: aiosqlite.Connection) -> None:
        """Ensure one row per detected GPU; release any orphaned leases.

        Note: the ``experiment_id`` column is overloaded — both experiments
        and eval_runs share this pool, identified by their primary key. We
        exempt both running tables from the orphan sweep.
        """
        cur = await conn.execute("SELECT gpu_index FROM gpu_leases")
        existing = {row[0] for row in await cur.fetchall()}

        for idx in self.indices:
            if idx not in existing:
                await conn.execute(
                    "INSERT INTO gpu_leases (gpu_index, experiment_id, leased_at) "
                    "VALUES (?, NULL, NULL)",
                    (idx,),
                )

        await conn.execute(
            "UPDATE gpu_leases SET experiment_id = NULL, leased_at = NULL "
            "WHERE experiment_id IS NOT NULL "
            "AND experiment_id NOT IN (SELECT id FROM experiments WHERE status = 'running') "
            "AND experiment_id NOT IN (SELECT id FROM eval_runs WHERE status = 'running')"
        )
        await conn.commit()

    async def try_allocate(
        self,
        conn: aiosqlite.Connection,
        count: int,
        experiment_id: str,
    ) -> list[int] | None:
        """Reserve ``count`` free GPUs for ``experiment_id``. Returns indices or None."""
        async with self._lock:
            cur = await conn.execute(
                "SELECT gpu_index FROM gpu_leases WHERE experiment_id IS NULL "
                "ORDER BY gpu_index LIMIT ?",
                (count,),
            )
            rows = await cur.fetchall()
            if len(rows) < count:
                return None
            indices = [int(r[0]) for r in rows]
            now = datetime.now(timezone.utc).isoformat()
            await conn.executemany(
                "UPDATE gpu_leases SET experiment_id = ?, leased_at = ? WHERE gpu_index = ?",
                [(experiment_id, now, idx) for idx in indices],
            )
            await conn.commit()
            return indices

    async def release(self, conn: aiosqlite.Connection, experiment_id: str) -> None:
        async with self._lock:
            await conn.execute(
                "UPDATE gpu_leases SET experiment_id = NULL, leased_at = NULL "
                "WHERE experiment_id = ?",
                (experiment_id,),
            )
            await conn.commit()

    async def status(self, conn: aiosqlite.Connection) -> list[dict]:
        cur = await conn.execute(
            "SELECT gpu_index, experiment_id, leased_at FROM gpu_leases ORDER BY gpu_index"
        )
        rows = await cur.fetchall()
        out: list[dict] = []
        for r in rows:
            idx = int(r[0])
            gpu = self.gpus.get(idx)
            out.append(
                {
                    "index": idx,
                    "name": gpu.name if gpu else "unknown",
                    "memory_total_mb": gpu.memory_total_mb if gpu else 0,
                    "experiment_id": r[1],
                    "leased_at": r[2],
                }
            )
        return out
