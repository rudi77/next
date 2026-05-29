"""Subprocess lifecycle for a single training run.

The process is started in its own process group on POSIX so SIGTERM (and a
SIGKILL fallback) take down any torchrun/python children. Stdout and stderr
are merged and tailed into a per-experiment log file.
"""

import asyncio
import logging
import os
import signal
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RunningProcess:
    def __init__(
        self,
        experiment_id: str,
        process: asyncio.subprocess.Process,
        log_path: Path,
        tee_task: asyncio.Task,
    ) -> None:
        self.experiment_id = experiment_id
        self.process = process
        self.log_path = log_path
        self.tee_task = tee_task
        self.cancelled = False

    @property
    def pid(self) -> int:
        return self.process.pid

    async def wait(self) -> int:
        rc = await self.process.wait()
        try:
            await self.tee_task
        except Exception:
            logger.exception("tee task failed for %s", self.experiment_id)
        return rc

    async def cancel(self, term_grace_sec: float = 10.0) -> None:
        self.cancelled = True
        if self.process.returncode is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self.process.wait(), timeout=term_grace_sec)
        except asyncio.TimeoutError:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            except ProcessLookupError:
                return
            await self.process.wait()


async def spawn_training_subprocess(
    experiment_id: str,
    argv: list[str],
    env_overrides: dict[str, str],
    log_path: Path,
    cwd: Path | None = None,
) -> RunningProcess:
    """Start ``argv`` and return a RunningProcess with a tee task draining stdout."""
    env = os.environ.copy()
    env.update(env_overrides)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {}
    if os.name == "posix":
        kwargs["preexec_fn"] = os.setsid

    process = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd=str(cwd) if cwd else None,
        **kwargs,
    )

    async def tee() -> None:
        # Per-line flush would burn the event loop on tqdm-style progress
        # output. Buffer in user space and flush at most once a second so the
        # SSE log tail still sees fresh data without blocking the scheduler.
        assert process.stdout is not None
        last_flush = time.monotonic()
        with log_path.open("ab") as f:
            while True:
                chunk = await process.stdout.readline()
                if not chunk:
                    break
                f.write(chunk)
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    f.flush()
                    last_flush = now

    tee_task = asyncio.create_task(
        tee(), name=f"trainpipe-tee-{experiment_id}"
    )
    return RunningProcess(
        experiment_id=experiment_id,
        process=process,
        log_path=log_path,
        tee_task=tee_task,
    )
