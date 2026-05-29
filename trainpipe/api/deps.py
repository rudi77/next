from fastapi import Request

from ..core.db import Database
from ..scheduler.gpu_pool import GpuPool
from ..scheduler.loop import Scheduler


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


def get_gpu_pool(request: Request) -> GpuPool:
    return request.app.state.gpu_pool
