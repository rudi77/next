from fastapi import Request

from ..autoresearch.manager import StudyManager
from ..core.db import Database
from ..evals.dispatcher import EvalDispatcher
from ..scheduler.gpu_pool import GpuPool
from ..scheduler.loop import Scheduler


def get_db(request: Request) -> Database:
    return request.app.state.db


def get_scheduler(request: Request) -> Scheduler:
    return request.app.state.scheduler


def get_gpu_pool(request: Request) -> GpuPool:
    return request.app.state.gpu_pool


def get_study_manager(request: Request) -> StudyManager:
    return request.app.state.study_manager


def get_eval_dispatcher(request: Request) -> EvalDispatcher:
    return request.app.state.eval_dispatcher
