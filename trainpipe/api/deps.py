from fastapi import Request

from ..acquisition.manager import AcquisitionManager
from ..autoresearch.manager import StudyManager
from ..core.db import Database
from ..evals.dispatcher import EvalDispatcher
from ..inference.service import InferenceService
from ..pipelines.manager import PipelineManager
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


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def get_pipeline_manager(request: Request) -> PipelineManager:
    return request.app.state.pipeline_manager


def get_acquisition_manager(request: Request) -> AcquisitionManager:
    return request.app.state.acquisition_manager
