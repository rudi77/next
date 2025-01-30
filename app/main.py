from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
import asyncio
from .rabbitmq.publisher import RabbitMQPublisher
from contextlib import asynccontextmanager
from .gpu.manager import GPUManager
from .job.store import JobStore, JobStatus
import uuid
from .utils.logger import setup_logger
import time

# Setup logger
logger = setup_logger("backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application")
    global rabbitmq_publisher
    rabbitmq_publisher = RabbitMQPublisher()
    await rabbitmq_publisher.connect()
    app.state.gpu_manager = GPUManager()
    app.state.job_store = JobStore()
    yield
    # Shutdown
    logger.info("Shutting down FastAPI application")
    if rabbitmq_publisher:
        await rabbitmq_publisher.close()

app = FastAPI(title="AI Training Pipeline", lifespan=lifespan)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"Status: {response.status_code} "
        f"Duration: {duration:.2f}s"
    )
    return response

class TrainingJob(BaseModel):
    model_name: str
    dataset_path: str
    hyperparameters: dict
    gpu_count: Optional[int] = 1

@app.post("/submit_job")
async def submit_job(job: TrainingJob):
    try:
        job_id = str(uuid.uuid4())
        logger.info(f"Submitting job {job_id}")
        # Store job in JobStore
        await app.state.job_store.add_job(job_id, job.dict())
        
        # Check if required GPUs are available
        gpu_manager = app.state.gpu_manager
        free_gpus = await gpu_manager.get_free_gpus()
        if len(free_gpus) < job.gpu_count:
            return {
                "status": "queued",
                "message": f"Not enough GPUs available. Job queued. Available GPUs: {len(free_gpus)}",
                "job_id": job_id
            }

        # Add job_id to the message
        job_message = job.dict()
        job_message["job_id"] = job_id
        
        # Publish job to RabbitMQ
        await rabbitmq_publisher.publish_job(job_message)
        logger.info(f"Job {job_id} submitted successfully")
        return {
            "status": "success", 
            "message": "Job submitted successfully",
            "job_id": job_id
        }
    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/gpu_status")
async def get_gpu_status():
    """Get current GPU status"""
    try:
        gpu_manager = app.state.gpu_manager
        gpu_info = await gpu_manager.get_gpu_info()
        free_gpus = await gpu_manager.get_free_gpus()
        return {
            "gpu_info": gpu_info,
            "free_gpus": free_gpus,
            "total_gpus": len(gpu_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    job = await app.state.job_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/jobs")
async def list_jobs():
    """Get all jobs and their status"""
    return app.state.job_store.jobs 