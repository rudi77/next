from typing import Dict
from enum import Enum
from datetime import datetime

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStore:
    def __init__(self):
        self.jobs: Dict[str] = {}
        
    async def add_job(self, job_id: str, job_data: dict):
        self.jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.QUEUED,
            "data": job_data,
            "created_at": datetime.utcnow(),
            "started_at": None,
            "completed_at": None,
            "gpu_indices": None
        }
    
    async def update_job_status(self, job_id: str, status: JobStatus, gpu_indices=None):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = status
            if status == JobStatus.RUNNING:
                self.jobs[job_id]["started_at"] = datetime.utcnow()
                self.jobs[job_id]["gpu_indices"] = gpu_indices
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                self.jobs[job_id]["completed_at"] = datetime.utcnow()
    
    async def get_job(self, job_id: str):
        return self.jobs.get(job_id) 