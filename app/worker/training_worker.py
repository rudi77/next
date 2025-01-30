import pika
import json
import os
from typing import List
import asyncio
from ..gpu.manager import GPUManager
import aio_pika
from ..job.store import JobStore, JobStatus
from ..utils.logger import setup_logger

class TrainingWorker:
    def __init__(self):
        self.logger = setup_logger("worker")
        self.connection = None
        self.channel = None
        self.queue_name = "training_jobs"
        self.gpu_manager = GPUManager()
        self.job_store = JobStore()

    async def connect(self):
        self.logger.info("Connecting to RabbitMQ")
        try:
            self.connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)
            
            self.queue = await self.channel.declare_queue(
                self.queue_name,
                durable=True
            )
            self.logger.info("Successfully connected to RabbitMQ")
        except Exception as e:
            self.logger.error(f"Failed to connect to RabbitMQ: {str(e)}", exc_info=True)
            raise

    async def process_training_job(self, gpu_indices: List[int], job_data: dict):
        job_id = job_data.get('job_id', 'unknown')
        self.logger.info(f"Processing job {job_id} on GPUs: {gpu_indices}")
        
        try:
            gpu_str = ','.join(map(str, gpu_indices))
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
            
            self.logger.info(f"Job {job_id}: Starting training process")
            await asyncio.sleep(10)  # Simulate training
            self.logger.info(f"Job {job_id}: Training completed")
            
            return {"status": "completed", "gpu_indices": gpu_indices}
        except Exception as e:
            self.logger.error(f"Job {job_id}: Training failed: {str(e)}", exc_info=True)
            raise

    async def callback(self, message):
        job_data = json.loads(message.body)
        job_id = job_data.pop("job_id")
        gpu_count = job_data.get('gpu_count', 1)
        
        # Try to allocate GPUs
        gpu_indices = await self.gpu_manager.allocate_gpus(gpu_count)
        
        if gpu_indices:
            try:
                # Update status to running
                await self.job_store.update_job_status(
                    job_id, 
                    JobStatus.RUNNING, 
                    gpu_indices
                )
                
                result = await self.process_training_job(gpu_indices, job_data)
                
                # Update status to completed
                await self.job_store.update_job_status(
                    job_id, 
                    JobStatus.COMPLETED
                )
                
                await message.ack()
            except Exception as e:
                # Update status to failed
                await self.job_store.update_job_status(
                    job_id, 
                    JobStatus.FAILED
                )
                await message.nack()
                print(f"Training failed: {str(e)}")
        else:
            # Keep status as queued
            await message.nack()
            print("No GPUs available, requeueing job")

    async def start(self):
        await self.connect()
        
        async with self.queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    await self.callback(message) 