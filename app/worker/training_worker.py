import pika
import json
import os
from typing import List
import asyncio
from ..gpu.manager import GPUManager
import aio_pika
from ..job.store import JobStore, JobStatus
from ..utils.logger import setup_logger
import subprocess
from ..training.swift_config import SwiftTrainingConfig

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
            # Extract training config from job_data
            training_config_data = {
                "model_type": job_data["model_type"],
                "model_id_or_path": job_data["model_id_or_path"],
                "dataset": job_data["dataset"],
                "gpu_indices": gpu_indices,
                # Add optional fields if they exist
                "val_dataset": job_data.get("val_dataset"),
                # Merge in the training_config dict
                **job_data.get("training_config", {})
            }
            
            # Create Swift training config
            training_config = SwiftTrainingConfig(**training_config_data)
            
            # Generate command
            command = training_config.get_command()
            self.logger.info(f"Running command: {command}")
            
            # Run the training process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.logger.info(output.strip())
            
            # Get return code
            return_code = process.poll()
            if return_code != 0:
                raise Exception(f"Training failed with return code {return_code}")
            
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