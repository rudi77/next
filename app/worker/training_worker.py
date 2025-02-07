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
import traceback

class TrainingWorker:
    def __init__(self):
        self.logger = setup_logger("worker")
        self.connection = None
        self.channel = None
        self.queue_name = "training_jobs"
        self.gpu_manager = GPUManager()
        self.job_store = JobStore()
        self.running = True

    async def connect(self):
        """Connect to RabbitMQ with retry logic"""
        while self.running:
            try:
                self.logger.info("Connecting to RabbitMQ")
                self.connection = await aio_pika.connect_robust("amqp://guest:guest@localhost/")
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=1)
                
                self.queue = await self.channel.declare_queue(
                    self.queue_name,
                    durable=True
                )
                self.logger.info("Successfully connected to RabbitMQ")
                return
            except Exception as e:
                self.logger.error(f"Failed to connect to RabbitMQ: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

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
            
            # Stream output and error in parallel
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()
                
                if output:
                    self.logger.info(output.strip())
                if error:
                    self.logger.error(error.strip())
                    
                if output == '' and error == '' and process.poll() is not None:
                    break
            
            # Get return code
            return_code = process.poll()
            if return_code != 0:
                raise Exception(f"Training failed with return code {return_code}")
            
            return {"status": "completed", "gpu_indices": gpu_indices}
        except Exception as e:
            self.logger.error(f"Job {job_id}: Training failed: {str(e)}\n{traceback.format_exc()}")
            raise

    async def handle_message(self, message: aio_pika.IncomingMessage):
        """Handle a single message with proper error handling"""
        try:
            async with message.process():
                job_data = json.loads(message.body)
                job_id = job_data.pop("job_id", "unknown")
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
                        
                        await self.process_training_job(gpu_indices, job_data)
                        
                        # Update status to completed
                        await self.job_store.update_job_status(
                            job_id, 
                            JobStatus.COMPLETED
                        )
                    except Exception as e:
                        # Update status to failed
                        self.logger.error(f"Job {job_id} failed: {str(e)}\n{traceback.format_exc()}")
                        await self.job_store.update_job_status(
                            job_id, 
                            JobStatus.FAILED
                        )
                        # Don't requeue failed jobs
                else:
                    # Keep status as queued and requeue message
                    self.logger.info(f"No GPUs available for job {job_id}, requeueing")
                    await message.nack(requeue=True)
                    
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}\n{traceback.format_exc()}")
            # In case of processing error, nack without requeue to prevent infinite loop
            try:
                await message.nack(requeue=False)
            except:
                pass

    async def start(self):
        """Main worker loop with error recovery"""
        while self.running:
            try:
                if not self.connection or self.connection.is_closed:
                    await self.connect()
                
                async with self.queue.iterator() as queue_iter:
                    self.logger.info("Worker started and waiting for messages")
                    async for message in queue_iter:
                        await self.handle_message(message)
                        
            except Exception as e:
                self.logger.error(f"Worker error: {str(e)}\n{traceback.format_exc()}")
                # Wait before reconnecting
                await asyncio.sleep(5)

    async def stop(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down worker...")
        self.running = False
        if self.connection:
            await self.connection.close() 