import subprocess
from typing import List, Optional
import json
import logging

# Set up logging
logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self):
        self.max_gpus = 4
        logger.info(f"Initialized GPUManager with max_gpus={self.max_gpus}")

    async def get_gpu_info(self) -> List[dict]:
        try:
            logger.debug("Querying nvidia-smi for GPU information")
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', 
                 '--format=csv,nounits,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            gpus = []
            for line in result.stdout.strip().split('\n'):
                index, memory_used, memory_total, utilization = line.split(', ')
                gpus.append({
                    'index': int(index),
                    'memory_used': float(memory_used),
                    'memory_total': float(memory_total),
                    'utilization': float(utilization)
                })
            logger.debug(f"Found {len(gpus)} GPUs: {gpus}")
            return gpus
        except subprocess.CalledProcessError:
            logger.error("Failed to get GPU information from nvidia-smi")
            return []

    async def get_free_gpus(self, required_memory: float = 8000) -> List[int]:
        """Return indices of GPUs with enough free memory (in MB)"""
        logger.debug(f"Searching for GPUs with at least {required_memory}MB free memory")
        gpu_info = await self.get_gpu_info()
        free_gpus = []
        
        for gpu in gpu_info:
            if (gpu['memory_total'] - gpu['memory_used'] >= required_memory and 
                gpu['utilization'] < 10):
                free_gpus.append(gpu['index'])
                
        logger.info(f"Found {len(free_gpus)} free GPUs: {free_gpus}")
        return free_gpus

    async def allocate_gpus(self, num_gpus: int = 1) -> Optional[List[int]]:
        """Attempt to allocate specified number of GPUs"""
        logger.info(f"Attempting to allocate {num_gpus} GPUs")
        free_gpus = await self.get_free_gpus()
        if len(free_gpus) >= num_gpus:
            allocated = free_gpus[:num_gpus]
            logger.info(f"Successfully allocated GPUs: {allocated}")
            return allocated
        logger.warning(f"Failed to allocate {num_gpus} GPUs. Only {len(free_gpus)} available")
        return None 