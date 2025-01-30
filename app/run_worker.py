import asyncio
from .worker.training_worker import TrainingWorker

async def main():
    worker = TrainingWorker()
    await worker.start()

if __name__ == "__main__":
    asyncio.run(main()) 