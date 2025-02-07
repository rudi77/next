import asyncio
import signal
import sys
import platform
from .worker.training_worker import TrainingWorker
from .utils.logger import setup_logger

logger = setup_logger("worker_runner")
worker = None

async def shutdown():
    """Cleanup tasks tied to the service's shutdown."""
    global worker
    logger.info("Shutting down worker...")
    
    if worker:
        await worker.stop()
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)

def handle_exception(loop, context):
    """Handle exceptions that escape the async tasks."""
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception: {msg}")

async def main():
    global worker
    
    # Setup exception handling
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    
    # Setup platform-specific signal handling
    if platform.system() != 'Windows':
        # Unix-like systems can use loop.add_signal_handler
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(shutdown())
            )
    else:
        # Windows needs a different approach
        import win32api
        def handler(sig):
            if sig == signal.SIGINT:
                asyncio.create_task(shutdown())
                sys.exit(0)
        win32api.SetConsoleCtrlHandler(handler, True)
    
    try:
        worker = TrainingWorker()
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await shutdown()
    except Exception as e:
        logger.error(f"Worker failed: {str(e)}", exc_info=True)
        if worker:
            await worker.stop()

if __name__ == "__main__":
    try:
        if platform.system() == 'Windows':
            # Add pywin32 to requirements.txt
            import win32api
            asyncio.run(main())
        else:
            asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...") 