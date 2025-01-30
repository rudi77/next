import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str, log_level=logging.INFO):
    """Setup logger with consistent formatting across applications"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create file handler
    file_handler = logging.FileHandler(
        os.path.join(logs_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    )
    file_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Clear any existing handlers
    logger.handlers = []

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger 