"""
Logging configuration for the ABSA Agent system.

This module provides a simple function to configure logging for the entire application.
You can call this at the start of your main scripts to set up logging with appropriate levels.
"""

import logging
import sys
import os

def setup_logging(level=logging.INFO, log_file=None):
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR)
        log_file: Optional path to a log file. If None, logs only to console.
    
    Example usage in your main script:
        from logging_config import setup_logging
        setup_logging(level=logging.INFO, log_file="app.log")
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Optionally reduce verbosity of third-party libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('vllm').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    return root_logger


if __name__ == "__main__":
    # Example usage
    setup_logging(level=logging.DEBUG, log_file="absa_agent.log")
    
    logger = logging.getLogger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
