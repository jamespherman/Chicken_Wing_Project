import logging
import sys
from logging.handlers import RotatingFileHandler

def configure_logging(log_level=logging.INFO, log_file='batch_processing.log'):
    """
    Configures logging for the entire application.
    """
    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Prevent duplicate handlers
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # File Handler (Rotating)
    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set the logger for the application
    app_logger = logging.getLogger('app')
    app_logger.info("Logging configured.")
    return app_logger

def get_logger(name):
    """
    Returns a logger with the specified name.
    """
    return logging.getLogger(name)
