"""Logging setup for the application."""

import logging
import sys

# Consider using structlog for more structured logging later

DEFAULT_LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def setup_logging(level: int = DEFAULT_LOG_LEVEL, log_format: str = LOG_FORMAT) -> None:
    """Configures basic logging for the application.

    Args:
        level: The minimum logging level (e.g., logging.DEBUG, logging.INFO).
        log_format: The format string for log messages.
    """
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance for a specific module.

    Args:
        name: The name for the logger (usually __name__ of the calling module).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)

# Example of setting up logging at application start (e.g., in main script or __init__):
# setup_logging()
# logger = get_logger(__name__)
# logger.info("Logging initialized.") 