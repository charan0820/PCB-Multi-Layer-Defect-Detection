"""
Logging utilities shared across all modules.
"""

import logging
from pathlib import Path


def get_logger(name: str, log_dir: str = "outputs/reports", level: str = "INFO") -> logging.Logger:
    """
    Create (or retrieve) a configured logger instance.

    Args:
        name: Logger name, typically __name__ of the calling module.
        log_dir: Directory where log files should be written.
        level: Logging level (e.g. "INFO", "DEBUG").

    Returns:
        Configured logging.Logger instance.
    """
    pass
