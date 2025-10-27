"""Utility functions for project-wide logging configuration."""
from __future__ import annotations

import logging
from typing import Optional


_LOGGER_NAME = "strikezone"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a module-level logger configured for the application.

    Parameters
    ----------
    name:
        Optional child logger name. When omitted the project root logger is
        returned.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the global logging handler once.

    Parameters
    ----------
    level:
        Desired logging level for the application.
    """

    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        # Already configured
        logger.setLevel(level)
        return

    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
