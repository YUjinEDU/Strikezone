"""Lightweight logging helpers for the Strikezone project."""
from __future__ import annotations

import logging
from typing import Optional

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_logging(level: int = logging.INFO, name: str = "strikezone") -> logging.Logger:
    """Configure and return the root logger used by the application.

    Args:
        level: Logging level to apply to the root logger.
        name: Logger name to retrieve.

    Returns:
        Configured :class:`logging.Logger` instance.
    """

    logging.basicConfig(level=level, format=_LOG_FORMAT)
    return logging.getLogger(name)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger using the application namespace.

    Args:
        name: Optional suffix appended to the base application logger name.

    Returns:
        Child :class:`logging.Logger` instance.
    """

    base_name = "strikezone"
    full_name = base_name if name is None else f"{base_name}.{name}"
    return logging.getLogger(full_name)


__all__ = ["configure_logging", "get_logger"]
