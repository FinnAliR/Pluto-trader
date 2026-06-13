"""Logging setup for the trading bot."""

from __future__ import annotations

import logging
import sys


def setup_logger(log_level: str = "INFO") -> logging.Logger:
    """Configure a console logger once and reuse it across modules."""
    logger = logging.getLogger("trading_bot")

    if logger.handlers:
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logger.addHandler(handler)
    return logger

