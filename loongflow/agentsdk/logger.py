"""Logging utilities for LoongFlow."""

from __future__ import annotations

import logging
from typing import Any


class MessageLogger:
    """Simple message logger wrapper."""

    def __init__(self, name: str = "loongflow") -> None:
        self.logger = logging.getLogger(name)

    def log(self, message: str, **kwargs: Any) -> None:
        self.logger.info(message, extra=kwargs)


__all__ = ["MessageLogger"]
