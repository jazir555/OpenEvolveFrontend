"""Utils Logging module."""
from typing import Any, Dict, List, Optional
import logging

def get_logger(name: str) -> logging.Logger:
    """Get logger."""
    return logging.getLogger(name)
