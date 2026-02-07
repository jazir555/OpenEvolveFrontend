"""General Utils module."""
from typing import Any, Dict, List, Optional
import logging

def setup_logging():
    """Setup logging."""
    logging.basicConfig(level=logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Get logger."""
    return logging.getLogger(name)
