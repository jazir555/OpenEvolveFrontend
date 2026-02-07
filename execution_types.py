"""Execution Types module."""
from typing import Any, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

class ExecutionStatus(Enum):
    """Execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExecutionResult:
    """Execution result."""
    status: ExecutionStatus = ExecutionStatus.PENDING
    output: Any = None
    error: Optional[str] = None
