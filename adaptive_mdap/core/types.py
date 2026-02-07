"""Adaptive MDAP Core Types."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

class TaskType(Enum):
    """Task type."""
    SIMPLE = "simple"
    COMPLEX = "complex"
    ADAPTIVE = "adaptive"

class ComplexityLevel(Enum):
    """Complexity level."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class TaskConfig:
    """Task configuration."""
    task_type: TaskType = TaskType.SIMPLE
    complexity: ComplexityLevel = ComplexityLevel.LOW
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

@dataclass
class ResourceAllocation:
    """Resource allocation."""
    cpu: float = 1.0
    memory: float = 1.0
    gpu: float = 0.0

@dataclass
class ExecutionContext:
    """Execution context."""
    task_id: str = ""
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
