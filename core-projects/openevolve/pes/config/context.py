# -*- coding: utf-8 -*-
"""
Context for runtime execution in OpenEvolve PES framework.
Adapted from LoongFlow.
"""

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Context:
    """
    Runtime context for all PES stage components.

    This class encapsulates the execution context for a single evolution cycle,
    including task information, iteration state, and metadata.
    """

    task: str
    base_path: str | Path = "./workspace"
    init_solution: str = ""
    init_evaluation: str = ""
    init_score: float = 0.0
    task_id: uuid.UUID = field(default_factory=uuid.uuid4)
    island_id: int = 0
    current_iteration: int = 0
    total_iterations: int = 1000
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def increment_iteration(self):
        """
        Increment the current iteration counter.
        """
        self.current_iteration += 1
