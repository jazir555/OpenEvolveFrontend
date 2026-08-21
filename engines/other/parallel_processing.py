"""
Facade for ``parallel_processing`` so the flat ``engines/`` scripts can resolve
it without changes. The real implementation lives in
``engines/orchestration/parallel_processing.py``. It is loaded under a private
module name (via ``importlib``) to avoid a name collision with this facade, and
its public symbols are re-exported here.
"""
from __future__ import annotations


import importlib.util
import os

_real_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "orchestration", "parallel_processing.py")
)

_spec = importlib.util.spec_from_file_location("_orchestration_parallel_processing", _real_path)
_mod = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules["_orchestration_parallel_processing"] = _mod
_spec.loader.exec_module(_mod)

ParallelTaskResult = _mod.ParallelTaskResult
TaskScheduler = _mod.TaskScheduler
ParallelDecompositionProcessor = _mod.ParallelDecompositionProcessor
AsyncSolutionProcessor = _mod.AsyncSolutionProcessor
ResourceAwareParallelProcessor = _mod.ResourceAwareParallelProcessor
ParallelWorkflowOrchestrator = _mod.ParallelWorkflowOrchestrator
integrate_with_system = _mod.integrate_with_system
example_usage = _mod.example_usage

__all__ = [
    "ParallelTaskResult",
    "TaskScheduler",
    "ParallelDecompositionProcessor",
    "AsyncSolutionProcessor",
    "ResourceAwareParallelProcessor",
    "ParallelWorkflowOrchestrator",
    "integrate_with_system",
    "example_usage",
]
