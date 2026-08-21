"""
Facade for ``distributed_processing`` so the flat ``engines/`` scripts can
resolve it without changes. The real implementation lives in
``engines/orchestration/distributed_processing.py``. It is loaded under a
private module name (via ``importlib``) to avoid a name collision with this
facade, and its public symbols are re-exported here.
"""
from __future__ import annotations


import importlib.util
import os

_real_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "orchestration", "distributed_processing.py")
)

_spec = importlib.util.spec_from_file_location("_orchestration_distributed_processing", _real_path)
_mod = importlib.util.module_from_spec(_spec)
import sys as _sys
_sys.modules["_orchestration_distributed_processing"] = _mod
_spec.loader.exec_module(_mod)

WorkerStatus = _mod.WorkerStatus
WorkerInfo = _mod.WorkerInfo
TaskInfo = _mod.TaskInfo
SyncManager = _mod.SyncManager
WorkerNode = _mod.WorkerNode
DistributedCoordinator = _mod.DistributedCoordinator
DistributedProcessor = _mod.DistributedProcessor
LoadBalancer = _mod.LoadBalancer
TaskScheduler = _mod.TaskScheduler
DistributedWorkflowExecutor = _mod.DistributedWorkflowExecutor
get_distributed_executor = _mod.get_distributed_executor
enable_distributed_processing = _mod.enable_distributed_processing
run_distributed_evolution = _mod.run_distributed_evolution

__all__ = [
    "WorkerStatus",
    "WorkerInfo",
    "TaskInfo",
    "SyncManager",
    "WorkerNode",
    "DistributedCoordinator",
    "DistributedProcessor",
    "LoadBalancer",
    "TaskScheduler",
    "DistributedWorkflowExecutor",
    "get_distributed_executor",
    "enable_distributed_processing",
    "run_distributed_evolution",
]
