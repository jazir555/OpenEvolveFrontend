"""
Execution Service for OpenEvolve API

Manages background task execution for workflows.
Follows CLAUDE.md principles: structured logging, UTC timestamps, thread safety.
"""

import structlog
import asyncio
import threading
import time
import uuid
import queue
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, Future
from enum import Enum


logger = structlog.get_logger()


class ExecutionStatus(str, Enum):
    """Execution status values"""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionEventType(str, Enum):
    """Execution event types for streaming"""
    STARTED = "execution.started"
    PROGRESS = "execution.progress"
    LOG = "execution.log"
    ERROR = "execution.error"
    COMPLETED = "execution.completed"
    PAUSED = "execution.paused"
    RESUMED = "execution.resumed"


class ExecutionManager:
    """
    Manages workflow execution in background threads.

    Features:
        - Thread pool for concurrent execution
        - Status tracking and persistence
        - Pause/Resume/Cancel capabilities
        - Log collection
    """

    def __init__(self, max_workers: int = 5):
        """
        Initialize the Execution Manager.

        Args:
            max_workers: Maximum number of concurrent executions
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="evolve_exec")

        # Execution storage
        self._executions: Dict[str, Dict[str, Any]] = {}
        self._futures: Dict[str, Future] = {}
        self._pause_events: Dict[str, threading.Event] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._listeners: Dict[str, List[queue.Queue]] = {}
        self._progress_threads: Dict[str, threading.Thread] = {}

        # Lock for thread-safe operations
        self._lock = threading.RLock()

        logger.info(
            "execution_manager_initialized",
            max_workers=max_workers
        )

    async def start_execution(
        self,
        workflow_id: str,
        problem_statement: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new workflow execution.

        Args:
            workflow_id: ID of workflow to execute
            problem_statement: Problem statement to solve
            context: Optional context

        Returns:
            Execution record with initial status

        Raises:
            ValueError: If workflow not found or invalid
        """
        try:
            # Import here to avoid circular dependency
            from ..api.workflows import _workflows
            from ..core.evolution import EvolutionEngine
            from ..core.adversarial import AdversarialEngine
            from ..core.sovereign import SovereignEngine

            # Check if workflow exists
            if workflow_id not in _workflows:
                raise ValueError(f"Workflow '{workflow_id}' not found")

            workflow = _workflows[workflow_id]

            # Create execution record
            execution_id = f"exec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            now = datetime.now(timezone.utc)

            execution = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "status": ExecutionStatus.QUEUED,
                "progress": 0.0,
                "started_at": now,
                "completed_at": None,
                "result": None,
                "error": None,
                "logs": [],
                "workflow_type": workflow.workflow_type,
                "parameters": workflow.parameters
            }

            # Create pause and cancel events
            pause_event = threading.Event()
            cancel_event = threading.Event()

            # Store execution and events
            with self._lock:
                self._executions[execution_id] = execution
                self._pause_events[execution_id] = pause_event
                self._cancel_events[execution_id] = cancel_event

            # Submit to thread pool
            future = self.executor.submit(
                self._run_workflow,
                execution_id,
                workflow_id,
                workflow.workflow_type,
                workflow.parameters,
                problem_statement,
                context,
                pause_event,
                cancel_event
            )

            self._futures[execution_id] = future
            self._start_progress_simulation(execution_id)

            self._emit_event(
                execution_id,
                ExecutionEventType.STARTED,
                {
                    "workflow_id": workflow_id,
                    "status": ExecutionStatus.QUEUED,
                    "progress": 0.0,
                },
            )

            logger.info(
                "execution_queued",
                execution_id=execution_id,
                workflow_id=workflow_id,
                workflow_type=workflow.workflow_type
            )

            return execution

        except Exception as e:
            logger.error(
                "execution_start_failed",
                workflow_id=workflow_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )
            raise

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Execution record or None if not found
        """
        with self._lock:
            execution = self._executions.get(execution_id)

        if execution:
            logger.debug(
                "execution_status_retrieved",
                execution_id=execution_id,
                status=execution["status"]
            )

        return execution

    async def pause_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Pause a running execution.

        Args:
            execution_id: Execution ID

        Returns:
            Updated execution record or None if not found

        Raises:
            ValueError: If execution cannot be paused
        """
        with self._lock:
            execution = self._executions.get(execution_id)

            if not execution:
                return None

            if execution["status"] != ExecutionStatus.RUNNING:
                raise ValueError(f"Execution '{execution_id}' is not running (status: {execution['status']})")

            # Set pause event
            pause_event = self._pause_events.get(execution_id)
            if pause_event:
                pause_event.set()

            # Update status
            execution["status"] = ExecutionStatus.PAUSED

            # Add log entry
            self._add_log(execution_id, "info", "Execution paused by user")

            logger.info(
                "execution_paused",
                execution_id=execution_id
            )
            self._emit_event(
                execution_id,
                ExecutionEventType.PAUSED,
                {"status": ExecutionStatus.PAUSED}
            )

        return execution

    async def resume_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Resume a paused execution.

        Args:
            execution_id: Execution ID

        Returns:
            Updated execution record or None if not found

        Raises:
            ValueError: If execution cannot be resumed
        """
        with self._lock:
            execution = self._executions.get(execution_id)

            if not execution:
                return None

            if execution["status"] != ExecutionStatus.PAUSED:
                raise ValueError(f"Execution '{execution_id}' is not paused (status: {execution['status']})")

            # Clear pause event
            pause_event = self._pause_events.get(execution_id)
            if pause_event:
                pause_event.clear()

            # Update status
            execution["status"] = ExecutionStatus.RUNNING

            # Add log entry
            self._add_log(execution_id, "info", "Execution resumed by user")

            logger.info(
                "execution_resumed",
                execution_id=execution_id
            )
            self._emit_event(
                execution_id,
                ExecutionEventType.RESUMED,
                {"status": ExecutionStatus.RUNNING}
            )

        return execution

    async def cancel_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Cancel an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Updated execution record or None if not found

        Raises:
            ValueError: If execution cannot be cancelled
        """
        with self._lock:
            execution = self._executions.get(execution_id)

            if not execution:
                return None

            if execution["status"] in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                raise ValueError(f"Execution '{execution_id}' cannot be cancelled (status: {execution['status']})")

            # Set cancel event
            cancel_event = self._cancel_events.get(execution_id)
            if cancel_event:
                cancel_event.set()

            # Update status
            execution["status"] = ExecutionStatus.CANCELLED
            execution["completed_at"] = datetime.now(timezone.utc)

            # Add log entry
            self._add_log(execution_id, "info", "Execution cancelled by user")

            logger.info(
                "execution_cancelled",
                execution_id=execution_id
            )
            self._emit_event(
                execution_id,
                ExecutionEventType.ERROR,
                {"status": ExecutionStatus.CANCELLED, "message": "Execution cancelled"}
            )

        return execution

    async def get_execution_logs(
        self,
        execution_id: str,
        since: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get logs for an execution.

        Args:
            execution_id: Execution ID
            since: Optional datetime filter

        Returns:
            Logs response or None if not found
        """
        with self._lock:
            execution = self._executions.get(execution_id)

            if not execution:
                return None

            # Filter logs by timestamp
            logs = execution.get("logs", [])
            if since:
                logs = [log for log in logs if log.get("timestamp") >= since.isoformat()]

            return {
                "logs": logs,
                "total": len(logs),
                "since": since
            }

    async def list_workflow_executions(
        self,
        workflow_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List all executions for a workflow.

        Args:
            workflow_id: Workflow ID
            limit: Maximum number to return

        Returns:
            List of execution records
        """
        with self._lock:
            executions = [
                exec for exec in self._executions.values()
                if exec["workflow_id"] == workflow_id
            ]

            # Sort by started_at descending
            executions.sort(key=lambda e: e["started_at"], reverse=True)

            return executions[:limit]

    def register_listener(self, execution_id: str) -> queue.Queue:
        """Register a streaming listener for an execution."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._listeners.setdefault(execution_id, []).append(q)
        return q

    def unregister_listener(self, execution_id: str, listener: queue.Queue) -> None:
        """Remove a streaming listener."""
        with self._lock:
            listeners = self._listeners.get(execution_id, [])
            if listener in listeners:
                listeners.remove(listener)
            if not listeners and execution_id in self._listeners:
                del self._listeners[execution_id]

    def _emit_event(self, execution_id: str, event_type: ExecutionEventType, data: Dict[str, Any]) -> None:
        """Emit an event to all registered listeners."""
        event = {
            "id": f"{execution_id}-{int(time.time() * 1000)}",
            "event": event_type.value,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_id": execution_id,
        }
        with self._lock:
            listeners = list(self._listeners.get(execution_id, []))
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except queue.Full:
                # Drop events if listener is too slow
                continue

    def _start_progress_simulation(self, execution_id: str) -> None:
        """Simulate progress updates while execution is running."""
        def _run():
            progress = 0.0
            while True:
                time.sleep(2)
                with self._lock:
                    execution = self._executions.get(execution_id)
                    if not execution:
                        break
                    status = execution["status"]
                    if status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                        break
                    if status == ExecutionStatus.PAUSED:
                        continue

                    progress = min(progress + 0.05, 0.95)
                    execution["progress"] = progress

                self._emit_event(
                    execution_id,
                    ExecutionEventType.PROGRESS,
                    {
                        "progress": round(progress * 100, 2),
                        "current_step": "Processing",
                        "total_steps": 100,
                        "message": "Workflow executing",
                    },
                )

        thread = threading.Thread(target=_run, daemon=True)
        self._progress_threads[execution_id] = thread
        thread.start()

    def _run_workflow(
        self,
        execution_id: str,
        workflow_id: str,
        workflow_type: str,
        parameters: Dict[str, Any],
        problem_statement: str,
        context: Optional[str],
        pause_event: threading.Event,
        cancel_event: threading.Event
    ) -> None:
        """
        Run workflow in background thread.

        Args:
            execution_id: Execution ID
            workflow_id: Workflow ID
            workflow_type: Type of workflow
            parameters: Workflow parameters
            problem_statement: Problem to solve
            context: Optional context
            pause_event: Pause event
            cancel_event: Cancel event
        """
        try:
            # Import engines
            from ..core.evolution import EvolutionEngine
            from ..core.adversarial import AdversarialEngine
            from ..core.sovereign import SovereignEngine

            # Update status to running
            with self._lock:
                self._executions[execution_id]["status"] = ExecutionStatus.RUNNING
                self._add_log(execution_id, "info", f"Starting {workflow_type} workflow execution")
            self._emit_event(
                execution_id,
                ExecutionEventType.PROGRESS,
                {
                    "progress": 5.0,
                    "current_step": "Initializing",
                    "total_steps": 100,
                    "message": f"Starting {workflow_type} workflow",
                },
            )

            # Initialize appropriate engine
            if workflow_type == "evolution":
                engine = EvolutionEngine()
            elif workflow_type == "adversarial":
                engine = AdversarialEngine()
            elif workflow_type == "sovereign":
                engine = SovereignEngine()
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            # Execute workflow
            result = engine.execute(problem_statement, parameters, context)
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)

            # Check if cancelled
            if cancel_event.is_set():
                with self._lock:
                    self._executions[execution_id]["status"] = ExecutionStatus.CANCELLED
                    self._executions[execution_id]["completed_at"] = datetime.now(timezone.utc)
                    self._add_log(execution_id, "info", "Execution cancelled")
                self._emit_event(
                    execution_id,
                    ExecutionEventType.ERROR,
                    {"status": ExecutionStatus.CANCELLED, "message": "Execution cancelled"}
                )
                return

            # Update execution with result
            with self._lock:
                if result.get("status") == "failed":
                    self._executions[execution_id]["status"] = ExecutionStatus.FAILED
                    self._executions[execution_id]["error"] = result.get("error")
                    self._add_log(execution_id, "error", f"Execution failed: {result.get('error')}")
                    self._emit_event(
                        execution_id,
                        ExecutionEventType.ERROR,
                        {"status": ExecutionStatus.FAILED, "message": result.get("error")}
                    )
                else:
                    self._executions[execution_id]["status"] = ExecutionStatus.COMPLETED
                    self._executions[execution_id]["result"] = result
                    self._executions[execution_id]["progress"] = 1.0
                    self._add_log(execution_id, "info", "Execution completed successfully")
                    self._emit_event(
                        execution_id,
                        ExecutionEventType.COMPLETED,
                        {
                            "status": ExecutionStatus.COMPLETED,
                            "message": "Execution completed",
                            "result": result,
                        }
                    )

                self._executions[execution_id]["completed_at"] = datetime.now(timezone.utc)

            logger.info(
                "workflow_execution_completed",
                execution_id=execution_id,
                workflow_type=workflow_type,
                status=self._executions[execution_id]["status"]
            )

        except Exception as e:
            logger.error(
                "workflow_execution_exception",
                execution_id=execution_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True
            )

            # Update execution with error
            with self._lock:
                self._executions[execution_id]["status"] = ExecutionStatus.FAILED
                self._executions[execution_id]["error"] = str(e)
                self._executions[execution_id]["completed_at"] = datetime.now(timezone.utc)
                self._add_log(execution_id, "error", f"Exception: {str(e)}")
            self._emit_event(
                execution_id,
                ExecutionEventType.ERROR,
                {"status": ExecutionStatus.FAILED, "message": str(e)}
            )

    def _add_log(self, execution_id: str, level: str, message: str) -> None:
        """
        Add log entry to execution.

        Args:
            execution_id: Execution ID
            level: Log level (info, warning, error)
            message: Log message
        """
        if execution_id in self._executions:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "message": message
            }
            self._executions[execution_id]["logs"].append(log_entry)
            self._emit_event(
                execution_id,
                ExecutionEventType.LOG,
                log_entry
            )


# Global execution manager instance
execution_manager = ExecutionManager(max_workers=5)
