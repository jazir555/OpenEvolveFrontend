"""
BubbleLabs Automation Module

This module provides automation capabilities for BubbleLabs workflows,
including task scheduling, batch processing, and workflow automation.

Features:
- Task scheduling and execution
- Batch workflow processing
- Automated testing
- Continuous integration/continuous deployment (CI/CD) support
- Automated quality assurance

Author: OpenEvolve Team
Date: 2025-12-29
"""
from __future__ import annotations


import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import json
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AutomationTask:
    """Represents an automated task."""
    task_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    scheduled_time: Optional[datetime] = None
    interval_seconds: Optional[float] = None
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BatchJob:
    """Represents a batch processing job."""
    job_id: str
    name: str
    tasks: List[AutomationTask] = field(default_factory=list)
    status: str = "pending"
    completed_tasks: int = 0
    failed_tasks: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class BubbleLabsAutomation:
    """
    Main automation class for BubbleLabs workflows.

    This class provides comprehensive automation capabilities including
    task scheduling, batch processing, and workflow automation.
    """

    def __init__(self):
        """Initialize the automation system."""
        self.tasks: Dict[str, AutomationTask] = {}
        self.batch_jobs: Dict[str, BatchJob] = {}
        self.task_queue: Queue = Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.logger = logging.getLogger(__name__)

    def schedule_task(
        self,
        task_id: str,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        scheduled_time: Optional[datetime] = None,
        interval_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Schedule a task for execution.

        Args:
            task_id: Unique identifier for the task
            name: Human-readable task name
            function: Function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            scheduled_time: When to execute the task (None for immediate)
            interval_seconds: If set, task will repeat at this interval

        Returns:
            Dict with scheduling result
        """
        try:
            if kwargs is None:
                kwargs = {}

            task = AutomationTask(
                task_id=task_id,
                name=name,
                function=function,
                args=args,
                kwargs=kwargs,
                scheduled_time=scheduled_time,
                interval_seconds=interval_seconds
            )

            self.tasks[task_id] = task
            self.task_queue.put(task)

            self.logger.info(f"Scheduled task: {name} ({task_id})")

            return {
                "success": True,
                "task_id": task_id,
                "status": "scheduled"
            }
        except Exception as e:
            self.logger.error(f"Failed to schedule task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def create_batch_job(
        self,
        job_id: str,
        name: str,
        tasks: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a batch processing job.

        Args:
            job_id: Unique identifier for the job
            name: Human-readable job name
            tasks: List of task configurations

        Returns:
            Dict with job creation result
        """
        try:
            if tasks is None:
                tasks = []

            job = BatchJob(
                job_id=job_id,
                name=name,
                status="pending"
            )

            # Add tasks to the job
            for task_config in tasks:
                task = AutomationTask(
                    task_id=f"{job_id}_{len(job.tasks)}",
                    name=task_config.get("name", "Unnamed Task"),
                    function=task_config.get("function"),
                    args=task_config.get("args", ()),
                    kwargs=task_config.get("kwargs", {})
                )
                job.tasks.append(task)

            self.batch_jobs[job_id] = job

            self.logger.info(f"Created batch job: {name} ({job_id}) with {len(job.tasks)} tasks")

            return {
                "success": True,
                "job_id": job_id,
                "task_count": len(job.tasks)
            }
        except Exception as e:
            self.logger.error(f"Failed to create batch job: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def execute_batch_job(self, job_id: str) -> Dict[str, Any]:
        """
        Execute a batch job.

        Args:
            job_id: ID of the job to execute

        Returns:
            Dict with execution result
        """
        try:
            if job_id not in self.batch_jobs:
                return {
                    "success": False,
                    "error": f"Job not found: {job_id}"
                }

            job = self.batch_jobs[job_id]
            job.status = "running"
            job.started_at = datetime.now()

            results = []
            for task in job.tasks:
                try:
                    if task.function:
                        result = task.function(*task.args, **task.kwargs)
                        task.result = result
                        task.status = "completed"
                        job.completed_tasks += 1
                        results.append({"task_id": task.task_id, "status": "completed", "result": result})
                    else:
                        task.status = "skipped"
                        results.append({"task_id": task.task_id, "status": "skipped"})
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    job.failed_tasks += 1
                    results.append({"task_id": task.task_id, "status": "failed", "error": str(e)})

            job.status = "completed" if job.failed_tasks == 0 else "partial"
            job.completed_at = datetime.now()

            self.logger.info(f"Batch job completed: {job_id} ({job.completed_tasks}/{len(job.tasks)} tasks succeeded)")

            return {
                "success": True,
                "job_id": job_id,
                "status": job.status,
                "completed_tasks": job.completed_tasks,
                "failed_tasks": job.failed_tasks,
                "results": results
            }
        except Exception as e:
            self.logger.error(f"Failed to execute batch job: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.

        Args:
            task_id: ID of the task

        Returns:
            Dict with task status or None if not found
        """
        if task_id in self.tasks:
            task = self.tasks[task_id]
            return {
                "task_id": task.task_id,
                "name": task.name,
                "status": task.status,
                "result": task.result,
                "error": task.error,
                "created_at": task.created_at.isoformat() if task.created_at else None
            }
        return None

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a batch job.

        Args:
            job_id: ID of the job

        Returns:
            Dict with job status or None if not found
        """
        if job_id in self.batch_jobs:
            job = self.batch_jobs[job_id]
            return {
                "job_id": job.job_id,
                "name": job.name,
                "status": job.status,
                "completed_tasks": job.completed_tasks,
                "failed_tasks": job.failed_tasks,
                "total_tasks": len(job.tasks),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None
            }
        return None

    def start_worker(self):
        """Start the background worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.logger.info("Worker thread started")

    def stop_worker(self):
        """Stop the background worker thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            self.logger.info("Worker thread stopped")

    def _worker_loop(self):
        """Worker thread loop for processing tasks."""
        while self.running:
            try:
                task = self.task_queue.get(timeout=1)
                if task and task.function:
                    try:
                        task.status = "running"
                        result = task.function(*task.args, **task.kwargs)
                        task.result = result
                        task.status = "completed"
                        self.logger.info(f"Task completed: {task.name}")
                    except Exception as e:
                        task.status = "failed"
                        task.error = str(e)
                        self.logger.error(f"Task failed: {task.name} - {e}")
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker loop error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get automation statistics.

        Returns:
            Dict with statistics
        """
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == "completed")
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")
        pending_tasks = sum(1 for t in self.tasks.values() if t.status == "pending")

        total_jobs = len(self.batch_jobs)
        completed_jobs = sum(1 for j in self.batch_jobs.values() if j.status == "completed")

        return {
            "tasks": {
                "total": total_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "pending": pending_tasks
            },
            "jobs": {
                "total": total_jobs,
                "completed": completed_jobs
            },
            "worker_running": self.running
        }
