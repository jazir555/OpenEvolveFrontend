"""
Background status-sync for the :8000 -> :8001 decomposition-workflow engine link.

When :8000 proxies a workflow run to the :8001 engine (via ``POST
/api/workflows/{id}/run``), it records the engine's workflow id in
``workflow.parameters["last_engine_workflow_id"]`` and sets the :8000 workflow
status to RUNNING. The engine provides no completion callback, so the :8000
record never advances to COMPLETED/FAILED/etc.

This module polls ``GET :8001/workflows/{engine_id}`` and writes the engine
status back into the :8000 in-memory record (and SQLite store) so the BubbleLab
UI reflects the true run state. All work is best-effort and non-crashing.
"""

import os
import time
import threading
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
import structlog

from ..models import WorkflowStatus
from . import workflows as workflows_module
from .engine_proxy import ENGINE_API_BASE_URL

logger = structlog.get_logger()

# Engine status string -> :8000 WorkflowStatus. Anything not listed (created /
# draft / ready) is intentionally ignored (left unchanged).
ENGINE_STATUS_MAP: Dict[str, WorkflowStatus] = {
    "running": WorkflowStatus.RUNNING,
    "completed": WorkflowStatus.COMPLETED,
    "failed": WorkflowStatus.FAILED,
    "paused": WorkflowStatus.PAUSED,
    "cancelled": WorkflowStatus.CANCELLED,
}

# Terminal statuses we are willing to advance a RUNNING workflow into.
TERMINAL_STATUSES = {
    WorkflowStatus.COMPLETED,
    WorkflowStatus.FAILED,
    WorkflowStatus.CANCELLED,
    WorkflowStatus.PAUSED,
}

# Statuses we poll for. RUNNING is always present; QUEUED is included only if
# the enum defines it (it currently does not) so this stays forward-compatible.
_SYNC_SOURCE_STATUSES = {WorkflowStatus.RUNNING}
_queued = getattr(WorkflowStatus, "QUEUED", None)
if _queued is not None:
    _SYNC_SOURCE_STATUSES.add(_queued)

# Hard timeout so a downed engine fails fast instead of hanging the poller.
_STATUS_SYNC_HTTP_TIMEOUT = float(os.getenv("STATUS_SYNC_HTTP_TIMEOUT", "30"))

_sync_thread: Optional[threading.Thread] = None
_sync_lock = threading.Lock()


def _engine_api_key() -> Optional[str]:
    """Return the admin API key the engine was registered with, if any."""
    return os.getenv("OPENEVOLVE_API_KEY")


def _fetch_engine_status(engine_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """GET :8001/workflows/{engine_id}. Raises on transport/HTTP errors."""
    url = f"{ENGINE_API_BASE_URL}/workflows/{engine_id}"
    headers = {"X-API-Key": api_key}
    with httpx.Client(timeout=_STATUS_SYNC_HTTP_TIMEOUT) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()


def _fetch_engine_results(engine_id: str, api_key: str) -> Optional[Any]:
    """Best-effort GET :8001/workflows/{engine_id}/results. Returns None on 404."""
    url = f"{ENGINE_API_BASE_URL}/workflows/{engine_id}/results"
    headers = {"X-API-Key": api_key}
    with httpx.Client(timeout=_STATUS_SYNC_HTTP_TIMEOUT) as client:
        response = client.get(url, headers=headers)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()


def sync_engine_statuses() -> int:
    """Poll every linked RUNNING workflow and write back terminal engine status.

    Returns the number of workflows whose status was advanced this pass.
    """
    api_key = _engine_api_key()
    if not api_key:
        logger.warning(
            "status_sync_skipped",
            reason="OPENEVOLVE_API_KEY not set",
        )
        return 0

    updated = 0
    for workflow in list(workflows_module._workflows.values()):
        engine_id = (workflow.parameters or {}).get("last_engine_workflow_id")
        if not engine_id:
            continue
        if workflow.status not in _SYNC_SOURCE_STATUSES:
            continue

        try:
            engine_data = _fetch_engine_status(engine_id, api_key)
            engine_status_str = (engine_data or {}).get("status")
            if not engine_status_str:
                continue

            mapped = ENGINE_STATUS_MAP.get(engine_status_str)
            if mapped is None:
                # created/draft/ready -> leave the :8000 record unchanged.
                continue
            if mapped == workflow.status:
                continue
            # Only advance into terminal statuses; non-terminal transitions
            # (e.g. running->running) are not an advance.
            if mapped not in TERMINAL_STATUSES:
                continue

            workflow.status = mapped
            workflow.updated_at = datetime.now(timezone.utc)
            if mapped in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED,
                          WorkflowStatus.CANCELLED):
                workflow.completed_at = datetime.now(timezone.utc)

            # Best-effort: capture the engine results payload if available.
            try:
                results = _fetch_engine_results(engine_id, api_key)
                if results is not None:
                    params = dict(workflow.parameters or {})
                    params["engine_results"] = results
                    workflow.parameters = params
            except Exception as results_err:  # pragma: no cover - defensive
                logger.debug(
                    "status_sync_results_failed",
                    workflow_id=workflow.id,
                    engine_id=engine_id,
                    error=str(results_err),
                )

            workflows_module._save_workflow_to_db(workflow)
            updated += 1
            logger.info(
                "workflow_status_synced",
                workflow_id=workflow.id,
                engine_id=engine_id,
                status=mapped.value,
            )
        except Exception as exc:  # pragma: no cover - defensive
            # A single bad/missing engine workflow must never break the loop.
            logger.debug(
                "status_sync_workflow_failed",
                workflow_id=getattr(workflow, "id", None),
                engine_id=engine_id,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            continue

    return updated


def start_status_sync_loop(interval_seconds: int = 10) -> None:
    """Start (idempotently) a daemon thread that polls engine statuses."""
    global _sync_thread
    with _sync_lock:
        if _sync_thread is not None and _sync_thread.is_alive():
            return

        def _loop() -> None:
            while True:
                try:
                    sync_engine_statuses()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "status_sync_loop_error",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                time.sleep(interval_seconds)

        _sync_thread = threading.Thread(
            target=_loop,
            name="workflow-status-sync",
            daemon=True,
        )
        _sync_thread.start()
        logger.info("status_sync_loop_started", interval_seconds=interval_seconds)
