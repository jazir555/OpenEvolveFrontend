"""
RESE REST + WebSocket API server.

Exposes the RESE 4-phase pipeline (glue.orchestration.rese_pipeline.RESEPipeline)
over HTTP and WebSocket:

    POST   /api/v1/pipeline/run        - submit a problem; runs asynchronously
    GET    /pipeline/{id}/status       - current status + per-phase progress
    GET    /pipeline/{id}/result       - full pipeline result
    DELETE /pipeline/{id}              - cancel a running pipeline
    GET    /admin/stats                - service statistics
    POST   /admin/cache/clear          - clear the result cache
    WS     /ws/pipeline/{id}           - live progress streaming

The router imports and drives the REAL RESEPipeline orchestrator (read-only
reference to glue/orchestration/rese_pipeline.py). Each run builds its own
RESEPipeline instance (with the required env defaults injected) so concurrent
runs never share mutable phase-executor state.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# Ensure the repository root is importable so the real glue pipeline can be used.
# Walk up from this file until we find the directory that contains the `glue`
# package (the repo root), which is robust to intermediate directory depth.
_REPO_ROOT = Path(__file__).resolve().parent
for _ in range(8):
    if (_REPO_ROOT / "glue" / "orchestration" / "rese_pipeline.py").exists():
        break
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        break
    _REPO_ROOT = _parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# RESEPipeline() reads required env vars or calls sys.exit(1); inject safe
# defaults so the API process can never be killed by a missing env var.
for _var, _default in {
    "PIPELINE_TIMEOUT_MS": "300000",
    "PHASE_I_TIMEOUT_MS": "60000",
    "PHASE_II_TIMEOUT_MS": "60000",
    "PHASE_III_TIMEOUT_MS": "60000",
    "PHASE_IV_TIMEOUT_MS": "60000",
    "MAX_RETRIES": "3",
    "RETRY_INITIAL_DELAY_MS": "1000",
    "RETRY_MAX_DELAY_MS": "30000",
}.items():
    os.environ.setdefault(_var, _default)

from glue.orchestration.rese_pipeline import RESEPipeline  # noqa: E402
from glue.orchestration.config import PipelineConfig  # noqa: E402
from glue.orchestration.event_bus import EventBus  # noqa: E402


# ============================================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================================


class Constraint(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    formalization: Optional[str] = None


class PipelineRunRequest(BaseModel):
    description: str = Field(..., min_length=1, description="Problem statement")
    constraints: List[Constraint] = Field(default_factory=list)
    variables: Dict[str, Any] = Field(default_factory=dict)
    phases: List[str] = Field(
        default_factory=lambda: ["phase1", "phase2", "phase3", "phase4"]
    )
    use_cache: bool = True


class PipelineRunResponse(BaseModel):
    pipeline_id: str
    problem_id: str
    correlation_id: str
    status: str
    final_solution: Dict[str, Any] = Field(default_factory=dict)
    aci_history: List[float] = Field(default_factory=list)
    validation_score: Optional[float] = None
    confidence: Optional[float] = None
    elapsed_seconds: Optional[float] = None
    phase_results: Dict[str, Any] = Field(default_factory=dict)
    cached: bool = False


# ============================================================================
# RUN RECORD + MANAGER
# ============================================================================


from dataclasses import dataclass, field  # noqa: E402


@dataclass
class RunRecord:
    pipeline_id: str
    correlation_id: str
    problem_id: str
    status: str = "pending"
    created_at: str = ""
    start_time: float = 0.0
    end_time: Optional[float] = None
    request: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    use_cache: bool = True
    was_cached: bool = False
    cancelled: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class PipelineRunManager:
    """Holds run records, caches results, and runs RESEPipeline per request."""

    PHASE_EVENTS = [
        EventBus.PHASE_I_STARTED, EventBus.PHASE_I_COMPLETED, EventBus.PHASE_I_FAILED,
        EventBus.PHASE_II_STARTED, EventBus.PHASE_II_COMPLETED, EventBus.PHASE_II_FAILED,
        EventBus.PHASE_III_STARTED, EventBus.PHASE_III_COMPLETED, EventBus.PHASE_III_FAILED,
        EventBus.PHASE_IV_STARTED, EventBus.PHASE_IV_COMPLETED, EventBus.PHASE_IV_FAILED,
        EventBus.PIPELINE_STARTED, EventBus.PIPELINE_COMPLETED, EventBus.PIPELINE_FAILED,
    ]
    TERMINAL = {"completed", "failed", "cancelled"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: Dict[str, RunRecord] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._start_time = time.time()

    # --- store helpers ---
    def _put(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.pipeline_id] = record

    def get(self, pipeline_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(pipeline_id)

    def _cache_key(self, req: PipelineRunRequest) -> str:
        payload = {
            "description": req.description,
            "constraints": [c.model_dump() for c in req.constraints],
            "variables": req.variables,
            "phases": sorted(req.phases),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    # --- run lifecycle ---
    async def run(self, req: PipelineRunRequest) -> RunRecord:
        pipeline_id = f"rese_{uuid.uuid4().hex[:12]}"
        problem_id = f"problem_{uuid.uuid4().hex[:12]}"
        correlation_id = str(uuid.uuid4())

        record = RunRecord(
            pipeline_id=pipeline_id,
            correlation_id=correlation_id,
            problem_id=problem_id,
            status="pending",
            created_at=datetime.now(timezone.utc).isoformat(),
            start_time=time.time(),
            request=req.model_dump(),
            use_cache=req.use_cache,
        )
        self._put(record)

        if req.use_cache:
            key = self._cache_key(req)
            cached = self._cache.get(key)
            if cached:
                with record.lock:
                    record.status = cached["status"]
                    record.result = cached["result"]
                    record.end_time = time.time()
                    record.was_cached = True
                    record.events.append({
                        "type": "cache_hit",
                        "data": {"cache_key": key},
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                return record

        asyncio.create_task(self._worker(record, req, key if req.use_cache else None))
        return record

    async def _worker(
        self, record: RunRecord, req: PipelineRunRequest, cache_key: Optional[str]
    ) -> None:
        try:
            config = PipelineConfig.from_env()
            config.enable_phase_i = "phase1" in req.phases
            config.enable_phase_ii = "phase2" in req.phases
            config.enable_phase_iii = "phase3" in req.phases
            config.enable_phase_iv = "phase4" in req.phases

            pipeline = RESEPipeline(config=config)

            def _handler(event):
                if event.correlation_id != record.correlation_id:
                    return
                with record.lock:
                    record.events.append({
                        "type": event.event_type,
                        "data": event.data,
                        "timestamp": event.timestamp,
                    })

            for ev_type in self.PHASE_EVENTS:
                pipeline.event_bus.subscribe(ev_type, _handler)

            context = {
                "constraints": [c.model_dump() for c in req.constraints],
                "variables": req.variables,
            }
            with record.lock:
                record.status = "running"

            result = await asyncio.to_thread(
                pipeline.execute, req.description, context, record.correlation_id
            )

            with record.lock:
                if record.cancelled:
                    record.status = "cancelled"
                else:
                    record.status = result.get("status", "completed")
                record.result = result
                record.end_time = time.time()
                if cache_key and record.status == "completed":
                    self._cache[cache_key] = {
                        "status": record.status,
                        "result": result,
                    }
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("rese_worker_failed", pipeline_id=record.pipeline_id, error=str(exc))
            with record.lock:
                record.status = "failed"
                record.error = str(exc)
                record.end_time = time.time()

    def cancel(self, pipeline_id: str) -> Optional[RunRecord]:
        record = self.get(pipeline_id)
        if record is None:
            return None
        with record.lock:
            record.cancelled = True
            if record.status not in self.TERMINAL:
                record.status = "cancelled"
            if record.end_time is None:
                record.end_time = time.time()
        return record

    # --- status / stats ---
    def build_status(self, record: RunRecord) -> Dict[str, Any]:
        with record.lock:
            status = record.status
            events = list(record.events)
            result = record.result

        phases: Dict[str, Any] = {}
        mapping = {
            "phase_i": ["phase1"],
            "phase_ii": ["phase2"],
            "phase_iii": ["phase3"],
            "phase_iv": ["phase4"],
        }
        for key, _labels in mapping.items():
            phases[key] = {"status": "pending"}
        for ev in events:
            et = ev["type"]
            if et.endswith(".started"):
                phases[_phase_key(et)]["status"] = "running"
            elif et.endswith(".completed"):
                phases[_phase_key(et)]["status"] = "completed"
            elif et.endswith(".failed"):
                phases[_phase_key(et)]["status"] = "failed"

        if result and isinstance(result.get("results"), dict):
            for key, val in result["results"].items():
                if key in phases and isinstance(val, dict):
                    phases[key] = {
                        "status": val.get("status", phases[key]["status"]),
                        "elapsed": (val.get("execution_time_ms") or 0) / 1000.0,
                    }

        elapsed = None
        if record.start_time:
            end = record.end_time or time.time()
            elapsed = end - record.start_time

        return {
            "pipeline_id": record.pipeline_id,
            "problem_id": record.problem_id,
            "correlation_id": record.correlation_id,
            "status": status,
            "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
            "phases": phases,
        }

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            active = sum(1 for r in self._runs.values() if r.status not in self.TERMINAL)
            stored = len(self._runs)
        return {
            "active_pipelines": active,
            "stored_results": stored,
            "cached_results": len(self._cache),
            "websocket_connections": _ws_connection_count,
            "uptime_seconds": round(time.time() - self._start_time, 3),
        }

    def clear_cache(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
        return count


def _phase_key(event_type: str) -> str:
    if "phase_i." in event_type or event_type.startswith("phase_i."):
        return "phase_i"
    if "phase_ii." in event_type:
        return "phase_ii"
    if "phase_iii." in event_type:
        return "phase_iii"
    if "phase_iv." in event_type:
        return "phase_iv"
    return "phase_i"


# Track live websocket connections for /admin/stats.
_ws_connection_count = 0
_ws_count_lock = threading.Lock()


# ============================================================================
# ROUTER
# ============================================================================

manager = PipelineRunManager()
router = APIRouter(tags=["rese"])


def _response_from_record(record: RunRecord, cached: bool = False) -> Dict[str, Any]:
    with record.lock:
        result = record.result
        status = record.status
    aci_history: List[float] = []
    final_solution: Dict[str, Any] = {}
    validation_score = None
    confidence = None
    elapsed = None
    phase_results: Dict[str, Any] = {}

    if result:
        phase_results = result.get("results", {}) or {}
        iv = phase_results.get("phase_iv", {}).get("data", {})
        if isinstance(iv, dict):
            final_solution = iv.get("architecture", {}) or {}
            validation_score = iv.get("validation", {}).get("aci_reduction")
        iii = phase_results.get("phase_iii", {}).get("data", {})
        if isinstance(iii, dict) and iii.get("search_result"):
            confidence = iii["search_result"].get("best_hypothesis", {}).get("confidence")
        if record.start_time:
            end = record.end_time or time.time()
            elapsed = end - record.start_time

    return {
        "pipeline_id": record.pipeline_id,
        "problem_id": record.problem_id,
        "correlation_id": record.correlation_id,
        "status": status,
        "final_solution": final_solution,
        "aci_history": aci_history,
        "validation_score": validation_score,
        "confidence": confidence,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "phase_results": phase_results,
        "cached": cached,
    }


@router.post("/api/v1/pipeline/run", response_model=PipelineRunResponse)
async def pipeline_run(req: PipelineRunRequest) -> Dict[str, Any]:
    record = await manager.run(req)
    return _response_from_record(record, cached=record.was_cached)


@router.get("/pipeline/{pipeline_id}/status")
async def pipeline_status(pipeline_id: str) -> Dict[str, Any]:
    record = manager.get(pipeline_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return manager.build_status(record)


@router.get("/pipeline/{pipeline_id}/result", response_model=PipelineRunResponse)
async def pipeline_result(pipeline_id: str) -> Dict[str, Any]:
    record = manager.get(pipeline_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return _response_from_record(record)


@router.delete("/pipeline/{pipeline_id}")
async def pipeline_delete(pipeline_id: str) -> Dict[str, Any]:
    record = manager.cancel(pipeline_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return {
        "message": f"Pipeline {pipeline_id} cancelled",
        "pipeline_id": pipeline_id,
    }


@router.get("/admin/stats")
async def admin_stats() -> Dict[str, Any]:
    return manager.stats()


@router.post("/admin/cache/clear")
async def admin_cache_clear() -> Dict[str, Any]:
    cleared = manager.clear_cache()
    return {"message": "Cache cleared", "cleared_entries": cleared}


@router.websocket("/ws/pipeline/{pipeline_id}")
async def ws_pipeline(websocket: WebSocket, pipeline_id: str) -> None:
    await websocket.accept()
    global _ws_connection_count
    with _ws_count_lock:
        _ws_connection_count += 1
    record = manager.get(pipeline_id)
    client_id = f"client_{uuid.uuid4().hex[:8]}"
    try:
        if record is None:
            await websocket.send_json({
                "type": "error",
                "error": "pipeline not found",
                "pipeline_id": pipeline_id,
            })
            await websocket.close()
            return

        await websocket.send_json({
            "type": "subscribed",
            "pipeline_id": pipeline_id,
            "client_id": client_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        last_index = 0
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                if isinstance(msg, dict) and msg.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            except asyncio.TimeoutError:
                pass

            with record.lock:
                new_events = record.events[last_index:]
                status = record.status
                result = record.result

            for ev in new_events:
                await websocket.send_json({
                    "type": "pipeline_update",
                    "pipeline_id": pipeline_id,
                    "status": status,
                    "progress": {"event_type": ev["type"], "data": ev["data"]},
                })
                last_index += 1

            if status in manager.TERMINAL:
                await websocket.send_json({
                    "type": "pipeline_complete",
                    "pipeline_id": pipeline_id,
                    "status": status,
                    "result": result,
                })
                break

            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_count_lock:
            _ws_connection_count = max(0, _ws_connection_count - 1)
