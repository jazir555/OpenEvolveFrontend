"""
ICR Event Bridge API Routes for OpenEvolve

Provides endpoints for refinement events, reward calibration, and UI heatmap snapshots.
"""

from collections import deque
from datetime import datetime
from typing import Deque, Dict, Any, List, Optional
import uuid
import structlog
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..database import get_setting
from ..models import ICRConfig

logger = structlog.get_logger()
router = APIRouter()

# In-memory queues (best-effort)
ICR_REFINEMENT_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=200)
ICR_REWARD_CALIBRATION_QUEUE: Deque[Dict[str, Any]] = deque(maxlen=100)
ICR_REWARD_CALIBRATION_RESPONSES: Dict[str, Dict[str, Any]] = {}
ICR_HEATMAP_SNAPSHOTS: Deque[Dict[str, Any]] = deque(maxlen=100)

# Settings key
_ICR_CONFIG_KEY = "icr_config"


class IcrRefinementEvent(BaseModel):
    """Event signaling a refinement is needed."""
    reason: Optional[str] = None
    overall_score: Optional[float] = None
    weaknesses: Optional[List[str]] = None
    friction_points: Optional[List[str]] = None
    auto_refine: Optional[bool] = None


class IcrRewardCalibrationRequest(BaseModel):
    """Reward calibration request payload."""
    request_id: Optional[str] = None
    option_a: str
    option_b: str
    confidence: Optional[float] = None
    prompt: Optional[str] = None


class IcrRewardCalibrationResponse(BaseModel):
    """Reward calibration response payload."""
    request_id: Optional[str] = None
    choice: str


class IcrHeatmapPoint(BaseModel):
    """Heatmap point from UI interaction logging."""
    x: float
    y: float
    intensity: float = 0.0
    dwellMs: Optional[float] = None
    timestamp: Optional[float] = None
    type: Optional[str] = None


class IcrHeatmapSnapshot(BaseModel):
    """Heatmap snapshot payload for multimodal analysis."""
    snapshot_id: Optional[str] = None
    timestamp: Optional[float] = None
    screen_html: str
    heatmap_data_url: Optional[str] = None
    composite_data_url: Optional[str] = None
    points: List[IcrHeatmapPoint] = Field(default_factory=list)
    manual_code_delta: Optional[float] = None
    context_text: Optional[str] = None
    auto_refine: Optional[bool] = None


def _get_icr_config() -> ICRConfig:
    config_data = get_setting(_ICR_CONFIG_KEY)
    if config_data:
        try:
            return ICRConfig(**config_data)
        except Exception:
            logger.warning("failed_to_parse_stored_icr_config")
    return ICRConfig()


async def _maybe_generate_multimodal_healing(snapshot_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attempt to generate a multimodal healing prompt if analytics manager is available."""
    try:
        from analytics_manager import analytics_manager
    except Exception:
        return None

    try:
        heatmap_payload = {
            "points": snapshot_payload.get("points", []),
            "manual_code_delta": snapshot_payload.get("manual_code_delta"),
        }
        return analytics_manager.generate_multimodal_healing_prompt(
            snapshot_payload.get("context_text", "") or "",
            heatmap_snapshot=heatmap_payload,
            auto_refine_enabled=bool(snapshot_payload.get("auto_refine")),
        )
    except Exception as exc:
        logger.warning("multimodal_healing_prompt_failed", error=str(exc))
        return None


# --- Event Bridge Endpoints ---

@router.post("/events/refinement-needed")
async def icr_emit_refinement_needed(event: IcrRefinementEvent):
    payload = event.model_dump()
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REFINEMENT_EVENTS.append(payload)
    return {"queued": True}


@router.get("/events/refinement-needed")
async def icr_get_refinement_needed(limit: int = 5):
    items: List[Dict[str, Any]] = []
    while ICR_REFINEMENT_EVENTS and len(items) < limit:
        items.append(ICR_REFINEMENT_EVENTS.popleft())
    return items


@router.post("/reward-calibration/request")
async def icr_queue_reward_calibration(request: IcrRewardCalibrationRequest):
    config = _get_icr_config()
    if not config.reward_calibration_enabled:
        return {"queued": False, "disabled": True}

    payload = request.model_dump()
    if not payload.get("request_id"):
        payload["request_id"] = str(uuid.uuid4())
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REWARD_CALIBRATION_QUEUE.append(payload)
    return {"queued": True, "request_id": payload["request_id"]}


@router.get("/reward-calibration/next")
async def icr_next_reward_calibration():
    if not ICR_REWARD_CALIBRATION_QUEUE:
        return {}
    return ICR_REWARD_CALIBRATION_QUEUE.popleft()


@router.post("/reward-calibration/respond")
async def icr_reward_calibration_respond(response: IcrRewardCalibrationResponse):
    request_id = response.request_id or str(uuid.uuid4())
    payload = response.model_dump()
    payload["request_id"] = request_id
    payload["timestamp"] = datetime.utcnow().isoformat()
    ICR_REWARD_CALIBRATION_RESPONSES[request_id] = payload
    return {"received": True, "request_id": request_id}


@router.get("/reward-calibration/response/{request_id}")
async def icr_reward_calibration_response(request_id: str):
    return ICR_REWARD_CALIBRATION_RESPONSES.get(request_id, {})


@router.post("/heatmap/snapshot")
async def icr_heatmap_snapshot(snapshot: IcrHeatmapSnapshot):
    config = _get_icr_config()

    payload = snapshot.model_dump()
    if not payload.get("snapshot_id"):
        payload["snapshot_id"] = str(uuid.uuid4())
    if not payload.get("timestamp"):
        payload["timestamp"] = datetime.utcnow().timestamp()
    payload["received_at"] = datetime.utcnow().isoformat()
    ICR_HEATMAP_SNAPSHOTS.append(payload)

    analysis = None
    if config.heatmap_analysis_enabled:
        analysis = await _maybe_generate_multimodal_healing(payload)

    return {
        "queued": True,
        "snapshot_id": payload["snapshot_id"],
        "analysis": analysis,
    }


# --- Analytics / Dashboard Endpoints ---


@router.get("/analytics/overview")
async def icr_analytics_overview() -> Dict[str, Any]:
    """High-level ICR analytics overview (counts + config flags)."""
    config = _get_icr_config()
    return {
        "success": True,
        "status": "ok",
        "refinement_events": len(ICR_REFINEMENT_EVENTS),
        "reward_calibration_queued": len(ICR_REWARD_CALIBRATION_QUEUE),
        "reward_calibration_responses": len(ICR_REWARD_CALIBRATION_RESPONSES),
        "heatmap_snapshots": len(ICR_HEATMAP_SNAPSHOTS),
        "config": config.model_dump(),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/analytics/components")
async def icr_analytics_components() -> Dict[str, Any]:
    """List the ICR analytics components and their event counts."""
    components = {
        "refinement_events": len(ICR_REFINEMENT_EVENTS),
        "reward_calibration": {
            "queued": len(ICR_REWARD_CALIBRATION_QUEUE),
            "responses": len(ICR_REWARD_CALIBRATION_RESPONSES),
        },
        "heatmap": {
            "snapshots": len(ICR_HEATMAP_SNAPSHOTS),
        },
    }
    return {
        "success": True,
        "status": "ok",
        "components": components,
        "count": len(components),
    }


@router.get("/analytics/refinements")
async def icr_analytics_refinements(limit: int = 50) -> Dict[str, Any]:
    """Return a non-destructive snapshot of recent refinement events."""
    items = list(ICR_REFINEMENT_EVENTS)[-limit:]
    return {
        "success": True,
        "status": "ok",
        "refinements": items,
        "count": len(items),
    }


@router.get("/analytics/heatmap")
async def icr_analytics_heatmap(limit: int = 50) -> Dict[str, Any]:
    """Return a snapshot of recent heatmap snapshots."""
    items = list(ICR_HEATMAP_SNAPSHOTS)[-limit:]
    return {
        "success": True,
        "status": "ok",
        "heatmap": items,
        "count": len(items),
    }


@router.get("/analytics/patterns")
async def icr_analytics_patterns() -> Dict[str, Any]:
    """Return aggregated friction/weakness patterns across refinement events."""
    weakness_counter: Dict[str, int] = {}
    friction_counter: Dict[str, int] = {}
    for evt in ICR_REFINEMENT_EVENTS:
        for w in evt.get("weaknesses") or []:
            weakness_counter[w] = weakness_counter.get(w, 0) + 1
        for f in evt.get("friction_points") or []:
            friction_counter[f] = friction_counter.get(f, 0) + 1
    return {
        "success": True,
        "status": "ok",
        "weakness_patterns": weakness_counter,
        "friction_patterns": friction_counter,
    }


@router.get("/vlm/config")
async def icr_vlm_config() -> Dict[str, Any]:
    """Return the VLM configuration for multimodal analysis."""
    config = _get_icr_config()
    return {
        "success": True,
        "status": "ok",
        "vlm_provider": config.vlm_provider,
        "vlm_model": config.vlm_model,
        "heatmap_analysis_enabled": config.heatmap_analysis_enabled,
        "heatmap_snapshot_interval": config.heatmap_snapshot_interval,
    }


@router.get("/analytics/vlm")
async def icr_analytics_vlm() -> Dict[str, Any]:
    """Return VLM analytics (provider/model + recent heatmap snapshot count)."""
    config = _get_icr_config()
    recent = list(ICR_HEATMAP_SNAPSHOTS)[-10:]
    return {
        "success": True,
        "status": "ok",
        "vlm_provider": config.vlm_provider,
        "vlm_model": config.vlm_model,
        "heatmap_analysis_enabled": config.heatmap_analysis_enabled,
        "recent_snapshots": len(recent),
        "snapshots": recent,
    }


@router.get("/config")
async def icr_config_endpoint() -> Dict[str, Any]:
    """Return the current ICR config."""
    config = _get_icr_config()
    return {
        "success": True,
        "status": "ok",
        "config": config.model_dump(),
    }


@router.get("/dashboard")
async def icr_dashboard() -> Dict[str, Any]:
    """Return a consolidated ICR dashboard view."""
    config = _get_icr_config()
    return {
        "success": True,
        "status": "ok",
        "service": "icr",
        "refinement_events": len(ICR_REFINEMENT_EVENTS),
        "reward_calibration_queued": len(ICR_REWARD_CALIBRATION_QUEUE),
        "heatmap_snapshots": len(ICR_HEATMAP_SNAPSHOTS),
        "config": config.model_dump(),
        "generated_at": datetime.utcnow().isoformat(),
    }
