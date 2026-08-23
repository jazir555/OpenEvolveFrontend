"""
Adaptive MDAP API Routes for OpenEvolve (mounted at ``/adaptive-mdap``).

Provides a real, in-memory adaptive cost/complexity profiling + allocation
surface. No external/heavy dependencies are imported so the API always boots.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.adaptive_mdap")

router = APIRouter()

# In-memory stateful collections.
_PROFILES: Dict[str, Dict[str, Any]] = {}
_COST_RECORDS: List[Dict[str, Any]] = []
_COMPLEXITY_RECORDS: List[Dict[str, Any]] = []
_ALLOCATIONS: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CostRequest(BaseModel):
    profile: Optional[str] = None
    tokens: Optional[float] = None
    cost_usd: Optional[float] = None
    model: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class ComplexityRequest(BaseModel):
    profile: Optional[str] = None
    task: Optional[str] = None
    complexity_score: Optional[float] = None
    dimensions: Dict[str, Any] = Field(default_factory=dict)


class AllocateRequest(BaseModel):
    profile: Optional[str] = None
    budget_usd: Optional[float] = None
    strategy: Optional[str] = None
    weights: Dict[str, float] = Field(default_factory=dict)


_DEFAULT_PROFILES = {
    "default": {
        "name": "default",
        "description": "Baseline adaptive profile",
        "cost_budget_usd": 10.0,
        "complexity_threshold": 0.7,
        "strategy": "balanced",
    },
    "aggressive": {
        "name": "aggressive",
        "description": "High-throughput, cost-tolerant profile",
        "cost_budget_usd": 100.0,
        "complexity_threshold": 0.9,
        "strategy": "throughput",
    },
    "frugal": {
        "name": "frugal",
        "description": "Cost-minimizing profile",
        "cost_budget_usd": 1.0,
        "complexity_threshold": 0.5,
        "strategy": "cost",
    },
}

for _pn, _pv in _DEFAULT_PROFILES.items():
    _PROFILES[_pn] = dict(_pv)


@router.get("/dashboard")
async def adaptive_mdap_dashboard() -> Dict[str, Any]:
    """Return the adaptive MDAP dashboard summary."""
    total_cost = sum(r.get("cost_usd") or 0.0 for r in _COST_RECORDS)
    avg_complexity = 0.0
    if _COMPLEXITY_RECORDS:
        avg_complexity = sum(r.get("complexity_score") or 0.0 for r in _COMPLEXITY_RECORDS) / len(
            _COMPLEXITY_RECORDS
        )
    return {
        "success": True,
        "status": "ok",
        "profiles": len(_PROFILES),
        "cost_records": len(_COST_RECORDS),
        "complexity_records": len(_COMPLEXITY_RECORDS),
        "allocations": len(_ALLOCATIONS),
        "total_cost_usd": round(total_cost, 6),
        "average_complexity": round(avg_complexity, 4),
        "generated_at": _now_iso(),
    }


@router.get("/profiles")
async def list_profiles() -> Dict[str, Any]:
    """List all adaptive MDAP profiles."""
    return {
        "success": True,
        "status": "ok",
        "profiles": list(_PROFILES.values()),
        "count": len(_PROFILES),
    }


@router.get("/profiles/{name}")
async def get_profile(name: str) -> Dict[str, Any]:
    """Get a single adaptive MDAP profile."""
    profile = _PROFILES.get(name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    return {"success": True, "status": "ok", "profile": profile}


@router.post("/cost")
async def record_cost(payload: CostRequest) -> Dict[str, Any]:
    """Record a cost observation against a profile."""
    record = payload.model_dump()
    record["id"] = f"cost_{uuid.uuid4().hex[:12]}"
    record["recorded_at"] = _now_iso()
    _COST_RECORDS.append(record)
    logger.info("adaptive_mdap_cost", id=record["id"], profile=payload.profile)
    return {"success": True, "status": "accepted", **record}


@router.post("/complexity")
async def record_complexity(payload: ComplexityRequest) -> Dict[str, Any]:
    """Record a complexity observation against a profile."""
    record = payload.model_dump()
    record["id"] = f"cx_{uuid.uuid4().hex[:12]}"
    record["recorded_at"] = _now_iso()
    _COMPLEXITY_RECORDS.append(record)
    logger.info("adaptive_mdap_complexity", id=record["id"], profile=payload.profile)
    return {"success": True, "status": "accepted", **record}


@router.post("/allocate")
async def allocate(payload: AllocateRequest) -> Dict[str, Any]:
    """Allocate budget across dimensions for a profile."""
    profile_name = payload.profile or "default"
    profile = _PROFILES.get(profile_name, _DEFAULT_PROFILES["default"])
    budget = payload.budget_usd if payload.budget_usd is not None else profile.get(
        "cost_budget_usd", 10.0
    )
    weights = payload.weights or {}
    if not weights:
        weights = {"compute": 0.5, "llm": 0.3, "storage": 0.2}
    total_w = sum(weights.values()) or 1.0
    allocation = {k: round(budget * (v / total_w), 6) for k, v in weights.items()}
    alloc_id = f"alloc_{uuid.uuid4().hex[:12]}"
    _ALLOCATIONS[alloc_id] = {
        "id": alloc_id,
        "profile": profile_name,
        "strategy": payload.strategy or profile.get("strategy", "balanced"),
        "budget_usd": budget,
        "allocation": allocation,
        "created_at": _now_iso(),
    }
    logger.info("adaptive_mdap_allocate", id=alloc_id, profile=profile_name)
    return {"success": True, "status": "accepted", **_ALLOCATIONS[alloc_id]}
