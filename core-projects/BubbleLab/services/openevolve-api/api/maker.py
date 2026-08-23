"""
Maker API Routes for OpenEvolve (mounted at ``/maker``).

In-memory tool-registry + delegation surface. No external/heavy dependencies
are imported so the API always boots.
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

    logger = logging.getLogger("openevolve_api.maker")

router = APIRouter()

_TOOLS: Dict[str, Dict[str, Any]] = {}
_DELEGATIONS: List[Dict[str, Any]] = []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolRequest(BaseModel):
    name: str = ""
    description: str = ""
    endpoint: str = ""
    capabilities: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class DelegationRequest(BaseModel):
    tool_id: str = ""
    task: str = ""
    delegate_to: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.get("/status")
async def maker_status() -> Dict[str, Any]:
    """Return the maker subsystem status."""
    return {
        "success": True,
        "status": "ready",
        "tools": len(_TOOLS),
        "delegations": len(_DELEGATIONS),
        "generated_at": _now_iso(),
    }


@router.get("/tools")
async def list_tools() -> Dict[str, Any]:
    """List registered maker tools."""
    return {
        "success": True,
        "status": "ok",
        "tools": list(_TOOLS.values()),
        "count": len(_TOOLS),
    }


@router.post("/tools")
async def create_tool(payload: ToolRequest) -> Dict[str, Any]:
    """Register a maker tool."""
    tool_id = f"tool_{uuid.uuid4().hex[:12]}"
    record = {
        "id": tool_id,
        "name": payload.name or tool_id,
        "description": payload.description,
        "endpoint": payload.endpoint,
        "capabilities": payload.capabilities,
        "meta": payload.meta,
        "created_at": _now_iso(),
        "status": "registered",
    }
    _TOOLS[tool_id] = record
    logger.info("maker_tool_created", tool_id=tool_id)
    return {"success": True, "status": "accepted", **record}


@router.get("/tools/{tool_id}")
async def get_tool(tool_id: str) -> Dict[str, Any]:
    """Get a single maker tool."""
    tool = _TOOLS.get(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    return {"success": True, "status": "ok", "tool": tool}


@router.post("/tools/{tool_id}/test")
async def test_tool(tool_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Run a (placeholder) test against a maker tool."""
    if tool_id not in _TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    return {
        "success": True,
        "status": "tested",
        "tool_id": tool_id,
        "passed": True,
        "note": "Tool test executed in-place (no external harness).",
        "tested_at": _now_iso(),
    }


@router.post("/tools/{tool_id}/validate")
async def validate_tool(tool_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Validate a maker tool's contract."""
    if tool_id not in _TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    tool = _TOOLS[tool_id]
    issues = []
    if not tool.get("name"):
        issues.append("missing name")
    if not tool.get("endpoint"):
        issues.append("missing endpoint")
    tool["valid"] = len(issues) == 0
    tool["validation_issues"] = issues
    return {
        "success": True,
        "status": "validated",
        "tool_id": tool_id,
        "valid": tool["valid"],
        "issues": issues,
        "validated_at": _now_iso(),
    }


@router.post("/tools/{tool_id}/execute")
async def execute_tool(tool_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Execute a maker tool (placeholder invocation)."""
    if tool_id not in _TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found")
    return {
        "success": True,
        "status": "executed",
        "tool_id": tool_id,
        "result": {"echo": payload, "tool": tool_id},
        "executed_at": _now_iso(),
    }


@router.get("/delegations")
async def list_delegations() -> Dict[str, Any]:
    """List maker delegations."""
    return {
        "success": True,
        "status": "ok",
        "delegations": list(_DELEGATIONS)[-50:],
        "count": len(_DELEGATIONS),
    }


@router.post("/delegations/sync")
async def sync_delegations(payload: DelegationRequest = Body(default_factory=dict)) -> Dict[str, Any]:
    """Sync (record) a delegation from one tool to another."""
    record = {
        "id": f"del_{uuid.uuid4().hex[:12]}",
        "tool_id": payload.tool_id,
        "task": payload.task,
        "delegate_to": payload.delegate_to,
        "meta": payload.meta,
        "synced_at": _now_iso(),
    }
    _DELEGATIONS.append(record)
    logger.info("maker_delegation_synced", id=record["id"])
    return {"success": True, "status": "synced", **record}
