"""
Suggestions API Routes for OpenEvolve (mounted at ``/suggestions``).

In-memory content/classification/security/improvement suggestion surface. No
external/heavy dependencies are imported so the API always boots. Responses are
deterministic and structured.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.suggestions")

router = APIRouter()

_SUGGESTIONS: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContentRequest(BaseModel):
    content: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)


class ClassificationRequest(BaseModel):
    text: str = ""
    labels: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class SecurityRequest(BaseModel):
    code: str = ""
    language: str = "unknown"
    meta: Dict[str, Any] = Field(default_factory=dict)


class ImprovementRequest(BaseModel):
    target: str = ""
    objective: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)


@router.post("/content")
async def suggest_content(payload: ContentRequest) -> Dict[str, Any]:
    """Generate content suggestions for provided text."""
    sid = f"sug_{uuid.uuid4().hex[:12]}"
    words = [w for w in payload.content.split() if w]
    record = {
        "id": sid,
        "type": "content",
        "token_count": len(words),
        "suggestions": [
            "Consider clarifying the primary objective.",
            "Add a concrete example to improve comprehension.",
        ],
        "created_at": _now_iso(),
    }
    _SUGGESTIONS[sid] = record
    logger.info("suggestion_content", sid=sid)
    return {"success": True, "status": "ok", **record}


@router.post("/classification")
async def suggest_classification(payload: ClassificationRequest) -> Dict[str, Any]:
    """Classify text against candidate labels."""
    sid = f"cls_{uuid.uuid4().hex[:12]}"
    label = payload.labels[0] if payload.labels else "unlabeled"
    record = {
        "id": sid,
        "type": "classification",
        "text": payload.text,
        "labels": payload.labels,
        "predicted_label": label,
        "confidence": 0.0,
        "created_at": _now_iso(),
    }
    _SUGGESTIONS[sid] = record
    logger.info("suggestion_classification", sid=sid)
    return {"success": True, "status": "ok", **record}


@router.post("/security")
async def suggest_security(payload: SecurityRequest) -> Dict[str, Any]:
    """Produce security suggestions for code."""
    sid = f"sec_{uuid.uuid4().hex[:12]}"
    findings = []
    lowered = payload.code.lower()
    if "eval(" in lowered:
        findings.append({"severity": "high", "rule": "dynamic-eval", "message": "Avoid eval() on untrusted input."})
    if "password" in lowered or "secret" in lowered:
        findings.append({"severity": "medium", "rule": "hardcoded-secret", "message": "Avoid hardcoding secrets."})
    record = {
        "id": sid,
        "type": "security",
        "language": payload.language,
        "findings": findings,
        "finding_count": len(findings),
        "created_at": _now_iso(),
    }
    _SUGGESTIONS[sid] = record
    logger.info("suggestion_security", sid=sid, findings=len(findings))
    return {"success": True, "status": "ok", **record}


@router.post("/improvement")
async def suggest_improvement(payload: ImprovementRequest) -> Dict[str, Any]:
    """Produce improvement suggestions for a target/objective."""
    sid = f"imp_{uuid.uuid4().hex[:12]}"
    record = {
        "id": sid,
        "type": "improvement",
        "target": payload.target,
        "objective": payload.objective,
        "suggestions": [
            "Refactor into smaller, testable units.",
            "Add observability/metrics around the hot path.",
        ],
        "created_at": _now_iso(),
    }
    _SUGGESTIONS[sid] = record
    logger.info("suggestion_improvement", sid=sid)
    return {"success": True, "status": "ok", **record}
