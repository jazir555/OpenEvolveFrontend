"""
OpenEvolve BubbleLabs LeanAide proxy router (mounted at ``/api/bubblelabs/leanaide``).

LeanAide is a SEPARATE service (default ``http://localhost:7654``). This module
does NOT reimplement LeanAide; it only exposes a thin health proxy so the
BubbleLab client can probe LeanAide availability through the OpenEvolve API
without cross-origin/CORS friction.

Endpoint (path relative to the ``/api`` prefix in ``main.py``):
    GET /bubblelabs/leanaide/health -> proxied LeanAide health (or degraded stub)

The target URL is read from the ``LEANAIDE_API_URL`` environment variable
(default ``http://localhost:7654``). If LeanAide is unreachable, a 200 response
with ``leanaide_available: false`` is returned so the UI degrades gracefully.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Any, Dict

from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("openevolve_api.leanaide")

router = APIRouter()

_LEANAIDE_API_URL = os.environ.get("LEANAIDE_API_URL", "http://localhost:7654")


@router.get("/bubblelabs/leanaide/health")
async def leanaide_health() -> Dict[str, Any]:
    target = f"{_LEANAIDE_API_URL.rstrip('/')}/health"
    try:
        req = urllib.request.Request(target, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            body = resp.read().decode("utf-8", "replace")
        try:
            parsed = __import__("json").loads(body)
        except Exception:
            parsed = {"raw": body}
        return {"leanaide_available": True, "server": _LEANAIDE_API_URL, **parsed}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("leanaide health probe failed", error=str(exc))
        return {
            "leanaide_available": False,
            "server": _LEANAIDE_API_URL,
            "detail": f"LeanAide health probe failed: {exc}",
        }
