"""
OpenEvolve BubbleLabs LeanAide proxy router (mounted at ``/api/bubblelabs/leanaide``).

LeanAide is a SEPARATE service (default ``http://localhost:7654``). This module
does NOT reimplement LeanAide; it only proxies the LeanAide HTTP surface so the
BubbleLab client can reach LeanAide through the OpenEvolve API without
cross-origin/CORS friction.

Endpoints (paths relative to the ``/api`` prefix in ``main.py``):
    GET  /bubblelabs/leanaide/health       -> {LEANAIDE_API_URL}/health
    GET  /bubblelabs/leanaide/status        -> {LEANAIDE_API_URL}/status
    POST /bubblelabs/leanaide/execute       -> {LEANAIDE_API_URL}/execute (body forwarded)
    GET  /bubblelabs/leanaide/trees         -> {LEANAIDE_API_URL}/trees
    GET  /bubblelabs/leanaide/trees/{id}    -> {LEANAIDE_API_URL}/trees/{id}
    GET  /bubblelabs/leanaide/proofs        -> {LEANAIDE_API_URL}/proofs
    GET  /bubblelabs/leanaide/proofs/{id}   -> {LEANAIDE_API_URL}/proofs/{id}
    POST /bubblelabs/leanaide/prove         -> {LEANAIDE_API_URL}/prove (body forwarded)

The target URL is read from the ``LEANAIDE_API_URL`` environment variable
(default ``http://localhost:7654``). If LeanAide is unreachable, a 502/503
response with ``leanaide_available: false`` is returned so the UI degrades
gracefully instead of crashing the server.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Request
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

_PROXY_TIMEOUT = float(os.environ.get("LEANAIDE_PROXY_TIMEOUT", "5"))


def _upstream_base() -> str:
    return _LEANAIDE_API_URL.rstrip("/")


async def _forward(
    upstream_path: str,
    method: str,
    request: Optional[Request] = None,
    *,
    json_body: Any = None,
) -> Tuple[int, Dict[str, Any]]:
    """Forward a request to LeanAide and return ``(status_code, body)``.

    On any transport failure (connection refused, timeout, DNS, bad status
    while reading) this returns a 502/503 with a graceful error payload that
    includes ``leanaide_available: false`` rather than raising.
    """
    target = f"{_upstream_base()}/{upstream_path.lstrip('/')}"

    headers: Dict[str, str] = {"Accept": "application/json"}
    data: Optional[bytes] = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif request is not None:
        raw = await request.body()
        if raw:
            data = raw
            ctype = request.headers.get("content-type")
            if ctype:
                headers["Content-Type"] = ctype

    try:
        req = urllib.request.Request(
            target, data=data, method=method.upper(), headers=headers
        )
        with urllib.request.urlopen(req, timeout=_PROXY_TIMEOUT) as resp:
            status = resp.status
            raw_body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        # Upstream returned an error status; surface it as-is when possible.
        status = exc.code
        try:
            raw_body = exc.read().decode("utf-8", "replace")
        except Exception:
            raw_body = ""
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.warning("leanaide proxy upstream unreachable", error=str(exc))
        return (
            502,
            {
                "error": f"LeanAide upstream unreachable: {exc}",
                "upstream": target,
                "leanaide_available": False,
            },
        )

    try:
        parsed: Dict[str, Any] = json.loads(raw_body) if raw_body else {}
    except Exception:
        parsed = {"raw": raw_body}

    # Annotate success payloads with availability + upstream for the client.
    if isinstance(parsed, dict) and status < 400:
        parsed.setdefault("leanaide_available", True)
        parsed.setdefault("server", _LEANAIDE_API_URL)

    return status, parsed


@router.get("/bubblelabs/leanaide/health")
async def leanaide_health() -> Dict[str, Any]:
    status, body = await _forward("health", "GET")
    if status >= 400:
        return {
            "leanaide_available": False,
            "server": _LEANAIDE_API_URL,
            "detail": body.get("error", "LeanAide health probe failed"),
        }
    return body


@router.get("/bubblelabs/leanaide/status")
async def leanaide_status() -> Any:
    status, body = await _forward("status", "GET")
    return JSONResponse(status_code=status, content=body)


@router.post("/bubblelabs/leanaide/execute")
async def leanaide_execute(request: Request) -> Any:
    try:
        json_body = await request.json()
    except Exception:
        json_body = None
    status, body = await _forward("execute", "POST", json_body=json_body)
    return JSONResponse(status_code=status, content=body)


@router.get("/bubblelabs/leanaide/trees")
async def leanaide_trees() -> Any:
    status, body = await _forward("trees", "GET")
    return JSONResponse(status_code=status, content=body)


@router.get("/bubblelabs/leanaide/trees/{tree_id}")
async def leanaide_tree(tree_id: str) -> Any:
    status, body = await _forward(f"trees/{tree_id}", "GET")
    return JSONResponse(status_code=status, content=body)


@router.get("/bubblelabs/leanaide/proofs")
async def leanaide_proofs() -> Any:
    status, body = await _forward("proofs", "GET")
    return JSONResponse(status_code=status, content=body)


@router.get("/bubblelabs/leanaide/proofs/{proof_id}")
async def leanaide_proof(proof_id: str) -> Any:
    status, body = await _forward(f"proofs/{proof_id}", "GET")
    return JSONResponse(status_code=status, content=body)


@router.post("/bubblelabs/leanaide/prove")
async def leanaide_prove(request: Request) -> Any:
    try:
        json_body = await request.json()
    except Exception:
        json_body = None
    status, body = await _forward("prove", "POST", json_body=json_body)
    return JSONResponse(status_code=status, content=body)
