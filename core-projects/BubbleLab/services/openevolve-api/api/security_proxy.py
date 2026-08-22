"""
Security proxy for the OpenEvolve Decomposition-Workflow engine (:8001).

This mirrors ``api/engine_proxy.py``: the ``:8000`` OpenEvolve API service acts
as the unified authority for the BubbleLab UI, so key/role/audit management is
delegated to the ``:8001`` engine's ``/security/*`` surface.

Auth: the frontend sends ``X-API-Key`` (value of ``OPENEVOLVE_API_KEY``). The
header is forwarded verbatim so ``:8001``'s RBAC enforcement can validate it
(the engine itself enforces admin auth + RBAC; the proxy only forwards the
caller's credentials). The ``Authorization`` header is forwarded too when
present. No secrets are hardcoded.
"""

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import Response

from .engine_proxy import ENGINE_API_BASE_URL

# Hard timeout so a downed engine fails fast instead of hanging the proxy.
SECURITY_HTTP_TIMEOUT = 30

router = APIRouter()


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy_security(path: str, request: Request):
    """
    Forward any ``/security/{path}`` request to the ``:8001`` engine.

    The engine owns key creation/listing/deletion, role management, and the
    audit-log read endpoints; this proxy is a transparent pass-through that
    preserves the upstream status code, headers, and JSON body.
    """
    url = f"{ENGINE_API_BASE_URL}/security/{path}"

    headers = {}
    api_key = request.headers.get("X-API-Key")
    if api_key:
        headers["X-API-Key"] = api_key
    authorization = request.headers.get("Authorization")
    if authorization:
        headers["Authorization"] = authorization

    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    try:
        async with httpx_client() as client:
            response = await client.request(
                request.method,
                url,
                headers=headers,
                content=body,
            )
    except (httpx.ConnectError, httpx.HTTPError):
        raise HTTPException(
            status_code=502,
            detail="OpenEvolve engine (security) unreachable",
        ) from None

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )


def httpx_client():
    """Construct the ``httpx.AsyncClient`` used for forwarding.

    Split out so tests can monkeypatch ``security_proxy.httpx_client`` without
    touching the network.
    """
    return httpx.AsyncClient(timeout=SECURITY_HTTP_TIMEOUT)
