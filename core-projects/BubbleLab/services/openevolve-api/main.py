"""OpenEvolve API Service"""
import structlog
import asyncio
import json
import queue
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    # Try relative imports first (when run as module)
    from .api import (
        workflows,
        teams,
        gauntlets,
        execution,
        settings,
        icr,
        determinism,
        decomposition,
    )
    from .api.openevolve_v1 import router as openevolve_v1_router
    from .api.parameters import router as parameters_router
    from .api.monitoring import router as monitoring_router
    from .api.validation import router as validation_router
    from .api.analytics import router as analytics_router
    from .api.crewai import router as crewai_router
    from .api.version_control import router as version_control_router
    from .api.evaluators import router as evaluators_router
    from .api.integrated import router as integrated_router
    from .api.leanaide import router as leanaide_router
    from .api.knowledge import router as knowledge_router
    from .api.mdap_maker import router as mdap_maker_router
    from .api.rese import router as rese_router
    from .api.gateway import gateway_router
    from .api.bubblelabs_control import router as bubblelabs_control_router
    from .api.security_proxy import router as security_proxy_router
    from .services.execution_service import execution_manager
except ImportError:
    # Fall back to absolute imports (when run directly)
    from api import (
        workflows,
        teams,
        gauntlets,
        execution,
        settings,
        icr,
        determinism,
        decomposition,
    )
    from api.openevolve_v1 import router as openevolve_v1_router
    from api.parameters import router as parameters_router
    from api.monitoring import router as monitoring_router
    from api.validation import router as validation_router
    from api.analytics import router as analytics_router
    from api.crewai import router as crewai_router
    from api.version_control import router as version_control_router
    from api.evaluators import router as evaluators_router
    from api.integrated import router as integrated_router
    from api.leanaide import router as leanaide_router
    from api.knowledge import router as knowledge_router
    from api.mdap_maker import router as mdap_maker_router
    from api.rese import router as rese_router
    from api.gateway import gateway_router
    from api.bubblelabs_control import router as bubblelabs_control_router
    from api.security_proxy import router as security_proxy_router
    from services.execution_service import execution_manager

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    # Startup
    logger.info("OpenEvolve API starting")
    yield
    # Shutdown
    logger.info("OpenEvolve API shutting down")

# Create FastAPI app
app = FastAPI(
    title="OpenEvolve Workflow API",
    description="REST API for OpenEvolve workflow execution",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # BubbleLab frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request metrics middleware (dependency-free, stdlib + Starlette only)
try:
    from .api.metrics import MetricsMiddleware
except ImportError:  # pragma: no cover - absolute import fallback
    from api.metrics import MetricsMiddleware

app.add_middleware(MetricsMiddleware)

# Include routers
app.include_router(workflows.router, prefix="/api/workflows", tags=["workflows"])
app.include_router(teams.router, prefix="/api/teams", tags=["teams"])
app.include_router(gauntlets.router, prefix="/api/gauntlets", tags=["gauntlets"])
app.include_router(execution.router, prefix="/api/executions", tags=["executions"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(icr.router, prefix="/icr", tags=["icr"])
app.include_router(determinism.router, prefix="/determinism", tags=["determinism"])
app.include_router(decomposition.router, prefix="/api/decomposition", tags=["decomposition"])

# OpenEvolve /api/v1/* dialect (mirrors openevolve/server_stdlib.py) so the
# BubbleLab integration bubbles can drive the REAL engine through this service.
app.include_router(openevolve_v1_router, prefix="/api/v1", tags=["openevolve-v1"])

# UI feature routers (real data where possible). These fill the route groups the
# BubbleLab client expects but the original service did not implement.
app.include_router(parameters_router, prefix="/api", tags=["parameters"])
app.include_router(monitoring_router, prefix="/api", tags=["monitoring"])
app.include_router(validation_router, prefix="/api", tags=["validation"])
app.include_router(analytics_router, prefix="/api", tags=["analytics"])
app.include_router(crewai_router, prefix="/api", tags=["crewai"])
app.include_router(version_control_router, prefix="/api", tags=["version-control"])
app.include_router(evaluators_router, prefix="/api", tags=["evaluators"])
app.include_router(integrated_router, prefix="/api", tags=["integrated"])
app.include_router(leanaide_router, prefix="/api", tags=["bubblelabs-leanaide"])
app.include_router(knowledge_router, prefix="/api", tags=["knowledge"])

# BubbleLabs control plane + workflow definitions/instances lifecycle.
app.include_router(bubblelabs_control_router, prefix="/bubblelabs", tags=["bubblelabs"])

# MDAP-MAKER / ROMA-MDAP-MAKER routes (previously defined but not mounted).
app.include_router(mdap_maker_router, tags=["mdap-maker"])

# RESE 4-phase pipeline REST + WebSocket API, backed by the real RESEPipeline.
app.include_router(rese_router, tags=["rese"])

# Unified API Gateway capability surface (auth/registry/lb/cache/stats).
if gateway_router is not None:
    app.include_router(gateway_router, prefix="/gateway", tags=["gateway"])

# Security proxy: transparently forward /security/* to the :8001 engine so the
# BubbleLab UI can manage OpenEvolve API keys / roles / audit logs through :8000.
app.include_router(security_proxy_router, prefix="/security", tags=["security"])

# PES Enhanced router. The module lives at the repo root
# (openevolve_pes_enhanced/) and is NOT part of this service package, so it is
# imported defensively. If the package is importable we mount its real router
# (it internally degrades to HTTP 503 when its own heavy deps, e.g. LoongFlow /
# torch, are absent). If the package itself cannot be imported we still register
# a fallback router so the /api/pes-enhanced group is reachable (HTTP 501) and
# honest instead of returning 404.
_pes_router = None
_pes_unavailable_reason = None
try:
    import os as _os
    import sys as _sys

    _pes_base = _os.path.dirname(_os.path.abspath(__file__))
    _pes_root = _pes_base
    for _ in range(6):
        if _os.path.isdir(_os.path.join(_pes_root, "openevolve_pes_enhanced")):
            break
        _parent = _os.path.dirname(_pes_root)
        if _parent == _pes_root:
            break
        _pes_root = _parent

    if _pes_root not in _sys.path:
        _sys.path.insert(0, _pes_root)

    if _os.path.isdir(_os.path.join(_pes_root, "openevolve_pes_enhanced")):
        from openevolve_pes_enhanced.api_routes import router as _pes_router
    else:
        _pes_unavailable_reason = "openevolve_pes_enhanced package not found on path"
except Exception as _pes_exc:
    _pes_router = None
    _pes_unavailable_reason = str(_pes_exc)

if _pes_router is not None:
    app.include_router(_pes_router, prefix="/api", tags=["pes-enhanced"])
else:
    from fastapi import APIRouter as _PESFallbackRouter
    from fastapi.responses import JSONResponse as _PESJSONResponse

    _pes_fallback = _PESFallbackRouter(prefix="/api/pes-enhanced", tags=["pes-enhanced"])

    async def _pes_unavailable():
        return _PESJSONResponse(
            status_code=501,
            content={
                "error": "PES enhanced unavailable",
                "detail": (
                    "PES enhanced unavailable: missing dependency "
                    f"{_pes_unavailable_reason or 'unknown'}"
                ),
            },
        )

    for _pes_path in (
        "/health",
        "/runs",
        "/cost-estimate",
        "/recommend-strategy",
        "/runs/{run_id}",
        "/runs/{run_id}/budget",
        "/runs/{run_id}/stop",
    ):
        _pes_fallback.add_api_route(
            _pes_path, _pes_unavailable, methods=["GET", "POST"]
        )
    app.include_router(_pes_fallback, tags=["pes-enhanced"])

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "openevolve-api",
        "version": "0.1.0",
        "features": {
            "evolution": True,
            "adversarial": True,
            "sovereign": True,
            "web3": True,
        }
    }


# SSE streaming endpoint for workflow execution
@app.get("/stream/workflow/{workflow_id}")
async def stream_workflow_execution(workflow_id: str):
    """Stream workflow execution events via SSE."""

    async def event_generator():
        listener: queue.Queue | None = None
        active_execution_id: str | None = None
        try:
            while True:
                execution_id = workflows._workflow_executions.get(workflow_id)
                if execution_id and execution_id != active_execution_id:
                    if listener and active_execution_id:
                        execution_manager.unregister_listener(active_execution_id, listener)
                    listener = execution_manager.register_listener(execution_id)
                    active_execution_id = execution_id

                if not listener:
                    yield ": waiting\n\n"
                    await asyncio.sleep(1)
                    continue

                try:
                    event = await asyncio.to_thread(listener.get, timeout=30)
                    payload = json.dumps(event)
                    yield f"data: {payload}\n\n"
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            if listener and active_execution_id:
                execution_manager.unregister_listener(active_execution_id, listener)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "OpenEvolve Workflow API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }
