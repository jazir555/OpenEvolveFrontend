"""OpenEvolve API Service"""
import structlog
import asyncio
import json
import queue
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

try:
    # Try relative imports first (when run as module)
    from .api import workflows, teams, gauntlets, execution, settings, icr, determinism, decomposition
    from .api.openevolve_v1 import router as openevolve_v1_router
    from .services.execution_service import execution_manager
except ImportError:
    # Fall back to absolute imports (when run directly)
    from api import workflows, teams, gauntlets, execution, settings, icr, determinism, decomposition
    from api.openevolve_v1 import router as openevolve_v1_router
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
