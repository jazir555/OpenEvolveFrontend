"""
API Bridge Layer - Connects BubbleLab React Frontend to Python Backend
Provides CORS, SSE streaming, and authentication bridging
"""

import os
import sys
import json
import logging
import asyncio
import time
from typing import Optional, Dict, Any, AsyncGenerator
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import existing backend
try:
    from api_server import app as backend_app
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logging.warning("Backend api_server not available. Running in bridge-only mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

api_bridge = FastAPI(
    title="OpenEvolve API Bridge",
    description="Bridge layer between BubbleLab React UI and Python backend",
    version="1.0.0"
)

# ============================================================================
# CORS Middleware Configuration
# ============================================================================

# Configure allowed origins from environment
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")

api_bridge.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

logger.info(f"CORS configured for origins: {allowed_origins}")

# ============================================================================
# Request Logging Middleware
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time
        response.headers["X-Process-Time"] = str(duration)

        # Log response
        logger.info(f"Response: {response.status_code} - {duration:.4f}s")

        return response

api_bridge.add_middleware(RequestLoggingMiddleware)

# ============================================================================
# Health Check
# ============================================================================

startup_time = datetime.now()

class HealthResponse(BaseModel):
    status: str = "healthy"
    backend_connected: bool
    uptime_seconds: float
    version: str
    timestamp: str

@api_bridge.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - startup_time).total_seconds()

    return HealthResponse(
        status="healthy",
        backend_connected=BACKEND_AVAILABLE,
        uptime_seconds=uptime,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )

# ============================================================================
# Mount Backend Application
# ============================================================================

if BACKEND_AVAILABLE:
    # Mount the existing backend at /api prefix
    api_bridge.mount("/api", backend_app)
    logger.info("Backend api_server mounted at /api")
else:
    logger.warning("Backend not available. Only bridge endpoints are active.")

# ============================================================================
# SSE Streaming Infrastructure
# ============================================================================

class SSEEvent(BaseModel):
    """Server-Sent Event model"""
    event: Optional[str] = None
    data: Dict[str, Any]
    id: Optional[str] = None
    retry: Optional[int] = None

async def sse_event_generator(
    event_id: str,
    event_queue: asyncio.Queue
) -> AsyncGenerator[str, None]:
    """
    Generate SSE events from a queue

    Args:
        event_id: Event stream identifier
        event_queue: Queue to pull events from

    Yields:
        SSE-formatted event strings
    """
    try:
        retry_count = 0
        max_retries = 3

        while True:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(event_queue.get(), timeout=30.0)

                # Format SSE event
                sse_data = {
                    "id": event.get("id", f"{event_id}-{int(time.time())}"),
                    "event": event.get("event", "message"),
                    "data": json.dumps(event.get("data", {})),
                    "retry": event.get("retry", 3000)
                }

                # Build SSE string
                sse_str = ""
                if sse_data["id"]:
                    sse_str += f"id: {sse_data['id']}\n"
                if sse_data["event"]:
                    sse_str += f"event: {sse_data['event']}\n"
                if sse_data["retry"]:
                    sse_str += f"retry: {sse_data['retry']}\n"
                sse_str += f"data: {sse_data['data']}\n\n"

                yield sse_str
                retry_count = 0  # Reset retry count on success

            except asyncio.TimeoutError:
                # Send keep-alive comment
                yield ": keep-alive\n\n"
                retry_count = 0

            except Exception as e:
                logger.error(f"Error generating SSE event: {e}")
                retry_count += 1

                if retry_count >= max_retries:
                    logger.error("Max retries exceeded, closing SSE connection")
                    break

                # Send error event
                error_data = {
                    "error": str(e),
                    "retry_count": retry_count
                }
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    except asyncio.CancelledError:
        logger.info(f"SSE stream {event_id} cancelled")
        raise

# ============================================================================
# Workflow Execution Streaming
# ============================================================================

# Store active execution streams
active_streams: Dict[str, asyncio.Queue] = {}

@api_bridge.get("/stream/workflow/{workflow_id}")
async def stream_workflow_execution(workflow_id: str):
    """
    Stream workflow execution events via SSE

    Args:
        workflow_id: ID of the workflow to stream

    Returns:
        StreamingResponse with SSE events
    """
    logger.info(f"Starting SSE stream for workflow: {workflow_id}")

    # Create event queue for this stream
    event_queue = asyncio.Queue()
    active_streams[workflow_id] = event_queue

    async def event_generator():
        try:
            async for event in sse_event_generator(workflow_id, event_queue):
                yield event
        finally:
            # Cleanup
            if workflow_id in active_streams:
                del active_streams[workflow_id]
            logger.info(f"SSE stream closed for workflow: {workflow_id}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

# ============================================================================
# Event Emitters (for testing and integration)
# ============================================================================

async def emit_workflow_event(workflow_id: str, event_type: str, data: Dict[str, Any]):
    """
    Emit an event to a workflow stream

    Args:
        workflow_id: ID of the workflow
        event_type: Type of event
        data: Event data
    """
    if workflow_id in active_streams:
        event = {
            "id": f"{workflow_id}-{int(time.time() * 1000)}",
            "event": event_type,
            "data": data
        }
        await active_streams[workflow_id].put(event)
        logger.debug(f"Emitted event {event_type} to workflow {workflow_id}")
    else:
        logger.warning(f"No active stream for workflow: {workflow_id}")

# ============================================================================
# Clerk JWT Validation
# ============================================================================

class ClerkConfig(BaseModel):
    """Clerk authentication configuration"""
    jwt_secret: str = Field(..., description="Clerk JWT secret")
    api_key: str = Field(..., description="Clerk API key")

# Clerk configuration (will be loaded from environment)
clerk_config = None

def init_clerk():
    """Initialize Clerk authentication"""
    global clerk_config
    jwt_secret = os.getenv("CLERK_JWT_SECRET")
    api_key = os.getenv("CLERK_API_KEY")

    if jwt_secret and api_key:
        clerk_config = ClerkConfig(jwt_secret=jwt_secret, api_key=api_key)
        logger.info("Clerk authentication initialized")
    else:
        logger.warning("Clerk credentials not found. Authentication disabled.")

# Initialize Clerk on startup
init_clerk()

# ============================================================================
# Authentication Middleware
# ============================================================================

@api_bridge.middleware("http")
async def auth_middleware(request: Request, call_next):
    """
    Authentication middleware for validating Clerk JWT tokens
    """
    # Skip auth for health check and OPTIONS
    if request.url.path in ["/health", "/docs", "/openapi.json"] or request.method == "OPTIONS":
        return await call_next(request)

    # Extract authorization header
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        # Allow request without auth for now (will be enforced later)
        return await call_next(request)

    if not auth_header.startswith("Bearer "):
        return await call_next(request)

    token = auth_header.replace("Bearer ", "")

    # Validate token (placeholder - will implement full validation)
    try:
        if clerk_config:
            # TODO: Implement actual JWT validation
            # For now, just log the token presence
            logger.debug(f"Auth token present: {token[:20]}...")
        else:
            logger.debug("Clerk not configured, skipping auth validation")

    except Exception as e:
        logger.error(f"Auth validation error: {e}")
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid authentication token"}
        )

    # Continue with request
    response = await call_next(request)
    return response

# ============================================================================
# Startup Event
# ============================================================================

@api_bridge.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 60)
    logger.info("OpenEvolve API Bridge Starting")
    logger.info(f"Version: 1.0.0")
    logger.info(f"Backend Available: {BACKEND_AVAILABLE}")
    logger.info(f"Clerk Configured: {clerk_config is not None}")
    logger.info(f"Allowed Origins: {allowed_origins}")
    logger.info("=" * 60)

@api_bridge.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("OpenEvolve API Bridge Shutting Down")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_BRIDGE_PORT", 8001))
    host = os.getenv("API_BRIDGE_HOST", "0.0.0.0")

    logger.info(f"Starting API Bridge on {host}:{port}")

    uvicorn.run(
        "api_bridge:api_bridge",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
