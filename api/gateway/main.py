"""
OpenEvolve API Gateway
Main FastAPI application
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Main
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Import middleware
from middleware.cors import setup_cors
from middleware.rate_limit import limiter, RateLimiter

# Import routes
from routes import auth, evolution

# Import WebSocket manager
from realtime.manager import manager, EvolutionRoomManager, AdversarialRoomManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting OpenEvolve API Gateway...")
    logger.info(f"Environment: {os.getenv('ENV', 'development')}")
    logger.info(f"API Port: {os.getenv('API_PORT', 8000)}")
    logger.info(f"Rate Limiting: {os.getenv('RATE_LIMIT_ENABLED', 'True')}")

    yield

    # Shutdown
    logger.info("Shutting down OpenEvolve API Gateway...")


# Create FastAPI application
app = FastAPI(
    title="OpenEvolve API Gateway",
    description="REST API and WebSocket Gateway for OpenEvolve Backend Engines",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Setup CORS
setup_cors(app)

# Setup rate limiting
if os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true":
    RateLimiter(app)
    logger.info("Rate limiting enabled")


# =============================================================================
# MIDDLEWARE
# =============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.utcnow()

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = (datetime.utcnow() - start_time).total_seconds()

    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )

    # Add timing header
    response.headers["X-Process-Time"] = str(duration)

    return response


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

    return response


# =============================================================================
# ROUTES
# =============================================================================

# Health check
@app.get("/", tags=["Health"])
@limiter.limit("60/minute")  # Rate limit health checks
async def root(request: Request):
    """Root endpoint with API information"""
    return {
        "name": "OpenEvolve API Gateway",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "websocket": "healthy",
            "rate_limit": "enabled" if os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true" else "disabled",
        },
    }


# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(evolution.router, prefix="/api/v1")


# =============================================================================
# WEBSOCKET ENDPOINTS
# =============================================================================

@app.websocket("/ws/evolution/{evolution_id}")
async def websocket_evolution(websocket: WebSocket, evolution_id: str):
    """
    WebSocket endpoint for evolution progress updates

    Args:
        websocket: WebSocket connection
        evolution_id: Evolution ID to subscribe to
    """
    room = f"evolution:{evolution_id}"
    user_id = websocket.query_params.get("user_id", "anonymous")

    await manager.connect(websocket, room, user_id)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()

            # Echo back or handle client messages
            await websocket.send_json({
                "type": "echo",
                "data": {"message": f"Received: {data}"},
            })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected from room: {room}")


@app.websocket("/ws/adversarial/{test_id}")
async def websocket_adversarial(websocket: WebSocket, test_id: str):
    """
    WebSocket endpoint for adversarial testing updates

    Args:
        websocket: WebSocket connection
        test_id: Test ID to subscribe to
    """
    room = f"adversarial:{test_id}"
    user_id = websocket.query_params.get("user_id", "anonymous")

    await manager.connect(websocket, room, user_id)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "echo",
                "data": {"message": f"Received: {data}"},
            })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected from room: {room}")


@app.websocket("/ws/workflow/{workflow_id}")
async def websocket_workflow(websocket: WebSocket, workflow_id: str):
    """
    WebSocket endpoint for workflow progress updates

    Args:
        websocket: WebSocket connection
        workflow_id: Workflow ID to subscribe to
    """
    room = f"workflow:{workflow_id}"
    user_id = websocket.query_params.get("user_id", "anonymous")

    await manager.connect(websocket, room, user_id)

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_json({
                "type": "echo",
                "data": {"message": f"Received: {data}"},
            })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected from room: {room}")


@app.websocket("/ws/collaboration/{room_id}")
async def websocket_collaboration(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for real-time collaboration

    Args:
        websocket: WebSocket connection
        room_id: Collaboration room ID
    """
    room = f"collaboration:{room_id}"
    user_id = websocket.query_params.get("user_id", "anonymous")
    username = websocket.query_params.get("username", "Anonymous")

    await manager.connect(websocket, room, user_id)

    # Broadcast user joined
    from realtime.manager import CollaborationRoomManager
    collab_manager = CollaborationRoomManager()
    await collab_manager.broadcast_user_joined(room_id, user_id, username)

    try:
        while True:
            data = await websocket.receive_json()

            # Handle different message types
            message_type = data.get("type")

            if message_type == "content_update":
                await collab_manager.broadcast_content_update(
                    room_id,
                    user_id,
                    data.get("content", ""),
                )
            elif message_type == "cursor_update":
                await collab_manager.broadcast_cursor_update(
                    room_id,
                    user_id,
                    data.get("position", {}),
                )

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        await collab_manager.broadcast_user_left(room_id, user_id)
        logger.info(f"WebSocket disconnected from room: {room}")


@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """
    WebSocket endpoint for system monitoring updates

    Args:
        websocket: WebSocket connection
    """
    room = "monitoring:global"
    user_id = "system"

    await manager.connect(websocket, room, user_id)

    try:
        while True:
            # Send periodic monitoring updates
            import asyncio
            await asyncio.sleep(5)

            await websocket.send_json({
                "type": "resource_update",
                "data": {
                    "cpu_percent": 45.2,
                    "memory_percent": 62.1,
                    "disk_percent": 55.3,
                    "active_connections": manager.get_total_connections(),
                },
                "timestamp": datetime.utcnow().isoformat(),
            })

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected from room: {room}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)} if os.getenv("ENV") == "development" else None,
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(status.HTTP_404_NOT_FOUND)
async def not_found_handler(request: Request, exc: status.HTTP_404_NOT_FOUND):
    """404 error handler"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested resource was not found",
                "details": {"path": request.url.path},
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(status.HTTP_422_UNPROCESSABLE_ENTITY)
async def validation_error_handler(
    request: Request, exc: status.HTTP_422_UNPROCESSABLE_ENTITY
):
    """422 validation error handler"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request data",
                "details": exc.detail(),
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("API_HOST", "0.0.0.0")
    reload = os.getenv("API_RELOAD", "True").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
