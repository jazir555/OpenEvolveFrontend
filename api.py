"""
RESE REST API and WebSocket Interface

Complete API for RESE pipeline with:
- REST endpoints for pipeline control
- WebSocket for real-time updates
- Authentication and authorization
- Rate limiting

Author: Agent Z1 (Integration Specialist)
Created: 2025-12-31
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Set
from dataclasses import asdict

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, \
                        BackgroundTasks, Depends, status, Header
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.security import APIKeyHeader
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Warning: FastAPI not available. Install with: pip install fastapi uvicorn")

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import RESEConfig, get_config, APIConfig
from .rese_pipeline import RESEPipeline, ProblemInput, PipelineStatus, PipelineResult


# =============================================================================
# API Data Models
# =============================================================================

if FASTAPI_AVAILABLE:
    class ProblemRequest(BaseModel):
        """Request model for problem submission"""
        description: str = Field(..., description="Problem description")
        constraints: List[Dict[str, Any]] = Field(default_factory=list, description="List of constraints")
        variables: Dict[str, Any] = Field(default_factory=dict, description="Problem variables")
        objective: Optional[str] = Field(None, description="Objective function")
        domain: str = Field(default="general", description="Problem domain")
        phases: Optional[List[str]] = Field(None, description="Phases to run (default: all)")
        use_cache: bool = Field(default=True, description="Use cached results")

    class PipelineStatusResponse(BaseModel):
        """Response model for pipeline status"""
        pipeline_id: str
        problem_id: str
        status: str
        elapsed_seconds: float
        phases: Dict[str, Any] = Field(default_factory=dict)

    class PipelineResultResponse(BaseModel):
        """Response model for pipeline result"""
        pipeline_id: str
        problem_id: str
        status: str
        final_solution: Optional[Dict[str, Any]] = None
        aci_history: List[float] = Field(default_factory=list)
        validation_score: float = 0.0
        confidence: float = 0.0
        elapsed_seconds: float = 0.0
        phase_results: Dict[str, Any] = Field(default_factory=dict)

    class HealthResponse(BaseModel):
        """Response model for health check"""
        status: str
        version: str
        timestamp: str
        uptime_seconds: float

    class ErrorResponse(BaseModel):
        """Response model for errors"""
        error: str
        detail: str = ""
        timestamp: str


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """
    Manages WebSocket connections for real-time updates.
    """

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Connect a new WebSocket client.

        Args:
            websocket: WebSocket connection
            client_id: Client identifier
        """
        await websocket.accept()
        if client_id not in self.active_connections:
            self.active_connections[client_id] = set()
        self.active_connections[client_id].add(websocket)
        self.client_subscriptions[websocket] = set()

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Disconnect a WebSocket client.

        Args:
            websocket: WebSocket connection
        """
        # Remove from all client groups
        for client_id, connections in self.active_connections.items():
            if websocket in connections:
                connections.remove(websocket)

        # Remove subscriptions
        if websocket in self.client_subscriptions:
            del self.client_subscriptions[websocket]

    def subscribe(self, websocket: WebSocket, pipeline_id: str) -> None:
        """
        Subscribe a WebSocket connection to pipeline updates.

        Args:
            websocket: WebSocket connection
            pipeline_id: Pipeline to subscribe to
        """
        if websocket in self.client_subscriptions:
            self.client_subscriptions[websocket].add(pipeline_id)

    def unsubscribe(self, websocket: WebSocket, pipeline_id: str) -> None:
        """
        Unsubscribe a WebSocket connection from pipeline updates.

        Args:
            websocket: WebSocket connection
            pipeline_id: Pipeline to unsubscribe from
        """
        if websocket in self.client_subscriptions:
            self.client_subscriptions[websocket].discard(pipeline_id)

    async def broadcast_pipeline_update(
        self,
        pipeline_id: str,
        update: Dict[str, Any]
    ) -> None:
        """
        Broadcast pipeline update to all subscribed clients.

        Args:
            pipeline_id: Pipeline ID
            update: Update data
        """
        # Find all connections subscribed to this pipeline
        for websocket, subscriptions in self.client_subscriptions.items():
            if pipeline_id in subscriptions:
                try:
                    await websocket.send_json(update)
                except Exception:  # TODO: Catch specific exception instead of Exception
                    # Connection may be closed
                    self.disconnect(websocket)

    async def send_personal_message(
        self,
        message: Dict[str, Any],
        websocket: WebSocket
    ) -> None:
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Message to send
            websocket: WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception:  # TODO: Catch specific exception instead of Exception
            self.disconnect(websocket)


# =============================================================================
# API Authentication
# =============================================================================

class APIAuthenticator:
    """
    Handles API authentication and authorization.
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self.api_keys: Set[str] = set()
        self._load_api_keys()

    def _load_api_keys(self) -> None:
        """Load API keys from environment or file"""
        import os

        # Load from environment
        env_keys = os.environ.get('RESE_API_KEYS', '')
        if env_keys:
            self.api_keys.update(env_keys.split(','))

        # Load from file
        keys_file = Path.cwd() / 'data' / 'api_keys.txt'
        if keys_file.exists():
            with open(keys_file, 'r') as f:
                for line in f:
                    key = line.strip()
                    if key:
                        self.api_keys.add(key)

    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate API key.

        Args:
            api_key: API key to validate

        Returns:
            True if valid
        """
        if not self.config.api_key_required:
            return True

        return api_key in self.api_keys


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm.
    """

    def __init__(self, rate_per_minute: int = 60):
        self.rate_per_minute = rate_per_minute
        self.requests: Dict[str, List[datetime]] = {}

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Client identifier

        Returns:
            True if request is allowed
        """
        now = datetime.now()

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if (now - req_time).total_seconds() < 60
            ]
        else:
            self.requests[client_id] = []

        # Check rate limit
        if len(self.requests[client_id]) >= self.rate_per_minute:
            return False

        # Record request
        self.requests[client_id].append(now)
        return True


# =============================================================================
# API Application Factory
# =============================================================================

def create_app(config: Optional[RESEConfig] = None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config: Optional configuration (uses default if None)

    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")

    config = config or get_config()
    app = FastAPI(
        title="RESE API",
        description="Recursive Epistemic Solvability Engine - Complete Pipeline API",
        version=config.version,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # State
    manager = ConnectionManager()
    authenticator = APIAuthenticator(config.api)
    rate_limiter = RateLimiter(config.api.rate_limit_per_minute)
    pipelines: Dict[str, RESEPipeline] = {}
    results: Dict[str, PipelineResult] = {}

    # Startup time
    start_time = datetime.now()

    # =============================================================================
    # Middleware Dependencies
    # =============================================================================

    async def get_api_key(
        x_api_key: str = Header(None, alias=config.api.api_key_header)
    ) -> Optional[str]:
        """Validate API key if required"""
        if config.api.enable_auth:
            if not x_api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required"
                )
            if not authenticator.validate_api_key(x_api_key):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
        return x_api_key

    async def check_rate_limit(
        x_client_id: str = Header(None, alias="X-Client-ID")
    ) -> None:
        """Check rate limit"""
        client_id = x_client_id or "anonymous"
        if not rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )

    # =============================================================================
    # Pipeline Progress Callback
    # =============================================================================

    def pipeline_progress_callback(result: PipelineResult) -> None:
        """Callback for pipeline progress updates"""
        # Store result
        results[result.pipeline_id] = result

        # Broadcast to WebSocket subscribers
        update = {
            'type': 'pipeline_update',
            'pipeline_id': result.pipeline_id,
            'status': result.status.value,
            'progress': result.to_dict()
        }

        # Schedule broadcast
        asyncio.create_task(manager.broadcast_pipeline_update(
            result.pipeline_id,
            update
        ))

    # =============================================================================
    # Health Endpoints
    # =============================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check():
        """
        Health check endpoint.

        Returns API status and uptime.
        """
        uptime = (datetime.now() - start_time).total_seconds()

        return HealthResponse(
            status="healthy",
            version=config.version,
            timestamp=datetime.now().isoformat(),
            uptime_seconds=uptime
        )

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "RESE API",
            "version": config.version,
            "description": "Recursive Epistemic Solvability Engine",
            "docs": "/docs",
            "health": "/health"
        }

    # =============================================================================
    # Pipeline Endpoints
    # =============================================================================

    @app.post("/api/v1/pipeline/run", response_model=PipelineResultResponse, tags=["Pipeline"])
    async def run_pipeline(
        request: ProblemRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(get_api_key),
        client_check: None = Depends(check_rate_limit)
    ):
        """
        Run RESE pipeline on a problem.

        Submits problem to RESE pipeline and returns results.
        Use WebSocket for real-time progress updates.
        """
        # Create problem
        problem = ProblemInput(
            id=f"problem_{uuid.uuid4().hex[:8]}",
            description=request.description,
            constraints=request.constraints,
            variables=request.variables,
            objective=request.objective,
            domain=request.domain
        )

        # Create pipeline
        pipeline = RESEPipeline(config)
        pipeline.add_progress_callback(pipeline_progress_callback)

        # Store pipeline
        pipeline_id = f"rese_{problem.id}"
        pipelines[pipeline_id] = pipeline

        # Run pipeline
        try:
            result = pipeline.run(
                problem,
                phases=request.phases,
                use_cache=request.use_cache
            )

            results[pipeline_id] = result

            return PipelineResultResponse(**result.to_dict())

        except Exception as e:  # TODO: Catch specific exception instead of Exception
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Pipeline execution failed: {str(e)}"
            )

    @app.get("/api/v1/pipeline/{pipeline_id}/status", response_model=PipelineStatusResponse, tags=["Pipeline"])
    async def get_pipeline_status(
        pipeline_id: str,
        api_key: str = Depends(get_api_key)
    ):
        """
        Get status of a running pipeline.

        Returns current status and progress information.
        """
        if pipeline_id not in pipelines:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline {pipeline_id} not found"
            )

        pipeline = pipelines[pipeline_id]
        progress = pipeline.get_progress()

        return PipelineStatusResponse(
            pipeline_id=pipeline_id,
            problem_id=progress.get('pipeline_id', pipeline_id),
            status=progress['status'],
            elapsed_seconds=progress.get('elapsed_seconds', 0.0),
            phases=progress.get('phases', {})
        )

    @app.get("/api/v1/pipeline/{pipeline_id}/result", response_model=PipelineResultResponse, tags=["Pipeline"])
    async def get_pipeline_result(
        pipeline_id: str,
        api_key: str = Depends(get_api_key)
    ):
        """
        Get complete result of a pipeline execution.

        Returns final result with all phase outputs and metrics.
        """
        if pipeline_id not in results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Result for pipeline {pipeline_id} not found"
            )

        result = results[pipeline_id]

        return PipelineResultResponse(**result.to_dict())

    @app.delete("/api/v1/pipeline/{pipeline_id}", tags=["Pipeline"])
    async def cancel_pipeline(
        pipeline_id: str,
        api_key: str = Depends(get_api_key)
    ):
        """
        Cancel a running pipeline.
        """
        if pipeline_id not in pipelines:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pipeline {pipeline_id} not found"
            )

        pipeline = pipelines[pipeline_id]
        pipeline.cancel()

        return {
            "message": f"Pipeline {pipeline_id} cancelled",
            "pipeline_id": pipeline_id
        }

    # =============================================================================
    # WebSocket Endpoint
    # =============================================================================

    @app.websocket("/ws/pipeline/{pipeline_id}")
    async def pipeline_websocket(websocket: WebSocket, pipeline_id: str):
        """
        WebSocket endpoint for real-time pipeline updates.

        Connect to receive live updates as pipeline progresses.

        Message format:
        {
            "type": "subscribe",
            "pipeline_id": "..."
        }
        """
        # Generate client ID
        client_id = f"client_{uuid.uuid4().hex[:8]}"

        await manager.connect(websocket, client_id)

        try:
            # Subscribe to pipeline
            manager.subscribe(websocket, pipeline_id)

            # Send confirmation
            await manager.send_personal_message({
                "type": "subscribed",
                "pipeline_id": pipeline_id,
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }, websocket)

            # Handle incoming messages
            while True:
                data = await websocket.receive_json()

                if data.get("type") == "unsubscribe":
                    manager.unsubscribe(websocket, pipeline_id)
                    await manager.send_personal_message({
                        "type": "unsubscribed",
                        "pipeline_id": pipeline_id
                    }, websocket)

                elif data.get("type") == "ping":
                    await manager.send_personal_message({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:  # TODO: Catch specific exception instead of Exception
            print(f"WebSocket error: {e}")
            manager.disconnect(websocket)

    # =============================================================================
    # Admin Endpoints
    # =============================================================================

    @app.get("/api/v1/admin/stats", tags=["Admin"])
    async def get_admin_stats(
        api_key: str = Depends(get_api_key)
    ):
        """
        Get system statistics (admin only).
        """
        return {
            "active_pipelines": len(pipelines),
            "stored_results": len(results),
            "websocket_connections": sum(
                len(conns) for conns in manager.active_connections.values()
            ),
            "uptime_seconds": (datetime.now() - start_time).total_seconds()
        }

    @app.post("/api/v1/admin/cache/clear", tags=["Admin"])
    async def clear_cache(
        api_key: str = Depends(get_api_key)
    ):
        """
        Clear pipeline cache (admin only).
        """
        # Clear cache in all pipelines
        for pipeline in pipelines.values():
            pipeline.cache.clear()

        return {
            "message": "Cache cleared"
        }

    return app


# =============================================================================
# Server Launch
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 4,
    config: Optional[RESEConfig] = None
) -> None:
    """
    Run RESE API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        config: Optional configuration
    """
    import uvicorn

    config = config or get_config()

    app = create_app(config)

    uvicorn.run(
        app,
        host=host or config.api.host,
        port=port or config.api.port,
        workers=workers or config.api.workers,
        log_level=config.monitoring.log_level.lower()
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'create_app',
    'run_server',
    'ConnectionManager',
    'APIAuthenticator',
    'RateLimiter',
]
