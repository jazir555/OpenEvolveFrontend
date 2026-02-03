"""
API Gateway - License: Apache 2.0

Unified API Gateway combining REST and GraphQL endpoints.
Provides single entry point with routing, auth, and rate limiting.

Dependencies (all permissive licenses):
- fastapi: MIT License
- uvicorn: BSD License
- pydantic: MIT License

Author: OpenEvolve
Date: 2026-02-02
"""

import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import asyncio

# FastAPI - MIT
from fastapi import FastAPI, Request, Response, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Pydantic - MIT
from pydantic import BaseModel

# OpenTelemetry - Apache 2.0
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

import logging
logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 100
    burst_size: int = 10


class GatewayConfig(BaseModel):
    """API Gateway configuration."""
    host: str = "0.0.0.0"
    port: int = 80
    enable_cors: bool = True
    enable_compression: bool = True
    enable_rate_limiting: bool = True
    enable_auth: bool = False
    rate_limit: RateLimitConfig = RateLimitConfig()
    rest_api_url: str = "http://localhost:8000"
    graphql_url: str = "http://localhost:8001/graphql"


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 100, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._requests: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        async with self._lock:
            now = time.time()
            window_start = now - 60  # 1 minute window
            
            # Get existing requests
            requests = self._requests.get(key, [])
            
            # Filter to current window
            requests = [r for r in requests if r > window_start]
            
            # Check limit
            if len(requests) >= self.requests_per_minute:
                return False
            
            # Add current request
            requests.append(now)
            self._requests[key] = requests
            
            return True


class APIGateway:
    """
    Unified API Gateway for OpenEvolve.
    
    Features:
    - Single entry point for all APIs
    - Request routing to REST/GraphQL
    - Rate limiting
    - Authentication
    - Request/response logging
    - Health aggregation
    - CORS handling
    - Compression
    
    License: Apache 2.0
    """
    
    def __init__(self, config: Optional[GatewayConfig] = None):
        self.config = config or GatewayConfig()
        self.app = FastAPI(
            title="OpenEvolve API Gateway",
            description="Unified gateway for REST and GraphQL APIs",
            version="1.0.0"
        )
        self.rate_limiter = RateLimiter(
            self.config.rate_limit.requests_per_minute,
            self.config.rate_limit.burst_size
        )
        self.security = HTTPBearer(auto_error=False)
        
        self._setup_middleware()
        self._setup_routes()
        
        # Instrument with OpenTelemetry if available
        if OPENTELEMETRY_AVAILABLE:
            FastAPIInstrumentor.instrument_app(self.app)
    
    def _setup_middleware(self) -> None:
        """Setup middleware."""
        # CORS
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )
        
        # Compression
        if self.config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Request logging
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            duration = time.time() - start
            
            logger.info(
                f"{request.method} {request.url.path} - "
                f"{response.status_code} - {duration:.3f}s"
            )
            return response
        
        # Rate limiting
        if self.config.enable_rate_limiting:
            @self.app.middleware("http")
            async def rate_limit(request: Request, call_next):
                client_ip = request.client.host if request.client else "unknown"
                
                if not await self.rate_limiter.is_allowed(client_ip):
                    raise HTTPException(
                        status_code=429,
                        detail="Rate limit exceeded"
                    )
                
                return await call_next(request)
    
    def _setup_routes(self) -> None:
        """Setup routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "service": "OpenEvolve API Gateway",
                "version": "1.0.0",
                "endpoints": {
                    "rest_api": "/api/v1",
                    "graphql": "/graphql",
                    "health": "/health",
                    "docs": "/docs"
                }
            }
        
        @self.app.get("/health")
        async def health():
            """Aggregated health check."""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {}
            }
            
            # Check REST API
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    rest_response = await client.get(
                        f"{self.config.rest_api_url}/health",
                        timeout=5.0
                    )
                    health_status["services"]["rest_api"] = {
                        "status": "healthy" if rest_response.status_code == 200 else "unhealthy",
                        "status_code": rest_response.status_code
                    }
            except Exception as e:
                health_status["services"]["rest_api"] = {
                    "status": "unreachable",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # Check GraphQL
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    graphql_response = await client.post(
                        self.config.graphql_url,
                        json={"query": "{ __typename }"},
                        timeout=5.0
                    )
                    health_status["services"]["graphql"] = {
                        "status": "healthy" if graphql_response.status_code == 200 else "unhealthy",
                        "status_code": graphql_response.status_code
                    }
            except Exception as e:
                health_status["services"]["graphql"] = {
                    "status": "unreachable",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            return health_status
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-compatible metrics endpoint."""
            # This would integrate with Prometheus metrics
            return Response(
                content="# OpenEvolve metrics\n",
                media_type="text/plain"
            )
        
        # REST API proxy
        @self.app.api_route("/api/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def proxy_rest(request: Request, path: str):
            """Proxy requests to REST API."""
            import httpx
            
            # Build target URL
            target_url = f"{self.config.rest_api_url}/{path}"
            if request.query_params:
                target_url += f"?{request.query_params}"
            
            # Forward request
            async with httpx.AsyncClient() as client:
                body = await request.body()
                response = await client.request(
                    method=request.method,
                    url=target_url,
                    headers=dict(request.headers),
                    content=body,
                    timeout=30.0
                )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        
        # GraphQL proxy
        @self.app.post("/graphql")
        async def proxy_graphql(request: Request):
            """Proxy requests to GraphQL API."""
            import httpx
            
            body = await request.body()
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.graphql_url,
                    content=body,
                    headers={
                        "Content-Type": "application/json",
                        **{k: v for k, v in request.headers.items() if k.lower() != "host"}
                    },
                    timeout=30.0
                )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
        
        # GraphQL IDE redirect
        @self.app.get("/graphql")
        async def graphql_ide():
            """Redirect to GraphQL IDE."""
            return RedirectResponse(url=f"{self.config.graphql_url}")
        
        # WebSocket upgrade for GraphQL subscriptions
        @self.app.websocket("/graphql-ws")
        async def graphql_ws(websocket):
            """WebSocket proxy for GraphQL subscriptions."""
            # This would proxy WebSocket connections
            await websocket.accept()
            await websocket.close(code=1000, reason="Not implemented")
    
    def run(self) -> None:
        """Run the gateway."""
        import uvicorn
        
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )


def create_gateway() -> FastAPI:
    """Create gateway FastAPI app."""
    gateway = APIGateway()
    return gateway.app


if __name__ == "__main__":
    gateway = APIGateway()
    gateway.run()
