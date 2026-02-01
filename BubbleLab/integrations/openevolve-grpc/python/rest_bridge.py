"""
REST to gRPC Bridge

Provides backward compatibility for existing HTTP REST clients.
Translates REST API calls to gRPC calls to the Python backend.

This allows gradual migration:
1. Existing TypeScript code continues using REST
2. New code can use gRPC directly
3. Eventually REST can be deprecated
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Import the gRPC client
from client import OpenEvolveGRPCClient, ExecutionRequest

logger = logging.getLogger(__name__)


class RESTToGRPCBridge:
    """
    Bridge that translates REST API calls to gRPC calls.
    
    Maintains API compatibility with the existing bubblelabs_nodes API server
    while using gRPC internally for better performance.
    """
    
    def __init__(
        self,
        grpc_host: str = "localhost",
        grpc_port: int = 50051,
        rest_host: str = "0.0.0.0",
        rest_port: int = 8000
    ):
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port
        self.rest_host = rest_host
        self.rest_port = rest_port
        
        self.grpc_client: Optional[OpenEvolveGRPCClient] = None
        self.app = self._create_app()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with all routes"""
        
        app = FastAPI(
            title="OpenEvolve REST to gRPC Bridge",
            description="Backward-compatible REST API backed by gRPC",
            version="2.0.0"
        )
        
        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Startup/shutdown events
        @app.on_event("startup")
        async def startup():
            await self._connect_grpc()
        
        @app.on_event("shutdown")
        async def shutdown():
            await self._disconnect_grpc()
        
        # Health check
        @app.get("/health")
        async def health():
            health_status = await self._check_health()
            return JSONResponse(
                content=health_status,
                status_code=200 if health_status["status"] == "healthy" else 503
            )
        
        # Root
        @app.get("/")
        async def root():
            return {
                "message": "OpenEvolve REST to gRPC Bridge",
                "version": "2.0.0",
                "docs": "/docs",
                "health": "/health"
            }
        
        # List integrations (nodes)
        @app.get("/api/integrations")
        async def list_integrations(category: Optional[str] = None):
            try:
                nodes = await self.grpc_client.list_nodes(category)
                return {
                    "success": True,
                    "nodes": [self._map_node_to_rest(n) for n in nodes]
                }
            except Exception as e:
                logger.error(f"List nodes failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Get node info
        @app.get("/api/integrations/{node_type}")
        async def get_integration_info(node_type: str):
            try:
                node = await self.grpc_client.get_node_schema(node_type)
                return {
                    "success": True,
                    "data": self._map_node_to_rest(node)
                }
            except Exception as e:
                if "not found" in str(e).lower():
                    raise HTTPException(status_code=404, detail=str(e))
                raise HTTPException(status_code=500, detail=str(e))
        
        # Execute node
        @app.post("/api/integrations/{node_type}/execute")
        async def execute_node(node_type: str, request: Request):
            try:
                body = await request.json()
                
                execution_request = ExecutionRequest(
                    nodeType=node_type,
                    inputs=body.get("inputs", {}),
                    config=body.get("config", {}),
                    options=body.get("options", {})
                )
                
                # Check if streaming is requested
                if body.get("streaming", False):
                    return StreamingResponse(
                        self._stream_execution(execution_request),
                        media_type="text/event-stream"
                    )
                
                # Synchronous execution
                result = await self.grpc_client.execute_node(execution_request)
                return self._map_execution_result_to_rest(result)
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Execute node failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Get execution status
        @app.get("/executions/{execution_id}")
        async def get_execution_status(execution_id: str):
            try:
                result = await self.grpc_client.get_execution_status(execution_id)
                return self._map_execution_result_to_rest(result)
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        # List executions
        @app.get("/executions")
        async def list_executions(limit: int = 50):
            # This would need to be implemented in the gRPC client
            return {
                "executions": [],
                "limit": limit
            }
        
        # Get node schema
        @app.get("/api/integrations/{node_type}/schema")
        async def get_node_schema(node_type: str):
            try:
                node = await self.grpc_client.get_node_schema(node_type)
                return {
                    "success": True,
                    "node_type": node_type,
                    "schema": node.parameterSchema
                }
            except Exception as e:
                raise HTTPException(status_code=404, detail=str(e))
        
        return app
    
    async def _connect_grpc(self):
        """Connect to gRPC server"""
        from client import create_grpc_client
        
        self.grpc_client = create_grpc_client(
            host=self.grpc_host,
            port=self.grpc_port
        )
        await self.grpc_client.connect()
        logger.info(f"Connected to gRPC server at {self.grpc_host}:{self.grpc_port}")
    
    async def _disconnect_grpc(self):
        """Disconnect from gRPC server"""
        if self.grpc_client:
            await self.grpc_client.close()
            logger.info("Disconnected from gRPC server")
    
    async def _check_health(self) -> Dict:
        """Check health of both REST and gRPC"""
        health = {
            "status": "healthy",
            "online": True,
            "services": {},
            "timestamp": ""
        }
        
        # Check gRPC health
        try:
            grpc_health = await self.grpc_client.check_health()
            health["services"]["grpc"] = {
                "status": grpc_health.status.lower(),
                "response_time_ms": grpc_health.responseTimeMs
            }
            if grpc_health.status != "HEALTHY":
                health["status"] = "degraded"
        except Exception as e:
            health["services"]["grpc"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health["status"] = "unhealthy"
            health["online"] = False
        
        return health
    
    async def _stream_execution(self, request: ExecutionRequest):
        """Stream execution results as server-sent events"""
        
        async def progress_handler(progress):
            data = json.dumps({
                "type": "progress",
                "percent": progress.percent,
                "message": progress.message,
                "stage": progress.stage,
                "timestamp": progress.timestamp.isoformat()
            })
            yield f"data: {data}\n\n"
        
        try:
            result = await self.grpc_client.execute_node_streaming(
                request,
                lambda p: asyncio.create_task(progress_handler(p))
            )
            
            # Send final result
            data = json.dumps({
                "type": "complete",
                "result": result.result,
                "state": result.state
            })
            yield f"data: {data}\n\n"
            
        except Exception as e:
            data = json.dumps({
                "type": "error",
                "error": str(e)
            })
            yield f"data: {data}\n\n"
    
    def _map_node_to_rest(self, node) -> Dict:
        """Map gRPC NodeInfo to REST format"""
        return {
            "node_id": node.nodeId,
            "node_type": node.nodeType,
            "display_name": node.displayName,
            "description": node.description,
            "icon": node.icon,
            "category": node.category,
            "version": node.version,
            "tags": node.tags,
            "capabilities": {
                "supports_streaming": node.capabilities.supportsStreaming,
                "supports_cancellation": node.capabilities.supportsCancellation,
                "supports_progress": node.capabilities.supportsProgress,
                "supports_checkpointing": node.capabilities.supportsCheckpointing,
                "max_timeout_seconds": node.capabilities.maxTimeoutSeconds,
                "required_resources": node.capabilities.requiredResources
            },
            "parameter_schema": node.parameterSchema
        }
    
    def _map_execution_result_to_rest(self, result) -> Dict:
        """Map gRPC ExecutionResult to REST format"""
        response = {
            "execution_id": result.executionId,
            "status": result.state.toLowerCase(),
        }
        
        if result.result:
            response["result"] = result.result
        
        if result.error:
            response["error"] = {
                "code": result.error.code,
                "message": result.error.message,
                "retryable": result.error.retryable
            }
        
        if result.progress:
            response["progress"] = {
                "percent": result.progress.percent,
                "message": result.progress.message
            }
        
        return response
    
    def run(self):
        """Run the REST bridge server"""
        logger.info(f"Starting REST to gRPC Bridge on {self.rest_host}:{self.rest_port}")
        
        uvicorn.run(
            self.app,
            host=self.rest_host,
            port=self.rest_port,
            log_level="info"
        )


# Convenience function
def create_bridge(
    grpc_host: str = "localhost",
    grpc_port: int = 50051,
    rest_host: str = "0.0.0.0",
    rest_port: int = 8000
) -> RESTToGRPCBridge:
    """Create a REST to gRPC bridge"""
    return RESTToGRPCBridge(grpc_host, grpc_port, rest_host, rest_port)


if __name__ == "__main__":
    import os
    
    bridge = create_bridge(
        grpc_host=os.getenv("GRPC_HOST", "localhost"),
        grpc_port=int(os.getenv("GRPC_PORT", "50051")),
        rest_host=os.getenv("REST_HOST", "0.0.0.0"),
        rest_port=int(os.getenv("REST_PORT", "8000"))
    )
    
    bridge.run()
