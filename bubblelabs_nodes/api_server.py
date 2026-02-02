"""
FastAPI Server for BubbleLabs OpenEvolve Nodes

Provides REST API endpoints for all OpenEvolve nodes to be used by the
TypeScript integration library and BubbleLab plugin.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uvicorn
import logging
from contextlib import asynccontextmanager
import signal
import asyncio


# Import node registry with error handling
try:
    from . import NodeRegistry, get_node
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import NodeRegistry: {e}")
    # Define minimal fallback implementations
    class FallbackNodeRegistry:
        @classmethod
        def list_nodes(cls):
            return {}

        @classmethod
        def get_node_info(cls, node_type: str):
            raise ValueError(f"Node type '{node_type}' not available")

    NodeRegistry = FallbackNodeRegistry()

    def get_node(node_type: str, config=None):
        raise ValueError(f"Node type '{node_type}' not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response Models
class ExecutionRequest(BaseModel):
    """Request model for node execution"""
    executionId: str
    inputs: Dict[str, Any]
    options: Optional[Dict[str, Any]] = None


class ExecutionResponse(BaseModel):
    """Response model for successful execution"""
    executionId: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    online: bool
    nodes: Dict[str, Any]
    timestamp: str


class NodeListResponse(BaseModel):
    """Response for node list endpoint"""
    nodes: Dict[str, Dict[str, Any]]


# Global state
active_executions: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting OpenEvolve Node API Server")
    try:
        logger.info(f"Registered nodes: {list(NodeRegistry.list_nodes().keys())}")

        # Log node info
        for node_type, node_class in NodeRegistry.list_nodes().items():
            try:
                node_info = NodeRegistry.get_node_info(node_type)
                logger.info(f"  - {node_type}: {node_info['display_name']}")
            except Exception as e:
                logger.warning(f"  - {node_type}: Could not get node info - {e}")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

    yield

    # Shutdown
    logger.info("Shutting down OpenEvolve Node API Server")
    try:
        # Clean up active executions
        active_executions.clear()
        logger.info("Active executions cleared")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")


# Create FastAPI app
app = FastAPI(
    title="OpenEvolve Node API",
    description="REST API for executing OpenEvolve workflow nodes",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Helper class for execution context
class APIExecutionContext:
    """Execution context that mimics BubbleLabs workflow context"""

    def __init__(self, execution_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.execution_id = execution_id
        self.progress = 0
        self.status_message = ""
        self.artifacts = {}
        self.metadata = metadata or {}

    def update_progress(self, progress: int, message: str = ""):
        """Update execution progress"""
        try:
            self.progress = progress
            self.status_message = message
            logger.info(f"[{self.execution_id}] Progress: {progress}% - {message}")

            # Update active execution
            if self.execution_id in active_executions:
                active_executions[self.execution_id]["progress"] = progress
                active_executions[self.execution_id]["status_message"] = message
        except Exception as e:
            logger.error(f"Error updating progress for execution {self.execution_id}: {e}")

    def add_artifact(self, name: str, data: Any):
        """Add execution artifact"""
        try:
            self.artifacts[name] = data
        except Exception as e:
            logger.error(f"Error adding artifact '{name}' for execution {self.execution_id}: {e}")

    def generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        return self.execution_id


# Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "OpenEvolve Node API Server",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime

    try:
        # Get info about all registered nodes
        nodes_info = {}
        try:
            node_types = NodeRegistry.list_nodes().keys()
        except Exception as e:
            logger.error(f"Could not list nodes for health check: {e}")
            node_types = []

        for node_type in node_types:
            try:
                node_info = NodeRegistry.get_node_info(node_type)
                nodes_info[node_type] = {
                    "status": "available",
                    "display_name": node_info["display_name"],
                    "description": node_info["description"],
                    "version": node_info["version"],
                    "category": node_info["category"]
                }
            except Exception as e:
                nodes_info[node_type] = {
                    "status": "error",
                    "error": str(e)
                }

        return HealthResponse(
            status="healthy" if len(nodes_info) > 0 else "unhealthy",
            online=True,
            nodes=nodes_info,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            online=False,
            nodes={},
            timestamp=datetime.utcnow().isoformat()
        )


@app.get("/api/integrations", response_model=NodeListResponse)
async def list_integrations():
    """List all available integrations (nodes)"""
    try:
        nodes_info = NodeRegistry.list_all_info()
        return NodeListResponse(nodes=nodes_info)
    except Exception as e:
        logger.error(f"Failed to list integrations: {e}")
        return NodeListResponse(nodes={})


@app.get("/api/integrations/{node_type}")
async def get_integration_info(node_type: str):
    """Get information about a specific integration"""
    try:
        node_info = NodeRegistry.get_node_info(node_type)
        return {
            "success": True,
            "data": node_info
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get integration info for {node_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error retrieving node info: {str(e)}")


@app.post("/api/integrations/{node_type}/execute", response_model=ExecutionResponse)
async def execute_node(node_type: str, request: ExecutionRequest):
    """
    Execute a node

    This endpoint is called by the integration library to execute OpenEvolve nodes.
    The inputs are passed to the Python node implementation, and the result is returned.
    """
    from .base_node import NodeExecutionError

    logger.info(f"Executing node: {node_type} (executionId: {request.executionId})")

    # Initialize active execution
    active_executions[request.executionId] = {
        "node_type": node_type,
        "status": "running",
        "progress": 0,
        "status_message": "Initializing",
        "start_time": None
    }

    try:
        # Get node instance
        try:
            node = get_node(node_type, request.options.get("config") if request.options else None)
        except ValueError as e:
            # Node not found
            logger.error(f"Node not found: {node_type} - {str(e)}")
            active_executions[request.executionId]["status"] = "failed"
            active_executions[request.executionId]["error"] = str(e)

            raise HTTPException(
                status_code=404,
                detail=f"Node type '{node_type}' not found: {str(e)}"
            )

        # Create execution context
        metadata = request.options.get("metadata") if request.options else None
        context = APIExecutionContext(request.executionId, metadata=metadata)

        # Validate inputs
        try:
            validation_errors = node.validate_inputs(request.inputs)
            if validation_errors:
                logger.warning(f"Validation errors for {node_type}: {validation_errors}")
                # Continue anyway - nodes may have their own error handling
        except Exception as e:
            logger.error(f"Input validation failed unexpectedly: {e}", exc_info=True)
            # Continue anyway - nodes may have their own error handling

        # Execute node
        logger.info(f"Executing {node_type} with inputs: {list(request.inputs.keys())}")

        # Use the timeout execution method if available, otherwise use safe execution
        timeout_seconds = request.options.get('timeout', 600) if request.options else 600  # Default to 10 minutes
        if hasattr(node, 'execute_with_timeout'):
            result = node.execute_with_timeout(request.inputs, context, timeout_seconds=timeout_seconds)
        elif hasattr(node, 'execute_safe'):
            result = node.execute_safe(request.inputs, context)
        else:
            result = node.execute(request.inputs, context)

        # Update active execution
        active_executions[request.executionId]["status"] = "completed"
        active_executions[request.executionId]["progress"] = 100
        active_executions[request.executionId]["status_message"] = "Completed"

        logger.info(f"Execution complete: {request.executionId}")

        return ExecutionResponse(
            executionId=request.executionId,
            status="success",
            result=result,
            metadata={
                "node_type": node_type,
                "artifacts": context.artifacts,
                "progress": context.progress,
                "context_metadata": context.metadata
            }
        )

    except ValueError as e:
        # Node not found
        logger.error(f"Node not found: {node_type} - {str(e)}")
        active_executions[request.executionId]["status"] = "failed"
        active_executions[request.executionId]["error"] = str(e)

        raise HTTPException(
            status_code=404,
            detail=f"Node type '{node_type}' not found: {str(e)}"
        )

    except NodeExecutionError as e:
        # Node execution error
        logger.error(f"Node execution error: {str(e)}", exc_info=True)
        active_executions[request.executionId]["status"] = "failed"
        active_executions[request.executionId]["error"] = str(e)

        return ExecutionResponse(
            executionId=request.executionId,
            status="error",
            error=str(e),
            metadata={
                "node_type": node_type,
                "error_details": e.details if hasattr(e, 'details') else {}
            }
        )

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error executing {node_type}: {str(e)}", exc_info=True)
        active_executions[request.executionId]["status"] = "failed"
        active_executions[request.executionId]["error"] = str(e)

        return ExecutionResponse(
            executionId=request.executionId,
            status="error",
            error=f"Unexpected error: {str(e)}",
            metadata={
                "node_type": node_type,
                "exception_type": type(e).__name__
            }
        )

    finally:
        # Clean up old executions (keep last 100)
        try:
            if len(active_executions) > 100:
                oldest = sorted(active_executions.keys())[0]
                del active_executions[oldest]
        except Exception as e:
            logger.error(f"Error during active executions cleanup: {e}")


@app.get("/api/integrations/{node_type}/schema")
async def get_node_schema(node_type: str):
    """Get parameter schema for a node"""
    try:
        node = get_node(node_type)
        schema = node.get_parameter_schema()
        return {
            "success": True,
            "node_type": node_type,
            "schema": schema
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get schema for {node_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error retrieving schema: {str(e)}")


@app.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get status of an active or recent execution"""
    try:
        if execution_id not in active_executions:
            raise HTTPException(
                status_code=404,
                detail=f"Execution {execution_id} not found"
            )

        return active_executions[execution_id]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get execution status for {execution_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error retrieving execution status: {str(e)}")


@app.get("/executions")
async def list_executions(limit: int = 50):
    """List recent executions"""
    try:
        executions = list(active_executions.values())
        # Sort by start time (most recent first)
        try:
            executions = sorted(
                executions,
                key=lambda x: x.get("start_time", ""),
                reverse=True
            )
        except Exception:
            # If sorting fails, return as is
            pass
        return executions[:limit]
    except Exception as e:
        logger.error(f"Failed to list executions: {e}")
        return []


def start_server(host: str = "127.0.0.1", port: int = 8000, log_level: str = "info"):
    """Start the API server"""
    try:
        logger.info(f"Starting OpenEvolve Node API Server on {host}:{port}")
        uvicorn.run(
            "bubblelabs_nodes.api_server:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=True,  # Enable auto-reload for development
            timeout_keep_alive=30,  # Prevent hanging connections
            timeout_graceful_shutdown=10,  # Timeout for graceful shutdown
            loop="asyncio"  # Explicitly specify event loop
        )
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down gracefully...")
    except SystemExit:
        logger.info("System exit signal received, shutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    start_server()
