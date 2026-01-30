"""
FastAPI Gateway Server for Unified MCP Gateway.

This module provides the HTTP API server that exposes the unified gateway
to clients, including CREWAI agents and ROMA.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .unified_mcp_gateway import UnifiedMCPGateway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global gateway instance
gateway: Optional[UnifiedMCPGateway] = None


# Request/Response Models
class ToolCallRequest(BaseModel):
    """Request model for tool calls."""
    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolCallResponse(BaseModel):
    """Response model for tool calls."""
    success: bool
    tool_name: str
    namespace: str
    server_name: str
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: str


class ToolsListResponse(BaseModel):
    """Response model for tools list."""
    tools: List[Dict[str, Any]]
    total_count: int
    namespaces: List[str]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    initialized: bool
    servers: Dict[str, Dict[str, Any]]
    tools: Dict[str, int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI app.

    Handles startup and shutdown events.
    """
    # Startup
    global gateway
    logger.info("Starting Unified MCP Gateway server...")

    try:
        gateway = UnifiedMCPGateway()
        await gateway.initialize()
        gateway.is_running = True
        logger.info("Gateway server started successfully")
    except Exception as e:
        logger.error(f"Failed to start gateway: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Unified MCP Gateway server...")
    if gateway:
        await gateway.shutdown()
    logger.info("Gateway server shut down")


# Create FastAPI app
app = FastAPI(
    title="Unified MCP Gateway",
    description="Gateway for coordinating tools from multiple MCP servers",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Unified MCP Gateway",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the gateway and all connected servers.
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    health_status = await gateway.get_health_status()

    return HealthResponse(
        status=health_status["gateway"]["status"],
        initialized=health_status["gateway"]["initialized"],
        servers=health_status["servers"],
        tools=health_status["tools"],
    )


@app.get("/api/tools", response_model=ToolsListResponse, tags=["Tools"])
async def list_tools(
    namespace: str = "",
    category: Optional[str] = None,
):
    """
    List all available tools.

    Args:
        namespace: Optional namespace filter (e.g., "kggen", "graphiti")
        category: Optional category filter (e.g., "knowledge", "evolution")

    Returns:
        List of available tools
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    tools = await gateway.list_tools(namespace=namespace, category=category)

    # Get unique namespaces
    namespaces = set()
    for tool in tools:
        ns = tool.get("namespace", "")
        if ns:
            namespaces.add(ns)

    return ToolsListResponse(
        tools=tools,
        total_count=len(tools),
        namespaces=sorted(list(namespaces)),
    )


@app.post("/api/tools/{tool_name}", response_model=ToolCallResponse, tags=["Tools"])
async def call_tool(
    tool_name: str,
    request: ToolCallRequest,
):
    """
    Execute a tool call.

    Args:
        tool_name: Name of the tool to call (can include namespace prefix)
        request: ToolCallRequest with parameters

    Returns:
        ToolCallResponse with execution result
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    try:
        # Call the tool
        result = await gateway.call_tool(
            tool_name=request.tool_name or tool_name,
            params=request.parameters,
        )

        return ToolCallResponse(
            success=result.success,
            tool_name=result.tool_name,
            namespace=result.namespace,
            server_name=result.server_name,
            result=result.result,
            error=result.error,
            execution_time=result.execution_time,
            timestamp=result.timestamp.isoformat(),
        )

    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tools/{tool_name}", tags=["Tools"])
async def get_tool_info(tool_name: str):
    """
    Get information about a specific tool.

    Args:
        tool_name: Name of the tool (can include namespace prefix)

    Returns:
        Tool information
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    # Parse tool name
    namespace = None
    name = tool_name
    if "/" in tool_name:
        namespace, name = tool_name.split("/", 1)

    # Get tool from registry
    tool = gateway.tool_registry.get_tool(name, namespace)

    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")

    return tool.to_dict()


@app.get("/api/namespaces", tags=["Tools"])
async def list_namespaces():
    """
    List all available namespaces.

    Returns:
        List of namespace names
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    namespaces = gateway.tool_registry.list_namespaces()

    return {
        "namespaces": namespaces,
        "total_count": len(namespaces),
    }


@app.get("/api/categories", tags=["Tools"])
async def list_categories():
    """
    List all tool categories.

    Returns:
        List of categories with tool counts
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    categories = {}
    for category in gateway.tool_registry.tools_by_category.keys():
        tools = gateway.tool_registry.list_tools_by_category(category)
        categories[category.value] = len(tools)

    return {
        "categories": categories,
    }


@app.get("/api/analytics", tags=["Analytics"])
async def get_analytics():
    """
    Get tool usage analytics.

    Returns:
        Analytics data including popular tools, success rates, etc.
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    return {
        "tool_calls": gateway.tool_call_stats,
        "summary": {
            "total_tools": len(gateway.tool_call_stats),
            "total_calls": sum(
                stats["total_calls"] for stats in gateway.tool_call_stats.values()
            ),
        },
    }


@app.get("/api/stats", tags=["Stats"])
async def get_stats():
    """
    Get gateway statistics.

    Returns:
        Gateway statistics including tool counts, server status, etc.
    """
    if not gateway:
        raise HTTPException(status_code=503, detail="Gateway not initialized")

    health = await gateway.get_health_status()

    return {
        "tools": gateway.tool_registry.get_tool_count(),
        "servers": health["servers"],
        "router": health["router"],
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


def run_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    log_level: str = "info",
):
    """
    Run the gateway server.

    Args:
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level
    """
    uvicorn.run(
        "mcp.gateway.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,  # Set to True for development
    )


if __name__ == "__main__":
    run_server()
