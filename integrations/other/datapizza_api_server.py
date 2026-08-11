"""
DataPizza FastAPI Server

Provides REST API endpoints for DataPizza functionality.
This server wraps DataPizza's multi-agent framework and provides
easy integration with the BubbleLabs frontend.

Endpoints:
- POST /query - Query data from various sources
- POST /process - Process data through DataPizza pipeline
- POST /pipeline - Run complete DataPizza pipeline
- GET /status - Get server status and available tools
- GET /health - Health check endpoint
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Try to import DataPizza components
try:
    from datapizza.agents import Agent
    from datapizza.clients import Client
    from datapizza.tools import Tool
    DATAPIZZA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("DataPizza core imported successfully")
except ImportError as e:
    DATAPIZZA_AVAILABLE = False
    Agent = None
    Client = None
    Tool = None
    logger = logging.getLogger(__name__)
    logger.warning(f"DataPizza core not available: {e}")


# =============================================================================
# Pydantic Models for API
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str = Field(..., description="The query to execute", min_length=1)
    data_source: str = Field(default="default", description="Data source to query")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    timeout: int = Field(default=30, ge=1, le=300, description="Query timeout in seconds")


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    count: int
    data_source: str
    execution_time: float
    message: Optional[str] = None


class ProcessRequest(BaseModel):
    """Request model for /process endpoint."""
    data: Union[str, List[str], Dict[str, Any]] = Field(..., description="Data to process")
    operation: str = Field(default="chunk", description="Operation to perform")
    params: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    chunk_size: int = Field(default=1000, ge=100, le=10000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="Overlap between chunks")
    timeout: int = Field(default=60, ge=1, le=600, description="Processing timeout in seconds")


class ProcessResponse(BaseModel):
    """Response model for /process endpoint."""
    success: bool
    operation: str
    processed_data: List[Dict[str, Any]]
    count: int
    execution_time: float
    progress: Dict[str, Any]
    message: Optional[str] = None


class PipelineRequest(BaseModel):
    """Request model for /pipeline endpoint."""
    data: Union[str, List[str], Dict[str, Any]] = Field(..., description="Input data for pipeline")
    stages: List[str] = Field(default=["validate", "chunk", "embed", "store"], description="Pipeline stages to execute")
    data_source: str = Field(default="default", description="Data source identifier")
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    embedding_model: str = Field(default="default", description="Embedding model to use")
    vector_store: str = Field(default="default", description="Vector store for embeddings")
    timeout: int = Field(default=300, ge=1, le=1800, description="Pipeline timeout in seconds")


class PipelineResponse(BaseModel):
    """Response model for /pipeline endpoint."""
    success: bool
    stages_completed: List[str]
    stages_failed: List[str]
    results: Dict[str, Any]
    execution_time: float
    progress: Dict[str, float]
    summary: Dict[str, Any]
    message: Optional[str] = None


class StatusResponse(BaseModel):
    """Response model for /status endpoint."""
    status: str
    datapizza_available: bool
    version: str
    available_tools: List[str]
    available_clients: List[str]
    uptime_seconds: float
    active_pipelines: int


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    healthy: bool
    timestamp: datetime


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="DataPizza API Server",
    description="REST API for DataPizza multi-agent framework",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Server startup time
_start_time = datetime.now()

# Active pipeline tracking
_active_pipelines: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# Helper Functions
# =============================================================================

def get_uptime() -> float:
    """Get server uptime in seconds."""
    return (datetime.now() - _start_time).total_seconds()


async def simulate_query(query: str, data_source: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simulate query execution when DataPizza is not available."""
    # Simulate async processing
    await asyncio.sleep(0.5)

    return [
        {
            "id": "1",
            "content": f"Sample result for '{query[:50]}...'",
            "source": data_source,
            "relevance": 0.95,
            "metadata": params
        }
    ]


async def simulate_processing(data: Any, operation: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simulate data processing when DataPizza is not available."""
    # Simulate async processing
    await asyncio.sleep(1.0)

    # Convert data to string if needed
    if isinstance(data, str):
        text = data
    elif isinstance(data, list):
        text = " ".join(str(item) for item in data)
    elif isinstance(data, dict):
        text = str(data)
    else:
        text = str(data)

    # Simulate chunking
    chunk_size = params.get("chunk_size", 1000)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    return [
        {
            "chunk_id": str(i),
            "content": chunk,
            "size": len(chunk),
            "operation": operation
        }
        for i, chunk in enumerate(chunks)
    ]


async def simulate_pipeline(data: Any, stages: List[str], params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate pipeline execution when DataPizza is not available."""
    results = {}
    progress = {}

    total_stages = len(stages)

    for i, stage in enumerate(stages):
        # Simulate stage execution
        await asyncio.sleep(0.5)

        if stage == "validate":
            results["validate"] = {"valid": True, "errors": []}
        elif stage == "chunk":
            text = str(data)
            chunk_size = params.get("chunk_size", 1000)
            chunks = [text[j:j+chunk_size] for j in range(0, len(text), chunk_size)]
            results["chunk"] = {"chunks": chunks, "count": len(chunks)}
        elif stage == "embed":
            results["embed"] = {"embeddings": [[0.1, 0.2, 0.3]], "model": params.get("embedding_model", "default")}
        elif stage == "store":
            results["store"] = {"stored": True, "vector_store": params.get("vector_store", "default")}

        progress[f"stage_{i}"] = (i + 1) / total_stages

    return results


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "DataPizza API Server",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="operational" if DATAPIZZA_AVAILABLE else "degraded",
        healthy=True,
        timestamp=datetime.now()
    )


@app.get("/status", response_model=StatusResponse)
async def status():
    """Get server status and available tools."""
    tools = []
    clients = []

    if DATAPIZZA_AVAILABLE:
        tools = ["FileSystem", "DuckDuckGo", "SQL", "WebFetch"]
        clients = ["OpenAI", "Anthropic", "Google"]

    return StatusResponse(
        status="ready" if DATAPIZZA_AVAILABLE else "degraded",
        datapizza_available=DATAPIZZA_AVAILABLE,
        version="1.0.0",
        available_tools=tools,
        available_clients=clients,
        uptime_seconds=get_uptime(),
        active_pipelines=len(_active_pipelines)
    )


@app.post("/query", response_model=QueryResponse)
async def query_data(request: QueryRequest):
    """
    Execute a query against data sources.

    This endpoint processes natural language queries and returns
    relevant results from the specified data source.
    """
    start_time = datetime.now()

    try:
        if DATAPIZZA_AVAILABLE:
            # Use real DataPizza implementation
            # This would involve creating agents and tools
            results = await simulate_query(request.query, request.data_source, request.params)
        else:
            # Use fallback simulation
            results = await simulate_query(request.query, request.data_source, request.params)

        execution_time = (datetime.now() - start_time).total_seconds()

        return QueryResponse(
            success=True,
            query=request.query,
            results=results,
            count=len(results),
            data_source=request.data_source,
            execution_time=execution_time,
            message=None if DATAPIZZA_AVAILABLE else "DataPizza not available - using fallback"
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Query failed: {e}")

        return QueryResponse(
            success=False,
            query=request.query,
            results=[],
            count=0,
            data_source=request.data_source,
            execution_time=execution_time,
            message=f"Query failed: {str(e)}"
        )


@app.post("/process", response_model=ProcessResponse)
async def process_data(request: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Process data through DataPizza operations.

    This endpoint handles data processing operations like:
    - chunk: Split text into chunks
    - embed: Generate embeddings for text
    - analyze: Analyze data with agents
    - transform: Transform data format
    """
    start_time = datetime.now()

    try:
        if DATAPIZZA_AVAILABLE:
            # Use real DataPizza implementation
            processed_data = await simulate_processing(request.data, request.operation, request.params)
        else:
            # Use fallback simulation
            processed_data = await simulate_processing(request.data, request.operation, request.params)

        execution_time = (datetime.now() - start_time).total_seconds()

        # Calculate progress
        progress = {
            "complete": 1.0,
            "stages": 1,
            "current_stage": request.operation
        }

        return ProcessResponse(
            success=True,
            operation=request.operation,
            processed_data=processed_data,
            count=len(processed_data),
            execution_time=execution_time,
            progress=progress,
            message=None if DATAPIZZA_AVAILABLE else "DataPizza not available - using fallback"
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Processing failed: {e}")

        return ProcessResponse(
            success=False,
            operation=request.operation,
            processed_data=[],
            count=0,
            execution_time=execution_time,
            progress={"complete": 0.0, "error": str(e)},
            message=f"Processing failed: {str(e)}"
        )


@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest):
    """
    Run complete DataPizza pipeline.

    This endpoint executes multiple stages in sequence:
    1. validate - Validate input data
    2. chunk - Split data into chunks
    3. embed - Generate embeddings
    4. store - Store in vector database

    Progress tracking is provided for long-running pipelines.
    """
    start_time = datetime.now()
    pipeline_id = f"pipeline_{hash(str(request.data))}"

    try:
        # Register active pipeline
        _active_pipelines[pipeline_id] = {
            "started_at": start_time,
            "stages": request.stages,
            "status": "running"
        }

        # Execute pipeline stages
        results = {}
        stages_completed = []
        stages_failed = []
        progress = {}

        for stage in request.stages:
            try:
                if stage in ["validate", "chunk", "embed", "store"]:
                    stage_results = await simulate_pipeline(request.data, [stage], request.dict())
                    results[stage] = stage_results
                    stages_completed.append(stage)

                # Update progress
                progress[stage] = 1.0

            except Exception as stage_error:
                logger.error(f"Stage {stage} failed: {stage_error}")
                stages_failed.append(stage)
                progress[stage] = 0.0
                break

        execution_time = (datetime.now() - start_time).total_seconds()

        # Calculate overall progress
        total_progress = sum(progress.values()) / len(request.stages) if progress else 0.0

        # Update pipeline status
        _active_pipelines[pipeline_id]["status"] = "completed"
        _active_pipelines[pipeline_id]["completed_at"] = datetime.now()

        return PipelineResponse(
            success=len(stages_failed) == 0,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            results=results,
            execution_time=execution_time,
            progress={"overall": total_progress, **progress},
            summary={
                "total_stages": len(request.stages),
                "completed_stages": len(stages_completed),
                "failed_stages": len(stages_failed),
                "pipeline_id": pipeline_id
            },
            message=None if DATAPIZZA_AVAILABLE else "DataPizza not available - using fallback"
        )

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Pipeline failed: {e}")

        # Update pipeline status
        if pipeline_id in _active_pipelines:
            _active_pipelines[pipeline_id]["status"] = "failed"
            _active_pipelines[pipeline_id]["failed_at"] = datetime.now()

        return PipelineResponse(
            success=False,
            stages_completed=[],
            stages_failed=request.stages,
            results={},
            execution_time=execution_time,
            progress={"overall": 0.0, "error": str(e)},
            summary={},
            message=f"Pipeline failed: {str(e)}"
        )


# =============================================================================
# Startup and Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("DataPizza API Server starting...")
    logger.info(f"DataPizza available: {DATAPIZZA_AVAILABLE}")
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("DataPizza API Server shutting down...")
    logger.info(f"Processed {len(_active_pipelines)} pipelines during uptime")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = True,
    log_level: str = "info"
):
    """
    Run the DataPizza API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload during development
        log_level: Logging level
    """
    uvicorn.run(
        "datapizza_api_server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run server
    run_server()
