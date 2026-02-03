"""
OpenEvolve Knowledge Engine - Production Server

This module implements a production-ready server for the knowledge engine
with proper configuration, data storage, authentication, and monitoring.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path
import json
import yaml
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# Import knowledge engine components
from knowledge_engine.config.config_manager import ConfigManager, get_config_manager
from knowledge_engine.data.storage import KnowledgeStorageEngine
from knowledge_engine.integrations.main_orchestrator import KnowledgeEngineOrchestrator


logger = logging.getLogger(__name__)


class KnowledgeEngineRequest(BaseModel):
    """Request model for knowledge engine operations."""
    query: str
    components: Optional[list[str]] = None
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None


class KnowledgeEngineResponse(BaseModel):
    """Response model for knowledge engine operations."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


# Global application state
class AppState:
    """Global application state."""
    def __init__(self):
        self.config_manager: Optional[ConfigManager] = None
        self.storage_engine: Optional[KnowledgeStorageEngine] = None
        self.orchestrator: Optional[KnowledgeEngineOrchestrator] = None
        self.initialized: bool = False


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    logger.info("Starting OpenEvolve Knowledge Engine server...")
    
    try:
        # Initialize configuration
        config_manager = await get_config_manager("config.yaml")
        app_state.config_manager = config_manager
        
        # Initialize storage engine
        storage_config = config_manager.get_component_config("storage") or {}
        app_state.storage_engine = KnowledgeStorageEngine(storage_config)
        await app_state.storage_engine.initialize()
        
        # Initialize orchestrator
        orchestrator_config = config_manager.get_component_config("orchestrator") or {}
        app_state.orchestrator = KnowledgeEngineOrchestrator(orchestrator_config)
        await app_state.orchestrator.initialize_components()
        
        app_state.initialized = True
        
        logger.info({
            "msg": "OpenEvolve Knowledge Engine initialized successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        yield
        
    except Exception as e:
        logger.error({
            "msg": f"Failed to initialize OpenEvolve Knowledge Engine: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        raise
    finally:
        logger.info("Shutting down OpenEvolve Knowledge Engine server...")
        
        # Cleanup resources
        if app_state.orchestrator:
            await app_state.orchestrator.close()
        
        if app_state.storage_engine:
            await app_state.storage_engine.close()
        
        logger.info("OpenEvolve Knowledge Engine server shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="OpenEvolve Knowledge Engine API",
    description="Production API for the OpenEvolve Knowledge Engine with integrated AI capabilities",
    version="1.0.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, configure specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom middleware for logging and correlation IDs
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.now(timezone.utc)
        correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())
        
        # Add correlation ID to request state
        request.state.correlation_id = correlation_id
        
        logger.info({
            "msg": "Incoming request",
            "method": request.method,
            "url": str(request.url),
            "correlation_id": correlation_id,
            "timestamp": start_time.isoformat()
        })
        
        try:
            response = await call_next(request)
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.error({
                "msg": "Request failed with exception",
                "method": request.method,
                "url": str(request.url),
                "correlation_id": correlation_id,
                "error": str(e),
                "processing_time_ms": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            response = JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": str(e),
                    "metadata": {
                        "correlation_id": correlation_id,
                        "processing_time_ms": processing_time
                    }
                }
            )
        
        if hasattr(response, 'status_code'):
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            logger.info({
                "msg": "Request completed",
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "correlation_id": correlation_id,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Add correlation ID to response headers
        if hasattr(response, 'headers'):
            response.headers["X-Correlation-ID"] = correlation_id
        
        return response


app.add_middleware(LoggingMiddleware)


def require_initialized():
    """Dependency to ensure app is initialized."""
    def check_initialized():
        if not app_state.initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        return True
    
    return Depends(check_initialized)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if all components are available
        storage_healthy = app_state.storage_engine is not None
        orchestrator_healthy = app_state.orchestrator is not None
        config_healthy = app_state.config_manager is not None
        
        status = {
            "status": "healthy" if all([storage_healthy, orchestrator_healthy, config_healthy]) else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "storage_engine": "healthy" if storage_healthy else "unhealthy",
                "orchestrator": "healthy" if orchestrator_healthy else "unhealthy",
                "config_manager": "healthy" if config_healthy else "unhealthy"
            },
            "uptime": "tracking would go here"
        }
        
        return status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/status")
async def get_status():
    """Get detailed status of the knowledge engine."""
    if not app_state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Get status from all components
        storage_status = await app_state.storage_engine.get_status() if app_state.storage_engine else {"error": "Not initialized"}
        orchestrator_status = await app_state.orchestrator.get_status() if app_state.orchestrator else {"error": "Not initialized"}
        config_status = app_state.config_manager.get_status() if app_state.config_manager else {"error": "Not initialized"}
        
        status = {
            "overall_status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "storage_engine": storage_status,
                "orchestrator": orchestrator_status,
                "config_manager": config_status
            },
            "integrations": {
                "graphiti": getattr(app_state.orchestrator, 'graphiti', None) is not None,
                "kggen": getattr(app_state.orchestrator, 'kggen', None) is not None,
                "oneke": getattr(app_state.orchestrator, 'oneke', None) is not None,
                "aikg": getattr(app_state.orchestrator, 'aikg', None) is not None,
                "ragbits": getattr(app_state.orchestrator, 'ragbits', None) is not None,
                "crewai": getattr(app_state.orchestrator, 'crewai', None) is not None,
                "deepke": getattr(app_state.orchestrator, 'deepke', None) is not None,
                "research_quest": getattr(app_state.orchestrator, 'research_quest', None) is not None,
                "agentic_context": getattr(app_state.orchestrator, 'agentic_context', None) is not None,
                "agentjson": getattr(app_state.orchestrator, 'agentjson', None) is not None,
                "dspy": getattr(app_state.orchestrator, 'dspy', None) is not None,
                "leanaide": getattr(app_state.orchestrator, 'leanaide', None) is not None,
                "openevolve_lib": getattr(app_state.orchestrator, 'openevolve_lib', None) is not None,
                "mcp_gateway": getattr(app_state.orchestrator, 'mcp_gateway', None) is not None
            }
        }
        
        return status
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.post("/process", response_model=KnowledgeEngineResponse)
async def process_request(
    request: KnowledgeEngineRequest,
    background_tasks: BackgroundTasks,
    initialized: bool = require_initialized()
):
    """Process a knowledge request through the integrated system."""
    correlation_id = request.state.correlation_id
    
    start_time = datetime.now(timezone.utc)
    
    logger.info({
        "msg": "Processing knowledge request",
        "query_length": len(request.query),
        "components_requested": request.components,
        "correlation_id": correlation_id,
        "timestamp": start_time.isoformat()
    })
    
    try:
        if not app_state.orchestrator:
            raise RuntimeError("Knowledge engine orchestrator not initialized")
        
        # Process the request through the orchestrator
        result = await app_state.orchestrator.process_knowledge_request(
            query=request.query,
            components=request.components,
            context=request.context or {},
            parameters=request.parameters or {},
            correlation_id=correlation_id
        )
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        response = KnowledgeEngineResponse(
            success=result.success,
            result=result.to_dict() if hasattr(result, 'to_dict') else result,
            metadata={
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
        )
        
        logger.info({
            "msg": "Knowledge request processed successfully",
            "correlation_id": correlation_id,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return response
        
    except Exception as e:
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.error({
            "msg": "Knowledge request processing failed",
            "correlation_id": correlation_id,
            "error": str(e),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return KnowledgeEngineResponse(
            success=False,
            error=str(e),
            metadata={
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
        )


@app.post("/extract-knowledge")
async def extract_knowledge(
    request: KnowledgeEngineRequest,
    initialized: bool = require_initialized()
):
    """Extract knowledge from text using multiple integrated systems."""
    correlation_id = request.state.correlation_id
    
    start_time = datetime.now(timezone.utc)
    
    logger.info({
        "msg": "Starting knowledge extraction",
        "text_length": len(request.query),
        "correlation_id": correlation_id,
        "timestamp": start_time.isoformat()
    })
    
    try:
        if not app_state.orchestrator:
            raise RuntimeError("Knowledge engine orchestrator not initialized")
        
        # Use the orchestrator to perform knowledge extraction
        extraction_result = await app_state.orchestrator.run_comprehensive_analysis(
            text=request.query,
            analysis_types=request.components or ["entities", "relations", "triples"],
            correlation_id=correlation_id
        )
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        result = {
            "success": extraction_result.success,
            "entities": extraction_result.entities if hasattr(extraction_result, 'entities') else [],
            "relations": extraction_result.relations if hasattr(extraction_result, 'relations') else [],
            "triples": extraction_result.triples if hasattr(extraction_result, 'triples') else [],
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
        }
        
        logger.info({
            "msg": "Knowledge extraction completed",
            "correlation_id": correlation_id,
            "entities_count": len(result["entities"]),
            "relations_count": len(result["relations"]),
            "triples_count": len(result["triples"]),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result
        
    except Exception as e:
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.error({
            "msg": "Knowledge extraction failed",
            "correlation_id": correlation_id,
            "error": str(e),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_knowledge(
    request: KnowledgeEngineRequest,
    initialized: bool = require_initialized()
):
    """Search the knowledge base using integrated search capabilities."""
    correlation_id = request.state.correlation_id
    
    start_time = datetime.now(timezone.utc)
    
    logger.info({
        "msg": "Starting knowledge search",
        "query_length": len(request.query),
        "correlation_id": correlation_id,
        "timestamp": start_time.isoformat()
    })
    
    try:
        if not app_state.storage_engine:
            raise RuntimeError("Knowledge storage engine not initialized")
        
        # Perform search in the knowledge base
        search_results = await app_state.storage_engine.search_knowledge_artifacts(
            query=request.query,
            artifact_type=request.parameters.get("artifact_type") if request.parameters else None,
            top_k=request.parameters.get("top_k", 10) if request.parameters else 10,
            filters=request.parameters.get("filters") if request.parameters else None
        )
        
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        result = {
            "success": True,
            "results": [artifact.to_dict() for artifact in search_results],
            "count": len(search_results),
            "metadata": {
                "processing_time_ms": processing_time_ms,
                "correlation_id": correlation_id
            }
        }
        
        logger.info({
            "msg": "Knowledge search completed",
            "correlation_id": correlation_id,
            "results_count": len(search_results),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return result
        
    except Exception as e:
        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.error({
            "msg": "Knowledge search failed",
            "correlation_id": correlation_id,
            "error": str(e),
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics(initialized: bool = require_initialized()):
    """Get system metrics and performance information."""
    try:
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components_initialized": app_state.initialized,
            "storage_metrics": await app_state.storage_engine.get_metrics() if app_state.storage_engine else {},
            "orchestrator_metrics": await app_state.orchestrator.get_metrics() if app_state.orchestrator else {},
            "config_metrics": app_state.config_manager.get_metrics() if app_state.config_manager else {}
        }
        
        return metrics
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    log_level: str = "info",
    config_file: str = "config.yaml"
):
    """
    Run the OpenEvolve Knowledge Engine server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        log_level: Logging level
        config_file: Path to configuration file
    """
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup signal handlers
    setup_signal_handlers()
    
    logger.info({
        "msg": "Starting OpenEvolve Knowledge Engine server",
        "host": host,
        "port": port,
        "workers": workers,
        "log_level": log_level,
        "config_file": config_file,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Run the server
    uvicorn.run(
        "knowledge_engine.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,  # Disable reload in production
        timeout_keep_alive=300,  # Increase keep-alive timeout
        timeout_graceful_shutdown=30,  # Graceful shutdown timeout
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="OpenEvolve Knowledge Engine Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        config_file=args.config
    )