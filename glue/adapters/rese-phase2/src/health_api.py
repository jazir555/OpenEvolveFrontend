"""
RESE Phase II Health Check API

FastAPI health check endpoints for Phase II Isomorphic Mapping Executor.

Endpoints:
- GET /health - Liveness check (returns 200 if alive)
- GET /ready - Readiness check (executor can initialize)
- GET /metrics - Circuit breaker status, I_mech cache size, domain KB size

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON with correlation_id
- UTC timestamps: All temporal data in UTC
- Error Handling: Graceful degradation

Author: RESE Team
Created: 2026-02-04
Phase: II - Isomorphic Mapping
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import uuid
import traceback

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("FATAL: FastAPI or uvicorn not installed")
    print("Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Import executor components
try:
    from phase2_executor import (
        IsomorphicMappingExecutor,
        Phase2Config,
        Phase2Logger,
    )
except ImportError as e:
    print(f"FATAL: Cannot import Phase II executor: {e}")
    sys.exit(1)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RESE Phase II Health API",
    description="Health check endpoints for Phase II Isomorphic Mapping Executor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# GLOBAL STATE (Singleton executor instance)
# ============================================================================

_executor: Optional[IsomorphicMappingExecutor] = None
_initialization_error: Optional[str] = None
_startup_time: datetime = datetime.now(timezone.utc)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_executor() -> IsomorphicMappingExecutor:
    """
    Get or create executor instance.

    Returns:
        IsomorphicMappingExecutor instance

    Raises:
        HTTPException: If executor initialization fails
    """
    global _executor, _initialization_error

    if _initialization_error:
        raise HTTPException(
            status_code=503,
            detail=f"Executor initialization failed: {_initialization_error}"
        )

    if _executor is None:
        try:
            # Load configuration from environment
            config = Phase2Config.from_env()
            _executor = IsomorphicMappingExecutor(config=config)
        except Exception as e:
            _initialization_error = str(e)
            raise HTTPException(
                status_code=503,
                detail=f"Failed to initialize executor: {str(e)}"
            )

    return _executor


def create_response(
    status: str,
    checks: Dict[str, Any],
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized health check response.

    Args:
        status: Health status (healthy/degraded/unhealthy)
        checks: Dictionary of check results
        correlation_id: Optional correlation ID

    Returns:
        Standardized response dictionary
    """
    return {
        "status": status,
        "phase": "phase2_isomorphic_mapping",
        "version": "1.0.0",
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def liveness():
    """
    Liveness check - returns 200 if the service is alive.

    This endpoint checks if the API process is running and responsive.
    It does not check if the executor can initialize.

    Returns:
        JSON response with liveness status
    """
    return create_response(
        status="healthy",
        checks={
            "alive": True,
            "uptime_seconds": (datetime.now(timezone.utc) - _startup_time).total_seconds(),
        }
    )


@app.get("/ready")
async def readiness():
    """
    Readiness check - returns 200 if the executor can initialize.

    This endpoint checks if:
    - Executor can be initialized
    - Configuration is valid
    - Domain KB is loaded

    Returns:
        JSON response with readiness status

    Raises:
        HTTPException: If service is not ready
    """
    checks = {}

    try:
        # Try to get executor
        executor = get_executor()
        checks["executor"] = "pass"
        checks["configuration"] = "valid"

        # Check if domain KB is loaded
        if hasattr(executor.structure_identifier, 'domain_kb'):
            checks["domain_kb_loaded"] = "pass"
            checks["domain_count"] = len(executor.structure_identifier.domain_kb)
        else:
            checks["domain_kb_loaded"] = "unknown"

    except HTTPException as e:
        raise e
    except Exception as e:
        checks["executor"] = "fail"
        checks["configuration"] = "unknown"
        checks["error"] = str(e)
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )

    return create_response(
        status="ready",
        checks=checks
    )


@app.get("/metrics")
async def metrics():
    """
    Metrics endpoint - returns detailed executor metrics.

    Returns:
        JSON response with:
        - Circuit breaker state
        - I_mech threshold
        - Domain KB size
        - Configuration summary
    """
    checks = {}

    try:
        executor = get_executor()

        # Circuit breaker stats
        if hasattr(executor, 'circuit_breaker'):
            cb = executor.circuit_breaker
            checks["circuit_breaker"] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time,
            }
        else:
            checks["circuit_breaker"] = {"status": "not_available"}

        # I_mech threshold
        checks["i_mech_threshold"] = executor.config.i_mech_threshold

        # Domain KB size
        if hasattr(executor.structure_identifier, 'domain_kb'):
            domain_kb = executor.structure_identifier.domain_kb
            checks["domain_kb_size"] = len(domain_kb)
            checks["domains"] = list(domain_kb.keys())

        # Configuration summary
        checks["config"] = {
            "max_target_domains": executor.config.max_target_domains,
            "max_mappings": executor.config.max_mappings,
            "timeout_ms": executor.config.timeout_ms,
            "enable_constraint_inversion": executor.config.enable_constraint_inversion,
        }

    except HTTPException:
        raise
    except Exception as e:
        checks["error"] = str(e)
        checks["traceback"] = traceback.format_exc()

    return create_response(
        status="healthy",
        checks=checks
    )


@app.get("/")
async def root():
    """
    Root endpoint - API information.

    Returns:
        JSON response with API information
    """
    return {
        "name": "RESE Phase II Health API",
        "version": "1.0.0",
        "phase": "phase2_isomorphic_mapping",
        "description": "Health check endpoints for Phase II Isomorphic Mapping Executor",
        "endpoints": {
            "GET /health": "Liveness check",
            "GET /ready": "Readiness check",
            "GET /metrics": "Detailed metrics",
            "GET /docs": "API documentation (Swagger UI)",
            "GET /redoc": "API documentation (ReDoc)",
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
        },
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "phase": "phase2_isomorphic_mapping",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for running the health API server.

    Environment Variables:
        PHASE2_HEALTH_PORT: Port to listen on (default: 8002)
        PHASE2_HEALTH_HOST: Host to bind to (default: 0.0.0.0)
    """
    port = int(os.getenv("PHASE2_HEALTH_PORT", "8002"))
    host = os.getenv("PHASE2_HEALTH_HOST", "0.0.0.0")

    print(f"Starting RESE Phase II Health API on {host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
