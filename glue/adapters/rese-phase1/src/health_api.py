"""
RESE Phase I Health Check API

FastAPI health check endpoints for Phase I Epistemic Audit Executor.

Endpoints:
- GET /health - Liveness check (returns 200 if alive)
- GET /ready - Readiness check (executor can initialize)
- GET /metrics - Circuit breaker status, DLQ size, last execution time

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON with correlation_id
- UTC timestamps: All temporal data in UTC
- Error Handling: Graceful degradation

Author: RESE Team
Created: 2026-02-04
Phase: I - Epistemic Audit
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
    from phase1_executor import (
        EpistemicAuditExecutor,
        Phase1Config,
        StructuredLogger,
        CircuitBreaker,
        DeadLetterQueue,
    )
except ImportError as e:
    print(f"FATAL: Cannot import Phase I executor: {e}")
    sys.exit(1)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RESE Phase I Health API",
    description="Health check endpoints for Phase I Epistemic Audit Executor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# GLOBAL STATE (Singleton executor instance)
# ============================================================================

_executor: Optional[EpistemicAuditExecutor] = None
_initialization_error: Optional[str] = None
_startup_time: datetime = datetime.now(timezone.utc)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_executor() -> EpistemicAuditExecutor:
    """
    Get or create executor instance.

    Returns:
        EpistemicAuditExecutor instance

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
            config = Phase1Config.from_env()
            _executor = EpistemicAuditExecutor(config=config)
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
        "phase": "phase1_epistemic_audit",
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
    - Dependencies are available

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
    except HTTPException as e:
        checks["executor"] = "fail"
        checks["configuration"] = "unknown"
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
        - Dead letter queue size
        - Executor statistics
    """
    checks = {}

    try:
        executor = get_executor()

        # Circuit breaker stats
        cb_stats = executor.circuit_breaker.get_stats()
        checks["circuit_breaker"] = {
            "state": cb_stats["state"],
            "failure_count": cb_stats["failure_count"],
            "last_failure_time": cb_stats["last_failure_time"],
        }

        # DLQ stats
        checks["dlq_size"] = executor.dlq.size()

        # Executor stats
        stats = executor.get_stats()
        checks.update(stats)

        # Configuration summary
        checks["config"] = {
            "max_assumptions": executor.config.MAX_ASSUMPTIONS,
            "max_constraints": executor.config.MAX_CONSTRAINTS,
            "timeout_ms": executor.config.TIMEOUT_MS,
            "enable_tacit_mining": executor.config.ENABLE_TACIT_MINING,
            "enable_red_team": executor.config.ENABLE_RED_TEAM_PROTOCOL,
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
        "name": "RESE Phase I Health API",
        "version": "1.0.0",
        "phase": "phase1_epistemic_audit",
        "description": "Health check endpoints for Phase I Epistemic Audit Executor",
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
            "phase": "phase1_epistemic_audit",
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
        PHASE1_HEALTH_PORT: Port to listen on (default: 8001)
        PHASE1_HEALTH_HOST: Host to bind to (default: 0.0.0.0)
    """
    port = int(os.getenv("PHASE1_HEALTH_PORT", "8001"))
    host = os.getenv("PHASE1_HEALTH_HOST", "0.0.0.0")

    print(f"Starting RESE Phase I Health API on {host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
