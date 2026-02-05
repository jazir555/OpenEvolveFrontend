"""
RESE Phase IV Health Check API

FastAPI health check endpoints for Phase IV Architecture Assembly Executor.

Endpoints:
- GET /health - Liveness check (returns 200 if alive)
- GET /ready - Readiness check (executor can initialize)
- GET /metrics - Circuit breaker status, validation level, knowledge items

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON with correlation_id
- UTC timestamps: All temporal data in UTC
- Error Handling: Graceful degradation

Author: RESE Team
Created: 2026-02-04
Phase: IV - Architecture Assembly
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
    from phase4_executor import (
        ArchitectureAssemblyExecutor,
        Phase4Config,
        StructuredLogger,
    )
except ImportError as e:
    print(f"FATAL: Cannot import Phase IV executor: {e}")
    sys.exit(1)


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RESE Phase IV Health API",
    description="Health check endpoints for Phase IV Architecture Assembly Executor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


# ============================================================================
# GLOBAL STATE (Singleton executor instance)
# ============================================================================

_executor: Optional[ArchitectureAssemblyExecutor] = None
_initialization_error: Optional[str] = None
_startup_time: datetime = datetime.now(timezone.utc)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_executor() -> ArchitectureAssemblyExecutor:
    """
    Get or create executor instance.

    Returns:
        ArchitectureAssemblyExecutor instance

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
            config = Phase4Config.from_env()
            _executor = ArchitectureAssemblyExecutor(config=config)
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
        "phase": "phase4_architecture_assembly",
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
    - Assembly components are ready

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

        # Check if assembly components are initialized
        if hasattr(executor, 'paradigm_assembler'):
            checks["paradigm_assembler"] = "pass"
        if hasattr(executor, 'knowledge_integrator'):
            checks["knowledge_integrator"] = "pass"
        if hasattr(executor, 'architecture_validator'):
            checks["architecture_validator"] = "pass"

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
        - Validation level
        - Assembly configuration
        - Confidence thresholds
    """
    checks = {}

    try:
        executor = get_executor()

        # Circuit breaker stats
        if hasattr(executor.circuit_breaker, 'can_execute'):
            cb = executor.circuit_breaker
            checks["circuit_breaker"] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "can_execute": cb.can_execute(),
            }
        else:
            checks["circuit_breaker"] = {"status": "not_available"}

        # Validation level
        checks["validation_level"] = executor.config.validation_level.value

        # Integration strategy
        checks["integration_strategy"] = executor.config.integration_strategy.value

        # Configuration summary
        checks["config"] = {
            "min_confidence_threshold": executor.config.min_confidence_threshold,
            "max_paradigm_shifts": executor.config.max_paradigm_shifts,
            "assembly_timeout_ms": executor.config.assembly_timeout_ms,
            "enable_cross_validation": executor.config.enable_cross_validation,
            "enable_formal_verification": executor.config.enable_formal_verification,
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
        "name": "RESE Phase IV Health API",
        "version": "1.0.0",
        "phase": "phase4_architecture_assembly",
        "description": "Health check endpoints for Phase IV Architecture Assembly Executor",
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
            "phase": "phase4_architecture_assembly",
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
        PHASE4_HEALTH_PORT: Port to listen on (default: 8004)
        PHASE4_HEALTH_HOST: Host to bind to (default: 0.0.0.0)
    """
    port = int(os.getenv("PHASE4_HEALTH_PORT", "8004"))
    host = os.getenv("PHASE4_HEALTH_HOST", "0.0.0.0")

    print(f"Starting RESE Phase IV Health API on {host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
