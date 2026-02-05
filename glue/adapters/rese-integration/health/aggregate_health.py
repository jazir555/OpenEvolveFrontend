"""
RESE Aggregate Health Checker

Aggregates health status from all RESE phase adapters.

This module provides a unified health check endpoint that:
1. Queries all phase health APIs
2. Aggregates results
3. Returns overall system health

Following CLAUDE.md principles:
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON with correlation_id
- UTC timestamps: All temporal data in UTC
- Timeout Enforcement: All health checks timeout (default 5000ms)
- Graceful Degradation: Partial failures don't break aggregation

Author: RESE Team
Created: 2026-02-04
Phase: Integration - Aggregate Health
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import json
import uuid

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lib"))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    import aiohttp
except ImportError as e:
    print("FATAL: Required packages not installed")
    print(f"Error: {e}")
    print("Install with: pip install fastapi uvicorn aiohttp")
    sys.exit(1)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class PhaseHealth:
    """Health check result for a single phase"""
    phase_name: str
    status: HealthStatus
    uptime_seconds: float
    checks: Dict[str, Any]
    error: Optional[str] = None
    response_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "phase": self.phase_name,
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "checks": self.checks,
            "error": self.error,
            "response_time_ms": self.response_time_ms,
        }


# ============================================================================
# CONFIGURATION
# ============================================================================

class AggregateHealthConfig:
    """Configuration for aggregate health checker"""

    def __init__(self):
        # Phase health API URLs (from env vars with defaults)
        self.phase1_url = os.getenv(
            "PHASE1_HEALTH_URL",
            "http://localhost:8001"
        )
        self.phase2_url = os.getenv(
            "PHASE2_HEALTH_URL",
            "http://localhost:8002"
        )
        self.phase3_url = os.getenv(
            "PHASE3_HEALTH_URL",
            "http://localhost:8003"
        )
        self.phase4_url = os.getenv(
            "PHASE4_HEALTH_URL",
            "http://localhost:8004"
        )

        # Timeout for health checks (milliseconds)
        self.timeout_ms = int(os.getenv("AGGREGATE_HEALTH_TIMEOUT_MS", "5000"))

        # Health check endpoints
        self.liveness_endpoint = "/health"
        self.readiness_endpoint = "/ready"
        self.metrics_endpoint = "/metrics"


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="RESE Aggregate Health API",
    description="Aggregate health check for all RESE phases",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global config
config = AggregateHealthConfig()
_startup_time = datetime.now(timezone.utc)


# ============================================================================
# HEALTH CHECKER
# ============================================================================

class AggregateHealthChecker:
    """
    Aggregates health status from all RESE phases.

    Queries all phase health APIs in parallel and aggregates results.
    """

    def __init__(self, config: AggregateHealthConfig):
        self.config = config
        self.phase_urls = {
            "phase1_epistemic_audit": config.phase1_url,
            "phase2_isomorphic_mapping": config.phase2_url,
            "phase3_mcts_refinement": config.phase3_url,
            "phase4_architecture_assembly": config.phase4_url,
        }

    async def check_phase_health(
        self,
        session: aiohttp.ClientSession,
        phase_name: str,
        base_url: str,
        endpoint: str = "/health"
    ) -> PhaseHealth:
        """
        Check health of a single phase.

        Args:
            session: aiohttp session
            phase_name: Name of the phase
            base_url: Base URL of phase health API
            endpoint: Health endpoint to query

        Returns:
            PhaseHealth object with check results
        """
        url = f"{base_url}{endpoint}"
        start_time = asyncio.get_event_loop().time()

        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_ms / 1000.0)
            ) as response:
                response_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                if response.status == 200:
                    data = await response.json()

                    return PhaseHealth(
                        phase_name=phase_name,
                        status=HealthStatus.HEALTHY,
                        uptime_seconds=data.get("checks", {}).get("uptime_seconds", 0.0),
                        checks=data.get("checks", {}),
                        response_time_ms=response_time_ms,
                    )
                else:
                    return PhaseHealth(
                        phase_name=phase_name,
                        status=HealthStatus.UNHEALTHY,
                        uptime_seconds=0.0,
                        checks={},
                        error=f"HTTP {response.status}",
                        response_time_ms=response_time_ms,
                    )

        except asyncio.TimeoutError:
            return PhaseHealth(
                phase_name=phase_name,
                status=HealthStatus.UNHEALTHY,
                uptime_seconds=0.0,
                checks={},
                error="Health check timeout",
                response_time_ms=None,
            )

        except Exception as e:
            return PhaseHealth(
                phase_name=phase_name,
                status=HealthStatus.UNKNOWN,
                uptime_seconds=0.0,
                checks={},
                error=str(e),
                response_time_ms=None,
            )

    async def check_all_phases(self, endpoint: str = "/health") -> Dict[str, PhaseHealth]:
        """
        Check health of all phases in parallel.

        Args:
            endpoint: Health endpoint to query (/health, /ready, /metrics)

        Returns:
            Dictionary mapping phase names to PhaseHealth objects
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.check_phase_health(session, phase_name, base_url, endpoint)
                for phase_name, base_url in self.phase_urls.items()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Build result dictionary
            phase_health = {}
            for phase_name, result in zip(self.phase_urls.keys(), results):
                if isinstance(result, Exception):
                    phase_health[phase_name] = PhaseHealth(
                        phase_name=phase_name,
                        status=HealthStatus.UNKNOWN,
                        uptime_seconds=0.0,
                        checks={},
                        error=str(result),
                    )
                else:
                    phase_health[phase_name] = result

            return phase_health

    def compute_overall_health(self, phase_health: Dict[str, PhaseHealth]) -> HealthStatus:
        """
        Compute overall health from all phases.

        Overall health logic:
        - HEALTHY: All phases are healthy
        - DEGRADED: At least one phase is degraded/unknown but none are unhealthy
        - UNHEALTHY: At least one phase is unhealthy

        Args:
            phase_health: Dictionary of phase health results

        Returns:
            Overall HealthStatus
        """
        if not phase_health:
            return HealthStatus.UNKNOWN

        statuses = [ph.status for ph in phase_health.values()]

        # If any phase is unhealthy, overall is unhealthy
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY

        # If any phase is unknown, overall is degraded
        if HealthStatus.UNKNOWN in statuses:
            return HealthStatus.DEGRADED

        # If all are healthy, overall is healthy
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        # Default to degraded
        return HealthStatus.DEGRADED


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_aggregate_response(
    overall_status: HealthStatus,
    phase_health: Dict[str, PhaseHealth],
    correlation_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized aggregate health response.

    Args:
        overall_status: Overall health status
        phase_health: Dictionary of phase health results
        correlation_id: Optional correlation ID

    Returns:
        Standardized response dictionary
    """
    return {
        "status": overall_status.value,
        "system": "rese_pipeline",
        "version": "1.0.0",
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phases": {name: ph.to_dict() for name, ph in phase_health.items()},
        "summary": {
            "total_phases": len(phase_health),
            "healthy_phases": sum(1 for ph in phase_health.values() if ph.status == HealthStatus.HEALTHY),
            "degraded_phases": sum(1 for ph in phase_health.values() if ph.status == HealthStatus.DEGRADED),
            "unhealthy_phases": sum(1 for ph in phase_health.values() if ph.status == HealthStatus.UNHEALTHY),
            "unknown_phases": sum(1 for ph in phase_health.values() if ph.status == HealthStatus.UNKNOWN),
        },
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

checker = AggregateHealthChecker(config)


@app.get("/health")
async def aggregate_liveness():
    """
    Aggregate liveness check - returns 200 if any phase is alive.

    This endpoint queries all phase /health endpoints and returns
    the overall system liveness status.

    Returns:
        JSON response with aggregate liveness status
    """
    phase_health = await checker.check_all_phases(endpoint="/health")
    overall_status = checker.compute_overall_health(phase_health)

    return create_aggregate_response(
        overall_status=overall_status,
        phase_health=phase_health
    )


@app.get("/ready")
async def aggregate_readiness():
    """
    Aggregate readiness check - returns 200 if all phases are ready.

    This endpoint queries all phase /ready endpoints and returns
    the overall system readiness status.

    Returns:
        JSON response with aggregate readiness status
    """
    phase_health = await checker.check_all_phases(endpoint="/ready")
    overall_status = checker.compute_overall_health(phase_health)

    # For readiness, require all phases to be ready
    if overall_status != HealthStatus.HEALTHY:
        raise HTTPException(
            status_code=503,
            detail=f"System not ready: {overall_status.value}"
        )

    return create_aggregate_response(
        overall_status=overall_status,
        phase_health=phase_health
    )


@app.get("/metrics")
async def aggregate_metrics():
    """
    Aggregate metrics endpoint - returns metrics from all phases.

    Returns:
        JSON response with aggregated metrics from all phases
    """
    phase_health = await checker.check_all_phases(endpoint="/metrics")
    overall_status = checker.compute_overall_health(phase_health)

    response = create_aggregate_response(
        overall_status=overall_status,
        phase_health=phase_health
    )

    # Add aggregate metrics
    response["aggregate_metrics"] = {
        "uptime_seconds": (datetime.now(timezone.utc) - _startup_time).total_seconds(),
        "timeout_ms": config.timeout_ms,
        "phase_urls": checker.phase_urls,
    }

    return response


@app.get("/")
async def root():
    """
    Root endpoint - API information.

    Returns:
        JSON response with API information
    """
    return {
        "name": "RESE Aggregate Health API",
        "version": "1.0.0",
        "system": "rese_pipeline",
        "description": "Aggregate health check for all RESE phases",
        "phases": list(checker.phase_urls.keys()),
        "endpoints": {
            "GET /health": "Aggregate liveness check",
            "GET /ready": "Aggregate readiness check",
            "GET /metrics": "Aggregate metrics",
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
            "system": "rese_pipeline",
            "error": str(exc),
            "correlation_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for running the aggregate health API server.

    Environment Variables:
        AGGREGATE_HEALTH_PORT: Port to listen on (default: 8000)
        AGGREGATE_HEALTH_HOST: Host to bind to (default: 0.0.0.0)
        PHASE1_HEALTH_URL: Phase I health API URL
        PHASE2_HEALTH_URL: Phase II health API URL
        PHASE3_HEALTH_URL: Phase III health API URL
        PHASE4_HEALTH_URL: Phase IV health API URL
        AGGREGATE_HEALTH_TIMEOUT_MS: Health check timeout (default: 5000)
    """
    port = int(os.getenv("AGGREGATE_HEALTH_PORT", "8000"))
    host = os.getenv("AGGREGATE_HEALTH_HOST", "0.0.0.0")

    print(f"Starting RESE Aggregate Health API on {host}:{port}")
    print(f"Phase URLs:")
    for phase, url in checker.phase_urls.items():
        print(f"  {phase}: {url}")
    print(f"Documentation: http://{host}:{port}/docs")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
