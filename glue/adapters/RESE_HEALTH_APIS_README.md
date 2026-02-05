# RESE Health Check APIs

Production monitoring endpoints for all RESE phase adapters.

## Overview

This directory contains FastAPI-based health check endpoints for monitoring the RESE (Recursive Epistemic Solvability Engine) pipeline in production environments.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Aggregate Health (8000)                   │
│              glue/adapters/rese-integration/health/         │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Phase I (8001)│  │ Phase II(8002)│  │Phase III(8003)│
│  Epistemic    │  │  Isomorphic   │  │   MCTS        │
│  Audit        │  │  Mapping      │  │  Refinement   │
└───────────────┘  └───────────────┘  └───────────────┘
                                                        │
                                                        ▼
                                              ┌───────────────┐
                                              │ Phase IV(8004)│
                                              │ Architecture  │
                                              │  Assembly     │
                                              └───────────────┘
```

## Health Endpoints

Each phase provides three standardized endpoints:

### 1. Liveness Check: `GET /health`

Returns `200 OK` if the service process is alive and responsive.

**Response:**
```json
{
  "status": "healthy",
  "phase": "phase1_epistemic_audit",
  "version": "1.0.0",
  "correlation_id": "uuid",
  "timestamp": "2026-02-04T22:00:00Z",
  "checks": {
    "alive": true,
    "uptime_seconds": 123.45
  }
}
```

### 2. Readiness Check: `GET /ready`

Returns `200 OK` if the service is ready to handle requests (executor initialized, config valid).

**Response:**
```json
{
  "status": "ready",
  "phase": "phase1_epistemic_audit",
  "version": "1.0.0",
  "correlation_id": "uuid",
  "timestamp": "2026-02-04T22:00:00Z",
  "checks": {
    "executor": "pass",
    "configuration": "valid"
  }
}
```

**Error Response (503):**
```json
{
  "detail": "Service not ready: Executor initialization failed: ..."
}
```

### 3. Metrics: `GET /metrics`

Returns detailed metrics about the executor state.

**Phase I Response:**
```json
{
  "status": "healthy",
  "phase": "phase1_epistemic_audit",
  "version": "1.0.0",
  "correlation_id": "uuid",
  "timestamp": "2026-02-04T22:00:00Z",
  "checks": {
    "circuit_breaker": {
      "state": "closed",
      "failure_count": 0,
      "last_failure_time": null
    },
    "dlq_size": 0,
    "config": {
      "max_assumptions": 100,
      "max_constraints": 1000,
      "timeout_ms": 15000,
      "enable_tacit_mining": true,
      "enable_red_team": true
    }
  }
}
```

**Phase II Response:**
```json
{
  "checks": {
    "circuit_breaker": {
      "state": "CLOSED",
      "failure_count": 0
    },
    "i_mech_threshold": 0.7,
    "domain_kb_size": 4,
    "domains": ["physics", "biology", "economics", "computer_science"]
  }
}
```

**Phase III Response:**
```json
{
  "checks": {
    "circuit_breaker": {
      "is_open": false,
      "failure_count": 0
    },
    "mcts_tree": {
      "total_nodes": 0,
      "max_depth": 0,
      "leaf_nodes": 0,
      "expanded_nodes": 0
    },
    "convergence_detector": {
      "confidence_history_size": 0,
      "aci_window_size": 100
    },
    "dlq_size": 0
  }
}
```

**Phase IV Response:**
```json
{
  "checks": {
    "circuit_breaker": {
      "state": "closed",
      "can_execute": true
    },
    "validation_level": "standard",
    "integration_strategy": "weighted_average",
    "config": {
      "min_confidence_threshold": 0.7,
      "max_paradigm_shifts": 50,
      "assembly_timeout_ms": 25000
    }
  }
}
```

### 4. Root: `GET /`

Returns API information and available endpoints.

**Response:**
```json
{
  "name": "RESE Phase I Health API",
  "version": "1.0.0",
  "phase": "phase1_epistemic_audit",
  "description": "Health check endpoints for Phase I Epistemic Audit Executor",
  "endpoints": {
    "GET /health": "Liveness check",
    "GET /ready": "Readiness check",
    "GET /metrics": "Detailed metrics",
    "GET /docs": "API documentation (Swagger UI)",
    "GET /redoc": "API documentation (ReDoc)"
  },
  "documentation": {
    "swagger": "/docs",
    "redoc": "/redoc"
  }
}
```

## Aggregate Health Endpoint

The aggregate health API monitors all phases and provides unified status.

### Endpoints

- `GET /health` - Aggregate liveness (returns healthy if any phase is alive)
- `GET /ready` - Aggregate readiness (returns healthy only if ALL phases are ready)
- `GET /metrics` - Aggregated metrics from all phases

### Response

```json
{
  "status": "healthy",
  "system": "rese_pipeline",
  "version": "1.0.0",
  "correlation_id": "uuid",
  "timestamp": "2026-02-04T22:00:00Z",
  "phases": {
    "phase1_epistemic_audit": {
      "phase": "phase1_epistemic_audit",
      "status": "healthy",
      "uptime_seconds": 123.45,
      "checks": {...},
      "response_time_ms": 15.2
    },
    "phase2_isomorphic_mapping": {
      ...
    },
    "phase3_mcts_refinement": {
      ...
    },
    "phase4_architecture_assembly": {
      ...
    }
  },
  "summary": {
    "total_phases": 4,
    "healthy_phases": 4,
    "degraded_phases": 0,
    "unhealthy_phases": 0,
    "unknown_phases": 0
  }
}
```

## Configuration

All health APIs follow the **Law of Configuration Explicitness** (CLAUDE.md):

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PHASE1_HEALTH_PORT` | Phase I health API port | 8001 |
| `PHASE1_HEALTH_HOST` | Phase I health API host | 0.0.0.0 |
| `PHASE2_HEALTH_PORT` | Phase II health API port | 8002 |
| `PHASE2_HEALTH_HOST` | Phase II health API host | 0.0.0.0 |
| `PHASE3_HEALTH_PORT` | Phase III health API port | 8003 |
| `PHASE3_HEALTH_HOST` | Phase III health API host | 0.0.0.0 |
| `PHASE4_HEALTH_PORT` | Phase IV health API port | 8004 |
| `PHASE4_HEALTH_HOST` | Phase IV health API host | 0.0.0.0 |
| `AGGREGATE_HEALTH_PORT` | Aggregate health API port | 8000 |
| `AGGREGATE_HEALTH_HOST` | Aggregate health API host | 0.0.0.0 |
| `PHASE1_HEALTH_URL` | Phase I URL (for aggregate) | http://localhost:8001 |
| `PHASE2_HEALTH_URL` | Phase II URL (for aggregate) | http://localhost:8002 |
| `PHASE3_HEALTH_URL` | Phase III URL (for aggregate) | http://localhost:8003 |
| `PHASE4_HEALTH_URL` | Phase IV URL (for aggregate) | http://localhost:8004 |
| `AGGREGATE_HEALTH_TIMEOUT_MS` | Health check timeout | 5000 |

## Installation

### Prerequisites

```bash
pip install fastapi uvicorn aiohttp
```

### Files Created

1. **Phase I**: `glue/adapters/rese-phase1/src/health_api.py`
2. **Phase II**: `glue/adapters/rese-phase2/src/health_api.py`
3. **Phase III**: `glue/adapters/rese-phase3/src/health_api.py`
4. **Phase IV**: `glue/adapters/rese-phase4/src/health_api.py`
5. **Aggregate**: `glue/adapters/rese-integration/health/aggregate_health.py`

## Usage

### Starting Individual Health APIs

Each health API can be started independently:

```bash
# Phase I
cd glue/adapters/rese-phase1/src
python health_api.py

# Phase II
cd glue/adapters/rese-phase2/src
python health_api.py

# Phase III
cd glue/adapters/rese-phase3/src
python health_api.py

# Phase IV
cd glue/adapters/rese-phase4/src
python health_api.py

# Aggregate
cd glue/adapters/rese-integration/health
python aggregate_health.py
```

### Using the Startup Script

The `start_rese_health_apis.py` script starts all health APIs:

```bash
# Start all APIs
python start_rese_health_apis.py

# Start specific phases
python start_rese_health_apis.py --phases phase1,phase2

# Start and run tests
python start_rese_health_apis.py --test

# Start only aggregate health
python start_rese_health_apis.py --phases aggregate
```

### Testing Health Endpoints

#### Verify Single Phase

```bash
python verify_rese_health_apis.py phase1
python verify_rese_health_apis.py phase2
python verify_rese_health_apis.py phase3
python verify_rese_health_apis.py phase4
python verify_rese_health_apis.py aggregate
```

#### Run Comprehensive Tests

```bash
python test_rese_health_endpoints.py
```

This tests all endpoints on all phases with detailed output.

## Example curl Commands

```bash
# Check Phase I liveness
curl http://localhost:8001/health

# Check Phase II readiness
curl http://localhost:8002/ready

# Get Phase III metrics
curl http://localhost:8003/metrics

# Check Phase IV liveness
curl http://localhost:8004/health

# Check aggregate health
curl http://localhost:8000/health

# Get aggregate metrics
curl http://localhost:8000/metrics | jq
```

## Monitoring Integration

### Kubernetes Liveness/Readiness Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8001
  initialDelaySeconds: 5
  periodSeconds: 5
```

### Prometheus Metrics

The `/metrics` endpoints return structured JSON that can be scraped:

```python
import requests
import json

response = requests.get("http://localhost:8001/metrics")
data = response.json()

# Extract metrics
circuit_breaker_state = data["checks"]["circuit_breaker"]["state"]
dlq_size = data["checks"]["dlq_size"]

print(f"Circuit Breaker: {circuit_breaker_state}")
print(f"DLQ Size: {dlq_size}")
```

### Grafana Dashboard

Use the aggregate health endpoint to build a Grafana dashboard showing:

- Overall system health (status from `/health`)
- Phase-specific metrics (from each phase's `/metrics`)
- Response times
- Circuit breaker states
- DLQ sizes
- Configuration values

## Design Principles

Following CLAUDE.md guidelines:

1. **Law of Configuration Explicitness**: All config via environment variables
2. **Structured Logging**: JSON responses with correlation_id
3. **UTC Timestamps**: All temporal data in UTC
4. **Timeout Enforcement**: All operations timeout (default 5000ms for aggregate)
5. **Graceful Degradation**: Partial failures don't break aggregation
6. **Error Handling**: Proper HTTP status codes (200, 503)

## Troubleshooting

### Port Already in Use

If you get "Address already in use", change the port:

```bash
export PHASE1_HEALTH_PORT=8011
python health_api.py
```

### Import Errors

Ensure you're in the correct directory:

```bash
cd glue/adapters/rese-phase1/src
python health_api.py
```

### Executor Initialization Failures

Check the `/ready` endpoint for detailed error messages:

```bash
curl http://localhost:8001/ready
```

If executor fails to initialize, check:
1. Environment variables are set correctly
2. Executor dependencies are installed
3. Configuration validation passes

### Circuit Breaker Open

Check `/metrics` endpoint to see circuit breaker state:

```bash
curl http://localhost:8001/metrics | jq .checks.circuit_breaker
```

If circuit breaker is open:
1. Check logs for failure reasons
2. Wait for timeout (default 60s)
3. Circuit breaker will automatically transition to half-open

## API Documentation

Each health API provides interactive documentation:

- Swagger UI: `http://localhost:<port>/docs`
- ReDoc: `http://localhost:<port>/redoc`

## Support

For issues or questions:
1. Check the logs in the terminal where the API is running
2. Use `/metrics` endpoint to get detailed state
3. Review the executor configuration
4. Check aggregate health for system-wide issues

## Version History

- **1.0.0** (2026-02-04): Initial implementation
  - All four phase health APIs
  - Aggregate health endpoint
  - Comprehensive testing scripts
