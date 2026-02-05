# RESE Health APIs - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install fastapi uvicorn aiohttp
```

### Step 2: Start a Single Health API

```bash
# From the Frontend directory
cd glue/adapters/rese-phase1/src
python health_api.py
```

You should see:
```
Starting RESE Phase I Health API on 0.0.0.0:8001
Documentation: http://0.0.0.0:8001/docs
```

### Step 3: Test the Endpoint

In another terminal:

```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "phase": "phase1_epistemic_audit",
  "version": "1.0.0",
  "correlation_id": "...",
  "timestamp": "2026-02-04T...",
  "checks": {
    "alive": true,
    "uptime_seconds": 5.23
  }
}
```

### Step 4: View Interactive Documentation

Open in browser:
```
http://localhost:8001/docs
```

## Starting All Health APIs

### Option 1: Using the Startup Script

```bash
# From the Frontend directory
python start_rese_health_apis.py
```

This starts all 5 APIs (phases 1-4 + aggregate).

### Option 2: Starting Specific Phases

```bash
python start_rese_health_apis.py --phases phase1,phase2,aggregate
```

### Option 3: Start and Test

```bash
python start_rese_health_apis.py --test
```

## Quick Testing

### Test Single Phase

```bash
python verify_rese_health_apis.py phase1
```

### Test All Phases

```bash
python test_rese_health_endpoints.py
```

Example output:
```
================================================================================
RESE Phase Health Endpoints Test
================================================================================
Time: 2026-02-04 22:00:00

Testing Phase I (Epistemic Audit)
URL: http://localhost:8001
--------------------------------------------------------------------------------
✓ / (200): 12.5ms
✓ /health (200): 8.3ms
✓ /ready (200): 15.2ms
✓ /metrics (200): 18.7ms

...
================================================================================
Test Summary
================================================================================
Total Tests: 20/20 (100.0%)

Response Time Statistics:
  Average: 14.23ms
  Min: 5.12ms
  Max: 25.40ms

All tests passed! ✓
```

## Common curl Commands

```bash
# Liveness check
curl http://localhost:8001/health | jq

# Readiness check
curl http://localhost:8001/ready | jq

# Detailed metrics
curl http://localhost:8001/metrics | jq

# API info
curl http://localhost:8001/ | jq

# Aggregate health
curl http://localhost:8000/health | jq

# Pretty print metrics
curl -s http://localhost:8001/metrics | python -m json.tool
```

## Endpoint URLs

| Service | Health | Ready | Metrics | Docs |
|---------|--------|-------|---------|------|
| Phase I | http://localhost:8001/health | http://localhost:8001/ready | http://localhost:8001/metrics | http://localhost:8001/docs |
| Phase II | http://localhost:8002/health | http://localhost:8002/ready | http://localhost:8002/metrics | http://localhost:8002/docs |
| Phase III | http://localhost:8003/health | http://localhost:8003/ready | http://localhost:8003/metrics | http://localhost:8003/docs |
| Phase IV | http://localhost:8004/health | http://localhost:8004/ready | http://localhost:8004/metrics | http://localhost:8004/docs |
| Aggregate | http://localhost:8000/health | http://localhost:8000/ready | http://localhost:8000/metrics | http://localhost:8000/docs |

## Troubleshooting

### "Address already in use"

Change the port:
```bash
export PHASE1_HEALTH_PORT=8011
python health_api.py
```

### "Module not found"

Make sure you're in the correct directory and dependencies are installed:
```bash
cd glue/adapters/rese-phase1/src
pip install fastapi uvicorn
python health_api.py
```

### "Executor initialization failed"

The executor may need configuration. Check the `/ready` endpoint for details:
```bash
curl http://localhost:8001/ready
```

### Slow response times

Check if the circuit breaker is open:
```bash
curl http://localhost:8001/metrics | jq .checks.circuit_breaker
```

## Next Steps

1. Read the full documentation: `RESE_HEALTH_APIS_README.md`
2. Integrate with your monitoring system (Prometheus, Grafana)
3. Set up Kubernetes probes
4. Configure alerts based on health status

## Support

For detailed information:
- Full README: `glue/adapters/RESE_HEALTH_APIS_README.md`
- API Documentation: http://localhost:<port>/docs
- Source code: `glue/adapters/rese-phase*/src/health_api.py`
