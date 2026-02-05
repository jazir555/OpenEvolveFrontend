# PES Enhanced API Integration Guide

This guide explains how to integrate the PES Enhanced API routes into the main API server.

## Quick Start

### 1. Import and Register Router

Add the following to `api_server.py`:

```python
# Add near other integration imports (around line 100-160)
try:
    from openevolve_pes_enhanced.api_routes import router as pes_enhanced_router
    PES_ENHANCED_API_AVAILABLE = True
except ImportError:
    PES_ENHANCED_API_AVAILABLE = False
    pes_enhanced_router = None
```

Then register the router (after `app = FastAPI(...)`):

```python
# Add after other router includes (around line 400)
if PES_ENHANCED_API_AVAILABLE and pes_enhanced_router:
    app.include_router(pes_enhanced_router)
    logger.info("PES Enhanced API routes registered")
```

### 2. Alternative: Use Helper Function

```python
from openevolve_pes_enhanced.api_routes import get_pes_enhanced_router

# In your app setup:
app.include_router(get_pes_enhanced_router())
```

## API Endpoints

Once integrated, the following endpoints are available:

### Evolution Runs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/pes-enhanced/runs` | Start a new evolution run |
| GET | `/pes-enhanced/runs` | List all runs |
| GET | `/pes-enhanced/runs/{run_id}` | Get run status/results |
| POST | `/pes-enhanced/runs/{run_id}/stop` | Stop a running evolution |

### Cost Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/pes-enhanced/cost-estimate` | Estimate cost before running |
| GET | `/pes-enhanced/runs/{run_id}/budget` | Get current budget status |

### Strategy

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/pes-enhanced/recommend-strategy` | Get strategy recommendation |
| GET | `/pes-enhanced/health` | Health check |

### WebSocket

| Method | Endpoint | Description |
|--------|----------|-------------|
| WS | `/pes-enhanced/ws/monitor/{run_id}` | Real-time monitoring |

## Example Usage

### Start Evolution Run

```bash
curl -X POST http://localhost:8000/pes-enhanced/runs \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "code": "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
    "problem_description": "Optimize Fibonacci for large inputs",
    "tests": [
      {"name": "test_10", "input": "fib(10)", "expected_output": "55"},
      {"name": "test_20", "input": "fib(20)", "expected_output": "6765"}
    ],
    "language": "python",
    "max_cost_usd": 5.0,
    "enable_cost_optimization": true,
    "enable_early_stopping": true
  }'
```

Response:
```json
{
  "run_id": "pes-enhanced-a1b2c3d4e5f6",
  "status": "pending",
  "success": false,
  "created_at": "2026-02-04T15:30:00.000000"
}
```

### Check Cost Estimate

```bash
curl -X POST http://localhost:8000/pes-enhanced/cost-estimate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "iterations": 50,
    "population_size": 20,
    "problem_complexity": "medium"
  }'
```

Response:
```json
{
  "estimated_cost_usd": 0.85,
  "estimated_tokens": 500000,
  "estimated_duration_ms": 75000,
  "recommended_strategy": "standard",
  "prompt_tokens": 350000,
  "completion_tokens": 150000,
  "prompt_cost_usd": 0.35,
  "completion_cost_usd": 0.45,
  "total_evaluations": 1000,
  "parameter_recommendations": {
    "iterations": 50,
    "population_size": 20,
    "early_stopping": true,
    "use_cheap_model": false
  }
}
```

### Get Run Results

```bash
curl http://localhost:8000/pes-enhanced/runs/pes-enhanced-a1b2c3d4e5f6 \
  -H "X-API-Key: your-api-key"
```

Response:
```json
{
  "run_id": "pes-enhanced-a1b2c3d4e5f6",
  "status": "completed",
  "success": true,
  "best_solution": "def fib(n, memo={}): ...",
  "best_fitness": 0.95,
  "total_cost_usd": 0.72,
  "efficiency_gain": 0.35,
  "iterations": 35,
  "converged": true,
  "stopped_early": true,
  "stop_reason": "Converged: fitness plateau detected",
  "strategy_used": "adaptive_mutation",
  "recommendations": [
    "Consider memoization for recursive functions",
    "Early stopping saved 15 iterations"
  ]
}
```

### WebSocket Monitoring

```javascript
const runId = 'pes-enhanced-a1b2c3d4e5f6';
const ws = new WebSocket(`ws://localhost:8000/pes-enhanced/ws/monitor/${runId}`);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  console.log(`[${msg.type}]`, msg.data);
  
  switch(msg.type) {
    case 'status':
      updateStatus(msg.data.status);
      break;
    case 'progress':
      updateProgress(msg.data.iteration, msg.data.fitness);
      break;
    case 'result':
      displayResults(msg.data);
      ws.close();
      break;
    case 'error':
      displayError(msg.data.error);
      ws.close();
      break;
  }
};

// Send ping every 30 seconds to keep connection alive
setInterval(() => ws.send('ping'), 30000);
```

## Request/Response Models

### PESEnhancedRunRequest

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| code | string | Yes | Source code to evolve |
| problem_description | string | Yes | Problem description (min 10 chars) |
| tests | array | No | Test cases |
| language | string | No | Programming language (default: "python") |
| max_cost_usd | float | No | Maximum budget in USD |
| max_tokens | int | No | Maximum token budget |
| max_time_seconds | int | No | Maximum time (default: 1800) |
| enable_cost_optimization | bool | No | Enable cost tracking (default: true) |
| enable_early_stopping | bool | No | Enable early stopping (default: true) |
| enable_planning | bool | No | Enable strategy planning (default: true) |
| enable_summarization | bool | No | Enable result summarization (default: true) |
| max_iterations | int | No | Maximum iterations |
| population_size | int | No | Population size |
| webhook_url | string | No | Webhook URL for notifications |

### PESEnhancedRunResponse

| Field | Type | Description |
|-------|------|-------------|
| run_id | string | Unique run identifier |
| status | string | Run status |
| success | bool | Whether evolution succeeded |
| best_solution | string | Best evolved code |
| best_fitness | float | Best fitness score |
| total_cost_usd | float | Total cost |
| efficiency_gain | float | Efficiency gain (0.0 to 1.0) |
| iterations | int | Iterations performed |
| converged | bool | Whether converged |
| stopped_early | bool | Whether stopped early |
| stop_reason | string | Reason for stopping |
| strategy_used | string | Evolution strategy used |
| recommendations | array | Post-run recommendations |
| metrics | object | Detailed metrics |
| budget_status | object | Final budget status |

## Error Handling

The API returns standard HTTP status codes:

| Status | Description |
|--------|-------------|
| 200 | Success |
| 202 | Accepted (run started) |
| 400 | Bad Request (validation error) |
| 404 | Not Found (run doesn't exist) |
| 422 | Validation Error (invalid parameters) |
| 503 | Service Unavailable (PES Enhanced not available) |

## Testing

Run the test suite:

```bash
# Run all tests
pytest openevolve_pes_enhanced/test_api_routes.py -v

# Run with coverage
pytest openevolve_pes_enhanced/test_api_routes.py --cov=openevolve_pes_enhanced -v

# Run specific test category
pytest openevolve_pes_enhanced/test_api_routes.py::TestHealthEndpoint -v
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   API Client    │────▶│  FastAPI Router  │────▶│  PES Enhanced   │
│                 │     │  (api_routes.py) │     │   (Wrapper)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │                           │
                                ▼                           ▼
                        ┌──────────────┐            ┌──────────────┐
                        │  WebSocket   │            │   Existing   │
                        │  Broadcast   │            │    OpenEvolve │
                        └──────────────┘            └──────────────┘
```

## Configuration

The API uses the same configuration as PES Enhanced:

```python
from openevolve_pes_enhanced.config import PESEnhancedConfig

config = PESEnhancedConfig(
    cost=CostOptimizationConfig(max_cost_usd=10.0),
    early_stopping=EarlyStoppingConfig(patience=5),
    enable_cost_optimization=True,
    enable_early_stopping=True
)
```

## Security Considerations

1. **API Key Authentication**: All endpoints should be protected with `verify_api_key` dependency
2. **Rate Limiting**: Consider adding rate limits for cost estimation endpoints
3. **Input Validation**: All inputs are validated using Pydantic models
4. **Webhook Security**: Verify webhook URLs are from trusted domains

## Troubleshooting

### PES Enhanced Not Available

If you see `503 Service Unavailable`:
1. Check that `openevolve_pes_enhanced` module is installed
2. Verify imports are working: `python -c "from openevolve_pes_enhanced import PESIntegrationWrapper"`

### Run Not Found

If you see `404 Not Found` for a run:
1. Run IDs expire after server restart (in-memory storage)
2. For production, implement persistent storage

### WebSocket Connection Issues

1. Ensure WebSocket support is enabled in your server
2. Check firewall rules for WebSocket connections
3. Use `wss://` for secure connections in production

## Migration from Direct API

If you're currently using `PESIntegrationWrapper` directly:

**Before:**
```python
from openevolve_pes_enhanced import PESIntegrationWrapper

wrapper = PESIntegrationWrapper()
result = await wrapper.enhance_with_planning(code, problem, tests)
```

**After:**
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/pes-enhanced/runs",
        json={"code": code, "problem_description": problem, "tests": tests}
    )
    run_id = response.json()["run_id"]
    
    # Poll for results
    while True:
        status = await client.get(f"http://localhost:8000/pes-enhanced/runs/{run_id}")
        if status.json()["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(1)
```

## Future Enhancements

1. **Persistent Storage**: Replace in-memory `_pe_runs` with database
2. **Queue Management**: Add Redis/RabbitMQ for distributed execution
3. **Metrics Export**: Add Prometheus metrics endpoint
4. **Batch Operations**: Support for batch evolution runs
5. **Caching**: Cache strategy recommendations for similar problems
