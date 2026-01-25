# RESE API Documentation

## Overview

The RESE (Recursive Epistemic Solvability Engine) API provides complete programmatic access to the RESE pipeline, including:

- REST API for pipeline control
- WebSocket for real-time updates
- Authentication and authorization
- Monitoring and metrics

**Base URL**: `http://localhost:8000`

**API Version**: v1

---

## Table of Contents

1. [Authentication](#authentication)
2. [Health Endpoints](#health-endpoints)
3. [Pipeline Endpoints](#pipeline-endpoints)
4. [WebSocket API](#websocket-api)
5. [Admin Endpoints](#admin-endpoints)
6. [Data Models](#data-models)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)

---

## Authentication

### API Key Authentication

All API requests require an API key (except health endpoints).

**Header**: `X-API-Key: <your-api-key>`

### Getting an API Key

API keys are managed via the `RESE_API_KEYS` environment variable or `data/api_keys.txt` file.

```bash
# Set API keys via environment
export RESE_API_KEYS="key1,key2,key3"

# Or add to data/api_keys.txt (one per line)
echo "your-secret-key-here" > data/api_keys.txt
```

---

## Health Endpoints

### Health Check

Check API health and uptime.

**Endpoint**: `GET /health`

**Authentication**: Not required

**Response**:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-31T12:00:00",
  "uptime_seconds": 3600.5
}
```

### Root Endpoint

Get API information and documentation links.

**Endpoint**: `GET /`

**Authentication**: Not required

**Response**:

```json
{
  "name": "RESE API",
  "version": "1.0.0",
  "description": "Recursive Epistemic Solvability Engine",
  "docs": "/docs",
  "health": "/health"
}
```

---

## Pipeline Endpoints

### Run Pipeline

Submit a problem to the RESE pipeline for processing.

**Endpoint**: `POST /api/v1/pipeline/run`

**Authentication**: Required

**Request Body**:

```json
{
  "description": "Optimize routing problem",
  "constraints": [
    {
      "id": "c1",
      "type": "hard",
      "description": "Cost constraint",
      "formalization": "cost < 1000",
      "source": "user"
    }
  ],
  "variables": {
    "cost": {"type": "real", "domain": "positive"},
    "distance": {"type": "real", "domain": "positive"}
  },
  "objective": "Minimize cost",
  "domain": "optimization",
  "phases": ["phase1", "phase2", "phase3", "phase4"],
  "use_cache": true
}
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| description | string | Yes | Problem description |
| constraints | array | No | List of constraints |
| variables | object | No | Problem variables |
| objective | string | No | Objective function |
| domain | string | No | Problem domain (default: "general") |
| phases | array | No | Phases to run (default: all) |
| use_cache | boolean | No | Use cached results (default: true) |

**Response**:

```json
{
  "pipeline_id": "rese_problem_abc123_20251231_120000",
  "problem_id": "problem_abc123",
  "status": "completed",
  "final_solution": {
    "architecture": []
  },
  "aci_history": [0.8, 0.6, 0.4, 0.3],
  "validation_score": 0.85,
  "confidence": 0.90,
  "elapsed_seconds": 45.2,
  "phase_results": {
    "phase1": {
      "phase_name": "phase1_epistemic_audit",
      "status": "completed",
      "metrics": {
        "num_constraints": 10,
        "bias_score": 0.3
      },
      "elapsed_seconds": 5.2
    }
  }
}
```

### Get Pipeline Status

Get current status of a running pipeline.

**Endpoint**: `GET /api/v1/pipeline/{pipeline_id}/status`

**Authentication**: Required

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pipeline_id | string | Yes | Pipeline identifier |

**Response**:

```json
{
  "pipeline_id": "rese_problem_abc123_20251231_120000",
  "problem_id": "problem_abc123",
  "status": "running",
  "elapsed_seconds": 15.5,
  "phases": {
    "phase1": {
      "status": "completed",
      "elapsed": 5.2,
      "metrics": {"num_constraints": 10}
    },
    "phase2": {
      "status": "running",
      "elapsed": 10.3,
      "metrics": {}
    }
  }
}
```

### Get Pipeline Result

Get complete result of a finished pipeline.

**Endpoint**: `GET /api/v1/pipeline/{pipeline_id}/result`

**Authentication**: Required

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pipeline_id | string | Yes | Pipeline identifier |

**Response**: Same as Run Pipeline response above.

### Cancel Pipeline

Cancel a running pipeline.

**Endpoint**: `DELETE /api/v1/pipeline/{pipeline_id}`

**Authentication**: Required

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pipeline_id | string | Yes | Pipeline identifier |

**Response**:

```json
{
  "message": "Pipeline rese_problem_abc123_20251231_120000 cancelled",
  "pipeline_id": "rese_problem_abc123_20251231_120000"
}
```

---

## WebSocket API

### Connect to Pipeline Updates

Connect to WebSocket for real-time pipeline updates.

**Endpoint**: `WS /ws/pipeline/{pipeline_id}`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| pipeline_id | string | Yes | Pipeline to subscribe to |

### WebSocket Messages

#### Server → Client Messages

**Pipeline Update**:

```json
{
  "type": "pipeline_update",
  "pipeline_id": "rese_problem_abc123_20251231_120000",
  "status": "running",
  "progress": {
    "pipeline_id": "rese_problem_abc123_20251231_120000",
    "status": "running",
    "phase_results": {
      "phase1": {"status": "completed", "elapsed_seconds": 5.2}
    }
  }
}
```

**Subscription Confirmation**:

```json
{
  "type": "subscribed",
  "pipeline_id": "rese_problem_abc123_20251231_120000",
  "client_id": "client_xyz789",
  "timestamp": "2025-12-31T12:00:00"
}
```

**Pong Response**:

```json
{
  "type": "pong",
  "timestamp": "2025-12-31T12:00:00"
}
```

#### Client → Server Messages

**Subscribe to Pipeline** (automatic on connect):

```json
{
  "type": "subscribe",
  "pipeline_id": "rese_problem_abc123_20251231_120000"
}
```

**Unsubscribe**:

```json
{
  "type": "unsubscribe",
  "pipeline_id": "rese_problem_abc123_20251231_120000"
}
```

**Ping** (keep-alive):

```json
{
  "type": "ping"
}
```

### WebSocket Example (Python)

```python
import asyncio
import websockets
import json

async def monitor_pipeline(pipeline_id):
    uri = f"ws://localhost:8000/ws/pipeline/{pipeline_id}"

    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data['type'] == 'pipeline_update':
                status = data['status']
                print(f"Status: {status}")

                if status in ['completed', 'failed']:
                    break

# Run
asyncio.run(monitor_pipeline("rese_problem_abc123_20251231_120000"))
```

### WebSocket Example (JavaScript)

```javascript
const pipelineId = "rese_problem_abc123_20251231_120000";
const ws = new WebSocket(`ws://localhost:8000/ws/pipeline/${pipelineId}`);

ws.onopen = () => {
  console.log("Connected to pipeline updates");
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'pipeline_update') {
    console.log("Status:", data.status);

    if (data.status === 'completed' || data.status === 'failed') {
      ws.close();
    }
  }
};

ws.onerror = (error) => {
  console.error("WebSocket error:", error);
};

// Send ping every 30 seconds
setInterval(() => {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({type: "ping"}));
  }
}, 30000);
```

---

## Admin Endpoints

### Get System Statistics

Get system-wide statistics (admin only).

**Endpoint**: `GET /api/v1/admin/stats`

**Authentication**: Required

**Response**:

```json
{
  "active_pipelines": 3,
  "stored_results": 15,
  "websocket_connections": 5,
  "uptime_seconds": 86400.5
}
```

### Clear Cache

Clear pipeline cache (admin only).

**Endpoint**: `POST /api/v1/admin/cache/clear`

**Authentication**: Required

**Response**:

```json
{
  "message": "Cache cleared"
}
```

---

## Data Models

### ProblemRequest

```typescript
{
  description: string;           // Required
  constraints?: Constraint[];    // Optional
  variables?: Record<string, any>; // Optional
  objective?: string;            // Optional
  domain?: string;               // Default: "general"
  phases?: string[];             // Default: all phases
  use_cache?: boolean;           // Default: true
}

interface Constraint {
  id: string;
  type: 'hard' | 'soft';
  description: string;
  formalization: string;
  source: string;
}
```

### PipelineResultResponse

```typescript
{
  pipeline_id: string;
  problem_id: string;
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';
  final_solution?: Record<string, any>;
  aci_history: number[];
  validation_score: number;      // 0.0 to 1.0
  confidence: number;            // 0.0 to 1.0
  elapsed_seconds: number;
  phase_results: Record<string, PhaseResult>;
}

interface PhaseResult {
  phase_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  output?: any;
  metrics: Record<string, number>;
  errors: string[];
  warnings: string[];
  start_time: string;            // ISO 8601
  end_time?: string;             // ISO 8601
  elapsed_seconds: number;
}
```

### ErrorResponse

```typescript
{
  error: string;
  detail?: string;
  timestamp: string;             // ISO 8601
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized (missing/invalid API key) |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "error": "Error type",
  "detail": "Detailed error message",
  "timestamp": "2025-12-31T12:00:00"
}
```

### Common Errors

**Missing API Key**:

```json
{
  "error": "Unauthorized",
  "detail": "API key required",
  "timestamp": "2025-12-31T12:00:00"
}
```

**Invalid API Key**:

```json
{
  "error": "Unauthorized",
  "detail": "Invalid API key",
  "timestamp": "2025-12-31T12:00:00"
}
```

**Pipeline Not Found**:

```json
{
  "error": "Not Found",
  "detail": "Pipeline rese_invalid_id not found",
  "timestamp": "2025-12-31T12:00:00"
}
```

**Rate Limit Exceeded**:

```json
{
  "error": "Rate Limit Exceeded",
  "detail": "Too many requests. Maximum 60 requests per minute.",
  "timestamp": "2025-12-31T12:00:00"
}
```

---

## Rate Limiting

### Default Limits

- **60 requests per minute** per API key
- Configurable via `rate_limit_per_minute` in API config

### Rate Limit Headers

Response includes rate limit information:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1704067200
```

### Handling Rate Limits

When rate limited:

1. Wait for the reset time (in `X-RateLimit-Reset` header)
2. Implement exponential backoff
3. Use WebSocket for real-time updates (doesn't count against rate limit)

**Example (Python)**:

```python
import time
import requests

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            # Rate limited - wait and retry
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            wait_time = max(reset_time - time.time(), 1)

            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)

            continue

        return response

    return None  # All retries exhausted
```

---

## Interactive API Documentation

The API includes interactive Swagger UI and ReDoc documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide:
- Interactive API exploration
- Request/response examples
- Try-it-out functionality
- Schema validation

---

## Python Client Example

```python
import requests
import json

class RESEClient:
    """Python client for RESE API"""

    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def run_pipeline(self, description, constraints=None, variables=None, **kwargs):
        """Run RESE pipeline"""
        url = f"{self.base_url}/api/v1/pipeline/run"

        payload = {
            "description": description,
            "constraints": constraints or [],
            "variables": variables or {},
            **kwargs
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        return response.json()

    def get_status(self, pipeline_id):
        """Get pipeline status"""
        url = f"{self.base_url}/api/v1/pipeline/{pipeline_id}/status"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_result(self, pipeline_id):
        """Get pipeline result"""
        url = f"{self.base_url}/api/v1/pipeline/{pipeline_id}/result"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def cancel_pipeline(self, pipeline_id):
        """Cancel pipeline"""
        url = f"{self.base_url}/api/v1/pipeline/{pipeline_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

# Usage
client = RESEClient(api_key="your-api-key")

result = client.run_pipeline(
    description="Optimize routing",
    constraints=[
        {
            "id": "c1",
            "type": "hard",
            "description": "Cost constraint",
            "formalization": "cost < 1000",
            "source": "user"
        }
    ],
    variables={"cost": {"type": "real"}},
    phases=["phase1", "phase2"]
)

print(f"Pipeline ID: {result['pipeline_id']}")
print(f"Status: {result['status']}")
print(f"Validation Score: {result['validation_score']}")
```

---

## Support

For issues or questions:

- **Documentation**: `rese/docs/`
- **Issues**: Via project issue tracker
- **Email**: support@example.com

---

*Last Updated: 2025-12-31*
*Version: 1.0.0*
