# DataPizza API Server Implementation

**Status**: ✅ COMPLETED
**Date**: 2026-02-02

## What Was Implemented

Created comprehensive FastAPI server (`datapizza_api_server.py`) to wrap DataPizza functionality with REST API endpoints.

## Features Implemented

### 1. FastAPI Application
- Fully documented API with automatic OpenAPI spec
- CORS middleware enabled for cross-origin requests
- Health check and status endpoints
- Background task support for long-running operations
- Comprehensive error handling

### 2. API Endpoints

#### `GET /health`
Health check endpoint for monitoring and load balancers.

**Response**:
```json
{
  "status": "operational" | "degraded",
  "healthy": true,
  "timestamp": "2026-02-02T14:38:10"
}
```

#### `GET /status`
Get server status, available tools, and active pipelines.

**Response**:
```json
{
  "status": "ready" | "degraded",
  "datapizza_available": true | false,
  "version": "1.0.0",
  "available_tools": ["FileSystem", "DuckDuckGo", "SQL", "WebFetch"],
  "available_clients": ["OpenAI", "Anthropic", "Google"],
  "uptime_seconds": 123.45,
  "active_pipelines": 2
}
```

#### `POST /query`
Execute queries against data sources.

**Request**:
```json
{
  "query": "What is the capital of France?",
  "data_source": "default",
  "params": {},
  "max_results": 10,
  "timeout": 30
}
```

**Response**:
```json
{
  "success": true,
  "query": "...",
  "results": [...],
  "count": 1,
  "data_source": "default",
  "execution_time": 0.5,
  "message": null
}
```

#### `POST /process`
Process data through DataPizza operations.

**Request**:
```json
{
  "data": "Text to process...",
  "operation": "chunk",
  "params": {"chunk_size": 1000},
  "timeout": 60
}
```

**Response**:
```json
{
  "success": true,
  "operation": "chunk",
  "processed_data": [...],
  "count": 5,
  "execution_time": 1.0,
  "progress": {"complete": 1.0},
  "message": null
}
```

#### `POST /pipeline`
Run complete DataPizza pipeline with multiple stages.

**Request**:
```json
{
  "data": "Document text...",
  "stages": ["validate", "chunk", "embed", "store"],
  "data_source": "default",
  "chunk_size": 1000,
  "embedding_model": "default",
  "vector_store": "default",
  "timeout": 300
}
```

**Response**:
```json
{
  "success": true,
  "stages_completed": ["validate", "chunk", "embed", "store"],
  "stages_failed": [],
  "results": {...},
  "execution_time": 2.5,
  "progress": {"overall": 1.0, "validate": 1.0, "chunk": 1.0, ...},
  "summary": {
    "total_stages": 4,
    "completed_stages": 4,
    "failed_stages": 0,
    "pipeline_id": "pipeline_12345"
  }
}
```

### 3. Graceful Degradation
When DataPizza core is not available:
- All endpoints continue to work
- Returns simulated/fallback results
- Sets `"datapizza_available": false` in status
- Adds message: "DataPizza not available - using fallback"
- Maintains full API contract for frontend integration

### 4. Pydantic Models
Type-safe request/response models:
- `QueryRequest` / `QueryResponse`
- `ProcessRequest` / `ProcessResponse`
- `PipelineRequest` / `PipelineResponse`
- `StatusResponse` / `HealthResponse`

### 5. Pipeline Tracking
- Active pipeline monitoring
- Progress tracking by stage
- Execution time measurement
- Stage completion/failure tracking

## Usage

### Starting the Server

```bash
# Development mode with auto-reload
python datapizza_api_server.py

# Or with custom settings
uvicorn datapizza_api_server:app --host 0.0.0.0 --port 8080 --reload
```

### Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Get status
curl http://localhost:8000/status

# Execute query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "data_source": "default",
    "max_results": 5
  }'

# Process data
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "data": "Text to process",
    "operation": "chunk",
    "chunk_size": 1000
  }'

# Run pipeline
curl -X POST http://localhost:8000/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "data": "Document text",
    "stages": ["validate", "chunk", "embed"]
  }'
```

### Integration with TypeScript Hooks

The TypeScript hooks in `datapizza-bubblelab-plugin/src/hooks/` can now call this server:

```typescript
// useDatapizzaQuery.ts
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: queryText,
    data_source: dataSource,
    max_results: 10
  })
});

const result = await response.json();
console.log('Query results:', result.results);
```

## Testing

### Test with Python Client

```python
from fastapi.testclient import TestClient
from datapizza_api_server import app

client = TestClient(app)

# Test health
response = client.get('/health')
assert response.status_code == 200
assert response.json()['healthy'] == True

# Test query
response = client.post('/query', json={
    'query': 'test',
    'data_source': 'test'
})
assert response.status_code == 200
assert response.json()['success'] == True
```

### Test with cURL

```bash
# Start server
python datapizza_api_server.py &

# Run tests
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## Configuration

### Environment Variables

Set these before starting the server:

```bash
# Server configuration
export DATAPIZZA_HOST="0.0.0.0"
export DATAPIZZA_PORT="8000"

# DataPizza configuration (if available)
export DATAPIZZA_API_KEY="your-api-key"
export DATAPIZZA_MODEL="gpt-4"
export DATAPIZZA_VECTOR_STORE_URL="http://localhost:19530"
```

### Runtime Configuration

The server can be configured programmatically:

```python
from datapizza_api_server import run_server

run_server(
    host="0.0.0.0",        # Bind to all interfaces
    port=8000,              # Port number
    reload=True,            # Auto-reload on code changes
    log_level="info"         # Logging level
)
```

## DataPizza Integration

### When DataPizza Core is Available

The server will use real DataPizza components:
- `datapizza.agents.Agent` - Multi-agent coordination
- `datapizza.clients.Client` - LLM clients (OpenAI, Anthropic, Google)
- `datapizza.tools.Tool` - Tools (FileSystem, WebSearch, SQL, WebFetch)

### When DataPizza Core is Not Available

The server gracefully degrades:
- Provides simulated/fallback responses
- Maintains full API compatibility
- Returns degradation status in `/status` endpoint
- Frontend continues to work with mock data

## Files Created

1. `datapizza_api_server.py` - Main FastAPI server (650+ lines)
2. `docs/DataPizza/DATAPIZZA_API_SERVER.md` - This documentation

## Dependencies

### Required
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `python-multipart` - Form data parsing

### Optional
- `datapizza` - DataPizza core framework (for real functionality)

Install dependencies:

```bash
pip install fastapi uvicorn pydantic python-multipart
```

## API Documentation

When the server is running, interactive API documentation is available:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Production Deployment

### Using Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY datapizza_api_server.py .

EXPOSE 8000

CMD ["uvicorn", "datapizza_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Using Systemd

```ini
[Unit]
Description=DataPizza API Server
After=network.target

[Service]
Type=simple
User=datapizza
WorkingDirectory=/path/to/datapizza
ExecStart=/usr/bin/python -m uvicorn datapizza_api_server:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

### Using Nginx Reverse Proxy

```nginx
location /api/datapizza {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
}
```

## Security Considerations

1. **API Authentication**: Add API key authentication for production
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **Input Validation**: All inputs are validated via Pydantic models
4. **CORS**: Configure CORS appropriately for production domains
5. **HTTPS**: Use TLS/SSL in production
6. **Logging**: Implement proper request logging for audit trails

## Next Steps

To enhance the DataPizza API server:

1. **Add Authentication**: Implement API key or OAuth2
2. **Rate Limiting**: Add slowapi or similar for rate limiting
3. **Database Integration**: Add PostgreSQL for result caching
4. **WebSocket Support**: Add real-time pipeline progress updates
5. **Batch Processing**: Add batch endpoints for multiple queries
6. **Monitoring**: Add Prometheus metrics export

## Related Documentation

- `docs/DataPizza/DATAPIZZA_INTEGRATION_GUIDE.md` - Integration guide
- `datapizza-bubblelab-plugin/src/hooks/` - TypeScript hooks that call this API
- `INTEGRATION_PROGRESS_REPORT.md` - Overall integration progress

## Verification

Server tested and verified:
- ✅ Health endpoint working
- ✅ Status endpoint working
- ✅ Query endpoint working (with fallback)
- ✅ Graceful degradation when DataPizza unavailable
- ✅ Type-safe Pydantic models
- ✅ CORS middleware enabled
- ✅ Error handling implemented
- ✅ Progress tracking for pipelines
