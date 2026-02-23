# LoongFlow HTTP API

FastAPI wrapper that provides HTTP endpoints for LoongFlow's Plan-Execute-Summarize (PES) evolution system.

## Overview

LoongFlow is primarily a CLI tool, but this API wrapper exposes its capabilities via REST endpoints for integration with the OpenEvolve federation.

## Quick Start

### Local Development

```bash
# Set required environment variables
export LOONGFLOW_LLM_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements-api.txt

# Run the API server
python api_server.py
```

The API will be available at `http://localhost:8000`

### Docker

```bash
# Set required environment variables
export LOONGFLOW_LLM_API_KEY="your-api-key-here"

# Build and run
docker-compose -f docker-compose.loongflow-core.yml up -d

# View logs
docker-compose -f docker-compose.loongflow-core.yml logs -f loongflow-core

# Stop
docker-compose -f docker-compose.loongflow-core.yml down
```

## API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "loongflow-api",
  "version": "1.0.0",
  "timestamp": "2026-02-22T12:34:56.789Z"
}
```

### Start Evolution

```http
POST /api/v1/evolve
Content-Type: application/json

{
  "name": "my-evolution",
  "task": "Solve the packing problem for unit square",
  "max_generations": 10,
  "population_size": 50,
  "config": {
    "custom_param": "value"
  }
}
```

**Response:**
```json
{
  "evolution_id": "evo_1a2b3c4d5e6f",
  "status": "PENDING",
  "message": "Evolution started successfully"
}
```

### Get Evolution Status

```http
GET /api/v1/status/{evolution_id}
```

**Response:**
```json
{
  "evolution_id": "evo_1a2b3c4d5e6f",
  "name": "my-evolution",
  "status": "RUNNING",
  "current_generation": 5,
  "max_generations": 10,
  "best_fitness": 0.5,
  "created_at": "2026-02-22T12:00:00Z",
  "updated_at": "2026-02-22T12:05:00Z"
}
```

Status values: `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`

### Get Solution

```http
GET /api/v1/solutions/{evolution_id}
```

**Response:**
```json
{
  "evolution_id": "evo_1a2b3c4d5e6f",
  "name": "my-evolution",
  "solution": "# Optimal solution code...",
  "fitness": 0.95,
  "generations_completed": 10,
  "metadata": {}
}
```

### List Evolutions

```http
GET /api/v1/evolutions?status=COMPLETED&limit=100
```

**Response:**
```json
{
  "evolutions": [
    {
      "evolution_id": "evo_1a2b3c4d5e6f",
      "name": "my-evolution",
      "status": "COMPLETED",
      ...
    }
  ],
  "count": 1
}
```

### Delete Evolution

```http
DELETE /api/v1/evolutions/{evolution_id}
```

## Environment Variables

### Required

- `LOONGFLOW_LLM_API_KEY`: API key for LLM provider

### Optional

- `LOONGFLOW_API_HOST`: Host to bind to (default: `0.0.0.0`)
- `LOONGFLOW_API_PORT`: Port to bind to (default: `8000`)
- `LOONGFLOW_API_WORKERS`: Number of worker processes (default: `1`)
- `LOONGFLOW_LLM_URL`: LLM API endpoint (default: `https://api.openai.com/v1`)
- `LOONGFLOW_LLM_MODEL`: Model name (default: `gpt-4`)
- `LOONGFLOW_LLM_TEMPERATURE`: Temperature (default: `0.7`)
- `LOONGFLOW_LLM_MAX_TOKENS`: Max tokens (default: `2000`)
- `LOONGFLOW_ENABLE_CHECKPOINTING`: Enable checkpointing (default: `true`)
- `LOONGFLOW_CHECKPOINT_DIR`: Checkpoint directory (default: `/app/checkpoints`)
- `LOONGFLOW_EVAL_TIMEOUT`: Evaluator timeout in seconds (default: `300`)

## Architecture

### Current Implementation (Phase 1)

The current implementation is a **first pass** that:

1. ✅ Provides HTTP endpoints for evolution management
2. ✅ Uses background tasks for async evolution
3. ⚠️ **Simulates** evolution progress (not yet integrated with actual LoongFlow PES logic)
4. ⚠️ Returns placeholder solutions

### Future Integration (Phase 2)

To fully integrate with LoongFlow's PES system, the following work is needed:

1. **Refactor `GeneralPESAgent`** to support programmatic calls (not just CLI)
2. **Hook into progress callbacks** to track generation count and fitness in real-time
3. **Extract final solutions** from LoongFlow's internal state
4. **Support multiple agent types** (General, Math, ML)
5. **Add streaming/websocket support** for real-time progress updates

### Integration Challenges

The main challenge is that LoongFlow was designed as a CLI tool with these characteristics:

- **Blocking execution**: Runs to completion before returning
- **File-based config**: Requires YAML config files
- **Logging to stdout**: Progress via print statements, not callbacks
- **No internal state API**: No way to query running evolution state

To properly integrate, we need to:

1. Refactor `BasePESRunner` to support both CLI and API modes
2. Add progress callbacks to `PESAgent`
3. Create a state manager that exposes evolution state
4. Handle long-running evolutions gracefully

## Design Principles

This API follows the **Federation Constitution** (CLAUDE.md):

1. **Law of Configuration Explicitness**: All config via environment variables; service crashes if required vars missing
2. **Law of Runtime Truth**: Health checks verify actual API functionality, not just process running
3. **Law of Idempotency**: Safe to retry operations; duplicate requests handled gracefully
4. **Law of UTC**: All timestamps in UTC ISO-8601 format
5. **Structured Logging**: JSON Lines format with correlation IDs

## Testing

### Manual Testing

```bash
# Start the server
python api_server.py

# In another terminal, test the endpoints
curl http://localhost:8000/health

# Start an evolution
curl -X POST http://localhost:8000/api/v1/evolve \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-evolution",
    "task": "Test task",
    "max_generations": 5
  }'

# Check status
curl http://localhost:8000/api/v1/status/evo_XXXXXX

# Get solution (when complete)
curl http://localhost:8000/api/v1/solutions/evo_XXXXXX
```

### Automated Testing

```bash
# Run API tests (to be implemented)
pytest tests/api/
```

## Production Considerations

### Scalability

- Use multiple workers (`LOONGFLOW_API_WORKERS > 1`) for horizontal scaling
- Add Redis for distributed evolution state management
- Implement rate limiting to prevent abuse

### Reliability

- Add circuit breakers for LLM API calls
- Implement proper error handling and retries
- Add dead letter queues for failed evolutions

### Security

- Add authentication (OIDC/OAuth2)
- Add authorization (who can start evolutions?)
- Sanitize task inputs to prevent injection attacks
- Use secrets management for API keys

### Observability

- Structured logging with correlation IDs
- Metrics for evolution success rates, durations
- Distributed tracing for request flows

## Contributing

When extending this API:

1. Follow the existing patterns for request/response models
2. Add proper error handling with HTTP status codes
3. Include correlation IDs in all log messages
4. Update this README with new endpoints
5. Add tests for new functionality

## License

Same as LoongFlow project.
