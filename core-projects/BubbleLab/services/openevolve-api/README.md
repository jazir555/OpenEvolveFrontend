# OpenEvolve API Service

FastAPI service for OpenEvolve workflow execution with evolutionary algorithms, adversarial testing, and sovereign decomposition.

## Architecture

### Core Components

1. **Evolution Engine** (`core/evolution.py`)
   - Evolutionary code generation and optimization
   - Supports population-based search with LLM guidance
   - Fitness-based refinement and selection

2. **Adversarial Engine** (`core/adversarial.py`)
   - Red team testing and vulnerability discovery
   - Multiple attack types: fuzzing, prompt injection, code injection, SQL injection, XSS
   - Circuit breaker pattern for failure isolation

3. **Sovereign Engine** (`core/sovereign.py`)
   - Problem decomposition into parallel sub-problems
   - Hierarchical decomposition with configurable depth
   - Solution synthesis and verification

### API Layer

1. **Workflow API** (`api/workflows.py`)
   - CRUD operations for workflow definitions
   - Support for evolution, adversarial, and sovereign workflows
   - Parameter validation per workflow type

2. **Execution API** (`api/execution.py`)
   - Start, pause, resume, cancel executions
   - Real-time status tracking
   - Log retrieval with filtering

3. **Team API** (`api/teams.py`)
   - AI agent team management
   - Multi-agent orchestration

4. **Gauntlet API** (`api/gauntlets.py`)
   - Solution validation workflows
   - Multi-round evaluation

### Services

- **Execution Manager** (`services/execution_service.py`)
  - Thread pool for background execution
  - Status tracking and persistence
  - Pause/Resume/Cancel capabilities
  - Structured logging collection

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

## Configuration

All configuration via environment variables (Law of Configuration Explicitness):

```bash
# Server Configuration
export OPENEVOLVE_HOST="0.0.0.0"
export OPENEVOLVE_PORT="8000"
export OPENEVOLVE_WORKERS="4"

# LLM Configuration
export OPENEVOLVE_LLM_PROVIDER="openai"  # or "anthropic", "local"
export OPENEVOLVE_LLM_API_KEY="sk-..."
export OPENEVOLVE_LLM_MODEL="gpt-4"
export OPENEVOLVE_LLM_BASE_URL="https://api.openai.com/v1"

# Execution Configuration
export OPENEVOLVE_MAX_WORKERS="5"
export OPENEVOLVE_EXECUTION_TIMEOUT="300"  # seconds
export OPENEVOLVE_LOG_LEVEL="INFO"

# Storage Configuration (Future)
export OPENEVOLVE_DATABASE_URL="sqlite:///openevolve.db"
export OPENEVOLVE_CACHE_TTL="3600"
```

## Running the Service

### Development

```bash
# Run with uvicorn for development
uvicorn openevolve_api.main:app --reload --host 0.0.0.0 --port 8000

# Or using the Makefile
make dev
```

### Production

```bash
# Run with gunicorn
gunicorn openevolve_api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --access-logfile - \
  --error-logfile -

# Or using Docker
docker-compose up -d
```

## API Documentation

Once running, access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Usage Examples

### Create Evolution Workflow

```bash
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "REST API Generator",
    "description": "Generate REST API for user management",
    "workflow_type": "evolution",
    "parameters": {
      "max_iterations": 50,
      "population_size": 25,
      "temperature": 0.7,
      "top_p": 1.0
    }
  }'
```

### Execute Workflow

```bash
curl -X POST http://localhost:8000/api/executions/workflows/{workflow_id}/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Create a REST API with user CRUD operations, authentication, and rate limiting",
    "context": "Use FastAPI framework, PostgreSQL database, JWT authentication"
  }'
```

### Get Execution Status

```bash
curl http://localhost:8000/api/executions/workflows/{workflow_id}/executions/{execution_id}
```

### Get Execution Logs

```bash
curl "http://localhost:8000/api/executions/workflows/{workflow_id}/executions/{execution_id}/logs?since=2024-01-01T00:00:00Z"
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "openevolve-api",
  "version": "0.1.0",
  "features": {
    "evolution": true,
    "adversarial": true,
    "sovereign": true
  }
}
```

### Metrics

Structured JSON logging with:
- `correlation_id`: Request tracing
- `execution_id`: Execution tracking
- `timestamp`: UTC ISO-8601
- `level`: Log level
- All relevant context

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=openevolve_api --cov-report=html

# Run specific test
pytest tests/test_evolution_engine.py
```

## Architecture Principles

Following CLAUDE.md federation constitution:

1. **Air Gap Law**: No direct imports from core-projects
2. **Runtime Truth**: Probe before implementing
3. **Untouchable DB**: Read-only state (writes only for backups)
4. **Idempotency**: All operations safe to retry
5. **Configuration Explicitness**: All config via env vars
6. **Law of UTC**: All timestamps in UTC

## Troubleshooting

### Execution Stuck in QUEUED

Check thread pool capacity:
```bash
curl http://localhost:8000/health
```

Increase workers:
```bash
export OPENEVOLVE_MAX_WORKERS="10"
```

### High Memory Usage

Reduce concurrent executions:
```bash
export OPENEVOLVE_MAX_WORKERS="2"
```

### LLM API Errors

Check configuration:
```bash
echo $OPENEVOLVE_LLM_API_KEY
echo $OPENEVOLVE_LLM_BASE_URL
```

## Booting the service

The package directory is named `openevolve-api` (hyphen), so boot it as a
module path rather than relying on an installed top-level package:

```bash
cd core-projects/BubbleLab/services/openevolve-api
uvicorn openevolve_api.main:app --host 127.0.0.1 --port 8000
# control plane + docs
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/docs
```

> The `openevolve_api` import name resolves because `main.py` uses a
> relative/absolute import fallback, and the real OpenEvolve engine is reached
> through `core/openevolve_bridge.py` (offline deterministic mock LLM by
> default; no API keys needed).

### New UI-facing route groups (real data)

These were added to stop the BubbleLab client (`apps/bubble-studio`) 404ing on
them. All numbers are derived from the live in-memory run registry
(`api/openevolve_v1.RUNS`) or a real parameter catalog — no fabricated values.

- `GET  /api/parameters/schema`   — OpenEvolve evolution parameter catalog
  (`max_iterations`, `population_size`, `temperature`, `elite_ratio`, ...).
- `GET  /api/parameters/defaults` — default values for each parameter.
- `GET  /api/parameters/categories` — parameter groupings.
- `POST /api/parameters/validate` — type/range/option validation of submitted params.
- `GET  /api/monitoring/dashboard` — health + aggregate run stats (total/completed/failed, last best score, uptime).
- `GET  /api/monitoring/{alerts,services,logs,metrics,health}` — service health surfaces.
- `GET/POST/PUT/DELETE /api/validation/rules` — in-memory content-validation rules.
- `POST /api/validation/run` — run rules against content (`ValidationRunResult`).
- `POST /api/validation/compliance` — compliance check (`ComplianceCheckResult`).
- `GET  /api/analytics/performance-metrics` — per-run performance metrics.
- `GET  /api/analytics/workflow-metrics` — per-run workflow metrics.
- `GET  /api/analytics/knowledge-stats` — knowledge-base stats (empty state until wired).
- `GET  /api/statistics` — workflow/team/gauntlet counts.

### Boot smoke test

`scripts/smoke_boot.py` launches the real uvicorn server, waits for `/health`,
orchestrates a real (offline mock) evolution run, and polls until `completed`,
asserting `result.best_code` is non-empty. Exit code is non-zero on failure.

```bash
python scripts/smoke_boot.py
```

## License

MIT

## Contributing

1. Follow CLAUDE.md principles
2. Add structured logging
3. Write tests
4. Update documentation
