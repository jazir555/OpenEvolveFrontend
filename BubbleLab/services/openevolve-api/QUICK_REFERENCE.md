# OpenEvolve API - Quick Reference

## Fast Start

```bash
# Install
pip install -r requirements.txt

# Run
make dev
# OR
uvicorn openevolve_api.main:app --reload --host 0.0.0.0 --port 8000

# Access
open http://localhost:8000/docs
```

## API Endpoints at a Glance

### Workflows
```
POST   /api/workflows                        Create workflow
GET    /api/workflows                        List workflows (paginated)
GET    /api/workflows/{id}                   Get workflow
PUT    /api/workflows/{id}                   Update workflow
DELETE /api/workflows/{id}                   Delete workflow
```

### Executions
```
POST   /api/executions/workflows/{id}/execute                     Start execution
GET    /api/executions/workflows/{id}/executions/{exec_id}        Get status
POST   /api/executions/workflows/{id}/executions/{exec_id}/pause  Pause
POST   /api/executions/workflows/{id}/executions/{exec_id}/resume Resume
POST   /api/executions/workflows/{id}/executions/{exec_id}/cancel Cancel
GET    /api/executions/workflows/{id}/executions/{exec_id}/logs   Get logs
GET    /api/executions/workflows/{id}/executions                  List executions
```

### Teams
```
POST   /api/teams           Create team
GET    /api/teams           List teams
GET    /api/teams/{id}      Get team
```

### Gauntlets
```
POST   /api/gauntlets       Create gauntlet
GET    /api/gauntlets       List gauntlets
GET    /api/gauntlets/{id}  Get gauntlet
```

### Health
```
GET    /health              Health check
GET    /                    API info
```

## Common Usage Patterns

### 1. Create and Execute Evolution Workflow

```bash
# Create workflow
WORKFLOW_ID=$(curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API Generator",
    "description": "Generate REST API",
    "workflow_type": "evolution",
    "parameters": {
      "max_iterations": 50,
      "temperature": 0.7
    }
  }' | jq -r '.id')

# Execute
EXECUTION_ID=$(curl -X POST http://localhost:8000/api/executions/workflows/$WORKFLOW_ID/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Create a user management API",
    "context": "Use FastAPI and PostgreSQL"
  }' | jq -r '.execution_id')

# Check status
curl http://localhost:8000/api/executions/workflows/$WORKFLOW_ID/executions/$EXECUTION_ID | jq
```

### 2. Run Adversarial Testing

```bash
# Create adversarial workflow
WORKFLOW_ID=$(curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Security Test",
    "description": "Test login endpoint",
    "workflow_type": "adversarial",
    "parameters": {
      "attack_types": ["prompt_injection", "sql_injection"],
      "rounds": 3
    }
  }' | jq -r '.id')

# Execute
curl -X POST http://localhost:8000/api/executions/workflows/$WORKFLOW_ID/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Test this login endpoint: POST /auth/login",
    "context": "Accepts email and password"
  }' | jq
```

### 3. Decompose Complex Problem

```bash
# Create sovereign workflow
WORKFLOW_ID=$(curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "name": "System Builder",
    "description": "Build e-commerce system",
    "workflow_type": "sovereign",
    "parameters": {
      "decomposition_depth": 3,
      "parallel_subproblems": 5
    }
  }' | jq -r '.id')

# Execute
curl -X POST http://localhost:8000/api/executions/workflows/$WORKFLOW_ID/execute \
  -H "Content-Type: application/json" \
  -d '{
    "problem_statement": "Build a complete e-commerce platform with user auth, product catalog, shopping cart, checkout, and payment processing"
  }' | jq
```

## Workflow Parameters Quick Reference

### Evolution Parameters
```json
{
  "max_iterations": 100,      // 1-200 (default: 100)
  "population_size": 50,      // 1-100 (default: 50)
  "temperature": 0.7,         // 0.0-2.0 (default: 0.7)
  "top_p": 1.0,              // 0.0-1.0 (default: 1.0)
  "max_tokens": 4096,        // 1-100000 (default: 4096)
  "frequency_penalty": 0.0,   // -2.0-2.0 (default: 0.0)
  "presence_penalty": 0.0,    // -2.0-2.0 (default: 0.0)
  "seed": 42                 // -1-999999 (default: 42, -1=random)
}
```

### Adversarial Parameters
```json
{
  "test_cases": [],          // Custom test cases
  "attack_types": [          // Attack types to test
    "fuzzing",
    "prompt_injection",
    "code_injection",
    "sql_injection",
    "xss"
  ],
  "rounds": 3                // 1-10 (default: 3)
}
```

### Sovereign Parameters
```json
{
  "decomposition_depth": 3,           // 1-10 (default: 3)
  "parallel_subproblems": 5,          // 1-20 (default: 5)
  "verification_strictness": "standard" // "lenient"|"standard"|"strict"
}
```

## Execution Status Values

- `queued` - Waiting in queue
- `running` - Currently executing
- `paused` - Paused by user
- `completed` - Successfully finished
- `failed` - Failed with error
- `cancelled` - Cancelled by user

## Environment Variables

```bash
# Server
export OPENEVOLVE_HOST="0.0.0.0"
export OPENEVOLVE_PORT="8000"
export OPENEVOLVE_WORKERS="4"

# LLM
export OPENEVOLVE_LLM_PROVIDER="openai"
export OPENEVOLVE_LLM_API_KEY="sk-..."
export OPENEVOLVE_LLM_MODEL="gpt-4"

# Execution
export OPENEVOLVE_MAX_WORKERS="5"
export OPENEVOLVE_EXECUTION_TIMEOUT="300"
```

## Docker Commands

```bash
# Build and run
make docker-build
make docker-up

# View logs
make docker-logs

# Stop
make docker-down
```

## Testing Commands

```bash
# Run tests
make test

# With coverage
make test-cov

# Lint
make lint

# Format
make format
```

## Monitoring

```bash
# Health check
curl http://localhost:8000/health

# Response
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

## Troubleshooting

### Execution stuck in QUEUED
```bash
# Increase workers
export OPENEVOLVE_MAX_WORKERS="10"
```

### LLM API errors
```bash
# Check configuration
echo $OPENEVOLVE_LLM_API_KEY
echo $OPENEVOLVE_LLM_BASE_URL
```

### High memory usage
```bash
# Reduce concurrent executions
export OPENEVOLVE_MAX_WORKERS="2"
```

## File Structure

```
openevolve-api/
├── api/
│   ├── workflows.py       # Workflow CRUD
│   ├── execution.py       # Execution management
│   ├── teams.py           # Team management
│   └── gauntlets.py       # Gauntlet management
├── core/
│   ├── evolution.py       # Evolution engine
│   ├── adversarial.py     # Adversarial engine
│   └── sovereign.py       # Sovereign engine
├── services/
│   └── execution_service.py  # Background execution
├── models/
│   └── __init__.py        # Pydantic models
├── main.py                # FastAPI app
├── Dockerfile             # Production image
├── docker-compose.yml     # Docker compose
├── Makefile               # Dev commands
└── requirements.txt       # Dependencies
```

## Key Features

✅ Evolutionary code generation
✅ Adversarial security testing
✅ Problem decomposition
✅ Background execution
✅ Pause/Resume/Cancel
✅ Real-time logs
✅ Structured logging
✅ Thread-safe operations
✅ Circuit breaker pattern
✅ Health checks
✅ OpenAPI documentation

## Documentation Links

- Full README: `README.md`
- API Docs: `API_DOCUMENTATION.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
