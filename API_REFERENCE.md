# OpenEvolve API Reference

**REST API & GraphQL Reference** | **License: Apache 2.0**

---

## REST API

Base URL: `http://localhost:8000`

### Health Endpoints

#### GET /health
Check system health.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-02-02T14:30:00",
  "services": {
    "rest_api": {"status": "healthy"},
    "graphql": {"status": "healthy"},
    "event_bus": {"status": "healthy"}
  }
}
```

#### GET /ready
Readiness probe for Kubernetes.

**Response**:
```json
{
  "ready": true
}
```

---

### Workflow Endpoints

#### GET /api/v1/workflows
List all workflows.

**Query Parameters**:
- `status` (optional): Filter by status
- `limit` (optional): Maximum results (default: 100)
- `offset` (optional): Pagination offset

**Response**:
```json
{
  "workflows": [
    {
      "id": "wf_001",
      "status": "completed",
      "created_at": "2026-02-02T14:00:00",
      "updated_at": "2026-02-02T14:30:00"
    }
  ],
  "total": 1
}
```

#### POST /api/v1/workflows
Create a new workflow.

**Request Body**:
```json
{
  "problem_description": "Optimize neural network",
  "strategy": "hybrid",
  "parameters": {
    "max_depth": 5,
    "population_size": 50
  }
}
```

**Response**:
```json
{
  "workflow_id": "wf_002",
  "status": "created",
  "created_at": "2026-02-02T14:35:00"
}
```

#### GET /api/v1/workflows/{id}
Get workflow details.

**Response**:
```json
{
  "id": "wf_001",
  "status": "completed",
  "problem_description": "Optimize neural network",
  "stages": [
    {
      "name": "decomposition",
      "status": "completed",
      "result": {"subproblems": 5}
    }
  ],
  "result": {
    "fitness": 0.95,
    "solution": "optimized_architecture"
  }
}
```

#### DELETE /api/v1/workflows/{id}
Delete a workflow.

**Response**:
```json
{
  "message": "Workflow deleted"
}
```

---

### Decomposition Endpoints

#### POST /api/v1/decompose
Decompose a problem.

**Request Body**:
```json
{
  "problem_description": "Optimize neural network architecture",
  "strategy": "hybrid",
  "parameters": {
    "max_depth": 3,
    "min_complexity": 0.2
  }
}
```

**Response**:
```json
{
  "decomposition_id": "dec_001",
  "subproblems": [
    {
      "id": "sub_001",
      "description": "Optimize layer configuration",
      "complexity": 0.7
    }
  ],
  "entanglement_matrix": {
    "sub_001": ["sub_002"],
    "sub_002": ["sub_001"]
  }
}
```

#### GET /api/v1/decompose/{id}/status
Get decomposition status.

**Response**:
```json
{
  "id": "dec_001",
  "status": "completed",
  "progress": 100,
  "subproblems_count": 5
}
```

---

### Knowledge Endpoints

#### GET /api/v1/knowledge/patterns
List extracted patterns.

**Query Parameters**:
- `type` (optional): Pattern type filter
- `min_confidence` (optional): Minimum confidence (0-1)

**Response**:
```json
{
  "patterns": [
    {
      "id": "pat_001",
      "type": "sequence",
      "description": "Common stage sequence",
      "confidence": 0.95,
      "occurrences": 10
    }
  ]
}
```

#### GET /api/v1/knowledge/artifacts
List knowledge artifacts.

**Response**:
```json
{
  "artifacts": [
    {
      "id": "art_001",
      "name": "Neural Network Strategy",
      "type": "strategy",
      "validity_score": 0.92
    }
  ]
}
```

#### POST /api/v1/knowledge/query
Query for applicable knowledge.

**Request Body**:
```json
{
  "problem_description": "Optimize neural network",
  "min_validity": 0.8
}
```

**Response**:
```json
{
  "artifacts": [
    {
      "id": "art_001",
      "name": "Neural Network Strategy",
      "relevance": 0.95
    }
  ]
}
```

---

### Service Management Endpoints

#### GET /api/v1/services
List all services.

**Response**:
```json
{
  "services": [
    {
      "name": "rest_api",
      "status": "running",
      "port": 8000,
      "uptime_seconds": 3600
    }
  ]
}
```

#### POST /api/v1/services/{name}/start
Start a service.

**Response**:
```json
{
  "message": "Service started",
  "status": "running"
}
```

#### POST /api/v1/services/{name}/stop
Stop a service.

**Response**:
```json
{
  "message": "Service stopped",
  "status": "stopped"
}
```

#### GET /api/v1/services/{name}/health
Get service health.

**Response**:
```json
{
  "name": "rest_api",
  "status": "healthy",
  "checks": {
    "database": "ok",
    "cache": "ok"
  }
}
```

---

### Metrics Endpoints

#### GET /metrics
Prometheus metrics.

**Response** (text/plain):
```
# HELP openevolve_workflows_total Total workflows
# TYPE openevolve_workflows_total counter
openevolve_workflows_total 42

# HELP openevolve_request_duration_seconds Request duration
# TYPE openevolve_request_duration_seconds histogram
openevolve_request_duration_seconds_bucket{le="0.1"} 5
```

---

## GraphQL API

Endpoint: `http://localhost:8001/graphql`

### Queries

#### Get Workflows
```graphql
query GetWorkflows($limit: Int = 10) {
  workflows(limit: $limit) {
    id
    status
    createdAt
    stages {
      name
      status
    }
  }
}
```

#### Get Workflow Details
```graphql
query GetWorkflow($id: ID!) {
  workflow(id: $id) {
    id
    status
    problemDescription
    stages {
      name
      status
      result
    }
    result
  }
}
```

#### Get Patterns
```graphql
query GetPatterns($type: String) {
  patterns(type: $type) {
    id
    type
    description
    confidence
    occurrences
  }
}
```

#### Get Artifacts
```graphql
query GetArtifacts {
  artifacts {
    id
    name
    type
    validityScore
  }
}
```

#### Query Knowledge
```graphql
query QueryKnowledge($problem: String!) {
  knowledge(problemDescription: $problem) {
    artifacts {
      id
      name
      relevance
    }
  }
}
```

#### Get Services
```graphql
query GetServices {
  services {
    name
    status
    port
    uptime
  }
}
```

---

### Mutations

#### Create Workflow
```graphql
mutation CreateWorkflow($input: WorkflowInput!) {
  createWorkflow(input: $input) {
    id
    status
    createdAt
  }
}
```

**Variables**:
```json
{
  "input": {
    "problemDescription": "Optimize neural network",
    "strategy": "hybrid",
    "parameters": {
      "maxDepth": 5
    }
  }
}
```

#### Decompose Problem
```graphql
mutation DecomposeProblem($input: DecompositionInput!) {
  decomposeProblem(input: $input) {
    id
    subproblems {
      id
      description
      complexity
    }
    entanglementMatrix
  }
}
```

#### Start Service
```graphql
mutation StartService($name: String!) {
  startService(name: $name) {
    name
    status
    message
  }
}
```

#### Stop Service
```graphql
mutation StopService($name: String!) {
  stopService(name: $name) {
    name
    status
    message
  }
}
```

---

### Subscriptions

#### Workflow Updates
```graphql
subscription WorkflowUpdates($workflowId: ID!) {
  workflowUpdates(workflowId: $workflowId) {
    type
    payload
    timestamp
  }
}
```

#### Service Status Changes
```graphql
subscription ServiceStatusChanges {
  serviceStatusChanges {
    name
    status
    timestamp
  }
}
```

---

## Error Responses

### Standard Error Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": {
      "field": "problem_description",
      "issue": "Required field missing"
    }
  }
}
```

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Input validation failed | 400 |
| `NOT_FOUND` | Resource not found | 404 |
| `INTERNAL_ERROR` | Internal server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |

---

## Authentication

### API Key

Include API key in header:
```
X-API-Key: your-api-key
```

### Bearer Token

Include JWT token in header:
```
Authorization: Bearer <token>
```

---

## Rate Limiting

Default limits:
- 100 requests per minute per IP
- 10 burst requests

Headers included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1643723400
```

---

## Examples

### cURL Examples

**Create Workflow**:
```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "problem_description": "Optimize neural network",
    "strategy": "hybrid"
  }'
```

**Query Knowledge**:
```bash
curl -X POST http://localhost:8000/api/v1/knowledge/query \
  -H "Content-Type: application/json" \
  -d '{
    "problem_description": "neural network optimization"
  }'
```

**GraphQL Query**:
```bash
curl -X POST http://localhost:8001/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ workflows { id status } }"
  }'
```

### Python Examples

**Using httpx**:
```python
import httpx

async with httpx.AsyncClient() as client:
    # Create workflow
    response = await client.post(
        "http://localhost:8000/api/v1/workflows",
        json={
            "problem_description": "Optimize neural network",
            "strategy": "hybrid"
        }
    )
    workflow = response.json()
    
    # Query knowledge
    response = await client.post(
        "http://localhost:8000/api/v1/knowledge/query",
        json={"problem_description": "neural network"}
    )
    knowledge = response.json()
```

---

**For more details, see Swagger UI at http://localhost:8000/docs**
