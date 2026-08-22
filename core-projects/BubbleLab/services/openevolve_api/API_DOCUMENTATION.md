# OpenEvolve API Documentation

Complete API reference for the OpenEvolve workflow execution service.

## Base URL

```
Development: http://localhost:8000
Production: https://openevolve.example.com
```

## Authentication

Currently no authentication required. Future versions will support:
- API Key authentication
- OAuth2 Bearer tokens
- JWT tokens

## Common Headers

```http
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>  # Future
```

## Common Response Codes

- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `202 Accepted` - Request accepted for processing
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Workflows API

### Create Workflow

Create a new workflow definition.

**Endpoint:** `POST /api/workflows`

**Request Body:**

```json
{
  "name": "string (1-100 chars)",
  "description": "string (1-500 chars)",
  "workflow_type": "evolution|adversarial|sovereign",
  "parameters": {}
}
```

**Evolution Parameters:**
```json
{
  "max_iterations": 100,      // 1-200
  "population_size": 50,       // 1-100
  "temperature": 0.7,          // 0.0-2.0
  "top_p": 1.0,               // 0.0-1.0
  "max_tokens": 4096,         // 1-100000
  "frequency_penalty": 0.0,    // -2.0-2.0
  "presence_penalty": 0.0,     // -2.0-2.0
  "seed": 42                  // -1-999999 (-1 for random)
}
```

**Adversarial Parameters:**
```json
{
  "test_cases": [],
  "attack_types": ["fuzzing", "prompt_injection", "code_injection"],
  "rounds": 3                 // 1-10
}
```

**Sovereign Parameters:**
```json
{
  "decomposition_depth": 3,           // 1-10
  "parallel_subproblems": 5,          // 1-20
  "verification_strictness": "standard" // "lenient"|"standard"|"strict"
}
```

**Response:** `201 Created`

```json
{
  "id": "wf_20240127_123456_789012",
  "name": "REST API Generator",
  "description": "Generate REST API for user management",
  "workflow_type": "evolution",
  "parameters": {},
  "status": "draft",
  "created_at": "2024-01-27T12:34:56.789012Z",
  "updated_at": "2024-01-27T12:34:56.789012Z"
}
```

### List Workflows

Retrieve paginated list of workflows.

**Endpoint:** `GET /api/workflows`

**Query Parameters:**
- `page` (integer, default: 1) - Page number
- `page_size` (integer, default: 10, max: 100) - Items per page
- `workflow_type` (string, optional) - Filter by workflow type
- `status` (string, optional) - Filter by status

**Response:** `200 OK`

```json
{
  "workflows": [],
  "total": 100,
  "page": 1,
  "page_size": 10
}
```

### Get Workflow

Retrieve specific workflow by ID.

**Endpoint:** `GET /api/workflows/{workflow_id}`

**Response:** `200 OK`

```json
{
  "id": "wf_20240127_123456_789012",
  "name": "REST API Generator",
  "description": "Generate REST API for user management",
  "workflow_type": "evolution",
  "parameters": {},
  "status": "ready",
  "created_at": "2024-01-27T12:34:56.789012Z",
  "updated_at": "2024-01-27T12:34:56.789012Z"
}
```

### Update Workflow

Update existing workflow.

**Endpoint:** `PUT /api/workflows/{workflow_id}`

**Request Body:**
```json
{
  "name": "string (optional)",
  "description": "string (optional)",
  "parameters": {}  // Merges with existing
}
```

**Response:** `200 OK`

```json
{
  "id": "wf_20240127_123456_789012",
  "name": "Updated Name",
  "description": "Updated description",
  "workflow_type": "evolution",
  "parameters": {},
  "status": "ready",
  "created_at": "2024-01-27T12:34:56.789012Z",
  "updated_at": "2024-01-27T13:00:00.000000Z"
}
```

### Delete Workflow

Delete a workflow.

**Endpoint:** `DELETE /api/workflows/{workflow_id}`

**Response:** `204 No Content`

---

## Execution API

### Start Execution

Start execution of a workflow.

**Endpoint:** `POST /api/executions/workflows/{workflow_id}/execute`

**Request Body:**
```json
{
  "problem_statement": "Create a REST API with user CRUD operations",
  "context": "Use FastAPI framework, PostgreSQL database"  // optional
}
```

**Response:** `202 Accepted`

```json
{
  "execution_id": "exec_20240127_123456_789012",
  "workflow_id": "wf_20240127_123456_789012",
  "status": "queued",
  "progress": 0.0,
  "started_at": "2024-01-27T12:34:56.789012Z",
  "completed_at": null,
  "result": null,
  "error": null
}
```

### Get Execution Status

Get current status of an execution.

**Endpoint:** `GET /api/executions/workflows/{workflow_id}/executions/{execution_id}`

**Response:** `200 OK`

```json
{
  "execution_id": "exec_20240127_123456_789012",
  "workflow_id": "wf_20240127_123456_789012",
  "status": "running",
  "progress": 0.45,
  "started_at": "2024-01-27T12:34:56.789012Z",
  "completed_at": null,
  "result": null,
  "error": null
}
```

**Status Values:**
- `queued` - Execution queued, waiting to start
- `running` - Currently executing
- `paused` - Paused by user
- `completed` - Successfully completed
- `failed` - Failed with error
- `cancelled` - Cancelled by user

### Pause Execution

Pause a running execution.

**Endpoint:** `POST /api/executions/workflows/{workflow_id}/executions/{execution_id}/pause`

**Response:** `200 OK`

```json
{
  "execution_id": "exec_20240127_123456_789012",
  "workflow_id": "wf_20240127_123456_789012",
  "status": "paused",
  "progress": 0.45,
  "started_at": "2024-01-27T12:34:56.789012Z",
  "completed_at": null,
  "result": null,
  "error": null
}
```

### Resume Execution

Resume a paused execution.

**Endpoint:** `POST /api/executions/workflows/{workflow_id}/executions/{execution_id}/resume`

**Response:** `200 OK`

```json
{
  "execution_id": "exec_20240127_123456_789012",
  "workflow_id": "wf_20240127_123456_789012",
  "status": "running",
  "progress": 0.45,
  "started_at": "2024-01-27T12:34:56.789012Z",
  "completed_at": null,
  "result": null,
  "error": null
}
```

### Cancel Execution

Cancel an execution.

**Endpoint:** `POST /api/executions/workflows/{workflow_id}/executions/{execution_id}/cancel`

**Response:** `200 OK`

```json
{
  "execution_id": "exec_20240127_123456_789012",
  "workflow_id": "wf_20240127_123456_789012",
  "status": "cancelled",
  "progress": 0.45,
  "started_at": "2024-01-27T12:34:56.789012Z",
  "completed_at": "2024-01-27T12:40:00.000000Z",
  "result": null,
  "error": null
}
```

### Get Execution Logs

Retrieve logs for an execution.

**Endpoint:** `GET /api/executions/workflows/{workflow_id}/executions/{execution_id}/logs`

**Query Parameters:**
- `since` (datetime, optional) - Filter logs since timestamp

**Response:** `200 OK`

```json
{
  "logs": [
    {
      "timestamp": "2024-01-27T12:34:56.789012Z",
      "level": "info",
      "message": "Starting evolution workflow execution"
    },
    {
      "timestamp": "2024-01-27T12:35:00.000000Z",
      "level": "info",
      "message": "Generating initial solution"
    }
  ],
  "total": 2,
  "since": "2024-01-27T12:34:00.000000Z"
}
```

### List Workflow Executions

List all executions for a workflow.

**Endpoint:** `GET /api/executions/workflows/{workflow_id}/executions`

**Query Parameters:**
- `limit` (integer, default: 10, max: 100) - Maximum executions to return

**Response:** `200 OK`

```json
[
  {
    "execution_id": "exec_20240127_123456_789012",
    "workflow_id": "wf_20240127_123456_789012",
    "status": "completed",
    "progress": 1.0,
    "started_at": "2024-01-27T12:34:56.789012Z",
    "completed_at": "2024-01-27T12:40:00.000000Z",
    "result": {},
    "error": null
  }
]
```

---

## Teams API

### Create Team

Create a new AI agent team.

**Endpoint:** `POST /api/teams`

**Request Body:**
```json
{
  "name": "string (1-100 chars)",
  "description": "string",
  "members": [
    {
      "name": "string",
      "role": "string",
      "model": "string",
      "temperature": 0.7,
      "max_tokens": 4096
    }
  ]
}
```

**Response:** `201 Created`

```json
{
  "id": "team_20240127_123456_789012",
  "name": "Development Team",
  "description": "Team for code generation tasks",
  "members": [],
  "created_at": "2024-01-27T12:34:56.789012Z"
}
```

### List Teams

**Endpoint:** `GET /api/teams`

**Response:** `200 OK`

```json
{
  "teams": [],
  "total": 10
}
```

### Get Team

**Endpoint:** `GET /api/teams/{team_id}`

**Response:** `200 OK`

```json
{
  "id": "team_20240127_123456_789012",
  "name": "Development Team",
  "description": "Team for code generation tasks",
  "members": [],
  "created_at": "2024-01-27T12:34:56.789012Z"
}
```

---

## Gauntlets API

### Create Gauntlet

Create a new validation gauntlet.

**Endpoint:** `POST /api/gauntlets`

**Request Body:**
```json
{
  "name": "string (1-100 chars)",
  "description": "string",
  "rounds": [
    {
      "name": "string",
      "quorum_threshold": 0.7,
      "confidence_threshold": 0.8,
      "evaluation_type": "string"
    }
  ]
}
```

**Response:** `201 Created`

```json
{
  "id": "gauntlet_20240127_123456_789012",
  "name": "Code Quality Gauntlet",
  "description": "Multi-round code quality validation",
  "rounds": [],
  "created_at": "2024-01-27T12:34:56.789012Z"
}
```

### List Gauntlets

**Endpoint:** `GET /api/gauntlets`

**Response:** `200 OK`

```json
{
  "gauntlets": [],
  "total": 5
}
```

### Get Gauntlet

**Endpoint:** `GET /api/gauntlets/{gauntlet_id}`

**Response:** `200 OK`

```json
{
  "id": "gauntlet_20240127_123456_789012",
  "name": "Code Quality Gauntlet",
  "description": "Multi-round code quality validation",
  "rounds": [],
  "created_at": "2024-01-27T12:34:56.789012Z"
}
```

---

## Health Check

### Health Status

Check service health and available features.

**Endpoint:** `GET /health`

**Response:** `200 OK`

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

### Root Endpoint

Get API information and documentation links.

**Endpoint:** `GET /`

**Response:** `200 OK`

```json
{
  "service": "OpenEvolve Workflow API",
  "version": "0.1.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

## Error Responses

All endpoints may return error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

Common errors:
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Rate Limiting

Currently no rate limiting. Future versions will implement:
- 100 requests per minute per IP
- 1000 requests per hour per IP

Rate limit headers will be included:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1706360400
```

---

## CORS

Enabled for:
- `http://localhost:5173` (Vite dev server)
- `http://localhost:3000` (React dev server)

Additional origins can be configured via environment variables.
