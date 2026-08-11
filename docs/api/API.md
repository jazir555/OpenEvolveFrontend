# Hybrid PES System API Documentation

Complete API reference for the OpenEvolve LoongFlow PES hybrid system.

## Table of Contents

1. [LoongFlow Adapter API](#loongflow-adapter-api)
2. [OpenEvolve Adapter API](#openevolve-adapter-api)
3. [Event Bus API](#event-bus-api)
4. [Schema Transformation APIs](#schema-transformation-apis)
5. [Workflow APIs](#workflow-apis)
6. [Health Check APIs](#health-check-apis)
7. [Error Handling](#error-handling)
8. [Rate Limiting](#rate-limiting)
9. [Authentication](#authentication)

## LoongFlow Adapter API

**Base URL**: `http://loongflow-adapter:8040`

**Content-Type**: `application/json`

### Workflow Management

#### POST /workflow/execute

Execute a Plan-Execute-Summarize (PES) workflow.

**Request Body**:
```typescript
{
  query: string;              // The problem statement or query
  maxIterations?: number;     // Maximum PES iterations (1-100, default: 5)
  enableCheckpointing?: boolean; // Enable state checkpointing (default: true)
  timeout?: number;           // Workflow timeout in milliseconds (default: 300000)
  metadata?: Record<string, any>; // Optional metadata
}
```

**Example Request**:
```bash
curl -X POST http://loongflow-adapter:8040/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Solve the traveling salesman problem for 10 cities",
    "maxIterations": 5,
    "enableCheckpointing": true
  }'
```

**Response** (200 OK):
```typescript
{
  workflowId: string;        // UUID v4
  status: "running" | "completed" | "failed";
  plan: {
    steps: Array<{
      stepId: string;
      description: string;
      estimatedDuration: number;
      dependencies: string[];
    }>;
    estimatedDuration: number;
    resources: string[];
  };
  execution: {
    executionId: string;
    status: string;
    startTime: string;       // ISO 8601 UTC
    results: Array<{
      stepId: string;
      output: any;
      duration: number;
    }>;
  };
  summary: {
    summaryId: string;
    summary: string;
    insights: string[];
    recommendations: string[];
  };
  metadata: Record<string, any>;
}
```

**Error Responses**:
- `400 Bad Request`: Invalid request body
- `422 Unprocessable Entity`: Schema validation failed
- `500 Internal Server Error`: Workflow execution failed
- `503 Service Unavailable`: LoongFlow core is down (circuit breaker open)

#### GET /workflow/{workflowId}

Get the status and results of a workflow.

**Path Parameters**:
- `workflowId` (string, required): Workflow UUID

**Example Request**:
```bash
curl http://loongflow-adapter:8040/workflow/abc-123-def
```

**Response** (200 OK):
```typescript
{
  workflowId: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  progress: number;           // 0.0 to 1.0
  currentPhase: "planning" | "execution" | "summarization";
  plan: Plan | null;
  execution: Execution | null;
  summary: Summary | null;
  error: string | null;       // Present if status is "failed"
  startTime: string;          // ISO 8601 UTC
  endTime: string | null;      // ISO 8601 UTC
  duration: number | null;     // Milliseconds
}
```

**Error Responses**:
- `404 Not Found`: Workflow not found
- `500 Internal Server Error`: Failed to retrieve workflow

#### GET /workflow

List workflows with optional filtering.

**Query Parameters**:
- `status` (string, optional): Filter by status (e.g., "running", "completed")
- `limit` (number, optional): Maximum number of results (default: 50)
- `offset` (number, optional): Pagination offset (default: 0)
- `sortBy` (string, optional): Sort field (default: "startTime")
- `sortOrder` (string, optional): "asc" or "desc" (default: "desc")

**Example Request**:
```bash
curl "http://loongflow-adapter:8040/workflow?status=completed&limit=10"
```

**Response** (200 OK):
```typescript
{
  workflows: Array<{
    workflowId: string;
    status: string;
    progress: number;
    startTime: string;
    endTime: string | null;
    duration: number | null;
  }>;
  total: number;
  limit: number;
  offset: number;
}
```

#### DELETE /workflow/{workflowId}

Cancel a running workflow.

**Path Parameters**:
- `workflowId` (string, required): Workflow UUID

**Example Request**:
```bash
curl -X DELETE http://loongflow-adapter:8040/workflow/abc-123-def
```

**Response** (200 OK):
```typescript
{
  workflowId: string;
  status: "cancelled";
  message: string;
}
```

**Error Responses**:
- `404 Not Found`: Workflow not found
- `409 Conflict`: Workflow already completed or failed
- `500 Internal Server Error`: Cancellation failed

### Checkpoint Management

#### GET /checkpoint/{workflowId}

Get checkpoint data for a workflow.

**Path Parameters**:
- `workflowId` (string, required): Workflow UUID

**Example Request**:
```bash
curl http://loongflow-adapter:8040/checkpoint/abc-123-def
```

**Response** (200 OK):
```typescript
{
  workflowId: string;
  checkpointId: string;
  timestamp: string;           // ISO 8601 UTC
  state: {
    phase: string;
    completedSteps: string[];
    partialResults: any;
  };
}
```

**Error Responses**:
- `404 Not Found`: No checkpoint found for workflow
- `500 Internal Server Error`: Failed to retrieve checkpoint

#### POST /checkpoint/{workflowId}/restore

Restore a workflow from checkpoint.

**Path Parameters**:
- `workflowId` (string, required): Workflow UUID

**Example Request**:
```bash
curl -X POST http://loongflow-adapter:8040/checkpoint/abc-123-def/restore
```

**Response** (200 OK):
```typescript
{
  workflowId: string;
  status: "running";
  restoredFromCheckpoint: string;
  message: string;
}
```

## OpenEvolve Adapter API

**Base URL**: `http://openevolve-adapter:8000`

**Content-Type**: `application/json`

### Evolutionary Optimization

#### POST /evolution/run

Run an evolutionary optimization session.

**Request Body**:
```typescript
{
  initialPrompt: string;      // Starting prompt to evolve
  generations: number;        // Number of generations (1-100)
  populationSize: number;     // Population size per generation (10-100)
  mutationRate: number;       // Mutation rate (0.0-1.0)
  crossoverRate?: number;     // Crossover rate (0.0-1.0, default: 0.7)
  selectionStrategy?: "tournament" | "roulette" | "rank"; // Selection strategy
  fitnessFunction?: string;   // Custom fitness function (optional)
  enableCheckpointing?: boolean; // Enable checkpointing (default: true)
  timeout?: number;           // Timeout in milliseconds (default: 600000)
}
```

**Example Request**:
```bash
curl -X POST http://openevolve-adapter:8000/evolution/run \
  -H "Content-Type: application/json" \
  -d '{
    "initialPrompt": "Write a Python function to sort a list",
    "generations": 10,
    "populationSize": 20,
    "mutationRate": 0.1
  }'
```

**Response** (200 OK):
```typescript
{
  runId: string;              // UUID v4
  status: "running" | "completed" | "failed";
  generation: number;         // Current generation
  bestPrompt: string;         // Best prompt found so far
  bestFitness: number;        // Fitness score of best prompt
  population: Array<{
    promptId: string;
    prompt: string;
    fitness: number;
    generation: number;
  }>;
  metrics: {
    averageFitness: number;
    fitnessImprovement: number;
    diversity: number;
    convergenceRate: number;
  };
  startTime: string;          // ISO 8601 UTC
  estimatedEndTime: string | null; // ISO 8601 UTC
}
```

#### GET /evolution/run/{runId}

Get the status and results of an evolution run.

**Path Parameters**:
- `runId` (string, required): Evolution run UUID

**Response** (200 OK):
```typescript
{
  runId: string;
  status: "running" | "completed" | "failed";
  generation: number;
  totalGenerations: number;
  bestPrompt: string;
  bestFitness: number;
  population: Array<Prompt>;
  fitnessHistory: Array<{
    generation: number;
    bestFitness: number;
    averageFitness: number;
  }>;
  metrics: EvolutionMetrics;
  startTime: string;
  endTime: string | null;
  duration: number | null;
}
```

#### POST /evolution/run/{runId}/cancel

Cancel an evolution run.

**Path Parameters**:
- `runId` (string, required): Evolution run UUID

**Response** (200 OK):
```typescript
{
  runId: string;
  status: "cancelled";
  generation: number;
  message: string;
}
```

### Knowledge Aggregation

#### POST /knowledge/aggregate

Aggregate knowledge from multiple sources.

**Request Body**:
```typescript
{
  sources: Array<{
    type: "database" | "api" | "file" | "graph";
    connection: string;       // Connection string or endpoint
    query?: string;           // Optional query
    credentials?: Record<string, string>; // Optional credentials
  }>;
  aggregationStrategy?: "union" | "intersection" | "weighted";
  enableDeduplication?: boolean; // Enable entity deduplication (default: true)
  enableReasoning?: boolean; // Enable reasoning engine (default: false)
}
```

**Example Request**:
```bash
curl -X POST http://openevolve-adapter:8000/knowledge/aggregate \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [
      {
        "type": "database",
        "connection": "postgresql://localhost:5432/kb"
      },
      {
        "type": "api",
        "connection": "http://knowledge-api:8080"
      }
    ],
    "enableDeduplication": true
  }'
```

**Response** (200 OK):
```typescript
{
  aggregationId: string;
  status: "completed";
  knowledgeGraph: {
    nodes: Array<{
      entityId: string;
      entityType: string;
      attributes: Record<string, any>;
      confidence: number;
      sources: string[];
    }>;
    edges: Array<{
      from: string;
      to: string;
      relationType: string;
      confidence: number;
      sources: string[];
    }>;
  };
  statistics: {
    totalNodes: number;
    totalEdges: number;
    deduplicationCount: number;
    aggregationTime: number; // Milliseconds
  };
}
```

### Integration Coordination

#### POST /integration/coordinate

Coordinate multiple adapters for a complex workflow.

**Request Body**:
```typescript
{
  workflow: {
    name: string;
    description?: string;
    steps: Array<{
      stepId: string;
      adapter: string;        // Adapter name (e.g., "loongflow", "openevolve")
      action: string;         // Action to perform
      parameters: Record<string, any>;
      dependencies: string[]; // Step IDs this step depends on
    }>;
  };
  timeout?: number;           // Overall timeout (default: 300000)
  enableCheckpointing?: boolean;
  enableParallelism?: boolean; // Enable parallel execution (default: true)
}
```

**Example Request**:
```bash
curl -X POST http://openevolve-adapter:8000/integration/coordinate \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "name": "Hybrid PES Evolution",
      "steps": [
        {
          "stepId": "plan",
          "adapter": "loongflow",
          "action": "executePESWorkflow",
          "parameters": {
            "query": "Design an algorithm"
          },
          "dependencies": []
        },
        {
          "stepId": "evolve",
          "adapter": "openevolve",
          "action": "evolveSystem",
          "parameters": {
            "initialPrompt": "${plan.summary}"
          },
          "dependencies": ["plan"]
        }
      ]
    }
  }'
```

**Response** (200 OK):
```typescript
{
  coordinationId: string;
  status: "running" | "completed" | "failed";
  workflow: Workflow;
  results: Array<{
    stepId: string;
    status: "completed" | "failed";
    output: any;
    duration: number;
    error?: string;
  }>;
  startTime: string;
  endTime: string | null;
  duration: number | null;
}
```

## Event Bus API

**Base URL**: `http://event-bus:6379` (Redis)

**Protocol**: Redis Pub/Sub

### Publish Events

#### Command: PUBLISH

Publish an event to a channel.

**Format**:
```bash
PUBLISH <channel> <payload>
```

**Example**:
```bash
redis-cli PUBLISH loongflow:pes.completed '{
  "workflowId": "abc-123",
  "status": "completed",
  "timestamp": "2024-02-22T12:00:00Z"
}'
```

**Event Payload Schema**:
```typescript
{
  eventId: string;            // UUID v4
  eventType: string;          // Event type (e.g., "pes.completed")
  timestamp: string;          // ISO 8601 UTC
  correlationId?: string;     // Correlation ID for tracing
  source: string;             // Source service
  data: Record<string, any>;  // Event-specific data
}
```

### Subscribe to Events

#### Command: SUBSCRIBE

Subscribe to events from a channel.

**Format**:
```bash
SUBSCRIBE <channel>
```

**Example**:
```bash
redis-cli SUBSCRIBE loongflow:pes.completed
```

**Response** (async):
```
1) "message"
2) "loongflow:pes.completed"
3) "{\"eventId\":\"...\",\"eventType\":\"pes.completed\",...}"
```

#### Command: PSUBSCRIBE

Subscribe to events using pattern matching.

**Format**:
```bash
PSUBSCRIBE <pattern>
```

**Example**:
```bash
# Subscribe to all loongflow events
redis-cli PSUBSCRIBE loongflow:*

# Subscribe to all completed events
redis-cli PSUBSCRIBE *.completed
```

### Dead Letter Queue

#### Command: LPUSH

Push failed event to Dead Letter Queue.

**Format**:
```bash
LPUSH dlq:events <payload>
```

**Example**:
```bash
redis-cli LPUSH dlq:events '{
  "eventId": "failed-event-123",
  "eventType": "pes.failed",
  "error": "Execution timeout",
  "originalEvent": {...}
}'
```

#### Command: LRANGE

Get failed events from DLQ.

**Format**:
```bash
LRANGE dlq:events 0 -1
```

#### Command: LPOP

Pop and process a failed event from DLQ.

**Format**:
```bash
LPOP dlq:events
```

## Schema Transformation APIs

Schema transformation functions for converting between different system formats.

### LoongFlow → Canonical

**Function**: `toCanonicalWorkflow(response: LoongFlowResponse): CanonicalWorkflow`

```typescript
import { toCanonicalWorkflow } from '@/schemas/loongflow-canonical';

const loongflowResponse = {
  workflow_id: "abc-123",
  plan_steps: [...],
  exec_results: [...]
};

const canonical = toCanonicalWorkflow(loongflowResponse);
// Returns: { workflowId: "abc-123", planSteps: [...], executionResults: [...] }
```

### Canonical → OpenEvolve

**Function**: `toEvolutionRequest(workflow: CanonicalWorkflow): EvolutionRequest`

```typescript
import { toEvolutionRequest } from '@/schemas/hybrid-pes-evolution-canonical';

const canonicalWorkflow = {
  workflowId: "abc-123",
  summary: { summary: "Algorithm design..." }
};

const evolutionRequest = toEvolutionRequest(canonicalWorkflow);
// Returns: { initialPrompt: "Algorithm design...", ... }
```

### Cross-System Transformation

**Function**: `transformSchema(source: any, targetSchema: ZodSchema): any`

```typescript
import { transformSchema } from '@/schemas';
import { OpenEvolveResponseSchema } from '@/schemas/openevolve-canonical';

const sourceData = { /* ... */ };
const transformed = transformSchema(sourceData, OpenEvolveResponseSchema);
```

## Workflow APIs

### Hybrid Workflow Execution

**Endpoint**: `POST /workflow/hybrid`

Execute a hybrid PES + Evolution workflow.

**Request Body**:
```typescript
{
  query: string;
  pesConfig?: {
    maxIterations?: number;
    enableCheckpointing?: boolean;
  };
  evolutionConfig?: {
    generations?: number;
    populationSize?: number;
    mutationRate?: number;
  };
  enableAdversarialTesting?: boolean;
  timeout?: number;
}
```

**Response**:
```typescript
{
  workflowId: string;
  status: string;
  phases: {
    planning: { status: string; result: any };
    evolution: { status: string; result: any };
    adversarialTesting?: { status: string; result: any };
    summarization: { status: string; result: any };
  };
  finalResult: any;
  startTime: string;
  endTime: string | null;
  duration: number | null;
}
```

### Workflow Templates

**Endpoint**: `GET /workflow/templates`

List available workflow templates.

**Response**:
```typescript
{
  templates: Array<{
    templateId: string;
    name: string;
    description: string;
    category: string;
    parameters: Array<{
      name: string;
      type: string;
      required: boolean;
      defaultValue?: any;
    }>;
  }>;
}
```

**Endpoint**: `POST /workflow/template/{templateId}/execute`

Execute a workflow from a template.

## Health Check APIs

All services expose health check endpoints.

### GET /health

Standard health check.

**Response** (200 OK):
```typescript
{
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;          // ISO 8601 UTC
  service: string;            // Service name
  version: string;            // Service version
  uptime: number;             // Uptime in seconds
  checks: {
    database: { status: "healthy" | "unhealthy"; latency?: number };
    redis: { status: "healthy" | "unhealthy"; latency?: number };
    externalAPIs: Array<{
      name: string;
      status: "healthy" | "unhealthy";
      latency?: number;
    }>;
  };
}
```

**Example**:
```bash
curl http://loongflow-adapter:8040/health
curl http://openevolve-adapter:8000/health
```

### GET /health/ready

Readiness probe (for Kubernetes).

**Response**: 200 OK if ready, 503 Service Unavailable if not ready.

### GET /health/live

Liveness probe (for Kubernetes).

**Response**: 200 OK if alive, 503 Service Unavailable if not alive.

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```typescript
{
  error: {
    code: string;             // Error code (e.g., "VALIDATION_ERROR")
    message: string;          // Human-readable error message
    details?: Record<string, any>; // Additional error details
    correlationId: string;    // For tracing
    timestamp: string;        // ISO 8601 UTC
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 400 | Request body validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict (e.g., already exists) |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service down or circuit breaker open |

### Example Error Response

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request body",
    "details": {
      "errors": [
        {
          "field": "query",
          "message": "Query is required and must be at least 1 character"
        }
      ]
    },
    "correlationId": "abc-123-def",
    "timestamp": "2024-02-22T12:00:00Z"
  }
}
```

## Rate Limiting

API requests are rate-limited to prevent abuse.

### Rate Limit Headers

All API responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1708600800
```

### Rate Limit Rules

| Endpoint | Limit | Window |
|----------|-------|--------|
| POST /workflow/execute | 10 requests | 1 minute |
| GET /workflow/{id} | 100 requests | 1 minute |
| POST /evolution/run | 5 requests | 1 minute |
| Other endpoints | 100 requests | 1 minute |

### Handling Rate Limits

When rate limit is exceeded:

```typescript
try {
  await api.executeWorkflow(request);
} catch (error) {
  if (error.response?.status === 429) {
    const retryAfter = error.response.headers['retry-after'];
    await sleep(retryAfter * 1000);
    // Retry request
  }
}
```

## Authentication

### OIDC Authentication (Production)

**Authorization Header**:
```
Authorization: Bearer <token>
```

**Example**:
```bash
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  http://loongflow-adapter:8040/workflow/execute
```

### Header Injection (Development)

For development, headers can be injected by an auth sidecar:

```
X-Remote-User: user123
X-Remote-Email: user@example.com
X-Remote-Groups: admin,developer
```

### Token Validation

Tokens are validated on each request:

```typescript
try {
  const decoded = await jwtVerifier.verify(token);
  // User is authenticated
} catch (error) {
  // Return 401 Unauthorized
}
```

---

**API Version**: 1.0.0
**Last Updated**: 2024-02-22
**Base URLs**:
- LoongFlow Adapter: `http://loongflow-adapter:8040`
- OpenEvolve Adapter: `http://openevolve-adapter:8000`
- Event Bus (Redis): `redis://event-bus:6379`
