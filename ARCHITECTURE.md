# Hybrid PES System Architecture

Complete architectural documentation for the OpenEvolve LoongFlow PES hybrid system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [API Endpoints](#api-endpoints)
5. [Event Types](#event-types)
6. [Schema Definitions](#schema-definitions)
7. [Integration Patterns](#integration-patterns)
8. [Deployment Architecture](#deployment-architecture)
9. [Security Architecture](#security-architecture)
10. [Failure Management](#failure-management)

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Application Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Web UI     │  │   CLI Tool   │  │  API Client  │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                 │                 │                      │
│         └─────────────────┼─────────────────┘                      │
│                           ▼                                        │
│                  ┌──────────────┐                                  │
│                  │ API Gateway  │                                  │
│                  │ (Optional)   │                                  │
│                  └──────┬───────┘                                  │
└──────────────────────────┼────────────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────────────┐
│                           ▼                                        │
│                    ┌──────────────┐                                │
│                    │   Event Bus  │                                │
│                    │   (Redis)    │                                │
│                    └──────┬───────┘                                │
│                           │                                        │
│         ┌─────────────────┼─────────────────┐                     │
│         │                 │                 │                     │
│         ▼                 ▼                 ▼                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ LoongFlow    │  │ OpenEvolve   │  │ Other        │           │
│  │ Adapter      │  │ Adapter      │  │ Adapters     │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                     │
│         ▼                 ▼                 ▼                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ LoongFlow    │  │ OpenEvolve   │  │ Core         │           │
│  │ Core         │  │ Core         │  │ Systems      │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Canonical Schemas (Zod)                      │  │
│  │  - Type Validation  - Transformation  - Enforcement      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Orchestration Layer                          │  │
│  │  - Circuit Breakers  - Retry Logic  - DLQ                │  │
│  │  - Correlation ID    - Logging       - Monitoring        │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Separation of Concerns**: Each component has a single responsibility
2. **Loose Coupling**: Components communicate via events, not direct calls
3. **Strong Typing**: All data validated against Zod schemas
4. **Fault Tolerance**: Circuit breakers and retries prevent cascading failures
5. **Observability**: Structured logging with correlation IDs throughout
6. **Configuration Explicitness**: No magic defaults, all config via env vars

## Component Architecture

### 1. LoongFlow Adapter

**Purpose**: Integrate LoongFlow's Plan-Execute-Summarize capabilities

**Location**: `glue/adapters/loongflow-adapter/`

**Components**:
```
loongflow-adapter/
├── src/
│   ├── adapter.ts          # Main adapter class
│   ├── index.ts            # Exports
│   └── types.ts            # TypeScript types
├── tests/
│   ├── contract.test.ts    # Contract tests
│   ├── fixtures/           # Test data
│   └── jest.config.js      # Jest config
├── probes/                 # API verification scripts
├── dist/                   # Compiled JavaScript
├── package.json
├── tsconfig.json
└── Dockerfile
```

**Key Features**:
- HTTP client with Axios
- Timeout and retry logic
- Schema validation on all requests/responses
- Circuit breaker integration
- Health check endpoint

**API Interface**:
```typescript
class LoongFlowAdapter {
  // Execute PES workflow
  executePESWorkflow(request: PESWorkflowRequest): Promise<PESWorkflowResponse>

  // Get workflow status
  getWorkflowStatus(workflowId: string): Promise<WorkflowStatus>

  // List workflows
  listWorkflows(filter?: WorkflowFilter): Promise<WorkflowList>

  // Cancel workflow
  cancelWorkflow(workflowId: string): Promise<void>

  // Health check
  healthCheck(): Promise<HealthStatus>
}
```

### 2. OpenEvolve Adapter

**Purpose**: Coordinate OpenEvolve's evolutionary optimization

**Location**: `glue/adapters/openevolve-adapter/`

**Components**:
```
openevolve-adapter/
├── src/
│   ├── adapter.ts                  # Main adapter
│   ├── integration-coordinator.ts  # Multi-system coordination
│   ├── knowledge-aggregator.ts     # Knowledge consolidation
│   ├── workflow-orchestrator.ts    # Workflow management
│   └── index.ts
├── tests/
│   └── contract.test.ts
├── probes/
├── dist/
├── package.json
└── Dockerfile
```

**Key Features**:
- Multi-system workflow coordination
- Knowledge aggregation from multiple sources
- Evolutionary optimization orchestration
- Integration with event bus
- Distributed tracing

**API Interface**:
```typescript
class OpenEvolveAdapter {
  // Run evolutionary optimization
  evolveSystem(request: EvolutionRequest): Promise<EvolutionResult>

  // Aggregate knowledge
  aggregateKnowledge(sources: KnowledgeSource[]): Promise<KnowledgeGraph>

  // Coordinate integrations
  coordinateIntegrations(workflow: Workflow): Promise<CoordinationResult>

  // Health check
  healthCheck(): Promise<HealthStatus>
}
```

### 3. Canonical Schemas

**Purpose**: Define canonical data models for type-safe communication

**Location**: `glue/schemas/`

**Schema Categories**:

#### PES Schemas (`pes-canonical.ts`)
```typescript
// Core PES schemas
const PlanSchema = z.object({
  steps: z.array(StepSchema),
  dependencies: z.array(DependencySchema),
  estimatedDuration: z.number(),
  resources: z.array(ResourceSchema)
});

const ExecutionSchema = z.object({
  planId: z.string(),
  status: z.enum(['pending', 'running', 'completed', 'failed']),
  results: z.array(ResultSchema),
  metrics: ExecutionMetricsSchema
});

const SummarySchema = z.object({
  executionId: z.string(),
  summary: z.string(),
  insights: z.array(InsightSchema),
  recommendations: z.array(RecommendationSchema)
});
```

#### LoongFlow Schemas (`loongflow-canonical.ts`)
```typescript
// LoongFlow-specific schemas
const LoongFlowWorkflowRequestSchema = z.object({
  query: z.string().min(1),
  maxIterations: z.number().min(1).max(100).optional(),
  enableCheckpointing: z.boolean().optional(),
  timeout: z.number().optional()
});

const LoongFlowWorkflowResponseSchema = z.object({
  workflowId: z.string().uuid(),
  status: WorkflowStatusSchema,
  plan: PlanSchema,
  execution: ExecutionSchema,
  summary: SummarySchema,
  metadata: z.record(z.any())
});
```

#### Evolution Schemas (`hybrid-pes-evolution-canonical.ts`)
```typescript
// Hybrid evolution schemas
const EvolutionRequestSchema = z.object({
  initialPrompt: z.string(),
  generations: z.number().min(1),
  populationSize: z.number().min(1),
  mutationRate: z.number().min(0).max(1),
  selectionStrategy: SelectionStrategySchema
});

const EvolutionResultSchema = z.object({
  generation: z.number(),
  bestPrompt: z.string(),
  fitness: z.number(),
  population: z.array(PromptSchema),
  metrics: EvolutionMetricsSchema
});
```

### 4. Event Bus

**Purpose**: Enable event-driven communication between components

**Location**: `glue/orchestration/event-bus.ts`

**Implementations**:
1. **In-Memory Event Bus**: For single-process scenarios
2. **Redis Event Bus**: For distributed scenarios

**Event Bus Interface**:
```typescript
interface IEventBus {
  // Publish event
  publish(eventName: string, data: any): Promise<void>;

  // Subscribe to event
  subscribe(eventName: string, handler: EventHandler): Promise<UnsubscribeFunction>;

  // Unsubscribe from event
  unsubscribe(eventName: string, handler: EventHandler): Promise<void>;

  // Start event bus
  start(): Promise<void>;

  // Stop event bus
  stop(): Promise<void>;
}
```

**Event Types**:
```typescript
// PES Events
'pes.workflow.started'
'pes.workflow.completed'
'pes.workflow.failed'
'pes.plan.generated'
'pes.execution.completed'
'pes.summary.generated'

// Evolution Events
'evolution.started'
'evolution.completed'
'evolution.failed'
'evolution.generation.completed'
'evolution.best.solution.found'

// Hybrid Events
'hybrid.workflow.started'
'hybrid.pes.phase.completed'
'hybrid.evolution.phase.completed'
'hybrid.workflow.completed'

// System Events
'system.health.check'
'system.error'
'system.warning'
```

### 5. Orchestration Layer

**Purpose**: Coordinate workflows across multiple adapters

**Location**: `glue/orchestration/`

**Components**:
- **Event Bus**: Event-driven communication
- **Dead Letter Queue**: Failed event handling
- **Circuit Breaker**: Fault tolerance
- **Correlation Tracker**: Request tracing
- **Retry Logic**: Automatic retry with exponential backoff

**Circuit Breaker**:
```typescript
interface CircuitBreakerConfig {
  threshold: number;      // Failures before opening
  timeout: number;        // Milliseconds to wait before trying again
  halfOpenAttempts: number; // Attempts in half-open state
}

enum CircuitBreakerState {
  CLOSED = 'closed',     // Normal operation
  OPEN = 'open',         // Circuit is open, blocking requests
  HALF_OPEN = 'half_open' // Testing if service has recovered
}
```

## Data Flow

### 1. PES Workflow Flow

```
User Request
    │
    ▼
┌──────────────────┐
│ API Gateway      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ LoongFlow        │
│ Adapter          │
└────────┬─────────┘
         │
         ├──► Validate Request (Zod Schema)
         │
         ├──► Call LoongFlow Core API
         │    │
         │    ├──► Plan Generation
         │    ├──► Execution
         │    └──► Summarization
         │
         ├──► Transform Response (Canonical Schema)
         │
         ├──► Publish Event (Event Bus)
         │    │
         │    └──► 'pes.workflow.completed'
         │
         └──► Return Response to User
```

### 2. Hybrid Workflow Flow

```
User Request
    │
    ▼
┌──────────────────┐
│ Workflow         │
│ Orchestrator     │
└────────┬─────────┘
         │
         ├──► Publish 'hybrid.workflow.started'
         │
         ├──► Phase 1: PES Planning
         │    │
         │    ├──► Call LoongFlow Adapter
         │    ├──► Generate Plan
         │    └──► Publish 'hybrid.pes.phase.completed'
         │
         ├──► Phase 2: Evolutionary Optimization
         │    │
         │    ├──► Call OpenEvolve Adapter
         │    ├──► Evolve Solution
         │    └──► Publish 'hybrid.evolution.phase.completed'
         │
         ├──► Phase 3: Aggregation
         │    │
         │    ├──► Aggregate Results
         │    └──► Generate Final Summary
         │
         ├──► Publish 'hybrid.workflow.completed'
         │
         └──► Return Results to User
```

### 3. Event-Driven Flow

```
Service A          Event Bus          Service B
    │                   │                   │
    │───Publish Event──▶│                   │
    │   'task.created'  │                   │
    │                   │───Deliver Event──▶│
    │                   │                   │
    │                   │◀──Acknowledge─────│
    │                   │                   │
    │                   │                   │───Process Task───┐
    │                   │                   │                  │
    │                   │                   │◀──Task Complete──┘
    │                   │                   │
    │◀──Success Event───│                   │
    │   'task.completed'│                   │
```

## API Endpoints

### LoongFlow Adapter API

**Base URL**: `http://loongflow-adapter:8040`

#### POST /workflow/execute
Execute a PES workflow.

**Request**:
```json
{
  "query": "Solve the traveling salesman problem",
  "maxIterations": 5,
  "enableCheckpointing": true
}
```

**Response**:
```json
{
  "workflowId": "uuid-123",
  "status": "running",
  "plan": { ... },
  "execution": { ... },
  "summary": { ... }
}
```

#### GET /workflow/{workflowId}
Get workflow status.

**Response**:
```json
{
  "workflowId": "uuid-123",
  "status": "completed",
  "progress": 1.0,
  "metrics": { ... }
}
```

#### GET /health
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-02-22T12:00:00Z"
}
```

### OpenEvolve Adapter API

**Base URL**: `http://openevolve-adapter:8000`

#### POST /evolution/run
Run evolutionary optimization.

**Request**:
```json
{
  "initialPrompt": "Write a sorting function",
  "generations": 10,
  "populationSize": 20,
  "mutationRate": 0.1
}
```

**Response**:
```json
{
  "runId": "uuid-456",
  "status": "running",
  "currentGeneration": 0,
  "bestFitness": 0.5
}
```

#### POST /knowledge/aggregate
Aggregate knowledge from sources.

**Request**:
```json
{
  "sources": [
    { "type": "database", "connection": "..." },
    { "type": "api", "endpoint": "..." }
  ]
}
```

**Response**:
```json
{
  "knowledgeGraph": { ... },
  "entities": 150,
  "relationships": 300
}
```

#### GET /health
Health check endpoint.

## Schema Definitions

### Complete Schema List

| Schema File | Purpose | Key Types |
|-------------|---------|-----------|
| `pes-canonical.ts` | Core PES data structures | Plan, Execution, Summary |
| `loongflow-canonical.ts` | LoongFlow-specific types | WorkflowRequest, WorkflowResponse |
| `hybrid-pes-evolution-canonical.ts` | Hybrid workflow types | EvolutionRequest, HybridWorkflow |
| `bubblelab-canonical.ts` | BubbleLab integration | Bubble, Scope, Context |
| `ragbits-canonical.ts` | RAGbits integration | Document, RetrievalResult |
| `graphiti-canonical.ts` | Graphiti integration | Graph, Node, Edge |
| `openevolve-canonical.ts` | OpenEvolve types | EvolutionRequest, EvolutionResult |
| `rese-canonical.ts` | RESE pipeline types | Phase, Result, Metrics |

### Schema Validation Example

```typescript
import { LoongFlowWorkflowRequestSchema } from '@/schemas';

// Validate request
const result = LoongFlowWorkflowRequestSchema.safeParse(request);

if (!result.success) {
  console.error('Validation errors:', result.error.errors);
  throw new Error('Invalid request');
}

// Use validated data
const workflowId = await executeWorkflow(result.data);
```

## Integration Patterns

### 1. Anti-Corruption Layer (ACL)

**Problem**: Different systems use different data formats and conventions.

**Solution**: Transform all data to/from canonical schemas.

```typescript
// Source system format (snake_case)
const sourceData = {
  workflow_id: "123",
  plan_steps: [...],
  exec_results: [...]
};

// Transform to canonical (camelCase)
const canonicalData = {
  workflowId: "123",
  planSteps: [...],
  executionResults: [...]
};

// Validate with Zod
const validated = CanonicalSchema.parse(canonicalData);
```

### 2. Circuit Breaker Pattern

**Problem**: Cascading failures when a service is down.

**Solution**: Circuit breaker stops requests to failing services.

```typescript
const circuitBreaker = new CircuitBreaker({
  threshold: 5,
  timeout: 60000,
  halfOpenAttempts: 3
});

// All API calls go through circuit breaker
const result = await circuitBreaker.execute(async () => {
  return await loongFlowAPI.executeWorkflow(request);
});
```

### 3. Retry with Exponential Backoff

**Problem**: Transient failures (network blips).

**Solution**: Retry with increasing delays.

```typescript
const result = await retryWithBackoff(
  async () => await apiCall(),
  {
    maxRetries: 3,
    initialDelay: 1000,
    maxDelay: 10000,
    backoffMultiplier: 2.0
  }
);
```

### 4. Event-Driven Coordination

**Problem**: Tightly coupled integrations.

**Solution**: Publish-subscribe pattern.

```typescript
// Service A publishes event
await eventBus.publish('task.completed', {
  taskId: '123',
  result: { ... }
});

// Service B subscribes to event
await eventBus.subscribe('task.completed', async (event) => {
  await processResult(event.data);
});
```

## Deployment Architecture

### Docker Compose Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Docker Network                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │  loongflow-core          (Port 8050)               │ │
│  │  - LoongFlow PES engine                            │ │
│  │  - Redis for state                                 │ │
│  │  - Volume: loongflow-data                          │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  loongflow-adapter      (Port 8040)                │ │
│  │  - TypeScript adapter                              │ │
│  │  - Health checks                                   │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  openevolve-adapter    (Port 8000)                 │ │
│  │  - Python adapter                                 │ │
│  │  - Integration coordinator                         │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  event-bus (Redis)      (Port 6379)                │ │
│  │  - Event streaming                               │ │
│  │  - Dead letter queue                             │ │
│  │  - Volume: redis-data                            │ │
│  └────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Kubernetes Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                          │
│  Namespace: hybrid-pes                                   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Deployment: loongflow-core                        │ │
│  │  - Replicas: 3                                     │ │
│  │  - Service: loongflow-core (ClusterIP)             │ │
│  │  - Ingress: loongflow.example.com                  │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Deployment: loongflow-adapter                     │ │
│  │  - Replicas: 2                                     │ │
│  │  - Service: loongflow-adapter (ClusterIP)          │ │
│  │  - HPA: CPU > 70%                                  │ │
│  └────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────┐ │
│  │  StatefulSet: redis                                │ │
│  │  - Replicas: 1                                     │ │
│  │  - Service: redis (ClusterIP)                      │ │
│  │  - PVC: redis-data (100Gi)                         │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ConfigMaps:                                            │
│  - loongflow-config                                     │
│  - adapter-config                                       │
│                                                          │
│  Secrets:                                               │
│  - loongflow-api-keys                                   │
│  - redis-password                                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Security Architecture

### 1. Authentication & Authorization

**OIDC Authentication** (Recommended for production):
```typescript
// User authenticates via OIDC provider
const token = await oidcProvider.authenticate(userId, password);

// Token included in requests
const response = await axios.get('/api/workflow', {
  headers: {
    Authorization: `Bearer ${token}`
  }
});
```

**Header Injection** (Fallback for development):
```typescript
// Auth sidecar injects headers
const request = {
  headers: {
    'X-Remote-User': 'user123',
    'X-Remote-Email': 'user@example.com',
    'X-Remote-Groups': 'admin,developer'
  }
};
```

### 2. Network Security

- **Service Mesh**: All inter-service communication via encrypted channels
- **Network Policies**: Kubernetes network policies restrict access
- **Secrets Management**: Kubernetes secrets or external vault (e.g., HashiCorp Vault)

### 3. Audit Logging

```typescript
// All actions logged with correlation ID
logger.info({
  msg: 'Workflow executed',
  correlation_id: ctx.correlationId,
  user_id: ctx.userId,
  workflow_id: workflowId,
  timestamp: new Date().toISOString()
});
```

## Failure Management

### 1. Transient Failures

**Detection**: Network timeout, 5xx errors

**Strategy**: Retry with exponential backoff + jitter

```typescript
const result = await retryWithBackoff(apiCall, {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 10000,
  backoffMultiplier: 2.0,
  jitter: true // Add randomness to prevent thundering herd
});
```

### 2. Logic Failures

**Detection**: 4xx errors, validation failures

**Strategy**: Dead Letter Queue (DLQ)

```typescript
try {
  await processEvent(event);
} catch (error) {
  if (isLogicError(error)) {
    // Send to DLQ for manual inspection
    await deadLetterQueue.add({
      event,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  } else {
    throw error; // Re-throw for retry logic
  }
}
```

### 3. System Failures

**Detection**: Service unresponsive, health check failures

**Strategy**: Circuit Breaker

```typescript
// Circuit breaker opens after threshold failures
// Requests fail fast instead of hanging
const result = await circuitBreaker.execute(apiCall);

if (circuitBreaker.state === CircuitBreakerState.OPEN) {
  // Use fallback or cached data
  return await fallbackStrategy();
}
```

### 4. Failure Recovery

**Automatic Recovery**:
- Circuit breaker closes after timeout
- Retry logic eventually succeeds
- Dead letter queue processed manually

**Manual Recovery**:
- Inspect DLQ messages
- Fix underlying issue
- Replay DLQ events

---

**Document Version**: 1.0
**Last Updated**: 2024-02-22
**Maintainer**: OpenEvolve Federation
