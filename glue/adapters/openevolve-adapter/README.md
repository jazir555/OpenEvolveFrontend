# OpenEvolve Main Orchestration Adapter

The primary orchestration adapter that coordinates all integrated systems within the OpenEvolve federation.

## Overview

The OpenEvolve adapter serves as the central coordination hub for 30+ integrated open-source projects. It provides:

- **Multi-adapter coordination** across Z3, LeanAide, RAGBits, Vector DB, Graphiti, KarateClub, and more
- **Workflow orchestration** for complex multi-step processes
- **Knowledge aggregation** from all sources
- **Circuit breaker resilience** to prevent cascading failures
- **Canonical schema enforcement** (Anti-Corruption Layer)
- **Structured JSON Lines logging** with correlation IDs

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenEvolve Main Adapter                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Anti-Corruption Layer (ACL)                        │ │
│  │  - Canonical schema validation                            │ │
│  │  - Data transformation                                     │ │
│  │  - Contract enforcement                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Integration Coordinator                            │ │
│  │  - Adapter selection                                       │ │
│  │  - Execution planning (parallel/sequential)                │ │
│  │  - Health monitoring                                       │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                  │
│  ┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │   Workflow   │  │     Knowledge    │  │   Circuit       │  │
│  │ Orchestrator │  │    Aggregator    │  │   Breakers      │  │
│  └──────────────┘  └──────────────────┘  └─────────────────┘  │
│         ↓                  ↓                      ↓             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Integrated Adapters                           │ │
│  │  ┌────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │ │
│  │  │ Z3 │ │LeanAide │ │RAGBits │ │VectorDB│ │Graphiti│ ... │ │
│  │  └────┘ └─────────┘ └────────┘ └────────┘ └────────┘     │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Environment Variables

### Required (No Defaults - Fails Fast if Missing)

```bash
# OpenEvolve API endpoint
export OPENEVOLVE_API_URL="http://localhost:8002"

# Request timeout in milliseconds
export TIMEOUT_MS=5000
```

### Optional (With Defaults)

```bash
# Event bus URL for pub/sub patterns
export EVENT_BUS_URL="http://localhost:8087"

# Logging level (debug, info, warn, error)
export LOG_LEVEL="info"

# Coordination timeout
export COORDINATION_TIMEOUT_MS=10000

# Maximum concurrent workflows
export MAX_CONCURRENT_WORKFLOWS=5

# Workflow checkpoint interval
export WORKFLOW_CHECKPOINT_INTERVAL_MS=30000
```

### Adapter Endpoints

```bash
# Individual adapter URLs
export Z3_ADAPTER_URL="http://localhost:8080"
export LEANAIDE_ADAPTER_URL="http://localhost:8081"
export RAGBITS_ADAPTER_URL="http://localhost:8082"
export VECTOR_DB_URL="http://localhost:8083"
export GRAPHITI_ADAPTER_URL="http://localhost:8084"
export KARATECLUB_ADAPTER_URL="http://localhost:8085"
export KNOWLEDGE_ENGINE_URL="http://localhost:8086"
export EVENT_BUS_URL="http://localhost:8087"
```

## Installation

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run tests
npm test
```

## Usage

### Basic Setup

```typescript
import { createOpenEvolveAdapter } from '@openevolve/openevolve-adapter';

// Create adapter instance (validates required env vars)
const adapter = createOpenEvolveAdapter({
  api_url: process.env.OPENEVOLVE_API_URL!, // Required, no default
  timeout_ms: parseInt(process.env.TIMEOUT_MS!), // Required, no default
  log_level: process.env.LOG_LEVEL, // Optional
  circuit_breaker: {
    failureThreshold: 5,
    successThreshold: 2,
    timeout: 60000,
  },
  retry: {
    maxRetries: 3,
    baseDelay: 1000,
    maxDelay: 10000,
    jitter: true,
  },
});
```

### Health Check

```typescript
// Check adapter and integration health
const health = await adapter.healthCheck();

console.log(health.status); // "healthy"
console.log(health.integrations); // Array of integration health statuses
```

### Team Management

```typescript
// Create a new team
const team = await adapter.createTeam({
  name: 'solver-team-1',
  role: 'Blue',
  members: [
    {
      model_id: 'gpt-4',
      api_key: '',
      api_base: 'http://localhost:8001',
      temperature: 0.7,
      max_tokens: 4096,
    },
  ],
  description: 'Math problem solving team',
});

// List all teams
const teams = await adapter.getTeams();

// Get specific team
const teamDetails = await adapter.getTeam('solver-team-1');

// Update team
await adapter.updateTeam('solver-team-1', { ...team, description: 'Updated' });

// Delete team
await adapter.deleteTeam('solver-team-1');
```

### Gauntlet Management

```typescript
// Create a gauntlet
const gauntlet = await adapter.createGauntlet({
  name: 'verification-gauntlet',
  team_name: 'red-team-1',
  rounds: [
    {
      round_number: 1,
      quorum_required_approvals: 2,
      quorum_from_panel_size: 3,
      min_overall_confidence: 0.8,
    },
  ],
});

// List gauntlets
const gauntlets = await adapter.getGauntlets();

// Get gauntlet details
const gauntletDetails = await adapter.getGauntlet('verification-gauntlet');
```

### Workflow Orchestration

```typescript
import { createWorkflowOrchestrator, createIntegrationCoordinator } from '@openevolve/openevolve-adapter';

// Create coordinators
const integrationCoordinator = createIntegrationCoordinator(adapter, 10000);
const workflowOrchestrator = createWorkflowOrchestrator(adapter, integrationCoordinator, 5);

// Execute workflow
const result = await workflowOrchestrator.executeWorkflow({
  workflow: {
    workflow_id: 'math-proof-001',
    name: 'Math Proof Workflow',
    problem_statement: 'Prove the Pythagorean theorem',
    max_refinement_loops: 3,
    auto_approval_enabled: false,
    sub_problems: [
      {
        id: 'sub-1',
        description: 'Define triangle properties',
        dependencies: [],
        solver_team_name: 'blue-team',
        gold_team_gauntlet_name: 'verification-gauntlet',
      },
    ],
  },
  priority: 'high',
  timeout_ms: 300000,
  checkpoint_enabled: true,
  on_progress: (update) => {
    console.log(`Progress: ${update.progress * 100}% - ${update.message}`);
  },
  on_complete: (result) => {
    console.log(`Workflow completed: ${result.status}`);
  },
  on_error: (error) => {
    console.error(`Workflow error: ${error.error_message}`);
  },
});
```

### Knowledge Aggregation

```typescript
import { createKnowledgeAggregator } from '@openevolve/openevolve-adapter';

// Create knowledge aggregator
const knowledgeAggregator = createKnowledgeAggregator(adapter, integrationCoordinator);

// Query knowledge across all sources
const results = await knowledgeAggregator.queryKnowledge({
  query: 'Pythagorean theorem proof techniques',
  domain: 'mathematics',
  problem_type: 'formal_verification',
  max_results: 20,
  similarity_threshold: 0.7,
});

console.log(`Found ${results.total_results} results`);
console.log(`Sources: ${Array.from(results.results_by_source.keys()).join(', ')}`);

// Extract knowledge from workflow
const artifacts = await knowledgeAggregator.extractKnowledge({
  workflow_id: 'math-proof-001',
  extraction_types: ['solution_pattern', 'team_performance', 'gauntlet_effectiveness'],
  domain: 'mathematics',
  problem_type: 'formal_verification',
});

// Build knowledge graph
const graph = await knowledgeAggregator.buildKnowledgeGraph(artifacts);
console.log(`Graph: ${graph.nodes.length} nodes, ${graph.edges.length} edges`);
```

## Coordination Flow

The adapter coordinates integrations through the following flow:

```
1. Request Received
   ↓
2. Validate Against Canonical Schema
   ↓
3. Plan Coordination
   - Select adapters based on problem type
   - Determine execution order (parallel/sequential)
   - Estimate duration
   ↓
4. Check Circuit Breakers
   - If OPEN: Skip or use fallback
   - If CLOSED/HALF_OPEN: Proceed
   ↓
5. Execute with Retry
   - Try request
   - If failure: exponential backoff retry
   - Update circuit breaker state
   ↓
6. Transform Results
   - Convert to canonical schema
   - Validate against schema
   ↓
7. Aggregate & Return
   - Combine results from multiple adapters
   - Apply fusion logic
   - Return canonical format
```

## Adapter Selection Logic

Adapters are selected based on problem type and capabilities:

```typescript
// Problem Type Mapping
formal_verification → [z3, leanaide]
proof_assistant → [leanaide]
retrieval → [ragbits, vectordb]
knowledge_graph → [graphiti, vectordb]
graph_ml → [karateclub, graphiti]
semantic_search → [vectordb, ragbits]
code_analysis → [z3, leanaide, graphiti]

// Capability Mapping
smt_solving → [z3]
tactic_execution → [leanaide]
vector_search → [vectordb]
graph_traversal → [graphiti]
node_embedding → [karateclub]
document_retrieval → [ragbits]
```

## Circuit Breaker States

```
CLOSED (Normal)
  ↓ Failure threshold reached
OPEN (Failing - reject requests)
  ↓ Timeout elapsed
HALF_OPEN (Testing recovery)
  ↓ Success threshold reached → CLOSED
  ↓ Failure encountered → OPEN
```

## Logging Format

All logs are structured as JSON Lines:

```json
{
  "timestamp": "2025-02-03T12:34:56.789Z",
  "level": "info",
  "message": "Team created",
  "service": "openevolve-adapter",
  "correlation_id": "abc-123-def",
  "source_service": "openevolve-adapter",
  "target_service": "openevolve-api",
  "team_name": "solver-team-1",
  "role": "Blue"
}
```

## Testing

### Contract Tests

Validate API contracts:

```bash
npm run test:contract
```

### Integration Tests

Test adapter coordination:

```bash
npm run test:integration
```

### Full Test Suite

```bash
npm test
```

### Coverage Report

```bash
npm run test:coverage
```

## Probes

Before running operations, validate that systems are available:

```bash
# Check OpenEvolve API
./probes/check_api.sh

# Check all integrations
./probes/check_integrations.sh

# Check workflow execution
./probes/check_workflows.sh
```

## Idempotency

All operations are idempotent and safe to retry:

- **Create**: Check for duplicates before creating
- **Update**: Upsert logic
- **Delete**: Idempotent (no error if already deleted)
- **Query**: Safe to retry
- **Workflow**: Checkpoint-based recovery

## UTC Timestamps

All timestamps are:
- Stored in UTC
- Formatted as ISO-8601
- End with 'Z' indicator
- No timezone offsets

Example: `2025-02-03T12:34:56.789Z`

## Error Handling

### Transient Failures

- Automatic retry with exponential backoff
- Jitter to prevent thundering herd
- Max 3 retries (configurable)

### Logic Failures

- Dead Letter Queue (DLQ)
- Don't block the pipeline
- Log structured error details

### System Failures

- Circuit breaker opens
- Stop hammering the service
- Wait for health check to pass

## Performance Considerations

### Latency

- ACL transformation: ~1-5ms per request
- Circuit breaker check: <1ms
- Retry overhead: Exponential backoff adds latency
- Parallel execution: Reduces total latency

### Throughput

- Limited by `MAX_CONCURRENT_WORKFLOWS`
- Circuit breakers protect against overload
- Connection pooling for HTTP clients

### Memory

- Knowledge cache: Configurable TTL
- Workflow state: In-memory during execution
- Checkpoint persistence: Database (future)

## Troubleshooting

### Circuit Breaker Open

```json
{
  "error": "Circuit breaker 'z3-adapter' is OPEN",
  "reason": "Too many failures",
  "next_attempt": "2025-02-03T12:40:00Z"
}
```

**Solution**: Wait for timeout or investigate adapter health.

### Schema Validation Failed

```json
{
  "error": "Schema validation failed",
  "field": "team.role",
  "expected": ["Blue", "Red", "Gold"],
  "received": "InvalidRole"
}
```

**Solution**: Use canonical schema values.

### Timeout Exceeded

```json
{
  "error": "Request timeout",
  "timeout_ms": 5000,
  "duration_ms": 5200,
  "operation": "createTeam"
}
```

**Solution**: Increase `TIMEOUT_MS` or investigate performance.

## Architecture Decision Record

See [ADR.md](./ADR.md) for detailed architectural decisions, alternatives considered, and implementation notes.

## Contributing

When contributing to this adapter:

1. **Maintain the Air Gap**: Don't import from `core-projects/`
2. **Write Probes First**: Validate APIs before implementing
3. **Enforce Schemas**: All data must pass through ACL
4. **Add Contract Tests**: Validate API contracts
5. **Use UTC Timestamps**: All times in UTC ISO-8601
6. **Make It Idempotent**: All operations safe to retry
7. **Log Structured**: JSON Lines with correlation IDs
8. **Handle Failures**: Circuit breakers and retries

## License

MIT

## Contact

OpenEvolve Federation
