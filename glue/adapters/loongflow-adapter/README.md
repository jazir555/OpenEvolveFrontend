# LoongFlow Adapter

Production-ready adapter for integrating the [LoongFlow PES (Plan-Execute-Summary)](../../../../core-projects/LoongFlow/) evolutionary AI framework into the OpenEvolve federation.

## Overview

LoongFlow is an expert-grade AI agent framework that uses evolutionary algorithms with PES (Plan-Execute-Summary) paradigm to iteratively improve solutions through structured thinking and learning.

**Key Features:**
- 🧬 **Evolutionary Memory**: MAP-Elites algorithm with Boltzmann sampling
- 🏝️ **Multi-Island Populations**: Parallel evolution across isolated populations
- 🔄 **Concurrent Execution**: Multiple evolution cycles running simultaneously
- 💾 **Checkpoint System**: State persistence and recovery
- 📊 **Adaptive Selection**: Dynamic exploration rate based on stagnation detection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  LoongFlow Adapter (Node.js)                │
│  - Circuit Breaker & Retry Logic                            │
│  - Structured JSON Logging                                  │
│  - Canonical Schema Transformation                          │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Python Sidecar Service                         │
│  - FastAPI Endpoints                                        │
│  - Async LoongFlow Execution                                │
│  - Worker Management                                        │
└────────────────────┬────────────────────────────────────────┘
                     │ Direct Python Imports
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  LoongFlow Library                          │
│  - PESAgent                                                │
│  - EvolveDatabase                                          │
│  - Planner/Executor/Summary Workers                        │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
cd glue/adapters/loongflow-adapter
npm install
npm run build
```

## Configuration

### Required Environment Variables

```bash
# LoongFlow Sidecar API (REQUIRED - no default)
export LOONGFLOW_API_URL="http://localhost:8000"

# Request timeout in milliseconds (REQUIRED - no default)
export LOONGFLOW_TIMEOUT_MS=30000
```

### Optional Environment Variables

```bash
# Maximum retry attempts for transient failures
export LOONGFLOW_MAX_RETRIES=3

# Logging level
export LOG_LEVEL="info"

# Service port for health checks
export SERVICE_PORT=8040

# Timezone (always UTC per Federation Constitution)
export TZ="UTC"
```

## Usage

### Basic Example

```typescript
import { createLoongFlowAdapter } from '@loongflow/loongflow-adapter';

// Initialize adapter (crashes if required env vars missing)
const adapter = createLoongFlowAdapter({
  api_url: process.env.LOONGFLOW_API_URL!,
  timeout_ms: 30000,
});

// Submit problem for evolution
const { agent_id } = await adapter.submitProblem({
  task: 'Optimize the circle packing algorithm',
  max_iterations: 100,
  target_score: 0.95,
  concurrency: 4,
});

// Monitor progress
const state = await adapter.getAgentState(agent_id);
console.log(`Iteration: ${state.current_iteration}`);
console.log(`Best Score: ${state.best_score}`);

// Get final result
const result = await adapter.getExecutionResult(agent_id);
console.log(`Final Solution: ${result.final_solution}`);
console.log(`Total Cost: $${result.total_cost}`);
```

### Advanced Usage

#### Query Evolutionary Database

```typescript
// Get best solutions
const bestSolutions = await adapter.getBestSolutions({
  island_id: 0,
  top_k: 10,
});

console.log('Top 10 solutions:', bestSolutions);

// Sample solution (Boltzmann selection)
const sampled = await adapter.sampleSolution(0);

// Get database status
const status = await adapter.getDatabaseStatus();
console.log('Total solutions:', status.global_status.total_solutions);
console.log('Best score:', status.global_status.best_score);
```

#### Checkpoint Management

```typescript
// Save checkpoint
await adapter.saveCheckpoint(
  '/path/to/checkpoints',
  'checkpoint-iter-50-25'
);

// List available checkpoints
const checkpoints = await adapter.listCheckpoints('/path/to/checkpoints');

// Load checkpoint
await adapter.loadCheckpoint('/path/to/checkpoints/checkpoint-iter-50-25');
```

#### Manual Intervention

```typescript
// Interrupt running agent
await adapter.interruptAgent(agent_id);

// Reset circuit breaker after sidecar recovery
adapter.resetCircuitBreaker();
```

## API Reference

### PES Agent Management

#### `submitProblem(request)`
Submit a problem to the PES Agent for evolution.

**Parameters:**
- `task` (string): Problem description
- `max_iterations?` (number): Maximum evolution iterations
- `target_score?` (number): Target score to stop evolution
- `concurrency?` (number): Number of concurrent workers
- `initial_code?` (string): Initial solution code
- `initial_score?` (number): Initial solution score
- `metadata?` (object): Additional metadata

**Returns:** `SubmitProblemResponse`
- `agent_id` (string): Unique identifier for the agent
- `status` (string): Agent status
- `message` (string): Status message

#### `getAgentState(agentId)`
Get the current state of a running PES Agent.

**Returns:** `PESAgentState`
- `status` (string): 'idle' | 'running' | 'interrupted' | 'completed' | 'failed'
- `current_iteration` (number): Current iteration number
- `best_score` (number): Best score achieved so far
- `total_tokens` (number): Total tokens consumed
- `total_cost` (number): Total cost in USD

#### `interruptAgent(agentId)`
Interrupt a running PES Agent gracefully.

**Idempotent:** Safe to call multiple times.

#### `getExecutionResult(agentId)`
Get the final execution result of a completed agent.

**Returns:** `ExecutionResult`
- `final_solution` (string): Best solution found
- `final_score` (number): Score of final solution
- `best_solutions` (Solution[]): Top N solutions
- `was_interrupted` (boolean): Whether execution was interrupted
- `total_cost` (number): Total execution cost

### Evolutionary Database

#### `sampleSolution(islandId?)`
Sample a solution using Boltzmann selection.

**Parameters:**
- `island_id?` (number): Island to sample from (optional)

**Returns:** `Solution` or empty object if database empty

#### `addSolution(solution)`
Add a solution to the database.

**Parameters:**
- `solution` (string): Solution code/content
- `evaluation` (string): Evaluation result
- `score` (number): Numerical score (0-1)
- `island_id` (number): Island identifier
- `generate_plan` (string): Plan from planner worker
- `summary` (string): Summary from summary worker

**Returns:** `solution_id` (string)

**Idempotent:** Same solution_id updates existing solution.

#### `getBestSolutions(islandId?, topK?)`
Get the best solutions from the database.

**Parameters:**
- `island_id?` (number): Filter by island
- `top_k?` (number): Return top K solutions

**Returns:** `Solution[]`

### Checkpoint Operations

#### `saveCheckpoint(checkpointPath, tag)`
Save a checkpoint of the current evolutionary state.

**Parameters:**
- `checkpoint_path` (string): Directory to save checkpoint
- `tag` (string): Checkpoint tag (e.g., 'checkpoint-iter-50-25')

#### `loadCheckpoint(checkpointPath)`
Load a checkpoint and restore evolutionary state.

**Parameters:**
- `checkpoint_path` (string): Full path to checkpoint directory

#### `listCheckpoints(checkpointPath)`
List available checkpoints in a directory.

**Returns:** `CheckpointInfo[]`

## Federation Constitution Compliance

This adapter follows all 6 Immutable Laws:

### ✅ 1. Law of Air Gap
No imports from `core-projects/LoongFlow/`. LoongFlow runs in isolated sidecar.

### ✅ 2. Law of Runtime Truth
Probe scripts verify LoongFlow structure before integration:
- `./probes/check_api.sh` - Verify source code structure
- `./probes/check_pes_api.sh` - Test PES framework imports
- `./probes/check_database.sh` - Test database operations

### ✅ 3. Law of Untouchable DB
Adapter has read-only access to evolutionary database. Writes only through LoongFlow API.

### ✅ 4. Law of Idempotency
All operations safe to retry:
- `submitProblem` - Same task_id returns existing agent
- `interruptAgent` - No-op if already stopped
- `addSolution` - Same solution_id updates existing
- `updateSolution` - UPSERT semantics

### ✅ 5. Law of Configuration Explicitness
Required environment variables crash service if missing:
```typescript
if (!config.api_url) {
  throw new Error('LOONGFLOW_API_URL is required');
}
```

### ✅ 6. Law of UTC
All timestamps in UTC ISO-8601 format:
```typescript
created_at: "2025-02-22T10:30:00.000Z"
```

## Error Handling

### Circuit Breaker
Prevents cascading failures when LoongFlow sidecar is down:
- **CLOSED**: Normal operation
- **OPEN**: Rejecting requests (sidecar down)
- **HALF_OPEN**: Testing recovery

```typescript
const stats = adapter.getCircuitBreakerState();
if (stats.state === 'open') {
  console.log('LoongFlow is down, using fallback');
}
```

### Retry Logic
Automatic retry with exponential backoff for transient failures:
- Max retries: 3 (configurable)
- Base delay: 1 second
- Max delay: 10 seconds
- Jitter: ±500ms

## Testing

```bash
# Run contract tests
npm test

# Run tests in watch mode
npm run test:watch

# Run with coverage
npm run test:coverage

# Execute probe scripts (Law of Runtime Truth)
cd probes
./check_api.sh
./check_pes_api.sh
./check_database.sh
```

## Development

```bash
# Build TypeScript
npm run build

# Watch mode
npm run watch

# Lint code
npm run lint
npm run lint:fix

# Clean build artifacts
npm run clean
```

## Deployment

### Docker
```bash
docker build -t loongflow-adapter:1.0.0 .
docker run -e LOONGFLOW_API_URL=http://loongflow-sidecar:8000 loongflow-adapter:1.0.0
```

### Kubernetes
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: loongflow-adapter
spec:
  containers:
  - name: adapter
    image: loongflow-adapter:1.0.0
    env:
    - name: LOONGFLOW_API_URL
      value: "http://loongflow-sidecar:8000"
    - name: LOONGFLOW_TIMEOUT_MS
      value: "30000"
    ports:
    - containerPort: 8040
    livenessProbe:
      httpGet:
        path: /health
        port: 8040
```

## Troubleshooting

### "LOONGFLOW_API_URL is required"
**Cause:** Missing required environment variable
**Fix:** Set `LOONGFLOW_API_URL` before starting adapter

### "Circuit breaker is OPEN"
**Cause:** LoongFlow sidecar is down or unreachable
**Fix:**
1. Check sidecar is running: `curl $LOONGFLOW_API_URL/health`
2. Verify network connectivity
3. Manual reset: `adapter.resetCircuitBreaker()`

### "All retries exhausted"
**Cause:** Transient failure persisted through all retries
**Fix:**
1. Check logs for specific error
2. Verify LoongFlow sidecar health
3. Increase timeout if needed

## References

- [LoongFlow Source](../../../../core-projects/LoongFlow/)
- [PES Agent Documentation](../../../../core-projects/LoongFlow/src/loongflow/framework/pes/pes_agent.py)
- [Federation Constitution](../../../CLAUDE.md)
- [ADR: LoongFlow Integration](./ADR.md)

## License

MIT

## Maintainer

OpenEvolve Federation
