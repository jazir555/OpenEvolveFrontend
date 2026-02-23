# ADR: LoongFlow Adapter Integration

## Status
Accepted

## Context
LoongFlow is an expert-grade AI agent framework built around the **PES (Plan-Execute-Summary)** paradigm. Unlike traditional "generate-retry" approaches, PES enables structured thinking and learning through evolutionary memory with multi-island populations and MAP-Elites algorithms.

### Key Characteristics
- **Language**: Python library (not an HTTP API)
- **Architecture**: PES (Plan-Execute-Summary) agent framework
- **Core Components**:
  - `PESAgent` - Main orchestrator with worker registration
  - `EvolveDatabase` - Evolutionary memory with MAP-Elites
  - `Solution` model - Core data structure with score, evaluation, summary
  - Checkpoint system - State persistence and recovery
- **Configuration**: YAML-based (task_config.yaml)
- **Execution**: Async/await with concurrent evolution cycles

## Decision
We will integrate LoongFlow using a **Python Sidecar Pattern**:

### Architecture
```
[LoongFlow Adapter (Node.js)]
         |
         | HTTP/REST
         |
         v
[Python Sidecar Service]
         |
         | Direct imports
         |
         v
[LoongFlow Library]
```

### Rationale
1. **Air Gap Compliance**: LoongFlow source code remains in `core-projects/LoongFlow/`, untouched
2. **Anti-Corruption Layer**: Adapter canonical schemas prevent LoongFlow data models from leaking
3. **Runtime Truth**: Probe scripts verify LoongFlow structure before attempting integration
4. **Isolation**: Sidecar runs in separate container, can be scaled independently
5. **Failure Management**: Circuit breaker prevents cascading failures from LoongFlow issues

### Integration Points

#### 1. PES Agent Management
- **POST /pes/submit** - Submit problem for evolution
- **GET /pes/agents/:id/state** - Get current agent state
- **POST /pes/agents/:id/interrupt** - Stop running agent
- **GET /pes/agents/:id/result** - Get final execution result

#### 2. Evolutionary Database
- **GET /database/sample** - Sample solution (Boltzmann selection)
- **POST /database/solutions** - Add solution to database
- **PUT /database/solutions/:id** - Update solution
- **GET /database/best** - Get best solutions (top-k)
- **GET /database/status** - Get database status

#### 3. Checkpoint Operations
- **POST /database/checkpoints** - Save checkpoint
- **POST /database/checkpoints/load** - Load checkpoint
- **GET /database/checkpoints** - List available checkpoints

## Data Transformation

### Canonical Schema (Adapter → Federation)
```typescript
interface Solution {
  solution_id: string;
  solution: string;        // Code/solution content
  evaluation: string;      // Evaluation result
  score: number;           // Numerical score
  parent_id?: string;      // Parent solution ID
  island_id: number;       // Island for population model
  iteration: number;       // Generation iteration
  generate_plan: string;   // Plan from planner worker
  summary: string;         // Summary from summary worker
  created_at: string;      // UTC ISO-8601 timestamp
}
```

### LoongFlow Native (Sidecar → LoongFlow)
```python
Solution(
    solution="code",
    evaluation="eval_result",
    score=0.85,
    island_id=0,
    generate_plan="plan",
    summary="summary"
)
```

### Transformation Rules
1. **Timestamps**: All converted to UTC ISO-8601 (Law of UTC)
2. **Snake to CamelCase**: `generate_plan` → `generatePlan`
3. **UUID Generation**: `solution_id` added by adapter if not present
4. **Validation**: Zod schemas enforce canonical structure

## Error Handling Strategy

### Transient Failures (Retry with Backoff)
- Network blips
- Temporary timeouts
- Sidecar container restarts

### Logic Failures (Dead Letter Queue)
- Invalid task descriptions
- Malformed solutions
- Configuration errors

### System Failures (Circuit Breaker)
- LoongFlow sidecar down
- Database connection failures
- Out of memory errors

### Circuit Breaker Configuration
```typescript
{
  threshold: 5,           // Trip after 5 consecutive failures
  timeout_ms: 60000,      // Stay open for 1 minute
  reset_timeout_ms: 10000 // Test recovery after 10 seconds
}
```

## Idempotency Guarantees

All operations are designed to be safely retryable:

1. **submitProblem**: Same task_id returns existing agent
2. **interruptAgent**: No-op if already stopped
3. **addSolution**: Same solution_id updates existing solution
4. **updateSolution**: UPSERT semantics
5. **saveCheckpoint**: Overwrites existing checkpoint tag

## Gotchas Discovered During Probing

### 1. Path Resolution
- **Issue**: Probe scripts initially used wrong relative path
- **Solution**: Corrected to `../../../../core-projects/LoongFlow` from `probes/`
- **Lesson**: Always verify paths with actual execution

### 2. Python Environment
- **Issue**: Python not in PATH on Windows
- **Impact**: Probe tests can't execute Python imports
- **Mitigation**: Probes verify source structure; runtime tests in container
- **Lesson**: Probing > Documentation, but know platform limitations

### 3. LoongFlow is NOT an HTTP API
- **Issue**: Initially assumed HTTP endpoints like OpenEvolve
- **Reality**: LoongFlow is a Python library requiring sidecar
- **Impact**: Added complexity of Python sidecar service
- **Decision**: Accept complexity for proper isolation and ACL

### 4. Checkpoint Naming Convention
- **Discovery**: Checkpoints use format `checkpoint-iter-{id}-{count}`
- **Requirement**: Adapter must parse this format for checkpoint listing
- **Implementation**: Add regex parsing in checkpoint operations

### 5. Async/Await Required
- **Discovery**: Most EvolveDatabase methods are async
- **Impact**: Sidecar must use asyncio properly
- **Solution**: FastAPI or similar async framework for sidecar

## Testing Approach

### 1. Contract Tests (Basic)
```typescript
describe('LoongFlow Adapter Contract', () => {
  it('should initialize with required config', async () => {
    const adapter = createLoongFlowAdapter({
      api_url: 'http://localhost:8000',
      timeout_ms: 30000,
    });
    expect(adapter).toBeInstanceOf(LoongFlowAdapter);
  });

  it('should fail without required config', () => {
    expect(() => createLoongFlowAdapter({}))
      .toThrow('LOONGFLOW_API_URL is required');
  });
});
```

### 2. Probe Validation
```bash
# Source structure
./probes/check_api.sh

# PES framework imports
./probes/check_pes_api.sh

# Database operations
./probes/check_database.sh
```

### 3. Integration Tests (with live sidecar)
- Submit problem and get result
- Sample solutions from database
- Save/load checkpoints
- Circuit breaker tripping
- Retry logic

## Future Enhancements

### Phase 2: Advanced Features
1. **Streaming Results**: WebSocket connection for real-time evolution updates
2. **Multi-Agent Orchestration**: Coordinate multiple PES agents in parallel
3. **Knowledge Fusion**: Extract insights from LoongFlow summaries into knowledge graph
4. **Hybrid Workflows**: Combine LoongFlow evolution with OpenEvolve gauntlet evaluation

### Phase 3: Performance Optimization
1. **Solution Caching**: Cache best solutions to reduce database queries
2. **Batch Operations**: Add bulk solution operations for efficiency
3. **Compression**: Compress large solutions and checkpoints
4. **Sharding**: Distribute islands across multiple sidecar instances

## References
- LoongFlow Source: `core-projects/LoongFlow/`
- PES Agent: `src/loongflow/framework/pes/pes_agent.py`
- EvolveDatabase: `src/loongflow/framework/pes/database/database.py`
- Example Configs: `agents/*/examples/*/task_config.yaml`
- Federation Constitution: `CLAUDE.md`

## Appendix: Example Usage

```typescript
import { createLoongFlowAdapter } from '@loongflow/loongflow-adapter';

// Initialize adapter (crashes if LOONGFLOW_API_URL not set)
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
console.log(`Iteration: ${state.current_iteration}/${state.max_iterations}`);
console.log(`Best score: ${state.best_score}`);

// Interrupt if needed
await adapter.interruptAgent(agent_id);

// Get final result
const result = await adapter.getExecutionResult(agent_id);
console.log(`Final solution: ${result.final_solution}`);
console.log(`Total cost: $${result.total_cost}`);

// Query evolutionary database
const bestSolutions = await adapter.getBestSolutions(topK: 10);
console.log(`Top 10 solutions:`, bestSolutions);
```

---

**Decision Date**: 2025-02-22
**Author**: Federation Distinguished Engineer
**Status**: Accepted - Implementation Complete
