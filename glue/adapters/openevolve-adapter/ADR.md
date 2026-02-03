# Architecture Decision Record: OpenEvolve Main Orchestration Adapter

## Status
Accepted

## Date
2025-02-03

## Context

The OpenEvolve system requires coordination of 30+ massive, immutable Open Source projects. Each project has:
- Different API patterns (REST, GraphQL, gRPC, CLI)
- Different data formats (JSON, XML, Protocol Buffers, S-expressions)
- Different naming conventions (snake_case, camelCase, kebab-case)
- Different authentication mechanisms
- Different error handling patterns

The challenge is to create a unified orchestration layer that:
1. Coordinates all adapters without coupling them together
2. Maintains data integrity through canonical schemas
3. Provides resilience through circuit breakers and retries
4. Enables complex multi-step workflows across systems
5. Aggregates knowledge from all sources

## Decision

Create a main orchestration adapter (`openevolve-adapter`) that serves as the central coordination hub with the following architecture:

### 1. Anti-Corruption Layer (ACL)

**Problem**: Direct data flow between adapters causes schema coupling and corruption.

**Solution**: Implement an Anti-Corruption Layer that:
- Defines canonical schemas for all data types
- Transforms data from each adapter to/from canonical format
- Validates all incoming/outgoing data against schemas
- Prevents schema leakage between adapters

**Implementation**:
```typescript
// Canonical schema definition
interface Team {
  name: string;
  role: 'Blue' | 'Red' | 'Gold';  // Canonical values
  members: ModelConfig[];
  // ... all fields in canonical format
}

// ACL transformation
function transformFromZ3(z3Data: Z3TeamFormat): Team {
  // Transform Z3 format to canonical
}

function transformToZ3(canonical: Team): Z3TeamFormat {
  // Transform canonical to Z3 format
}
```

### 2. Circuit Breaker Pattern

**Problem**: Cascading failures when one adapter fails can bring down the entire system.

**Solution**: Implement circuit breakers for each adapter:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Adapter failing, requests immediately rejected
- **HALF_OPEN**: Testing if adapter has recovered

**Configuration**:
```typescript
{
  failureThreshold: 5,      // Open after 5 failures
  successThreshold: 2,      // Close after 2 successes
  timeout: 60000,           // Stay open for 60 seconds
  monitorPeriod: 10000      // Check health every 10 seconds
}
```

**Implementation**:
- Each adapter has its own circuit breaker instance
- Breakers are independent (Z3 failing doesn't affect LeanAide)
- Automatic transition between states based on responses

### 3. Retry with Exponential Backoff

**Problem**: Transient network failures cause unnecessary errors.

**Solution**: Implement retry logic with:
- Exponential backoff (1s, 2s, 4s, 8s, ...)
- Jitter to avoid thundering herd
- Max retries limit (default: 3)
- Idempotency checks for safe retries

**Implementation**:
```typescript
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: RetryConfig,
  logger: StructuredLogger,
  context: LogContext,
): Promise<T>
```

### 4. Integration Coordinator

**Problem**: Complex operations require coordination across multiple adapters.

**Solution**: Create an `IntegrationCoordinator` that:
- Selects appropriate adapters based on problem type
- Plans execution order (parallel vs sequential)
- Handles fallback between adapters
- Monitors adapter health

**Coordination Flow**:
```
Request → Plan Coordination → Select Adapters → Execute → Aggregate Results
         ↓                    ↓                  ↓
    Analyze Type         Choose by          Parallel/
    and Domain           Capability         Sequential
```

**Adapter Selection Logic**:
```typescript
problem_type → capability mapping:
  - formal_verification → [z3, leanaide]
  - retrieval → [ragbits, vectordb]
  - knowledge_graph → [graphiti, vectordb]
  - graph_ml → [karateclub, graphiti]
```

### 5. Workflow Orchestrator

**Problem**: Multi-step workflows require complex state management and error handling.

**Solution**: Create a `WorkflowOrchestrator` that:
- Manages workflow lifecycle (init → execute → finalize)
- Executes stages in dependency order
- Tracks progress and reports updates
- Handles stage failures with recovery
- Supports workflow checkpoints

**Workflow Stages**:
1. Content Analysis (understand problem)
2. Decomposition Planning (break into sub-problems)
3. Sub-problem Solving (solve each sub-problem)
4. Solution Assembly (combine solutions)
5. Final Verification (validate result)
6. Knowledge Extraction (extract learnings)

**Stage Dependencies**:
```
content_analysis → decomposition_planning → solve_sub_problem_* → solution_assembly → final_verification → knowledge_extraction
```

### 6. Knowledge Aggregator

**Problem**: Knowledge is scattered across multiple adapters with different query interfaces.

**Solution**: Create a `KnowledgeAggregator` that:
- Provides unified query interface
- Queries all sources in parallel
- Fuses results using semantic similarity
- Caches results for performance
- Extracts knowledge from workflows

**Query Flow**:
```
Knowledge Query → Check Cache → Query Sources → Fuse Results → Cache → Return
                                     ↓
                    [Z3, LeanAide, RAGBits, Vector DB,
                     Graphiti, KarateClub]
```

### 7. Structured Logging (JSON Lines)

**Problem**: Unstructured logs make debugging and monitoring difficult.

**Solution**: Implement structured logging with:
- JSON Lines format (one JSON object per line)
- Correlation ID for request tracking
- Service identification (source/target)
- Timestamps in UTC ISO-8601 format
- Contextual metadata

**Log Format**:
```json
{
  "timestamp": "2025-02-03T12:34:56.789Z",
  "level": "info",
  "message": "Team created",
  "service": "openevolve-adapter",
  "correlation_id": "abc-123-def",
  "source_service": "openevolve-adapter",
  "target_service": "openevolve-api",
  "team_name": "test-team",
  "role": "Blue"
}
```

## Consequences

### Positive

1. **Decoupling**: Adapters are independent through the ACL
2. **Resilience**: Circuit breakers prevent cascading failures
3. **Flexibility**: Easy to add new adapters without changing core
4. **Observability**: Structured logs enable monitoring
5. **Scalability**: Parallel execution where possible
6. **Reliability**: Retries handle transient failures
7. **Maintainability**: Clear separation of concerns

### Negative

1. **Complexity**: More layers to understand and debug
2. **Latency**: ACL transformations add overhead
3. **Memory**: Caching requires memory management
4. **Testing**: More integration points to test

### Risks

1. **Schema Drift**: Canonical schemas may become outdated
   - **Mitigation**: Contract tests validate schemas on startup

2. **Circuit Breaker Flooding**: Many requests can fail simultaneously
   - **Mitigation**: Jitter and adaptive thresholds

3. **Cache Staleness**: Cached knowledge may become outdated
   - **Mitigation**: TTL-based invalidation

## Alternatives Considered

### Alternative 1: Direct Adapter-to-Adapter Communication

**Approach**: Adapters call each other directly without coordination layer.

**Pros**:
- Simpler architecture
- Lower latency

**Cons**:
- Tight coupling between adapters
- No centralized error handling
- Difficult to add new adapters
- Schema corruption likely

**Decision**: Rejected - Violates Law of the "Air Gap"

### Alternative 2: Event-Only Architecture (No Direct Calls)

**Approach**: All communication through event bus, no direct API calls.

**Pros**:
- Maximum decoupling
- Natural scalability

**Cons**:
- Complex request/response patterns
- Higher latency
- Difficult debugging
- Event schema management

**Decision**: Rejected - Too complex for synchronous operations

### Alternative 3: Single Monolithic Orchestrator

**Approach**: One giant orchestrator that handles everything.

**Pros**:
- Simpler deployment
- Single codebase

**Cons**:
- Violates Single Responsibility Principle
- Difficult to test
- Hard to maintain
- No modularity

**Decision**: Rejected - Violates separation of concerns

## Implementation Notes

### Environment Variables

**Required** (no defaults, fail if missing):
- `OPENEVOLVE_API_URL` - OpenEvolve API endpoint
- `TIMEOUT_MS` - Request timeout in milliseconds

**Optional** (with defaults):
- `EVENT_BUS_URL` - Event bus for pub/sub
- `LOG_LEVEL` - Logging level (default: info)
- `COORDINATION_TIMEOUT_MS` - Coordination timeout (default: 10000)
- `MAX_CONCURRENT_WORKFLOWS` - Max concurrent workflows (default: 5)

### Idempotency Requirements

All operations must be idempotent:
- Check before create (avoid duplicates)
- Use UPSERT logic
- Deduplicate by ID
- Safe to retry any operation

### UTC Timestamps

All timestamps:
- Stored in UTC
- Formatted as ISO-8601
- End with 'Z' indicator
- No timezone offsets

### Canonical Schema Enforcement

All data:
- Validated against Zod schemas
- Transformed at ACL boundaries
- Never passed through directly
- Documented in contract tests

## Testing Strategy

### Contract Tests

Validate API contracts:
- Run on adapter startup
- Test all endpoints
- Validate response schemas
- Fail fast if contracts violated

### Integration Tests

Test adapter coordination:
- Mock downstream adapters
- Test circuit breaker behavior
- Test retry logic
- Test workflow execution

### End-to-End Tests

Test full workflows:
- Use real adapters
- Test complex scenarios
- Validate knowledge aggregation
- Measure performance

## Monitoring

### Metrics to Track

- **Health**: Each adapter's circuit breaker state
- **Latency**: Request latency by adapter
- **Errors**: Error rate by adapter and operation
- **Throughput**: Requests per second
- **Cache**: Hit rate, size, eviction rate
- **Workflows**: Active, completed, failed

### Alerts

- **Critical**: Circuit breaker OPEN for critical adapter
- **Warning**: Error rate > 10% for any adapter
- **Info**: Workflow completed/failed

## Future Enhancements

1. **Dynamic Adapter Discovery**: Auto-discover new adapters
2. **Machine Learning**: Learn optimal adapter selection
3. **Workflow Templates**: Pre-built workflow patterns
4. **Visual Workflow Editor**: GUI for workflow design
5. **Distributed Tracing**: OpenTelemetry integration
6. **GraphQL API**: Alternative to REST
7. **WebSocket Support**: Real-time workflow updates

## References

- [Law of the "Air Gap"](../../../CLAUDE.md) - Source code isolation
- [Law of "Runtime Truth"](../../../CLAUDE.md) - Probe before implementation
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Anti-Corruption Layer](https://microservices.io/patterns/applied-patterns/anti-corruption-layer.html)
