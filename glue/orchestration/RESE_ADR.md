# Architecture Decision Records (ADR) - RESE Pipeline

## ADR-001: Event-Driven Architecture for Phase Coordination

### Status
Accepted

### Context
The RESE pipeline consists of four phases that need to coordinate with each other. Each phase produces outputs that become inputs to the next phase. We need a mechanism for:

1. Phase coordination and sequencing
2. Error handling and recovery
3. Correlation tracking across phases
4. Monitoring and observability

### Decision
Use an event-driven architecture with a publish/subscribe event bus.

**Rationale:**

- **Decoupling**: Phases are loosely coupled through events, not direct calls
- **Flexibility**: Easy to add new phases or change orchestration logic
- **Observability**: All events are logged with correlation IDs
- **Scalability**: Multiple instances can subscribe to same events
- **Idempotency**: Event deduplication enables safe replay

**Alternatives Considered:**

1. **Direct function calls**: Too tightly coupled, hard to monitor
2. **Message queue (RabbitMQ/Kafka)**: Overkill for single-host deployment
3. **Actor model (Ray/dask)**: Too complex, steep learning curve

### Consequences

**Positive:**
- Easy to add new phases without modifying existing code
- Natural observability through event logs
- Supports both sync and async execution

**Negative:**
- Slightly more complex than direct calls
- Event ordering must be managed carefully

**Mitigation:**
- Use correlation IDs for tracking
- Document event types and flow
- Provide helper methods for common patterns

---

## ADR-002: Circuit Breaker Pattern for Failure Handling

### Status
Accepted

### Context
The pipeline calls external services (DEE, LLTL, SCE adapters) that may fail or become slow. We need to:

1. Prevent cascading failures
2. Stop calling failing services
3. Automatically recover when services are healthy
4. Provide visibility into service health

### Decision
Implement the Circuit Breaker pattern for all external service calls.

**Rationale:**

- **Prevents cascading failures**: Failing fast instead of hanging
- **Automatic recovery**: Probes service health periodically
- **Visibility**: Circuit state (OPEN/CLOSED/HALF_OPEN) is observable
- **Proven pattern**: Used by Netflix, Google, etc.

**States:**
- **CLOSED**: Normal operation, calls go through
- **OPEN**: Failing, calls are blocked
- **HALF_OPEN**: Testing if service recovered

**Thresholds:**
- Open after N consecutive failures (default: 5)
- Stay open for T milliseconds (default: 60000ms)
- Close after N successful attempts (default: 3)

**Alternatives Considered:**

1. **Retry only**: Would continue hammering failing service
2. **Timeout only**: Would still waste resources on failing calls
3. **Manual intervention**: Too slow, not automated

### Consequences

**Positive:**
- Prevents cascading failures
- Conserves resources during outages
- Automatic recovery

**Negative:**
- Adds complexity to error handling
- Need to tune thresholds for each service

**Mitigation:**
- Make thresholds configurable via env vars
- Log all state transitions
- Provide manual reset endpoint

---

## ADR-003: Exponential Backoff with Jitter for Retries

### Status
Accepted

### Context
Transient failures (network blips, timeouts) are common. We need to retry failed operations, but:

1. Don't overwhelm the service with rapid retries
2. Distribute retry attempts to avoid thundering herd
3. Give service time to recover
4. Bound the total retry time

### Decision
Use exponential backoff with jitter for all transient failures.

**Rationale:**

- **Exponential backoff**: Gives service time to recover (1s, 2s, 4s, 8s...)
- **Jitter**: Distributes attempts to avoid synchronization
- **Bounded**: Max delay prevents unbounded waits
- **Proven**: Used by AWS, Google, etc.

**Formula:**
```
delay = min(initial_delay * (multiplier ^ attempt) * random(0.5, 1.5), max_delay)
```

**Defaults:**
- Initial delay: 1000ms
- Max delay: 30000ms
- Multiplier: 2.0
- Max retries: 3

**Alternatives Considered:**

1. **Fixed delay**: Would cause thundering herd
2. **Linear backoff**: Slower to adapt
3. **No retries**: Would fail too easily

### Consequences

**Positive:**
- Handles transient failures gracefully
- Avoids overwhelming services
- Proven pattern

**Negative:**
- Adds latency on retries
- Need to tune parameters

**Mitigation:**
- Make parameters configurable
- Log retry attempts
- Use circuit breaker to stop retries eventually

---

## ADR-004: Dead Letter Queue for Logic Failures

### Status
Accepted

### Context
Some failures are permanent (bad data, validation errors). Retrying these won't help. We need to:

1. Not block the pipeline on logic errors
2. Preserve failed requests for analysis
3. Enable manual intervention/retry
4. Provide visibility into failure patterns

### Decision
Use a Dead Letter Queue (DLQ) for all logic failures.

**Rationale:**

- **Don't lose data**: Failed requests are preserved
- **Don't block**: Pipeline continues despite failures
- **Visibility**: Can analyze failure patterns
- **Manual retry**: Can fix and retry items

**Error Classification:**

1. **Transient** (timeout, network): Retry with backoff
2. **Logic** (validation, bad data): Send to DLQ
3. **System** (circuit breaker): Stop calling

**DLQ Features:**
- Max size limit (default: 1000)
- Persist to disk (optional)
- Expose via API for inspection
- Support manual retry

**Alternatives Considered:**

1. **Log and discard**: Would lose data
2. **Retry forever**: Would waste resources
3. **Block pipeline**: Would stop all progress

### Consequences

**Positive:**
- Preserves failed requests
- Doesn't block pipeline
- Enables analysis

**Negative:**
- Adds operational complexity
- Need to monitor DLQ size

**Mitigation:**
- Alert on DLQ size threshold
- Provide API to manage DLQ
- Document error types

---

## ADR-005: Correlation IDs for End-to-End Tracing

### Status
Accepted

### Context
A pipeline execution spans multiple phases, services, and components. We need to:

1. Trace a single request across all components
2. Debug issues across phase boundaries
3. Aggregate logs by request
4. Measure end-to-end latency

### Decision
Use UUID v4 correlation IDs for all pipeline executions.

**Rationale:**

- **Universally unique**: No collisions
- **Traces everywhere**: Logs, events, metrics
- **No coordination**: Distributed generation
- **Standard**: Used by most distributed systems

**Implementation:**

- Generated at pipeline start
- Passed to all phases
- Included in all log entries
- Published in all events
- Returned in final result

**Benefits:**
- Can grep logs by correlation_id
- Can aggregate events by correlation_id
- Can measure per-request latency
- Can debug specific executions

**Alternatives Considered:**

1. **Trace ID (OpenTelemetry)**: Overkill for now
2. **Request ID**: Same concept, different name
3. **No tracing**: Would be impossible to debug

### Consequences

**Positive:**
- Excellent observability
- Easy debugging
- Standard practice

**Negative:**
- Slightly more complex code
- Need to pass ID everywhere

**Mitigation:**
- Use CorrelationManager to simplify
- Include ID in logger
- Document tracing patterns

---

## ADR-006: Phase Timeouts for Bounded Execution

### Status
Accepted

### Context
Each phase can take unbounded time (e.g., MCTS search). We need to:

1. Guarantee overall pipeline completes
2. Prevent runaway phases
3. Provide time budgets per phase
4. Fail fast when phases hang

### Decision
Enforce per-phase and overall pipeline timeouts.

**Rationale:**

- **Bounded latency**: Guarantees max execution time
- **Resource management**: Prevents resource exhaustion
- **User experience**: Fails fast instead of hanging
- **Production safety**: Prevents cascade failures

**Timeouts:**
- Phase I: 60s (Epistemic Audit)
- Phase II: 90s (Isomorphic Mapping)
- Phase III: 120s (MCTS Search)
- Phase IV: 60s (Architecture Assembly)
- Pipeline: 300s (overall)

**Implementation:**
- Use `threading.Timer` for timeout
- Kill phase thread on timeout
- Return TIMEOUT status
- Log timeout events

**Alternatives Considered:**

1. **No timeout**: Could hang forever
2. **Global timeout only**: Would allow one phase to monopolize time
3. **Adaptive timeout**: Too complex, unpredictable

### Consequences

**Positive:**
- Bounded execution time
- Prevents resource exhaustion
- Predictable performance

**Negative:**
- May timeout legitimate long-running phases
- Need to tune timeouts per use case

**Mitigation:**
- Make timeouts configurable
- Log timeout events prominently
- Support checkpoint/resume in future

---

## ADR-007: JSON Lines Logging for Structured Logs

### Status
Accepted

### Context
We need to log pipeline execution for debugging and monitoring. Requirements:

1. Machine-parseable logs
2. Queryable (grep, jq, etc.)
3. Include context (correlation_id, phase, etc.)
4. Standard format

### Decision
Use JSON Lines (jsonl) format for all logs.

**Rationale:**

- **Structured**: Each log line is valid JSON
- **Queryable**: Easy to grep, jq, etc.
- **Context-rich**: Include metadata in each log
- **Standard**: Used by most modern systems

**Log Entry Format:**
```json
{
  "msg": "Phase I completed",
  "level": "INFO",
  "correlation_id": "abc-123",
  "source_service": "phase_i_executor",
  "timestamp": "2026-02-04T12:34:56.789Z",
  "phase": "Phase_I",
  "execution_time_ms": 1234
}
```

**Required Fields:**
- `msg`: Log message
- `level`: INFO, WARNING, ERROR, DEBUG
- `correlation_id`: For tracing
- `source_service`: Component name
- `timestamp`: ISO 8601 UTC

**Optional Fields:**
- Phase-specific context
- Error details
- Performance metrics

**Alternatives Considered:**

1. **Plain text**: Not machine-parseable
2. **Binary logs**: Not human-readable
3. **Syslog**: Too complex, requires external daemon

### Consequences

**Positive:**
- Excellent observability
- Easy to query and analyze
- Standard tooling (jq, etc.)

**Negative:**
- More verbose than plain text
- Slightly more complex to generate

**Mitigation:**
- Use helper class (StructuredLogger)
- Provide human-readable summary endpoint
- Document log format

---

## ADR-008: Configuration via Environment Variables

### Status
Accepted

### Context
The pipeline has many configuration options. We need to:

1. Support different environments (dev, staging, prod)
2. Avoid hardcoding values
3. Make configuration explicit
4. Enable containerization

### Decision
Load all configuration from environment variables.

**Rationale:**

- **12-factor app**: Follows best practices
- **Container-friendly**: Works with Docker/Kubernetes
- **Explicit**: No magic defaults
- **Security**: Can inject secrets via env vars

**Categories:**
- Timeouts (all phases)
- Retry (max, delays, backoff)
- Circuit breaker (threshold, timeout)
- DLQ (max size, persist path)
- Event bus (max events, persist)
- Phase enablement (flags)
- Logging (level, format)

**Validation:**
- Crash immediately if required vars missing
- Validate all values at startup
- Provide clear error messages

**Alternatives Considered:**

1. **Config files**: Harder to manage in containers
2. **Command-line args**: Too many options
3. **Hardcoded**: Not flexible, violates 12-factor

### Consequences

**Positive:**
- Follows 12-factor app principles
- Container-friendly
- Explicit configuration

**Negative:**
- Many environment variables to manage
- Need to document all variables

**Mitigation:**
- Provide docker-compose.yml with all vars
- Use Kubernetes ConfigMaps
- Document all variables in README
- Provide --config flag to show current config

---

## ADR-009: Sequential Phase Execution

### Status
Accepted

### Context
The four phases have dependencies: Phase II needs Phase I output, Phase III needs Phase II output, etc. We need to:

1. Respect phase dependencies
2. Support future parallel phases
3. Enable selective phase execution
4. Provide progress feedback

### Decision
Execute phases sequentially by default, with option to skip phases.

**Rationale:**

- **Correctness**: Respects dependencies
- **Simplicity**: Easy to understand and debug
- **Flexibility**: Can skip phases via flags
- **Progress**: Can report progress after each phase

**Dependencies:**
```
Phase I (constraints)
  ↓
Phase II (isomorphisms)
  ↓
Phase III (hypotheses)
  ↓
Phase IV (architecture)
```

**Flags:**
- `ENABLE_PHASE_I`: default true
- `ENABLE_PHASE_II`: default true
- `ENABLE_PHASE_III`: default true
- `ENABLE_PHASE_IV`: default true

**Future Enhancement:**
- Support parallel execution of independent phases
- Checkpoint/resume for long-running pipelines

**Alternatives Considered:**

1. **Parallel execution**: Would violate dependencies
2. **All-or-nothing**: Not flexible enough
3. **Directed acyclic graph (DAG)**: Overkill for now

### Consequences

**Positive:**
- Correct by default
- Simple to understand
- Flexible via flags

**Negative:**
- Slower than parallel (if applicable)
- Less flexible than DAG

**Mitigation:**
- Document dependencies clearly
- Log phase completion events
- Consider DAG for future enhancements

---

## ADR-010: Idempotency for Safe Replay

### Status
Accepted

### Context
Network failures and retries are common. The same operation may be called multiple times. We need to:

1. Ensure safe replay
2. Avoid duplicate work
3. Prevent duplicate side effects
4. Enable exactly-once semantics

### Decision
Design all operations to be idempotent.

**Rationale:**

- **Safe replay**: Can retry without side effects
- **DEDUP**: Check if operation already performed
- **Exactly-once**: Avoid duplicates despite retries
- **Standard**: Used by most distributed systems

**Patterns:**

1. **Event deduplication**: Track processed event IDs
2. **UPSERT**: Update or insert (database)
3. **Check-before-create**: Check if resource exists first
4. **Idempotency keys**: Include unique key in requests

**Examples:**
- Event bus: Deduplicate by event_id
- Pipeline: Check correlation_id before starting
- Adapters: UPSERT instead of INSERT

**Alternatives Considered:**

1. **At-least-once**: Could cause duplicates
2. **At-most-once**: Could lose data
3. **Distributed transactions**: Too complex, not needed

### Consequences

**Positive:**
- Safe to retry operations
- No duplicate side effects
- Simplifies error handling

**Negative:**
- Slightly more complex code
- Need to track state

**Mitigation:**
- Use correlation IDs
- Provide helper methods
- Document idempotency patterns
