# Orchestration Layer

Event bus and workflow orchestration for the OpenEvolve Mega-Structure.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orchestration Layer                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │ Event Bus    │───▶│ Workflow      │───▶│ Correlation     │  │
│  │              │    │ Engine        │    │ Tracker         │  │
│  │ - Publish    │    │               │    │                 │  │
│  │ - Subscribe  │    │ - Sequential  │    │ - UUID v4       │  │
│  │ - Replay     │    │ - Parallel    │    │ - Distributed   │  │
│  │ - Persist    │    │ - State Mgmt  │    │   Tracing       │  │
│  └──────────────┘    └───────────────┘    └─────────────────┘  │
│         │                    │                     │            │
│         └────────────────────┼─────────────────────┘            │
│                              ▼                                  │
│                    ┌──────────────────┐                         │
│                    │ Dead Letter Queue│                         │
│                    │                  │                         │
│                    │ - Retry Logic    │                         │
│                    │ - Exponential    │                         │
│                    │   Backoff        │                         │
│                    │ - Monitoring     │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Adapters Layer                            │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Z3       │  │LeanAide  │  │RAGBits   │  │Vector DB │       │
│  │ Adapter  │  │ Adapter  │  │ Adapter  │  │ Adapter  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Graphiti  │  │KarateClub│  │BubbleLab │  │Main      │       │
│  │ Adapter  │  │ Adapter  │  │ Adapter  │  │ Adapter  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Event Bus (`event-bus.ts`)

Publish/subscribe interface for distributed event handling.

**Features:**
- Multiple backends: memory, Redis, RabbitMQ, Kafka
- Event persistence for replay (idempotency)
- Dead letter queue routing
- Circuit breaker integration
- Correlation ID propagation

**Configuration:**
```bash
EVENT_BUS_TYPE=memory|redis|rabbitmq|kafka
EVENT_BUS_URL=redis://localhost:6379  # Required for non-memory backends
```

**Usage:**
```typescript
import { eventBus } from './orchestration';
import { createBaseEvent } from './orchestration';

// Subscribe to events
const subscription = eventBus.subscribe('KnowledgeExtracted', async (event) => {
  console.log(`Received event: ${event.type}`);
  await processChunks(event.data.chunks);
});

// Publish event
const event = createBaseEvent(
  'KnowledgeExtracted',
  'ragbits-adapter',
  'corr-123',
  {
    document_id: 'doc-456',
    chunk_count: 10,
    chunks: [...],
    extraction_method: 'recursive'
  }
);

await eventBus.publish(event);

// Replay events
await eventBus.replay({
  event_type: 'KnowledgeExtracted',
  from_timestamp: '2025-01-01T00:00:00Z'
});
```

### 2. Workflow Engine (`workflow-engine.ts`)

Multi-step workflow orchestration with state management.

**Features:**
- Sequential and parallel step execution
- Workflow state management
- Circuit breakers for external services
- Timeout handling
- Built-in workflows for common patterns

**Predefined Workflows:**

1. **Z3 → LeanAide Cross-Validation**
   - Verify proofs with both systems
   - Cross-validate results
   - Ideal for formal verification

2. **RAGBits → Vector DB → Knowledge Graph**
   - Extract knowledge chunks
   - Create embeddings
   - Index in vector DB
   - Update knowledge graph
   - Complete RAG pipeline

3. **Document → Embedding → Index**
   - Extract document content
   - Generate embeddings
   - Store in vector DB
   - Quick document indexing

**Usage:**
```typescript
import { workflowEngine, PREDEFINED_WORKFLOWS } from './orchestration';

// Execute predefined workflow
const result = await workflowEngine.execute(
  PREDEFINED_WORKFLOWS['rag-pipeline'],
  {
    document_id: 'doc-123',
    document_path: '/path/to/document.pdf'
  }
);

console.log('Workflow result:', result);
// {
//   execution_id: '...',
//   state: 'completed',
//   duration_ms: 5000,
//   steps_completed: 4,
//   steps_failed: 0,
//   output_data: { ... }
// }

// Define custom workflow
const customWorkflow: WorkflowDefinition = {
  workflow_id: 'custom-analysis',
  workflow_name: 'Custom Analysis',
  description: 'My custom analysis workflow',
  steps: [
    {
      step_id: 'step1',
      step_name: 'First Step',
      service: 'my-service',
      operation: 'analyze',
      handler: async (context) => {
        // Do work
        return { result: 'success' };
      },
      timeout_ms: 30000,
      retry_on_failure: true,
      max_retries: 3,
      circuit_breaker: true
    }
  ],
  parallel: false,
  on_failure: 'stop',
  timeout_ms: 120000
};

const customResult = await workflowEngine.execute(customWorkflow, { data: 'test' });
```

### 3. Dead Letter Queue (`dead-letter-queue.ts`)

Handles failed events with retry logic and monitoring.

**Features:**
- Event persistence
- Exponential backoff retry
- Configurable retry policies
- DLQ monitoring endpoints
- Alert integration hooks

**Retry Policy:**
```typescript
{
  max_retries: 3,
  initial_delay_ms: 1000,     // Start with 1 second
  max_delay_ms: 60000,        // Max 60 seconds
  backoff_multiplier: 2,       // Double each time
  retry_on: ['NetworkError', 'TimeoutError', 'TransientError']
}
```

**Usage:**
```typescript
import { deadLetterQueue } from './orchestration';

// Enqueue failed event
try {
  await processEvent(event);
} catch (error) {
  await deadLetterQueue.enqueue(event, error, {
    handler: 'vector-db-adapter',
    operation: 'index-embeddings'
  });
}

// Process retries (run periodically)
setInterval(async () => {
  const processed = await deadLetterQueue.processRetry(async (event) => {
    await processEvent(event);
  });
  console.log(`Retried ${processed} events`);
}, 30000); // Every 30 seconds

// Get statistics
const stats = deadLetterQueue.getStats();
console.log('DLQ Stats:', stats);
// {
//   total_entries: 10,
//   pending_entries: 5,
//   processed_entries: 3,
//   failed_permanently: 2,
//   by_event_type: {
//     'KnowledgeExtracted': 4,
//     'VectorIndexed': 3,
//     ...
//   }
// }

// Manual retry
await deadLetterQueue.retryEntry('entry-id', async (event) => {
  await processEvent(event);
});

// Get pending entries
const pending = deadLetterQueue.getEntries({
  processed: false,
  event_type: 'KnowledgeExtracted'
});
```

### 4. Correlation Tracker (`correlation-tracker.ts`)

Distributed tracing and request lineage tracking.

**Features:**
- UUID v4 correlation ID generation
- Service path tracking
- Distributed tracing spans
- Request lineage visualization
- Integration with logging

**Usage:**
```typescript
import { correlationTracker } from './orchestration';

// Create correlation context
const context = correlationTracker.createContext({
  user_id: '12345',
  workflow: 'rag-pipeline'
});

// Record service calls
correlationTracker.recordServiceCall(context, 'ragbits-adapter', 'extract-chunks');
correlationTracker.recordServiceCall(context, 'vector-db-adapter', 'index-embeddings');

// Create distributed trace span
const span = correlationTracker.createSpan(
  context.trace_id!,
  undefined,
  'orchestration',
  'process-document',
  { document_id: 'doc-123' }
);

// Do work...

// Complete span
correlationTracker.completeSpan(span, 'ok');

// Calculate duration
const duration = correlationTracker.calculateDuration(context);
console.log(`Request took ${duration}ms`);

// Format for logging
const logContext = correlationTracker.formatForLogging(context);
logger.info('Request completed', logContext);

// Extract correlation ID from incoming headers
const correlationId = correlationTracker.extractOrGenerate({
  'x-correlation-id': 'incoming-123',
  'x-request-id': 'req-456'
});

// Inject correlation ID into outgoing headers
const headers = correlationTracker.injectIntoHeaders(context);
// {
//   'x-correlation-id': '...',
//   'x-trace-id': '...',
//   'x-parent-id': '...'
// }
```

## Event Types

All events follow the canonical schema:

```typescript
interface BaseEvent {
  id: string;                    // UUID for idempotency
  type: string;                  // Event type
  timestamp: string;             // ISO-8601 UTC
  correlation_id: string;        // Correlation ID
  source_service: string;        // Source service name
  data: any;                     // Event data
  metadata?: Record<string, any>;
}
```

**Available Event Types:**

1. **KnowledgeExtracted** - Knowledge chunks extracted from document
2. **ProofVerified** - Formal proof verified by Z3/LeanAide
3. **GraphUpdated** - Knowledge graph updated
4. **VectorIndexed** - Embeddings indexed in vector DB
5. **RAGRetrieved** - RAG retrieved relevant chunks
6. **WorkflowStarted** - Workflow execution started
7. **WorkflowCompleted** - Workflow completed successfully
8. **WorkflowFailed** - Workflow execution failed

## Integration Examples

### Example 1: RAG Pipeline with Event Bus

```typescript
import { eventBus } from './orchestration';
import { createBaseEvent } from './orchestration';

// RAGBits adapter publishes event
async function processDocument(document: Document) {
  const chunks = await ragbitsAdapter.extractChunks(document);

  const event = createBaseEvent(
    'KnowledgeExtracted',
    'ragbits-adapter',
    correlationId,
    {
      document_id: document.id,
      chunk_count: chunks.length,
      chunks: chunks,
      extraction_method: 'recursive'
    }
  );

  await eventBus.publish(event);
}

// Vector DB adapter subscribes to event
eventBus.subscribe('KnowledgeExtracted', async (event) => {
  if (event.type === 'KnowledgeExtracted') {
    const embeddings = await createEmbeddings(event.data.chunks);
    await indexEmbeddings(embeddings);

    // Publish next event
    const vectorEvent = createBaseEvent(
      'VectorIndexed',
      'vector-db-adapter',
      event.correlation_id,
      {
        vector_db_type: 'chroma',
        index_id: 'idx-123',
        embedding_count: embeddings.length,
        embedding_model: 'text-embedding-ada-002',
        dimension: 1536,
        index_type: 'create'
      }
    );

    await eventBus.publish(vectorEvent);
  }
});
```

### Example 2: Cross-Validation Workflow

```typescript
import { workflowEngine, PREDEFINED_WORKFLOWS } from './orchestration';

// Execute Z3 → LeanAide cross-validation
const result = await workflowEngine.execute(
  PREDEFINED_WORKFLOWS['z3-lean-validation'],
  {
    proof_id: 'proof-123',
    theorem_name: 'MyTheorem',
    proof_content: '...'
  }
);

if (result.state === 'completed') {
  const validationResult = result.output_data['cross-validate'];

  if (validationResult.cross_validated) {
    console.log('✓ Proof verified by both Z3 and LeanAide');
  } else {
    console.log('⚠ Proof verification mismatch');
    console.log('  Z3:', validationResult.z3_result);
    console.log('  LeanAide:', validationResult.lean_result);
  }
}
```

### Example 3: Error Handling with DLQ

```typescript
import { deadLetterQueue } from './orchestration';

// Adapter with error handling
async function processWithDLQ(event: Event) {
  try {
    // Retry transient failures
    await retryWithBackoff(
      async () => {
        return await processEvent(event);
      },
      { max_retries: 3 }
    );
  } catch (error) {
    // Logic failures go to DLQ
    await deadLetterQueue.enqueue(event, error, {
      handler: 'my-adapter',
      operation: 'process-event'
    });
  }
}

// Periodic DLQ processor
setInterval(async () => {
  const processed = await deadLetterQueue.processRetry(async (event) => {
    await processWithDLQ(event);
  });

  if (processed > 0) {
    logger.info('DLQ retry batch completed', {
      processed,
      timestamp: new Date().toISOString()
    });
  }
}, 60000); // Every minute
```

## Configuration

### Environment Variables

```bash
# Event Bus Configuration
EVENT_BUS_TYPE=memory              # memory|redis|rabbitmq|kafka
EVENT_BUS_URL=redis://localhost:6379  # Required for non-memory

# Optional: Persistence
EVENT_PERSISTENCE_ENABLED=true

# Optional: Circuit Breaker
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_MS=60000

# Optional: DLQ
DLQ_ENABLED=true
DLQ_MAX_RETRIES=3
DLQ_INITIAL_DELAY_MS=1000
DLQ_MAX_DELAY_MS=60000
```

## Monitoring

### Event Bus Statistics

```typescript
const stats = eventBus.getStats();
console.log(stats);
// {
//   type: 'memory',
//   events_published: 1234,
//   events_received: 4567,
//   events_failed: 5,
//   subscriptions: 12,
//   uptime_seconds: 3600
// }
```

### DLQ Statistics

```typescript
const stats = deadLetterQueue.getStats();
console.log(stats);
// {
//   total_entries: 10,
//   pending_entries: 5,
//   processed_entries: 3,
//   failed_permanently: 2,
//   by_event_type: {
//     'KnowledgeExtracted': 4,
//     'VectorIndexed': 3,
//     'GraphUpdated': 3
//   }
// }
```

### Workflow Statistics

```typescript
const active = workflowEngine.getActiveWorkflows();
const completed = workflowEngine.getCompletedWorkflows();

console.log(`Active workflows: ${active.length}`);
console.log(`Completed workflows: ${completed.length}`);
```

## Failure Management

### Transient Failures (Network Blips)
- **Strategy**: Exponential backoff retry with jitter
- **Implementation**: Use `retryWithBackoff()` from lib
- **Example**: Timeout, connection refused

### Logic Failures (Bad Data)
- **Strategy**: Dead Letter Queue
- **Implementation**: Enqueue to DLQ with context
- **Example**: Validation error, schema mismatch

### System Failures (Target Down)
- **Strategy**: Circuit Breaker
- **Implementation**: Use `CircuitBreaker` class
- **Example**: Service crash, database down

## Best Practices

### 1. Always Use Correlation IDs
```typescript
const context = correlationTracker.createContext();
logger.info('Processing started', {
  correlation_id: context.correlation_id
});
```

### 2. Make Operations Idempotent
```typescript
// Check before creating
if (!await resourceExists(id)) {
  await createResource(id);
}
```

### 3. Use Timeouts
```typescript
const result = await Promise.race([
  operation(),
  timeout(5000) // 5 second timeout
]);
```

### 4. Log Structured Data
```typescript
logger.info('Chunk indexed', {
  correlation_id: context.correlation_id,
  chunk_id: chunk.id,
  index_id: index.id,
  duration_ms: duration
});
```

### 5. Handle Failures Gracefully
```typescript
try {
  await riskyOperation();
} catch (error) {
  if (isTransient(error)) {
    // Retry
  } else if (isLogicError(error)) {
    // Send to DLQ
  } else {
    // Circuit breaker
  }
}
```

## Testing

```typescript
import { eventBus, workflowEngine } from './orchestration';

describe('Orchestration', () => {
  it('should publish and receive events', async () => {
    let receivedEvent = null;

    eventBus.subscribe('TestEvent', async (event) => {
      receivedEvent = event;
    });

    const testEvent = createBaseEvent(
      'TestEvent',
      'test-service',
      'test-123',
      { test: 'data' }
    );

    await eventBus.publish(testEvent);

    expect(receivedEvent).toBeTruthy();
    expect(receivedEvent.type).toBe('TestEvent');
  });

  it('should execute workflow', async () => {
    const workflow: WorkflowDefinition = {
      workflow_id: 'test-workflow',
      workflow_name: 'Test Workflow',
      description: 'Test',
      steps: [
        {
          step_id: 'step1',
          step_name: 'Step 1',
          service: 'test',
          operation: 'test',
          handler: async () => ({ result: 'ok' })
        }
      ]
    };

    const result = await workflowEngine.execute(workflow, {});

    expect(result.state).toBe('completed');
    expect(result.steps_completed).toBe(1);
  });
});
```

## License

MIT

## Contributing

See [CLAUDE.md](../../CLAUDE.md) for the Federation Constitution.
