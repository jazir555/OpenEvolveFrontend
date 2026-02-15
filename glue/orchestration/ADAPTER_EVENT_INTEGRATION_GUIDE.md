# Adapter Event Bus Integration Guide

## Overview

This guide explains how to integrate adapters with the central event bus following the Federation Constitution's failure management strategy:

- **Transient Failure** → Exponential Backoff Retry (Jittered)
- **Logic Failure** → Dead Letter Queue (DLQ)
- **System Failure** → Circuit Breaker

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Event Bus (Central)                        │
│  - Publish/Subscribe pattern                                      │
│  - Event persistence for replay                                   │
│  - Circuit breaker integration                                   │
│  - DLQ routing                                                    │
└─────────────────────────────────────────────────────────────────┘
         ↑                                    ↓
    Publish                             Subscribe
         │                                    │
    ┌────┴────┐                        ┌─────┴─────┐
    │ Adapter │                        │  Adapter  │
    │  (RAG)   │                        │ (Vector)  │
    └─────────┘                        └───────────┘
```

## Event Flow

### 1. Successful Operation

```
Adapter → Execute Operation → Publish Event → Event Bus → Subscribers
         (with retry,          (success)
          circuit breaker)
```

### 2. Transient Failure (Network Timeout)

```
Adapter → Execute Operation → Fail → Retry with Backoff → Success → Publish Event
                                      ↑
                                      └── Max 3 retries
```

### 3. Logic Failure (Bad Data)

```
Adapter → Execute Operation → Fail → Send to DLQ → Publish Failure Event
                                      (manual retry)
```

### 4. System Failure (Service Down)

```
Adapter → Execute Operation → Fail → Circuit Breaker Opens → Immediate Fail
                                                            ↓
                                                      Retry after timeout
```

## Available Events

### Knowledge Extracted
Fired when RAGBits extracts knowledge chunks from a document.

```typescript
{
  type: 'KnowledgeExtracted',
  data: {
    document_id: string;
    chunk_count: number;
    chunks: Array<{
      chunk_id: string;
      content: string;
      metadata: Record<string, any>;
    }>;
    extraction_method: string;
  }
}
```

### Proof Verified
Fired when Z3 or LeanAide verifies a formal proof.

```typescript
{
  type: 'ProofVerified',
  data: {
    proof_id: string;
    theorem_name: string;
    verification_system: 'z3' | 'lean-aide' | 'both';
    status: 'valid' | 'invalid' | 'unknown';
    verification_time_ms: number;
  }
}
```

### Graph Updated
Fired when Graphiti or KarateClub updates the knowledge graph.

```typescript
{
  type: 'GraphUpdated',
  data: {
    graph_id: string;
    update_type: 'node_added' | 'edge_added' | 'node_updated' | 'graph_merged';
    node_count?: number;
    edge_count?: number;
    graph_system: 'graphiti' | 'karate-club' | 'both';
  }
}
```

### Vector Indexed
Fired when Vector DB indexes embeddings.

```typescript
{
  type: 'VectorIndexed',
  data: {
    vector_db_type: 'chroma' | 'pinecone' | 'weaviate' | 'qdrant';
    index_id: string;
    embedding_count: number;
    embedding_model: string;
    dimension: number;
    index_type: 'create' | 'update' | 'delete';
  }
}
```

## Creating an Event-Enabled Adapter

### Step 1: Extend EventEnabledAdapter

```typescript
import { EventEnabledAdapter } from '../../lib/event-enabled-adapter';
import { EventBus } from '../../orchestration/event-bus';
import { Event, isEventType } from '../../orchestration/event-types';

class MyAdapter extends EventEnabledAdapter {
  private client: MyClient;

  constructor(config: {
    api_url: string;
    timeout_ms: number;
    eventBus: EventBus;
  }) {
    super('my-adapter', {
      eventBus: config.eventBus,
      publishEvents: true,
      subscribeToEvents: true,
      dlqEnabled: true,
      circuitBreakerEnabled: true,
      retryConfig: {
        max_retries: 3,
        base_delay_ms: 1000,
      },
    });

    this.client = new MyClient({
      api_url: config.api_url,
      timeout_ms: config.timeout_ms,
    });

    this.setupEventSubscriptions();
  }
}
```

### Step 2: Setup Event Subscriptions

```typescript
private setupEventSubscriptions(): void {
  // Subscribe to relevant events
  this.subscribeToEvent('KnowledgeExtracted', async (event) => {
    if (isEventType(event, 'KnowledgeExtracted')) {
      await this.handleKnowledgeExtracted(event);
    }
  });
}

private async handleKnowledgeExtracted(event: Event): Promise<void> {
  if (!isEventType(event, 'KnowledgeExtracted')) {
    return;
  }

  this.logger.info('Processing KnowledgeExtracted event', {
    event_id: event.id,
    correlation_id: event.correlation_id,
    document_id: event.data.document_id,
  });

  // Process the event
  for (const chunk of event.data.chunks) {
    await this.processChunk(chunk.content, chunk.metadata);
  }
}
```

### Step 3: Implement Operations with Event Publishing

```typescript
async processData(input: string): Promise<AdapterOperationResult> {
  return this.executeOperation(
    'process-data',
    async () => {
      return await this.client.process({ input });
    },
    'DataProcessed',
    {
      input_id: randomUUID(),
      input_length: input.length,
    }
  );
}
```

### Step 4: Handle Failures Appropriately

The `EventEnabledAdapter` base class automatically handles:

1. **Transient Failures**: Retry with exponential backoff
2. **Logic Failures**: Send to DLQ
3. **System Failures**: Circuit breaker opens

Customize failure classification by overriding `isLogicFailure`:

```typescript
protected isLogicFailure(error: Error): boolean {
  // Custom logic to determine if error is logic or transient
  const transientPatterns = [
    'timeout',
    'ECONNREFUSED',
    'rate limit',
  ];

  const errorMsg = error.message.toLowerCase();
  const isTransient = transientPatterns.some((pattern) =>
    errorMsg.includes(pattern.toLowerCase())
  );

  return !isTransient; // If not transient, it's a logic failure
}
```

## Complete Example

```typescript
import { EventEnabledAdapter, AdapterOperationResult } from '../../lib/event-enabled-adapter';
import { EventBus } from '../../orchestration/event-bus';
import { Event, isEventType, createBaseEvent } from '../../orchestration/event-types';
import { randomUUID } from 'crypto';

interface GraphitiAdapterConfig {
  api_url: string;
  timeout_ms: number;
  eventBus: EventBus;
}

export class GraphitiEventAdapter extends EventEnabledAdapter {
  private client: any; // GraphitiClient

  constructor(config: GraphitiAdapterConfig) {
    super('graphiti-adapter', {
      eventBus: config.eventBus,
      publishEvents: true,
      subscribeToEvents: true,
      dlqEnabled: true,
      circuitBreakerEnabled: true,
      retryConfig: {
        max_retries: 3,
        base_delay_ms: 1000,
      },
    });

    this.client = new GraphitiClient({
      api_url: config.api_url,
      timeout_ms: config.timeout_ms,
    });

    this.setupEventSubscriptions();
  }

  private setupEventSubscriptions(): void {
    // Subscribe to knowledge extraction events to build graph
    this.subscribeToEvent('KnowledgeExtracted', async (event) => {
      if (isEventType(event, 'KnowledgeExtracted')) {
        await this.handleKnowledgeExtracted(event);
      }
    });
  }

  private async handleKnowledgeExtracted(event: Event): Promise<void> {
    if (!isEventType(event, 'KnowledgeExtracted')) {
      return;
    }

    this.logger.info('Building graph from extracted knowledge', {
      event_id: event.id,
      correlation_id: event.correlation_id,
      chunk_count: event.data.chunk_count,
    });

    // Add nodes for each chunk
    for (const chunk of event.data.chunks) {
      await this.addNode({
        id: chunk.chunk_id,
        type: 'knowledge_chunk',
        properties: {
          content: chunk.content,
          ...chunk.metadata,
        },
      }, event.correlation_id);
    }

    // Publish graph update event
    await this.publishEvent(
      'GraphUpdated',
      {
        graph_id: event.data.document_id,
        update_type: 'node_added',
        node_count: event.data.chunk_count,
        graph_system: 'graphiti',
      },
      event.correlation_id
    );
  }

  async addNode(
    node: { id: string; type: string; properties: any },
    correlationId?: string
  ): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'add-node',
      async () => {
        return await this.client.addNode(node);
      },
      'GraphNodeAdded',
      {
        node_id: node.id,
        node_type: node.type,
      }
    );
  }

  async addEdge(
    edge: { source: string; target: string; type: string; properties: any },
    correlationId?: string
  ): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'add-edge',
      async () => {
        return await this.client.addEdge(edge);
      },
      'GraphEdgeAdded',
      {
        edge_id: `${edge.source}-${edge.target}`,
        edge_type: edge.type,
      }
    );
  }

  async search(
    query: string,
    limit = 10,
    correlationId?: string
  ): Promise<AdapterOperationResult> {
    return this.executeOperation(
      'graph-search',
      async () => {
        return await this.client.search({ query, limit });
      },
      'GraphSearched',
      {
        query,
        result_count: limit,
      }
    );
  }
}
```

## Environment Variables

All adapters require the following environment variables:

```bash
# Event Bus Configuration
EVENT_BUS_TYPE=memory              # memory | redis | rabbitmq | kafka
EVENT_BUS_URL=                     # Required for redis, rabbitmq, kafka

# Adapter Configuration
YOUR_ADAPTER_API_URL=http://localhost:8000
TIMEOUT_MS=5000
MAX_RETRIES=3
BASE_DELAY_MS=1000

# Optional: Circuit Breaker
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_MS=30000
```

## Testing

### Unit Test Example

```typescript
import { EventBus } from '../../orchestration/event-bus';
import { MyAdapter } from './my-adapter';

describe('MyAdapter', () => {
  let adapter: MyAdapter;
  let eventBus: EventBus;

  beforeEach(() => {
    eventBus = new EventBus({ type: 'memory' });
    adapter = new MyAdapter({
      api_url: 'http://localhost:8000',
      timeout_ms: 5000,
      eventBus,
    });
  });

  it('should publish event on successful operation', async () => {
    let eventReceived = false;

    eventBus.subscribe('DataProcessed', async (event) => {
      eventReceived = true;
      expect(event.data.input_length).toBe(100);
    });

    const result = await adapter.processData('x'.repeat(100));

    expect(result.success).toBe(true);
    expect(result.event_published).toBe(true);
    expect(eventReceived).toBe(true);
  });

  it('should retry on transient failure', async () => {
    let attemptCount = 0;

    // Mock client to fail first 2 attempts
    adapter.client = {
      process: async () => {
        attemptCount++;
        if (attemptCount < 3) {
          throw new Error('Timeout');
        }
        return { success: true };
      },
    };

    const result = await adapter.processData('test');

    expect(result.success).toBe(true);
    expect(attemptCount).toBe(3);
  });

  it('should send logic failures to DLQ', async () => {
    const dlq = eventBus.getDLQ()!;

    // Mock client to fail with validation error
    adapter.client = {
      process: async () => {
        throw new Error('Validation failed: invalid input');
      },
    };

    const result = await adapter.processData('');

    expect(result.success).toBe(false);

    const dlqStats = dlq.getStats();
    expect(dlqStats.total_entries).toBe(1);
  });
});
```

## Monitoring

### Check Event Bus Stats

```typescript
const stats = eventBus.getStats();
console.log('Event Bus Stats:', stats);
// {
//   type: 'memory',
//   events_published: 1234,
//   events_received: 4567,
//   events_failed: 2,
//   subscriptions: 10,
//   uptime_seconds: 3600
// }
```

### Check DLQ Stats

```typescript
const dlq = eventBus.getDLQ();
const dlqStats = dlq.getStats();
console.log('DLQ Stats:', dlqStats);
// {
//   total_entries: 5,
//   pending_entries: 3,
//   processed_entries: 2,
//   failed_permanently: 0,
//   by_event_type: {
//     KnowledgeExtracted: 2,
//     VectorIndexed: 1,
//     GraphUpdated: 2
//   }
// }
```

### Check Circuit Breaker State

```typescript
const cbState = adapter.getCircuitBreakerState();
console.log('Circuit Breaker State:', cbState);
// {
//   state: 'closed',
//   failure_count: 2,
//   success_count: 10,
//   last_failure_time: undefined,
//   last_state_change: 2025-01-15T10:30:00.000Z
// }
```

### Replay Events

```typescript
// Replay all KnowledgeExtracted events from the last hour
const replayed = await eventBus.replay({
  event_type: 'KnowledgeExtracted',
  from_timestamp: new Date(Date.now() - 3600000).toISOString(),
});

console.log(`Replayed ${replayed} events`);
```

## Best Practices

### 1. Always Use Correlation IDs

```typescript
const correlationId = randomUUID();
const result = await adapter.processData(input, correlationId);
```

### 2. Check Circuit Breaker Before Critical Operations

```typescript
const cbState = adapter.getCircuitBreakerState();
if (cbState && cbState.state === 'open') {
  logger.warn('Circuit breaker is open, using fallback');
  return fallbackData();
}
```

### 3. Monitor DLQ and Process Manually When Needed

```typescript
const dlq = eventBus.getDLQ();
const entries = dlq.getEntries({ processed: false });

for (const entry of entries) {
  if (entry.retry_count < entry.max_retries) {
    // Manual retry
    await dlq.retryEntry(entry.id, async (event) => {
      return await processEvent(event);
    });
  }
}
```

### 4. Use Idempotent Operations

```typescript
async upsertVector(vector: VectorEntry): Promise<AdapterOperationResult> {
  return this.executeOperation(
    'upsert-vector',
    async () => {
      // Check if vector exists first (idempotency)
      const existing = await this.client.getVector(vector.id);
      if (existing) {
        return await this.client.updateVector(vector);
      } else {
        return await this.client.createVector(vector);
      }
    },
    'VectorIndexed',
    { vector_id: vector.id }
  );
}
```

### 5. Log Structured Events

```typescript
this.logger.info('Processing started', {
  correlation_id: correlationId,
  operation: 'process-data',
  input_size: input.length,
  adapter: this.adapterName,
});
```

## Troubleshooting

### Events Not Being Received

1. Check if adapter is subscribing to correct event type
2. Verify event bus type is configured correctly
3. Check correlation IDs match

### Circuit Breaker Not Opening

1. Check failure threshold configuration
2. Verify errors are being thrown correctly
3. Check circuit breaker state logs

### DLQ Filling Up

1. Check for logic failures vs transient failures
2. Verify error classification in `isLogicFailure`
3. Process DLQ entries manually if needed

### High Memory Usage

1. Clear event history periodically: `eventBus.clearHistory()`
2. Process DLQ entries to free memory
3. Adjust event persistence settings

## Federation Constitution Compliance

This implementation follows all 6 commandments:

1. ✅ **Air Gap**: No imports from core-projects
2. ✅ **Runtime Truth**: Probe before use
3. ✅ **Untouchable DB**: Read-only via adapters
4. ✅ **Idempotency**: Safe to retry operations
5. ✅ **Configuration Explicitness**: All via ENV vars
6. ✅ **UTC**: All timestamps in UTC ISO-8601
