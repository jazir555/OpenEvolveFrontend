# RAGBits-Graphiti Bidirectional Sync Adapter

**Production-ready bidirectional synchronization adapter between RAGBits (RAG system) and Graphiti (Temporal Knowledge Graph).**

## Overview

This adapter implements the Federation Constitution's Anti-Corruption Layer pattern, providing seamless bidirectional synchronization between document-based retrieval (RAGBits) and knowledge graph-based temporal reasoning (Graphiti).

### Key Features

- **Bidirectional Sync**: RAGBits ↔ Graphiti data synchronization
- **Conflict Detection**: Automatic detection of entity, temporal, and semantic conflicts
- **Conflict Resolution**: Multiple resolution strategies (source_wins, target_wins, newest_wins, merge)
- **Circuit Breakers**: Prevents cascading failures with automatic recovery
- **Idempotent Operations**: Safe to run multiple times (Law of Idempotency)
- **Structured Logging**: JSON Lines format with correlation IDs for distributed tracing
- **Configuration Explicitness**: All settings via environment variables (no magic defaults)

## Architecture

```
┌─────────────────┐
│   RAGBits       │
│  (RAG System)   │
└────────┬────────┘
         │
         │ Document Ingestion
         ↓
┌─────────────────────────────────────────┐
│         Sync Manager                     │
│  ┌─────────────────────────────────┐    │
│  │  RAGBits → Graphiti Sync        │    │
│  │  - Chunk to Episode conversion  │    │
│  │  - Entity extraction            │    │
│  │  - Temporal metadata            │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  Graphiti → RAGBits Sync        │    │
│  │  - Entity to keyword mapping    │    │
│  │  - Boost factors for retrieval  │    │
│  │  - Query enhancement            │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │  Conflict Detector              │    │
│  │  - Entity mismatches            │    │
│  │  - Temporal inconsistencies     │    │
│  │  - Semantic conflicts           │    │
│  └─────────────────────────────────┘    │
└────────┬────────────────────────────────┘
         │
         │ Knowledge Graph Updates
         ↓
┌─────────────────┐
│   Graphiti      │
│ (Knowledge      │
│    Graph)       │
└─────────────────┘
```

## Installation

```bash
# Install dependencies
npm install

# Build TypeScript
npm run build

# Run contract tests
npm test
```

## Configuration

All configuration via environment variables (Law of Configuration Explicitness):

```bash
# Required
export RAGBITS_API_URL="http://ragbits:8000"
export GRAPHITI_API_URL="http://graphiti:8000"

# Optional (with defaults)
export SYNC_TIMEOUT_MS=30000
export SYNC_MAX_RETRIES=3
export SYNC_BATCH_SIZE=10
export SYNC_INTERVAL_MS=300000
export SYNC_ENABLED=true
export SYNC_BIDIRECTIONAL=true
```

## Usage

### Basic Sync Manager Setup

```typescript
import SyncManager from '@openevolve/ragbits-graphiti-sync';

// Initialize sync manager (reads from environment)
const syncManager = new SyncManager();

// Sync on document ingestion
const document = {
  id: 'doc-123',
  content: 'Document content here...',
  source: 'manual-upload',
  metadata: { author: 'John Doe' }
};

const result = await syncManager.syncOnIngest(document, correlationId);

console.log('Sync completed:', result.sync_result);
console.log('Conflicts detected:', result.conflict_report?.total_conflicts || 0);
```

### Manual Sync with Specification

```typescript
import { SyncSpec, SyncDirection } from '@openevolve/ragbits-graphiti-sync';

const spec: SyncSpec = {
  direction: SyncDirection.bidirectional,
  entity_ids: ['entity-1', 'entity-2'],
  episode_ids: ['episode-1'],
  conflict_resolution: 'newest_wins',
  timeout_ms: 30000
};

const result = await syncManager.syncManual(spec, correlationId);
```

### Scheduled Sync

```typescript
// Run scheduled sync every N minutes
setInterval(async () => {
  const correlationId = uuidv4();
  const result = await syncManager.syncOnSchedule(correlationId);
  console.log('Scheduled sync completed:', result.sync_result);
}, 5 * 60 * 1000); // Every 5 minutes
```

### Enhanced Retrieval with Graph Context

```typescript
import GraphitiToRAGBitsSync from '@openevolve/ragbits-graphiti-sync';

const graphitiToRAGBits = new GraphitiToRAGBitsSync(config);

// Enhance query with knowledge graph entities
const query = "What are the relationships between entities?";

const enhancedQuery = await graphitiToRAGBits.enhanceRetrieval(query, correlationId);

console.log('Original query:', enhancedQuery.original_query);
console.log('Enhanced query:', enhancedQuery.enhanced_query);
console.log('Entities found:', enhancedQuery.entities);
console.log('Boost factors:', enhancedQuery.boost_factors);
```

### Conflict Resolution

```typescript
import { ConflictResolution } from '@openevolve/ragbits-graphiti-sync';

// Resolve detected conflicts
const conflicts = conflictReport.conflicts;

const resolutionResult = await syncManager.resolveConflicts(
  conflicts,
  ConflictResolution.newest_wins,
  correlationId
);

console.log('Resolved:', resolutionResult.resolved.length);
console.log('Failed:', resolutionResult.failed.length);
```

## Sync Triggers

### 1. On Document Ingestion
Automatically triggered when documents are added to RAGBits:

```typescript
// In your document ingestion pipeline
await syncManager.syncOnIngest(document, correlationId);
```

### 2. On Entity Update
Triggered when knowledge graph entities are updated:

```typescript
// Sync entity metadata to RAGBits for enhanced retrieval
await graphitiToRAGBits.syncEntity(entity, correlationId);
```

### 3. Scheduled Sync
Periodic sync to keep systems aligned:

```typescript
// Configure in your scheduler
setInterval(() => {
  syncManager.syncOnSchedule(correlationId);
}, config.sync.interval_ms);
```

### 4. Manual Sync
On-demand sync via API endpoint:

```bash
curl -X POST http://sync-adapter:8000/api/sync/manual \
  -H "Content-Type: application/json" \
  -d '{
    "direction": "bidirectional",
    "entity_ids": ["entity-1"],
    "conflict_resolution": "newest_wins"
  }'
```

## Monitoring & Observability

### Sync Statistics

```typescript
const stats = syncManager.getStats();

console.log('Total syncs:', stats.total_syncs);
console.log('Success rate:', stats.success_rate.toFixed(2) + '%');
console.log('Avg duration:', stats.avg_duration_ms.toFixed(2) + 'ms');
console.log('Conflict rate:', stats.conflict_rate.toFixed(2) + '%');
console.log('Active operations:', stats.active_operations);
```

### Logging

All logs use structured JSON Lines format:

```json
{"level":"info","msg":"Sync on ingest started","timestamp":"2025-02-03T12:34:56.789Z","correlation_id":"550e8400-e29b-41d4-a716-446655440000","source_service":"ragbits","target_service":"graphiti","document_id":"doc-123"}
```

### Metrics

Key metrics to monitor:

- `sync_success_rate` - Percentage of successful sync operations
- `avg_sync_latency_ms` - Average duration of sync operations
- `conflict_count` - Number of conflicts detected
- `conflict_resolution_rate` - Percentage of conflicts auto-resolved
- `circuit_breaker_state` - Current state of circuit breaker

## Probe Scripts

Verify sync functionality before deployment:

```bash
# Test basic sync operations
npm run probe:sync

# Test conflict detection
npm run probe:conflicts

# Test bidirectional sync
npm run probe:bidirectional

# Run all probes
npm run probe:all
```

## Error Handling

The adapter implements comprehensive error handling:

```typescript
try {
  const result = await syncManager.syncOnIngest(document, correlationId);

  if (result.sync_result.status === 'completed') {
    console.log('Sync successful');
  } else if (result.sync_result.status === 'partially_completed') {
    console.warn('Partial sync:', result.sync_result.operations_failed, 'failed');
  } else {
    console.error('Sync failed:', result.errors);
  }
} catch (error) {
  // Handle unexpected errors
  console.error('Sync error:', error);

  // Check circuit breaker state
  const cbStats = ragbitsToGraphiti.getCircuitBreakerStats();
  if (cbStats.state === 'open') {
    console.error('Circuit breaker is OPEN, service is down');
  }
}
```

## Federation Constitution Compliance

This adapter follows all Federation Constitution laws:

### ✓ Law of the "AIR GAP" (Source Code Isolation)
No imports from `core-projects/`. All integration via API calls.

### ✓ Law of "RUNTIME TRUTH" (Anti-Hallucination)
Probe scripts verify API functionality before implementation.

### ✓ Law of the "UNTOUCHABLE DB" (Read-Only State)
SELECT privileges only. No direct database writes.

### ✓ Law of IDEMPOTENCY (The Replayability Pact)
All operations safe to run multiple times.

### ✓ Law of Configuration Explicitness
All configurable values via environment variables. No magic defaults.

### ✓ Law of UTC
All timestamps in UTC ISO-8601 format.

## Development

### Running Tests

```bash
cd tests
npm install
npm test
npm run test:coverage
```

### Building

```bash
npm run build
```

### Linting

```bash
npm run lint
npm run format
```

## API Reference

See `ADR.md` for detailed architectural decisions and API documentation.

## Troubleshooting

### Circuit Breaker is Open

```bash
# Check circuit breaker status
curl http://sync-adapter:8000/api/health

# Reset circuit breaker (after service recovery)
curl -X POST http://sync-adapter:8000/api/circuit-breaker/reset
```

### High Conflict Rate

```bash
# Check conflict statistics
curl http://sync-adapter:8000/api/sync/stats

# Adjust conflict resolution strategy
export SYNC_CONFLICT_RESOLUTION=merge
```

### Sync Latency Issues

```bash
# Check sync metrics
curl http://sync-adapter:8000/api/sync/metrics

# Adjust batch size for better throughput
export SYNC_BATCH_SIZE=20

# Increase timeout if needed
export SYNC_TIMEOUT_MS=60000
```

## License

Apache-2.0

## Support

For issues and questions, please open an issue in the OpenEvolve Federation repository.
