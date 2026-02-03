# RAGBits Adapter

Following the **Federation Constitution**, this adapter provides a normalized interface to RAGBits (Retrieval-Augmented Generation) services.

## Architecture

```
[OpenEvolve Core] --> [RAGBits Adapter (Canonical Layer)] --> [RAGBits Server]
                                                                       |
                                                                       +-- [Vector DB]
                                                                       +-- [Embedding Model]
```

## Features

- **Circuit Breaker**: Fail-fast when RAGBits is down
- **Exponential Backoff Retry**: Jittered retries for transient failures
- **Structured Logging**: JSON Lines with correlation IDs
- **Canonical Schema**: Anti-corruption layer for data normalization
- **Idempotency**: Safe to retry all operations
- **UTC Timestamps**: All times in UTC ISO-8601 format

## Installation

```bash
cd glue/adapters/ragbits-adapter
npm install
```

## Configuration

### Required Environment Variables

```bash
# MANDATORY - No defaults
export RAGBITS_API_URL=http://ragbits-core:8002
export TIMEOUT_MS=5000
```

### Optional Environment Variables

```bash
# Circuit Breaker
export CIRCUIT_FAILURE_THRESHOLD=5
export CIRCUIT_SUCCESS_THRESHOLD=2
export CIRCUIT_TIMEOUT_MS=60000

# Retry
export RETRY_MAX_ATTEMPTS=3
export RETRY_BASE_DELAY_MS=1000
export RETRY_MAX_DELAY_MS=10000
export RETRY_EXPONENTIAL=2.0
export RETRY_JITTER=0.1

# Authentication
export RAGBITS_API_KEY=your-api-key-here
```

## Usage

### Basic Usage

```typescript
import { RAGBitsAdapter } from '@openevolve/ragbits-adapter';

// Create adapter (will crash if RAGBITS_API_URL missing)
const adapter = new RAGBitsAdapter({
  api_url: process.env.RAGBITS_API_URL!,
  timeout_ms: parseInt(process.env.TIMEOUT_MS || '5000'),
  api_key: process.env.RAGBITS_API_KEY,
});

// Test connection
const connected = await adapter.testConnection();
if (!connected) {
  throw new Error('RAGBits is not available');
}

// Search documents
const results = await adapter.search(
  'machine learning algorithms',  // query
  5,                                // top_k
  { document_type: 'markdown' },    // filters
  'correlation-id-123'              // correlation_id
);

console.log(`Found ${results.total_results} results`);
results.results.forEach(result => {
  console.log(`[${result.score.toFixed(2)}] ${result.content}`);
});
```

### Document Ingestion

```typescript
// Ingest single document
const ingestResult = await adapter.ingest(
  'This is the document content to ingest...',
  {
    source: 'user-guide.pdf',
    author: 'John Doe',
    tags: ['guide', 'tutorial'],
  },
  'manual',  // source
  'correlation-id-456'
);

console.log(`Ingested ${ingestResult.chunks_created} chunks`);
console.log(`Document ID: ${ingestResult.document_id}`);
```

### Batch Ingestion

```typescript
// Ingest multiple documents
const documents = [
  { content: 'Document 1 content...', metadata: { source: 'doc1.pdf' } },
  { content: 'Document 2 content...', metadata: { source: 'doc2.pdf' } },
  { content: 'Document 3 content...', metadata: { source: 'doc3.pdf' } },
];

const batchResult = await adapter.batchIngest(documents);
console.log(`Successfully ingested ${batchResult.success_count}/${documents.length} documents`);
```

### Statistics

```typescript
// Get system statistics
const stats = await adapter.getStats();
console.log(`Total documents: ${stats.total_documents}`);
console.log(`Total chunks: ${stats.total_chunks}`);
console.log(`Vector store: ${stats.vector_store_type}`);
console.log(`Embedding model: ${stats.embedding_model}`);
```

### Circuit Breaker

```typescript
// Check circuit state
const state = adapter.getCircuitState();
console.log(`Circuit: ${state.state}`);
console.log(`Failures: ${state.failureCount}`);

// Reset circuit breaker (manual recovery)
await adapter.resetCircuitBreaker('correlation-id-789');
```

## Federation Constitution Compliance

### Law of "Air Gap" (Source Code Isolation)

✅ **Compliant**: No imports from `core-projects/`

All RAGBits utilities are rewritten in the adapter layer. No direct dependencies on RAGBits core code.

### Law of "Runtime Truth" (Anti-Hallucination)

✅ **Compliant**: Probes verify API before use

Run probes before using:
```bash
# Test API endpoints
./probes/check_api.sh

# Test database connectivity
./probes/check_database.sh

# Test retrieval operations
./probes/check_retrieval.sh
```

### Law of "Untouchable DB" (Read-Only State)

✅ **Compliant**: SELECT privileges only

Probes verify read-only access. No write operations except through ingest API.

### Law of Idempotency (The Replayability Pact)

✅ **Compliant**: Safe to retry

All operations are idempotent:
- Search queries return same results
- Ingest operations check for existing documents
- Clear cache is idempotent

### Law of Configuration Explicitness

✅ **Compliant**: No magic defaults

Configuration:
- `RAGBITS_API_URL`: Required, crashes if missing
- `TIMEOUT_MS`: Required, crashes if missing
- All timeouts are explicit
- No `localhost` defaults

### Law of UTC

✅ **Compliant**: All timestamps in UTC

All timestamps use UTC ISO-8601 format (`2025-02-03T12:34:56.789Z`).

## Probes

### check_api.sh

Tests RAGBits API endpoints:
- `GET /health` - Health check
- `POST /search` - Search endpoint
- `POST /ingest` - Ingest endpoint
- `GET /stats` - Statistics endpoint

```bash
export RAGBITS_API_URL=http://ragbits-core:8002
export TIMEOUT_MS=5000
./probes/check_api.sh
```

### check_database.sh

Tests vector database connectivity:
- Connection status
- Query execution (SELECT)
- Statistics retrieval
- Read-only access verification
- Latency measurement

```bash
export RAGBITS_API_URL=http://ragbits-core:8002
export TIMEOUT_MS=5000
./probes/check_database.sh
```

### check_retrieval.sh

Tests retrieval operations:
- Semantic search
- Hybrid search
- Filtered search
- Score threshold filtering
- Idempotency verification
- Performance measurement

```bash
export RAGBITS_API_URL=http://ragbits-core:8002
export TIMEOUT_MS=10000
./probes/check_retrieval.sh
```

## Contract Tests

Validate the adapter contract with RAGBits core:

```bash
cd tests
npm install
npm test
```

Or run contract tests only:
```bash
npm run contract
```

With coverage:
```bash
npm run test:coverage
```

## API Reference

### RAGBitsAdapter

#### Constructor

```typescript
constructor(config: RAGClientConfig)
```

**Parameters**:
- `config.api_url` (string, required): RAGBits server URL
- `config.timeout_ms` (number, required): Request timeout in milliseconds
- `config.api_key` (string, optional): API authentication key

**Throws**:
- `Error`: If `api_url` is missing
- `Error`: If `timeout_ms` is missing or not positive

#### Methods

##### search

```typescript
async search(
  query: string,
  topK?: number,
  filters?: Record<string, any>,
  correlationId?: string
): Promise<any>
```

Search for similar documents.

**Parameters**:
- `query` (string, required): Search query
- `topK` (number, optional): Number of results (default: 5)
- `filters` (object, optional): Metadata filters
- `correlationId` (string, optional): Correlation ID for tracing

**Returns**: Search results with scores and metadata

**Throws**:
- `Error`: If circuit breaker is OPEN
- `Error`: If all retry attempts fail

##### ingest

```typescript
async ingest(
  content: string,
  metadata: Record<string, any>,
  source?: string,
  correlationId?: string
): Promise<any>
```

Ingest a document into the vector store.

**Parameters**:
- `content` (string, required): Document content
- `metadata` (object, required): Document metadata
- `source` (string, optional): Source identifier
- `correlationId` (string, optional): Correlation ID for tracing

**Returns**: Ingest result with document ID and chunk count

**Throws**:
- `Error`: If circuit breaker is OPEN
- `Error`: If all retry attempts fail

##### batchIngest

```typescript
async batchIngest(
  documents: Array<{ content: string; metadata: Record<string, any> }>,
  correlationId?: string
): Promise<any>
```

Ingest multiple documents.

**Parameters**:
- `documents` (array, required): Documents to ingest
- `correlationId` (string, optional): Correlation ID for tracing

**Returns**: Batch ingest result with success count

**Throws**:
- `Error`: If circuit breaker is OPEN
- `Error`: If all retry attempts fail

##### getStats

```typescript
async getStats(correlationId?: string): Promise<any>
```

Get system statistics.

**Returns**: Statistics including document counts, vector store type, etc.

##### clearCache

```typescript
async clearCache(correlationId?: string): Promise<any>
```

Clear the query cache.

**Returns**: Success confirmation

##### testConnection

```typescript
async testConnection(correlationId?: string): Promise<boolean>
```

Test connection to RAGBits server.

**Returns**: `true` if connected, `false` otherwise

##### getCircuitState

```typescript
getCircuitState(): {
  state: CircuitState;
  failureCount: number;
  successCount: number;
}
```

Get current circuit breaker state.

**Returns**: Circuit state information

##### resetCircuitBreaker

```typescript
resetCircuitBreaker(correlationId?: string): void
```

Reset the circuit breaker to CLOSED state.

## Error Handling

### Circuit Breaker States

- **CLOSED**: Normal operation
- **OPEN**: Service unavailable, fail fast
- **HALF_OPEN**: Testing if service recovered

### Retry Strategy

- **Max Attempts**: 3 (configurable)
- **Backoff**: Exponential with jitter
- **Base Delay**: 1000ms
- **Max Delay**: 10000ms

### Error Types

| Error Type | Retryable | Circuit Breaker | Fallback |
|------------|-----------|-----------------|----------|
| Timeout | Yes | Yes | Cached results |
| Connection Refused | Yes | Yes | Empty results |
| HTTP 4xx | No | No | Error message |
| HTTP 5xx | Yes | Yes | Cached results |
| Parse Error | No | No | Error message |

## Performance

### Expected Latency

- **Health Check**: < 100ms
- **Simple Search**: < 500ms
- **Complex Search**: < 1500ms
- **Single Ingest**: < 3000ms
- **Batch Ingest**: < 10000ms

### Optimization Tips

1. **Use Hybrid Search**: Faster than pure semantic for keyword queries
2. **Set Reasonable top_k**: Smaller values are faster
3. **Cache Results**: Use cache for frequent queries
4. **Batch Ingest**: More efficient than single ingest
5. **Warm-up**: Run warm-up query on startup

## Troubleshooting

### Circuit Breaker Open

```typescript
const state = adapter.getCircuitState();
if (state.state === 'OPEN') {
  console.log('Circuit breaker is open');
  console.log(`Failures: ${state.failureCount}`);

  // Wait for timeout (60s default)
  // Or manually reset:
  await adapter.resetCircuitBreaker();
}
```

### Timeout Errors

```typescript
// Increase timeout
adapter.configure({
  client: {
    timeout_ms: 10000,  // 10 seconds
  },
});
```

### Connection Refused

```bash
# Check RAGBits server is running
curl http://ragbits-core:8002/health

# Check environment variables
echo $RAGBITS_API_URL
echo $TIMEOUT_MS
```

### Poor Search Results

```typescript
// Try hybrid search
const results = await adapter.search(
  query,
  10,  // Increase top_k
  {},  // No filters
  correlationId,
  { enable_hybrid_search: true }  // Enable hybrid
);
```

## Contributing

When modifying this adapter:

1. **Run Probes**: Verify API still works
2. **Update Tests**: Add contract tests for new features
3. **Update ADR**: Document architecture decisions
4. **Validate Compliance**: Ensure Federation Constitution compliance

## References

- **Canonical Schema**: `/glue/schemas/ragbits-canonical.ts`
- **ADR**: `ADR.md`
- **Probes**: `./probes/`
- **Tests**: `./tests/`

## License

MIT

---

**Created**: 2026-02-03
**Maintainer**: OpenEvolve Architecture Team
**Status**: Production Ready
