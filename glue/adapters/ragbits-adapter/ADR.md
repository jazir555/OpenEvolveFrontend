# Architecture Decision Record: RAGBits Integration

## Status
**Accepted**

## Context

RAGBits is a Retrieval-Augmented Generation (RAG) system that provides semantic document search and retrieval capabilities for the OpenEvolve platform. It enables:

- **Semantic Search**: Vector-based similarity search using embeddings
- **Hybrid Search**: Combination of semantic and keyword search
- **Document Ingestion**: Chunking, embedding, and indexing of documents
- **Metadata Filtering**: Search with custom metadata filters
- **Retrieval at Scale**: Efficient retrieval from large document corpora

RAGBits is used throughout OpenEvolve for:
- Knowledge retrieval for AI agents
- Document context for LLM prompts
- Semantic search across codebases
- Historical pattern retrieval
- Cross-reference discovery

The integration must support:
1. **REST API Interface**: HTTP-based communication with RAGBits server
2. **Circuit Breaker**: Fail-fast when service is down
3. **Exponential Backoff**: Retry with jitter for transient failures
4. **Canonical Schema**: Anti-corruption layer for data normalization
5. **Idempotency**: Safe to retry all operations

## Decision

### Architecture Pattern: Sidecar Adapter with Circuit Breaker

We chose a **Sidecar Adapter Pattern** with the following characteristics:

```
[Core OpenEvolve] --> [RAGBits Adapter (Canonical Layer)] --> [RAGBits Server]
                                                                      |
                                                                      +-- [Vector DB (ChromaDB)]
                                                                      +-- [Embedding Model]
```

### Key Design Choices

1. **Adapter Location**: `/glue/adapters/ragbits-adapter/`
   - Isolated from core-projects (Law of Air Gap)
   - Rewritten RAGBits utilities in adapter layer (no imports from core)
   - Canonical schema at `/glue/schemas/ragbits-canonical.ts`

2. **Interface Strategy**: REST API over HTTP
   - Primary: HTTP/JSON communication
   - Endpoints: `/health`, `/search`, `/ingest`, `/stats`
   - Timeout enforcement on all requests

3. **Data Flow**:
   ```
   Input (Canonical Format)
       --> RAGBitsAdapter.search()
       --> RAGClient.fetch()
       --> RAGBits Server API
       --> Response (Canonical Format)
       --> Output (Canonical Format)
   ```

4. **Failure Management**:
   - **Transient Failure**: Exponential backoff retry (3 attempts)
   - **System Failure**: Circuit breaker (open after 5 failures)
   - **Logic Failure**: Return error (no retry)

## Consequences

### Positive Benefits

1. **Performance**: Sub-second retrieval for typical queries
2. **Reliability**: Circuit breaker prevents cascading failures
3. **Flexibility**: Supports multiple search modes (semantic, hybrid, keyword)
4. **Integration**: Direct bridge to canonical knowledge schemas
5. **Extensibility**: Easy to add new search modes or filters
6. **Isolation**: Core system never directly depends on RAGBits

### Negative Tradeoffs

1. **Duplication**: RAGBits utilities rewritten in adapter layer (law of air gap)
2. **Overhead**: Adapter layer adds ~10-20ms per call
3. **Complexity**: Circuit breaker and retry logic increases maintenance
4. **Memory**: Embedding operations can consume significant memory
5. **State**: No persistent state between calls (must be stateless)

### Known Limitations

1. **Embedding Size**: Large documents (>10MB) may cause timeouts
2. **Concurrent Requests**: Limited by RAGBits server capacity
3. **Vector Store**: Depends on ChromaDB availability
4. **Search Relevance**: Quality depends on embedding model
5. **Index Build Time**: Initial indexing can be slow for large corpora

## Implementation Details

### Core Components

#### 1. RAGClient
```typescript
class RAGClient {
  search(request: RAGSearchRequest): Promise<any>
  ingest(request: RAGIngestRequest): Promise<any>
  batchIngest(requests: RAGIngestRequest[]): Promise<any>
  getStats(): Promise<any>
  clearCache(): Promise<any>
  testConnection(): Promise<boolean>
}
```

**Capabilities**:
- HTTP communication with RAGBits server
- Timeout enforcement (MANDATORY)
- Structured logging (JSON Lines)
- Correlation ID propagation

**Example**:
```typescript
const client = new RAGClient({
  api_url: 'http://ragbits-core:8002',
  timeout_ms: 5000,
});

const results = await client.search({
  query: 'machine learning algorithms',
  top_k: 5,
  enable_hybrid_search: true,
});
```

#### 2. RAGBitsAdapter
```typescript
class RAGBitsAdapter {
  search(query: string, topK: number, filters?): Promise<any>
  ingest(content: string, metadata: {}): Promise<any>
  batchIngest(documents: Array<{content, metadata}>): Promise<any>
  getStats(): Promise<any>
  clearCache(): Promise<any>
  testConnection(): Promise<boolean>
  getCircuitState(): CircuitState
  resetCircuitBreaker(): void
}
```

**Capabilities**:
- Circuit breaker (open after 5 failures)
- Exponential backoff retry (3 attempts)
- Structured logging with correlation IDs
- Configuration validation

**Example**:
```typescript
const adapter = new RAGBitsAdapter({
  api_url: 'http://ragbits-core:8002',
  timeout_ms: 5000,
});

const results = await adapter.search(
  'How to implement authentication?',
  5,
  { document_type: 'markdown' }
);
```

### API Endpoints

| Endpoint | Purpose | Timeout | Retry Strategy |
|----------|---------|---------|----------------|
| `GET /health` | Health check | 5s | No retry (fast) |
| `POST /search` | Semantic search | 10s | 3 attempts, exponential backoff |
| `POST /ingest` | Ingest document | 30s | 3 attempts, exponential backoff |
| `POST /ingest/batch` | Batch ingest | 60s | 2 attempts, linear backoff |
| `GET /stats` | System statistics | 5s | No retry (fast) |
| `POST /clear-cache` | Clear cache | 5s | No retry (idempotent) |

### Data Flow Diagrams

#### Search Flow
```
[Client]
  --> {query, top_k, filters}
[RAGBits Adapter]
  --> Normalize to Canonical Schema
  --> Check Circuit Breaker
  --> Execute with Retry
[RAGClient]
  --> HTTP POST /search
  --> Timeout enforcement
  --> Structured logging
[RAGBits Server]
  --> Vector search
  --> Rank results
  --> Return matches
[RAGBits Adapter]
  --> Transform to Canonical Format
  --> Update circuit breaker
  --> Log metrics
[Client]
  <-- {results, total_results, execution_time_ms}
```

#### Ingest Flow
```
[Client]
  --> {content, metadata, source}
[RAGBits Adapter]
  --> Normalize to Canonical Schema
  --> Validate input
  --> Check Circuit Breaker
[RAGClient]
  --> HTTP POST /ingest
  --> Timeout enforcement
[RAGBits Server]
  --> Chunk document
  --> Generate embeddings
  --> Store in vector DB
  --> Return document_id
[RAGBits Adapter]
  --> Transform to Canonical Format
  --> Log success
[Client]
  <-- {success, document_id, chunks_created}
```

### Configuration Requirements

#### Environment Variables
```bash
# RAGBits Configuration
RAGBITS_API_URL=http://ragbits-core:8002  # RAGBits server URL
TIMEOUT_MS=5000                            # Default timeout (ms)

# Circuit Breaker Configuration
CIRCUIT_FAILURE_THRESHOLD=5                # Open after N failures
CIRCUIT_SUCCESS_THRESHOLD=2                # Close after N successes
CIRCUIT_TIMEOUT_MS=60000                   # Open state duration (ms)

# Retry Configuration
RETRY_MAX_ATTEMPTS=3                       # Max retry attempts
RETRY_BASE_DELAY_MS=1000                   # Base delay (ms)
RETRY_MAX_DELAY_MS=10000                   # Max delay (ms)
RETRY_EXPONENTIAL=2.0                      # Backoff multiplier
RETRY_JITTER=0.1                           # Random jitter (0-1)
```

#### TypeScript Configuration
```typescript
const config: RAGClientConfig = {
  api_url: process.env.RAGBITS_API_URL,  // Required, no default
  timeout_ms: parseInt(process.env.TIMEOUT_MS || '5000'),
  api_key: process.env.RAGBITS_API_KEY,  // Optional
};
```

## Gotchas

### API Quirks Discovered

1. **Timeout Unreliability**:
   - RAGBits server may not respect client timeout
   - **Solution**: Use AbortController with process-level timeout

2. **Empty Results**:
   - Search returns empty array when no matches found
   - **Gotcha**: Not an error, just no results
   - **Solution**: Check `total_results` field

3. **Batch Ingest Limits**:
   - Large batches (>100 docs) may timeout
   - **Solution**: Split into smaller batches

4. **Metadata Filtering**:
   - Filters must match exact metadata structure
   - **Gotcha**: Case-sensitive matching
   - **Solution**: Normalize metadata before ingest

5. **Embedding Model Loading**:
   - First search may be slow (model loading)
   - **Solution**: Warm-up query on startup

6. **Vector Store Locking**:
   - ChromaDB may lock during ingest
   - **Solution**: Retry with backoff

### Version Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|---------------------|-------|
| RAGBits Server | 1.0.0 | 2.0.0+ | Latest stable |
| Node.js | 18.0 | 20.0+ | ESM support |
| TypeScript | 5.0 | 5.3+ | ESM modules |

### Non-Obvious Behaviors

1. **Caching**:
   - RAGBits caches query results
   - **Gotcha**: Stale data after recent ingest
   - **Solution**: Call `/clear-cache` after bulk ingest

2. **Chunk Overlap**:
   - Default overlap may cause duplicate results
   - **Gotcha**: Same content appears multiple times
   - **Solution**: Deduplicate by document_id

3. **Score Normalization**:
   - Scores not normalized across searches
   - **Gotcha**: Scores vary by query complexity
   - **Solution**: Use relative ranking, not absolute scores

4. **Concurrent Ingest**:
   - Multiple ingest operations may conflict
   - **Gotcha**: Race conditions in vector DB
   - **Solution**: Queue ingest operations

5. **Memory Growth**:
   - Embedding cache grows unbounded
   - **Solution**: Periodic cache clearing

## Circuit Breaker Configuration

### Timeout Values
```typescript
TIMEOUTS = {
  search: 10000,        // 10 seconds
  ingest: 30000,        // 30 seconds
  batch_ingest: 60000,  // 60 seconds
  stats: 5000,          // 5 seconds
  health: 5000,         // 5 seconds
  clear_cache: 5000,    // 5 seconds
}
```

### Retry Strategies

#### Exponential Backoff (Default)
```typescript
{
  max_attempts: 3,
  base_delay_ms: 1000,
  max_delay_ms: 10000,
  exponential: 2.0,
  jitter: 0.1,
}
```
**Usage**: Search, ingest

#### Linear Backoff (Long Operations)
```typescript
{
  max_attempts: 2,
  base_delay_ms: 2000,
  max_delay_ms: 5000,
  exponential: 1.0,
}
```
**Usage**: Batch ingest

#### No Retry (Fast/Stateful)
```typescript
// No retry decorator
```
**Usage**: Health, stats, clear-cache

### Failure Thresholds

```typescript
CIRCUIT_BREAKER = {
  failure_threshold: 5,        // open after 5 failures
  success_threshold: 2,        // close after 2 successes
  timeout: 60000,              // open state duration (ms)
  half_open_max_calls: 1,      // test call in half-open state
}
```

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit tripped, requests fail immediately
- **HALF_OPEN**: Test if service recovered, allow 1 call

**Triggers**:
- 5 consecutive failures (timeout or exception)
- 3 consecutive timeouts (>10s)
- HTTP 5xx errors

**Recovery**:
- 2 consecutive successes → CLOSE
- 60s timeout → HALF_OPEN
- Manual reset via `resetCircuitBreaker()`

### Error Classification

| Error Type | Retryable | Circuit Breaker | Fallback |
|------------|-----------|-----------------|----------|
| `ETIMEDOUT` | Yes | Yes | Return cached results |
| `ECONNREFUSED` | Yes | Yes | Return empty results |
| HTTP 4xx | No | No | Return error |
| HTTP 5xx | Yes | Yes | Return cached results |
| Parse Error | No | No | Return error |

## Security Considerations

### Input Validation

#### Query Sanitization
```typescript
function validateQuery(query: string): boolean {
  // Max length
  if (query.length > 10000) {
    throw new Error('Query too long');
  }

  // Block shell commands
  const shellIndicators = ['; rm', '| rm', '$(', '`'];
  if (shellIndicators.some(indicator => query.includes(indicator))) {
    throw new Error('Shell commands not allowed');
  }

  return true;
}
```

#### Content Sanitization
```typescript
function validateContent(content: string): boolean {
  // Max size
  if (content.length > 1000000) {  // 1MB
    throw new Error('Content too large');
  }

  // Check for malicious patterns
  const maliciousPatterns = [
    '<script>',
    'javascript:',
    'data:text/html',
  ];

  if (maliciousPatterns.some(pattern =>
    content.toLowerCase().includes(pattern)
  )) {
    throw new Error('Malicious content detected');
  }

  return true;
}
```

### Authentication Requirements

**RAGBits has optional authentication**:

```typescript
const config: RAGClientConfig = {
  api_url: process.env.RAGBITS_API_URL,
  timeout_ms: 5000,
  api_key: process.env.RAGBITS_API_KEY,  // Optional Bearer token
};
```

### Data Sensitivity

#### Sensitive Data in Documents
```typescript
// WARNING: Documents may contain sensitive information
// Example: "The API key is sk-1234567890abcdef"

// Best practice: Hash sensitive values before logging
function sanitizeForLogging(content: string): string {
  // Redact potential secrets
  return content
    .replace(/sk-[a-zA-Z0-9]{20,}/g, 'sk-REDACTED')
    .replace(/password["\s:=]+[^\s"]+/gi, 'password=REDACTED')
    .replace(/api[_-]?key["\s:=]+[^\s"]+/gi, 'api_key=REDACTED');
}
```

#### Logging Security
```typescript
// NEVER log document contents
logger.info('Ingesting document');  // OK
logger.info(`Content: ${content}`); // BAD - leaks data

// Log only metadata
logger.info({
  msg: 'Ingesting document',
  content_length: content.length,
  document_type: metadata.type,
  correlation_id: ctx.id,
});
```

### Resource Limits

```typescript
// Prevent resource exhaustion
MAX_CONTENT_LENGTH = 1000000;  // 1MB
MAX_BATCH_SIZE = 100;          // documents
MAX_QUERY_LENGTH = 10000;      // characters
MAX_EXECUTION_TIME = 300000;   // 5 minutes

function enforceLimits(content: string, batchSize: number): void {
  if (content.length > MAX_CONTENT_LENGTH) {
    throw new Error(`Content too large: ${content.length} > ${MAX_CONTENT_LENGTH}`);
  }

  if (batchSize > MAX_BATCH_SIZE) {
    throw new Error(`Batch too large: ${batchSize} > ${MAX_BATCH_SIZE}`);
  }
}
```

## References

- **RAGBits GitHub**: (Internal repository)
- **RAGBits Documentation**: (Internal docs)
- **Canonical Schema**: `/glue/schemas/ragbits-canonical.ts`
- **Adapter Source**: `/glue/adapters/ragbits-adapter/src/`
- **Probes**: `/glue/adapters/ragbits-adapter/probes/`
- **Contract Tests**: `/glue/adapters/ragbits-adapter/tests/`

**Created**: 2026-02-03
**Author**: OpenEvolve Architecture Team
**Status**: Accepted, Implemented
**Last Updated**: 2026-02-03
