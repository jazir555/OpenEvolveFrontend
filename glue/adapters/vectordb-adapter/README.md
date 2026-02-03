# Vector DB Adapter

Multi-backend vector database adapter for the OpenEvolve Federation.

## Overview

The Vector DB Adapter provides a unified interface to multiple vector database backends:

- **Qdrant**: High-performance, self-hosted vector database (primary)
- **Pinecone**: Managed vector service (cloud production)
- **Chroma**: Lightweight embedded store (local development)
- **pgvector**: PostgreSQL vector extension (existing infrastructure)

Following the OpenEvolve Federation Constitution:
- ✅ Law of Configuration Explicitness (environment variables only)
- ✅ Law of Runtime Truth (validated at runtime)
- ✅ Law of Idempotency (all operations safe to retry)
- ✅ Law of UTC (all timestamps in UTC ISO-8601)
- ✅ JSON Lines logging with correlation_id

## Installation

```bash
npm install @openevolve/vectordb-adapter
```

## Quick Start

### 1. Configure Environment Variables

```bash
# Required: Backend type
export VECTORDB_TYPE=qdrant  # or pinecone, chroma, pgvector

# Backend-specific configuration
export VECTORDB_URL=http://localhost:6333  # Qdrant, Chroma
export VECTORDB_API_KEY=your-api-key      # Pinecone
export VECTORDB_CONNECTION_STRING=postgresql://...  # pgvector

# Optional: Timeouts and retries
export TIMEOUT_MS=5000
export MAX_RETRIES=3
```

### 2. Create Adapter

```typescript
import { createVectorDBAdapter } from '@openevolve/vectordb-adapter';

// Auto-configured from environment
const adapter = createVectorDBAdapter();

// Or with explicit configuration
import { createVectorDBAdapterWithConfig } from '@openevolve/vectordb-adapter';
const adapter = createVectorDBAdapterWithConfig({
  backendType: 'qdrant',
  url: 'http://localhost:6333',
  timeout: 5000,
  maxRetries: 3,
});
```

### 3. Use the Adapter

```typescript
// Health check
const health = await adapter.healthCheck();
console.log(health.status); // 'healthy', 'degraded', 'unhealthy'

// Create collection
await adapter.createCollection({
  name: 'documents',
  dimension: 384,
  distance_metric: 'cosine',
  format: 'dense',
});

// Upsert vectors
await adapter.upsert({
  collection_name: 'documents',
  entries: [
    {
      id: '550e8400-e29b-41d4-a716-446655440000',
      vector: [0.1, 0.2, 0.3, ...],
      text: 'Document text here',
      metadata: { source: 'test' },
      created_at: new Date().toISOString(),
    },
  ],
});

// Search similar vectors
const results = await adapter.search('documents', {
  vector: [0.1, 0.2, 0.3, ...],
  k: 10,
  score_threshold: 0.7,
});

// Delete vectors
await adapter.delete({
  collection_name: 'documents',
  ids: ['550e8400-e29b-41d4-a716-446655440000'],
});
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Vector DB Adapter                         │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Canonical Schema Layer                  │   │
│  │  • VectorEntry (dense/sparse)                        │   │
│  │  • SearchQuery (k, score_threshold, filter)          │   │
│  │  • UpsertRequest, DeleteRequest                      │   │
│  │  • CollectionConfig (dimension, distance_metric)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Adapter Router                          │   │
│  │  • Backend detection (VECTORDB_TYPE)                 │   │
│  │  • Request validation                               │   │
│  │  • Dimension validation                             │   │
│  │  • Circuit breaker per backend                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌──────────────┬──────────────┬──────────────┬───────────┐│
│  │   Qdrant     │   Pinecone   │    Chroma    │  pgvector ││
│  │   Client     │    Client    │    Client    │   Client ││
│  └──────────────┴──────────────┴──────────────┴───────────┘│
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### VectorDBAdapter

#### `healthCheck()`
Check if the vector database backend is accessible.

```typescript
async healthCheck(): Promise<HealthCheckResponse>
```

**Returns**: `HealthCheckResponse` with status, latency, and connection info.

#### `createCollection(config)`
Create a new collection (index/table).

```typescript
async createCollection(config: CollectionConfig): Promise<void>
```

**Parameters**:
- `config.name`: Collection name
- `config.dimension`: Vector dimension
- `config.distance_metric`: 'cosine' | 'euclidean' | 'dot_product'
- `config.format`: 'dense' | 'sparse'

#### `upsert(request)`
Insert or update vectors in a collection.

```typescript
async upsert(request: UpsertRequest): Promise<UpsertResponse>
```

**Parameters**:
- `request.collection_name`: Target collection
- `request.entries`: Array of VectorEntry objects
- `request.namespace`: Namespace (Pinecone only)

**Returns**: `UpsertResponse` with count and timestamp.

#### `search(collectionName, query)`
Search for similar vectors.

```typescript
async search(collectionName: string, query: SearchQuery): Promise<SearchResult[]>
```

**Parameters**:
- `collectionName`: Target collection
- `query.vector`: Query vector (dense or sparse)
- `query.k`: Number of results (1-1000)
- `query.score_threshold`: Minimum similarity score (optional)
- `query.filter`: Metadata filter (optional)

**Returns**: Array of `SearchResult` with entry, score, and distance.

#### `delete(request)`
Delete vectors from a collection.

```typescript
async delete(request: DeleteRequest): Promise<DeleteResponse>
```

**Parameters**:
- `request.collection_name`: Target collection
- `request.ids`: Array of vector IDs to delete
- `request.delete_all`: Delete all vectors (dangerous!)

**Returns**: `DeleteResponse` with count and timestamp.

#### `getCollectionInfo(collectionName)`
Get information about a collection.

```typescript
async getCollectionInfo(collectionName: string): Promise<CollectionInfo>
```

**Returns**: `CollectionInfo` with name, dimension, count, and timestamps.

#### `listCollections()`
List all collections in the backend.

```typescript
async listCollections(): Promise<string[]>
```

**Returns**: Array of collection names.

## Canonical Schemas

### VectorEntry
A single vector with metadata.

```typescript
interface VectorEntry {
  id: string; // UUID
  vector: number[] | {  // Dense or sparse
    indices: number[];
    values: number[];
    dimension: number;
  };
  text?: string;
  metadata?: Record<string, unknown>;
  vector_name?: string;
  created_at?: string; // UTC ISO-8601
}
```

### SearchQuery
Vector similarity search query.

```typescript
interface SearchQuery {
  vector: number[] | SparseVector;
  k: number; // 1-1000
  score_threshold?: number; // 0-1
  filter?: Record<string, unknown>;
  vector_name?: string;
}
```

### SearchResult
Search result with similarity score.

```typescript
interface SearchResult {
  entry: VectorEntry;
  score: number; // 0-1
  distance?: number;
}
```

## Backend-Specific Notes

### Qdrant
- **Distance Metrics**: Cosine, Euclidean, Dot
- **Sparse Vectors**: ✅ Supported
- **Named Vectors**: ✅ Supported (multi-vector collections)
- **Filtering**: Payload filtering with match/any/range
- **Configuration**:
  ```bash
  export VECTORDB_TYPE=qdrant
  export VECTORDB_URL=http://localhost:6333
  export VECTORDB_API_KEY=optional-api-key
  ```

### Pinecone
- **Distance Metrics**: Cosine, Euclidean, Dotproduct
- **Sparse Vectors**: ❌ Not supported (rejected by adapter)
- **Named Vectors**: ❌ Not supported
- **Namespaces**: ✅ Supported (multi-tenancy)
- **Configuration**:
  ```bash
  export VECTORDB_TYPE=pinecone
  export VECTORDB_API_KEY=your-api-key
  export PINECONE_ENVIRONMENT=us-east1-aws
  ```

### Chroma
- **Distance Metrics**: Cosine, L2 (Euclidean), IP (Dot)
- **Sparse Vectors**: ❌ Not supported
- **Named Vectors**: ❌ Not supported
- **Where Filters**: ✅ Supported (metadata filtering)
- **Configuration**:
  ```bash
  export VECTORDB_TYPE=chroma
  export VECTORDB_URL=http://localhost:8000
  ```

### pgvector
- **Distance Metrics**: Cosine (<=>), L2 (<->), IP (<#>)
- **Sparse Vectors**: ✅ Supported (sparsevec type)
- **Named Vectors**: ❌ Not supported
- **Filtering**: ✅ Supported (metadata JSONB)
- **Configuration**:
  ```bash
  export VECTORDB_TYPE=pgvector
  export VECTORDB_CONNECTION_STRING=postgresql://user:pass@host:5432/db
  ```

## Probes (Runtime Validation)

The adapter includes probe scripts to validate backend APIs at runtime:

### check_api.sh
Test basic API connectivity and health.

```bash
./probes/check_api.sh
```

**Tests**:
1. Health check endpoint
2. Response time (< TIMEOUT_MS)
3. Response structure validation

### check_collections.sh
Test collection operations.

```bash
./probes/check_collections.sh [collection_name]
```

**Tests**:
1. List collections
2. Create collection
3. Get collection info
4. Delete collection

### check_search.sh
Test vector similarity search.

```bash
./probes/check_search.sh [collection_name]
```

**Tests**:
1. Create test collection
2. Insert test vectors
3. Perform similarity search
4. Clean up test collection

## Contract Tests

Validate backend API contracts:

```bash
npm run test:contract
```

The contract tests verify:
- Canonical schema conformance
- Vector dimension validation
- Distance metric mapping
- Idempotency of operations
- Edge case handling

## Error Handling

The adapter implements:

1. **Circuit Breakers**: Per-backend circuit breakers (5 failures → open for 60s)
2. **Retry Logic**: Exponential backoff with jitter (max 3 retries)
3. **Validation**: Schema validation with Zod (fail fast)
4. **Structured Logging**: JSON Lines with correlation_id

### Example Error Response

```json
{
  "level": "error",
  "msg": "Vector dimension validation failed",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_service": "vectordb-adapter",
  "entry_id": "550e8400-e29b-41d4-a716-446655440000",
  "expected_dimension": 384,
  "error_name": "ValidationError",
  "error_message": "Vector dimension 128 does not match expected dimension 384"
}
```

## Best Practices

### 1. Use Correlation IDs
```typescript
import { createCorrelationId } from '@openevolve/lib';

const correlationId = createCorrelationId();
// All logs will include this correlation_id
```

### 2. Handle Idempotency
```typescript
// Upsert is idempotent (safe to retry)
await adapter.upsert({
  collection_name: 'docs',
  entries: [vectorEntry],
});

// Delete by ID is idempotent
await adapter.delete({
  collection_name: 'docs',
  ids: [vectorId],
});
```

### 3. Validate Dimensions
```typescript
// Adapter validates dimensions before upsert
// Mismatched dimensions will throw ValidationError
try {
  await adapter.upsert(request);
} catch (error) {
  if (error.message.includes('dimension')) {
    console.error('Vector dimension mismatch');
  }
}
```

### 4. Use Score Thresholds
```typescript
// Filter out low-quality results
const results = await adapter.search('docs', {
  vector: queryVector,
  k: 100, // Get more results
  score_threshold: 0.8, // Filter to high-quality
});
```

## Troubleshooting

### Backend Not Accessible
```bash
# Test connectivity
./probes/check_api.sh

# Check environment variables
echo $VECTORDB_TYPE
echo $VECTORDB_URL
```

### Dimension Mismatch
```bash
# Check collection dimension
curl http://localhost:6333/collections/my-collection
```

### Circuit Breaker Open
Wait 60 seconds for automatic reset, or restart the adapter.

## Contributing

See [ADR.md](ADR.md) for architecture decisions and design rationale.

## License

MIT

## Authors

OpenEvolve Federation
