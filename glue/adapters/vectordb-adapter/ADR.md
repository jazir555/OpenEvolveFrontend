# Architecture Decision Record (ADR): Vector DB Adapter

## Status
Accepted

## Date
2024-01-15

## Context
The OpenEvolve Federation requires a unified interface to multiple vector database backends for semantic search, RAG (Retrieval-Augmented Generation), and knowledge graph operations. Different projects within the federation use different vector databases:

- **Qdrant**: Used in RAGBits and BubbleLab for high-performance vector search
- **Pinecone**: Managed vector service used in some production deployments
- **Chroma**: Lightweight vector store used in local development and testing
- **pgvector**: PostgreSQL extension used in existing data infrastructure

Each backend has:
- Different API conventions (REST, gRPC, SQL)
- Different data models (named vectors, namespaces, collections)
- Different distance metrics and similarity functions
- Different performance characteristics

Following the **Law of the "Air Gap" (Source Code Isolation)** from CLAUDE.md, we cannot directly import or depend on the core projects' vector database code. We need an **Anti-Corruption Layer (ACL)** that normalizes these differences.

## Decision
Create a multi-backend Vector DB Adapter with:

1. **Canonical Schema**: Unified data model for all vector operations
2. **Backend Detection**: Runtime backend selection via `VECTORDB_TYPE` environment variable
3. **Transformation Layer**: Bidirectional mapping between canonical and backend-specific formats
4. **Circuit Breakers**: Per-backend circuit breakers for fault tolerance
5. **Retry Logic**: Exponential backoff with jitter for transient failures
6. **Contract Tests**: Validate backend APIs on adapter startup

### Architecture

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
                          ↓
        ┌─────────────────┴─────────────────┐
        │     Backend APIs (REST/SQL)       │
        └────────────────────────────────────┘
```

## Backend Selection Strategy

### Primary Backend: Qdrant
**Rationale**:
- Open source with permissive license (Apache 2.0)
- Self-hosted (no vendor lock-in)
- Excellent performance with HNSW indexing
- Supports both dense and sparse vectors
- Native support for named vectors (multi-vector collections)
- Active development and community

### Fallback Strategy

| Scenario | Backend | Reason |
|----------|---------|--------|
| Local development | Chroma | Lightweight, embedded, easy to set up |
| Cloud production | Pinecone | Fully managed, auto-scaling, low ops overhead |
| Existing PostgreSQL | pgvector | Leverage existing database infrastructure |
| High-performance | Qdrant | Best performance/cost ratio for self-hosted |

**Configuration**:
```bash
# Primary (Qdrant)
export VECTORDB_TYPE=qdrant
export VECTORDB_URL=http://qdrant:6333

# Fallback to Chroma (local dev)
export VECTORDB_TYPE=chroma
export VECTORDB_URL=http://localhost:8000

# Fallback to Pinecone (cloud)
export VECTORDB_TYPE=pinecone
export VECTORDB_API_KEY=your-api-key
export PINECONE_ENVIRONMENT=us-east1-aws

# Fallback to pgvector (existing DB)
export VECTORDB_TYPE=pgvector
export VECTORDB_CONNECTION_STRING=postgresql://...
```

## Canonical Schema Design

### Vector Entry
Supports both dense and sparse vectors:

```typescript
interface VectorEntry {
  id: string; // UUID
  vector: number[] | { indices: number[]; values: number[]; dimension: number };
  text?: string;
  metadata?: Record<string, unknown>;
  vector_name?: string; // For multi-vector collections
  created_at?: string; // UTC ISO-8601
}
```

**Rationale**:
- **Dense vectors**: Standard arrays (compatible with all backends)
- **Sparse vectors**: Object with indices/values (Qdrant, pgvector)
- **vector_name**: Enables multi-vector collections (e.g., dense + sparse)
- **text field**: RAG workflows need both embeddings and source text
- **metadata**: Flexible payload for filtering

### Search Query
Unified search interface:

```typescript
interface SearchQuery {
  vector: number[] | SparseVector;
  k: number; // Number of results (1-1000)
  score_threshold?: number; // Minimum similarity (0-1)
  filter?: Record<string, unknown>; // Metadata filter
  vector_name?: string; // For multi-vector search
}
```

**Rationale**:
- **k**: Standard top-k search (all backends)
- **score_threshold**: Client-side filtering (backend support varies)
- **filter**: Metadata filtering (syntax varies by backend, adapter normalizes)
- **vector_name**: Multi-vector search support

## Consequences

### Positive
1. **Backend Flexibility**: Switch backends via environment variable (no code changes)
2. **Unified API**: Single interface for all vector operations
3. **Type Safety**: Canonical schemas validated with Zod
4. **Fault Tolerance**: Circuit breakers prevent cascade failures
5. **Testability**: Contract tests validate backend compatibility
6. **Zero Vendor Lock-in**: Easy migration between backends

### Negative
1. **Complexity**: Additional abstraction layer adds maintenance overhead
2. **Performance**: Transformation layer adds ~1-5ms latency
3. **Feature Parity**: Lowest common denominator (can't use backend-specific features)
4. **Testing**: Need to test all backends (maintenance burden)

### Mitigation
1. **Complexity**: Well-documented patterns, shared library (circuit breaker, retry)
2. **Performance**: Minimal overhead (transformations are simple mappings)
3. **Feature Parity**: Extension points for backend-specific features
4. **Testing**: Contract tests run on adapter startup (fail fast)

## Implementation Details

### Distance Metric Mapping

| Canonical | Qdrant | Pinecone | Chroma | pgvector |
|-----------|--------|----------|--------|----------|
| cosine | Cosine | cosine | cosine | <=> |
| euclidean | Euclid | euclidean | l2 | <-> |
| dot_product | Dot | dotproduct | ip | <#> |

### Sparse Vector Support

| Backend | Support | Implementation |
|---------|---------|----------------|
| Qdrant | ✅ Native | SparseVector structure |
| Pinecone | ❌ | Dense only (rejected by client) |
| Chroma | ❌ | Dense only (rejected by client) |
| pgvector | ✅ Native | sparsevec type |

### Idempotency (Law of Idempotency)

All operations are idempotent:
- **Upsert**: Insert or update (based on UUID)
- **Delete**: Delete by ID (safe to retry)
- **Create Collection**: Create if not exists (409 OK)

## Migration Path

### From Qdrant-Only to Multi-Backend

1. **Phase 1**: Refactor existing code to use canonical schema
2. **Phase 2**: Add transformation layer for Qdrant
3. **Phase 3**: Add backend detection and client router
4. **Phase 4**: Add additional backends (Pinecone, Chroma, pgvector)
5. **Phase 5**: Deploy with backend selection via environment

### Rollback Strategy

If a backend fails:
1. Circuit breaker opens (stop sending requests)
2. Log error with correlation_id
3. Return error to caller
4. Caller can retry with fallback backend

## Alternatives Considered

### Alternative 1: LangChain Vector Stores
**Rejected**: Too high-level, doesn't follow Federation Constitution (Law of Air Gap)

### Alternative 2: LlamaIndex Vector Stores
**Rejected**: Same as LangChain, plus tight coupling to LlamaIndex ecosystem

### Alternative 3: Direct Backend Clients
**Rejected**: Would violate Law of Configuration Explicitness (no unified interface)

### Alternative 4: Separate Adapters per Backend
**Rejected**: Duplicate code, harder to maintain, no backend flexibility

## References

- [CLAUDE.md - Federation Constitution](../../../../CLAUDE.md)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [Canonical Schema: vectordb-canonical.ts](../../../schemas/vectordb-canonical.ts)

## Authors
OpenEvolve Federation - Distinguished Engineer

## Related ADRs
- [ADR: Event Bus Orchestration](../../orchestration/ADR.md)
- [ADR: Anti-Corruption Layer Pattern](../README.md)
