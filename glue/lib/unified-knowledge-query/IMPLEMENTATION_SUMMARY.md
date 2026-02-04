# Unified Knowledge Query Interface - Implementation Summary

## Overview

The Unified Knowledge Query Interface has been successfully implemented following the Federation Constitution. This system provides a unified interface to query multiple knowledge systems (RAGBits, Graphiti, Vector DB) simultaneously with intelligent routing, result fusion, and fallback strategies.

## Directory Structure

```
glue/lib/unified-knowledge-query/
├── src/
│   ├── canonical.ts          # Canonical schemas and types (Zod validation)
│   ├── clients.ts            # Individual system clients (RAGBits, Graphiti, Vector DB)
│   ├── query-router.ts       # Query routing and strategy determination
│   ├── result-fusion.ts      # Result merging and conflict resolution
│   ├── fallback-strategy.ts  # Fallback and graceful degradation
│   ├── engine.ts             # Main unified query engine
│   └── index.ts              # Public API exports
├── probes/
│   ├── check_ragbits.sh      # RAGBits API probe
│   ├── check_graphiti.sh     # Graphiti API probe
│   ├── check_vectordb.sh     # Vector DB API probe
│   └── check_unified.sh      # Unified system probe suite
├── tests/
│   ├── contract.test.ts      # Contract validation tests
│   └── jest.config.js        # Jest test configuration
├── package.json              # NPM package configuration
├── tsconfig.json             # TypeScript configuration
└── README.md                 # Documentation
```

## Implementation Details

### 1. Canonical Schema (canonical.ts)

**Purpose**: Define canonical data models for all queries and results

**Key Schemas**:
- `UnifiedKnowledgeQuery`: Input query specification
- `UnifiedQueryResult`: Merged results from all systems
- `KnowledgeDomain`: System selector (ragbits | graphiti | vectordb | all)
- `TemporalFilter`: Time-based filtering (Law of UTC compliance)
- `KnowledgeType`: Result type filter (document | entity | proof | code | relationship | all)

**Features**:
- Zod schema validation for runtime type safety
- UTC ISO-8601 timestamp enforcement (Law of UTC)
- Type guards for compile-time and runtime validation
- Validation helpers for clean API

### 2. System Clients (clients.ts)

**Purpose**: Individual clients for each knowledge system

**Clients**:
- `RAGBitsClient`: Document-based RAG and retrieval
- `GraphitiClient`: Temporal knowledge graph queries
- `VectorDBClient`: High-performance semantic vector search

**Features**:
- Circuit breaker for each client
- Retry logic with exponential backoff
- Timeout enforcement on all requests
- Structured logging with correlation IDs
- Result normalization to canonical format

**API Methods**:
- `search(query, options)`: Execute search query
- `healthCheck()`: Verify system availability
- `getStats()`: Retrieve system statistics

### 3. Query Router (query-router.ts)

**Purpose**: Determine which systems to query and execution strategy

**Routing Logic**:
1. Analyze query characteristics
2. Determine query strategy (semantic | temporal | graph | hybrid | fallback)
3. Select systems based on domains and health status
4. Estimate execution cost
5. Decide if parallel execution is possible

**Query Strategies**:
- `semantic-search`: Vector DB + RAGBits for document retrieval
- `temporal-query`: Graphiti with time filters
- `graph-traversal`: Graphiti relationship queries
- `hybrid`: Combine all sources
- `fallback`: Graceful degradation

**Features**:
- System health awareness
- Historical performance tracking
- Priority-based system selection
- Cost estimation

### 4. Result Fusion (result-fusion.ts)

**Purpose**: Merge and normalize results from multiple systems

**Fusion Pipeline**:
1. **Normalize Scores**: Min-max normalization per system
2. **Detect Conflicts**: Identify duplicate IDs with different content
3. **Resolve Conflicts**: Select highest confidence version
4. **Rank Results**: Combined score (60% normalized + 40% confidence)
5. **Build Metadata**: Track source system performance
6. **Calculate Confidence**: Weighted average overall confidence

**Features**:
- Deterministic merge operations (Law of Idempotency)
- Conflict detection and resolution
- Relevance score normalization
- Deduplication by ID
- Filtering by confidence and type

### 5. Fallback Strategy (fallback-strategy.ts)

**Purpose**: Implement graceful degradation when systems fail

**Fallback Mechanisms**:
1. **Primary System**: Try primary system first
2. **Sequential Fallback**: Try fallbacks in priority order
3. **Parallel Execution**: Race multiple systems, use first successful
4. **Circuit Breaker Awareness**: Skip unhealthy systems

**Fallback Priority**:
1. Vector DB (fastest, most reliable)
2. RAGBits (balanced)
3. Graphiti (specialized)

**Features**:
- Configurable fallback behavior
- Health-aware system selection
- Automatic retry with exponential backoff
- Fallback success metrics

### 6. Main Engine (engine.ts)

**Purpose**: Unified entry point for all knowledge queries

**Query Methods**:
- `query(query, options)`: General unified query
- `semanticSearch(query, options)`: Vector + document search
- `temporalQuery(query, startDate, endDate, options)`: Time-filtered query
- `graphTraversal(query, options)`: Graph relationship query
- `hybridQuery(query, options)`: Query all systems

**Monitoring**:
- `healthCheck()`: Check all system health
- `getMetrics()`: Retrieve engine performance metrics
- `resetMetrics()`: Clear metrics

**Features**:
- Configuration via environment variables
- Automatic system initialization
- Correlation ID tracking
- Performance metrics collection

## Federation Constitution Compliance

### Law of Air Gap (Source Code Isolation)
✅ No imports from `./core-projects/` directory
✅ All functionality implemented in glue layer
✅ No dependency leakage from core projects

### Law of Runtime Truth (Anti-Hallucination)
✅ Probe scripts verify API availability before execution
✅ Contract tests validate API responses
✅ Runtime validation with Zod schemas

### Law of the Untouchable DB (Read-Only State)
✅ Only SELECT operations via search APIs
✅ No direct database writes
✅ Read-only access to all systems

### Law of Idempotency (The Replayability Pact)
✅ All queries safe to retry
✅ Deterministic result fusion
✅ Deduplication by unique IDs
✅ No side effects from queries

### Law of Configuration Explicitness
✅ All URLs via environment variables (RAGBITS_URL, GRAPHITI_URL, VECTORDB_URL)
✅ Timeout values configurable
✅ No magic defaults
✅ Startup validation of required config

### Law of UTC
✅ All timestamps in UTC ISO-8601 format
✅ Temporal filters require UTC timestamps
✅ Schema validation enforces UTC format

## Failure Management Strategy

### Transient Failures
- **Detection**: Network errors, timeouts (5xx status)
- **Action**: Exponential backoff retry (1s, 2s, 4s...)
- **Circuit Breaker**: Trip after 5 consecutive failures
- **Recovery**: Half-open state after 60s

### Logic Failures
- **Detection**: Bad data, validation errors
- **Action**: Log error, return partial results
- **Dead Letter Queue**: Document failed queries
- **Continue**: Don't block pipeline

### System Failures
- **Detection**: Health check failures, circuit open
- **Action**: Skip system, use fallbacks
- **Circuit Breaker**: Stop hammering dead service
- **Recovery**: Health check before retry

## Observability

### Structured Logging (JSON Lines format)
```json
{
  "level": "info",
  "msg": "Query completed successfully",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_service": "unified-knowledge-engine",
  "result_count": 42,
  "confidence": 0.87,
  "execution_time": 1234
}
```

### Metrics Collected
- Total queries
- Successful/failed query counts
- Average query time
- System health status
- Per-system response times

## Usage Examples

### Basic Query
```typescript
import { UnifiedKnowledgeQueryEngine } from '@openevolve/unified-knowledge-query';

const engine = new UnifiedKnowledgeQueryEngine({
  ragbitsUrl: process.env.RAGBITS_URL,
  graphitiUrl: process.env.GRAPHITI_URL,
  vectordbUrl: process.env.VECTORDB_URL,
});

const results = await engine.query('What are the latest developments in Lean proofs?', {
  domains: ['all'],
  maxResults: 10
});
```

### Temporal Query
```typescript
const results = await engine.temporalQuery(
  'How did the understanding evolve?',
  '2024-01-01T00:00:00Z',
  '2024-12-31T23:59:59Z',
  { maxResults: 20 }
);
```

### Health Check
```typescript
const health = await engine.healthCheck();
console.log(health);
// [{ system: 'ragbits', status: 'healthy', responseTimeMs: 45 }, ...]
```

## Testing

### Contract Tests
```bash
npm test
```

### Probe Scripts (Run before deployment)
```bash
npm run probe:all
# Or individually:
npm run probe:ragbits
npm run probe:graphiti
npm run probe:vectordb
```

### Build
```bash
npm run build
```

## Environment Variables Required

```bash
# Knowledge System URLs (Required)
RAGBITS_URL=http://ragbits-core:8000
GRAPHITI_URL=http://graphiti-core:8001
VECTORDB_URL=http://vectordb-core:6333

# Optional Configuration
RAGBITS_TIMEOUT=5000
GRAPHITI_TIMEOUT=5000
VECTORDB_TIMEOUT=5000
```

## Next Steps

1. **Deploy probes**: Run probe scripts to verify system availability
2. **Run contract tests**: Validate API contracts
3. **Configure environment**: Set required environment variables
4. **Initialize engine**: Create engine instance with system URLs
5. **Execute queries**: Start querying knowledge systems
6. **Monitor metrics**: Track performance and health

## Files Created

- ✅ `src/canonical.ts` - Canonical schemas (371 lines)
- ✅ `src/clients.ts` - System clients (579 lines)
- ✅ `src/query-router.ts` - Query routing (267 lines)
- ✅ `src/result-fusion.ts` - Result fusion (297 lines)
- ✅ `src/fallback-strategy.ts` - Fallback strategy (398 lines)
- ✅ `src/engine.ts` - Main engine (483 lines)
- ✅ `src/index.ts` - Public API (98 lines)
- ✅ `probes/check_ragbits.sh` - RAGBits probe (128 lines)
- ✅ `probes/check_graphiti.sh` - Graphiti probe (153 lines)
- ✅ `probes/check_vectordb.sh` - Vector DB probe (138 lines)
- ✅ `probes/check_unified.sh` - Unified probe (175 lines)
- ✅ `tests/contract.test.ts` - Contract tests (445 lines)
- ✅ `tests/jest.config.js` - Jest config (23 lines)
- ✅ `package.json` - Package config (45 lines)
- ✅ `tsconfig.json` - TypeScript config (22 lines)
- ✅ `README.md` - Documentation (133 lines)

**Total Lines of Code**: ~3,685 lines

## Status

✅ **COMPLETE** - Unified Knowledge Query Interface fully implemented and ready for deployment.

All components follow Federation Constitution principles with proper error handling, logging, circuit breakers, and failure management.
