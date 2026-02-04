# Quick Start Guide - Unified Knowledge Query Interface

## Installation

```bash
cd glue/lib/unified-knowledge-query
npm install
```

## Configuration

Set environment variables:

```bash
export RAGBITS_URL="http://ragbits-core:8000"
export GRAPHITI_URL="http://graphiti-core:8001"
export VECTORDB_URL="http://vectordb-core:6333"
```

## Pre-Deployment Verification

### 1. Run Probes (Federation Constitution - Law of Runtime Truth)

```bash
# Probe all systems
npm run probe:all

# Or probe individually
npm run probe:ragbits
npm run probe:graphiti
npm run probe:vectordb
```

**Expected Output**: All systems should show "OPERATIONAL"

### 2. Run Contract Tests

```bash
npm test
```

**Expected Output**: All contract tests should pass

### 3. Build TypeScript

```bash
npm run build
```

**Expected Output**: Compiled JavaScript in `dist/` directory

## Basic Usage

### Initialize Engine

```typescript
import { UnifiedKnowledgeQueryEngine } from '@openevolve/unified-knowledge-query';

const engine = new UnifiedKnowledgeQueryEngine({
  ragbitsUrl: process.env.RAGBITS_URL,
  graphitiUrl: process.env.GRAPHITI_URL,
  vectordbUrl: process.env.VECTORDB_URL,
});
```

### Query Examples

#### 1. Basic Unified Query

```typescript
const results = await engine.query('What are the latest Lean proofs?', {
  domains: ['all'],
  maxResults: 10
});

console.log(`Found ${results.results.length} results`);
console.log(`Overall confidence: ${results.confidence}`);
```

#### 2. Semantic Search (Documents)

```typescript
const results = await engine.semanticSearch('mathematical proofs', {
  maxResults: 20,
  minConfidence: 0.7
});
```

#### 3. Temporal Query (Time-Filtered)

```typescript
const results = await engine.temporalQuery(
  'How did this concept evolve?',
  '2024-01-01T00:00:00Z',
  '2024-12-31T23:59:59Z',
  { maxResults: 15 }
);
```

#### 4. Graph Traversal (Relationships)

```typescript
const results = await engine.graphTraversal('Show related concepts', {
  maxDepth: 3,
  knowledgeTypes: ['entity', 'relationship']
});
```

#### 5. Hybrid Query (All Systems)

```typescript
const results = await engine.hybridQuery(
  'Comprehensive search across all knowledge',
  {
    maxResults: 50,
    minConfidence: 0.5,
    knowledgeTypes: ['document', 'entity', 'relationship']
  }
);
```

### Result Structure

```typescript
{
  query: "What are the latest Lean proofs?",
  results: [
    {
      content: "Result content...",
      source: "ragbits",  // or 'graphiti' or 'vectordb'
      id: "unique-id",
      type: "document",   // or 'entity', 'relationship', etc.
      confidence: 0.85,
      relevance: 0.92,
      timestamp: "2024-01-15T10:30:00Z",
      metadata: { /* additional data */ }
    },
    // ... more results
  ],
  sources: [
    {
      system: "ragbits",
      queryTimeMs: 123,
      resultCount: 5,
      success: true
    },
    // ... more sources
  ],
  confidence: 0.87,
  executionTimeMs: 456,
  correlationId: "550e8400-e29b-41d4-a716-446655440000"
}
```

## Health Monitoring

### Check System Health

```typescript
const health = await engine.healthCheck();

health.forEach(h => {
  console.log(`${h.system}: ${h.status} (${h.responseTimeMs}ms)`);
});
```

### Get Engine Metrics

```typescript
const metrics = await engine.getMetrics();

console.log(`Total queries: ${metrics.totalQueries}`);
console.log(`Success rate: ${metrics.successfulQueries / metrics.totalQueries * 100}%`);
console.log(`Average time: ${metrics.averageQueryTime}ms`);
```

## Query Options

### Available Options

```typescript
interface QueryOptions {
  // Which systems to query
  domains?: ('ragbits' | 'graphiti' | 'vectordb' | 'all')[];

  // Knowledge types to retrieve
  knowledgeTypes?: ('document' | 'entity' | 'proof' | 'code' | 'relationship' | 'all')[];

  // Maximum number of results
  maxResults?: number; // default: 50, max: 1000

  // Minimum confidence threshold (0.0 - 1.0)
  minConfidence?: number; // default: 0.0

  // Temporal filter for time-based queries
  temporalFilter?: {
    startDate?: string;  // ISO-8601 UTC
    endDate?: string;    // ISO-8601 UTC
    pointInTime?: string; // ISO-8601 UTC
  };

  // Query strategy
  queryType?: 'semantic-search' | 'temporal-query' | 'graph-traversal' | 'hybrid' | 'fallback';

  // Graph traversal depth
  maxDepth?: number; // default: 2, max: 10
}
```

## Query Types

### 1. Semantic Search
- **Systems**: Vector DB + RAGBits
- **Best for**: Document retrieval, semantic similarity
- **Execution**: Parallel

### 2. Temporal Query
- **Systems**: Graphiti
- **Best for**: Historical analysis, time-based filtering
- **Execution**: Sequential

### 3. Graph Traversal
- **Systems**: Graphiti
- **Best for**: Relationship exploration, entity connections
- **Execution**: Sequential

### 4. Hybrid Query
- **Systems**: All systems
- **Best for**: Comprehensive search
- **Execution**: Parallel when possible

### 5. Fallback Query
- **Systems**: Available systems
- **Best for**: Degraded mode operation
- **Execution**: Sequential with fallbacks

## Error Handling

```typescript
try {
  const results = await engine.query('test query', { maxResults: 10 });
  console.log('Success:', results);
} catch (error) {
  if (error.message.includes('Circuit breaker is OPEN')) {
    console.error('System is down, using fallback');
  } else if (error.message.includes('All systems failed')) {
    console.error('No knowledge systems available');
  } else {
    console.error('Query failed:', error);
  }
}
```

## Troubleshooting

### Issue: "No knowledge systems configured"
**Solution**: Set environment variables for RAGBITS_URL, GRAPHITI_URL, VECTORDB_URL

### Issue: "Circuit breaker is OPEN"
**Solution**: System is down. Check health with `engine.healthCheck()`. Wait for automatic recovery (60s) or use fallback systems.

### Issue: "All systems failed"
**Solution**: All configured systems are unavailable. Check system status and network connectivity.

### Issue: Low confidence results
**Solution**: Adjust `minConfidence` threshold or try different query type.

## Architecture Diagram

```
User Query
    ↓
[UnifiedKnowledgeQueryEngine]
    ↓
[QueryRouter] → Determine strategy & select systems
    ↓
┌─────────────────────────────────────┐
│  Execute Queries (Parallel/Serial)  │
├──────────────┬──────────────┬───────┤
│  RAGBits     │   Graphiti   │VectorDB│
│  Client      │    Client    │ Client │
└──────────────┴──────────────┴───────┘
    ↓
[FallbackStrategy] → Graceful degradation
    ↓
[ResultFusion] → Merge, normalize, rank
    ↓
UnifiedQueryResult
```

## Federation Constitution Compliance

- ✅ **Air Gap**: No imports from core-projects
- ✅ **Runtime Truth**: Probe scripts verify APIs
- ✅ **Untouchable DB**: Read-only queries only
- ✅ **Idempotency**: Safe to retry
- ✅ **Configuration**: All via environment variables
- ✅ **UTC**: All timestamps in ISO-8601 UTC

## Support

For detailed documentation, see:
- `README.md` - Complete API reference
- `IMPLEMENTATION_SUMMARY.md` - Implementation details
- `src/canonical.ts` - Type definitions

## Files Reference

**Core Implementation**:
- `src/engine.ts` - Main engine
- `src/clients.ts` - System clients
- `src/query-router.ts` - Query routing
- `src/result-fusion.ts` - Result merging
- `src/fallback-strategy.ts` - Fallback logic
- `src/canonical.ts` - Schemas and types

**Testing & Probes**:
- `probes/check_*.sh` - API probe scripts
- `tests/contract.test.ts` - Contract tests

**Configuration**:
- `package.json` - Dependencies
- `tsconfig.json` - TypeScript config
- `tests/jest.config.js` - Jest config
