# Unified Knowledge Query Interface

**Federation Constitution Compliant Multi-System Knowledge Query Engine**

## Overview

This library provides a unified interface to query multiple knowledge systems simultaneously:
- **RAGBits**: Document-based RAG and knowledge retrieval
- **Graphiti**: Temporal knowledge graph with entity relationships
- **Vector DB**: High-performance semantic vector search

## Architecture

```
User Query
    ↓
Parse & Analyze
    ↓
Select Systems (RAGBits, Graphiti, Vector DB)
    ↓
Execute Queries (Parallel when possible)
    ↓
Merge & Normalize Results
    ↓
Resolve Conflicts
    ↓
Rank & Filter
    ↓
Return Unified Result
```

## Installation

```bash
npm install @openevolve/unified-knowledge-query
```

## Usage

### Basic Query

```typescript
import { UnifiedKnowledgeQueryEngine } from '@openevolve/unified-knowledge-query';

const engine = new UnifiedKnowledgeQueryEngine({
  ragbits: { url: process.env.RAGBITS_URL },
  graphiti: { url: process.env.GRAPHITI_URL },
  vectordb: { url: process.env.VECTORDB_URL },
});

const results = await engine.query('What are the latest developments in Lean proofs?', {
  domains: ['all'],
  knowledgeTypes: ['document', 'entity'],
  maxResults: 10
});
```

### Temporal Query

```typescript
const results = await engine.query('How did the understanding of this concept evolve?', {
  domains: ['graphiti'],
  temporalFilter: {
    startDate: '2024-01-01T00:00:00Z',
    endDate: '2024-12-31T23:59:59Z'
  }
});
```

### Graph Traversal

```typescript
const results = await engine.query('Show related concepts and their relationships', {
  domains: ['graphiti'],
  queryType: 'graph-traversal',
  maxDepth: 3
});
```

## Query Types

1. **Semantic Search**: Vector DB + RAGBits for document retrieval
2. **Temporal Query**: Graphiti with time filters for historical analysis
3. **Graph Traversal**: Graphiti relationship queries for knowledge exploration
4. **Hybrid Multi-System**: Combine all sources for comprehensive results
5. **Fallback Query**: Graceful degradation when systems are unavailable

## Federation Constitution Compliance

- **Law of Air Gap**: No imports from core-projects directory
- **Law of Runtime Truth**: Probe scripts verify API availability before execution
- **Law of Idempotency**: All queries are safe to retry
- **Law of Configuration Explicitness**: All URLs and timeouts via environment variables
- **Failure Management**: Circuit breakers and fallback strategies for all system calls
- **Observability**: Structured JSON logging with correlation IDs

## Probes

Before deploying, run probe scripts to verify system availability:

```bash
npm run probe:all
```

## Testing

```bash
npm test
npm run test:coverage
```

## API Reference

See `/dist/index.d.ts` for complete TypeScript definitions.

## Canonical Schema

All queries and results conform to the canonical schemas defined in `canonical.ts`:
- `UnifiedKnowledgeQuery`: Input query specification
- `UnifiedQueryResult`: Merged results from all systems
- `KnowledgeDomain`: System selector
- `TemporalFilter`: Time-based filtering
- `KnowledgeType`: Result type filter
