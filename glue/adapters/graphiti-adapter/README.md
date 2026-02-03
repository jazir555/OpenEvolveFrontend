# Graphiti Adapter

Temporal knowledge graph adapter for the OpenEvolve Federation.

## Overview

This adapter integrates [Graphiti](https://github.com/getgraphiti/core), a temporal knowledge graph system that enables:

- **Temporal knowledge tracking** - Track what was true at specific points in time
- **Entity extraction** - Automatic extraction of entities and relationships from text
- **Hybrid search** - Semantic + keyword + graph traversal search
- **Community detection** - Automatically group related entities

## Architecture

```
OpenEvolve System
        │
        ▼
┌─────────────────────────────────────────┐
│  Graphiti Adapter (TypeScript)          │
│  - Canonical schema validation          │
│  - Circuit breaker                      │
│  - Retry logic with exponential backoff │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Graphiti Core (Python Service)         │
│  - LLM-based entity extraction          │
│  - Temporal graph operations            │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  Neo4j Graph Database                   │
│  - Entity nodes                         │
│  - Relationship edges                   │
│  - Temporal episode nodes               │
└─────────────────────────────────────────┘
```

## Installation

```bash
npm install @openevolve/graphiti-adapter
```

## Configuration

### Environment Variables

Required:
- `NEO4J_URI` - Neo4j connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER` - Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` - Neo4j password (required)

Optional:
- `OPENAI_API_KEY` - OpenAI API key for LLM-based entity extraction
- `ANTHROPIC_API_KEY` - Anthropic API key (alternative to OpenAI)
- `TIMEOUT_MS` - Request timeout in milliseconds (default: `30000`)

### Usage

```typescript
import { GraphitiAdapter } from '@openevolve/graphiti-adapter';

// Initialize adapter
const adapter = new GraphitiAdapter({
  graphiti_api_url: 'http://localhost:8000',
  neo4j_uri: process.env.NEO4J_URI,
  neo4j_user: process.env.NEO4J_USER,
  neo4j_password: process.env.NEO4J_PASSWORD,
  openai_api_key: process.env.OPENAI_API_KEY,
  timeout_ms: 30000,
});

// Initialize and verify connection
await adapter.initialize();

// Add an episode (temporal knowledge)
const result = await adapter.addEpisode({
  name: 'Employee Onboarding',
  content: 'John Doe joined OpenEvolve as a Senior Software Engineer on January 15, 2024.',
  episode_type: 'text',
  valid_at: '2024-01-15T09:00:00.000Z',
  source_description: 'HR system',
});

console.log(`Extracted ${result.entities_extracted} entities`);
console.log(`Extracted ${result.relationships_extracted} relationships`);
```

## Features

### 1. Episode Operations

Add temporal knowledge episodes:

```typescript
const result = await adapter.addEpisode({
  name: 'Project Meeting',
  content: 'The team decided to use TypeScript for the new project.',
  episode_type: 'text',
  valid_at: new Date().toISOString(),
});

// Bulk operations
const results = await adapter.addEpisodesBulk([episode1, episode2, episode3]);
```

### 2. Triplet Operations

Add direct relationships between entities:

```typescript
const result = await adapter.addTriplet({
  subject: {
    name: 'John Doe',
    labels: ['Person', 'Employee'],
  },
  predicate: {
    relation_type: 'WORKS_AT',
    fact: 'is employed by',
  },
  object: {
    name: 'OpenEvolve',
    labels: ['Organization'],
  },
});
```

### 3. Search Operations

Search the knowledge graph:

```typescript
const result = await adapter.search({
  query: 'John Doe employer',
  temporal_filter: 'current',
  max_results: 10,
});

console.log('Found edges:', result.edges);
console.log('Found nodes:', result.nodes);
```

### 4. Temporal Queries

Query knowledge at specific points in time:

```typescript
// Point-in-time query
const result = await adapter.queryAtPointInTime(
  'Who worked at OpenEvolve?',
  '2024-06-01T00:00:00.000Z',
  10
);

// Entity timeline
const timeline = await adapter.getEntityTimeline(
  'John Doe',
  '2024-01-01T00:00:00.000Z',
  '2024-12-31T23:59:59.999Z'
);
```

### 5. Entity Operations

```typescript
// Get entity by UUID
const entity = await adapter.getEntity(entityUuid);

console.log(entity.name, entity.labels, entity.summary);
```

## Canonical Schema

All data is normalized to the canonical schema defined in `../../schemas/graphiti-canonical.ts`:

### CanonicalEntity
```typescript
{
  id: string;              // UUID
  name: string;
  labels: string[];
  summary?: string;
  created_at: string;      // UTC ISO-8601
  attributes: Record<string, unknown>;
}
```

### CanonicalEpisode
```typescript
{
  id: string;              // UUID
  name: string;
  content: string;
  episode_type: EpisodeType;
  valid_at: string;        // When the event occurred (UTC)
  created_at: string;      // When added to graph (UTC)
}
```

### CanonicalEntityEdge
```typescript
{
  id: string;              // UUID
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  fact: string;
  created_at: string;      // UTC ISO-8601
}
```

## Probes

Before using the adapter, run the probe scripts to verify the Graphiti/Neo4j setup:

```bash
# Test Neo4j connection and basic operations
cd glue/adapters/graphiti-adapter/probes
./check_api.sh

# Test graph operations (nodes, edges, episodes)
./check_graph.sh

# Test full CRUD operations
./check_entities.sh
```

## Tests

```bash
cd glue/adapters/graphiti-adapter

# Install test dependencies
cd tests && npm install

# Run all tests
npm test

# Run with coverage
npm run test:coverage

# Skip integration tests
export SKIP_INTEGRATION_TESTS=true
npm test
```

## Compliance

This adapter follows the Federation Constitution:

- **Law of Configuration Explicitness**: All config via environment variables
- **Law of Runtime Truth**: Probe scripts verify API before use
- **Law of Idempotency**: All operations safe to run multiple times
- **Law of UTC**: All timestamps in UTC ISO-8601 format
- **Circuit Breaker**: Prevents cascading failures
- **Retry Logic**: Exponential backoff for transient failures

## Error Handling

The adapter uses a circuit breaker to prevent cascading failures:

```typescript
try {
  const result = await adapter.search({ query: 'test', max_results: 10 });
} catch (error) {
  // Check if circuit is open
  const health = await adapter.healthCheck();
  if (health.circuit_state === 'open') {
    console.error('Graphiti service is down, using fallback');
    // Use fallback or cached data
  }
}
```

## Health Checks

```typescript
const health = await adapter.healthCheck();

console.log({
  healthy: health.healthy,
  initialized: health.initialized,
  graphiti_connected: health.graphiti_connected,
  circuit_state: health.circuit_state,
});
```

## Statistics

```typescript
const stats = await adapter.getStatistics();

console.log({
  entities_count: stats.entities_count,
  relationships_count: stats.relationships_count,
  episodes_count: stats.episodes_count,
  communities_count: stats.communities_count,
});
```

## Cleanup

```typescript
await adapter.close();
```

## References

- Graphiti Core: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\core-projects\graphiti`
- Graphiti Python Integration: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\knowledge_engine\integrations\graphiti_integration.py`
- Canonical Schema: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\glue\schemas\graphiti-canonical.ts`
- ADR: `C:\Users\mmeadow\Documents\OpenEvolve\Frontend\glue\adapters\graphiti-adapter\ADR.md`

## License

Apache-2.0

## Authors

OpenEvolve Federation
