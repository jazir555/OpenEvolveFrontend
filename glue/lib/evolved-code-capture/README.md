# Evolved Code Capture System

Stores OpenEvolve's evolved code in knowledge systems (Vector DB + Graphiti) for semantic search and lineage tracking.

## Overview

This system captures evolved code solutions from OpenEvolve and stores them in:
- **Vector Database** (e.g., Qdrant, Pinecone): For semantic similarity search
- **Graphiti** (Neo4j-based temporal knowledge graph): For lineage tracking and evolution history

Following the Federation Constitution:
- **Law of the Air Gap**: No imports from core-projects
- **Law of Runtime Truth**: Verify connections before use
- **Law of Idempotency**: All operations safe to run multiple times
- **Law of Configuration Explicitness**: All config via environment variables
- **Law of UTC**: All timestamps in UTC ISO-8601 format

## Architecture

```
OpenEvolve Evolution Complete
        ↓
Extract Best Solution
        ↓
[Generate Embedding] + [Create Graph Episode]
        ↓                    ↓
   Vector DB          Graphiti
        ↓                    ↓
   Semantic Search    Lineage Tracking
        ↓                    ↓
        Unified Knowledge Storage
```

## Installation

```bash
npm install
```

## Configuration

### Environment Variables

Required:
```bash
# Vector DB Adapter
VECTORDB_ADAPTER_URL=http://vectordb-adapter:8000
EVOLVED_CODE_COLLECTION=evolved_code

# Graphiti Adapter
GRAPHITI_ADAPTER_URL=http://graphiti-adapter:8000
```

Optional:
```bash
# Embedding Configuration
EMBEDDING_DIMENSION=1536
OPENAI_API_KEY=sk-...  # For OpenAI embeddings (recommended)

# Feature Flags
ENABLE_VECTOR_STORAGE=true
ENABLE_GRAPH_STORAGE=true

# Metrics
TRACK_METRICS=true
```

## Usage

### Basic Example

```typescript
import { EvolvedCodeCapturer, Problem, EvolvedCode } from '@openevolve/evolved-code-capture';

// Create capturer
const capturer = new EvolvedCodeCapturer({
  vector_storage: {
    vectordb_adapter_url: 'http://vectordb-adapter:8000',
    collection_name: 'evolved_code',
    embedding_dimension: 1536,
    embedding_api_key: process.env.OPENAI_API_KEY,
  },
  graph_storage: {
    graphiti_adapter_url: 'http://graphiti-adapter:8000',
  },
  enable_vector_storage: true,
  enable_graph_storage: true,
});

// Initialize
await capturer.initialize();

// Capture evolved code
const problem: Problem = {
  description: 'Optimize matrix multiplication',
  type: 'algorithm_optimization',
  constraints: {
    max_memory_mb: 512,
    max_runtime_ms: 5000,
  },
};

const solution: EvolvedCode = {
  id: '123e4567-e89b-12d3-a456-426614174000',
  problem,
  language: 'python',
  code: 'def matrix_multiply(A, B): return np.dot(A, B)',
  metrics: {
    iterations: 100,
    fitness_score: 0.95,
    fitness_improvement: 0.45,
    duration_ms: 15000,
  },
  timestamp_utc: new Date().toISOString(),
  is_valid: true,
};

const result = await capturer.captureEvolution(
  problem,
  solution,
  solution.metrics
);

console.log('Capture result:', result);
```

### Search for Similar Problems

```typescript
// Find similar previously-solved problems
const similar = await capturer.searchSimilarProblems(
  problem,
  10  // max results
);

similar.forEach((solution) => {
  console.log(`Similarity: ${solution.similarity_score}`);
  console.log(`Fitness: ${solution.evolved_code.metrics.fitness_score}`);
  console.log(`Code: ${solution.evolved_code.code}`);
});
```

### Get Evolution Lineage

```typescript
// Track full evolution tree from initial to final solution
const lineage = await capturer.getEvolutionLineage(codeId);

console.log(`Total nodes: ${lineage.total_nodes}`);
console.log(`Depth: ${lineage.depth}`);
console.log(`Branches: ${lineage.branches}`);

lineage.nodes.forEach((node) => {
  console.log(`Generation ${node.generation}: fitness=${node.fitness_score}`);
});
```

### Get Metrics

```typescript
const metrics = await capturer.getMetrics();

console.log(`Total captures: ${metrics.total_captures}`);
console.log(`Success rate: ${metrics.successful_captures / metrics.total_captures}`);
console.log(`Avg processing time: ${metrics.average_processing_time_ms}ms`);
console.log(`Problem types:`, metrics.problem_type_distribution);
console.log(`Languages:`, metrics.language_distribution);
```

## Canonical Schemas

### Problem

```typescript
interface Problem {
  description: string;
  type: ProblemType;  // algorithm_optimization, bug_fix, etc.
  constraints?: Constraints;
  input_spec?: string;
  output_spec?: string;
  test_cases?: TestCase[];
  difficulty?: 'easy' | 'medium' | 'hard' | 'expert';
  tags?: string[];
}
```

### Evolved Code

```typescript
interface EvolvedCode {
  id: string;  // UUID
  problem: Problem;
  language: Language;  // python, javascript, etc.
  code: string;
  metrics: EvolutionMetrics;
  timestamp_utc: string;  // ISO-8601
  is_valid: boolean;
  parent_code_id?: string;  // For lineage tracking
  generation_number?: number;
  tags?: string[];
}
```

### Evolution Metrics

```typescript
interface EvolutionMetrics {
  iterations: number;
  fitness_score: number;
  fitness_improvement: number;
  duration_ms: number;
  generations?: number;
  population_size?: number;
  mutation_rate?: number;
  convergence_generation?: number;
  success_rate?: number;
  benchmark_score?: number;
}
```

## Probes

Before using the system, verify storage backends are available:

```bash
# Check storage connectivity
./probes/check_storage.sh

# Check retrieval operations
./probes/check_retrieval.sh
```

## Testing

### Contract Tests

Contract tests validate that schemas and interfaces match expected contracts:

```bash
npm run test:contract
```

These tests run on container startup. If they fail, the adapter refuses to start.

### Full Test Suite

```bash
npm test
npm run test:coverage
```

## Integration with OpenEvolve

### OpenEvolve Adapter

The `openevolve-adapter` integrates this capture system with OpenEvolve:

```typescript
import { OpenEvolveAdapter } from '@openevolve/openevolve-adapter';
import { EvolvedCodeCapturer } from '@openevolve/evolved-code-capture';

// Create adapters
const openEvolve = new OpenEvolveAdapter({...});
const capturer = new EvolvedCodeCapturer({...});

// Initialize both
await openEvolve.initialize();
await capturer.initialize();

// Run evolution and capture result
const result = await openEvolve.evolve(problem);

if (result.best_solution) {
  await capturer.captureEvolution(
    problem,
    result.best_solution,
    result.metrics
  );
}
```

### Webhook Integration

Configure OpenEvolve to webhook on evolution completion:

```bash
# OpenEvolve configuration
EVOLUTION_WEBHOOK_URL=http://capturer:8000/webhooks/evolution
EVOLUTION_WEBHOOK_EVENTS=completion
```

## Data Flow

### Capture Flow

1. OpenEvolve completes evolution
2. Extract best solution from generation
3. Validate against canonical schemas
4. Generate embedding (OpenAI or hash-based)
5. Store in Vector DB with metadata
6. Create episode in Graphiti
7. Link problem to solution entities
8. Track lineage if parent_code_id exists
9. Return capture result

### Retrieval Flow

1. Query by problem description
2. Generate embedding for query
3. Search Vector DB for similar problems
4. Fetch full solutions from storage
5. Return ranked by similarity score

### Lineage Flow

1. Query by code_id
2. Search Graphiti for related episodes
3. Build evolution tree structure
4. Calculate depth and branches
5. Return full lineage

## Failure Management

Following Federation Constitution failure strategies:

- **Transient Failure**: Exponential backoff with jitter
- **System Failure**: Circuit breaker (stop hammering dead services)
- **Logic Failure**: Return error in CaptureResult, don't block pipeline

## Metrics Tracking

The system tracks:

- Total/successful/failed captures
- Average processing time
- Problem type distribution
- Language distribution
- Last capture timestamp

Access via `getMetrics()` or reset with `resetMetrics()`.

## Federation Constitution Compliance

This system enforces:

1. **Law of the Air Gap**: No imports from `core-projects/`
2. **Law of Runtime Truth**: Probes verify before use
3. **Law of the Untouchable DB**: Read-only to application databases
4. **Law of Idempotency**: All operations safe to retry
5. **Law of Configuration Explicitness**: Crash on missing config
6. **Law of UTC**: All timestamps in UTC ISO-8601

## Troubleshooting

### Vector DB Connection Failed

```bash
# Check Vector DB adapter health
curl http://vectordb-adapter:8000/health

# Check collection exists
curl http://vectordb-adapter:8000/collections/evolved_code
```

### Graphiti Connection Failed

```bash
# Check Graphiti adapter health
curl http://graphiti-adapter:8000/health

# Check Neo4j connection
curl http://graphiti-adapter:8000/statistics
```

### Embedding Generation Failed

If using OpenAI embeddings:
- Check `OPENAI_API_KEY` is valid
- Verify API quota not exceeded
- Check network connectivity to api.openai.com

Falls back to hash-based embeddings if API key missing (not semantically meaningful).

## License

MIT

## Author

OpenEvolve Federation
