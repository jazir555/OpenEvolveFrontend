# KarateClub Adapter

Graph machine learning adapter for KarateClub - providing 51 state-of-the-art algorithms for community detection, node embeddings, and graph embeddings.

## Overview

KarateClub is a Python library for unsupervised learning on graph structured data. This adapter provides a TypeScript/Node.js interface to KarateClub's algorithms, following OpenEvolve's federation architecture principles.

### Supported Algorithms

**Community Detection (10 algorithms)**:
- Label Propagation
- BigClam
- DANMF
- GEMSEC
- EdMot
- SCD
- Ego-Splitting
- SymmNMF
- M-NMF
- NNSED

**Node Embedding (32 algorithms)**:
- DeepWalk, Node2Vec, Walklets
- GraRep, HOPE, NetMF
- GraphWave, Role2Vec, SINR
- FEATHER-N, TADW, MUSAE, AE, FSCNMF, SINE, BANE, TENE, ASNE
- And more...

**Graph Embedding (10 algorithms)**:
- Graph2Vec
- Feather Graph
- NetLSD
- GeoScattering
- IGE
- LDP
- GL2Vec
- SF (Statistical Features)
- FGSD

## Installation

### Prerequisites

1. **Python 3.8+** with KarateClub installed:
```bash
pip install karateclub networkx numpy
```

2. **Node.js 18+** for the adapter

### Setup

```bash
# Install adapter dependencies
npm install

# Build TypeScript
npm run build

# Run contract tests
npm run test:contract
```

## Usage

### Basic Node Embedding

```typescript
import { createAdapter } from '@openevolve/karateclub-adapter';

const adapter = createAdapter({
  pythonPath: 'python3',
  timeoutMs: 120000,
});

// Generate node embeddings
const response = await adapter.generateNodeEmbeddings({
  algorithm: 'node2vec',
  graph: {
    nodes: [
      { id: 'node1', features: [1.0, 0.5, 0.3] },
      { id: 'node2', features: [0.8, 0.2, 0.9] },
    ],
    edges: [
      { source: 'node1', target: 'node2', weight: 1.0 },
    ],
    directed: false,
    weighted: true,
  },
  parameters: {
    dimensions: 128,
    walk_length: 80,
    walk_number: 10,
  },
  timeout_ms: 120000,
});

if (response.success) {
  console.log('Embeddings:', response.embeddings);
  console.log('Dimensions:', response.dimensions);
}
```

### Community Detection

```typescript
const response = await adapter.detectCommunities({
  algorithm: 'label_propagation',
  graph: myGraph,
  timeout_ms: 60000,
});

if (response.success) {
  console.log('Communities:', response.memberships);
  console.log('Num Communities:', response.num_communities);
  console.log('Community Sizes:', response.community_sizes);
}
```

### Combined Graph Analysis

```typescript
const response = await adapter.analyzeGraph({
  graph: myGraph,
  analyses: [
    'node_embeddings',
    'community_detection',
    'graph_statistics',
  ],
  node_embedding_algorithm: 'node2vec',
  community_algorithm: 'label_propagation',
  parameters: {
    embedding_dimensions: 128,
  },
  timeout_ms: 300000,
});

if (response.success) {
  console.log('Results:', response.results);
  console.log('Algorithms Used:', response.algorithms_used);
}
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_PATH` | Path to Python executable | `python3` |
| `TIMEOUT_MS` | Default timeout in milliseconds | `120000` |
| `MAX_RETRIES` | Maximum retry attempts | `2` |
| `TEMP_DIR` | Temporary directory for graph files | `/tmp/karateclub` |
| `KARATECLUB_LOG_LEVEL` | Logging level | `info` |

### Adapter Configuration

```typescript
const config = {
  pythonPath: 'python3',
  timeoutMs: 120000,
  maxRetries: 2,
  tempDir: '/tmp/karateclub',
  circuitBreaker: {
    failureThreshold: 5,
    successThreshold: 2,
    timeout: 60000,
    halfOpenMaxCalls: 1,
  },
  enableMetrics: true,
  logLevel: 'info',
};

const adapter = createAdapter(config);
```

## Architecture

### Data Flow

```
[Client Request]
    ↓
[Validate with Zod Schema]
    ↓
[Generate Python Script]
    ↓
[Write Graph to Temp File]
    ↓
[Spawn Python Subprocess]
    ↓
[KarateClub Executes]
    ↓
[Parse JSON Output]
    ↓
[Validate Response]
    ↓
[Return to Client]
```

### Key Components

1. **KarateClubAdapter**: Main adapter interface
2. **KarateClubMLClient**: Python subprocess execution
3. **Algorithm Registry**: Algorithm metadata and defaults
4. **Canonical Schemas**: Zod validation schemas

## Testing

### Contract Tests

```bash
# Run all tests
npm test

# Run contract tests only
npm run test:contract

# Run with coverage
npm run test:coverage

# Run in watch mode
npm run test:watch
```

### Probes

```bash
# Check KarateClub installation
bash probes/check_api.sh

# Check algorithm availability
bash probes/check_algorithms.sh

# Check embedding operations
bash probes/check_embeddings.sh
```

## Error Handling

### Circuit Breaker

The adapter includes a circuit breaker to prevent cascading failures:

- **CLOSED**: Normal operation
- **OPEN**: Failing, reject requests immediately
- **HALF_OPEN**: Testing if recovered

Configuration:
```typescript
circuitBreaker: {
  failureThreshold: 5,      // Open after 5 failures
  successThreshold: 2,      // Close after 2 successes
  timeout: 60000,           // Stay OPEN for 60 seconds
  halfOpenMaxCalls: 1,      // Allow 1 call in HALF_OPEN
}
```

### Retry Logic

ML operations use fewer retries due to their cost:

```typescript
// Node embeddings: 2 retries with exponential backoff
// Community detection: 1 retry (fast)
// Graph embeddings: 1 retry (very slow)
```

## Performance

### Typical Execution Times

| Algorithm | Graph Size | Execution Time |
|-----------|------------|----------------|
| Node2Vec | 1K nodes, 5K edges | 30-60s |
| DeepWalk | 1K nodes, 5K edges | 20-40s |
| Label Propagation | 10K nodes, 50K edges | 5-10s |
| Graph2Vec | 100 graphs | 60-120s |

### Resource Limits

- **Max Nodes**: 100,000
- **Max Edges**: 1,000,000
- **Max Timeout**: 10 minutes
- **Memory per Request**: ~2-4GB (Python subprocess)

## Known Limitations

1. **Stateless**: Each request spawns a new Python process
2. **No GPU**: KarateClub doesn't support GPU acceleration
3. **Graph Size**: Limited by available memory (typically 2-4GB)
4. **Timeout**: Must manually kill Python processes on timeout
5. **Subprocess Overhead**: ~100-500ms per request

## Troubleshooting

### Python Not Found

```bash
Error: spawn python3 ENOENT
```

**Solution**: Set `PYTHON_PATH` environment variable to correct Python executable.

### KarateClub Not Installed

```bash
ModuleNotFoundError: No module named 'karateclub'
```

**Solution**: Install KarateClub in Python environment:
```bash
pip install karateclub networkx numpy
```

### Timeout Errors

```bash
Error: Script execution timeout after 120000ms
```

**Solution**: Increase `timeout_ms` in request or reduce graph size.

### Memory Errors

```bash
MemoryError: Unable to allocate array
```

**Solution**: Reduce graph size or increase available memory.

## Contributing

When adding new algorithms:

1. Update `algorithms.ts` with algorithm metadata
2. Add script generation in `ml-client.ts`
3. Update canonical schemas if needed
4. Add contract tests
5. Update this README

## License

MIT License - See LICENSE file for details.

## References

- [KarateClub GitHub](https://github.com/benedekrozemberczki/karateclub)
- [KarateClub Paper](https://arxiv.org/abs/2008.13797)
- [OpenEvolve Federation](https://github.com/openevolve)

## Authors

OpenEvolve Federation Architecture Team
