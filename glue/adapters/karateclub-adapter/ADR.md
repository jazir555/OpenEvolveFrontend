# Architecture Decision Record: KarateClub Integration

## Status
**Accepted**

## Context

KarateClub is a Python library for graph representation learning with 51 state-of-the-art algorithms across three categories:
- **Community Detection** (10 algorithms): Discover communities and clusters in graphs
- **Node Embedding** (32 algorithms): Generate low-dimensional representations of nodes
- **Graph Embedding** (10 algorithms): Generate representations of entire graphs

KarateClub is used throughout OpenEvolve for:
- Knowledge graph analysis and structure discovery
- Node similarity and relationship extraction
- Graph classification and comparison
- Community detection in knowledge networks
- Feature generation for downstream ML tasks

The integration must support:
1. **Python Bridge**: KarateClub is Python-only, requires subprocess execution
2. **Long-Running Operations**: ML operations can take minutes
3. **Large Graph Support**: Handle graphs with thousands of nodes
4. **Algorithm Diversity**: Support 51 different algorithms with varying parameters
5. **Idempotency**: ML operations should be reproducible with same inputs

## Decision

### Architecture Pattern: Subprocess-Based Adapter with Canonical Schema

```
[Core OpenEvolve] --> [KarateClub Adapter (Canonical Layer)] --> [Python Subprocess] --> [KarateClub Engine]
```

### Key Design Choices

1. **Adapter Location**: `/glue/adapters/karateclub-adapter/`
   - Isolated from core-projects (Law of Air Gap)
   - Rewritten KarateClub utilities in adapter layer
   - Canonical schema at `/glue/schemas/karateclub-canonical.ts`

2. **Interface Strategy**: Python subprocess execution
   - Generate Python scripts dynamically
   - Execute via `child_process.spawn()`
   - Parse JSON output from Python scripts
   - Timeout handling via process termination

3. **Data Flow**:
   ```
   Input (Canonical Format)
       --> KarateClubAdapter.generateNodeEmbeddings()
       --> Generate Python script
       --> Execute Python subprocess
       --> Parse JSON output
       --> Convert to Canonical Format
       --> Output (Canonical Format)
   ```

4. **Algorithm Support**:
   - Node Embedding: DeepWalk, Node2Vec, Walklets, GraRep, HOPE, NetMF, GraphWave, Role2Vec, etc. (32 algorithms)
   - Community Detection: Label Propagation, BigClam, DANMF, GEMSEC, EdMot, SCD, etc. (10 algorithms)
   - Graph Embedding: Graph2Vec, Feather Graph, NetLSD, GeoScattering, etc. (10 algorithms)

## Consequences

### Positive Benefits

1. **Algorithm Coverage**: Access to 51 state-of-the-art graph ML algorithms
2. **Isolation**: Python process isolation prevents crashes from affecting Node.js
3. **Flexibility**: Easy to add new algorithms by updating Python script generation
4. **Type Safety**: TypeScript + Zod schemas ensure data structure validity
5. **Idempotency**: Same input graph + parameters = same output (with fixed random seed)
6. **Monitoring**: Built-in circuit breaker and metrics collection

### Negative Tradeoffs

1. **Process Overhead**: Spawning Python processes adds ~100-500ms overhead per request
2. **Serialization**: Graph data must be serialized to JSON for Python
3. **Memory**: Each Python subprocess has its own memory footprint
4. **Timeout Complexity**: Must manually kill Python processes on timeout
5. **Error Handling**: Must parse both stdout and stderr for errors

### Known Limitations

1. **Graph Size**: Limited by subprocess memory (typically 2-4GB per process)
2. **Execution Time**: Large graphs can take minutes to process
3. **Concurrency**: Each operation spawns a new process (not thread-safe)
4. **Statelessness**: Cannot persist models between requests (must re-train)
5. **GPU Support**: Standard KarateClub doesn't support GPU acceleration

## Implementation Details

### Core Components

#### 1. KarateClubMLClient
```typescript
class KarateClubMLClient {
  async generateNodeEmbeddings(
    request: NodeEmbeddingRequest
  ): Promise<NodeEmbeddingResponse>

  async detectCommunities(
    request: CommunityDetectionRequest
  ): Promise<CommunityDetectionResponse>

  async healthCheck(): Promise<{healthy: boolean, version?: string}>
}
```

**Capabilities**:
- Spawn Python subprocesses for ML operations
- Generate dynamic Python scripts for each algorithm
- Parse JSON output from KarateClub
- Circuit breaker integration
- Retry logic (fewer retries for long ML operations)

**Example**:
```typescript
const client = new KarateClubMLClient({
  pythonPath: 'python3',
  timeoutMs: 120000,
});

const response = await client.generateNodeEmbeddings({
  algorithm: 'node2vec',
  graph: myGraph,
  parameters: {
    dimensions: 128,
    walk_length: 80,
    walk_number: 10,
  },
  timeout_ms: 120000,
});
// response.embeddings[node_id] -> embedding vector
```

#### 2. KarateClubAdapter
```typescript
class KarateClubAdapter {
  async generateNodeEmbeddings(
    request: NodeEmbeddingRequest
  ): Promise<NodeEmbeddingResponse>

  async detectCommunities(
    request: CommunityDetectionRequest
  ): Promise<CommunityDetectionResponse>

  async analyzeGraph(
    request: GraphAnalysisRequest
  ): Promise<GraphAnalysisResponse>

  async healthCheck(): Promise<{healthy: boolean, version?: string}>
}
```

**Capabilities**:
- Validate requests using canonical schemas
- Execute ML operations via MLClient
- Metrics collection (success rate, execution time)
- Combined graph analysis (embeddings + communities + statistics)
- Structured JSON Lines logging

#### 3. Algorithm Registry
```typescript
const NODE_EMBEDDING_ALGORITHMS = {
  deepwalk: {
    name: 'DeepWalk',
    parameters: ['dimensions', 'walk_length', 'walk_number', 'window_size', 'seed'],
    defaultTimeout: 120000, // 2 minutes
  },
  node2vec: {
    name: 'Node2Vec',
    parameters: ['dimensions', 'walk_length', 'walk_number', 'p', 'q', 'window_size', 'seed'],
    defaultTimeout: 120000,
  },
  // ... 30 more algorithms
};
```

### API Endpoints (MCP Tools)

| Tool | Purpose | Timeout | Retry Strategy |
|------|---------|---------|----------------|
| `karateclub_node_embeddings` | Generate node embeddings | 120s | 2 attempts, exponential backoff |
| `karateclub_community_detection` | Detect communities | 60s | 1 attempt (fast) |
| `karateclub_graph_embeddings` | Generate graph embeddings | 300s | 1 attempt (very slow) |
| `karateclub_analyze_graph` | Combined analysis | 300s | No retry (complex) |

### Data Flow Diagrams

#### Node Embedding Flow
```
[Client]
  --> {algorithm: "node2vec", graph: {...}, timeout_ms: 120000}
[KarateClub Adapter]
  --> Validate request with Zod schema
  --> Generate Python script for node2vec
  --> Write graph to temporary JSON file
[Python Subprocess]
  --> Load graph from JSON
  --> Initialize Node2Vec model with parameters
  --> model.fit(graph)
  --> embedding = model.get_embedding()
  --> print(json.dumps({embeddings: {...}}))
[KarateClub Adapter]
  --> Parse JSON output from Python
  --> Validate response with Zod schema
  --> Delete temporary files
[Client]
  <-- {success: true, embeddings: {...}, dimensions: 128, ...}
```

### Configuration Requirements

#### Environment Variables
```bash
# Python Configuration
PYTHON_PATH=python3           # Python executable path

# KarateClub Configuration
KARATECLUB_API_URL=http://localhost:8000  # Optional: Future API service
TIMEOUT_MS=120000             # Default timeout (milliseconds)
MAX_RETRIES=2                 # Maximum retry attempts

# Temporary Files
TEMP_DIR=/tmp/karateclub      # Directory for temporary graph files

# Adapter Configuration
KARATECLUB_LOG_LEVEL=info     # Logging level
```

#### TypeScript Configuration
```typescript
const config: AdapterConfig = {
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
};
```

## Gotchas

### Algorithm Quirks Discovered

1. **Module Import Paths**:
   - Some algorithms are in submodules (e.g., `karateclub.community_detection.non_overlapping`)
   - **Solution**: Try multiple import paths in Python script

2. **Graph Directionality**:
   - Community detection algorithms expect undirected graphs
   - Node embedding algorithms can handle both
   - **Gotcha**: Passing directed graph to community detection may cause errors
   - **Solution**: Convert to undirected in Python script

3. **Memory Usage**:
   - Large graphs (>10K nodes) can consume >2GB RAM
   - **Gotcha**: Default Node.js memory limit may be exceeded
   - **Solution**: Run Python subprocesses in separate processes

4. **Timeout Handling**:
   - Python processes don't respect timeouts automatically
   - **Gotcha**: Must use `timeout` command or manual `kill`
   - **Solution**: Use `setTimeout()` + `process.kill()` in Node.js

5. **Random Seeds**:
   - Without seed, results are non-deterministic
   - **Gotcha**: Different Python versions may have different RNG
   - **Solution**: Always set `seed` parameter explicitly

6. **Empty Graphs**:
   - Some algorithms crash on empty graphs (0 nodes or 0 edges)
   - **Gotcha**: Error messages are cryptic
   - **Solution**: Validate graph size in request handler

### Version Requirements

| Component | Minimum Version | Recommended Version | Notes |
|-----------|----------------|---------------------|-------|
| karateclub (Python) | 1.2.0 | 1.3.0+ | 1.3 adds more algorithms |
| networkx (Python) | 2.8 | 3.0+ | Required for graph structures |
| numpy (Python) | 1.21 | 1.24+ | Required for numerical ops |
| Python | 3.8 | 3.10+ | 3.10 improves performance |

### Non-Obvious Behaviors

1. **Subprocess Overhead**:
   - Spawning Python process takes 100-500ms
   - **Gotcha**: This overhead is significant for small graphs
   - **Solution**: Batch operations when possible (future enhancement)

2. **JSON Serialization**:
   - NumPy arrays don't serialize to JSON directly
   - **Gotcha**: Must convert to `.tolist()` before serializing
   - **Solution**: Always use `.tolist()` in Python scripts

3. **Graph Features**:
   - Some algorithms require node features (attributed embeddings)
   - **Gotcha**: Passing features when algorithm doesn't support them wastes memory
   - **Solution**: Check algorithm requirements before including features

4. **File Cleanup**:
   - Temporary files may remain if process crashes
   - **Gotcha**: Disk space can fill up over time
   - **Solution**: Use temp directory with automatic cleanup

5. **Circuit Breaker State**:
   - Circuit breaker state is in-memory only
   - **Gotcha**: State lost on adapter restart
   - **Solution**: Persistent state for production (future enhancement)

## Circuit Breaker Configuration

### Timeout Values
```typescript
TIMEOUTS = {
  "node_embeddings": 120000,    // 120 seconds (2 minutes)
  "community_detection": 60000, // 60 seconds
  "graph_embeddings": 300000,   // 300 seconds (5 minutes)
  "graph_analysis": 300000,     // 300 seconds (5 minutes)
}
```

### Retry Strategies

#### Reduced Retries for ML Operations
```typescript
@retry(
  attempts=2,                   // Only 2 attempts (reduced from default 3)
  base_delay=2.0,               // 2 second delay
  max_delay=10.0,
  exponential=2.0,
)
async function generateNodeEmbeddings(...): Promise<NodeEmbeddingResponse>
```

**Usage**: Node embeddings, community detection, graph embeddings

**Reasoning**: ML operations are expensive; fewer retries save resources

### Failure Thresholds

```typescript
CIRCUIT_BREAKER = {
  "failure_threshold": 5,        // open after 5 failures
  "success_threshold": 2,        // close after 2 successes
  "timeout": 60000,              // open state duration (60 seconds)
  "half_open_max_calls": 1       // test call in half-open state
}
```

**States**:
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit tripped, requests fail immediately
- **HALF_OPEN**: Test if service recovered, allow 1 call

**Triggers**:
- 5 consecutive failures (timeout or exception)
- 3 consecutive timeouts (>60s)
- Memory errors (OOM)

**Recovery**:
- 2 consecutive successes → CLOSE
- 60s timeout → HALF_OPEN

## Security Considerations

### Input Validation

#### Graph Structure Validation
```typescript
function validateGraphStructure(graph: GraphStructure): boolean {
  // Max nodes to prevent OOM
  if (graph.nodes.length > 100000) {
    throw new Error('Graph too large: >100K nodes');
  }

  // Max edges to prevent OOM
  if (graph.edges.length > 1000000) {
    throw new Error('Graph too large: >1M edges');
  }

  // Check for self-loops (may cause issues in some algorithms)
  const hasSelfLoops = graph.edges.some(e => e.source === e.target);
  if (hasSelfLoops) {
    // Warn but allow (some algorithms handle this)
  }

  return true;
}
```

### Resource Limits

```typescript
const MAX_NODES = 100000;
const MAX_EDGES = 1000000;
const MAX_EXECUTION_TIME = 600000; // 10 minutes
const MAX_MEMORY_MB = 8192; // 8GB

function enforceResourceLimits(request: MLRequest): void {
  if (request.graph.nodes.length > MAX_NODES) {
    throw new Error(`Graph too large: ${request.graph.nodes.length} > ${MAX_NODES}`);
  }

  if (request.timeout_ms > MAX_EXECUTION_TIME) {
    throw new Error(`Timeout too long: ${request.timeout_ms} > ${MAX_EXECUTION_TIME}`);
  }
}
```

### Python Code Injection Prevention

```typescript
function sanitizeAlgorithmName(algorithm: string): void {
  // Only allow alphanumeric and underscore
  const validPattern = /^[a-z_][a-z0-9_]*$/;

  if (!validPattern.test(algorithm)) {
    throw new Error(`Invalid algorithm name: ${algorithm}`);
  }

  // Check against whitelist
  const validAlgorithms = [...NODE_EMBEDDING_ALGORITHMS, ...COMMUNITY_ALGORITHMS];

  if (!validAlgorithms.includes(algorithm)) {
    throw new Error(`Unknown algorithm: ${algorithm}`);
  }
}
```

---

## References

- **KarateClub GitHub**: https://github.com/benedekrozemberczki/karateclub
- **KarateClub Paper**: Rozemberczki, B., et al. "Karate Club: A Toolkit for Graph Representation Learning"
- **OpenEvolve Integration**: `/core-projects/openevolve/openevolve/knowledge_engine/core/backends/karateclub_backend.py`
- **Canonical Schema**: `/glue/schemas/karateclub-canonical.ts`

**Created**: 2026-02-03
**Author**: OpenEvolve Architecture Team
**Status**: Accepted, Implemented
**Last Updated**: 2026-02-03
