# OpenEvolve gRPC Integration

High-performance gRPC integration layer between BubbleLab (TypeScript) and OpenEvolve Python backend.

## Overview

This integration provides:

- **gRPC Protocol**: Binary serialization for 5-10x performance improvement over REST
- **Streaming Support**: Real-time progress updates for long-running operations
- **Type Safety**: Protobuf-generated types for both TypeScript and Python
- **Service Mesh**: Load balancing, health checks, and circuit breakers
- **Backward Compatibility**: REST bridge for gradual migration

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  BubbleLab Studio (React + TypeScript)                              │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ gRPC (HTTP/2)
┌─────────────────────────────▼───────────────────────────────────────┐
│  TypeScript gRPC Client                                             │
│  - Connection pooling                                               │
│  - Automatic retries                                                │
│  - Streaming support                                                │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ Protocol Buffers
┌─────────────────────────────▼───────────────────────────────────────┐
│  Service Mesh                                                       │
│  - Load balancing (round-robin, weighted, health-based)             │
│  - Health checking                                                  │
│  - Circuit breakers                                                 │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ gRPC
┌─────────────────────────────▼───────────────────────────────────────┐
│  Python gRPC Server                                                 │
│  - Wraps bubblelabs_nodes                                           │
│  - Streaming execution                                              │
│  - Execution management                                             │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ Python imports
┌─────────────────────────────▼───────────────────────────────────────┐
│  OpenEvolve Core (bubblelabs_nodes)                                 │
│  - 90+ nodes (decomposition, knowledge, math, gauntlets)            │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

**Python:**
```bash
cd python
pip install -r requirements.txt
```

**TypeScript:**
```bash
cd typescript
npm install
```

### 2. Generate Code from Protobuf

```bash
cd scripts
./generate.sh
```

This generates:
- Python gRPC stubs in `python/generated/`
- TypeScript gRPC stubs in `typescript/generated/`

### 3. Start the gRPC Server

```bash
cd python
python server.py
```

Server starts on port 50051 by default.

### 4. Use in TypeScript

```typescript
import { createGRPCClient } from '@openevolve/grpc-client';

const client = createGRPCClient({
  host: 'localhost',
  port: 50051
});

await client.connect();

// List available nodes
const nodes = await client.listNodes();

// Execute a node
const result = await client.executeNode({
  nodeType: 'decomposition',
  inputs: {
    problem_statement: 'Design a scalable ML pipeline'
  }
});

// Or with streaming for long-running operations
await client.executeNodeStreaming(
  { nodeType: 'gauntlet', inputs: { ... } },
  (progress) => console.log(`${progress.percent}%: ${progress.message}`)
);
```

### 5. Use REST Bridge (Backward Compatible)

For existing code using REST:

```bash
cd python
python rest_bridge.py
```

The REST API at port 8000 proxies to the gRPC server:

```typescript
// Existing code continues to work
const response = await fetch('http://localhost:8000/api/integrations/decomposition/execute', {
  method: 'POST',
  body: JSON.stringify({ inputs: { ... } })
});
```

## Project Structure

```
bubblelab/integrations/openevolve-grpc/
├── proto/                          # Protobuf definitions
│   ├── common.proto               # Shared types
│   ├── nodes.proto                # Node registry
│   ├── decomposition.proto        # Decomposition service
│   ├── knowledge.proto            # Knowledge service
│   ├── math.proto                 # Math/verification service
│   └── gauntlet.proto             # Gauntlet service
├── python/                         # Python gRPC server
│   ├── server.py                  # Main gRPC server
│   ├── client.py                  # Python client (for testing)
│   ├── service_mesh.py            # Load balancing & circuit breakers
│   ├── rest_bridge.py             # REST compatibility layer
│   ├── requirements.txt           # Python dependencies
│   └── generated/                 # Generated protobuf code
├── typescript/                     # TypeScript gRPC client
│   ├── client.ts                  # Main client library
│   ├── package.json               # NPM manifest
│   ├── tsconfig.json              # TypeScript config
│   └── generated/                 # Generated protobuf code
└── scripts/                        # Build scripts
    └── generate.sh                # Code generation script
```

## Protocol Buffer Definitions

### Common Types

```protobuf
message RequestMetadata {
  string request_id = 1;
  string correlation_id = 2;
  google.protobuf.Timestamp timestamp = 3;
}

message ExecutionOptions {
  int32 timeout_seconds = 1;
  bool enable_streaming = 2;
  bool enable_checkpointing = 3;
  string execution_priority = 4;
}

enum ExecutionState {
  EXECUTION_STATE_PENDING = 1;
  EXECUTION_STATE_RUNNING = 2;
  EXECUTION_STATE_COMPLETED = 3;
  EXECUTION_STATE_FAILED = 4;
  EXECUTION_STATE_CANCELLED = 5;
}
```

### Node Registry Service

```protobuf
service NodeRegistry {
  rpc ListNodes(ListNodesRequest) returns (ListNodesResponse);
  rpc GetNodeSchema(GetNodeSchemaRequest) returns (GetNodeSchemaResponse);
  rpc ExecuteNode(NodeExecutionRequest) returns (NodeExecutionResponse);
  rpc ExecuteNodeStreaming(NodeExecutionRequest) returns (stream ExecutionUpdate);
  rpc GetExecutionStatus(GetExecutionStatusRequest) returns (GetExecutionStatusResponse);
  rpc CancelExecution(CancelExecutionRequest) returns (CancelExecutionResponse);
}
```

### Decomposition Service

```protobuf
service DecompositionService {
  rpc Decompose(DecompositionRequest) returns (DecompositionResult);
  rpc DecomposeStreaming(DecompositionRequest) returns (stream Progress);
  rpc Recompose(RecompositionRequest) returns (RecompositionResult);
  rpc RecommendStrategy(ProblemDefinition) returns (StrategyRecommendation);
}
```

## Service Mesh Features

### Load Balancing Strategies

- **Round Robin**: Even distribution across endpoints
- **Weighted**: Distribution based on endpoint weights
- **Least Connections**: Route to endpoint with fewest active connections
- **Health-Based**: Route based on health scores and response times

### Health Checking

```python
from service_mesh import HealthTracker

tracker = HealthTracker(
    check_interval_seconds=30,
    unhealthy_threshold=3,
    healthy_threshold=2
)

tracker.add_endpoint(Endpoint("localhost", 50051))
tracker.add_endpoint(Endpoint("localhost", 50052))

# Get healthy endpoints
healthy = tracker.get_healthy_endpoints()
```

### Circuit Breaker

```python
from service_mesh import CircuitBreaker, CircuitBreakerConfig

breaker = CircuitBreaker(
    endpoint=Endpoint("localhost", 50051),
    config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=3,
        timeout_seconds=30
    )
)

# Check if request can proceed
if await breaker.can_execute():
    try:
        result = await execute()
        await breaker.record_success()
    except Exception:
        await breaker.record_failure()
```

## Streaming Support

For long-running operations (MCTS, Evolution, Gauntlets):

### TypeScript Client

```typescript
await client.executeNodeStreaming(
  {
    nodeType: 'gauntlet',
    inputs: { target: 'neural_network' }
  },
  (progress) => {
    updateProgressBar(progress.percent);
    showStatusMessage(progress.message);
  }
);
```

### Python Server

```python
async def ExecuteNodeStreaming(self, request, context):
    for update in execution_updates:
        if context.is_active():
            yield ExecutionUpdate(
                progress=Progress(
                    percent=update.percent,
                    message=update.message
                )
            )
```

## Performance Comparison

| Metric | REST (HTTP/1.1) | gRPC (HTTP/2) | Improvement |
|--------|-----------------|---------------|-------------|
| Serialization | JSON (text) | Protobuf (binary) | 3-5x smaller |
| Connection | New per request | Persistent | 10x fewer connections |
| Latency | ~50ms | ~5ms | 10x faster |
| Streaming | Polling | Native | Real-time |
| Type Safety | Runtime | Compile-time | Errors caught early |

## Configuration

### Environment Variables

**Server:**
```bash
GRPC_HOST=0.0.0.0
GRPC_PORT=50051
GRPC_MAX_WORKERS=10
GRPC_ENABLE_REFLECTION=true
GRPC_ENABLE_HEALTH=true
```

**Client:**
```bash
GRPC_HOST=localhost
GRPC_PORT=50051
GRPC_MAX_RETRIES=3
GRPC_RETRY_DELAY_MS=1000
GRPC_POOL_SIZE=5
GRPC_TIMEOUT_MS=60000
```

**REST Bridge:**
```bash
REST_HOST=0.0.0.0
REST_PORT=8000
GRPC_HOST=localhost
GRPC_PORT=50051
```

## Migration Guide

### Phase 1: Deploy gRPC Server (No Code Changes)

1. Start gRPC server alongside existing REST server
2. Verify with health checks
3. Monitor performance

### Phase 2: Use REST Bridge

1. Point existing REST clients to bridge
2. Bridge proxies to gRPC
3. Gradually update clients

### Phase 3: Direct gRPC (New Code)

1. New features use gRPC client directly
2. Existing code continues via bridge
3. Migrate existing code incrementally

### Phase 4: Deprecate REST

1. Remove REST bridge
2. All clients use gRPC
3. Remove legacy REST server

## Testing

### Unit Tests

```bash
# Python
cd python
pytest

# TypeScript
cd typescript
npm test
```

### Integration Tests

```bash
# Start server
python python/server.py &

# Run integration tests
python python/test_integration.py
```

### Load Tests

```bash
# Usingghz (gRPC load tester)
ghz --proto=proto/nodes.proto \
    --call=openevolve.grpc.NodeRegistry.ListNodes \
    localhost:50051
```

## Troubleshooting

### Connection Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Code Generation Errors

```bash
# Regenerate from clean state
rm -rf python/generated typescript/generated
./scripts/generate.sh
```

### Performance Issues

1. Check connection pooling settings
2. Verify compression is enabled
3. Monitor server resource usage
4. Review load balancing strategy

## Contributing

1. Update proto definitions
2. Regenerate code: `./scripts/generate.sh`
3. Update both TypeScript and Python implementations
4. Add tests
5. Update documentation

## License

MIT License - See LICENSE file for details.
