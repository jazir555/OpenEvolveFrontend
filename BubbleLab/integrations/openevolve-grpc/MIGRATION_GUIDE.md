# Migration Guide: HTTP REST to gRPC

This guide helps you migrate from the existing HTTP REST API to the new gRPC integration.

## Overview

The migration maintains **100% backward compatibility** through the REST bridge. You can migrate incrementally without breaking existing code.

## Current State

```
Before (Current):
┌──────────────┐     HTTP      ┌──────────────┐
│  TypeScript  │ ────────────> │  Python API  │
│   Client     │    REST       │   Server     │
└──────────────┘               └──────────────┘
                                    │
                                    │ imports
                                    ▼
                              ┌──────────────┐
                              │ bubblelabs_  │
                              │    nodes     │
                              └──────────────┘

After (Target):
┌──────────────┐    gRPC      ┌──────────────┐
│  TypeScript  │ ───────────> │ Python gRPC  │
│  gRPC Client │   HTTP/2      │   Server     │
└──────────────┘               └──────────────┘
                                    │
                                    │ wraps
                                    ▼
                              ┌──────────────┐
                              │ bubblelabs_  │
                              │    nodes     │
                              └──────────────┘

Migration Path (Backward Compatible):
┌──────────────┐     REST      ┌─────────────┐     gRPC      ┌──────────────┐
│  TypeScript  │ ────────────> │    REST     │ ───────────> │ Python gRPC  │
│   Client     │               │   Bridge    │              │   Server     │
└──────────────┘               └─────────────┘              └──────────────┘
```

## Migration Phases

### Phase 1: Deploy Infrastructure (Day 1)

**Goal:** Run gRPC server alongside existing REST API

#### Step 1.1: Install gRPC Server

```bash
cd bubblelab/integrations/openevolve-grpc/python
pip install -r requirements.txt
```

#### Step 1.2: Start gRPC Server

```bash
# Terminal 1: Start gRPC server (port 50051)
python server.py

# Terminal 2: Start REST bridge (port 8001)
python rest_bridge.py

# Terminal 3: Keep existing REST API running (port 8000)
python bubblelabs_nodes/api_server.py
```

#### Step 1.3: Verify Health

```bash
# Check gRPC health
curl http://localhost:8001/health

# Should return:
# {
#   "status": "healthy",
#   "services": {
#     "grpc": { "status": "healthy", "response_time_ms": 12 }
#   }
# }
```

### Phase 2: Point Traffic to Bridge (Day 2-3)

**Goal:** Route existing REST calls through the bridge to gRPC

#### Step 2.1: Update Environment Variables

```bash
# Before
API_URL=http://localhost:8000

# After (point to bridge)
API_URL=http://localhost:8001
```

#### Step 2.2: Test Existing Code

Your existing TypeScript code should work unchanged:

```typescript
// This code continues to work without changes
const response = await fetch('http://localhost:8001/api/integrations/decomposition/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    executionId: 'exec-123',
    inputs: { problem_statement: '...' },
    options: { timeout: 300 }
  })
});
```

#### Step 2.3: Monitor for Issues

Watch logs for:
- Translation errors between REST and gRPC
- Performance differences
- Any changed behavior

### Phase 3: New Code Uses gRPC Directly (Week 1+)

**Goal:** New features use gRPC client for better performance

#### Step 3.1: Install TypeScript gRPC Client

```bash
cd bubblelab/integrations/openevolve-grpc/typescript
npm install
npm run build
```

#### Step 3.2: Use in New Code

```typescript
// New code uses gRPC directly
import { createGRPCClient } from '@openevolve/grpc-client';

const client = createGRPCClient({
  host: 'localhost',
  port: 50051
});

await client.connect();

// List nodes
const nodes = await client.listNodes();

// Execute with streaming
await client.executeNodeStreaming(
  {
    nodeType: 'gauntlet',
    inputs: { target: 'neural_network' }
  },
  (progress) => {
    console.log(`${progress.percent}%: ${progress.message}`);
  }
);
```

#### Step 3.3: Gradually Migrate Existing Code

Replace REST calls incrementally:

```typescript
// BEFORE (REST)
const response = await fetch(`${API_URL}/api/integrations/${nodeType}/execute`, {
  method: 'POST',
  body: JSON.stringify({ inputs, options })
});
const result = await response.json();

// AFTER (gRPC)
const result = await grpcClient.executeNode({
  nodeType,
  inputs,
  options: { enableStreaming: false }
});
```

### Phase 4: Remove REST Bridge (Month 2+)

**Goal:** All code uses gRPC, remove bridge

```bash
# Stop REST bridge
kill $(pgrep -f rest_bridge.py)

# Update TypeScript client to use gRPC directly
# Remove REST fallback code
```

## API Mapping

### REST to gRPC Method Mapping

| REST Endpoint | gRPC Method | Notes |
|---------------|-------------|-------|
| `GET /health` | Health.Check | Health check |
| `GET /api/integrations` | NodeRegistry.ListNodes | List all nodes |
| `GET /api/integrations/{type}` | NodeRegistry.GetNodeSchema | Get node info |
| `POST /api/integrations/{type}/execute` | NodeRegistry.ExecuteNode | Execute node |
| `POST /api/integrations/{type}/execute` (stream) | NodeRegistry.ExecuteNodeStreaming | Stream execution |
| `GET /executions/{id}` | NodeRegistry.GetExecutionStatus | Get status |
| `POST /cancel` | NodeRegistry.CancelExecution | Cancel execution |

### Request/Response Mapping

#### List Nodes

```json
// REST Request
GET /api/integrations?category=analysis

// gRPC Request
{
  "metadata": { "request_id": "..." },
  "category": "analysis"
}
```

```json
// REST Response
{
  "nodes": [
    {
      "node_id": "decomposition",
      "display_name": "Decomposition",
      "category": "analysis"
    }
  ]
}

// gRPC Response
{
  "nodes": [
    {
      "node_id": "decomposition",
      "display_name": "Decomposition",
      "category": "analysis"
    }
  ]
}
```

#### Execute Node

```json
// REST Request
POST /api/integrations/decomposition/execute
{
  "executionId": "exec-123",
  "inputs": {
    "problem_statement": "Design ML pipeline"
  },
  "options": {
    "timeout": 300
  }
}

// gRPC Request
{
  "metadata": { "request_id": "exec-123" },
  "node_type": "decomposition",
  "inputs": {
    "problem_statement": "Design ML pipeline"
  },
  "options": {
    "timeout_seconds": 300
  }
}
```

## Common Issues & Solutions

### Issue 1: Connection Refused

**Symptom:** `Error: 14 UNAVAILABLE: Connection refused`

**Solution:**
```bash
# Check if gRPC server is running
curl http://localhost:8001/health

# Start server if needed
python python/server.py
```

### Issue 2: Type Mismatch

**Symptom:** `Error: 3 INVALID_ARGUMENT: Type mismatch`

**Cause:** gRPC is more strict about types than REST

**Solution:** Ensure all fields match the protobuf schema exactly:

```typescript
// Wrong - missing required field
await client.executeNode({
  nodeType: 'decomposition'
  // missing inputs!
});

// Correct
await client.executeNode({
  nodeType: 'decomposition',
  inputs: { problem_statement: '...' }  // required
});
```

### Issue 3: Streaming Not Working

**Symptom:** No progress updates received

**Solution:**
```typescript
// Ensure streaming is enabled
await client.executeNodeStreaming(
  {
    nodeType: 'gauntlet',
    inputs: { ... }
  },
  (progress) => console.log(progress)  // This callback is required!
);
```

### Issue 4: Performance Worse Than REST

**Symptom:** gRPC slower than expected

**Causes & Solutions:**
1. **Connection not reused**: Ensure client.connect() is called once and reused
2. **No compression**: Enable compression in config
3. **Wrong load balancing**: Use health-based strategy

```typescript
const client = createGRPCClient({
  compression: CompressionAlgorithms.gzip,
  poolSize: 5  // Connection pooling
});
```

## Performance Comparison

### Before (REST)

```typescript
// 10 sequential requests
for (let i = 0; i < 10; i++) {
  await fetch('http://localhost:8000/api/integrations/decomposition/execute', {
    method: 'POST',
    body: JSON.stringify({ inputs: { ... } })
  });
}
// Time: ~500ms (50ms per request)
```

### After (gRPC)

```typescript
// 10 sequential requests
for (let i = 0; i < 10; i++) {
  await client.executeNode({
    nodeType: 'decomposition',
    inputs: { ... }
  });
}
// Time: ~50ms (5ms per request)
```

### With Streaming (Real-time Progress)

```typescript
// REST: Polling (inefficient)
const poll = setInterval(async () => {
  const status = await fetch(`/executions/${id}`);
  if (status.completed) clearInterval(poll);
}, 1000);  // Poll every second

// gRPC: Native streaming
await client.executeNodeStreaming(
  { nodeType: 'gauntlet', inputs: { ... } },
  (progress) => updateUI(progress)  // Real-time!
);
```

## Testing Migration

### Test Script

```typescript
// test_migration.ts
import { createGRPCClient } from '@openevolve/grpc-client';

async function testMigration() {
  // Test REST (existing)
  const restResponse = await fetch('http://localhost:8001/api/integrations/decomposition/execute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      executionId: 'test-rest',
      inputs: { problem_statement: 'Test problem' },
      options: {}
    })
  });
  const restResult = await restResponse.json();
  console.log('REST Result:', restResult);

  // Test gRPC (new)
  const client = createGRPCClient();
  await client.connect();
  
  const grpcResult = await client.executeNode({
    nodeType: 'decomposition',
    inputs: { problem_statement: 'Test problem' }
  });
  console.log('gRPC Result:', grpcResult);

  // Compare
  console.log('Results match:', JSON.stringify(restResult) === JSON.stringify(grpcResult));
}

testMigration();
```

## Rollback Plan

If issues occur:

1. **Immediate**: Point back to old REST API
   ```bash
   # Change environment variable back
   API_URL=http://localhost:8000  # Old REST API
   ```

2. **Short-term**: Keep bridge running but disable gRPC
   ```bash
   # Bridge falls back to direct Python calls
   export BRIDGE_FALLBACK=true
   ```

3. **Long-term**: Fix issues and retry migration

## Checklist

- [ ] Phase 1: gRPC server deployed and healthy
- [ ] Phase 2: REST bridge tested and routing correctly
- [ ] Phase 3: New features using gRPC directly
- [ ] Phase 3: Existing code migration complete
- [ ] Phase 4: REST bridge removed
- [ ] Performance tests passing
- [ ] All integration tests passing

## Support

For issues during migration:

1. Check logs: `tail -f python/server.log`
2. Verify health: `curl http://localhost:8001/health`
3. Review this guide for common issues
4. File issue with logs and reproduction steps
