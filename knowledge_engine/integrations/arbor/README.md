# Arbor Integration for Knowledge Engine

Real-time code intelligence integration between [Arbor](https://github.com/Anandb71/arbor) and OpenEvolve Knowledge Engine.

## Overview

Arbor is a Rust-based code graph intelligence layer using Tree-sitter. This integration enables:

- **Code Graph Ingestion**: Parse codebases into navigable knowledge graphs
- **Real-time Sync**: Incremental updates as code changes
- **AI Agent Tools**: MCP tools for code navigation and analysis
- **Impact Analysis**: Understand what code would be affected by changes

## Architecture

```
┌─────────────────┐     WebSocket      ┌─────────────────┐
│   Arbor Server  │◄──────────────────►│   ArborClient   │
│   (Rust)        │    JSON-RPC        │   (Python)      │
└─────────────────┘                    └────────┬────────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
          ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
          │  GraphAdapter   │      │   SyncManager   │      │  ArborMCPBridge │
          │  (Schema Map)   │      │  (Incremental)  │      │  (AI Tools)     │
          └────────┬────────┘      └─────────────────┘      └─────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  KnowledgeGraph │
          └─────────────────┘
```

## Quick Start

### 1. Start Arbor Server

```bash
cd arbor/
cargo run --release
```

Arbor will start on `ws://localhost:7433` by default.

### 2. Use the Integration

```python
import asyncio
from knowledge_engine.integrations.arbor import (
    ArborClient, ArborConfig, ArborMCPBridge
)

async def main():
    # Connect to Arbor
    config = ArborConfig.from_env()
    client = ArborClient(config)
    await client.connect()
    
    # Use MCP tools for AI agents
    mcp = ArborMCPBridge(client, config.mcp)
    
    # Find code definitions
    result = await mcp.execute_tool(
        "arbor_find_definition",
        {"symbol": "ArborClient"}
    )
    
    if result.success:
        print(f"Found: {result.data['name']} at {result.data['file']}")
    
    await client.disconnect()

asyncio.run(main())
```

## MCP Tools

The MCP bridge exposes these tools to AI agents:

| Tool | Description |
|------|-------------|
| `arbor_find_definition` | Find where a symbol is defined |
| `arbor_get_callers` | Find functions that call a given function |
| `arbor_get_callees` | Find functions called by a given function |
| `arbor_find_path` | Find logic flow between two components |
| `arbor_analyze_impact` | Analyze impact of changing a symbol |
| `arbor_get_context` | Get contextual code information |
| `arbor_search` | Search for code by name or content |

## Configuration

Set via environment variables:

```bash
export ARBOR_WS_URL="ws://localhost:7433"
export ARBOR_ENABLED="true"
export ARBOR_DEBUG="false"
export ARBOR_SYNC_MODE="realtime"
export ARBOR_BATCH_SIZE="1000"
```

Or programmatically:

```python
from knowledge_engine.integrations.arbor import ArborConfig

config = ArborConfig(
    connection=ArborConnectionConfig(
        ws_url="ws://arbor-server:7433",
        reconnect_interval=5.0
    ),
    sync=ArborSyncConfig(
        mode="realtime",
        batch_size=1000
    )
)
```

## Examples

### Run the Demo

```bash
python -m knowledge_engine.integrations.arbor.examples.mcp_demo
```

### Basic Usage

```python
from knowledge_engine.integrations.arbor import (
    ArborClient, ArborConfig, ArborGraphAdapter
)

# Connect and query
config = ArborConfig.from_env()
client = ArborClient(config)
await client.connect()

# Get full graph
graph = await client.export_graph()
print(f"Graph has {len(graph['nodes'])} nodes, {len(graph['edges'])} edges")

# Import to Knowledge Graph
adapter = ArborGraphAdapter(knowledge_graph)
result = await adapter.import_graph(graph)
print(f"Imported {result.nodes_imported} nodes")

await client.disconnect()
```

## Phases

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Foundation (Client, Config, Health) | ✓ Complete |
| 2 | Graph Bridge (Adapter, Sync) | ✓ Complete |
| 3 | Intelligence (Query, Context, Impact) | ✓ Complete |
| 4 | MCP Tools (AI Agent Integration) | ✓ Complete |
| 5 | Visualization | Future |

## Testing

```bash
# Run unit tests
pytest knowledge_engine/integrations/arbor/tests/

# Run integration tests (requires Arbor server)
pytest knowledge_engine/integrations/arbor/tests/test_integration.py -v

# Run MCP demo
python -m knowledge_engine.integrations.arbor.examples.mcp_demo
```

## License

Same as OpenEvolve Knowledge Engine (AGPL-3.0)
