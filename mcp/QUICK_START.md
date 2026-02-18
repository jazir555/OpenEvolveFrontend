# Unified MCP Gateway - Quick Start Guide

## 30-Second Setup

```bash
# 1. Install dependencies
pip install -r mcp/requirements.txt

# 2. Start the gateway
python -m mcp.gateway.server

# 3. Test it
curl http://localhost:8080/health
```

## Tool Namespaces

| Namespace | Description | Example Tools |
|-----------|-------------|---------------|
| `kggen/` | Knowledge graph memory | `kggen/add_memories`, `kggen/retrieve_relevant_memories` |
| `graphiti/` | Knowledge graph | `graphiti/add_episode`, `graphiti/search_facts` |
| `openevolve/` | Evolutionary computation | `openevolve/run_evolution`, `openevolve/evolve_code` |
| `crewai/` | Orchestration | `crewai/delegate_task` |
| `roma/` | Recursive meta-agent | `roma/solve_recursive` |
| `ace/` | Learning engine | `ace/execute_task`, `ace/learn_from_execution` |

## Quick API Examples

### List All Tools
```bash
curl http://localhost:8080/api/tools
```

### Call a Tool
```bash
curl -X POST http://localhost:8080/api/tools/kggen/add_memories \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "kggen/add_memories",
    "parameters": {
      "text": "John works at OpenAI as a software engineer."
    }
  }'
```

### Get Tool Info
```bash
curl http://localhost:8080/api/tools/kggen/add_memories
```

### Filter by Namespace
```bash
curl "http://localhost:8080/api/tools?namespace=kggen"
```

### Get Analytics
```bash
curl http://localhost:8080/api/analytics
```

## Python Integration

### Basic Usage
```python
from mcp.gateway import UnifiedMCPGateway

async def main():
    # Initialize
    gateway = UnifiedMCPGateway()
    await gateway.initialize()

    # List tools
    tools = await gateway.list_tools()
    print(f"Available tools: {len(tools)}")

    # Call tool
    result = await gateway.call_tool(
        tool_name="kggen/add_memories",
        params={"text": "Sample text"}
    )

    print(f"Result: {result.success}")
    print(f"Output: {result.result}")

    # Shutdown
    await gateway.shutdown()

# Run
import asyncio
asyncio.run(main())
```

### With CrewAI
```python
from CrewAI.src.mcp.gateway_integration import (
    get_crewai_gateway,
    delegate_to_agent_with_mcp_tools
)

async def use_crewai():
    gateway = await get_crewai_gateway()

    result = await delegate_to_agent_with_mcp_tools(
        agent=my_agent,
        task="Solve this problem",
        tools_available=["kggen/retrieve_relevant_memories"]
    )

    return result
```

### With ROMA
```python
from ROMA.src.roma_dspy.mcp.gateway_client import (
    get_roma_gateway_client
)

async def use_roma():
    client = get_roma_gateway_client()

    # List tools
    tools = await client.list_tools(namespace="kggen")

    # Invoke tool
    result = await client.invoke_tool(
        tool_name="kggen/add_memories",
        params={"text": "Sample text"}
    )

    return result
```

## Docker Deployment

```bash
# Build
docker build -t mcp-gateway:latest -f mcp/gateway/Dockerfile .

# Run
docker run -d \
  --name mcp-gateway \
  -p 8080:8080 \
  -v $(pwd)/mcp/config/gateway.yaml:/app/config/gateway.yaml \
  mcp-gateway:latest

# Check logs
docker logs -f mcp-gateway

# Check health
curl http://localhost:8080/health
```

## Configuration

Edit `mcp/config/gateway.yaml`:

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  log_level: "INFO"

servers:
  kggen:
    enabled: true
    url: "http://localhost:8001"
    timeout: 30

routing:
  circuit_breaker_threshold: 5
  max_retries: 3

monitoring:
  metrics_enabled: true
  log_tool_calls: true
```

## Testing

```bash
# Run tests
pytest mcp/tests/test_gateway.py -v

# Run with coverage
pytest mcp/tests/test_gateway.py --cov=mcp/gateway --cov-report=html
```

## Troubleshooting

### Gateway won't start
```bash
# Check if port is in use
lsof -i :8080

# Use different port
# Edit gateway.yaml: port: 8081
```

### Tools not showing up
```bash
# Check health
curl http://localhost:8080/health

# Check logs
docker logs mcp-gateway

# Verify backend servers are running
curl http://localhost:8001/health  # kg-gen
```

### Circuit breaker tripping
```bash
# Check analytics
curl http://localhost:8080/api/analytics

# Increase threshold in config
# routing.circuit_breaker_threshold: 10
```

## Health Check

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
  "status": "running",
  "initialized": true,
  "servers": {
    "kggen": {
      "status": "online",
      "url": "http://localhost:8001"
    }
  },
  "tools": {
    "total_tools": 25
  }
}
```

## Analytics

```bash
curl http://localhost:8080/api/analytics
```

Shows:
- Popular tools
- Slowest tools
- Success rates
- Usage trends

## Key Files

| File | Purpose |
|------|---------|
| `mcp/gateway/unified_mcp_gateway.py` | Main gateway class |
| `mcp/gateway/server.py` | FastAPI HTTP server |
| `mcp/gateway/tool_registry.py` | Tool registration |
| `mcp/gateway/tool_router.py` | Routing & circuit breaking |
| `mcp/gateway/analytics.py` | Usage analytics |
| `mcp/config/gateway.yaml` | Configuration |
| `mcp/gateway/Dockerfile` | Docker deployment |
| `mcp/tests/test_gateway.py` | Test suite |

## Support

- Full Documentation: `mcp/gateway/README.md`
- Implementation Report: `PHASE7_MCP_GATEWAY_COMPLETE.md`
- Test Suite: `mcp/tests/test_gateway.py`
- Configuration: `mcp/config/gateway.yaml`

## Performance Tips

1. **Enable Caching** - Set `cache.enabled: true` in config
2. **Adjust Workers** - Set `gateway.max_workers: 16` for high load
3. **Timeout Tuning** - Reduce server timeouts for faster failure detection
4. **Load Balancing** - Use `round_robin` for even distribution

## Security Notes

For production:
1. Enable HTTPS/TLS
2. Add API key authentication
3. Configure rate limiting
4. Restrict CORS origins
5. Use secrets management

---

**Version:** 1.0.0
**Last Updated:** 2026-01-07
**Status:** Production Ready ✅
