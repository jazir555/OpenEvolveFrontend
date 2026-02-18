# Unified MCP Gateway

A centralized gateway that coordinates tools from multiple MCP (Model Context Protocol) servers including kg-gen, Graphiti, OpenEvolve, CrewAI, ROMA, and ACE into a single unified namespace for agent consumption.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Unified MCP Gateway                      │
│                                                               │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Tool Registry │  │ Tool Router  │  │    Analytics    │ │
│  └───────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
    ┌─────────┐      ┌─────────┐     ┌─────────┐
    │ kg-gen  │      │Graphiti │     │OpenEvolve│
    │ Server  │      │ Server  │     │  Server  │
    └─────────┘      └─────────┘     └─────────┘
         │                │                │
         └────────────────┴────────────────┘
                          │
                    ┌─────▼─────┐
                    │  Agents   │
                    │(CrewAI│
                    │   ROMA    │
                    │    ACE)   │
                    └───────────┘
```

## Features

### Core Capabilities

- **Unified Tool Namespace**: Access tools from all servers through a single API
- **Automatic Tool Discovery**: Dynamic registration of tools from all connected servers
- **Intelligent Routing**: Smart routing with load balancing and circuit breaking
- **Fallback & Retry**: Automatic fallback to alternative servers on failure
- **Performance Analytics**: Track tool usage, success rates, and performance
- **Health Monitoring**: Real-time health checks for all servers

### Advanced Features

- **Circuit Breaking**: Prevent cascading failures by tripping circuits on failing servers
- **Load Balancing**: Distribute load across multiple server instances
- **Tool Categorization**: Organize tools by category (knowledge, evolution, learning, etc.)
- **Version Management**: Track tool versions and deprecations
- **Caching**: Cache tool results for improved performance

## Installation

### Local Development

```bash
# Clone the repository
cd /path/to/OpenEvolve/Frontend

# Install dependencies
pip install -r mcp/requirements.txt

# Start the gateway
python -m mcp.gateway.server
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t mcp-gateway:latest -f mcp/gateway/Dockerfile .

# Run the gateway
docker run -d \
  --name mcp-gateway \
  -p 8080:8080 \
  -v $(pwd)/mcp/config/gateway.yaml:/app/config/gateway.yaml \
  mcp-gateway:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  mcp-gateway:
    build:
      context: .
      dockerfile: mcp/gateway/Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - ./mcp/config/gateway.yaml:/app/config/gateway.yaml
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

## Configuration

The gateway is configured via `mcp/config/gateway.yaml`:

```yaml
gateway:
  host: "0.0.0.0"
  port: 8080
  log_level: "INFO"
  max_workers: 10
  request_timeout: 120

servers:
  kggen:
    enabled: true
    url: "http://localhost:8001"
    timeout: 30
    namespace: "kggen"

  graphiti:
    enabled: true
    url: "http://localhost:8002"
    timeout: 30
    namespace: "graphiti"

tool_registry:
  categorization_enabled: true
  versioning_enabled: true

routing:
  load_balancing: "round_robin"
  circuit_breaker_threshold: 5
  circuit_breaker_timeout: 60
  fallback_enabled: true
  max_retries: 3

monitoring:
  metrics_enabled: true
  log_tool_calls: true
```

## Tool Namespaces

Tools are organized into namespaces for clarity:

- **kggen/**: Knowledge graph memory tools (kg-gen)
  - `kggen/add_memories`
  - `kggen/retrieve_relevant_memories`
  - `kggen/visualize_memories`
  - `kggen/get_memory_stats`

- **graphiti/**: Knowledge graph tools (Graphiti)
  - `graphiti/add_episode`
  - `graphiti/search_nodes`
  - `graphiti/search_facts`

- **openevolve/**: Evolutionary computation tools (OpenEvolve)
  - `openevolve/run_evolution`
  - `openevolve/evolve_code`
  - `openevolve/evolve_function`

- **crewai/**: Orchestration tools (CrewAI)
  - `crewai/delegate_task`
  - `crewai/coordinate_agents`

- **roma/**: Recursive meta-agent tools (ROMA)
  - `roma/solve_recursive`
  - `roma/decompose_task`

- **ace/**: Learning tools (ACE)
  - `ace/initialize_agent`
  - `ace/execute_task`
  - `ace/learn_from_execution`

## API Usage

### List All Tools

```bash
curl http://localhost:8080/api/tools
```

Response:
```json
{
  "tools": [
    {
      "name": "add_memories",
      "namespace": "kggen",
      "description": "Extract and store memories from text",
      "parameters": {
        "type": "object",
        "properties": {
          "text": {"type": "string"}
        }
      }
    }
  ],
  "total_count": 1,
  "namespaces": ["kggen"]
}
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

Response:
```json
{
  "success": true,
  "tool_name": "add_memories",
  "namespace": "kggen",
  "server_name": "kggen",
  "result": {
    "message": "Extracted 2 entities: John, OpenAI"
  },
  "execution_time": 0.523,
  "timestamp": "2026-01-07T12:00:00Z"
}
```

### Filter by Namespace

```bash
curl "http://localhost:8080/api/tools?namespace=kggen"
```

### Filter by Category

```bash
curl "http://localhost:8080/api/tools?category=knowledge"
```

## Integration with CrewAI

```python
from CrewAI.src.mcp.gateway_integration import (
    get_crewai_gateway,
    delegate_to_agent_with_mcp_tools
)

# Initialize gateway
gateway = await get_crewai_gateway()

# Delegate task with MCP tools
result = await delegate_to_agent_with_mcp_tools(
    agent=my_agent,
    task="Solve this problem using available knowledge tools",
    tools_available=["kggen/retrieve_relevant_memories"]
)
```

## Integration with ROMA

```python
from ROMA.src.roma_dspy.mcp.gateway_client import (
    get_roma_gateway_client
)

# Get client
client = get_roma_gateway_client()

# List tools
tools = await client.list_tools(namespace="kggen")

# Invoke tool
result = await client.invoke_tool(
    tool_name="kggen/add_memories",
    params={"text": "Sample text"}
)
```

## Analytics

The gateway provides detailed analytics on tool usage:

```bash
curl http://localhost:8080/api/analytics
```

Response includes:
- Popular tools (most used)
- Slowest tools (highest execution time)
- Least reliable tools (lowest success rate)
- Server statistics
- Usage trends over time

## Health Monitoring

Check gateway health:

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "running",
  "initialized": true,
  "servers": {
    "kggen": {
      "status": "online",
      "url": "http://localhost:8001",
      "enabled": true
    }
  },
  "tools": {
    "total_tools": 25,
    "namespaces": 6,
    "categories": 5
  }
}
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install -r mcp/requirements.txt

# Run tests
pytest mcp/tests/test_gateway.py -v

# Run with coverage
pytest mcp/tests/test_gateway.py --cov=mcp/gateway --cov-report=html
```

## Troubleshooting

### Gateway fails to start

1. Check if port 8080 is available
2. Verify configuration file exists
3. Check logs: `docker logs mcp-gateway`

### Tools not appearing

1. Verify backend servers are running
2. Check server URLs in configuration
3. Review health check endpoint
4. Check gateway logs for connection errors

### Circuit breaker keeps tripping

1. Increase `circuit_breaker_threshold` in config
2. Check backend server health
3. Review server logs for errors
4. Consider scaling backend servers

### Slow tool execution

1. Check analytics for slow tools
2. Review server timeout settings
3. Consider enabling caching
4. Check network latency

## Performance Optimization

1. **Enable Caching**: Cache frequently used tool results
2. **Load Balancing**: Use round-robin or least-connections
3. **Connection Pooling**: Reuse HTTP connections
4. **Timeout Tuning**: Adjust timeouts based on tool characteristics
5. **Circuit Breakers**: Prevent cascading failures

## Security Considerations

1. **API Keys**: Add authentication for production
2. **Rate Limiting**: Configure appropriate rate limits
3. **Input Validation**: Validate all tool parameters
4. **CORS**: Configure CORS for production domains
5. **HTTPS**: Use TLS in production

## Contributing

When adding new MCP servers:

1. Create wrapper in `mcp/servers/{server}_mcp_wrapper.py`
2. Register tools in gateway initialization
3. Add server configuration to `gateway.yaml`
4. Update documentation
5. Add tests for new tools

## License

This project is part of OpenEvolve and follows the same license.

## Support

For issues and questions:
- GitHub Issues: [OpenEvolve Issues]
- Documentation: [OpenEvolve Docs]
- Discord/Slack: [Community Channels]
