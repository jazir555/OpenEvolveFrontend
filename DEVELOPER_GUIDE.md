# OpenEvolve Developer Guide

**Integration System Development Guide** | **License: Apache 2.0**

---

## 🎯 Overview

This guide helps developers understand, extend, and contribute to the OpenEvolve Integration System.

---

## 📚 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Component Guide](#component-guide)
5. [Testing](#testing)
6. [Deployment](#deployment)
7. [Best Practices](#best-practices)

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│  (Claude/Cursor, Web UI, API Clients)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway (Port 80)                        │
│              (Rate Limiting, Routing, Auth)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   REST API   │    │  GraphQL API │    │   WebSocket  │
│  (Port 8000) │    │  (Port 8001) │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Service Orchestrator                           │
│              (Lifecycle, Health, Dependencies)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Event Bus   │    │    MCP       │    │  Telemetry   │
│  (Valkey)    │    │   Server     │    │(OpenTelemetry│
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Stage 6 Knowledge Extraction                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Prerequisites

- Python 3.11+ (3.10+ supported)
- Git
- Docker (optional, for full stack)
- Make (optional, for convenience)

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd openevolve

# Automated setup
python setup_integration.py --dev

# Or manual setup
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements_integration.txt
pip install -r requirements_with_testing.txt

# Verify installation
python verify_integration.py --full
```

### Project Structure

```
openevolve/
├── Core Integration
│   ├── unified_mcp_server.py          # MCP server
│   ├── event_bus.py                   # Messaging
│   ├── graphql_server.py              # GraphQL API
│   ├── service_orchestrator.py        # Service management
│   ├── plugin_registry.py             # Plugin system
│   ├── stage6_knowledge_extraction.py # Knowledge extraction
│   ├── integration_config.py          # Configuration
│   ├── telemetry.py                   # Observability
│   ├── api_gateway.py                 # API gateway
│   └── openevolve_cli.py              # CLI tool
│
├── Operations
│   ├── docker-compose.yml             # Docker stack
│   ├── Makefile                       # Build commands
│   ├── backup_restore.py              # Backup utility
│   └── run_integration_tests.py       # Test runner
│
├── Utilities
│   ├── setup_integration.py           # Setup script
│   ├── demo_integration.py            # Demo showcase
│   ├── verify_integration.py          # Verification
│   ├── system_health.py               # Health check
│   └── benchmark_integrations.py      # Benchmarks
│
├── Tests
│   └── test_integrations_comprehensive.py
│
└── Documentation
    ├── INTEGRATION_GUIDE.md
    ├── DEVELOPER_GUIDE.md
    └── README_INTEGRATION.md
```

---

## Development Workflow

### 1. Start Development Server

```bash
# Start all services
make dev-start

# Or start specific services
python -m openevolve_cli services start --rest --graphql
```

### 2. Make Changes

Edit files in your IDE. The services support hot-reload in development mode.

### 3. Run Tests

```bash
# Run all tests
make test

# Run specific test
pytest test_integrations_comprehensive.py::TestStage6KnowledgeExtraction -v

# Run with coverage
make test-coverage
```

### 4. Check Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type check
make mypy

# Security scan
make security
```

### 5. Verify Integration

```bash
# Run full verification
python verify_integration.py --full

# Run system integration test
python test_full_system_integration.py
```

---

## Component Guide

### Unified MCP Server

**File**: `unified_mcp_server.py`

**Purpose**: Provides unified Model Context Protocol server for Claude/Cursor integration.

**Key Features**:
- 25+ organized tools across 5 categories
- Health monitoring
- Graceful degradation

**Adding a New Tool**:

```python
@mcp.tool()
async def my_custom_tool(
    param1: str,
    param2: int,
    ctx: Context
) -> str:
    """
    Description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Result description
    """
    try:
        # Implementation
        result = await process(param1, param2)
        return json.dumps({"status": "success", "result": result})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
```

### Event Bus

**File**: `event_bus.py`

**Purpose**: Provides event-driven messaging using Valkey.

**Usage**:

```python
from event_bus import WorkflowEventBus, WorkflowEvent, EventType

# Initialize
bus = WorkflowEventBus()
await bus.connect()

# Subscribe
async def handler(event):
    print(f"Received: {event.type}")

await bus.subscribe("my_channel", handler)

# Publish
event = WorkflowEvent(
    id="evt_001",
    type=EventType.WORKFLOW_STARTED,
    payload={"data": "value"},
    timestamp=datetime.now(),
    priority=1
)

await bus.publish("my_channel", event)
```

### Stage 6 Knowledge Extraction

**File**: `stage6_knowledge_extraction.py`

**Purpose**: Extracts patterns and generates artifacts from workflow execution.

**Usage**:

```python
from stage6_knowledge_extraction import (
    Stage6KnowledgeExtraction,
    ExecutionTrace
)

# Initialize
engine = Stage6KnowledgeExtraction()

# Process trace
result = await engine.process_trace(trace)

# Get applicable knowledge
artifacts = engine.get_applicable_artifacts("problem description")
```

**Adding Custom Pattern Extractor**:

```python
class MyPatternExtractor:
    def extract_patterns(self, traces: List[ExecutionTrace]) -> List[ExtractedPattern]:
        patterns = []
        # Implementation
        return patterns
```

### Plugin System

**File**: `plugin_registry.py`

**Purpose**: Dynamic plugin loading and management.

**Creating a Plugin**:

```python
from plugin_registry import IntegrationPlugin, PluginMetadata, PluginType

class MyPlugin(IntegrationPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
            author="Developer",
            license="Apache-2.0",
            plugin_type=PluginType.MCP_TOOL,
            capabilities=[]
        )
    
    async def initialize(self, config: dict) -> bool:
        # Initialization logic
        return True
    
    async def shutdown(self) -> bool:
        # Cleanup logic
        return True
```

---

## Testing

### Test Organization

```
tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
├── e2e/               # End-to-end tests
└── benchmarks/        # Performance tests
```

### Writing Tests

```python
import pytest
from stage6_knowledge_extraction import Stage6KnowledgeExtraction

@pytest.mark.asyncio
async def test_knowledge_extraction():
    """Test knowledge extraction."""
    engine = Stage6KnowledgeExtraction()
    
    # Create test trace
    trace = ExecutionTrace(...)
    
    # Process
    result = await engine.process_trace(trace)
    
    # Assert
    assert result['patterns_extracted'] >= 0
```

### Test Markers

- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.e2e` - End-to-end tests

---

## Deployment

### Local Deployment

```bash
# Using deployment script
deploy/deploy_local.sh full

# Or using Make
make start
```

### Docker Deployment

```bash
# Using deployment script
deploy/deploy_docker.sh full

# Or using Make
make docker-up
```

### Production Checklist

- [ ] Update `.env` with production values
- [ ] Change `SECRET_KEY`
- [ ] Set `DEBUG=false`
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS
- [ ] Configure backup schedule
- [ ] Set up monitoring alerts
- [ ] Run full test suite
- [ ] Verify health checks

---

## Best Practices

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Keep functions small
- Use async/await for I/O

### Error Handling

```python
try:
    result = await operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Handle specific error
except Exception as e:
    logger.exception("Unexpected error")
    # Handle generic error
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# Good logging
logger.info("Processing workflow: %s", workflow_id)
logger.debug("Debug data: %s", data)
logger.error("Failed to process: %s", error, exc_info=True)
```

### Configuration

```python
from integration_config import get_config

config = get_config()

# Use config values
port = config.rest_api.port
log_level = config.log_level
```

### Testing

- Write tests before code (TDD)
- Use fixtures for setup
- Mock external dependencies
- Test edge cases
- Keep tests fast

---

## Debugging

### Enable Debug Mode

```bash
export OPENEVOLVE_LOG_LEVEL=DEBUG
make dev-start
```

### View Logs

```bash
# View all logs
make logs

# View specific service
python -m openevolve_cli services logs <service_name>

# Docker logs
make docker-logs
```

### Common Issues

**Issue**: Services fail to start
```bash
# Check for port conflicts
lsof -i :8000

# Check logs
python system_health.py
```

**Issue**: Import errors
```bash
# Reinstall dependencies
pip install -r requirements_integration.txt --force-reinstall
```

**Issue**: Event bus connection failed
```bash
# Check Valkey
redis-cli ping

# Restart event bus
python -m openevolve_cli services restart event_bus
```

---

## Contributing

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run full test suite
5. Submit pull request

### Code Review Checklist

- [ ] Tests pass
- [ ] Code formatted (Black)
- [ ] Linting passes (Flake8)
- [ ] Type hints added
- [ ] Docstrings written
- [ ] No security issues (Bandit)

---

## Resources

- **Documentation**: `INTEGRATION_GUIDE.md`
- **API Docs**: http://localhost:8000/docs
- **GraphQL Playground**: http://localhost:8001/graphql
- **Issues**: GitHub Issues

---

**Happy Coding!** 🚀
