# OpenEvolve Integration Guide

**Version**: 1.0.0  
**License**: Apache 2.0  
**Last Updated**: 2026-02-02

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [Integration Components](#integration-components)
5. [Configuration](#configuration)
6. [Deployment](#deployment)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The OpenEvolve Integration System provides a unified platform for orchestrating decomposition, recomposition, and evolutionary optimization workflows. It consolidates 24 scattered MCP files into a cohesive architecture with REST API, GraphQL, event-driven messaging, and observability.

### Key Features

- **Unified MCP Server**: Single entry point for Claude/Cursor
- **REST API**: FastAPI-based with auto-generated docs
- **GraphQL API**: Strawberry-based with subscriptions
- **Event Bus**: Valkey-based messaging (Apache 2.0)
- **OpenTelemetry**: Distributed tracing and metrics
- **Stage 6 Knowledge Extraction**: Pattern recognition and artifact generation
- **Plugin System**: Dynamic extensibility
- **CLI Management**: Complete command-line interface

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway (Port 80)                     │
│                    (Rate Limiting, Auth, Routing)                │
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
│              (Lifecycle Management, Health Checks)               │
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
│         (Pattern Recognition, Artifact Generation)               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core OpenEvolve Engine                        │
│      (Decomposition, Recomposition, Evolution, Z3, Lean)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Valkey (Redis alternative) - Apache 2.0 licensed
- Git

### Installation

```bash
# Clone repository
git clone <repository-url>
cd openevolve

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements_integration.txt

# Start Valkey (if using Docker)
docker run -d --name valkey -p 6379:6379 valkey/valkey:latest
```

### Start Services

```bash
# Start all services
python -m openevolve_cli services start --all

# Or start individually
python -m openevolve_cli services start --rest --graphql

# Check status
python -m openevolve_cli services status
```

### Test Installation

```bash
# Health check
curl http://localhost:8000/health

# Test REST API
curl http://localhost:8000/api/v1/workflows

# Test GraphQL
curl -X POST http://localhost:8001/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __typename }"}'
```

---

## Integration Components

### 1. Unified MCP Server (`unified_mcp_server.py`)

Consolidates 24 scattered MCP files into a single server.

**Features**:
- 25+ organized tools across 5 categories
- Health monitoring
- Graceful degradation
- Claude/Cursor integration

**Usage**:
```python
from unified_mcp_server import UnifiedMCPServer

server = UnifiedMCPServer()
await server.start()
```

### 2. Event Bus (`event_bus.py`)

Apache 2.0 compliant messaging using Valkey.

**Features**:
- Pub/sub messaging
- Priority queues
- Event persistence
- Dead letter queues

**Usage**:
```python
from event_bus import WorkflowEventBus

bus = WorkflowEventBus()
await bus.publish_workflow_started("wf_001", "Test problem")
```

### 3. GraphQL API (`graphql_server.py`)

Strawberry-based GraphQL (MIT License).

**Features**:
- Queries, mutations, subscriptions
- Real-time workflow updates
- Flexible data fetching

**Usage**:
```python
from graphql_server import create_graphql_app

app = create_graphql_app()
```

### 4. REST API (`api_server.py`)

FastAPI-based REST API (MIT License).

**Features**:
- Auto-generated documentation
- Request validation
- Error handling

**Endpoints**:
- `GET /health` - Health check
- `POST /api/v1/workflows` - Create workflow
- `GET /api/v1/workflows/{id}` - Get workflow
- `POST /api/v1/decompose` - Decompose problem

### 5. OpenTelemetry Integration (`telemetry.py`)

Distributed tracing and metrics (Apache 2.0).

**Features**:
- Automatic instrumentation
- Custom spans
- Metrics collection
- OTLP export

**Usage**:
```python
from telemetry import WorkflowTracer

tracer = WorkflowTracer(config)
result = await tracer.trace_decomposition(
    "problem_001",
    decompose_function,
    problem_data
)
```

### 6. Stage 6 Knowledge Extraction (`stage6_knowledge_extraction.py`)

Advanced pattern recognition and artifact generation.

**Features**:
- Sequence pattern extraction
- Semantic clustering
- Parametric pattern detection
- Structural analysis
- Knowledge artifact generation

**Usage**:
```python
from stage6_knowledge_extraction import (
    Stage6KnowledgeExtraction,
    ExecutionTrace
)

engine = Stage6KnowledgeExtraction()
result = await engine.process_trace(trace)
artifacts = engine.get_applicable_artifacts("problem description")
```

### 7. Service Orchestrator (`service_orchestrator.py`)

Manages service lifecycle and health.

**Features**:
- Dependency management
- Health monitoring
- Graceful startup/shutdown
- REST API for management

**Usage**:
```python
from service_orchestrator import ServiceOrchestrator

orchestrator = ServiceOrchestrator()
orchestrator.register_service(name="api", start_func=start_api)
await orchestrator.start_all()
```

### 8. Plugin Registry (`plugin_registry.py`)

Dynamic plugin loading system.

**Features**:
- File-based discovery
- Module loading
- Lifecycle management
- Capability registration

**Usage**:
```python
from plugin_registry import PluginRegistry

registry = PluginRegistry()
await registry.load_from_directory("./plugins")
```

### 9. API Gateway (`api_gateway.py`)

Unified entry point for all APIs.

**Features**:
- Request routing
- Rate limiting
- CORS handling
- Health aggregation

### 10. CLI Tool (`openevolve_cli.py`)

Complete management CLI.

**Commands**:
```bash
# Services
openevolve services start --all
openevolve services stop
openevolve services status
openevolve services health

# Plugins
openevolve plugins list
openevolve plugins load <path>

# Config
openevolve config show
openevolve config validate

# Docker
openevolve docker generate
```

---

## Configuration

### Configuration Files

**`integration_config.yaml`**:
```yaml
log_level: INFO
orchestrator_port: 8080

services:
  rest_api: true
  graphql_api: true
  event_bus: true
  mcp_server: true
  telemetry: true

rest_api:
  host: 0.0.0.0
  port: 8000

graphql:
  host: 0.0.0.0
  port: 8001
  enable_playground: true

event_bus:
  enabled: true
  backend: valkey
  host: localhost
  port: 6379

telemetry:
  enabled: true
  service_name: openevolve
  otlp_endpoint: http://localhost:4317
```

### Environment Variables

```bash
# Required
OPENEVOLVE_LOG_LEVEL=INFO
OPENEVOLVE_ORCHESTRATOR_PORT=8080

# Services
OPENEVOLVE_REST_API__PORT=8000
OPENEVOLVE_GRAPHQL__PORT=8001

# Event Bus
OPENEVOLVE_EVENT_BUS__HOST=localhost
OPENEVOLVE_EVENT_BUS__PORT=6379

# Telemetry
OPENEVOLVE_TELEMETRY__ENABLED=true
OPENEVOLVE_TELEMETRY__OTLP_ENDPOINT=http://localhost:4317
```

---

## Deployment

### Docker Compose

Generated by `openevolve docker generate`:

```yaml
version: '3.8'

services:
  valkey:
    image: valkey/valkey:latest
    ports:
      - "6379:6379"
    volumes:
      - valkey_data:/data

  openevolve:
    build: .
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8080:8080"
    environment:
      - VALKEY_HOST=valkey
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
    depends_on:
      - valkey

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"
      - "4317:4317"

volumes:
  valkey_data:
```

### Production Deployment

```bash
# Build image
docker build -t openevolve:latest .

# Deploy stack
docker-compose up -d

# Verify
docker-compose ps
docker-compose logs -f
```

---

## Monitoring

### BubbleLab UI Dashboard

```bash
BubbleLab UI run monitoring_dashboard.py
```

Access at: http://localhost:8501

### Health Checks

```bash
# Overall health
curl http://localhost:8080/health

# Service health
curl http://localhost:8000/health  # REST
curl http://localhost:8001/health  # GraphQL
```

### Metrics

Prometheus-compatible endpoint:
```
http://localhost:8080/metrics
```

### OpenTelemetry

View traces in Jaeger: http://localhost:16686

---

## Troubleshooting

### Common Issues

**Issue**: Services fail to start
```bash
# Check logs
python -m openevolve_cli services status

# Verify dependencies
pip check

# Check ports
lsof -i :8000
lsof -i :8001
```

**Issue**: Event bus connection failed
```bash
# Verify Valkey is running
docker ps | grep valkey
redis-cli -p 6379 ping
```

**Issue**: Import errors
```bash
# Reinstall dependencies
pip install -r requirements_integration.txt --force-reinstall
```

### Debug Mode

```bash
# Enable debug logging
export OPENEVOLVE_LOG_LEVEL=DEBUG
python -m openevolve_cli services start --all --verbose
```

### Getting Help

1. Check logs: `docker-compose logs`
2. Run diagnostics: `python -m openevolve_cli status`
3. Verify installation: `pytest test_integrations_comprehensive.py`

---

## License Compliance

All integration components use permissive licenses:

| Component | License |
|-----------|---------|
| FastAPI | MIT |
| Strawberry GraphQL | MIT |
| Valkey | Apache 2.0 |
| OpenTelemetry | Apache 2.0 |
| MCP | MIT |
| Pydantic | MIT |
| scikit-learn | BSD |
| NetworkX | BSD |

**No GPL/AGPL dependencies** are included.

---

## Migration Guide

### From Scattered MCP Files

```bash
# 1. Analyze existing files
python migrate_to_unified_mcp.py --analyze

# 2. Create backup
python migrate_to_unified_mcp.py --backup-old

# 3. Generate config
python migrate_to_unified_mcp.py --generate-config

# 4. Start unified server
python -m openevolve_cli services start --mcp
```

---

## API Reference

See individual component documentation:

- [REST API](docs/api/rest.md)
- [GraphQL Schema](docs/api/graphql.md)
- [MCP Tools](docs/api/mcp.md)
- [Event Bus](docs/api/events.md)

---

**Maintained by**: OpenEvolve Team  
**Issues**: Report on GitHub Issues  
**License**: Apache 2.0

