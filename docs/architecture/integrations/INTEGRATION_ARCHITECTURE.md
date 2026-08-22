# OpenEvolve Integration Architecture

**Author**: Agent 8 (Integration Orchestrator)
**Created**: 2026-01-02
**Status**: ✅ Base Architecture Complete
**Version**: 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [Integration Architecture](#integration-architecture)
4. [Base Interfaces](#base-interfaces)
5. [Adapter Pattern](#adapter-pattern)
6. [Factory Pattern](#factory-pattern)
7. [Health Monitoring](#health-monitoring)
8. [Configuration System](#configuration-system)
9. [Error Handling](#error-handling)
10. [Testing Strategy](#testing-strategy)
11. [Deployment](#deployment)

---

## Overview

OpenEvolve integrates 7 external projects to extend its capabilities in knowledge graphs, extraction, experimentation, optimization, uncertainty quantification, visualization, and domain knowledge.

### Integrated Projects

| Project | Priority | Domain | Interface |
|---------|----------|--------|-----------|
| **Graphiti** | P1 | Temporal Knowledge Graphs | `KnowledgeGraphInterface` |
| **OneKE** | P2 | Schema-Guided Extraction | `ExtractionInterface` |
| **Curie** | P1.5 | Scientific Experimentation | `ExperimentationInterface` |
| **NeuroMANCER** | P3 | Physics-Informed Optimization | `OptimizationInterface` |
| **pygraphistry** | P2 | Graph Visualization | `VisualizationInterface` |
| **uqtestfuns** | P3 | Uncertainty Quantification | `UncertaintyQuantificationInterface` |
| **global-chem** | P4 | Chemical Knowledge | `DomainKnowledgeInterface` |

### Key Features

- **Zero Modification**: No changes to external project source code
- **Graceful Degradation**: System functions even if integrations are unavailable
- **Decoupled Design**: Adapters isolate external dependencies
- **Configuration-Driven**: All behavior configurable via YAML/JSON
- **Async-First**: Built for asynchronous operations
- **Type Safety**: Full type hints for all interfaces
- **Health Monitoring**: Continuous health checks and alerts

---

## Architecture Principles

### 1. Zero Modification Policy

**Never modify external project source code.**

Instead, create adapter layers that wrap external functionality.

```python
# ✅ CORRECT: Create adapter
class GraphitiAdapter(KnowledgeGraphInterface):
    def __init__(self, config):
        # Wrap Graphiti without modification
        self.client = GraphitiClient(config)

# ❌ WRONG: Modify Graphiti source
# Do not edit files in projects/graphiti/
```

### 2. Adapter Pattern

All integrations implement abstract base interfaces:

```python
from abc import ABC, abstractmethod

class DomainInterface(ABC):
    @abstractmethod
    async def initialize(self, config: Dict) -> None:
        pass

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    async def validate(self) -> bool:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass
```

### 3. Graceful Degradation

System continues operating even when integrations fail:

```python
# Get integration (returns None if unavailable)
graphiti = await factory.get_knowledge_graph()

if graphiti:
    # Use Graphiti
    await graphiti.add_episode(...)
else:
    # Fall back to default behavior
    logger.warning("Graphiti unavailable, using fallback")
    await fallback_method(...)
```

### 4. Dependency Isolation

Use separate environments when needed:

```yaml
# config.yaml
integration:
  name: neuromancer
  isolation:
    method: conda
    environment: neuromancer_env  # Separate PyTorch environment
```

### 5. Configuration-Driven

All integration behavior via configuration files:

```yaml
project:
  name: graphiti
  version: 1.0.0
  enabled: true

connection:
  uri: bolt://localhost:7687
  username: ${NEO4J_USERNAME}  # Environment variable
  password: ${NEO4J_PASSWORD}

integration:
  auto_start: true
  cache_enabled: true
  fallback_on_error: true
```

---

## Integration Architecture

### Directory Structure

```
openevolve/
├── integrations/
│   ├── __init__.py                 # Factory pattern entry point
│   ├── registry.py                 # Dynamic integration loading
│   ├── health_monitor.py           # Health monitoring system
│   ├── config_loader.py            # Configuration management
│   │
│   ├── base/                       # Abstract interfaces
│   │   ├── knowledge_interface.py
│   │   ├── extraction_interface.py
│   │   ├── experimentation_interface.py
│   │   ├── optimization_interface.py
│   │   ├── uq_interface.py
│   │   ├── visualization_interface.py
│   │   └── domain_knowledge_interface.py
│   │
│   ├── graphiti/                   # Graphiti integration
│   │   ├── adapter.py
│   │   ├── bridge.py
│   │   └── config.yaml
│   │
│   ├── oneke/                      # OneKE integration
│   │   ├── adapter.py
│   │   ├── bridge.py
│   │   └── config.yaml
│   │
│   ├── curie/                      # Curie integration
│   │   ├── adapter.py
│   │   ├── bridge.py
│   │   └── config.yaml
│   │
│   ├── neuromancer/                # NeuroMANCER integration
│   │   ├── adapter.py
│   │   ├── bridge.py
│   │   └── config.yaml
│   │
│   ├── pygraphistry/               # pygraphistry integration
│   │   ├── adapter.py
│   │   ├── bridge.py
│   │   └── config.yaml
│   │
│   ├── uqtestfuns/                 # uqtestfuns integration
│   │   ├── adapter.py
│   │   ├── bridge.py
│   │   └── config.yaml
│   │
│   └── global_chem/                # global-chem integration
│       ├── adapter.py
│       ├── bridge.py
│       └── config.yaml
│
├── projects/                       # External projects (git submodules)
│   ├── graphiti/
│   ├── OneKE/
│   ├── curie/
│   ├── neuromancer/
│   ├── pygraphistry/
│   ├── uqtestfuns/
│   └── global-chem/
│
└── docs/
    └── integrations/
        ├── INTEGRATION_ARCHITECTURE.md  # This document
        └── INTEGRATION_STATUS.md         # Status dashboard
```

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    OpenEvolve Core                           │
│  (Workflow Engine, Knowledge Engine, SOP Generator, etc.)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   IntegrationFactory                         │
│  (Factory Pattern - Single Entry Point for All Integrations) │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌──────────┐ ┌──────────────┐
    │  Registry   │ │   Health  │ │    Config    │
    │             │ │  Monitor  │ │    Loader    │
    └──────┬──────┘ └─────┬────┘ └──────┬───────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Base Interfaces                         │
│  (Abstract contracts all adapters must implement)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌─────────────┐ ┌──────────┐ ┌──────────────┐
    │   Adapter   │ │   Bridge │ │    Config    │
    │   (Wrap)    │ │ (Connect)│ │   (.yaml)    │
    └──────┬──────┘ └─────┬────┘ └──────────────┘
           │               │
           └───────────────┼───────────────┐
                           ▼               ▼
                    ┌─────────────┐ ┌────────────┐
                    │  External   │ │  Fallback  │
                    │  Projects   │ │  Behavior  │
                    └─────────────┘ └────────────┘
```

---

## Base Interfaces

All integrations implement abstract base interfaces that define the contract between OpenEvolve and external projects.

### Interface Hierarchy

```
DomainInterface (ABC)
├── KnowledgeGraphInterface
│   ├── initialize(config)
│   ├── add_episode(...)
│   ├── search(...)
│   ├── validate()
│   └── shutdown()
│
├── ExtractionInterface
│   ├── initialize(config)
│   ├── extract_ner(...)
│   ├── extract_re(...)
│   ├── extract_schema_guided(...)
│   ├── validate()
│   └── shutdown()
│
├── ExperimentationInterface
│   ├── initialize(config)
│   ├── design_experiment(...)
│   ├── run_experiment(...)
│   ├── analyze_results(...)
│   ├── validate()
│   └── shutdown()
│
├── OptimizationInterface
│   ├── initialize(config)
│   ├── solve(...)
│   ├── constrained_optimization(...)
│   ├── system_identification(...)
│   ├── validate()
│   └── shutdown()
│
├── UncertaintyQuantificationInterface
│   ├── initialize(config)
│   ├── sample_inputs(...)
│   ├── compute_statistics(...)
│   ├── compute_sensitivity(...)
│   ├── validate()
│   └── shutdown()
│
├── VisualizationInterface
│   ├── initialize(config)
│   ├── visualize_graph(...)
│   ├── compute_embeddings(...)
│   ├── cluster_nodes(...)
│   ├── validate()
│   └── shutdown()
│
└── DomainKnowledgeInterface
    ├── initialize(config)
    ├── query_chemical(...)
    ├── search_smiles(...)
    ├── get_properties(...)
    ├── validate()
    └── shutdown()
```

### Interface Methods

Every interface follows the standard lifecycle:

1. **`initialize(config)`** - Set up the integration with configuration
2. **Domain-specific methods** - Methods specific to the integration type
3. **`validate()`** - Check integration health and availability
4. **`shutdown()`** - Clean shutdown and resource cleanup

All methods are `async` to support concurrent operations.

---

## Adapter Pattern

Adapters wrap external projects and implement base interfaces.

### Adapter Template

```python
from integrations.base.knowledge_interface import KnowledgeGraphInterface
import external_project

class ExternalProjectAdapter(KnowledgeGraphInterface):
    """Adapter for External Project"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self._initialized = False

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize external project client"""
        try:
            # Create client without modifying external source
            self.client = external_project.Client(**config)
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def add_episode(self, name, body, reference_time, **kwargs):
        """Add episode using external project"""
        if not self._initialized:
            raise RuntimeError("Adapter not initialized")

        # Call external project API
        return await self.client.add_episode(
            name=name,
            body=body,
            reference_time=reference_time,
            **kwargs
        )

    async def validate(self) -> Dict[str, Any]:
        """Validate adapter is working"""
        try:
            # Test basic operation
            result = await self.client.ping()
            return {
                'is_valid': result,
                'checks': {'connection': result},
                'issues': []
            }
        except Exception as e:
            return {
                'is_valid': False,
                'issues': [str(e)]
            }

    async def shutdown(self) -> bool:
        """Shutdown adapter"""
        if self.client:
            await self.client.close()
        self._initialized = False
        return True
```

### Key Adapter Responsibilities

1. **Wrap External API** - Call external project methods
2. **Transform Data** - Convert between OpenEvolve and external formats
3. **Handle Errors** - Catch and transform exceptions
4. **Implement Interface** - Fulfill contract defined by base interface
5. **Maintain State** - Track initialization and connection status

---

## Factory Pattern

The `IntegrationFactory` provides a unified interface for accessing all integrations.

### Usage Example

```python
from integrations import IntegrationFactory

# Create factory
factory = IntegrationFactory(config_dir="integrations/configs")

# Get integrations
graphiti = await factory.get_knowledge_graph()  # Returns None if unavailable
oneke = await factory.get_extraction()
curie = await factory.get_experimentation()

# Use integrations with graceful degradation
if graphiti:
    await graphiti.add_episode(...)
else:
    # Fallback behavior
    logger.warning("Graphiti unavailable")

# Check health
health = await factory.check_all_health()
for name, status in health.items():
    print(f"{name}: {status.status}")

# List all integrations
integrations = factory.list_integrations()
for info in integrations:
    print(f"{info.name} - {info.status}")

# Shutdown
await factory.shutdown_all()
```

### Factory Methods

```python
class IntegrationFactory:
    # Knowledge Graph
    async def get_knowledge_graph(name="graphiti", config=None)
        -> Optional[KnowledgeGraphInterface]

    # Extraction
    async def get_extraction(name="oneke", config=None)
        -> Optional[ExtractionInterface]

    # Experimentation
    async def get_experimentation(name="curie", config=None)
        -> Optional[ExperimentationInterface]

    # Optimization
    async def get_optimization(name="neuromancer", config=None)
        -> Optional[OptimizationInterface]

    # Uncertainty Quantification
    async def get_uncertainty_quantification(name="uqtestfuns", config=None)
        -> Optional[UncertaintyQuantificationInterface]

    # Visualization
    async def get_visualization(name="pygraphistry", config=None)
        -> Optional[VisualizationInterface]

    # Domain Knowledge
    async def get_domain_knowledge(name="global_chem", config=None)
        -> Optional[DomainKnowledgeInterface]

    # Generic getter
    async def get_integration(name, config=None) -> Optional[Any]

    # Health monitoring
    async def start_health_monitoring()
    async def stop_health_monitoring()
    async def check_health(integration: str) -> Optional[IntegrationHealth]
    async def check_all_health() -> Dict[str, IntegrationHealth]

    # Registry management
    def list_integrations(status_filter=None, type_filter=None)
        -> List[IntegrationInfo]
    def get_integration_info(name: str) -> Optional[IntegrationInfo]
    async def is_available(name: str) -> bool

    # Lifecycle
    async def shutdown_integration(name: str) -> bool
    async def shutdown_all() -> Dict[str, bool]
    async def validate_all() -> Dict[str, Dict[str, Any]]
```

---

## Health Monitoring

The health monitoring system provides continuous health checks for all integrations.

### Health Monitor Features

- **Periodic Health Checks** - Automatic health monitoring at configurable intervals
- **Performance Metrics** - Track response times, error rates, uptime
- **Alert Generation** - Automatic alerts for unhealthy states
- **Historical Data** - Track health history over time
- **Metrics Export** - Export metrics in JSON or Prometheus format

### Usage Example

```python
from integrations import IntegrationFactory

factory = IntegrationFactory()

# Start monitoring (runs in background)
await factory.start_health_monitoring()

# Get health summary
summary = factory.get_health_summary()
print(f"Healthy: {summary['healthy']}")
print(f"Unhealthy: {summary['unhealthy']}")
print(f"Active alerts: {summary['active_alerts']}")

# Get specific integration health
health = await factory.check_health("graphiti")
print(f"Status: {health.status}")
print(f"Error rate: {health.error_rate}")
print(f"Response time: {health.avg_response_time}ms")

# Export metrics
prometheus_metrics = factory.health_monitor.export_metrics("prometheus")
print(prometheus_metrics)

# Stop monitoring
await factory.stop_health_monitoring()
```

### Health Status Levels

| Status | Description | Action |
|--------|-------------|--------|
| `HEALTHY` | Integration functioning normally | None |
| `DEGRADED` | Integration slow but working | Monitor |
| `UNHEALTHY` | Integration failed | Investigate |
| `UNKNOWN` | Health check failed | Check configuration |

### Alert Levels

| Level | Description | Example |
|-------|-------------|---------|
| `INFO` | Informational | Integration started |
| `WARNING` | Warning condition | High error rate |
| `ERROR` | Error condition | Health check failed |
| `CRITICAL` | Critical condition | Integration down |

---

## Configuration System

All integration behavior is configurable via YAML or JSON files.

### Configuration Structure

```yaml
# Project metadata
project:
  name: graphiti
  version: 1.0.0
  description: Temporal knowledge graph
  enabled: true

# Connection settings
connection:
  uri: bolt://localhost:7687
  username: ${NEO4J_USERNAME}  # Environment variable
  password: ${NEO4J_PASSWORD}
  database: neo4j
  timeout: 30
  retries: 3

# Feature flags
features:
  temporal_metadata: true
  hybrid_search: true
  graph_traversal: true

# Integration behavior
integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true

# Performance settings
performance:
  max_workers: 4
  batch_size: 100
  timeout: 30

# Logging
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Environment Variables

Configuration supports environment variable interpolation:

```yaml
connection:
  api_key: ${API_KEY}                    # Required
  database: ${DATABASE_NAME:neo4j}       # Optional with default
  timeout: ${TIMEOUT:30}                 # Optional with default
```

### Config Loader API

```python
from integrations.config_loader import ConfigLoader

loader = ConfigLoader()

# Load configuration
config = loader.load("integrations/graphiti/config.yaml")

# Save configuration
loader.save(config, "integrations/graphiti/config.yaml")

# Create example config
loader.create_example_config("graphiti", "example_config.yaml")

# Clear cache
loader.clear_cache()
```

---

## Error Handling

### Exception Hierarchy

Each interface defines specific exceptions:

```python
# Base exception
class IntegrationError(Exception):
    pass

# Common exceptions
class ConfigurationError(IntegrationError):
    """Invalid configuration"""

class ConnectionError(IntegrationError):
    """Connection failed"""

class ValidationError(IntegrationError):
    """Validation failed"""

class ShutdownError(IntegrationError):
    """Shutdown failed"""

# Domain-specific exceptions (per interface)
class ExtractionError(IntegrationError):
    """Extraction operation failed"""

class OptimizationError(IntegrationError):
    """Optimization failed"""
```

### Error Handling Pattern

```python
from integrations import IntegrationFactory
from integrations.base.knowledge_interface import ConnectionError

factory = IntegrationFactory()

try:
    graphiti = await factory.get_knowledge_graph()
    if graphiti:
        await graphiti.add_episode(...)
except ConnectionError as e:
    logger.error(f"Connection failed: {e}")
    # Use fallback
    await fallback_add_episode(...)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

### Graceful Degradation Pattern

```python
async def add_knowledge_with_fallback(data):
    """Add knowledge with fallback on error"""
    graphiti = await factory.get_knowledge_graph()

    if graphiti:
        try:
            return await graphiti.add_episode(**data)
        except Exception as e:
            logger.warning(f"Graphiti failed: {e}, using fallback")

    # Fallback to default behavior
    return await default_add_knowledge(data)
```

---

## Testing Strategy

### Unit Testing

Test each adapter in isolation:

```python
import pytest
from integrations.graphiti.adapter import GraphitiAdapter

@pytest.mark.asyncio
async def test_graphiti_adapter_initialize():
    adapter = GraphitiAdapter(config={...})
    result = await adapter.initialize({})
    assert result is True

@pytest.mark.asyncio
async def test_graphiti_adapter_add_episode():
    adapter = GraphitiAdapter(config={...})
    await adapter.initialize({})

    result = await adapter.add_episode(
        name="test",
        body="test episode",
        reference_time=datetime.now()
    )

    assert result["uuid"] is not None
```

### Integration Testing

Test adapter with actual external project:

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_graphiti_integration():
    factory = IntegrationFactory()
    graphiti = await factory.get_knowledge_graph()

    assert graphiti is not None
    health = await graphiti.validate()
    assert health["is_valid"] is True
```

### Health Testing

Test health monitoring:

```python
@pytest.mark.asyncio
async def test_health_monitoring():
    factory = IntegrationFactory()

    # Start monitoring
    await factory.start_health_monitoring()

    # Check health
    health = await factory.check_health("graphiti")
    assert health is not None
    assert health.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]

    # Stop monitoring
    await factory.stop_health_monitoring()
```

### Coverage Requirements

- **Unit tests**: >80% code coverage
- **Integration tests**: Cover all major operations
- **Health tests**: Verify monitoring system
- **Error tests**: Verify graceful degradation

---

## Deployment

### Environment Setup

1. **Create conda environments for isolated integrations**:

```bash
# PyTorch environment for NeuroMANCER
conda create -n neuromancer_env python=3.10
conda activate neuromancer_env
conda install pytorch torchvision torchaudio

# Default environment for other integrations
conda create -n openevolve python=3.10
conda activate openevolve
pip install -r requirements.txt
```

2. **Install external projects**:

```bash
# Add as git submodules
git submodule add https://github.com/getgraphiti/graphiti projects/graphiti
git submodule add https://github.com/xxx/OneKE projects/OneKE
# ... etc

# Update submodules
git submodule update --init --recursive
```

3. **Configure environment variables**:

```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
```

### Configuration Deployment

```bash
# Copy example configs
cp integrations/configs/example/*.yaml integrations/configs/

# Edit configs for your environment
vim integrations/configs/graphiti.yaml
vim integrations/configs/oneke.yaml
# ... etc
```

### Health Monitoring Setup

```python
# In your application startup
from integrations import IntegrationFactory

factory = IntegrationFactory()

# Start health monitoring
await factory.start_health_monitoring()

# Configure alert callbacks
async def alert_callback(alert):
    if alert.level == AlertLevel.CRITICAL:
        # Send PagerDuty alert
        await send_pagerduty_alert(alert)
    elif alert.level == AlertLevel.ERROR:
        # Send Slack notification
        await send_slack_notification(alert)

factory.health_monitor.alert_callbacks.append(alert_callback)
```

### Production Checklist

- [ ] All external projects installed as git submodules
- [ ] Conda environments created for isolated dependencies
- [ ] Configuration files deployed and customized
- [ ] Environment variables set
- [ ] Health monitoring enabled
- [ ] Alert callbacks configured
- [ ] Logging configured and tested
- [ ] Integration tests passing
- [ ] Fallback behaviors tested
- [ ] Documentation updated

---

## Summary

The OpenEvolve integration architecture provides:

1. **Decoupled Design** - Adapters isolate external dependencies
2. **Zero Modification** - No changes to external project source
3. **Graceful Degradation** - System continues operating on failures
4. **Configuration-Driven** - All behavior via YAML/JSON
5. **Health Monitoring** - Continuous health checks and alerts
6. **Type Safety** - Full type hints and async support
7. **Extensibility** - Easy to add new integrations

All 7 external projects can now be integrated following this architecture, with each specialist agent implementing the specific adapter for their assigned project.

---

**Next Steps**: See [INTEGRATION_STATUS.md](INTEGRATION_STATUS.md) for integration progress dashboard.
