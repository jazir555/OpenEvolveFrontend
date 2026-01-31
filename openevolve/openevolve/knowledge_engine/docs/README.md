# Knowledge Engine Documentation

Complete documentation for the OpenEvolve Knowledge Engine Phase 1 implementation.

## Overview

The OpenEvolve Knowledge Engine provides a comprehensive framework for managing, extracting, and reasoning over knowledge with support for temporal operations, graph-based knowledge representation, and advanced extraction pipelines.

## Key Features

- **Temporal Knowledge Tracking**: Track knowledge validity over time with point-in-time queries
- **Hybrid Search**: Combine BM25, vector similarity, and graph traversal for optimal retrieval
- **Advanced Extraction Pipelines**: Multi-stage entity and relationship extraction with deduplication
- **Graph Visualization**: Interactive knowledge graph exploration and visualization
- **Multilingual Support**: Bilingual document processing and extraction
- **Contradiction Detection**: Automated detection of conflicting knowledge

## Documentation Structure

### Getting Started
- [5-Minute Quick Start](quickstart/5_minute_quickstart.md) - Get up and running in 5 minutes
- [Developer Setup Guide](quickstart/developer_setup.md) - Complete development environment setup
- [Docker Deployment](quickstart/docker_quickstart.md) - Deploy with Docker
- [Sample Data Tutorial](quickstart/sample_data_tutorial.md) - Learn with sample data

### Integration Guides
- [Graphiti Integration Guide](temporal_kg_integration_guide.md) - Temporal knowledge graph integration
- [KG-Gen Pipeline Guide](kg_generation_pipeline_guide.md) - Knowledge graph generation pipeline
- [Bilingual Extraction Guide](multilingual_extraction_guide.md) - Multilingual document processing
- [Visualization Guide](graph_visualization_guide.md) - Graph visualization and exploration
- **OneKE Integration Guide** - [OneKE Bilingual Extraction](../integrations/oneke/ONEKE_INTEGRATION_GUIDE.md) - Complete OneKE framework documentation
- **OneKE Quick Start** - [Get Started in 5 Minutes](../integrations/oneke/ONEKE_QUICK_START.md) - Quick start for OneKE

### API Reference
- [Temporal Bridge API](api/temporal_bridge_api.md) - Temporal operations API
- [Extraction Pipeline API](api/extraction_pipeline_api.md) - Knowledge extraction API
- [Bilingual Extraction API](api/bilingual_extraction_api.md) - Multilingual extraction API
- [Visualization API](api/visualization_api.md) - Visualization and rendering API
- [Agent Memory API](api/agent_memory_api.md) - Agent memory management API
- **OneKE Tutorials** - [Bilingual Extraction Tutorial](../integrations/oneke/BILINGUAL_EXTRACTION_TUTORIAL.md) - Learn EN/CN extraction
- **Schema Guide** - [Schema Definition Guide](../integrations/oneke/SCHEMA_DEFINITION_GUIDE.md) - Create custom schemas

### Tutorials
- [Temporal Knowledge Queries](tutorials/temporal_queries_tutorial.md) - Query knowledge over time
- [Knowledge Graph Generation](tutorials/kg_generation_tutorial.md) - Generate knowledge graphs
- [Multilingual Processing](tutorials/multilingual_processing_tutorial.md) - Process documents in multiple languages
- [Graph Exploration](tutorials/graph_exploration_tutorial.md) - Explore and navigate knowledge graphs
- [Contradiction Detection](tutorials/contradiction_detection_tutorial.md) - Detect conflicting knowledge
- [Advanced Deduplication](tutorials/advanced_deduplication_tutorial.md) - Advanced deduplication techniques

### Architecture
- [Phase 1 Architecture](architecture/phase1_architecture.md) - System architecture overview
- [Temporal System Design](architecture/temporal_system_design.md) - Temporal knowledge system
- [Extraction Pipeline Architecture](architecture/extraction_pipeline_architecture.md) - Pipeline architecture
- [Visualization Architecture](architecture/visualization_architecture.md) - Visualization system design
- [Data Flow Diagrams](architecture/data_flow.md) - Data flow and processing

### Operations
- [Deployment Guide](operations/deployment_guide.md) - Production deployment
- [Configuration Guide](operations/configuration_guide.md) - System configuration
- [Monitoring Guide](operations/monitoring_guide.md) - Monitoring and observability
- [Troubleshooting Guide](operations/troubleshooting_guide.md) - Common issues and solutions
- [Performance Tuning](operations/performance_tuning_guide.md) - Optimization and tuning

## Quick Links

### Installation
```bash
# Clone the repository
git clone https://github.com/openevolve/frontend.git
cd Frontend

# Install dependencies
pip install -r requirements.txt

# Initialize the knowledge engine
python -m knowledge_engine.initialize
```

### Basic Usage
```python
from knowledge_engine.core.temporal_knowledge_engine import TemporalKnowledgeEngine
from datetime import datetime

# Initialize the engine
engine = TemporalKnowledgeEngine()

# Add temporal knowledge
await engine.add_knowledge_temporal(
    content="The API endpoint changed from /v1/users to /v2/accounts",
    artifact_type="solution_pattern",
    valid_at=datetime(2024, 1, 1),
    metadata={"source": "migration_guide"}
)

# Query at a point in time
results = await engine.query_at_time(
    query="API endpoint",
    timestamp=datetime(2024, 6, 1)
)
```

## Version Information

- **Current Version**: 1.0.0
- **Release Date**: January 2025
- **Status**: Production Ready

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../CONTRIBUTING.md) for details.

## Support

- **Documentation**: [https://docs.openevolve.org](https://docs.openevolve.org)
- **GitHub Issues**: [https://github.com/openevolve/frontend/issues](https://github.com/openevolve/frontend/issues)
- **Discord Community**: [https://discord.gg/openevolve](https://discord.gg/openevolve)

## License

Apache License 2.0 - see [LICENSE](../LICENSE) for details.

---

**Next Steps**: Start with the [5-Minute Quick Start](quickstart/5_minute_quickstart.md) to get up and running quickly.
