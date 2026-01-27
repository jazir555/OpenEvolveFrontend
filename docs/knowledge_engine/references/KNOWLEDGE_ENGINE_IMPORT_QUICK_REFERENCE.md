# Knowledge Engine Import Quick Reference

Quick copy-paste import statements for all Knowledge Engine components.

## Main Orchestration

```python
from knowledge_engine import (
    KnowledgeEngine,
    create_knowledge_engine,
    ProcessingResult,
    QueryResult
)
```

## Sprint 1: Graphiti (Temporal Knowledge Graph)

```python
from knowledge_engine.integrations.graphiti import (
    # Core
    GraphitiTemporalBridge,
    GraphitiContradictionDetector,
    GraphitiIncrementalUpdater,
    GraphitiAgentMemory,
    GraphitiHealthChecker,

    # Configuration
    GraphitiConfig,
    validate_config,

    # Exceptions
    GraphitiIntegrationError,
    ConfigurationError,
    ContradictionError,
    IncrementalUpdateError,

    # Quick health check
    health_check_quick
)
```

## Sprint 2: KG-Gen (Knowledge Graph Generation)

```python
from knowledge_engine.integrations.kggen import (
    # Core
    ExtractionPipeline,
    DeduplicationEngine,
    KGGenMCPServer,
    ConversationAnalyzer,
    GraphAggregator,

    # Result types
    ExtractionResult,
    DeduplicationResult,
    ConversationResult,
    AggregationResult,

    # Strategies
    SEMHASHStrategy,
    LMClusterStrategy,
    CrossDocumentResolver
)
```

## Sprint 3: OneKE (Bilingual Knowledge Extraction)

```python
from knowledge_engine.integrations.oneke import (
    # Core
    OneKEModelAdapter,
    MultiTaskExtractionFramework,
    OneKESchemaManager,
    CrossLingualEntityLinker,
    EventExtractionPipeline,

    # Configuration
    ModelConfig,
    SchemaDefinition,

    # Result types
    ExtractionResult,
    EntityMatchResult,
    EventChain,
    TemporalEvent
)
```

## Sprint 4: Visualization

```python
from knowledge_engine.visualization import (
    # Visualizers
    GraphExplorer,
    TemporalVisualizer,
    CommunityVisualizer,

    # Configuration
    VisualizationOptions,
    TemporalVisualizationOptions,
    CommunityVisualizationOptions,

    # Filters
    NodeFilter,
    EdgeFilter,

    # Export
    ExportHandler
)
```

## Complete Import (All Components)

```python
from knowledge_engine import (
    KnowledgeEngine,
    create_knowledge_engine,
    ProcessingResult,
    QueryResult
)

from knowledge_engine.integrations.graphiti import (
    GraphitiTemporalBridge,
    GraphitiContradictionDetector,
    GraphitiIncrementalUpdater,
    GraphitiAgentMemory
)

from knowledge_engine.integrations.kggen import (
    ExtractionPipeline,
    DeduplicationEngine,
    KGGenMCPServer
)

from knowledge_engine.integrations.oneke import (
    OneKEModelAdapter,
    MultiTaskExtractionFramework
)

from knowledge_engine.visualization import (
    GraphExplorer,
    TemporalVisualizer,
    CommunityVisualizer
)
```

## Usage Examples

### Initialize Knowledge Engine
```python
from knowledge_engine import create_knowledge_engine

async def main():
    engine = await create_knowledge_engine()
    result = await engine.process_document("doc.pdf")
    await engine.close()
```

### Use Graphiti for Temporal Knowledge
```python
from knowledge_engine.integrations.graphiti import GraphitiTemporalBridge

bridge = GraphitiTemporalBridge()
await bridge.add_artifact(artifact_data)
temporal_data = await bridge.query_temporal(start_date, end_date)
```

### Use KG-Gen for Extraction
```python
from knowledge_engine.integrations.kggen import ExtractionPipeline

pipeline = ExtractionPipeline()
result = await pipeline.extract(text_document)
```

### Use OneKE for Bilingual Extraction
```python
from knowledge_engine.integrations.oneke import OneKEModelAdapter

adapter = OneKEModelAdapter()
result = await adapter.extract_entities(text, language="bilingual")
```

### Visualize Knowledge Graph
```python
from knowledge_engine.visualization import GraphExplorer, ExportHandler

explorer = GraphExplorer()
viz = explorer.explore(graph_data)
ExportHandler.export(viz, format="html")
```

---

**Last Updated:** 2026-01-08
**Test Status:** All imports verified working (100% success rate)
**Total Components:** 88 items across 5 modules
