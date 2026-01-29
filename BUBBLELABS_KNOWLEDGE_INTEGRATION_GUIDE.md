# BubbleLabs Knowledge Engine Integration Guide

## Overview

This integration connects BubbleLabs UI with the OpenEvolve Knowledge Engine, providing comprehensive knowledge exploration, visualization, and extraction capabilities.

**Created:** 2026-01-03
**Status:** Production Ready
**Version:** 1.0.0

---

## Features

### 1. Knowledge Graph Visualization
- Interactive network graphs with force-directed layouts
- Entity and relationship exploration
- Confidence score visualization
- Multiple layout algorithms (spring, circular, kamada-kawai)
- Graph statistics and metrics
- Neighbor exploration and pathfinding

### 2. Knowledge Query Interface
- **Multi-source querying:**
  - Bedrock Knowledge Base
  - Graphiti temporal knowledge graph
  - Local code indexes
  - Elasticsearch indices

- **Query features:**
  - Unified query across multiple sources
  - Temporal search with Graphiti
  - Query history tracking
  - Result caching
  - Configurable result limits

### 3. Knowledge Extraction Workflow
- Document loading (PDF, Office, text files)
- LLM-powered entity extraction
- Relationship extraction
- Knowledge graph construction
- Batch extraction support
- Extraction history tracking

### 4. Knowledge Statistics Dashboard
- Entity type distribution
- Relationship type distribution
- Graph density metrics
- Query execution statistics
- Interactive visualizations

---

## Installation

### Requirements

```bash
pip install streamlit plotly networkx pandas
pip install boto3  # For Bedrock
pip install elasticsearch  # For Elasticsearch
```

### Setup

1. **Configure API Keys:**

   Edit `mcp_agent.secrets.yaml`:

   ```yaml
   aws:
     region_name: "us-east-1"
     access_key_id: "YOUR_ACCESS_KEY"
     secret_access_key: "YOUR_SECRET_KEY"

   anthropic:
     api_key: "sk-ant-..."

   openai:
     api_key: "sk-..."
   ```

2. **Configure Knowledge Engine:**

   Edit `knowledge_engine/indexer_config.yaml`:

   ```yaml
   llm:
     model_provider: "anthropic"
     anthropic_default_model: "claude-sonnet-4-20250514"
     temperature: 0.3
     max_tokens: 1000
   ```

3. **Run the Application:**

   ```bash
   streamlit run bubblelabs_knowledge_integration.py
   ```

---

## Usage

### 1. Knowledge Query Interface

#### Query Bedrock Knowledge Base

```python
from bubblelabs_knowledge_integration import BubbleLabsKnowledgeUI

# Initialize UI
ui = BubbleLabsKnowledgeUI()
ui.initialize_engine()

# Execute query
results = await ui.query_interface.query_bedrock(
    knowledge_base_id="YOUR_KB_ID",
    query_text="How does adversarial validation work?",
    use_temporal_search=True
)
```

#### Query Graphiti Temporal Knowledge

```python
results = await ui.query_interface.query_graphiti(
    query="decomposition workflow improvements",
    temporal_filters={
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
)
```

#### Unified Query Across Sources

```python
results = await ui.query_interface.unified_query(
    query="MCTS optimization strategies",
    sources=['bedrock', 'graphiti', 'local'],
    bedrock_kb_id="YOUR_KB_ID",
    index_path="knowledge_index"
)
```

### 2. Knowledge Graph Visualization

#### Create Interactive Visualization

```python
from bubblelabs_knowledge_integration import KnowledgeGraphVisualizer

# Initialize visualizer
visualizer = KnowledgeGraphVisualizer()

# Build graph from data
visualizer.build_graph_from_data(
    entities=[
        {"name": "MCTS", "type": "algorithm", "attributes": {"confidence": 0.95}},
        {"name": "MDAP", "type": "framework", "attributes": {}}
    ],
    relationships=[
        {"source": "MCTS", "relation": "optimizes", "target": "MDAP"}
    ]
)

# Create Plotly figure
fig = visualizer.create_interactive_plot(
    layout='spring',
    node_size_multiplier=1.5,
    show_labels=True
)

# Display in Streamlit
st.plotly_chart(fig, use_container_width=True)
```

#### Get Graph Statistics

```python
stats = visualizer.get_graph_statistics()
print(f"Nodes: {stats['total_nodes']}")
print(f"Edges: {stats['total_edges']}")
print(f"Density: {stats['density']}")
```

#### Find Shortest Path

```python
path = visualizer.find_shortest_path("MCTS", "Adversarial")
print(f"Path: {' -> '.join(path)}")
```

### 3. Knowledge Extraction Workflow

#### Extract from Document

```python
from bubblelabs_knowledge_integration import KnowledgeExtractionWorkflow

# Initialize workflow
workflow = KnowledgeExtractionWorkflow(knowledge_engine)

# Extract from PDF
results = await workflow.extract_from_document(
    document_path_or_url="https://arxiv.org/pdf/2301.07041",
    extraction_config={
        "extract_entities": True,
        "extract_relationships": True,
        "min_confidence": 0.7
    }
)

print(f"Extracted {results['statistics']['total_entities']} entities")
print(f"Extracted {results['statistics']['total_relationships']} relationships")
```

#### Extract from Text

```python
text = """
MCTS (Monte Carlo Tree Search) is used for optimization in MDAP.
It improves decision making through tree traversal and backpropagation.
"""

entities, relationships = await workflow._extract_knowledge(text)

print("Entities:")
for entity in entities:
    print(f"  - {entity['name']} ({entity['type']})")

print("Relationships:")
for rel in relationships:
    print(f"  - {rel['source']} {rel['relation']} {rel['target']}")
```

---

## Streamlit UI Components

### Main Interface

```python
from bubblelabs_knowledge_integration import BubbleLabsKnowledgeUI

# Create UI
ui = BubbleLabsKnowledgeUI()

# Render explorer
ui.render_knowledge_explorer()
```

### Individual Components

#### Query Interface

```python
ui.render_query_interface()
```

Features:
- Multi-source query builder
- Query history
- Result visualization
- Source-specific configuration

#### Graph Visualization

```python
ui.render_graph_visualization()
```

Features:
- Interactive network plot
- Layout selection
- Entity exploration
- Graph statistics

#### Extraction Workflow

```python
ui.render_extraction_workflow()
```

Features:
- Document upload
- URL input
- Text input
- Real-time extraction
- Result preview

#### Statistics Dashboard

```python
ui.render_statistics_dashboard()
```

Features:
- Entity/relationship counts
- Type distribution charts
- Query statistics
- Interactive metrics

---

## Knowledge Engine Integration

### Direct Knowledge Engine Usage

```python
from knowledge_engine.engine import KnowledgeEngine

# Initialize engine
engine = KnowledgeEngine()

# Query Bedrock
results = await engine.query_bedrock_knowledge_base(
    knowledge_base_id="YOUR_KB_ID",
    query="temporal knowledge graph"
)

# Add document
text = await engine.add_document(
    path_or_url="document.pdf",
    output_dir="processed_docs"
)

# Index project
index_result = await engine.index_project(
    project_path="./src",
    target_structure="Software architecture",
    output_dir="indexes"
)

# Load and query index
index_data = engine.load_index("indexes/project_index.json")
matches = engine.query_index_by_keyword(index_data, "MCTS")
```

### Knowledge Graph Operations

```python
from knowledge_engine.core import EntityKnowledgeGraph

# Create knowledge graph
kg = EntityKnowledgeGraph()

# Add entities
kg.add_entity("MCTS", {"type": "algorithm", "confidence": 0.95})
kg.add_entity("MDAP", {"type": "framework"})

# Add relationships
kg.add_relationship("MCTS", "optimizes", "MDAP")
kg.add_relationship("MDAP", "uses", "MCTS", {"strength": "high"})

# Query
entity = kg.get_entity("MCTS")
relationships = kg.get_relationships_for_entity("MCTS")

# Serialize
kg_dict = kg.to_dict()
```

---

## Advanced Features

### 1. Temporal Knowledge with Graphiti

```python
# Enable Graphiti integration
bedrock_client = BedrockKnowledgeBaseClient(
    region_name='us-east-1',
    use_graphiti=True,
    graphiti_config_path='graphiti/config.yaml'
)

# Query with temporal context
results = await bedrock_client.query_knowledge_base(
    knowledge_base_id="YOUR_KB_ID",
    query_text="MCTS improvements over time",
    use_temporal_search=True
)

# Access temporal metadata
temporal_edges = results['temporal_metadata']['edges_with_temporal_metadata']
for edge in temporal_edges:
    print(f"{edge['fact']}")
    print(f"  Created: {edge['created_at']}")
    print(f"  Valid: {edge['valid_at']}")
```

### 2. Knowledge Provenance Tracking

```python
# Add episode to Graphiti
await bedrock_client.add_episode_to_graphiti(
    name="MCTS Optimization Episode",
    body="Implemented improved MCTS with ACI guidance",
    reference_time=datetime.now(),
    metadata={
        "source": "workflow_execution",
        "confidence": 0.9
    }
)
```

### 3. Confidence Filtering

```python
# Filter by confidence
fig = visualizer.create_interactive_plot(
    layout='spring',
    min_confidence=0.7  # Only show edges with confidence >= 0.7
)
```

### 4. Entity Type Filtering

```python
# Filter graph by entity types
filtered_entities = [
    e['name'] for e in entities
    if e.get('type') in ['algorithm', 'framework']
]

fig = visualizer.create_interactive_plot(
    layout='spring',
    filter_entities=filtered_entities
)
```

---

## Configuration

### Knowledge Engine Configuration

**File:** `knowledge_engine/indexer_config.yaml`

```yaml
# LLM Configuration
llm:
  model_provider: "anthropic"  # or "openai", "google"
  anthropic_default_model: "claude-sonnet-4-20250514"
  openai_default_model: "o3-mini"
  google_default_model: "gemini-2.0-flash"
  temperature: 0.3
  max_tokens: 1000
  system_prompt: "You are a helpful assistant."

# Indexing Configuration
paths:
  code_base_path: "code_base"
  output_dir: "indexes"

# File Analysis
file_analysis:
  supported_extensions:
    - ".py"
    - ".js"
    - ".ts"
    - ".md"
    - ".txt"
  skip_directories:
    - "__pycache__"
    - "node_modules"
  max_file_size: 1048576  # 1MB
  max_content_length: 3000

# Relationship Configuration
relationships:
  min_confidence_score: 0.3
  high_confidence_threshold: 0.7
  relationship_types:
    direct_match: 1.0
    partial_match: 0.8
    reference: 0.6
    utility: 0.4

# Performance
performance:
  enable_concurrent_analysis: false
  max_concurrent_files: 5
  enable_content_caching: true
  max_cache_size: 100

# Output
output:
  generate_summary: true
  generate_statistics: true
  include_metadata: true
```

### Bedrock Configuration

**File:** `mcp_agent.secrets.yaml`

```yaml
aws:
  region_name: "us-east-1"
  access_key_id: "YOUR_ACCESS_KEY"
  secret_access_key: "YOUR_SECRET_KEY"

# Bedrock Knowledge Base IDs
bedrock:
  knowledge_bases:
    default: "YOUR_DEFAULT_KB_ID"
    research: "YOUR_RESEARCH_KB_ID"
    code: "YOUR_CODE_KB_ID"
```

---

## API Reference

### KnowledgeGraphVisualizer

```python
class KnowledgeGraphVisualizer:
    def __init__(self)
    def build_graph_from_data(self, entities, relationships) -> nx.DiGraph
    def create_interactive_plot(self, layout, node_size_multiplier, ...) -> go.Figure
    def get_graph_statistics(self) -> Dict[str, Any]
    def find_shortest_path(self, source, target) -> Optional[List[str]]
    def get_entity_neighbors(self, entity, depth) -> Dict[str, List[str]]
```

### KnowledgeQueryInterface

```python
class KnowledgeQueryInterface:
    def __init__(self, knowledge_engine: KnowledgeEngine)
    async def query_bedrock(self, knowledge_base_id, query_text, ...) -> Dict[str, Any]
    async def query_graphiti(self, query, temporal_filters, ...) -> Optional[Dict[str, Any]]
    async def query_local_index(self, index_path, keyword) -> List[Dict[str, Any]]
    async def unified_query(self, query, sources, ...) -> Dict[str, Any]
    def get_query_history(self, limit: int) -> List[Dict[str, Any]]
```

### KnowledgeExtractionWorkflow

```python
class KnowledgeExtractionWorkflow:
    def __init__(self, knowledge_engine: KnowledgeEngine)
    async def extract_from_document(self, document_path_or_url, ...) -> Dict[str, Any]
    async def _extract_knowledge(self, text: str) -> Tuple[List, List]
    def get_extraction_history(self) -> List[Dict[str, Any]]
```

### BubbleLabsKnowledgeUI

```python
class BubbleLabsKnowledgeUI:
    def __init__(self)
    def initialize_engine(self)
    def render_knowledge_explorer(self)
    def render_query_interface(self)
    def render_graph_visualization(self)
    def render_extraction_workflow(self)
    def render_statistics_dashboard(self)
```

---

## Examples

### Example 1: Query and Visualize Research Knowledge

```python
import asyncio
from bubblelabs_knowledge_integration import BubbleLabsKnowledgeUI

# Initialize
ui = BubbleLabsKnowledgeUI()
ui.initialize_engine()

# Query research knowledge
results = asyncio.run(ui.query_interface.unified_query(
    query="MCTS adversarial validation improvements",
    sources=['bedrock', 'graphiti'],
    bedrock_kb_id="RESEARCH_KB_ID"
))

# Extract entities and relationships from results
if 'graphiti' in results['sources']:
    graphiti_data = results['sources']['graphiti']
    entities = [
        {"name": n['name'], "type": "concept"}
        for n in graphiti_data.get('nodes', [])
    ]
    relationships = [
        {
            "source": e.get('source', ''),
            "relation": e.get('fact', 'related_to'),
            "target": e.get('target', '')
        }
        for e in graphiti_data.get('edges', [])
    ]

    # Visualize
    ui.visualizer.build_graph_from_data(entities, relationships)
    fig = ui.visualizer.create_interactive_plot(layout='spring')
    st.plotly_chart(fig)
```

### Example 2: Extract Knowledge from Technical Paper

```python
import asyncio
from bubblelabs_knowledge_integration import KnowledgeExtractionWorkflow

# Initialize
workflow = KnowledgeExtractionWorkflow(knowledge_engine)

# Extract from paper
results = asyncio.run(workflow.extract_from_document(
    document_path_or_url="https://arxiv.org/pdf/2301.07041"
))

# Display results
print(f"Extracted {len(results['entities'])} entities")
print(f"Extracted {len(results['relationships'])} relationships")

# Build and visualize graph
ui.visualizer.build_graph_from_data(
    results['entities'],
    results['relationships']
)
```

### Example 3: Multi-Source Knowledge Query

```python
import asyncio
from bubblelabs_knowledge_integration import BubbleLabsKnowledgeUI

ui = BubbleLabsKnowledgeUI()
ui.initialize_engine()

# Query across all sources
results = asyncio.run(ui.query_interface.unified_query(
    query="How does MDAP use MCTS for optimization?",
    sources=['bedrock', 'graphiti', 'local'],
    bedrock_kb_id="MAIN_KB_ID",
    index_path="knowledge_index"
))

# Display combined results
for source, data in results['sources'].items():
    print(f"\n=== {source.upper()} RESULTS ===")
    if isinstance(data, dict) and 'error' not in data:
        if 'merged_context' in data:
            print(data['merged_context'])
        elif 'nodes' in data:
            print(f"Found {len(data['nodes'])} entities")
    elif isinstance(data, list):
        print(f"Found {len(data)} matching files")
```

---

## Troubleshooting

### Issue: Knowledge Engine Initialization Fails

**Solution:**
- Check API keys in `mcp_agent.secrets.yaml`
- Verify AWS credentials for Bedrock
- Ensure LLM API keys are valid

### Issue: Graphiti Integration Not Working

**Solution:**
- Verify Graphiti bridge installation
- Check Graphiti config path
- Ensure Graphiti service is running

### Issue: Graph Visualization Too Slow

**Solution:**
- Filter entities by type
- Reduce node size multiplier
- Use simpler layout (circular vs spring)
- Limit the number of nodes displayed

### Issue: Extraction Returns Empty Results

**Solution:**
- Check document format is supported
- Ensure LLM client is initialized
- Try smaller document chunks
- Increase max_tokens in config

---

## Performance Optimization

### 1. Enable Caching

```yaml
# indexer_config.yaml
performance:
  enable_content_caching: true
  max_cache_size: 100
```

### 2. Use Concurrent Processing

```yaml
performance:
  enable_concurrent_analysis: true
  max_concurrent_files: 5
```

### 3. Filter Results

```python
# Only get high-confidence results
results = await interface.query_bedrock(
    knowledge_base_id="KB_ID",
    query_text="...",
    num_results=5  # Limit results
)
```

### 4. Use Local Indexes

```python
# Build local index once
await engine.index_project(
    project_path="./src",
    target_structure="...",
    output_dir="indexes"
)

# Query fast from local index
results = await interface.query_local_index(
    index_path="indexes/index.json",
    keyword="MCTS"
)
```

---

## Security Considerations

1. **API Key Storage:**
   - Never commit `mcp_agent.secrets.yaml` to version control
   - Use environment variables for production deployments
   - Rotate keys regularly

2. **Input Validation:**
   - All user inputs are sanitized
   - File size limits are enforced
   - Path traversal is prevented

3. **Access Control:**
   - Bedrock KB IDs should be configured per user
   - Graphiti access requires authentication
   - Local indexes should have proper permissions

---

## Future Enhancements

- [ ] Real-time collaboration on knowledge graphs
- [ ] Knowledge graph versioning and history
- [ ] Advanced NLP extraction techniques
- [ ] Multi-modal knowledge extraction (images, tables)
- [ ] Knowledge graph reasoning and inference
- [ ] Integration with more knowledge sources
- [ ] Knowledge provenance visualization
- [ ] Automated knowledge quality assessment

---

## Contributing

Contributions are welcome! Please ensure:

1. Code follows PEP 8 style guidelines
2. All functions have docstrings
3. Tests are included for new features
4. Documentation is updated

---

## License

This integration is part of the OpenEvolve project.

---

## Contact

For questions or issues, please contact the OpenEvolve integration team.

---

**Last Updated:** 2026-01-03
**Documentation Version:** 1.0.0
