# OneKE Integration Guide

## Overview

OneKE (One Knowledge Extraction) is a schema-guided information extraction framework that provides powerful capabilities for extracting structured knowledge from unstructured text. It has been integrated into OpenEvolve to enhance knowledge extraction across physics and chemistry domains.

### What is OneKE?

OneKE is an open-source framework developed by ZJU NLP that supports:
- **Named Entity Recognition (NER)**: Identify and classify entities in text
- **Relation Extraction (RE)**: Extract relationships between entities
- **Event Extraction (EE)**: Identify and classify events
- **Triple Extraction**: Extract subject-relation-object triples
- **Schema-Guided Extraction**: Custom schemas for domain-specific extraction
- **Multi-Agent Workflows**: Parallel extraction with specialized agents

### Repository
- **GitHub**: https://github.com/zjunlp/OneKE
- **License**: Apache 2.0
- **Language**: Python

## Purpose

OneKE fills two critical gaps in OpenEvolve:

### GAP-2: Physics Domain Knowledge
OneKE provides schema-guided extraction of:
- Quantum systems and states
- Observables and operators
- Dynamical equations and their properties
- Approximation methods and symmetries
- Computational physics concepts

This enables OpenEvolve to understand and reason about physics problems, extract relevant concepts from solutions, and build a knowledge base of physics domain expertise.

### GAP-10: Knowledge Extraction
OneKE enhances OpenEvolve's knowledge extraction capabilities by:
- Extracting structured knowledge from workflow executions
- Identifying solution patterns and approaches
- Building knowledge graphs from unstructured text
- Supporting domain-specific schemas
- Providing multi-agent extraction workflows

This enables better knowledge capture, reuse, and transfer across workflow executions.

## Technical Implementation

### Architecture

The OneKE integration follows a **decoupled adapter pattern**:

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenEvolve Workflow                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              workflow_knowledge_extractor.py                 │
│  (Extracts knowledge from workflow executions)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   OneKEBridge                                │
│  (High-level bridge for workflow integration)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  OneKEAdapter                                │
│  (Implements ExtractionInterface)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      OneKE Framework                         │
│  (Schema-guided information extraction)                     │
└─────────────────────────────────────────────────────────────┘
```

### Components

#### 1. Base Interface (`integrations/base/extraction_interface.py`)
- **ExtractionInterface**: Abstract interface for all extraction implementations
- **ExtractionResult**: Dataclass for extraction results
- **SchemaDefinition**: Dataclass for schema definitions
- **ExtractionType**: Enum for extraction types (NER, RE, EE, Triple, Schema)

#### 2. OneKE Adapter (`integrations/oneke/adapter.py`)
- **OneKEAdapter**: Implementation of ExtractionInterface for OneKE
- Features:
  - Async API for all extraction methods
  - Batch extraction support
  - Schema loading from YAML
  - Fallback extraction on errors
  - Docker/Conda environment support
  - Connection pooling and caching

#### 3. OneKE Bridge (`integrations/oneke/bridge.py`)
- **OneKEBridge**: High-level bridge for workflow integration
- Features:
  - Workflow-to-text conversion
  - Domain-specific extraction methods
  - Batch processing of workflows
  - Integration with workflow_knowledge_extractor.py
  - Result caching

#### 4. Schemas (`integrations/oneke/schemas/`)
- **physics.yaml**: Physics concepts schema
- **chemistry.yaml**: Chemical entities schema
- **relations.yaml**: General relations schema

### Key Design Principles

1. **Zero Modifications**: No changes to OneKE source code
2. **Decoupled**: Adapter pattern for clean separation
3. **Extensible**: Easy to add new schemas and extraction types
4. **Performant**: Async API, batching, caching
5. **Reliable**: Fallback extraction on errors
6. **Testable**: Clear interfaces and separation of concerns

## Integration Points

### 1. Workflow Knowledge Extractor

The main integration point is with `workflow_knowledge_extractor.py`:

```python
from integrations.oneke import OneKEBridge
from workflow_structures import WorkflowState

# Initialize bridge
bridge = OneKEBridge()
await bridge.initialize()

# Extract from workflow
workflow = WorkflowState(...)
knowledge = await bridge.extract_from_workflow(workflow)

# Extract physics domain knowledge
physics_knowledge = await bridge.extract_physics_knowledge(workflow)

# Extract solution patterns
patterns = await bridge.extract_solution_patterns(workflow, domain='physics')
```

### 2. Direct Adapter Usage

For fine-grained control, use the adapter directly:

```python
from integrations.oneke import OneKEAdapter

# Initialize adapter
adapter = OneKEAdapter(config_path='integrations/oneke/config.yaml')
await adapter.initialize()

# Load schema
schema = adapter.load_schema('integrations/oneke/schemas/physics.yaml')

# Perform extraction
result = await adapter.extract_schema_guided(
    text="The harmonic oscillator has energy levels E_n = ℏω(n + 1/2)",
    schema=schema
)

# Access results
print(result.entities)
print(result.relations)
print(result.confidence)
```

### 3. Batch Processing

Process multiple workflows efficiently:

```python
workflows = [workflow1, workflow2, workflow3]

results = await bridge.batch_extract_from_workflows(
    workflows,
    schemas=['physics_concepts', 'relations']
)
```

## Configuration

### Configuration File

Located at `integrations/oneke/config.yaml`:

```yaml
project:
  name: OneKE
  version: 0.1.0
  enabled: true

connection:
  model_category: ChatGPT
  model_name_or_path: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  docker: false
  conda_env: oneke

features:
  ner: true
  re: true
  ee: true
  triple: true
  multi_agent: true

schemas:
  - physics_concepts
  - chemical_entities
  - relations

integration:
  auto_start: true
  cache_enabled: true
  cache_ttl: 3600
  fallback_on_error: true

performance:
  max_workers: 4
  timeout: 30
  batch_size: 100
```

### Environment Variables

Set these in your environment or `.env` file:

```bash
# Required for ChatGPT models
export OPENAI_API_KEY="your-api-key"

# Optional: OneKE installation path
export ONEKE_PATH="/path/to/OneKE"

# Optional: Conda environment name
export ONEKE_CONDA_ENV="oneke"
```

### Configuration Options

#### Project
- **enabled**: Enable/disable OneKE integration
- **version**: Integration version

#### Connection
- **model_category**: Model type (ChatGPT, LLaMA, etc.)
- **model_name_or_path**: Model identifier
- **api_key**: API key for the model
- **docker**: Use Docker environment (default: false)
- **conda_env**: Conda environment name (if not using Docker)

#### Features
- **ner**: Enable Named Entity Recognition
- **re**: Enable Relation Extraction
- **ee**: Enable Event Extraction
- **triple**: Enable Triple Extraction
- **multi_agent**: Enable multi-agent workflows

#### Integration
- **auto_start**: Auto-start OneKE server on initialization
- **cache_enabled**: Enable result caching
- **cache_ttl**: Cache time-to-live in seconds
- **fallback_on_error**: Use fallback extraction on errors

#### Performance
- **max_workers**: Maximum parallel workers for batch processing
- **timeout**: Request timeout in seconds
- **batch_size**: Batch size for batch extraction

## Schema Definitions

### Schema Structure

Schemas are defined in YAML files:

```yaml
name: schema_name
description: Schema description

entity_types:
  - name: entity_type_name
    description: Entity type description
    examples:
      - example1
      - example2

relation_types:
  - name: relation_type_name
    description: Relation type description
    examples:
      - example1

event_types:
  - name: event_type_name
    description: Event type description
    attributes:
      - attribute1
      - attribute2

constraints:
  min_confidence: 0.7
  domains:
    - domain1
    - domain2

examples:
  - text: Example text
    entities:
      - text: entity text
        type: entity_type
    relations:
      - source: entity1
        type: relation
        target: entity2
```

### Physics Schema

**Location**: `integrations/oneke/schemas/physics.yaml`

**Entity Types**:
- quantum_system: Quantum mechanical systems
- observable: Physical observables
- state: Quantum states and wavefunctions
- operator: Mathematical operators
- dynamic_equation: Equations of motion
- approximation_method: Approximation techniques
- symmetry: Symmetries and conservation laws

**Relation Types**:
- describes: Observable describes system
- evolves_by: State evolves by equation
- acts_on: Operator acts on state
- approximates: Method approximates system
- conserves: Symmetry conserves quantity
- couples: Systems couple through interaction

### Chemistry Schema

**Location**: `integrations/oneke/schemas/chemistry.yaml`

**Entity Types**:
- substance: Chemical compounds
- element: Chemical elements
- functional_group: Organic functional groups
- reaction: Chemical reactions
- catalyst: Catalysts and enzymes
- property: Chemical properties
- structure: Molecular structures

**Relation Types**:
- contains: Substance contains element
- undergoes: Substance undergoes reaction
- catalyzes: Catalyst catalyzes reaction
- has_property: Substance has property
- reacts_with: Substances react
- decomposes_to: Substance decomposes to products

### Relations Schema

**Location**: `integrations/oneke/schemas/relations.yaml`

**Entity Types**:
- concept: General concepts
- property: Properties and attributes
- constraint: Constraints and requirements

**Relation Types**:
- causes: Causal relationship
- enables: Enablement relationship
- requires: Dependency relationship
- improves: Improvement relationship
- reduces: Reduction relationship
- part_of: Compositional relationship
- similar_to: Similarity relationship
- different_from: Difference relationship

### Creating Custom Schemas

To create a custom schema:

1. Create a new YAML file in `integrations/oneke/schemas/`
2. Define the schema structure
3. Add entity types with examples
4. Add relation types with examples
5. Add constraints and examples
6. Reference the schema in config.yaml

Example:

```yaml
name: biology_entities
description: Extract biological entities

entity_types:
  - name: gene
    description: Gene
    examples:
      - BRCA1
      - TP53

  - name: protein
    description: Protein
    examples:
      - hemoglobin
      - insulin

relation_types:
  - name: encodes
    description: Gene encodes protein
    examples:
      - BRCA1 encodes protein

constraints:
  min_confidence: 0.7

examples:
  - text: The BRCA1 gene encodes a protein involved in DNA repair.
    entities:
      - text: BRCA1
        type: gene
      - text: protein
        type: protein
    relations:
      - source: BRCA1
        type: encodes
        target: protein
```

## Usage Examples

### Example 1: Extract Physics Knowledge

```python
from integrations.oneke import OneKEBridge
from workflow_structures import WorkflowState

# Create workflow
workflow = WorkflowState(
    workflow_id="phys_001",
    problem_statement="Solve the quantum harmonic oscillator using the variational method",
    final_solution="The ground state energy is E_0 = (1/2)ℏω"
)

# Initialize bridge
bridge = OneKEBridge()
await bridge.initialize()

# Extract physics knowledge
physics = await bridge.extract_physics_knowledge(workflow)

print("Concepts:", physics['concepts'])
print("Observables:", physics['observables'])
print("Dynamics:", physics['dynamics'])
print("Quantum:", physics['quantum'])
```

### Example 2: Extract Chemical Knowledge

```python
# Extract from chemistry workflow
workflow = WorkflowState(
    workflow_id="chem_001",
    problem_statement="Analyze the combustion reaction of methane",
    final_solution="CH4 + 2O2 → CO2 + 2H2O"
)

chemistry = await bridge.extract_chemistry_knowledge(workflow)

print("Substances:", chemistry['substances'])
print("Reactions:", chemistry['reactions'])
print("Properties:", chemistry['properties'])
```

### Example 3: Named Entity Recognition

```python
from integrations.oneke import OneKEAdapter

adapter = OneKEAdapter()
await adapter.initialize()

schema = adapter.load_schema('integrations/oneke/schemas/physics.yaml')

text = """
The hydrogen atom is a quantum system with energy levels
E_n = -13.6 eV / n^2. The Hamiltonian includes kinetic and
potential energy terms.
"""

result = await adapter.extract_ner(text, schema)

for entity in result.entities:
    print(f"{entity['text']} ({entity['type']})")
```

### Example 4: Relation Extraction

```python
result = await adapter.extract_re(text, schema)

for relation in result.relations:
    print(f"{relation['source']} --{relation['type']}--> {relation['target']}")
```

### Example 5: Schema-Guided Extraction

```python
from integrations.oneke.adapter import SchemaDefinition

# Custom schema
custom_schema = SchemaDefinition(
    name='algorithm_analysis',
    description='Extract algorithm properties',
    entity_types=[
        {'name': 'algorithm', 'description': 'Algorithm'},
        {'name': 'complexity', 'description': 'Time complexity'}
    ],
    relation_types=[
        {'name': 'has_complexity', 'description': 'Algorithm has complexity'}
    ]
)

result = await adapter.extract_schema_guided(
    text="QuickSort has average time complexity O(n log n)",
    schema=custom_schema
)
```

### Example 6: Batch Extraction

```python
texts = [
    "Text 1 about physics...",
    "Text 2 about chemistry...",
    "Text 3 about algorithms..."
]

results = await adapter.batch_extract(
    texts=texts,
    extraction_type=ExtractionType.NER,
    schema=schema
)

for i, result in enumerate(results):
    print(f"Text {i}: {len(result.entities)} entities")
```

### Example 7: Integration with Workflow Knowledge Extractor

```python
from workflow_knowledge_extractor import WorkflowKnowledgeExtractor
from integrations.oneke import OneKEBridge

# Initialize both
extractor = WorkflowKnowledgeExtractor()
bridge = OneKEBridge()
await bridge.initialize()

# Extract standard artifacts
artifacts = extractor.extract_all_knowledge(workflow)

# Extract domain knowledge
domain_knowledge = await bridge.extract_from_workflow(
    workflow,
    schemas=['physics_concepts', 'relations']
)

# Combine results
combined_knowledge = {
    'standard_artifacts': artifacts,
    'domain_knowledge': domain_knowledge
}
```

## API Reference

### OneKEAdapter

#### Methods

##### `async initialize(config: Optional[Dict] = None) -> bool`
Initialize the adapter with optional configuration override.

**Returns**: True if successful

**Raises**:
- `ConfigurationError`: Invalid configuration
- `ConnectionError`: Connection failed

##### `async extract_ner(text: str, schema: Optional[SchemaDefinition] = None) -> ExtractionResult`
Perform Named Entity Recognition.

**Parameters**:
- `text`: Input text
- `schema`: Optional schema

**Returns**: ExtractionResult with entities

##### `async extract_re(text: str, schema: Optional[SchemaDefinition] = None) -> ExtractionResult`
Perform Relation Extraction.

**Parameters**:
- `text`: Input text
- `schema`: Optional schema

**Returns**: ExtractionResult with relations

##### `async extract_ee(text: str, schema: Optional[SchemaDefinition] = None) -> ExtractionResult`
Perform Event Extraction.

**Parameters**:
- `text`: Input text
- `schema`: Optional schema

**Returns**: ExtractionResult with events

##### `async extract_triple(text: str, schema: Optional[SchemaDefinition] = None) -> ExtractionResult`
Perform Triple Extraction.

**Parameters**:
- `text`: Input text
- `schema`: Optional schema

**Returns**: ExtractionResult with triples

##### `async extract_schema_guided(text: str, schema: SchemaDefinition) -> ExtractionResult`
Perform schema-guided extraction (most flexible).

**Parameters**:
- `text`: Input text
- `schema`: Schema definition

**Returns**: ExtractionResult with all extracted info

##### `async batch_extract(texts: List[str], extraction_type: ExtractionType, schema: Optional[SchemaDefinition] = None) -> List[ExtractionResult]`
Perform batch extraction.

**Parameters**:
- `texts`: List of input texts
- `extraction_type`: Type of extraction
- `schema`: Optional schema

**Returns**: List of ExtractionResult objects

##### `async validate() -> Dict[str, Any]`
Validate adapter configuration and connection.

**Returns**: Validation results dictionary

##### `async shutdown() -> bool`
Shutdown the adapter.

**Returns**: True if successful

##### `load_schema(schema_path: str) -> SchemaDefinition`
Load schema from YAML file.

**Parameters**:
- `schema_path`: Path to schema YAML

**Returns**: SchemaDefinition object

### OneKEBridge

#### Methods

##### `async initialize() -> bool`
Initialize the bridge and adapter.

**Returns**: True if successful

##### `async extract_from_workflow(workflow: Union[WorkflowState, Dict], schemas: Optional[List[str]] = None) -> Dict[str, ExtractionResult]`
Extract knowledge from workflow.

**Parameters**:
- `workflow`: WorkflowState or dictionary
- `schemas`: Optional list of schema names

**Returns**: Dictionary of schema results

##### `async extract_physics_knowledge(workflow: Union[WorkflowState, Dict]) -> Dict[str, Any]`
Extract physics domain knowledge.

**Parameters**:
- `workflow`: Workflow state or dictionary

**Returns**: Physics knowledge dictionary

##### `async extract_chemistry_knowledge(workflow: Union[WorkflowState, Dict]) -> Dict[str, Any]`
Extract chemistry domain knowledge.

**Parameters**:
- `workflow`: Workflow state or dictionary

**Returns**: Chemistry knowledge dictionary

##### `async extract_solution_patterns(workflow: Union[WorkflowState, Dict], domain: str = 'general') -> Dict[str, Any]`
Extract solution patterns.

**Parameters**:
- `workflow`: Workflow state or dictionary
- `domain`: Domain for extraction

**Returns**: Pattern information dictionary

##### `async batch_extract_from_workflows(workflows: List[Union[WorkflowState, Dict]], schemas: Optional[List[str]] = None) -> List[Dict[str, ExtractionResult]]`
Extract from multiple workflows.

**Parameters**:
- `workflows`: List of workflows
- `schemas`: Optional schema names

**Returns**: List of extraction results

##### `async validate_integration() -> Dict[str, Any]`
Validate OneKE integration.

**Returns**: Validation results

##### `async shutdown() -> bool`
Shutdown the bridge.

**Returns**: True if successful

### Data Classes

#### ExtractionResult

```python
@dataclass
class ExtractionResult:
    extraction_type: ExtractionType
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    triples: List[Dict[str, Any]]
    schema: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]
    raw_response: Optional[Dict[str, Any]] = None
```

#### SchemaDefinition

```python
@dataclass
class SchemaDefinition:
    name: str
    description: str
    entity_types: List[Dict[str, Any]]
    relation_types: Optional[List[Dict[str, Any]]] = None
    event_types: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    examples: Optional[List[Dict[str, Any]]] = None
```

## Testing

### Unit Tests

Located at `tests/integrations/test_oneke_integration.py`:

```python
import pytest
from integrations.oneke import OneKEAdapter, OneKEBridge

@pytest.mark.asyncio
async def test_adapter_initialization():
    adapter = OneKEAdapter()
    assert await adapter.initialize() == True

@pytest.mark.asyncio
async def test_ner_extraction():
    adapter = OneKEAdapter()
    await adapter.initialize()

    schema = adapter.load_schema('integrations/oneke/schemas/physics.yaml')
    result = await adapter.extract_ner("The harmonic oscillator...", schema)

    assert len(result.entities) > 0
    assert result.confidence > 0

@pytest.mark.asyncio
async def test_physics_extraction():
    bridge = OneKEBridge()
    await bridge.initialize()

    workflow = create_test_workflow()
    knowledge = await bridge.extract_physics_knowledge(workflow)

    assert 'concepts' in knowledge
    assert 'observables' in knowledge
```

### Integration Tests

Test the full integration with workflow_knowledge_extractor.py:

```python
@pytest.mark.asyncio
async def test_workflow_integration():
    from workflow_knowledge_extractor import WorkflowKnowledgeExtractor
    from integrations.oneke import OneKEBridge

    extractor = WorkflowKnowledgeExtractor()
    bridge = OneKEBridge()
    await bridge.initialize()

    workflow = create_test_workflow()

    # Extract standard artifacts
    standard = extractor.extract_all_knowledge(workflow)

    # Extract domain knowledge
    domain = await bridge.extract_from_workflow(workflow)

    # Verify integration
    assert 'solution_patterns' in standard
    assert 'physics_concepts' in domain
```

### Running Tests

```bash
# Run all OneKE tests
pytest tests/integrations/test_oneke_integration.py -v

# Run specific test
pytest tests/integrations/test_oneke_integration.py::test_physics_extraction -v

# Run with coverage
pytest tests/integrations/test_oneke_integration.py --cov=integrations/oneke
```

### Manual Testing

Test the integration manually:

```python
import asyncio
from integrations.oneke import OneKEBridge

async def test():
    bridge = OneKEBridge()
    await bridge.initialize()

    # Validate
    validation = await bridge.validate_integration()
    print(validation)

    # Test extraction
    result = await bridge.extract_physics_knowledge(test_workflow)
    print(result)

asyncio.run(test())
```

## Troubleshooting

### Common Issues

#### 1. OneKE Not Found

**Error**: `OneKE not found. Please install OneKE or set ONEKE_PATH environment variable.`

**Solutions**:
- Set `ONEKE_PATH` environment variable
- Install OneKE: `git clone https://github.com/zjunlp/OneKE.git`
- Update `oneke_path` in config.yaml

#### 2. API Key Not Set

**Error**: `OPENAI_API_KEY not set`

**Solutions**:
- Set environment variable: `export OPENAI_API_KEY="your-key"`
- Add to `.env` file: `OPENAI_API_KEY=your-key`
- Update `api_key` in config.yaml

#### 3. Schema Loading Failed

**Error**: `Failed to load schema from physics.yaml`

**Solutions**:
- Verify YAML syntax
- Check schema file exists
- Validate schema structure
- Check file permissions

#### 4. Extraction Timeout

**Error**: `Extraction timeout after 30 seconds`

**Solutions**:
- Increase `timeout` in config.yaml
- Reduce batch size
- Check network connectivity
- Use faster model

#### 5. Low Confidence Results

**Issue**: Extraction results have low confidence (< 0.5)

**Solutions**:
- Improve schema with more examples
- Use more specific entity/relation types
- Adjust `min_confidence` constraint
- Use better model (e.g., gpt-4 instead of gpt-4o-mini)

#### 6. Docker Issues

**Error**: Docker container fails to start

**Solutions**:
- Use Conda instead: Set `docker: false` in config.yaml
- Check Docker is installed and running
- Verify container image
- Check port availability

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

adapter = OneKEAdapter()
await adapter.initialize()
```

### Validation

Always validate the integration:

```python
validation = await bridge.validate_integration()
print(validation)
```

Check:
- `is_valid`: Overall status
- `checks`: Individual checks
- `issues`: List of issues

## Future Enhancements

### Planned Features

1. **Additional Schemas**
   - Biology entities (genes, proteins)
   - Mathematical concepts (theorems, proofs)
   - Computer science (algorithms, data structures)

2. **Enhanced Extraction**
   - Few-shot learning support
   - Active learning for schema refinement
   - Confidence calibration
   - Ensemble extraction

3. **Performance**
   - GPU acceleration
   - Distributed processing
   - Streaming extraction
   - Incremental updates

4. **Integration**
   - Real-time extraction during workflows
   - Knowledge graph construction
   - Automatic schema discovery
   - Multi-modal extraction (text + code)

5. **UI**
   - Schema editor interface
   - Extraction result visualization
   - Interactive debugging
   - Performance dashboard

### Contribution Guidelines

To add new features:

1. Update this guide
2. Add unit tests
3. Update API reference
4. Add usage examples
5. Document breaking changes

### Roadmap

- **Q1 2025**: Core integration complete
- **Q2 2025**: Additional schemas and performance improvements
- **Q3 2025**: Knowledge graph integration
- **Q4 2025**: Multi-modal extraction

## References

- **OneKE Paper**: [Link to paper]
- **OneKE Documentation**: https://github.com/zjunlp/OneKE
- **OpenEvolve GAP Analysis**: PROJECT_GAP_ANALYSIS_AND_RECOMMENDATIONS.md
- **Integration Tasks**: MULTI_AGENT_INTEGRATION_TASK.md

## Support

For issues or questions:
1. Check this guide
2. Review examples
3. Check OneKE documentation
4. Open GitHub issue

---

**Version**: 0.1.0
**Last Updated**: 2025-01-02
**Author**: OpenEvolve Integration Team
