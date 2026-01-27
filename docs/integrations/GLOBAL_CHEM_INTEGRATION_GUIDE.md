# GlobalChem Integration Guide

**Author**: Agent 7 (GlobalChem Integration Specialist)
**Date**: 2026-01-02
**Version**: 0.1.0
**Status**: Complete

---

## Table of Contents

1. [Overview](#1-overview)
2. [Purpose and Gap Analysis](#2-purpose-and-gap-analysis)
3. [Technical Implementation](#3-technical-implementation)
4. [Architecture](#4-architecture)
5. [Integration Points](#5-integration-points)
6. [Configuration](#6-configuration)
7. [SMILES/SMARTS Support](#7-smilessmarts-support)
8. [Usage Examples](#8-usage-examples)
9. [API Reference](#9-api-reference)
10. [Testing](#10-testing)
11. [Troubleshooting](#11-troubleshooting)
12. [Future Enhancements](#12-future-enhancements)

---

## 1. Overview

### What is GlobalChem?

**GlobalChem** is a community-curated chemical knowledge graph that provides:
- **50+ Chemical Lists**: Organic compounds, biomolecules, drugs, food additives, environmental chemicals, etc.
- **SMILES/SMARTS Support**: Parse and validate chemical structure notation
- **Domain-Specific Knowledge**: Specialized knowledge for chemistry and biology
- **Community-Driven**: Open-source with contributions from the scientific community

**Repository**: https://github.com/Sulstice/global-chem

### Why Integrate GlobalChem?

GlobalChem fills critical gaps in OpenEvolve's knowledge base:
- **GAP-13 (Chemical/Biological Knowledge)**: Provides comprehensive chemical knowledge
- **GAP-2 (Domain Knowledge)**: Enhances domain-specific knowledge for chemistry/biology

### Integration Approach

We use a **decoupled adapter pattern** that:
- **Zero modifications** to GlobalChem source code
- **Graceful degradation** if GlobalChem is unavailable
- **Caching** for performance optimization
- **OneKE integration** for enhanced entity recognition

---

## 2. Purpose and Gap Analysis

### Gaps Filled

#### GAP-13: Chemical/Biological Knowledge

**Before Integration**:
- No domain-specific chemical knowledge
- Limited understanding of biochemical pathways
- No chemical entity recognition

**After Integration**:
- ✅ 50+ community-curated chemical lists
- ✅ SMILES/SMARTS parsing and validation
- ✅ Chemical property prediction
- ✅ Entity recognition for chemical compounds
- ✅ Integration with biochemical knowledge

#### GAP-2: Domain Knowledge

**Enhancement**:
- Comprehensive chemistry domain knowledge
- Medicinal chemistry knowledge (drugs, scaffolds, warheads)
- Food chemistry knowledge (additives, vitamins)
- Environmental chemistry (alternative fuels, biomass)

### Priority Level

**P4 (OPTIONAL)**: 1 week effort
- Not critical for core functionality
- Enhances chemical/biological domain knowledge
- Optional but valuable for chemistry use cases

---

## 3. Technical Implementation

### Adapter Pattern

The integration uses a decoupled adapter pattern:

```
OpenEvolve ──► GlobalChemAdapter ──► GlobalChem
                    │
                    └──► Knowledge Base
```

**Key Components**:

1. **GlobalChemAdapter** (`integrations/global_chem/adapter.py`)
   - Implements `KnowledgeGraphInterface`
   - Wraps GlobalChem functionality
   - Provides SMILES/SMARTS parsing
   - Manages chemical list caching

2. **GlobalChemBridge** (`integrations/global_chem/bridge.py`)
   - Connects to OpenEvolve knowledge base
   - Provides entity recognition
   - Extracts chemical relationships
   - Integrates with OneKE

3. **Configuration** (`integrations/global_chem/config.yaml`)
   - Chemical lists to load
   - Feature toggles
   - Performance settings

### Implementation Details

#### No Modifications to GlobalChem

```python
# Add GlobalChem to path (no installation required)
global_chem_path = "../../projects to analyze/global-chem"
sys.path.insert(0, global_chem_path)

# Import and use directly
from global_chem.global_chem.global_chem import GlobalChem
```

#### Graceful Degradation

```python
try:
    from global_chem.global_chem.global_chem import GlobalChem
    GLOBAL_CHEM_AVAILABLE = True
except ImportError:
    GLOBAL_CHEM_AVAILABLE = False
    # Adapter handles unavailability gracefully
```

---

## 4. Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenEvolve Core                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │           Knowledge Graph Interface                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              GlobalChem Adapter                       │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  - Chemical List Queries                        │  │  │
│  │  │  - SMILES/SMARTS Parsing                        │  │  │
│  │  │  - Property Prediction                          │  │  │
│  │  │  - Caching Layer                                │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              GlobalChem Bridge                        │  │
│  │  ┌─────────────────────────────────────────────────┐  │  │
│  │  │  - Entity Recognition                           │  │  │
│  │  │  - Relationship Extraction                      │  │  │
│  │  │  - Knowledge Graph Generation                   │  │  │
│  │  │  - OneKE Integration                            │  │  │
│  │  └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     GlobalChem                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Organic     │  │  Bio         │  │  Medicinal   │      │
│  │  Chemistry   │  │  molecules   │  │  Chemistry   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Food        │  │  Environ-    │  │  Controlled  │      │
│  │  Chemistry   │  │  mental      │  │  Substances  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Text Input
    │
    ▼
Entity Recognition (Bridge)
    │
    ├──► SMILES Validation (Adapter)
    ├──► Chemical List Query (Adapter)
    └──► Property Prediction (Adapter)
    │
    ▼
Knowledge Graph Generation (Bridge)
    │
    ├──► Nodes: Chemical Entities
    └──► Edges: Relationships
    │
    ▼
OneKE Integration (Optional)
    │
    └──► Merge with General Entities
    │
    ▼
Knowledge Base Storage
```

---

## 5. Integration Points

### 5.1 Knowledge Graph Interface

```python
from integrations.global_chem import GlobalChemAdapter

# Initialize adapter
adapter = GlobalChemAdapter()
await adapter.initialize(config)

# Search chemical knowledge
results = await adapter.search("aspirin", num_results=10)
```

### 5.2 Entity Recognition

```python
from integrations.global_chem import GlobalChemBridge

# Initialize bridge
bridge = GlobalChemBridge(adapter)
await bridge.initialize(config)

# Recognize chemical entities
entities = await bridge.recognize_chemical_entities(
    "Aspirin reacts with acetic anhydride to form acetylsalicylic acid"
)
```

### 5.3 OneKE Integration

```python
# Integrate with OneKE for enhanced extraction
result = await bridge.integrate_with_oneke(
    text="The reaction produces acetaminophen",
    oneke_results=oneke_output
)
```

### 5.4 Knowledge Graph Generation

```python
# Generate knowledge graph from text
kg = await bridge.generate_knowledge_graph(
    "CBD is a phytocannabinoid found in Cannabis sativa"
)
```

---

## 6. Configuration

### Configuration File: `config.yaml`

```yaml
project:
  name: global-chem
  version: 0.1.0
  enabled: true

features:
  chemical_lists: true
  smiles_parsing: true
  smarts_parsing: true
  property_prediction: true
  oneke_integration: true

chemical_lists:
  - organic_and_inorganic_bronsted_acids
  - amino_acids
  - vitamins
  - phyto_cannabinoids
  # ... (see full list in config.yaml)

integration:
  auto_start: true
  oneke_integration: true
  cache_enabled: true
  cache_ttl: 3600

entity_recognition:
  enabled: true
  confidence_threshold: 0.7

performance:
  max_workers: 4
  timeout: 30
  batch_size: 100
```

### Loading Configuration

```python
import yaml

# Load config
with open('integrations/global_chem/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with config
adapter = GlobalChemAdapter()
await adapter.initialize(config)
```

### Environment Variables

```bash
# Optional: Override configuration
GLOBAL_CHEM_CACHE_ENABLED=true
GLOBAL_CHEM_CACHE_TTL=3600
GLOBAL_CHEM_ONEKE_INTEGRATION=true
```

---

## 7. SMILES/SMARTS Support

### SMILES Parsing

**SMILES** (Simplified Molecular-Input Line-Entry System) is a chemical notation system.

```python
# Parse SMILES string
result = await adapter.parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")

print(result)
# {
#     "is_valid": true,
#     "canonical_form": "CC(=O)OC1=CC=CC=C1C(=O)O",
#     "molecular_formula": "C9H8O4",
#     "molecular_weight": 180.16,
#     "source_list": "phyto_cannabinoids",
#     "error": null
# }
```

### SMARTS Parsing

**SMARTS** (SMILES Arbitrary Target Specification) is a language for specifying substructural patterns.

```python
# Parse SMARTS pattern
result = await adapter.parse_smarts("[C][C]")

print(result)
# {
#     "is_valid": true,
#     "pattern_type": "atom_query",
#     "error": null
# }
```

### SMILES/SMARTS Pipeline

```
Input SMILES/SMARTS
    │
    ▼
Validation (Basic syntax check)
    │
    ▼
Database Lookup (GlobalChem lists)
    │
    ▼
Property Extraction
    ├──► Molecular Formula
    ├──► Molecular Weight
    └──► Source List
    │
    ▼
Caching (if enabled)
    │
    ▼
Return Result
```

---

## 8. Usage Examples

### Example 1: Basic Chemical Search

```python
from integrations.global_chem import GlobalChemAdapter

# Initialize
adapter = GlobalChemAdapter()
await adapter.initialize({
    'auto_start': True,
    'cache_enabled': True
})

# Search for chemicals
results = await adapter.search("CBD", num_results=5)

for chemical in results['chemicals']:
    print(f"{chemical['name']}: {chemical['smiles']}")
```

### Example 2: Entity Recognition

```python
from integrations.global_chem import GlobalChemBridge

# Initialize bridge
bridge = GlobalChemBridge(adapter)
await bridge.initialize({
    'entity_recognition': {
        'enabled': True,
        'confidence_threshold': 0.7
    }
})

# Recognize entities in text
text = """
The reaction between acetic acid and salicylic acid produces
acetylsalicylic acid (aspirin), which is commonly used as
a pain reliever and anti-inflammatory drug.
"""

entities = await bridge.recognize_chemical_entities(text)

for entity in entities:
    print(f"{entity.name} ({entity.entity_type.value})")
    print(f"  SMILES: {entity.smiles}")
    print(f"  Confidence: {entity.confidence}")
```

### Example 3: Knowledge Graph Generation

```python
# Generate knowledge graph
kg = await bridge.generate_knowledge_graph(text)

print(f"Nodes: {kg['metadata']['num_entities']}")
print(f"Edges: {kg['metadata']['num_relationships']}")

for node in kg['nodes']:
    print(f"Entity: {node['id']} (Type: {node['type']})")

for edge in kg['edges']:
    print(f"{edge['source']} --[{edge['relationship']}]--> {edge['target']}")
```

### Example 4: OneKE Integration

```python
# Assume OneKE has extracted entities
oneke_results = {
    "entities": [
        {"name": "acetic acid", "type": "CHEMICAL"},
        {"name": "salicylic acid", "type": "CHEMICAL"}
    ]
}

# Integrate with GlobalChem
integrated = await bridge.integrate_with_oneke(
    text=text,
    oneke_results=oneke_results
)

print(f"Chemical entities: {len(integrated['chemical_entities'])}")
print(f"OneKE entities: {len(integrated['oneke_entities'])}")
```

### Example 5: Chemical List Query

```python
# Query specific chemical list
results = await adapter.query_chemical_list(
    list_name="amino_acids",
    query="glycine",
    limit=10
)

print(f"Found {results['total']} amino acids matching 'glycine'")

for chemical in results['chemicals']:
    print(f"  {chemical['name']}: {chemical['smiles']}")
```

---

## 9. API Reference

### GlobalChemAdapter

#### `__init__()`
Initialize the adapter.

#### `async initialize(config: Dict[str, Any]) -> bool`
Initialize GlobalChem with configuration.

**Parameters**:
- `config`: Configuration dictionary

**Returns**: True if successful

**Raises**:
- `ConfigurationError`: If config is invalid
- `ConnectionError`: If initialization fails

#### `async parse_smiles(smiles_string: str) -> Dict[str, Any]`
Parse and validate SMILES string.

**Parameters**:
- `smiles_string`: SMILES string

**Returns**: Parse result dictionary

**Raises**:
- `SMILESParsingError`: If parsing fails

#### `async parse_smarts(smarts_string: str) -> Dict[str, Any]`
Parse and validate SMARTS string.

**Parameters**:
- `smarts_string`: SMARTS string

**Returns**: Parse result dictionary

**Raises**:
- `SMARTSParsingError`: If parsing fails

#### `async query_chemical_list(list_name: str, query: Optional[str] = None, limit: int = 100) -> Dict[str, Any]`
Query a specific chemical list.

**Parameters**:
- `list_name`: Name of chemical list
- `query`: Optional search query
- `limit`: Maximum results

**Returns**: Query results dictionary

#### `async search(query: str, num_results: int = 10) -> Dict[str, Any]`
Search GlobalChem for chemical knowledge.

**Parameters**:
- `query`: Search query
- `num_results`: Maximum results

**Returns**: Search results dictionary

#### `async get_available_chemical_lists() -> List[str]`
Get list of available chemical lists.

**Returns**: List of chemical list names

#### `async validate() -> Dict[str, Any]`
Validate GlobalChem state.

**Returns**: Validation results dictionary

#### `async shutdown() -> bool`
Shutdown GlobalChem connection.

**Returns**: True if successful

### GlobalChemBridge

#### `__init__(adapter: GlobalChemAdapter)`
Initialize the bridge with an adapter.

#### `async initialize(config: Dict[str, Any]) -> bool`
Initialize the bridge with configuration.

#### `async recognize_chemical_entities(text: str, threshold: float = 0.7) -> List[ChemicalEntity]`
Recognize chemical entities in text.

**Parameters**:
- `text`: Input text
- `threshold`: Minimum confidence threshold

**Returns**: List of ChemicalEntity objects

#### `async extract_chemical_relationships(entities: List[ChemicalEntity], text: str) -> List[ChemicalRelationship]`
Extract relationships between chemical entities.

**Parameters**:
- `entities`: List of recognized entities
- `text`: Input text

**Returns**: List of ChemicalRelationship objects

#### `async generate_knowledge_graph(text: str) -> Dict[str, Any]`
Generate knowledge graph from text.

**Parameters**:
- `text`: Input text

**Returns**: Knowledge graph dictionary

#### `async query_chemical_knowledge(query: str, entity_type: Optional[ChemicalEntityType] = None) -> List[Dict[str, Any]]`
Query chemical knowledge base.

**Parameters**:
- `query`: Search query
- `entity_type`: Optional entity type filter

**Returns**: List of matching entities

#### `async integrate_with_oneke(text: str, oneke_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]`
Integrate with OneKE for enhanced entity recognition.

**Parameters**:
- `text`: Input text
- `oneke_results`: Optional OneKE results

**Returns**: Integrated results dictionary

#### `async get_statistics() -> Dict[str, Any]`
Get bridge statistics.

**Returns**: Statistics dictionary

#### `async shutdown() -> bool`
Shutdown the bridge.

**Returns**: True if successful

### Data Classes

#### `ChemicalEntity`
```python
@dataclass
class ChemicalEntity:
    name: str
    smiles: Optional[str]
    entity_type: ChemicalEntityType
    source_list: str
    properties: Dict[str, Any]
    confidence: float
```

#### `ChemicalRelationship`
```python
@dataclass
class ChemicalRelationship:
    source_entity: str
    relationship_type: str
    target_entity: str
    confidence: float
    metadata: Dict[str, Any]
```

#### `ChemicalEntityType` (Enum)
```python
class ChemicalEntityType(Enum):
    ORGANIC_COMPOUND = "organic_compound"
    INORGANIC_COMPOUND = "inorganic_compound"
    BIOMOLECULE = "biomolecule"
    DRUG = "drug"
    POLYMER = "polymer"
    SOLVENT = "solvent"
    NARCOTIC = "narcotic"
    FOOD_ADDITIVE = "food_additive"
    ENVIRONMENTAL_CHEMICAL = "environmental_chemical"
    WARFARE_AGENT = "warfare_agent"
    UNKNOWN = "unknown"
```

---

## 10. Testing

### Running Tests

```bash
# Run all GlobalChem integration tests
pytest tests/integrations/test_global_chem_integration.py

# Run with coverage
pytest tests/integrations/test_global_chem_integration.py --cov=integrations/global_chem

# Run specific test
pytest tests/integrations/test_global_chem_integration.py::test_smiles_parsing
```

### Test Structure

```python
# tests/integrations/test_global_chem_integration.py
import pytest
from integrations.global_chem import GlobalChemAdapter, GlobalChemBridge

@pytest.mark.asyncio
async def test_adapter_initialization():
    """Test adapter initialization."""
    adapter = GlobalChemAdapter()
    result = await adapter.initialize({'auto_start': False})
    assert result is True
    assert adapter.is_initialized is True

@pytest.mark.asyncio
async def test_smiles_parsing():
    """Test SMILES parsing."""
    adapter = GlobalChemAdapter()
    await adapter.initialize({'auto_start': False})

    result = await adapter.parse_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert result['is_valid'] is True

@pytest.mark.asyncio
async def test_entity_recognition():
    """Test entity recognition."""
    adapter = GlobalChemAdapter()
    await adapter.initialize({'auto_start': True})

    bridge = GlobalChemBridge(adapter)
    await bridge.initialize({'entity_recognition': {'enabled': True}})

    entities = await bridge.recognize_chemical_entities(
        "Aspirin is acetylsalicylic acid"
    )
    assert len(entities) > 0
```

### Test Coverage

The test suite covers:
- ✅ Adapter initialization
- ✅ SMILES/SMARTS parsing
- ✅ Chemical list queries
- ✅ Entity recognition
- ✅ Relationship extraction
- ✅ Knowledge graph generation
- ✅ OneKE integration
- ✅ Caching functionality
- ✅ Error handling
- ✅ Graceful degradation

---

## 11. Troubleshooting

### Issue 1: GlobalChem Not Available

**Symptoms**:
- `ConfigurationError: GlobalChem is not available`
- ImportError when trying to import GlobalChem

**Solutions**:
1. Verify GlobalChem is in "projects to analyze" directory
2. Check that the path is correct in adapter.py
3. Ensure no namespace conflicts

```python
# Check if GlobalChem is available
import sys
import os

global_chem_path = "../../projects to analyze/global-chem"
print(f"Path exists: {os.path.exists(global_chem_path)}")

# Add to path
sys.path.insert(0, global_chem_path)

# Try import
try:
    from global_chem.global_chem.global_chem import GlobalChem
    print("GlobalChem imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
```

### Issue 2: Empty Chemical Lists

**Symptoms**:
- Chemical list queries return empty results
- `get_available_chemical_lists()` returns empty list

**Solutions**:
1. Check if `auto_start` is enabled in config
2. Verify chemical lists are specified in config
3. Check GlobalChem data files exist

```python
# Check available lists
lists = await adapter.get_available_chemical_lists()
print(f"Available lists: {len(lists)}")

# Load specific list
await adapter.query_chemical_list("amino_acids")
```

### Issue 3: SMILES Parsing Failures

**Symptoms**:
- `SMILESParsingError: Failed to parse SMILES`
- All SMILES return `is_valid: False`

**Solutions**:
1. Verify SMILES string format
2. Check if SMILES exists in GlobalChem database
3. Use canonical SMILES format

```python
# Test with known SMILES
test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
result = await adapter.parse_smiles(test_smiles)
print(result)

# Search for SMILES
results = await adapter.search("aspirin")
```

### Issue 4: Entity Recognition Not Working

**Symptoms**:
- `recognize_chemical_entities()` returns empty list
- No entities recognized in text

**Solutions**:
1. Lower confidence threshold
2. Verify chemical lists are loaded
3. Check text contains known chemical names

```python
# Lower threshold
entities = await bridge.recognize_chemical_entities(
    text,
    threshold=0.5  # Lower from 0.7
)

# Check available entities
stats = await bridge.get_statistics()
print(f"Cached entities: {stats['cached_entities']}")
```

### Issue 5: Performance Issues

**Symptoms**:
- Slow initialization
- Slow entity recognition
- High memory usage

**Solutions**:
1. Enable caching
2. Limit chemical lists loaded
3. Use batch processing

```yaml
# config.yaml
integration:
  cache_enabled: true
  cache_ttl: 3600

chemical_lists:
  - amino_acids  # Only load necessary lists
  - vitamins

performance:
  max_workers: 4
  batch_size: 100
```

### Issue 6: OneKE Integration Failures

**Symptoms**:
- `OneKE integration failed` error
- Entities not merged with OneKE results

**Solutions**:
1. Verify OneKE is properly initialized
2. Check OneKE results format
3. Enable OneKE integration in config

```python
# Check OneKE integration
stats = await bridge.get_statistics()
print(f"OneKE enabled: {stats['oneke_integration_enabled']}")

# Test integration
result = await bridge.integrate_with_oneke(
    text="Test text",
    oneke_results={'entities': []}
)
```

---

## 12. Future Enhancements

### Planned Improvements

#### Phase 1: Enhanced SMILES/SMARTS Support (Priority: P3)
- Integrate RDKit for advanced SMILES validation
- Add molecular descriptor calculation
- Implement substructure search
- Support reaction SMILES

#### Phase 2: Advanced Property Prediction (Priority: P3)
- Integrate with chemical property databases
- Add physicochemical property prediction
- Implement toxicity prediction
- Add drug-likeness scores

#### Phase 3: Knowledge Graph Expansion (Priority: P4)
- Add biochemical pathway knowledge
- Integrate with ChEMBL database
- Add protein-ligand interaction data
- Implement reaction prediction

#### Phase 4: Performance Optimization (Priority: P2)
- Implement parallel SMILES parsing
- Add incremental list loading
- Optimize caching strategy
- Add database backend for chemical data

#### Phase 5: Enhanced Entity Recognition (Priority: P3)
- Add NER model fine-tuning for chemicals
- Implement chemical synonym resolution
- Add IUPAC name parsing
- Support chemical formula recognition

### Community Contributions

We welcome contributions to enhance the GlobalChem integration:

1. **Additional Chemical Lists**: Add new chemical list categories
2. **Property Predictors**: Contribute property prediction models
3. **Performance Improvements**: Optimize parsing and caching
4. **Documentation**: Improve guides and examples
5. **Tests**: Add test coverage for edge cases

---

## Appendix

### A. Available Chemical Lists

| Category | Lists | Description |
|----------|-------|-------------|
| Organic Chemistry | 3 | Acids, solvents, monomers |
| Biomolecules | 15 | Amino acids, vitamins, cannabis compounds |
| Medicinal Chemistry | 20 | Drugs, scaffolds, kinase inhibitors |
| Food Chemistry | 12 | Additives, vitamins, fruit compounds |
| Environmental | 5 | Fuels, biomass, interstellar |
| Controlled Substances | 7 | Narcotics, scheduled compounds |
| Warfare Agents | 1 | Nerve agents |

### B. SMILES/SMARTS Resources

- **SMILES Tutorial**: https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html
- **SMARTS Tutorial**: https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html
- **OpenSMILES**: http://opensmiles.org/

### C. Related Integrations

- **OneKE**: General entity recognition (P2 priority)
- **Graphiti**: Temporal knowledge graph (P1 priority)
- **DeepKE**: Knowledge extraction (P2 priority)

### D. References

1. GlobalChem Repository: https://github.com/Sulstice/global-chem
2. SMILES Specification: Daylight Chemical Information Systems
3. Chemical Knowledge Graphs: AI-KG and DeepKE projects

---

## Summary

The GlobalChem integration provides OpenEvolve with comprehensive chemical knowledge:

✅ **50+ Chemical Lists**: Community-curated chemical knowledge
✅ **SMILES/SMARTS Support**: Parse and validate chemical structures
✅ **Entity Recognition**: Extract chemical entities from text
✅ **Property Prediction**: Calculate molecular properties
✅ **OneKE Integration**: Enhanced with general entity recognition
✅ **Decoupled Design**: Zero modifications to GlobalChem source
✅ **Graceful Degradation**: Handles unavailability gracefully
✅ **Extensible**: Easy to add new features and lists

**Gaps Filled**: GAP-13 (Chemical/Biological Knowledge), GAP-2 (Domain Knowledge)
**Priority**: P4 (OPTIONAL)
**Effort**: 1 week
**Status**: Complete

---

**Last Updated**: 2026-01-02
**Next Review**: After user feedback and testing
