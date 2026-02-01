# Knowledge Engine Bubbles for BubbleLab

## Overview

This directory contains BubbleLabs integration nodes (bubbles) that expose the OpenEvolve Knowledge Engine functionality within BubbleLab workflows.

## Available Bubbles

### 1. KnowledgeExtractionNode
**File:** `knowledge_extraction_node.py`

Extract structured knowledge from unstructured text using multiple NLP strategies.

**Operations:**
- Extract entities, relationships, and triples from text
- Support for DeepKE, OneKE, KG-Gen extractors
- Auto-extractor selection based on content

**Configuration:**
- `extractor`: enum ["deepke", "oneke", "kg_gen", "auto"]
- `min_confidence`: number (0.0-1.0, default 0.7)
- `include_metadata`: boolean (default true)
- `domain`: string (optional) - Domain hint

**Input:** `{ "text": "OpenAI released GPT-4..." }`

**Output:**
```json
{
  "triples": [
    {"subject": "OpenAI", "predicate": "released", "object": "GPT-4", "confidence": 0.95}
  ],
  "entities": ["OpenAI", "GPT-4"],
  "confidence": 0.92,
  "statistics": {"total_triples": 5, "total_entities": 3}
}
```

---

### 2. KnowledgeQueryNode
**File:** `knowledge_query_node.py`

Query the unified knowledge graph for entities, relationships, and paths.

**Operations:**
- Query triples by subject, predicate, or object
- Find paths between entities
- Get entity neighborhoods
- Export knowledge graph

**Configuration:**
- `query_type`: enum ["triples", "paths", "neighbors", "export"]
- `subject`, `predicate`, `object`: string filters
- `min_confidence`: number (0.0-1.0)
- `source_entity`, `target_entity`: for path queries
- `max_path_length`: number (default 3)
- `depth`: number (default 1) - for neighbor queries

**Input:** Depends on query_type

**Output:** Query results including triples, paths, or exported data

---

### 3. KnowledgeReasoningNode
**File:** `knowledge_reasoning_node.py`

Verify, validate, and reason over knowledge using formal methods.

**Operations:**
- Verify knowledge consistency using Z3
- Detect contradictions
- Infer new facts
- Validate logical statements

**Configuration:**
- `reasoning_type`: enum ["verify", "contradiction_check", "infer", "validate"]
- `premises`: array of strings - Starting facts
- `conclusion`: string - Statement to verify
- `include_explanation`: boolean (default true)
- `timeout`: integer (1-300 seconds)

**Input:** `{ "premises": ["All humans are mortal", "Socrates is human"], "conclusion": "Socrates is mortal" }`

**Output:**
```json
{
  "valid": true,
  "explanation": "Modus ponens: From A->B and A, derive B",
  "contradictions": [],
  "inferred": ["Socrates is mortal"],
  "confidence": 1.0
}
```

---

### 4. KnowledgeIntegrationNode
**File:** `knowledge_integration_node.py`

Integrate knowledge from multiple sources using the Unified KG Integration Hub.

**Operations:**
- Initialize knowledge hub
- Extract using multiple extractors in parallel
- Merge knowledge from multiple sources
- Export integrated knowledge
- Health check all integrations

**Configuration:**
- `operation`: enum ["initialize", "extract", "merge", "export", "health_check"]
- `extractors`: array ["deepke", "oneke", "kg_gen"]
- `export_format`: enum ["json", "triples", "networkx"]
- `enable_reasoning`: boolean (default true)
- `enable_temporal`: boolean (default true)

**Input:** Depends on operation

**Output:** Integration results, health status, or exported data

---

### 5. EntityProfileNode
**File:** `entity_profile_node.py`

Create and manage rich entity profiles with relationships and provenance.

**Operations:**
- Create entity profiles
- Update existing profiles
- Get profile by name
- Merge multiple profiles

**Configuration:**
- `operation`: enum ["create", "update", "get", "merge"]
- `entity_name`: string
- `entity_types`: array of strings
- `properties`: object - Key-value properties
- `relationships`: array of relationship objects
- `confidence`: number (0.0-1.0)

**Input:**
```json
{
  "operation": "create",
  "entity_name": "Alice Chen",
  "entity_types": ["Person", "Researcher"],
  "properties": {"expertise": ["AI", "NLP"]},
  "relationships": [{"predicate": "works_at", "target": "OpenAI"}]
}
```

**Output:** Entity profile data

---

## Workflow Examples

### Example 1: Research Paper Analysis

```
[Document Input] -> [KnowledgeExtractionNode] -> [EntityProfileNode] 
    -> [KnowledgeReasoningNode] -> [KnowledgeIntegrationNode] -> [Export]
```

1. Extract knowledge from research paper
2. Create profiles for discovered entities
3. Verify consistency of extracted facts
4. Integrate with existing knowledge base
5. Export enriched knowledge

### Example 2: Customer 360° View

```
[CRM Data] -> [EntityProfileNode: create customer profile]
    -> [KnowledgeQueryNode: find related entities]
    -> [KnowledgeReasoningNode: infer risk factors]
    -> [Report Generation]
```

### Example 3: Knowledge Base Construction

```
[Multiple Sources] -> [KnowledgeIntegrationNode: extract]
    -> [KnowledgeReasoningNode: verify]
    -> [EntityProfileNode: enrich]
    -> [KnowledgeIntegrationNode: merge]
    -> [KnowledgeQueryNode: validate]
```

---

## Implementation Details

### Base Class
All nodes inherit from `BubbleLabsNode` in `base_node.py`:
- Standardized error handling via `NodeExecutionError`
- Configuration validation
- Safe imports with fallbacks
- Progress tracking
- Health checks

### Safe Import Pattern
```python
SomeClass = self.safe_import(
    'module.path.SomeClass',
    error_msg="Description of what's missing"
)
```

### Error Handling
All nodes use `NodeExecutionError` for consistent error reporting:
```python
raise NodeExecutionError(
    node_name=self.__class__.__name__,
    message="Description of error",
    details={"context": "additional info"}
)
```

---

## Integration with BubbleLab

### Node Registration
Nodes are automatically discovered by BubbleLab based on:
- File location in `bubblelabs_nodes/`
- Inheritance from `BubbleLabsNode`
- Metadata attributes (DISPLAY_NAME, CATEGORY, etc.)

### UI Configuration
Each node provides `get_parameter_schema()` returning JSON Schema:
- UI generates configuration panels from schema
- Type-safe parameter validation
- Conditional field visibility
- Default values

### Workflow State
Nodes access workflow state via `context` parameter:
```python
def execute(self, inputs: Dict, context: 'WorkflowState') -> Dict:
    # Access previous node outputs
    previous_result = context.get_artifact('previous_node_id')
    
    # Store results for downstream nodes
    context.store_artifact('my_output', result)
    
    # Update progress
    context.update_progress(50, "Halfway done")
```

---

## Testing

### Unit Tests
Run tests for individual nodes:
```bash
cd knowledge_engine
python -m pytest tests/test_knowledge_bubbles.py -v
```

### Integration Tests
Test full workflows:
```bash
python -m pytest tests/integrations/test_knowledge_workflows.py -v
```

---

## Dependencies

### Required
- `knowledge_engine` - Core knowledge engine
- `bubblelabs_nodes/base_node.py` - Base node class

### Optional (with graceful degradation)
- `deepke`, `oneke`, `kg_gen` - Knowledge extraction
- `z3` - Formal reasoning
- `networkx` - Graph operations
- `karateclub` - Graph analytics

---

## Version History

### v1.0.0 (2026-02-01)
- Initial release
- 5 knowledge engine bubbles
- Full integration with BubbleLab
- Safe import pattern for all dependencies

---

## Support

For issues or questions:
- Check `SYSTEM_ARCHITECTURE_AND_USAGE_GUIDE.md` in knowledge_engine
- Review example workflows in `examples/`
- Run diagnostic: `python -m bubblelabs_nodes.diagnostic`
