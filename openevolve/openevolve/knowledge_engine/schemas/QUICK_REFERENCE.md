# Entity Schema System - Quick Reference

## Installation

```python
from knowledge_engine.schemas import (
    EntitySchemaManager,
    Entity,
    SOFTWARE_ENGINEERING_SCHEMA,
    MATHEMATICAL_REASONING_SCHEMA,
    WORKFLOW_PROVENANCE_SCHEMA
)
```

## Common Operations

### Initialize Manager

```python
# Without config
manager = EntitySchemaManager()

# With config
manager = EntitySchemaManager(config_path="knowledge_engine/config/schemas.yaml")
```

### Register Schema

```python
manager.register_schema('domain_name', schema_dict)
# or
manager.register_schema('domain_name', EntitySchema_object)
```

### Validate Entity

```python
result = manager.validate_entity(entity, 'domain_name')

if result.is_valid:
    print("Valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

### Validate Batch

```python
result = manager.validate_entities(entities, 'domain_name')
print(f"Valid: {result.valid_count}/{result.entity_count}")
```

### Map Entities

```python
mapped = manager.map_entities(
    source_entities=entities,
    target_schema='target_domain',
    mapping_name='mapping_name'
)
```

### Merge Cross-Domain

```python
merged = manager.merge_cross_domain([
    (entities1, 'domain1'),
    (entities2, 'domain2'),
    (entities3, 'domain3')
])
```

### Generate LLM Prompt

```python
prompt = manager.generate_schema_prompt('domain_name')
```

### Export/Import Schema

```python
manager.export_schema('domain', 'output.yaml')
manager.import_schema('domain', 'input.yaml')
```

## Entity Types

### Software Engineering

- `CodeEntity`: Functions, classes, modules
- `DependencyEntity`: Imports, dependencies
- `APISchema`: API endpoints
- `BugPattern`: Bug patterns

### Mathematical Reasoning

- `TheoremEntity`: Theorems, lemmas
- `ConceptEntity`: Definitions, concepts
- `TechniqueEntity`: Methods, techniques
- `ProofStepEntity`: Proof steps

### Workflow/Provenance

- `WorkflowEntity`: Workflow definitions
- `TaskEntity`: Tasks
- `AgentEntity`: Agents, services
- `ExecutionEntity`: Executions

## Property Types

- `STRING`: Text values
- `INTEGER`: Whole numbers
- `FLOAT`: Decimal numbers
- `BOOLEAN`: True/False
- `ENUM`: Fixed set of values
- `ARRAY`: Lists
- `OBJECT`: Dictionaries
- `DATE`: Dates
- `DATETIME`: Date+Time

## Validation Results

```python
result.is_valid          # Overall validity
result.errors           # List of errors
result.warnings         # List of warnings
result.entity_count     # Total entities
result.valid_count      # Valid entities
result.invalid_count    # Invalid entities
```

## Mappings

Available mappings:
- `knowledge_engine_to_graphiti`
- `graphiti_to_knowledge_engine`
- `knowledge_engine_to_oneke`
- `oneke_to_knowledge_engine`
- `openevolve_to_neo4j`

## Configuration

Config file: `knowledge_engine/config/schemas.yaml`

Key settings:
- `default_schema`: Default schema to use
- `schemas.{domain}.enabled`: Enable/disable schemas
- `schemas.{domain}.validation_strict`: Strict validation
- `validation.auto_validate`: Auto-validate on creation
- `cross_domain.enabled`: Enable cross-domain merging

## Creating Custom Schema

```python
from knowledge_engine.schemas.base import (
    EntitySchema,
    EntityType,
    PropertyDefinition,
    PropertyType
)

schema = EntitySchema(
    domain="my_domain",
    description="My custom schema",
    entity_types={
        "MyEntity": EntityType(
            name="MyEntity",
            properties={
                "name": PropertyDefinition(
                    name="name",
                    type=PropertyType.STRING,
                    required=True
                )
            }
        )
    }
)
```

## Testing

```bash
# Run all tests
python -m pytest knowledge_engine/schemas/test_schemas.py -v

# Run specific test
python -m pytest knowledge_engine/schemas/test_schemas.py::TestEntitySchemaManager -v

# Run with coverage
python -m pytest knowledge_engine/schemas/test_schemas.py --cov=knowledge_engine.schemas
```

## Examples

See `knowledge_engine/schemas/example_usage.py` for complete examples.

## Documentation

Full documentation: `knowledge_engine/schemas/README.md`
