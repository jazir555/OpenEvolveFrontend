# Knowledge Engine Schema System

Unified entity schema management system for OpenEvolve's knowledge graph projects.

## Overview

The Schema System provides a unified way to define, validate, and manage entity schemas across multiple knowledge graph domains including:
- **Software Engineering** (code, dependencies, APIs, bugs)
- **Mathematical Reasoning** (theorems, concepts, proofs)
- **Workflow/Provenance** (workflows, tasks, agents, executions)

## Features

- **Schema Definition**: Define entity types with properties, validation rules, and examples
- **Entity Validation**: Validate entities against schema definitions
- **Cross-Project Mapping**: Map entities between different schema systems (Graphiti, OneKE, etc.)
- **Cross-Domain Integration**: Merge entities from multiple domains
- **Schema Migration**: Validate and plan schema migrations
- **LLM Prompt Generation**: Generate extraction prompts for LLMs

## Installation

The schema system is part of the Knowledge Engine:

```python
from knowledge_engine.schemas import (
    EntitySchemaManager,
    SOFTWARE_ENGINEERING_SCHEMA,
    MATHEMATICAL_REASONING_SCHEMA,
    WORKFLOW_PROVENANCE_SCHEMA
)
```

## Quick Start

### 1. Initialize Schema Manager

```python
from knowledge_engine.schemas import EntitySchemaManager

# Initialize with config
manager = EntitySchemaManager(
    config_path="knowledge_engine/config/schemas.yaml"
)

# Or initialize without config
manager = EntitySchemaManager()
```

### 2. Register Schemas

```python
from knowledge_engine.schemas import SOFTWARE_ENGINEERING_SCHEMA

# Register schema from dict
manager.register_schema(
    'software_engineering',
    SOFTWARE_ENGINEERING_SCHEMA.to_dict()
)

# Or register from EntitySchema object
manager.register_schema('software_engineering', SOFTWARE_ENGINEERING_SCHEMA)
```

### 3. Validate Entities

```python
from knowledge_engine.schemas import Entity

# Create entity
entity = Entity(
    entity_id="func-001",
    entity_type="CodeEntity",
    properties={
        "name": "calculate_hash",
        "code_type": "function",
        "language": "Python",
        "file_path": "src/utils/crypto.py"
    }
)

# Validate
result = manager.validate_entity(entity, 'software_engineering')

if result.is_valid:
    print("Entity is valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

### 4. Map Entities Between Schemas

```python
# Register mapping
manager.register_mapping(
    'knowledge_engine_to_graphiti',
    {
        'CodeEntity': 'Activity',
        'TheoremEntity': 'Requirement',
        'WorkflowEntity': 'Document'
    }
)

# Map entities
mapped_entities = manager.map_entities(
    source_entities=entities,
    target_schema='graphiti',
    mapping_name='knowledge_engine_to_graphiti'
)
```

### 5. Merge Cross-Domain Entities

```python
# Merge entities from different domains
merged = manager.merge_cross_domain([
    (software_entities, 'software_engineering'),
    (workflow_entities, 'workflow_provenance'),
    (math_entities, 'mathematical_reasoning')
])
```

## Schema Definitions

### Software Engineering Schema

Domain: `software_engineering`

**Entity Types:**
- `CodeEntity`: Functions, classes, modules, packages
- `DependencyEntity`: Import and dependency relationships
- `APISchema`: API endpoints and specifications
- `BugPattern`: Known bug patterns with fixes

**Relationship Types:**
- `calls`: Function/method calls
- `imports`: Import dependencies
- `implements`: Implementation relationships
- `exposes`: API exposure
- `has_bug_pattern`: Bug pattern associations

**Example:**

```python
from knowledge_engine.schemas import Entity

code_entity = Entity(
    entity_id="func-001",
    entity_type="CodeEntity",
    properties={
        "name": "calculate_hash",
        "code_type": "function",
        "signature": "calculate_hash(data: bytes, algorithm: str) -> str",
        "file_path": "src/utils/crypto.py",
        "line_start": 42,
        "line_end": 58,
        "language": "Python",
        "complexity": 3
    }
)
```

### Mathematical Reasoning Schema

Domain: `mathematical_reasoning`

**Entity Types:**
- `TheoremEntity`: Theorems, lemmas, propositions
- `ConceptEntity`: Definitions, axioms, concepts
- `TechniqueEntity`: Mathematical techniques and methods
- `ProofStepEntity`: Individual proof steps

**Relationship Types:**
- `uses`: Uses theorems/concepts
- `generalizes`: Generalization relationships
- `defines`: Concept definitions
- `has_proof_step`: Proof structure

**Example:**

```python
theorem = Entity(
    entity_id="thm-pythagorean",
    entity_type="TheoremEntity",
    properties={
        "name": "Pythagorean Theorem",
        "theorem_type": "theorem",
        "statement": "In a right-angled triangle, the square of the hypotenuse equals the sum of squares of the other two sides",
        "proof_status": "proven",
        "domain": "geometry"
    }
)
```

### Workflow/Provenance Schema

Domain: `workflow_provenance`

**Entity Types:**
- `WorkflowEntity`: Workflow definitions
- `TaskEntity`: Individual tasks
- `AgentEntity`: Agents and services
- `ExecutionEntity`: Execution records

**Relationship Types:**
- `contains`: Workflow contains tasks
- `executed_by`: Task execution by agents
- `has_execution`: Execution records
- `preceded_by`: Temporal ordering

**Example:**

```python
workflow = Entity(
    entity_id="workflow-001",
    entity_type="WorkflowEntity",
    properties={
        "workflow_id": "code-analysis-pipeline",
        "name": "Code Analysis Pipeline",
        "stages": ["parse", "analyze", "report"],
        "status": "active"
    }
)
```

## Defining Custom Schemas

### 1. Define Property Types

```python
from knowledge_engine.schemas.base import (
    PropertyDefinition,
    PropertyType,
    EntityType,
    EntitySchema,
    RelationshipType
)

# Define properties
name_prop = PropertyDefinition(
    name="name",
    type=PropertyType.STRING,
    required=True,
    description="Entity name",
    validation_pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$"
)

age_prop = PropertyDefinition(
    name="age",
    type=PropertyType.INTEGER,
    required=False,
    min_value=0,
    max_value=150
)
```

### 2. Define Entity Types

```python
user_type = EntityType(
    name="User",
    description="User account",
    properties={
        "username": PropertyDefinition(
            name="username",
            type=PropertyType.STRING,
            required=True
        ),
        "email": PropertyDefinition(
            name="email",
            type=PropertyType.STRING,
            required=True,
            validation_pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
    },
    examples=[
        {
            "username": "john_doe",
            "email": "john@example.com"
        }
    ]
)
```

### 3. Define Relationship Types

```python
follows_rel = RelationshipType(
    name="follows",
    description="User follows another user",
    source_types=["User"],
    target_types=["User"],
    directed=True
)
```

### 4. Create Schema

```python
schema = EntitySchema(
    domain="social_network",
    description="Social network schema",
    version="1.0.0",
    entity_types={
        "User": user_type
    },
    relationship_types={
        "follows": follows_rel
    }
)
```

## Validation Rules

### Property Validation

Properties support automatic validation:

```python
from knowledge_engine.schemas.base import PropertyDefinition, PropertyType, ValidationRule

# String with pattern
email_prop = PropertyDefinition(
    name="email",
    type=PropertyType.STRING,
    required=True,
    validation_pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Numeric with range
age_prop = PropertyDefinition(
    name="age",
    type=PropertyType.INTEGER,
    min_value=0,
    max_value=150
)

# Enum
status_prop = PropertyDefinition(
    name="status",
    type=PropertyType.ENUM,
    allowed_values=["active", "inactive", "pending"]
)
```

### Custom Validation Rules

```python
def username_not_admin(entity_data):
    username = entity_data.get("username", "")
    if username.lower() == "admin":
        return False, "Username cannot be 'admin'"
    return True, None

user_type = EntityType(
    name="User",
    properties={"username": username_prop},
    validation_rules=[
        ValidationRule(
            name="username_not_admin",
            description="Username cannot be admin",
            validator=username_not_admin,
            severity="error"
        )
    ]
)
```

## Cross-Project Mappings

### Available Mappings

- `knowledge_engine_to_graphiti`: Map to Graphiti schema
- `graphiti_to_knowledge_engine`: Map from Graphiti schema
- `knowledge_engine_to_oneke`: Map to OneKE schema
- `oneke_to_knowledge_engine`: Map from OneKE schema
- `openevolve_to_neo4j`: Map to Neo4j property graph

### Using Mappings

```python
from knowledge_engine.schemas.schema_mappings import (
    ENTITY_MAPPINGS,
    get_mapping,
    apply_mapping
)

# Get mapping
mapping = get_mapping('knowledge_engine_to_graphiti')

# Apply mapping
mapped_type = apply_mapping('CodeEntity', mapping)
# Returns: 'Activity'

# Register in manager
manager.register_mapping('my_mapping', {
    'MyEntity': 'TargetEntity'
})
```

### Creating Custom Mappings

```python
manager.register_mapping(
    'custom_to_neo4j',
    {
        'CodeEntity': 'Node',
        'DependencyEntity': 'RELATIONSHIP',
        'APISchema': 'Node'
    }
)

# Use mapping
mapped = manager.map_entities(
    source_entities=entities,
    target_schema='neo4j',
    mapping_name='custom_to_neo4j'
)
```

## Schema Migration

### Validating Migrations

```python
from knowledge_engine.schemas.validators import SchemaMigrationValidator

# Validate migration from v1 to v2
result = SchemaMigrationValidator.validate_migration(
    old_schema=v1_schema,
    new_schema=v2_schema
)

if not result.is_valid:
    for error in result.errors:
        print(f"Breaking change: {error}")
```

### Migration Planning

```python
# Generate migration plan
plan = SchemaMigrationValidator.generate_migration_plan(
    old_schema=v1_schema,
    new_schema=v2_schema
)

print(f"Breaking changes: {plan['breaking_changes']}")
print(f"Additive changes: {plan['additive_changes']}")
print(f"Migration steps: {plan['migration_steps']}")
```

## LLM Integration

### Generate Extraction Prompts

```python
# Generate prompt for entity extraction
prompt = manager.generate_schema_prompt(
    domain='software_engineering',
    include_examples=True
)

# Use with LLM
from knowledge_engine import KnowledgeEngine

engine = KnowledgeEngine()
response = await engine._call_llm(
    prompt=f"""
{prompt}

Text to extract from:
{document_text}
""",
    system_prompt="You are an expert entity extractor."
)
```

### Prompt Structure

Generated prompts include:
1. Domain description
2. Entity types with properties
3. Property requirements and constraints
4. Example entities
5. Relationship types
6. Extraction instructions

## Configuration

### Schema Configuration File

Located at `knowledge_engine/config/schemas.yaml`:

```yaml
default_schema: knowledge_engine

schemas:
  software_engineering:
    enabled: true
    validation_strict: true
    auto_mapping: true

mappings:
  knowledge_engine_to_graphiti:
    enabled: true
    auto_apply: true

cross_domain:
  enabled: true
  merge_strategy: "union"
  id_deconfliction: "prefix"

validation:
  strict_mode: false
  auto_validate: true
  on_validation_failure: warning
```

### Loading Configuration

```python
# Initialize with config
manager = EntitySchemaManager(
    config_path="knowledge_engine/config/schemas.yaml"
)

# Access config
config = manager.config
default_schema = manager.default_schema
```

## Advanced Usage

### Batch Processing

```python
from knowledge_engine.schemas.validators import EntityBatchProcessor

validator = SchemaValidator(schema)
processor = EntityBatchProcessor(validator, batch_size=100)

def on_batch_complete(start, end, result):
    print(f"Processed {start}-{end}: {result.valid_count} valid")

result = processor.process_entities(
    entities=large_entity_list,
    on_batch_complete=on_batch_complete
)
```

### Schema Export/Import

```python
# Export schema
manager.export_schema(
    domain='software_engineering',
    output_path='schemas/software_engineering.yaml'
)

# Import schema
manager.import_schema(
    domain='software_engineering',
    input_path='schemas/software_engineering.yaml'
)
```

### Schema Statistics

```python
stats = manager.get_statistics(domain='software_engineering')

print(f"Entity types: {stats['entity_types']}")
print(f"Relationship types: {stats['relationship_types']}")
print(f"Total properties: {stats['total_properties']}")
```

## Testing

### Running Tests

```bash
# Run all schema tests
python -m pytest knowledge_engine/schemas/test_schemas.py -v

# Run specific test class
python -m pytest knowledge_engine/schemas/test_schemas.py::TestEntitySchemaManager -v

# Run with coverage
python -m pytest knowledge_engine/schemas/test_schemas.py --cov=knowledge_engine.schemas
```

### Test Coverage

The test suite includes:
- Property validation tests
- Entity type validation tests
- Schema manager tests
- Validator tests
- Mapping tests
- Integration tests

## Best Practices

### 1. Schema Design

- Use clear, descriptive names for entity types
- Define required vs optional properties carefully
- Provide examples for complex entity types
- Use validation rules for business logic

### 2. Property Definitions

- Always provide descriptions
- Use appropriate types (STRING, INTEGER, ENUM, etc.)
- Define validation patterns for strings
- Set min/max values for numerics
- Use allowed_values for enums

### 3. Entity Validation

- Validate early and often
- Use batch validation for performance
- Handle validation errors gracefully
- Log validation results

### 4. Cross-Domain Integration

- Use consistent entity IDs across domains
- Document mapping strategies
- Test mappings thoroughly
- Handle mapping failures gracefully

### 5. Schema Evolution

- Version your schemas
- Document breaking changes
- Use migration planning
- Test migrations before deployment

## Troubleshooting

### Common Issues

**Issue**: Validation fails for valid entities

**Solution**: Check that schema is registered and entity type exists

```python
# List available schemas
print(manager.list_schemas())

# List entity types in schema
schema = manager.get_schema('software_engineering')
print(schema.list_entity_types())
```

**Issue**: Mapping returns wrong entity types

**Solution**: Verify mapping is registered and entity type exists in mapping

```python
# List available mappings
print(manager.list_mappings())

# Check mapping
mapping = get_mapping('knowledge_engine_to_graphiti')
print(mapping)
```

**Issue**: Cross-domain merge loses entities

**Solution**: Entities are merged by ID. Ensure IDs are consistent or use different IDs

```python
# Check for duplicate IDs
entity_ids = [e.entity_id for e in entities]
if len(entity_ids) != len(set(entity_ids)):
    print("Duplicate IDs found!")
```

## API Reference

### EntitySchemaManager

Main class for schema management.

**Methods:**
- `register_schema(domain, schema_definition)`: Register a schema
- `get_schema(domain)`: Get schema by domain
- `list_schemas()`: List all registered schemas
- `validate_entity(entity, schema)`: Validate single entity
- `validate_entities(entities, schema)`: Validate multiple entities
- `map_entities(source_entities, target_schema, mapping_name)`: Map entities
- `merge_cross_domain(entity_sets)`: Merge cross-domain entities
- `generate_schema_prompt(domain, include_examples)`: Generate LLM prompt
- `export_schema(domain, output_path)`: Export schema to YAML
- `import_schema(domain, input_path)`: Import schema from YAML

### SchemaValidator

Validates entities and relationships.

**Methods:**
- `validate_entity(entity)`: Validate entity
- `validate_relationship(relationship, source_type, target_type)`: Validate relationship
- `validate_batch(entities, fail_fast)`: Batch validation
- `validate_entity_consistency(entities)`: Check consistency
- `validate_cross_schema_compatibility(other_schema)`: Check compatibility

### SchemaMigrationValidator

Validates schema migrations.

**Methods:**
- `validate_migration(old_schema, new_schema)`: Validate migration
- `generate_migration_plan(old_schema, new_schema)`: Generate migration plan

## Contributing

When contributing new schemas:

1. Define clear entity types with properties
2. Provide examples for each entity type
3. Define relationship types
4. Add validation rules where needed
5. Document the schema
6. Add tests for the schema
7. Update this README

## License

Part of the OpenEvolve project. See main project LICENSE file.
