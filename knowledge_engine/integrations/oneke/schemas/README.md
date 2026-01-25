# OneKE Schema Definitions

This directory contains schema definitions for the OneKE bilingual knowledge extraction system.

## Available Schemas

### 1. general_schema.json
General-purpose schema for common entity and relation types.

**Entity Types:**
- PERSON (Individuals)
- ORGANIZATION (Companies, institutions)
- LOCATION (Geographical locations)
- PRODUCT (Products, services)
- EVENT (Named events)
- DATE (Temporal expressions)
- MONEY (Monetary values)

**Relation Types:**
- WORKS_FOR / EMPLOYS
- FOUNDED_BY / FOUNDED
- LOCATED_IN / CONTAINS
- ACQUIRED / ACQUIRED_BY
- LAUNCHED / LAUNCHED_BY
- PARTNERED_WITH

**Event Types:**
- ACQUISITION (Mergers and acquisitions)
- LAUNCH (Product launches)
- APPOINTMENT (Personnel appointments)
- LEGAL (Legal proceedings)

### 2. biomedical_schema.json
Domain-specific schema for biomedical and healthcare extraction.

**Entity Types:**
- DISEASE (Diseases, disorders)
- DRUG (Medications, treatments)
- GENE (Genes, proteins)
- SYMPTOM (Clinical manifestations)
- PROCEDURE (Medical procedures)
- ANATOMY (Body parts)
- MEDICAL_INSTITUTION (Healthcare organizations)

**Relation Types:**
- TREATS / TREATED_BY
- CAUSES / CAUSED_BY
- ASSOCIATED_WITH
- INTERACTS_WITH
- DIAGNOSES / DIAGNOSED_BY
- MANUFACTURED_BY / MANUFACTURES

**Event Types:**
- CLINICAL_TRIAL
- OUTBREAK
- DRUG_APPROVAL

### 3. legal_schema.json
Domain-specific schema for legal document extraction.

**Entity Types:**
- PERSON (Legal actors)
- ORGANIZATION (Companies, government bodies)
- COURT (Judicial bodies)
- LAW (Statutes, regulations)
- CASE (Legal cases)
- DATE (Legal dates)
- MONEY (Damages, settlements)
- LEGAL_CONCEPT (Legal principles)

**Relation Types:**
- SUES / SUED_BY
- REPRESENTS / REPRESENTED_BY
- VIOLATES / VIOLATED_BY
- RULES_ON / RULED_BY
- SETS_PRECEDENT / PRECEDENT_SET_BY
- AWARDS / AWARDED_TO

**Event Types:**
- LAWSUIT_FILING
- COURT_DECISION
- SETTLEMENT
- APPEAL
- REGULATION_ENACTMENT

## Using Schemas

### Loading a Schema

```python
from knowledge_engine.integrations.oneke import OneKESchemaManager

manager = OneKESchemaManager()

# Load schema from file
schema = await manager.load_schema("schemas/general_schema.json")

# Or load by name
schema = await manager.load_schema_by_name("general_extraction")
```

### Using Schema for Extraction

```python
from knowledge_engine.integrations.oneke import OneKEModelAdapter

adapter = OneKEModelAdapter()

# Extract using schema
result = await adapter.extract(
    text="Apple announced the new iPhone today.",
    schema=schema,
    language=Language.ENGLISH
)
```

### Custom Schema Creation

Create a custom schema following this structure:

```json
{
  "schema_name": "my_custom_schema",
  "schema_type": "entity_extraction",
  "description": "Description of schema",
  "version": "1.0",
  "languages": ["en", "zh"],
  "entity_types": [...],
  "relation_types": [...],
  "event_types": [...],
  "extraction_rules": {...}
}
```

## Schema Validation

Schemas are automatically validated when loaded:

- Required fields must be present
- Entity types must have examples
- Relation types must define inverse relationships
- Event types must specify arguments

## Best Practices

1. **Be Specific**: Use descriptive entity and relation type names
2. **Provide Examples**: Include bilingual examples for each type
3. **Define Inverses**: Specify inverse relationships for all relations
4. **Set Thresholds**: Configure appropriate confidence thresholds
5. **Test Thoroughly**: Validate schemas on sample documents

## Contributing

To add a new schema:

1. Create a new JSON file in this directory
2. Follow the schema structure
3. Include bilingual examples
4. Test with sample documents
5. Update this README
