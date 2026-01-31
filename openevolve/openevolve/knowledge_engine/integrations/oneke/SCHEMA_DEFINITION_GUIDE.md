# Schema Definition Guide

**OpenEvolve Knowledge Engine - OneKE Schema System**

Complete guide for defining, validating, and managing schemas for bilingual knowledge extraction.

---

## Table of Contents

1. [Schema Format](#schema-format)
2. [Schema Validation](#schema-validation)
3. [Schema Versioning](#schema-versioning)
4. [Schema Migration](#schema-migration)
5. [Built-in Schemas](#built-in-schemas)
6. [Custom Schema Creation](#custom-schema-creation)
7. [Examples](#examples)
8. [Best Practices](#best-practices)

---

## Schema Format

### JSON Format

```json
{
  "name": "company_domain",
  "version": "1.0.0",
  "description": "Schema for company-related extraction",
  "entity_types": [
    {
      "name": "Company",
      "description": "Business organization",
      "examples": ["Apple", "Microsoft", "Google"],
      "attributes": ["founded", "headquarters", "industry"],
      "validation_rules": {
        "min_length": 2
      }
    }
  ],
  "relation_types": [
    {
      "name": "founded_by",
      "description": "Company was founded by person",
      "domain": "Company",
      "range": "Person",
      "examples": ["Apple was founded by Steve Jobs"],
      "symmetric": false,
      "inverse_of": null
    }
  ],
  "event_types": [
    {
      "name": "Acquisition",
      "description": "Company acquisition event",
      "arguments": ["acquirer", "target", "amount", "date"],
      "examples": ["Microsoft acquired Activision for $68.7B"]
    }
  ],
  "metadata": {
    "author": "OpenEvolve",
    "created_at": "2026-01-08",
    "domain": "business"
  }
}
```

### YAML Format

```yaml
name: company_domain
version: 1.0.0
description: Schema for company-related extraction

entity_types:
  - name: Company
    description: Business organization
    examples:
      - Apple
      - Microsoft
      - Google
    attributes:
      - founded
      - headquarters
      - industry

relation_types:
  - name: founded_by
    description: Company was founded by person
    domain: Company
    range: Person
    examples:
      - Apple was founded by Steve Jobs

event_types:
  - name: Acquisition
    description: Company acquisition event
    arguments:
      - acquirer
      - target
      - amount
      - date
```

---

## Schema Validation

### Pydantic Validation

All schemas are validated using Pydantic models:

```python
from knowledge_engine.integrations.oneke.schema_manager import SchemaDefinition
from pydantic import ValidationError

# Valid schema
schema_data = {
    "name": "test_schema",
    "version": "1.0.0",
    "entity_types": [
        {"name": "Person", "description": "A person"}
    ]
}

try:
    schema = SchemaDefinition(**schema_data)
    print("Schema is valid!")
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Validation Rules

#### 1. Version Format

```python
# ✅ Valid: Semantic versioning
"version": "1.0.0"
"version": "2.3.15"

# ❌ Invalid: Wrong format
"version": "1.0"
"version": "1.0.0.0"
"version": "latest"
```

#### 2. Entity Type Validation

```python
# ✅ Valid: Entity types with unique names
"entity_types": [
    {"name": "Person", "description": "A person"},
    {"name": "Company", "description": "A company"}
]

# ❌ Invalid: Duplicate entity type names
"entity_types": [
    {"name": "Person", "description": "A person"},
    {"name": "Person", "description": "Individual"}  # Duplicate!
]

# ❌ Invalid: Missing name field
"entity_types": [
    {"description": "A person"}  # No name!
]
```

#### 3. Relation Type Validation

```python
# ✅ Valid: Relations reference existing entities
"entity_types": [
    {"name": "Company"},
    {"name": "Person"}
],
"relation_types": [
    {
        "name": "founded_by",
        "domain": "Company",   # References existing entity
        "range": "Person"      # References existing entity
    }
]

# ❌ Invalid: Relations reference non-existent entities
"relation_types": [
    {
        "name": "led_by",
        "domain": "Company",
        "range": "CEO"  # CEO entity type doesn't exist!
    }
]
```

### Custom Validation Rules

```python
schema_data = {
    "name": "custom_schema",
    "entity_types": [
        {
            "name": "Company",
            "validation_rules": {
                "min_length": 2,        # Minimum name length
                "max_length": 100,      # Maximum name length
                "pattern": "^[A-Z]",    # Must start with uppercase
                "required_attributes": ["founded", "industry"]
            }
        }
    ]
}
```

---

## Schema Versioning

### Semantic Versioning

Schemas use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes (removed/renamed types)
- **MINOR**: New features (added types)
- **PATCH**: Bug fixes (metadata updates)

### Version History

```python
from knowledge_engine.integrations.oneke.schema_manager import OneKESchemaManager

schema_manager = OneKESchemaManager()

# Create version 1.0.0
schema_v1 = SchemaDefinition(
    name="company_schema",
    version="1.0.0",
    entity_types=[
        {"name": "Company", "description": "Business org"}
    ]
)
await schema_manager.save_schema(schema_v1)

# Create version 1.1.0 (add new entity type)
schema_v2 = SchemaDefinition(
    name="company_schema",
    version="1.1.0",
    entity_types=[
        {"name": "Company", "description": "Business org"},
        {"name": "Startup", "description": "New company"}  # Added
    ]
)
await schema_manager.save_schema(schema_v2)

# List all versions
versions = await schema_manager.get_schema_versions("company_schema")
print(versions)  # ["1.0.0", "1.1.0"]
```

### Automatic Versioning

```python
# Update schema with auto-versioning
updates = {
    "entity_types": [
        {"name": "Product", "description": "Company product"}
    ]
}

# Automatically creates version 1.1.1
updated = await schema_manager.update_schema(
    name="company_schema",
    updates=updates,
    create_version=True
)
```

---

## Schema Migration

### Migration Steps

```python
migration_steps = [
    {
        "type": "add_entity_type",
        "entity_type": {
            "name": "Product",
            "description": "Company product"
        }
    },
    {
        "type": "rename_entity_type",
        "old_name": "Startup",
        "new_name": "StartupCompany"
    },
    {
        "type": "remove_entity_type",
        "entity_name": "OldType"
    },
    {
        "type": "add_relation_type",
        "relation_type": {
            "name": "produces",
            "description": "Company produces product",
            "domain": "Company",
            "range": "Product"
        }
    },
    {
        "type": "update_version",
        "version": "2.0.0"
    }
]
```

### Execute Migration

```python
# Migrate from version 1.0.0 to 2.0.0
migrated = await schema_manager.migrate_schema(
    name="company_schema",
    from_version="1.0.0",
    to_version="2.0.0",
    migration_steps=migration_steps
)

print(f"Migrated to version: {migrated.version}")
```

### Migration Example

```python
# Original schema (1.0.0)
schema_v1 = {
    "name": "company_schema",
    "version": "1.0.0",
    "entity_types": [
        {"name": "Company"},
        {"name": "Person"}
    ],
    "relation_types": [
        {"name": "founded_by", "domain": "Company", "range": "Person"}
    ]
}

# Migration: Add products and update relations
migration_steps = [
    {
        "type": "add_entity_type",
        "entity_type": {
            "name": "Product",
            "description": "Product or service"
        }
    },
    {
        "type": "add_relation_type",
        "relation_type": {
            "name": "produces",
            "description": "Company produces product",
            "domain": "Company",
            "range": "Product"
        }
    },
    {
        "type": "update_version",
        "version": "2.0.0"
    }
]

# Execute migration
schema_v2 = await schema_manager.migrate_schema(
    name="company_schema",
    from_version="1.0.0",
    to_version="2.0.0",
    migration_steps=migration_steps
)
```

---

## Built-in Schemas

### General Schema

```python
schema = await schema_manager.load_schema("general")

# Entity Types:
# - Person: A person
# - Organization: An organization
# - Location: A location
# - Date: A date or time
# - Number: A numerical value

# Relation Types:
# - located_in: Something is located in a place
# - works_for: Person works for organization
# - founded: Person founded organization
# - happened_on: Event happened on date
```

### Biomedical Schema

```python
schema = await schema_manager.load_schema("biomedical")

# Entity Types:
# - Gene: A gene
# - Protein: A protein
# - Disease: A disease
# - Drug: A drug or medication

# Relation Types:
# - associates_with: Gene associates with disease
# - interacts_with: Protein interacts with protein
# - treats: Drug treats disease
```

### Legal Schema

```python
schema = await schema_manager.load_schema("legal")

# Entity Types:
# - Court: A court
# - Case: A legal case
# - Law: A law or statute
# - Judge: A judge

# Relation Types:
# - heard_in: Case was heard in court
# - presided_over: Judge presided over case
# - established_by: Case established law
```

---

## Custom Schema Creation

### Step 1: Define Schema

```python
# Create custom schema for financial news
financial_news_schema = {
    "name": "financial_news",
    "version": "1.0.0",
    "description": "Schema for financial news extraction",
    "entity_types": [
        {
            "name": "Company",
            "description": "Publicly traded company",
            "examples": ["Apple", "Microsoft", "Tesla"],
            "attributes": ["ticker", "exchange", "market_cap"]
        },
        {
            "name": "Stock",
            "description": "Stock or security",
            "examples": ["AAPL", "MSFT", "TSLA"],
            "attributes": ["price", "change", "volume"]
        },
        {
            "name": "CEO",
            "description": "Chief Executive Officer",
            "examples": ["Tim Cook", "Elon Musk"],
            "attributes": ["tenure", "compensation"]
        },
        {
            "name": "FinancialMetric",
            "description": "Financial metric or indicator",
            "examples": ["revenue", "profit", "earnings", "growth"],
            "attributes": ["value", "period", "comparison"]
        }
    ],
    "relation_types": [
        {
            "name": "trades_as",
            "description": "Company trades as stock",
            "domain": "Company",
            "range": "Stock",
            "examples": ["Apple trades as AAPL"]
        },
        {
            "name": "led_by",
            "description": "Company led by CEO",
            "domain": "Company",
            "range": "CEO",
            "examples": ["Apple is led by Tim Cook"]
        },
        {
            "name": "reported",
            "description": "Company reported metric",
            "domain": "Company",
            "range": "FinancialMetric",
            "examples": ["Apple reported revenue of $94.9B"]
        },
        {
            "name": "beat",
            "description": "Metric beat expectations",
            "domain": "FinancialMetric",
            "range": "FinancialMetric",
            "examples": ["Revenue beat analyst expectations"]
        }
    ]
}
```

### Step 2: Validate Schema

```python
from knowledge_engine.integrations.oneke.schema_manager import SchemaDefinition

# Create and validate
try:
    schema_def = SchemaDefinition(**financial_news_schema)
    print("Schema validation passed!")
    print(f"Schema hash: {schema_def.get_hash()}")
except ValidationError as e:
    print(f"Schema validation failed: {e}")
```

### Step 3: Save Schema

```python
# Save to JSON file
schema_path = await schema_manager.save_schema(
    schema=schema_def,
    format=SchemaFormat.JSON,
    create_version=False
)

print(f"Schema saved to: {schema_path}")
```

### Step 4: Use Schema for Extraction

```python
from knowledge_engine.integrations.oneke.model_adapter import OneKEModelAdapter, Language

adapter = OneKEModelAdapter()
await adapter.load_model()

# Extract with custom schema
text = """
Apple reported Q4 revenue of $94.9 billion, beating analyst expectations
of $91.3 billion. The stock (AAPL) rose 2% in after-hours trading.
CEO Tim Cook attributed the strong results to iPhone sales.
"""

result = await adapter.extract_triples(
    text=text,
    schema=schema_def.dict(),
    language=Language.ENGLISH
)

print("Extracted triples:")
for triple in result.triples:
    print(f"  {triple}")

# Expected:
# (Apple, reported, $94.9B revenue)
# (Apple, trades_as, AAPL)
# (Apple, led_by, Tim Cook)
# (Revenue, beat, expectations)
```

---

## Examples

### Example 1: Simple Entity Schema

```json
{
  "name": "person_organization",
  "version": "1.0.0",
  "entity_types": [
    {
      "name": "Person",
      "description": "A person",
      "examples": ["John Doe", "Jane Smith"]
    },
    {
      "name": "Organization",
      "description": "An organization",
      "examples": ["Apple", "UN"]
    }
  ],
  "relation_types": [
    {
      "name": "works_for",
      "description": "Person works for organization",
      "domain": "Person",
      "range": "Organization"
    }
  ]
}
```

### Example 2: Complex Event Schema

```json
{
  "name": "news_events",
  "version": "1.0.0",
  "entity_types": [
    {"name": "Person", "examples": ["Joe Biden", "Vladimir Putin"]},
    {"name": "Country", "examples": ["USA", "Russia", "China"]},
    {"name": "Organization", "examples": ["UN", "NATO"]}
  ],
  "relation_types": [
    {"name": "visited", "domain": "Person", "range": "Country"},
    {"name": "sanctioned", "domain": "Country", "range": "Country"}
  ],
  "event_types": [
    {
      "name": "DiplomaticMeeting",
      "description": "Diplomatic meeting between leaders",
      "arguments": ["participant1", "participant2", "location", "date"],
      "examples": [
        "Biden met with Putin in Geneva in June 2021",
        "习近平与普京会晤"
      ]
    },
    {
      "name": "TradeAgreement",
      "description": "Trade agreement signed",
      "arguments": ["country1", "country2", "value", "date"],
      "examples": [
        "China and US signed phase one trade deal worth $200B"
      ]
    }
  ]
}
```

### Example 3: Bilingual Schema

```json
{
  "name": "bilingual_tech",
  "version": "1.0.0",
  "entity_types": [
    {
      "name": "Company/公司",
      "description": "Technology company / 技术公司",
      "examples_en": ["Apple", "Microsoft", "Google"],
      "examples_zh": ["苹果", "微软", "谷歌"],
      "translations": {
        "Apple": "苹果",
        "Microsoft": "微软",
        "Google": "谷歌",
        "Huawei": "华为"
      }
    },
    {
      "name": "Product/产品",
      "description": "Product / 产品",
      "examples_en": ["iPhone", "Windows", "Android"],
      "examples_zh": ["iPhone", "Windows", "安卓"],
      "translations": {
        "iPhone": "iPhone",
        "Windows": "Windows",
        "Android": "安卓"
      }
    }
  ],
  "relation_types": [
    {
      "name": "released/发布",
      "description": "Company released product / 公司发布产品",
      "domain": "Company/公司",
      "range": "Product/产品"
    }
  ]
}
```

---

## Best Practices

### 1. Schema Design

```python
# ✅ Good: Clear, specific entity types
entity_types = [
    {"name": "CEO", "description": "Chief Executive Officer"},
    {"name": "CTO", "description": "Chief Technology Officer"},
    {"name": "CFO", "description": "Chief Financial Officer"}
]

# ❌ Bad: Too generic
entity_types = [
    {"name": "Person", "description": "A person"}  # Too broad
]
```

### 2. Examples

```python
# ✅ Good: Provide plenty of examples
entity_type = {
    "name": "Company",
    "examples": [
        "Apple",
        "Microsoft",
        "Google",
        "Amazon",
        "Tesla"
    ]
}

# ❌ Bad: Too few examples
entity_type = {
    "name": "Company",
    "examples": ["Apple"]  # Not enough
}
```

### 3. Version Management

```python
# ✅ Good: Semantic versioning
"version": "1.0.0"  # MAJOR.MINOR.PATCH

# ❌ Bad: Inconsistent versions
"version": "v1"
"version": "1.0"
"version": "latest"
```

### 4. Relation Domains

```python
# ✅ Good: Specify domain and range
relation_type = {
    "name": "founded_by",
    "domain": "Company",   # Subject entity type
    "range": "Person"      # Object entity type
}

# ❌ Bad: No domain/range specified
relation_type = {
    "name": "founded_by"
}
```

### 5. Schema Reusability

```python
# ✅ Good: Modular schemas
# Base schema
base_schema = {
    "entity_types": [
        {"name": "Person"},
        {"name": "Organization"}
    ]
}

# Extend for specific domain
tech_schema = {
    **base_schema,
    "entity_types": [
        *base_schema["entity_types"],
        {"name": "Product"},
        {"name": "Technology"}
    ]
}

# ❌ Bad: Duplicate definitions
# Copy-pasting common entities across schemas
```

---

## Next Steps

- [OneKE Integration Guide](ONEKE_INTEGRATION_GUIDE.md) - Complete integration documentation
- [Bilingual Extraction Tutorial](BILINGUAL_EXTRACTION_TUTORIAL.md) - Bilingual extraction examples
- [Event Extraction Guide](EVENT_EXTRACTION_GUIDE.md) - Extract events with schemas
- [API Reference](ONEKE_API_REFERENCE.md) - Complete API documentation

---

**Version:** 1.0.0
**Last Updated:** 2026-01-08
