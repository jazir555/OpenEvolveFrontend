# Schema Definition Guide

Complete guide for creating and managing OneKE extraction schemas.

## Table of Contents

1. [Schema Overview](#schema-overview)
2. [Schema Structure](#schema-structure)
3. [Entity Type Definition](#entity-type-definition)
4. [Relation Type Definition](#relation-type-definition)
5. [Event Type Definition](#event-type-definition)
6. [Schema Validation](#schema-validation)
7. [Creating Custom Schemas](#creating-custom-schemas)
8. [Schema Best Practices](#schema-best-practices)
9. [Domain-Specific Schemas](#domain-specific-schemas)
10. [Schema Management](#schema-management)

---

## Schema Overview

### What is a Schema?

A schema defines what to extract from text:
- **Entity Types**: What kinds of entities to extract (PERSON, ORGANIZATION, etc.)
- **Relation Types**: How entities relate (WORKS_FOR, LOCATED_IN, etc.)
- **Event Types**: What events to detect (LAUNCH, ACQUISITION, etc.)
- **Extraction Rules**: Confidence thresholds, limits, validation rules

### Why Use Schemas?

- **Accuracy**: Guides extraction to relevant types
- **Consistency**: Ensures uniform output across documents
- **Flexibility**: Customize for different domains
- **Efficiency**: Reduces false positives

---

## Schema Structure

### Complete Schema Template

```json
{
  "schema_name": "my_custom_schema",
  "schema_type": "entity_extraction",
  "description": "Description of what this schema extracts",
  "version": "1.0",
  "languages": ["en", "zh"],
  "entity_types": [...],
  "relation_types": [...],
  "event_types": [...],
  "extraction_rules": {
    "minimum_confidence": 0.6,
    "max_entities_per_document": 100,
    "max_relations_per_document": 150,
    "require_entity_types": true,
    "allow_overlapping_entities": false
  }
}
```

### Schema Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_name` | string | Yes | Unique schema identifier |
| `schema_type` | string | Yes | Schema type (entity_extraction, domain_specific) |
| `description` | string | Yes | Human-readable description |
| `version` | string | Yes | Schema version |
| `languages` | array | Yes | Supported languages |
| `entity_types` | array | Yes | Entity type definitions |
| `relation_types` | array | No | Relation type definitions |
| `event_types` | array | No | Event type definitions |
| `extraction_rules` | object | No | Extraction configuration |

---

## Entity Type Definition

### Basic Entity Type

```json
{
  "type": "PERSON",
  "description": "Person, including individuals",
  "examples_en": ["Steve Jobs", "Barack Obama"],
  "examples_zh": ["史蒂夫·乔布斯", "巴拉克·奥巴马"]
}
```

### Entity Type Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Entity type name (uppercase) |
| `description` | string | Yes | What this type represents |
| `examples_en` | array | Yes | English examples |
| `examples_zh` | array | Yes | Chinese examples |

### Standard Entity Types

```json
{
  "entity_types": [
    {
      "type": "PERSON",
      "description": "People, including fictional characters",
      "examples_en": ["Steve Jobs", "Harry Potter"],
      "examples_zh": ["史蒂夫·乔布斯", "哈利·波特"]
    },
    {
      "type": "ORGANIZATION",
      "description": "Companies, institutions, organizations",
      "examples_en": ["Apple Inc.", "United Nations"],
      "examples_zh": ["苹果公司", "联合国"]
    },
    {
      "type": "LOCATION",
      "description": "Geographical locations, cities, countries",
      "examples_en": ["San Francisco", "United States"],
      "examples_zh": ["旧金山", "美国"]
    },
    {
      "type": "PRODUCT",
      "description": "Products, services, applications",
      "examples_en": ["iPhone", "Windows 11"],
      "examples_zh": ["iPhone", "Windows 11"]
    },
    {
      "type": "DATE",
      "description": "Dates, times, durations",
      "examples_en": ["January 9, 2007", "Q4 2024"],
      "examples_zh": ["2007年1月9日", "2024年第四季度"]
    },
    {
      "type": "MONEY",
      "description": "Monetary values, currency amounts",
      "examples_en": ["$1 billion", "¥100 million"],
      "examples_zh": ["10亿美元", "1亿人民币"]
    },
    {
      "type": "PERCENT",
      "description": "Percentages and ratios",
      "examples_en": ["50%", "0.75"],
      "examples_zh": ["50%", "0.75"]
    },
    {
      "type": "EVENT",
      "description": "Named events, conferences, meetings",
      "examples_en": ["WWDC 2024", "Olympic Games"],
      "examples_zh": ["WWDC 2024", "奥运会"]
    }
  ]
}
```

---

## Relation Type Definition

### Basic Relation Type

```json
{
  "type": "WORKS_FOR",
  "description": "Employment relationship: person works for organization",
  "inverse": "EMPLOYS",
  "examples_en": ["Steve Jobs WORKS_FOR Apple"],
  "examples_zh": ["史蒂夫·乔布斯 工作于 苹果公司"]
}
```

### Relation Type Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Relation type name (uppercase) |
| `description` | string | Yes | What this relation represents |
| `inverse` | string | Yes | Inverse relation type |
| `examples_en` | array | Yes | English examples |
| `examples_zh` | array | Yes | Chinese examples |

### Standard Relation Types

```json
{
  "relation_types": [
    {
      "type": "WORKS_FOR",
      "description": "Person works for organization",
      "inverse": "EMPLOYS",
      "examples_en": ["Tim Cook WORKS_FOR Apple"],
      "examples_zh": ["蒂姆·库克 工作于 苹果"]
    },
    {
      "type": "FOUNDED_BY",
      "description": "Organization founded by person",
      "inverse": "FOUNDED",
      "examples_en": ["Apple FOUNDED_BY Steve Jobs"],
      "examples_zh": ["苹果公司 由 史蒂夫·乔布斯 创立"]
    },
    {
      "type": "LOCATED_IN",
      "description": "Entity located in location",
      "inverse": "CONTAINS",
      "examples_en": ["Apple LOCATED_IN Cupertino"],
      "examples_zh": ["苹果公司 位于 库比蒂诺"]
    },
    {
      "type": "ACQUIRED",
      "description": "Organization acquired another",
      "inverse": "ACQUIRED_BY",
      "examples_en": ["Microsoft ACQUIRED LinkedIn"],
      "examples_zh": ["微软 收购 了 领英"]
    },
    {
      "type": "LAUNCHED",
      "description": "Organization launched product",
      "inverse": "LAUNCHED_BY",
      "examples_en": ["Apple LAUNCHED iPhone"],
      "examples_zh": ["苹果公司 发布 了 iPhone"]
    },
    {
      "type": "PARTNERED_WITH",
      "description": "Partnership between organizations",
      "inverse": "PARTNERED_WITH",
      "examples_en": ["Google PARTNERED_WITH NASA"],
      "examples_zh": ["谷歌 与 NASA 合作"]
    }
  ]
}
```

---

## Event Type Definition

### Basic Event Type

```json
{
  "type": "ACQUISITION",
  "description": "Company acquisition or merger",
  "triggers_en": ["acquired", "bought", "purchased", "merged"],
  "triggers_zh": ["收购", "并购", "收购了", "合并"],
  "arguments": ["subject", "object", "time", "location", "amount"]
}
```

### Event Type Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Event type name (uppercase) |
| `description` | string | Yes | What this event represents |
| `triggers_en` | array | Yes | English trigger words |
| `triggers_zh` | array | Yes | Chinese trigger words |
| `arguments` | array | Yes | Event argument roles |

### Standard Event Types

```json
{
  "event_types": [
    {
      "type": "ACQUISITION",
      "description": "Company acquisition or merger",
      "triggers_en": ["acquired", "bought", "purchased", "merged", "acquisition"],
      "triggers_zh": ["收购", "并购", "收购了", "合并"],
      "arguments": ["acquirer", "acquired", "time", "location", "amount"]
    },
    {
      "type": "LAUNCH",
      "description": "Product or service launch",
      "triggers_en": ["launched", "released", "introduced", "announced", "unveiled"],
      "triggers_zh": ["发布", "推出", "宣布", "亮相"],
      "arguments": ["organization", "product", "time", "location"]
    },
    {
      "type": "APPOINTMENT",
      "description": "Person appointment to position",
      "triggers_en": ["appointed", "named", "hired", "joined"],
      "triggers_zh": ["任命", "聘用", "加入"],
      "arguments": ["person", "position", "organization", "time"]
    },
    {
      "type": "LEGAL",
      "description": "Legal proceedings, lawsuits",
      "triggers_en": ["sued", "lawsuit", "litigation", "court", "legal action"],
      "triggers_zh": ["起诉", "诉讼", "法庭"],
      "arguments": ["plaintiff", "defendant", "time", "location", "reason"]
    },
    {
      "type": "FINANCIAL",
      "description": "Financial events (earnings, investments, etc.)",
      "triggers_en": ["reported earnings", "raised funding", "IPO", "investment"],
      "triggers_zh": ["财报", "融资", "上市", "投资"],
      "arguments": ["organization", "amount", "time", "investors"]
    }
  ]
}
```

---

## Schema Validation

### Validation Rules

1. **Required Fields**: All required fields must be present
2. **Bilingual Examples**: Must provide examples in both languages
3. **Inverse Relations**: All relations must have inverse defined
4. **Unique Types**: Type names must be unique
5. **Valid Arguments**: Event arguments must be from standard set

### Validation Example

```python
from knowledge_engine.integrations.oneke import OneKESchemaManager

async def validate_schema_example():
    manager = OneKESchemaManager()

    # Load and validate schema
    schema = await manager.load_schema("schemas/general_schema.json")

    # Check validity
    is_valid = await manager.validate_schema(schema)

    if is_valid:
        print("Schema is valid!")
    else:
        print("Schema validation failed")

    return is_valid

asyncio.run(validate_schema_example())
```

### Common Validation Errors

#### Error 1: Missing Bilingual Examples

```json
// INVALID
{
  "type": "PERSON",
  "description": "Person",
  "examples_en": ["Steve Jobs"]
  // Missing examples_zh
}

// VALID
{
  "type": "PERSON",
  "description": "Person",
  "examples_en": ["Steve Jobs"],
  "examples_zh": ["史蒂夫·乔布斯"]
}
```

#### Error 2: Missing Inverse Relation

```json
// INVALID
{
  "type": "WORKS_FOR",
  "description": "Employment",
  "examples_en": ["Steve WORKS_FOR Apple"]
  // Missing inverse field
}

// VALID
{
  "type": "WORKS_FOR",
  "description": "Employment",
  "inverse": "EMPLOYS",
  "examples_en": ["Steve WORKS_FOR Apple"]
}
```

---

## Creating Custom Schemas

### Example 1: E-commerce Schema

```json
{
  "schema_name": "ecommerce_extraction",
  "schema_type": "domain_specific",
  "description": "E-commerce product and customer extraction",
  "version": "1.0",
  "languages": ["en", "zh"],
  "entity_types": [
    {
      "type": "PRODUCT",
      "description": "Products for sale",
      "examples_en": ["iPhone 15 Pro", "Samsung Galaxy S24"],
      "examples_zh": ["iPhone 15 Pro", "三星Galaxy S24"]
    },
    {
      "type": "BRAND",
      "description": "Product brands",
      "examples_en": ["Apple", "Samsung", "Nike"],
      "examples_zh": ["苹果", "三星", "耐克"]
    },
    {
      "type": "CUSTOMER",
      "description": "Customers",
      "examples_en": ["John Doe", "Jane Smith"],
      "examples_zh": ["张三", "李四"]
    },
    {
      "type": "REVIEW",
      "description": "Product reviews",
      "examples_en": ["5 stars", "great product"],
      "examples_zh": ["五星好评", "很棒的产品"]
    },
    {
      "type": "PRICE",
      "description": "Product prices",
      "examples_en": ["$999", "¥6999"],
      "examples_zh": ["999美元", "6999人民币"]
    }
  ],
  "relation_types": [
    {
      "type": "MANUFACTURED_BY",
      "description": "Product manufactured by brand",
      "inverse": "MANUFACTURES",
      "examples_en": ["iPhone MANUFACTURED_BY Apple"],
      "examples_zh": ["iPhone 由 苹果 制造"]
    },
    {
      "type": "REVIEWED_BY",
      "description": "Product reviewed by customer",
      "inverse": "REVIEWED",
      "examples_en": ["iPhone REVIEWED_BY John"],
      "examples_zh": ["iPhone 被 张三 评价"]
    },
    {
      "type": "HAS_PRICE",
      "description": "Product has price",
      "inverse": "PRICE_OF",
      "examples_en": ["iPhone HAS_PRICE $999"],
      "examples_zh": ["iPhone 价格 $999"]
    }
  ],
  "event_types": [
    {
      "type": "PURCHASE",
      "description": "Product purchase event",
      "triggers_en": ["bought", "purchased", "ordered"],
      "triggers_zh": ["购买", "下单"],
      "arguments": ["customer", "product", "time", "amount"]
    },
    {
      "type": "REVIEW",
      "description": "Product review event",
      "triggers_en": ["reviewed", "rated", "commented"],
      "triggers_zh": ["评价", "评论", "评分"],
      "arguments": ["customer", "product", "rating", "time"]
    }
  ],
  "extraction_rules": {
    "minimum_confidence": 0.7,
    "max_entities_per_document": 50,
    "max_relations_per_document": 80,
    "require_entity_types": true,
    "allow_overlapping_entities": false
  }
}
```

### Example 2: Academic Paper Schema

```json
{
  "schema_name": "academic_paper_extraction",
  "schema_type": "domain_specific",
  "description": "Academic paper and citation extraction",
  "version": "1.0",
  "languages": ["en", "zh"],
  "entity_types": [
    {
      "type": "PAPER",
      "description": "Research papers",
      "examples_en": ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers"],
      "examples_zh": ["注意力机制就是你所需的全部", "BERT论文"]
    },
    {
      "type": "AUTHOR",
      "description": "Paper authors",
      "examples_en": ["Geoffrey Hinton", "Yann LeCun"],
      "examples_zh": ["杰弗里·辛顿", "杨立昆"]
    },
    {
      "type": "INSTITUTION",
      "description": "Research institutions",
      "examples_en": ["Google Research", "MIT", "Stanford"],
      "examples_zh": ["谷歌研究院", "麻省理工", "斯坦福"]
    },
    {
      "type": "VENUE",
      "description": "Publication venues (conferences, journals)",
      "examples_en": ["NeurIPS", "ICML", "Nature"],
      "examples_zh": ["神经信息处理系统大会", "国际机器学习会议", "自然"]
    },
    {
      "type": "METHOD",
      "description": "Methods and algorithms",
      "examples_en": ["Transformer", "CNN", "LSTM"],
      "examples_zh": ["Transformer", "卷积神经网络", "长短期记忆网络"]
    }
  ],
  "relation_types": [
    {
      "type": "WRITTEN_BY",
      "description": "Paper written by author",
      "inverse": "WROTE",
      "examples_en": ["Attention paper WRITTEN_BY Vaswani"],
      "examples_zh": ["注意力论文 由 Vaswani 撰写"]
    },
    {
      "type": "PUBLISHED_AT",
      "description": "Paper published at venue",
      "inverse": "PUBLISHED",
      "examples_en": ["BERT PUBLISHED_AT NAACL"],
      "examples_zh": ["BERT 发表于 NAACL"]
    },
    {
      "type": "AFFILIATED_WITH",
      "description": "Author affiliated with institution",
      "inverse": "AFFILIATES",
      "examples_en": ["Hinton AFFILIATED_WITH University of Toronto"],
      "examples_zh": ["辛顿 隶属于 多伦多大学"]
    },
    {
      "type": "CITES",
      "description": "Paper cites another paper",
      "inverse": "CITED_BY",
      "examples_en": ["BERT CITES Attention paper"],
      "examples_zh": ["BERT 引用 了 注意力论文"]
    },
    {
      "type": "USES_METHOD",
      "description": "Paper uses method",
      "inverse": "USED_IN",
      "examples_en": ["GPT USES_METHOD Transformer"],
      "examples_zh": ["GPT 使用 了 Transformer"]
    }
  ],
  "event_types": [
    {
      "type": "PUBLICATION",
      "description": "Paper publication event",
      "triggers_en": ["published", "appeared", "presented at"],
      "triggers_zh": ["发表", "发布", "出现在"],
      "arguments": ["paper", "venue", "year", "authors"]
    },
    {
      "type": "CITATION",
      "description": "Citation event",
      "triggers_en": ["cites", "references", "builds on"],
      "triggers_zh": ["引用", "参考", "基于"],
      "arguments": ["citing_paper", "cited_paper", "context"]
    }
  ],
  "extraction_rules": {
    "minimum_confidence": 0.65,
    "max_entities_per_document": 100,
    "max_relations_per_document": 150,
    "require_entity_types": true,
    "allow_overlapping_entities": false
  }
}
```

---

## Schema Best Practices

### 1. Naming Conventions

**DO:**
- Use UPPERCASE for type names
- Use descriptive names
- Be consistent

```json
{
  "type": "PERSON",           // Good
  "type": "WORKS_FOR",        // Good
}
```

**DON'T:**
- Use lowercase
- Use abbreviations
- Be inconsistent

```json
{
  "type": "person",           // Bad
  "type": "WRK_4",            // Bad
  "type": "Person",           // Bad
}
```

### 2. Provide Examples

**DO:**
- Provide 2-4 examples per type
- Include bilingual examples
- Use representative examples

```json
{
  "type": "ORGANIZATION",
  "examples_en": ["Apple Inc.", "Google LLC", "Microsoft Corporation"],
  "examples_zh": ["苹果公司", "谷歌", "微软公司"]
}
```

**DON'T:**
- Use single example
- Use English only
- Use obscure examples

```json
{
  "type": "ORGANIZATION",
  "examples_en": ["Apple Inc."],
  // Missing Chinese examples
}
```

### 3. Define Inverses

**DO:**
- Always define inverse
- Make inverse clear
- Use consistent naming

```json
{
  "type": "WORKS_FOR",
  "inverse": "EMPLOYS",
  "description": "Person works for organization"
}
```

**DON'T:**
- Skip inverse
- Use unclear inverse
- Forget to define inverse relation

```json
{
  "type": "WORKS_FOR",
  // Missing inverse field
}
```

### 4. Set Appropriate Thresholds

**DO:**
- Adjust threshold based on use case
- Consider precision vs recall
- Test on sample documents

```json
{
  "extraction_rules": {
    "minimum_confidence": 0.7,  // High precision
    "max_entities_per_document": 100
  }
}
```

**DON'T:**
- Use default threshold always
- Set too high (miss entities)
- Set too low (noise)

```json
{
  "extraction_rules": {
    "minimum_confidence": 0.99,  // Too high
    "minimum_confidence": 0.1    // Too low
  }
}
```

### 5. Test Thoroughly

**DO:**
- Test on diverse documents
- Validate both languages
- Check extraction quality

```python
# Test schema
schema = await manager.load_schema("my_schema.json")

# Test on sample
result = await adapter.extract(
    text=sample_text,
    schema=schema,
    language=Language.BILINGUAL
)

# Review results
print(f"Extracted {len(result.entities)} entities")
print(f"Extracted {len(result.relations)} relations")
```

---

## Domain-Specific Schemas

### Available Schemas

1. **general_schema.json**: General-purpose extraction
2. **biomedical_schema.json**: Biomedical domain
3. **legal_schema.json**: Legal domain

### Creating Domain Schemas

**Steps:**

1. **Identify Domain Entities**: What entities are important?
2. **Define Relations**: How do entities relate?
3. **Specify Events**: What events occur?
4. **Provide Examples**: Domain-specific examples
5. **Set Rules**: Domain-specific thresholds
6. **Test**: Validate on domain documents

**Example Domain Schema:**

```json
{
  "schema_name": "finance_extraction",
  "schema_type": "domain_specific",
  "description": "Financial news and reports extraction",
  "version": "1.0",
  "languages": ["en", "zh"],
  "entity_types": [
    {
      "type": "COMPANY",
      "description": "Publicly traded companies",
      "examples_en": ["Apple Inc.", "Tesla Inc."],
      "examples_zh": ["苹果公司", "特斯拉公司"]
    },
    {
      "type": "STOCK",
      "description": "Stock symbols",
      "examples_en": ["AAPL", "TSLA"],
      "examples_zh": ["AAPL", "TSLA"]
    },
    {
      "type": "FINANCIAL_METRIC",
      "description": "Financial metrics (revenue, profit, etc.)",
      "examples_en": ["revenue", "net income", "EBITDA"],
      "examples_zh": ["收入", "净利润", "息税前利润"]
    },
    {
      "type": "EARNINGS_PERIOD",
      "description": "Fiscal periods",
      "examples_en": ["Q1 2024", "fiscal year 2023"],
      "examples_zh": ["2024年第一季度", "2023财年"]
    }
  ],
  "relation_types": [
    {
      "type": "REPORTED",
      "description": "Company reported metric",
      "inverse": "REPORTED_BY",
      "examples_en": ["Apple REPORTED revenue $94.3B"],
      "examples_zh": ["苹果 报告 收入 943亿美元"]
    }
  ],
  "event_types": [
    {
      "type": "EARNINGS_RELEASE",
      "description": "Earnings report release",
      "triggers_en": ["reported earnings", "released financial results"],
      "triggers_zh": ["发布财报", "公布业绩"],
      "arguments": ["company", "period", "revenue", "profit"]
    }
  ],
  "extraction_rules": {
    "minimum_confidence": 0.75,
    "max_entities_per_document": 80,
    "max_relations_per_document": 120
  }
}
```

---

## Schema Management

### Loading Schemas

```python
from knowledge_engine.integrations.oneke import OneKESchemaManager

manager = OneKESchemaManager()

# Load from file
schema = await manager.load_schema("schemas/general_schema.json")

# Load by name
schema = await manager.load_schema_by_name("general_extraction")

# List available schemas
schemas = await manager.list_schemas()
print(f"Available schemas: {schemas}")
```

### Creating Schemas Programmatically

```python
# Create custom schema
custom_schema = await manager.create_custom_schema(
    entity_types=["PERSON", "ORG", "PRODUCT"],
    relation_types=["WORKS_FOR", "LAUNCHED"],
    event_types=["RELEASE", "UPDATE"]
)

# Customize
custom_schema.entity_types.append({
    "type": "CUSTOM_TYPE",
    "description": "Custom entity type",
    "examples_en": ["example"],
    "examples_zh": ["示例"]
})

# Save schema
await manager.save_schema(custom_schema, "schemas/custom_schema.json")
```

### Validating Schemas

```python
# Validate schema
is_valid = await manager.validate_schema(schema)

if not is_valid:
    errors = await manager.get_validation_errors(schema)
    for error in errors:
        print(f"Error: {error}")
```

### Updating Schemas

```python
# Load existing schema
schema = await manager.load_schema("schemas/general_schema.json")

# Add new entity type
schema['entity_types'].append({
    "type": "NEW_TYPE",
    "description": "New entity type",
    "examples_en": ["example"],
    "examples_zh": ["示例"]
})

# Validate updated schema
is_valid = await manager.validate_schema(schema)

# Save if valid
if is_valid:
    await manager.save_schema(schema, "schemas/general_schema_v2.json")
```

---

## Summary

This guide covered:

1. **Schema Structure**: Complete schema format
2. **Entity Types**: Defining entity types with examples
3. **Relation Types**: Defining relations with inverses
4. **Event Types**: Defining events with triggers
5. **Schema Validation**: Ensuring schema correctness
6. **Custom Schemas**: Creating domain-specific schemas
7. **Best Practices**: Naming, examples, thresholds
8. **Domain Schemas**: Specialized schemas for different domains
9. **Schema Management**: Loading, creating, updating schemas

### Key Takeaways

- Schemas guide extraction to relevant types
- Provide bilingual examples for all types
- Define inverse relations for all relations
- Set appropriate confidence thresholds
- Test schemas on domain documents
- Use domain-specific schemas for better accuracy

### Next Steps

- Explore existing schemas in `schemas/` directory
- Create custom schema for your domain
- Test schema on sample documents
- Iterate based on extraction quality

For more details:
- [ONEKE_INTEGRATION_GUIDE.md](ONEKE_INTEGRATION_GUIDE.md)
- [BILINGUAL_EXTRACTION_TUTORIAL.md](BILINGUAL_EXTRACTION_TUTORIAL.md)
