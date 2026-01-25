# OneKE Integration Guide

**OpenEvolve Knowledge Engine - Sprint 3: OneKE Bilingual Extraction**

Complete guide for integrating and using the OneKE multi-task extraction framework for bilingual (English/Chinese) knowledge extraction.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Model Setup](#model-setup)
4. [Configuration](#configuration)
5. [Bilingual Extraction](#bilingual-extraction)
6. [Schema System](#schema-system)
7. [Entity Linking](#entity-linking)
8. [Event Extraction](#event-extraction)
9. [API Reference](#api-reference)
10. [Examples](#examples)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Overview

OneKE is a bilingual (English/Chinese) knowledge extraction model that provides:

- **Multi-Task Extraction**: NER, Relation Extraction, Event Extraction, Triple Extraction
- **Schema-Guided Extraction**: Define what to extract using schemas
- **Bilingual Support**: Extract from English, Chinese, or mixed-language documents
- **Few-Shot Learning**: Provide examples to improve extraction
- **Model Quantization**: INT8/INT4 quantization for memory efficiency

### Key Features

- **Named Entity Recognition (NER)**: Extract entities with W2NER model
- **Relation Extraction**: Extract relationships between entities
- **Event Extraction**: Extract events and event arguments
- **Triple Joint Extraction**: Extract (subject, predicate, object) triples
- **Automatic Task Detection**: Auto-select appropriate extraction task
- **Schema Management**: Define, validate, and version extraction schemas

---

## Installation

### Prerequisites

```bash
# Python 3.9+
python --version

# CUDA 11.8+ (for GPU acceleration)
nvidia-smi

# 16GB+ RAM recommended (32GB for full model)
```

### Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and dependencies
pip install transformers>=4.35.0
pip install bitsandbytes>=0.41.0
pip install accelerate>=0.24.0
pip install pydantic>=2.0.0

# Install additional dependencies
pip install pyyaml
```

### Clone Repository

```bash
git clone https://github.com/openevolve/frontend.git
cd Frontend
```

---

## Model Setup

### Option 1: Use HuggingFace Model

```python
from knowledge_engine.integrations.oneke.model_adapter import OneKEModelAdapter, ModelConfig

# Configure model
config = ModelConfig(
    model_name="oneke/OneKE-13B",
    device="cuda",
    quantization="int4"  # Use int4 quantization for memory efficiency
)

# Initialize adapter
adapter = OneKEModelAdapter(config)

# Load model
await adapter.load_model()
```

### Option 2: Use Local Model Path

```bash
# Download model from HuggingFace
git clone https://huggingface.co/oneke/OneKE-13B ./models/oneke

# Or download manually
wget https://huggingface.co/oneke/OneKE-13B/resolve/main/pytorch_model.bin
```

```python
# Configure with local path
config = ModelConfig(
    model_path="./models/oneke",
    device="cuda"
)

adapter = OneKEModelAdapter(config)
await adapter.load_model()
```

### Model Quantization

```python
from knowledge_engine.integrations.oneke.model_adapter import QuantizationMode

# No quantization (full precision, requires ~26GB VRAM)
config = ModelConfig(quantization=QuantizationMode.NONE)

# INT8 quantization (~13GB VRAM)
config = ModelConfig(quantization=QuantizationMode.INT8)

# INT4 quantization (~8GB VRAM) - Recommended
config = ModelConfig(quantization=QuantizationMode.INT4)

# FP16 mixed precision (~16GB VRAM)
config = ModelConfig(quantization=QuantizationMode.FP16)
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Model Configuration
ONEKE_MODEL_NAME=oneke/OneKE-13B
ONEKE_MODEL_PATH=/path/to/model
ONEKE_DEVICE=cuda
ONEKE_MAX_LENGTH=4096

# Quantization
ONEKE_QUANTIZATION=int4

# Generation Parameters
ONEKE_TEMPERATURE=0.1
ONEKE_TOP_P=0.9
ONEKE_TOP_K=50
ONEKE_NUM_BEAMS=1
ONEKE_DO_SAMPLE=true

# Schema Configuration
ONEKE_SCHEMA_DIR=./knowledge_engine/integrations/oneke/schemas

# Task Configuration
ONEKE_TASK_TIMEOUT=300
ONEKE_MAX_RETRIES=3
```

### Python Configuration

```python
from knowledge_engine.integrations.oneke.model_adapter import ModelConfig
from knowledge_engine.integrations.oneke.extraction_framework import TaskConfig

# Model configuration
model_config = ModelConfig(
    model_name="oneke/OneKE-13B",
    device="cuda",
    max_length=4096,
    quantization=QuantizationMode.INT4,
    temperature=0.1,
    top_p=0.9
)

# Task configuration
task_config = TaskConfig(
    ner_model="oneke/W2NER",
    re_model="oneke/TransformerRE",
    ee_model="oneke/EventExtractor",
    triple_model="oneke/OneKE-13B",
    task_timeout=300,
    max_retries=3
)
```

---

## Bilingual Extraction

### English Extraction

```python
from knowledge_engine.integrations.oneke.model_adapter import Language

# English text
text = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
The company is headquartered in Cupertino, California.
"""

# Extract entities
result = await adapter.extract_entities(
    text=text,
    language=Language.ENGLISH,
    correlation_id="ext_en_001"
)

print(result.entities)
# [
#   {"name": "Apple Inc.", "type": "Organization"},
#   {"name": "Steve Jobs", "type": "Person"},
#   {"name": "Steve Wozniak", "type": "Person"},
#   {"name": "Ronald Wayne", "type": "Person"},
#   {"name": "Cupertino", "type": "Location"},
#   {"name": "California", "type": "Location"}
# ]
```

### Chinese Extraction

```python
# Chinese text
text = """
苹果公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年创立。
公司总部位于加利福尼亚州库比蒂诺。
"""

# Extract with Chinese language
result = await adapter.extract_entities(
    text=text,
    language=Language.CHINESE,
    correlation_id="ext_zh_001"
)

print(result.entities)
# [
#   {"name": "苹果公司", "type": "Organization"},
#   {"name": "史蒂夫·乔布斯", "type": "Person"},
#   {"name": "史蒂夫·沃兹尼亚克", "type": "Person"},
#   {"name": "罗纳德·韦恩", "type": "Person"},
#   {"name": "库比蒂诺", "type": "Location"},
#   {"name": "加利福尼亚州", "type": "Location"}
# ]
```

### Mixed-Language Documents

```python
# Mixed text
text = """
Apple Inc. (苹果公司) is a multinational technology company headquartered in
Cupertino, California (加利福尼亚州库比蒂诺). It was founded by Steve Jobs (史蒂夫·乔布斯).
"""

# Use bilingual mode
result = await adapter.extract_entities(
    text=text,
    language=Language.BILINGUAL,
    correlation_id="ext_bi_001"
)

# Extracts entities from both languages
print(result.entities)
```

---

## Schema System

### Load Built-in Schema

```python
from knowledge_engine.integrations.oneke.schema_manager import OneKESchemaManager

# Initialize schema manager
schema_manager = OneKESchemaManager()

# Load built-in general schema
general_schema = await schema_manager.load_schema("general")

print(general_schema.entity_types)
# [
#   {"name": "Person", "description": "A person"},
#   {"name": "Organization", "description": "An organization"},
#   {"name": "Location", "description": "A location"},
#   ...
# ]
```

### Create Custom Schema

```python
# Define schema in JSON
company_schema = {
    "name": "company_domain",
    "version": "1.0.0",
    "description": "Schema for company-related extraction",
    "entity_types": [
        {
            "name": "Company",
            "description": "Business organization",
            "examples": ["Apple", "Microsoft", "Google"]
        },
        {
            "name": "CEO",
            "description": "Chief Executive Officer",
            "examples": ["Tim Cook", "Satya Nadella"]
        }
    ],
    "relation_types": [
        {
            "name": "led_by",
            "description": "Company is led by CEO",
            "domain": "Company",
            "range": "CEO"
        }
    ]
}

# Save schema
from knowledge_engine.integrations.oneke.schema_manager import SchemaDefinition

schema_def = SchemaDefinition(**company_schema)
schema_path = await schema_manager.save_schema(schema_def)
```

### Use Schema for Extraction

```python
# Load schema
schema = await schema_manager.load_schema("company_domain")

# Extract with schema guidance
result = await adapter.extract_entities(
    text="Tim Cook is the CEO of Apple Inc.",
    schema=schema.dict(),
    language=Language.ENGLISH
)

print(result.entities)
# Extracts entities matching schema types
```

### Schema Versioning

```python
# Update schema
updates = {
    "entity_types": [
        {
            "name": "Product",
            "description": "Company product",
            "examples": ["iPhone", "MacBook"]
        }
    ]
}

# This creates version 1.0.1 automatically
updated_schema = await schema_manager.update_schema(
    name="company_domain",
    updates=updates,
    create_version=True
)

# List versions
versions = await schema_manager.get_schema_versions("company_domain")
print(versions)  # ["1.0.0", "1.0.1"]
```

---

## Entity Linking

### Cross-Lingual Entity Linking

```python
# Link entities across languages
entities_en = await adapter.extract_entities(
    text="Steve Jobs founded Apple",
    language=Language.ENGLISH
)

entities_zh = await adapter.extract_entities(
    text="史蒂夫·乔布斯创立了苹果公司",
    language=Language.CHINESE
)

# Link entities (example)
# "Steve Jobs" <-> "史蒂夫·乔布斯"
# "Apple" <-> "苹果公司"
```

### Translation-Aware Linking

```python
# Use schema to link translations
schema = {
    "entity_types": [
        {
            "name": "Person",
            "translations": {
                "en": "Steve Jobs",
                "zh": "史蒂夫·乔布斯"
            }
        }
    ]
}
```

---

## Event Extraction

### Extract Events

```python
# Define event schema
event_schema = {
    "name": "business_events",
    "event_types": [
        {
            "name": "Acquisition",
            "description": "Company acquisition event",
            "arguments": ["acquirer", "target", "amount", "date"]
        },
        {
            "name": "ProductLaunch",
            "description": "Product launch event",
            "arguments": ["company", "product", "date"]
        }
    ]
}

# Extract events
from knowledge_engine.integrations.oneke.extraction_framework import MultiTaskExtractionFramework

framework = MultiTaskExtractionFramework()

result = await framework.extract_events(
    text="Microsoft acquired GitHub for $7.5 billion in 2018",
    schema=event_schema,
    language=Language.ENGLISH
)

print(result.events)
# [
#   {
#     "type": "Acquisition",
#     "acquirer": "Microsoft",
#     "target": "GitHub",
#     "amount": "$7.5 billion",
#     "date": "2018"
#   }
# ]
```

### Event Chains

```python
# Extract event sequences
text = """
In 1976, Steve Jobs founded Apple. In 1985, he left Apple and founded NeXT.
In 1997, Apple acquired NeXT, and Jobs returned to Apple.
"""

result = await framework.extract_events(
    text=text,
    schema=event_schema,
    language=Language.ENGLISH
)

# Events are temporally ordered
events = result.events
# Event 1: Found(Apple, Steve Jobs, 1976)
# Event 2: Leave(Apple, Steve Jobs, 1985)
# Event 3: Found(NeXT, Steve Jobs, 1985)
# Event 4: Acquire(Apple, NeXT, 1997)
# Event 5: Return(Apple, Steve Jobs, 1997)
```

---

## API Reference

### OneKEModelAdapter

#### `__init__(config: ModelConfig)`

Initialize model adapter.

```python
adapter = OneKEModelAdapter(config)
```

#### `async load_model() -> bool`

Load model into memory.

```python
success = await adapter.load_model()
```

#### `async extract_entities(text, schema, language, few_shot_examples, correlation_id) -> ExtractionResult`

Extract entities from text.

**Parameters:**
- `text` (str): Input text
- `schema` (Dict, optional): Schema definition
- `language` (Language): Target language (EN, ZH, BILINGUAL)
- `few_shot_examples` (List, optional): Example extractions
- `correlation_id` (str, optional): Tracking ID

**Returns:** `ExtractionResult`

#### `async extract_relations(text, entities, schema, language, few_shot_examples, correlation_id) -> ExtractionResult`

Extract relations between entities.

#### `async extract_triples(text, schema, language, few_shot_examples, correlation_id) -> ExtractionResult`

Extract (subject, predicate, object) triples.

#### `async unload()`

Unload model from memory.

```python
await adapter.unload()
```

### MultiTaskExtractionFramework

#### `__init__(task_config, model_config)`

Initialize multi-task framework.

```python
framework = MultiTaskExtractionFramework(
    task_config=TaskConfig(),
    model_config=ModelConfig()
)
```

#### `async extract(text, task, schema, language, few_shot_examples, correlation_id) -> ExtractionResult`

Extract with automatic task selection.

**Parameters:**
- `text` (str): Input text
- `task` (TaskType): Task type (NER, RE, EE, TRIPLE, AUTO)
- `schema` (Dict, optional): Schema definition
- `language` (Language): Target language
- `few_shot_examples` (List, optional): Examples
- `correlation_id` (str, optional): Tracking ID

**Returns:** `ExtractionResult`

#### `async extract_ner(...)` / `extract_relations(...)` / `extract_events(...)` / `extract_triples(...)`

Task-specific extraction methods.

### OneKESchemaManager

#### `async load_schema(name, version, format, correlation_id) -> SchemaDefinition`

Load schema from file or built-in library.

```python
schema = await schema_manager.load_schema("general")
```

#### `async save_schema(schema, format, create_version, correlation_id) -> str`

Save schema to file.

```python
path = await schema_manager.save_schema(schema_def)
```

#### `async update_schema(name, updates, create_version, correlation_id) -> SchemaDefinition`

Update existing schema.

```python
updated = await schema_manager.update_schema(
    name="general",
    updates={"entity_types": [...]}
)
```

#### `async list_schemas() -> List[Dict]`

List all available schemas.

```python
schemas = await schema_manager.list_schemas()
```

---

## Examples

### Example 1: Extract from News Article

```python
import asyncio
from knowledge_engine.integrations.oneke.model_adapter import OneKEModelAdapter, ModelConfig, Language
from knowledge_engine.integrations.oneke.schema_manager import OneKESchemaManager

async def extract_news():
    # Initialize
    adapter = OneKEModelAdapter(ModelConfig())
    await adapter.load_model()

    schema_manager = OneKESchemaManager()
    schema = await schema_manager.load_schema("general")

    # Extract
    text = """
    The European Union has approved a $2.1 billion climate fund to help developing
    countries reduce greenhouse gas emissions. The agreement was reached at the
    COP28 summit in Dubai, with representatives from 198 countries participating.
    """

    result = await adapter.extract_triples(
        text=text,
        schema=schema.dict(),
        language=Language.ENGLISH
    )

    print("Extracted Triples:")
    for triple in result.triples:
        print(f"  {triple}")

    await adapter.unload()

asyncio.run(extract_news())
```

### Example 2: Bilingual Document Processing

```python
async def process_bilingual():
    adapter = OneKEModelAdapter(ModelConfig())
    await adapter.load_model()

    # Mixed document
    text = """
    苹果公司 (Apple Inc.) 发布了新款 iPhone 15。
    The new phone features a titanium design and USB-C charging.
    售价从 799 美元起。 (Starting at $799)
    """

    result = await adapter.extract_entities(
        text=text,
        language=Language.BILINGUAL
    )

    # Entities from both languages
    for entity in result.entities:
        print(f"{entity['name']} ({entity['language']})")

    await adapter.unload()
```

### Example 3: Event Extraction from Financial News

```python
from knowledge_engine.integrations.oneke.extraction_framework import MultiTaskExtractionFramework

async def extract_financial_events():
    framework = MultiTaskExtractionFramework()

    schema = {
        "event_types": [
            {
                "name": "Merger",
                "arguments": ["company1", "company2", "value"]
            },
            {
                "name": "Investment",
                "arguments": ["investor", "company", "amount"]
            }
        ]
    }

    text = """
    Microsoft announced it will acquire Activision Blizzard for $68.7 billion.
    Separately, Sequoia Capital invested $50 million in the AI startup Anthropic.
    """

    result = await framework.extract_events(
        text=text,
        schema=schema,
        language=Language.ENGLISH
    )

    print("Extracted Events:")
    for event in result.events:
        print(f"  {event['type']}: {event}")

asyncio.run(extract_financial_events())
```

---

## Troubleshooting

### Model Won't Load

**Problem**: `RuntimeError: CUDA requested but not available`

**Solution**:
```python
# Use CPU instead
config = ModelConfig(device="cpu")
```

### Out of Memory

**Problem**: `CUDA out of memory`

**Solutions**:
```python
# 1. Use quantization
config = ModelConfig(quantization=QuantizationMode.INT4)

# 2. Reduce max length
config = ModelConfig(max_length=2048)

# 3. Use CPU
config = ModelConfig(device="cpu")
```

### Poor Extraction Quality

**Problem**: Model not extracting expected entities

**Solutions**:
```python
# 1. Use schema guidance
schema = {
    "entity_types": [
        {
            "name": "Company",
            "examples": ["Apple", "Microsoft"]
        }
    ]
}

# 2. Provide few-shot examples
examples = [
    {
        "text": "Tim Cook is CEO of Apple",
        "entities": [
            {"name": "Tim Cook", "type": "Person"},
            {"name": "Apple", "type": "Company"}
        ]
    }
]

# 3. Adjust temperature
config = ModelConfig(temperature=0.01)  # More deterministic
```

### Language Detection Issues

**Problem**: Wrong language detected

**Solution**:
```python
# Explicitly specify language
result = await adapter.extract_entities(
    text=text,
    language=Language.CHINESE  # Force Chinese
)
```

---

## Best Practices

### 1. Schema Design

- **Be Specific**: Define clear entity types with examples
- **Use Hierarchies**: Create parent-child relationships
- **Version Control**: Track schema changes over time

### 2. Extraction Quality

- **Use Schemas**: Always provide schema for better results
- **Few-Shot Learning**: Provide 2-3 examples for complex patterns
- **Temperature**: Use low temperature (0.1) for consistent results

### 3. Performance

- **Quantization**: Use INT4 for production deployments
- **Batch Processing**: Process multiple documents in parallel
- **Model Unloading**: Unload model when not in use

### 4. Bilingual Processing

- **Language Detection**: Let model auto-detect for mixed documents
- **Separate Processing**: Process large documents by language chunks
- **Translation**: Use consistent entity names across translations

### 5. Error Handling

```python
try:
    result = await adapter.extract_entities(text, language=Language.ENGLISH)
except RuntimeError as e:
    logger.error(f"Extraction failed: {e}")
    # Fallback to simpler extraction
    result = await simple_extract(text)
```

---

## Next Steps

- [Quick Start Guide](ONEKE_QUICK_START.md) - Get started in 5 minutes
- [Bilingual Extraction Tutorial](BILINGUAL_EXTRACTION_TUTORIAL.md) - Learn bilingual processing
- [Schema Definition Guide](SCHEMA_DEFINITION_GUIDE.md) - Create custom schemas
- [Event Extraction Guide](EVENT_EXTRACTION_GUIDE.md) - Extract events
- [API Reference](ONEKE_API_REFERENCE.md) - Complete API documentation

---

**Version:** 1.0.0
**Last Updated:** 2026-01-08
**Maintainer:** OpenEvolve Team
