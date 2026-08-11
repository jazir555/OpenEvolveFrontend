# OneKE Integration Guide

Complete guide for integrating OneKE bilingual knowledge extraction into OpenEvolve.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Core Components](#core-components)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## Overview

OneKE (One Knowledge Extraction) is a bilingual (English/Chinese) knowledge extraction system that provides:

- **Named Entity Recognition (NER)**: Extract entities like people, organizations, locations
- **Relation Extraction (RE)**: Identify relationships between entities
- **Event Extraction (EE)**: Detect and extract events with temporal information
- **Cross-Lingual Entity Linking**: Match entities across languages
- **Schema-Guided Extraction**: Use custom schemas for domain-specific extraction

### Key Features

- **Bilingual Support**: Native English and Chinese language processing
- **High Accuracy**: State-of-the-art extraction models
- **Flexible Schemas**: Define custom entity, relation, and event types
- **Event Chains**: Build temporal sequences of events
- **Causal Relations**: Extract cause-effect relationships
- **Production Ready**: Robust error handling, logging, and monitoring

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     OneKE Integration                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Model       │  │ Extraction   │  │   Schema     │     │
│  │  Adapter     │  │ Framework    │  │   Manager    │     │
│  │              │  │              │  │              │     │
│  │ - Load model │  │ - NER        │  │ - Load/valid │     │
│  │ - Inference  │  │ - RE         │  │ - Customize  │     │
│  │ - Quantize   │  │ - EE         │  │ - Cache      │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│           │                 │                  │            │
│           └─────────────────┴──────────────────┘            │
│                            │                                │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Cross-Lingual Entity Linker          │      │
│  │  - Bilingual matching  - Translation-aware       │      │
│  │  - Deduplication      - KG format                │      │
│  └──────────────────────────────────────────────────┘      │
│                            │                                │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Event Extraction Pipeline            │      │
│  │  - Event detection    - Argument extraction      │      │
│  │  - Event chains       - Causal relations         │      │
│  │  - Temporal ordering  - Complete pipeline        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Text document (EN/CN/bilingual)
2. **Schema Selection**: Load or define extraction schema
3. **Model Inference**: Run extraction models
4. **Entity Linking**: Match and deduplicate entities
5. **Event Processing**: Extract and chain events
6. **Output**: Structured knowledge graph (JSON)

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM (32GB recommended)

### Install Dependencies

```bash
# Core dependencies
pip install torch>=2.0.0 transformers>=4.30.0
pip install scikit-learn rapidfuzz numpy
pip install pydantic python-dotenv

# Optional: for development
pip install pytest pytest-asyncio pytest-cov
```

### Environment Variables

Create `.env` file:

```bash
# OneKE Model Configuration
ONEKE_MODEL_NAME=oneke/OneKE-13B
ONEKE_MODEL_PATH=/path/to/model
ONEKE_DEVICE=cuda  # or cpu
ONEKE_MAX_LENGTH=4096
ONEKE_QUANTIZATION=none  # none/int8/int4/fp16

# Extraction Configuration
ONEKE_CONFIDENCE_THRESHOLD=0.6
ONEKE_MAX_EVENTS_PER_DOC=50
ONEKE_ENABLE_CAUSAL_EXTRACTION=true

# Translation Service (optional)
ONEKE_TRANSLATION_API=https://api.translation.service
ONEKE_TRANSLATION_MODEL=google
ONEKE_ENABLE_TRANSLATION=true
```

---

## Configuration

### Model Configuration

```python
from knowledge_engine.integrations.oneke import ModelConfig, QuantizationMode

config = ModelConfig(
    model_name="oneke/OneKE-13B",
    device="cuda",
    max_length=4096,
    quantization=QuantizationMode.INT4,  # Reduce memory
    temperature=0.1,
    top_p=0.9
)
```

### Task Configuration

```python
from knowledge_engine.integrations.oneke import TaskConfig

task_config = TaskConfig(
    ner_model="oneke/W2NER",
    re_model="oneke/TransformerRE",
    ee_model="oneke/EventExtractor",
    task_timeout=300,
    max_retries=3
)
```

### Linker Configuration

```python
from knowledge_engine.integrations.oneke import LinkerConfig

linker_config = LinkerConfig(
    fuzzy_threshold=85,
    semantic_threshold=0.7,
    enable_translation=True,
    cache_translations=True
)
```

---

## Core Components

### 1. Model Adapter

Handles model loading and inference.

```python
from knowledge_engine.integrations.oneke import OneKEModelAdapter, ModelConfig

# Initialize
config = ModelConfig(model_name="oneke/OneKE-13B")
adapter = OneKEModelAdapter(config)

# Load model
await adapter.load_model()

# Extract knowledge
result = await adapter.extract(
    text="Apple announced the iPhone in 2007.",
    schema=custom_schema,
    language=Language.ENGLISH
)

print(result.entities)
# [{'id': 'E1', 'type': 'ORGANIZATION', 'text': 'Apple'},
#  {'id': 'E2', 'type': 'PRODUCT', 'text': 'iPhone'}]
```

### 2. Extraction Framework

Coordinates multiple extraction models.

```python
from knowledge_engine.integrations.oneke import MultiTaskExtractionFramework

# Initialize
framework = MultiTaskExtractionFramework(task_config, model_config)

# Extract entities
entities = await framework.extract_ner(
    text="Steve Jobs founded Apple in 1976.",
    language=Language.ENGLISH
)

# Extract relations
relations = await framework.extract_relations(
    text="Steve Jobs founded Apple",
    entities=entities,
    language=Language.ENGLISH
)

# Extract events
events = await framework.extract_events(
    text="Apple launched the iPhone in 2007",
    language=Language.ENGLISH
)
```

### 3. Schema Manager

Manages extraction schemas.

```python
from knowledge_engine.integrations.oneke import OneKESchemaManager

manager = OneKESchemaManager()

# Load schema
schema = await manager.load_schema("schemas/general_schema.json")

# Validate schema
is_valid = await manager.validate_schema(schema)

# Customize schema
custom_schema = await manager.create_custom_schema(
    entity_types=["PERSON", "ORG", "PRODUCT"],
    relation_types=["WORKS_FOR", "FOUNDED_BY"],
    event_types=["LAUNCH", "ACQUISITION"]
)
```

### 4. Entity Linker

Cross-lingual entity matching and linking.

```python
from knowledge_engine.integrations.oneke import CrossLingualEntityLinker, Entity

# Initialize
linker = CrossLingualEntityLinker()

# Create entities
entity1 = Entity(
    entity_id="E1",
    name_en=["Apple Inc."],
    name_zh=["苹果公司"],
    type="ORGANIZATION"
)

entity2 = Entity(
    entity_id="E2",
    name_en=["Apple"],
    name_zh=["苹果"],
    type="ORGANIZATION"
)

# Match entities
match_result = await linker.match_entities(
    entity1, entity2,
    strategy=MatchStrategy.HYBRID
)

print(match_result.matched, match_result.confidence)
# True, 0.95

# Deduplicate
clusters = await linker.deduplicate_entities([entity1, entity2])

# Export to bilingual KG
kg = linker.to_bilingual_kg([entity1, entity2])
```

### 5. Event Extractor

Event extraction and chaining.

```python
from knowledge_engine.integrations.oneke import EventExtractionPipeline

# Initialize
pipeline = EventExtractionPipeline()

# Extract events from text
events = await pipeline.extract_events(
    text="In 2007, Apple announced the iPhone. The device was released in June.",
    language=Language.ENGLISH
)

# Build event chains
chains = await pipeline.build_event_chains(events)

# Extract causal relations
causal_relations = await pipeline.extract_causal_relations(
    events,
    text="The announcement caused excitement in the tech industry",
    language=Language.ENGLISH
)

# Run complete pipeline
result = await pipeline.extract_complete_pipeline(
    text=document_text,
    language=Language.ENGLISH
)
```

---

## Usage Examples

### Example 1: Basic Entity Extraction

```python
import asyncio
from knowledge_engine.integrations.oneke import (
    OneKEModelAdapter, ModelConfig, Language
)

async def extract_entities():
    # Initialize
    config = ModelConfig(model_name="oneke/OneKE-13B")
    adapter = OneKEModelAdapter(config)
    await adapter.load_model()

    # Extract
    text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak,
    and Ronald Wayne in 1976. The company is headquartered
    in Cupertino, California.
    """

    result = await adapter.extract(
        text=text,
        schema=general_schema,
        language=Language.ENGLISH
    )

    # Process results
    for entity in result.entities:
        print(f"{entity['type']}: {entity['text']}")

    for relation in result.relations:
        print(f"{relation['subject']} -> {relation['type']} -> {relation['object']}")

asyncio.run(extract_entities())
```

### Example 2: Bilingual Extraction

```python
from knowledge_engine.integrations.oneke import (
    CrossLingualEntityLinker, Entity, LinkerLanguage
)

async def bilingual_extraction():
    linker = CrossLingualEntityLinker()

    # Detect language
    lang = await linker.detect_language("这是中文文本")
    print(f"Detected: {lang}")

    # Create bilingual entities
    entities = [
        Entity(
            entity_id="E1",
            name_en=["Microsoft"],
            name_zh=["微软"],
            type="ORGANIZATION",
            language=LinkerLanguage.BILINGUAL
        ),
        Entity(
            entity_id="E2",
            name_en=["Google"],
            name_zh=["谷歌"],
            type="ORGANIZATION",
            language=LinkerLanguage.BILINGUAL
        )
    ]

    # Match across languages
    for entity in entities:
        await linker.add_entity(entity)

    # Create bilingual KG
    kg = linker.to_bilingual_kg(entities)

    return kg

asyncio.run(bilingual_extraction())
```

### Example 3: Event Extraction

```python
from knowledge_engine.integrations.oneke import EventExtractionPipeline, Language

async def extract_events():
    pipeline = EventExtractionPipeline()

    text = """
    In January 2007, Apple announced the iPhone at Macworld.
    Steve Jobs introduced the device to the public.
    The iPhone was released on June 29, 2007.
    This launch revolutionized the smartphone industry.
    """

    result = await pipeline.extract_complete_pipeline(
        text=text,
        language=Language.ENGLISH
    )

    print(f"Extracted {result['metadata']['num_events']} events")
    print(f"Built {result['metadata']['num_chains']} chains")
    print(f"Found {result['metadata']['num_causal_relations']} causal relations")

    return result

asyncio.run(extract_events())
```

### Example 4: Domain-Specific Extraction

```python
from knowledge_engine.integrations.oneke import OneKESchemaManager

async def biomedical_extraction():
    # Load biomedical schema
    manager = OneKESchemaManager()
    schema = await manager.load_schema("schemas/biomedical_schema.json")

    # Extract using schema
    text = """
    COVID-19 is caused by the SARS-CoV-2 virus.
    Common symptoms include fever, cough, and fatigue.
    Vaccines from Pfizer and Moderna are effective treatments.
    """

    result = await adapter.extract(
        text=text,
        schema=schema,
        language=Language.ENGLISH
    )

    # Results will include DISEASE, DRUG, SYMPTOM entities
    # and TREATS, CAUSES relations

    return result

asyncio.run(biomedical_extraction())
```

---

## API Reference

### ModelConfig

Configuration for OneKE model adapter.

**Parameters:**
- `model_name` (str): HuggingFace model name
- `device` (str): Device to run on ("cuda" or "cpu")
- `max_length` (int): Maximum sequence length
- `quantization` (QuantizationMode): Quantization mode
- `temperature` (float): Generation temperature
- `top_p` (float): Top-p sampling parameter

### ExtractionResult

Result from knowledge extraction.

**Attributes:**
- `entities` (List[Dict]): Extracted entities
- `relations` (List[Dict]): Extracted relations
- `events` (List[Dict]): Extracted events
- `confidence` (float): Overall confidence
- `language` (Language): Detected language
- `timestamp` (datetime): Extraction timestamp (UTC)

### Entity

Bilingual entity representation.

**Parameters:**
- `entity_id` (str): Unique identifier
- `name_en` (List[str]): English names
- `name_zh` (List[str]): Chinese names
- `aliases_en` (List[str]): English aliases
- `aliases_zh` (List[str]): Chinese aliases
- `type` (str): Entity type
- `language` (Language): Primary language

**Methods:**
- `get_all_names()`: Get all names by language

### TemporalEvent

Temporal event representation.

**Parameters:**
- `event_id` (str): Unique identifier
- `event_type` (EventType): Event type
- `trigger` (str): Trigger text
- `arguments` (List[EventArgument]): Event arguments
- `timestamp` (datetime): Event timestamp (UTC)
- `certainty` (float): Certainty score

**Methods:**
- `get_argument(role)`: Get argument by role
- `to_dict()`: Convert to dictionary

---

## Performance Optimization

### Model Quantization

Reduce memory usage with quantization:

```python
from knowledge_engine.integrations.oneke import QuantizationMode

# INT4 quantization (smallest, fastest)
config = ModelConfig(quantization=QuantizationMode.INT4)

# INT8 quantization (balanced)
config = ModelConfig(quantization=QuantizationMode.INT8)

# FP16 (half precision)
config = ModelConfig(quantization=QuantizationMode.FP16)
```

### Batch Processing

Process multiple documents efficiently:

```python
async def batch_extract(documents):
    results = []
    for doc in documents:
        result = await adapter.extract(
            text=doc['text'],
            schema=schema,
            language=doc['language']
        )
        results.append(result)
    return results
```

### Caching

Enable caching for repeated extractions:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
async def cached_extract(text_hash):
    return await adapter.extract(text=text, schema=schema)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Problem:** CUDA out of memory error

**Solutions:**
- Reduce `max_length` in config
- Use quantization (INT4/INT8)
- Reduce batch size
- Use CPU instead of GPU

```python
config = ModelConfig(
    max_length=2048,  # Reduce from 4096
    quantization=QuantizationMode.INT4
)
```

#### 2. Low Confidence Results

**Problem:** Extraction confidence below threshold

**Solutions:**
- Lower confidence threshold
- Improve input text quality
- Use domain-specific schema
- Adjust model parameters

```python
# Lower threshold
config = ExtractorConfig(confidence_threshold=0.5)

# Adjust temperature
model_config = ModelConfig(temperature=0.3)
```

#### 3. Slow Inference

**Problem:** Extraction taking too long

**Solutions:**
- Use GPU acceleration
- Enable quantization
- Reduce max_length
- Use smaller model

```python
config = ModelConfig(
    device="cuda",
    quantization=QuantizationMode.INT4,
    max_length=2048
)
```

#### 4. Language Detection Errors

**Problem:** Incorrect language detection

**Solutions:**
- Manually specify language
- Use bilingual mode
- Check input text encoding

```python
# Specify language manually
result = await adapter.extract(
    text=text,
    schema=schema,
    language=Language.CHINESE  # Force Chinese
)
```

---

## Best Practices

### 1. Schema Design

- **Be Specific**: Use descriptive type names
- **Provide Examples**: Include bilingual examples
- **Define Inverses**: Specify inverse relationships
- **Test Thoroughly**: Validate on sample documents

### 2. Entity Linking

- **Use Hybrid Strategy**: Combine multiple matching methods
- **Set Appropriate Thresholds**: Tune fuzzy and semantic thresholds
- **Enable Translation**: For cross-lingual matching
- **Validate Results**: Review match confidence scores

### 3. Event Extraction

- **Preprocess Text**: Clean and structure input
- **Use Domain Schemas**: Improve extraction accuracy
- **Build Event Chains**: Understand event sequences
- **Extract Causal Relations**: Identify cause-effect relationships

### 4. Performance

- **Use Quantization**: Reduce memory and speed up inference
- **Batch Processing**: Process multiple documents together
- **Cache Results**: Avoid repeated extractions
- **Monitor Resources**: Track GPU/CPU usage

### 5. Error Handling

- **Validate Inputs**: Check text quality and length
- **Handle Failures**: Implement retry logic
- **Log Errors**: Use structured logging
- **Fallback Strategies**: Provide alternative extraction methods

---

## Next Steps

- See [BILINGUAL_EXTRACTION_TUTORIAL.md](BILINGUAL_EXTRACTION_TUTORIAL.md) for detailed examples
- See [SCHEMA_DEFINITION_GUIDE.md](SCHEMA_DEFINITION_GUIDE.md) for schema creation
- Run probe scripts to verify installation:
  ```bash
  python knowledge_engine/integrations/oneke/probes/check_model_adapter.py
  python knowledge_engine/integrations/oneke/probes/check_bilingual_extraction.py
  python knowledge_engine/integrations/oneke/probes/check_event_extraction.py
  ```

---

## Support

For issues or questions:

1. Check troubleshooting section
2. Run probe scripts for diagnostics
3. Review test examples
4. Check logs for error details

---

**Version**: 1.0.0
**Last Updated**: 2025-01-08
**Status**: Production Ready
