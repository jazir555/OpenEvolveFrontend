# OneKE Quick Start Guide

**Get Started with Bilingual Knowledge Extraction in 5 Minutes**

---

## Prerequisites Check

```bash
# Check Python version (3.9+)
python --version

# Check CUDA availability (optional, for GPU)
nvidia-smi

# Check available RAM (16GB+ recommended)
free -h  # Linux
systeminfo | findstr /C:"Memory" # Windows
```

---

## Installation (2 minutes)

```bash
# Step 1: Install dependencies
pip install transformers>=4.35.0 torch pydantic pyyaml

# Step 2: Set environment variables
export ONEKE_MODEL_NAME="oneke/OneKE-13B"
export ONEKE_DEVICE="cpu"  # Use "cuda" for GPU
export ONEKE_QUANTIZATION="int4"

# Step 3: Verify installation
python -c "import transformers; print(transformers.__version__)"
```

---

## Your First Extraction (3 minutes)

### Step 1: Initialize Model

```python
import asyncio
from knowledge_engine.integrations.oneke.model_adapter import (
    OneKEModelAdapter, ModelConfig, Language
)

async def quick_start():
    # Initialize adapter with defaults
    adapter = OneKEModelAdapter(ModelConfig(
        device="cpu",        # Use "cuda" for GPU
        quantization="none"  # No quantization for CPU
    ))

    # Load model (takes ~1-2 minutes first time)
    await adapter.load_model()
    print("Model loaded!")

    return adapter

# Run
adapter = asyncio.run(quick_start())
```

### Step 2: Extract Entities (English)

```python
async def extract_english():
    text = "Apple Inc. was founded by Steve Jobs in 1976."

    # Extract entities
    result = await adapter.extract_entities(
        text=text,
        language=Language.ENGLISH
    )

    print("Extracted Entities:")
    for entity in result.entities:
        print(f"  - {entity.get('name')} ({entity.get('type', 'Unknown')})")

asyncio.run(extract_english())

# Output:
# Extracted Entities:
#   - Apple Inc. (Organization)
#   - Steve Jobs (Person)
#   - 1976 (Date)
```

### Step 3: Extract Entities (Chinese)

```python
async def extract_chinese():
    text = "苹果公司由史蒂夫·乔布斯于1976年创立。"

    # Extract entities
    result = await adapter.extract_entities(
        text=text,
        language=Language.CHINESE
    )

    print("提取的实体:")
    for entity in result.entities:
        print(f"  - {entity.get('name')} ({entity.get('type', 'Unknown')})")

asyncio.run(extract_chinese())

# Output:
# 提取的实体:
#   - 苹果公司 (Organization)
#   - 史蒂夫·乔布斯 (Person)
#   - 1976 (Date)
```

### Step 4: Extract Relations

```python
async def extract_relations():
    text = "Tim Cook is the CEO of Apple, succeeding Steve Jobs."

    # Extract relations
    result = await adapter.extract_relations(
        text=text,
        language=Language.ENGLISH
    )

    print("Extracted Relations:")
    for rel in result.relations:
        print(f"  - {rel}")

asyncio.run(extract_relations())

# Output:
# Extracted Relations:
#   - (Tim Cook, CEO of, Apple)
#   - (Tim Cook, succeeding, Steve Jobs)
```

### Step 5: Clean Up

```python
async def cleanup():
    # Unload model to free memory
    await adapter.unload()
    print("Model unloaded!")

asyncio.run(cleanup())
```

---

## Using Schema Guidance

### Load Built-in Schema

```python
from knowledge_engine.integrations.oneke.schema_manager import OneKESchemaManager

async def extract_with_schema():
    adapter = OneKEModelAdapter(ModelConfig(device="cpu"))
    await adapter.load_model()

    # Load built-in schema
    schema_manager = OneKESchemaManager()
    schema = await schema_manager.load_schema("general")

    # Extract with schema
    text = "The United Nations headquarters is in New York City."
    result = await adapter.extract_entities(
        text=text,
        schema=schema.dict(),
        language=Language.ENGLISH
    )

    print("Entities with schema:")
    for entity in result.entities:
        print(f"  - {entity.get('name')} ({entity.get('type', 'Unknown')})")

    await adapter.unload()

asyncio.run(extract_with_schema())
```

---

## Common Use Cases

### Extract from News Article

```python
async def extract_news():
    adapter = OneKEModelAdapter(ModelConfig(device="cpu"))
    await adapter.load_model()

    text = """
    Microsoft announced it will acquire Activision Blizzard for $68.7 billion.
    The deal, Microsoft's largest ever, will position the company as a leading
    gaming company. Activision Blizzard is known for games like Call of Duty
    and World of Warcraft.
    """

    # Extract triples (subject, predicate, object)
    result = await adapter.extract_triples(
        text=text,
        language=Language.ENGLISH
    )

    print("News Triples:")
    for triple in result.triples:
        print(f"  ({triple.get('subject')}, {triple.get('predicate')}, {triple.get('object')})")

    await adapter.unload()

asyncio.run(extract_news())
```

### Extract from Academic Paper

```python
async def extract_academic():
    adapter = OneKEModelAdapter(ModelConfig(device="cpu"))
    await adapter.load_model()

    text = """
    Deep learning has revolutionized computer vision. In 2012, AlexNet won
    the ImageNet competition with a top-5 error rate of 15.3%. This was a
    significant improvement over previous methods.
    """

    result = await adapter.extract_entities(
        text=text,
        language=Language.ENGLISH
    )

    print("Academic Entities:")
    for entity in result.entities:
        print(f"  - {entity.get('name')} ({entity.get('type', 'Unknown')})")

    await adapter.unload()

asyncio.run(extract_academic())
```

---

## Configuration Tips

### For CPU (Slower but works everywhere)

```python
config = ModelConfig(
    device="cpu",
    quantization="none",  # CPU doesn't support quantization
    max_length=2048       # Reduce for memory efficiency
)
```

### For GPU (Faster)

```python
config = ModelConfig(
    device="cuda",
    quantization="int4",   # Use int4 for 8GB+ VRAM
    max_length=4096
)
```

### For Better Quality

```python
config = ModelConfig(
    temperature=0.01,     # More deterministic
    top_p=0.9,           # Nucleus sampling
    num_beams=1          # Greedy decoding
)
```

---

## Troubleshooting

### Problem: Out of Memory

```python
# Solution: Use CPU or reduce max_length
config = ModelConfig(
    device="cpu",
    max_length=1024
)
```

### Problem: Slow Extraction

```python
# Solution: Use GPU if available
config = ModelConfig(
    device="cuda",
    quantization="int4"
)
```

### Problem: Poor Results

```python
# Solution: Use schema with examples
schema = {
    "entity_types": [
        {
            "name": "Company",
            "examples": ["Apple", "Microsoft", "Google"]
        }
    ]
}
result = await adapter.extract_entities(
    text=text,
    schema=schema
)
```

---

## Next Steps

- [OneKE Integration Guide](ONEKE_INTEGRATION_GUIDE.md) - Complete documentation
- [Bilingual Extraction Tutorial](BILINGUAL_EXTRACTION_TUTORIAL.md) - Learn bilingual processing
- [Schema Definition Guide](SCHEMA_DEFINITION_GUIDE.md) - Create custom schemas
- [API Reference](ONEKE_API_REFERENCE.md) - Complete API reference

---

**Version:** 1.0.0
**Last Updated:** 2026-01-08

**Quick Start Complete!** You successfully extracted entities from English and Chinese text.
