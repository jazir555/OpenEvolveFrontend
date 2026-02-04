# DeepKE Integration Guide

## Overview

The DeepKE integration provides advanced knowledge extraction capabilities to the Knowledge Engine. DeepKE is a comprehensive toolkit for knowledge graph construction, supporting entity recognition, relation extraction, and triple extraction from unstructured text.

### Key Features
- Named Entity Recognition (NER)
- Relation Extraction (RE)
- Triple Extraction
- Document-level extraction
- Few-shot learning support
- Multi-modal capabilities
- Pre-trained model support

### Use Cases
- Building knowledge graphs from text
- Extracting entity relationships from documents
- Information extraction pipelines
- Knowledge base construction
- Semantic analysis of text corpora

## Installation

```bash
# Core installation
pip install deepke

# With specific modules
pip install deepke[relation-extraction]  # For relation extraction
pip install deepke[entity-extraction]  # For entity extraction
pip install deepke[document-level]  # For document-level extraction
pip install deepke[multimodal]  # For multi-modal extraction

# With Knowledge Engine
pip install knowledge-engine[deepke]

# Optional: GPU support
pip install deepke[gpu]  # For CUDA acceleration
```

### Configuration

Set up environment variables:

```bash
export DEEPKE_MODEL_DIR="/path/to/models"
export DEEPKE_DEVICE="cuda"  # or "cpu"
export DEEPKE_CACHE_DIR="/path/to/cache"
```

## Quick Start

### Basic Usage

```python
from knowledge_engine.integrations import DeepKEIntegration

# Initialize with default configuration
integration = DeepKEIntegration()

# Extract entities and relations
result = await integration.extract_entities_relations(
    text="Apple Inc. was founded by Steve Jobs in Cupertino, California."
)

if result.success:
    print(f"Entities: {result.entities}")
    print(f"Relations: {result.relations}")
    print(f"Triples: {result.triples}")
    # Output:
    # Entities: [
    #   {"text": "Apple Inc.", "type": "ORGANIZATION"},
    #   {"text": "Steve Jobs", "type": "PERSON"},
    #   {"text": "Cupertino", "type": "LOCATION"},
    #   {"text": "California", "type": "LOCATION"}
    # ]
    # Relations: [
    #   {"head": "Apple Inc.", "relation": "founded_by", "tail": "Steve Jobs"},
    #   {"head": "Apple Inc.", "relation": "located_in", "tail": "Cupertino"}
    # ]
    # Triples: [
    #   ("Apple Inc.", "founded_by", "Steve Jobs"),
    #   ("Apple Inc.", "located_in", "Cupertino")
    # ]
```

### Extraction Only

```python
# Extract just entities
entities_result = await integration.extract_entities(
    text="Barack Obama was born in Hawaii."
)

# Extract just relations
relations_result = await integration.extract_relations(
    text="Google acquired YouTube in 2006.",
    entities=[{"text": "Google", "type": "ORG"},
              {"text": "YouTube", "type": "ORG"}]
)

# Extract triples directly
triples_result = await integration.extract_triples(
    text="Elon Musk founded SpaceX in 2002."
)
```

## Configuration Options

### Full Configuration Schema

```python
config = {
    # Model Configuration
    "model_type": "standard",  # standard, document, few_shot, multimodal
    "model_name": "deepke/relation-extraction",
    "device": "cuda",  # cuda, cpu, mps
    "max_length": 512,
    "batch_size": 16,

    # Training Configuration
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "valid_steps": 100,
    "save_steps": 500,
    "logging_steps": 10,
    "output_dir": "./checkpoints",
    "overwrite_cache": False,

    # Extraction Configuration
    "entity_types": [
        "PERSON", "ORGANIZATION", "LOCATION",
        "DATE", "NUMBER", "MISC"
    ],
    "relation_types": [
        "founded_by", "located_in", "acquired",
        "works_for", "born_in", "part_of"
    ],
    "confidence_threshold": 0.5,
    "max_triples": 50,

    # Few-shot Configuration
    "few_shot": {
        "enabled": False,
        "k": 5,  # Number of examples
        "shots": [
            {"text": "Example 1", "entities": [], "relations": []},
            # ... more examples
        ]
    },

    # Document-level Configuration
    "document_level": {
        "enabled": False,
        "max_sentences": 100,
        "cross_sentence": True,
        "coreference": True
    },

    # Post-processing
    "post_processing": {
        "deduplication": True,
        "entity_normalization": True,
        "relation_validation": True
    }
}
```

## API Reference

### Core Methods

#### `extract_entities(text, options)`

Extract named entities from text.

**Parameters:**
- `text` (str): Input text
- `options` (dict, optional): Extraction options

**Returns:** `DeepKEResult` object
- `entities` (List[dict]): Extracted entities
- Each entity contains:
  - `text` (str): Entity text
  - `type` (str): Entity type
  - `start` (int): Start position
  - `end` (int): End position
  - `confidence` (float): Confidence score

**Example:**
```python
result = await integration.extract_entities(
    text="Apple Inc. is based in Cupertino.",
    options={"entity_types": ["ORGANIZATION", "LOCATION"]}
)
```

#### `extract_relations(text, entities, options)`

Extract relations between entities.

**Parameters:**
- `text` (str): Input text
- `entities` (List[dict], optional): Pre-extracted entities
- `options` (dict, optional): Extraction options

**Returns:** `DeepKEResult` object
- `relations` (List[dict]): Extracted relations
- Each relation contains:
  - `head` (str): Head entity
  - `relation` (str): Relation type
  - `tail` (str): Tail entity
  - `confidence` (float): Confidence score

**Example:**
```python
result = await integration.extract_relations(
    text="Steve Jobs co-founded Apple Inc.",
    options={"relation_types": ["co-founded_by", "founder_of"]}
)
```

#### `extract_triples(text, options)`

Extract knowledge triples (subject, predicate, object).

**Parameters:**
- `text` (str): Input text
- `options` (dict, optional): Extraction options

**Returns:** `DeepKEResult` object
- `triples` (List[Tuple[str, str, str]]): Knowledge triples
- Each triple is (subject, predicate, object)

**Example:**
```python
result = await integration.extract_triples(
    text="Python was created by Guido van Rossum in 1991."
)
# Returns: [("Python", "created_by", "Guido van Rossum")]
```

#### `extract_entities_relations(text, options)`

Extract both entities and relations in one call.

**Parameters:**
- `text` (str): Input text
- `options` (dict, optional): Extraction options

**Returns:** `DeepKEResult` object with both `entities` and `relations`

#### `batch_extract(texts, options)`

Extract from multiple texts in batch.

**Parameters:**
- `texts` (List[str]): List of input texts
- `options` (dict, optional): Extraction options

**Returns:** List of `DeepKEResult` objects

**Example:**
```python
texts = [
    "Text 1...",
    "Text 2...",
    "Text 3..."
]
results = await integration.batch_extract(texts, batch_size=8)
```

## Advanced Usage

### Document-Level Extraction

Extract relations across sentences:

```python
config = {
    "document_level": {
        "enabled": True,
        "cross_sentence": True,
        "coreference": True  # Resolve coreferences
    }
}
integration = DeepKEIntegration(config=config)

text = """
Apple Inc. is a technology company.
It was founded by Steve Jobs.
The company is headquartered in Cupertino.
"""

result = await integration.extract_entities_relations(text, config)
# Can extract relations between "Apple Inc." and "Steve Jobs"
# even though they're in different sentences
```

### Few-Shot Learning

Extract with few examples:

```python
config = {
    "few_shot": {
        "enabled": True,
        "k": 3,
        "shots": [
            {
                "text": "Microsoft was founded by Bill Gates.",
                "triples": [("Microsoft", "founded_by", "Bill Gates")]
            },
            {
                "text": "Amazon was founded by Jeff Bezos.",
                "triples": [("Amazon", "founded_by", "Jeff Bezos")]
            },
            {
                "text": "Tesla is led by Elon Musk.",
                "triples": [("Tesla", "led_by", "Elon Musk")]
            }
        ]
    }
}
integration = DeepKEIntegration(config=config)

result = await integration.extract_triples(
    text="SpaceX was founded by Elon Musk."
)
# Will use the few-shot examples to guide extraction
```

### Custom Entity and Relation Types

Define custom types for your domain:

```python
config = {
    "entity_types": [
        "COMPANY",
        "PERSON",
        "PRODUCT",
        "TECHNOLOGY",
        "DATE"
    ],
    "relation_types": [
        "develops",
        "acquired",
        "competes_with",
        "uses",
        "released_on"
    ]
}
integration = DeepKEIntegration(config=config)

result = await integration.extract_entities_relations(
    text="Apple developed the iPhone, which competes with Android."
)
```

### Training on Custom Data

Train DeepKE on your own data:

```python
# Prepare training data
train_data = [
    {
        "text": "Apple acquired Siri in 2010.",
        "entities": [
            {"text": "Apple", "type": "COMPANY", "start": 0, "end": 5},
            {"text": "Siri", "type": "PRODUCT", "start": 15, "end": 19},
            {"text": "2010", "type": "DATE", "start": 23, "end": 27}
        ],
        "relations": [
            {"head": "Apple", "relation": "acquired", "tail": "Siri"}
        ]
    },
    # ... more examples
]

# Train
await integration.train(
    train_data=train_data,
    output_dir="./custom_model",
    num_epochs=5,
    learning_rate=3e-5
)

# Use custom model
custom_integration = DeepKEIntegration(config={
    "model_name": "./custom_model"
})
```

## Integration with Knowledge Engine

### Using with Entity Knowledge Graph

```python
from knowledge_engine.integrations import DeepKEIntegration, ROMAEntityExtractor

# Extract entities and relations
deepke = DeepKEIntegration()
result = await deepke.extract_entities_relations(
    text="Your document text here..."
)

# Add to knowledge graph
roma = ROMAEntityExtractor()
for entity in result.entities:
    await roma.add_entity(
        entity_type=entity["type"],
        name=entity["text"],
        properties={"confidence": entity["confidence"]}
    )

for relation in result.relations:
    await roma.add_relation(
        from_entity=relation["head"],
        relation_type=relation["relation"],
        to_entity=relation["tail"],
        properties={"confidence": relation["confidence"]}
   )
```

### Using with ROMA-DeepKE Integration

```python
from knowledge_engine.integrations import ROMADeepKEIntegration

# Use the integrated ROMA-DeepKE pipeline
roma_deepke = ROMADeepKEIntegration()

# Extract and store in knowledge graph
result = await roma_deepke.extract_and_store(
    text="Your document text...",
    graph_name="knowledge_graph"
)
```

### Pipeline with DSPy

```python
from knowledge_engine.integrations import DeepKEIntegration, DSPyIntegration

# Extract knowledge
deepke = DeepKEIntegration()
kg_result = await deepke.extract_entities_relations(text)

# Reason about the knowledge
dspy = DSPyIntegration()
reasoning = await dspy.chain_of_thought(
    query="What relationships exist between the entities?",
    context={"triples": kg_result.triples}
)
```

## Performance Considerations

### GPU Acceleration

```python
import torch

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "device": device,
    "batch_size": 32 if device == "cuda" else 8
}
integration = DeepKEIntegration(config=config)
```

### Batch Processing

Process multiple documents efficiently:

```python
texts = [doc1, doc2, doc3, ...]  # Large corpus

# Process in batches
results = await integration.batch_extract(
    texts=texts,
    batch_size=16,  # Adjust based on GPU memory
    num_workers=4  # Parallel processing
)
```

### Memory Management

```python
config = {
    "max_length": 256,  # Reduce max length
    "batch_size": 8,  # Reduce batch size
    "gradient_checkpointing": True  # Enable gradient checkpointing
}
```

### Caching

```python
# Enable model caching
config = {
    "cache_dir": "./deepke_cache",
    "use_cache": True
}
```

## Error Handling

### Common Errors

1. **CUDA Out of Memory**
   ```python
   # Solution: Reduce batch size or use CPU
   config = {
       "device": "cpu",
       "batch_size": 4
   }
   ```

2. **Model Not Found**
   ```python
   # Solution: Download model first or specify correct path
   config = {
       "model_name": "deepke/relation-extraction-mapping",
       "local_files_only": False  # Allow download
   }
   ```

3. **Invalid Input**
   ```python
   # Solution: Validate and preprocess input
   if not text or len(text.strip()) == 0:
       raise ValueError("Input text cannot be empty")
   ```

### Validation

```python
# Validate extraction results
def validate_triples(triples):
    valid = []
    for subj, pred, obj in triples:
        if subj and pred and obj:
            valid.append((subj, pred, obj))
    return valid

result = await integration.extract_triples(text)
result.triples = validate_triples(result.triples)
```

## Troubleshooting

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = {"verbose": True}
integration = DeepKEIntegration(config=config)
```

### Model Download Issues

```python
# Set mirror for model download
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

integration = DeepKEIntegration()
```

### Performance Issues

```python
# Profile extraction time
import time

start = time.time()
result = await integration.extract_entities_relations(text)
elapsed = time.time() - start

print(f"Extraction took {elapsed:.2f}s")
print(f"Extracted {len(result.entities)} entities")
print(f"Extracted {len(result.relations)} relations")
```

## Examples

See the DeepKE examples in `examples/deepke/`:
- `basic_extraction.py` - Basic entity and relation extraction
- `document_level.py` - Document-level extraction
- `few_shot.py` - Few-shot learning
- `custom_training.py` - Training on custom data
- `pipeline_integration.py` - Integration with other systems

## References

- [DeepKE Documentation](https://github.com/zjunlp/DeepKE)
- [DeepKE Paper](https://arxiv.org/abs/2207.14289)
- [Knowledge Graph Construction Tutorial](https://github.com/zjunlp/DeepKE/blob/main/doc/usage.md)

---

**Last Updated**: 2025-02-03
**Integration Version**: 1.0.0
