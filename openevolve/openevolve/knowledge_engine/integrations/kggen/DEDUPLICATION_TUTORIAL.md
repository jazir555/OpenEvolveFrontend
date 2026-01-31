# KG-Gen Deduplication Tutorial

Task 2.6.5: Create deduplication tutorial

## Overview

This tutorial provides a comprehensive guide to using KG-Gen's advanced deduplication engine. You'll learn how to remove duplicate entities from knowledge graphs using semantic hashing and language model clustering.

## Table of Contents

1. [Introduction](#introduction)
2. [Deduplication Methods](#deduplication-methods)
3. [Basic Usage](#basic-usage)
4. [Advanced Techniques](#advanced-techniques)
5. [Cross-Document Resolution](#cross-document-resolution)
6. [Temporal Tracking](#temporal-tracking)
7. [Performance Tuning](#performance-tuning)
8. [Best Practices](#best-practices)

## Introduction

### What is Deduplication?

Deduplication identifies and merges duplicate or near-duplicate entities in knowledge graphs. For example:
- "Apple", "apple", "APPLE" → "Apple"
- "Apple Inc", "Apple Corporation" → "Apple Inc"
- "Google LLC", "Google Inc", "Alphabet Inc" → Single cluster

### Why Deduplicate?

1. **Improve Quality** - Remove redundancy and inconsistencies
2. **Reduce Noise** - Focus on unique entities
3. **Better Analytics** - Accurate entity counts
4. **Save Storage** - Store only unique entities

## Deduplication Methods

### Method 1: SEMHASH (Semantic Hashing)

**Best for:** Exact and near-exact duplicates

```python
from knowledge_engine.integrations.kggen import DeduplicationEngine, DeduplicationMethod

engine = DeduplicationEngine()

entities = ["Apple", "apple", "APPLE", "Google", "google"]

result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.SEMHASH
)

# Result: 2 unique entities instead of 5
```

**How it works:**
- Creates semantic hash for each entity
- Groups entities with similar hashes
- Fast and efficient
- Threshold-based similarity (default: 0.95)

**Use when:**
- Case variations ("Apple", "apple")
- Minor spelling differences
- Performance is critical

### Method 2: LM_CLUSTER (Language Model Clustering)

**Best for:** Semantic duplicates

```python
entities = [
    "Apple Inc",
    "Apple Corporation",
    "Apple Company",
    "Google LLC",
    "Google Inc"
]

result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.LM_CLUSTER
)

# Result: 2 clusters (Apple variants, Google variants)
```

**How it works:**
- Generates embeddings for each entity
- Uses KNN clustering to group similar entities
- More accurate but slower
- Similarity threshold (default: 0.85)

**Use when:**
- Different names for same entity
- Abbreviations vs full names
- Organizational variants

### Method 3: FULL (Combined)

**Best for:** Comprehensive deduplication

```python
result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.FULL
)

# Applies both SEMHASH and LM_CLUSTER for maximum reduction
```

**How it works:**
1. First pass: SEMHASH for exact/near-exact duplicates
2. Second pass: LM_CLUSTER for semantic duplicates
3. Combines results from both methods

**Use when:**
- Maximum deduplication needed
- Quality is more important than speed
- Mixed types of duplicates

## Basic Usage

### Simple Deduplication

```python
import asyncio
from knowledge_engine.integrations.kggen import DeduplicationEngine, DeduplicationMethod

async def basic_dedup():
    # Initialize engine
    engine = DeduplicationEngine()

    # Input entities with duplicates
    entities = [
        "Apple",
        "apple",
        "APPLE",
        "Apple Inc",
        "Google",
        "google",
        "Microsoft",
        "microsoft"
    ]

    # Deduplicate using FULL method
    result = await engine.deduplicate(
        entities=entities,
        method=DeduplicationMethod.FULL
    )

    # Print results
    print(f"Original count: {result.original_count}")
    print(f"Final count: {result.final_count}")
    print(f"Duplicates removed: {result.duplicates_removed}")
    print(f"Reduction rate: {result.reduction_rate:.1%}")

    print(f"\nUnique entities:")
    for entity in result.unique_entities:
        print(f"  - {entity}")

    # Print clusters
    print(f"\nClusters found:")
    for cluster in result.entity_clusters:
        print(f"  Cluster: {cluster.cluster_id}")
        print(f"    Canonical: {cluster.canonical_entity}")
        print(f"    Variants: {cluster.variants}")
        print(f"    Confidence: {cluster.confidence:.2f}")

    await engine.close()

asyncio.run(basic_dedup())
```

**Output:**
```
Original count: 8
Final count: 3
Duplicates removed: 5
Reduction rate: 62.5%

Unique entities:
  - Apple
  - Google
  - Microsoft

Clusters found:
  Cluster: cluster-abc123
    Canonical: Apple
    Variants: ['apple', 'APPLE', 'Apple Inc']
    Confidence: 0.95
  Cluster: cluster-def456
    Canonical: Google
    Variants: ['google']
    Confidence: 0.95
```

### Deduplicating Relationships

```python
async def dedup_relationships():
    engine = DeduplicationEngine()

    relationships = [
        {"subject": "Apple", "predicate": "owns", "object": "iOS"},
        {"subject": "Apple", "predicate": "owns", "object": "iOS"},  # Duplicate
        {"subject": "Apple", "predicate": "owns", "object": "iOS"},  # Duplicate
        {"subject": "Google", "predicate": "owns", "object": "Android"},
        {"subject": "Google", "predicate": "owns", "object": "Android"},  # Duplicate
    ]

    unique = await engine.deduplicate_relationships(
        relationships=relationships
    )

    print(f"Original: {len(relationships)}")
    print(f"Unique: {len(unique)}")

    await engine.close()

asyncio.run(dedup_relationships())
```

## Advanced Techniques

### Custom Configuration

```python
from knowledge_engine.integrations.kggen import DeduplicationConfig

# Custom configuration
config = DeduplicationConfig(
    semhash_threshold=0.90,  # Lower threshold = more aggressive
    lm_cluster_size=256,  # Larger clusters
    lm_similarity_threshold=0.80,  # Lower = more merging
    enable_temporal=True  # Enable temporal tracking
)

engine = DeduplicationEngine(config)
```

### Cross-Document Resolution

Track entities across multiple documents:

```python
async def cross_document_resolution():
    engine = DeduplicationEngine()

    # Document 1
    engine.cross_doc_resolver.register_document_entities(
        document_id="doc1",
        entities=["Apple", "Google", "Microsoft"]
    )

    # Document 2
    engine.cross_doc_resolver.register_document_entities(
        document_id="doc2",
        entities=["Apple", "Amazon", "Facebook"]
    )

    # Find entities common to both documents
    common = engine.cross_doc_resolver.find_common_entities(
        document_ids=["doc1", "doc2"]
    )

    print(f"Common entities: {common}")
    # Output: ['Apple']

    # Find all documents mentioning "Apple"
    related = engine.cross_doc_resolver.get_related_documents(
        entity="Apple"
    )

    print(f"Documents mentioning Apple: {related}")
    # Output: ['doc1', 'doc2']

    await engine.close()

asyncio.run(cross_document_resolution())
```

### Temporal Tracking

Track entity appearances over time:

```python
async def temporal_tracking():
    config = DeduplicationConfig(enable_temporal=True)
    engine = DeduplicationEngine(config)

    # Deduplicate with document tracking
    result = await engine.deduplicate(
        entities=["Apple", "Google"],
        method=DeduplicationMethod.FULL,
        document_id="doc1"
    )

    # Get entity history
    history = engine.get_entity_history("Apple")

    print(f"Entity 'Apple' history:")
    for entry in history:
        print(f"  - {entry['timestamp']}: {entry['document_id']}")

    await engine.close()

asyncio.run(temporal_tracking())
```

## Cross-Document Resolution

### Problem: Same entity in multiple documents

```
Document 1: "Apple is a tech company"
Document 2: "Apple Inc reported earnings"
Document 3: "Apple Corporation launched iPhone"
```

### Solution: Cross-document deduplication

```python
async def multi_document_dedup():
    engine = DeduplicationEngine()

    documents = {
        "doc1": ["Apple", "Google", "Microsoft"],
        "doc2": ["Apple Inc", "Google LLC", "Amazon"],
        "doc3": ["Apple Corporation", "Microsoft Corp", "Facebook"]
    }

    # Register all documents
    for doc_id, entities in documents.items():
        engine.cross_doc_resolver.register_document_entities(
            document_id=doc_id,
            entities=entities
        )

    # Find common entities across all documents
    common = engine.cross_doc_resolver.find_common_entities(
        document_ids=["doc1", "doc2", "doc3"]
    )

    print(f"Entities in all documents: {common}")
    # Output: ['Apple']

    # Find entities in doc1 and doc2
    common_12 = engine.cross_doc_resolver.find_common_entities(
        document_ids=["doc1", "doc2"]
    )

    print(f"Common to doc1 and doc2: {common_12}")
    # Output: ['Apple', 'Google']

    await engine.close()

asyncio.run(multi_document_dedup())
```

### Entity Resolution Across Documents

```python
async def entity_resolution():
    engine = DeduplicationEngine()

    # Register documents
    docs = [
        ("doc1", ["Apple", "Google", "Microsoft"]),
        ("doc2", ["Apple Inc", "Google LLC"]),
        ("doc3", ["Apple Corporation", "Amazon"])
    ]

    for doc_id, entities in docs:
        engine.cross_doc_resolver.register_document_entities(doc_id, entities)

    # Get all documents mentioning "Apple" (in any form)
    apple_docs = engine.cross_doc_resolver.get_related_documents("Apple")

    print(f"'Apple' mentioned in: {apple_docs}")
    # Output: ['doc1', 'doc2', 'doc3']

    await engine.close()

asyncio.run(entity_resolution())
```

## Temporal Tracking

### Enable Temporal Tracking

```python
from knowledge_engine.integrations.kggen import DeduplicationConfig

config = DeduplicationConfig(
    enable_temporal=True,
    temporal_window_hours=24  # Track entities for 24 hours
)

engine = DeduplicationEngine(config)
```

### Track Entity Evolution

```python
async def track_entity_evolution():
    config = DeduplicationConfig(enable_temporal=True)
    engine = DeduplicationEngine(config)

    # Simulate multiple extractions over time
    extractions = [
        ("2024-01-01 10:00", "doc1", ["Apple", "Google"]),
        ("2024-01-02 14:00", "doc2", ["Apple Inc", "Google LLC"]),
        ("2024-01-03 09:00", "doc3", ["Apple Corp", "Microsoft"]),
    ]

    for timestamp, doc_id, entities in extractions:
        await engine.deduplicate(
            entities=entities,
            method=DeduplicationMethod.FULL,
            document_id=doc_id
        )

    # Get history for "Apple"
    history = engine.get_entity_history("Apple")

    print("Entity 'Apple' timeline:")
    for entry in sorted(history, key=lambda x: x['timestamp']):
        print(f"  {entry['timestamp']}: {entry['document_id']}")

    await engine.close()

asyncio.run(track_entity_evolution())
```

## Performance Tuning

### For Large Datasets

```python
# Reduce memory usage
config = DeduplicationConfig(
    parallel_workers=2,  # Fewer workers
    batch_size=50  # Smaller batches
)
```

### For Speed

```python
# Use SEMHASH only (faster but less accurate)
result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.SEMHASH
)
```

### For Quality

```python
# Use FULL method (slower but more accurate)
result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.FULL
)
```

### Adjust Thresholds

```python
config = DeduplicationConfig(
    # Lower SEMHASH threshold = more aggressive merging
    semhash_threshold=0.85,  # Default: 0.95

    # Lower LM threshold = more clusters
    lm_similarity_threshold=0.75,  # Default: 0.85
)
```

## Best Practices

### 1. Always Deduplicate After Extraction

```python
# BAD: Use raw extraction results
result = await pipeline.extract(text=text)
entities = result.entities  # May contain duplicates

# GOOD: Deduplicate after extraction
result = await pipeline.extract(text=text)
dedup_result = await dedup.deduplicate(entities=result.entities)
entities = dedup_result.unique_entities  # Clean, unique
```

### 2. Choose Method Based on Use Case

```python
# Case variations only → SEMHASH
entities = ["Apple", "apple", "APPLE"]
method = DeduplicationMethod.SEMHASH

# Semantic variants → LM_CLUSTER
entities = ["Apple Inc", "Apple Corporation"]
method = DeduplicationMethod.LM_CLUSTER

# Mixed → FULL
entities = ["Apple", "apple", "Apple Inc", "Apple Corporation"]
method = DeduplicationMethod.FULL
```

### 3. Inspect Clusters

```python
result = await engine.deduplicate(entities, method=DeduplicationMethod.FULL)

# Always check clusters
for cluster in result.entity_clusters:
    print(f"Cluster: {cluster.canonical_entity}")
    print(f"  Variants: {cluster.variants}")

    # Verify merging is correct
    if len(cluster.variants) > 5:
        print(f"  WARNING: Large cluster - may be over-merging")
```

### 4. Use Correlation IDs

```python
# Always provide correlation ID for tracking
result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.FULL,
    correlation_id="my-batch-001"  # For logging/debugging
)
```

### 5. Handle Edge Cases

```python
# Very similar but distinct entities
entities = ["Apple", "Apple Records"]

# SEMHASH might merge them incorrectly
# Solution: Use higher threshold or LM clustering
config = DeduplicationConfig(
    semhash_threshold=0.98  # Higher threshold
)
```

### 6. Batch Processing

```python
# Process in batches for large datasets
async def batch_dedup(all_entities, batch_size=1000):
    engine = DeduplicationEngine()

    results = []
    for i in range(0, len(all_entities), batch_size):
        batch = all_entities[i:i+batch_size]
        result = await engine.deduplicate(
            entities=batch,
            method=DeduplicationMethod.FULL
        )
        results.append(result)

    await engine.close()
    return results
```

### 7. Validate Results

```python
result = await engine.deduplicate(entities, method=DeduplicationMethod.FULL)

# Check reduction rate
if result.reduction_rate > 0.8:
    print("WARNING: Very high reduction rate - check for over-merging")

# Check cluster sizes
for cluster in result.entity_clusters:
    if len(cluster.variants) > 10:
        print(f"WARNING: Large cluster: {cluster.canonical_entity}")
```

## Complete Example

```python
import asyncio
from knowledge_engine.integrations.kggen import (
    ExtractionPipeline,
    DeduplicationEngine,
    DeduplicationMethod,
    DeduplicationConfig
)

async def complete_deduplication_workflow():
    # Configuration
    config = DeduplicationConfig(
        semhash_threshold=0.95,
        lm_cluster_size=128,
        enable_temporal=True
    )

    # Initialize components
    pipeline = ExtractionPipeline()
    dedup = DeduplicationEngine(config)

    # Extract from document
    text = """
    Apple is a technology company. Apple Inc was founded by Steve Jobs.
    apple makes the iPhone. APPLE is headquartered in Cupertino.
    Google was founded by Larry Page. Google LLC owns YouTube.
    google is a search company. Microsoft was founded by Bill Gates.
    microsoft corporation makes Windows.
    """

    print("Extracting knowledge...")
    extraction_result = await pipeline.extract(text=text)

    print(f"Extracted {extraction_result.entity_count} entities:")
    for entity in extraction_result.entities:
        print(f"  - {entity}")

    # Deduplicate
    print("\nDeduplicating...")
    dedup_result = await dedup.deduplicate(
        entities=extraction_result.entities,
        method=DeduplicationMethod.FULL
    )

    print(f"Results:")
    print(f"  Original: {dedup_result.original_count}")
    print(f"  Final: {dedup_result.final_count}")
    print(f"  Removed: {dedup_result.duplicates_removed}")
    print(f"  Reduction: {dedup_result.reduction_rate:.1%}")

    print(f"\nUnique entities:")
    for entity in dedup_result.unique_entities:
        print(f"  - {entity}")

    print(f"\nClusters:")
    for cluster in dedup_result.entity_clusters:
        print(f"  {cluster.canonical_entity}:")
        for variant in cluster.variants:
            print(f"    - {variant}")

    # Deduplicate relationships
    print("\nDeduplicating relationships...")
    unique_relationships = await dedup.deduplicate_relationships(
        extraction_result.relationships
    )

    print(f"Relationships: {len(extraction_result.relationships)} → {len(unique_relationships)}")

    # Cleanup
    await pipeline.close()
    await dedup.close()

asyncio.run(complete_deduplication_workflow())
```

## Troubleshooting

### Problem: Over-merging

```
Entities: ["Apple", "Apple Records"]
Result: Merged into single cluster ❌
```

**Solution:** Increase thresholds
```python
config = DeduplicationConfig(
    semhash_threshold=0.98,
    lm_similarity_threshold=0.90
)
```

### Problem: Under-merging

```
Entities: ["Apple", "apple", "APPLE"]
Result: All kept as separate ❌
```

**Solution:** Decrease thresholds
```python
config = DeduplicationConfig(
    semhash_threshold=0.90
)
```

### Problem: Poor Performance

**Solution:** Use SEMHASH only
```python
result = await engine.deduplicate(
    entities=entities,
    method=DeduplicationMethod.SEMHASH  # Faster
)
```

### Problem: Memory Issues

**Solution:** Reduce batch size
```python
config = DeduplicationConfig(
    batch_size=50  # Smaller batches
)
```

## Summary

- **SEMHASH**: Fast, best for exact duplicates
- **LM_CLUSTER**: Slower, best for semantic duplicates
- **FULL**: Most thorough, combines both methods
- Always inspect clusters to validate results
- Adjust thresholds based on your use case
- Use correlation IDs for tracking
- Enable temporal tracking for multi-document scenarios

For more examples, see `PIPELINE_USAGE_EXAMPLES.md`.
