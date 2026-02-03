"""
Example usage of the Unified Deduplication System

Demonstrates various use cases and strategies.
"""

import asyncio
import time
from datetime import datetime
from knowledge_engine.core.deduplication import (
    UnifiedDeduplicationManager,
    Entity
)


async def example_basic_usage():
    """Basic deduplication example."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)

    # Create manager
    manager = UnifiedDeduplicationManager()

    # Create sample entities with duplicates
    entities = [
        Entity(
            id="e1",
            name="Machine Learning",
            entity_type="concept",
            description="AI and ML technologies"
        ),
        Entity(
            id="e2",
            name="machine learning",  # Duplicate (case insensitive)
            entity_type="concept",
            description="Artificial Intelligence and ML"
        ),
        Entity(
            id="e3",
            name="Deep Learning",
            entity_type="concept",
            description="Neural networks"
        ),
        Entity(
            id="e4",
            name="TensorFlow",
            entity_type="tool",
            description="ML framework"
        ),
        Entity(
            id="e5",
            name="TensorFlow",  # Exact duplicate
            entity_type="tool",
            description="Machine Learning Framework"
        ),
    ]

    print(f"\nOriginal entities: {len(entities)}")
    for e in entities:
        print(f"  - {e.id}: {e.name}")

    # Deduplicate with auto strategy
    result = await manager.deduplicate(entities, strategy='auto')

    print(f"\nAfter deduplication: {len(result.canonical_entities)}")
    print(f"Strategy used: {result.strategy_used}")
    print(f"Processing time: {result.processing_time_ms:.2f}ms")
    print(f"Duplicate groups found: {len(result.duplicate_groups)}")

    print("\nCanonical entities:")
    for e in result.canonical_entities:
        print(f"  - {e.id}: {e.name}")

    if result.duplicate_groups:
        print("\nDuplicate groups:")
        for i, group in enumerate(result.duplicate_groups, 1):
            print(f"  Group {i}:")
            for e in group:
                print(f"    - {e.id}: {e.name}")


async def example_strategy_comparison():
    """Compare different strategies."""
    print("\n" + "=" * 80)
    print("Example 2: Strategy Comparison")
    print("=" * 80)

    # Create larger dataset
    entities = []
    concepts = [
        "Machine Learning", "Deep Learning", "Neural Networks",
        "Natural Language Processing", "Computer Vision"
    ]

    for i, concept in enumerate(concepts):
        # Add variations
        entities.append(Entity(
            id=f"e{i*3}",
            name=concept,
            entity_type="concept"
        ))
        entities.append(Entity(
            id=f"e{i*3+1}",
            name=concept.lower(),  # Case variation
            entity_type="concept"
        ))
        entities.append(Entity(
            id=f"e{i*3+2}",
            name=f"{concept} Algorithms",  # Subset
            entity_type="concept"
        ))

    print(f"\nTotal entities: {len(entities)}")

    # Test each strategy
    manager = UnifiedDeduplicationManager()
    results = {}

    for strategy_name in manager.strategies.keys():
        start = time.time()
        result = await manager.deduplicate(entities, strategy=strategy_name)
        elapsed = time.time() - start

        results[strategy_name] = {
            'canonical': len(result.canonical_entities),
            'groups': len(result.duplicate_groups),
            'time': elapsed * 1000
        }

        print(f"\n{strategy_name.upper()}:")
        print(f"  Canonical entities: {len(result.canonical_entities)}")
        print(f"  Duplicate groups: {len(result.duplicate_groups)}")
        print(f"  Processing time: {elapsed * 1000:.2f}ms")


async def example_cache_usage():
    """Demonstrate caching functionality."""
    print("\n" + "=" * 80)
    print("Example 3: Cache Usage")
    print("=" * 80)

    manager = UnifiedDeduplicationManager()

    entities = [
        Entity(id=f"e{i}", name=f"Concept {i % 20}", entity_type="test")
        for i in range(100)
    ]

    print(f"\nProcessing {len(entities)} entities...")

    # First run (no cache)
    start = time.time()
    result1 = await manager.deduplicate(entities, use_cache=True)
    time1 = (time.time() - start) * 1000

    # Second run (with cache)
    start = time.time()
    result2 = await manager.deduplicate(entities, use_cache=True)
    time2 = (time.time() - start) * 1000

    print(f"\nFirst run (no cache): {time1:.2f}ms")
    print(f"Second run (cached): {time2:.2f}ms")
    print(f"Speedup: {time1 / time2:.2f}x")

    # Show cache stats
    stats = manager.get_stats()
    print(f"\nCache size: {stats['cache_size']} entries")


async def example_entity_merging():
    """Demonstrate entity merging."""
    print("\n" + "=" * 80)
    print("Example 4: Entity Merging")
    print("=" * 80)

    manager = UnifiedDeduplicationManager()

    # Create duplicate entities with different properties
    entities = [
        Entity(
            id="e1",
            name="TensorFlow",
            entity_type="tool",
            description="ML framework",
            properties={"version": "2.0", "language": "Python"},
            source="documentation"
        ),
        Entity(
            id="e2",
            name="tensorflow",
            entity_type="tool",
            description="Machine Learning framework",
            properties={"version": "2.1", "category": "deep learning"},
            source="wiki"
        ),
    ]

    print("\nOriginal entities:")
    for e in entities:
        print(f"  {e.name}:")
        print(f"    Properties: {e.properties}")
        print(f"    Source: {e.source}")

    # Merge
    merged = await manager.merge_entities(entities)

    print("\nMerged entity:")
    print(f"  Name: {merged.name}")
    print(f"  Properties: {merged.properties}")
    print(f"  Source: {merged.source}")


async def example_canonical_mapping():
    """Track canonical-to-variant mappings."""
    print("\n" + "=" * 80)
    print("Example 5: Canonical Mapping")
    print("=" * 80)

    manager = UnifiedDeduplicationManager()

    entities = [
        Entity(id="e1", name="ML", entity_type="concept"),
        Entity(id="e2", name="Machine Learning", entity_type="concept"),
        Entity(id="e3", name="ml", entity_type="concept"),
        Entity(id="e4", name="Deep Learning", entity_type="concept"),
        Entity(id="e5", name="deep learning", entity_type="concept"),
    ]

    # Deduplicate
    result = await manager.deduplicate(entities)

    # Get mappings
    mappings = manager.get_canonical_mapping()

    print("\nCanonical-to-variant mappings:")
    for canonical_id, variant_ids in mappings.items():
        print(f"\nCanonical: {canonical_id}")
        print(f"  Variants: {variant_ids}")

    print(f"\nTotal mappings: {len(mappings)}")


async def example_performance_benchmark():
    """Benchmark performance with different dataset sizes."""
    print("\n" + "=" * 80)
    print("Example 6: Performance Benchmark")
    print("=" * 80)

    manager = UnifiedDeduplicationManager()

    sizes = [50, 100, 500, 1000]

    print("\nDataset size | Strategy | Time (ms) | Reduction")
    print("-" * 70)

    for size in sizes:
        # Create test data
        entities = [
            Entity(
                id=f"e{i}",
                name=f"Concept {i % (size // 10)}",  # Create duplicates
                entity_type="test"
            )
            for i in range(size)
        ]

        # Test with auto strategy
        start = time.time()
        result = await manager.deduplicate(entities, strategy='auto')
        elapsed = (time.time() - start) * 1000

        reduction = (1 - len(result.canonical_entities) / len(entities)) * 100

        print(f"{size:12} | {result.strategy_used:8} | {elapsed:8.2f} | {reduction:6.1f}%")


async def example_real_world_usage():
    """Real-world example: Deduplicating knowledge base articles."""
    print("\n" + "=" * 80)
    print("Example 7: Real-World Usage - Knowledge Base Deduplication")
    print("=" * 80)

    manager = UnifiedDeduplicationManager()

    # Simulate knowledge base articles
    articles = [
        Entity(
            id="article1",
            name="Getting Started with TensorFlow",
            entity_type="tutorial",
            description="Introduction to TensorFlow basics",
            properties={"author": "John Doe", "date": "2024-01-01"}
        ),
        Entity(
            id="article2",
            name="getting started with tensorflow",  # Duplicate
            entity_type="tutorial",
            description="Learn TensorFlow fundamentals",
            properties={"author": "Jane Smith", "date": "2024-01-15"}
        ),
        Entity(
            id="article3",
            name="TensorFlow Tutorial for Beginners",
            entity_type="tutorial",
            description="Complete guide to TensorFlow",
            properties={"author": "Bob Johnson", "date": "2024-02-01"}
        ),
        Entity(
            id="article4",
            name="Advanced PyTorch Techniques",
            entity_type="tutorial",
            description="Deep dive into PyTorch",
            properties={"author": "Alice Williams", "date": "2024-01-20"}
        ),
        Entity(
            id="article5",
            name="PyTorch Advanced Guide",  # Duplicate
            entity_type="tutorial",
            description="Advanced PyTorch methods",
            properties={"author": "Charlie Brown", "date": "2024-02-10"}
        ),
    ]

    print(f"\nOriginal articles: {len(articles)}")

    # Deduplicate
    result = await manager.deduplicate(articles, strategy='semantic')

    print(f"After deduplication: {len(result.canonical_entities)}")
    print(f"Removed {len(articles) - len(result.canonical_entities)} duplicates")

    print("\nDuplicate groups found:")
    for i, group in enumerate(result.duplicate_groups, 1):
        print(f"\nGroup {i}:")
        for article in group:
            print(f"  - {article.name} (by {article.properties.get('author', 'Unknown')})")


async def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("UNIFIED DEDUPLICATION SYSTEM - EXAMPLES")
    print("=" * 80)

    await example_basic_usage()
    await example_strategy_comparison()
    await example_cache_usage()
    await example_entity_merging()
    await example_canonical_mapping()
    await example_performance_benchmark()
    await example_real_world_usage()

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
