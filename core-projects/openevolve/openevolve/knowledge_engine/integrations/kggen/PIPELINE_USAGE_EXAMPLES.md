# KG-Gen Pipeline Usage Examples

Task 2.6.4: Add pipeline usage examples

## Table of Contents
1. [Basic Extraction](#basic-extraction)
2. [Advanced Deduplication](#advanced-deduplication)
3. [Memory Management](#memory-management)
4. [Conversation Analysis](#conversation-analysis)
5. [Graph Aggregation](#graph-aggregation)
6. [Complete Workflows](#complete-workflows)

## Basic Extraction

### Simple Entity Extraction

```python
import asyncio
from knowledge_engine.integrations.kggen import ExtractionPipeline

async def simple_extraction():
    pipeline = ExtractionPipeline()

    text = """
    Apple Inc is a multinational technology company headquartered in Cupertino,
    California. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
    in April 1976. Google was founded by Larry Page and Sergey Brin in 1998.
    """

    result = await pipeline.extract(text=text)

    print(f"Extracted {len(result.entities)} entities:")
    for entity in result.entities[:10]:
        print(f"  - {entity}")

    print(f"\nExtracted {len(result.relationships)} relationships:")
    for rel in result.relationships[:5]:
        print(f"  - {rel['subject']} -> {rel['predicate']} -> {rel['object']}")

    await pipeline.close()

asyncio.run(simple_extraction())
```

### Extraction with Progress Tracking

```python
async def extraction_with_progress():
    pipeline = ExtractionPipeline()

    def progress_callback(status):
        print(f"Progress: {status.progress:.1%} - Stage: {status.stage.value}")

    text = "Large document text..."  # Your large text here

    result = await pipeline.extract(
        text=text,
        progress_callback=progress_callback
    )

    await pipeline.close()

asyncio.run(extraction_with_progress())
```

## Advanced Deduplication

### SEMHASH Deduplication

```python
from knowledge_engine.integrations.kggen import (
    DeduplicationEngine,
    DeduplicationMethod
)

async def semhash_dedup():
    engine = DeduplicationEngine()

    entities = [
        "Apple", "apple", "APPLE", "Apple Inc",
        "Google", "google", "Google LLC",
        "Microsoft", "Microsoft Corp"
    ]

    result = await engine.deduplicate(
        entities=entities,
        method=DeduplicationMethod.SEMHASH
    )

    print(f"Original: {result.original_count}")
    print(f"After SEMHASH: {result.final_count}")
    print(f"Reduction: {result.reduction_rate:.1%}")
    print(f"Clusters: {len(result.entity_clusters)}")

    for cluster in result.entity_clusters[:3]:
        print(f"  - {cluster.canonical_entity}: {cluster.variants}")

    await engine.close()

asyncio.run(semhash_dedup())
```

### LM Clustering Deduplication

```python
async def lm_cluster_dedup():
    engine = DeduplicationEngine()

    entities = [
        "Apple Inc", "Apple Corporation", "Apple Company",
        "Google LLC", "Google Inc", "Alphabet Inc",
        "Microsoft Corp", "Microsoft Corporation"
    ]

    result = await engine.deduplicate(
        entities=entities,
        method=DeduplicationMethod.LM_CLUSTER
    )

    print(f"After LM Clustering: {result.final_count}")

    for cluster in result.entity_clusters:
        print(f"  Cluster: {cluster.cluster_id}")
        print(f"    Canonical: {cluster.canonical_entity}")
        print(f"    Variants: {cluster.variants}")
        print(f"    Confidence: {cluster.confidence:.2f}")

    await engine.close()

asyncio.run(lm_cluster_dedup())
```

### Full Deduplication (SEMHASH + LM)

```python
async def full_dedup():
    engine = DeduplicationEngine()

    entities = [
        # Exact/near-exact duplicates
        "Apple", "apple", "APPLE", "Apple Inc",
        # Semantic variants
        "Apple Corporation", "Apple Company",
        "Google", "google", "Google LLC",
        "Microsoft", "Microsoft Corp"
    ]

    result = await engine.deduplicate(
        entities=entities,
        method=DeduplicationMethod.FULL  # Both SEMHASH and LM
    )

    print(f"Full deduplication results:")
    print(f"  Original: {result.original_count}")
    print(f"  Final: {result.final_count}")
    print(f"  Removed: {result.duplicates_removed}")
    print(f"  Reduction: {result.reduction_rate:.1%}")

    await engine.close()

asyncio.run(full_dedup())
```

## Memory Management

### Adding Memories

```python
from knowledge_engine.integrations.kggen import (
    KGGenMCPServer,
    MemoryType
)

async def add_memories():
    server = KGGenMCPServer()

    memories_data = [
        {
            "content": "Apple is a technology company",
            "memory_type": "fact",
            "importance": 0.8,
            "confidence": 0.95
        },
        {
            "content": "Google owns Android",
            "memory_type": "fact",
            "importance": 0.7,
            "confidence": 0.9
        },
        {
            "content": "Steve Jobs founded Apple",
            "memory_type": "fact",
            "importance": 0.9,
            "confidence": 1.0
        }
    ]

    result = await server.add_memories(
        memories=memories_data,
        session_id="tech_companies_session"
    )

    print(f"Added {result['count']} memories")
    for memory in result['memories']:
        print(f"  - {memory['content']} (importance: {memory['importance']})")

    await server.close()

asyncio.run(add_memories())
```

### Retrieving Relevant Memories

```python
async def retrieve_memories():
    server = KGGenMCPServer()

    # First add some memories
    await server.add_memories(
        memories=[
            {"content": "Python is a programming language", "memory_type": "fact"},
            {"content": "JavaScript is used for web development", "memory_type": "fact"},
            {"content": "Java is widely used in enterprise", "memory_type": "fact"}
        ],
        session_id="programming_session"
    )

    # Retrieve relevant memories
    result = await server.retrieve_relevant_memories(
        query_text="programming languages",
        session_id="programming_session",
        max_results=10
    )

    print(f"Retrieved {result['count']} memories:")
    for memory in result['memories']:
        print(f"  - {memory['content']}")
        print(f"    Importance: {memory['importance']}")
        print(f"    Accessed: {memory['access_count']} times")

    await server.close()

asyncio.run(retrieve_memories())
```

### Memory Aggregation

```python
async def aggregate_session_memories():
    server = KGGenMCPServer()

    # Add memories to a session
    for i in range(10):
        await server.memory_manager.add_memory(
            content=f"Memory number {i}",
            memory_type=MemoryType.FACT,
            session_id="test_session",
            importance=0.5 + (i * 0.05)
        )

    # Aggregate
    aggregation = await server.memory_manager.aggregate_session_memories(
        session_id="test_session"
    )

    print(f"Session aggregation:")
    print(f"  Total memories: {aggregation['total_memories']}")
    print(f"  By type: {aggregation['by_type']}")
    print(f"  Avg importance: {aggregation['avg_importance']:.2f}")
    print(f"  Avg confidence: {aggregation['avg_confidence']:.2f}")

    await server.close()

asyncio.run(aggregate_session_memories())
```

## Conversation Analysis

### Analyzing a Conversation

```python
from knowledge_engine.integrations.kggen import ConversationAnalyzer

async def analyze_conversation():
    analyzer = ConversationAnalyzer()

    messages = [
        {
            "role": "user",
            "content": "Can you tell me about Apple and Google?",
            "speaker_id": "user_123"
        },
        {
            "role": "assistant",
            "content": "Apple and Google are both major technology companies. "
                      "Apple was founded by Steve Jobs and is known for the iPhone. "
                      "Google was founded by Larry Page and Sergey Brin, and is known "
                      "for search and Android.",
            "speaker_id": "assistant"
        },
        {
            "role": "user",
            "content": "What about Microsoft?",
            "speaker_id": "user_123"
        },
        {
            "role": "assistant",
            "content": "Microsoft was founded by Bill Gates and Paul Allen. "
                      "They're known for Windows and Office.",
            "speaker_id": "assistant"
        }
    ]

    result = await analyzer.analyze(messages=messages)

    print(f"Conversation analysis:")
    print(f"  Speakers: {result.total_speakers}")
    print(f"  Entities extracted: {result.total_entities}")
    print(f"  Relations: {result.total_relations}")

    if result.summary:
        print(f"\nSummary:")
        print(f"  Topic: {result.summary.topic}")
        print(f"  Participants: {result.summary.participants}")
        print(f"  Key points: {result.summary.key_points}")

    print(f"\nEntities by speaker:")
    for entity in result.speaker_entities[:5]:
        print(f"  {entity.speaker_id}: {entity.entity_name} ({entity.entity_type})")

    await analyzer.close()

asyncio.run(analyze_conversation())
```

### Conversation to Knowledge Graph

```python
async def conversation_to_kg():
    analyzer = ConversationAnalyzer()

    messages = [
        {"role": "user", "content": "Apple competes with Google", "speaker_id": "user1"},
        {"role": "assistant", "content": "Yes, both are tech giants", "speaker_id": "bot"}
    ]

    result = await analyzer.analyze(messages=messages)

    print("Knowledge Graph from Conversation:")
    print(f"\nEntities ({len(result.entities)}):")
    for entity in result.entities:
        print(f"  - {entity}")

    print(f"\nRelationships ({len(result.relationships)}):")
    for rel in result.relationships:
        print(f"  - {rel['subject']} -> {rel['predicate']} -> {rel['object']}")

    await analyzer.close()

asyncio.run(conversation_to_kg())
```

## Graph Aggregation

### Aggregating Multiple Graphs

```python
from knowledge_engine.integrations.kggen import GraphAggregator

async def aggregate_graphs():
    aggregator = GraphAggregator()

    graphs = [
        {
            "entities": ["Apple", "Google", "Microsoft"],
            "relationships": [
                {"subject": "Apple", "predicate": "competes_with", "object": "Google"},
                {"subject": "Apple", "predicate": "competes_with", "object": "Microsoft"}
            ]
        },
        {
            "entities": ["Apple", "Amazon", "Facebook"],
            "relationships": [
                {"subject": "Amazon", "predicate": "competes_with", "object": "Apple"},
                {"subject": "Facebook", "predicate": "competes_with", "object": "Google"}
            ]
        },
        {
            "entities": ["Google", "Netflix"],
            "relationships": [
                {"subject": "Google", "predicate": "partners_with", "object": "Netflix"}
            ]
        }
    ]

    result = await aggregator.aggregate(graphs=graphs)

    print(f"Aggregated {len(graphs)} graphs:")
    print(f"  Total entities: {result.total_entities}")
    print(f"  Total relationships: {result.total_relationships}")
    print(f"  Conflicts resolved: {result.conflicts_resolved}")
    print(f"  Processing time: {result.processing_time_seconds:.2f}s")

    print(f"\nAggregated entities:")
    for entity in result.aggregated_graph.entities:
        print(f"  - {entity}")

    await aggregator.close()

asyncio.run(aggregate_graphs())
```

### Graph Versioning

```python
async def versioned_graphs():
    aggregator = GraphAggregator()

    # Version 1
    graph1 = {
        "entities": ["Apple", "Google"],
        "relationships": []
    }

    result1 = await aggregator.aggregate([graph1])
    version1_id = result1.aggregated_graph.version_id

    # Version 2
    graph2 = {
        "entities": ["Apple", "Google", "Microsoft"],
        "relationships": []
    }

    result2 = await aggregator.aggregate([graph2])
    version2_id = result2.aggregated_graph.version_id

    # List versions
    versions = await aggregator.list_versions(limit=10)

    print(f"Graph versions ({len(versions)}):")
    for version in versions:
        print(f"  v{version.version_number}: {version.version_id}")
        print(f"    Entities: {len(version.entities)}")
        print(f"    Created: {version.created_at}")

    await aggregator.close()

asyncio.run(versioned_graphs())
```

### Graph Comparison

```python
async def compare_graphs():
    aggregator = GraphAggregator()

    # Create two versions
    graph1 = {
        "entities": ["Apple", "Google"],
        "relationships": []
    }

    result1 = await aggregator.aggregate([graph1])

    graph2 = {
        "entities": ["Apple", "Google", "Microsoft"],
        "relationships": []
    }

    result2 = await aggregator.aggregate([graph2])

    # Compare
    diff = await aggregator.compare_versions(
        version_id1=result1.aggregated_graph.version_id,
        version_id2=result2.aggregated_graph.version_id
    )

    print(f"Graph comparison:")
    print(f"  Entities added: {diff.entities_added}")
    print(f"  Entities removed: {diff.entities_removed}")
    print(f"  Total changes: {diff.change_count}")
    print(f"  Similarity: {diff.similarity_score:.2%}")

    await aggregator.close()

asyncio.run(compare_graphs())
```

## Complete Workflows

### End-to-End: Extract → Deduplicate → Aggregate

```python
import asyncio
from knowledge_engine.integrations.kggen import (
    ExtractionPipeline,
    DeduplicationEngine,
    DeduplicationMethod,
    GraphAggregator
)

async def end_to_end_workflow():
    # Document text
    text = """
    Apple Inc is a technology company founded by Steve Jobs.
    Google was founded by Larry Page and Sergey Brin.
    Microsoft was founded by Bill Gates.
    Apple competes with both Google and Microsoft.
    Google owns Android and YouTube.
    Microsoft owns Windows and Office.
    """

    # Step 1: Extract
    print("Step 1: Extracting knowledge...")
    pipeline = ExtractionPipeline()
    extraction_result = await pipeline.extract(text=text)

    print(f"  Extracted {extraction_result.entity_count} entities")
    print(f"  Extracted {extraction_result.relationship_count} relationships")

    # Step 2: Deduplicate
    print("\nStep 2: Deduplicating...")
    dedup = DeduplicationEngine()
    dedup_result = await dedup.deduplicate(
        entities=extraction_result.entities,
        method=DeduplicationMethod.FULL
    )

    print(f"  Before: {dedup_result.original_count} entities")
    print(f"  After: {dedup_result.final_count} entities")
    print(f"  Reduction: {dedup_result.reduction_rate:.1%}")

    # Step 3: Aggregate into versioned graph
    print("\nStep 3: Aggregating...")
    aggregator = GraphAggregator()

    # Also deduplicate relationships
    unique_relationships = await dedup.deduplicate_relationships(
        extraction_result.relationships
    )

    graph = {
        "entities": dedup_result.unique_entities,
        "relationships": unique_relationships
    }

    agg_result = await aggregator.aggregate([graph])

    print(f"  Final graph: v{agg_result.aggregated_graph.version_number}")
    print(f"  Entities: {agg_result.total_entities}")
    print(f"  Relationships: {agg_result.total_relationships}")

    # Print results
    print("\n" + "="*50)
    print("FINAL KNOWLEDGE GRAPH")
    print("="*50)
    print("\nEntities:")
    for entity in agg_result.aggregated_graph.entities:
        print(f"  - {entity}")

    print("\nRelationships:")
    for rel in agg_result.aggregated_graph.relationships:
        print(f"  - {rel['subject']} -> {rel['predicate']} -> {rel['object']}")

    # Cleanup
    await pipeline.close()
    await dedup.close()
    await aggregator.close()

asyncio.run(end_to_end_workflow())
```

### Multi-Document Processing

```python
async def multi_document_processing():
    # Multiple documents
    documents = [
        "Apple is a tech company founded by Steve Jobs. It makes the iPhone.",
        "Google was founded by Larry Page. Google owns Android and YouTube.",
        "Microsoft was founded by Bill Gates. They make Windows and Office."
    ]

    # Initialize components
    pipeline = ExtractionPipeline()
    dedup = DeduplicationEngine()
    aggregator = GraphAggregator()

    # Process each document
    graphs = []

    for i, doc in enumerate(documents):
        print(f"\nProcessing document {i+1}...")

        # Extract
        result = await pipeline.extract(text=doc)

        # Deduplicate entities
        dedup_result = await dedup.deduplicate(
            entities=result.entities,
            method=DeduplicationMethod.FULL
        )

        # Create graph
        graph = {
            "entities": dedup_result.unique_entities,
            "relationships": result.relationships
        }

        graphs.append(graph)

        print(f"  Entities: {len(graph['entities'])}")
        print(f"  Relationships: {len(graph['relationships'])}")

    # Aggregate all graphs
    print("\nAggregating all documents...")
    final_result = await aggregator.aggregate(graphs=graphs)

    print(f"\nFinal aggregated graph:")
    print(f"  Total entities: {final_result.total_entities}")
    print(f"  Total relationships: {final_result.total_relationships}")

    # Print final knowledge graph
    print("\n" + "="*50)
    print("AGGREGATED KNOWLEDGE GRAPH")
    print("="*50)
    print("\nEntities:")
    for entity in sorted(final_result.aggregated_graph.entities):
        print(f"  - {entity}")

    print("\nRelationships:")
    for rel in final_result.aggregated_graph.relationships:
        print(f"  - {rel['subject']} -> {rel['predicate']} -> {rel['object']}")

    # Cleanup
    await pipeline.close()
    await dedup.close()
    await aggregator.close()

asyncio.run(multi_document_processing())
```

### Conversation + Memory + Aggregation

```python
async def conversation_memory_workflow():
    # Initialize all components
    analyzer = ConversationAnalyzer()
    server = KGGenMCPServer()
    aggregator = GraphAggregator()

    # Conversation
    messages = [
        {"role": "user", "content": "Tell me about tech companies", "speaker_id": "user1"},
        {"role": "assistant", "content": "Apple, Google, and Microsoft are major tech companies", "speaker_id": "bot"},
        {"role": "user", "content": "What about their founders?", "speaker_id": "user1"},
        {"role": "assistant", "content": "Steve Jobs founded Apple, Larry Page founded Google, Bill Gates founded Microsoft", "speaker_id": "bot"}
    ]

    # Analyze conversation
    print("Analyzing conversation...")
    conv_result = await analyzer.analyze(messages=messages)

    print(f"  Extracted {conv_result.total_entities} entities")

    # Store in memory
    print("\nStoring in memory...")
    for entity in conv_result.entities:
        await server.memory_manager.add_memory(
            content=entity,
            memory_type=MemoryType.ENTITY,
            session_id="tech_conversation",
            importance=0.7
        )

    # Create graph from conversation
    graph = {
        "entities": conv_result.entities,
        "relationships": conv_result.relationships
    }

    # Aggregate
    print("\nAggregating graph...")
    agg_result = await aggregator.aggregate([graph])

    print(f"  Graph version: {agg_result.aggregated_graph.version_number}")

    # Retrieve relevant memories
    print("\nRetrieving memories about companies...")
    memories = await server.memory_manager.retrieve_relevant_memories(
        query=MemoryQuery(
            query_text="companies",
            session_id="tech_conversation",
            max_results=5
        )
    )

    print(f"  Retrieved {len(memories)} memories")

    # Cleanup
    await analyzer.close()
    await server.close()
    await aggregator.close()

asyncio.run(conversation_memory_workflow())
```

## Tips and Best Practices

### 1. Error Handling

```python
async def safe_extraction():
    pipeline = ExtractionPipeline()

    try:
        result = await pipeline.extract(text=your_text)
        # Process result
    except TimeoutError:
        print("Extraction timed out - try reducing text size or increasing timeout")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await pipeline.close()

asyncio.run(safe_extraction())
```

### 2. Batch Processing

```python
async def batch_extract(texts):
    pipeline = ExtractionPipeline()

    # Process in batch
    results = await pipeline.extract_batch(texts=texts)

    for i, result in enumerate(results):
        print(f"Document {i+1}: {result.entity_count} entities")

    await pipeline.close()

asyncio.run(batch_extract(text_list))
```

### 3. Progress Monitoring

```python
async def monitored_extraction():
    pipeline = ExtractionPipeline()

    class ProgressTracker:
        def __call__(self, status):
            print(f"[{status.stage.value}] {status.progress:.1%} complete")

    result = await pipeline.extract(
        text=large_text,
        progress_callback=ProgressTracker()
    )

    await pipeline.close()

asyncio.run(monitored_extraction())
```

### 4. Memory Optimization

```python
# For large documents, reduce memory usage
config = PipelineConfig(
    chunk_size=3000,  # Smaller chunks
    parallel_workers=2,  # Fewer workers
    enable_metrics=False  # Disable metrics
)

pipeline = ExtractionPipeline(config)
```

### 5. Idempotency

```python
# All operations are idempotent - safe to retry
async def retry_extraction():
    pipeline = ExtractionPipeline()

    # First attempt
    result1 = await pipeline.extract(text=text)

    # Retry with same text - will give consistent results
    result2 = await pipeline.extract(text=text)

    assert result1.entities == result2.entities  # Idempotent!

    await pipeline.close()

asyncio.run(retry_extraction())
```
