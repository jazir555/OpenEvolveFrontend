# Phase 4: Enhanced Knowledge Base

## Overview

Phase 4 implements an advanced RAG-powered knowledge base with automatic knowledge extraction, vector indexing optimization, and knowledge enrichment capabilities.

## Components

### 1. Knowledge Extractor

#### Overview (`extraction/knowledge_extractor.py`)

Automatically extracts structured knowledge from workflow artifacts using pattern matching and LLM-based extraction.

**Key Features**:
- 10 knowledge entity types (Solution Patterns, Best Practices, Lessons Learned, Anti-Patterns, etc.)
- Pattern-based extraction with configurable confidence thresholds
- LLM-based extraction for complex patterns
- Automatic entity deduplication
- Tag extraction and entity linking
- Multi-artifact batch processing

**Knowledge Entity Types**:
- `SOLUTION_PATTERN` - Architectural and implementation patterns
- `BEST_PRACTICE` - Recommended approaches and conventions
- `LESSON_LEARNED` - Insights from experience
- `ANTI_PATTERN` - Practices to avoid
- `TECHNIQUE` - Specific techniques and methods
- `PRINCIPLE` - Design principles and guidelines
- `REQUIREMENT` - Requirements and specifications
- `CONSTRAINT` - Constraints and limitations
- `ASSUMPTION` - Assumptions and dependencies
- `DEPENDENCY` - External dependencies

**Usage**:
```python
from ragbits_integration.knowledge_base import KnowledgeExtractor

extractor = KnowledgeExtractor(hephaestus_client)

# Extract from single artifact
result = await extractor.extract_from_artifact(
    artifact_id="solution_123",
    content="""
    Implement JWT authentication for the API.
    Best practice: Use httpOnly cookies for token storage.
    Pattern: Middleware-based authentication flow.
    Lesson learned: Always validate tokens at each request.
    """,
    artifact_type="solution",
    use_llm=True,
    min_confidence=0.3
)

# Access extracted entities
for entity in result.entities:
    print(f"{entity.entity_type.value}: {entity.content}")
    print(f"  Confidence: {entity.confidence}")
    print(f"  Tags: {entity.tags}")

# Get extraction summary
print(f"Extracted {len(result.entities)} entities")
print(f"Summary: {result.extraction_summary}")

# Extract from multiple artifacts
artifacts = [
    {"artifact_id": "art_1", "content": "...", "artifact_type": "solution"},
    {"artifact_id": "art_2", "content": "...", "artifact_type": "solution"}
]
results = await extractor.extract_from_multiple_artifacts(artifacts)
```

### 2. Knowledge Enricher

#### Overview (`enrichment/knowledge_enricher.py`)

Enriches extracted knowledge with additional context, relationships, and quality scores.

**Key Features**:
- Contextual information retrieval
- Related pattern discovery
- Success rate estimation
- Quality score calculation
- Usage tracking
- Metadata enrichment

**Usage**:
```python
from ragbits_integration.knowledge_base import KnowledgeEnricher

enricher = KnowledgeEnricher(
    storage_manager=storage,
    hephaestus_client=hephaestus
)

# Enrich entities
result = await enricher.enrich_entities(
    entities=extracted_entities,
    artifact_type="solution",
    add_context=True,
    find_relationships=True
)

# Access enriched entities
for enriched in result.enriched_entities:
    print(f"Original: {enriched.original_entity.content}")
    print(f"Quality Score: {enriched.quality_score}")
    print(f"Additional Context: {len(enriched.additional_context)} items")
    print(f"Related Patterns: {len(enriched.related_patterns)}")

# Get enrichment summary
print(f"Enriched {result.enrichment_summary['total_entities']} entities")
print(f"Added context: {result.enrichment_summary['context_added']} items")
```

### 3. Vector Index Optimizer

#### Overview (`indexing/vector_optimizer.py`)

Optimizes vector indexing strategies for improved search performance and relevance.

**Indexing Strategies**:
- `BASIC` - Simple flat indexing for small datasets
- `HNSW` - Hierarchical Navigable Small World for read-heavy workloads
- `IVF` - Inverted File Index for write-heavy workloads
- `HNSW_IVF` - Hybrid strategy for large-scale read-heavy datasets
- `ADAPTIVE` - Adaptive strategy that adjusts based on patterns

**Key Features**:
- Automatic strategy recommendation
- Configuration optimization
- Performance estimation
- Memory usage calculation
- Build time estimation
- Optimization history tracking

**Usage**:
```python
from ragbits_integration.knowledge_base import VectorIndexOptimizer, IndexingStrategy

optimizer = VectorIndexOptimizer()

# Analyze and recommend strategy
report = await optimizer.analyze_and_recommend(
    document_count=50000,
    dimension=1536,
    query_pattern="read_heavy",
    current_strategy=IndexingStrategy.BASIC
)

print(f"Recommended Strategy: {report.optimized_strategy.value}")
print(f"Expected Improvement: {report.performance_improvement:.1f}%")
print(f"Index Size Reduction: {report.index_size_reduction:.1f}%")
print(f"Search Accuracy: {report.search_accuracy}")

# View recommendations
for rec in report.recommendations:
    print(f"- {rec}")

# Check estimated resources
print(f"Build Time: {report.metrics['estimated_build_time']:.1f}s")
print(f"Memory Usage: {report.metrics['estimated_memory_usage']:.1f} MB")

# Optimize existing configuration
from ragbits_integration.knowledge_base import IndexConfiguration

current_config = IndexConfiguration(
    strategy=IndexingStrategy.HNSW,
    dimension=1536,
    ef_construction=100,
    M=8
)

optimized_config, changes = await optimizer.optimize_index(current_config)

for change in changes:
    print(f"Change: {change}")
```

### 4. Advanced RAG Engine

#### Overview (`rag_engine/advanced_rag.py`)

Advanced Retrieval-Augmented Generation engine with hybrid search, reranking, and query expansion.

**Search Types**:
- `SEMANTIC` - Pure vector similarity search
- `KEYWORD` - Keyword-based search
- `HYBRID` - Combined semantic + keyword (default)
- `RERANKED` - Search with LLM reranking
- `EXPANDED` - Query expansion for better coverage

**Key Features**:
- Hybrid search combining semantic and keyword matching
- LLM-based query expansion
- Result reranking with multiple strategies
- Metadata filtering
- Query expansion with LLM
- Result deduplication
- Performance tracking

**Usage**:
```python
from ragbits_integration.knowledge_base import AdvancedRAGEngine, SearchType

engine = AdvancedRAGEngine(
    document_search=document_search,
    hephaestus_client=hephaestus
)

# Simple hybrid search
result = await engine.query(
    query_text="How to implement JWT authentication?",
    search_type=SearchType.HYBRID,
    top_k=5
)

# Access results
for doc in result.ranked_documents:
    print(f"Score: {doc.get('score')}")
    print(f"Content: {doc.get('content')[:200]}")

# Advanced search with reranking and expansion
result = await engine.query(
    query_text="Authentication best practices",
    search_type=SearchType.RERANKED,
    filters={"artifact_type": "solution"},
    top_k=10,
    expand_query=True,
    rerank=True
)

# Check query expansion
if result.query_expansion:
    print("Expanded queries:")
    for query in result.query_expansion:
        print(f"  - {query}")

# Search metadata
print(f"Retrieved: {result.search_metadata['total_retrieved']} documents")
print(f"Time: {result.retrieval_time_ms:.0f}ms")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Extraction                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Pattern      │  │ LLM-Based    │  │ Entity       │    │
│  │ Extraction   │  │ Extraction   │  │ Linking      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Enrichment                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Context      │  │ Related      │  │ Quality      │    │
│  │ Retrieval    │  │ Patterns     │  │ Scoring      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                  Vector Index Optimization                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Strategy     │  │ Configuration │  │ Performance  │    │
│  │ Analysis     │  │ Tuning       │  │ Estimation   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Advanced RAG Engine                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Hybrid       │  │ Query        │  │ LLM          │    │
│  │ Search       │  │ Expansion    │  │ Reranking    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Complete Workflow Example

```python
import asyncio
from ragbits_integration.knowledge_base import (
    KnowledgeExtractor,
    KnowledgeEnricher,
    VectorIndexOptimizer,
    AdvancedRAGEngine
)

async def complete_knowledge_workflow():
    # Step 1: Extract knowledge from artifacts
    extractor = KnowledgeExtractor(hephaestus_client)

    content = """
    Implement JWT-based authentication for the REST API.
    Best practice: Use httpOnly cookies for secure token storage.
    Pattern: Middleware-based authentication flow.
    Lesson learned: Always validate tokens at each request.
    Avoid: Storing tokens in localStorage (security risk).
    """

    extraction_result = await extractor.extract_from_artifact(
        artifact_id="auth_solution",
        content=content,
        artifact_type="solution",
        use_llm=True
    )

    print(f"Extracted {len(extraction_result.entities)} knowledge entities")

    # Step 2: Enrich extracted knowledge
    enricher = KnowledgeEnricher(storage_manager, hephaestus_client)

    enrichment_result = await enricher.enrich_entities(
        entities=extraction_result.entities,
        artifact_type="solution"
    )

    print(f"Enriched {len(enrichment_result.enriched_entities)} entities")

    # Step 3: Optimize vector indexing
    optimizer = VectorIndexOptimizer()

    optimization_report = await optimizer.analyze_and_recommend(
        document_count=10000,
        dimension=1536,
        query_pattern="read_heavy"
    )

    print(f"Recommended strategy: {optimization_report.optimized_strategy.value}")

    # Step 4: Query knowledge base with advanced RAG
    rag_engine = AdvancedRAGEngine(document_search, hephaestus_client)

    query_result = await rag_engine.query(
        query_text="Best practices for JWT authentication",
        search_type=SearchType.RERANKED,
        top_k=5,
        expand_query=True
    )

    print(f"Retrieved {len(query_result.ranked_documents)} documents")

    for doc in query_result.ranked_documents[:3]:
        print(f"\n- {doc.get('content', '')[:150]}...")

# Run workflow
asyncio.run(complete_knowledge_workflow())
```

## Knowledge Entity Types

### Solution Patterns
- Architectural patterns (Microservices, Event-Driven, etc.)
- Implementation patterns (Repository, Factory, etc.)
- Design patterns (Singleton, Observer, etc.)

### Best Practices
- Security practices (Input validation, encryption, etc.)
- Performance practices (Caching, optimization, etc.)
- Code quality practices (Testing, documentation, etc.)

### Lessons Learned
- Insights from previous implementations
- Common pitfalls and how to avoid them
- Experience-based recommendations

### Anti-Patterns
- Common mistakes to avoid
- Practices that lead to problems
- Negative examples

### Techniques
- Specific implementation techniques
- Algorithms and data structures
- Development methodologies

### Principles
- Design principles (SOLID, DRY, etc.)
- Architectural principles
- Best practice guidelines

### Requirements
- Functional requirements
- Non-functional requirements
- Business requirements

### Constraints
- Technical constraints
- Business constraints
- Resource limitations

### Assumptions
- Architectural assumptions
- Dependency assumptions
- Environmental assumptions

### Dependencies
- Library dependencies
- Service dependencies
- Platform dependencies

## Testing

Run Phase 4 tests:

```bash
# Run all Phase 4 tests
python -m pytest ragbits_integration/knowledge_base/tests/test_phase4_knowledge_base.py

# Run manually
python ragbits_integration/knowledge_base/tests/test_phase4_knowledge_base.py
```

## Files Structure

```
ragbits_integration/knowledge_base/
├── __init__.py
├── README.md                          # This file
├── extraction/
│   ├── __init__.py
│   └── knowledge_extractor.py         # Knowledge extraction
├── enrichment/
│   ├── __init__.py
│   └── knowledge_enricher.py          # Knowledge enrichment
├── indexing/
│   ├── __init__.py
│   └── vector_optimizer.py            # Vector optimization
├── rag_engine/
│   ├── __init__.py
│   └── advanced_rag.py                # Advanced RAG engine
└── tests/
    ├── __init__.py
    └── test_phase4_knowledge_base.py  # Comprehensive tests
```

## Integration with Decomposition Workflow

Phase 4 components integrate at multiple points:

**Stage 0: Content Analysis**
- Extract requirements and constraints from problem description

**Stage 1: Decomposition**
- Extract decomposition patterns and techniques

**Stage 3: Solution Generation**
- Extract solution patterns and best practices
- Enrich with contextual information
- Query for similar solutions using advanced RAG

**Stage 5: Knowledge Extraction**
- Extract lessons learned from completed solutions
- Identify anti-patterns to avoid
- Build knowledge base for future reference

## Next Steps

Phase 5: UI/CLI Integration (remaining in integration plan)
- Review interface enhancements
- CLI tools for knowledge base operations
- Interactive knowledge exploration dashboards

## Status

✅ **COMPLETE** - All Phase 4 components implemented and tested

- Automatic knowledge extraction with 10 entity types
- Knowledge enrichment with context and quality scoring
- Vector indexing optimization with 5 strategies
- Advanced RAG engine with hybrid search and reranking
- Complete test coverage
