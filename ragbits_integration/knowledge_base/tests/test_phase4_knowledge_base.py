"""
Phase 4 Comprehensive Tests

Tests for enhanced knowledge base components including:
- Knowledge extraction
- Knowledge enrichment
- Vector indexing optimization
- Advanced RAG engine
"""

import pytest
import asyncio

from ragbits_integration.knowledge_base.extraction.knowledge_extractor import (
    KnowledgeExtractor,
    KnowledgeEntity,
    KnowledgeEntityType
)
from ragbits_integration.knowledge_base.enrichment.knowledge_enricher import (
    KnowledgeEnricher
)
from ragbits_integration.knowledge_base.indexing.vector_optimizer import (
    VectorIndexOptimizer,
    IndexingStrategy,
    IndexConfiguration
)
from ragbits_integration.knowledge_base.rag_engine.advanced_rag import (
    AdvancedRAGEngine,
    RAGQuery,
    SearchType
)


# Knowledge Extractor Tests

@pytest.mark.asyncio
async def test_knowledge_extractor_initialization():
    """Test knowledge extractor initialization"""
    extractor = KnowledgeExtractor()

    assert extractor is not None
    assert extractor.extraction_stats["artifacts_processed"] == 0


@pytest.mark.asyncio
async def test_extract_solution_patterns():
    """Test extracting solution patterns"""
    extractor = KnowledgeExtractor()

    content = """
    Implement JWT authentication for the API.
    Pattern: Use bearer token authentication with middleware.
    Best practice: Store tokens securely with httpOnly cookies.
    """

    result = await extractor.extract_from_artifact(
        artifact_id="test_artifact",
        content=content,
        artifact_type="solution"
    )

    assert result is not None
    assert result.artifact_id == "test_artifact"
    assert len(result.entities) > 0
    assert result.processing_time_ms > 0


@pytest.mark.asyncio
async def test_extract_best_practices():
    """Test extracting best practices"""
    extractor = KnowledgeExtractor()

    content = """
    Best practice: Use connection pooling for database access.
    Recommended: Implement retry logic with exponential backoff.
    """

    result = await extractor.extract_from_artifact(
        artifact_id="test_artifact",
        content=content,
        artifact_type="solution"
    )

    assert result is not None

    # Check for best practice entities
    bp_entities = [
        e for e in result.entities
        if e.entity_type == KnowledgeEntityType.BEST_PRACTICE
    ]

    assert len(bp_entities) >= 1


@pytest.mark.asyncio
async def test_extract_lessons_learned():
    """Test extracting lessons learned"""
    extractor = KnowledgeExtractor()

    content = """
    Lesson learned: Always validate input at the API boundary.
    Found that rate limiting prevents abuse.
    """

    result = await extractor.extract_from_artifact(
        artifact_id="test_artifact",
        content=content,
        artifact_type="solution"
    )

    assert result is not None

    # Check for lesson learned entities
    lesson_entities = [
        e for e in result.entities
        if e.entity_type == KnowledgeEntityType.LESSON_LEARNED
    ]

    assert len(lesson_entities) >= 1


@pytest.mark.asyncio
async def test_extract_anti_patterns():
    """Test extracting anti-patterns"""
    extractor = KnowledgeExtractor()

    content = """
    Avoid: Hardcoding configuration values.
    Don't use synchronous I/O in async contexts.
    """

    result = await extractor.extract_from_artifact(
        artifact_id="test_artifact",
        content=content,
        artifact_type="solution"
    )

    assert result is not None

    # Check for anti-pattern entities
    anti_pattern_entities = [
        e for e in result.entities
        if e.entity_type == KnowledgeEntityType.ANTI_PATTERN
    ]

    assert len(anti_pattern_entities) >= 1


@pytest.mark.asyncio
async def test_entity_deduplication():
    """Test entity deduplication"""
    extractor = KnowledgeExtractor()

    content = """
    Best practice: Use connection pooling.
    Best practice: Use connection pooling.
    Pattern: Connection pooling for efficiency.
    """

    result = await extractor.extract_from_artifact(
        artifact_id="test_artifact",
        content=content,
        artifact_type="solution"
    )

    # Should deduplicate similar entities
    assert len(result.entities) <= 3


@pytest.mark.asyncio
async def test_entity_tagging():
    """Test automatic entity tagging"""
    extractor = KnowledgeExtractor()

    content = """
    Best practice: Implement REST API with JWT authentication for security.
    Recommended: Use Docker for containerization.
    Pattern: Deploy to Kubernetes cluster for scalability.
    """

    result = await extractor.extract_from_artifact(
        artifact_id="test_artifact",
        content=content,
        artifact_type="solution"
    )

    # Check that entities have tags
    entities_with_tags = [
        e for e in result.entities
        if len(e.tags) > 0
    ]

    assert len(entities_with_tags) > 0


@pytest.mark.asyncio
async def test_extraction_statistics():
    """Test extraction statistics tracking"""
    extractor = KnowledgeExtractor()

    # Extract from multiple artifacts
    for i in range(3):
        await extractor.extract_from_artifact(
            artifact_id=f"artifact_{i}",
            content="Best practice: Write clean code.",
            artifact_type="solution"
        )

    stats = extractor.get_statistics()

    assert stats["artifacts_processed"] == 3
    assert stats["entities_extracted"] > 0


# Knowledge Enricher Tests

@pytest.mark.asyncio
async def test_knowledge_enricher_initialization():
    """Test knowledge enricher initialization"""
    enricher = KnowledgeEnricher()

    assert enricher is not None
    assert enricher.enrichment_stats["entities_enriched"] == 0


@pytest.mark.asyncio
async def test_enrich_entities():
    """Test entity enrichment"""
    enricher = KnowledgeEnricher()

    # Create test entities
    entities = [
        KnowledgeEntity(
            entity_type=KnowledgeEntityType.SOLUTION_PATTERN,
            content="Use JWT for authentication",
            confidence=0.8,
            source_artifact_id="test_artifact",
            tags={"security", "api"}
        )
    ]

    result = await enricher.enrich_entities(
        entities=entities,
        artifact_type="solution"
    )

    assert result is not None
    assert len(result.enriched_entities) == 1
    assert result.enrichment_summary["total_entities"] == 1


@pytest.mark.asyncio
async def test_quality_score_calculation():
    """Test quality score calculation"""
    enricher = KnowledgeEnricher()

    entities = [
        KnowledgeEntity(
            entity_type=KnowledgeEntityType.SOLUTION_PATTERN,
            content="A" * 200,  # Long content
            confidence=0.9,
            source_artifact_id="test_artifact",
            tags={"tag1", "tag2", "tag3", "tag4"}
        )
    ]

    result = await enricher.enrich_entities(
        entities=entities,
        artifact_type="solution"
    )

    enriched = result.enriched_entities[0]
    assert enriched.quality_score > 0.5


@pytest.mark.asyncio
async def test_enrichment_statistics():
    """Test enrichment statistics"""
    enricher = KnowledgeEnricher()

    entities = [
        KnowledgeEntity(
            entity_type=KnowledgeEntityType.BEST_PRACTICE,
            content="Test content",
            confidence=0.7,
            source_artifact_id="test_artifact"
        )
    ]

    await enricher.enrich_entities(entities, "solution")

    stats = enricher.get_statistics()
    assert stats["entities_enriched"] == 1


# Vector Index Optimizer Tests

@pytest.mark.asyncio
async def test_vector_optimizer_initialization():
    """Test vector optimizer initialization"""
    optimizer = VectorIndexOptimizer()

    assert optimizer is not None
    assert len(optimizer.optimization_history) == 0


@pytest.mark.asyncio
async def test_recommend_strategy_small_dataset():
    """Test strategy recommendation for small dataset"""
    optimizer = VectorIndexOptimizer()

    report = await optimizer.analyze_and_recommend(
        document_count=5000,
        dimension=1536,
        query_pattern="read_heavy"
    )

    assert report is not None
    assert report.optimized_strategy == IndexingStrategy.BASIC
    assert len(report.recommendations) > 0


@pytest.mark.asyncio
async def test_recommend_strategy_medium_dataset():
    """Test strategy recommendation for medium dataset"""
    optimizer = VectorIndexOptimizer()

    report = await optimizer.analyze_and_recommend(
        document_count=50000,
        dimension=1536,
        query_pattern="read_heavy"
    )

    assert report is not None
    assert report.optimized_strategy == IndexingStrategy.HNSW


@pytest.mark.asyncio
async def test_recommend_strategy_large_dataset():
    """Test strategy recommendation for large dataset"""
    optimizer = VectorIndexOptimizer()

    report = await optimizer.analyze_and_recommend(
        document_count=2000000,
        dimension=1536,
        query_pattern="read_heavy"
    )

    assert report is not None
    assert report.optimized_strategy in [
        IndexingStrategy.HNSW_IVF,
        IndexingStrategy.ADAPTIVE
    ]


@pytest.mark.asyncio
async def test_optimize_existing_configuration():
    """Test optimizing existing configuration"""
    optimizer = VectorIndexOptimizer()

    current_config = IndexConfiguration(
        strategy=IndexingStrategy.HNSW,
        dimension=1536,
        ef_construction=100,  # Suboptimal
        M=8  # Suboptimal
    )

    optimized_config, changes = await optimizer.optimize_index(current_config)

    assert optimized_config is not None
    assert len(changes) > 0
    assert optimized_config.ef_construction > current_config.ef_construction


@pytest.mark.asyncio
async def test_estimate_build_time():
    """Test index build time estimation"""
    optimizer = VectorIndexOptimizer()

    report = await optimizer.analyze_and_recommend(
        document_count=10000,
        dimension=1536
    )

    assert "estimated_build_time" in report.metrics
    assert report.metrics["estimated_build_time"] > 0


@pytest.mark.asyncio
async def test_estimate_memory_usage():
    """Test memory usage estimation"""
    optimizer = VectorIndexOptimizer()

    report = await optimizer.analyze_and_recommend(
        document_count=50000,
        dimension=1536
    )

    assert "estimated_memory_usage" in report.metrics
    assert report.metrics["estimated_memory_usage"] > 0


# Advanced RAG Engine Tests

@pytest.mark.asyncio
async def test_rag_engine_initialization():
    """Test RAG engine initialization"""
    engine = AdvancedRAGEngine()

    assert engine is not None
    assert engine.search_stats["total_queries"] == 0


@pytest.mark.asyncio
async def test_create_rag_query():
    """Test RAG query creation"""
    query = RAGQuery(
        query_text="How to implement JWT authentication?",
        search_type=SearchType.HYBRID,
        top_k=5
    )

    assert query.query_text == "How to implement JWT authentication?"
    assert query.search_type == SearchType.HYBRID
    assert query.top_k == 5


@pytest.mark.asyncio
async def test_semantic_search_type():
    """Test semantic search"""
    engine = AdvancedRAGEngine()

    result = await engine.query(
        query_text="Authentication best practices",
        search_type=SearchType.SEMANTIC,
        top_k=3
    )

    assert result is not None
    assert result.query.search_type == SearchType.SEMANTIC


@pytest.mark.asyncio
async def test_hybrid_search_type():
    """Test hybrid search"""
    engine = AdvancedRAGEngine()

    result = await engine.query(
        query_text="How to implement OAuth?",
        search_type=SearchType.HYBRID,
        top_k=5
    )

    assert result is not None
    assert result.query.search_type == SearchType.HYBRID


@pytest.mark.asyncio
async def test_query_with_filters():
    """Test query with metadata filters"""
    engine = AdvancedRAGEngine()

    result = await engine.query(
        query_text="Authentication patterns",
        filters={"artifact_type": "solution"},
        top_k=5
    )

    assert result is not None
    assert result.query.filters == {"artifact_type": "solution"}


@pytest.mark.asyncio
async def test_keyword_extraction():
    """Test keyword extraction from queries"""
    engine = AdvancedRAGEngine()

    keywords = engine._extract_keywords(
        "How to implement JWT authentication in REST API?"
    )

    assert "implement" in keywords
    assert "jwt" in keywords
    assert "authentication" in keywords
    assert len(keywords) > 0


@pytest.mark.asyncio
async def test_result_deduplication():
    """Test result deduplication"""
    engine = AdvancedRAGEngine()

    documents = [
        {"id": "1", "content": "Test document"},
        {"id": "2", "content": "Test document"},  # Duplicate
        {"id": "3", "content": "Different document"}
    ]

    deduplicated = engine._deduplicate_results(documents)

    assert len(deduplicated) < len(documents)


@pytest.mark.asyncio
async def test_apply_filters():
    """Test applying filters to results"""
    engine = AdvancedRAGEngine()

    documents = [
        {"id": "1", "content": "Doc 1", "metadata": {"type": "solution"}},
        {"id": "2", "content": "Doc 2", "metadata": {"type": "critique"}},
        {"id": "3", "content": "Doc 3", "metadata": {"type": "solution"}}
    ]

    filtered = engine._apply_filters(
        documents,
        {"type": "solution"}
    )

    assert len(filtered) == 2
    assert all(doc["metadata"]["type"] == "solution" for doc in filtered)


# Integration Tests

@pytest.mark.asyncio
async def test_full_knowledge_pipeline():
    """Test complete knowledge pipeline"""
    # Initialize components
    extractor = KnowledgeExtractor()
    enricher = KnowledgeEnricher()

    # Step 1: Extract knowledge
    content = """
    Implement JWT-based authentication for the REST API.
    Best practice: Use httpOnly cookies for token storage.
    Pattern: Middleware-based authentication flow.
    Lesson learned: Always validate tokens at each request.
    """

    extraction_result = await extractor.extract_from_artifact(
        artifact_id="pipeline_test",
        content=content,
        artifact_type="solution"
    )

    assert extraction_result is not None
    assert len(extraction_result.entities) > 0

    # Step 2: Enrich knowledge
    enrichment_result = await enricher.enrich_entities(
        entities=extraction_result.entities,
        artifact_type="solution"
    )

    assert enrichment_result is not None
    assert len(enrichment_result.enriched_entities) > 0

    # Step 3: Verify enrichment
    enriched_entity = enrichment_result.enriched_entities[0]

    assert enriched_entity.quality_score >= 0.0
    assert enriched_entity.quality_score <= 1.0


@pytest.mark.asyncio
async def test_vector_optimization_with_rag():
    """Test vector optimization integration with RAG"""
    optimizer = VectorIndexOptimizer()
    engine = AdvancedRAGEngine()

    # Optimize for medium dataset
    optimization_report = await optimizer.analyze_and_recommend(
        document_count=50000,
        dimension=1536,
        query_pattern="read_heavy"
    )

    assert optimization_report is not None

    # Perform search
    search_result = await engine.query(
        query_text="Best authentication practices",
        search_type=SearchType.HYBRID,
        top_k=5
    )

    assert search_result is not None
    assert search_result.retrieval_time_ms >= 0


if __name__ == "__main__":
    # Run tests manually
    import sys

    async def run_tests():
        print("Running Phase 4 Knowledge Base Tests...\n")

        tests = [
            ("Extractor Initialization", test_knowledge_extractor_initialization),
            ("Extract Solution Patterns", test_extract_solution_patterns),
            ("Extract Best Practices", test_extract_best_practices),
            ("Extract Lessons Learned", test_extract_lessons_learned),
            ("Extract Anti-Patterns", test_extract_anti_patterns),
            ("Entity Deduplication", test_entity_deduplication),
            ("Entity Tagging", test_entity_tagging),
            ("Enricher Initialization", test_knowledge_enricher_initialization),
            ("Enrich Entities", test_enrich_entities),
            ("Quality Score Calculation", test_quality_score_calculation),
            ("Vector Optimizer Initialization", test_vector_optimizer_initialization),
            ("Recommend Strategy - Small Dataset", test_recommend_strategy_small_dataset),
            ("Recommend Strategy - Medium Dataset", test_recommend_strategy_medium_dataset),
            ("Recommend Strategy - Large Dataset", test_recommend_strategy_large_dataset),
            ("Optimize Configuration", test_optimize_existing_configuration),
            ("RAG Engine Initialization", test_rag_engine_initialization),
            ("Keyword Extraction", test_keyword_extraction),
            ("Result Deduplication", test_result_deduplication),
            ("Apply Filters", test_apply_filters),
            ("Full Knowledge Pipeline", test_full_knowledge_pipeline),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                await test_func()
                passed += 1
                print(f"✅ PASSED: {name}")
            except Exception as e:
                failed += 1
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")

        print(f"\n{'='*70}")
        print(f"Passed: {passed}/{passed + failed}")
        print('='*70)

        if failed > 0:
            sys.exit(1)

    asyncio.run(run_tests())
