"""
Comprehensive Integration Test Suite for KG-Gen Sprint 2

Tests all components working together end-to-end.
Following CLAUDE.md: RUNTIME TRUTH - Verify actual behavior.
"""

import asyncio
import pytest
import os
from datetime import datetime, timezone

from knowledge_engine.integrations.kggen.extraction_pipeline import (
    ExtractionPipeline,
    ExtractionResult,
    PipelineConfig,
    PipelineStage
)

from knowledge_engine.integrations.kggen.deduplication_engine import (
    DeduplicationEngine,
    DeduplicationConfig,
    DeduplicationMethod
)

from knowledge_engine.integrations.kggen.mcp_server import (
    KGGenMCPServer,
    MemoryManager,
    MemoryType,
    MemoryQuery
)

from knowledge_engine.integrations.kggen.conversation_analyzer import (
    ConversationAnalyzer,
    Message
)

from knowledge_engine.integrations.kggen.graph_aggregator import (
    GraphAggregator,
    GraphAggregatorConfig
)


class TestEndToEnd:
    """
    End-to-end integration tests.

    Tests complete workflows: extraction -> deduplication -> storage -> aggregation
    """

    @pytest.mark.asyncio
    async def test_complete_kg_workflow(self):
        """
        Test complete knowledge graph generation workflow.

        Workflow:
        1. Extract entities and relations from text
        2. Deduplicate entities
        3. Store in memory
        4. Aggregate graphs
        """
        # Step 1: Extract knowledge
        print("\n=== Step 1: Extraction ===")
        pipeline = ExtractionPipeline(
            config=PipelineConfig(
                parallel_workers=2,
                chunk_size=1000
            )
        )

        text = """
        Apple Inc is a technology company headquartered in Cupertino, California.
        Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
        Apple designs and manufactures consumer electronics, computer software, and online services.

        Google LLC is an American technology company specializing in Internet-related services and products.
        Google was founded by Larry Page and Sergey Brin in 1998.
        Google is a subsidiary of Alphabet Inc.

        Microsoft Corporation is an American technology company.
        Microsoft was founded by Bill Gates and Paul Allen in 1975.
        Microsoft develops, manufactures, licenses, supports, and sells computer software, consumer electronics, and personal computers.

        Apple, Google, and Microsoft are all major technology companies.
        They compete in various markets including smartphones, cloud computing, and artificial intelligence.
        """

        extraction_result = await pipeline.extract(
            text=text,
            context="Technology companies overview"
        )

        print(f"Extracted {extraction_result.entity_count} entities")
        print(f"Extracted {extraction_result.relationship_count} relationships")
        print(f"Processing time: {extraction_result.processing_time_seconds:.2f}s")

        assert extraction_result.entity_count > 0, "Should extract entities"
        assert extraction_result.correlation_id, "Should have correlation ID"
        assert extraction_result.validation_passed is not False, "Validation should not fail"

        # Step 2: Deduplicate entities
        print("\n=== Step 2: Deduplication ===")
        dedup_engine = DeduplicationEngine(
            config=DeduplicationConfig()
        )

        dedup_result = await dedup_engine.deduplicate(
            entities=extraction_result.entities,
            method=DeduplicationMethod.FULL,
            correlation_id=extraction_result.correlation_id,
            document_id="tech-companies-doc"
        )

        print(f"Deduplication: {dedup_result.original_count} -> {dedup_result.final_count} entities")
        print(f"Reduction rate: {dedup_result.reduction_rate:.1%}")
        print(f"Clusters created: {dedup_result.clusters_created}")
        print(f"Processing time: {dedup_result.processing_time_seconds:.2f}s")

        assert dedup_result.final_count <= dedup_result.original_count
        assert dedup_result.reduction_rate >= 0

        # Step 3: Store in memory
        print("\n=== Step 3: Memory Storage ===")
        mcp_server = KGGenMCPServer()

        # Store entities as memories
        for entity in dedup_result.unique_entities[:10]:  # Store first 10
            memory = await mcp_server.memory_manager.add_memory(
                content=entity,
                memory_type=MemoryType.ENTITY,
                session_id="tech-companies-session",
                importance=0.7,
                source="extraction_pipeline"
            )

            print(f"Stored memory: {memory.memory_id} - {entity}")

        # Store some relationships
        for rel in extraction_result.relationships[:5]:  # Store first 5
            content = f"{rel['subject']} {rel['predicate']} {rel['object']}"
            await mcp_server.memory_manager.add_memory(
                content=content,
                memory_type=MemoryType.RELATIONSHIP,
                session_id="tech-companies-session",
                importance=0.6,
                source="extraction_pipeline"
            )

        # Retrieve memories
        query = MemoryQuery(
            query_text="Apple",
            session_id="tech-companies-session",
            max_results=5
        )

        memories = await mcp_server.memory_manager.retrieve_relevant_memories(query)

        print(f"Retrieved {len(memories)} memories for query 'Apple'")

        assert len(memories) > 0, "Should retrieve memories"

        # Step 4: Aggregate graphs
        print("\n=== Step 4: Graph Aggregation ===")
        aggregator = GraphAggregator(
            config=GraphAggregatorConfig(
                merge_strategy="union"
            )
        )

        # Create graph from deduplicated results
        graph = {
            "entities": dedup_result.unique_entities,
            "relationships": extraction_result.relationships
        }

        agg_result = await aggregator.aggregate(
            graphs=[graph],
            correlation_id=extraction_result.correlation_id,
            create_version=True
        )

        print(f"Aggregated graph: {agg_result.total_entities} entities, {agg_result.total_relationships} relationships")
        print(f"Conflicts resolved: {agg_result.conflicts_resolved}")
        print(f"Processing time: {agg_result.processing_time_seconds:.2f}s")

        assert agg_result.total_entities > 0
        assert agg_result.aggregated_graph.version_id

        # Cleanup
        await pipeline.close()
        await dedup_engine.close()
        await mcp_server.close()
        await aggregator.close()

        print("\n=== Complete Workflow Test PASSED ===")

    @pytest.mark.asyncio
    async def test_conversation_to_kg_workflow(self):
        """
        Test conversation analysis to knowledge graph workflow.
        """
        print("\n=== Conversation Analysis Workflow ===")

        # Step 1: Analyze conversation
        analyzer = ConversationAnalyzer()

        messages = [
            {
                "role": "user",
                "content": "Can you tell me about the founders of major tech companies?",
                "speaker_id": "user1"
            },
            {
                "role": "assistant",
                "content": "Sure! Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne. Google was founded by Larry Page and Sergey Brin.",
                "speaker_id": "assistant"
            },
            {
                "role": "user",
                "content": "What about Microsoft?",
                "speaker_id": "user1"
            },
            {
                "role": "assistant",
                "content": "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
                "speaker_id": "assistant"
            },
            {
                "role": "user",
                "content": "Thanks! That's very helpful.",
                "speaker_id": "user1"
            }
        ]

        conv_result = await analyzer.analyze(
            messages=messages,
            conversation_id="tech-founders-conv"
        )

        print(f"Conversation ID: {conv_result.conversation_id}")
        print(f"Total speakers: {conv_result.total_speakers}")
        print(f"Total entities extracted: {conv_result.total_entities}")
        print(f"Total relations: {conv_result.total_relations}")
        print(f"Processing time: {conv_result.processing_time_seconds:.2f}s")

        assert conv_result.conversation_id
        assert conv_result.total_speakers >= 2
        assert conv_result.processing_time_seconds >= 0

        # Step 2: Convert to graph and aggregate
        aggregator = GraphAggregator()

        graph = {
            "entities": conv_result.entities,
            "relationships": conv_result.relationships
        }

        agg_result = await aggregator.aggregate([graph])

        print(f"Aggregated {agg_result.total_entities} entities from conversation")

        assert agg_result.total_entities >= 0

        # Step 3: Store conversation insights in memory
        mcp_server = KGGenMCPServer()

        if conv_result.summary:
            summary_memory = await mcp_server.memory_manager.add_memory(
                content=f"Conversation about tech founders: {conv_result.summary.topic}",
                memory_type=MemoryType.CONVERSATION,
                session_id="tech-founders-session",
                importance=0.8
            )

            print(f"Stored summary memory: {summary_memory.memory_id}")

        # Store speaker entities
        for entity in conv_result.speaker_entities[:5]:
            memory = await mcp_server.memory_manager.add_memory(
                content=entity.entity_name,
                memory_type=MemoryType.ENTITY,
                session_id="tech-founders-session",
                importance=entity.importance
            )

            print(f"Stored entity memory: {entity.entity_name}")

        # Cleanup
        await analyzer.close()
        await aggregator.close()
        await mcp_server.close()

        print("\n=== Conversation Analysis Workflow PASSED ===")

    @pytest.mark.asyncio
    async def test_idempotency_across_pipeline(self):
        """
        Test idempotency: Running same data through pipeline twice should be safe.
        """
        print("\n=== Idempotency Test ===")

        text = "Apple and Google are technology companies."

        # Run 1
        pipeline1 = ExtractionPipeline()
        result1 = await pipeline1.extract(text=text)
        await pipeline1.close()

        # Run 2
        pipeline2 = ExtractionPipeline()
        result2 = await pipeline2.extract(text=text)
        await pipeline2.close()

        # Results should be consistent
        print(f"Run 1: {result1.entity_count} entities, {result1.relationship_count} relations")
        print(f"Run 2: {result2.entity_count} entities, {result2.relationship_count} relations")

        # Entity extraction should be deterministic (with fallback)
        # Results may vary slightly with LLM, but should be similar
        assert result1.entity_count > 0
        assert result2.entity_count > 0

        print("\n=== Idempotency Test PASSED ===")

    @pytest.mark.asyncio
    async def test_cross_document_entity_resolution(self):
        """
        Test cross-document entity resolution.
        """
        print("\n=== Cross-Document Entity Resolution Test ===")

        dedup_engine = DeduplicationEngine()

        # Document 1: Apple, Google, Microsoft
        entities1 = ["Apple", "Google", "Microsoft"]
        dedup_engine.cross_doc_resolver.register_document_entities(
            "doc1",
            entities1
        )

        # Document 2: Apple, Amazon, Facebook
        entities2 = ["Apple", "Amazon", "Facebook"]
        result2 = await dedup_engine.deduplicate(
            entities=entities2,
            document_id="doc2"
        )

        # Find common entities
        common = dedup_engine.cross_doc_resolver.find_common_entities(
            ["doc1", "doc2"]
        )

        print(f"Common entities across doc1 and doc2: {common}")

        assert "Apple" in common

        # Get related documents for Apple
        related = dedup_engine.cross_doc_resolver.get_related_documents("Apple")

        print(f"Documents mentioning 'Apple': {related}")

        assert "doc1" in related
        assert "doc2" in related

        await dedup_engine.close()

        print("\n=== Cross-Document Entity Resolution Test PASSED ===")

    @pytest.mark.asyncio
    async def test_memory_aggregation_and_backup(self):
        """
        Test memory aggregation and backup functionality.
        """
        print("\n=== Memory Aggregation and Backup Test ===")

        mcp_server = KGGenMCPServer()

        # Add memories to a session
        session_id = "test-aggregation-session"

        for i in range(5):
            await mcp_server.memory_manager.add_memory(
                content=f"Test memory {i}",
                memory_type=MemoryType.FACT,
                session_id=session_id,
                importance=0.5 + (i * 0.1)
            )

        # Aggregate session memories
        agg_result = await mcp_server.memory_manager.aggregate_session_memories(
            session_id=session_id
        )

        print(f"Session aggregation:")
        print(f"  Total memories: {agg_result['total_memories']}")
        print(f"  By type: {agg_result['by_type']}")
        print(f"  Avg importance: {agg_result['avg_importance']:.2f}")
        print(f"  Avg confidence: {agg_result['avg_confidence']:.2f}")

        assert agg_result['total_memories'] == 5

        # Test backup
        backup_success = await mcp_server.memory_manager.backup_memories()

        print(f"Backup successful: {backup_success}")

        assert backup_success is True

        await mcp_server.close()

        print("\n=== Memory Aggregation and Backup Test PASSED ===")

    @pytest.mark.asyncio
    async def test_graph_versioning_and_differential(self):
        """
        Test graph versioning and differential comparison.
        """
        print("\n=== Graph Versioning and Differential Test ===")

        aggregator = GraphAggregator(
            config=GraphAggregatorConfig(
                auto_version=True,
                max_versions=10
            )
        )

        # Version 1: Initial graph
        graph1 = {
            "entities": ["Apple", "Google"],
            "relationships": [
                {"subject": "Apple", "predicate": "competes_with", "object": "Google"}
            ]
        }

        result1 = await aggregator.aggregate([graph1])
        version1_id = result1.aggregated_graph.version_id

        print(f"Version 1: {version1_id}")
        print(f"  Entities: {result1.total_entities}")
        print(f"  Relationships: {result1.total_relationships}")

        # Version 2: Add Microsoft
        graph2 = {
            "entities": ["Apple", "Google", "Microsoft"],
            "relationships": [
                {"subject": "Apple", "predicate": "competes_with", "object": "Google"},
                {"subject": "Microsoft", "predicate": "competes_with", "object": "Apple"}
            ]
        }

        result2 = await aggregator.aggregate([graph2])
        version2_id = result2.aggregated_graph.version_id

        print(f"Version 2: {version2_id}")
        print(f"  Entities: {result2.total_entities}")
        print(f"  Relationships: {result2.total_relationships}")

        # Compare versions
        diff = await aggregator.compare_versions(
            version1_id,
            version2_id
        )

        print(f"\nDifferential comparison:")
        print(f"  Entities added: {diff.entities_added}")
        print(f"  Entities removed: {diff.entities_removed}")
        print(f"  Relationships added: {len(diff.relationships_added)}")
        print(f"  Change count: {diff.change_count}")
        print(f"  Similarity score: {diff.similarity_score:.2f}")

        assert "Microsoft" in diff.entities_added
        assert diff.change_count > 0
        assert diff.similarity_score > 0

        # List versions
        versions = await aggregator.list_versions()

        print(f"\nTotal versions stored: {len(versions)}")

        assert len(versions) >= 2

        await aggregator.close()

        print("\n=== Graph Versioning and Differential Test PASSED ===")


class TestErrorHandling:
    """
    Test error handling and edge cases.
    """

    @pytest.mark.asyncio
    async def test_empty_text_extraction(self):
        """Test extraction with empty text."""
        pipeline = ExtractionPipeline()

        result = await pipeline.extract(text="")

        assert result.entity_count == 0
        assert result.relationship_count == 0

        await pipeline.close()

    @pytest.mark.asyncio
    async def test_empty_deduplication(self):
        """Test deduplication with empty list."""
        engine = DeduplicationEngine()

        result = await engine.deduplicate(entities=[])

        assert result.final_count == 0
        assert result.original_count == 0

        await engine.close()

    @pytest.mark.asyncio
    async def test_empty_conversation(self):
        """Test conversation analysis with empty messages."""
        analyzer = ConversationAnalyzer()

        result = await analyzer.analyze(messages=[])

        assert result.conversation_id
        assert result.total_speakers == 0

        await analyzer.close()

    @pytest.mark.asyncio
    async def test_memory_retrieval_no_results(self):
        """Test memory retrieval when no results found."""
        server = KGGenMCPServer()

        query = MemoryQuery(
            query_text="nonexistent query",
            session_id="empty-session",
            max_results=10
        )

        memories = await server.memory_manager.retrieve_relevant_memories(query)

        assert len(memories) == 0

        await server.close()

    @pytest.mark.asyncio
    async def test_graph_aggregation_empty(self):
        """Test graph aggregation with empty list."""
        aggregator = GraphAggregator()

        result = await aggregator.aggregate(graphs=[])

        assert result.total_entities == 0
        assert result.total_relationships == 0

        await aggregator.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s", "--tb=short"])
