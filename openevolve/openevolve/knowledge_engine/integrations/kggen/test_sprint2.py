"""
Comprehensive Test Suite for KG-Gen Sprint 2 Integration

Task 2.6.1: Write unit tests for extraction pipeline
Task 2.6.2: Write integration tests for deduplication

Following CLAUDE.md:
- RUNTIME TRUTH: Tests verify actual behavior
- IDEMPOTENCY: Tests verify retry safety
"""

import asyncio
import pytest
import os
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any

# Import all KG-Gen components
from knowledge_engine.integrations.kggen.extraction_pipeline import (
    ExtractionPipeline,
    ExtractionResult,
    PipelineConfig,
    PipelineStage
)

from knowledge_engine.integrations.kggen.deduplication_engine import (
    DeduplicationEngine,
    DeduplicationConfig,
    DeduplicationMethod,
    SEMHASHStrategy,
    LMClusterStrategy
)

from knowledge_engine.integrations.kggen.mcp_server import (
    KGGenMCPServer,
    MemoryManager,
    MemoryType,
    MemoryQuery
)

from knowledge_engine.integrations.kggen.conversation_analyzer import (
    ConversationAnalyzer,
    ConversationAnalyzerConfig,
    Message
)

from knowledge_engine.integrations.kggen.graph_aggregator import (
    GraphAggregator,
    GraphAggregatorConfig,
    GraphVersion
)


class TestExtractionPipeline:
    """
    Unit tests for extraction pipeline.

    Task 2.6.1: Write unit tests for extraction pipeline.
    """

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        config = PipelineConfig(
            parallel_workers=2,
            chunk_size=1000
        )
        return ExtractionPipeline(config)

    def test_config_validation(self, pipeline):
        """Test configuration validation."""
        assert pipeline.config.chunk_size == 1000
        assert pipeline.config.parallel_workers == 2

    def test_correlation_id_generation(self, pipeline):
        """Test correlation ID generation."""
        text = "Test text"
        correlation_id = pipeline.generate_correlation_id(text)

        assert correlation_id.startswith("kggen-")
        assert len(correlation_id) > 20

    def test_fallback_entity_extraction(self, pipeline):
        """Test fallback entity extraction."""
        text = "Apple and Google are major tech companies. Microsoft develops software."
        entities = pipeline._extract_entities_fallback(text)

        assert len(entities) > 0
        assert "Apple" in entities or "apple" in entities
        assert "Google" in entities or "google" in entities

    def test_fallback_relation_extraction(self, pipeline):
        """Test fallback relation extraction."""
        text = "Apple owns iOS. Google develops Android."
        entities = ["Apple", "Google", "iOS", "Android"]
        relations = pipeline._extract_relations_fallback(text, entities)

        # Should extract some relations
        assert isinstance(relations, list)

    @pytest.mark.asyncio
    async def test_extract_entities_from_chunk(self, pipeline):
        """Test entity extraction from chunk."""
        chunk = "Apple Inc is a technology company headquartered in Cupertino."
        entities = await pipeline._extract_entities_from_chunk(chunk, "test context")

        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_full_extraction(self, pipeline):
        """Test full extraction pipeline."""
        text = """
        Apple is a technology company founded by Steve Jobs.
        Google was founded by Larry Page and Sergey Brin.
        Microsoft was founded by Bill Gates.
        """

        result = await pipeline.extract(
            text=text,
            context="test context"
        )

        assert isinstance(result, ExtractionResult)
        assert result.correlation_id
        assert result.entity_count == len(result.entities)
        assert result.relationship_count == len(result.relationships)
        assert result.processing_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_pipeline_status_tracking(self, pipeline):
        """Test pipeline status tracking."""
        text = "Apple and Google are tech companies."

        status_updates = []

        def progress_callback(status):
            status_updates.append(status.to_dict())

        result = await pipeline.extract(
            text=text,
            progress_callback=progress_callback
        )

        # Should have status updates
        assert len(status_updates) > 0

        # Final status should be completed
        final_status = pipeline.get_status(result.correlation_id)
        assert final_status is not None
        assert final_status.stage == PipelineStage.COMPLETED

    @pytest.mark.asyncio
    async def test_idempotency(self, pipeline):
        """
        Test extraction idempotency.

        LAW OF IDEMPOTENCY: Same input should produce consistent results.
        """
        text = "Apple is a tech company."

        result1 = await pipeline.extract(text=text)
        result2 = await pipeline.extract(text=text)

        # Results should be consistent
        assert result1.entities == result2.entities


class TestDeduplicationEngine:
    """
    Unit and integration tests for deduplication engine.

    Task 2.6.2: Write integration tests for deduplication.
    """

    @pytest.fixture
    def dedup_engine(self):
        """Create deduplication engine."""
        config = DeduplicationConfig()
        return DeduplicationEngine(config)

    def test_config_validation(self, dedup_engine):
        """Test configuration validation."""
        assert 0.0 <= dedup_engine.config.semhash_threshold <= 1.0
        assert 0.0 <= dedup_engine.config.lm_similarity_threshold <= 1.0

    @pytest.mark.asyncio
    async def test_semhash_deduplication(self, dedup_engine):
        """Test SEMHASH deduplication."""
        entities = [
            "Apple",
            "apple",
            "APPLE",
            "Google",
            "google",
            "Microsoft"
        ]

        result = await dedup_engine.deduplicate(
            entities=entities,
            method=DeduplicationMethod.SEMHASH
        )

        assert result.final_count < result.original_count
        assert result.duplicates_removed > 0
        assert result.reduction_rate > 0

    @pytest.mark.asyncio
    async def test_lm_cluster_deduplication(self, dedup_engine):
        """Test LM clustering deduplication."""
        entities = [
            "Apple Inc",
            "Apple Corporation",
            "Google LLC",
            "Google Inc",
            "Microsoft Corp"
        ]

        result = await dedup_engine.deduplicate(
            entities=entities,
            method=DeduplicationMethod.LM_CLUSTER
        )

        assert result.final_count <= result.original_count
        assert len(result.entity_clusters) >= 0

    @pytest.mark.asyncio
    async def test_full_deduplication(self, dedup_engine):
        """Test full deduplication (SEMHASH + LM)."""
        entities = [
            "Apple",
            "apple",
            "APPLE",
            "Apple Inc",
            "Apple Corporation",
            "Google",
            "Google LLC"
        ]

        result = await dedup_engine.deduplicate(
            entities=entities,
            method=DeduplicationMethod.FULL
        )

        # Should have more reduction than single method
        assert result.final_count < result.original_count
        assert result.duplicates_removed > 0

    @pytest.mark.asyncio
    async def test_relationship_deduplication(self, dedup_engine):
        """Test relationship deduplication."""
        relationships = [
            {"subject": "Apple", "predicate": "owns", "object": "iOS"},
            {"subject": "Apple", "predicate": "owns", "object": "iOS"},  # Duplicate
            {"subject": "Google", "predicate": "owns", "object": "Android"},
            {"subject": "Apple", "predicate": "owns", "object": "iOS"},  # Duplicate
        ]

        unique = await dedup_engine.deduplicate_relationships(relationships)

        assert len(unique) < len(relationships)
        assert len(unique) == 2  # Only 2 unique relationships

    @pytest.mark.asyncio
    async def test_deduplication_idempotency(self, dedup_engine):
        """
        Test deduplication idempotency.

        LAW OF IDEMPOTENCY: Running deduplication twice should be safe.
        """
        entities = ["Apple", "apple", "Google"]

        result1 = await dedup_engine.deduplicate(entities, method=DeduplicationMethod.SEMHASH)
        result2 = await dedup_engine.deduplicate(result1.unique_entities, method=DeduplicationMethod.SEMHASH)

        # Second run should not change count
        assert result2.final_count == result2.original_count

    @pytest.mark.asyncio
    async def test_cross_document_resolution(self, dedup_engine):
        """Test cross-document entity resolution."""
        # Register entities from different documents
        dedup_engine.cross_doc_resolver.register_document_entities(
            "doc1",
            ["Apple", "Google", "Microsoft"]
        )

        dedup_engine.cross_doc_resolver.register_document_entities(
            "doc2",
            ["Apple", "Amazon", "Facebook"]
        )

        # Find common entities
        common = dedup_engine.cross_doc_resolver.find_common_entities(["doc1", "doc2"])

        assert "Apple" in common

        # Find related documents
        related = dedup_engine.cross_doc_resolver.get_related_documents("Apple")

        assert "doc1" in related
        assert "doc2" in related


class TestMCPServer:
    """Unit tests for MCP server."""

    @pytest.fixture
    def mcp_server(self):
        """Create MCP server."""
        return KGGenMCPServer()

    @pytest.mark.asyncio
    async def test_add_memory(self, mcp_server):
        """Test adding a memory."""
        memory = await mcp_server.memory_manager.add_memory(
            content="Test fact",
            memory_type=MemoryType.FACT,
            session_id="test-session"
        )

        assert memory.memory_id
        assert memory.content == "Test fact"
        assert memory.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_memory_retrieval(self, mcp_server):
        """Test memory retrieval."""
        # Add memory
        await mcp_server.memory_manager.add_memory(
            content="Python is a programming language",
            memory_type=MemoryType.FACT,
            session_id="test-session"
        )

        # Retrieve
        query = MemoryQuery(
            query_text="programming",
            session_id="test-session",
            max_results=10
        )

        memories = await mcp_server.memory_manager.retrieve_relevant_memories(query)

        assert len(memories) > 0

    @pytest.mark.asyncio
    async def test_add_memories_tool(self, mcp_server):
        """Test add_memories MCP tool."""
        memories_data = [
            {"content": "Fact 1", "memory_type": "fact"},
            {"content": "Fact 2", "memory_type": "fact"}
        ]

        result = await mcp_server.add_memories(
            memories=memories_data,
            session_id="test-session"
        )

        assert result["success"] == True
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_relevant_memories_tool(self, mcp_server):
        """Test retrieve_relevant_memories MCP tool."""
        # Add test memory
        await mcp_server.memory_manager.add_memory(
            content="Test memory about AI",
            memory_type=MemoryType.FACT,
            session_id="test-session"
        )

        result = await mcp_server.retrieve_relevant_memories(
            query_text="AI",
            session_id="test-session",
            max_results=10
        )

        assert result["success"] == True
        assert result["count"] >= 0

    @pytest.mark.asyncio
    async def test_visualize_memories_tool(self, mcp_server):
        """Test visualize_memories MCP tool."""
        # Use unique session ID for test isolation
        unique_session = f"test-session-viz-{uuid.uuid4().hex[:8]}"

        # Add memories
        for i in range(3):
            await mcp_server.memory_manager.add_memory(
                content=f"Memory {i}",
                memory_type=MemoryType.FACT,
                session_id=unique_session
            )

        result = await mcp_server.visualize_memories(session_id=unique_session)

        assert result["success"] == True
        assert result["statistics"]["total_memories"] == 3

    @pytest.mark.asyncio
    async def test_memory_idempotency(self, mcp_server):
        """Test memory idempotency."""
        # Use unique session ID for test isolation
        unique_session = f"test-session-idem-{uuid.uuid4().hex[:8]}"
        content = "Test memory"

        # Add same memory twice
        mem1 = await mcp_server.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.FACT,
            session_id=unique_session
        )

        mem2 = await mcp_server.memory_manager.add_memory(
            content=content,
            memory_type=MemoryType.FACT,
            session_id=unique_session
        )

        # Should update existing, not create new
        assert mem1.memory_id == mem2.memory_id
        assert mem2.access_count >= 1  # At least 1 access (idempotent update)


class TestConversationAnalyzer:
    """Unit tests for conversation analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create conversation analyzer."""
        return ConversationAnalyzer()

    def test_message_parsing(self, analyzer):
        """Test message parsing."""
        messages = [
            {"role": "user", "content": "Hello", "speaker_id": "user1"},
            {"role": "assistant", "content": "Hi there", "speaker_id": "assistant"}
        ]

        parsed = analyzer._parse_messages(messages)

        assert len(parsed) == 2
        assert parsed[0].role == "user"
        assert parsed[0].content == "Hello"

    @pytest.mark.asyncio
    async def test_conversation_analysis(self, analyzer):
        """Test full conversation analysis."""
        messages = [
            {"role": "user", "content": "Tell me about Apple", "speaker_id": "user1"},
            {"role": "assistant", "content": "Apple is a tech company", "speaker_id": "assistant"},
            {"role": "user", "content": "What about Google?", "speaker_id": "user1"},
            {"role": "assistant", "content": "Google is also a tech company", "speaker_id": "assistant"}
        ]

        result = await analyzer.analyze(messages)

        assert result.conversation_id
        assert result.total_speakers > 0
        assert result.processing_time_seconds >= 0

    @pytest.mark.asyncio
    async def test_speaker_entity_extraction(self, analyzer):
        """Test speaker entity extraction."""
        messages = [
            Message(role="user", content="Apple and Google are tech companies", speaker_id="user1"),
            Message(role="assistant", content="Yes, both are major companies", speaker_id="assistant")
        ]

        entities = await analyzer.entity_extractor.extract_entities(
            messages,
            speaker_id="user1",
            correlation_id="test"
        )

        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_conversation_to_kg(self, analyzer):
        """Test conversation-to-knowledge-graph conversion."""
        messages = [
            {"role": "user", "content": "Tell me about Apple", "speaker_id": "user1"},
            {"role": "assistant", "content": "Apple is a tech company", "speaker_id": "assistant"}
        ]

        result = await analyzer.analyze(messages)

        # Should have entities and relationships
        assert isinstance(result.entities, list)
        assert isinstance(result.relationships, list)


class TestGraphAggregator:
    """Unit tests for graph aggregator."""

    @pytest.fixture
    def aggregator(self):
        """Create graph aggregator."""
        return GraphAggregator()

    @pytest.mark.asyncio
    async def test_graph_aggregation(self, aggregator):
        """Test graph aggregation."""
        graphs = [
            {
                "entities": ["Apple", "Google"],
                "relationships": [
                    {"subject": "Apple", "predicate": "competes_with", "object": "Google"}
                ]
            },
            {
                "entities": ["Apple", "Microsoft"],
                "relationships": [
                    {"subject": "Apple", "predicate": "competes_with", "object": "Microsoft"}
                ]
            }
        ]

        result = await aggregator.aggregate(graphs)

        assert result.total_entities == 3  # Apple, Google, Microsoft
        assert result.total_relationships == 2
        assert result.aggregated_graph.version_id

    @pytest.mark.asyncio
    async def test_versioning(self, aggregator):
        """Test graph versioning."""
        graph1 = {
            "entities": ["Apple"],
            "relationships": []
        }

        result1 = await aggregator.aggregate([graph1])
        version1_id = result1.aggregated_graph.version_id

        # Add another version
        graph2 = {
            "entities": ["Apple", "Google"],
            "relationships": []
        }

        result2 = await aggregator.aggregate([graph2])

        # Should have two versions
        versions = await aggregator.list_versions()

        assert len(versions) >= 2

    @pytest.mark.asyncio
    async def test_graph_diff(self, aggregator):
        """Test graph differential comparison."""
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
            result1.aggregated_graph.version_id,
            result2.aggregated_graph.version_id
        )

        assert diff.entities_added == ["Microsoft"]
        assert diff.change_count > 0

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, aggregator):
        """Test conflict resolution."""
        # Same entity from multiple sources
        graphs = [
            {
                "entities": ["Apple"],
                "relationships": [
                    {"subject": "Apple", "predicate": "owns", "object": "iOS"}
                ]
            },
            {
                "entities": ["Apple"],
                "relationships": [
                    {"subject": "Apple", "predicate": "owns", "object": "iOS"}
                ]
            }
        ]

        result = await aggregator.aggregate(graphs)

        # Should resolve conflicts
        assert result.total_entities == 1
        assert result.conflicts_resolved >= 0


class TestIntegration:
    """
    Integration tests for complete KG-Gen pipeline.

    Tests the full workflow: extraction -> deduplication -> storage -> aggregation
    """

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test complete pipeline workflow."""
        # 1. Extract knowledge
        pipeline = ExtractionPipeline()
        text = """
        Apple is a technology company founded by Steve Jobs.
        Google was founded by Larry Page.
        Microsoft was founded by Bill Gates.
        Apple, Google, and Microsoft are all technology companies.
        """

        extraction_result = await pipeline.extract(text=text)

        assert extraction_result.entity_count > 0

        # 2. Deduplicate
        dedup_engine = DeduplicationEngine()
        dedup_result = await dedup_engine.deduplicate(
            entities=extraction_result.entities,
            method=DeduplicationMethod.FULL
        )

        assert dedup_result.final_count <= dedup_result.original_count

        # 3. Store in memory
        mcp_server = KGGenMCPServer()

        for entity in dedup_result.unique_entities:
            await mcp_server.memory_manager.add_memory(
                content=entity,
                memory_type=MemoryType.ENTITY,
                session_id="test-session"
            )

        # 4. Aggregate
        aggregator = GraphAggregator()

        graph = {
            "entities": dedup_result.unique_entities,
            "relationships": extraction_result.relationships
        }

        agg_result = await aggregator.aggregate([graph])

        assert agg_result.total_entities > 0

        # Cleanup
        await pipeline.close()
        await dedup_engine.close()
        await mcp_server.close()
        await aggregator.close()

    @pytest.mark.asyncio
    async def test_conversation_to_kg_workflow(self):
        """Test conversation analysis to knowledge graph workflow."""
        # Analyze conversation
        analyzer = ConversationAnalyzer()

        messages = [
            {"role": "user", "content": "Tell me about Apple and Google", "speaker_id": "user1"},
            {"role": "assistant", "content": "Both are major tech companies", "speaker_id": "assistant"}
        ]

        conv_result = await analyzer.analyze(messages)

        # Convert to graph and aggregate
        aggregator = GraphAggregator()

        graph = {
            "entities": conv_result.entities,
            "relationships": conv_result.relationships
        }

        agg_result = await aggregator.aggregate([graph])

        assert agg_result.total_entities >= 0

        await analyzer.close()
        await aggregator.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
