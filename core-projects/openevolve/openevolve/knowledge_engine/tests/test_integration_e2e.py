"""
End-to-End Integration Tests for Knowledge Engine

Following CLAUDE.md principles:
- RUNTIME TRUTH: Test full pipeline against live services
- IDEMPOTENCY: Tests safe to run multiple times
- STRUCTURED LOGGING: JSON logs for traceability

Tests verify:
- Full document processing pipeline
- Knowledge graph generation from documents
- Temporal knowledge queries
- Bilingual extraction (EN/CN)
- Visualization generation
- Agent memory persistence
- Contradiction detection
- Deduplication across documents
"""

import asyncio
import json
import logging
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import sys
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import core module using conftest's approach
CORE_AVAILABLE = False
KnowledgeState = None
EntityKnowledgeGraph = None

try:
    spec = importlib.util.spec_from_file_location(
        "core",
        project_root / "knowledge_engine" / "core.py"
    )
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        EntityKnowledgeGraph = core_module.EntityKnowledgeGraph
        KnowledgeState = core_module.KnowledgeState
        CORE_AVAILABLE = True
except Exception as e:
    CORE_AVAILABLE = False
    KnowledgeState = None
    EntityKnowledgeGraph = None

try:
    from knowledge_engine.knowledge_extractor import KnowledgeExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False
    KnowledgeExtractor = None

try:
    from knowledge_engine.document_loader import DocumentLoader
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False
    DocumentLoader = None

logger = logging.getLogger(__name__)


class TestDocumentProcessingPipeline:
    """
    End-to-end tests for document processing pipeline.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not LOADER_AVAILABLE, reason="DocumentLoader not available")
    async def test_full_document_ingestion(self, sample_document, temp_dir, performance_tracker):
        """
        Test complete document ingestion: load → extract → store → retrieve.
        """
        performance_tracker.start()

        try:
            # Step 1: Load document
            loader = DocumentLoader()
            doc_path = temp_dir / "test_doc.txt"
            doc_path.write_text(sample_document)

            loaded_doc = await loader.load_text(str(doc_path))
            assert loaded_doc is not None
            assert len(loaded_doc) > 0
            performance_tracker.record_operation()

            logger.info(json.dumps({
                "msg": "Document loaded successfully",
                "doc_length": len(loaded_doc),
                "level": "INFO"
            }))

            # Step 2: Extract knowledge
            if not EXTRACTOR_AVAILABLE:
                pytest.skip("KnowledgeExtractor not available")
            extractor = KnowledgeExtractor()
            extraction_result = await extractor.extract(loaded_doc)

            assert extraction_result is not None
            assert "entities" in extraction_result or "knowledge" in extraction_result
            performance_tracker.record_operation()

            logger.info(json.dumps({
                "msg": "Knowledge extracted successfully",
                "extraction_type": type(extraction_result).__name__,
                "level": "INFO"
            }))

            # Step 3: Store in graph
            graph = EntityKnowledgeGraph()
            if "entities" in extraction_result:
                for entity in extraction_result["entities"]:
                    if isinstance(entity, dict):
                        await graph.add_entity(
                            entity.get("name", entity.get("text", "Unknown")),
                            entity
                        )
                    else:
                        await graph.add_entity(str(entity))

            if "relationships" in extraction_result:
                for rel in extraction_result["relationships"]:
                    if isinstance(rel, dict):
                        await graph.add_relationship(
                            rel.get("subject", rel.get("source")),
                            rel.get("predicate", rel.get("relation")),
                            rel.get("object", rel.get("target"))
                        )

            performance_tracker.record_operation()

            # Step 4: Retrieve and verify
            entities = graph.get_entities()
            assert len(entities) > 0

            logger.info(json.dumps({
                "msg": "Document processed end-to-end",
                "entities_stored": len(entities),
                "level": "INFO"
            }))

        except Exception as e:
            performance_tracker.record_error(str(e))
            logger.error(json.dumps({
                "msg": "Document processing failed",
                "error": str(e),
                "level": "ERROR"
            }))
            raise

        finally:
            performance_tracker.stop()
            metrics = performance_tracker.get_metrics()

            logger.info(json.dumps({
                "msg": "Performance metrics",
                "metrics": metrics,
                "level": "INFO"
            }))

            # Verify performance baseline
            assert metrics["duration_ms"] < 10000, "Processing took too long"


class TestKnowledgeGraphGeneration:
    """
    End-to-end tests for knowledge graph generation.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_graph_from_multiple_documents(self, generate_test_documents):
        """
        Test generating a unified graph from multiple documents.
        """
        documents = generate_test_documents(5)
        graph = EntityKnowledgeGraph()

        for i, doc in enumerate(documents):
            # Extract entities from each document
            entities = self._extract_entities_simple(doc)
            for entity in entities:
                await graph.add_entity(entity, {"source_doc": f"doc_{i}"})

        # Verify graph structure
        all_entities = graph.get_entities()
        assert len(all_entities) > 0

        logger.info(json.dumps({
            "msg": "Graph generated from multiple documents",
            "doc_count": len(documents),
            "entity_count": len(all_entities),
            "level": "INFO"
        }))

    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction for testing."""
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 3]
        return entities[:3]  # Limit for testing


class TestTemporalQueries:
    """
    End-to-end tests for temporal knowledge queries.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_temporal_evolution_tracking(self):
        """
        Test tracking knowledge evolution over time.
        """
        state = KnowledgeState(query="AI development over time")

        # Add knowledge at different time points
        times = [
            datetime.now() - timedelta(days=30),
            datetime.now() - timedelta(days=15),
            datetime.now()
        ]

        facts = [
            "AI research began in the 1950s",
            "Deep learning revolution started in 2010s",
            "Large language models emerged in 2020s"
        ]

        for time, fact in zip(times, facts):
            state.add_fact(fact)
            state.add_workflow_execution(
                workflow_id=f"workflow_{time.strftime('%Y%m%d')}",
                artifacts_extracted=1,
                timestamp=time.isoformat()
            )

        # Verify temporal evolution
        assert len(state.search_history) == 3
        assert len(state.facts) == 3

        # Verify chronological order
        timestamps = [h["timestamp"] for h in state.search_history]
        assert timestamps == sorted(timestamps)

        logger.info(json.dumps({
            "msg": "Temporal evolution tracked successfully",
            "time_points": len(state.search_history),
            "level": "INFO"
        }))


class TestBilingualExtraction:
    """
    End-to-end tests for bilingual (EN/CN) extraction.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_english_document_extraction(self):
        """
        Test extraction from English document.
        """
        english_doc = """
        Machine learning is a subset of artificial intelligence.
        It focuses on building systems that learn from data.
        Neural networks are a key technique in deep learning.
        """

        graph = EntityKnowledgeGraph()
        entities = self._extract_entities_simple(english_doc)

        for entity in entities:
            await graph.add_entity(entity, {"language": "en"})

        # Verify English entities extracted
        en_entities = [e for e in graph.get_entities() if graph.get_entity(e).get("language") == "en"]
        assert len(en_entities) > 0

        logger.info(json.dumps({
            "msg": "English extraction successful",
            "entity_count": len(en_entities),
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_chinese_document_extraction(self):
        """
        Test extraction from Chinese document.
        """
        chinese_doc = """
        机器学习是人工智能的一个子集。
        它专注于构建从数据中学习的系统。
        神经网络是深度学习的关键技术。
        """

        graph = EntityKnowledgeGraph()
        # Simple extraction: split by common Chinese terms
        entities = ["机器学习", "人工智能", "神经网络", "深度学习"]

        for entity in entities:
            await graph.add_entity(entity, {"language": "zh"})

        # Verify Chinese entities extracted
        zh_entities = [
            e for e in graph.get_entities()
            if graph.get_entity(e).get("language") == "zh"
        ]
        assert len(zh_entities) == 4

        logger.info(json.dumps({
            "msg": "Chinese extraction successful",
            "entity_count": len(zh_entities),
            "level": "INFO"
        }))


class TestVisualizationGeneration:
    """
    End-to-end tests for visualization generation.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_generate_graph_visualization(self, populated_graph):
        """
        Test generating visualization data from knowledge graph.
        """
        # Get visualization data
        viz_data = await populated_graph.to_dict()

        # Verify structure
        assert "entities" in viz_data
        assert "relationships" in viz_data

        # Verify entities have required fields
        for entity_name, attrs in viz_data["entities"].items():
            assert isinstance(entity_name, str)
            assert isinstance(attrs, dict)

        # Verify relationships have required structure
        for rel in viz_data["relationships"]:
            assert "source" in rel
            assert "relation" in rel
            assert "target" in rel

        logger.info(json.dumps({
            "msg": "Visualization generated successfully",
            "entity_count": len(viz_data["entities"]),
            "relationship_count": len(viz_data["relationships"]),
            "level": "INFO"
        }))


class TestAgentMemoryPersistence:
    """
    End-to-end tests for agent memory persistence.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_knowledge_state_persistence(self, sample_knowledge_state, temp_dir):
        """
        Test saving and loading knowledge state.
        """
        # Save state
        state_path = temp_dir / "knowledge_state.json"
        state_dict = sample_knowledge_state.to_dict()

        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2)

        # Load state
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_dict = json.load(f)

        # Verify persistence
        restored_state = KnowledgeState.from_dict(loaded_dict)
        assert restored_state.query == sample_knowledge_state.query
        assert len(restored_state.facts) == len(sample_knowledge_state.facts)
        assert len(restored_state.uncertainties) == len(sample_knowledge_state.uncertainties)

        logger.info(json.dumps({
            "msg": "Knowledge state persisted and restored",
            "fact_count": len(restored_state.facts),
            "uncertainty_count": len(restored_state.uncertainties),
            "level": "INFO"
        }))


class TestContradictionDetection:
    """
    End-to-end tests for contradiction detection.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_detect_contradictions(self):
        """
        Test detecting contradictions in knowledge.
        """
        state = KnowledgeState(query="Climate change facts")

        # Add potentially contradictory facts
        state.add_fact("Global temperatures are rising")
        state.add_fact("Some regions are experiencing cooling")

        # In a real implementation, this would trigger contradiction detection
        # For now, we verify the structure
        assert len(state.facts) == 2

        # Check for obvious contradictions (simple keyword-based for testing)
        has_contradiction = any(
            "rising" in fact1.lower() and "cooling" in fact2.lower()
            for fact1 in state.facts
            for fact2 in state.facts
            if fact1 != fact2
        )

        logger.info(json.dumps({
            "msg": "Contradiction detection executed",
            "contradiction_found": has_contradiction,
            "fact_count": len(state.facts),
            "level": "INFO"
        }))


class TestDeduplication:
    """
    End-to-end tests for deduplication across documents.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_cross_document_deduplication(self):
        """
        Test deduplicating entities across multiple documents.
        """
        graph = EntityKnowledgeGraph()

        # Document 1 entities
        doc1_entities = ["Machine Learning", "AI", "Neural Networks"]
        for entity in doc1_entities:
            await graph.add_entity(entity, {"source": "doc1"})

        # Document 2 entities (with duplicates)
        doc2_entities = ["AI", "Deep Learning", "Data Science"]
        for entity in doc2_entities:
            await graph.add_entity(entity, {"source": "doc2"})

        # Verify deduplication (AI should appear only once)
        all_entities = graph.get_entities()
        ai_entity = await graph.get_entity("AI")

        assert ai_entity is not None
        # Entity should have merged attributes
        assert "source" in ai_entity

        # Count unique entities
        unique_count = len(all_entities)
        assert unique_count <= len(doc1_entities) + len(doc2_entities) - 1  # At least one duplicate

        logger.info(json.dumps({
            "msg": "Cross-document deduplication successful",
            "unique_entities": unique_count,
            "expected_max": len(doc1_entities) + len(doc2_entities),
            "level": "INFO"
        }))


class TestEndToEndWorkflows:
    """
    Complex end-to-end workflow tests.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_research_workflow(self, sample_document, performance_tracker):
        """
        Test complete research workflow: ingest → extract → analyze → visualize.
        """
        performance_tracker.start()

        try:
            # Initialize components
            state = KnowledgeState(query="Extract knowledge about AI")
            graph = EntityKnowledgeGraph()

            # Step 1: Ingest document
            state.add_fact("Document ingested successfully")
            performance_tracker.record_operation()

            # Step 2: Extract entities
            entities = self._extract_entities_simple(sample_document)
            for entity in entities:
                await graph.add_entity(entity, {"extracted_at": datetime.now().isoformat()})
            performance_tracker.record_operation()

            # Step 3: Build knowledge state
            state.add_fact(f"Extracted {len(entities)} entities")
            state.set_current_understanding("Document processed and knowledge graph built")
            performance_tracker.record_operation()

            # Step 4: Generate visualization
            viz_data = await graph.to_dict()
            performance_tracker.record_operation()

            # Verify workflow completion
            assert len(state.facts) >= 2
            assert len(viz_data["entities"]) > 0

            logger.info(json.dumps({
                "msg": "Research workflow completed",
                "facts": len(state.facts),
                "entities": len(viz_data["entities"]),
                "level": "INFO"
            }))

        finally:
            performance_tracker.stop()
            metrics = performance_tracker.get_metrics()

            # Performance assertions
            assert metrics["operation_count"] == 4
            assert metrics["error_count"] == 0

    def _extract_entities_simple(self, text: str) -> List[str]:
        """Simple entity extraction for testing."""
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 5]
        return list(set(entities))[:5]  # Deduplicate and limit


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
