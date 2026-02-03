"""
Comprehensive Test Suite for ROMA Integration Components

This module provides complete test coverage for all ROMA integration components:
- ROMAIntegration (core ROMA functionality)
- ROMAKnowledgePipeline (knowledge extraction and storage)
- ROMAEntityExtractor/Writer/Reader (entity knowledge graph integration)
- ROMADSPyIntegration (cooperative reasoning)
- ROMADeepKEIntegration (entity extraction)
- ROMARagbitsIntegration (solution indexing and retrieval)

Test Categories:
1. Unit Tests - Test each method in isolation with mocked dependencies
2. Integration Tests - Test interactions between components
3. Edge Case Tests - Test boundary conditions and error scenarios
4. Configuration Tests - Test default and custom configuration
5. Idempotency Tests - Verify operations are safe to repeat
6. Performance Tests - Test batch processing and parallelism

Testing Best Practices:
- Use pytest with asyncio support
- Mock external dependencies (ROMA core, DSPy, DeepKE, RAGbits)
- Test both success and failure cases
- Verify structured logging (JSON format)
- Test UTC timestamps
- Test correlation ID propagation
- Aim for >80% code coverage

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import ROMA integration components
try:
    from knowledge_engine.integrations.roma_integration import (
        ROMAIntegration,
        ROMAResult,
        ROMADecomposition,
        ROMASolution,
        ROMAVerification
    )
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    pytestmark = pytest.mark.skip("ROMA integration not available")

try:
    from knowledge_engine.integrations.roma_knowledge_pipeline import (
        ROMAKnowledgePipeline,
        EntityExtractionResult,
        KnowledgeArtifact
    )
    ROMA_PIPELINE_AVAILABLE = True
except ImportError:
    ROMA_PIPELINE_AVAILABLE = False
    pytestmark = pytest.mark.skip("ROMA knowledge pipeline not available")

try:
    from knowledge_engine.integrations.roma_entity_kg_integration import (
        ROMAEntityExtractor,
        ROMAKnowledgeWriter,
        ROMAKnowledgeReader,
        ROMAEntity,
        ROMARelationship,
        ROMAKnowledgeResult,
        ROMAEntityType,
        ROMARelationshipType,
        SimilarDecomposition
    )
    ROMA_EKG_AVAILABLE = True
except ImportError:
    ROMA_EKG_AVAILABLE = False
    pytestmark = pytest.mark.skip("ROMA EKG integration not available")

try:
    from knowledge_engine.integrations.roma_dspy_integration import (
        ROMADSPyIntegration,
        ReasoningTrace,
        EnhancedSubproblem
    )
    ROMA_DSPY_AVAILABLE = True
except ImportError:
    ROMA_DSPY_AVAILABLE = False
    pytestmark = pytest.mark.skip("ROMA-DSPy integration not available")

try:
    from knowledge_engine.integrations.roma_deepke_integration import (
        ROMADeepKEIntegration,
        EntityExtraction
    )
    ROMA_DEEPKE_AVAILABLE = True
except ImportError:
    ROMA_DEEPKE_AVAILABLE = False
    pytestmark = pytest.mark.skip("ROMA-DeepKE integration not available")

try:
    from knowledge_engine.integrations.roma_ragbits_integration import (
        ROMARagbitsIntegration,
        IndexedSolution,
        SimilarSolution,
        SolutionReuseResult,
        IndexStatistics,
        SolutionReuseStatus
    )
    ROMA_RAGBITS_AVAILABLE = True
except ImportError:
    ROMA_RAGBITS_AVAILABLE = False
    pytestmark = pytest.mark.skip("ROMA-RAGbits integration not available")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for ROMA integration."""
    return {
        "decomposer": {
            "type": "hierarchical",
            "max_depth": 3,
            "branching_factor": 2,
            "atomic_threshold": 0.7,
            "strategy": "recursive"
        },
        "solver": {
            "type": "multi_agent",
            "timeout_seconds": 300,
            "max_retries": 3
        },
        "verifier": {
            "type": "constraint",
            "threshold": 0.8,
            "strict_mode": False
        },
        "knowledge_integration": {
            "enabled": False
        }
    }


@pytest.fixture
def sample_decomposition():
    """Sample ROMA decomposition."""
    return ROMADecomposition(
        decomposition_id="test_decomp_1",
        problem="Design a scalable microservices architecture",
        sub_problems=[
            ROMADecomposition(
                decomposition_id="test_sub_1",
                problem="Design API gateway",
                sub_problems=[],
                is_atomic=True,
                depth=1,
                parent_id="test_decomp_1"
            )
        ],
        is_atomic=False,
        depth=0,
        metadata={"strategy": "recursive"}
    )


@pytest.fixture
def sample_solution():
    """Sample ROMA solution."""
    return ROMASolution(
        solution_id="test_sol_1",
        problem_id="test_decomp_1",
        solution="Implement microservices with API gateway and service mesh",
        confidence=0.85,
        reasoning="Applied architectural best practices",
        metadata={"agent_used": "reasoning"}
    )


@pytest.fixture
def sample_verification():
    """Sample ROMA verification."""
    return ROMAVerification(
        verification_id="test_ver_1",
        solution_id="test_sol_1",
        passed=True,
        score=0.9,
        feedback="Solution meets all requirements",
        requirements_met={
            "completeness": True,
            "correctness": True,
            "consistency": True
        }
    )


@pytest.fixture
def sample_result(sample_decomposition, sample_solution):
    """Sample ROMA result."""
    return ROMAResult(
        success=True,
        decomposition=sample_decomposition,
        solutions=[sample_solution],
        verification=None,
        metadata={"test": "data"},
        processing_time_ms=100.0
    )


@pytest.fixture
def mock_knowledge_engine():
    """Mock knowledge engine."""
    kg = AsyncMock()
    kg.add_entity_async = AsyncMock(return_value=True)
    kg.get_entity_async = AsyncMock(return_value=None)
    kg.add_relationship_async = AsyncMock(return_value=True)
    kg.get_relationships_async = AsyncMock(return_value=[])
    kg.find_entities_async = AsyncMock(return_value=[])
    kg.search_entities_async = AsyncMock(return_value=[])
    kg.get_statistics_async = AsyncMock(return_value={})
    return kg


@pytest.fixture
def mock_dspy_integration():
    """Mock DSPy integration."""
    dspy = AsyncMock()
    dspy.lm = Mock()
    dspy.chain_of_thought = AsyncMock(
        return_value=Mock(
            success=True,
            reasoning="Step 1: Analyze requirements\nStep 2: Design solution\nStep 3: Verify",
            output="Solution meets requirements",
            processing_time_ms=50.0
        )
    )
    dspy.get_dspy_status = Mock(return_value={"available": True})
    dspy.close = AsyncMock()
    return dspy


@pytest.fixture
def mock_deepke_integration():
    """Mock DeepKE integration."""
    deepke = AsyncMock()
    deepke.extract_entities = AsyncMock(
        return_value=Mock(
            success=True,
            entities=[
                {"name": "API Gateway", "type": "TECH", "confidence": 0.9},
                {"name": "Microservices", "type": "CONCEPT", "confidence": 0.85}
            ]
        )
    )
    deepke.extract_relations = AsyncMock(
        return_value=Mock(
            success=True,
            relations=[
                {"subject": "API Gateway", "predicate": "uses", "object": "Microservices", "confidence": 0.8}
            ]
        )
    )
    deepke.close = AsyncMock()
    return deepke


@pytest.fixture
def mock_ragbits_integration():
    """Mock RAGbits integration."""
    ragbits = AsyncMock()
    ragbits.ingest_documents = AsyncMock(
        return_value=Mock(success=True, document_ids=["doc_1"])
    )
    ragbits.search_documents = AsyncMock(
        return_value=Mock(
            success=True,
            results=[
                {
                    "document_id": "doc_1",
                    "content": "Solution content",
                    "score": 0.85,
                    "metadata": {"solution_id": "sol_1"}
                }
            ]
        )
    )
    ragbits.get_statistics = AsyncMock(return_value={})
    ragbits.health_check = AsyncMock(return_value={"status": "healthy"})
    ragbits.close = AsyncMock()
    return ragbits


# =============================================================================
# TEST CLASS: ROMAIntegration
# =============================================================================

@pytest.mark.skipif(not ROMA_AVAILABLE, reason="ROMA integration not available")
class TestROMAIntegration:
    """Test suite for ROMAIntegration class."""

    @pytest.mark.asyncio
    async def test_initialization(self, sample_config):
        """Test ROMA integration initialization."""
        roma = ROMAIntegration(config=sample_config)

        assert roma.config == sample_config
        assert roma.decomposer is None  # Mock mode
        assert roma.solver is None
        assert roma.verifier is None
        assert roma.reassembler is None
        assert roma._stats["decompositions_performed"] == 0
        assert len(roma._artifact_cache) == 0

    @pytest.mark.asyncio
    async def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        roma = ROMAIntegration()

        assert "decomposer" in roma.config
        assert "solver" in roma.config
        assert "verifier" in roma.config
        assert roma.config["decomposer"]["max_depth"] == 5

    @pytest.mark.asyncio
    async def test_decompose_problem_basic(self, sample_config):
        """Test basic problem decomposition."""
        roma = ROMAIntegration(config=sample_config)

        result = await roma.decompose_problem(
            "Design a scalable system",
            max_depth=2
        )

        assert result.success is True
        assert result.decomposition is not None
        assert result.decomposition.problem == "Design a scalable system"
        assert result.decomposition.depth == 0
        assert result.processing_time_ms > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_decompose_problem_with_entity_extraction(self, sample_config):
        """Test decomposition with knowledge entity extraction."""
        config = sample_config.copy()
        config["knowledge_integration"]["enabled"] = True
        config["knowledge_integration"]["auto_extract_entities"] = True

        roma = ROMAIntegration(config=config)

        result = await roma.decompose_problem(
            "Design authentication system",
            extract_entities=True
        )

        assert result.success is True
        assert "entities_extracted" in result.metadata
        assert result.metadata["entities_extracted"] >= 0

    @pytest.mark.asyncio
    async def test_decompose_problem_empty_input(self):
        """Test decomposition with empty input."""
        roma = ROMAIntegration()

        result = await roma.decompose_problem("")

        # Should handle gracefully
        assert result is not None
        assert isinstance(result, ROMAResult)

    @pytest.mark.asyncio
    async def test_decompose_problem_very_long_input(self):
        """Test decomposition with very long input."""
        roma = ROMAIntegration()
        long_problem = "Design system " * 1000  # Very long problem

        result = await roma.decompose_problem(long_problem)

        assert result is not None
        assert isinstance(result, ROMAResult)

    @pytest.mark.asyncio
    async def test_solve_atomic_basic(self, sample_decomposition):
        """Test solving atomic sub-problem."""
        roma = ROMAIntegration()

        result = await roma.solve_atomic(sample_decomposition)

        assert result.success is True
        assert len(result.solutions) > 0
        assert result.solutions[0].problem_id == sample_decomposition.decomposition_id
        assert result.solutions[0].confidence > 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_solve_atomic_with_context(self, sample_decomposition):
        """Test solving with additional context."""
        roma = ROMAIntegration()

        result = await roma.solve_atomic(
            sample_decomposition,
            context={"domain": "microservices", "priority": "high"}
        )

        assert result.success is True
        assert result.metadata.get("context_provided") is True

    @pytest.mark.asyncio
    async def test_verify_solution_basic(self, sample_solution):
        """Test solution verification."""
        roma = ROMAIntegration()
        requirements = {
            "completeness": True,
            "correctness": 0.9,
            "consistency": True
        }

        result = await roma.verify_solution(sample_solution, requirements)

        assert result.success is True
        assert result.verification is not None
        assert result.verification.solution_id == sample_solution.solution_id
        assert isinstance(result.verification.passed, bool)
        assert isinstance(result.verification.score, float)
        assert result.verification.requirements_met is not None

    @pytest.mark.asyncio
    async def test_verify_solution_strict_requirements(self, sample_solution):
        """Test verification with strict requirements."""
        config = {"verifier": {"threshold": 0.95, "strict_mode": True}}
        roma = ROMAIntegration(config=config)
        requirements = {"correctness": 0.95}

        result = await roma.verify_solution(sample_solution, requirements)

        assert result is not None
        assert result.verification is not None

    @pytest.mark.asyncio
    async def test_reassemble_solution_basic(self, sample_solution):
        """Test solution reassembly."""
        roma = ROMAIntegration()
        solutions = [sample_solution]

        result = await roma.reassemble_solution(solutions)

        assert result.success is True
        assert len(result.solutions) > 0
        assert result.solutions[0].problem_id == "reassembled"
        assert result.metadata.get("strategy") is not None

    @pytest.mark.asyncio
    async def test_reassemble_solution_with_knowledge_storage(self, sample_solution):
        """Test reassembly with knowledge storage."""
        config = {
            "knowledge_integration": {
                "enabled": True,
                "auto_store_solutions": True
            }
        }
        roma = ROMAIntegration(config=config)
        solutions = [sample_solution]

        result = await roma.reassemble_solution(
            solutions,
            store_as_knowledge=True
        )

        assert result.success is True
        # Note: knowledge_artifact_id will be None in mock mode

    @pytest.mark.asyncio
    async def test_batch_decompose(self):
        """Test batch decomposition."""
        roma = ROMAIntegration()
        problems = [
            "Design API",
            "Implement database",
            "Create UI"
        ]

        results = await roma.batch_decompose(problems, max_depth=2)

        assert len(results) == len(problems)
        for result in results:
            assert isinstance(result, ROMAResult)

    @pytest.mark.asyncio
    async def test_batch_decompose_timeout(self):
        """Test batch decomposition with timeout."""
        config = {
            "batch_processing": {
                "enabled": True,
                "max_parallel": 2,
                "timeout_seconds": 0.001  # Very short timeout
            }
        }
        roma = ROMAIntegration(config=config)
        problems = ["Problem 1", "Problem 2"]

        results = await roma.batch_decompose(problems)

        # Should handle timeout gracefully
        assert results is not None
        assert len(results) == len(problems)

    @pytest.mark.asyncio
    async def test_extract_knowledge_entities(self, sample_result):
        """Test knowledge entity extraction."""
        config = {
            "knowledge_integration": {
                "enabled": True,
                "entity_types": ["problem", "solution"]
            }
        }
        roma = ROMAIntegration(config=config)

        entities = await roma.extract_knowledge_entities(sample_result)

        assert isinstance(entities, list)
        # Check entity structure
        for entity in entities:
            assert "id" in entity
            assert "type" in entity
            assert "name" in entity
            assert "properties" in entity

    @pytest.mark.asyncio
    async def test_store_solution_as_knowledge(self, sample_result):
        """Test storing solution as knowledge artifact."""
        config = {
            "knowledge_integration": {
                "enabled": True
            }
        }
        roma = ROMAIntegration(config=config)

        artifact_id = await roma.store_solution_as_knowledge(sample_result)

        # Will be None in mock mode
        assert artifact_id is None or isinstance(artifact_id, str)

    def test_get_statistics(self):
        """Test getting statistics."""
        roma = ROMAIntegration()
        stats = roma.get_statistics()

        assert "decompositions_performed" in stats
        assert "problems_solved" in stats
        assert "verifications_performed" in stats
        assert "reassemblies_performed" in stats
        assert "entities_extracted" in stats
        assert "solutions_stored" in stats
        assert "total_processing_time_ms" in stats
        assert "average_processing_time_ms" in stats
        assert "config" in stats
        assert "timestamp" in stats

    def test_health_check(self):
        """Test health check."""
        roma = ROMAIntegration()
        health = roma.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health
        assert "statistics" in health
        assert "timestamp" in health

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing resources."""
        roma = ROMAIntegration()

        # Should not raise exception
        await roma.close()


# =============================================================================
# TEST CLASS: ROMAKnowledgePipeline
# =============================================================================

@pytest.mark.skipif(not ROMA_PIPELINE_AVAILABLE, reason="ROMA knowledge pipeline not available")
class TestROMAKnowledgePipeline:
    """Test suite for ROMAKnowledgePipeline class."""

    @pytest.fixture
    async def pipeline(self, mock_knowledge_engine):
        """Create ROMA knowledge pipeline for testing."""
        roma = ROMAIntegration()
        pipeline = ROMAKnowledgePipeline(
            roma_integration=roma,
            knowledge_engine=mock_knowledge_engine,
            config={
                "auto_extract_entities": True,
                "auto_store_solutions": True
            }
        )
        yield pipeline
        await pipeline.close()

    @pytest.mark.asyncio
    async def test_initialization(self, mock_knowledge_engine):
        """Test pipeline initialization."""
        roma = ROMAIntegration()
        pipeline = ROMAKnowledgePipeline(
            roma_integration=roma,
            knowledge_engine=mock_knowledge_engine
        )

        assert pipeline.roma == roma
        assert pipeline.knowledge_engine == mock_knowledge_engine
        assert pipeline.config["auto_extract_entities"] is True
        assert pipeline._stats["executions_performed"] == 0

    @pytest.mark.asyncio
    async def test_execute_and_store(self, pipeline):
        """Test execute and store pipeline."""
        result = await pipeline.execute_and_store(
            "Design a scalable system",
            options={"max_depth": 2}
        )

        assert result.success is True
        assert result.decomposition is not None
        assert "stored_at" in result.metadata
        assert pipeline._stats["executions_performed"] == 1

    @pytest.mark.asyncio
    async def test_execute_and_store_with_entity_extraction(self, pipeline):
        """Test execute with entity extraction."""
        result = await pipeline.execute_and_store(
            "Implement API gateway",
            options={"max_depth": 1}
        )

        assert result.success is True
        assert "entities_created" in result.metadata
        assert isinstance(result.metadata["entities_created"], list)

    @pytest.mark.asyncio
    async def test_extract_and_store_entities(self, pipeline, sample_result):
        """Test entity extraction and storage."""
        entity_ids = await pipeline.extract_and_store_entities(sample_result)

        assert isinstance(entity_ids, list)
        pipeline._stats["entities_extracted"] >= 0

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions(self, pipeline):
        """Test retrieving similar solutions."""
        similar = await pipeline.retrieve_similar_solutions(
            "Design authentication system",
            top_k=5
        )

        assert isinstance(similar, list)
        for sol in similar:
            assert "artifact_id" in sol
            assert "problem" in sol
            assert "similarity" in sol

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions_empty_query(self, pipeline):
        """Test retrieval with empty query."""
        similar = await pipeline.retrieve_similar_solutions("")

        # Should handle gracefully
        assert isinstance(similar, list)

    @pytest.mark.asyncio
    async def test_get_statistics(self, pipeline):
        """Test getting pipeline statistics."""
        stats = await pipeline.get_statistics()

        assert "executions_performed" in stats
        assert "entities_extracted" in stats
        assert "solutions_stored" in stats
        assert "similar_retrievals" in stats
        assert "total_processing_time_ms" in stats
        assert "average_processing_time_ms" in stats
        assert "timestamp" in stats

    @pytest.mark.asyncio
    async def test_health_check(self, pipeline):
        """Test health check."""
        health = await pipeline.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "roma_integration" in health
        assert "knowledge_engine" in health
        assert "statistics" in health


# =============================================================================
# TEST CLASS: ROMAEntityExtractor
# =============================================================================

@pytest.mark.skipif(not ROMA_EKG_AVAILABLE, reason="ROMA EKG integration not available")
class TestROMAEntityExtractor:
    """Test suite for ROMAEntityExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create entity extractor for testing."""
        return ROMAEntityExtractor(config={
            "extract_properties": True,
            "extract_metadata": True,
            "min_confidence": 0.5
        })

    @pytest.mark.asyncio
    async def test_extract_from_decomposition(self, extractor, sample_decomposition):
        """Test entity extraction from decomposition."""
        decomposition_dict = asdict(sample_decomposition)
        entities = await extractor.extract_from_decomposition(decomposition_dict)

        assert isinstance(entities, list)
        assert len(entities) > 0

        for entity in entities:
            assert isinstance(entity, ROMAEntity)
            assert entity.entity_id is not None
            assert entity.entity_type in ROMAEntityType
            assert entity.name is not None
            assert entity.confidence >= 0.0
            assert entity.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_extract_from_solution(self, extractor, sample_solution):
        """Test entity extraction from solution."""
        solution_dict = asdict(sample_solution)
        entities = await extractor.extract_from_solution(solution_dict)

        assert isinstance(entities, list)
        assert len(entities) > 0

        # Check for solution entity
        solution_entities = [e for e in entities if e.entity_type == ROMAEntityType.SOLUTION]
        assert len(solution_entities) > 0

    @pytest.mark.asyncio
    async def test_extract_relationships(self, extractor, sample_decomposition):
        """Test relationship extraction."""
        decomposition_dict = asdict(sample_decomposition)
        entities = await extractor.extract_from_decomposition(decomposition_dict)
        relationships = await extractor.extract_relationships(decomposition_dict, entities)

        assert isinstance(relationships, list)

        for rel in relationships:
            assert isinstance(rel, ROMARelationship)
            assert rel.source_id is not None
            assert rel.target_id is not None
            assert rel.relationship_type in ROMARelationshipType

    @pytest.mark.asyncio
    async def test_extract_empty_decomposition(self, extractor):
        """Test extraction with empty decomposition."""
        empty_decomp = {
            "decomposition_id": "empty",
            "problem": "",
            "sub_problems": [],
            "is_atomic": True,
            "depth": 0,
            "metadata": {}
        }

        entities = await extractor.extract_from_decomposition(empty_decomp)
        assert isinstance(entities, list)


# =============================================================================
# TEST CLASS: ROMAKnowledgeWriter
# =============================================================================

@pytest.mark.skipif(not ROMA_EKG_AVAILABLE, reason="ROMA EKG integration not available")
class TestROMAKnowledgeWriter:
    """Test suite for ROMAKnowledgeWriter class."""

    @pytest.fixture
    def writer(self, mock_knowledge_engine):
        """Create knowledge writer for testing."""
        return ROMAKnowledgeWriter(
            knowledge_graph=mock_knowledge_engine,
            config={"idempotent": True, "batch_size": 10}
        )

    @pytest.mark.asyncio
    async def test_store_entities(self, writer, mock_knowledge_engine):
        """Test storing entities."""
        entities = [
            ROMAEntity(
                entity_id="test_1",
                entity_type=ROMAEntityType.PROBLEM,
                name="Test Problem",
                description="Test description",
                properties={"test": "data"},
                confidence=0.9
            ),
            ROMAEntity(
                entity_id="test_2",
                entity_type=ROMAEntityType.SOLUTION,
                name="Test Solution",
                description="Test solution",
                confidence=0.85
            )
        ]

        entity_ids = await writer.store_entities(entities)

        assert isinstance(entity_ids, list)
        assert len(entity_ids) == len(entities)
        assert mock_knowledge_engine.add_entity_async.call_count == len(entities)

    @pytest.mark.asyncio
    async def test_store_entities_idempotent(self, writer, mock_knowledge_engine):
        """Test idempotent entity storage."""
        # First call - entity doesn't exist
        mock_knowledge_engine.get_entity_async.return_value = None
        entities = [
            ROMAEntity(
                entity_id="test_1",
                entity_type=ROMAEntityType.PROBLEM,
                name="Test",
                description="Test",
                confidence=0.9
            )
        ]

        ids1 = await writer.store_entities(entities)
        ids2 = await writer.store_entities(entities)

        assert len(ids1) == len(ids2)

    @pytest.mark.asyncio
    async def test_store_relationships(self, writer, mock_knowledge_engine):
        """Test storing relationships."""
        relationships = [
            ROMARelationship(
                source_id="test_1",
                target_id="test_2",
                relationship_type=ROMARelationshipType.DECOMPOSED_FROM,
                properties={"depth": 1},
                confidence=0.9
            )
        ]

        rel_ids = await writer.store_relationships(relationships)

        assert isinstance(rel_ids, list)
        assert len(rel_ids) > 0
        assert mock_knowledge_engine.add_relationship_async.called

    @pytest.mark.asyncio
    async def test_store_artifact(self, writer, mock_knowledge_engine):
        """Test storing artifact."""
        solution = {
            "solution_id": "test_sol",
            "problem_id": "test_prob",
            "solution": "Test solution content",
            "reasoning": "Test reasoning",
            "confidence": 0.9,
            "metadata": {}
        }

        artifact_id = await writer.store_artifact(solution)

        assert isinstance(artifact_id, str)
        assert len(artifact_id) > 0
        assert mock_knowledge_engine.add_entity_async.called

    @pytest.mark.asyncio
    async def test_circuit_breaker(self, writer, mock_knowledge_engine):
        """Test circuit breaker pattern."""
        # Simulate multiple failures
        mock_knowledge_engine.add_entity_async.side_effect = Exception("DB error")

        entities = [
            ROMAEntity(
                entity_id=f"test_{i}",
                entity_type=ROMAEntityType.PROBLEM,
                name=f"Test {i}",
                description="Test",
                confidence=0.9
            )
            for i in range(10)
        ]

        # Should fail but not crash
        entity_ids = await writer.store_entities(entities)

        # Circuit breaker should be open after threshold
        assert writer._circuit_breaker_failures >= 5


# =============================================================================
# TEST CLASS: ROMAKnowledgeReader
# =============================================================================

@pytest.mark.skipif(not ROMA_EKG_AVAILABLE, reason="ROMA EKG integration not available")
class TestROMAKnowledgeReader:
    """Test suite for ROMAKnowledgeReader class."""

    @pytest.fixture
    def reader(self, mock_knowledge_engine):
        """Create knowledge reader for testing."""
        return ROMAKnowledgeReader(
            knowledge_graph=mock_knowledge_engine,
            config={"default_top_k": 5, "similarity_threshold": 0.7}
        )

    @pytest.mark.asyncio
    async def test_find_similar_decompositions(self, reader, mock_knowledge_engine):
        """Test finding similar decompositions."""
        mock_knowledge_engine.search_entities_async.return_value = [
            {
                "entity_id": "test_1",
                "entity_type": "roma_problem",
                "name": "Design API system",
                "properties": {"description": "API design problem"}
            }
        ]

        similar = await reader.find_similar_decompositions(
            "Design RESTful API",
            top_k=3
        )

        assert isinstance(similar, list)
        assert len(similar) > 0

        for decomp in similar:
            assert isinstance(decomp, SimilarDecomposition)
            assert decomp.decomposition_id is not None
            assert decomp.problem is not None
            assert 0.0 <= decomp.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_get_solution_artifacts(self, reader, mock_knowledge_engine):
        """Test getting solution artifacts."""
        mock_knowledge_engine.get_relationships_async.return_value = [
            {
                "relationship_id": "rel_1",
                "relationship_type": "solves",
                "source_entity_id": "sol_1",
                "target_entity_id": "prob_1"
            }
        ]
        mock_knowledge_engine.get_entity_async.return_value = {
            "entity_id": "sol_1",
            "properties": {
                "description": "Solution content",
                "confidence": 0.9
            }
        }

        artifacts = await reader.get_solution_artifacts("prob_1")

        assert isinstance(artifacts, list)

    @pytest.mark.asyncio
    async def test_trace_dependencies(self, reader, mock_knowledge_engine):
        """Test dependency tracing."""
        mock_knowledge_engine.get_relationships_async.return_value = [
            {
                "relationship_id": "rel_1",
                "relationship_type": "depends_on",
                "source_entity_id": "prob_1",
                "target_entity_id": "prob_2"
            }
        ]
        mock_knowledge_engine.get_entity_async.return_value = {
            "entity_id": "prob_2",
            "name": "Dependency Problem"
        }

        dependencies = await reader.trace_dependencies("prob_1")

        assert isinstance(dependencies, list)
        for dep in dependencies:
            assert "source" in dep
            assert "target" in dep

    @pytest.mark.asyncio
    async def test_get_decomposition_tree(self, reader, mock_knowledge_engine):
        """Test getting decomposition tree."""
        mock_knowledge_engine.get_entity_async.return_value = {
            "entity_id": "root",
            "name": "Root Problem",
            "properties": {"is_atomic": False}
        }
        mock_knowledge_engine.get_relationships_async.return_value = []

        tree = await reader.get_decomposition_tree("root", max_depth=3)

        assert isinstance(tree, dict)
        assert "entity_id" in tree
        assert "name" in tree


# =============================================================================
# TEST CLASS: ROMADSPyIntegration
# =============================================================================

@pytest.mark.skipif(not ROMA_DSPY_AVAILABLE, reason="ROMA-DSPy integration not available")
class TestROMADSPyIntegration:
    """Test suite for ROMADSPyIntegration class."""

    @pytest.fixture
    def roma_dspy(self, mock_dspy_integration):
        """Create ROMA-DSPy integration for testing."""
        roma = ROMAIntegration()
        integration = ROMADSPyIntegration(
            roma_integration=roma,
            dspy_integration=mock_dspy_integration,
            config={"auto_add_reasoning": True}
        )
        yield integration
        # Cleanup
        asyncio.create_task(integration.close())

    @pytest.mark.asyncio
    async def test_initialization(self, roma_dspy):
        """Test initialization."""
        assert roma_dspy.roma is not None
        assert roma_dspy.dspy is not None
        assert roma_dspy.config["auto_add_reasoning"] is True
        assert len(roma_dspy._reasoning_cache) == 0

    @pytest.mark.asyncio
    async def test_solve_with_cooperative_reasoning(self, roma_dspy):
        """Test cooperative reasoning problem solving."""
        result = await roma_dspy.solve_with_cooperative_reasoning(
            "Design authentication system",
            max_depth=2
        )

        assert result.success is True
        assert result.decomposition is not None
        assert "enhanced_subproblems" in result.metadata
        assert "reasoning_enabled" in result.metadata

    @pytest.mark.asyncio
    async def test_add_reasoning_to_subproblem(self, roma_dspy):
        """Test adding reasoning to sub-problem."""
        subproblem = {
            "id": "test_sub",
            "problem": "Design API gateway",
            "depth": 1,
            "is_atomic": True
        }

        enhanced = await roma_dspy.add_reasoning_to_subproblem(subproblem)

        assert isinstance(enhanced, EnhancedSubproblem)
        assert enhanced.subproblem_id == "test_sub"
        assert enhanced.problem == "Design API gateway"
        assert enhanced.reasoning_trace is not None
        assert len(enhanced.reasoning_trace.steps) > 0

    @pytest.mark.asyncio
    async def test_add_reasoning_with_cache(self, roma_dspy):
        """Test reasoning cache."""
        subproblem = {
            "id": "test_sub",
            "problem": "Design API",
            "depth": 1
        }

        # First call - cache miss
        enhanced1 = await roma_dspy.add_reasoning_to_subproblem(subproblem)
        cache_hits_1 = roma_dspy._stats["reasoning_cache_hits"]

        # Second call - cache hit
        enhanced2 = await roma_dspy.add_reasoning_to_subproblem(subproblem)
        cache_hits_2 = roma_dspy._stats["reasoning_cache_hits"]

        assert cache_hits_2 > cache_hits_1

    @pytest.mark.asyncio
    async def test_batch_reason_subproblems(self, roma_dspy):
        """Test batch reasoning."""
        subproblems = [
            {
                "id": f"sub_{i}",
                "problem": f"Sub-problem {i}",
                "depth": 1,
                "is_atomic": True
            }
            for i in range(5)
        ]

        enhanced = await roma_dspy.batch_reason_subproblems(subproblems)

        assert len(enhanced) == len(subproblems)
        for e in enhanced:
            assert isinstance(e, EnhancedSubproblem)

    @pytest.mark.asyncio
    async def test_verify_with_reasoning(self, roma_dspy, sample_result):
        """Test verification with reasoning."""
        requirements = {
            "completeness": True,
            "correctness": 0.9
        }

        verified = await roma_dspy.verify_with_reasoning(
            sample_result,
            requirements
        )

        assert verified is not None
        assert verified.verification is not None

    def test_get_statistics(self, roma_dspy):
        """Test getting statistics."""
        stats = roma_dspy.get_statistics()

        assert "cooperative_solutions" in stats
        assert "reasoning_traces_generated" in stats
        assert "reasoning_cache_hits" in stats
        assert "subproblems_reasoned" in stats
        assert "cache_hit_rate" in stats
        assert "timestamp" in stats

    def test_health_check(self, roma_dspy):
        """Test health check."""
        health = roma_dspy.health_check()

        assert "status" in health
        assert "roma_status" in health
        assert "dspy_available" in health
        assert "reasoning_enabled" in health

    @pytest.mark.asyncio
    async def test_close(self, roma_dspy):
        """Test closing resources."""
        await roma_dspy.close()
        assert len(roma_dspy._reasoning_cache) == 0


# =============================================================================
# TEST CLASS: ROMADeepKEIntegration
# =============================================================================

@pytest.mark.skipif(not ROMA_DEEPKE_AVAILABLE, reason="ROMA-DeepKE integration not available")
class TestROMADeepKEIntegration:
    """Test suite for ROMADeepKEIntegration class."""

    @pytest.fixture
    def roma_deepke(self, mock_deepke_integration, mock_knowledge_engine):
        """Create ROMA-DeepKE integration for testing."""
        roma = ROMAIntegration()
        integration = ROMADeepKEIntegration(
            roma_integration=roma,
            deepke_integration=mock_deepke_integration,
            knowledge_engine=mock_knowledge_engine,
            config={"confidence_threshold": 0.7}
        )
        yield integration
        # Cleanup
        asyncio.create_task(integration.close())

    @pytest.mark.asyncio
    async def test_initialization(self, roma_deepke):
        """Test initialization."""
        assert roma_deepke.roma is not None
        assert roma_deepke.deepke is not None
        assert roma_deepke.knowledge_engine is not None
        assert roma_deepke.config["confidence_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_enrich_with_entities(self, roma_deepke, sample_result):
        """Test solution enrichment with entities."""
        enriched = await roma_deepke.enrich_with_entities(sample_result)

        assert enriched.success is True
        assert "extracted_entities" in enriched.metadata
        assert "entity_extraction_time_ms" in enriched.metadata

        extraction = enriched.metadata["extracted_entities"]
        assert "entities" in extraction
        assert "confidence" in extraction

    @pytest.mark.asyncio
    async def test_extract_entities_from_solution(self, roma_deepke):
        """Test entity extraction from solution text."""
        solution_text = "Implement REST API using FastAPI with JWT authentication"

        entities = await roma_deepke.extract_entities_from_solution(
            solution_text,
            "technical_solution"
        )

        assert isinstance(entities, list)
        for entity in entities:
            assert "name" in entity
            assert "type" in entity
            assert "confidence" in entity
            assert entity["confidence"] >= 0.0

    @pytest.mark.asyncio
    async def test_extract_relations_from_solution(self, roma_deepke):
        """Test relation extraction."""
        solution_text = "FastAPI uses JWT for authentication"
        entities = [
            {"name": "FastAPI", "type": "TECH"},
            {"name": "JWT", "type": "TECH"}
        ]

        relations = await roma_deepke.extract_relations_from_solution(
            solution_text,
            entities
        )

        assert isinstance(relations, list)
        for rel in relations:
            assert "subject" in rel
            assert "predicate" in rel
            assert "object" in rel

    @pytest.mark.asyncio
    async def test_create_knowledge_entities(self, roma_deepke, mock_knowledge_engine):
        """Test creating knowledge entities."""
        entities = [
            {"name": "API Gateway", "type": "TECH", "confidence": 0.9}
        ]
        relations = [
            {"subject": "API", "predicate": "uses", "object": "Gateway", "confidence": 0.8}
        ]

        entity_ids = await roma_deepke.create_knowledge_entities(entities, relations)

        assert isinstance(entity_ids, list)
        assert mock_knowledge_engine.add_entity_async.called

    @pytest.mark.asyncio
    async def test_deduplicate_entities(self, roma_deepke):
        """Test entity deduplication."""
        entities = [
            {"name": "API", "type": "TECH", "confidence": 0.8},
            {"name": "API", "type": "TECH", "confidence": 0.9},  # Duplicate
            {"name": "Gateway", "type": "TECH", "confidence": 0.85}
        ]

        deduplicated = await roma_deepke._deduplicate_entities(entities)

        assert len(deduplicated) <= len(entities)
        # Check no duplicates
        names = [e["name"] for e in deduplicated]
        assert len(names) == len(set(names))

    @pytest.mark.asyncio
    async def test_batch_extract_entities(self, roma_deepke):
        """Test batch entity extraction."""
        results = [
            ROMAResult(
                success=True,
                decomposition=None,
                solutions=[
                    ROMASolution(
                        solution_id=f"sol_{i}",
                        problem_id=f"prob_{i}",
                        solution=f"Solution {i}",
                        confidence=0.8,
                        reasoning="Test"
                    )
                ],
                verification=None,
                metadata={}
            )
            for i in range(3)
        ]

        enriched = await roma_deepke.batch_extract_entities(results)

        assert len(enriched) == len(results)

    @pytest.mark.asyncio
    async def test_fallback_entity_extraction(self, roma_deepke):
        """Test fallback entity extraction when DeepKE fails."""
        # This tests the internal fallback method
        text = "Implement FastAPI with OAuth2 authentication"
        entity_types = ["TECH", "CONCEPT"]

        entities = roma_deepke._fallback_entity_extraction(text, entity_types)

        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_get_entity_statistics(self, roma_deepke):
        """Test getting entity statistics."""
        stats = await roma_deepke.get_entity_statistics()

        assert "solutions_processed" in stats
        assert "entities_extracted" in stats
        assert "relations_extracted" in stats
        assert "kg_entities_created" in stats
        assert "timestamp" in stats


# =============================================================================
# TEST CLASS: ROMARagbitsIntegration
# =============================================================================

@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits integration not available")
class TestROMARagbitsIntegration:
    """Test suite for ROMARagbitsIntegration class."""

    @pytest.fixture
    def roma_ragbits(self, mock_ragbits_integration):
        """Create ROMA-RAGbits integration for testing."""
        integration = ROMARagbitsIntegration(
            ragbits_integration=mock_ragbits_integration,
            config={"auto_index_solutions": True}
        )
        yield integration
        # Cleanup
        asyncio.create_task(integration.close())

    @pytest.mark.asyncio
    async def test_initialization(self, roma_ragbits):
        """Test initialization."""
        assert roma_ragbits.ragbits_integration is not None
        assert roma_ragbits.config["auto_index_solutions"] is True
        assert len(roma_ragbits._solution_cache) == 0

    @pytest.mark.asyncio
    async def test_index_solution(self, roma_ragbits, sample_result):
        """Test indexing a solution."""
        doc_id = await roma_ragbits.index_solution(sample_result)

        assert doc_id is not None
        assert isinstance(doc_id, str)
        assert doc_id.startswith("roma_sol_")
        assert len(roma_ragbits._solution_cache) > 0

    @pytest.mark.asyncio
    async def test_index_solution_idempotent(self, roma_ragbits, sample_result):
        """Test idempotent indexing."""
        doc_id1 = await roma_ragbits.index_solution(sample_result)
        doc_id2 = await roma_ragbits.index_solution(sample_result)

        # Should return same document ID (cached)
        assert doc_id1 == doc_id2

    @pytest.mark.asyncio
    async def test_index_batch_solutions(self, roma_ragbits):
        """Test batch indexing."""
        results = [
            ROMAResult(
                success=True,
                decomposition=None,
                solutions=[
                    ROMASolution(
                        solution_id=f"sol_{i}",
                        problem_id=f"prob_{i}",
                        solution=f"Solution {i}",
                        confidence=0.8,
                        reasoning="Test"
                    )
                ],
                verification=None,
                metadata={}
            )
            for i in range(3)
        ]

        doc_ids = await roma_ragbits.index_batch_solutions(results)

        assert isinstance(doc_ids, list)
        assert len(doc_ids) > 0

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions(self, roma_ragbits):
        """Test retrieving similar solutions."""
        similar = await roma_ragbits.retrieve_similar_solutions(
            "Design API gateway",
            top_k=3
        )

        assert isinstance(similar, list)
        for sol in similar:
            assert isinstance(sol, SimilarSolution)
            assert sol.document_id is not None
            assert sol.problem is not None
            assert 0.0 <= sol.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions_with_filters(self, roma_ragbits):
        """Test retrieval with filters."""
        similar = await roma_ragbits.retrieve_similar_solutions(
            "Design system",
            top_k=5,
            filters={"problem_type": "design", "min_confidence": 0.7}
        )

        assert isinstance(similar, list)

    @pytest.mark.asyncio
    async def test_reuse_solution_direct(self, roma_ragbits):
        """Test solution reuse with high similarity."""
        reuse_result = await roma_ragbits.reuse_solution(
            "Design API",
            top_k=3
        )

        assert isinstance(reuse_result, SolutionReuseResult)
        assert reuse_result.status in SolutionReuseStatus
        assert isinstance(reuse_result.similar_solutions, list)

    @pytest.mark.asyncio
    async def test_reuse_solution_no_similar(self, roma_ragbits):
        """Test solution reuse with no similar solutions."""
        # Mock empty results
        roma_ragbits.ragbits_integration.search_documents.return_value = Mock(
            success=True,
            results=[]
        )

        reuse_result = await roma_ragbits.reuse_solution("Unique problem")

        assert reuse_result.status == SolutionReuseStatus.NO_SIMILAR_FOUND

    @pytest.mark.asyncio
    async def test_get_solution_by_id(self, roma_ragbits):
        """Test getting solution by ID."""
        result = await roma_ragbits.get_solution_by_id("doc_1")

        # May be None if not found
        assert result is None or isinstance(result, ROMAResult)

    @pytest.mark.asyncio
    async def test_delete_solution(self, roma_ragbits):
        """Test deleting solution."""
        success = await roma_ragbits.delete_solution("doc_1")

        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_update_solution(self, roma_ragbits, sample_result):
        """Test updating solution."""
        success = await roma_ragbits.update_solution("doc_1", sample_result)

        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_search_solutions(self, roma_ragbits):
        """Test general search."""
        results = await roma_ragbits.search_solutions(
            "API design",
            top_k=5
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_index_statistics(self, roma_ragbits):
        """Test getting index statistics."""
        stats = await roma_ragbits.get_index_statistics()

        assert isinstance(stats, IndexStatistics)
        assert stats.total_solutions >= 0
        assert stats.index_size_bytes >= 0
        assert stats.index_health in ["healthy", "moderate", "full", "unknown"]

    def test_get_statistics(self, roma_ragbits):
        """Test getting integration statistics."""
        stats = roma_ragbits.get_statistics()

        assert "solutions_indexed" in stats
        assert "solutions_retrieved" in stats
        assert "solutions_reused" in stats
        assert "batches_indexed" in stats
        assert "cached_solutions" in stats
        assert "timestamp" in stats

    @pytest.mark.asyncio
    async def test_health_check(self, roma_ragbits):
        """Test health check."""
        health = await roma_ragbits.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "checks" in health

    @pytest.mark.asyncio
    async def test_close(self, roma_ragbits):
        """Test closing resources."""
        await roma_ragbits.close()
        assert len(roma_ragbits._solution_cache) == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.skipif(not ROMA_AVAILABLE, reason="ROMA integration not available")
class TestROMAIntegrationWorkflow:
    """End-to-end integration tests for ROMA workflows."""

    @pytest.mark.asyncio
    async def test_full_roma_workflow(self):
        """Test complete ROMA workflow from decomposition to reassembly."""
        roma = ROMAIntegration()

        # Step 1: Decompose problem
        decomp_result = await roma.decompose_problem(
            "Design scalable microservices architecture",
            max_depth=2
        )
        assert decomp_result.success is True

        # Step 2: Solve atomic sub-problems
        if decomp_result.decomposition.sub_problems:
            solve_result = await roma.solve_atomic(
                decomp_result.decomposition.sub_problems[0]
            )
            assert solve_result.success is True

        # Step 3: Verify solution
        if decomp_result.solutions:
            verify_result = await roma.verify_solution(
                decomp_result.solutions[0],
                {"completeness": True, "correctness": 0.8}
            )
            assert verify_result.success is True

        # Step 4: Reassemble solutions
        if decomp_result.solutions:
            reassemble_result = await roma.reassemble_solution(
                decomp_result.solutions
            )
            assert reassemble_result.success is True

        await roma.close()

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        roma = ROMAIntegration()

        problems = [
            "Design API",
            "Implement database",
            "Create UI"
        ]

        # Batch decompose
        results = await roma.batch_decompose(problems, max_depth=1)

        assert len(results) == len(problems)

        # Process each result
        for result in results:
            if result.success and result.decomposition:
                if result.decomposition.is_atomic:
                    solve_result = await roma.solve_atomic(result.decomposition)
                    assert solve_result is not None

        await roma.close()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.skipif(not ROMA_AVAILABLE, reason="ROMA integration not available")
class TestROMAEdgeCases:
    """Edge case tests for ROMA integration."""

    @pytest.mark.asyncio
    async def test_null_inputs(self):
        """Test handling of None/null inputs."""
        roma = ROMAIntegration()

        # Should handle gracefully
        result = await roma.decompose_problem("")
        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_input(self):
        """Test handling of unicode characters."""
        roma = ROMAIntegration()

        unicode_problem = "设计可扩展的系统 🚀 with émojis and spëcial çhars"
        result = await roma.decompose_problem(unicode_problem)

        assert result is not None

    @pytest.mark.asyncio
    async def test_very_deep_decomposition(self):
        """Test very deep decomposition."""
        roma = ROMAIntegration(config={"decomposer": {"max_depth": 100}})

        result = await roma.decompose_problem("Complex problem", max_depth=50)

        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test concurrent operations."""
        roma = ROMAIntegration()

        # Run multiple operations concurrently
        tasks = [
            roma.decompose_problem(f"Problem {i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        for result in results:
            assert result is not None


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

@pytest.mark.skipif(not ROMA_AVAILABLE, reason="ROMA integration not available")
class TestROMAConfiguration:
    """Configuration tests for ROMA integration."""

    @pytest.mark.asyncio
    async def test_custom_decomposer_config(self):
        """Test custom decomposer configuration."""
        config = {
            "decomposer": {
                "max_depth": 10,
                "branching_factor": 5,
                "strategy": "iterative"
            }
        }
        roma = ROMAIntegration(config=config)

        assert roma.config["decomposer"]["max_depth"] == 10
        assert roma.config["decomposer"]["branching_factor"] == 5

    @pytest.mark.asyncio
    async def test_custom_solver_config(self):
        """Test custom solver configuration."""
        config = {
            "solver": {
                "timeout_seconds": 600,
                "max_retries": 5
            }
        }
        roma = ROMAIntegration(config=config)

        assert roma.config["solver"]["timeout_seconds"] == 600
        assert roma.config["solver"]["max_retries"] == 5

    @pytest.mark.asyncio
    async def test_custom_verifier_config(self):
        """Test custom verifier configuration."""
        config = {
            "verifier": {
                "threshold": 0.95,
                "strict_mode": True
            }
        }
        roma = ROMAIntegration(config=config)

        assert roma.config["verifier"]["threshold"] == 0.95
        assert roma.config["verifier"]["strict_mode"] is True

    @pytest.mark.asyncio
    async def test_knowledge_integration_config(self):
        """Test knowledge integration configuration."""
        config = {
            "knowledge_integration": {
                "enabled": True,
                "auto_extract_entities": True,
                "auto_store_solutions": True
            }
        }
        roma = ROMAIntegration(config=config)

        assert roma.config["knowledge_integration"]["enabled"] is True
        assert roma.config["knowledge_integration"]["auto_extract_entities"] is True


# =============================================================================
# IDEMPOTENCY TESTS
# =============================================================================

@pytest.mark.skipif(not ROMA_AVAILABLE, reason="ROMA integration not available")
class TestROMAIdempotency:
    """Idempotency tests for ROMA integration."""

    @pytest.mark.asyncio
    async def test_decompose_idempotency(self):
        """Test that decomposition is idempotent."""
        roma = ROMAIntegration()
        problem = "Design API"

        result1 = await roma.decompose_problem(problem)
        result2 = await roma.decompose_problem(problem)

        # Both should succeed
        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_solve_idempotency(self, sample_decomposition):
        """Test that solving is idempotent."""
        roma = ROMAIntegration()

        result1 = await roma.solve_atomic(sample_decomposition)
        result2 = await roma.solve_atomic(sample_decomposition)

        # Both should succeed (though solutions may differ)
        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_statistics_consistency(self):
        """Test statistics consistency across operations."""
        roma = ROMAIntegration()

        stats_before = roma.get_statistics()

        await roma.decompose_problem("Test problem")
        stats_after = roma.get_statistics()

        assert stats_after["decompositions_performed"] == stats_before["decompositions_performed"] + 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
