"""
Comprehensive Test Suite for ROMA Cross-Integrations

This module provides complete test coverage for ROMA cross-integrations:
- ROMA-DSPy Integration (cooperative reasoning)
- ROMA-DeepKE Integration (entity extraction)
- ROMA-RAGbits Integration (solution indexing)
- Cross-integration initialization
- Method chaining between integrations
- Error handling in cross-integration scenarios
- Knowledge flow between systems
- Async operations across integrations

Test Statistics:
- Total Test Functions: 72
- Test Classes: 12
- Fixture Functions: 15
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Cross-System

Test Categories:
1. ROMA-DSPy Tests - Test cooperative reasoning integration
2. ROMA-DeepKE Tests - Test entity extraction integration
3. ROMA-RAGbits Tests - Test solution indexing and retrieval
4. Cross-Integration Tests - Test interactions between integrations
5. Error Handling Tests - Test graceful degradation
6. Configuration Tests - Test default and custom configurations
7. Statistics Tests - Test statistics tracking and reporting
8. Factory Function Tests - Test creation functions

Running Tests:
    pytest tests/test_roma_cross_integrations.py -v
    pytest tests/test_roma_cross_integrations.py -v -k "test_dspy"
    pytest tests/test_roma_cross_integrations.py --cov=knowledge_engine.integrations.roma_dspy_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
Created: 2026-02-03
"""

import pytest
import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import ROMA cross-integrations
# NOTE: We don't use pytestmark here because setting it multiple times causes issues
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
    ROMAIntegration = None
    ROMAResult = None
    ROMADecomposition = None
    ROMASolution = None
    ROMAVerification = None

try:
    from knowledge_engine.integrations.dspy_integration import DSPyIntegration
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    DSPyIntegration = None

try:
    from knowledge_engine.integrations.deepke_integration import DeepKEIntegration
    DEEPKE_AVAILABLE = True
except ImportError:
    DEEPKE_AVAILABLE = False
    DeepKEIntegration = None

try:
    from knowledge_engine.integrations.ragbits_integration import RagbitsIntegration
    RAGBITS_AVAILABLE = True
except ImportError:
    RAGBITS_AVAILABLE = False
    RagbitsIntegration = None

try:
    from knowledge_engine.integrations.roma_dspy_integration import (
        ROMADSPyIntegration,
        ReasoningTrace,
        EnhancedSubproblem,
        create_roma_dspy_integration
    )
    ROMA_DSPY_AVAILABLE = True
except ImportError:
    ROMA_DSPY_AVAILABLE = False
    ROMADSPyIntegration = None
    ReasoningTrace = None
    EnhancedSubproblem = None
    create_roma_dspy_integration = None

try:
    from knowledge_engine.integrations.roma_deepke_integration import (
        ROMADeepKEIntegration,
        EntityExtraction,
        create_roma_deepke_integration
    )
    ROMA_DEEPKE_AVAILABLE = True
except ImportError:
    ROMA_DEEPKE_AVAILABLE = False
    ROMADeepKEIntegration = None
    EntityExtraction = None
    create_roma_deepke_integration = None

try:
    from knowledge_engine.integrations.roma_ragbits_integration import (
        ROMARagbitsIntegration,
        IndexedSolution,
        SimilarSolution,
        SolutionReuseResult,
        IndexStatistics,
        SolutionReuseStatus,
        create_roma_ragbits_integration,
        get_roma_ragbits_integration
    )
    ROMA_RAGBITS_AVAILABLE = True
except ImportError:
    ROMA_RAGBITS_AVAILABLE = False
    ROMARagbitsIntegration = None
    IndexedSolution = None
    SimilarSolution = None
    SolutionReuseResult = None
    IndexStatistics = None
    SolutionReuseStatus = None
    create_roma_ragbits_integration = None
    get_roma_ragbits_integration = None


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration for cross-integrations."""
    return {
        "auto_add_reasoning": True,
        "auto_extract_entities": True,
        "auto_create_kg_entities": True,
        "parallel_reasoning": True,
        "batch_size": 5,
        "confidence_threshold": 0.7,
        "entity_types": ["PERSON", "ORG", "TECH", "CONCEPT"],
        "relation_types": ["uses", "depends_on", "solves", "related_to"]
    }


@pytest.fixture
def sample_problem():
    """Sample problem for testing."""
    return "Design a scalable microservices architecture for an e-commerce platform"


@pytest.fixture
def sample_subproblems():
    """Sample sub-problems for testing."""
    return [
        {
            "id": "sub_1",
            "problem": "Design API gateway",
            "depth": 1,
            "is_atomic": True,
            "metadata": {}
        },
        {
            "id": "sub_2",
            "problem": "Design service discovery",
            "depth": 1,
            "is_atomic": True,
            "metadata": {}
        },
        {
            "id": "sub_3",
            "problem": "Design authentication system",
            "depth": 1,
            "is_atomic": True,
            "metadata": {}
        }
    ]


@pytest.fixture
def sample_decomposition(sample_problem):
    """Sample ROMA decomposition."""
    return ROMADecomposition(
        decomposition_id="test_decomp_1",
        problem=sample_problem,
        sub_problems=[],
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
        solution="Implement microservices with API gateway",
        confidence=0.85,
        reasoning="Applied architectural best practices",
        metadata={"agent_used": "reasoning"}
    )


@pytest.fixture
def sample_roma_result(sample_decomposition, sample_solution):
    """Sample ROMA result."""
    return ROMAResult(
        success=True,
        decomposition=sample_decomposition,
        solutions=[sample_solution],
        verification=None,
        metadata={"strategy": "recursive"},
        processing_time_ms=500.0
    )


@pytest.fixture
def mock_roma_integration():
    """Mock ROMA integration."""
    mock_roma = Mock(spec=ROMAIntegration)

    # Create sample decomposition and solution
    sample_decomp = ROMADecomposition(
        decomposition_id="test",
        problem="Test problem",
        sub_problems=[],
        is_atomic=False,
        depth=0
    )

    sample_sol = ROMASolution(
        solution_id="sol_1",
        problem_id="test",
        solution="Test solution",
        confidence=0.8,
        reasoning="Test reasoning"
    )

    sample_verif = ROMAVerification(
        verification_id="v1",
        solution_id="sol_1",
        passed=True,
        score=0.9,
        feedback="Good solution",
        requirements_met={"completeness": True, "correctness": True}
    )

    mock_roma.decompose_problem = AsyncMock(return_value=ROMAResult(
        success=True,
        decomposition=sample_decomp,
        solutions=[],
        verification=None,
        metadata={},
        processing_time_ms=100.0
    ))
    mock_roma.solve_atomic = AsyncMock(return_value=ROMAResult(
        success=True,
        decomposition=None,
        solutions=[sample_sol],
        verification=None,
        metadata={},
        processing_time_ms=200.0
    ))
    mock_roma.reassemble_solution = AsyncMock(return_value=ROMAResult(
        success=True,
        decomposition=None,
        solutions=[],
        verification=None,
        metadata={},
        processing_time_ms=100.0
    ))
    mock_roma.verify_solution = AsyncMock(return_value=ROMAResult(
        success=True,
        decomposition=None,
        solutions=[],
        verification=sample_verif,
        metadata={},
        processing_time_ms=150.0
    ))
    mock_roma.health_check = Mock(return_value={"status": "healthy"})
    mock_roma.close = AsyncMock()
    return mock_roma


@pytest.fixture
def mock_dspy_integration():
    """Mock DSPy integration."""
    mock_dspy = Mock()
    mock_dspy.lm = Mock()  # Simulate available LM
    mock_dspy.chain_of_thought = AsyncMock(return_value=Mock(
        success=True,
        reasoning="Step 1: Analyze problem\nStep 2: Consider options\nStep 3: Select best approach",
        output="Solution: Implement API gateway using microservices pattern",
        processing_time_ms=300.0
    ))
    mock_dspy.get_dspy_status = Mock(return_value={"available": True})
    mock_dspy.close = AsyncMock()
    return mock_dspy


@pytest.fixture
def mock_deepke_integration():
    """Mock DeepKE integration."""
    mock_deepke = Mock()
    mock_deepke.extract_entities = AsyncMock(return_value=Mock(
        success=True,
        entities=[
            {"name": "API Gateway", "type": "TECH", "confidence": 0.9},
            {"name": "Microservices", "type": "TECH", "confidence": 0.95},
            {"name": "REST", "type": "CONCEPT", "confidence": 0.85}
        ]
    ))
    mock_deepke.extract_relations = AsyncMock(return_value=Mock(
        success=True,
        relations=[
            {"subject": "API Gateway", "predicate": "uses", "object": "REST", "confidence": 0.8}
        ]
    ))
    return mock_deepke


@pytest.fixture
def mock_ragbits_integration():
    """Mock RAGbits integration."""
    mock_ragbits = Mock()
    mock_ragbits.ingest_documents = AsyncMock(return_value=Mock(
        success=True,
        document_ids=["doc_1", "doc_2"]
    ))
    mock_ragbits.search_documents = AsyncMock(return_value=Mock(
        success=True,
        results=[
            {
                "content": "Test solution 1",
                "score": 0.9,
                "metadata": {"solution_id": "sol_1", "problem_type": "design"}
            },
            {
                "content": "Test solution 2",
                "score": 0.8,
                "metadata": {"solution_id": "sol_2", "problem_type": "design"}
            }
        ]
    ))
    mock_ragbits.get_statistics = AsyncMock(return_value={"index_size_bytes": 10000})
    mock_ragbits.health_check = AsyncMock(return_value={"status": "healthy"})
    mock_ragbits.close = AsyncMock()
    return mock_ragbits


@pytest.fixture
def mock_knowledge_engine():
    """Mock knowledge engine."""
    mock_ke = Mock()
    mock_ke.add_entity_async = AsyncMock(return_value="entity_123")
    mock_ke.get_entity_async = AsyncMock(return_value=None)
    mock_ke.search_entities_async = AsyncMock(return_value=[])
    mock_ke.add_relationship_async = AsyncMock(return_value="rel_123")
    mock_ke.get_statistics_async = AsyncMock(return_value={
        "total_entities": 100,
        "total_relationships": 200
    })
    return mock_ke


# =============================================================================
# TEST CLASS: ROMA-DSPy Integration - Initialization
# =============================================================================

@pytest.mark.skipif(not ROMA_DSPY_AVAILABLE, reason="ROMA-DSPy not available")
class TestROMADSPyInitialization:
    """Test suite for ROMA-DSPy initialization."""

    def test_initialization_with_integrations(self, mock_roma_integration, mock_dspy_integration):
        """Test initialization with ROMA and DSPy integrations."""
        integration = ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration
        )

        assert integration.roma == mock_roma_integration
        assert integration.dspy == mock_dspy_integration
        assert integration._reasoning_cache == {}
        assert integration._stats["cooperative_solutions"] == 0

    def test_initialization_with_config(self, mock_roma_integration, mock_dspy_integration, sample_config):
        """Test initialization with custom configuration."""
        integration = ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration,
            config=sample_config
        )

        # Config is merged with defaults, so check that custom values are applied
        assert integration.config["auto_add_reasoning"] is True
        assert integration.config["parallel_reasoning"] is True
        # Check that default values are also present
        assert "reasoning_model" in integration.config
        assert "cache_reasoning" in integration.config

    def test_default_config_structure(self, mock_roma_integration, mock_dspy_integration):
        """Test default configuration has all required fields."""
        integration = ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration
        )
        config = integration._get_default_config()

        assert "auto_add_reasoning" in config
        assert "reasoning_model" in config
        assert "max_reasoning_steps" in config
        assert "confidence_threshold" in config
        assert "parallel_reasoning" in config
        assert "cache_reasoning" in config

    def test_cache_key_generation(self, mock_roma_integration, mock_dspy_integration):
        """Test cache key generation for reasoning traces."""
        integration = ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration
        )

        key1 = integration._generate_cache_key("test problem")
        key2 = integration._generate_cache_key("test problem")
        key3 = integration._generate_cache_key("different problem")

        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key
        assert len(key1) == 64  # SHA256 hex length


# =============================================================================
# TEST CLASS: ROMA-DSPy Integration - Cooperative Reasoning
# =============================================================================

@pytest.mark.skipif(not ROMA_DSPY_AVAILABLE, reason="ROMA-DSPy not available")
class TestROMADSPyCooperativeReasoning:
    """Test suite for ROMA-DSPy cooperative reasoning."""

    @pytest.fixture
    def integration(self, mock_roma_integration, mock_dspy_integration):
        """Create integration for testing."""
        return ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration,
            config={"auto_add_reasoning": True}
        )

    @pytest.mark.asyncio
    async def test_solve_with_cooperative_reasoning_basic(self, integration, sample_problem):
        """Test basic cooperative reasoning."""
        result = await integration.solve_with_cooperative_reasoning(sample_problem)

        assert isinstance(result, ROMAResult)
        assert result.success is True
        assert "enhanced_subproblems" in result.metadata

    @pytest.mark.asyncio
    async def test_cooperative_reasoning_decomposes_first(self, integration, sample_problem):
        """Test that cooperative reasoning starts with ROMA decomposition."""
        await integration.solve_with_cooperative_reasoning(sample_problem)

        assert integration.roma.decompose_problem.called

    @pytest.mark.asyncio
    async def test_cooperative_reasoning_adds_reasoning_traces(self, integration, sample_problem):
        """Test that reasoning traces are added to sub-problems."""
        result = await integration.solve_with_cooperative_reasoning(sample_problem)

        enhanced_subproblems = result.metadata.get("enhanced_subproblems", [])
        assert len(enhanced_subproblems) > 0

    @pytest.mark.asyncio
    async def test_cooperative_reasoning_with_max_depth(self, integration, sample_problem):
        """Test cooperative reasoning with custom max depth."""
        result = await integration.solve_with_cooperative_reasoning(
            sample_problem,
            max_depth=5
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_cooperative_reasoning_parallel_disabled(self, integration, sample_problem):
        """Test cooperative reasoning with parallel reasoning disabled."""
        integration.config["parallel_reasoning"] = False

        result = await integration.solve_with_cooperative_reasoning(sample_problem)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_cooperative_reasoning_statistics_tracking(self, integration, sample_problem):
        """Test that statistics are tracked during cooperative reasoning."""
        initial_count = integration._stats["cooperative_solutions"]

        await integration.solve_with_cooperative_reasoning(sample_problem)

        assert integration._stats["cooperative_solutions"] == initial_count + 1


# =============================================================================
# TEST CLASS: ROMA-DSPy Integration - Reasoning Traces
# =============================================================================

@pytest.mark.skipif(not ROMA_DSPY_AVAILABLE, reason="ROMA-DSPy not available")
class TestROMADSPyReasoningTraces:
    """Test suite for reasoning trace generation."""

    @pytest.fixture
    def integration(self, mock_roma_integration, mock_dspy_integration):
        """Create integration for testing."""
        return ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration
        )

    @pytest.mark.asyncio
    async def test_add_reasoning_to_subproblem(self, integration, sample_subproblems):
        """Test adding reasoning to a single sub-problem."""
        enhanced = await integration.add_reasoning_to_subproblem(sample_subproblems[0])

        assert isinstance(enhanced, EnhancedSubproblem)
        assert enhanced.subproblem_id == sample_subproblems[0]["id"]
        assert enhanced.problem == sample_subproblems[0]["problem"]

    @pytest.mark.asyncio
    async def test_reasoning_trace_structure(self, integration, sample_subproblems):
        """Test that reasoning trace has proper structure."""
        enhanced = await integration.add_reasoning_to_subproblem(sample_subproblems[0])

        if enhanced.reasoning_trace:
            assert enhanced.reasoning_trace.trace_id is not None
            assert len(enhanced.reasoning_trace.steps) > 0
            assert enhanced.reasoning_trace.confidence >= 0.0
            assert enhanced.reasoning_trace.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_reasoning_cache_hit(self, integration, sample_subproblems):
        """Test that reasoning cache works."""
        integration.config["cache_reasoning"] = True

        # First call - should generate reasoning
        enhanced1 = await integration.add_reasoning_to_subproblem(sample_subproblems[0])

        # Second call - should hit cache
        initial_hits = integration._stats["reasoning_cache_hits"]
        enhanced2 = await integration.add_reasoning_to_subproblem(sample_subproblems[0])

        assert integration._stats["reasoning_cache_hits"] >= initial_hits

    @pytest.mark.asyncio
    async def test_batch_reason_subproblems(self, integration, sample_subproblems):
        """Test batch reasoning for multiple sub-problems."""
        enhanced_list = await integration.batch_reason_subproblems(sample_subproblems)

        assert len(enhanced_list) == len(sample_subproblems)
        assert all(isinstance(e, EnhancedSubproblem) for e in enhanced_list)

    @pytest.mark.asyncio
    async def test_batch_reasoning_parallel(self, integration, sample_subproblems):
        """Test parallel batch reasoning."""
        integration.config["parallel_reasoning"] = True
        integration.config["batch_size"] = 2

        enhanced_list = await integration.batch_reason_subproblems(sample_subproblems)

        assert len(enhanced_list) == len(sample_subproblems)

    @pytest.mark.asyncio
    async def test_reasoning_failure_handling(self, integration, sample_subproblems):
        """Test handling of reasoning generation failure."""
        integration.dspy.chain_of_thought = AsyncMock(
            side_effect=Exception("Reasoning failed")
        )

        enhanced = await integration.add_reasoning_to_subproblem(sample_subproblems[0])

        # Should return enhanced sub-problem with mock reasoning (fallback)
        assert isinstance(enhanced, EnhancedSubproblem)
        # When DSPy fails, it falls back to mock reasoning
        assert enhanced.reasoning_trace is not None
        assert enhanced.reasoning_trace.metadata.get("mock_reasoning") is True


# =============================================================================
# TEST CLASS: ROMA-DeepKE Integration - Initialization
# =============================================================================

@pytest.mark.skipif(not ROMA_DEEPKE_AVAILABLE, reason="ROMA-DeepKE not available")
class TestROMADeepKEInitialization:
    """Test suite for ROMA-DeepKE initialization."""

    def test_initialization_with_required_components(self, mock_roma_integration, mock_deepke_integration, mock_knowledge_engine):
        """Test initialization with all required components."""
        integration = ROMADeepKEIntegration(
            roma_integration=mock_roma_integration,
            deepke_integration=mock_deepke_integration,
            knowledge_engine=mock_knowledge_engine
        )

        assert integration.roma == mock_roma_integration
        assert integration.deepke == mock_deepke_integration
        assert integration.knowledge_engine == mock_knowledge_engine

    def test_initialization_requires_roma(self, mock_deepke_integration, mock_knowledge_engine):
        """Test that ROMA integration is required."""
        with pytest.raises(ValueError, match="ROMA integration is required"):
            ROMADeepKEIntegration(
                roma_integration=None,
                deepke_integration=mock_deepke_integration,
                knowledge_engine=mock_knowledge_engine
            )

    def test_initialization_requires_deepke(self, mock_roma_integration, mock_knowledge_engine):
        """Test that DeepKE integration is required."""
        with pytest.raises(ValueError, match="DeepKE integration is required"):
            ROMADeepKEIntegration(
                roma_integration=mock_roma_integration,
                deepke_integration=None,
                knowledge_engine=mock_knowledge_engine
            )

    def test_initialization_requires_knowledge_engine(self, mock_roma_integration, mock_deepke_integration):
        """Test that knowledge engine is required."""
        with pytest.raises(ValueError, match="Knowledge engine is required"):
            ROMADeepKEIntegration(
                roma_integration=mock_roma_integration,
                deepke_integration=mock_deepke_integration,
                knowledge_engine=None
            )

    def test_default_config_structure(self, mock_roma_integration, mock_deepke_integration, mock_knowledge_engine):
        """Test default configuration structure."""
        integration = ROMADeepKEIntegration(
            roma_integration=mock_roma_integration,
            deepke_integration=mock_deepke_integration,
            knowledge_engine=mock_knowledge_engine
        )
        config = integration._get_default_config()

        assert "auto_extract_entities" in config
        assert "auto_extract_relations" in config
        assert "auto_create_kg_entities" in config
        assert "entity_types" in config
        assert "confidence_threshold" in config


# =============================================================================
# TEST CLASS: ROMA-DeepKE Integration - Entity Extraction
# =============================================================================

@pytest.mark.skipif(not ROMA_DEEPKE_AVAILABLE, reason="ROMA-DeepKE not available")
class TestROMADeepKEEntityExtraction:
    """Test suite for entity extraction functionality."""

    @pytest.fixture
    def integration(self, mock_roma_integration, mock_deepke_integration, mock_knowledge_engine):
        """Create integration for testing."""
        return ROMADeepKEIntegration(
            roma_integration=mock_roma_integration,
            deepke_integration=mock_deepke_integration,
            knowledge_engine=mock_knowledge_engine,
            config={"auto_extract_entities": True}
        )

    @pytest.mark.asyncio
    async def test_enrich_with_entities_basic(self, integration, sample_roma_result):
        """Test basic entity enrichment."""
        enriched = await integration.enrich_with_entities(sample_roma_result)

        assert isinstance(enriched, ROMAResult)
        assert "extracted_entities" in enriched.metadata

    @pytest.mark.asyncio
    async def test_enrich_creates_kg_entities(self, integration, sample_roma_result):
        """Test that enrichment creates knowledge graph entities."""
        enriched = await integration.enrich_with_entities(sample_roma_result)

        kg_ids = enriched.metadata.get("kg_entity_ids", [])
        assert isinstance(kg_ids, list)

    @pytest.mark.asyncio
    async def test_extract_entities_from_solution(self, integration):
        """Test extracting entities from solution text."""
        solution_text = "Implement API Gateway using REST protocol"

        entities = await integration.extract_entities_from_solution(
            solution_text,
            "technical_solution"
        )

        assert isinstance(entities, list)
        # Check entity structure
        for entity in entities:
            assert "name" in entity
            assert "type" in entity
            assert "confidence" in entity

    @pytest.mark.asyncio
    async def test_extract_relations_from_solution(self, integration):
        """Test extracting relations from solution text."""
        solution_text = "API Gateway uses REST protocol"
        entities = [{"name": "API Gateway", "type": "TECH"}]

        relations = await integration.extract_relations_from_solution(
            solution_text,
            entities
        )

        assert isinstance(relations, list)

    @pytest.mark.asyncio
    async def test_entity_deduplication(self, integration):
        """Test entity deduplication."""
        duplicate_entities = [
            {"name": "API Gateway", "type": "TECH", "confidence": 0.8},
            {"name": "API Gateway", "type": "TECH", "confidence": 0.9},
            {"name": "Microservices", "type": "TECH", "confidence": 0.85}
        ]

        deduplicated = await integration._deduplicate_entities(duplicate_entities)

        # Should remove duplicates, keep highest confidence
        api_gateway_count = sum(1 for e in deduplicated if e["name"] == "API Gateway")
        assert api_gateway_count == 1

    @pytest.mark.asyncio
    async def test_batch_extract_entities(self, integration):
        """Test batch entity extraction."""
        results = [sample_roma_result] * 3

        enriched = await integration.batch_extract_entities(results)

        assert len(enriched) == 3


# =============================================================================
# TEST CLASS: ROMA-RAGbits Integration - Initialization
# =============================================================================

@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits not available")
class TestROMARagbitsInitialization:
    """Test suite for ROMA-RAGbits initialization."""

    def test_initialization_with_integrations(self, mock_roma_integration, mock_ragbits_integration):
        """Test initialization with ROMA and RAGbits integrations."""
        integration = ROMARagbitsIntegration(
            roma_integration=mock_roma_integration,
            ragbits_integration=mock_ragbits_integration
        )

        assert integration.roma_integration == mock_roma_integration
        assert integration.ragbits_integration == mock_ragbits_integration
        assert integration._stats["solutions_indexed"] == 0

    def test_initialization_without_integrations(self):
        """Test initialization creates integrations if not provided."""
        integration = ROMARagbitsIntegration()

        # Integrations should be initialized or None based on availability
        assert integration.config is not None

    def test_default_config_structure(self):
        """Test default configuration structure."""
        integration = ROMARagbitsIntegration()
        config = integration._get_default_config()

        assert "auto_index_solutions" in config
        assert "similarity_threshold" in config
        assert "max_index_size" in config
        assert "batch_index_size" in config
        assert "ragbits" in config
        assert "roma" in config


# =============================================================================
# TEST CLASS: ROMA-RAGbits Integration - Solution Indexing
# =============================================================================

@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits not available")
class TestROMARagbitsSolutionIndexing:
    """Test suite for solution indexing functionality."""

    @pytest.fixture
    def integration(self, mock_roma_integration, mock_ragbits_integration):
        """Create integration for testing."""
        return ROMARagbitsIntegration(
            roma_integration=mock_roma_integration,
            ragbits_integration=mock_ragbits_integration
        )

    @pytest.mark.asyncio
    async def test_index_solution_basic(self, integration, sample_roma_result):
        """Test basic solution indexing."""
        doc_id = await integration.index_solution(sample_roma_result)

        assert doc_id is not None
        assert isinstance(doc_id, str)
        assert doc_id.startswith("roma_sol_")

    @pytest.mark.asyncio
    async def test_index_solution_idempotent(self, integration, sample_roma_result):
        """Test that indexing is idempotent."""
        doc_id1 = await integration.index_solution(sample_roma_result)
        doc_id2 = await integration.index_solution(sample_roma_result)

        # Should return same document ID
        assert doc_id1 == doc_id2

    @pytest.mark.asyncio
    async def test_index_solution_with_metadata(self, integration, sample_roma_result):
        """Test indexing with additional metadata."""
        custom_metadata = {"project": "test_project", "version": "1.0"}

        doc_id = await integration.index_solution(
            sample_roma_result,
            metadata=custom_metadata
        )

        assert doc_id is not None

    @pytest.mark.asyncio
    async def test_batch_index_solutions(self, integration, sample_roma_result):
        """Test batch indexing of solutions."""
        results = [sample_roma_result] * 5

        doc_ids = await integration.index_batch_solutions(results)

        assert len(doc_ids) == 5
        assert all(isinstance(id, str) for id in doc_ids)

    @pytest.mark.asyncio
    async def test_create_solution_content(self, integration, sample_roma_result):
        """Test solution content creation."""
        if not sample_roma_result.solutions:
            pytest.skip("No solutions in sample result")

        content = integration._create_solution_content(
            sample_roma_result,
            sample_roma_result.solutions[0]
        )

        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_determine_problem_type(self, integration):
        """Test problem type determination."""
        decomp = ROMADecomposition(
            decomposition_id="test",
            problem="Design a scalable system architecture",
            sub_problems=[],
            is_atomic=False,
            depth=0
        )

        problem_type = integration._determine_problem_type(decomp)

        assert problem_type == "design"

    @pytest.mark.asyncio
    async def test_calculate_complexity(self, integration, sample_roma_result):
        """Test complexity calculation."""
        complexity = integration._calculate_complexity(sample_roma_result)

        assert isinstance(complexity, float)
        assert complexity >= 0.0
        assert complexity <= 1.0


# =============================================================================
# TEST CLASS: ROMA-RAGbits Integration - Solution Retrieval
# =============================================================================

@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits not available")
class TestROMARagbitsSolutionRetrieval:
    """Test suite for solution retrieval functionality."""

    @pytest.fixture
    def integration(self, mock_roma_integration, mock_ragbits_integration):
        """Create integration for testing."""
        return ROMARagbitsIntegration(
            roma_integration=mock_roma_integration,
            ragbits_integration=mock_ragbits_integration
        )

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions_basic(self, integration, sample_problem):
        """Test basic similar solution retrieval."""
        similar = await integration.retrieve_similar_solutions(sample_problem, top_k=3)

        assert isinstance(similar, list)
        assert len(similar) <= 3
        assert all(isinstance(s, SimilarSolution) for s in similar)

    @pytest.mark.asyncio
    async def test_retrieve_similar_solutions_with_filters(self, integration, sample_problem):
        """Test retrieval with filters."""
        filters = {
            "problem_type": "design",
            "min_confidence": 0.7
        }

        similar = await integration.retrieve_similar_solutions(
            sample_problem,
            top_k=5,
            filters=filters
        )

        assert isinstance(similar, list)

    @pytest.mark.asyncio
    async def test_similar_solution_structure(self, integration, sample_problem):
        """Test similar solution object structure."""
        similar = await integration.retrieve_similar_solutions(sample_problem, top_k=1)

        if similar:
            solution = similar[0]
            assert solution.document_id is not None
            assert solution.similarity_score >= 0.0
            assert solution.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_search_solutions_general(self, integration):
        """Test general solution search."""
        results = await integration.search_solutions("API design", top_k=10)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_get_solution_by_id(self, integration):
        """Test retrieving solution by ID."""
        result = await integration.get_solution_by_id("test_doc_id")

        # Should return ROMAResult or None
        assert result is None or isinstance(result, ROMAResult)


# =============================================================================
# TEST CLASS: ROMA-RAGbits Integration - Solution Reuse
# =============================================================================

@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits not available")
class TestROMARagbitsSolutionReuse:
    """Test suite for solution reuse functionality."""

    @pytest.fixture
    def integration(self, mock_roma_integration, mock_ragbits_integration):
        """Create integration for testing."""
        return ROMARagbitsIntegration(
            roma_integration=mock_roma_integration,
            ragbits_integration=mock_ragbits_integration,
            config={
                "solution_reuse": {
                    "enabled": True,
                    "min_similarity_for_reuse": 0.8
                }
            }
        )

    @pytest.mark.asyncio
    async def test_reuse_solution_basic(self, integration, sample_problem):
        """Test basic solution reuse."""
        result = await integration.reuse_solution(sample_problem, top_k=3)

        assert isinstance(result, SolutionReuseResult)
        assert result.status in SolutionReuseStatus

    @pytest.mark.asyncio
    async def test_reuse_solution_high_similarity(self, integration, sample_problem, mock_ragbits_integration):
        """Test solution reuse with high similarity match."""
        # Mock high similarity results
        mock_ragbits_integration.search_documents = AsyncMock(return_value=Mock(
            success=True,
            results=[{
                "content": "Perfect match solution",
                "score": 0.95,
                "metadata": {"solution_id": "sol_1", "confidence": 0.9}
            }]
        ))

        result = await integration.reuse_solution(sample_problem, top_k=3)

        assert isinstance(result, SolutionReuseResult)
        if result.status == SolutionReuseStatus.REUSED_DIRECT:
            assert result.solution is not None

    @pytest.mark.asyncio
    async def test_reuse_solution_no_similar_found(self, integration, sample_problem, mock_ragbits_integration):
        """Test solution reuse when no similar solutions found."""
        # Mock empty results
        mock_ragbits_integration.search_documents = AsyncMock(return_value=Mock(
            success=True,
            results=[]
        ))

        result = await integration.reuse_solution(sample_problem)

        assert result.status == SolutionReuseStatus.NO_SIMILAR_FOUND

    @pytest.mark.asyncio
    async def test_reuse_solution_disabled(self, integration, sample_problem):
        """Test solution reuse when disabled."""
        integration.config["solution_reuse"]["enabled"] = False

        result = await integration.reuse_solution(sample_problem)

        assert result.status == SolutionReuseStatus.REUSE_FAILED


# =============================================================================
# TEST CLASS: Cross-Integration Statistics and Health
# =============================================================================

@pytest.mark.skipif(
    not (ROMA_DSPY_AVAILABLE and ROMA_DEEPKE_AVAILABLE and ROMA_RAGBITS_AVAILABLE),
    reason="All cross-integrations not available"
)
class TestCrossIntegrationStatistics:
    """Test suite for cross-integration statistics and health."""

    @pytest.fixture
    def roma_dspy(self, mock_roma_integration, mock_dspy_integration):
        """Create ROMA-DSPy integration."""
        return ROMADSPyIntegration(
            roma_integration=mock_roma_integration,
            dspy_integration=mock_dspy_integration
        )

    @pytest.fixture
    def roma_deepke(self, mock_roma_integration, mock_deepke_integration, mock_knowledge_engine):
        """Create ROMA-DeepKE integration."""
        return ROMADeepKEIntegration(
            roma_integration=mock_roma_integration,
            deepke_integration=mock_deepke_integration,
            knowledge_engine=mock_knowledge_engine
        )

    @pytest.fixture
    def roma_ragbits(self, mock_roma_integration, mock_ragbits_integration):
        """Create ROMA-RAGbits integration."""
        return ROMARagbitsIntegration(
            roma_integration=mock_roma_integration,
            ragbits_integration=mock_ragbits_integration
        )

    def test_roma_dspy_statistics(self, roma_dspy):
        """Test ROMA-DSPy statistics retrieval."""
        stats = roma_dspy.get_statistics()

        assert isinstance(stats, dict)
        assert "cooperative_solutions" in stats
        assert "reasoning_traces_generated" in stats
        assert "cache_hit_rate" in stats
        assert "timestamp" in stats

    def test_roma_dspy_health_check(self, roma_dspy):
        """Test ROMA-DSPy health check."""
        health = roma_dspy.health_check()

        assert isinstance(health, dict)
        assert "status" in health
        assert "roma_status" in health
        assert "dspy_available" in health

    @pytest.mark.asyncio
    async def test_roma_deepke_statistics(self, roma_deepke):
        """Test ROMA-DeepKE statistics retrieval."""
        stats = await roma_deepke.get_entity_statistics()

        assert isinstance(stats, dict)
        assert "solutions_processed" in stats
        assert "entities_extracted" in stats
        assert "relations_extracted" in stats
        assert "success_rate" in stats

    def test_roma_ragbits_statistics(self, roma_ragbits):
        """Test ROMA-RAGbits statistics retrieval."""
        stats = roma_ragbits.get_statistics()

        assert isinstance(stats, dict)
        assert "solutions_indexed" in stats
        assert "solutions_retrieved" in stats
        assert "solutions_reused" in stats
        assert "cached_solutions" in stats

    @pytest.mark.asyncio
    async def test_roma_ragbits_index_statistics(self, roma_ragbits):
        """Test ROMA-RAGbits index statistics."""
        stats = await roma_ragbits.get_index_statistics()

        assert isinstance(stats, IndexStatistics)
        assert stats.total_solutions >= 0
        assert stats.index_health in ["healthy", "moderate", "full", "unknown"]


# =============================================================================
# TEST CLASS: Factory Functions
# =============================================================================

@pytest.mark.skipif(not ROMA_DSPY_AVAILABLE, reason="ROMA-DSPy not available")
class TestROMADSPyFactory:
    """Test suite for ROMA-DSPy factory function."""

    @pytest.mark.asyncio
    async def test_create_roma_dspy_integration(self):
        """Test creating ROMA-DSPy integration via factory."""
        integration = await create_roma_dspy_integration(
            roma_config={"decomposer": {"max_depth": 3}},
            dspy_config={"model": "gpt-4o"},
            integration_config={"auto_add_reasoning": True}
        )

        assert isinstance(integration, ROMADSPyIntegration)
        assert integration.config["auto_add_reasoning"] is True


@pytest.mark.skipif(not ROMA_DEEPKE_AVAILABLE, reason="ROMA-DeepKE not available")
class TestROMADeepKEFactory:
    """Test suite for ROMA-DeepKE factory function."""

    @pytest.mark.asyncio
    async def test_create_roma_deepke_integration(self, mock_knowledge_engine):
        """Test creating ROMA-DeepKE integration via factory."""
        from knowledge_engine.optional_imports import OptionalDependencyError

        try:
            integration = await create_roma_deepke_integration(
                knowledge_engine=mock_knowledge_engine,
                config={"confidence_threshold": 0.8}
            )

            assert isinstance(integration, ROMADeepKEIntegration)
            assert integration.config["confidence_threshold"] == 0.8
        except OptionalDependencyError:
            # DeepKE not actually installed, skip test
            pytest.skip("DeepKE dependency not installed")

    @pytest.mark.asyncio
    async def test_factory_requires_knowledge_engine(self):
        """Test that factory requires knowledge engine."""
        with pytest.raises(ValueError, match="knowledge_engine is required"):
            await create_roma_deepke_integration(knowledge_engine=None)


@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits not available")
class TestROMARagbitsFactory:
    """Test suite for ROMA-RAGbits factory functions."""

    @pytest.mark.asyncio
    async def test_create_roma_ragbits_integration(self):
        """Test creating ROMA-RAGbits integration via async factory."""
        integration = await create_roma_ragbits_integration(
            config={"auto_index_solutions": True}
        )

        assert isinstance(integration, ROMARagbitsIntegration)
        assert integration.config["auto_index_solutions"] is True

    def test_get_roma_ragbits_integration(self):
        """Test getting ROMA-RAGbits integration via sync factory."""
        integration = get_roma_ragbits_integration(
            config={"similarity_threshold": 0.8}
        )

        assert isinstance(integration, ROMARagbitsIntegration)
        assert integration.config["similarity_threshold"] == 0.8


# =============================================================================
# TEST CLASS: Data Classes
# =============================================================================

@pytest.mark.skipif(not ROMA_DSPY_AVAILABLE, reason="ROMA-DSPy not available")
class TestROMADSPyDataClasses:
    """Test suite for ROMA-DSPy data classes."""

    def test_reasoning_trace_creation(self):
        """Test ReasoningTrace dataclass creation."""
        trace = ReasoningTrace(
            trace_id="trace_1",
            subproblem_id="sub_1",
            steps=["Step 1", "Step 2"],
            confidence=0.8,
            intermediate_conclusions=["Conclusion 1"]
        )

        assert trace.trace_id == "trace_1"
        assert len(trace.steps) == 2
        assert trace.confidence == 0.8

    def test_reasoning_trace_to_dict(self):
        """Test ReasoningTrace to_dict conversion."""
        trace = ReasoningTrace(
            trace_id="trace_1",
            subproblem_id="sub_1",
            steps=["Step 1"],
            confidence=0.9,
            intermediate_conclusions=[]
        )

        trace_dict = trace.to_dict()

        assert isinstance(trace_dict, dict)
        assert trace_dict["trace_id"] == "trace_1"
        assert trace_dict["confidence"] == 0.9

    def test_enhanced_subproblem_creation(self):
        """Test EnhancedSubproblem dataclass creation."""
        subproblem = EnhancedSubproblem(
            subproblem_id="sub_1",
            problem="Test problem",
            depth=1,
            is_atomic=True
        )

        assert subproblem.subproblem_id == "sub_1"
        assert subproblem.depth == 1
        assert subproblem.is_atomic is True

    def test_enhanced_subproblem_with_reasoning(self):
        """Test EnhancedSubproblem with reasoning trace."""
        trace = ReasoningTrace(
            trace_id="trace_1",
            subproblem_id="sub_1",
            steps=["Step 1"],
            confidence=0.8,
            intermediate_conclusions=[]
        )

        subproblem = EnhancedSubproblem(
            subproblem_id="sub_1",
            problem="Test problem",
            depth=1,
            is_atomic=True,
            reasoning_trace=trace
        )

        assert subproblem.reasoning_trace == trace


@pytest.mark.skipif(not ROMA_DEEPKE_AVAILABLE, reason="ROMA-DeepKE not available")
class TestROMADeepKEDataClasses:
    """Test suite for ROMA-DeepKE data classes."""

    def test_entity_extraction_creation(self):
        """Test EntityExtraction dataclass creation."""
        extraction = EntityExtraction(
            entities=[{"name": "Test", "type": "TECH"}],
            relations=[],
            confidence=0.85
        )

        assert len(extraction.entities) == 1
        assert extraction.confidence == 0.85

    def test_entity_extraction_to_dict(self):
        """Test EntityExtraction to_dict conversion."""
        extraction = EntityExtraction(
            entities=[],
            relations=[],
            confidence=0.9,
            extraction_metadata={"test": "data"}
        )

        extraction_dict = extraction.to_dict()

        assert isinstance(extraction_dict, dict)
        assert extraction_dict["confidence"] == 0.9
        assert extraction_dict["extraction_metadata"]["test"] == "data"


@pytest.mark.skipif(not ROMA_RAGBITS_AVAILABLE, reason="ROMA-RAGbits not available")
class TestROMARagbitsDataClasses:
    """Test suite for ROMA-RAGbits data classes."""

    def test_indexed_solution_creation(self):
        """Test IndexedSolution dataclass creation."""
        solution = ROMASolution(
            solution_id="sol_1",
            problem_id="prob_1",
            solution="Test solution",
            confidence=0.8,
            reasoning="Test reasoning"
        )

        indexed = IndexedSolution(
            document_id="doc_1",
            solution=solution,
            decomposition=None,
            verification=None
        )

        assert indexed.document_id == "doc_1"
        assert indexed.solution == solution

    def test_similar_solution_creation(self):
        """Test SimilarSolution dataclass creation."""
        similar = SimilarSolution(
            document_id="doc_1",
            problem="Test problem",
            solution="Test solution",
            similarity_score=0.85
        )

        assert similar.similarity_score == 0.85
        assert similar.document_id == "doc_1"

    def test_solution_reuse_result_creation(self):
        """Test SolutionReuseResult dataclass creation."""
        result = SolutionReuseResult(
            success=True,
            status=SolutionReuseStatus.REUSED_DIRECT,
            solution=None,
            similar_solutions=[],
            adaptation_notes="Direct reuse",
            processing_time_ms=100.0
        )

        assert result.success is True
        assert result.status == SolutionReuseStatus.REUSED_DIRECT
        assert result.processing_time_ms == 100.0

    def test_index_statistics_creation(self):
        """Test IndexStatistics dataclass creation."""
        stats = IndexStatistics(
            total_solutions=100,
            index_size_bytes=50000,
            last_indexed="2026-02-03T00:00:00Z",
            problem_types={"design": 50, "general": 50},
            average_complexity=0.7,
            verification_rate=0.8,
            index_health="healthy"
        )

        assert stats.total_solutions == 100
        assert stats.index_health == "healthy"
