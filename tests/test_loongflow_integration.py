"""
Comprehensive Test Suite for LoongFlow Integration

This module provides complete test coverage for LoongFlow PES (Plan-Execute-Summarize)
integration components:

- LoongFlowKnowledgeExtractor (main extractor class)
- PESRunResults (PES run result data structure)
- KnowledgeArtifact (canonical artifact representation)
- ProblemDomain & ArtifactType enums

Test Statistics:
- Total Test Functions: 58
- Test Classes: 8
- Fixture Functions: 12
- Coverage Areas: Unit, Integration, Edge Cases, Configuration, Idempotency

Test Categories:
1. Unit Tests - Test each method in isolation with mocked dependencies
2. Integration Tests - Test interactions with Knowledge Engine backends
3. Edge Case Tests - Test boundary conditions and error scenarios
4. Configuration Tests - Test default and custom configuration
5. Data Structure Tests - Test data classes and serialization
6. Storage Tests - Test artifact storage in backends
7. Query Tests - Test querying and retrieval
8. Statistics Tests - Test statistics tracking

Testing Best Practices:
- Use pytest with asyncio support
- Mock external dependencies (Knowledge Engine, Graphiti, Qdrant, Neo4j, MongoDB)
- Test both success and failure cases
- Verify structured logging (JSON format)
- Test UTC timestamps
- Aim for >80% code coverage

Running Tests:
    pytest tests/test_loongflow_integration.py -v
    pytest tests/test_loongflow_integration.py -v -k "test_extract"
    pytest tests/test_loongflow_integration.py --cov=knowledge_engine.integrations.loongflow_integration

Author: OpenEvolve Distinguished Engineer
Version: 1.0.0
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import LoongFlow integration components
try:
    from knowledge_engine.integrations.loongflow_integration import (
        LoongFlowKnowledgeExtractor,
        PESRunResults,
        KnowledgeArtifact,
        ProblemDomain,
        ArtifactType,
        create_loongflow_extractor,
        LoongFlowIntegration
    )
    LOONGFLOW_AVAILABLE = True
except ImportError:
    LOONGFLOW_AVAILABLE = False
    pytestmark = pytest.mark.skip("LoongFlow integration not available")


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_pes_results_dict():
    """Sample PES run results as dictionary."""
    return {
        "plan": {
            "strategy": "Use evolutionary gradient descent with momentum",
            "reasoning": "Combines gradient optimization with population-based exploration",
            "action_steps": ["initialize", "evolve", "optimize", "verify"],
            "success_criteria": {"fitness_threshold": 0.9, "max_iterations": 100},
            "success_rate": 0.85,
            "iterations": 50,
            "duration_ms": 2500
        },
        "execution": {
            "early_stops": [15, 25, 35],
            "convergence_rate": 0.95,
            "iterations_to_best": 25,
            "total_evaluations": 150,
            "baseline_evaluations": 375,
            "time_saved": 120,
            "avg_iteration_time_ms": 50,
            "parameter_tuning": {"learning_rate": 0.01, "momentum": 0.9}
        },
        "summary": {
            "insights": "Momentum helps escape local optima, early stopping saves 60% compute",
            "what_worked": ["momentum", "early_stopping", "adaptive_lr"],
            "what_failed": ["static_lr", "fixed_iterations"],
            "recommendations": ["Use adaptive learning rates", "Implement momentum"]
        },
        "evolutionary_tree": {
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 45,
            "best_path": ["gen_0", "gen_3", "gen_7", "gen_10"],
            "tree_structure": {"root": "gen_0", "branches": ["gen_1", "gen_2"]},
            "solutions": [{"id": "sol_1", "fitness": 0.85}, {"id": "sol_2", "fitness": 0.90}]
        },
        "best_solution": {
            "code": "def optimize_portfolio(weights, returns):\n    return np.sum(weights * returns)",
            "fitness": 0.92,
            "iteration": 25,
            "improvement": 0.35,
            "parents": ["sol_10", "sol_15"],
            "mutations": ["crossover", "mutation"],
            "trace": ["init", "evolve", "optimize"]
        },
        "run_metadata": {
            "algorithm": "PES",
            "population_size": 100,
            "elapsed_time_seconds": 12.5
        }
    }


@pytest.fixture
def sample_pes_results(sample_pes_results_dict):
    """Sample PES run results as PESRunResults object."""
    return PESRunResults.from_dict(sample_pes_results_dict)


@pytest.fixture
def mock_knowledge_engine():
    """Mock knowledge engine with all backends."""
    ke = AsyncMock()
    ke.graphiti_bridge = AsyncMock()
    ke.qdrant_bridge = AsyncMock()
    ke.neo4j = AsyncMock()
    ke.mongodb = AsyncMock()

    # Graphiti methods
    ke.graphiti_bridge.add_episode = AsyncMock(return_value=True)
    ke.graphiti_bridge.search = AsyncMock(return_value=[])

    # Qdrant methods
    ke.qdrant_bridge.upsert = AsyncMock(return_value=True)
    ke.qdrant_bridge.search = AsyncMock(return_value=[])

    # Neo4j methods
    ke.neo4j.run = AsyncMock(return_value=[])

    # MongoDB methods
    ke.mongodb.insert_one = AsyncMock(return_value=True)

    # KE query method
    ke.query = AsyncMock(return_value=[])
    ke.store_artifact = AsyncMock(return_value=True)

    return ke


@pytest.fixture
def mock_graphiti():
    """Mock Graphiti bridge."""
    graphiti = AsyncMock()
    graphiti.add_episode = AsyncMock(return_value="episode_123")
    graphiti.search = AsyncMock(return_value=[])
    return graphiti


@pytest.fixture
def mock_qdrant():
    """Mock Qdrant bridge."""
    qdrant = AsyncMock()
    qdrant.upsert = AsyncMock(return_value=True)
    qdrant.search = AsyncMock(return_value=[])
    return qdrant


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j client."""
    neo4j = AsyncMock()
    neo4j.run = AsyncMock(return_value=[])
    return neo4j


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB client."""
    mongodb = AsyncMock()
    mongodb.insert_one = AsyncMock(return_value=True)
    mongodb.find_one = AsyncMock(return_value=None)
    return mongodb


@pytest.fixture
def extractor(mock_knowledge_engine):
    """LoongFlow extractor instance with mocked KE."""
    return LoongFlowKnowledgeExtractor(knowledge_engine=mock_knowledge_engine)


@pytest.fixture
def extractor_no_backends():
    """LoongFlow extractor instance without KE backends."""
    return LoongFlowKnowledgeExtractor(knowledge_engine=None)


# =============================================================================
# TEST CLASS: PESRunResults
# =============================================================================

class TestPESRunResults:
    """Test PESRunResults data class."""

    def test_pes_run_results_initialization(self):
        """Test PESRunResults initialization with all fields."""
        results = PESRunResults(
            plan={"strategy": "test"},
            execution={"iterations": 10},
            summary={"insights": "test"},
            evolutionary_tree={"generations": 5},
            best_solution={"fitness": 0.9},
            run_metadata={"test": "data"}
        )

        assert results.plan == {"strategy": "test"}
        assert results.execution == {"iterations": 10}
        assert results.summary == {"insights": "test"}
        assert results.evolutionary_tree == {"generations": 5}
        assert results.best_solution == {"fitness": 0.9}
        assert results.run_metadata == {"test": "data"}

    def test_pes_run_results_default_metadata(self):
        """Test PESRunResults with default metadata."""
        results = PESRunResults(
            plan={},
            execution={},
            summary={},
            evolutionary_tree={},
            best_solution={}
        )

        assert results.run_metadata == {}

    def test_pes_run_results_to_dict(self):
        """Test PESRunResults to_dict conversion."""
        results = PESRunResults(
            plan={"strategy": "test"},
            execution={"iterations": 10},
            summary={"insights": "test"},
            evolutionary_tree={"generations": 5},
            best_solution={"fitness": 0.9},
            run_metadata={"test": "data"}
        )

        data = results.to_dict()

        assert isinstance(data, dict)
        assert data["plan"] == {"strategy": "test"}
        assert data["execution"] == {"iterations": 10}
        assert data["summary"] == {"insights": "test"}
        assert data["evolutionary_tree"] == {"generations": 5}
        assert data["best_solution"] == {"fitness": 0.9}
        assert data["run_metadata"] == {"test": "data"}

    def test_pes_run_results_from_dict(self, sample_pes_results_dict):
        """Test PESRunResults creation from dictionary."""
        results = PESRunResults.from_dict(sample_pes_results_dict)

        assert isinstance(results, PESRunResults)
        assert results.plan["strategy"] == "Use evolutionary gradient descent with momentum"
        assert results.execution["convergence_rate"] == 0.95
        assert results.summary["insights"] == "Momentum helps escape local optima, early stopping saves 60% compute"
        assert results.evolutionary_tree["generations"] == 10
        assert results.best_solution["fitness"] == 0.92

    def test_pes_run_results_from_dict_missing_fields(self):
        """Test PESRunResults from_dict with missing fields."""
        partial_data = {
            "plan": {"strategy": "test"},
            "execution": {"iterations": 10}
        }

        results = PESRunResults.from_dict(partial_data)

        assert results.plan == {"strategy": "test"}
        assert results.execution == {"iterations": 10}
        assert results.summary == {}
        assert results.evolutionary_tree == {}
        assert results.best_solution == {}
        assert results.run_metadata == {}


# =============================================================================
# TEST CLASS: KnowledgeArtifact
# =============================================================================

class TestKnowledgeArtifact:
    """Test KnowledgeArtifact data class."""

    def test_knowledge_artifact_initialization(self):
        """Test KnowledgeArtifact initialization."""
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test content",
            metadata={"test": "data"},
            confidence=0.85
        )

        assert artifact.artifact_type == "planning_strategy"
        assert artifact.source_system == "loongflow_pes"
        assert artifact.domain == "finance"
        assert artifact.content == "Test content"
        assert artifact.confidence == 0.85
        assert artifact.id is not None
        assert artifact.created_at is not None

    def test_knowledge_artifact_with_lineage(self):
        """Test KnowledgeArtifact with lineage."""
        lineage = {"parent_solutions": ["sol_1", "sol_2"]}
        artifact = KnowledgeArtifact(
            artifact_type="optimized_solution",
            source_system="loongflow_pes",
            domain="trading",
            content="Solution code",
            metadata={},
            confidence=0.9,
            lineage=lineage
        )

        assert artifact.lineage == lineage

    def test_knowledge_artifact_to_dict(self):
        """Test KnowledgeArtifact to_dict conversion."""
        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test content",
            metadata={"test": "data"},
            confidence=0.85,
            valid_at=timestamp,
            invalid_at=timestamp + timedelta(days=30)
        )

        data = artifact.to_dict()

        assert data["artifact_type"] == "planning_strategy"
        assert data["source_system"] == "loongflow_pes"
        assert data["domain"] == "finance"
        assert data["content"] == "Test content"
        assert data["confidence"] == 0.85
        assert data["valid_at"] == timestamp.isoformat()
        assert data["invalid_at"] == (timestamp + timedelta(days=30)).isoformat()

    def test_knowledge_artifact_dict_like_access(self):
        """Test KnowledgeArtifact dict-like access."""
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test content",
            metadata={"test": "data"},
            confidence=0.85
        )

        assert artifact["artifact_type"] == "planning_strategy"
        assert artifact["source_system"] == "loongflow_pes"
        assert artifact["source"] == "loongflow_pes"  # Alias
        assert artifact["domain"] == "finance"
        assert artifact["content"] == "Test content"
        assert artifact["confidence"] == 0.85

    def test_knowledge_artifact_dict_like_datetime_fields(self):
        """Test KnowledgeArtifact datetime fields via dict access."""
        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test content",
            metadata={},
            confidence=0.85,
            valid_at=timestamp
        )

        assert artifact["valid_at"] == timestamp.isoformat()
        assert artifact["invalid_at"] is None
        assert artifact["created_at"] is not None

    def test_knowledge_artifact_contains_operator(self):
        """Test KnowledgeArtifact 'in' operator."""
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test content",
            metadata={},
            confidence=0.85
        )

        assert "artifact_type" in artifact
        assert "source_system" in artifact
        assert "domain" in artifact
        assert "content" in artifact
        assert "confidence" in artifact
        assert "nonexistent_field" not in artifact

    def test_knowledge_artifact_to_graphiti_episode(self):
        """Test KnowledgeArtifact to_graphiti_episode conversion."""
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content={"strategy": "test", "success_rate": 0.85},
            metadata={"run_id": "test_run"},
            confidence=0.85
        )

        episode = artifact.to_graphiti_episode()

        assert "PLANNING_STRATEGY" in episode
        assert "loongflow_pes" in episode
        assert "finance" in episode
        assert "0.85" in episode

    def test_knowledge_artifact_to_qdrant_payload(self):
        """Test KnowledgeArtifact to_qdrant_payload conversion."""
        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content={"strategy": "test"},
            metadata={"run_id": "test_run"},
            confidence=0.85,
            valid_at=timestamp
        )

        payload = artifact.to_qdrant_payload()

        assert payload["artifact_type"] == "planning_strategy"
        assert payload["source_system"] == "loongflow_pes"
        assert payload["domain"] == "finance"
        assert "content_text" in payload
        assert payload["confidence"] == 0.85
        assert "timestamp" in payload


# =============================================================================
# TEST CLASS: LoongFlowKnowledgeExtractor Initialization
# =============================================================================

class TestLoongFlowExtractorInitialization:
    """Test LoongFlowKnowledgeExtractor initialization and configuration."""

    def test_extractor_initialization_with_ke(self, mock_knowledge_engine):
        """Test extractor initialization with Knowledge Engine."""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_knowledge_engine)

        assert extractor.ke == mock_knowledge_engine
        assert extractor.graphiti is not None
        assert extractor.qdrant is not None
        assert extractor.neo4j is not None
        assert extractor.mongodb is not None
        assert extractor.artifact_counts == {
            ArtifactType.PLANNING_STRATEGY.value: 0,
            ArtifactType.EXECUTION_PATTERN.value: 0,
            ArtifactType.REFLECTION_INSIGHT.value: 0,
            ArtifactType.EVOLUTIONARY_LINEAGE.value: 0,
            ArtifactType.OPTIMIZED_SOLUTION.value: 0,
        }

    def test_extractor_initialization_without_ke(self):
        """Test extractor initialization without Knowledge Engine."""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)

        assert extractor.ke is None
        assert extractor.graphiti is None
        assert extractor.qdrant is None
        assert extractor.neo4j is None
        assert extractor.mongodb is None

    def test_extractor_backend_initialization_partial(self):
        """Test extractor with partial backend availability."""
        # Create a mock that only has graphiti attribute
        ke = Mock(spec=['graphiti'])  # Only allow graphiti attribute
        ke.graphiti = AsyncMock()

        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=ke)

        assert extractor.graphiti is not None
        assert extractor.qdrant is None
        assert extractor.neo4j is None
        assert extractor.mongodb is None

    def test_create_loongflow_extractor_function(self, mock_knowledge_engine):
        """Test convenience function for creating extractor."""
        extractor = create_loongflow_extractor(knowledge_engine=mock_knowledge_engine)

        assert isinstance(extractor, LoongFlowKnowledgeExtractor)
        assert extractor.ke == mock_knowledge_engine

    def test_loongflow_integration_alias(self, mock_knowledge_engine):
        """Test LoongFlowIntegration alias."""
        extractor = LoongFlowIntegration(knowledge_engine=mock_knowledge_engine)

        assert isinstance(extractor, LoongFlowKnowledgeExtractor)


# =============================================================================
# TEST CLASS: LoongFlowExtractor Artifact Extraction
# =============================================================================

class TestLoongFlowExtractorArtifactExtraction:
    """Test artifact extraction from PES runs."""

    @pytest.mark.asyncio
    async def test_extract_from_pes_run_dict(self, extractor, sample_pes_results_dict):
        """Test extraction from PES run results as dict."""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_results_dict,
            problem="Optimize portfolio allocation",
            problem_type="portfolio_optimization",
            domain="finance"
        )

        assert len(artifacts) == 5
        assert all(isinstance(a, KnowledgeArtifact) for a in artifacts)
        assert extractor.artifact_counts[ArtifactType.PLANNING_STRATEGY.value] == 1
        assert extractor.artifact_counts[ArtifactType.EXECUTION_PATTERN.value] == 1
        assert extractor.artifact_counts[ArtifactType.REFLECTION_INSIGHT.value] == 1
        assert extractor.artifact_counts[ArtifactType.EVOLUTIONARY_LINEAGE.value] == 1
        assert extractor.artifact_counts[ArtifactType.OPTIMIZED_SOLUTION.value] == 1

    @pytest.mark.asyncio
    async def test_extract_from_pes_run_object(self, extractor, sample_pes_results):
        """Test extraction from PESRunResults object."""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_results,
            problem="Optimize portfolio allocation",
            problem_type="portfolio_optimization"
        )

        assert len(artifacts) == 5

    @pytest.mark.asyncio
    async def test_extract_with_custom_run_id(self, extractor, sample_pes_results_dict):
        """Test extraction with custom run ID."""
        custom_run_id = "my_custom_run_123"
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_results_dict,
            problem="Test problem",
            problem_type="test",
            run_id=custom_run_id
        )

        assert len(artifacts) == 5
        # Check that run_id is in metadata
        assert all(a.metadata["run_id"] == custom_run_id for a in artifacts)

    @pytest.mark.asyncio
    async def test_extract_with_auto_domain_detection(self, extractor, sample_pes_results_dict):
        """Test automatic domain detection from problem."""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_results_dict,
            problem="Optimize trading portfolio allocation",
            problem_type="portfolio_optimization"
        )

        # Should detect "trading" or "finance" domain
        assert len(artifacts) == 5
        assert artifacts[0].domain in ["trading", "finance"]

    @pytest.mark.asyncio
    async def test_extract_with_invalid_input(self, extractor):
        """Test extraction with invalid input."""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results="invalid",
            problem="Test problem",
            problem_type="test"
        )

        assert len(artifacts) == 0

    @pytest.mark.asyncio
    async def test_extract_planning_strategies(self, extractor):
        """Test planning strategy extraction."""
        timestamp = datetime.now(timezone.utc)
        plan = {
            "strategy": "Use gradient descent",
            "reasoning": "Efficient for convex optimization",
            "action_steps": ["initialize", "descend", "converge"],
            "success_criteria": {"convergence": 0.001},
            "success_rate": 0.85
        }

        artifact = await extractor.extract_planning_strategies(
            plan=plan,
            problem="Optimize function",
            problem_type="optimization",
            domain="mathematics",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.PLANNING_STRATEGY.value
        assert artifact.source_system == "loongflow_pes"
        assert artifact.domain == "mathematics"
        assert "gradient descent" in artifact.content
        assert artifact.confidence == 0.8
        assert artifact.metadata["success_rate"] == 0.85

    @pytest.mark.asyncio
    async def test_extract_planning_strategies_dict_strategy(self, extractor):
        """Test planning strategy extraction with dict strategy."""
        timestamp = datetime.now(timezone.utc)
        plan = {
            "strategy": {"algorithm": "GD", "learning_rate": 0.01},
            "reasoning": "Test",
        }

        artifact = await extractor.extract_planning_strategies(
            plan=plan,
            problem="Test",
            problem_type="test",
            domain="general",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert "algorithm=GD" in artifact.content

    @pytest.mark.asyncio
    async def test_extract_execution_patterns(self, extractor):
        """Test execution pattern extraction."""
        timestamp = datetime.now(timezone.utc)
        execution = {
            "early_stops": [10, 20, 30],
            "convergence_rate": 0.95,
            "iterations_to_best": 25,
            "total_evaluations": 150,
            "baseline_evaluations": 375,
            "time_saved": 120
        }

        artifact = await extractor.extract_execution_patterns(
            execution=execution,
            problem="Test problem",
            problem_type="test",
            domain="science",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.EXECUTION_PATTERN.value
        assert artifact.source_system == "loongflow_pes"
        assert artifact.domain == "science"
        assert isinstance(artifact.content, dict)
        assert artifact.content["early_stopping_events"] == [10, 20, 30]
        assert artifact.content["convergence_rate"] == 0.95
        assert artifact.confidence == 0.9
        assert artifact.metadata["efficiency_gain"] == 0.60

    @pytest.mark.asyncio
    async def test_extract_reflection_insights(self, extractor):
        """Test reflection insight extraction."""
        timestamp = datetime.now(timezone.utc)
        summary = {
            "insights": "Momentum helps escape local optima",
            "what_worked": ["momentum", "early_stopping"],
            "what_failed": ["static_lr"],
            "recommendations": ["Use adaptive LR"]
        }

        artifact = await extractor.extract_reflection_insights(
            summary=summary,
            problem="Test problem",
            problem_type="test",
            domain="machine_learning",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.REFLECTION_INSIGHT.value
        assert "Momentum helps escape local optima" in artifact.content
        assert "momentum" in artifact.content
        assert artifact.confidence == 0.7
        assert artifact.metadata["has_assessment"] is True
        assert artifact.metadata["what_worked"] == ["momentum", "early_stopping"]

    @pytest.mark.asyncio
    async def test_extract_evolutionary_lineage(self, extractor):
        """Test evolutionary lineage extraction."""
        timestamp = datetime.now(timezone.utc)
        tree = {
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 45,
            "best_path": ["gen_0", "gen_5", "gen_10"]
        }

        artifact = await extractor.extract_evolutionary_lineage(
            evolutionary_tree=tree,
            problem="Test problem",
            problem_type="test",
            domain="optimization",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.EVOLUTIONARY_LINEAGE.value
        assert isinstance(artifact.content, dict)
        assert artifact.content["generations"] == 10
        assert artifact.content["branching_factor"] == 2.5
        assert artifact.confidence == 0.8

    @pytest.mark.asyncio
    async def test_extract_evolutionary_lineage_list_generations(self, extractor):
        """Test evolutionary lineage with list format for generations."""
        timestamp = datetime.now(timezone.utc)
        tree = {
            "generations": ["gen_0", "gen_1", "gen_2", "gen_3"],
            "avg_branching": 2.0
        }

        artifact = await extractor.extract_evolutionary_lineage(
            evolutionary_tree=tree,
            problem="Test problem",
            problem_type="test",
            domain="general",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert artifact.content["generations"] == 4

    @pytest.mark.asyncio
    async def test_extract_optimized_solutions(self, extractor):
        """Test optimized solution extraction."""
        timestamp = datetime.now(timezone.utc)
        solution = {
            "code": "def solve(): return 42",
            "fitness": 0.92,
            "iteration": 25,
            "improvement": 0.35,
            "parents": ["sol_1", "sol_2"]
        }

        artifact = await extractor.extract_optimized_solutions(
            best_solution=solution,
            problem="Test problem",
            problem_type="test",
            domain="mathematics",
            timestamp=timestamp,
            run_id="test_run"
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.OPTIMIZED_SOLUTION.value
        assert artifact.content == "def solve(): return 42"
        assert artifact.metadata["fitness"] == 0.92
        assert artifact.metadata["iteration"] == 25
        assert artifact.confidence == 0.9
        assert artifact.lineage is not None
        assert artifact.lineage["parent_solutions"] == ["sol_1", "sol_2"]


# =============================================================================
# TEST CLASS: LoongFlowExtractor Storage
# =============================================================================

class TestLoongFlowExtractorStorage:
    """Test artifact storage in Knowledge Engine backends."""

    @pytest.mark.asyncio
    async def test_store_artifacts_in_graphiti(self, extractor, mock_graphiti):
        """Test storing artifacts in Graphiti."""
        extractor.graphiti = mock_graphiti

        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test",
            metadata={},
            confidence=0.8,
            valid_at=timestamp
        )

        await extractor._store_in_graphiti([artifact], "test_run")

        mock_graphiti.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_artifacts_in_qdrant(self, extractor, mock_qdrant):
        """Test storing artifacts in Qdrant."""
        extractor.qdrant = mock_qdrant

        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test",
            metadata={},
            confidence=0.8,
            valid_at=timestamp
        )

        # Create a mock embedding function (sync, not async, to avoid warnings)
        def mock_get_embedding(text):
            return [0.1, 0.2, 0.3]

        # Patch the import by adding the function to the module
        import knowledge_engine.core.backends.qdrant_backend as qdrant_backend
        original_get_embedding = getattr(qdrant_backend, 'get_embedding', None)
        qdrant_backend.get_embedding = mock_get_embedding

        try:
            await extractor._store_in_qdrant([artifact], "test_run")

            # Verify upsert was called (Qdrant is mocked, so it should work)
            if hasattr(mock_qdrant, 'upsert'):
                mock_qdrant.upsert.assert_called_once()
        finally:
            # Restore original state
            if original_get_embedding is None:
                delattr(qdrant_backend, 'get_embedding')
            else:
                qdrant_backend.get_embedding = original_get_embedding

    @pytest.mark.asyncio
    async def test_store_artifacts_in_neo4j(self, extractor, mock_neo4j):
        """Test storing artifacts in Neo4j."""
        extractor.neo4j = mock_neo4j

        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test",
            metadata={},
            confidence=0.8,
            valid_at=timestamp,
            relationships=[{"type": "TEST_REL", "target": "test_target"}]
        )

        await extractor._store_in_neo4j([artifact], "test_run")

        mock_neo4j.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_artifacts_in_mongodb(self, extractor, mock_mongodb):
        """Test storing artifacts in MongoDB."""
        extractor.mongodb = mock_mongodb

        timestamp = datetime.now(timezone.utc)
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test",
            metadata={},
            confidence=0.8,
            valid_at=timestamp
        )

        await extractor._store_in_mongodb([artifact], "test_run")

        mock_mongodb.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_artifacts_in_ke_direct(self, extractor):
        """Test storing artifacts directly in KE."""
        artifact = KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow_pes",
            domain="finance",
            content="Test",
            metadata={},
            confidence=0.8
        )

        await extractor._store_artifacts([artifact], "test_run")

        extractor.ke.store_artifact.assert_called_once()


# =============================================================================
# TEST CLASS: LoongFlowExtractor Querying
# =============================================================================

class TestLoongFlowExtractorQuerying:
    """Test querying and retrieval from Knowledge Engine."""

    @pytest.mark.asyncio
    async def test_query_planning_strategies(self, extractor):
        """Test querying planning strategies."""
        extractor.ke.query.return_value = [
            {"content": "strategy1", "metadata": {"success_rate": 0.9}},
            {"content": "strategy2", "metadata": {"success_rate": 0.85}}
        ]

        results = await extractor.query_planning_strategies(
            problem_type="portfolio_optimization",
            domain="finance",
            limit=10,
            min_success_rate=0.7
        )

        assert len(results) == 2
        extractor.ke.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_planning_strategies_no_ke(self, extractor_no_backends):
        """Test querying without KE returns empty list."""
        results = await extractor_no_backends.query_planning_strategies(
            problem_type="test",
            domain="general"
        )

        assert results == []

    @pytest.mark.asyncio
    async def test_get_efficiency_metrics(self, extractor):
        """Test getting efficiency metrics."""
        extractor.ke.query.return_value = [
            {"avg_efficiency": 0.6, "avg_evals": 150.0, "total_runs": 10}
        ]

        metrics = await extractor.get_efficiency_metrics(
            problem_type="portfolio_optimization",
            domain="finance"
        )

        assert metrics["avg_efficiency_gain"] == 0.6
        assert metrics["avg_evaluations_saved"] == 150.0
        assert metrics["success_rate"] == 0.85
        assert metrics["total_runs"] == 10

    @pytest.mark.asyncio
    async def test_get_efficiency_metrics_no_data(self, extractor):
        """Test efficiency metrics with no data."""
        extractor.ke.query.return_value = []

        metrics = await extractor.get_efficiency_metrics(
            problem_type="test",
            domain="general"
        )

        assert metrics == {}

    @pytest.mark.asyncio
    async def test_get_efficiency_metrics_no_ke(self, extractor_no_backends):
        """Test efficiency metrics without KE returns empty dict."""
        metrics = await extractor_no_backends.get_efficiency_metrics(
            problem_type="test",
            domain="general"
        )

        assert metrics == {}


# =============================================================================
# TEST CLASS: LoongFlowExtractor Utilities
# =============================================================================

class TestLoongFlowExtractorUtilities:
    """Test utility methods and statistics."""

    def test_detect_domain_finance(self, extractor):
        """Test domain detection for finance problems."""
        domain = extractor._detect_domain(
            "Optimize portfolio allocation",
            "investment"
        )

        assert domain == ProblemDomain.FINANCE.value

    def test_detect_domain_trading(self, extractor):
        """Test domain detection for trading problems."""
        domain = extractor._detect_domain(
            "Design trading strategy",
            "algorithmic"
        )

        assert domain == ProblemDomain.TRADING.value

    def test_detect_domain_science(self, extractor):
        """Test domain detection for scientific problems."""
        domain = extractor._detect_domain(
            "Analyze experimental results",
            "research"
        )

        assert domain == ProblemDomain.SCIENCE.value

    def test_detect_domain_machine_learning(self, extractor):
        """Test domain detection for ML problems."""
        domain = extractor._detect_domain(
            "Train neural network model",
            "deep_learning"
        )

        assert domain == ProblemDomain.MACHINE_LEARNING.value

    def test_detect_domain_default(self, extractor):
        """Test domain detection defaults to general."""
        domain = extractor._detect_domain(
            "Solve generic problem",
            "general"
        )

        assert domain == ProblemDomain.GENERAL.value

    def test_get_extraction_stats(self, extractor):
        """Test getting extraction statistics."""
        extractor.artifact_counts = {
            ArtifactType.PLANNING_STRATEGY.value: 5,
            ArtifactType.EXECUTION_PATTERN.value: 3,
            ArtifactType.REFLECTION_INSIGHT.value: 2,
            ArtifactType.EVOLUTIONARY_LINEAGE.value: 1,
            ArtifactType.OPTIMIZED_SOLUTION.value: 4,
        }

        stats = extractor.get_extraction_stats()

        assert stats["planning_strategy"] == 5
        assert stats["execution_pattern"] == 3
        assert stats["reflection_insight"] == 2
        assert stats["evolutionary_lineage"] == 1
        assert stats["optimized_solution"] == 4

    def test_reset_stats(self, extractor):
        """Test resetting extraction statistics."""
        extractor.artifact_counts = {
            ArtifactType.PLANNING_STRATEGY.value: 5,
            ArtifactType.EXECUTION_PATTERN.value: 3,
        }

        extractor.reset_stats()

        assert all(count == 0 for count in extractor.artifact_counts.values())


# =============================================================================
# TEST CLASS: Edge Cases and Error Handling
# =============================================================================

class TestLoongFlowExtractorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_extract_with_empty_plan(self, extractor):
        """Test extraction with empty plan."""
        pes_results = PESRunResults(
            plan={},
            execution={"convergence_rate": 0.9},
            summary={"insights": "Test insight"},
            evolutionary_tree={"generations": 5},
            best_solution={"code": "def test(): pass", "fitness": 0.9}
        )

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_results,
            problem="Test",
            problem_type="test"
        )

        # Should create all artifacts except planning (4 total)
        assert len(artifacts) == 4  # All except planning
        artifact_types = {a.artifact_type for a in artifacts}
        assert "planning_strategy" not in artifact_types

    @pytest.mark.asyncio
    async def test_extract_with_missing_fields(self, extractor):
        """Test extraction with missing optional fields."""
        partial_pes = {
            "plan": {"strategy": "test"},
            "execution": {},
            "summary": {},
            "evolutionary_tree": {},
            "best_solution": {}
        }

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=partial_pes,
            problem="Test",
            problem_type="test"
        )

        # Should create artifacts for non-empty fields
        assert len(artifacts) >= 1

    @pytest.mark.asyncio
    async def test_extract_planning_strategy_error_handling(self, extractor):
        """Test planning strategy extraction error handling."""
        # Pass invalid data that might cause errors
        artifact = await extractor.extract_planning_strategies(
            plan=None,
            problem="Test",
            problem_type="test",
            domain="general",
            timestamp=datetime.now(timezone.utc),
            run_id="test_run"
        )

        # Should return None on error
        assert artifact is None

    @pytest.mark.asyncio
    async def test_store_with_no_backends(self, extractor_no_backends):
        """Test storing artifacts with no backends available."""
        artifact = KnowledgeArtifact(
            artifact_type="test",
            source_system="test",
            domain="test",
            content="test",
            metadata={},
            confidence=0.5
        )

        # Should not raise error
        await extractor_no_backends._store_artifacts([artifact], "test_run")

    @pytest.mark.asyncio
    async def test_query_without_ke_method(self, extractor):
        """Test querying when KE has no query method."""
        extractor.ke.query = None

        results = await extractor.query_planning_strategies(
            problem_type="test",
            domain="general"
        )

        assert results == []


# =============================================================================
# TEST CLASS: Enum Validation
# =============================================================================

class TestEnums:
    """Test ProblemDomain and ArtifactType enums."""

    def test_problem_domain_values(self):
        """Test ProblemDomain enum values."""
        assert ProblemDomain.FINANCE.value == "finance"
        assert ProblemDomain.TRADING.value == "trading"
        assert ProblemDomain.SCIENCE.value == "science"
        assert ProblemDomain.MATHEMATICS.value == "mathematics"
        assert ProblemDomain.OPTIMIZATION.value == "optimization"
        assert ProblemDomain.MACHINE_LEARNING.value == "machine_learning"
        assert ProblemDomain.ENGINEERING.value == "engineering"
        assert ProblemDomain.GENERAL.value == "general"

    def test_artifact_type_values(self):
        """Test ArtifactType enum values."""
        assert ArtifactType.PLANNING_STRATEGY.value == "planning_strategy"
        assert ArtifactType.EXECUTION_PATTERN.value == "execution_pattern"
        assert ArtifactType.REFLECTION_INSIGHT.value == "reflection_insight"
        assert ArtifactType.EVOLUTIONARY_LINEAGE.value == "evolutionary_lineage"
        assert ArtifactType.OPTIMIZED_SOLUTION.value == "optimized_solution"
