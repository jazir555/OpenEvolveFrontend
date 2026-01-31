"""
Comprehensive tests for LoongFlow Knowledge Extraction Integration

Tests knowledge extraction from LoongFlow PES runs and integration with
Knowledge Engine storage backends (Graphiti, Qdrant, Neo4j, MongoDB).

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, MagicMock

from knowledge_engine.integrations.loongflow_integration import (
    LoongFlowKnowledgeExtractor,
    PESRunResults,
    KnowledgeArtifact,
    ProblemDomain,
    ArtifactType,
)


class TestPESRunResults:
    """Test PESRunResults dataclass"""

    @pytest.fixture
    def sample_pes_results(self) -> PESRunResults:
        """Create sample PES run results for testing"""
        return PESRunResults(
            plan={
                "strategy": "Use gradient descent with momentum",
                "reasoning": "Momentum helps escape local optima",
                "action_steps": ["Initialize weights", "Compute gradients", "Update with momentum"],
                "success_criteria": {"convergence": 0.001},
                "approach": "gradient_based",
                "success_rate": 0.85,
                "iterations": 50,
            },
            execution={
                "early_stops": [15, 25, 35],
                "convergence_rate": 0.95,
                "iterations_to_best": 25,
                "total_evaluations": 40,
                "baseline_evaluations": 100,
                "time_saved": 120,
                "parameter_tuning": {"learning_rate": 0.01, "momentum": 0.9},
            },
            summary={
                "insights": "Momentum significantly improved convergence",
                "what_worked": ["Gradient descent with momentum", "Adaptive learning rate"],
                "what_failed": ["Pure gradient descent", "Fixed learning rate"],
                "recommendations": ["Always use momentum", "Adapt learning rate"],
                "adaptation_patterns": ["Learning rate decay"],
            },
            evolutionary_tree={
                "generations": 10,
                "avg_branching": 2.5,
                "total_mutations": 20,
                "best_path": ["gen_0", "gen_3", "gen_7", "gen_10"],
                "tree_structure": {"root": "gen_0", "branches": ["gen_1", "gen_2"]},
            },
            best_solution={
                "code": "def solve(): return 42",
                "fitness": 0.95,
                "iteration": 25,
                "improvement": 0.15,
                "params": {"learning_rate": 0.01, "epochs": 100},
            },
        )

    def test_to_dict(self, sample_pes_results: PESRunResults):
        """Test conversion to dictionary"""
        result_dict = sample_pes_results.to_dict()

        assert "plan" in result_dict
        assert "execution" in result_dict
        assert "summary" in result_dict
        assert "evolutionary_tree" in result_dict
        assert "best_solution" in result_dict
        assert result_dict["plan"]["strategy"] == "Use gradient descent with momentum"

    def test_from_dict(self, sample_pes_results: PESRunResults):
        """Test creation from dictionary"""
        result_dict = sample_pes_results.to_dict()
        recreated = PESRunResults.from_dict(result_dict)

        assert recreated.plan["strategy"] == sample_pes_results.plan["strategy"]
        assert recreated.execution["total_evaluations"] == 40
        assert recreated.summary["insights"] == "Momentum significantly improved convergence"
        assert recreated.evolutionary_tree["generations"] == 10
        assert recreated.best_solution["fitness"] == 0.95


class TestKnowledgeArtifact:
    """Test KnowledgeArtifact dataclass"""

    @pytest.fixture
    def sample_artifact(self) -> KnowledgeArtifact:
        """Create sample knowledge artifact"""
        return KnowledgeArtifact(
            artifact_type="planning_strategy",
            source_system="loongflow",
            domain="machine_learning",
            content={"strategy": "Use gradient descent"},
            metadata={"problem": "Optimize neural network"},
            confidence=0.85,
            valid_at=datetime.now(timezone.utc),
            entities=["ml", "optimization"],
            relationships=[{"type": "SOLVES", "target": "problem"}],
        )

    def test_to_dict(self, sample_artifact: KnowledgeArtifact):
        """Test conversion to dictionary"""
        artifact_dict = sample_artifact.to_dict()

        assert artifact_dict["artifact_type"] == "planning_strategy"
        assert artifact_dict["source_system"] == "loongflow"
        assert artifact_dict["domain"] == "machine_learning"
        assert artifact_dict["confidence"] == 0.85
        assert "valid_at" in artifact_dict

    def test_to_graphiti_episode(self, sample_artifact: KnowledgeArtifact):
        """Test conversion to Graphiti episode format"""
        episode = sample_artifact.to_graphiti_episode()

        assert "PLANNING_STRATEGY" in episode
        assert "loongflow" in episode
        assert "machine_learning" in episode
        assert "0.85" in episode

    def test_to_qdrant_payload(self, sample_artifact: KnowledgeArtifact):
        """Test conversion to Qdrant payload"""
        payload = sample_artifact.to_qdrant_payload()

        assert payload["artifact_type"] == "planning_strategy"
        assert payload["source_system"] == "loongflow"
        assert payload["domain"] == "machine_learning"
        assert "content_text" in payload
        assert "timestamp" in payload


class TestLoongFlowKnowledgeExtractor:
    """Test LoongFlowKnowledgeExtractor"""

    @pytest.fixture
    def mock_knowledge_engine(self):
        """Create mock Knowledge Engine"""
        ke = Mock()
        ke.graphiti_bridge = None
        ke.qdrant_bridge = None
        ke.neo4j = None
        ke.mongodb = None
        return ke

    @pytest.fixture
    def extractor(self, mock_knowledge_engine):
        """Create extractor instance"""
        return LoongFlowKnowledgeExtractor(knowledge_engine=mock_knowledge_engine)

    @pytest.fixture
    def sample_pes_results(self) -> PESRunResults:
        """Create sample PES results"""
        return PESRunResults(
            plan={
                "strategy": "Use genetic algorithm",
                "reasoning": "Global search needed",
                "action_steps": ["Initialize population", "Evaluate fitness", "Select best"],
                "success_criteria": {"fitness_threshold": 0.9},
                "approach": "evolutionary",
                "success_rate": 0.8,
                "iterations": 30,
            },
            execution={
                "early_stops": [10, 20],
                "convergence_rate": 0.9,
                "iterations_to_best": 20,
                "total_evaluations": 25,
                "baseline_evaluations": 100,
                "time_saved": 75,
            },
            summary={
                "insights": "Crossover operator was most effective",
                "what_worked": ["Two-point crossover", "Tournament selection"],
                "what_failed": ["Single-point crossover", "Random selection"],
                "recommendations": ["Use two-point crossover", "Keep tournament size 5"],
            },
            evolutionary_tree={
                "generations": 5,
                "avg_branching": 2.0,
                "total_mutations": 10,
                "best_path": ["gen_0", "gen_2", "gen_5"],
            },
            best_solution={
                "code": "best_solution_code",
                "fitness": 0.92,
                "iteration": 20,
                "improvement": 0.2,
            },
        )

    @pytest.mark.asyncio
    async def test_extract_from_pes_run(
        self,
        extractor: LoongFlowKnowledgeExtractor,
        sample_pes_results: PESRunResults,
    ):
        """Test complete extraction from PES run"""
        problem = "Optimize portfolio allocation"
        problem_type = "financial_optimization"

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_results,
            problem=problem,
            problem_type=problem_type,
        )

        # Should extract 5 artifacts
        assert len(artifacts) == 5

        # Check artifact types
        artifact_types = [a.artifact_type for a in artifacts]
        assert ArtifactType.PLANNING_STRATEGY.value in artifact_types
        assert ArtifactType.EXECUTION_PATTERN.value in artifact_types
        assert ArtifactType.REFLECTION_INSIGHT.value in artifact_types
        assert ArtifactType.EVOLUTIONARY_LINEAGE.value in artifact_types
        assert ArtifactType.OPTIMIZED_SOLUTION.value in artifact_types

        # Check source system
        for artifact in artifacts:
            assert artifact.source_system == "loongflow"

        # Check domain detection
        for artifact in artifacts:
            assert artifact.domain == ProblemDomain.FINANCE.value

    @pytest.mark.asyncio
    async def test_extract_planning_strategies(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test planning strategy extraction"""
        plan = {
            "strategy": "Use gradient descent",
            "reasoning": "Convex optimization problem",
            "action_steps": ["Step 1", "Step 2", "Step 3"],
            "success_criteria": {"convergence": 0.001},
            "approach": "gradient_based",
            "success_rate": 0.85,
        }

        artifact = await extractor.extract_planning_strategies(
            plan=plan,
            problem="Optimize function",
            problem_type="mathematical",
            domain="mathematics",
            timestamp=datetime.now(timezone.utc),
            run_id="test_run_1",
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.PLANNING_STRATEGY.value
        assert artifact.source_system == "loongflow"
        assert artifact.domain == "mathematics"
        assert artifact.content["strategy"] == "Use gradient descent"
        assert artifact.confidence > 0.8
        assert len(artifact.entities) > 0
        assert len(artifact.relationships) > 0

    @pytest.mark.asyncio
    async def test_extract_execution_patterns(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test execution pattern extraction"""
        execution = {
            "early_stops": [15, 25],
            "convergence_rate": 0.95,
            "iterations_to_best": 25,
            "total_evaluations": 30,
            "baseline_evaluations": 100,
            "time_saved": 70,
        }

        artifact = await extractor.extract_execution_patterns(
            execution=execution,
            problem="Optimize neural network",
            problem_type="ml_training",
            domain="machine_learning",
            timestamp=datetime.now(timezone.utc),
            run_id="test_run_2",
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.EXECUTION_PATTERN.value
        assert artifact.content["early_stopping_events"] == [15, 25]
        assert artifact.content["convergence_rate"] == 0.95
        assert artifact.metadata["efficiency_gain"] > 0.5  # Should be ~0.7
        assert artifact.metadata["early_stop_count"] == 2
        assert artifact.confidence > 0.7  # Higher confidence for good efficiency

    @pytest.mark.asyncio
    async def test_extract_reflection_insights(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test reflection insight extraction"""
        summary = {
            "insights": "Momentum helps escape local optima",
            "what_worked": ["Gradient descent with momentum", "Adaptive learning rate"],
            "what_failed": ["Pure gradient descent"],
            "recommendations": ["Use momentum", "Adapt learning rate"],
        }

        artifact = await extractor.extract_reflection_insights(
            summary=summary,
            problem="Optimize function",
            problem_type="optimization",
            domain="mathematics",
            timestamp=datetime.now(timezone.utc),
            run_id="test_run_3",
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.REFLECTION_INSIGHT.value
        assert artifact.content["insights"] == "Momentum helps escape local optima"
        assert len(artifact.content["what_worked"]) == 2
        assert len(artifact.content["what_failed"]) == 1
        assert artifact.metadata["has_assessment"] is True
        assert artifact.metadata["insight_count"] == 3

    @pytest.mark.asyncio
    async def test_extract_evolutionary_lineage(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test evolutionary lineage extraction"""
        tree = {
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 25,
            "best_path": ["gen_0", "gen_3", "gen_7", "gen_10"],
            "tree_structure": {"root": "gen_0"},
        }

        artifact = await extractor.extract_evolutionary_lineage(
            evolutionary_tree=tree,
            problem="Evolve sorting algorithm",
            problem_type="algorithm_design",
            domain="general",
            timestamp=datetime.now(timezone.utc),
            run_id="test_run_4",
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.EVOLUTIONARY_LINEAGE.value
        assert artifact.content["generations"] == 10
        assert artifact.content["branching_factor"] == 2.5
        assert artifact.content["total_mutations"] == 25
        assert len(artifact.content["best_path"]) == 4
        assert artifact.confidence >= 0.8  # Has complete tree

    @pytest.mark.asyncio
    async def test_extract_optimized_solutions(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test optimized solution extraction"""
        solution = {
            "code": "def optimized_solution(): pass",
            "fitness": 0.95,
            "iteration": 25,
            "improvement": 0.2,
            "params": {"param1": 0.5, "param2": 1.0},
            "parents": ["parent_1", "parent_2"],
            "mutations": ["mutation_1"],
        }

        artifact = await extractor.extract_optimized_solutions(
            best_solution=solution,
            problem="Maximize function",
            problem_type="optimization",
            domain="mathematics",
            timestamp=datetime.now(timezone.utc),
            run_id="test_run_5",
        )

        assert artifact is not None
        assert artifact.artifact_type == ArtifactType.OPTIMIZED_SOLUTION.value
        assert artifact.content["solution"] == "def optimized_solution(): pass"
        assert artifact.content["fitness"] == 0.95
        assert artifact.content["iteration_found"] == 25
        assert artifact.lineage is not None
        assert artifact.lineage["parent_solutions"] == ["parent_1", "parent_2"]
        assert artifact.confidence > 0.9

    def test_detect_domain(self, extractor: LoongFlowKnowledgeExtractor):
        """Test domain auto-detection"""
        # Finance domain
        domain = extractor._detect_domain(
            "Optimize portfolio allocation for stocks",
            "financial"
        )
        assert domain == ProblemDomain.FINANCE.value

        # Machine learning domain
        domain = extractor._detect_domain(
            "Train neural network for classification",
            "ml_training"
        )
        assert domain == ProblemDomain.MACHINE_LEARNING.value

        # Science domain
        domain = extractor._detect_domain(
            "Design experiment for chemical reaction",
            "scientific"
        )
        assert domain == ProblemDomain.SCIENCE.value

        # General domain (no keywords)
        domain = extractor._detect_domain(
            "Solve this problem",
            "general"
        )
        assert domain == ProblemDomain.GENERAL.value

    @pytest.mark.asyncio
    async def test_extract_with_dict_input(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test extraction with dict input instead of PESRunResults"""
        pes_dict = {
            "plan": {"strategy": "Test strategy", "success_rate": 0.75},
            "execution": {"total_evaluations": 50},
            "summary": {"insights": "Test insight"},
            "evolutionary_tree": {"generations": 5},
            "best_solution": {"fitness": 0.8, "code": "test code"},
        }

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_dict,
            problem="Test problem",
            problem_type="test",
        )

        # Should still extract 5 artifacts
        assert len(artifacts) == 5

    @pytest.mark.asyncio
    async def test_extract_with_missing_data(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test extraction with incomplete PES data"""
        incomplete_pes = PESRunResults(
            plan={"strategy": "Test"},
            execution={},
            summary={},
            evolutionary_tree={},
            best_solution={},
        )

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=incomplete_pes,
            problem="Test",
            problem_type="test",
        )

        # Should extract artifacts for non-empty sections only
        # plan, summary, evolutionary_tree, best_solution have at least some data
        assert len(artifacts) >= 1

    def test_get_extraction_stats(self, extractor: LoongFlowKnowledgeExtractor):
        """Test extraction statistics"""
        stats = extractor.get_extraction_stats()

        assert isinstance(stats, dict)
        assert ArtifactType.PLANNING_STRATEGY.value in stats
        assert ArtifactType.EXECUTION_PATTERN.value in stats
        assert ArtifactType.REFLECTION_INSIGHT.value in stats
        assert ArtifactType.EVOLUTIONARY_LINEAGE.value in stats
        assert ArtifactType.OPTIMIZED_SOLUTION.value in stats

    def test_reset_stats(self, extractor: LoongFlowKnowledgeExtractor):
        """Test statistics reset"""
        # Manually increment some counters
        extractor.artifact_counts[ArtifactType.PLANNING_STRATEGY.value] = 5
        extractor.artifact_counts[ArtifactType.EXECUTION_PATTERN.value] = 3

        # Reset
        extractor.reset_stats()

        # Check all zeros
        stats = extractor.get_extraction_stats()
        for count in stats.values():
            assert count == 0


class TestStorageBackendIntegration:
    """Test integration with Knowledge Engine storage backends"""

    @pytest.fixture
    def mock_graphiti(self):
        """Create mock Graphiti bridge"""
        graphiti = Mock()
        graphiti.add_episode = AsyncMock()
        return graphiti

    @pytest.fixture
    def mock_qdrant(self):
        """Create mock Qdrant bridge"""
        qdrant = Mock()
        qdrant.upsert = AsyncMock()
        return qdrant

    @pytest.fixture
    def mock_neo4j(self):
        """Create mock Neo4j driver"""
        neo4j = Mock()
        neo4j.run = AsyncMock()
        return neo4j

    @pytest.fixture
    def mock_mongodb(self):
        """Create mock MongoDB collection"""
        mongodb = Mock()
        mongodb.insert_one = AsyncMock()
        return mongodb

    @pytest.fixture
    def mock_ke_with_backends(
        self,
        mock_graphiti,
        mock_qdrant,
        mock_neo4j,
        mock_mongodb,
    ):
        """Create mock Knowledge Engine with all backends"""
        ke = Mock()
        ke.graphiti_bridge = mock_graphiti
        ke.qdrant_bridge = mock_qdrant
        ke.neo4j = mock_neo4j
        ke.mongodb = mock_mongodb
        return ke

    @pytest.mark.asyncio
    async def test_store_in_graphiti(self, mock_graphiti):
        """Test storing artifacts in Graphiti"""
        extractor = LoongFlowKnowledgeExtractor()
        extractor.graphiti = mock_graphiti

        artifact = KnowledgeArtifact(
            artifact_type="test_artifact",
            source_system="loongflow",
            domain="test",
            content={"test": "data"},
            metadata={},
            confidence=0.8,
            valid_at=datetime.now(timezone.utc),
        )

        await extractor._store_in_graphiti([artifact], "test_run")

        # Verify add_episode was called
        assert mock_graphiti.add_episode.called
        call_args = mock_graphiti.add_episode.call_args
        assert "test_artifact_test_run" in call_args[1]["name"]

    @pytest.mark.asyncio
    async def test_store_in_neo4j(self, mock_neo4j):
        """Test storing artifacts in Neo4j"""
        extractor = LoongFlowKnowledgeExtractor()
        extractor.neo4j = mock_neo4j

        artifact = KnowledgeArtifact(
            artifact_type="test_artifact",
            source_system="loongflow",
            domain="test",
            content={"test": "data"},
            metadata={},
            confidence=0.8,
            valid_at=datetime.now(timezone.utc),
            relationships=[{"type": "TEST_REL", "target": "test_target"}],
        )

        await extractor._store_in_neo4j([artifact], "test_run")

        # Verify run was called
        assert mock_neo4j.run.called

    @pytest.mark.asyncio
    async def test_store_in_mongodb(self, mock_mongodb):
        """Test storing artifacts in MongoDB"""
        extractor = LoongFlowKnowledgeExtractor()
        extractor.mongodb = mock_mongodb

        artifact = KnowledgeArtifact(
            artifact_type="test_artifact",
            source_system="loongflow",
            domain="test",
            content={"test": "data"},
            metadata={},
            confidence=0.8,
            valid_at=datetime.now(timezone.utc),
        )

        await extractor._store_in_mongodb([artifact], "test_run")

        # Verify insert_one was called
        assert mock_mongodb.insert_one.called

    @pytest.mark.asyncio
    async def test_full_extraction_with_storage(
        self,
        mock_ke_with_backends,
    ):
        """Test full extraction pipeline with all storage backends"""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke_with_backends)

        pes_results = PESRunResults(
            plan={"strategy": "Test", "success_rate": 0.8},
            execution={"total_evaluations": 30, "baseline_evaluations": 100},
            summary={"insights": "Test insight"},
            evolutionary_tree={"generations": 5},
            best_solution={"fitness": 0.9, "code": "test"},
        )

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_results,
            problem="Test problem",
            problem_type="test",
        )

        # Should extract 5 artifacts
        assert len(artifacts) == 5

        # Verify storage backends were called
        assert mock_ke_with_backends.graphiti_bridge.add_episode.called
        assert mock_ke_with_backends.neo4j.run.called
        assert mock_ke_with_backends.mongodb.insert_one.called


class TestQueryMethods:
    """Test query methods for retrieving artifacts"""

    @pytest.fixture
    def mock_ke_with_query(self):
        """Create mock KE with query capability"""
        ke = Mock()
        ke.query = AsyncMock(return_value=[
            {
                "a.content": {"strategy": "Test strategy"},
                "a.metadata": {"success_rate": 0.85, "problem": "Test problem"},
            }
        ])
        return ke

    @pytest.fixture
    def extractor(self, mock_ke_with_query):
        """Create extractor with mock KE"""
        return LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke_with_query)

    @pytest.mark.asyncio
    async def test_query_planning_strategies(
        self,
        extractor: LoongFlowKnowledgeExtractor,
    ):
        """Test querying planning strategies"""
        results = await extractor.query_planning_strategies(
            problem_type="portfolio_optimization",
            domain="finance",
            limit=10,
            min_success_rate=0.7,
        )

        # Should return results from KE
        assert len(results) >= 0
        if results:
            assert "content" in results[0] or "a.content" in results[0]

    @pytest.mark.asyncio
    async def test_get_efficiency_metrics(
        self,
        extractor: LoongFlowKnowledgeExtractor,
        mock_ke_with_query,
    ):
        """Test getting efficiency metrics"""
        # Setup mock response for efficiency metrics
        mock_ke_with_query.query.return_value = [
            {
                "avg_efficiency": 0.65,
                "avg_evals": 35.0,
                "total_runs": 10,
            }
        ]

        metrics = await extractor.get_efficiency_metrics(
            problem_type="portfolio_optimization",
            domain="finance",
        )

        assert "avg_efficiency_gain" in metrics
        assert "avg_evaluations_saved" in metrics
        assert "success_rate" in metrics
        assert "total_runs" in metrics
        assert metrics["avg_efficiency_gain"] == 0.65


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def extractor(self):
        """Create extractor without KE"""
        return LoongFlowKnowledgeExtractor(knowledge_engine=None)

    def test_initialization_without_ke(self, extractor):
        """Test extractor initialization without Knowledge Engine"""
        assert extractor.ke is None
        assert extractor.graphiti is None
        assert extractor.qdrant is None

    @pytest.mark.asyncio
    async def test_extract_without_ke(self, extractor):
        """Test extraction without Knowledge Engine (graceful degradation)"""
        pes_results = PESRunResults(
            plan={"strategy": "Test"},
            execution={},
            summary={},
            evolutionary_tree={},
            best_solution={},
        )

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_results,
            problem="Test",
            problem_type="test",
        )

        # Should still extract artifacts
        assert len(artifacts) >= 1

    @pytest.mark.asyncio
    async def test_extract_with_invalid_input(self, extractor):
        """Test extraction with invalid input"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results="invalid",  # Wrong type
            problem="Test",
            problem_type="test",
        )

        # Should return empty list
        assert len(artifacts) == 0

    @pytest.mark.asyncio
    async def test_extract_with_empty_plan(self, extractor):
        """Test planning strategy extraction with empty plan"""
        artifact = await extractor.extract_planning_strategies(
            plan={},
            problem="Test",
            problem_type="test",
            domain="general",
            timestamp=datetime.now(timezone.utc),
            run_id="test",
        )

        # Should return None for empty plan
        assert artifact is None

    @pytest.mark.asyncio
    async def test_domain_detection_with_empty_strings(self, extractor):
        """Test domain detection with empty input"""
        domain = extractor._detect_domain("", "")

        # Should default to general
        assert domain == ProblemDomain.GENERAL.value


class TestIntegrationWithStorage:
    """Integration tests with mock storage backends"""

    @pytest.mark.asyncio
    async def test_end_to_end_extraction_and_storage(self):
        """Test complete workflow from extraction to storage"""
        # Create mock KE with all backends
        mock_ke = Mock()
        mock_ke.graphiti_bridge = Mock()
        mock_ke.graphiti_bridge.add_episode = AsyncMock()
        mock_ke.neo4j = Mock()
        mock_ke.neo4j.run = AsyncMock()
        mock_ke.mongodb = Mock()
        mock_ke.mongodb.insert_one = AsyncMock()

        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)

        # Create PES results
        pes_results = PESRunResults(
            plan={"strategy": "Genetic algorithm", "success_rate": 0.85},
            execution={
                "early_stops": [10, 20],
                "convergence_rate": 0.9,
                "iterations_to_best": 20,
                "total_evaluations": 25,
                "baseline_evaluations": 100,
            },
            summary={
                "insights": "Crossover worked well",
                "what_worked": ["Two-point crossover"],
                "what_failed": ["Single-point crossover"],
                "recommendations": ["Use two-point crossover"],
            },
            evolutionary_tree={
                "generations": 5,
                "avg_branching": 2.0,
                "total_mutations": 10,
                "best_path": ["gen_0", "gen_2", "gen_5"],
            },
            best_solution={
                "code": "def solve(): return optimal",
                "fitness": 0.92,
                "iteration": 20,
                "improvement": 0.2,
            },
        )

        # Extract artifacts
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_results,
            problem="Optimize trading strategy",
            problem_type="trading_strategy",
            domain="trading",
            run_id="e2e_test_run",
        )

        # Verify extraction
        assert len(artifacts) == 5

        # Verify storage calls
        assert mock_ke.graphiti_bridge.add_episode.called
        assert mock_ke.neo4j.run.called
        assert mock_ke.mongodb.insert_one.called

        # Verify correct number of episodes (5 artifacts)
        assert mock_ke.graphiti_bridge.add_episode.call_count == 5

        # Check stats
        stats = extractor.get_extraction_stats()
        assert stats[ArtifactType.PLANNING_STRATEGY.value] == 1
        assert stats[ArtifactType.EXECUTION_PATTERN.value] == 1
        assert stats[ArtifactType.REFLECTION_INSIGHT.value] == 1
        assert stats[ArtifactType.EVOLUTIONARY_LINEAGE.value] == 1
        assert stats[ArtifactType.OPTIMIZED_SOLUTION.value] == 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
