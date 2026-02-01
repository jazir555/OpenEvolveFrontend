"""
Integration Tests for LoongFlow PES Knowledge Extraction

Tests the complete workflow of extracting knowledge artifacts from
LoongFlow PES runs and storing them in the Knowledge Engine.

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
from typing import Dict, Any, List

from knowledge_engine.integrations.loongflow_integration import (
    LoongFlowKnowledgeExtractor,
    PESRunResults,
    ProblemDomain,
    create_loongflow_extractor,
)


class MockKnowledgeEngine:
    """Mock Knowledge Engine for testing"""

    def __init__(self):
        self.stored_artifacts: List[Dict[str, Any]] = []

    async def store_artifact(self, artifact: Dict[str, Any]):
        """Store an artifact"""
        self.stored_artifacts.append(artifact)

    async def query(self, query: str) -> List[Dict[str, Any]]:
        """Mock query method"""
        return []

    async def search(self, query_text: str, filters: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Mock search method"""
        return []


@pytest.fixture
def mock_ke():
    """Create a mock Knowledge Engine"""
    return MockKnowledgeEngine()


@pytest.fixture
def extractor(mock_ke):
    """Create a LoongFlow extractor with mock KE"""
    return LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)


@pytest.fixture
def sample_pes_run() -> Dict[str, Any]:
    """Sample PES run results for testing"""
    return {
        "plan": {
            "strategy": "Use gradient descent with momentum and adaptive learning rate",
            "success_rate": 0.85,
            "iterations": 50,
            "approach": "hybrid_evolutionary",
        },
        "execution": {
            "early_stops": [15, 25],
            "convergence_rate": 0.95,
            "iterations_to_best": 25,
            "total_evaluations": 30,
            "efficiency_gain": 0.60,
            "time_saved": 1200,
        },
        "summary": {
            "insights": "Momentum helps escape local optima effectively",
            "what_worked": ["momentum", "adaptive_learning_rate", "early_stopping"],
            "what_failed": ["fixed_learning_rate", "large_batch_size"],
            "recommendations": [
                "Use momentum in future runs",
                "Implement adaptive learning rate scheduling",
            ],
        },
        "evolutionary_tree": {
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 45,
            "best_lineage": ["root", "gen1", "gen2", "gen3", "best"],
        },
        "best_solution": {
            "code": """
def optimize_portfolio(weights, returns, risk_tolerance):
    # Momentum-based optimization
    velocity = np.zeros_like(weights)
    momentum = 0.9

    for i in range(100):
        gradient = compute_gradient(weights, returns)
        velocity = momentum * velocity + 0.01 * gradient
        weights = weights - velocity

    return weights
            """,
            "fitness": 0.95,
            "iteration": 25,
            "improvement": 0.40,
        },
    }


class TestLoongFlowKnowledgeExtractor:
    """Test suite for LoongFlow Knowledge Extractor"""

    def test_initialization(self, mock_ke):
        """Test extractor initialization"""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)

        assert extractor.ke == mock_ke
        assert extractor.artifact_counts["planning_strategy"] == 0
        assert extractor.artifact_counts["execution_pattern"] == 0
        assert extractor.artifact_counts["reflection_insight"] == 0
        assert extractor.artifact_counts["evolutionary_lineage"] == 0
        assert extractor.artifact_counts["optimized_solution"] == 0

    def test_create_extractor_function(self):
        """Test convenience function for creating extractor"""
        extractor = create_loongflow_extractor()

        assert extractor is not None
        assert isinstance(extractor, LoongFlowKnowledgeExtractor)
        assert extractor.ke is None

    @pytest.mark.asyncio
    async def test_extract_all_artifacts(self, extractor, sample_pes_run):
        """Test extracting all 5 artifact types"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Optimize neural network training",
            problem_type="scientific",
        )

        # Should extract 5 artifacts
        assert len(artifacts) == 5

        # Check artifact types (KnowledgeArtifact objects, not dicts)
        artifact_types = [a.artifact_type for a in artifacts]
        assert "planning_strategy" in artifact_types
        assert "execution_pattern" in artifact_types
        assert "reflection_insight" in artifact_types
        assert "evolutionary_lineage" in artifact_types
        assert "optimized_solution" in artifact_types

    @pytest.mark.asyncio
    async def test_planning_strategy_artifact(self, extractor, sample_pes_run):
        """Test planning strategy artifact extraction"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Portfolio optimization",
            problem_type="finance",
        )

        planning = next(a for a in artifacts if a["artifact_type"] == "planning_strategy")

        # Check structure
        assert "id" in planning
        assert planning["artifact_type"] == "planning_strategy"
        assert planning["source"] == "loongflow_pes"
        assert planning["confidence"] == 0.8

        # Check content
        assert "gradient descent" in planning["content"].lower()
        assert "momentum" in planning["content"].lower()

        # Check metadata
        assert planning["metadata"]["problem"] == "Portfolio optimization"
        assert planning["metadata"]["problem_type"] == "finance"
        assert planning["metadata"]["success_rate"] == 0.85
        assert planning["metadata"]["iterations_planned"] == 50

    @pytest.mark.asyncio
    async def test_execution_pattern_artifact(self, extractor, sample_pes_run):
        """Test execution pattern artifact extraction"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Algorithm optimization",
            problem_type="scientific",
        )

        execution = next(a for a in artifacts if a["artifact_type"] == "execution_pattern")

        # Check structure
        assert execution["artifact_type"] == "execution_pattern"
        assert execution["source"] == "loongflow_pes"
        assert execution["confidence"] == 0.9

        # Check metadata
        assert execution["metadata"]["efficiency_gain"] == 0.60
        assert execution["metadata"]["time_saved_seconds"] == 1200
        assert execution["metadata"]["early_stop_count"] == 2

        # Check content has execution data
        content = execution["content"]
        assert "early_stopping_events" in content
        assert "convergence_rate" in content

    @pytest.mark.asyncio
    async def test_reflection_insight_artifact(self, extractor, sample_pes_run):
        """Test reflection insight artifact extraction"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Neural network training",
            problem_type="machine_learning",
        )

        reflection = next(a for a in artifacts if a["artifact_type"] == "reflection_insight")

        # Check structure
        assert reflection["artifact_type"] == "reflection_insight"
        assert reflection["source"] == "loongflow_pes"
        assert reflection["confidence"] == 0.7

        # Check content
        assert "momentum" in reflection["content"].lower()
        assert "local optima" in reflection["content"].lower()

        # Check metadata
        assert "momentum" in reflection["metadata"]["what_worked"]
        assert "adaptive_learning_rate" in reflection["metadata"]["what_worked"]
        assert "fixed_learning_rate" in reflection["metadata"]["what_failed"]
        assert len(reflection["metadata"]["recommendations"]) > 0

    @pytest.mark.asyncio
    async def test_evolutionary_lineage_artifact(self, extractor, sample_pes_run):
        """Test evolutionary lineage artifact extraction"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Mathematical optimization",
            problem_type="mathematics",
        )

        lineage = next(a for a in artifacts if a["artifact_type"] == "evolutionary_lineage")

        # Check structure
        assert lineage["artifact_type"] == "evolutionary_lineage"
        assert lineage["source"] == "loongflow_pes"
        assert lineage["confidence"] == 0.8

        # Check metadata
        assert lineage["metadata"]["generations"] == 10
        assert lineage["metadata"]["branching_factor"] == 2.5
        assert lineage["metadata"]["total_mutations"] == 45

    @pytest.mark.asyncio
    async def test_optimized_solution_artifact(self, extractor, sample_pes_run):
        """Test optimized solution artifact extraction"""
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Portfolio optimization",
            problem_type="finance",
        )

        solution = next(a for a in artifacts if a["artifact_type"] == "optimized_solution")

        # Check structure
        assert solution["artifact_type"] == "optimized_solution"
        assert solution["source"] == "loongflow_pes"
        assert solution["confidence"] == 0.9

        # Check content has code
        assert "def optimize_portfolio" in solution["content"]
        assert "momentum" in solution["content"]

        # Check metadata
        assert solution["metadata"]["fitness"] == 0.95
        assert solution["metadata"]["iteration"] == 25
        assert solution["metadata"]["improvement_over_baseline"] == 0.40

    @pytest.mark.asyncio
    async def test_artifact_storage_in_ke(self, extractor, sample_pes_run, mock_ke):
        """Test that artifacts are stored in Knowledge Engine"""
        await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Test problem",
            problem_type="test",
        )

        # Check KE received artifacts
        assert len(mock_ke.stored_artifacts) == 5

        # Verify each artifact has required fields
        for artifact in mock_ke.stored_artifacts:
            assert "id" in artifact
            assert "content" in artifact
            assert "artifact_type" in artifact
            assert "valid_at" in artifact
            assert "source" in artifact
            assert "metadata" in artifact
            assert "confidence" in artifact

    @pytest.mark.asyncio
    async def test_temporal_metadata(self, extractor, sample_pes_run):
        """Test that artifacts have correct temporal metadata"""
        before = datetime.now(timezone.utc)

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Test",
            problem_type="test",
        )

        after = datetime.now(timezone.utc)

        # Check all artifacts have timestamps
        for artifact in artifacts:
            valid_at = datetime.fromisoformat(artifact["valid_at"])
            created_at = datetime.fromisoformat(artifact["created_at"])

            assert before <= valid_at <= after
            assert before <= created_at <= after
            assert artifact["invalid_at"] is None  # Still valid

    @pytest.mark.asyncio
    async def test_partial_pes_run(self, extractor):
        """Test handling of partial/missing PES data"""
        partial_run = {
            "plan": {"strategy": "Simple approach", "success_rate": 0.5},
            # Missing other sections
        }

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=partial_run,
            problem="Partial test",
            problem_type="test",
        )

        # Should only extract planning strategy
        assert len(artifacts) == 1
        assert artifacts[0]["artifact_type"] == "planning_strategy"

    @pytest.mark.asyncio
    async def test_invalid_pes_run(self, extractor):
        """Test handling of invalid PES run data"""
        # Not a dict
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=None,  # type: ignore
            problem="Invalid test",
            problem_type="test",
        )

        assert len(artifacts) == 0

    @pytest.mark.asyncio
    async def test_extraction_stats(self, extractor, sample_pes_run):
        """Test extraction statistics tracking"""
        # Reset stats
        extractor.reset_stats()

        # Extract artifacts
        await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Test",
            problem_type="test",
        )

        # Check stats
        stats = extractor.get_extraction_stats()
        assert stats["planning_strategy"] == 1
        assert stats["execution_pattern"] == 1
        assert stats["reflection_insight"] == 1
        assert stats["evolutionary_lineage"] == 1
        assert stats["optimized_solution"] == 1

    @pytest.mark.asyncio
    async def test_query_planning_strategies(self, extractor, sample_pes_run, mock_ke):
        """Test querying planning strategies"""
        # First store some artifacts
        await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Portfolio optimization",
            problem_type="finance",
        )

        # Query strategies
        strategies = await extractor.query_planning_strategies(
            problem_type="finance",
            limit=10,
        )

        # Mock KE returns empty, but should not error
        assert isinstance(strategies, list)

    @pytest.mark.asyncio
    async def test_get_efficiency_metrics(self, extractor, sample_pes_run, mock_ke):
        """Test getting efficiency metrics"""
        # First store execution pattern
        await extractor.extract_from_pes_run(
            pes_run_results=sample_pes_run,
            problem="Optimization",
            problem_type="scientific",
        )

        # Get metrics
        metrics = await extractor.get_efficiency_metrics(problem_type="scientific")

        # Mock KE returns empty, but should not error
        assert isinstance(metrics, dict)

    def test_pes_run_results_dataclass(self):
        """Test PESRunResults dataclass"""
        pes_data = {
            "plan": {"strategy": "test"},
            "execution": {"early_stops": [1, 2]},
            "summary": {"insights": "test"},
            "evolutionary_tree": {"generations": 5},
            "best_solution": {"code": "def test(): pass"},
        }

        pes_run = PESRunResults.from_dict(pes_data)

        assert pes_run.plan == pes_data["plan"]
        assert pes_run.execution == pes_data["execution"]
        assert pes_run.summary == pes_data["summary"]
        assert pes_run.evolutionary_tree == pes_data["evolutionary_tree"]
        assert pes_run.best_solution == pes_data["best_solution"]

        # Test to_dict
        pes_dict = pes_run.to_dict()
        assert pes_dict["plan"] == pes_data["plan"]
        assert pes_dict["execution"] == pes_data["execution"]


class TestLoongFlowIntegrationEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_strategy_string(self, mock_ke):
        """Test handling of empty strategy string"""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)

        pes_run = {
            "plan": {"strategy": "", "success_rate": 0.0},
            "execution": {},
            "summary": {},
            "evolutionary_tree": {},
            "best_solution": {},
        }

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_run,
            problem="Test",
            problem_type="test",
        )

        # Should still create artifacts, even with empty content
        assert len(artifacts) > 0

    @pytest.mark.asyncio
    async def test_dict_strategy_conversion(self, mock_ke):
        """Test that dict strategies are converted to JSON strings"""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)

        pes_run = {
            "plan": {
                "strategy": {"method": "gradient_descent", "lr": 0.01, "momentum": 0.9},
                "success_rate": 0.8,
            },
            "execution": {},
            "summary": {},
            "evolutionary_tree": {},
            "best_solution": {},
        }

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_run,
            problem="Test",
            problem_type="test",
        )

        planning = next((a for a in artifacts if a["artifact_type"] == "planning_strategy"), None)
        assert planning is not None
        assert isinstance(planning["content"], str)
        assert "gradient_descent" in planning["content"]

    @pytest.mark.asyncio
    async def test_extractor_without_ke(self):
        """Test extractor works without Knowledge Engine"""
        extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)

        pes_run = {
            "plan": {"strategy": "Test strategy", "success_rate": 0.8},
            "execution": {},
            "summary": {},
            "evolutionary_tree": {},
            "best_solution": {},
        }

        # Should not error, just not store
        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_run,
            problem="Test",
            problem_type="test",
        )

        assert len(artifacts) > 0

    @pytest.mark.asyncio
    async def test_multiple_extractions(self, extractor, sample_pes_run):
        """Test multiple extraction runs accumulate stats"""
        extractor.reset_stats()

        # Run extraction 3 times
        for _ in range(3):
            await extractor.extract_from_pes_run(
                pes_run_results=sample_pes_run,
                problem="Test",
                problem_type="test",
            )

        stats = extractor.get_extraction_stats()
        assert stats["planning_strategy"] == 3
        assert stats["execution_pattern"] == 3
        assert stats["reflection_insight"] == 3
        assert stats["evolutionary_lineage"] == 3
        assert stats["optimized_solution"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
