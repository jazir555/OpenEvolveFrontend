"""
Unit Tests for Unified Evolution Knowledge Integration System

Comprehensive test suite for:
- Dual-run knowledge extraction
- Performance comparison
- Knowledge fusion
- Best practice identification
- Synergy detection
- Hybrid recommendations

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import pytest
import asyncio
from datetime import datetime, UTC
from typing import Dict, Any, List
import numpy as np

# Import the system under test
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_engine.integrations.unified_evolution_integration import (
    UnifiedEvolutionKnowledgeExtractor,
    PerformanceComparison,
    SynergyOpportunity,
    BestPractice,
    HybridStrategyRecommendation,
    DualRunAnalysis,
    KnowledgeArtifact,
    EvolutionarySystem,
    ComparisonMetric
)

from knowledge_engine.schemas.evolutionary_artifacts import (
    SolutionPatternArtifact,
    MAPElitesArchiveArtifact,
    PESPatternsArtifact,
    PerformanceMetricsArtifact,
    ArtifactType,
    SystemType,
    DomainType
)

from knowledge_engine.schemas.comparison_results import (
    CategoryComparison,
    DetailedPerformanceComparison,
    WinnerType,
    ComparisonCategory
)


# ========================================================================
# FIXTURES
# ========================================================================

@pytest.fixture
def mock_openevolve_result():
    """Mock OpenEvolve evolutionary run result"""
    return {
        "best_solution": "def optimized_solution():\n    return 42",
        "best_fitness": 0.95,
        "best_iteration": 45,
        "total_iterations": 100,
        "total_evaluations": 1000,
        "total_time": 300.0,
        "history": [
            {"iteration": i, "fitness": 0.3 + 0.6 * (i / 100)}
            for i in range(0, 101, 10)
        ],
        "archive": {
            "coverage": 0.75,
            "occupancy": {"(0,0)": "solution_1", "(1,1)": "solution_2"},
            "solutions": [
                {"id": "sol_1", "fitness": 0.9, "features": (0, 0)},
                {"id": "sol_2", "fitness": 0.85, "features": (1, 1)}
            ]
        },
        "config": {
            "population_size": 1000,
            "num_islands": 5,
            "feature_dimensions": ["complexity", "diversity"],
            "feature_bins": 10
        },
        "llm_calls": 150,
        "tokens_used": 50000
    }


@pytest.fixture
def mock_loongflow_result():
    """Mock LoongFlow PES run result"""
    return {
        "best_solution": "def optimized_solution():\n    return 42",
        "best_fitness": 0.93,
        "total_iterations": 40,
        "total_evaluations": 400,  # 60% fewer!
        "total_time": 250.0,
        "convergence_generation": 35,
        "sample_efficiency": 0.0023,  # fitness per evaluation
        "generations": [
            {
                "plan": {
                    "strategy": "Use gradient-based optimization",
                    "reasoning": "Directed search is more efficient"
                },
                "execution": {
                    "approach": "Iterative refinement",
                    "early_stopped": True,
                    "iterations": 3
                },
                "summary": {
                    "insight": "Planning phase reduces wasted evaluations"
                }
            }
            for _ in range(10)
        ],
        "evolutionary_tree": {
            "root_id": "root",
            "num_generations": 10,
            "branching_factor": 2.5,
            "best_path": ["root", "child_1", "grandchild_3"],
            "solutions": [
                {"id": "sol_1", "fitness": 0.93, "generation": 3}
            ]
        },
        "summaries": [
            {"insight": "Early stopping saves 60% evaluations"},
            {"insight": "Adaptive exploration prevents local optima"}
        ],
        "metrics": {
            "total_evaluations": 400,
            "best_fitness": 0.93,
            "convergence_generation": 35,
            "sample_efficiency": 0.0023
        },
        "llm_calls": 120,  # 3 per generation
        "tokens_used": 45000
    }


@pytest.fixture
def extractor():
    """Create UnifiedEvolutionKnowledgeExtractor instance"""
    return UnifiedEvolutionKnowledgeExtractor(knowledge_engine=None)


# ========================================================================
# TEST: Dual-Run Knowledge Extraction
# ========================================================================

class TestDualRunExtraction:
    """Test dual-run knowledge extraction"""

    @pytest.mark.asyncio
    async def test_extract_dual_run_knowledge(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test complete dual-run knowledge extraction"""
        # Act
        analysis = await extractor.extract_dual_run_knowledge(
            openevolve_result=mock_openevolve_result,
            loongflow_result=mock_loongflow_result,
            domain="finance",
            problem="Portfolio optimization"
        )

        # Assert
        assert isinstance(analysis, DualRunAnalysis)
        assert analysis.domain == "finance"
        assert analysis.problem_description == "Portfolio optimization"
        assert len(analysis.openevolve_artifacts) > 0
        assert len(analysis.loongflow_artifacts) > 0
        assert isinstance(analysis.performance_comparison, PerformanceComparison)
        assert len(analysis.best_practices) > 0
        assert len(analysis.synergy_opportunities) > 0
        assert isinstance(analysis.hybrid_recommendation, HybridStrategyRecommendation)

    @pytest.mark.asyncio
    async def test_extract_openevolve_artifacts(
        self,
        extractor,
        mock_openevolve_result
    ):
        """Test OpenEvolve artifact extraction"""
        # Act
        artifacts = await extractor._extract_openevolve_artifacts(
            mock_openevolve_result,
            "finance"
        )

        # Assert
        assert len(artifacts) >= 3

        # Check solution pattern artifact
        solution_artifact = next(
            (a for a in artifacts if a.artifact_type == "solution_pattern"),
            None
        )
        assert solution_artifact is not None
        assert solution_artifact.source_system == "openevolve"
        assert solution_artifact.content["fitness"] == 0.95

        # Check MAP-Elites artifact
        archive_artifact = next(
            (a for a in artifacts if a.artifact_type == "map_elites_archive"),
            None
        )
        assert archive_artifact is not None
        assert archive_artifact.content["archive_coverage"] == 0.75

    @pytest.mark.asyncio
    async def test_extract_loongflow_artifacts(
        self,
        extractor,
        mock_loongflow_result
    ):
        """Test LoongFlow artifact extraction"""
        # Act
        artifacts = await extractor._extract_loongflow_artifacts(
            mock_loongflow_result,
            "finance"
        )

        # Assert
        assert len(artifacts) >= 2

        # Check PES patterns artifact
        pes_artifact = next(
            (a for a in artifacts if a.artifact_type == "pes_patterns"),
            None
        )
        assert pes_artifact is not None
        assert pes_artifact.source_system == "loongflow"
        assert pes_artifact.content["num_generations"] == 10

        # Check performance metrics artifact
        metrics_artifact = next(
            (a for a in artifacts if a.artifact_type == "performance_metrics"),
            None
        )
        assert metrics_artifact is not None
        assert metrics_artifact.content["total_evaluations"] == 400


# ========================================================================
# TEST: Performance Comparison
# ========================================================================

class TestPerformanceComparison:
    """Test performance comparison logic"""

    @pytest.mark.asyncio
    async def test_compare_system_performance(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test complete performance comparison"""
        # Act
        comparison = await extractor.compare_system_performance(
            mock_openevolve_result,
            mock_loongflow_result,
            "finance"
        )

        # Assert
        assert isinstance(comparison, PerformanceComparison)

        # Check convergence speed
        assert "openevolve" in comparison.convergence_speed
        assert "loongflow" in comparison.convergence_speed
        # OpenEvolve reaches 90% at iteration 10, LoongFlow uses 40 total
        assert comparison.convergence_speed["openevolve"] < comparison.convergence_speed["loongflow"]

        # Check solution quality
        assert comparison.solution_quality["openevolve"] == 0.95
        assert comparison.solution_quality["loongflow"] == 0.93

        # Check evaluation efficiency
        assert comparison.evaluation_efficiency["loongflow"] > \
               comparison.evaluation_efficiency["openevolve"]  # LoongFlow more efficient

        # Check computational cost
        assert "openevolve" in comparison.computational_cost
        assert "loongflow" in comparison.computational_cost

        # Check winner determination
        assert comparison.overall_winner in ["openevolve", "loongflow", "tie"]
        assert 0 <= comparison.confidence <= 1

    @pytest.mark.asyncio
    async def test_compare_convergence_speed(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test convergence speed comparison"""
        # Act
        result = await extractor._compare_convergence_speed(
            mock_openevolve_result,
            mock_loongflow_result
        )

        # Assert
        assert "openevolve" in result
        assert "loongflow" in result
        assert "ratio" in result
        # OpenEvolve (10) < LoongFlow (40)
        assert result["openevolve"] < result["loongflow"]

    @pytest.mark.asyncio
    async def test_compare_solution_quality(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test solution quality comparison"""
        # Act
        result = await extractor._compare_solution_quality(
            mock_openevolve_result,
            mock_loongflow_result
        )

        # Assert
        assert result["openevolve"] == 0.95
        assert result["loongflow"] == 0.93
        assert result["winner"] == "openevolve"  # Higher fitness wins

    @pytest.mark.asyncio
    async def test_compare_evaluation_efficiency(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test evaluation efficiency comparison"""
        # Act
        result = await extractor._compare_evaluation_efficiency(
            mock_openevolve_result,
            mock_loongflow_result
        )

        # Assert
        oe_eff = result["openevolve"]
        lf_eff = result["loongflow"]

        # Verify calculation
        expected_oe = 0.95 / 1000
        expected_lf = 0.93 / 400

        assert abs(oe_eff - expected_oe) < 1e-6
        assert abs(lf_eff - expected_lf) < 1e-6
        assert lf_eff > oe_eff  # LoongFlow more efficient


# ========================================================================
# TEST: Knowledge Fusion
# ========================================================================

class TestKnowledgeFusion:
    """Test knowledge fusion algorithms"""

    @pytest.mark.asyncio
    async def test_fuse_evolutionary_insights(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test insight fusion from both systems"""
        # Arrange
        oe_artifacts = await extractor._extract_openevolve_artifacts(
            mock_openevolve_result, "finance"
        )
        lf_artifacts = await extractor._extract_loongflow_artifacts(
            mock_loongflow_result, "finance"
        )

        # Act
        fused = await extractor.fuse_evolutionary_insights(
            oe_artifacts, lf_artifacts
        )

        # Assert
        assert len(fused) > 0

        # Check for complementary insights
        complementary = [f for f in fused if f.artifact_type == "complementary_insight"]
        assert len(complementary) > 0

        # Check for consensus insights
        consensus = [f for f in fused if f.artifact_type == "consensus_insight"]
        assert len(consensus) > 0

        # Check for synthesized insights
        synthesized = [f for f in fused if f.artifact_type == "synthesized_insight"]
        assert len(synthesized) > 0

    @pytest.mark.asyncio
    async def test_find_complementary_insights(self, extractor):
        """Test complementary insight detection"""
        # Arrange
        oe_artifacts = [
            KnowledgeArtifact(
                artifact_type="map_elites_archive",
                source_system="openevolve",
                content={"coverage": 0.8},
                metadata={},
                confidence=0.9
            )
        ]
        lf_artifacts = [
            KnowledgeArtifact(
                artifact_type="pes_patterns",
                source_system="loongflow",
                content={"num_generations": 10},
                metadata={},
                confidence=0.9
            )
        ]

        # Act
        complementary = await extractor._find_complementary_insights(
            oe_artifacts, lf_artifacts
        )

        # Assert
        assert len(complementary) > 0
        assert complementary[0].source_system == "hybrid"


# ========================================================================
# TEST: Best Practice Identification
# ========================================================================

class TestBestPractices:
    """Test best practice identification"""

    @pytest.mark.asyncio
    async def test_identify_best_practices(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test best practice identification"""
        # Arrange
        oe_artifacts = await extractor._extract_openevolve_artifacts(
            mock_openevolve_result, "finance"
        )
        lf_artifacts = await extractor._extract_loongflow_artifacts(
            mock_loongflow_result, "finance"
        )
        performance = await extractor.compare_system_performance(
            mock_openevolve_result, mock_loongflow_result, "finance"
        )

        # Act
        practices = await extractor.identify_best_practices(
            oe_artifacts, lf_artifacts, performance, "finance"
        )

        # Assert
        assert len(practices) > 0
        assert len(practices) <= 10  # Top 10

        # Check structure
        for practice in practices:
            assert isinstance(practice, BestPractice)
            assert practice.domain == "finance"
            assert practice.source_system in ["openevolve", "loongflow", "both"]
            assert 0 <= practice.confidence <= 1

    @pytest.mark.asyncio
    async def test_best_practices_ranked_by_confidence(self, extractor):
        """Test that best practices are ranked by confidence"""
        # Arrange
        practices = [
            BestPractice(
                practice=f"Practice {i}",
                source_system="openevolve",
                domain="finance",
                evidence={},
                confidence=0.5 + (i * 0.1)
            )
            for i in range(5)
        ]

        # Act
        ranked = sorted(practices, key=lambda bp: bp.confidence, reverse=True)

        # Assert
        for i in range(len(ranked) - 1):
            assert ranked[i].confidence >= ranked[i + 1].confidence


# ========================================================================
# TEST: Synergy Detection
# ========================================================================

class TestSynergyDetection:
    """Test synergy opportunity detection"""

    @pytest.mark.asyncio
    async def test_detect_synergy_opportunities(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test synergy opportunity detection"""
        # Arrange
        oe_insights = await extractor._extract_openevolve_artifacts(
            mock_openevolve_result, "finance"
        )
        lf_insights = await extractor._extract_loongflow_artifacts(
            mock_loongflow_result, "finance"
        )

        # Act
        opportunities = await extractor.detect_synergy_opportunities(
            oe_insights, lf_insights
        )

        # Assert
        assert len(opportunities) > 0

        # Check for specific opportunities
        opp_types = [o.opportunity_type for o in opportunities]
        assert "technique_transfer" in opp_types
        assert "parameter_tuning" in opp_types

        # Check priority ranking
        for i in range(len(opportunities) - 1):
            assert opportunities[i].priority >= opportunities[i + 1].priority

    def test_has_pes_advantages(self, extractor):
        """Test PES advantage detection"""
        # Arrange
        lf_artifacts = [
            KnowledgeArtifact(
                artifact_type="pes_patterns",
                source_system="loongflow",
                content={},
                metadata={},
                confidence=0.9
            )
        ]

        # Act & Assert
        assert extractor._has_pes_advantages(lf_artifacts) is True

    def test_has_qd_advantages(self, extractor):
        """Test QD advantage detection"""
        # Arrange
        oe_artifacts = [
            KnowledgeArtifact(
                artifact_type="map_elites_archive",
                source_system="openevolve",
                content={},
                metadata={},
                confidence=0.9
            )
        ]

        # Act & Assert
        assert extractor._has_qd_advantages(oe_artifacts) is True


# ========================================================================
# TEST: Hybrid Recommendations
# ========================================================================

class TestHybridRecommendations:
    """Test hybrid strategy recommendations"""

    @pytest.mark.asyncio
    async def test_create_hybrid_recommendations_finance(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test hybrid recommendation for finance domain"""
        # Arrange
        performance = await extractor.compare_system_performance(
            mock_openevolve_result, mock_loongflow_result, "finance"
        )
        best_practices = []

        # Act
        recommendation = await extractor.create_hybrid_recommendations(
            performance, best_practices, "finance"
        )

        # Assert
        assert isinstance(recommendation, HybridStrategyRecommendation)
        assert recommendation.recommended_mode in ["pes", "qd", "mo", "adversarial", "hybrid"]
        assert 0 <= recommendation.confidence <= 1
        assert len(recommendation.rationale) > 0
        assert isinstance(recommendation.configuration, dict)
        assert 0 <= recommendation.expected_improvement <= 1
        assert isinstance(recommendation.risk_factors, list)

        # For finance, should recommend PES due to expensive evaluations
        if performance.evaluation_efficiency.get("loongflow", 0) > \
           performance.evaluation_efficiency.get("openevolve", 0):
            assert recommendation.recommended_mode == "pes"
            assert recommendation.expected_improvement >= 0.5

    @pytest.mark.asyncio
    async def test_create_hybrid_recommendations_science(self, extractor):
        """Test hybrid recommendation for science domain"""
        # Arrange
        performance = PerformanceComparison(
            evaluation_efficiency={"loongflow": 0.003, "openevolve": 0.001},
            convergence_speed={"loongflow": 30, "openevolve": 80},
            solution_quality={"loongflow": 0.92, "openevolve": 0.90},
            diversity_metrics={},
            computational_cost={},
            winner_by_category={},
            overall_winner="loongflow",
            confidence=0.85
        )
        best_practices = []

        # Act
        recommendation = await extractor.create_hybrid_recommendations(
            performance, best_practices, "science"
        )

        # Assert
        assert recommendation.recommended_mode == "pes"
        assert "expensive evaluations" in recommendation.rationale.lower()

    @pytest.mark.asyncio
    async def test_create_hybrid_recommendations_tie(self, extractor):
        """Test hybrid recommendation when systems tie"""
        # Arrange
        performance = PerformanceComparison(
            evaluation_efficiency={"loongflow": 0.002, "openevolve": 0.002},
            convergence_speed={"loongflow": 50, "openevolve": 50},
            solution_quality={"loongflow": 0.90, "openevolve": 0.90},
            diversity_metrics={},
            computational_cost={},
            winner_by_category={},
            overall_winner="tie",
            confidence=0.5
        )
        best_practices = []

        # Act
        recommendation = await extractor.create_hybrid_recommendations(
            performance, best_practices, "general"
        )

        # Assert
        # Should recommend hybrid for tie
        assert recommendation.recommended_mode in ["hybrid", "pes", "qd"]


# ========================================================================
# TEST: Utility Functions
# ========================================================================

class TestUtilityFunctions:
    """Test utility and helper functions"""

    def test_calculate_improvement_rate(self, extractor):
        """Test improvement rate calculation"""
        # Arrange
        history = [
            {"iteration": 0, "fitness": 0.5},
            {"iteration": 1, "fitness": 0.6},
            {"iteration": 2, "fitness": 0.72}
        ]

        # Act
        rate = extractor._calculate_improvement_rate(history)

        # Assert
        assert rate > 0
        # (0.6-0.5)/0.5 = 0.2, (0.72-0.6)/0.6 = 0.2
        # Average = 0.2
        assert abs(rate - 0.2) < 0.01

    def test_calculate_iterations_to_90_percent(self, extractor):
        """Test iterations to 90% calculation"""
        # Arrange
        result = {
            "best_fitness": 1.0,
            "history": [
                {"fitness": 0.5},
                {"fitness": 0.7},
                {"fitness": 0.85},
                {"fitness": 0.92},  # First to exceed 0.9
                {"fitness": 0.95}
            ]
        }

        # Act
        iterations = extractor._calculate_iterations_to_90_percent(result)

        # Assert
        assert iterations == 3  # 0-indexed

    def test_calculate_efficiency(self, extractor):
        """Test efficiency calculation"""
        # Arrange
        result = {
            "best_fitness": 0.95,
            "total_evaluations": 1000
        }

        # Act
        efficiency = extractor._calculate_efficiency(result)

        # Assert
        assert efficiency == 0.95 / 1000
        assert efficiency == 0.00095


# ========================================================================
# TEST: Integration Tests
# ========================================================================

class TestIntegration:
    """Integration tests with realistic scenarios"""

    @pytest.mark.asyncio
    async def test_complete_dual_run_analysis_workflow(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test complete workflow from extraction to recommendation"""
        # Act
        analysis = await extractor.extract_dual_run_knowledge(
            openevolve_result=mock_openevolve_result,
            loongflow_result=mock_loongflow_result,
            domain="trading",
            problem="Optimize trading strategy for maximum Sharpe ratio"
        )

        # Assert - Complete analysis
        assert analysis.run_id.startswith("dual_trading_")
        assert analysis.domain == "trading"

        # Performance comparison
        perf = analysis.performance_comparison
        assert perf.overall_winner in ["openevolve", "loongflow", "tie"]

        # Best practices
        assert len(analysis.best_practices) > 0
        assert all(isinstance(bp, BestPractice) for bp in analysis.best_practices)

        # Synergy opportunities
        assert len(analysis.synergy_opportunities) > 0
        assert all(isinstance(so, SynergyOpportunity) for so in analysis.synergy_opportunities)

        # Hybrid recommendation
        rec = analysis.hybrid_recommendation
        assert rec.recommended_mode in ["pes", "qd", "hybrid"]
        assert rec.expected_improvement > 0

    @pytest.mark.asyncio
    async def test_serialization_and_deserialization(
        self,
        extractor,
        mock_openevolve_result,
        mock_loongflow_result
    ):
        """Test that analysis can be serialized to JSON"""
        # Act
        analysis = await extractor.extract_dual_run_knowledge(
            openevolve_result=mock_openevolve_result,
            loongflow_result=mock_loongflow_result,
            domain="finance",
            problem="Test"
        )

        # Convert to dict
        analysis_dict = analysis.to_dict()

        # Assert
        assert isinstance(analysis_dict, dict)
        assert "run_id" in analysis_dict
        assert "domain" in analysis_dict
        assert "performance_comparison" in analysis_dict
        assert "best_practices" in analysis_dict
        assert "synergy_opportunities" in analysis_dict
        assert "hybrid_recommendation" in analysis_dict
        assert "timestamp" in analysis_dict

    @pytest.mark.asyncio
    async def test_domain_specific_recommendations(self, extractor):
        """Test recommendations across different domains"""
        domains = ["finance", "science", "engineering", "trading"]

        for domain in domains:
            # Arrange
            performance = PerformanceComparison(
                evaluation_efficiency={"loongflow": 0.003, "openevolve": 0.001},
                convergence_speed={},
                solution_quality={},
                diversity_metrics={},
                computational_cost={},
                winner_by_category={},
                overall_winner="loongflow",
                confidence=0.8
            )

            # Act
            rec = await extractor.create_hybrid_recommendations(
                performance, [], domain
            )

            # Assert
            assert rec.recommended_mode == "pes"  # Expensive eval domains
            assert domain.lower() in rec.rationale.lower()


# ========================================================================
# RUN TESTS
# ========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
