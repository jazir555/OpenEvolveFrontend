"""
Comprehensive Tests for Memory Fusion System

Tests the EvolutionaryMemoryFusion class which combines OpenEvolve and
LoongFlow memory for enhanced learning.

Test Categories:
1. Memory fusion basic operations
2. Complementary pattern detection
3. Conflict detection and resolution
4. Unified lineage creation
5. Cross-system pollination
6. Temporal queries
7. Unified insights generation
8. Integration scenarios
9. Edge cases and error handling

Copyright 2026 OpenEvolve
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import sys
import os

# Add knowledge_engine to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from knowledge_engine.integrations.memory_fusion import (
    EvolutionaryMemoryFusion,
    OpenEvolveMemory,
    LoongFlowMemory,
    FusedMemory,
    ComplementaryPattern,
    MemoryConflict,
    ConflictResolution,
    UnifiedLineage,
    LineageNode,
    CrossSystemEdge,
    KnowledgeGraph,
    PollinationOpportunity,
    PollinationResult,
    UnifiedInsights,
    PatternType,
    ConflictSeverity,
    ResolutionStrategy,
    PollinationKnowledgeType,
    ImplementationComplexity,
    create_memory_fusion,
    fuse_and_analyze,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_openevolve_memory():
    """Create sample OpenEvolve memory for testing"""
    return OpenEvolveMemory(
        population_archive={
            "cell_0_0": {"fitness": 0.8, "solution": "solution_a"},
            "cell_1_1": {"fitness": 0.85, "solution": "solution_b"},
            "cell_2_2": {"fitness": 0.9, "solution": "solution_c"},
        },
        evolutionary_lineage=[
            {
                "generation": 0,
                "individual": 0,
                "solution": "initial_solution",
                "fitness": 0.5,
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=2),
                "parent_ids": [],
                "children_ids": ["gen_1_indiv_0"],
            },
            {
                "generation": 1,
                "individual": 0,
                "solution": "improved_solution",
                "fitness": 0.7,
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=1),
                "parent_ids": ["gen_0_indiv_0"],
                "children_ids": [],
            },
        ],
        fitness_history=[
            {"fitness": 0.5, "timestamp": datetime.now(timezone.utc) - timedelta(hours=2)},
            {"fitness": 0.7, "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)},
            {"fitness": 0.9, "timestamp": datetime.now(timezone.utc)},
        ],
        elite_solutions=[
            {"solution": "elite_1", "fitness": 0.9, "generation": 5},
            {"solution": "elite_2", "fitness": 0.88, "generation": 4},
            {"solution": "elite_3", "fitness": 0.85, "generation": 6},
        ],
        diversity_metrics=[
            {"diversity": 0.8, "metric": "behavioral"},
            {"diversity": 0.75, "metric": "genetic"},
        ],
        convergence_data={
            "convergence_generation": 50,
            "threshold": 0.001,
            "avg_evaluations": 250,
        },
        metadata={
            "mutation_rate": 0.1,
            "population_size": 1000,
            "selection_strategy": "archival",
            "evaluation_mode": "full",
            "adversarial_enabled": True,
            "adversarial_rounds": 20,
        },
    )


@pytest.fixture
def sample_loongflow_memory():
    """Create sample LoongFlow memory for testing"""
    return LoongFlowMemory(
        planning_strategies=[
            {
                "strategy": "Use gradient descent with momentum",
                "success_rate": 0.85,
                "iterations": 10,
            },
            {
                "strategy": "Simulated annealing approach",
                "success_rate": 0.75,
                "iterations": 8,
            },
        ],
        execution_patterns=[
            {
                "early_stopped": True,
                "iteration": 15,
                "convergence_rate": 0.95,
            },
            {
                "early_stopped": True,
                "iteration": 20,
                "convergence_rate": 0.90,
            },
        ],
        reflection_insights=[
            {
                "insights": "Momentum helps escape local optima",
                "what_worked": ["Adaptive learning rate", "Momentum"],
                "what_failed": ["Fixed learning rate"],
            }
        ],
        summarization_episodes=[
            {
                "summary": "Iteration 1-10 showed steady improvement",
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=30),
            },
            {
                "summary": "Early stopping at iteration 15 saved 40% time",
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=15),
            },
        ],
        pes_lineage=[
            {
                "iteration": 0,
                "variant": 0,
                "plan": "Initial plan",
                "fitness": 0.6,
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=45),
                "parent_plan_ids": [],
                "child_plan_ids": ["iter_1_var_0"],
            },
            {
                "iteration": 1,
                "variant": 0,
                "plan": "Refined plan",
                "fitness": 0.8,
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=30),
                "parent_plan_ids": ["iter_0_var_0"],
                "child_plan_ids": [],
            },
        ],
        efficiency_metrics={
            "efficiency_gain": 0.6,
            "convergence_rate": 0.95,
            "avg_evaluations": 100,
            "convergence_threshold": 0.01,
        },
        metadata={
            "mutation_rate": 0.3,
            "population_size": 100,
            "selection_strategy": "boltzmann",
            "evaluation_mode": "cascade",
        },
    )


# ============================================================================
# TEST CLASS 1: BASIC FUSION OPERATIONS
# ============================================================================


class TestBasicFusion:
    """Test basic memory fusion operations"""

    @pytest.mark.asyncio
    async def test_fuse_memories_creates_fused_memory(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that fuse_memories creates a valid FusedMemory object"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
            domain="finance",
        )

        assert isinstance(fused, FusedMemory)
        assert fused.domain == "finance"
        assert fused.fusion_timestamp is not None
        assert fused.fusion_quality_score > 0.0

    @pytest.mark.asyncio
    async def test_fuse_memories_with_dicts(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that fuse_memories works with dict inputs"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory.to_dict(),
            loongflow_memory=sample_loongflow_memory.to_dict(),
        )

        assert isinstance(fused, FusedMemory)
        assert isinstance(fused.openevolve_component, OpenEvolveMemory)
        assert isinstance(fused.loongflow_component, LoongFlowMemory)

    @pytest.mark.asyncio
    async def test_fuse_memories_creates_all_components(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that fuse_memories creates all required components"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        # Check all components are created
        assert isinstance(fused.complementary_patterns, list)
        assert isinstance(fused.conflicts, list)
        assert isinstance(fused.conflict_resolutions, list)
        assert isinstance(fused.unified_lineage, UnifiedLineage)
        assert isinstance(fused.unified_knowledge_graph, KnowledgeGraph)
        assert isinstance(fused.pollination_opportunities, list)

    @pytest.mark.asyncio
    async def test_fusion_tracks_statistics(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that fusion operations are tracked in statistics"""
        fusion = create_memory_fusion()

        await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        stats = fusion.get_stats()
        assert stats["total_fusions"] == 1
        assert stats["patterns_detected"] >= 0
        assert stats["conflicts_resolved"] >= 0


# ============================================================================
# TEST CLASS 2: COMPLEMENTARY PATTERNS
# ============================================================================


class TestComplementaryPatterns:
    """Test complementary pattern detection"""

    @pytest.mark.asyncio
    async def test_detect_exploration_refinement_pattern(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test detection of exploration + refinement pattern"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        # Should detect exploration + refinement pattern
        patterns = [
            p for p in fused.complementary_patterns
            if p.pattern_type == PatternType.EXPLORATION_REFINEMENT.value
        ]

        assert len(patterns) > 0
        pattern = patterns[0]
        assert "diversity" in pattern.openevolve_contribution.lower()
        # Check for either "efficiency" or "efficient" in the contribution
        assert "efficien" in pattern.loongflow_contribution.lower()
        assert pattern.expected_improvement > 0.0
        assert pattern.confidence > 0.0

    @pytest.mark.asyncio
    async def test_detect_diversity_efficiency_pattern(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test detection of diversity + efficiency pattern"""
        # Create memories with clear diversity and efficiency
        oe_memory = OpenEvolveMemory(
            elite_solutions=[
                {"solution": f"elite_{i}", "fitness": 0.8 + i * 0.01, "generation": i}
                for i in range(10)
            ],
            convergence_data={"avg_evaluations": 250},
        )

        lf_memory = LoongFlowMemory(
            execution_patterns=[{"early_stopped": True, "iteration": i} for i in range(5)],
            efficiency_metrics={"avg_evaluations": 100},
        )

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
        )

        patterns = [
            p for p in fused.complementary_patterns
            if p.pattern_type == PatternType.DIVERSITY_EFFICIENCY.value
        ]

        assert len(patterns) > 0

    @pytest.mark.asyncio
    async def test_pattern_includes_evidence(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that patterns include supporting evidence"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        for pattern in fused.complementary_patterns:
            assert isinstance(pattern.evidence, list)
            assert len(pattern.evidence) > 0


# ============================================================================
# TEST CLASS 3: CONFLICT DETECTION
# ============================================================================


class TestConflictDetection:
    """Test conflict detection and resolution"""

    @pytest.mark.asyncio
    async def test_detect_parameter_conflicts(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test detection of parameter value conflicts"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        # Should detect mutation rate conflict (0.1 vs 0.3)
        mutation_conflicts = [
            c for c in fused.conflicts
            if c.conflict_type == "parameter_value" and "mutation" in c.description.lower()
        ]

        assert len(mutation_conflicts) > 0
        conflict = mutation_conflicts[0]
        assert conflict.severity in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_detect_strategy_conflicts(self):
        """Test detection of strategy effectiveness conflicts"""
        oe_memory = OpenEvolveMemory(
            metadata={"selection_strategy": "archival"},
        )

        lf_memory = LoongFlowMemory(
            metadata={"selection_strategy": "boltzmann"},
        )

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
        )

        strategy_conflicts = [
            c for c in fused.conflicts
            if c.conflict_type == "strategy_effectiveness"
        ]

        assert len(strategy_conflicts) > 0

    @pytest.mark.asyncio
    async def test_resolve_low_severity_conflicts(self):
        """Test resolution of low severity conflicts"""
        conflict = MemoryConflict(
            conflict_type="test",
            openevolve_position="Position A",
            loongflow_position="Position B",
            severity=ConflictSeverity.LOW.value,
            description="Test conflict",
        )

        fusion = create_memory_fusion()
        resolutions = await fusion.resolve_conflicts([conflict])

        assert len(resolutions) == 1
        resolution = resolutions[0]
        assert resolution.resolution_strategy in [s.value for s in ResolutionStrategy]
        assert resolution.confidence > 0.0
        assert resolution.expected_accuracy > 0.0

    @pytest.mark.asyncio
    async def test_resolve_high_severity_conflicts(self):
        """Test resolution of high severity conflicts"""
        conflict = MemoryConflict(
            conflict_type="test",
            openevolve_position="Position A",
            loongflow_position="Position B",
            severity=ConflictSeverity.HIGH.value,
            description="High severity test conflict",
        )

        fusion = create_memory_fusion()
        resolutions = await fusion.resolve_conflicts([conflict])

        assert len(resolutions) == 1
        resolution = resolutions[0]
        # High severity should require investigation
        assert resolution.resolution_strategy == ResolutionStrategy.INVESTIGATE.value


# ============================================================================
# TEST CLASS 4: UNIFIED LINEAGE
# ============================================================================


class TestUnifiedLineage:
    """Test unified lineage creation"""

    @pytest.mark.asyncio
    async def test_create_unified_lineage(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that unified lineage is created correctly"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        lineage = fused.unified_lineage
        assert isinstance(lineage, UnifiedLineage)
        assert isinstance(lineage.lineage_nodes, list)
        assert isinstance(lineage.cross_system_edges, list)

    @pytest.mark.asyncio
    async def test_trace_solution_origin(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test tracing solution origin through lineage"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        lineage = fused.unified_lineage

        if lineage.lineage_nodes:
            # Test tracing a node
            node = lineage.lineage_nodes[0]
            path = lineage.trace_solution_origin(node.node_id)

            assert isinstance(path, list)
            assert len(path) >= 1

    @pytest.mark.asyncio
    async def test_find_common_ancestors(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test finding common ancestors of two solutions"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        lineage = fused.unified_lineage

        if len(lineage.lineage_nodes) >= 2:
            # Find common ancestors of first two nodes
            node1 = lineage.lineage_nodes[0]
            node2 = lineage.lineage_nodes[1]

            ancestors = lineage.find_common_ancestors(node1.node_id, node2.node_id)

            assert isinstance(ancestors, list)


# ============================================================================
# TEST CLASS 5: CROSS-SYSTEM POLLINATION
# ============================================================================


class TestCrossSystemPollination:
    """Test cross-system pollination opportunities"""

    @pytest.mark.asyncio
    async def test_find_pollination_opportunities(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that pollination opportunities are found"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        opportunities = fused.pollination_opportunities
        assert len(opportunities) > 0

    @pytest.mark.asyncio
    async def test_pollination_includes_all_fields(self):
        """Test that pollination opportunities include all required fields"""
        oe_memory = OpenEvolveMemory(
            elite_solutions=[{"solution": "test", "fitness": 0.9} for _ in range(10)],
        )

        lf_memory = LoongFlowMemory(
            planning_strategies=[{"strategy": "test", "success_rate": 0.8}],
            execution_patterns=[
                {"early_stopped": True, "iteration": i} for i in range(5)
            ],
        )

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
        )

        for opp in fused.pollination_opportunities:
            assert opp.opportunity_id is not None
            assert opp.source_system in ["openevolve", "loongflow", "both"]
            assert opp.target_system in ["openevolve", "loongflow", "both"]
            assert opp.knowledge_type in [kt.value for kt in PollinationKnowledgeType]
            assert opp.expected_benefit > 0.0
            assert opp.confidence > 0.0
            assert opp.implementation_complexity in [ic.value for ic in ImplementationComplexity]
            assert len(opp.description) > 0

    @pytest.mark.asyncio
    async def test_apply_pollination_opportunity(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test applying a pollination opportunity"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        if fused.pollination_opportunities:
            opportunity = fused.pollination_opportunities[0]
            result = await fusion.apply_pollination(opportunity)

            assert isinstance(result, PollinationResult)
            assert isinstance(result.success, bool)
            assert result.actual_improvement >= 0.0

    @pytest.mark.asyncio
    async def test_apply_pollination_tracks_stats(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that pollination is tracked in statistics"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        if fused.pollination_opportunities:
            await fusion.apply_pollination(fused.pollination_opportunities[0])

            stats = fusion.get_stats()
            assert stats["pollinations_applied"] >= 1


# ============================================================================
# TEST CLASS 6: TEMPORAL QUERIES
# ============================================================================


class TestTemporalQueries:
    """Test temporal query functionality"""

    @pytest.mark.asyncio
    async def test_temporal_query_finds_artifacts(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that temporal queries find artifacts in time range"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        # Query for last 3 hours
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=3)

        results = await fusion.temporal_query(
            fused_memory=fused,
            query="fitness",
            time_range=(start_time, end_time),
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_temporal_query_respects_limit(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that temporal queries respect the limit parameter"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=3)

        # Query with limit
        results = await fusion.temporal_query(
            fused_memory=fused,
            query="fitness",
            time_range=(start_time, end_time),
            limit=2,
        )

        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_temporal_query_ranks_by_relevance(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that temporal query results are ranked by relevance"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=3)

        results = await fusion.temporal_query(
            fused_memory=fused,
            query="fitness improvement",
            time_range=(start_time, end_time),
        )

        # Check that results are sorted by relevance
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["relevance"] >= results[i + 1]["relevance"]


# ============================================================================
# TEST CLASS 7: UNIFIED INSIGHTS
# ============================================================================


class TestUnifiedInsights:
    """Test unified insights generation"""

    @pytest.mark.asyncio
    async def test_get_unified_insights(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that unified insights are generated"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        insights = await fusion.get_unified_insights(fused)

        assert isinstance(insights, UnifiedInsights)
        assert insights.domain == fused.domain
        assert len(insights.best_practices) > 0
        assert len(insights.anti_patterns) > 0
        assert insights.confidence > 0.0

    @pytest.mark.asyncio
    async def test_insights_include_performance_comparison(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that insights include performance comparison"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        insights = await fusion.get_unified_insights(fused)

        assert "openevolve" in insights.overall_performance_comparison
        assert "loongflow" in insights.overall_performance_comparison

    @pytest.mark.asyncio
    async def test_insights_include_configurations(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that insights include recommended configurations"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        insights = await fusion.get_unified_insights(fused)

        assert "openevolve" in insights.recommended_configurations
        assert "loongflow" in insights.recommended_configurations


# ============================================================================
# TEST CLASS 8: INTEGRATION SCENARIOS
# ============================================================================


class TestIntegrationScenarios:
    """Test real-world integration scenarios"""

    @pytest.mark.asyncio
    async def test_finance_domain_scenario(self):
        """Test memory fusion for finance domain"""
        oe_memory = OpenEvolveMemory(
            elite_solutions=[
                {"solution": f"portfolio_{i}", "fitness": 0.7 + i * 0.05, "generation": i}
                for i in range(5)
            ],
            fitness_history=[
                {"fitness": 0.7 + i * 0.05, "timestamp": datetime.now(timezone.utc) - timedelta(hours=i)}
                for i in range(5)
            ],
            diversity_metrics=[  # Add diversity metrics for pattern detection with "diversity" key
                {"diversity": 0.85, "metric": "behavioral_diversity"},
                {"diversity": 0.78, "metric": "genotypic_diversity"}
            ],
            metadata={"mutation_rate": 0.1, "population_size": 1000},
        )

        lf_memory = LoongFlowMemory(
            planning_strategies=[
                {"strategy": "Risk-adjusted optimization", "success_rate": 0.85}
            ],
            efficiency_metrics={"efficiency_gain": 0.8, "avg_evaluations": 100, "convergence_rate": 0.9},  # Higher efficiency
            metadata={"mutation_rate": 0.25, "population_size": 100},
        )

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
            domain="finance",
        )

        assert fused.domain == "finance"
        assert len(fused.complementary_patterns) > 0
        assert len(fused.pollination_opportunities) > 0

    @pytest.mark.asyncio
    async def test_science_domain_scenario(self):
        """Test memory fusion for scientific experiments"""
        oe_memory = OpenEvolveMemory(
            population_archive={
                f"cell_{i}_{j}": {"fitness": 0.6 + (i + j) * 0.1, "experiment": f"exp_{i}"}
                for i in range(3) for j in range(3)
            },
            elite_solutions=[  # Add elite_solutions for diversity pattern detection
                {"solution": f"experiment_{i}", "fitness": 0.6 + i * 0.1, "generation": i}
                for i in range(10)  # More than 5 for pattern detection
            ],
            diversity_metrics=[{"diversity": 0.85, "metric": "experimental_design"}],
            convergence_data={"avg_evaluations": 300},  # Higher than LF
            metadata={"mutation_rate": 0.15, "population_size": 800},
        )

        lf_memory = LoongFlowMemory(
            planning_strategies=[
                {"strategy": "Sequential experimental design", "success_rate": 0.90}
            ],
            reflection_insights=[
                {"insights": "Fewer experiments with better planning"}
            ],
            execution_patterns=[{"pattern": "early_stopping", "count": 5}],  # Add for pattern detection
            efficiency_metrics={"efficiency_gain": 0.75, "avg_evaluations": 100},  # 30%+ more efficient than OE
            metadata={"mutation_rate": 0.20, "population_size": 80},
        )

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
            domain="science",
        )

        assert fused.domain == "science"
        # Science domain should benefit from diversity + efficiency
        diversity_efficiency_patterns = [
            p for p in fused.complementary_patterns
            if p.pattern_type == PatternType.DIVERSITY_EFFICIENCY.value
        ]
        assert len(diversity_efficiency_patterns) > 0


# ============================================================================
# TEST CLASS 9: EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_empty_memories(self):
        """Test fusion with empty memories"""
        oe_memory = OpenEvolveMemory()
        lf_memory = LoongFlowMemory()

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
        )

        # Should still create fused memory
        assert isinstance(fused, FusedMemory)
        assert fused.fusion_quality_score >= 0.0

    @pytest.mark.asyncio
    async def test_minimal_data_fusion(self):
        """Test fusion with minimal data"""
        oe_memory = OpenEvolveMemory(
            fitness_history=[{"fitness": 0.5}],
        )

        lf_memory = LoongFlowMemory(
            planning_strategies=[{"strategy": "test"}],
        )

        fusion = create_memory_fusion()
        fused = await fusion.fuse_memories(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
        )

        assert isinstance(fused, FusedMemory)

    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test the convenience function fuse_and_analyze"""
        oe_memory = OpenEvolveMemory(
            fitness_history=[{"fitness": 0.5, "timestamp": datetime.now(timezone.utc)}],
        )

        lf_memory = LoongFlowMemory(
            planning_strategies=[{"strategy": "test"}],
        )

        fused, insights = await fuse_and_analyze(
            openevolve_memory=oe_memory,
            loongflow_memory=lf_memory,
            domain="general",
        )

        assert isinstance(fused, FusedMemory)
        assert isinstance(insights, UnifiedInsights)


# ============================================================================
# TEST CLASS 10: KNOWLEDGE GRAPH
# ============================================================================


class TestKnowledgeGraph:
    """Test unified knowledge graph creation"""

    @pytest.mark.asyncio
    async def test_create_knowledge_graph(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test that knowledge graph is created"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        graph = fused.unified_knowledge_graph
        assert isinstance(graph, KnowledgeGraph)
        assert isinstance(graph.entities, dict)
        assert isinstance(graph.relationships, list)

    @pytest.mark.asyncio
    async def test_graph_query_entities(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test querying entities from knowledge graph"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        graph = fused.unified_knowledge_graph

        # Query all entities
        all_entities = graph.query_entities()
        assert isinstance(all_entities, list)

        # Query by type
        solutions = graph.query_entities(entity_type="solution")
        assert isinstance(solutions, list)

    @pytest.mark.asyncio
    async def test_graph_get_related_entities(
        self, sample_openevolve_memory, sample_loongflow_memory
    ):
        """Test getting related entities from graph"""
        fusion = create_memory_fusion()

        fused = await fusion.fuse_memories(
            openevolve_memory=sample_openevolve_memory,
            loongflow_memory=sample_loongflow_memory,
        )

        graph = fused.unified_knowledge_graph

        if graph.entities:
            entity_id = list(graph.entities.keys())[0]
            related = graph.get_related_entities(entity_id)
            assert isinstance(related, list)


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
