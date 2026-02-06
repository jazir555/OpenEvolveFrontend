"""
Tests for Sovereign UI Components.
Tests data preparation and logic, not actual UI rendering.
"""

import pytest
from datetime import datetime

from sovereign_ui_components import _calculate_node_levels
from sovereign_data_models import (
    DecompositionPlan, SubProblem, DecompositionStrategy,
    SubProblemType, ComplexityScore, DependencyGraph,
    QualityScores, generate_id
)
from sovereign_quality_assessment import QualityReport, QualityMetrics
from sovereign_refinement import RefinementCycle


class TestNodeLevelCalculation:
    """Test dependency graph level calculation."""
    
    def test_simple_linear_dependencies(self):
        """Test level calculation for linear dependencies."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'}
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
            {'source': 'b', 'target': 'c'}
        ]
        
        levels = _calculate_node_levels(nodes, edges)
        
        assert levels['a'] == 0
        assert levels['b'] == 1
        assert levels['c'] == 2
    
    def test_parallel_branches(self):
        """Test level calculation for parallel branches."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
            {'id': 'd'}
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
            {'source': 'a', 'target': 'c'},
            {'source': 'b', 'target': 'd'},
            {'source': 'c', 'target': 'd'}
        ]
        
        levels = _calculate_node_levels(nodes, edges)
        
        assert levels['a'] == 0
        assert levels['b'] == 1
        assert levels['c'] == 1
        assert levels['d'] == 2
    
    def test_no_dependencies(self):
        """Test level calculation with no dependencies."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'}
        ]
        edges = []
        
        levels = _calculate_node_levels(nodes, edges)
        
        # All should be at level 0
        assert all(level == 0 for level in levels.values())
    
    def test_complex_graph(self):
        """Test level calculation for complex dependency graph."""
        nodes = [
            {'id': 'a'},
            {'id': 'b'},
            {'id': 'c'},
            {'id': 'd'},
            {'id': 'e'}
        ]
        edges = [
            {'source': 'a', 'target': 'b'},
            {'source': 'a', 'target': 'c'},
            {'source': 'b', 'target': 'd'},
            {'source': 'c', 'target': 'd'},
            {'source': 'd', 'target': 'e'}
        ]
        
        levels = _calculate_node_levels(nodes, edges)
        
        assert levels['a'] == 0
        assert levels['b'] == 1
        assert levels['c'] == 1
        assert levels['d'] == 2
        assert levels['e'] == 3


class TestUIDataPreparation:
    """Test UI data preparation functions."""
    
    @pytest.fixture
    def sample_plan(self):
        """Create a sample decomposition plan."""
        sub_problems = [
            SubProblem(
                id="sp1",
                parent_id="problem1",
                title="Sub-problem 1",
                description="First sub-problem",
                type=SubProblemType.ANALYSIS,
                complexity_score=ComplexityScore(6, 5, 7, 6, 6, "test"),
                dependencies=[],
                success_criteria=[],
                validation_gauntlet="coherence",
                priority=8,
                estimated_effort=10
            ),
            SubProblem(
                id="sp2",
                parent_id="problem1",
                title="Sub-problem 2",
                description="Second sub-problem",
                type=SubProblemType.IMPLEMENTATION,
                complexity_score=ComplexityScore(7, 8, 6, 7, 7, "test"),
                dependencies=["sp1"],
                success_criteria=[],
                validation_gauntlet="feasibility",
                priority=6,
                estimated_effort=20
            )
        ]
        
        return DecompositionPlan(
            id="plan1",
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=sub_problems,
            dependency_graph=DependencyGraph(
                nodes={sp.id: sp for sp in sub_problems},
                edges={"sp2": ["sp1"]},
                critical_path=["sp1", "sp2"],
                parallel_groups=[],
                execution_order=["sp1", "sp2"]
            ),
            validation_checkpoints=[],
            quality_scores=QualityScores(
                coherence_score=0.85,
                completeness_score=0.90,
                feasibility_score=0.80,
                integration_score=0.85,
                overall_score=0.85,
                meets_thresholds=True,
                details={},
                timestamp=datetime.now()
            ),
            confidence_level=0.85,
            created_by="test",
            approved_by=None,
            status="approved",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def test_plan_has_sub_problems(self, sample_plan):
        """Test plan contains sub-problems."""
        assert len(sample_plan.sub_problems) == 2
    
    def test_plan_has_dependencies(self, sample_plan):
        """Test plan has dependency information."""
        assert sample_plan.dependency_graph is not None
        assert len(sample_plan.dependency_graph.critical_path) == 2
    
    def test_total_effort_calculation(self, sample_plan):
        """Test total effort can be calculated."""
        total_effort = sum(sp.estimated_effort for sp in sample_plan.sub_problems)
        assert total_effort == 30  # 10 + 20
    
    def test_complexity_data_extraction(self, sample_plan):
        """Test complexity data can be extracted for visualization."""
        sp = sample_plan.sub_problems[0]
        
        complexity_data = {
            'Cognitive': sp.complexity_score.cognitive_complexity,
            'Computational': sp.complexity_score.computational_complexity,
            'Domain': sp.complexity_score.domain_complexity,
            'Integration': sp.complexity_score.integration_complexity
        }
        
        assert len(complexity_data) == 4
        assert all(0 <= v <= 10 for v in complexity_data.values())


class TestQualityReportData:
    """Test quality report data preparation."""
    
    @pytest.fixture
    def sample_report(self):
        """Create a sample quality report."""
        metrics = QualityMetrics(
            coherence_score=0.85,
            completeness_score=0.90,
            feasibility_score=0.80,
            integration_score=0.85,
            balance_score=0.88,
            clarity_score=0.82,
            overall_score=0.85,
            details={}
        )
        
        return QualityReport(
            plan_id="plan1",
            metrics=metrics,
            strengths=["Good coherence", "Complete coverage"],
            weaknesses=["Could improve feasibility"],
            recommendations=["Add more validation", "Simplify complex sub-problems"],
            meets_thresholds=True,
            generated_at=datetime.now()
        )
    
    def test_metrics_extraction(self, sample_report):
        """Test metrics can be extracted for visualization."""
        metrics = sample_report.metrics
        
        values = [
            metrics.coherence_score,
            metrics.completeness_score,
            metrics.feasibility_score,
            metrics.integration_score,
            metrics.balance_score,
            metrics.clarity_score
        ]
        
        assert len(values) == 6
        assert all(0 <= v <= 1 for v in values)
    
    def test_radar_chart_data(self, sample_report):
        """Test data preparation for radar chart."""
        categories = ['Coherence', 'Completeness', 'Feasibility', 'Integration', 'Balance', 'Clarity']
        metrics = sample_report.metrics
        
        values = [
            metrics.coherence_score,
            metrics.completeness_score,
            metrics.feasibility_score,
            metrics.integration_score,
            metrics.balance_score,
            metrics.clarity_score
        ]
        
        assert len(categories) == len(values)
    
    def test_strengths_and_weaknesses(self, sample_report):
        """Test strengths and weaknesses are available."""
        assert len(sample_report.strengths) > 0
        assert len(sample_report.weaknesses) > 0
        assert len(sample_report.recommendations) > 0


class TestRefinementHistoryData:
    """Test refinement history data preparation."""
    
    @pytest.fixture
    def sample_cycles(self):
        """Create sample refinement cycles."""
        return [
            RefinementCycle(
                cycle_number=1,
                plan_id="plan1",
                feedback_received=[],
                improvements_applied=["Improved coherence", "Added dependencies"],
                quality_before=0.70,
                quality_after=0.80,
                gauntlet_results={},
                converged=False
            ),
            RefinementCycle(
                cycle_number=2,
                plan_id="plan1",
                feedback_received=[],
                improvements_applied=["Balanced complexity"],
                quality_before=0.80,
                quality_after=0.85,
                gauntlet_results={},
                converged=False
            ),
            RefinementCycle(
                cycle_number=3,
                plan_id="plan1",
                feedback_received=[],
                improvements_applied=[],
                quality_before=0.85,
                quality_after=0.86,
                gauntlet_results={},
                converged=True
            )
        ]
    
    def test_quality_progression_data(self, sample_cycles):
        """Test quality progression data extraction."""
        cycle_numbers = [c.cycle_number for c in sample_cycles]
        quality_before = [c.quality_before for c in sample_cycles]
        quality_after = [c.quality_after for c in sample_cycles]
        
        assert len(cycle_numbers) == 3
        assert len(quality_before) == 3
        assert len(quality_after) == 3
        
        # Quality should improve
        assert quality_after[-1] > quality_before[0]
    
    def test_convergence_detection(self, sample_cycles):
        """Test convergence can be detected."""
        converged_cycles = [c for c in sample_cycles if c.converged]
        assert len(converged_cycles) == 1
        assert converged_cycles[0].cycle_number == 3
    
    def test_improvement_tracking(self, sample_cycles):
        """Test improvements are tracked."""
        total_improvements = sum(len(c.improvements_applied) for c in sample_cycles)
        assert total_improvements == 3


class TestGraphVisualizationData:
    """Test graph visualization data preparation."""
    
    def test_node_data_preparation(self):
        """Test node data can be prepared for visualization."""
        sp = SubProblem(
            id="sp1",
            parent_id="problem1",
            title="Test Sub-problem with a very long title that needs truncation",
            description="Test description",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(6, 5, 7, 6, 6, "test"),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=8,
            estimated_effort=10
        )
        
        node = {
            'id': sp.id,
            'label': sp.title[:30] + '...' if len(sp.title) > 30 else sp.title,
            'complexity': sp.complexity_score.overall_complexity,
            'priority': sp.priority
        }
        
        assert len(node['label']) <= 33  # 30 + '...'
        assert node['complexity'] == 6
        assert node['priority'] == 8
    
    def test_edge_data_preparation(self):
        """Test edge data can be prepared for visualization."""
        sp1 = SubProblem(
            id="sp1",
            parent_id="problem1",
            title="Sub-problem 1",
            description="First",
            type=SubProblemType.ANALYSIS,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=[],
            success_criteria=[],
            validation_gauntlet="coherence",
            priority=5,
            estimated_effort=8
        )
        
        sp2 = SubProblem(
            id="sp2",
            parent_id="problem1",
            title="Sub-problem 2",
            description="Second",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
            dependencies=["sp1"],
            success_criteria=[],
            validation_gauntlet="feasibility",
            priority=5,
            estimated_effort=8
        )
        
        edges = []
        for dep in sp2.dependencies:
            edges.append({'source': dep, 'target': sp2.id})
        
        assert len(edges) == 1
        assert edges[0]['source'] == 'sp1'
        assert edges[0]['target'] == 'sp2'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

