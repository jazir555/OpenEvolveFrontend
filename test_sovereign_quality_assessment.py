"""
Tests for Sovereign Quality Assessment System
Task 7.5: Unit tests for quality assessment
"""

import pytest
from datetime import datetime

from sovereign_data_models import (
    DecompositionPlan, SubProblem, DecompositionStrategy, SubProblemType,
    ComplexityScore, SuccessCriterion, DependencyGraph, generate_id
)
from sovereign_quality_assessment import QualityAssessor, QualityMetrics, QualityReport


@pytest.fixture
def sample_complexity():
    """Create a sample complexity score."""
    return ComplexityScore(
        cognitive_complexity=5.0,
        computational_complexity=5.0,
        domain_complexity=5.0,
        integration_complexity=5.0,
        overall_complexity=5.0,
        explanation="Medium complexity"
    )


@pytest.fixture
def good_plan(sample_complexity):
    """Create a high-quality decomposition plan."""
    sub_problems = []
    
    for i in range(4):
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title=f"Well-Defined Sub-Problem {i+1}",
            description=f"This is a detailed description for sub-problem {i+1} with sufficient information to understand the task clearly.",
            type=[SubProblemType.RESEARCH, SubProblemType.ANALYSIS, 
                  SubProblemType.IMPLEMENTATION, SubProblemType.VALIDATION][i],
            complexity_score=sample_complexity,
            dependencies=[sub_problems[i-1].id] if i > 0 else [],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description=f"Clear success criterion for sub-problem {i+1}",
                    metric="completion",
                    threshold=0.9,
                    validation_method="review"
                )
            ],
            validation_gauntlet="coherence",
            priority=8 - i,
            estimated_effort=10
        )
        sub_problems.append(sp)
    
    return DecompositionPlan(
        id=generate_id("plan"),
        problem_id="problem1",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=sub_problems,
        dependency_graph=DependencyGraph(
            nodes={sp.id: sp for sp in sub_problems},
            edges={sp.id: sp.dependencies for sp in sub_problems},
            execution_order=[sp.id for sp in sub_problems]
        ),
        confidence_level=0.85
    )


@pytest.fixture
def poor_plan():
    """Create a low-quality decomposition plan."""
    complexity = ComplexityScore(
        cognitive_complexity=9.5,
        computational_complexity=9.5,
        domain_complexity=9.5,
        integration_complexity=9.5,
        overall_complexity=9.5,
        explanation="Very high complexity"
    )
    
    sp = SubProblem(
        id=generate_id("subproblem"),
        parent_id="problem1",
        title="Vague",  # Too short
        description="Short",  # Too short
        type=SubProblemType.ANALYSIS,
        complexity_score=complexity,  # Too complex
        dependencies=[],
        success_criteria=[],  # No criteria
        validation_gauntlet="",
        priority=0,
        estimated_effort=100  # Too much effort
    )
    
    return DecompositionPlan(
        id=generate_id("plan"),
        problem_id="problem1",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=[sp],
        confidence_level=0.3
    )


class TestQualityAssessor:
    """Test QualityAssessor class."""
    
    def test_initialization(self):
        assessor = QualityAssessor()
        assert assessor.thresholds is not None
        assert len(assessor.thresholds) == 7
    
    def test_calculate_coherence_score_good_plan(self, good_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_coherence_score(good_plan)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.7  # Good plan should have good coherence
    
    def test_calculate_coherence_score_poor_plan(self, poor_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_coherence_score(poor_plan)
        
        assert 0.0 <= score <= 1.0
        assert score < 0.7  # Poor plan should have low coherence
    
    def test_calculate_completeness_score_good_plan(self, good_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_completeness_score(good_plan)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.7  # Good plan should be complete
    
    def test_calculate_completeness_score_single_subproblem(self, poor_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_completeness_score(poor_plan)
        
        assert score < 0.8  # Single sub-problem is incomplete
    
    def test_calculate_feasibility_score_good_plan(self, good_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_feasibility_score(good_plan)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.7  # Good plan should be feasible
    
    def test_calculate_feasibility_score_high_complexity(self, poor_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_feasibility_score(poor_plan)
        
        assert score < 0.9  # High complexity reduces feasibility
    
    def test_calculate_integration_score_good_plan(self, good_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_integration_score(good_plan)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.7  # Good plan should integrate well
    
    def test_calculate_integration_score_no_graph(self, sample_complexity):
        assessor = QualityAssessor()
        
        # Create plan with multiple sub-problems but no dependency graph
        sub_problems = []
        for i in range(3):
            sp = SubProblem(
                id=generate_id("subproblem"),
                parent_id="problem1",
                title=f"Sub-Problem {i+1}",
                description=f"Description {i+1}",
                type=SubProblemType.ANALYSIS,
                complexity_score=sample_complexity,
                dependencies=[],
                success_criteria=[],
                estimated_effort=10
            )
            sub_problems.append(sp)
        
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=sub_problems,
            dependency_graph=None,  # No dependency graph
            confidence_level=0.5
        )
        
        score = assessor.calculate_integration_score(plan)
        
        assert score < 1.0  # Missing dependency graph reduces score
    
    def test_calculate_balance_score_good_plan(self, good_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_balance_score(good_plan)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.7  # Good plan should be balanced
    
    def test_calculate_clarity_score_good_plan(self, good_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_clarity_score(good_plan)
        
        assert 0.0 <= score <= 1.0
        assert score >= 0.7  # Good plan should be clear
    
    def test_calculate_clarity_score_poor_descriptions(self, poor_plan):
        assessor = QualityAssessor()
        score = assessor.calculate_clarity_score(poor_plan)
        
        assert score < 0.7  # Poor descriptions reduce clarity


class TestQualityReport:
    """Test quality report generation."""
    
    def test_generate_quality_report_good_plan(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        
        assert report.plan_id == good_plan.id
        assert isinstance(report.metrics, QualityMetrics)
        assert isinstance(report.strengths, list)
        assert isinstance(report.weaknesses, list)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.meets_thresholds, bool)
        assert isinstance(report.generated_at, datetime)
    
    def test_good_plan_has_strengths(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        
        # Good plan should have some strengths
        assert len(report.strengths) > 0
    
    def test_poor_plan_has_weaknesses(self, poor_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(poor_plan)
        
        # Poor plan should have weaknesses
        assert len(report.weaknesses) > 0
    
    def test_poor_plan_has_recommendations(self, poor_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(poor_plan)
        
        # Poor plan should have recommendations
        assert len(report.recommendations) > 0
    
    def test_overall_score_calculation(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        
        # Overall score should be weighted average
        assert 0.0 <= report.metrics.overall_score <= 1.0
        
        # Verify it's reasonable given individual scores
        avg_score = (
            report.metrics.coherence_score +
            report.metrics.completeness_score +
            report.metrics.feasibility_score +
            report.metrics.integration_score +
            report.metrics.balance_score +
            report.metrics.clarity_score
        ) / 6.0
        
        # Overall should be close to average (within 0.2)
        assert abs(report.metrics.overall_score - avg_score) < 0.2
    
    def test_meets_thresholds_good_plan(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        
        # Good plan should meet thresholds
        assert report.meets_thresholds is True
    
    def test_meets_thresholds_poor_plan(self, poor_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(poor_plan)
        
        # Poor plan should not meet thresholds
        assert report.meets_thresholds is False


class TestQualityMetrics:
    """Test quality metrics."""
    
    def test_all_metrics_in_range(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        metrics = report.metrics
        
        # All scores should be between 0 and 1
        assert 0.0 <= metrics.coherence_score <= 1.0
        assert 0.0 <= metrics.completeness_score <= 1.0
        assert 0.0 <= metrics.feasibility_score <= 1.0
        assert 0.0 <= metrics.integration_score <= 1.0
        assert 0.0 <= metrics.balance_score <= 1.0
        assert 0.0 <= metrics.clarity_score <= 1.0
        assert 0.0 <= metrics.overall_score <= 1.0
    
    def test_metrics_details(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        metrics = report.metrics
        
        # Details should contain useful information
        assert 'sub_problem_count' in metrics.details
        assert 'avg_complexity' in metrics.details
        assert 'total_effort' in metrics.details
        assert 'strategy' in metrics.details
        
        assert metrics.details['sub_problem_count'] == len(good_plan.sub_problems)


class TestThresholdChecking:
    """Test threshold validation."""
    
    def test_check_quality_thresholds_pass(self, good_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(good_plan)
        
        result = assessor.check_quality_thresholds(report.metrics)
        assert result is True
    
    def test_check_quality_thresholds_fail(self, poor_plan):
        assessor = QualityAssessor()
        report = assessor.generate_quality_report(poor_plan)
        
        result = assessor.check_quality_thresholds(report.metrics)
        assert result is False
    
    def test_update_plan_quality_scores(self, good_plan):
        assessor = QualityAssessor()
        
        # Initially no quality scores
        assert good_plan.quality_scores is None
        
        # Update with quality scores
        quality_scores = assessor.update_plan_quality_scores(good_plan)
        
        # Now plan should have quality scores
        assert good_plan.quality_scores is not None
        assert good_plan.quality_scores == quality_scores
        assert quality_scores.overall_score > 0.0


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_plan(self):
        assessor = QualityAssessor()
        
        empty_plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=[],
            confidence_level=0.0
        )
        
        report = assessor.generate_quality_report(empty_plan)
        
        # Empty plan should have zero scores
        assert report.metrics.coherence_score == 0.0
        assert report.metrics.completeness_score == 0.0
        assert report.meets_thresholds is False
    
    def test_single_subproblem_plan(self, sample_complexity):
        assessor = QualityAssessor()
        
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title="Single Sub-Problem",
            description="A single sub-problem with good description",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=sample_complexity,
            dependencies=[],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description="Success criterion",
                    metric="completion",
                    threshold=1.0,
                    validation_method="review"
                )
            ],
            estimated_effort=10
        )
        
        plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id="problem1",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=[sp],
            confidence_level=0.7
        )
        
        report = assessor.generate_quality_report(plan)
        
        # Single sub-problem should have reduced completeness
        assert report.metrics.completeness_score < 0.8
        # But integration should be fine (trivial case)
        assert report.metrics.integration_score >= 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
