"""
Tests for Sovereign Refinement Coordinator.
"""

import pytest
from datetime import datetime

from sovereign_refinement import (
    RefinementCoordinator, RefinementPlan, RefinementCycle, RefinementMetrics
)
from sovereign_data_models import (
    DecompositionPlan, SubProblem, Feedback, DecompositionStrategy,
    ProblemType, SubProblemType, ComplexityScore, SuccessCriterion,
    DependencyGraph, QualityScores, SolutionAttempt, generate_id
)
from problem_fractal_pipeline import FractalPipelineCoordinator
from z3prover_integration import generate_refutation_narrative


class TestRefinementCoordinator:
    """Test refinement coordinator functionality."""
    
    @pytest.fixture
    def coordinator(self):
        return RefinementCoordinator()
    
    @pytest.fixture
    def sample_plan(self):
        """Create a sample decomposition plan."""
        sub_problems = [
            SubProblem(
                id=generate_id("subproblem"),
                parent_id="test_problem",
                title="Sub-problem 1",
                description="First sub-problem",
                type=SubProblemType.ANALYSIS,
                complexity_score=ComplexityScore(5, 5, 5, 5, 5, "test"),
                dependencies=[],
                success_criteria=[],
                validation_gauntlet="coherence",
                priority=5,
                estimated_effort=8
            )
        ]
        
        return DecompositionPlan(
            id=generate_id("plan"),
            problem_id="test_problem",
            strategy=DecompositionStrategy.SEMANTIC,
            sub_problems=sub_problems,
            dependency_graph=DependencyGraph(
                nodes={sp.id: sp for sp in sub_problems},
                edges={},
                critical_path=[],
                parallel_groups=[],
                execution_order=[]
            ),
            validation_checkpoints=[],
            quality_scores=QualityScores(
                coherence_score=0.7,
                completeness_score=0.7,
                feasibility_score=0.7,
                integration_score=0.7,
                overall_score=0.7,
                meets_thresholds=False,
                details={},
                timestamp=datetime.now()
            ),
            confidence_level=0.7,
            created_by="test",
            approved_by=None,
            status="draft",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_feedback(self):
        """Create sample feedback."""
        return [
            Feedback(
                id=generate_id("feedback"),
                source="coherence_gauntlet",
                feedback_type="critique",
                content="Sub-problems lack clear dependencies",
                severity="major",
                actionable=True,
                timestamp=datetime.now(),
                metadata={'improvements': ['Add dependency relationships']}
            ),
            Feedback(
                id=generate_id("feedback"),
                source="completeness_gauntlet",
                feedback_type="critique",
                content="Missing validation criteria",
                severity="critical",
                actionable=True,
                timestamp=datetime.now(),
                metadata={'improvements': ['Add success criteria']}
            ),
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="suggestion",
                content="Consider edge cases",
                severity="minor",
                actionable=True,
                timestamp=datetime.now()
            )
        ]
    
    def test_coordinator_initialization(self, coordinator):
        """Test coordinator can be initialized."""
        assert coordinator is not None
        assert coordinator.gauntlet_system is not None
        assert coordinator.quality_assessor is not None
        assert coordinator.team_coordinator is not None
    
    def test_process_feedback(self, coordinator, sample_plan, sample_feedback):
        """Test feedback processing."""
        result = coordinator.process_feedback(sample_plan, sample_feedback)
        
        assert result['total_feedback'] == 3
        assert 'categorized' in result
        assert 'prioritized' in result
        assert 'improvements' in result
        assert result['critical_count'] == 1
        assert len(result['critical_issues']) == 1
        assert result['actionable'] is True
    
    def test_feedback_categorization(self, coordinator, sample_feedback):
        """Test feedback is categorized by type."""
        categorized = coordinator._categorize_feedback(sample_feedback)
        
        assert 'critique' in categorized
        assert 'suggestion' in categorized
        assert len(categorized['critique']) == 2
        assert len(categorized['suggestion']) == 1
    
    def test_feedback_prioritization(self, coordinator, sample_feedback):
        """Test feedback is prioritized by severity."""
        prioritized = coordinator._prioritize_feedback(sample_feedback)
        
        # Critical should be first
        assert prioritized[0].severity == 'critical'
        # Major should be second
        assert prioritized[1].severity == 'major'
        # Minor should be last
        assert prioritized[2].severity == 'minor'
    
    def test_generate_improvements(self, coordinator, sample_plan, sample_feedback):
        """Test improvement generation from feedback."""
        improvements = coordinator._generate_improvements(sample_feedback, sample_plan)
        
        assert len(improvements) > 0
        # Should extract improvements from metadata
        assert any('dependency' in imp.lower() for imp in improvements)
        assert any('criteria' in imp.lower() for imp in improvements)
    
    def test_generate_refinement_plan(self, coordinator, sample_plan, sample_feedback):
        """Test refinement plan generation."""
        refinement_plan = coordinator.generate_refinement_plan(sample_plan, sample_feedback)
        
        assert refinement_plan is not None
        assert refinement_plan.plan_id == sample_plan.id
        assert len(refinement_plan.issues) == 3
        assert len(refinement_plan.improvements) > 0
        assert len(refinement_plan.priority_order) == 3
        assert refinement_plan.estimated_effort > 0
        
        # Critical issue should be first in priority
        first_issue = next(i for i in refinement_plan.issues 
                          if i['id'] == refinement_plan.priority_order[0])
        assert first_issue['severity'] == 'critical'
    
    def test_estimate_refinement_effort(self, coordinator):
        """Test effort estimation."""
        issues = [
            {'severity': 'critical'},
            {'severity': 'major'},
            {'severity': 'minor'}
        ]
        
        effort = coordinator._estimate_refinement_effort(issues)
        
        # Critical=4h, Major=2h, Minor=1h = 7h total
        assert effort == 7
    
    def test_execute_refinement(self, coordinator, sample_plan, sample_feedback):
        """Test refinement execution."""
        refinement_plan = coordinator.generate_refinement_plan(sample_plan, sample_feedback)
        
        refined_plan, metrics = coordinator.execute_refinement(sample_plan, refinement_plan)
        
        assert refined_plan is not None
        assert isinstance(metrics, RefinementMetrics)
        assert metrics.total_cycles == 1
        assert metrics.time_spent >= 0
    
    def test_track_refinement_cycles(self, coordinator, sample_plan):
        """Test refinement cycle tracking."""
        result = coordinator.track_refinement_cycles(
            sample_plan,
            max_cycles=3,
            convergence_threshold=0.01
        )
        
        assert result['plan_id'] == sample_plan.id
        assert result['total_cycles'] > 0
        assert result['total_cycles'] <= 3
        assert 'converged' in result
        assert 'final_quality' in result
        assert 'cycles' in result
    
    def test_refinement_history_tracking(self, coordinator, sample_plan):
        """Test refinement history is tracked."""
        # Run refinement cycles
        coordinator.track_refinement_cycles(sample_plan, max_cycles=2)
        
        # Check history
        history = coordinator.get_refinement_history(sample_plan.id)
        
        assert len(history) > 0
        assert all(isinstance(cycle, RefinementCycle) for cycle in history)
        assert all(cycle.plan_id == sample_plan.id for cycle in history)
        assert all(cycle.plan_id == sample_plan.id for cycle in history)
    
    def test_convergence_metrics(self, coordinator, sample_plan):
        """Test convergence metrics calculation."""
        # Run refinement cycles
        coordinator.track_refinement_cycles(sample_plan, max_cycles=2)
        
        # Get metrics
        metrics = coordinator.get_convergence_metrics(sample_plan.id)
        
        assert metrics['has_data'] is True
        assert metrics['total_cycles'] > 0
        assert 'quality_progression' in metrics
        assert 'total_improvement' in metrics
        assert 'converged' in metrics
    
    def test_convergence_detection(self, coordinator, sample_plan):
        """Test early stopping when converged."""
        result = coordinator.track_refinement_cycles(
            sample_plan,
            max_cycles=10,
            convergence_threshold=0.5  # High threshold for quick convergence
        )
        
        # Should converge before max cycles
        assert result['total_cycles'] < 10 or result['converged']
    
    def test_max_cycles_limit(self, coordinator, sample_plan):
        """Test max cycles limit is respected."""
        result = coordinator.track_refinement_cycles(
            sample_plan,
            max_cycles=2,
            convergence_threshold=0.001  # Low threshold to prevent early convergence
        )
        
        # Should stop at max cycles
        assert result['total_cycles'] <= 2
    
    def test_severity_scoring(self, coordinator):
        """Test severity to score conversion."""
        assert coordinator._severity_score('critical') == 4
        assert coordinator._severity_score('major') == 3
        assert coordinator._severity_score('minor') == 2
        assert coordinator._severity_score('info') == 1
        assert coordinator._severity_score('unknown') == 0


class TestRefinementDataModels:
    """Test refinement data models."""
    
    def test_refinement_plan_creation(self):
        """Test RefinementPlan can be created."""
        plan = RefinementPlan(
            id="test_plan",
            plan_id="decomp_plan",
            issues=[],
            improvements=[],
            priority_order=[],
            estimated_effort=5
        )
        
        assert plan.id == "test_plan"
        assert plan.estimated_effort == 5
    
    def test_refinement_cycle_creation(self):
        """Test RefinementCycle can be created."""
        cycle = RefinementCycle(
            cycle_number=1,
            plan_id="test_plan",
            feedback_received=[],
            improvements_applied=[],
            quality_before=0.7,
            quality_after=0.8,
            gauntlet_results={},
            converged=False
        )
        
        assert cycle.cycle_number == 1
        assert cycle.quality_after > cycle.quality_before
    
    def test_refinement_metrics_creation(self):
        """Test RefinementMetrics can be created."""
        metrics = RefinementMetrics(
            total_cycles=3,
            quality_improvement=0.15,
            issues_resolved=5,
            issues_remaining=2,
            convergence_rate=1.2,
            time_spent=2.5
        )
        
        assert metrics.total_cycles == 3
        assert metrics.quality_improvement > 0


class TestRefinementIntegration:
    """Test refinement integration with other systems."""
    
    def test_integration_with_gauntlets(self):
        """Test refinement integrates with gauntlet system."""
        from sovereign_gauntlets import GauntletSystem
        
        gauntlet_system = GauntletSystem()
        coordinator = RefinementCoordinator(gauntlet_system=gauntlet_system)
        
        assert coordinator.gauntlet_system is gauntlet_system
    
    def test_integration_with_quality_assessor(self):
        """Test refinement integrates with quality assessor."""
        from sovereign_quality_assessment import QualityAssessor
        
        assessor = QualityAssessor()
        coordinator = RefinementCoordinator(quality_assessor=assessor)
        
        assert coordinator.quality_assessor is assessor
    
    def test_integration_with_team_coordinator(self):
        """Test refinement integrates with team coordinator."""
        from sovereign_team_coordination import TeamCoordinator
        
        team_coord = TeamCoordinator()
        coordinator = RefinementCoordinator(team_coordinator=team_coord)
        
        assert coordinator.team_coordinator is team_coord


def test_entanglement_propagation_marks_consistency_refinement():
    """Ensure entanglement invalidation marks solved peers for consistency refinement."""
    complexity = ComplexityScore("test", 1, 1, 1, 1, 1)
    sp_a = SubProblem(
        id="A",
        parent_id="P",
        title="Sub A",
        description="def func_a(): pass",
        type=SubProblemType.ANALYSIS,
        complexity_score=complexity,
        dependencies=[],
        success_criteria=[],
        validation_gauntlet="coherence",
        priority=5,
        estimated_effort=1
    )
    sp_b = SubProblem(
        id="B",
        parent_id="P",
        title="Sub B",
        description="def func_b(): func_a()",
        type=SubProblemType.ANALYSIS,
        complexity_score=complexity,
        dependencies=[],
        success_criteria=[],
        validation_gauntlet="coherence",
        priority=5,
        estimated_effort=1
    )
    plan = DecompositionPlan(
        id="plan",
        problem_id="P",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=[sp_a, sp_b],
        dependency_graph=DependencyGraph(
            nodes={"A": sp_a, "B": sp_b},
            edges={},
            critical_path=[],
            parallel_groups=[],
            execution_order=[]
        ),
        quality_scores=QualityScores(
            coherence_score=0.7,
            completeness_score=0.7,
            feasibility_score=0.7,
            integration_score=0.7,
            overall_score=0.7,
            meets_thresholds=False,
            details={},
            timestamp=datetime.now()
        ),
        confidence_level=0.7,
        created_by="test",
        approved_by=None,
        status="draft",
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

    coordinator = FractalPipelineCoordinator()
    coordinator.entanglement_matrix = {"A": {"B"}, "B": set()}

    sub_solutions = {
        "A": SolutionAttempt(
            id=generate_id("solution_attempt"),
            sub_problem_id="A",
            approach="test",
            solution_content="ok",
            team_id="team",
            confidence_score=0.8,
            status="solved"
        ),
        "B": SolutionAttempt(
            id=generate_id("solution_attempt"),
            sub_problem_id="B",
            approach="test",
            solution_content="ok",
            team_id="team",
            confidence_score=0.8,
            status="solved"
        )
    }

    coordinator._propagate_entanglement("A", sub_solutions, plan)

    assert sub_solutions["B"].status == "needs_consistency_refinement"
    assert sub_solutions["B"].metadata.get("entanglement_invalidation") == ["A"]
    assert sp_b.metadata.get("needs_consistency_refinement") is True


def test_z3_refutation_narrative_includes_constraints():
    """Verify refutation narrative includes contradictions and constraints."""
    constraints = ["x > 10", "x < 5"]
    narrative = generate_refutation_narrative("unsat", constraints=constraints)

    assert "contradiction" in narrative.lower()
    assert "x > 10" in narrative
    assert "x < 5" in narrative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
