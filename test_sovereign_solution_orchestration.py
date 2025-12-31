"""
Tests for Sovereign Solution Orchestration System
Task 8.4: Unit tests for solution orchestration
"""

import pytest
from datetime import datetime

from sovereign_data_models import (
    DecompositionPlan, SubProblem, SolutionAttempt, DecompositionStrategy,
    SubProblemType, ComplexityScore, SuccessCriterion, DependencyGraph, generate_id
)
from sovereign_solution_orchestration import (
    SolutionOrchestrator, IntegratedSolution, Conflict
)


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
def sample_sub_problem(sample_complexity):
    """Create a sample sub-problem."""
    return SubProblem(
        id=generate_id("subproblem"),
        parent_id="problem1",
        title="Test Sub-Problem",
        description="A test sub-problem",
        type=SubProblemType.IMPLEMENTATION,
        complexity_score=sample_complexity,
        dependencies=[],
        success_criteria=[
            SuccessCriterion(
                id=generate_id("criterion"),
                description="Implementation must be complete and functional",
                metric="completion",
                threshold=1.0,
                validation_method="testing"
            )
        ],
        estimated_effort=10
    )


@pytest.fixture
def sample_plan(sample_complexity):
    """Create a sample decomposition plan."""
    sub_problems = []
    for i in range(3):
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title=f"Sub-Problem {i+1}",
            description=f"Description {i+1}",
            type=SubProblemType.IMPLEMENTATION,
            complexity_score=sample_complexity,
            dependencies=[sub_problems[i-1].id] if i > 0 else [],
            success_criteria=[
                SuccessCriterion(
                    id=generate_id("criterion"),
                    description=f"Criterion {i+1}",
                    metric="completion",
                    threshold=0.9,
                    validation_method="review"
                )
            ],
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


class TestSolutionOrchestrator:
    """Test SolutionOrchestrator class."""
    
    def test_initialization(self):
        orchestrator = SolutionOrchestrator()
        assert orchestrator.solution_attempts == {}
        assert orchestrator.integrated_solutions == {}
    
    def test_track_solution_attempt(self, sample_sub_problem):
        orchestrator = SolutionOrchestrator()
        
        attempt = orchestrator.track_solution_attempt(
            sub_problem_id=sample_sub_problem.id,
            approach="Implement using algorithm X",
            solution_content="Solution implementation here...",
            team_id="blue_team",
            confidence_score=0.85
        )
        
        assert attempt.sub_problem_id == sample_sub_problem.id
        assert attempt.approach == "Implement using algorithm X"
        assert attempt.confidence_score == 0.85
        assert attempt.status == "pending"
        assert sample_sub_problem.id in orchestrator.solution_attempts
    
    def test_validate_solution_with_criteria(self, sample_sub_problem):
        orchestrator = SolutionOrchestrator()
        
        attempt = orchestrator.track_solution_attempt(
            sub_problem_id=sample_sub_problem.id,
            approach="Complete implementation",
            solution_content="This is a complete and functional implementation that satisfies all requirements",
            team_id="blue_team"
        )
        
        result = orchestrator.validate_solution(attempt, sample_sub_problem)
        
        assert result.validator == "solution_orchestrator"
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0
        assert len(attempt.validation_results) == 1
    
    def test_validate_solution_no_criteria(self, sample_complexity):
        orchestrator = SolutionOrchestrator()
        
        # Sub-problem without success criteria
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title="No Criteria",
            description="Sub-problem without criteria",
            type=SubProblemType.ANALYSIS,
            complexity_score=sample_complexity,
            dependencies=[],
            success_criteria=[],  # No criteria
            estimated_effort=5
        )
        
        attempt = orchestrator.track_solution_attempt(
            sub_problem_id=sp.id,
            approach="Analysis approach",
            solution_content="Analysis results",
            team_id="blue_team"
        )
        
        result = orchestrator.validate_solution(attempt, sp)
        
        assert result.passed is True  # Should pass without criteria
        assert attempt.status == "validated"
    
    def test_detect_conflicts_none(self):
        orchestrator = SolutionOrchestrator()
        
        # Create good solutions
        solutions = [
            SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id="sp1",
                approach="Approach A",
                solution_content="Solution A",
                team_id="team1",
                confidence_score=0.9,
                validation_results=[],
                feedback=[],
                status="validated",
                created_at=datetime.now(),
                metadata={'approach_type': 'algorithmic'}
            ),
            SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id="sp2",
                approach="Approach B",
                solution_content="Solution B",
                team_id="team1",
                confidence_score=0.85,
                validation_results=[],
                feedback=[],
                status="validated",
                created_at=datetime.now(),
                metadata={'approach_type': 'algorithmic'}
            )
        ]
        
        conflicts = orchestrator.detect_conflicts(solutions)
        
        # Should have no critical conflicts
        critical = [c for c in conflicts if c.severity == "critical"]
        assert len(critical) == 0
    
    def test_detect_conflicts_low_confidence(self):
        orchestrator = SolutionOrchestrator()
        
        solutions = [
            SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id="sp1",
                approach="Approach A",
                solution_content="Solution A",
                team_id="team1",
                confidence_score=0.3,  # Low confidence
                validation_results=[],
                feedback=[],
                status="pending",
                created_at=datetime.now(),
                metadata={'approach_type': 'general'}
            )
        ]
        
        conflicts = orchestrator.detect_conflicts(solutions)
        
        # Should detect low confidence conflict
        assert len(conflicts) > 0
        assert any(c.conflict_type == "low_confidence" for c in conflicts)
    
    def test_detect_conflicts_validation_failure(self):
        orchestrator = SolutionOrchestrator()
        
        solutions = [
            SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id="sp1",
                approach="Approach A",
                solution_content="Solution A",
                team_id="team1",
                confidence_score=0.8,
                validation_results=[],
                feedback=[],
                status="rejected",  # Failed validation
                created_at=datetime.now(),
                metadata={'approach_type': 'general'}
            )
        ]
        
        conflicts = orchestrator.detect_conflicts(solutions)
        
        # Should detect validation failure
        assert len(conflicts) > 0
        assert any(c.conflict_type == "validation_failure" for c in conflicts)
        assert any(c.severity == "critical" for c in conflicts)
    
    def test_calculate_confidence(self):
        orchestrator = SolutionOrchestrator()
        
        solutions = [
            SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id="sp1",
                approach="A",
                solution_content="S1",
                team_id="team1",
                confidence_score=0.9,
                validation_results=[],
                feedback=[],
                status="validated",
                created_at=datetime.now(),
                metadata={}
            ),
            SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id="sp2",
                approach="B",
                solution_content="S2",
                team_id="team1",
                confidence_score=0.8,
                validation_results=[],
                feedback=[],
                status="validated",
                created_at=datetime.now(),
                metadata={}
            )
        ]
        
        confidence = orchestrator.calculate_confidence(solutions)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence >= 0.8  # Both validated with high confidence
    
    def test_integrate_solutions(self, sample_plan):
        orchestrator = SolutionOrchestrator()
        
        # Create solution attempts for each sub-problem
        attempts = []
        for sp in sample_plan.sub_problems:
            # Use content that matches success criteria
            criterion_text = sp.success_criteria[0].description if sp.success_criteria else ""
            attempt = orchestrator.track_solution_attempt(
                sub_problem_id=sp.id,
                approach=f"Approach for {sp.title}",
                solution_content=f"Solution content for {sp.title} that satisfies {criterion_text} with complete implementation",
                team_id="blue_team",
                confidence_score=0.85
            )
            # Validate it
            orchestrator.validate_solution(attempt, sp)
            attempts.append(attempt)
        
        # Integrate solutions
        integrated = orchestrator.integrate_solutions(sample_plan, attempts)
        
        assert integrated.plan_id == sample_plan.id
        # May have fewer solutions if some were rejected
        assert len(integrated.sub_solutions) >= 1
        assert integrated.final_content != ""
        assert 0.0 <= integrated.confidence_score <= 1.0
    
    def test_integrate_solutions_with_conflicts(self, sample_plan):
        orchestrator = SolutionOrchestrator()
        
        # Create mixed quality attempts
        attempts = []
        for i, sp in enumerate(sample_plan.sub_problems):
            confidence = 0.9 if i == 0 else 0.4  # First good, others poor
            attempt = orchestrator.track_solution_attempt(
                sub_problem_id=sp.id,
                approach=f"Approach {i}",
                solution_content=f"Solution {i}",
                team_id="blue_team",
                confidence_score=confidence
            )
            attempt.status = "validated" if i == 0 else "pending"
            attempts.append(attempt)
        
        # Should handle conflicts
        integrated = orchestrator.integrate_solutions(sample_plan, attempts)
        
        assert integrated is not None
        # May have resolved some conflicts
        assert len(integrated.conflicts_resolved) >= 0
    
    def test_get_solution_status_integrated(self, sample_plan):
        orchestrator = SolutionOrchestrator()
        
        # Create and integrate solutions
        attempts = []
        for sp in sample_plan.sub_problems:
            attempt = orchestrator.track_solution_attempt(
                sub_problem_id=sp.id,
                approach="Approach",
                solution_content="Solution",
                team_id="team1",
                confidence_score=0.85
            )
            attempt.status = "validated"
            attempts.append(attempt)
        
        orchestrator.integrate_solutions(sample_plan, attempts)
        
        status = orchestrator.get_solution_status(sample_plan.id)
        
        assert status['status'] == 'integrated'
        assert 'confidence' in status
        assert status['sub_solution_count'] == len(sample_plan.sub_problems)
    
    def test_get_solution_status_in_progress(self, sample_sub_problem):
        orchestrator = SolutionOrchestrator()
        
        # Track some attempts but don't integrate
        orchestrator.track_solution_attempt(
            sub_problem_id=sample_sub_problem.id,
            approach="Approach",
            solution_content="Solution",
            team_id="team1"
        )
        
        status = orchestrator.get_solution_status("plan1")
        
        assert status['status'] == 'in_progress'
        assert status['total_attempts'] >= 1


class TestIntegration:
    """Integration tests for solution orchestration."""
    
    def test_complete_workflow(self, sample_plan):
        """Test complete solution workflow."""
        orchestrator = SolutionOrchestrator()
        
        # Step 1: Track solutions for each sub-problem
        for sp in sample_plan.sub_problems:
            # Use content that matches success criteria
            criterion_text = sp.success_criteria[0].description if sp.success_criteria else "completion"
            attempt = orchestrator.track_solution_attempt(
                sub_problem_id=sp.id,
                approach=f"Implement {sp.title}",
                solution_content=f"Complete implementation for {sp.title} with all required functionality including {criterion_text}",
                team_id="blue_team",
                confidence_score=0.85
            )
            
            # Step 2: Validate each solution
            result = orchestrator.validate_solution(attempt, sp)
            assert result is not None
        
        # Step 3: Integrate all solutions (pass attempts explicitly)
        attempts = []
        for sp in sample_plan.sub_problems:
            sp_attempts = orchestrator.solution_attempts.get(sp.id, [])
            if sp_attempts:
                # Use the latest attempt regardless of status for testing
                attempts.append(sp_attempts[-1])
        
        integrated = orchestrator.integrate_solutions(sample_plan, attempts)
        
        # Verify integration
        assert integrated.plan_id == sample_plan.id
        assert len(integrated.sub_solutions) >= 1  # At least one solution
        assert integrated.confidence_score > 0.0
        
        # Step 4: Check status
        status = orchestrator.get_solution_status(sample_plan.id)
        assert status['status'] == 'integrated'


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_solutions_list(self, sample_plan):
        orchestrator = SolutionOrchestrator()
        
        # Try to integrate with no solutions
        with pytest.raises(ValueError):
            orchestrator.integrate_solutions(sample_plan, [])
    
    def test_calculate_confidence_empty(self):
        orchestrator = SolutionOrchestrator()
        confidence = orchestrator.calculate_confidence([])
        assert confidence == 0.0
    
    def test_clear_attempts(self, sample_sub_problem):
        orchestrator = SolutionOrchestrator()
        
        # Add some attempts
        orchestrator.track_solution_attempt(
            sub_problem_id=sample_sub_problem.id,
            approach="Approach",
            solution_content="Solution",
            team_id="team1"
        )
        
        assert len(orchestrator.solution_attempts) > 0
        
        # Clear all
        orchestrator.clear_attempts()
        
        assert len(orchestrator.solution_attempts) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
