"""
Tests for Sovereign Team Coordination System
Task 6.6: Unit tests for team coordination
"""

import pytest
from datetime import datetime

from sovereign_data_models import (
    DecompositionPlan, SubProblem, DecompositionStrategy, SubProblemType,
    ComplexityScore, SuccessCriterion, Feedback, generate_id
)
from sovereign_team_coordination import (
    TeamAssignmentManager, TeamCoordinator, DecompositionWorkflow,
    TeamCapacity, RefinementRequest, GoldEvaluation
)
from sovereign_gauntlets import GauntletSystem


@pytest.fixture
def sample_plan():
    """Create a sample decomposition plan."""
    complexity = ComplexityScore(
        cognitive_complexity=5.0,
        computational_complexity=5.0,
        domain_complexity=5.0,
        integration_complexity=5.0,
        overall_complexity=5.0,
        explanation="Medium complexity"
    )
    
    sub_problems = []
    for i in range(3):
        sp = SubProblem(
            id=generate_id("subproblem"),
            parent_id="problem1",
            title=f"Sub-Problem {i+1}",
            description=f"Description for sub-problem {i+1}",
            type=[SubProblemType.RESEARCH, SubProblemType.ANALYSIS, SubProblemType.IMPLEMENTATION][i],
            complexity_score=complexity,
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
            priority=8 - i,
            estimated_effort=10
        )
        sub_problems.append(sp)
    
    return DecompositionPlan(
        id=generate_id("plan"),
        problem_id="problem1",
        strategy=DecompositionStrategy.SEMANTIC,
        sub_problems=sub_problems,
        confidence_level=0.85
    )


class TestTeamAssignmentManager:
    """Test TeamAssignmentManager."""
    
    def test_assign_to_team(self):
        manager = TeamAssignmentManager()
        
        assignment = manager.assign_to_team(
            task_id="task1",
            team="red",
            priority=8
        )
        
        assert assignment.task_id == "task1"
        assert assignment.team == "red"
        assert assignment.status == "assigned"
        assert assignment.metadata['priority'] == 8
    
    def test_track_capacity(self):
        manager = TeamAssignmentManager()
        
        # Assign some tasks
        manager.assign_to_team("task1", "red")
        manager.assign_to_team("task2", "red")
        
        capacity = manager.track_team_capacity("red")
        assert capacity.team_name == "red"
        assert capacity.current_tasks == 2
        assert capacity.max_concurrent_tasks == 5
    
    def test_complete_assignment(self):
        manager = TeamAssignmentManager()
        
        assignment = manager.assign_to_team("task1", "blue")
        initial_tasks = manager.team_capacity['blue'].current_tasks
        
        success = manager.complete_assignment(assignment.id)
        assert success is True
        assert assignment.status == "completed"
        assert manager.team_capacity['blue'].current_tasks == initial_tasks - 1
    
    def test_get_team_workload(self):
        manager = TeamAssignmentManager()
        
        # Assign multiple tasks
        for i in range(3):
            manager.assign_to_team(f"task{i}", "gold")
        
        workload = manager.get_team_workload("gold")
        assert workload['team'] == "gold"
        assert workload['current_tasks'] == 3
        assert workload['utilization'] == 3 / 3  # 3 out of 3 max
        assert workload['pending_assignments'] == 3
    
    def test_capacity_warning(self):
        manager = TeamAssignmentManager()
        
        # Fill up team capacity
        for i in range(6):  # More than max of 5
            manager.assign_to_team(f"task{i}", "red")
        
        capacity = manager.track_team_capacity("red")
        assert capacity.current_tasks > capacity.max_concurrent_tasks


class TestTeamCoordinator:
    """Test TeamCoordinator."""
    
    def test_assign_decomposition_review(self, sample_plan):
        coordinator = TeamCoordinator()
        
        assignment = coordinator.assign_decomposition_review(sample_plan)
        
        assert assignment.team == "red"
        assert assignment.task_id == sample_plan.id
        assert 'gauntlet_results' in assignment.metadata
    
    def test_process_red_team_feedback(self):
        coordinator = TeamCoordinator()
        
        feedback = [
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="critique",
                content="Critical issue found",
                severity="critical",
                actionable=True
            ),
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="critique",
                content="Minor issue",
                severity="minor",
                actionable=True
            )
        ]
        
        request = coordinator.process_red_team_feedback("plan1", feedback)
        
        assert request.plan_id == "plan1"
        assert len(request.feedback) == 2
        assert request.priority == 10  # Critical issue = priority 10
        assert request.requested_by == "red_team"
    
    def test_coordinate_refinement(self):
        coordinator = TeamCoordinator()
        
        feedback = [
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="critique",
                content="Needs improvement",
                severity="major",
                actionable=True
            )
        ]
        
        request = RefinementRequest(
            plan_id="plan1",
            feedback=feedback,
            priority=8,
            requested_by="red_team",
            requested_at=datetime.now()
        )
        
        assignment = coordinator.coordinate_refinement(request)
        
        assert assignment.team == "blue"
        assert assignment.task_id == "plan1"
        assert 'refinement_request' in assignment.metadata
    
    def test_request_gold_evaluation(self, sample_plan):
        coordinator = TeamCoordinator()
        
        assignment = coordinator.request_gold_evaluation(sample_plan)
        
        assert assignment.team == "gold"
        assert assignment.task_id == sample_plan.id
        assert 'gauntlet_check' in assignment.metadata
    
    def test_record_gold_evaluation(self):
        coordinator = TeamCoordinator()
        
        evaluation = coordinator.record_gold_evaluation(
            plan_id="plan1",
            approved=True,
            overall_score=0.92,
            strengths=["Well structured", "Clear criteria"],
            weaknesses=[],
            recommendations=["Consider edge cases"]
        )
        
        assert evaluation.plan_id == "plan1"
        assert evaluation.approved is True
        assert evaluation.overall_score == 0.92
        assert len(evaluation.strengths) == 2
        assert evaluation.evaluated_by == "gold_team"
    
    def test_balance_workload(self):
        coordinator = TeamCoordinator()
        
        # Assign tasks to different teams
        coordinator.assignment_manager.assign_to_team("task1", "red")
        coordinator.assignment_manager.assign_to_team("task2", "red")
        coordinator.assignment_manager.assign_to_team("task3", "blue")
        coordinator.assignment_manager.assign_to_team("task4", "gold")
        
        balance = coordinator.balance_workload()
        
        assert 'red_team' in balance
        assert 'blue_team' in balance
        assert 'gold_team' in balance
        assert 'avg_utilization' in balance
        assert 'balance_score' in balance
        assert isinstance(balance['needs_rebalancing'], bool)
    
    def test_get_plan_workflow_status(self, sample_plan):
        coordinator = TeamCoordinator()
        
        # Simulate workflow
        coordinator.assign_decomposition_review(sample_plan)
        
        feedback = [
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="critique",
                content="Issue",
                severity="major",
                actionable=True
            )
        ]
        coordinator.process_red_team_feedback(sample_plan.id, feedback)
        
        coordinator.record_gold_evaluation(
            plan_id=sample_plan.id,
            approved=True,
            overall_score=0.88,
            strengths=["Good"],
            weaknesses=[],
            recommendations=[]
        )
        
        status = coordinator.get_plan_workflow_status(sample_plan.id)
        
        assert status['plan_id'] == sample_plan.id
        assert status['total_refinements'] == 1
        assert status['total_evaluations'] == 1
        assert status['approved'] is True
        assert len(status['assignments']) > 0


class TestDecompositionWorkflow:
    """Test DecompositionWorkflow."""
    
    def test_validate_and_refine(self, sample_plan):
        workflow = DecompositionWorkflow()
        
        result = workflow.validate_and_refine(sample_plan, max_refinement_cycles=2)
        
        assert 'plan_id' in result
        assert 'approved' in result
        assert 'refinement_cycles' in result
        assert 'final_score' in result
        assert 'evaluation' in result
        assert 'workflow_status' in result
        
        assert result['plan_id'] == sample_plan.id
        assert isinstance(result['approved'], bool)
        assert result['refinement_cycles'] >= 0
        assert 0.0 <= result['final_score'] <= 1.0
    
    def test_workflow_with_good_plan(self, sample_plan):
        workflow = DecompositionWorkflow()
        
        # Good plan should be approved quickly
        result = workflow.validate_and_refine(sample_plan, max_refinement_cycles=3)
        
        # Should pass with minimal refinement
        assert result['refinement_cycles'] <= 2
        assert result['final_score'] >= 0.7
    
    def test_workflow_tracks_history(self, sample_plan):
        workflow = DecompositionWorkflow()
        
        result = workflow.validate_and_refine(sample_plan)
        
        # Check that history is tracked
        status = workflow.coordinator.get_plan_workflow_status(sample_plan.id)
        assert status['total_evaluations'] >= 1
        assert len(status['assignments']) > 0


class TestIntegration:
    """Integration tests for team coordination."""
    
    def test_complete_workflow_integration(self, sample_plan):
        """Test complete workflow from review to approval."""
        coordinator = TeamCoordinator()
        
        # Step 1: Red Team Review
        red_assignment = coordinator.assign_decomposition_review(sample_plan)
        assert red_assignment.team == "red"
        
        # Step 2: Process feedback
        feedback = [
            Feedback(
                id=generate_id("feedback"),
                source="red_team",
                feedback_type="critique",
                content="Minor improvement needed",
                severity="minor",
                actionable=True
            )
        ]
        refinement_request = coordinator.process_red_team_feedback(sample_plan.id, feedback)
        assert refinement_request.plan_id == sample_plan.id
        
        # Step 3: Blue Team Refinement
        blue_assignment = coordinator.coordinate_refinement(refinement_request)
        assert blue_assignment.team == "blue"
        
        # Step 4: Gold Team Evaluation
        gold_assignment = coordinator.request_gold_evaluation(sample_plan)
        assert gold_assignment.team == "gold"
        
        # Step 5: Record evaluation
        evaluation = coordinator.record_gold_evaluation(
            plan_id=sample_plan.id,
            approved=True,
            overall_score=0.90,
            strengths=["Excellent structure"],
            weaknesses=[],
            recommendations=[]
        )
        assert evaluation.approved is True
        
        # Verify complete workflow
        status = coordinator.get_plan_workflow_status(sample_plan.id)
        assert status['total_refinements'] == 1
        assert status['total_evaluations'] == 1
        assert status['approved'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
