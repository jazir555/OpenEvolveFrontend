"""
Sovereign-Grade Problem Decomposition System - Team Coordination
Integrates Red/Blue/Gold teams for decomposition validation and refinement.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from sovereign_data_models import (
    DecompositionPlan, SubProblem, TeamAssignment, Feedback,
    ValidationResult, SolutionAttempt, generate_id
)
from sovereign_gauntlets import GauntletSystem
from red_team import RedTeam, IssueFinding
from blue_team import BlueTeam
from evaluator_team import EvaluatorTeam, EvaluationThreshold
from sovereign_reliability import with_error_handling, ErrorSeverity

logger = logging.getLogger(__name__)


@dataclass
class TeamCapacity:
    """Tracks team capacity and workload."""
    team_name: str
    max_concurrent_tasks: int
    current_tasks: int
    avg_completion_time: float  # hours
    availability: float  # 0-1 scale


@dataclass
class RefinementRequest:
    """Request for decomposition refinement."""
    plan_id: str
    feedback: List[Feedback]
    priority: int
    requested_by: str
    requested_at: datetime
    error_message: Optional[str] = None


@dataclass
class GoldEvaluation:
    """Final evaluation from Gold Team."""
    plan_id: str
    approved: bool
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    evaluated_by: str
    evaluated_at: datetime
    error_message: Optional[str] = None


class TeamAssignmentManager:
    """Manages team assignments and workload balancing."""
    
    def __init__(self):
        self.assignments: Dict[str, TeamAssignment] = {}
        self.team_capacity: Dict[str, TeamCapacity] = {
            'red': TeamCapacity('red', 5, 0, 2.0, 1.0),
            'blue': TeamCapacity('blue', 5, 0, 3.0, 1.0),
            'gold': TeamCapacity('gold', 3, 0, 1.5, 1.0)
        }
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda task_id, team, priority, due_hours: TeamAssignment(
        id=generate_id("assignment"), task_id=task_id, team=team, status="error", metadata={"error": "Failed to create assignment"}
    ))
    def assign_to_team(
        self, 
        task_id: str, 
        team: str,
        priority: int = 5,
        due_hours: Optional[float] = None
    ) -> TeamAssignment:
        """
        Assigns task to appropriate team.
        
        Args:
            task_id: ID of the task (plan, sub-problem, etc.)
            team: Team name ('red', 'blue', 'gold')
            priority: Task priority (1-10)
            due_hours: Hours until due (optional)
            
        Returns:
            TeamAssignment object
        """
        self.logger.info(f"Assigning task {task_id} to {team} team")
        
        # Check capacity
        capacity = self.team_capacity.get(team)
        if capacity and capacity.current_tasks >= capacity.max_concurrent_tasks:
            self.logger.warning(f"{team} team at capacity ({capacity.current_tasks}/{capacity.max_concurrent_tasks})")
        
        # Calculate due date
        due_date = None
        if due_hours:
            due_date = datetime.now() + timedelta(hours=due_hours)
        elif capacity:
            due_date = datetime.now() + timedelta(hours=capacity.avg_completion_time)
        
        # Create assignment
        assignment = TeamAssignment(
            id=generate_id("assignment"),
            task_id=task_id,
            team=team,
            assigned_at=datetime.now(),
            due_date=due_date,
            status="assigned",
            metadata={'priority': priority}
        )
        
        # Track assignment
        self.assignments[assignment.id] = assignment
        if capacity:
            capacity.current_tasks += 1
        
        return assignment
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda assignment_id: False)
    def complete_assignment(self, assignment_id: str) -> bool:
        """Mark assignment as complete and update capacity."""
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            return False
        
        assignment.status = "completed"
        
        # Update capacity
        capacity = self.team_capacity.get(assignment.team)
        if capacity and capacity.current_tasks > 0:
            capacity.current_tasks -= 1
        
        return True
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda team: TeamCapacity(team, 0, 0, 0.0, 0.0))
    def track_team_capacity(self, team: str) -> TeamCapacity:
        """
        Monitors team workload and capacity.
        
        Args:
            team: Team name
            
        Returns:
            TeamCapacity object
        """
        return self.team_capacity.get(team, TeamCapacity(team, 0, 0, 0.0, 0.0))
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda: [])
    def optimize_assignments(self) -> List[TeamAssignment]:
        """
        Optimizes task assignments across teams.
        
        Returns:
            List of optimized assignments
        """
        # Simple optimization: balance load across teams
        optimized = []
        
        for assignment in self.assignments.values():
            if assignment.status == "assigned":
                capacity = self.team_capacity.get(assignment.team)
                if capacity and capacity.current_tasks > capacity.max_concurrent_tasks:
                    # Team overloaded - could reassign or delay
                    self.logger.warning(f"Team {assignment.team} overloaded")
                optimized.append(assignment)
        
        return optimized
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda team: {})
    def get_team_workload(self, team: str) -> Dict[str, Any]:
        """Get detailed workload information for a team."""
        capacity = self.team_capacity.get(team)
        if not capacity:
            return {}
        
        team_assignments = [a for a in self.assignments.values() 
                          if a.team == team and a.status == "assigned"]
        
        return {
            'team': team,
            'current_tasks': capacity.current_tasks,
            'max_concurrent': capacity.max_concurrent_tasks,
            'utilization': capacity.current_tasks / capacity.max_concurrent_tasks if capacity.max_concurrent_tasks > 0 else 0,
            'availability': capacity.availability,
            'avg_completion_time': capacity.avg_completion_time,
            'pending_assignments': len(team_assignments)
        }


class TeamCoordinator:
    """Coordinates AI teams for decomposition validation."""
    
    def __init__(self, openevolve_client=None):
        """
        Initialize team coordinator.
        
        Args:
            openevolve_client: Optional OpenEvolve client for LLM interactions.
        """
        self.openevolve_client = openevolve_client
        self.gauntlet_system = GauntletSystem(openevolve_client=self.openevolve_client)
        self.assignment_manager = TeamAssignmentManager()
        try:
            # Instantiate teams. They will handle their own OpenEvolve integration internally.
            self.red_team = RedTeam()
            self.blue_team = BlueTeam()
            self.evaluator_team = EvaluatorTeam()
        except Exception as e:
            self.logger.error(f"Failed to initialize one or more team instances: {e}", exc_info=True)
            # Fallback to dummy teams or raise a critical error
            self.red_team = None # type: ignore
            self.blue_team = None # type: ignore
            self.evaluator_team = None # type: ignore
            raise RuntimeError("Critical error: Team initialization failed. Cannot proceed with team coordination.") from e
        self.logger = logging.getLogger(__name__)
        
        # Track refinement history
        self.refinement_history: Dict[str, List[RefinementRequest]] = {}
        self.evaluation_history: Dict[str, List[GoldEvaluation]] = {}
    
    def _convert_issue_findings_to_feedback(
        self, 
        findings: List[IssueFinding]
    ) -> List[Feedback]:
        """Converts a list of IssueFinding objects to a list of Feedback objects."""
        feedback_list = []
        for issue in findings:
            try:
                feedback = Feedback(
                    id=generate_id("feedback"),
                    source='red_team',
                    feedback_type='critique',
                    content=issue.description,
                    severity=issue.severity.value if hasattr(issue.severity, 'value') else str(issue.severity),
                    actionable=bool(issue.suggested_fix),
                    timestamp=datetime.now(),
                    metadata={
                        'title': issue.title,
                        'category': issue.category.value if hasattr(issue.category, 'value') else str(issue.category),
                        'location': issue.location,
                        'confidence': issue.confidence,
                        'suggested_fix': issue.suggested_fix,
                        'exploit_example': issue.exploit_example
                    }
                )
                feedback_list.append(feedback)
            except Exception as e:
                self.logger.error(f"Failed to convert IssueFinding to Feedback for issue '{issue.title if hasattr(issue, 'title') else 'unknown'}': {e}", exc_info=True)
                # Continue to process other issues or return partial list
                continue
        return feedback_list

    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda plan, priority: TeamAssignment(
        id=generate_id("assignment"), task_id=plan.id, team='red', status="error", metadata={"error": "Failed to assign decomposition review"}
    ))
    def assign_decomposition_review(
        self, 
        plan: DecompositionPlan,
        priority: int = 5
    ) -> TeamAssignment:
        """
        Assigns decomposition to Red Team for review.
        
        Args:
            plan: The decomposition plan to review
            priority: Review priority (1-10)
            
        Returns:
            TeamAssignment for Red Team
        """
        self.logger.info(f"Assigning plan {plan.id} to Red Team for review")
        
        # First run gauntlets for automated validation
        gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(plan)
        
        # Create assignment
        assignment = self.assignment_manager.assign_to_team(
            task_id=plan.id,
            team='red',
            priority=priority,
            due_hours=2.0  # Red team review typically takes 2 hours
        )
        
        try:
            # Store gauntlet results in metadata
            assignment.metadata['gauntlet_results'] = {
                name: {'passed': result.passed, 'score': result.score}
                for name, result in gauntlet_results.items()
            }
        except Exception as e:
            self.logger.error(f"Failed to store gauntlet results in assignment metadata for plan {plan.id}: {e}", exc_info=True)
            # Continue without gauntlet results in metadata or set to empty
            assignment.metadata['gauntlet_results'] = {'error': f'Failed to retrieve gauntlet results: {e}'}
        
        return assignment
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda plan_id, feedback: RefinementRequest(
        plan_id=plan_id, feedback=[], priority=0, requested_by="system", requested_at=datetime.now(), error_message="Failed to process red team feedback"
    ))
    def process_red_team_feedback(
        self, 
        plan_id: str,
        feedback: List[Feedback]
    ) -> RefinementRequest:
        """
        Routes Red Team feedback to Blue Team.
        
        Args:
            plan_id: ID of the decomposition plan
            feedback: List of feedback from Red Team
            
        Returns:
            RefinementRequest for Blue Team
        """
        self.logger.info(f"Processing Red Team feedback for plan {plan_id}")
        
        try:
            # Analyze feedback severity
            critical_count = sum(1 for f in feedback if f.severity == "critical")
            major_count = sum(1 for f in feedback if f.severity == "major")
            
            # Determine priority based on severity
            if critical_count > 0:
                priority = 10
            elif major_count > 2:
                priority = 8
            elif major_count > 0:
                priority = 6
            else:
                priority = 4
            
            # Create refinement request
            request = RefinementRequest(
                plan_id=plan_id,
                feedback=feedback,
                priority=priority,
                requested_by='red_team',
                requested_at=datetime.now()
            )
            
            # Track in history
            if plan_id not in self.refinement_history:
                self.refinement_history[plan_id] = []
            self.refinement_history[plan_id].append(request)
            
            return request
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while processing Red Team feedback for plan {plan_id}: {e}", exc_info=True)
            # Fallback: return a RefinementRequest with an error message
            return RefinementRequest(
                plan_id=plan_id,
                feedback=[],
                priority=0,
                requested_by="system_error",
                requested_at=datetime.now(),
                error_message=f"Failed to process feedback: {e}"
            )
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda request: TeamAssignment(
        id=generate_id("assignment"), task_id=request.plan_id, team='blue', status="error", metadata={"error": "Failed to coordinate refinement"}
    ))
    def coordinate_refinement(
        self, 
        request: RefinementRequest
    ) -> TeamAssignment:
        """
        Blue Team refines decomposition based on feedback.
        
        Args:
            request: RefinementRequest with feedback
            
        Returns:
            TeamAssignment for Blue Team
        """
        self.logger.info(f"Coordinating refinement for plan {request.plan_id}")
        
        # Assign to Blue Team
        assignment = self.assignment_manager.assign_to_team(
            task_id=request.plan_id,
            team='blue',
            priority=request.priority,
            due_hours=3.0  # Blue team refinement typically takes 3 hours
        )
        
        try:
            # Store refinement context
            assignment.metadata['refinement_request'] = {
                'feedback_count': len(request.feedback),
                'priority': request.priority,
                'requested_by': request.requested_by
            }
        except Exception as e:
            self.logger.error(f"Failed to store refinement request in assignment metadata for plan {request.plan_id}: {e}", exc_info=True)
            # Continue without refinement request in metadata or set to empty
            assignment.metadata['refinement_request'] = {'error': f'Failed to retrieve refinement request: {e}'}
        
        return assignment
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda plan: TeamAssignment(
        id=generate_id("assignment"), task_id=plan.id, team='gold', status="error", metadata={"error": "Failed to request gold evaluation"}
    ))
    def request_gold_evaluation(
        self, 
        plan: DecompositionPlan
    ) -> TeamAssignment:
        """
        Submits to Gold Team for final evaluation.
        
        Args:
            plan: The decomposition plan to evaluate
            
        Returns:
            TeamAssignment for Gold Team
        """
        self.logger.info(f"Requesting Gold Team evaluation for plan {plan.id}")
        
        # Run final gauntlet check
        gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(plan)
        all_passed = self.gauntlet_system.all_passed(gauntlet_results)
        overall_quality = self.gauntlet_system.get_overall_quality(gauntlet_results)
        
        # Assign to Gold Team
        assignment = self.assignment_manager.assign_to_team(
            task_id=plan.id,
            team='gold',
            priority=7,  # Gold evaluation is high priority
            due_hours=1.5  # Gold team evaluation typically takes 1.5 hours
        )
        
        try:
            # Store evaluation context
            assignment.metadata['gauntlet_check'] = {
                'all_passed': all_passed,
                'overall_quality': overall_quality,
                'refinement_cycles': len(self.refinement_history.get(plan.id, []))
            }
        except Exception as e:
            self.logger.error(f"Failed to store gauntlet check results in assignment metadata for plan {plan.id}: {e}", exc_info=True)
            # Continue without gauntlet check results in metadata or set to empty
            assignment.metadata['gauntlet_check'] = {'error': f'Failed to retrieve gauntlet check results: {e}'}
        
        return assignment
    
    @with_error_handling(severity=ErrorSeverity.HIGH, fallback=lambda plan_id, approved, overall_score, strengths, weaknesses, recommendations: GoldEvaluation(
        plan_id=plan_id, approved=False, overall_score=0.0, strengths=[], weaknesses=[], recommendations=[], evaluated_by="system", evaluated_at=datetime.now(), error_message="Failed to record gold evaluation"
    ))
    def record_gold_evaluation(
        self,
        plan_id: str,
        approved: bool,
        overall_score: float,
        strengths: List[str],
        weaknesses: List[str],
        recommendations: List[str]
    ) -> GoldEvaluation:
        """
        Records Gold Team evaluation.
        
        Args:
            plan_id: ID of the decomposition plan
            approved: Whether plan is approved
            overall_score: Overall quality score (0-1)
            strengths: List of strengths
            weaknesses: List of weaknesses
            recommendations: List of recommendations
            
        Returns:
            GoldEvaluation object
        """
        evaluation = GoldEvaluation(
            plan_id=plan_id,
            approved=approved,
            overall_score=overall_score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            evaluated_by='gold_team',
            evaluated_at=datetime.now()
        )
        
        try:
            # Track in history
            if plan_id not in self.evaluation_history:
                self.evaluation_history[plan_id] = []
            self.evaluation_history[plan_id].append(evaluation)
            
            self.logger.info(f"Gold evaluation recorded: {'APPROVED' if approved else 'REJECTED'} (score: {overall_score:.2f})")
            
            return evaluation
        except Exception as e:
            self.logger.error(f"Failed to record Gold Team evaluation for plan {plan.id}: {e}", exc_info=True)
            # Fallback: return the evaluation object with an error message
            evaluation.error_message = f"Failed to record evaluation: {e}"
            return evaluation
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda: {})
    def balance_workload(self) -> Dict[str, Any]:
        """
        Balances work across teams.
        
        Returns:
            Dictionary with workload balance information
        """
        self.logger.info("Balancing workload across teams")
        
        try:
            # Get workload for each team
            red_workload = self.assignment_manager.get_team_workload('red')
            blue_workload = self.assignment_manager.get_team_workload('blue')
            gold_workload = self.assignment_manager.get_team_workload('gold')
            
            # Calculate balance metrics
            utilizations = [
                red_workload.get('utilization', 0),
                blue_workload.get('utilization', 0),
                gold_workload.get('utilization', 0)
            ]
            
            avg_utilization = sum(utilizations) / len(utilizations)
            max_utilization = max(utilizations)
            min_utilization = min(utilizations)
            
            balance_score = 1.0 - (max_utilization - min_utilization)
            
            return {
                'red_team': red_workload,
                'blue_team': blue_workload,
                'gold_team': gold_workload,
                'avg_utilization': avg_utilization,
                'balance_score': balance_score,
                'needs_rebalancing': balance_score < 0.7
            }
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while balancing workload: {e}", exc_info=True)
            # Fallback: return default workload balance information
            return {
                'red_team': {},
                'blue_team': {},
                'gold_team': {},
                'avg_utilization': 0.0,
                'balance_score': 0.0,
                'needs_rebalancing': True,
                'error_message': f"Failed to balance workload: {e}"
            }
    
    @with_error_handling(severity=ErrorSeverity.MEDIUM, fallback=lambda plan_id: {})
    def get_plan_workflow_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get complete workflow status for a plan.
        
        Args:
            plan_id: ID of the decomposition plan
            
        Returns:
            Dictionary with workflow status
        """
        try:
            refinements = self.refinement_history.get(plan_id, [])
            evaluations = self.evaluation_history.get(plan_id, [])
            
            # Get all assignments for this plan
            assignments = [a for a in self.assignment_manager.assignments.values()
                          if a.task_id == plan_id]
            
            return {
                'plan_id': plan_id,
                'total_refinements': len(refinements),
                'total_evaluations': len(evaluations),
                'assignments': [
                    {
                        'team': a.team,
                        'status': a.status,
                        'assigned_at': a.assigned_at.isoformat(),
                        'due_date': a.due_date.isoformat() if a.due_date else None
                    }
                    for a in assignments
                ],
                'latest_evaluation': evaluations[-1] if evaluations else None,
                'approved': evaluations[-1].approved if evaluations else False
            }
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while getting workflow status for plan {plan_id}: {e}", exc_info=True)
            # Fallback: return partial or empty status information
            return {
                'plan_id': plan_id,
                'total_refinements': 0,
                'total_evaluations': 0,
                'assignments': [],
                'latest_evaluation': None,
                'approved': False,
                'error_message': f"Failed to retrieve workflow status: {e}"
            }

    @with_error_handling(severity=ErrorSeverity.CRITICAL, fallback=lambda plan, max_refinement_cycles: {
        'plan_id': plan.id, 'approved': False, 'error_message': 'Validation and refinement workflow failed'
    })
    def execute_validation_and_refinement_workflow(
        self,
        plan: DecompositionPlan,
        max_refinement_cycles: int = 3
    ) -> Dict[str, Any]:
        """
        Complete validation and refinement workflow using real teams.
        
        Args:
            plan: The decomposition plan to validate
            max_refinement_cycles: Maximum number of refinement cycles
            
        Returns:
            Dictionary with workflow results
        """
        self.logger.info(f"Starting validation workflow for plan {plan.id}")
        
        import json
        
        cycle = 0
        approved = False
        current_plan = plan
        
        while cycle < max_refinement_cycles and not approved:
            cycle += 1
            self.logger.info(f"Refinement cycle {cycle}/{max_refinement_cycles}")
            
            # Step 1: Red Team Review
            try:
                self.assign_decomposition_review(current_plan)
                plan_str = json.dumps(current_plan.to_dict(), indent=2)
                red_team_assessment = self.red_team.assess_content(plan_str, content_type='json')
                
                # Convert findings to feedback
                feedback = self._convert_issue_findings_to_feedback(red_team_assessment.findings)

                # Check for critical issues
                critical_issues = [f for f in feedback if f.severity == "critical"]
            except Exception as e:
                self.logger.error(f"Red Team review failed for plan {current_plan.id}: {e}", exc_info=True)
                critical_issues = [] # Treat as no critical issues to proceed to Gold Team or break
                feedback = []
            
            if critical_issues:
                self.logger.info(f"Red Team found {len(critical_issues)} critical issues. Engaging Blue Team.")
                # Step 2: Blue Team Refinement
                try:
                    refinement_request = self.process_red_team_feedback(current_plan.id, feedback)
                    self.coordinate_refinement(refinement_request)
                    
                    blue_team_assessment = self.blue_team.apply_fixes(plan_str, red_team_assessment.findings, content_type='json')
                    
                    try:
                        # Assuming the fixed content is a JSON string of the plan
                        plan_dict = json.loads(blue_team_assessment.fixed_content)
                        current_plan = DecompositionPlan.from_dict(plan_dict)
                        self.logger.info(f"Blue Team refinement cycle {cycle} completed.")
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.error(f"Failed to decode Blue Team's refined plan: {e}. Proceeding with last valid plan.", exc_info=True)
                        # If refinement fails, break the loop and proceed with the last valid plan
                        break
                except Exception as e:
                    self.logger.error(f"Blue Team refinement failed for plan {current_plan.id}: {e}. Proceeding with last valid plan.", exc_info=True)
                    break
            else:
                self.logger.info("No critical issues found by Red Team. Proceeding to Gold Team.")
                break
        
        # Step 3: Gold Team Evaluation
        try:
            self.request_gold_evaluation(current_plan)
            plan_str = json.dumps(current_plan.to_dict(), indent=2)
            
            evaluation = self.evaluator_team.evaluate_content(plan_str, content_type='json', threshold=EvaluationThreshold.STANDARD_APPROVAL)
            
            approved = evaluation.final_verdict == "APPROVED"
            
            recorded_evaluation = self.record_gold_evaluation(
                plan_id=current_plan.id,
                approved=approved,
                overall_score=evaluation.consensus_score / 100.0, # scale to 0-1
                strengths=[rec for rec in evaluation.recommendations if "strength" in rec.lower()],
                weaknesses=[rec for rec in evaluation.recommendations if "weakness" in rec.lower()],
                recommendations=evaluation.recommendations
            )
            
            return {
                'plan_id': current_plan.id,
                'approved': approved,
                'refinement_cycles': cycle,
                'final_score': evaluation.consensus_score,
                'evaluation': recorded_evaluation,
                'workflow_status': self.get_plan_workflow_status(current_plan.id)
            }
        except Exception as e:
            self.logger.error(f"Gold Team evaluation failed for plan {current_plan.id}: {e}", exc_info=True)
            return {
                'plan_id': current_plan.id,
                'approved': False,
                'refinement_cycles': cycle,
                'final_score': 0.0,
                'evaluation': GoldEvaluation(
                    plan_id=current_plan.id, approved=False, overall_score=0.0, strengths=[], weaknesses=[], recommendations=[], evaluated_by="system_error", evaluated_at=datetime.now(), error_message=f"Gold Team evaluation failed: {e}"
                ),
                'workflow_status': self.get_plan_workflow_status(current_plan.id),
                'error_message': f"Validation and refinement workflow failed during Gold Team evaluation: {e}"
            }



class DecompositionWorkflow:
    """Complete workflow for decomposition validation and refinement."""
    
    def __init__(self, openevolve_client=None):
        """
        Initialize decomposition workflow.
        
        Args:
            openevolve_client: Optional OpenEvolve client for LLM interactions.
        """
        self.coordinator = TeamCoordinator(openevolve_client=openevolve_client)
        self.logger = logging.getLogger(__name__)
    
    @with_error_handling(severity=ErrorSeverity.CRITICAL, fallback=lambda plan, max_refinement_cycles: {
        'plan_id': plan.id, 'approved': False, 'error_message': 'Validation and refinement workflow failed'
    })
    def validate_and_refine(
        self,
        plan: DecompositionPlan,
        max_refinement_cycles: int = 3
    ) -> Dict[str, Any]:
        """
        Complete validation and refinement workflow using the TeamCoordinator.
        
        Args:
            plan: The decomposition plan to validate
            max_refinement_cycles: Maximum number of refinement cycles
            
        Returns:
            Dictionary with workflow results
        """
        return self.coordinator.execute_validation_and_refinement_workflow(
            plan, 
            max_refinement_cycles
        )
