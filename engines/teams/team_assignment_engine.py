"""
Intelligent Team Assignment Engine for Decomposition Workflow

This module implements intelligent team assignment to sub-problems based on:
- Team capabilities and expertise
- Historical performance
- Current workload
- Specialization matching
- Conflict avoidance
"""
from __future__ import annotations


import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict

from sovereign_data_models import (
    SubProblem, DecompositionPlan
)

try:
    # Preferred: shared model, if the schema facade ever exposes it.
    from sovereign_data_models import SubProblemTeamAssignment
except ImportError:
    # sovereign_data_models (openevolve.kernel.schema) only ships the generic
    # ``TeamAssignment`` (id/team_id/sub_problem_id), which cannot represent the
    # per-role assignment used here, so define the minimal record needed.
    @dataclass
    class SubProblemTeamAssignment:
        """Per-role team assignment for a single sub-problem."""

        solver: Optional[str] = None
        patcher: Optional[str] = None
        red_team: Optional[str] = None
        gold_team: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)
from openevolve_structures import Team
from team_manager import TeamManager

# **ACTUAL INTEGRATION**: Alerting and knowledge for team assignment
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TeamCapability:
    """
    Represents a team's capability in a specific domain.

    Attributes:
        team_id: Unique identifier for the team
        team_name: Name of the team
        domain: Domain of expertise
        expertise_areas: List of specific expertise areas
        capability_score: Overall capability score (0.0-1.0)
        success_rate: Historical success rate (0.0-1.0)
        total_assignments: Total number of assignments
        recent_performance: Last 10 assignment results (0.0-1.0 each)
        workload_score: Current workload (0.0-1.0, lower is better)
        specialization_fit: How well team specialization matches (0.0-1.0)
        confidence_score: Overall confidence in this assessment (0.0-1.0)
        metadata: Additional metadata
    """
    team_id: str
    team_name: str
    domain: str
    expertise_areas: List[str] = field(default_factory=list)
    capability_score: float = 0.5
    success_rate: float = 0.5
    total_assignments: int = 0
    recent_performance: List[float] = field(default_factory=list)
    workload_score: float = 0.0
    specialization_fit: float = 0.5
    confidence_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calculate_overall_capability(self) -> float:
        """
        Calculate overall capability score combining multiple factors.

        Returns:
            Overall capability score (0.0-1.0)
        """
        weights = {
            'capability': 0.35,
            'success_rate': 0.30,
            'workload': 0.20,
            'specialization': 0.15
        }

        # Invert workload so lower workload = higher score
        workload_factor = 1.0 - self.workload_score

        overall = (
            weights['capability'] * self.capability_score +
            weights['success_rate'] * self.success_rate +
            weights['workload'] * workload_factor +
            weights['specialization'] * self.specialization_fit
        )

        return min(1.0, max(0.0, overall))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TeamCapability':
        """Create TeamCapability from dictionary."""
        return cls(**data)


class TeamCapabilityAssessor:
    """
    Assesses team capabilities for different types of sub-problems.

    Considers:
    - Domain expertise matching
    - Required skills availability
    - Historical performance
    - Current workload
    - Specialization fit
    """

    def __init__(self, team_manager: TeamManager):
        """
        Initialize the capability assessor.

        Args:
            team_manager: TeamManager instance for accessing team data
        """
        self.team_manager = team_manager
        self.logger = logging.getLogger(f"{__name__}.TeamCapabilityAssessor")

    def assess_team_capability(
        self,
        team: Team,
        sub_problem: SubProblem
    ) -> TeamCapability:
        """
        Assess how capable a team is for a specific sub-problem.

        Args:
            team: Team to assess
            sub_problem: Sub-problem to assess against

        Returns:
            TeamCapability object with assessment results
        """
        try:
            # Extract problem domain
            domain = getattr(sub_problem, 'domain', 'general')

            # Assess expertise matching
            expertise_areas, expertise_match = self._assess_expertise_match(
                team, sub_problem
            )

            # Calculate capability score
            capability_score = self._calculate_capability_score(
                team, sub_problem, expertise_match
            )

            # Get historical performance
            success_rate = self._get_historical_success_rate(team.name, domain)

            # Assess current workload
            workload_score = self._assess_workload(team)

            # Assess specialization fit
            specialization_fit = self._assess_specialization_fit(
                team, sub_problem
            )

            # Get recent performance
            recent_performance = self._get_recent_performance(team.name, domain)

            # Calculate confidence
            confidence_score = self._calculate_confidence(
                capability_score, success_rate, specialization_fit
            )

            capability = TeamCapability(
                team_id=team.name,
                team_name=team.name,
                domain=domain,
                expertise_areas=expertise_areas,
                capability_score=capability_score,
                success_rate=success_rate,
                total_assignments=self._get_total_assignments(team.name),
                recent_performance=recent_performance,
                workload_score=workload_score,
                specialization_fit=specialization_fit,
                confidence_score=confidence_score,
                metadata={
                    'assessed_at': datetime.now().isoformat(),
                    'sub_problem_id': sub_problem.id,
                    'team_role': team.role
                }
            )

            self.logger.debug(
                f"Assessed capability for team {team.name}: "
                f"overall={capability.calculate_overall_capability():.2f}"
            )

            return capability

        except (AttributeError, TypeError, ValueError) as e:
            self.logger.error(f"Error assessing team capability: {e}", exc_info=True)
            # Return low-capability default
            return TeamCapability(
                team_id=team.name,
                team_name=team.name,
                domain='unknown',
                capability_score=0.1,
                success_rate=0.1,
                confidence_score=0.1
            )

    def assess_all_teams(
        self,
        sub_problem: SubProblem,
        available_teams: List[Team]
    ) -> Dict[str, TeamCapability]:
        """
        Assess all available teams for a sub-problem.

        Args:
            sub_problem: Sub-problem to assess against
            available_teams: List of available teams

        Returns:
            Dictionary mapping team names to TeamCapability objects
        """
        capabilities = {}

        for team in available_teams:
            try:
                capability = self.assess_team_capability(team, sub_problem)
                capabilities[team.name] = capability
            except (AttributeError, TypeError, ValueError) as e:
                self.logger.error(
                    f"Error assessing team {team.name}: {e}",
                    exc_info=True
                )
                continue

        self.logger.info(
            f"Assessed {len(capabilities)} teams for sub-problem {sub_problem.id}"
        )

        return capabilities

    def _assess_expertise_match(
        self,
        team: Team,
        sub_problem: SubProblem
    ) -> Tuple[List[str], float]:
        """
        Assess how well team expertise matches sub-problem requirements.

        Returns:
            Tuple of (matched_expertise_areas, match_score)
        """
        # Get required expertise from sub-problem
        required_expertise = getattr(sub_problem, 'required_expertise', [])

        # Get team domain specialization
        team_domains = getattr(team, 'domain_specialization', [])
        team_problem_types = getattr(team, 'problem_type_specialization', [])

        # Match expertise areas
        matched_areas = []
        for req in required_expertise:
            req_lower = req.lower()
            if any(req_lower in td.lower() for td in team_domains):
                matched_areas.append(req)
            elif any(req_lower in pt.lower() for pt in team_problem_types):
                matched_areas.append(req)

        # Calculate match score
        if not required_expertise:
            # No specific requirements = neutral score
            match_score = 0.5
        else:
            match_score = len(matched_areas) / len(required_expertise) if required_expertise else 0.5

        return matched_areas, min(1.0, match_score)

    def _calculate_capability_score(
        self,
        team: Team,
        sub_problem: SubProblem,
        expertise_match: float
    ) -> float:
        """Calculate base capability score."""
        base_score = 0.5  # Neutral baseline

        # Expertise matching (40%)
        expertise_factor = expertise_match * 0.4

        # Team role appropriateness (20%)
        role_factor = 0.2 if team.role == "Blue" else 0.1

        # Performance metrics (30%)
        perf_metrics = getattr(team, 'performance_metrics', {})
        performance_factor = 0.0
        if perf_metrics:
            avg_perf = sum(perf_metrics.values()) / len(perf_metrics)
            performance_factor = min(0.3, avg_perf * 0.3)

        # Team configuration (10%)
        team_config = getattr(team, 'team_config', {})
        config_factor = 0.1 if team_config else 0.05

        capability = base_score + expertise_factor + role_factor + performance_factor + config_factor
        return min(1.0, max(0.0, capability))

    def _get_historical_success_rate(self, team_name: str, domain: str) -> float:
        """
        Get historical success rate for a team in a domain.

        Args:
            team_name: Name of the team
            domain: Domain to check

        Returns:
            Success rate (0.0-1.0)
        """
        try:
            metrics = self.team_manager.aggregate_team_metrics(team_name)
            if metrics and 'avg_fitness' in metrics:
                # Normalize fitness to 0-1 range
                return min(1.0, max(0.0, metrics['avg_fitness']))
        except (KeyError, AttributeError, TypeError) as e:
            self.logger.debug(f"Could not get historical success rate: {e}")

        # Default to neutral if no data
        return 0.5

    def _assess_workload(self, team: Team) -> float:
        """
        Assess current workload of a team.

        Returns:
            Workload score (0.0-1.0, higher = more busy)
        """
        try:
            metrics = self.team_manager.aggregate_team_metrics(team.name)
            if metrics:
                total_ops = metrics.get('total_operations', 0)
                # Simple heuristic: more operations = higher workload
                # Could be enhanced with actual active task count
                workload = min(1.0, total_ops / 100.0)
                return workload
        except (KeyError, AttributeError, TypeError) as e:
            self.logger.debug(f"Could not assess workload: {e}")

        return 0.0  # No data = assume available

    def _assess_specialization_fit(
        self,
        team: Team,
        sub_problem: SubProblem
    ) -> float:
        """
        Assess how well team specialization matches sub-problem.

        Returns:
            Specialization fit score (0.0-1.0)
        """
        # Get team specializations
        domain_spec = getattr(team, 'domain_specialization', [])
        problem_type_spec = getattr(team, 'problem_type_specialization', [])

        if not domain_spec and not problem_type_spec:
            return 0.5  # No specialization = neutral

        # Check description for domain clues
        description_lower = sub_problem.description.lower()
        title_lower = sub_problem.title.lower() if hasattr(sub_problem, 'title') else ""

        matches = 0
        total_specs = len(domain_spec) + len(problem_type_spec)

        for spec in domain_spec + problem_type_spec:
            if spec.lower() in description_lower or spec.lower() in title_lower:
                matches += 1

        return matches / total_specs if total_specs > 0 else 0.5

    def _get_recent_performance(self, team_name: str, domain: str) -> List[float]:
        """
        Get recent performance scores for a team.

        Returns:
            List of recent performance scores (0.0-1.0 each)
        """
        try:
            metrics_list = self.team_manager.get_openevolve_metrics(team_name)

            # Extract fitness scores from recent metrics
            performances = []
            for entry in metrics_list[-10:]:  # Last 10
                metrics = entry.get('metrics', {})
                fitness = metrics.get('best_fitness', 0.5)
                # Normalize to 0-1
                performances.append(min(1.0, max(0.0, fitness)))

            return performances
        except (KeyError, AttributeError, TypeError) as e:
            self.logger.debug(f"Could not get recent performance: {e}")

        return []

    def _get_total_assignments(self, team_name: str) -> int:
        """Get total number of assignments for a team."""
        try:
            metrics = self.team_manager.aggregate_team_metrics(team_name)
            return metrics.get('total_operations', 0)
        except (KeyError, AttributeError, TypeError) as e:
            self.logger.debug(f"Could not get total assignments: {e}")
            return 0

    def _calculate_confidence(
        self,
        capability_score: float,
        success_rate: float,
        specialization_fit: float
    ) -> float:
        """
        Calculate confidence in the capability assessment.

        Higher confidence when:
        - High capability score
        - Strong historical performance
        - Good specialization fit
        """
        # Confidence based on agreement between metrics
        agreement = 1.0 - abs(capability_score - success_rate)
        specialization_boost = specialization_fit * 0.3

        confidence = (agreement * 0.7) + specialization_boost
        return min(1.0, max(0.0, confidence))


class TeamAssignmentEngine:
    """
    Intelligently assigns teams to sub-problems based on capabilities.

    Features:
    - Multi-factor capability assessment
    - Workload balancing
    - Conflict avoidance
    - Specialization utilization
    - Performance tracking integration
    """

    def __init__(
        self,
        team_manager: TeamManager,
        capability_assessor: Optional[TeamCapabilityAssessor] = None,
        performance_tracker: Optional['TeamPerformanceTracker'] = None
    ):
        """
        Initialize the team assignment engine.

        Args:
            team_manager: TeamManager instance
            capability_assessor: Optional custom capability assessor
            performance_tracker: Optional performance tracker
        """
        self.team_manager = team_manager
        self.capability_assessor = capability_assessor or TeamCapabilityAssessor(team_manager)
        self.performance_tracker = performance_tracker
        self.assignment_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.TeamAssignmentEngine")

    def assign_teams_to_subproblem(
        self,
        sub_problem: SubProblem,
        available_teams: List[Team]
    ) -> SubProblemTeamAssignment:
        """
        Assign teams to a sub-problem.

        Args:
            sub_problem: Sub-problem to assign teams to
            available_teams: List of available teams

        Returns:
            SubProblemTeamAssignment with recommended teams
        """
        try:
            # Assess all teams
            capabilities = self.capability_assessor.assess_all_teams(
                sub_problem, available_teams
            )

            if not capabilities:
                self.logger.warning(f"No teams available for sub-problem {sub_problem.id}")
                return SubProblemTeamAssignment()

            # Assign solver (best overall Blue team)
            solver = self._assign_solver(capabilities, sub_problem)

            # Assign patcher (may be same as solver or different)
            patcher = self._assign_patcher(capabilities, sub_problem, solver)

            # Assign red team (best critique team)
            red_team = self._assign_red_team(capabilities, sub_problem, solver)

            # Assign gold team (best verification team)
            gold_team = self._assign_gold_team(capabilities, sub_problem)

            assignment = SubProblemTeamAssignment(
                solver=solver,
                patcher=patcher,
                red_team=red_team,
                gold_team=gold_team,
                metadata={
                    'assigned_at': datetime.now().isoformat(),
                    'num_candidates': len(capabilities),
                    'solver_confidence': capabilities.get(solver, TeamCapability(
                        team_id=solver, team_name=solver, domain=''
                    )).calculate_overall_capability() if solver else 0.0
                }
            )

            # Record assignment
            self._record_assignment(sub_problem, assignment, capabilities)

            self.logger.info(
                f"Assigned teams to sub-problem {sub_problem.id}: "
                f"solver={solver}, red_team={red_team}, gold_team={gold_team}"
            )

            return assignment

        except (AttributeError, TypeError, ValueError) as e:
            self.logger.error(
                f"Error assigning teams to sub-problem {sub_problem.id}: {e}",
                exc_info=True
            )
            return SubProblemTeamAssignment()

    def assign_teams_to_plan(
        self,
        decomposition_plan: DecompositionPlan,
        available_teams: List[Team]
    ) -> DecompositionPlan:
        """
        Assign teams to all sub-problems in a decomposition plan.

        Optimizes for:
        - Best overall team assignments
        - Balanced workload across teams
        - Specialization utilization

        Args:
            decomposition_plan: Plan with sub-problems to assign
            available_teams: List of available teams

        Returns:
            Updated DecompositionPlan with team assignments
        """
        try:
            self.logger.info(
                f"Assigning teams to {len(decomposition_plan.sub_problems)} sub-problems"
            )

            # Track team usage for workload balancing
            team_usage = defaultdict(int)

            for sub_problem in decomposition_plan.sub_problems:
                # Assign teams to this sub-problem
                assignment = self.assign_teams_to_subproblem(
                    sub_problem, available_teams
                )

                # Update sub-problem with assignment
                sub_problem.ai_suggested_team_assignment = assignment

                # Track team usage
                if assignment.solver:
                    team_usage[assignment.solver] += 1
                if assignment.patcher and assignment.patcher != assignment.solver:
                    team_usage[assignment.patcher] += 1
                if assignment.red_team:
                    team_usage[assignment.red_team] += 1
                if assignment.gold_team:
                    team_usage[assignment.gold_team] += 1

            self.logger.info(
                f"Team assignment complete. Usage: {dict(team_usage)}"
            )

            # **ACTUAL INTEGRATION**: Extract assignment knowledge
            assignments = [
                sp.ai_suggested_team_assignment
                for sp in decomposition_plan.sub_problems
                if hasattr(sp, 'ai_suggested_team_assignment') and sp.ai_suggested_team_assignment
            ]
            self._extract_assignment_knowledge(decomposition_plan.id, assignments)

            # **ACTUAL INTEGRATION**: Track performance
            self._track_assignment_performance(decomposition_plan.id, len(assignments), True)

            return decomposition_plan

        except (AttributeError, TypeError, ValueError) as e:
            self.logger.error(
                f"Error assigning teams to plan: {e}",
                exc_info=True
            )

            # **ACTUAL INTEGRATION**: Trigger alert on failure
            self._trigger_assignment_alerts(
                getattr(decomposition_plan, 'id', 'unknown'),
                False,
                str(e)
            )

            return decomposition_plan

    def calculate_assignment_confidence(
        self,
        sub_problem: SubProblem,
        team: Team
    ) -> float:
        """
        Calculate confidence score for a team assignment.

        Returns 0.0-1.0 based on:
        - Capability match (40%)
        - Historical performance (30%)
        - Workload availability (20%)
        - Specialization fit (10%)

        Args:
            sub_problem: Sub-problem to assign
            team: Team being considered

        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            capability = self.capability_assessor.assess_team_capability(
                team, sub_problem
            )

            return capability.calculate_overall_capability()

        except (AttributeError, TypeError, ValueError) as e:
            self.logger.error(f"Error calculating assignment confidence: {e}")
            return 0.0

    def _assign_solver(
        self,
        capabilities: Dict[str, TeamCapability],
        sub_problem: SubProblem
    ) -> str:
        """
        Assign best solver team.

        Solver should be:
        - Blue team (creation/solving role)
        - High capability score
        - Good specialization match
        - Available workload
        """
        # Filter for Blue teams only
        blue_teams = {
            name: cap for name, cap in capabilities.items()
            if cap.metadata.get('team_role') == 'Blue'
        }

        if not blue_teams:
            # Fallback to any team
            blue_teams = capabilities

        # Sort by overall capability
        sorted_teams = sorted(
            blue_teams.items(),
            key=lambda x: x[1].calculate_overall_capability(),
            reverse=True
        )

        if sorted_teams:
            return sorted_teams[0][0]

        return ""

    def _assign_patcher(
        self,
        capabilities: Dict[str, TeamCapability],
        sub_problem: SubProblem,
        solver: str
    ) -> str:
        """
        Assign patcher team.

        Patcher may be:
        - Same as solver (default)
        - Different if specialized patching needed
        """
        # Default to solver
        return solver

    def _assign_red_team(
        self,
        capabilities: Dict[str, TeamCapability],
        sub_problem: SubProblem,
        solver: str
    ) -> str:
        """
        Assign red team for critique.

        Red team should be:
        - Red team role (critique)
        - Different from solver (conflict avoidance)
        - High capability for critique
        """
        # Filter for Red teams only
        red_teams = {
            name: cap for name, cap in capabilities.items()
            if cap.metadata.get('team_role') == 'Red'
        }

        # Remove solver from consideration (conflict avoidance)
        if solver in red_teams:
            # Check if there are other red teams
            if len(red_teams) > 1:
                red_teams.pop(solver)
            else:
                # Only one red team and it's the solver
                # Use it anyway (better than nothing)
                pass

        if not red_teams:
            # Fallback to any team except solver
            red_teams = {
                name: cap for name, cap in capabilities.items()
                if name != solver
            }

        if red_teams:
            # Sort by overall capability
            sorted_teams = sorted(
                red_teams.items(),
                key=lambda x: x[1].calculate_overall_capability(),
                reverse=True
            )
            return sorted_teams[0][0]

        return ""

    def _assign_gold_team(
        self,
        capabilities: Dict[str, TeamCapability],
        sub_problem: SubProblem
    ) -> str:
        """
        Assign gold team for verification.

        Gold team should be:
        - Gold team role (verification)
        - High capability for verification
        """
        # Filter for Gold teams only
        gold_teams = {
            name: cap for name, cap in capabilities.items()
            if cap.metadata.get('team_role') == 'Gold'
        }

        if not gold_teams:
            # Fallback to any team
            gold_teams = capabilities

        # Sort by overall capability
        sorted_teams = sorted(
            gold_teams.items(),
            key=lambda x: x[1].calculate_overall_capability(),
            reverse=True
        )

        if sorted_teams:
            return sorted_teams[0][0]

        return ""

    def _record_assignment(
        self,
        sub_problem: SubProblem,
        assignment: SubProblemTeamAssignment,
        capabilities: Dict[str, TeamCapability]
    ):
        """Record assignment for learning and tracking."""
        record = {
            'sub_problem_id': sub_problem.id,
            'assignment': assignment.to_dict(),
            'capabilities': {
                name: cap.to_dict() for name, cap in capabilities.items()
            },
            'timestamp': datetime.now().isoformat()
        }

        # Store in history
        self.assignment_history[sub_problem.id].append(record)

        # Also record in performance tracker if available
        if self.performance_tracker:
            for role, team_name in [
                ('solver', assignment.solver),
                ('patcher', assignment.patcher),
                ('red_team', assignment.red_team),
                ('gold_team', assignment.gold_team)
            ]:
                if team_name:
                    self.performance_tracker.record_assignment(
                        team_name, sub_problem.id, role, assignment
                    )

    # **ADAPTIVE MDAP INTEGRATION**: Complexity-based team sizing
    def compute_subproblem_complexity(
        self,
        sub_problem: SubProblem
    ) -> Optional[float]:
        """
        Compute complexity score for a sub-problem using Adaptive MDAP.
        
        Args:
            sub_problem: Sub-problem to analyze
            
        Returns:
            Complexity score (0.0-1.0) or None if Adaptive MDAP unavailable
        """
        if not ADAPTIVE_MDAP_AVAILABLE:
            return None
        
        try:
            # Convert to Adaptive MDAP SubProblem type
            adaptive_sp = SubProblem(
                id=sub_problem.id,
                description=sub_problem.description,
                domain=getattr(sub_problem, 'domain', 'general'),
                depth=getattr(sub_problem, 'depth', 1),
                dependencies=getattr(sub_problem, 'dependencies', []),
                metadata=getattr(sub_problem, 'metadata', {})
            )
            
            classifier = TaskComplexityClassifier()
            score = classifier.compute_complexity(adaptive_sp)
            
            self.logger.debug(
                f"Computed complexity for {sub_problem.id}: {score.overall_score:.3f}"
            )
            
            return score.overall_score
            
        except Exception as e:
            self.logger.warning(f"Failed to compute complexity: {e}")
            return None
    
    def get_optimal_team_size(
        self,
        sub_problem: SubProblem,
        base_size: int = 3
    ) -> int:
        """
        Get optimal team size based on sub-problem complexity.
        
        Uses Adaptive MDAP to determine team size:
        - Simple problems (≤0.2): Smaller teams (1-2 members)
        - Medium problems (0.2-0.6): Standard teams (3-5 members)
        - Complex problems (>0.6): Larger teams (5-7 members)
        
        Args:
            sub_problem: Sub-problem to analyze
            base_size: Base team size to adjust from
            
        Returns:
            Optimal team size
        """
        complexity = self.compute_subproblem_complexity(sub_problem)
        
        if complexity is None:
            return base_size
        
        # Adjust team size based on complexity
        if complexity <= 0.2:
            # Simple problem - minimal team
            return max(1, base_size - 2)
        elif complexity <= 0.4:
            # Light complexity - small team
            return max(2, base_size - 1)
        elif complexity <= 0.6:
            # Medium complexity - standard team
            return base_size
        elif complexity <= 0.8:
            # High complexity - larger team
            return base_size + 1
        else:
            # Very high complexity - full team
            return base_size + 2
    
    def assign_teams_with_complexity(
        self,
        sub_problem: SubProblem,
        available_teams: List[Team]
    ) -> SubProblemTeamAssignment:
        """
        Assign teams with complexity-based optimization.
        
        This method extends assign_teams_to_subproblem by:
        1. Computing sub-problem complexity
        2. Adjusting team size recommendations
        3. Logging complexity-based decisions
        
        Args:
            sub_problem: Sub-problem to assign teams to
            available_teams: List of available teams
            
        Returns:
            SubProblemTeamAssignment with complexity metadata
        """
        # Get complexity
        complexity = self.compute_subproblem_complexity(sub_problem)
        
        # Get base assignment
        assignment = self.assign_teams_to_subproblem(sub_problem, available_teams)
        
        # Add complexity metadata
        if complexity is not None:
            if assignment.metadata is None:
                assignment.metadata = {}
            
            assignment.metadata['complexity_score'] = complexity
            assignment.metadata['recommended_team_size'] = self.get_optimal_team_size(
                sub_problem
            )
            
            # Log complexity-based assignment
            self.logger.info(
                f"Complexity-based assignment for {sub_problem.id}: "
                f"complexity={complexity:.3f}, "
                f"recommended_size={assignment.metadata['recommended_team_size']}"
            )
        
        return assignment


class TeamPerformanceTracker:
    """
    Tracks team performance over time for better assignment decisions.

    Features:
    - Assignment tracking
    - Outcome recording
    - Performance statistics
    - Team ranking
    - Persistent storage
    """

    def __init__(self, storage_path: str = "team_performance.json"):
        """
        Initialize the performance tracker.

        Args:
            storage_path: Path to JSON file for persistent storage
        """
        self.storage_path = storage_path
        self.performance_data: Dict[str, Any] = self._load_performance_data()
        self.logger = logging.getLogger(f"{__name__}.TeamPerformanceTracker")

    def _load_performance_data(self) -> Dict[str, Any]:
        """Load performance data from persistent storage."""
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.logger.info(f"Loaded performance data from {self.storage_path}")
                    return data
        except (OSError, IOError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading performance data: {e}", exc_info=True)

        return {
            'teams': {},
            'assignments': [],
            'outcomes': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            }
        }

    def _save_performance_data(self):
        """Save performance data to persistent storage."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.performance_data, f, indent=2)
                self.logger.debug(f"Saved performance data to {self.storage_path}")
        except (OSError, IOError, TypeError) as e:
            self.logger.error(f"Error saving performance data: {e}", exc_info=True)

    def record_assignment(
        self,
        team_id: str,
        sub_problem_id: str,
        role: str,
        assignment: SubProblemTeamAssignment
    ):
        """
        Record a team assignment for tracking.

        Args:
            team_id: ID of the team being assigned
            sub_problem_id: ID of the sub-problem
            role: Role being assigned (solver, patcher, red_team, gold_team)
            assignment: Full assignment object
        """
        try:
            # Initialize team data if not exists
            if team_id not in self.performance_data['teams']:
                self.performance_data['teams'][team_id] = {
                    'total_assignments': 0,
                    'assignments_by_role': defaultdict(int),
                    'domains': defaultdict(int),
                    'first_assigned': datetime.now().isoformat()
                }

            # Update team stats
            team_data = self.performance_data['teams'][team_id]
            team_data['total_assignments'] += 1
            team_data['assignments_by_role'][role] += 1
            team_data['last_assigned'] = datetime.now().isoformat()

            # Record assignment
            assignment_record = {
                'team_id': team_id,
                'sub_problem_id': sub_problem_id,
                'role': role,
                'timestamp': datetime.now().isoformat(),
                'assignment_metadata': assignment.metadata
            }

            self.performance_data['assignments'].append(assignment_record)

            # Save to disk
            self._save_performance_data()

            self.logger.debug(
                f"Recorded assignment: team={team_id}, role={role}, "
                f"sub_problem={sub_problem_id}"
            )

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error recording assignment: {e}", exc_info=True)

    def record_outcome(
        self,
        team_id: str,
        sub_problem_id: str,
        success: bool,
        quality_score: float,
        time_taken: float
    ):
        """
        Record the outcome of a team's work.

        Args:
            team_id: ID of the team
            sub_problem_id: ID of the sub-problem
            success: Whether the team succeeded
            quality_score: Quality of work (0.0-1.0)
            time_taken: Time taken in seconds
        """
        try:
            # Update team stats
            if team_id in self.performance_data['teams']:
                team_data = self.performance_data['teams'][team_id]

                # Update success rate
                if 'successful_assignments' not in team_data:
                    team_data['successful_assignments'] = 0
                if success:
                    team_data['successful_assignments'] += 1

                # Update quality metrics
                if 'quality_scores' not in team_data:
                    team_data['quality_scores'] = []
                team_data['quality_scores'].append(quality_score)

                # Update time metrics
                if 'time_taken' not in team_data:
                    team_data['time_taken'] = []
                team_data['time_taken'].append(time_taken)

            # Record outcome
            outcome_record = {
                'team_id': team_id,
                'sub_problem_id': sub_problem_id,
                'success': success,
                'quality_score': quality_score,
                'time_taken': time_taken,
                'timestamp': datetime.now().isoformat()
            }

            self.performance_data['outcomes'].append(outcome_record)

            # Save to disk
            self._save_performance_data()

            self.logger.info(
                f"Recorded outcome: team={team_id}, success={success}, "
                f"quality={quality_score:.2f}"
            )

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error recording outcome: {e}", exc_info=True)

    def get_team_performance_stats(self, team_id: str) -> Dict[str, Any]:
        """
        Get performance statistics for a team.

        Args:
            team_id: ID of the team

        Returns:
            Dictionary with performance statistics
        """
        try:
            if team_id not in self.performance_data['teams']:
                return {
                    'error': f'Team {team_id} not found in performance data'
                }

            team_data = self.performance_data['teams'][team_id]

            # Calculate success rate
            total_assignments = team_data.get('total_assignments', 0)
            successful = team_data.get('successful_assignments', 0)
            success_rate = successful / total_assignments if total_assignments > 0 else 0.0

            # Calculate average quality
            quality_scores = team_data.get('quality_scores', [])
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

            # Calculate average time
            time_taken = team_data.get('time_taken', [])
            avg_time = sum(time_taken) / len(time_taken) if time_taken else 0.0

            # Get best domains
            domains = team_data.get('domains', {})
            best_domains = sorted(
                domains.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            # Get recent performance trend
            team_outcomes = [
                o for o in self.performance_data['outcomes']
                if o['team_id'] == team_id
            ]
            recent_outcomes = team_outcomes[-10:]  # Last 10
            recent_success_rate = (
                sum(1 for o in recent_outcomes if o['success']) / len(recent_outcomes)
                if recent_outcomes else 0.0
            )

            return {
                'team_id': team_id,
                'total_assignments': total_assignments,
                'success_rate': success_rate,
                'average_quality_score': avg_quality,
                'average_time_taken': avg_time,
                'best_domains': [d[0] for d in best_domains],
                'recent_performance_trend': recent_success_rate,
                'assignments_by_role': dict(team_data.get('assignments_by_role', {})),
                'first_assigned': team_data.get('first_assigned'),
                'last_assigned': team_data.get('last_assigned')
            }

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error getting team performance stats: {e}", exc_info=True)
            return {'error': str(e)}

    def get_team_ranking(self, domain: Optional[str] = None) -> List[Tuple[str, float]]:
        """
        Get teams ranked by performance.

        Args:
            domain: Optional domain filter

        Returns:
            List of (team_id, score) tuples, sorted by score descending
        """
        try:
            rankings = []

            for team_id in self.performance_data['teams']:
                stats = self.get_team_performance_stats(team_id)

                if 'error' in stats:
                    continue

                # Calculate ranking score
                # Weight: success rate (50%), quality (30%), recent performance (20%)
                score = (
                    stats['success_rate'] * 0.5 +
                    stats['average_quality_score'] * 0.3 +
                    stats['recent_performance_trend'] * 0.2
                )

                # Filter by domain if specified
                if domain:
                    if domain in stats['best_domains']:
                        # Boost score if team is good in this domain
                        domain_rank = stats['best_domains'].index(domain)
                        domain_boost = 1.0 - (domain_rank * 0.1)
                        score *= domain_boost
                    else:
                        # Penalize if domain not in best domains
                        score *= 0.5

                rankings.append((team_id, score))

            # Sort by score descending
            rankings.sort(key=lambda x: x[1], reverse=True)

            return rankings

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error getting team ranking: {e}", exc_info=True)
            return []

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for team assignment
    # =========================================================================

    def _trigger_assignment_alerts(
        self,
        plan_id: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for assignment failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.MEDIUM

                alert_manager.create_alert(
                    title=f"Team Assignment Failed: {plan_id}",
                    description=f"Team assignment failed for plan '{plan_id}'. " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="team_assignment_engine",
                    component="assignment",
                    metadata=metadata or {}
                )

        except Exception as e:
            self.logger.error(f"Failed to trigger assignment alert: {e}")

    def _extract_assignment_knowledge(
        self,
        plan_id: str,
        assignments: List['SubProblemTeamAssignment']
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract assignment knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            # Create knowledge artifact from assignments
            artifact = KnowledgeArtifact(
                artifact_id=f"assignment_{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="team_assignment",
                source_component="team_assignment_engine",
                title=f"Team Assignment: {plan_id}",
                content={
                    "plan_id": plan_id,
                    "num_assignments": len(assignments),
                    "assignments": [
                        {
                            "sub_problem_id": a.sub_problem_id,
                            "team_id": a.team_id,
                            "confidence": a.confidence_score
                        }
                        for a in assignments[:10]  # Limit to first 10
                    ],
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "total_assignments": len(assignments)
                },
                tags=["assignment", "team", "coordination"]
            )

            knowledge_engine.store_artifact(artifact)
            self.logger.debug(f"Extracted assignment knowledge for {plan_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to extract assignment knowledge: {e}")
            return False

    def _track_assignment_performance(
        self,
        plan_id: str,
        num_assignments: int,
        success: bool
    ):
        """**ACTUAL INTEGRATION**: Track assignment performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"team_assignment",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"plan_id": plan_id, "num_assignments": num_assignments}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                self.logger.debug(f"Tracked assignment performance: {plan_id}")

        except Exception as e:
            self.logger.error(f"Failed to track assignment performance: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get overall performance summary.

        Returns:
            Dictionary with summary statistics
        """
        try:
            total_teams = len(self.performance_data['teams'])
            total_assignments = len(self.performance_data['assignments'])
            total_outcomes = len(self.performance_data['outcomes'])

            # Overall success rate
            if total_outcomes > 0:
                overall_success = sum(
                    1 for o in self.performance_data['outcomes'] if o['success']
                ) / total_outcomes
            else:
                overall_success = 0.0

            # Top performers
            top_teams = self.get_team_ranking()[:5]

            return {
                'total_teams': total_teams,
                'total_assignments': total_assignments,
                'total_outcomes': total_outcomes,
                'overall_success_rate': overall_success,
                'top_performing_teams': [
                    {'team_id': team, 'score': score} for team, score in top_teams
                ],
                'metadata': self.performance_data.get('metadata', {})
            }

        except (KeyError, TypeError, AttributeError) as e:
            self.logger.error(f"Error getting performance summary: {e}", exc_info=True)
            return {}
