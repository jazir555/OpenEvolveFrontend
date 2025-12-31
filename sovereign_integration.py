"""
Sovereign-Grade Problem Decomposition System - Integration Orchestrator
Task 15: Complete system integration and end-to-end workflow orchestration.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from problem_analyzer import ProblemAnalyzer
from decomposition_engine import DecompositionEngine
from sovereign_gauntlets import GauntletSystem
from sovereign_team_coordination import TeamCoordinator
from sovereign_quality_assessment import QualityAssessor
from sovereign_solution_orchestration import SolutionOrchestrator
from sovereign_refinement import RefinementCoordinator
from sovereign_knowledge_manager import KnowledgeManager
from sovereign_data_models import (
    ProblemDefinition, DecompositionPlan, SubProblem, 
    SolutionAttempt, generate_id, SubProblemStatus
)
from sub_problem_solver import SubProblemSolver
from sovereign_persistence import SovereignDatabase # Import SovereignDatabase
from sovereign_reliability import ErrorHandler, ErrorSeverity, get_health_monitor # Import ErrorHandler and ErrorSeverity
from configuration_manager import config_manager # Import config_manager

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Result of complete integration workflow."""
    problem_id: str
    plan_id: str
    success: bool
    final_plan: DecompositionPlan
    quality_score: float
    refinement_cycles: int
    solutions: List[SolutionAttempt]
    knowledge_extracted: bool
    execution_time: float
    metadata: Dict[str, Any]


class SovereignIntegrationOrchestrator:
    """
    Orchestrates complete end-to-end workflow integrating all components.
    
    This is the main entry point for the sovereign decomposition system,
    coordinating problem analysis, decomposition, validation, refinement,
    solution tracking, and knowledge extraction.
    """
    
    def __init__(self, orchestrator=None):
        """Initialize all system components with team integration."""
        self.logger = logging.getLogger(__name__)
        self.orchestrator = orchestrator
        
        # Load configurations
        self.performance_config = config_manager.get_performance_config()
        self.reliability_config = config_manager.get_reliability_config()
        self.openevolve_config = config_manager.get_openevolve_config()

        # Initialize core components with configurations
        self.db = SovereignDatabase(db_path=self.reliability_config.get("database", {}).get("db_path", "sovereign_decomposition.db"))
        self.error_handler = ErrorHandler() # ErrorHandler doesn't take config in __init__
        self.health_monitor = get_health_monitor() # Global instance, configured externally if needed

        # Configure ErrorHandler (e.g., logging levels, alerting integration)
        # For now, assume default logging is sufficient, but this is where
        # external alerting could be configured.

        # Initialize ProblemAnalyzer with OpenEvolve client config
        self.analyzer = ProblemAnalyzer(openevolve_client_config=self.openevolve_config)

        # Initialize PerformanceOptimizationOrchestrator and configure optimizers
        from performance_optimization import PerformanceOptimizationOrchestrator, PerformanceOptimizationType
        self.optimizer_orchestrator = PerformanceOptimizationOrchestrator()
        for opt_type, opt_config in self.performance_config.items():
            optimizer = self.optimizer_orchestrator.get_optimizer(PerformanceOptimizationType[opt_type.upper()])
            if optimizer and opt_config.get("enabled", True):
                optimizer.configure(**opt_config)
                optimizer.enabled = True # Ensure enabled status is set
            elif optimizer:
                optimizer.enabled = False # Disable if not enabled in config

        # Get CachingOptimizer instance for passing to strategies
        self.caching_optimizer = self.optimizer_orchestrator.get_optimizer(PerformanceOptimizationType.CACHING)
        if not self.caching_optimizer:
            self.logger.warning("CachingOptimizer not found in orchestrator. Caching will be disabled for strategies.")

        # Initialize DecompositionEngine with analyzer and caching_optimizer
        self.engine = DecompositionEngine(self.analyzer, caching_optimizer=self.caching_optimizer)
        
        self.gauntlet_system = GauntletSystem()
        self.team_coordinator = TeamCoordinator(self.gauntlet_system)
        self.quality_assessor = QualityAssessor()
        self.solution_orchestrator = SolutionOrchestrator()
        self.refinement_coordinator = RefinementCoordinator(
            self.gauntlet_system,
            self.quality_assessor,
            self.team_coordinator
        )
        self.knowledge_manager = KnowledgeManager()
        self.sub_problem_solver = SubProblemSolver() # Instantiate the new solver
        
        # Register health checks
        self._register_health_checks()
        
        # Import and initialize dependency manager
        from dependency_manager import DependencyManager
        self.dependency_manager = DependencyManager()
        
        # Initialize team systems if available
        self.red_team = None
        self.blue_team = None
        self.evaluator_team = None
        
        try:
            from red_team import RedTeam
            from blue_team import BlueTeam
            from evaluator_team import EvaluatorTeam
            
            self.red_team = RedTeam(orchestrator=orchestrator)
            self.blue_team = BlueTeam(orchestrator=orchestrator)
            self.evaluator_team = EvaluatorTeam(orchestrator=orchestrator)
            self.logger.info("Team systems initialized successfully")
        except ImportError as e:
            self.logger.warning(f"Team systems not available: {e}")
        
        self.logger.info("Sovereign Integration Orchestrator initialized")
    
    def run_complete_workflow(
        self,
        problem_text: str,
        title: str = "",
        strategy: str = 'hybrid',
        max_refinement_cycles: int = 3,
        enable_knowledge_extraction: bool = True
    ) -> IntegrationResult:
        """
        Execute complete end-to-end workflow.
        
        Args:
            problem_text: Problem description
            title: Problem title
            strategy: Decomposition strategy to use
            max_refinement_cycles: Maximum refinement iterations
            enable_knowledge_extraction: Whether to extract knowledge patterns
            
        Returns:
            IntegrationResult with complete workflow results
        """
        start_time = datetime.now()
        self.logger.info(f"Starting complete workflow for: {title or 'Untitled'}")
        
        try:
            # Phase 1: Problem Analysis
            try:
                self.logger.info("Phase 1: Analyzing problem...")
                problem = self.analyzer.analyze_problem(problem_text, title)
            except Exception as e:
                raise AnalysisError(f"Problem analysis failed: {e}", severity=ErrorSeverity.CRITICAL) from e

            # Phase 2: Decomposition
            try:
                self.logger.info(f"Phase 2: Decomposing with {strategy} strategy...")
                plan = self.engine.decompose(problem, strategy=strategy)
            except Exception as e:
                raise DecompositionError(f"Decomposition failed: {e}", severity=ErrorSeverity.CRITICAL) from e
            
            # Phase 2.5: Build Dependency Graph
            try:
                self.logger.info("Phase 2.5: Building dependency graph...")
                if plan.sub_problems:
                    dependency_graph = self.dependency_manager.build_graph(plan.sub_problems)
                    plan.dependency_graph = dependency_graph
                    
                    # Validate dependencies
                    dep_validation = self.dependency_manager.validate_dependencies(dependency_graph)
                    if not dep_validation.passed:
                        self.logger.warning(f"Dependency validation issues: {dep_validation.feedback}")
            except Exception as e:
                self.logger.error(f"Dependency graph construction failed: {e}")

            # Phase 3: Validation
            try:
                self.logger.info("Phase 3: Running validation gauntlets...")
                gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(plan)
            except Exception as e:
                raise ValidationError(f"Validation gauntlets failed: {e}", severity=ErrorSeverity.HIGH) from e
            
            # Phase 4: Quality Assessment
            try:
                self.logger.info("Phase 4: Assessing quality...")
                quality_report = self.quality_assessor.generate_quality_report(plan)
            except Exception as e:
                self.logger.error(f"Quality assessment failed: {e}")
                quality_report = None

            # Phase 5: Refinement (if needed)
            refinement_cycles = 0
            if quality_report and not quality_report.meets_thresholds and max_refinement_cycles > 0:
                try:
                    self.logger.info("Phase 5: Refining decomposition...")
                    plan, refinement_cycles = self._refine_decomposition(
                        plan, 
                        gauntlet_results,
                        max_refinement_cycles
                    )
                    # Re-assess quality after refinement
                    quality_report = self.quality_assessor.generate_quality_report(plan)
                except Exception as e:
                    self.logger.error(f"Refinement failed: {e}")
            
            # Phase 6: Team Coordination with Red/Blue/Evaluator Teams
            try:
                self.logger.info("Phase 6: Coordinating team review...")
                team_feedback = self._run_team_review(plan)
                
                # Apply team feedback if significant issues found
                if team_feedback and any(f.severity in ['critical', 'major'] for f in team_feedback):
                    self.logger.info("Applying team feedback...")
                    plan = self._apply_team_feedback(plan, team_feedback)
            except Exception as e:
                self.logger.error(f"Team coordination failed: {e}")
                team_feedback = []
            
            # Phase 7: Solution Tracking Setup
            self.logger.info("Phase 7: Setting up solution tracking...")
            solutions = self._initialize_solution_tracking(plan)
            
            # Phase 8: Knowledge Extraction
            knowledge_extracted = False
            if enable_knowledge_extraction and quality_report and quality_report.meets_thresholds:
                try:
                    self.logger.info("Phase 8: Extracting knowledge patterns...")
                    self.knowledge_manager.extract_patterns(
                        plan,
                        success=True,
                        quality_score=quality_report.metrics.overall_score
                    )
                    knowledge_extracted = True
                except Exception as e:
                    self.logger.error(f"Knowledge extraction failed: {e}")
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = IntegrationResult(
                problem_id=problem.id,
                plan_id=plan.id,
                success=quality_report.meets_thresholds if quality_report else False,
                final_plan=plan,
                quality_score=quality_report.metrics.overall_score if quality_report else 0.0,
                refinement_cycles=refinement_cycles,
                solutions=solutions,
                knowledge_extracted=knowledge_extracted,
                execution_time=execution_time,
                metadata={
                    'problem_type': problem.problem_type.value,
                    'strategy': strategy,
                    'sub_problem_count': len(plan.sub_problems),
                    'gauntlet_results': {k: v.passed for k, v in gauntlet_results.items()},
                    'quality_metrics': {
                        'coherence': quality_report.metrics.coherence_score if quality_report else 0.0,
                        'completeness': quality_report.metrics.completeness_score if quality_report else 0.0,
                        'feasibility': quality_report.metrics.feasibility_score if quality_report else 0.0,
                        'integration': quality_report.metrics.integration_score if quality_report else 0.0
                    },
                    'team_feedback_count': len(team_feedback) if team_feedback else 0
                }
            )
            
            self.logger.info(
                f"Workflow complete: {result.success}, "
                f"quality={result.quality_score:.2f}, "
                f"time={result.execution_time:.2f}s"
            )
            
            return result
            
        except (AnalysisError, DecompositionError, ValidationError) as e:
            self.logger.error(f"Workflow failed in phase: {type(e).__name__} - {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Use the centralized error handler
            error_info = self.error_handler.handle_error(
                e,
                context={'workflow': 'run_complete_workflow', 'problem_title': title},
                severity=e.severity
            )
            
            # Return failure result
            return IntegrationResult(
                problem_id="",
                plan_id="",
                success=False,
                final_plan=None,
                quality_score=0.0,
                refinement_cycles=0,
                solutions=[],
                knowledge_extracted=False,
                execution_time=execution_time,
                metadata={'error': str(e), 'error_info': error_info}
            )
        except Exception as e:
            self.logger.error(f"An unexpected error occurred in the workflow: {e}", exc_info=True)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Use the centralized error handler
            error_info = self.error_handler.handle_error(
                e,
                context={'workflow': 'run_complete_workflow', 'problem_title': title},
                severity=ErrorSeverity.CRITICAL
            )
            
            # Return failure result
            return IntegrationResult(
                problem_id="",
                plan_id="",
                success=False,
                final_plan=None,
                quality_score=0.0,
                refinement_cycles=0,
                solutions=[],
                knowledge_extracted=False,
                execution_time=execution_time,
                metadata={'error': str(e), 'error_info': error_info}
            )
    
    def _refine_decomposition(
        self,
        plan: DecompositionPlan,
        gauntlet_results: Dict,
        max_cycles: int
    ) -> tuple[DecompositionPlan, int]:
        """
        Refine decomposition based on feedback.
        
        Args:
            plan: Current decomposition plan
            gauntlet_results: Results from gauntlets
            max_cycles: Maximum refinement cycles
            
        Returns:
            Tuple of (refined_plan, cycles_used)
        """
        feedback = self.gauntlet_system.process_gauntlet_feedback(gauntlet_results)
        
        if not feedback:
            self.logger.info("No feedback to process")
            return plan, 0
        
        current_plan = plan
        previous_quality = self.quality_assessor.generate_quality_report(plan).metrics.overall_score
        
        for cycle in range(max_cycles):
            self.logger.info(f"Refinement cycle {cycle + 1}/{max_cycles}")
            
            # Generate refinement plan
            refinement_plan = self.refinement_coordinator.generate_refinement_plan(
                current_plan,
                feedback
            )
            
            if not refinement_plan or not refinement_plan.improvements:
                self.logger.info("No improvements identified")
                return current_plan, cycle
            
            # Execute refinement
            refined_plan, metrics = self.refinement_coordinator.execute_refinement(
                current_plan,
                refinement_plan
            )
            
            # Rebuild dependency graph if structure changed
            if refined_plan.sub_problems:
                dependency_graph = self.dependency_manager.build_graph(refined_plan.sub_problems)
                refined_plan.dependency_graph = dependency_graph
            
            # Re-validate
            new_gauntlet_results = self.gauntlet_system.run_decomposition_gauntlets(refined_plan)
            new_quality = self.quality_assessor.generate_quality_report(refined_plan)
            current_quality = new_quality.metrics.overall_score
            
            # Check if improved
            if new_quality.meets_thresholds:
                self.logger.info(f"Quality thresholds met after {cycle + 1} cycles")
                return refined_plan, cycle + 1
            
            # Check for convergence (minimal improvement)
            improvement = current_quality - previous_quality
            if improvement < 0.01:
                self.logger.info(f"Converged after {cycle + 1} cycles (improvement: {improvement:.3f})")
                return refined_plan, cycle + 1
            
            # Update for next cycle
            current_plan = refined_plan
            previous_quality = current_quality
            feedback = self.gauntlet_system.process_gauntlet_feedback(new_gauntlet_results)
        
        self.logger.warning(f"Max refinement cycles ({max_cycles}) reached")
        return current_plan, max_cycles
    
    def _initialize_solution_tracking(
        self,
        plan: DecompositionPlan
    ) -> List[SolutionAttempt]:
        """
        Initialize solution tracking for all sub-problems.
        
        Args:
            plan: Decomposition plan
            
        Returns:
            List of initialized SolutionAttempt objects
        """
        solutions = []
        
        for sub_problem in plan.sub_problems:
            solution = SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id=sub_problem.id,
                approach="pending",
                solution_content="",
                team_id="",
                confidence_score=0.0,
                validation_results=[],
                feedback=[],
                status="pending",
                created_at=datetime.now(),
                metadata={}
            )
            solutions.append(solution)
            self.solution_orchestrator.track_solution_attempt(
                sub_problem_id=solution.sub_problem_id,
                approach=solution.approach,
                solution_content=solution.solution_content,
                team_id=solution.team_id,
                confidence_score=solution.confidence_score
            )
        
        return solutions
    
    def solve_sub_problem(
        self,
        plan: DecompositionPlan,
        sub_problem_id: str,
        approach: str,
        implementation: str
    ) -> SolutionAttempt:
        """
        Record and validate a solution attempt for a sub-problem.
        
        Args:
            plan: Decomposition plan
            sub_problem_id: ID of sub-problem being solved
            approach: Solution approach description
            implementation: Implementation details
            
        Returns:
            SolutionAttempt with validation results
        """
        self.logger.info(f"Recording solution for sub-problem: {sub_problem_id}")
        
        # Find sub-problem
        sub_problem = next((sp for sp in plan.sub_problems if sp.id == sub_problem_id), None)
        if not sub_problem:
            raise ValueError(f"Sub-problem not found: {sub_problem_id}")
        
        # Create solution attempt
        solution = SolutionAttempt(
            id=generate_id("solution"),
            sub_problem_id=sub_problem_id,
            approach=approach,
            solution_content=implementation,
            team_id="manual",
            confidence_score=0.9, # Manual solutions are highly confident
            validation_results=[],
            feedback=[],
            status="pending",
            created_at=datetime.now(),
            metadata={}
        )
        
        # Track and validate
        self.solution_orchestrator.track_solution_attempt(
            sub_problem_id=solution.sub_problem_id,
            approach=solution.approach,
            solution_content=solution.solution_content,
            team_id=solution.team_id,
            confidence_score=solution.confidence_score
        )
        validated_solution = self.solution_orchestrator.validate_solution(solution, sub_problem)
        
        self.logger.info(
            f"Solution validated: success={validated_solution.passed}, "
            f"confidence={validated_solution.score:.2f}"
        )
        
        return validated_solution
    
    def integrate_all_solutions(
        self,
        plan: DecompositionPlan,
        solutions: List[SolutionAttempt]
    ) -> Dict[str, Any]:
        """
        Integrate all sub-problem solutions into final solution.
        
        Args:
            plan: Decomposition plan
            solutions: List of solution attempts
            
        Returns:
            Integration result with final solution and conflicts
        """
        self.logger.info("Integrating all solutions...")
        
        result = self.solution_orchestrator.integrate_solutions(plan, solutions)
        
        self.logger.info(
            f"Integration complete: success={result.success}, "
            f"conflicts={len(result.conflicts_resolved)}"
        )
        
        return result.to_dict()
    
    def get_workflow_status(self, plan_id: str) -> Dict[str, Any]:
        """
        Get current status of a workflow by querying the persistence layer.
        
        Args:
            plan_id: ID of decomposition plan
            
        Returns:
            Status dictionary with progress information
        """
        plan = self.db.get_plan(plan_id)
        if not plan:
            return {
                'plan_id': plan_id,
                'status': 'not_found',
                'message': f'Decomposition plan with ID {plan_id} not found.',
                'timestamp': datetime.now().isoformat()
            }

        overall_status = plan.status.value
        sub_problem_statuses = {}
        
        total_sub_problems = len(plan.sub_problems)
        solved_sub_problems = 0
        in_progress_sub_problems = 0
        pending_sub_problems = 0
        failed_sub_problems = 0

        for sub_problem in plan.sub_problems:
            sub_problem_statuses[sub_problem.id] = sub_problem.status
            
            if sub_problem.status == "SOLVED":
                solved_sub_problems += 1
            elif sub_problem.status == "IN_PROGRESS":
                in_progress_sub_problems += 1
            elif sub_problem.status == "PENDING":
                pending_sub_problems += 1
            elif sub_problem.status == "FAILED":
                failed_sub_problems += 1

        # Determine overall status based on sub-problem progress
        if total_sub_problems == 0:
            progress_percentage = 0.0
        else:
            progress_percentage = (solved_sub_problems / total_sub_problems) * 100

        if solved_sub_problems == total_sub_problems and total_sub_problems > 0:
            overall_status = "completed"
        elif failed_sub_problems > 0:
            overall_status = "failed"
        elif in_progress_sub_problems > 0 or solved_sub_problems > 0:
            overall_status = "in_progress"
        else:
            overall_status = "pending" # All pending

        return {
            'plan_id': plan_id,
            'status': overall_status,
            'progress_percentage': round(progress_percentage, 2),
            'sub_problem_summary': {
                'total': total_sub_problems,
                'solved': solved_sub_problems,
                'in_progress': in_progress_sub_problems,
                'pending': pending_sub_problems,
                'failed': failed_sub_problems
            },
            'sub_problem_statuses': sub_problem_statuses,
            'timestamp': datetime.now().isoformat()
        }
    
    def apply_learned_patterns(
        self,
        problem: ProblemDefinition
    ) -> Optional[Dict[str, Any]]:
        """
        Apply previously learned patterns to a new problem.
        
        Args:
            problem: Problem to decompose
            
        Returns:
            Pattern guidance dictionary, or None if no patterns match
        """
        self.logger.info(f"Searching for applicable patterns for: {problem.title}")
        
        # Retrieve similar patterns
        patterns = self.knowledge_manager.retrieve_patterns(
            problem_type=problem.problem_type,
            domain=problem.domain_context.domain,
            min_success_rate=0.7
        )
        
        if not patterns:
            self.logger.info("No applicable patterns found")
            return None
        
        # Apply best pattern
        best_pattern = patterns[0]
        self.logger.info(f"Applying pattern: {best_pattern.id}")
        
        guidance = self.knowledge_manager.apply_pattern(best_pattern, problem.description)
        
        return guidance


# Convenience functions for common workflows

def decompose_problem(
    problem_text: str,
    title: str = "",
    strategy: str = 'hybrid',
    max_refinement_cycles: int = 3
) -> IntegrationResult:
    """
    Convenience function to decompose a problem using the complete workflow.
    
    Args:
        problem_text: Problem description
        title: Problem title
        strategy: Decomposition strategy ('semantic', 'dependency', 'complexity', 'hybrid')
        max_refinement_cycles: Maximum refinement iterations
        
    Returns:
        IntegrationResult with complete workflow results
    """
    orchestrator = SovereignIntegrationOrchestrator()
    return orchestrator.run_complete_workflow(
        problem_text,
        title,
        strategy,
        max_refinement_cycles
    )


def execute_complete_solution_workflow(
    problem_text: str,
    title: str = "",
    strategy: str = 'hybrid'
) -> Dict[str, Any]:
    """
    Execute complete workflow including solution execution.
    
    This is the full end-to-end workflow that:
    1. Analyzes and decomposes the problem
    2. Validates and refines the decomposition
    3. Executes solutions for each sub-problem
    4. Integrates all solutions into final result
    5. Extracts knowledge for future use
    
    Args:
        problem_text: Problem description
        title: Problem title
        strategy: Decomposition strategy
        
    Returns:
        Dictionary with complete workflow results
    """
    orchestrator = SovereignIntegrationOrchestrator()
    return orchestrator.execute_complete_solution_workflow(
        problem_text,
        title,
        strategy
    )

    def _run_team_review(self, plan: DecompositionPlan) -> List:
        """Run Red/Blue/Evaluator team review on decomposition plan."""
        from sovereign_data_models import Feedback
        
        all_feedback = []
        
        # Convert plan to content for team analysis
        plan_content = self._plan_to_content(plan)
        
        # Red Team Critique
        if self.red_team:
            try:
                self.logger.info("Running Red Team critique...")
                red_assessment = self.red_team.assess_content(plan_content, "protocol")
                for finding in red_assessment.findings:
                    all_feedback.append(Feedback(
                        id=generate_id("feedback"),
                        source="red_team",
                        feedback_type="critique",
                        content=f"{finding.title}: {finding.description}",
                        severity=self._map_severity(finding.severity),
                        actionable=True,
                        timestamp=datetime.now()
                    ))
                self.logger.info(f"Red Team found {len(red_assessment.findings)} issues")
            except Exception as e:
                self.logger.warning(f"Red Team analysis failed: {e}")
        
        # Blue Team Suggestions
        if self.blue_team and self.red_team:
            try:
                self.logger.info("Running Blue Team refinement...")
                red_assessment = self.red_team.assess_content(plan_content, "protocol")
                fix_suggestions = self.blue_team.suggest_fixes(
                    plan_content,
                    red_assessment.findings,
                    "protocol"
                )
                for suggestion in fix_suggestions:
                    all_feedback.append(Feedback(
                        id=generate_id("feedback"),
                        source="blue_team",
                        feedback_type="suggestion",
                        content=suggestion.fix_description,
                        severity="minor",
                        actionable=True,
                        timestamp=datetime.now()
                    ))
                self.logger.info(f"Blue Team provided {len(fix_suggestions)} suggestions")
            except Exception as e:
                self.logger.warning(f"Blue Team analysis failed: {e}")
        
        # Evaluator Team Assessment
        if self.evaluator_team:
            try:
                self.logger.info("Running Evaluator Team assessment...")
                evaluation = self.evaluator_team.evaluate_content(plan_content, "protocol")
                all_feedback.append(Feedback(
                    id=generate_id("feedback"),
                    source="evaluator_team",
                    feedback_type="approval" if evaluation.final_verdict == "APPROVED" else "critique",
                    content=f"Verdict: {evaluation.final_verdict}. Score: {evaluation.consensus_score:.1f}/100",
                    severity="info" if evaluation.final_verdict == "APPROVED" else "major",
                    actionable=evaluation.final_verdict != "APPROVED",
                    timestamp=datetime.now()
                ))
                self.logger.info(f"Evaluator Team verdict: {evaluation.final_verdict}")
            except Exception as e:
                self.logger.warning(f"Evaluator Team analysis failed: {e}")
        
        return all_feedback
    
    def _plan_to_content(self, plan: DecompositionPlan) -> str:
        """Convert decomposition plan to text content for team analysis."""
        content = f"DECOMPOSITION PLAN\n"
        content += f"Strategy: {plan.strategy.value}\n"
        content += f"Sub-problems: {len(plan.sub_problems)}\n\n"
        
        for i, sp in enumerate(plan.sub_problems, 1):
            content += f"{i}. {sp.title} ({sp.type.value})\n"
            content += f"   Description: {sp.description}\n"
            content += f"   Priority: {sp.priority}, Effort: {sp.estimated_effort}h\n"
            if sp.dependencies:
                content += f"   Dependencies: {len(sp.dependencies)}\n"
            content += "\n"
        
        return content
    
    def _map_severity(self, severity) -> str:
        """Map team severity levels to feedback severity."""
        severity_map = {
            "CRITICAL": "critical",
            "HIGH": "major",
            "MEDIUM": "minor",
            "LOW": "info"
        }
        severity_str = str(severity).upper() if hasattr(severity, 'value') else str(severity).upper()
        return severity_map.get(severity_str, "minor")
    
    def _apply_team_feedback(self, plan: DecompositionPlan, feedback: List) -> DecompositionPlan:
        """
        Apply team feedback to improve decomposition plan.
        This method now leverages the RefinementCoordinator to intelligently process
        and apply feedback, potentially triggering further refinement cycles.
        """
        self.logger.info(f"Received {len(feedback)} feedback items from teams")

        actionable_feedback = [f for f in feedback if f.actionable and f.severity in ['critical', 'major']]

        if not actionable_feedback:
            self.logger.info("No actionable critical or major feedback from teams to apply.")
            return plan

        self.logger.info(f"Applying {len(actionable_feedback)} actionable feedback items from teams.")

        # Generate a refinement plan based on team feedback
        refinement_plan = self.refinement_coordinator.generate_refinement_plan(
            plan,
            actionable_feedback # Pass the team feedback directly
        )

        if not refinement_plan or not refinement_plan.improvements:
            self.logger.info("Team feedback did not result in identified improvements.")
            return plan

        # Execute refinement based on team feedback
        refined_plan, metrics = self.refinement_coordinator.execute_refinement(
            plan,
            refinement_plan
        )
        
        # Rebuild dependency graph if structure changed
        if refined_plan.sub_problems:
            dependency_graph = self.dependency_manager.build_graph(refined_plan.sub_problems)
            refined_plan.dependency_graph = dependency_graph

        self.logger.info("Team feedback successfully applied and plan refined.")
        return refined_plan

    def _register_health_checks(self):
        """Register health checks for critical components."""
        from llm_cache import get_cache # Import get_cache here to avoid circular dependency

        self.health_monitor.register_check("database_connectivity", self._check_database_health)
        self.health_monitor.register_check("llm_service_availability", self._check_llm_service_health)
        self.health_monitor.register_check("cache_health", self._check_cache_health)


    def _check_database_health(self) -> bool:
        """Check database connectivity by performing a simple query."""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return cursor.fetchone()[0] == 1
        except Exception as e:
            self.error_handler.handle_error(e, context={"health_check": "database_connectivity"}, severity=ErrorSeverity.CRITICAL)
            return False

    def _check_llm_service_health(self) -> bool:
        """Check LLM service availability by making a dummy call."""
        if not self.analyzer.openevolve_client:
            return False # LLM client not initialized

        try:
            # Attempt a very small, cheap LLM call
            # This assumes openevolve_client.evolve can handle a simple ping
            result = self.analyzer.openevolve_client.evolve(
                content="ping",
                evolution_mode="standard",
                content_type="health_check",
                max_iterations=1,
                temperature=0.0,
                max_tokens=1
            )
            return result.success
        except Exception as e:
            self.error_handler.handle_error(e, context={"health_check": "llm_service_availability"}, severity=ErrorSeverity.CRITICAL)
            return False

    def _check_cache_health(self) -> bool:
        """Check cache health by performing a simple get/set operation."""
        try:
            from llm_cache import get_cache # Get the global LLMCache instance
            cache = get_cache()
            test_key = "health_check_key"
            test_value = "health_check_value"
            cache.set(model="health_check", messages=[{"role": "user", "content": test_key}], temperature=0.0, max_tokens=0, response=test_value)
            retrieved_value = cache.get(model="health_check", messages=[{"role": "user", "content": test_key}], temperature=0.0, max_tokens=0)
            cache.clear() # Clean up
            return retrieved_value == test_value
        except Exception as e:
            self.error_handler.handle_error(e, context={"health_check": "cache_health"}, severity=ErrorSeverity.CRITICAL)
            return False