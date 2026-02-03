"""
ICR-CrewAI Integration Bridge

This module integrates Iterative Contextual Refinement (ICR) with CrewAI workflows,
enabling continuous improvement of decomposition plans through feedback loops.

Features:
- Extracts feedback from CrewAI workflow results
- Applies RefinementCoordinator to process feedback
- Generates refinement suggestions
- Iteratively improves decomposition plans
- Tracks convergence and quality metrics
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

# ICR imports
try:
    from sovereign_refinement import RefinementCoordinator, RefinementPlan, RefinementCycle
    from sovereign_data_models import (
        DecompositionPlan, SubProblem, Feedback, ValidationResult,
        QualityScores, SolutionAttempt, generate_id
    )
    ICR_AVAILABLE = True
except ImportError as e:
    ICR_AVAILABLE = False
    logging.warning(f"ICR components not available: {e}")

# CrewAI imports
try:
    from decomposition_crewai_bridge import (
        execute_phase_1_setup,
        execute_phase_2_solution,
        execute_phase_3_critique,
        execute_phase_4_verify,
        execute_phase_5_reassembly,
        execute_phase_6_final_validation,
    )
    CREWAI_BRIDGE_AVAILABLE = True
except ImportError as e:
    CREWAI_BRIDGE_AVAILABLE = False
    logging.warning(f"CrewAI bridge not available: {e}")
    # Create stub functions for graceful degradation
    def execute_phase_1_setup(*args, **kwargs): return {}
    def execute_phase_2_solution(*args, **kwargs): return {'solutions': []}
    def execute_phase_3_critique(*args, **kwargs): return {'critiques': []}
    def execute_phase_4_verify(*args, **kwargs): return {'verifications': {}, 'overall_score': 0.7}
    def execute_phase_5_reassembly(*args, **kwargs): return {}
    def execute_phase_6_final_validation(*args, **kwargs): return {}

logger = logging.getLogger(__name__)


@dataclass
class ICRWorkflowConfig:
    """Configuration for ICR-enhanced CrewAI workflows."""
    max_refinement_cycles: int = 3
    quality_threshold: float = 0.85
    convergence_threshold: float = 0.05
    enable_auto_refinement: bool = True
    refinement_strategy: str = "adaptive"  # adaptive, conservative, aggressive
    track_metrics: bool = True


@dataclass
class ICRWorkflowResult:
    """Result from ICR-enhanced CrewAI workflow execution."""
    workflow_id: str
    original_plan: Optional[DecompositionPlan]
    refined_plan: Optional[DecompositionPlan]
    cycles_completed: int
    quality_scores: List[QualityScores]
    feedback_applied: List[str]
    converged: bool
    final_quality: float
    total_time_seconds: float
    refinement_history: List[RefinementCycle] = field(default_factory=list)


class ICRCrewAIIntegration:
    """
    Integrates ICR refinement with CrewAI decomposition workflows.

    This class orchestrates the feedback loop between CrewAI workflow execution
    and ICR refinement, enabling continuous improvement of decomposition plans.
    """

    def __init__(
        self,
        refinement_coordinator: Optional[RefinementCoordinator] = None,
        config: Optional[ICRWorkflowConfig] = None
    ):
        """
        Initialize ICR-CrewAI integration.

        Args:
            refinement_coordinator: Optional RefinementCoordinator instance
            config: Optional workflow configuration
        """
        self.config = config or ICRWorkflowConfig()

        # Initialize refinement coordinator
        if ICR_AVAILABLE:
            self.refinement_coordinator = refinement_coordinator or RefinementCoordinator()
        else:
            self.refinement_coordinator = None
            logger.warning("ICR not available - workflow will run without refinement")

        self.logger = logging.getLogger(__name__)

    def execute_with_refinement(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> ICRWorkflowResult:
        """
        Execute CrewAI workflow with ICR refinement enabled.

        Runs the full decomposition workflow with iterative refinement based
        on feedback from verification phases.

        Args:
            problem_statement: The problem to solve
            problem_type: Optional problem type
            domain: Optional problem domain
            **kwargs: Additional arguments for workflow phases

        Returns:
            ICRWorkflowResult with refinement details
        """
        start_time = datetime.now()
        workflow_id = generate_id()

        self.logger.info(f"Starting ICR-enhanced workflow {workflow_id} for: {problem_statement[:50]}...")

        # Phase 1: Initial decomposition
        self.logger.info("Phase 1: Initial decomposition")
        phase1_result = execute_phase_1_setup(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
            **kwargs
        )

        original_plan = self._extract_plan(phase1_result)
        current_plan = original_plan
        quality_scores = []
        feedback_history = []
        refinement_cycles = []

        # Iterative refinement loop
        for cycle in range(self.config.max_refinement_cycles):
            self.logger.info(f"Refinement cycle {cycle + 1}/{self.config.max_refinement_cycles}")

            # Phase 2-4: Execute solution, critique, and verification
            phase2_result = execute_phase_2_solution(
                decomposition_plan=current_plan,
                **kwargs
            )

            phase3_result = execute_phase_3_critique(
                solutions=phase2_result.get('solutions', []),
                **kwargs
            )

            phase4_result = execute_phase_4_verify(
                solutions=phase2_result.get('solutions', []),
                requirements=phase1_result.get('requirements', [])
            )

            # Extract quality metrics
            quality = self._extract_quality_score(phase4_result)
            quality_scores.append(quality)

            # Extract feedback
            feedback = self._extract_feedback(phase3_result, phase4_result)
            feedback_history.extend(feedback)

            # Check convergence
            if len(quality_scores) >= 2:
                improvement = quality_scores[-1].overall_score - quality_scores[-2].overall_score
                if improvement < self.config.convergence_threshold:
                    self.logger.info(f"Converged after {cycle + 1} cycles (improvement: {improvement:.3f})")
                    break

            # Check quality threshold
            if quality.overall_score >= self.config.quality_threshold:
                self.logger.info(f"Quality threshold met: {quality.overall_score:.3f} >= {self.config.quality_threshold}")
                break

            # Apply refinement if enabled
            if self.config.enable_auto_refinement and self.refinement_coordinator:
                refinement_plan = self.refinement_coordinator.generate_refinement_plan(
                    plan=current_plan,
                    feedback_list=feedback
                )

                # Apply refinements to plan
                current_plan = self._apply_refinement(current_plan, refinement_plan)

                # Track cycle
                cycle_data = RefinementCycle(
                    cycle_number=cycle + 1,
                    plan_id=workflow_id,
                    feedback_received=feedback,
                    improvements_applied=refinement_plan.improvements,
                    quality_before=quality_scores[-2].overall_score if len(quality_scores) >= 2 else 0.0,
                    quality_after=quality.overall_score,
                    gauntlet_results=phase4_result.get('verifications', {}),
                    converged=False
                )
                refinement_cycles.append(cycle_data)
            else:
                # No refinement, track cycle without changes
                cycle_data = RefinementCycle(
                    cycle_number=cycle + 1,
                    plan_id=workflow_id,
                    feedback_received=feedback,
                    improvements_applied=[],
                    quality_before=quality_scores[-2].overall_score if len(quality_scores) >= 2 else 0.0,
                    quality_after=quality.overall_score,
                    gauntlet_results=phase4_result.get('verifications', {}),
                    converged=False
                )
                refinement_cycles.append(cycle_data)

        # Phase 5-6: Final reassembly and validation
        if current_plan:
            phase5_result = execute_phase_5_reassembly(
                solutions=phase2_result.get('solutions', []),
                original_plan=current_plan
            )

            phase6_result = execute_phase_6_final_validation(
                reassembled_solution=phase5_result.get('reassembled_solution'),
                requirements=phase1_result.get('requirements', [])
            )
        else:
            phase5_result = {}
            phase6_result = {}

        # Calculate final metrics
        total_time = (datetime.now() - start_time).total_seconds()
        final_quality = quality_scores[-1].overall_score if quality_scores else 0.0
        converged = final_quality >= self.config.quality_threshold

        # Compile feedback applied
        feedback_applied = []
        for cycle in refinement_cycles:
            feedback_applied.extend(cycle.improvements_applied)

        result = ICRWorkflowResult(
            workflow_id=workflow_id,
            original_plan=original_plan,
            refined_plan=current_plan,
            cycles_completed=len(refinement_cycles),
            quality_scores=quality_scores,
            feedback_applied=feedback_applied,
            converged=converged,
            final_quality=final_quality,
            total_time_seconds=total_time,
            refinement_history=refinement_cycles
        )

        self.logger.info(
            f"ICR workflow completed: {result.cycles_completed} cycles, "
            f"final quality: {result.final_quality:.3f}, "
            f"converged: {result.converged}"
        )

        return result

    def _extract_plan(self, phase_result: Dict[str, Any]) -> Optional[DecompositionPlan]:
        """Extract DecompositionPlan from phase 1 result."""
        # This is a placeholder - real implementation would extract the actual plan
        if ICR_AVAILABLE:
            return DecompositionPlan(
                id=generate_id(),
                problem_statement=phase_result.get('problem_statement', ''),
                sub_problems=[],
                created_at=datetime.now()
            )
        return None

    def _extract_quality_score(self, verification_result: Dict[str, Any]) -> QualityScores:
        """Extract quality scores from verification result."""
        if ICR_AVAILABLE:
            return QualityScores(
                overall_score=verification_result.get('overall_score', 0.7),
                completeness=verification_result.get('completeness', 0.7),
                correctness=verification_result.get('correctness', 0.7),
                clarity=verification_result.get('clarity', 0.7),
                efficiency=verification_result.get('efficiency', 0.7),
                maintainability=verification_result.get('maintainability', 0.7),
                scalability=verification_result.get('scalability', 0.7),
                security=verification_result.get('security', 0.7),
                test_coverage=verification_result.get('test_coverage', 0.7)
            )
        # Fallback
        from dataclasses import dataclass
        @dataclass
        class FallbackQuality:
            overall_score: float = 0.7
        return FallbackQuality()

    def _extract_feedback(self, critique_result: Dict, verification_result: Dict) -> List[Feedback]:
        """Extract feedback from critique and verification results."""
        feedback_list = []

        if not ICR_AVAILABLE:
            return feedback_list

        # Extract from critique
        critiques = critique_result.get('critiques', [])
        for critique in critiques:
            feedback = Feedback(
                id=generate_id(),
                source='red_team',
                feedback_type='critique',
                content=critique.get('content', ''),
                severity=critique.get('severity', 'medium'),
                actionable=True
            )
            feedback_list.append(feedback)

        # Extract from verification
        verifications = verification_result.get('verifications', {})
        for sol_id, verification in verifications.items():
            if not verification.get('verified', True):
                feedback = Feedback(
                    id=generate_id(),
                    source='verification',
                    feedback_type='validation_failure',
                    content=verification.get('reason', 'Verification failed'),
                    severity='high',
                    actionable=True
                )
                feedback_list.append(feedback)

        return feedback_list

    def _apply_refinement(
        self,
        plan: DecompositionPlan,
        refinement_plan: RefinementPlan
    ) -> DecompositionPlan:
        """Apply refinement plan to decomposition plan.

        This is a placeholder - real implementation would:
        1. Analyze refinement suggestions
        2. Modify sub-problems based on feedback
        3. Reorder or merge sub-problems
        4. Update plan metadata
        """
        self.logger.info(f"Applying {len(refinement_plan.improvements)} refinements")

        # Placeholder: Create a new plan with updated metadata
        if ICR_AVAILABLE:
            return DecompositionPlan(
                id=generate_id(),
                problem_statement=plan.problem_statement,
                sub_problems=plan.sub_problems.copy(),
                metadata={
                    **(plan.metadata or {}),
                    'refined': True,
                    'refinements_applied': refinement_plan.improvements,
                    'refinement_count': len(refinement_plan.improvements)
                },
                created_at=datetime.now()
            )
        return plan


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def execute_icr_enhanced_workflow(
    problem_statement: str,
    problem_type: Optional[str] = None,
    domain: Optional[str] = None,
    max_cycles: int = 3,
    quality_threshold: float = 0.85,
    **kwargs
) -> ICRWorkflowResult:
    """
    Execute ICR-enhanced CrewAI workflow with convenience parameters.

    Args:
        problem_statement: The problem to solve
        problem_type: Optional problem type
        domain: Optional problem domain
        max_cycles: Maximum refinement cycles
        quality_threshold: Quality threshold for convergence
        **kwargs: Additional arguments

    Returns:
        ICRWorkflowResult with refinement details
    """
    config = ICRWorkflowConfig(
        max_refinement_cycles=max_cycles,
        quality_threshold=quality_threshold
    )

    integration = ICRCrewAIIntegration(config=config)
    return integration.execute_with_refinement(
        problem_statement=problem_statement,
        problem_type=problem_type,
        domain=domain,
        **kwargs
    )


def get_icr_integration_status() -> Dict[str, Any]:
    """Get status of ICR-CrewAI integration."""
    return {
        'icr_available': ICR_AVAILABLE,
        'crewai_bridge_available': CREWAI_BRIDGE_AVAILABLE,
        'integration_ready': ICR_AVAILABLE and CREWAI_BRIDGE_AVAILABLE,
        'refinement_coordinator_available': ICR_AVAILABLE,
    }


__all__ = [
    'ICRCrewAIIntegration',
    'ICRWorkflowConfig',
    'ICRWorkflowResult',
    'execute_icr_enhanced_workflow',
    'get_icr_integration_status',
]
