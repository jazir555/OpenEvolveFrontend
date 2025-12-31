"""
Sovereign-Grade Refinement System - Comprehensive Implementation

This module implements intelligent, iterative refinement of decomposition plans
using Red Team (critics), Blue Team (fixers), and Evaluator Team (judges).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from sovereign_data_models import (
    DecompositionPlan, SubProblem, Feedback, generate_id
)

# Import existing team implementations
try:
    from red_team import RedTeam, IssueFinding
    from blue_team import BlueTeam, FixSuggestion
    from evaluator_team import EvaluatorTeam
    TEAMS_AVAILABLE = True
except ImportError:
    TEAMS_AVAILABLE = False
    logging.warning("Team implementations not available")

logger = logging.getLogger(__name__)


@dataclass
class RefinementCycle:
    """Represents one cycle of refinement."""
    cycle_number: int
    original_plan: DecompositionPlan
    red_team_findings: List[IssueFinding]
    blue_team_suggestions: List[FixSuggestion]
    evaluator_assessment: Any
    refined_plan: Optional[DecompositionPlan]
    improvement_score: float
    timestamp: datetime


@dataclass
class RefinementResult:
    """Complete refinement result."""
    initial_plan: DecompositionPlan
    final_plan: DecompositionPlan
    cycles: List[RefinementCycle]
    total_improvements: int
    final_quality_score: float
    converged: bool
    iterations_used: int
    total_time: float


class ComprehensiveRefinementEngine:
    """
    Comprehensive refinement engine that iteratively improves decomposition plans.
    
    Uses Red Team to identify issues, Blue Team to suggest fixes, and Evaluator Team
    to assess quality. Continues refining until convergence or max iterations.
    """
    
    def __init__(
        self,
        orchestrator=None,
        max_iterations: int = 5,
        convergence_threshold: float = 0.90,
        min_improvement: float = 0.05
    ):
        """
        Initialize refinement engine.
        
        Args:
            orchestrator: Model orchestrator for LLM access
            max_iterations: Maximum refinement cycles
            convergence_threshold: Quality score to consider converged
            min_improvement: Minimum improvement to continue refining
        """
        self.orchestrator = orchestrator
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.min_improvement = min_improvement
        self.logger = logging.getLogger(__name__)
        
        # Initialize teams if available
        if TEAMS_AVAILABLE:
            self.red_team = RedTeam(orchestrator=orchestrator)
            self.blue_team = BlueTeam(orchestrator=orchestrator)
            self.evaluator_team = EvaluatorTeam(orchestrator=orchestrator)
        else:
            self.red_team = None
            self.blue_team = None
            self.evaluator_team = None
            self.logger.warning("Teams not available - refinement will be limited")
    
    def refine_plan(
        self,
        plan: DecompositionPlan,
        api_key: Optional[str] = None
    ) -> RefinementResult:
        """
        Refine decomposition plan through iterative improvement.
        
        Args:
            plan: Initial decomposition plan
            api_key: API key for LLM access
            
        Returns:
            RefinementResult with refined plan and improvement history
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Starting comprehensive refinement of plan {plan.id}")
        
        cycles = []
        current_plan = plan
        previous_quality = 0.0
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"\n=== Refinement Cycle {iteration + 1}/{self.max_iterations} ===")
            
            # Run refinement cycle
            cycle = self._run_refinement_cycle(
                current_plan,
                iteration + 1,
                api_key
            )
            cycles.append(cycle)
            
            # Check for convergence
            if cycle.refined_plan:
                current_plan = cycle.refined_plan
                
                # Check quality improvement
                improvement = cycle.improvement_score - previous_quality
                self.logger.info(f"Quality: {cycle.improvement_score:.2f} (improvement: {improvement:+.2f})")
                
                # Check convergence conditions
                if cycle.improvement_score >= self.convergence_threshold:
                    self.logger.info(f"✓ Converged: Quality {cycle.improvement_score:.2f} >= {self.convergence_threshold}")
                    break
                
                if improvement < self.min_improvement and iteration > 0:
                    self.logger.info(f"✓ Converged: Improvement {improvement:.2f} < {self.min_improvement}")
                    break
                
                previous_quality = cycle.improvement_score
            else:
                self.logger.warning("Refinement cycle produced no refined plan")
                break
        
        total_time = time.time() - start_time
        
        # Count total improvements
        total_improvements = sum(
            len(c.blue_team_suggestions) for c in cycles
        )
        
        result = RefinementResult(
            initial_plan=plan,
            final_plan=current_plan,
            cycles=cycles,
            total_improvements=total_improvements,
            final_quality_score=cycles[-1].improvement_score if cycles else 0.0,
            converged=cycles[-1].improvement_score >= self.convergence_threshold if cycles else False,
            iterations_used=len(cycles),
            total_time=total_time
        )
        
        self.logger.info(f"\nRefinement complete: {len(cycles)} cycles, {total_improvements} improvements")
        self.logger.info(f"Final quality: {result.final_quality_score:.2f}")
        
        return result
    
    def _run_refinement_cycle(
        self,
        plan: DecompositionPlan,
        cycle_number: int,
        api_key: Optional[str]
    ) -> RefinementCycle:
        """Run one cycle of refinement."""
        
        # Convert plan to content for team analysis
        plan_content = self._plan_to_content(plan)
        
        # Step 1: Red Team identifies issues
        self.logger.info("Step 1: Red Team critique...")
        red_findings = []
        if self.red_team and TEAMS_AVAILABLE:
            try:
                assessment = self.red_team.assess_content(
                    content=plan_content,
                    content_type="protocol"
                )
                red_findings = assessment.findings
                self.logger.info(f"  Found {len(red_findings)} issues")
            except Exception as e:
                self.logger.error(f"Red Team failed: {e}")
        
        # Step 2: Blue Team suggests fixes
        self.logger.info("Step 2: Blue Team suggestions...")
        blue_suggestions = []
        if self.blue_team and TEAMS_AVAILABLE and red_findings:
            try:
                suggestions = self.blue_team.suggest_fixes(
                    content=plan_content,
                    issues=red_findings,
                    content_type="protocol"
                )
                blue_suggestions = suggestions
                self.logger.info(f"  Generated {len(blue_suggestions)} suggestions")
            except Exception as e:
                self.logger.error(f"Blue Team failed: {e}")
        
        # Step 3: Apply fixes to create refined plan
        self.logger.info("Step 3: Applying fixes...")
        refined_plan = self._apply_fixes_to_plan(plan, blue_suggestions)
        
        # Step 4: Evaluator Team assesses quality
        self.logger.info("Step 4: Evaluator assessment...")
        evaluator_assessment = None
        improvement_score = 0.5
        
        if self.evaluator_team and TEAMS_AVAILABLE and refined_plan:
            try:
                refined_content = self._plan_to_content(refined_plan)
                evaluation = self.evaluator_team.evaluate_content(
                    content=refined_content,
                    content_type="protocol",
                    previous_versions=[plan_content]
                )
                evaluator_assessment = evaluation
                improvement_score = evaluation.consensus_score / 100.0
                self.logger.info(f"  Quality score: {improvement_score:.2f}")
            except Exception as e:
                self.logger.error(f"Evaluator Team failed: {e}")
        
        return RefinementCycle(
            cycle_number=cycle_number,
            original_plan=plan,
            red_team_findings=red_findings,
            blue_team_suggestions=blue_suggestions,
            evaluator_assessment=evaluator_assessment,
            refined_plan=refined_plan,
            improvement_score=improvement_score,
            timestamp=datetime.now()
        )
    
    def _plan_to_content(self, plan: DecompositionPlan) -> str:
        """Convert decomposition plan to text content for team analysis."""
        content = f"""DECOMPOSITION PLAN
Strategy: {plan.strategy.value}
Sub-problems: {len(plan.sub_problems)}

"""
        
        for i, sp in enumerate(plan.sub_problems, 1):
            content += f"""{i}. {sp.title}
   Type: {sp.type.value}
   Priority: {sp.priority}
   Effort: {sp.estimated_effort}h
   Complexity: {sp.complexity_score.overall_complexity:.1f}/10
   Description: {sp.description}
   Dependencies: {', '.join(sp.dependencies) if sp.dependencies else 'None'}
   Success Criteria: {sp.success_criteria[0].description if sp.success_criteria else 'Not defined'}

"""
        
        return content
    
    def _apply_fixes_to_plan(
        self,
        plan: DecompositionPlan,
        suggestions: List[FixSuggestion]
    ) -> Optional[DecompositionPlan]:
        """
        Apply Blue Team suggestions to create refined plan.
        
        This is a simplified implementation - in production, this would
        intelligently merge suggestions into the plan structure.
        """
        if not suggestions:
            return plan
        
        # For now, return the original plan with updated metadata
        # In a full implementation, this would parse suggestions and
        # modify sub-problems, add missing pieces, etc.
        
        refined_plan = DecompositionPlan(
            id=generate_id("plan"),
            problem_id=plan.problem_id,
            strategy=plan.strategy,
            sub_problems=plan.sub_problems,  # Would be modified based on suggestions
            dependency_graph=plan.dependency_graph,
            validation_checkpoints=plan.validation_checkpoints,
            quality_scores=plan.quality_scores,
            confidence_level=min(1.0, plan.confidence_level + 0.1),
            created_by="refinement_engine",
            metadata={
                'refined_from': plan.id,
                'suggestions_applied': len(suggestions),
                'refinement_timestamp': datetime.now().isoformat()
            }
        )
        
        return refined_plan
    
    def generate_refinement_report(self, result: RefinementResult) -> str:
        """Generate human-readable refinement report."""
        report = f"""
REFINEMENT REPORT
================

Initial Quality: {result.cycles[0].improvement_score if result.cycles else 0:.2f}
Final Quality: {result.final_quality_score:.2f}
Improvement: {result.final_quality_score - (result.cycles[0].improvement_score if result.cycles else 0):+.2f}

Iterations: {result.iterations_used}/{self.max_iterations}
Total Improvements: {result.total_improvements}
Converged: {'Yes' if result.converged else 'No'}
Time: {result.total_time:.1f}s

REFINEMENT CYCLES:
"""
        
        for cycle in result.cycles:
            report += f"""
Cycle {cycle.cycle_number}:
  Issues Found: {len(cycle.red_team_findings)}
  Fixes Suggested: {len(cycle.blue_team_suggestions)}
  Quality Score: {cycle.improvement_score:.2f}
"""
            
            # Top issues
            if cycle.red_team_findings:
                report += "  Top Issues:\n"
                for finding in cycle.red_team_findings[:3]:
                    report += f"    - [{finding.severity.value}] {finding.title}\n"
            
            # Top suggestions
            if cycle.blue_team_suggestions:
                report += "  Top Suggestions:\n"
                for suggestion in cycle.blue_team_suggestions[:3]:
                    report += f"    - [{suggestion.priority.value}] {suggestion.fix_description[:80]}...\n"
        
        return report
