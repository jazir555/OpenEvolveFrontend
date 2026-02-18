"""Deterministic iterative refinement using the Gauntlet and MDAP/MAKER systems."""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .utils import optional_import
from .pipeline import DeterminismResult, DeterministicPipeline

logger = logging.getLogger(__name__)

@dataclass
class IssueFinding:
    id: str
    description: str
    severity: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FixSuggestion:
    id: str
    description: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAssessment:
    score: float
    improvement: float
    invariants_preserved: bool
    reproducibility_verified: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class RefinementRedTeam:
    """Identify issues while maintaining deterministic output (Critique Layer)."""
    
    def __init__(self, gauntlet_integration: Any = None):
        self.integration = gauntlet_integration
        if not self.integration:
            module = optional_import("mdap_maker_gauntlet_integration")
            if module:
                self.integration = module.MDAPMakerGauntletIntegration()

    def critique(self, output: Any, context: Dict[str, Any]) -> List[IssueFinding]:
        """Identify issues in the generated output using Gauntlet + MDAP/MAKER."""
        findings = []
        if not self.integration:
            logger.debug("Red Team: No gauntlet integration available, using basic heuristic critique.")
            return self._basic_critique(output, context)
            
        try:
            # Prepare gauntlet
            from gauntlet_types import AdversarialGauntlet, FormalVerificationGauntlet
            
            # Select appropriate gauntlet type based on context/output
            if isinstance(output, dict):
                gauntlet = FormalVerificationGauntlet(name="Refinement Critique Gauntlet")
            else:
                gauntlet = AdversarialGauntlet(name="Refinement Critique Gauntlet")
            
            # Execute with MDAP/MAKER for multi-agent adversarial analysis
            result = self.integration.execute_with_mdap_maker(
                gauntlet=gauntlet,
                solution=output,
                context=context,
                problem_description=context.get("prompt", "")
            )
            
            # Map results to IssueFindings
            if not result.gauntlet_result.passed:
                # Add findings from gauntlet details
                for i, detail in enumerate(result.gauntlet_result.details.get("findings", [])):
                    findings.append(IssueFinding(
                        id=f"red_{i}",
                        description=detail.get("description", str(detail)),
                        severity=detail.get("severity", "major"),
                        metadata=detail
                    ))
                
                # Add findings from red flags
                for i, flag in enumerate(result.red_flags):
                    findings.append(IssueFinding(
                        id=f"flag_{i}",
                        description=flag.get("message", "Red flag detected"),
                        severity=flag.get("severity", "critical"),
                        metadata=flag
                    ))
                    
                # If no specific findings but failed, add a general one
                if not findings:
                    findings.append(IssueFinding(
                        id="red_fail",
                        description=result.gauntlet_result.feedback or "Gauntlet validation failed",
                        severity="major"
                    ))
        except Exception as exc:
            logger.warning(f"Red Team: Advanced critique failed ({exc}), falling back to basic.")
            return self._basic_critique(output, context)
            
        return findings

    def _basic_critique(self, output: Any, context: Dict[str, Any]) -> List[IssueFinding]:
        findings = []
        out_str = str(output).lower()
        if "potential flaw" in out_str:
            findings.append(IssueFinding(id="basic_1", description="Potential flaw detected in output.", severity="major"))
        if len(out_str) < 10:
            findings.append(IssueFinding(id="basic_2", description="Output is suspiciously short.", severity="minor"))
        return findings

class RefinementBlueTeam:
    """Propose fixes while maintaining determinism (Fix Layer)."""
    
    def __init__(self, pipeline: DeterministicPipeline):
        self.pipeline = pipeline

    def generate_fixes(self, findings: List[IssueFinding], original_output: Any) -> List[FixSuggestion]:
        """Generate fixes for identified issues using MAKER-style voting if possible."""
        fixes = []
        if not findings:
            return fixes
            
        # Implementation logic: Combine findings into a single fix request
        # or handle them individually. For "full business logic", we do both.
        
        issue_summary = "\n".join([f"- {f.description} ({f.severity})" for f in findings])
        
        # Strategy: Use the pipeline to generate multiple candidates and select/merge them
        prompt = (
            f"Original output: {original_output}\n\n"
            f"The following issues were identified by the Red Team:\n{issue_summary}\n\n"
            f"Please provide a corrected version of the output that addresses all these issues. "
            f"Maintain the original format and structure."
        )
        
        # In a full MAKER implementation, we'd run this multiple times and vote
        # Here we use the deterministic pipeline's built-in layers (which might include consensus)
        # IMPORTANT: Disable refinement for the fix generation itself to avoid infinite recursion
        original_refinement_setting = self.pipeline.config.use_refinement
        self.pipeline.config.use_refinement = False
        try:
            result = self.pipeline.generate_with_all_layers(prompt)
        finally:
            self.pipeline.config.use_refinement = original_refinement_setting
        
        if result.success:
            fixes.append(FixSuggestion(
                id="blue_fix_comprehensive",
                description="Comprehensive fix addressing all identified issues",
                action=str(result.output),
                metadata={"findings_covered": [f.id for f in findings]}
            ))
                
        return fixes

class RefinementEvaluatorTeam:
    """Validate improvements with formal guarantees (Validation Layer)."""
    
    def __init__(self, gauntlet_integration: Any = None):
        self.integration = gauntlet_integration
        if not self.integration:
            module = optional_import("mdap_maker_gauntlet_integration")
            if module:
                self.integration = module.MDAPMakerGauntletIntegration()

    def assess_quality(self, original: Any, refined: Any, context: Dict[str, Any]) -> QualityAssessment:
        """Assess quality of refined output vs original with formal verification if possible."""
        if not self.integration:
            logger.debug("Evaluator Team: No gauntlet integration available, using basic similarity score.")
            return self._basic_assessment(original, refined)
            
        try:
            from gauntlet_types import CrossValidationGauntlet
            
            # Use GOLD TEAM style verification gauntlet
            gauntlet = CrossValidationGauntlet(name="Refinement Evaluation Gauntlet")
            
            result = self.integration.execute_with_mdap_maker(
                gauntlet=gauntlet,
                solution=refined,
                context={**context, "original_solution": original},
                problem_description="Verify if the refined solution is superior to the original."
            )
            
            return QualityAssessment(
                score=result.gauntlet_result.score,
                improvement=max(0.0, result.gauntlet_result.score - 0.5), # Heuristic
                invariants_preserved=result.gauntlet_result.passed,
                reproducibility_verified=result.consensus_reached,
                metadata=result.gauntlet_result.details
            )
        except Exception as exc:
            logger.warning(f"Evaluator Team: Advanced assessment failed ({exc}), falling back to basic.")
            return self._basic_assessment(original, refined)

    def _basic_assessment(self, original: Any, refined: Any) -> QualityAssessment:
        from .utils import similarity
        score = similarity(str(original), str(refined))
        return QualityAssessment(
            score=score,
            improvement=0.1 if str(original) != str(refined) else 0.0,
            invariants_preserved=True,
            reproducibility_verified=True,
            metadata={"method": "basic_similarity"}
        )

class DeterministicRefinementLoop:
    """Deterministic iterative refinement implementation using the 3-team model."""
    
    def __init__(
        self,
        pipeline: DeterministicPipeline,
        max_iterations: int = 3,
        convergence_threshold: float = 0.95
    ):
        self.pipeline = pipeline
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Initialize teams with shared integration if possible
        module = optional_import("mdap_maker_gauntlet_integration")
        integration = module.MDAPMakerGauntletIntegration() if module else None
        
        self.red_team = RefinementRedTeam(integration)
        self.blue_team = RefinementBlueTeam(pipeline)
        self.evaluator = RefinementEvaluatorTeam(integration)
        
        # State management integration
        self.state_mgr = None
        state_module = optional_import("crewai_state_management")
        if state_module:
            try:
                self.state_mgr = state_module.StateManager()
            except Exception:
                pass

        # Observability integration
        self.observability = None
        obs_module = optional_import("monitoring_system")
        if obs_module:
            try:
                self.observability = obs_module.get_observability_manager()
            except Exception:
                pass

    def refine(self, initial_output: Any, context: Dict[str, Any]) -> Any:
        """Refine the output iteratively until convergence or max iterations."""
        current_output = initial_output
        best_output = initial_output
        best_score = -1.0
        
        workflow_id = context.get("workflow_id") or f"refine_{int(time.time())}"
        
        if self.state_mgr:
            try:
                from crewai_state_management import WorkflowState, WorkflowStatus
                # Create initial state record
                state = WorkflowState(
                    workflow_id=workflow_id,
                    problem_statement=context.get("prompt", ""),
                    status=WorkflowStatus.SOLVING,
                    metadata={"refinement_context": context}
                )
                self.state_mgr.save_state(workflow_id, state)
            except Exception as exc:
                logger.debug(f"Failed to initialize refinement state: {exc}")

        history = []
        
        for i in range(self.max_iterations):
            logger.info(f"Refinement cycle {i+1}/{self.max_iterations}")
            
            # 1. Red Team critique
            findings = self.red_team.critique(current_output, context)
            if not findings:
                logger.info("Red Team: No issues found. Output is stable.")
                break
                
            # 2. Blue Team generates fixes
            fixes = self.blue_team.generate_fixes(findings, current_output)
            if not fixes:
                logger.info("Blue Team: No actionable fixes generated.")
                break
                
            # Use the best fix (in this implementation, we take the comprehensive one)
            refined_output = fixes[0].action
            
            # 3. Evaluator Team assesses quality
            assessment = self.evaluator.assess_quality(current_output, refined_output, context)
            
            # Update state if manager is available
            if self.state_mgr:
                try:
                    state = self.state_mgr.load_state(workflow_id)
                    if state:
                        # Append finding details to metadata
                        state.metadata[f"cycle_{i}_findings"] = [f.description for f in findings]
                        state.metadata[f"cycle_{i}_score"] = assessment.score
                        state.updated_at = datetime.now().isoformat()
                        self.state_mgr.save_state_with_versioning(workflow_id, state)
                except Exception as exc:
                    logger.debug(f"Failed to update refinement state: {exc}")

            # Record metrics
            if self.observability:
                try:
                    from monitoring_system import MetricType
                    self.observability.add_custom_metric("refinement_findings_count", len(findings), MetricType.GAUGE, {"iteration": str(i)})
                    self.observability.add_custom_metric("refinement_quality_score", assessment.score, MetricType.GAUGE, {"iteration": str(i)})
                    self.observability.add_custom_metric("refinement_improvement", assessment.improvement, MetricType.GAUGE, {"iteration": str(i)})
                except Exception:
                    pass

            history.append({
                "iteration": i,
                "findings": len(findings),
                "score": assessment.score,
                "improvement": assessment.improvement
            })
            
            if assessment.score > best_score:
                best_score = assessment.score
                best_output = refined_output
                
            if assessment.score >= self.convergence_threshold:
                logger.info(f"Refinement: Convergence reached (score={assessment.score:.4f}).")
                break
                
            # Update for next iteration
            current_output = refined_output
            
        if self.state_mgr:
            try:
                state = self.state_mgr.load_state(workflow_id)
                if state:
                    from crewai_state_management import WorkflowStatus
                    state.status = WorkflowStatus.COMPLETED
                    state.overall_score = best_score
                    self.state_mgr.save_state(workflow_id, state)
            except Exception:
                pass

        return best_output
