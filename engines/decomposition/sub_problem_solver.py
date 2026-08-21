"""
Sub-Problem Solver for Sovereign-Grade Problem Decomposition System
"""
from __future__ import annotations


import logging
import time
from typing import Optional, Dict, Any, List

from sovereign_data_models import SubProblem, SolutionAttempt, generate_id
from sovereign_reliability import with_retry, with_error_handling, ErrorSeverity

# Adaptive MDAP Imports
try:
    from adaptive_mdap.integrations.subproblem_solver_integration import SubProblemSolverIntegration
    from adaptive_mdap.core.types import SubProblem as AdaptiveSubProblem
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Sub-Problem Solver
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# **ACTUAL INTEGRATION HELPER METHODS**: Sub-Problem Solver
def _trigger_subproblem_solver_alerts(operation, success, problem_id=None, error=None, metadata=None):
    """Trigger alerts for sub-problem solver operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.HIGH if operation == "solve" else AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"Sub-Problem Solver {operation} Failed",
            message=f"Sub-problem solver operation '{operation}' failed: {error}",
            severity=severity,
            source="SubProblemSolver",
            metadata=metadata or {"problem_id": problem_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger sub-problem solver alert: {e}")


def _extract_subproblem_solver_knowledge(operation, problem_id, approach, result):
    """Extract knowledge from sub-problem solver operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        from datetime import datetime
        artifact = KnowledgeArtifact(
            artifact_id=f"subproblem_solver_{operation}_{problem_id}",
            artifact_type="subproblem_solver_execution",
            source_component="SubProblemSolver",
            content={
                "operation": operation,
                "problem_id": problem_id,
                "approach": approach,
                "confidence": getattr(result, 'confidence_score', 0.0) if result else 0.0,
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract sub-problem solver knowledge: {e}")


def _track_subproblem_solver_performance(operation, success, duration_seconds, approach, confidence=0.0):
    """Track performance of sub-problem solver operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name=f"subproblem_solver_{approach}",
            component_name="SubProblemSolver",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "approach": approach,
                "confidence": confidence
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track sub-problem solver performance: {e}")


class SubProblemSolver:
    """Solves sub-problems using LLM-based solution generation."""

    def __init__(
        self, 
        openevolve_client=None, 
        enable_adaptive_allocation: bool = True,
        maker_config: Optional[Dict[str, Any]] = None,
        adaptive_config: Optional[Dict[str, Any]] = None,
        maker_preset: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize sub-problem solver.
        
        Args:
            openevolve_client: Client for LLM calls
            enable_adaptive_allocation: Whether to use adaptive tiers
            maker_config: Configuration for the MAKER engine
            adaptive_config: Configuration for adaptive components
            maker_preset: Name of a MAKER preset (FAST, BALANCED, ZERO_ERROR)
            config: Optional configuration dictionary (includes CAV-NLP settings)
        """
        self.config = config or {}
        self.openevolve_client = openevolve_client
        self.enable_adaptive_allocation = enable_adaptive_allocation and ADAPTIVE_AVAILABLE
        
        # Apply MAKER preset if provided
        self.maker_config = maker_config or {}
        if maker_preset:
            try:
                from openevolve_maker_integration import MAKER_PRESETS
                preset_cfg = MAKER_PRESETS.get(maker_preset.upper(), {})
                # Merge: config overrides preset
                self.maker_config = {**preset_cfg, **self.maker_config}
                logger.info(f"Applied MAKER preset: {maker_preset}")
            except ImportError:
                logger.warning("MAKER_PRESETS not available, skipping preset.")

        self.adaptive_config = adaptive_config or {}
        
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except ImportError:
                logger.warning("OpenEvolve client not available for sub-problem solver.")
        
        # Initialize adaptive integration with custom configs
        self.adaptive_integration = None
        if self.enable_adaptive_allocation:
            try:
                # Extract granular configs
                classifier_cfg = self.adaptive_config.get("classifier")
                allocator_cfg = self.adaptive_config.get("allocator")
                
                self.adaptive_integration = SubProblemSolverIntegration(
                    enable_adaptive=True,
                    classifier_config=classifier_cfg,
                    allocator_config=allocator_cfg
                )
                logger.info("Adaptive MDAP allocation enabled for SubProblemSolver")
            except Exception as e:
                logger.error(f"Failed to initialize Adaptive MDAP: {e}")
                self.enable_adaptive_allocation = False
        
        # CAV-NLP integration for sub-problem formalization
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for sub-problem formalization")

    @with_error_handling(fallback=lambda *args, **kwargs: SolutionAttempt(id=generate_id("solution"), sub_problem_id=args[1].id, approach="failed", solution_content="", team_id="error-fallback", confidence_score=0.0), severity=ErrorSeverity.HIGH)
    @with_retry(max_attempts=2, retry_on=(RuntimeError,))
    def solve(self, sub_problem: SubProblem) -> SolutionAttempt:
        """Generates a solution for a sub-problem using an LLM."""
        start_time = time.time()
        success = False
        approach = "unknown"

        logger.info(f"Solving sub-problem: {sub_problem.title}")

        try:
            # Try adaptive allocation if enabled
            if self.enable_adaptive_allocation and self.adaptive_integration:
                try:
                    solution = self._solve_adaptive(sub_problem)
                    # **ACTUAL INTEGRATION**: Extract knowledge and track performance
                    success = True
                    approach = f"adaptive-{solution.approach}"
                    duration = time.time() - start_time
                    _extract_subproblem_solver_knowledge("solve", sub_problem.id, approach, solution)
                    _track_subproblem_solver_performance("solve", True, duration, approach, solution.confidence_score)
                    return solution
                except Exception as e:
                    logger.warning(f"Adaptive solve failed, falling back to standard: {e}")
                    # Fall through to standard solve

            if not self.openevolve_client:
                raise RuntimeError("OpenEvolve client not available for sub-problem solver.")

            prompt = self._build_prompt(sub_problem)

            result = self.openevolve_client.evolve(
                content=prompt,
                evolution_mode="standard",
                content_type="code",
                max_iterations=1,
                temperature=0.5,
                max_tokens=1000,
            )

            if not result.success or not result.best_code:
                raise RuntimeError("LLM evolution failed to produce a solution.")

            solution = SolutionAttempt(
                id=generate_id("solution"),
                sub_problem_id=sub_problem.id,
                approach="llm-generated",
                solution_content=result.best_code,
                team_id="standard-llm",
                confidence_score=0.75,  # Initial confidence for LLM-generated solution
            )

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            success = True
            approach = "llm-generated"
            duration = time.time() - start_time
            _extract_subproblem_solver_knowledge("solve", sub_problem.id, approach, solution)
            _track_subproblem_solver_performance("solve", True, duration, approach, 0.75)
            return solution

        except Exception as e:
            duration = time.time() - start_time
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_subproblem_solver_alerts("solve", False, sub_problem.id, str(e))
            _track_subproblem_solver_performance("solve", False, duration, approach, 0.0)
            raise

    def _solve_adaptive(self, sub_problem: SubProblem) -> SolutionAttempt:
        """Solves sub-problem using adaptive resource allocation."""
        logger.info(f"Using adaptive allocation for sub-problem: {sub_problem.id}")
        
        # Convert to adaptive sub-problem type
        adaptive_sp = self._map_to_adaptive_type(sub_problem)
        
        # Solve using adaptive integration
        result = self.adaptive_integration.solve_adaptive(adaptive_sp)
        
        # Map back to SolutionAttempt
        return SolutionAttempt(
            id=generate_id("solution"),
            sub_problem_id=sub_problem.id,
            approach=f"adaptive-{result.strategy_used}",
            solution_content=str(result.solution),
            team_id=f"adaptive-team-{result.strategy_used}",
            confidence_score=0.8 if result.success else 0.4,
            status="solved" if result.success else "failed",
        )

    def _map_to_adaptive_type(self, sub_problem: SubProblem):
        """Maps sovereign_data_models.SubProblem to adaptive_mdap.core.types.SubProblem."""
        # Calculate depth (if not explicitly stored, estimate from parent_id depth)
        depth = 0
        if hasattr(sub_problem, 'metadata') and sub_problem.metadata:
            depth = sub_problem.metadata.get('depth', 0)
        
        return AdaptiveSubProblem(
            id=sub_problem.id,
            description=sub_problem.description,
            domain=sub_problem.type.value if hasattr(sub_problem.type, 'value') else str(sub_problem.type),
            depth=depth,
            dependencies=sub_problem.dependencies or [],
            metadata={
                "title": sub_problem.title,
                "original_complexity": sub_problem.complexity_score.overall_complexity if hasattr(sub_problem, 'complexity_score') else 0.5
            }
        )

    def formalize_subproblem_with_cav_nlp(
        self, 
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """
        Formalize a sub-problem using CAV-NLP.
        
        Args:
            sub_problem: Sub-problem to formalize
            
        Returns:
            Formalization result with constraints and properties
        """
        if not self.use_cav_nlp:
            return {
                'success': False,
                'error': 'CAV-NLP not available',
                'sub_problem_id': sub_problem.id
            }
        
        try:
            # Combine title and description for formalization
            problem_text = f"{sub_problem.title}\n{sub_problem.description}"
            
            # Use enhanced solver to formalize
            formalization = self.enhanced_solver.formalize_natural_language(
                problem_text,
                context={
                    'sub_problem_id': sub_problem.id,
                    'sub_problem_type': sub_problem.type.value if hasattr(sub_problem.type, 'value') else str(sub_problem.type),
                    'complexity': getattr(sub_problem, 'complexity_score', None),
                    'dependencies': sub_problem.dependencies or []
                }
            )
            
            result = {
                'success': formalization.get('success', False),
                'sub_problem_id': sub_problem.id,
                'constraints': formalization.get('constraints', []),
                'variables': formalization.get('variables', []),
                'properties': formalization.get('properties', {}),
                'z3_expr': formalization.get('z3_expression', None),
                'confidence': formalization.get('confidence', 0.0),
                'formalized_problem': formalization.get('formalized_problem', '')
            }
            
            logger.info(f"Formalized sub-problem {sub_problem.id} with CAV-NLP "
                       f"(confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"CAV-NLP formalization failed for {sub_problem.id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'sub_problem_id': sub_problem.id
            }
    
    def verify_subproblem_constraints(
        self,
        sub_problem: SubProblem
    ) -> Dict[str, Any]:
        """
        Verify sub-problem constraints using CAV-NLP and Z3.
        
        Args:
            sub_problem: Sub-problem to verify
            
        Returns:
            Verification result
        """
        if not self.use_cav_nlp:
            return {
                'verifiable': False,
                'message': 'CAV-NLP not available for verification'
            }
        
        try:
            # First formalize
            formalization = self.formalize_subproblem_with_cav_nlp(sub_problem)
            
            if not formalization['success']:
                return {
                    'verifiable': False,
                    'message': f"Formalization failed: {formalization.get('error')}",
                    'formalization': formalization
                }
            
            # Verify using math service
            if formalization.get('z3_expr'):
                verification = self.math_service.verify_expression(
                    formalization['z3_expr']
                )
                
                return {
                    'verifiable': True,
                    'valid': verification.get('valid', False),
                    'confidence': verification.get('confidence', 0.0),
                    'message': verification.get('message', 'Verification completed'),
                    'formalization': formalization,
                    'z3_result': verification.get('z3_result')
                }
            
            return {
                'verifiable': False,
                'message': 'No Z3 expression to verify',
                'formalization': formalization
            }
            
        except Exception as e:
            logger.error(f"Constraint verification failed for {sub_problem.id}: {e}")
            return {
                'verifiable': False,
                'message': f"Verification error: {e}"
            }
    
    def _build_prompt(self, sub_problem: SubProblem) -> str:
        """Builds the prompt for the LLM to solve the sub-problem."""
        return f"""You are an expert problem solver. Generate a solution for the following sub-problem.

SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}

TASK:
Provide a detailed solution to the sub-problem. The solution should be a combination of code and explanation, as appropriate.

SOLUTION:"""