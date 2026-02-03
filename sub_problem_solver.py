"""
Sub-Problem Solver for Sovereign-Grade Problem Decomposition System
"""

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

logger = logging.getLogger(__name__)

class SubProblemSolver:
    """Solves sub-problems using LLM-based solution generation."""

    def __init__(
        self, 
        openevolve_client=None, 
        enable_adaptive_allocation: bool = True,
        maker_config: Optional[Dict[str, Any]] = None,
        adaptive_config: Optional[Dict[str, Any]] = None,
        maker_preset: Optional[str] = None
    ):
        """
        Initialize sub-problem solver.
        
        Args:
            openevolve_client: Client for LLM calls
            enable_adaptive_allocation: Whether to use adaptive tiers
            maker_config: Configuration for the MAKER engine
            adaptive_config: Configuration for adaptive components
            maker_preset: Name of a MAKER preset (FAST, BALANCED, ZERO_ERROR)
        """
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

    @with_error_handling(fallback=lambda *args, **kwargs: SolutionAttempt(id=generate_id("solution"), sub_problem_id=args[1].id, approach="failed", solution_content="", team_id="error-fallback", confidence_score=0.0), severity=ErrorSeverity.HIGH)
    @with_retry(max_attempts=2, retry_on=(RuntimeError,))
    def solve(self, sub_problem: SubProblem) -> SolutionAttempt:
        """Generates a solution for a sub-problem using an LLM."""
        logger.info(f"Solving sub-problem: {sub_problem.title}")

        # Try adaptive allocation if enabled
        if self.enable_adaptive_allocation and self.adaptive_integration:
            try:
                return self._solve_adaptive(sub_problem)
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

        return SolutionAttempt(
            id=generate_id("solution"),
            sub_problem_id=sub_problem.id,
            approach="llm-generated",
            solution_content=result.best_code,
            team_id="standard-llm",
            confidence_score=0.75,  # Initial confidence for LLM-generated solution
        )

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

    def _build_prompt(self, sub_problem: SubProblem) -> str:
        """Builds the prompt for the LLM to solve the sub-problem."""
        return f"""You are an expert problem solver. Generate a solution for the following sub-problem.

SUB-PROBLEM:
Title: {sub_problem.title}
Description: {sub_problem.description}

TASK:
Provide a detailed solution to the sub-problem. The solution should be a combination of code and explanation, as appropriate.

SOLUTION:"""