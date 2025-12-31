"""
Sub-Problem Solver for Sovereign-Grade Problem Decomposition System
"""

import logging
from typing import Optional

from sovereign_data_models import SubProblem, SolutionAttempt, generate_id
from sovereign_reliability import with_retry, with_error_handling, ErrorSeverity

logger = logging.getLogger(__name__)

class SubProblemSolver:
    """Solves sub-problems using LLM-based solution generation."""

    def __init__(self, openevolve_client=None):
        self.openevolve_client = openevolve_client
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except ImportError:
                logger.warning("OpenEvolve client not available for sub-problem solver.")

    @with_error_handling(fallback=lambda *args, **kwargs: SolutionAttempt(id=generate_id("solution"), sub_problem_id=args[1].id, approach="failed", solution_content="", confidence_score=0.0), severity=ErrorSeverity.HIGH)
    @with_retry(max_attempts=2, retry_on=(RuntimeError,))
    def solve(self, sub_problem: SubProblem) -> SolutionAttempt:
        """Generates a solution for a sub-problem using an LLM."""
        logger.info(f"Solving sub-problem: {sub_problem.title}")

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
            confidence_score=0.75,  # Initial confidence for LLM-generated solution
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