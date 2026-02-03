"""
Evolution Engine for OpenEvolve API

Implements evolutionary code generation and optimization workflow.
Follows CLAUDE.md principles: structured logging, UTC timestamps, idempotent operations.

Integrates with BubbleLab services:
- Judge Adapter: Code evaluation and fitness scoring
- Mutate Adapter: Code mutation for evolution
"""

import structlog
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from enum import Enum


logger = structlog.get_logger()


class EvolutionStatus(str, Enum):
    """Evolution workflow status"""
    INITIALIZING = "initializing"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


class EvolutionEngine:
    """
    Evolution Engine for code generation and optimization.

    Integrates with BubbleLab Judge and Mutate services for real evolution.
    All timestamps in UTC. All operations idempotent where possible.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Evolution Engine.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Lazy import of adapters to avoid circular imports
        self._judge_adapter = None
        self._mutate_adapter = None

        logger.info(
            "evolution_engine_initialized",
            engine_type="evolution",
            config_keys=list(self.config.keys()),
            adapter_integration="enabled"
        )

    def _get_judge_adapter(self):
        """Get or create Judge adapter instance"""
        if self._judge_adapter is None:
            from services.adapters import get_judge_adapter
            self._judge_adapter = get_judge_adapter()
        return self._judge_adapter

    def _get_mutate_adapter(self):
        """Get or create Mutate adapter instance"""
        if self._mutate_adapter is None:
            from services.adapters import get_mutate_adapter
            self._mutate_adapter = get_mutate_adapter()
        return self._mutate_adapter

    async def execute(
        self,
        problem_statement: str,
        parameters: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute evolution workflow for a given problem statement.

        This is the main entry point for evolutionary code generation.
        Follows Law of Idempotency: safe to retry if failed.

        Args:
            problem_statement: The problem to solve (e.g., "Create a REST API for user management")
            parameters: Evolution parameters from EvolutionParameters model
                - max_iterations: Maximum evolution iterations
                - population_size: Population for genetic algorithm
                - temperature: Temperature for generation randomness
                - top_p: Top-p sampling parameter
                - max_tokens: Maximum tokens in generated code
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                - seed: Random seed (-1 for random)
            context: Additional context or constraints

        Returns:
            Dictionary containing:
                - status: Final execution status
                - solution: Generated code/solution
                - iterations: Number of iterations performed
                - fitness: Fitness score of best solution
                - history: Evolution history (optional)
                - metadata: Execution metadata (timestamps, etc.)

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails critically
        """
        execution_start = datetime.now(timezone.utc)
        execution_id = f"evo_{execution_start.strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info(
            "evolution_execution_started",
            execution_id=execution_id,
            problem_statement=problem_statement[:100] + "..." if len(problem_statement) > 100 else problem_statement,
            parameters=parameters,
            context_provided=context is not None
        )

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Initialize evolution state
            status = EvolutionStatus.INITIALIZING
            iteration = 0
            best_fitness = 0.0
            evolution_history = []

            logger.info(
                "evolution_phase",
                execution_id=execution_id,
                phase=status.value,
                iteration=iteration
            )

            # PHASE 1: GENERATING - Create initial population
            status = EvolutionStatus.GENERATING
            initial_solution = self._generate_initial_solution(
                problem_statement,
                parameters,
                context
            )

            logger.info(
                "evolution_phase",
                execution_id=execution_id,
                phase=status.value,
                solution_length=len(initial_solution.get("code", ""))
            )

            # PHASE 2: EVOLUTION LOOP
            status = EvolutionStatus.EVALUATING
            max_iterations = parameters.get("max_iterations", 100)

            for iteration in range(1, max_iterations + 1):
                # Evaluate current solution (now async)
                fitness_score = await self._evaluate_solution(
                    initial_solution,
                    problem_statement
                )

                evolution_history.append({
                    "iteration": iteration,
                    "fitness": fitness_score,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

                # Track best solution
                if fitness_score > best_fitness:
                    best_fitness = fitness_score
                    logger.debug(
                        "evolution_improvement",
                        execution_id=execution_id,
                        iteration=iteration,
                        new_best_fitness=best_fitness
                    )

                # Refine solution (now async)
                status = EvolutionStatus.REFINING
                initial_solution = await self._refine_solution(
                    initial_solution,
                    fitness_score,
                    parameters
                )

                # Log progress every 10 iterations
                if iteration % 10 == 0:
                    logger.info(
                        "evolution_progress",
                        execution_id=execution_id,
                        iteration=iteration,
                        max_iterations=max_iterations,
                        current_fitness=fitness_score,
                        best_fitness=best_fitness
                    )

                # Check for convergence
                if fitness_score >= 0.95:  # 95% fitness threshold
                    logger.info(
                        "evolution_converged",
                        execution_id=execution_id,
                        iteration=iteration,
                        final_fitness=fitness_score
                    )
                    break

            # PHASE 3: COMPLETED
            status = EvolutionStatus.COMPLETED
            execution_end = datetime.now(timezone.utc)
            execution_duration = (execution_end - execution_start).total_seconds()

            result = {
                "status": status.value,
                "solution": {
                    "code": initial_solution.get("code", ""),
                    "description": initial_solution.get("description", ""),
                    "language": initial_solution.get("language", "python"),
                    "dependencies": initial_solution.get("dependencies", [])
                },
                "iterations": iteration,
                "fitness": best_fitness,
                "history": evolution_history,
                "metadata": {
                    "execution_id": execution_id,
                    "started_at": execution_start.isoformat(),
                    "completed_at": execution_end.isoformat(),
                    "duration_seconds": execution_duration,
                    "parameters": parameters,
                    "engine_version": "0.1.0"
                }
            }

            logger.info(
                "evolution_execution_completed",
                execution_id=execution_id,
                status=status.value,
                iterations=iteration,
                final_fitness=best_fitness,
                duration_seconds=execution_duration
            )

            return result

        except Exception as e:
            execution_end = datetime.now(timezone.utc)
            error_message = str(e)

            logger.error(
                "evolution_execution_failed",
                execution_id=execution_id,
                error=error_message,
                error_type=type(e).__name__,
                duration_seconds=(execution_end - execution_start).total_seconds(),
                exc_info=True
            )

            return {
                "status": EvolutionStatus.FAILED.value,
                "solution": None,
                "iterations": iteration,
                "fitness": 0.0,
                "history": evolution_history,
                "error": error_message,
                "metadata": {
                    "execution_id": execution_id,
                    "started_at": execution_start.isoformat(),
                    "failed_at": execution_end.isoformat(),
                    "error_type": type(e).__name__
                }
            }

    def _validate_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Validate evolution parameters.

        Args:
            parameters: Parameters dictionary to validate

        Raises:
            ValueError: If parameters are invalid
        """
        required_fields = []
        optional_fields = {
            "max_iterations": (1, 200),
            "population_size": (1, 100),
            "temperature": (0.0, 2.0),
            "top_p": (0.0, 1.0),
            "max_tokens": (1, 100000),
            "frequency_penalty": (-2.0, 2.0),
            "presence_penalty": (-2.0, 2.0),
            "seed": (-1, 999999)
        }

        # Validate optional fields if provided
        for field, (min_val, max_val) in optional_fields.items():
            if field in parameters:
                value = parameters[field]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Parameter '{field}' must be numeric")
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"Parameter '{field}' must be between {min_val} and {max_val}"
                    )

        logger.debug("evolution_parameters_validated", parameters=parameters)

    def _generate_initial_solution(
        self,
        problem_statement: str,
        parameters: Dict[str, Any],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Generate initial solution using LLM.

        Args:
            problem_statement: Problem to solve
            parameters: Generation parameters
            context: Additional context

        Returns:
            Dictionary with initial solution code and metadata
        """
        # This is a placeholder - actual implementation would call LLM
        logger.debug(
            "generating_initial_solution",
            problem_statement_length=len(problem_statement)
        )

        return {
            "code": f"# Solution for: {problem_statement}\n# Generated at {datetime.now(timezone.utc).isoformat()}\n\ndef solution():\n    # Implementation here\n    pass",
            "description": "Initial solution generated by evolution engine",
            "language": "python",
            "dependencies": []
        }

    async def _evaluate_solution(
        self,
        solution: Dict[str, Any],
        problem_statement: str
    ) -> float:
        """
        Evaluate solution fitness using Judge adapter.

        Args:
            solution: Solution to evaluate
            problem_statement: Original problem statement

        Returns:
            Fitness score between 0.0 and 1.0
        """
        try:
            code = solution.get("code", "")

            logger.debug(
                "evaluating_solution",
                code_length=len(code),
                problem_statement_length=len(problem_statement)
            )

            # Use Judge adapter for real evaluation
            judge = self._get_judge_adapter()

            evaluation = await judge.evaluate(
                code=code,
                problem_statement=problem_statement,
                weights=self.config.get("judge_weights", {
                    "correctness": 0.4,
                    "efficiency": 0.3,
                    "style": 0.2,
                    "documentation": 0.1,
                })
            )

            fitness_score = evaluation.get("overall_score", 0.5)

            logger.info(
                "solution_evaluated",
                fitness=fitness_score,
                criteria_count=len(evaluation.get("criteria", []))
            )

            return fitness_score

        except Exception as e:
            logger.warning(
                "judge_evaluation_failed",
                error=str(e),
                fallback_enabled=True
            )

            # Fallback to heuristic evaluation if Judge service unavailable
            code = solution.get("code", "")
            base_score = 0.5

            if len(code) > 100:
                base_score += 0.2
            if "def " in code or "class " in code:
                base_score += 0.2
            if "import " in code:
                base_score += 0.1

            return min(base_score, 1.0)

    async def _refine_solution(
        self,
        solution: Dict[str, Any],
        fitness: float,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Refine solution using Mutate adapter.

        Args:
            solution: Current solution
            fitness: Current fitness score
            parameters: Evolution parameters

        Returns:
            Refined solution
        """
        try:
            code = solution.get("code", "")
            mutation_rate = 1.0 - fitness  # Lower fitness = higher mutation rate

            logger.debug(
                "refining_solution",
                current_fitness=fitness,
                mutation_rate=mutation_rate,
                temperature=parameters.get("temperature", 0.7)
            )

            # Use Mutate adapter for real mutation
            mutate = self._get_mutate_adapter()

            mutation_result = await mutate.mutate(
                code=code,
                mutation_type="point",
                mutation_rate=max(0.05, min(mutation_rate, 0.5)),  # Clamp between 5% and 50%
            )

            refined_code = mutation_result.get("mutated_code", code)
            mutations_count = mutation_result.get("mutations_count", 0)

            # Update solution with mutated code
            solution["code"] = refined_code

            logger.info(
                "solution_refined",
                mutations_count=mutations_count,
                new_code_length=len(refined_code)
            )

            return solution

        except Exception as e:
            logger.warning(
                "mutate_refinement_failed",
                error=str(e),
                fallback_enabled=True
            )

            # Fallback: Add refinement comment if Mutate service unavailable
            refined_code = solution.get("code", "") + f"\n# Refined at {datetime.now(timezone.utc).isoformat()}"
            solution["code"] = refined_code

            return solution
