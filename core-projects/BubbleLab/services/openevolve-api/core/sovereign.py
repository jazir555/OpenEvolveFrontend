"""
Sovereign Engine for OpenEvolve API

Implements sovereign decomposition workflow for breaking down complex problems.
Follows CLAUDE.md principles: structured logging, UTC timestamps, idempotent operations.

Integrates with BubbleLab services:
- LeanAide Adapter: Lean 4 theorem proving for formal verification
"""

import structlog
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum


logger = structlog.get_logger()


class SovereignStatus(str, Enum):
    """Sovereign decomposition status"""
    INITIALIZING = "initializing"
    DECOMPOSING = "decomposing"
    SOLVING = "solving"
    VERIFYING = "verifying"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubProblemStatus(str, Enum):
    """Sub-problem solving status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SOLVED = "solved"
    FAILED = "failed"


class SovereignEngine:
    """
    Sovereign Engine for problem decomposition and parallel solving.

    Integrates with BubbleLab LeanAide service for formal verification.
    All timestamps in UTC. Idempotent operations where possible.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Sovereign Engine.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Lazy import of adapters to avoid circular imports
        self._leanaide_adapter = None

        logger.info(
            "sovereign_engine_initialized",
            engine_type="sovereign",
            config_keys=list(self.config.keys()),
            adapter_integration="enabled"
        )

    def _get_leanaide_adapter(self):
        """Get or create LeanAide adapter instance"""
        if self._leanaide_adapter is None:
            from services.adapters import get_leanaide_adapter
            self._leanaide_adapter = get_leanaide_adapter()
        return self._leanaide_adapter

    async def execute(
        self,
        problem_statement: str,
        parameters: Dict[str, Any],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute sovereign decomposition workflow.

        Breaks down complex problems into smaller sub-problems, solves them in parallel,
        then synthesizes the results into a comprehensive solution.

        Args:
            problem_statement: The complex problem to solve
            parameters: Sovereign parameters from SovereignParameters model
                - decomposition_depth: Maximum depth of decomposition
                - parallel_subproblems: Number of sub-problems to solve in parallel
                - verification_strictness: "lenient", "standard", or "strict"
            context: Additional context or constraints

        Returns:
            Dictionary containing:
                - status: Final execution status
                - decomposition: Problem decomposition tree
                - sub_problems: List of sub-problems with solutions
                - final_solution: Synthesized final solution
                - verification_results: Results of solution verification
                - metadata: Execution metadata (timestamps, etc.)

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If execution fails critically
        """
        execution_start = datetime.now(timezone.utc)
        execution_id = f"sov_{execution_start.strftime('%Y%m%d_%H%M%S_%f')}"

        logger.info(
            "sovereign_execution_started",
            execution_id=execution_id,
            problem_statement=problem_statement[:100] + "..." if len(problem_statement) > 100 else problem_statement,
            parameters=parameters,
            context_provided=context is not None
        )

        try:
            # Validate parameters
            self._validate_parameters(parameters)

            # Initialize sovereign state
            status = SovereignStatus.INITIALIZING

            # Get parameters
            decomposition_depth = parameters.get("decomposition_depth", 3)
            parallel_subproblems = parameters.get("parallel_subproblems", 5)
            verification_strictness = parameters.get("verification_strictness", "standard")

            logger.info(
                "sovereign_phase",
                execution_id=execution_id,
                phase=status.value,
                decomposition_depth=decomposition_depth,
                parallel_subproblems=parallel_subproblems,
                verification_strictness=verification_strictness
            )

            # PHASE 2: DECOMPOSING - Break down the problem
            status = SovereignStatus.DECOMPOSING
            decomposition = self._decompose_problem(
                problem_statement,
                decomposition_depth
            )

            logger.info(
                "sovereign_phase",
                execution_id=execution_id,
                phase=status.value,
                total_subproblems=len(decomposition.get("sub_problems", []))
            )

            # PHASE 3: SOLVING - Solve sub-problems in parallel
            status = SovereignStatus.SOLVING
            sub_problems_solved = self._solve_sub_problems(
                decomposition.get("sub_problems", []),
                parallel_subproblems,
                context
            )

            # Count successful solves
            solved_count = len([sp for sp in sub_problems_solved if sp.get("status") == SubProblemStatus.SOLVED])
            failed_count = len([sp for sp in sub_problems_solved if sp.get("status") == SubProblemStatus.FAILED])

            logger.info(
                "sovereign_phase",
                execution_id=execution_id,
                phase=status.value,
                total_subproblems=len(sub_problems_solved),
                solved_count=solved_count,
                failed_count=failed_count
            )

            # PHASE 4: VERIFYING - Verify solutions
            status = SovereignStatus.VERIFYING
            verification_results = await self._verify_solutions(
                sub_problems_solved,
                verification_strictness
            )

            logger.info(
                "sovereign_phase",
                execution_id=execution_id,
                phase=status.value,
                verification_passed=verification_results["passed"],
                verification_failed=verification_results["failed"]
            )

            # PHASE 5: SYNTHESIZING - Combine solutions
            status = SovereignStatus.SYNTHESIZING
            final_solution = self._synthesize_solution(
                problem_statement,
                sub_problems_solved,
                verification_results
            )

            logger.info(
                "sovereign_phase",
                execution_id=execution_id,
                phase=status.value,
                final_solution_length=len(final_solution.get("content", ""))
            )

            # PHASE 6: COMPLETED
            status = SovereignStatus.COMPLETED
            execution_end = datetime.now(timezone.utc)
            execution_duration = (execution_end - execution_start).total_seconds()

            result = {
                "status": status.value,
                "decomposition": decomposition,
                "sub_problems": sub_problems_solved,
                "final_solution": final_solution,
                "verification_results": verification_results,
                "summary": {
                    "total_subproblems": len(sub_problems_solved),
                    "solved_count": solved_count,
                    "failed_count": failed_count,
                    "verification_passed": verification_results["passed"],
                    "verification_failed": verification_results["failed"],
                    "overall_success_rate": solved_count / len(sub_problems_solved) if sub_problems_solved else 0.0
                },
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
                "sovereign_execution_completed",
                execution_id=execution_id,
                status=status.value,
                total_subproblems=len(sub_problems_solved),
                solved_count=solved_count,
                success_rate=result["summary"]["overall_success_rate"],
                duration_seconds=execution_duration
            )

            return result

        except Exception as e:
            execution_end = datetime.now(timezone.utc)
            error_message = str(e)

            logger.error(
                "sovereign_execution_failed",
                execution_id=execution_id,
                error=error_message,
                error_type=type(e).__name__,
                duration_seconds=(execution_end - execution_start).total_seconds(),
                exc_info=True
            )

            return {
                "status": SovereignStatus.FAILED.value,
                "decomposition": decomposition if 'decomposition' in locals() else {},
                "sub_problems": sub_problems_solved if 'sub_problems_solved' in locals() else [],
                "final_solution": None,
                "verification_results": {},
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
        Validate sovereign parameters.

        Args:
            parameters: Parameters dictionary to validate

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate decomposition_depth
        if "decomposition_depth" in parameters:
            depth = parameters["decomposition_depth"]
            if not isinstance(depth, int) or not (1 <= depth <= 10):
                raise ValueError("Parameter 'decomposition_depth' must be an integer between 1 and 10")

        # Validate parallel_subproblems
        if "parallel_subproblems" in parameters:
            parallel = parameters["parallel_subproblems"]
            if not isinstance(parallel, int) or not (1 <= parallel <= 20):
                raise ValueError("Parameter 'parallel_subproblems' must be an integer between 1 and 20")

        # Validate verification_strictness
        if "verification_strictness" in parameters:
            strictness = parameters["verification_strictness"]
            valid_values = ["lenient", "standard", "strict"]
            if strictness not in valid_values:
                raise ValueError(f"Parameter 'verification_strictness' must be one of {valid_values}")

        logger.debug("sovereign_parameters_validated", parameters=parameters)

    def _decompose_problem(
        self,
        problem_statement: str,
        max_depth: int
    ) -> Dict[str, Any]:
        """
        Decompose problem into smaller sub-problems.

        Args:
            problem_statement: Original problem statement
            max_depth: Maximum decomposition depth

        Returns:
            Decomposition tree with sub-problems
        """
        logger.debug(
            "decomposing_problem",
            problem_length=len(problem_statement),
            max_depth=max_depth
        )

        # Placeholder - actual implementation would use LLM to decompose
        # This simulates decomposition logic
        sub_problems = []

        # Generate sample sub-problems based on problem complexity
        problem_complexity = len(problem_statement) // 100  # Simple heuristic
        num_subproblems = min(max(problem_complexity, 3), 10)

        for i in range(num_subproblems):
            sub_problems.append({
                "id": f"subproblem_{i + 1}",
                "title": f"Sub-problem {i + 1}",
                "description": f"Decomposed aspect {i + 1} of the main problem",
                "priority": "high" if i < 2 else "medium",
                "dependencies": [] if i == 0 else [f"subproblem_{i}"],
                "estimated_complexity": "medium",
                "depth": 1
            })

        decomposition = {
            "original_problem": problem_statement,
            "decomposition_strategy": "hierarchical",
            "max_depth": max_depth,
            "sub_problems": sub_problems,
            "total_complexity": sum(1 for sp in sub_problems if sp.get("priority") == "high")
        }

        logger.info(
            "problem_decomposed",
            subproblems_count=len(sub_problems),
            decomposition_strategy=decomposition["decomposition_strategy"]
        )

        return decomposition

    def _solve_sub_problems(
        self,
        sub_problems: List[Dict[str, Any]],
        parallel_count: int,
        context: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Solve sub-problems in parallel batches.

        Args:
            sub_problems: List of sub-problems to solve
            parallel_count: Number of sub-problems to solve in parallel
            context: Additional context

        Returns:
            List of sub-problems with solutions
        """
        logger.debug(
            "solving_subproblems",
            total_subproblems=len(sub_problems),
            parallel_count=parallel_count
        )

        solved_problems = []

        # Process in batches
        for i in range(0, len(sub_problems), parallel_count):
            batch = sub_problems[i:i + parallel_count]

            # Solve batch in parallel (placeholder - actual implementation would use asyncio/multiprocessing)
            for sub_problem in batch:
                solution = self._solve_single_subproblem(sub_problem, context)
                solved_problems.append(solution)

                logger.debug(
                    "subproblem_solved",
                    subproblem_id=sub_problem.get("id"),
                    status=solution.get("status")
                )

        return solved_problems

    def _solve_single_subproblem(
        self,
        sub_problem: Dict[str, Any],
        context: Optional[str]
    ) -> Dict[str, Any]:
        """
        Solve a single sub-problem.

        Args:
            sub_problem: Sub-problem to solve
            context: Additional context

        Returns:
            Sub-problem with solution
        """
        # Placeholder - actual implementation would use LLM
        import random

        # Simulate solving with some chance of failure
        success = random.random() > 0.1  # 90% success rate

        if success:
            return {
                **sub_problem,
                "status": SubProblemStatus.SOLVED,
                "solution": {
                    "content": f"Solution for {sub_problem.get('title')}",
                    "approach": "algorithmic",
                    "code": f"# Implementation for {sub_problem.get('id')}\ndef solution():\n    pass",
                    "confidence": random.uniform(0.7, 0.95)
                },
                "solved_at": datetime.now(timezone.utc).isoformat()
            }
        else:
            return {
                **sub_problem,
                "status": SubProblemStatus.FAILED,
                "error": "Unable to solve sub-problem - insufficient context",
                "failed_at": datetime.now(timezone.utc).isoformat()
            }

    async def _verify_solutions(
        self,
        sub_problems: List[Dict[str, Any]],
        strictness: str
    ) -> Dict[str, Any]:
        """
        Verify solutions against requirements.

        Integrates with LeanAide adapter for formal proof verification when available.
        Falls back to confidence-based verification for non-formal solutions.

        Args:
            sub_problems: Solved sub-problems
            strictness: Verification strictness level

        Returns:
            Verification results with formal verification status
        """
        logger.debug(
            "verifying_solutions",
            total_solutions=len(sub_problems),
            strictness=strictness,
            formal_verification_enabled=True
        )

        # Define strictness thresholds
        thresholds = {
            "lenient": 0.6,
            "standard": 0.75,
            "strict": 0.9
        }

        min_confidence = thresholds[strictness]

        verification_results = {
            "strictness": strictness,
            "min_confidence_threshold": min_confidence,
            "verified_solutions": [],
            "failed_solutions": [],
            "passed": 0,
            "failed": 0,
            "formal_proofs_verified": 0,
            "heuristic_verifications": 0
        }

        try:
            # Try to use LeanAide adapter for formal verification
            leanaide = self._get_leanaide_adapter()

            for sub_problem in sub_problems:
                if sub_problem.get("status") != SubProblemStatus.SOLVED:
                    continue

                solution = sub_problem.get("solution", {})
                subproblem_id = sub_problem.get("id")
                confidence = solution.get("confidence", 0.0)

                # Check if solution has formal proof
                proof = solution.get("proof")
                proposition = sub_problem.get("description", "")

                if proof and strictness in ["standard", "strict"]:
                    # Use LeanAide for formal verification
                    try:
                        logger.debug(
                            "attempting_formal_verification",
                            subproblem_id=subproblem_id,
                            proof_length=len(proof)
                        )

                        verification = await leanaide.verify_proof(
                            proof=proof,
                            proposition=proposition
                        )

                        if verification.get("is_valid"):
                            verification_results["verified_solutions"].append({
                                "subproblem_id": subproblem_id,
                                "confidence": confidence,
                                "passed": True,
                                "verification_method": "formal",
                                "proof_valid": True,
                                "tactics_used": verification.get("tactics", [])
                            })
                            verification_results["passed"] += 1
                            verification_results["formal_proofs_verified"] += 1
                        else:
                            # Formal proof failed
                            if strictness == "strict":
                                # Strict mode: formal proof required
                                verification_results["failed_solutions"].append({
                                    "subproblem_id": subproblem_id,
                                    "confidence": confidence,
                                    "passed": False,
                                    "verification_method": "formal",
                                    "reason": f"Formal proof verification failed: {verification.get('error', 'Unknown error')}"
                                })
                                verification_results["failed"] += 1
                            else:
                                # Standard mode: fall back to confidence check
                                if confidence >= min_confidence:
                                    verification_results["verified_solutions"].append({
                                        "subproblem_id": subproblem_id,
                                        "confidence": confidence,
                                        "passed": True,
                                        "verification_method": "heuristic",
                                        "proof_valid": False,
                                        "reason": "Formal proof failed but confidence sufficient"
                                    })
                                    verification_results["passed"] += 1
                                    verification_results["heuristic_verifications"] += 1
                                else:
                                    verification_results["failed_solutions"].append({
                                        "subproblem_id": subproblem_id,
                                        "confidence": confidence,
                                        "passed": False,
                                        "verification_method": "heuristic",
                                        "reason": f"Formal proof failed and confidence {confidence:.2f} below threshold {min_confidence}"
                                    })
                                    verification_results["failed"] += 1

                    except Exception as e:
                        logger.warning(
                            "leanaide_verification_error",
                            subproblem_id=subproblem_id,
                            error=str(e),
                            fallback_to_heuristic=True
                        )

                        # Fallback to confidence-based verification
                        if confidence >= min_confidence:
                            verification_results["verified_solutions"].append({
                                "subproblem_id": subproblem_id,
                                "confidence": confidence,
                                "passed": True,
                                "verification_method": "heuristic",
                                "reason": "LeanAide unavailable, using confidence"
                            })
                            verification_results["passed"] += 1
                            verification_results["heuristic_verifications"] += 1
                        else:
                            verification_results["failed_solutions"].append({
                                "subproblem_id": subproblem_id,
                                "confidence": confidence,
                                "passed": False,
                                "verification_method": "heuristic",
                                "reason": f"Confidence {confidence:.2f} below threshold {min_confidence}"
                            })
                            verification_results["failed"] += 1

                else:
                    # No formal proof provided, use confidence-based verification
                    verification_results["heuristic_verifications"] += 1

                    if confidence >= min_confidence:
                        verification_results["verified_solutions"].append({
                            "subproblem_id": subproblem_id,
                            "confidence": confidence,
                            "passed": True,
                            "verification_method": "heuristic",
                            "reason": "No formal proof provided, confidence sufficient"
                        })
                        verification_results["passed"] += 1
                    else:
                        verification_results["failed_solutions"].append({
                            "subproblem_id": subproblem_id,
                            "confidence": confidence,
                            "passed": False,
                            "verification_method": "heuristic",
                            "reason": f"Confidence {confidence:.2f} below threshold {min_confidence}"
                        })
                        verification_results["failed"] += 1

        except Exception as e:
            logger.error(
                "leanaide_adapter_unavailable",
                error=str(e),
                error_type=type(e).__name__,
                fallback_to_confidence_verification=True
            )

            # Complete fallback to confidence-based verification
            verification_results["heuristic_verifications"] = len([sp for sp in sub_problems if sp.get("status") == SubProblemStatus.SOLVED])

            for sub_problem in sub_problems:
                if sub_problem.get("status") != SubProblemStatus.SOLVED:
                    continue

                solution = sub_problem.get("solution", {})
                subproblem_id = sub_problem.get("id")
                confidence = solution.get("confidence", 0.0)

                if confidence >= min_confidence:
                    verification_results["verified_solutions"].append({
                        "subproblem_id": subproblem_id,
                        "confidence": confidence,
                        "passed": True,
                        "verification_method": "heuristic",
                        "reason": "LeanAide adapter unavailable, using confidence only"
                    })
                    verification_results["passed"] += 1
                else:
                    verification_results["failed_solutions"].append({
                        "subproblem_id": subproblem_id,
                        "confidence": confidence,
                        "passed": False,
                        "verification_method": "heuristic",
                        "reason": f"Confidence {confidence:.2f} below threshold {min_confidence}"
                    })
                    verification_results["failed"] += 1

        logger.info(
            "verification_completed",
            passed=verification_results["passed"],
            failed=verification_results["failed"],
            strictness=strictness,
            formal_proofs=verification_results["formal_proofs_verified"],
            heuristic_verifications=verification_results["heuristic_verifications"]
        )

        return verification_results

    def _synthesize_solution(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
        verification_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize final solution from sub-problem solutions.

        Args:
            problem_statement: Original problem statement
            sub_problems: Solved sub-problems
            verification_results: Verification results

        Returns:
            Synthesized final solution
        """
        logger.debug(
            "synthesizing_solution",
            problem_length=len(problem_statement),
            subproblem_count=len(sub_problems)
        )

        # Collect verified solutions
        verified_ids = {v["subproblem_id"] for v in verification_results.get("verified_solutions", [])}
        verified_solutions = [
            sp for sp in sub_problems
            if sp.get("id") in verified_ids and sp.get("status") == SubProblemStatus.SOLVED
        ]

        # Build synthesis
        solution_parts = []
        for sp in sorted(verified_solutions, key=lambda x: x.get("id", "")):
            solution = sp.get("solution", {})
            solution_parts.append(f"## {sp.get('title')}\n{solution.get('content', '')}")

        synthesis_content = f"# Solution for: {problem_statement}\n\n" + "\n\n".join(solution_parts)

        final_solution = {
            "content": synthesis_content,
            "approach": "compositional",
            "sub_solutions_count": len(verified_solutions),
            "integrity_score": len(verified_solutions) / len(sub_problems) if sub_problems else 0.0,
            "synthesis_timestamp": datetime.now(timezone.utc).isoformat()
        }

        logger.info(
            "solution_synthesized",
            final_solution_length=len(synthesis_content),
            integrity_score=final_solution["integrity_score"]
        )

        return final_solution
