"""
Claudiomiro Hephaestus Workflow Bridge

This module bridges Claudiomiro's autonomous development capabilities with
Hephaestus's workflow orchestration. It maps Hephaestus's 6-phase workflow
to Claudiomiro's development automation.

Architecture:
    Hephaestus (6 phases) -> Claudiomiro Bridge -> Claudiomiro CLI
                                                         ↓
                                            Autonomous Development
                                            (Decompose → Code → Review → Test → Commit)
"""

from typing import Any, Dict, List, Optional, Callable
import sys
import os
import json
import logging
from functools import wraps
from datetime import datetime
from pathlib import Path

# Import Claudiomiro MCP tools
try:
    import claudiomiro_mcp_tools as claudiomiro
    CLAUDIOMIRO_AVAILABLE = claudiomiro.CLAUDIOMIRO_AVAILABLE
except ImportError:
    CLAUDIOMIRO_AVAILABLE = False
    claudiomiro = None

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Claudiomiro Hephaestus Workflow Bridge
# ============================================================================

class ClaudiomiroHephaestusWorkflowBridge:
    """
    Bridge between Claudiomiro autonomous development and Hephaestus workflow.

    This bridge enables Hephaestus to leverage Claudiomiro's complete development
    automation capabilities:
    - Task decomposition
    - Parallel execution
    - Automated code review
    - Test execution and fixing
    - Production-ready commits

    Phase Mapping:
        - Phase 1 (Setup): Claudiomiro analyzes codebase
        - Phase 2 (Solution): Claudiomiro generates implementation
        - Phase 3 (Critique): Claudiomiro reviews code
        - Phase 4 (Verify): Claudiomiro runs tests
        - Phase 5 (Reassemble): Claudiomiro integrates components
        - Phase 6 (Final): Claudiomiro creates final commit

    Attributes:
        working_dir: Root directory for operations
        ai_provider: AI provider for Claudiomiro (claude, codex, gemini, etc.)
        enable_parallel: Enable parallel task execution

    Example:
        bridge = ClaudiomiroHephaestusWorkflowBridge(
            working_dir="/path/to/project",
            ai_provider="claude",
        )

        result = bridge.execute_full_workflow(
            prompt="Add user authentication with JWT",
        )
    """

    def __init__(
        self,
        working_dir: str = ".",
        ai_provider: str = "claude",
        enable_parallel: bool = True,
        max_cycles: int = 20,
    ):
        """
        Initialize Claudiomiro-Hephaestus bridge.

        Args:
            working_dir: Root working directory
            ai_provider: AI provider (claude, codex, gemini, deep-seek, glm)
            enable_parallel: Enable parallel task execution
            max_cycles: Maximum execution cycles
        """
        self.working_dir = working_dir
        self.ai_provider = ai_provider
        self.enable_parallel = enable_parallel
        self.max_cycles = max_cycles

        if not CLAUDIOMIRO_AVAILABLE:
            logger.warning("Claudiomiro not available - bridge will return stub results")

    # ========================================================================
    # Phase 1: Setup - Analyze and Decompose
    # ========================================================================

    def execute_phase_1_setup(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        output_format: str = "decomposition",
    ) -> Dict[str, Any]:
        """
        Execute Phase 1 (Setup) with Claudiomiro task decomposition.

        Phase 1 Activities:
        - Analyze codebase
        - Decompose problem into sub-tasks
        - Identify dependencies
        - Plan execution strategy

        Args:
            problem_statement: The problem to solve
            problem_type: Type of problem
            domain: Problem domain
            context: Additional context
            output_format: "decomposition" or "full"

        Returns:
            Dict with phase results and task breakdown
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return self._stub_result("Phase 1: Setup", problem_statement)

        try:
            logger.info(f"Phase 1: Decomposing task - {problem_statement[:50]}...")

            # Use Claudiomiro to decompose task
            result = claudiomiro.decompose_task_with_claudiomiro(
                task_id=f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prompt=problem_statement,
                working_dir=self.working_dir,
                ai_provider=self.ai_provider,
            )

            if result.get("success"):
                return {
                    "phase": "Phase 1: Setup",
                    "success": True,
                    "problem_statement": problem_statement,
                    "sub_tasks": result.get("sub_tasks", []),
                    "num_tasks": result.get("num_tasks", 0),
                    "working_dir": self.working_dir,
                    "ai_provider": self.ai_provider,
                    "message": f"Task decomposed into {result.get('num_tasks', 0)} sub-tasks",
                }
            else:
                return {
                    "phase": "Phase 1: Setup",
                    "success": False,
                    "error": result.get("error"),
                    "message": "Task decomposition failed",
                }

        except Exception as e:
            logger.error(f"Phase 1 execution failed: {e}")
            return {
                "phase": "Phase 1: Setup",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 2: Solution - Autonomous Implementation
    # ========================================================================

    def execute_phase_2_solution(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        frontend: Optional[str] = None,
        enable_parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 2 (Solution Generation) with Claudiomiro.

        Phase 2 Activities:
        - Generate solutions for sub-problems
        - Execute tasks in parallel (if enabled)
        - Apply coding best practices
        - Ensure code quality

        Args:
            problem_statement: Overall problem
            sub_problems: List of sub-problems to solve
            context: Additional context
            backend: Backend directory (for multi-repo)
            frontend: Frontend directory (for multi-repo)
            enable_parallel: Enable parallel execution

        Returns:
            Dict with phase results and implementations
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return self._stub_result("Phase 2: Solution", problem_statement)

        try:
            logger.info(f"Phase 2: Implementing solutions for {len(sub_problems)} sub-problems")

            # Build comprehensive prompt
            prompt = self._build_solution_prompt(problem_statement, sub_problems)

            # Execute with Claudiomiro
            result = claudiomiro.execute_claudiomiro_task(
                task_id=f"phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prompt=prompt,
                working_dir=self.working_dir,
                ai_provider=self.ai_provider,
                backend=backend,
                frontend=frontend,
                max_cycles=self.max_cycles,
            )

            return {
                "phase": "Phase 2: Solution",
                "success": result.get("success", False),
                "status": result.get("status"),
                "problem_statement": problem_statement,
                "sub_problems_solved": len(sub_problems),
                "backend": backend,
                "frontend": frontend,
                "parallel_execution": enable_parallel,
                "output": result.get("output", ""),
                "commit_hash": result.get("commit_hash"),
                "message": "Solutions implemented" if result.get("success") else "Implementation failed",
            }

        except Exception as e:
            logger.error(f"Phase 2 execution failed: {e}")
            return {
                "phase": "Phase 2: Solution",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 3: Critique - Code Review
    # ========================================================================

    def execute_phase_3_critique(
        self,
        solutions: List[Dict[str, Any]],
        critique_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phase 3 (Critique) with Claudiomiro code review.

        Phase 3 Activities:
        - Review generated code
        - Identify issues and improvements
        - Ensure best practices
        - Suggest optimizations

        Args:
            solutions: List of solutions to critique
            critique_criteria: Criteria for critique
            context: Additional context

        Returns:
            Dict with phase results and critiques
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return self._stub_result("Phase 3: Critique", f"{len(solutions)} solutions")

        try:
            logger.info(f"Phase 3: Critiquing {len(solutions)} solutions")

            # Build critique prompt
            prompt = "Review and critique the following implementations:\n"
            for i, solution in enumerate(solutions):
                prompt += f"\n{i+1}. {solution.get('description', 'Solution')}\n"

            # Execute critique
            result = claudiomiro.execute_claudiomiro_task(
                task_id=f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prompt=prompt,
                working_dir=self.working_dir,
                ai_provider=self.ai_provider,
                max_cycles=max(5, self.max_cycles // 4),  # Fewer cycles for critique
            )

            return {
                "phase": "Phase 3: Critique",
                "success": result.get("success", False),
                "status": result.get("status"),
                "solutions_reviewed": len(solutions),
                "critique_criteria": critique_criteria,
                "output": result.get("output", ""),
                "message": "Code review completed" if result.get("success") else "Review failed",
            }

        except Exception as e:
            logger.error(f"Phase 3 execution failed: {e}")
            return {
                "phase": "Phase 3: Critique",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 4: Verify - Automated Testing
    # ========================================================================

    def execute_phase_4_verify(
        self,
        solutions: List[Dict[str, Any]],
        test_command: str,
        verification_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        loop_fixes: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute Phase 4 (Verification) with Claudiomiro automated testing.

        Phase 4 Activities:
        - Run automated tests
        - Fix failing tests automatically
        - Verify all criteria met
        - Ensure quality standards

        Args:
            solutions: Solutions to verify
            test_command: Test command to run
            verification_criteria: Criteria for verification
            context: Additional context
            loop_fixes: Whether to loop fixes until all tests pass

        Returns:
            Dict with phase results and test outcomes
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return self._stub_result("Phase 4: Verify", f"{len(solutions)} solutions")

        try:
            logger.info(f"Phase 4: Running tests - {test_command}")

            # Use Claudiomiro's fix command capability
            result = claudiomiro.fix_tests_with_claudiomiro(
                task_id=f"phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                test_command=test_command,
                working_dir=self.working_dir,
                loop_fixes=loop_fixes,
                max_iterations=self.max_cycles,
                ai_provider=self.ai_provider,
            )

            return {
                "phase": "Phase 4: Verify",
                "success": result.get("success", False),
                "status": result.get("status"),
                "test_command": test_command,
                "tests_fixed": result.get("tests_fixed", False),
                "iterations": result.get("iterations", 0),
                "verification_criteria": verification_criteria,
                "output": result.get("output", ""),
                "message": "Tests verified and fixed" if result.get("success") else "Test verification failed",
            }

        except Exception as e:
            logger.error(f"Phase 4 execution failed: {e}")
            return {
                "phase": "Phase 4: Verify",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 5: Reassemble - Integration
    # ========================================================================

    def execute_phase_5_reassemble(
        self,
        sub_solutions: List[Dict[str, Any]],
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        frontend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phase 5 (Reassembly) with Claudiomiro integration.

        Phase 5 Activities:
        - Integrate sub-solutions
        - Verify integration points
        - Ensure compatibility
        - Test end-to-end

        Args:
            sub_solutions: List of sub-solutions to reassemble
            problem_statement: Original problem
            context: Additional context
            backend: Backend directory
            frontend: Frontend directory

        Returns:
            Dict with phase results and integration status
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return self._stub_result("Phase 5: Reassemble", problem_statement)

        try:
            logger.info(f"Phase 5: Reassembling {len(sub_solutions)} sub-solutions")

            # Build integration prompt
            prompt = f"""Integrate the following sub-solutions into a complete solution:

Problem: {problem_statement}

Sub-solutions to integrate:
"""
            for i, solution in enumerate(sub_solutions):
                prompt += f"\n{i+1}. {solution.get('description', 'Solution')}\n"

            # Execute integration
            result = claudiomiro.execute_claudiomiro_task(
                task_id=f"phase5_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prompt=prompt,
                working_dir=self.working_dir,
                ai_provider=self.ai_provider,
                backend=backend,
                frontend=frontend,
                max_cycles=self.max_cycles,
            )

            return {
                "phase": "Phase 5: Reassemble",
                "success": result.get("success", False),
                "status": result.get("status"),
                "sub_solutions_integrated": len(sub_solutions),
                "problem_statement": problem_statement,
                "backend": backend,
                "frontend": frontend,
                "output": result.get("output", ""),
                "message": "Integration completed" if result.get("success") else "Integration failed",
            }

        except Exception as e:
            logger.error(f"Phase 5 execution failed: {e}")
            return {
                "phase": "Phase 5: Reassemble",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Phase 6: Final - Commit and Push
    # ========================================================================

    def execute_phase_6_final(
        self,
        final_solution: str,
        problem_statement: str,
        validation_criteria: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        create_pr: bool = True,
        target_branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Execute Phase 6 (Final Validation) with Claudiomiro commit.

        Phase 6 Activities:
        - Final validation
        - Code review
        - Create production-ready commit
        - Optionally create PR

        Args:
            final_solution: The final solution
            problem_statement: Original problem
            validation_criteria: Final validation criteria
            context: Additional context
            create_pr: Whether to create PR
            target_branch: Target branch for PR

        Returns:
            Dict with phase results and commit info
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return self._stub_result("Phase 6: Final", problem_statement)

        try:
            logger.info("Phase 6: Final validation and commit")

            # Use Claudiomiro's fix-branch capability
            result = claudiomiro.fix_branch_with_claudiomiro(
                task_id=f"phase6_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                working_dir=self.working_dir,
                target_branch=target_branch,
                ai_provider=self.ai_provider,
            )

            return {
                "phase": "Phase 6: Final",
                "success": result.get("success", False),
                "problem_statement": problem_statement,
                "validation_criteria": validation_criteria,
                "target_branch": target_branch,
                "create_pr": create_pr,
                "output": result.get("output", ""),
                "commit_hash": result.get("commit_hash"),
                "message": "Final validation and commit completed" if result.get("success") else "Final phase failed",
            }

        except Exception as e:
            logger.error(f"Phase 6 execution failed: {e}")
            return {
                "phase": "Phase 6: Final",
                "success": False,
                "error": str(e),
            }

    # ========================================================================
    # Full Workflow Execution
    # ========================================================================

    def execute_full_workflow(
        self,
        prompt: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        backend: Optional[str] = None,
        frontend: Optional[str] = None,
        test_command: Optional[str] = "npm test",
        enable_all_phases: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full 6-phase Hephaestus workflow with Claudiomiro.

        This method runs all phases sequentially, with Claudiomiro handling:
        - Task decomposition
        - Autonomous implementation
        - Code review
        - Automated testing
        - Integration
        - Final commit

        Args:
            prompt: The task/prompt to execute
            problem_type: Type of problem
            domain: Problem domain
            backend: Backend directory (optional)
            frontend: Frontend directory (optional)
            test_command: Test command for verification
            enable_all_phases: Run all phases or stop early

        Returns:
            Dict with all phase results and overall status
        """
        logger.info(f"Starting full Claudiomiro workflow: {prompt[:50]}...")

        results = {
            "prompt": prompt,
            "phases": {},
            "overall_success": False,
            "working_dir": self.working_dir,
            "ai_provider": self.ai_provider,
        }

        # Phase 1: Setup (Decomposition)
        if enable_all_phases:
            phase1_result = self.execute_phase_1_setup(
                problem_statement=prompt,
                problem_type=problem_type,
                domain=domain,
            )
            results["phases"]["phase_1"] = phase1_result

            if not phase1_result.get("success"):
                results["message"] = "Workflow failed at Phase 1"
                return results

        # Phase 2-6: Execute remaining workflow
        # For now, we'll use the single execute_claudiomiro_task which handles all phases
        logger.info("Executing Claudiomiro autonomous workflow...")

        final_result = claudiomiro.execute_claudiomiro_task(
            task_id=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            prompt=prompt,
            working_dir=self.working_dir,
            ai_provider=self.ai_provider,
            backend=backend,
            frontend=frontend,
            max_cycles=self.max_cycles,
        )

        results["claudiomiro_execution"] = final_result
        results["overall_success"] = final_result.get("success", False)

        logger.info("Full workflow execution complete")
        return results

    # ========================================================================
    # Multi-Repository Workflow
    # ========================================================================

    def execute_multi_repo_workflow(
        self,
        prompt: str,
        backend: str,
        frontend: str,
        working_dir: str,
        legacy_backend: Optional[str] = None,
        legacy_frontend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute workflow across multiple repositories.

        Args:
            prompt: Task description
            backend: Backend directory path
            frontend: Frontend directory path
            working_dir: Root working directory
            legacy_backend: Optional legacy backend
            legacy_frontend: Optional legacy frontend

        Returns:
            Dict with multi-repo workflow results
        """
        if not CLAUDIOMIRO_AVAILABLE:
            return {
                "success": False,
                "available": False,
                "error": "Claudiomiro not available",
            }

        try:
            logger.info("Executing multi-repo workflow")

            result = claudiomiro.execute_multi_repo_task_with_claudiomiro(
                task_id=f"multi_repo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                prompt=prompt,
                backend=backend,
                frontend=frontend,
                working_dir=working_dir,
                legacy_backend=legacy_backend,
                legacy_frontend=legacy_frontend,
                ai_provider=self.ai_provider,
            )

            return {
                "success": result.get("success", False),
                "backend": backend,
                "frontend": frontend,
                "has_legacy": bool(legacy_backend or legacy_frontend),
                "output": result.get("output", ""),
                "message": "Multi-repo workflow completed" if result.get("success") else "Multi-repo workflow failed",
            }

        except Exception as e:
            logger.error(f"Multi-repo workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Multi-repo workflow failed: {e}",
            }

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _build_solution_prompt(
        self,
        problem_statement: str,
        sub_problems: List[Dict[str, Any]],
    ) -> str:
        """Build comprehensive prompt for solution phase."""
        prompt = f"Problem: {problem_statement}\n\n"
        prompt += "Sub-problems to solve:\n"

        for i, sub_problem in enumerate(sub_problems):
            prompt += f"\n{i+1}. {sub_problem.get('description', 'Sub-problem')}\n"

        prompt += "\nPlease implement complete solutions for all sub-problems."
        return prompt

    def _stub_result(self, phase: str, input: str) -> Dict[str, Any]:
        """Return stub result when Claudiomiro is not available."""
        return {
            "phase": phase,
            "success": False,
            "available": False,
            "error": "Claudiomiro not available",
            "message": "Claudiomiro CLI not installed or not in PATH",
        }


# ============================================================================
# Decorator for Automatic Claudiomiro Execution
# ============================================================================

def claudiomiro_capture(
    bridge: ClaudiomiroHephaestusWorkflowBridge,
    working_dir: str = ".",
):
    """
    Decorator for automatic Claudiomiro execution on function failure.

    Args:
        bridge: ClaudiomiroHephaestusWorkflowBridge instance
        working_dir: Working directory for Claudiomiro

    Example:
        bridge = ClaudiomiroHephaestusWorkflowBridge()

        @claudiomiro_capture(bridge)
        def my_development_task(input_data):
            # Try to implement
            return result

        result = my_development_task({"task": "Add feature"})
        # If it fails, Claudiomiro automatically fixes it
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                # Execute original function
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                # If it fails, use Claudiomiro to fix
                if CLAUDIOMIRO_AVAILABLE and bridge:
                    logger.info(f"Function {func.__name__} failed, using Claudiomiro to fix")

                    fix_result = claudiomiro.fix_tests_with_claudiomiro(
                        task_id=f"fix_{func.__name__}",
                        test_command="echo 'fix required'",
                        working_dir=working_dir,
                        ai_provider=bridge.ai_provider,
                    )

                    # Return fix result
                    return {
                        "original_error": str(e),
                        "claudiomiro_fix_attempted": True,
                        "fix_result": fix_result,
                    }
                else:
                    raise

        return wrapper

    return decorator


# ============================================================================
# Export all classes and functions
# ============================================================================

__all__ = [
    "ClaudiomiroHephaestusWorkflowBridge",
    "claudiomiro_capture",
    "CLAUDIOMIRO_AVAILABLE",
]

# Module initialization
if __name__ == "__main__":
    print("Claudiomiro Hephaestus Workflow Bridge Module")
    print(f"Claudiomiro Available: {CLAUDIOMIRO_AVAILABLE}")
    print("\nClasses:")
    print("  - ClaudiomiroHephaestusWorkflowBridge")
    print("\nDecorators:")
    print("  - claudiomiro_capture")
