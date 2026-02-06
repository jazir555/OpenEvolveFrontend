"""
Decomposition-Recomposition Integration Module

This module provides integration between problem decomposition and solution recomposition
to enable solving complex problems by breaking them down and reassembling solutions.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

from workflow_structures import ModelConfig, Team, SubProblem, SolutionAttempt
from mdap_maker_complete import MAKEREngine, RecursiveMAKERSolver

logger = logging.getLogger(__name__)


class SolverMode(Enum):
    """Different modes for the solver."""
    DECOMPOSITION_FIRST = "decomposition_first"
    RECURSIVE_DECOMPOSITION = "recursive_decomposition"
    PARALLEL_SUBSOLUTIONS = "parallel_subsolutions"
    HYBRID_APPROACH = "hybrid_approach"


@dataclass
class SolverConfig:
    """Configuration for the decomposition-recomposition solver."""
    mode: SolverMode = SolverMode.HYBRID_APPROACH
    max_decomposition_depth: int = 5
    max_subproblems: int = 10
    k_ahead: int = 3
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_steps: int = 1000
    timeout_seconds: int = 300
    enable_validation: bool = True
    validation_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


class SolutionSolver:
    """
    Solver that handles both decomposition and recomposition of solutions.
    
    This class manages the process of breaking down complex problems into subproblems,
    solving each subproblem, and then recombining the solutions into a cohesive whole.
    """
    
    def __init__(
        self,
        team: Team,
        config: SolverConfig
    ):
        self.config = config
        self.team = team
        
        # Initialize MAKER components
        self.maker_engine = MAKEREngine(
            team=team,
            k_ahead=config.k_ahead,
            max_token_length=config.max_token_length,
            max_steps=config.max_steps,
            enable_red_flagging=config.enable_red_flagging
        )
        
        self.recursive_solver = RecursiveMAKERSolver(
            team=team,
            max_depth=config.max_decomposition_depth,
            k_ahead=config.k_ahead,
            num_candidates=5,  # Default number of candidates
            max_token_length=config.max_token_length
        )
    
    def decompose_problem(
        self,
        problem_description: str,
        context: Dict[str, Any]
    ) -> List[SubProblem]:
        """
        Decompose a complex problem into subproblems.
        
        Args:
            problem_description: Description of the main problem
            context: Context information for decomposition
            
        Returns:
            List of SubProblem objects
        """
        try:
            # Build decomposition prompt
            prompt = f"""Decompose the following problem into smaller, manageable subproblems:

Main Problem: {problem_description}

Context: {json.dumps(context, indent=2)}

Provide the decomposition in the following format:
1. List the subproblems
2. Describe each subproblem clearly
3. Indicate dependencies between subproblems if any
4. Estimate effort for each subproblem

Return as a structured response."""
            
            system_prompt = """You are an expert problem decomposer. Your task is to break down complex problems into smaller, manageable subproblems that can be solved independently. Follow these principles:
1. Each subproblem should be clearly defined
2. Subproblems should be as independent as possible
3. The combination of solutions should solve the main problem
4. Estimate effort and priority for each subproblem"""
            
            # Use MAKER to generate decomposition
            action_list, final_state, metrics = self.maker_engine.generate_solution(
                initial_state={"problem": problem_description, "context": context},
                prompt_template=lambda state: prompt,
                system_prompt=system_prompt,
                stop_condition=lambda s: "decomposition" in s
            )
            
            # Parse the decomposition result
            decomposition_result = self._parse_decomposition_result(action_list, problem_description)
            
            # Create SubProblem objects
            subproblems = []
            for i, subproblem_data in enumerate(decomposition_result):
                subproblem = SubProblem(
                    id=f"subprob_{i+1}_{hash(problem_description) % 1000}",
                    title=subproblem_data.get("title", f"Subproblem {i+1}"),
                    description=subproblem_data.get("description", ""),
                    type=subproblem_data.get("type", "general"),
                    estimated_effort=subproblem_data.get("effort", 1),
                    priority=subproblem_data.get("priority", 1),
                    dependencies=subproblem_data.get("dependencies", []),
                    success_criteria=subproblem_data.get("success_criteria", [])
                )
                subproblems.append(subproblem)
            
            # Limit number of subproblems if needed
            if len(subproblems) > self.config.max_subproblems:
                subproblems = subproblems[:self.config.max_subproblems]
            
            logger.info(f"Decomposed problem into {len(subproblems)} subproblems")
            return subproblems
            
        except Exception as e:
            logger.error(f"Problem decomposition failed: {e}")
            # Return a single subproblem with the original problem if decomposition fails
            return [SubProblem(
                id=f"subprob_direct_{hash(problem_description) % 1000}",
                title="Direct Solution",
                description=problem_description,
                type="general",
                estimated_effort=5,
                priority=1
            )]
    
    def solve_subproblem(
        self,
        subproblem: SubProblem,
        context: Dict[str, Any]
    ) -> SolutionAttempt:
        """
        Solve a single subproblem.
        
        Args:
            subproblem: The subproblem to solve
            context: Context information for solving
            
        Returns:
            SolutionAttempt object
        """
        try:
            # Build solution prompt
            prompt = f"""Solve the following subproblem:

Title: {subproblem.title}
Description: {subproblem.description}

Context: {json.dumps(context, indent=2)}

Provide a detailed solution to this subproblem."""
            
            system_prompt = f"""You are solving a subproblem: {subproblem.title}

Description: {subproblem.description}

Context: {json.dumps(context, indent=2)}

Provide a comprehensive solution that addresses the subproblem requirements."""
            
            # Use MAKER to generate solution
            action_list, final_state, metrics = self.maker_engine.generate_solution(
                initial_state={"subproblem": subproblem.__dict__, "context": context},
                prompt_template=lambda state: prompt,
                system_prompt=system_prompt,
                stop_condition=lambda s: "solution" in s or "completed" in s
            )
            
            # Construct solution content
            solution_content = ""
            for action in action_list:
                if isinstance(action, dict):
                    solution_content += action.get("content", str(action)) + "\n"
                else:
                    solution_content += str(action) + "\n"
            
            if not solution_content.strip():
                solution_content = json.dumps(final_state, indent=2)
            
            # Create solution attempt
            solution_attempt = SolutionAttempt(
                sub_problem_id=subproblem.id,
                team_id=self.team.team_id,
                content=solution_content,
                metadata={
                    "subproblem_title": subproblem.title,
                    "subproblem_type": subproblem.type,
                    "effort_estimate": subproblem.estimated_effort,
                    "solution_metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else {},
                    "context_used": context
                }
            )
            
            logger.info(f"Solved subproblem: {subproblem.title}")
            return solution_attempt
            
        except Exception as e:
            logger.error(f"Subproblem solving failed for {subproblem.title}: {e}")
            # Return a failed solution attempt
            return SolutionAttempt(
                sub_problem_id=subproblem.id,
                team_id=self.team.team_id,
                content="",
                metadata={
                    "error": str(e),
                    "subproblem_title": subproblem.title,
                    "status": "failed"
                }
            )
    
    def recompose_solution(
        self,
        subproblem_solutions: List[SolutionAttempt],
        original_problem: str,
        context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Recompose individual subproblem solutions into a complete solution.
        
        Args:
            subproblem_solutions: List of solutions to subproblems
            original_problem: The original problem that was decomposed
            context: Context information for recomposition
            
        Returns:
            Tuple of (recomposed_solution, metadata)
        """
        try:
            # Build solution content from individual subproblem solutions
            solution_parts = []
            for sol in subproblem_solutions:
                solution_parts.append(f"Solution to '{sol.metadata.get('subproblem_title', 'Unknown')}':\n{sol.content}\n")
            
            all_solutions = "\n".join(solution_parts)
            
            # Build recomposition prompt
            prompt = f"""Recompose the following subproblem solutions into a complete solution for the original problem:

Original Problem: {original_problem}

Context: {json.dumps(context, indent=2)}

Subproblem Solutions:
{all_solutions}

Combine these solutions into a coherent, comprehensive solution that addresses the original problem."""
            
            system_prompt = f"""You are a solution recomposer. Your task is to take solutions to subproblems and combine them into a cohesive solution for the original problem.

Original Problem: {original_problem}

Context: {json.dumps(context, indent=2)}

Subproblem Solutions:
{all_solutions}

Create a unified solution that:
1. Addresses all aspects of the original problem
2. Maintains coherence between different solution parts
3. Resolves any conflicts between subproblem solutions
4. Provides a clear, actionable outcome"""
            
            # Use MAKER to generate recomposed solution
            action_list, final_state, metrics = self.maker_engine.generate_solution(
                initial_state={
                    "original_problem": original_problem,
                    "context": context,
                    "subproblem_solutions": [s.__dict__ for s in subproblem_solutions]
                },
                prompt_template=lambda state: prompt,
                system_prompt=system_prompt,
                stop_condition=lambda s: "final_solution" in s or "completed" in s
            )
            
            # Construct final solution
            final_solution = ""
            for action in action_list:
                if isinstance(action, dict):
                    final_solution += action.get("content", str(action)) + "\n"
                else:
                    final_solution += str(action) + "\n"
            
            if not final_solution.strip():
                final_solution = all_solutions  # Fallback to concatenated solutions
            
            metadata = {
                "subproblem_count": len(subproblem_solutions),
                "recomposition_metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else {},
                "original_problem": original_problem
            }
            
            logger.info(f"Recomposed solution from {len(subproblem_solutions)} subproblem solutions")
            return final_solution, metadata
            
        except Exception as e:
            logger.error(f"Solution recomposition failed: {e}")
            # Fallback: concatenate all solutions
            fallback_solution = ""
            for sol in subproblem_solutions:
                fallback_solution += f"Subproblem: {sol.metadata.get('subproblem_title', 'Unknown')}\n"
                fallback_solution += f"Solution: {sol.content}\n\n"
            
            return fallback_solution, {
                "error": str(e),
                "fallback_used": True,
                "subproblem_count": len(subproblem_solutions)
            }
    
    def solve(
        self,
        problem_description: str,
        context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Solve a problem using decomposition-recomposition approach.
        
        Args:
            problem_description: Description of the problem to solve
            context: Context information for solving
            
        Returns:
            Tuple of (final_solution, metadata)
        """
        try:
            logger.info(f"Starting decomposition-recomposition for: {problem_description[:100]}...")
            
            # Step 1: Decompose the problem
            subproblems = self.decompose_problem(problem_description, context)
            
            # Step 2: Solve each subproblem
            solutions = []
            for subproblem in subproblems:
                solution = self.solve_subproblem(subproblem, context)
                solutions.append(solution)
            
            # Step 3: Recompose the solutions
            final_solution, metadata = self.recompose_solution(
                solutions, problem_description, context
            )
            
            # Add problem-specific metadata
            metadata.update({
                "problem_decomposed": True,
                "decomposition_depth": len(subproblems),
                "subproblem_solutions_count": len(solutions)
            })
            
            logger.info(f"Completed decomposition-recomposition with {len(subproblems)} subproblems")
            return final_solution, metadata
            
        except Exception as e:
            logger.error(f"Complete problem solving failed: {e}")
            # Fallback: try to solve directly without decomposition
            return self._solve_directly(problem_description, context)
    
    def _solve_directly(
        self,
        problem_description: str,
        context: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """Fallback method to solve problem directly without decomposition."""
        try:
            logger.info("Using direct solving as fallback")
            
            prompt = f"""Solve the following problem directly:

Problem: {problem_description}

Context: {json.dumps(context, indent=2)}

Provide a comprehensive solution."""
            
            system_prompt = f"""You are solving the following problem: {problem_description}

Context: {json.dumps(context, indent=2)}

Provide a detailed, comprehensive solution."""
            
            # Use MAKER to generate solution directly
            action_list, final_state, metrics = self.maker_engine.generate_solution(
                initial_state={"problem": problem_description, "context": context},
                prompt_template=lambda state: prompt,
                system_prompt=system_prompt,
                stop_condition=lambda s: "solution" in s or "completed" in s
            )
            
            # Construct solution
            solution_content = ""
            for action in action_list:
                if isinstance(action, dict):
                    solution_content += action.get("content", str(action)) + "\n"
                else:
                    solution_content += str(action) + "\n"
            
            if not solution_content.strip():
                solution_content = json.dumps(final_state, indent=2)
            
            return solution_content, {
                "direct_solution": True,
                "fallback_used": True,
                "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else {}
            }
            
        except Exception as e:
            logger.error(f"Direct solving also failed: {e}")
            return f"Unable to solve problem: {str(e)}", {
                "error": str(e),
                "direct_solution": False,
                "fallback_used": True
            }
    
    def _parse_decomposition_result(
        self,
        action_list: List[Any],
        original_problem: str
    ) -> List[Dict[str, Any]]:
        """Parse the decomposition result from MAKER actions."""
        try:
            # Look for decomposition in the actions
            for action in action_list:
                if isinstance(action, dict):
                    # Check if this action contains decomposition data
                    if "decomposition" in action:
                        decomp_data = action["decomposition"]
                        if isinstance(decomp_data, list):
                            return decomp_data
                        elif isinstance(decomp_data, str):
                            # Try to parse as JSON
                            try:
                                import ast
                                parsed = ast.literal_eval(decomp_data)
                                if isinstance(parsed, list):
                                    return parsed
                            except:
                                pass
            
            # If no structured decomposition found, create a simple one
            return [{
                "title": "Primary Solution Component",
                "description": original_problem,
                "type": "general",
                "effort": 5,
                "priority": 1,
                "dependencies": [],
                "success_criteria": ["Problem is solved"]
            }]
            
        except Exception as e:
            logger.error(f"Failed to parse decomposition result: {e}")
            # Return a default decomposition
            return [{
                "title": "Default Subproblem",
                "description": original_problem,
                "type": "general",
                "effort": 5,
                "priority": 1,
                "dependencies": [],
                "success_criteria": ["Problem is addressed"]
            }]


def create_solution_solver(
    team: Team,
    mode: SolverMode = SolverMode.HYBRID_APPROACH,
    max_depth: int = 5
) -> SolutionSolver:
    """
    Factory function to create a solution solver.
    
    Args:
        team: Team of agents to use
        mode: Solver mode to use
        max_depth: Maximum decomposition depth
        
    Returns:
        SolutionSolver instance
    """
    config = SolverConfig(
        mode=mode,
        max_decomposition_depth=max_depth
    )
    
    return SolutionSolver(team, config)


__all__ = [
    "SolutionSolver",
    "SolverConfig",
    "SolverMode",
    "create_solution_solver"
]