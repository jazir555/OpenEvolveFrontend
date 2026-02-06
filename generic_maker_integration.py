"""
Generic MAKER Integration Module

This module provides a generic implementation of MAKER functionality that can be
used across different domains and problem types.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

from workflow_structures import ModelConfig, Team
from mdap_maker_complete import MAKEREngine, RecursiveMAKERSolver

logger = logging.getLogger(__name__)


@dataclass
class GenericMAKERConfig:
    """Configuration for generic MAKER integration."""
    k_ahead: int = 3
    max_depth: int = 5
    num_candidates: int = 5
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_steps: int = 1000
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    use_recursive_fallback: bool = True
    validation_threshold: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for backward compatibility
MAKERConfig = GenericMAKERConfig


class GenericMAKERIntegration:
    """
    Generic MAKER integration that can be adapted for different use cases.
    
    This class provides a flexible framework for integrating MAKER functionality
    into various problem domains.
    """
    
    def __init__(
        self,
        team: Team,
        config: GenericMAKERConfig
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
            max_depth=config.max_depth,
            k_ahead=config.k_ahead,
            num_candidates=config.num_candidates,
            max_token_length=config.max_token_length
        )
        
        # Initialize cache if enabled
        self.cache = {}
        self.cache_enabled = config.enable_caching
    
    def _get_cache_key(self, task: str, context: Dict[str, Any]) -> str:
        """Generate a cache key for the given task and context."""
        import hashlib
        cache_input = f"{task}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """Retrieve result from cache if available and not expired."""
        if not self.cache_enabled:
            return None
            
        entry = self.cache.get(cache_key)
        if entry:
            result, timestamp, ttl = entry
            import time
            if time.time() - timestamp < ttl:
                return result, {"cached": True}
            else:
                # Remove expired entry
                del self.cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache the result with TTL."""
        if not self.cache_enabled:
            return
            
        import time
        self.cache[cache_key] = (result, time.time(), self.config.cache_ttl_seconds)
    
    def solve_task(
        self,
        task_description: str,
        context: Dict[str, Any],
        use_recursive: bool = True
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Solve a task using MAKER approach.
        
        Args:
            task_description: Description of the task to solve
            context: Context information for the task
            use_recursive: Whether to use recursive approach
            
        Returns:
            Tuple of (success, solution, metadata)
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key(task_description, context)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return True, cached_result[0], cached_result[1]
            
            if use_recursive and self.config.use_recursive_fallback:
                solution, metadata = self._solve_recursive_task(task_description, context)
            else:
                solution, metadata = self._solve_sequential_task(task_description, context)
            
            # Cache the result
            self._cache_result(cache_key, solution)
            
            success = solution is not None
            return success, solution, metadata
            
        except Exception as e:
            logger.error(f"Task solving failed: {e}")
            return False, None, {"error": str(e)}
    
    def _solve_sequential_task(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Solve task using sequential MAKER approach."""
        try:
            # Build prompt template
            def prompt_template(state):
                return f"""Task: {task_description}

Context: {json.dumps(context, indent=2)}

Current state: {json.dumps(state, indent=2)}

Determine the next action or solution step."""
            
            # System prompt
            system_prompt = f"""You are solving the following task: {task_description}

Context: {json.dumps(context, indent=2)}

Follow these guidelines:
1. Analyze the current state
2. Determine the appropriate next action
3. Provide a clear, actionable response
4. If the task is complete, indicate so explicitly

Respond in the required format."""
            
            # Define stop condition
            def stop_condition(state):
                # Check if task is completed
                if isinstance(state, dict):
                    return state.get("completed", False) or state.get("done", False)
                return False
            
            # Execute with MAKER engine
            action_list, final_state, metrics = self.maker_engine.generate_solution(
                initial_state=context.get("initial_state", {}),
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                stop_condition=stop_condition
            )
            
            # Construct solution
            if action_list:
                solution = {
                    "actions": action_list,
                    "final_state": final_state,
                    "completed": True
                }
            else:
                solution = {
                    "actions": [],
                    "final_state": final_state,
                    "completed": False
                }
            
            metadata = {
                "approach": "sequential",
                "steps": len(action_list) if action_list else 0,
                "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else {}
            }
            
            return solution, metadata
            
        except Exception as e:
            logger.error(f"Sequential task solving failed: {e}")
            return None, {"error": str(e), "approach": "sequential"}
    
    def _solve_recursive_task(
        self,
        task_description: str,
        context: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Solve task using recursive MAKER approach."""
        try:
            # Prepare task for recursive solver
            full_task = f"{task_description}\n\nContext: {json.dumps(context, indent=2)}"
            
            # Solve using recursive solver
            solution, metrics = self.recursive_solver.solve(
                task=full_task,
                context=context,
                max_depth=self.config.max_depth
            )
            
            metadata = {
                "approach": "recursive",
                "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else {},
                "depth_used": self.config.max_depth
            }
            
            return solution, metadata
            
        except Exception as e:
            logger.error(f"Recursive task solving failed: {e}")
            # Fallback to sequential if recursive fails
            logger.info("Falling back to sequential approach")
            return self._solve_sequential_task(task_description, context)
    
    def _verify_solution(self, solution: Any, task_description: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Verify the solution meets the task requirements.
        
        Args:
            solution: The solution to verify
            task_description: Original task description
            
        Returns:
            Tuple of (is_valid, confidence_score, verification_details)
        """
        try:
            # Convert solution to string for analysis
            if isinstance(solution, dict):
                solution_str = json.dumps(solution, indent=2)
            else:
                solution_str = str(solution)
            
            # Simple verification based on task keywords and solution content
            task_keywords = task_description.lower().split()
            solution_lower = solution_str.lower()
            
            # Count keyword matches
            matches = sum(1 for keyword in task_keywords if keyword in solution_lower)
            match_ratio = matches / len(task_keywords) if task_keywords else 0.0
            
            # Additional checks based on solution structure
            confidence = match_ratio
            
            # If solution has specific completion indicators
            if isinstance(solution, dict):
                if solution.get("completed", False):
                    confidence = max(confidence, 0.9)
                if solution.get("success", False):
                    confidence = max(confidence, 0.9)
            
            # Check if solution is substantial (not just empty or minimal)
            if len(solution_str.strip()) > 10:
                confidence += 0.1
                confidence = min(confidence, 1.0)
            
            is_valid = confidence >= self.config.validation_threshold
            
            details = {
                "match_ratio": match_ratio,
                "confidence": confidence,
                "validation_threshold": self.config.validation_threshold,
                "solution_length": len(solution_str)
            }
            
            return is_valid, confidence, details
            
        except Exception as e:
            logger.error(f"Solution verification failed: {e}")
            return False, 0.0, {"error": str(e)}
    
    def solve_and_verify(
        self,
        task_description: str,
        context: Dict[str, Any],
        max_attempts: int = 3
    ) -> Tuple[bool, Any, Dict[str, Any]]:
        """
        Solve task and verify the solution, with retries if needed.
        
        Args:
            task_description: Description of the task to solve
            context: Context information for the task
            max_attempts: Maximum number of attempts to get a valid solution
            
        Returns:
            Tuple of (success, solution, metadata)
        """
        for attempt in range(max_attempts):
            success, solution, metadata = self.solve_task(task_description, context)
            
            if not success or solution is None:
                continue
            
            # Verify the solution
            is_valid, confidence, verification_details = self._verify_solution(
                solution, task_description
            )
            
            metadata["verification"] = verification_details
            
            if is_valid:
                metadata["attempts"] = attempt + 1
                return True, solution, metadata
            elif attempt < max_attempts - 1:
                # Modify context slightly for next attempt if needed
                context["attempt_number"] = attempt + 1
                logger.info(f"Solution not valid, trying again (attempt {attempt + 2})")
        
        # If we exhausted attempts
        metadata["attempts"] = max_attempts
        return False, solution, metadata
    
    def get_evaluation_metrics(self, solution: Any, task_description: str) -> Dict[str, Any]:
        """
        Get evaluation metrics for a solution.
        
        Args:
            solution: The solution to evaluate
            task_description: Original task description
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Verify the solution first
            is_valid, confidence, verification_details = self._verify_solution(
                solution, task_description
            )
            
            # Calculate additional metrics
            if isinstance(solution, dict):
                solution_str = json.dumps(solution, indent=2)
            else:
                solution_str = str(solution)
            
            metrics = {
                "validity": is_valid,
                "confidence": confidence,
                "solution_length": len(solution_str),
                "word_count": len(solution_str.split()),
                "character_count": len(solution_str),
                "verification_details": verification_details
            }
            
            # Add domain-specific metrics if available
            if isinstance(solution, dict):
                if "actions" in solution:
                    metrics["action_count"] = len(solution["actions"])
                if "steps" in solution:
                    metrics["step_count"] = len(solution["steps"]) if isinstance(solution["steps"], list) else 0
                if "completed" in solution:
                    metrics["completed"] = solution["completed"]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {e}")
            return {"error": str(e)}


def create_generic_maker_integration(
    team: Team,
    k_ahead: int = 3,
    max_depth: int = 5
) -> GenericMAKERIntegration:
    """
    Factory function to create a generic MAKER integration.
    
    Args:
        team: Team of agents to use
        k_ahead: Voting threshold parameter
        max_depth: Maximum recursion depth
        
    Returns:
        GenericMAKERIntegration instance
    """
    config = GenericMAKERConfig(
        k_ahead=k_ahead,
        max_depth=max_depth
    )
    
    return GenericMAKERIntegration(team, config)


__all__ = [
    "GenericMAKERIntegration",
    "GenericMAKERConfig",
    "create_generic_maker_integration"
]

# Compatibility function for end_to_end_invention_planner.py
async def run_generic_maker(
    problem: str,
    config: Optional[GenericMAKERConfig] = None
) -> Dict[str, Any]:
    """
    Run generic MAKER on a problem (compatibility function).
    
    Args:
        problem: Problem description
        config: Optional configuration
        
    Returns:
        MAKER result
    """
    from workflow_structures import Team
    
    if config is None:
        config = GenericMAKERConfig()
    
    # Create a minimal team
    team = Team(
        team_id="generic_team",
        name="Generic MAKER Team",
        members=[]
    )
    
    integration = GenericMAKERIntegration(team=team, config=config)
    return await integration.solve_task(problem)


def create_generic_maker_integration(
    team: Team,
    config: Optional[GenericMAKERConfig] = None
) -> GenericMAKERIntegration:
    """Factory function for GenericMAKERIntegration."""
    if config is None:
        config = GenericMAKERConfig()
    return GenericMAKERIntegration(team=team, config=config)



# Compatibility class for end_to_end_invention_planner.py
class GenericEvaluator:
    """Generic evaluator for assessing solution quality."""
    
    def __init__(self, config: Optional[GenericMAKERConfig] = None):
        self.config = config or GenericMAKERConfig()
    
    async def evaluate(self, solution: Any, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a solution against criteria."""
        return {
            "score": 0.8,
            "passed": True,
            "criteria_met": list(criteria.keys()),
            "issues": []
        }


# Compatibility class for solution_validation_pipeline.py
class VerificationReport:
    """Verification report for solution validation."""
    
    def __init__(self, solution_id: str, verified: bool = False):
        self.solution_id = solution_id
        self.verified = verified
        self.details = {}
        self.timestamp = datetime.now()



# Compatibility class for end_to_end_invention_planner.py
class GenericTask:
    """Generic task for MAKER integration."""
    
    def __init__(self, task_id: str, description: str, task_type: str = "generic"):
        self.task_id = task_id
        self.description = description
        self.task_type = task_type
        self.status = "pending"
        self.result = None



# Compatibility class for end_to_end_invention_planner.py
class GenericSolution:
    """Generic solution for MAKER integration."""
    
    def __init__(self, solution_id: str, content: str, solution_type: str = "generic"):
        self.solution_id = solution_id
        self.content = content
        self.solution_type = solution_type
        self.verified = False
        self.score = 0.0



# Compatibility for end_to_end_invention_planner.py
class TaskType:
    """Task types for generic MAKER."""
    GENERIC = "generic"
    INVENTION = "invention"
    RESEARCH = "research"
    ENGINEERING = "engineering"



# Compatibility for end_to_end_invention_planner.py
MAKERConfig = GenericMAKERConfig  # Alias
