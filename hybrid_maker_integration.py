"""
Hybrid MAKER Integration Module

This module integrates multiple MAKER approaches with other systems to provide
hybrid problem-solving capabilities.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from workflow_structures import ModelConfig, Team
from mdap_maker_complete import MAKEREngine, RecursiveMAKERSolver
from maker_engine import MakerEngine, MakerConfig, MakerState, MakerRunResult

logger = logging.getLogger(__name__)


class HybridMode(Enum):
    """Different modes for hybrid MAKER operation."""
    SEQUENTIAL = "sequential"
    RECURSIVE = "recursive"
    COMBINED = "combined"
    ADAPTIVE = "adaptive"


@dataclass
class HybridMAKERConfig:
    """Configuration for hybrid MAKER system."""
    mode: HybridMode = HybridMode.COMBINED
    k_ahead: int = 3
    max_depth: int = 5
    num_candidates: int = 5
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_steps: int = 1000
    timeout_seconds: int = 300
    use_mdap_fallback: bool = True
    enable_caching: bool = True


class HybridMAKEREngine:
    """
    Hybrid MAKER engine that combines multiple MAKER approaches with other systems.
    
    This engine can operate in different modes combining sequential, recursive,
    and other problem-solving approaches.
    """
    
    def __init__(
        self,
        team: Team,
        config: HybridMAKERConfig
    ):
        self.config = config
        self.team = team
        
        # Initialize different MAKER engines based on mode
        if config.mode in [HybridMode.SEQUENTIAL, HybridMode.COMBINED, HybridMode.ADAPTIVE]:
            self.sequential_engine = MAKEREngine(
                team=team,
                k_ahead=config.k_ahead,
                max_token_length=config.max_token_length,
                max_steps=config.max_steps
            )
        
        if config.mode in [HybridMode.RECURSIVE, HybridMode.COMBINED, HybridMode.ADAPTIVE]:
            self.recursive_solver = RecursiveMAKERSolver(
                team=team,
                max_depth=config.max_depth,
                k_ahead=config.k_ahead,
                num_candidates=config.num_candidates,
                max_token_length=config.max_token_length
            )
        
        if config.mode == HybridMode.ADAPTIVE:
            # For adaptive mode, we'll use both and choose based on problem characteristics
            pass
    
    def generate_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int = 100
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Generate proof using hybrid MAKER approach.
        
        Args:
            initial_state: Starting state for the proof generation
            prompt_template: Template for generating prompts
            system_prompt: System prompt for the LLM
            max_steps: Maximum number of steps to take
            
        Returns:
            Tuple of (success, proof_text, metadata)
        """
        try:
            if self.config.mode == HybridMode.SEQUENTIAL:
                return self._generate_sequential_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
            elif self.config.mode == HybridMode.RECURSIVE:
                return self._generate_recursive_proof(
                    initial_state, prompt_template, system_prompt
                )
            elif self.config.mode == HybridMode.COMBINED:
                return self._generate_combined_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
            elif self.config.mode == HybridMode.ADAPTIVE:
                return self._generate_adaptive_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
            else:
                raise ValueError(f"Unknown hybrid mode: {self.config.mode}")
                
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": self.config.mode.value}
    
    def _generate_sequential_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using sequential MAKER approach."""
        try:
            # Define step builder function
            def step_builder(current_state, history):
                from maker_engine import MakerStep
                prompt = prompt_template.format(
                    state=json.dumps(current_state, indent=2),
                    history=json.dumps(history, indent=2)
                )
                return MakerStep(
                    step_id=f"proof_step_{len(history)}",
                    prompt_template=prompt,
                    system_prompt=system_prompt
                )
            
            # Define apply action function
            def apply_action(current_state, action):
                # Apply the action to the current state
                if isinstance(action, dict):
                    new_state = {**current_state, **action}
                else:
                    new_state = {**current_state, "last_action": action}
                return new_state
            
            # Define stop condition
            def stop_condition(state):
                # Check if we've reached a proof conclusion
                current = state.current_state
                if isinstance(current, dict):
                    return current.get("proved", False) or current.get("conclusion", "").lower().startswith("qed")
                return False
            
            # Use MakerEngine to solve
            maker_config = MakerConfig(
                k_min=self.config.k_ahead - 1 if self.config.k_ahead > 1 else 1,
                k_max=self.config.k_ahead + 1,
                max_votes_per_step=20,
                max_steps=max_steps,
                timeout_seconds=self.config.timeout_seconds
            )
            
            maker_engine = MakerEngine(self.team, maker_config)
            result = maker_engine.solve(
                initial_state=initial_state,
                step_builder=step_builder,
                apply_action=apply_action,
                stop_condition=stop_condition
            )
            
            # Extract proof from history
            proof_steps = []
            for entry in result.state.history:
                action = entry.get("action", {})
                if isinstance(action, dict):
                    step_text = action.get("step", str(action))
                else:
                    step_text = str(action)
                proof_steps.append(step_text)
            
            proof_text = "\n".join(proof_steps)
            success = len(proof_steps) > 0
            
            return success, proof_text, {
                "steps_taken": len(result.state.history),
                "terminated_reason": result.terminated_reason,
                "metrics": result.metrics
            }
            
        except Exception as e:
            logger.error(f"Sequential proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": "sequential"}
    
    def _generate_recursive_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using recursive MAKER approach."""
        try:
            # Format the task for recursive solver
            task = f"{prompt_template}\n\nInitial state: {json.dumps(initial_state)}"
            
            # Context for the solver
            context = {
                "system_prompt": system_prompt,
                "initial_state": initial_state
            }
            
            # Solve using recursive solver
            solution, metrics = self.recursive_solver.solve(
                task=task,
                context=context,
                max_depth=self.config.max_depth
            )
            
            if solution:
                if isinstance(solution, dict):
                    proof_text = solution.get("proof", json.dumps(solution, indent=2))
                else:
                    proof_text = str(solution)
                success = True
            else:
                proof_text = ""
                success = False
                
            return success, proof_text, {
                "metrics": metrics.__dict__ if hasattr(metrics, '__dict__') else vars(metrics) if isinstance(metrics, object) else {},
                "depth_used": self.config.max_depth
            }
            
        except Exception as e:
            logger.error(f"Recursive proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": "recursive"}
    
    def _generate_combined_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using combined sequential and recursive approaches."""
        try:
            # Try sequential first
            seq_success, seq_proof, seq_meta = self._generate_sequential_proof(
                initial_state, prompt_template, system_prompt, max_steps
            )
            
            # Try recursive second
            rec_success, rec_proof, rec_meta = self._generate_recursive_proof(
                initial_state, prompt_template, system_prompt
            )
            
            # Combine results based on which performed better
            if seq_success and rec_success:
                # Both succeeded, return the shorter proof or combine them
                if len(seq_proof) <= len(rec_proof):
                    return True, seq_proof, {
                        "primary_approach": "sequential",
                        "sequential_result": seq_meta,
                        "recursive_result": rec_meta
                    }
                else:
                    return True, rec_proof, {
                        "primary_approach": "recursive",
                        "sequential_result": seq_meta,
                        "recursive_result": rec_meta
                    }
            elif seq_success:
                return True, seq_proof, {
                    "primary_approach": "sequential",
                    "sequential_result": seq_meta
                }
            elif rec_success:
                return True, rec_proof, {
                    "primary_approach": "recursive",
                    "recursive_result": rec_meta
                }
            else:
                # Both failed
                return False, "", {
                    "primary_approach": "none",
                    "sequential_result": seq_meta,
                    "recursive_result": rec_meta
                }
                
        except Exception as e:
            logger.error(f"Combined proof generation failed: {e}")
            return False, "", {"error": str(e), "mode": "combined"}
    
    def _generate_adaptive_proof(
        self,
        initial_state: Any,
        prompt_template: str,
        system_prompt: str,
        max_steps: int
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Generate proof using adaptive approach based on problem characteristics."""
        try:
            # Analyze the problem to determine the best approach
            problem_complexity = self._estimate_problem_complexity(
                prompt_template, initial_state
            )
            
            if problem_complexity > 0.7:  # High complexity
                # Use recursive approach for complex problems
                return self._generate_recursive_proof(
                    initial_state, prompt_template, system_prompt
                )
            else:  # Low to medium complexity
                # Use sequential approach for simpler problems
                return self._generate_sequential_proof(
                    initial_state, prompt_template, system_prompt, max_steps
                )
                
        except Exception as e:
            logger.error(f"Adaptive proof generation failed: {e}")
            # Fallback to sequential
            return self._generate_sequential_proof(
                initial_state, prompt_template, system_prompt, max_steps
            )
    
    def _estimate_problem_complexity(self, prompt: str, state: Any) -> float:
        """Estimate problem complexity to choose the best approach."""
        # Simple heuristic: longer prompts or more complex state suggest higher complexity
        prompt_length = len(prompt)
        state_complexity = 0
        
        if isinstance(state, dict):
            state_complexity = len(json.dumps(state))
        elif hasattr(state, '__dict__'):
            state_complexity = len(json.dumps(vars(state)))
        else:
            state_complexity = len(str(state))
        
        # Normalize to 0-1 scale
        complexity_score = min(1.0, (prompt_length + state_complexity) / 10000.0)
        return complexity_score


def create_hybrid_maker_engine(
    team: Team,
    mode: HybridMode = HybridMode.COMBINED,
    k_ahead: int = 3,
    max_depth: int = 5
) -> HybridMAKEREngine:
    """
    Factory function to create a hybrid MAKER engine.
    
    Args:
        team: Team of agents to use
        mode: Hybrid mode to use
        k_ahead: Voting threshold parameter
        max_depth: Maximum recursion depth
        
    Returns:
        HybridMAKEREngine instance
    """
    config = HybridMAKERConfig(
        mode=mode,
        k_ahead=k_ahead,
        max_depth=max_depth
    )
    
    return HybridMAKEREngine(team, config)


__all__ = [
    "HybridMAKEREngine",
    "HybridMAKERConfig", 
    "HybridMode",
    "create_hybrid_maker_engine"
]