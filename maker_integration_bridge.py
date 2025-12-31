"""
MAKER Integration Bridge

This module provides a unified integration layer for the MAKER (Maximal Agentic decomposition
with first-to-ahead-by-K Error correction and Red-flagging) framework with the existing
OpenEvolve decomposition infrastructure.

Based on the paper: "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)

Integration Points:
1. MAKER ←→ MDAP Engine
2. MAKER ←→ ROMA Decomposition
3. MAKER ←→ Sovereign Decomposition Engine
4. Unified API for all MAKER modes

Usage:
    from maker_integration_bridge import MAKERIntegrationBridge, create_maker_config

    # Create configuration
    config = create_maker_config(mode="recursive", k_ahead=3)

    # Initialize bridge
    bridge = MAKERIntegrationBridge(config, team)

    # Solve task
    result = bridge.solve("Solve Towers of Hanoi with 20 disks")
"""

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import MAKER components
from mdap_maker_complete import (
    MAKEREngine,
    RecursiveMAKERSolver,
    VotingEngine,
    VoteCollector,
    TaskDecomposition,
    MAKERRunMetrics,
    create_maker_config as base_maker_config,
    get_system_status as maker_system_status
)

# Import existing components
from workflow_structures import ModelConfig, Team
from llm_utils import _compose_messages, _request_openai_compatible_chat

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED CONFIGURATION
# =============================================================================

@dataclass
class MAKERIntegrationConfig:
    """
    Unified configuration for all MAKER modes.

    Supports three execution modes:
    1. "sequential": Algorithm 1 - generate_solution (for predetermined steps)
    2. "recursive": Algorithm 4 - Recursive multi-agent solve (for general tasks)
    3. "hybrid": ROMA decomposition + MAKER voting
    """
    # Execution mode
    mode: str = "recursive"  # "sequential", "recursive", "hybrid"

    # Voting parameters
    k_ahead: int = 3  # First-to-ahead-by-k threshold
    num_candidates: int = 5  # N = 2k - 1 candidates for voting
    enable_first_to_ahead: bool = True  # True = first-to-ahead-by-k, False = first-to-k

    # Red-flagging parameters
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_characters: Optional[int] = 6000

    # Execution limits
    max_steps: int = 1000  # For sequential mode
    max_depth: int = 5  # For recursive mode
    timeout_seconds: int = 300

    # ROMA integration (for hybrid mode)
    enable_roma: bool = False
    roma_max_depth: int = 3

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000

    # Provider settings
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature_first: float = 0.0
    temperature_subsequent: float = 0.1

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_maker_config(
    mode: str = "recursive",
    k_ahead: int = 3,
    max_depth: int = 5,
    enable_red_flagging: bool = True,
    **kwargs
) -> MAKERIntegrationConfig:
    """
    Create MAKER configuration.

    Args:
        mode: Execution mode ("sequential", "recursive", "hybrid")
        k_ahead: Voting threshold
        max_depth: Max recursion depth (for recursive mode)
        enable_red_flagging: Enable red-flagging
        **kwargs: Additional configuration

    Returns:
        MAKERIntegrationConfig object
    """
    return MAKERIntegrationConfig(
        mode=mode,
        k_ahead=k_ahead,
        max_depth=max_depth,
        enable_red_flagging=enable_red_flagging,
        **kwargs
    )


# =============================================================================
# MAIN INTEGRATION BRIDGE
# =============================================================================

class MAKERIntegrationBridge:
    """
    Unified integration bridge for all MAKER functionality.

    Provides a single API for:
    1. Sequential task solving (Algorithm 1)
    2. Recursive decomposition solving (Algorithm 4)
    3. Hybrid ROMA+MAKER solving
    4. Direct access to MAKER components
    """

    def __init__(
        self,
        config: MAKERIntegrationConfig,
        team: Optional[Team] = None
    ):
        """
        Initialize MAKER integration bridge.

        Args:
            config: MAKER configuration
            team: Optional team (will create default if not provided)
        """
        self.config = config
        self.team = team or self._create_default_team()

        # Initialize appropriate engine based on mode
        if config.mode == "sequential":
            self.engine = MAKEREngine(
                team=self.team,
                k_ahead=config.k_ahead,
                max_token_length=config.max_token_length,
                max_steps=config.max_steps,
                enable_first_to_ahead=config.enable_first_to_ahead,
                enable_red_flagging=config.enable_red_flagging
            )
            self.solver = None
        elif config.mode == "recursive":
            self.engine = None
            self.solver = RecursiveMAKERSolver(
                team=self.team,
                max_depth=config.max_depth,
                k_ahead=config.k_ahead,
                num_candidates=config.num_candidates,
                max_token_length=config.max_token_length
            )
        elif config.mode == "hybrid":
            # Hybrid uses both
            self.engine = MAKEREngine(
                team=self.team,
                k_ahead=config.k_ahead,
                max_token_length=config.max_token_length,
                max_steps=config.max_steps,
                enable_first_to_ahead=config.enable_first_to_ahead,
                enable_red_flagging=config.enable_red_flagging
            )
            self.solver = RecursiveMAKERSolver(
                team=self.team,
                max_depth=config.max_depth,
                k_ahead=config.k_ahead,
                num_candidates=config.num_candidates,
                max_token_length=config.max_token_length
            )
        else:
            raise ValueError(f"Unknown mode: {config.mode}")

        logger.info(f"MAKER integration bridge initialized in {config.mode} mode")

    def solve(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[Callable[[Any], str]] = None,
        system_prompt: Optional[str] = None,
        expected_schema: Optional[Dict[str, Any]] = None,
        parser: Optional[Callable[[str], Tuple[Any, Any]]] = None,
        initial_state: Optional[Any] = None,
        stop_condition: Optional[Callable[[Any], bool]] = None,
        progress_callback: Optional[Callable[[int, Any], None]] = None,
        max_depth_override: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main entry point: Solve task using MAKER.

        Automatically routes to appropriate solver based on configuration mode.

        Args:
            task: Task description or initial state
            context: Optional context dict
            prompt_template: Function(state) -> prompt (for sequential mode)
            system_prompt: System prompt
            expected_schema: Optional JSON schema for validation
            parser: Optional custom response parser
            initial_state: Initial state (for sequential mode)
            stop_condition: Function(state) -> bool (for sequential mode)
            progress_callback: Function(step, state) (for sequential mode)
            max_depth_override: Override max depth (for recursive mode)

        Returns:
            Dict with:
                - result: Solution/result
                - metrics: MAKERRunMetrics
                - mode: Execution mode used
                - success: Boolean
                - execution_time: Seconds
        """
        start_time = time.time()
        context = context or {}

        try:
            if self.config.mode == "sequential":
                return self._solve_sequential(
                    task, initial_state, prompt_template, system_prompt,
                    expected_schema, parser, stop_condition, progress_callback
                )
            elif self.config.mode == "recursive":
                return self._solve_recursive(
                    task, context, max_depth_override
                )
            elif self.config.mode == "hybrid":
                return self._solve_hybrid(
                    task, context, max_depth_override
                )
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

        except Exception as e:
            logger.error(f"MAKER solve failed: {e}", exc_info=True)
            return {
                "result": None,
                "error": str(e),
                "mode": self.config.mode,
                "success": False,
                "execution_time": time.time() - start_time
            }

    def _solve_sequential(
        self,
        task: str,
        initial_state: Any,
        prompt_template: Callable[[Any], str],
        system_prompt: str,
        expected_schema: Optional[Dict[str, Any]],
        parser: Optional[Callable[[str], Tuple[Any, Any]]],
        stop_condition: Optional[Callable[[Any], bool]],
        progress_callback: Optional[Callable[[int, Any], None]]
    ) -> Dict[str, Any]:
        """Solve using sequential mode (Algorithm 1)."""
        if not self.engine:
            raise RuntimeError("Sequential engine not initialized")

        # Default prompt template if not provided
        if prompt_template is None:
            prompt_template = lambda state: f"Current state: {json.dumps(state, indent=2)}\n\nNext step?"

        # Default system prompt
        if system_prompt is None:
            system_prompt = "You are a specialized task execution agent. Follow instructions precisely."

        # Execute
        action_list, final_state, metrics = self.engine.generate_solution(
            initial_state=initial_state or task,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            expected_schema=expected_schema,
            parser=parser,
            stop_condition=stop_condition,
            progress_callback=progress_callback
        )

        return {
            "result": {
                "actions": action_list,
                "final_state": final_state
            },
            "metrics": metrics,
            "mode": "sequential",
            "success": True,
            "execution_time": metrics.total_time
        }

    def _solve_recursive(
        self,
        task: str,
        context: Dict[str, Any],
        max_depth_override: Optional[int]
    ) -> Dict[str, Any]:
        """Solve using recursive mode (Algorithm 4)."""
        if not self.solver:
            raise RuntimeError("Recursive solver not initialized")

        # Execute
        solution, metrics = self.solver.solve(
            task=task,
            context=context,
            max_depth=max_depth_override or self.config.max_depth
        )

        return {
            "result": solution,
            "metrics": metrics,
            "mode": "recursive",
            "success": solution is not None,
            "execution_time": metrics.total_time
        }

    def _solve_hybrid(
        self,
        task: str,
        context: Dict[str, Any],
        max_depth_override: Optional[int]
    ) -> Dict[str, Any]:
        """
        Solve using hybrid mode (ROMA + MAKER).

        For tasks that benefit from both hierarchical decomposition
        and fine-grained voting.
        """
        # Try ROMA decomposition first if available
        try:
            from roma_mcp_tools import analyze_with_roma

            # ROMA analysis
            roma_result = analyze_with_roma(
                task=task,
                max_depth=self.config.roma_max_depth,
                execution_mode="recursive",
                provider=self.config.provider,
                model=self.config.model
            )

            if roma_result.get("decomposition"):
                # Apply MAKER voting to ROMA hierarchy
                if self.solver:
                    # For now, treat as recursive solve
                    # TODO: Implement true hybrid voting across ROMA tree
                    solution, metrics = self.solver.solve(
                        task=task,
                        context={**context, "roma_decomposition": roma_result["decomposition"]},
                        max_depth=max_depth_override or self.config.max_depth
                    )

                    return {
                        "result": solution,
                        "metrics": metrics,
                        "mode": "hybrid",
                        "roma_used": True,
                        "success": solution is not None,
                        "execution_time": metrics.total_time
                    }
        except ImportError:
            logger.warning("ROMA not available, falling back to recursive mode")
        except Exception as e:
            logger.warning(f"ROMA analysis failed: {e}, falling back to recursive mode")

        # Fallback to recursive
        return self._solve_recursive(task, context, max_depth_override)

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics from all engines."""
        metrics = {
            "mode": self.config.mode,
            "config": {
                "k_ahead": self.config.k_ahead,
                "max_depth": self.config.max_depth,
                "enable_red_flagging": self.config.enable_red_flagging
            }
        }

        if self.engine and self.engine.vote_collector:
            metrics["sequential"] = {
                "attempts": self.engine.vote_collector.attempt_count
            }

        if self.solver:
            metrics["recursive"] = self.solver.metrics.copy()

        return metrics

    def reset_metrics(self):
        """Reset all metrics."""
        if self.engine and self.engine.vote_collector:
            self.engine.vote_collector.attempt_count = 0

        if self.solver:
            self.solver.metrics = {
                "total_decompositions": 0,
                "atomic_solves": 0,
                "composition_votes": 0,
                "max_depth_reached": 0
            }

    def _create_default_team(self) -> Team:
        """Create default team if not provided."""
        from workflow_structures import ModelConfig

        model_config = ModelConfig(
            model_id=f"{self.config.provider}_{self.config.model}",
            provider=self.config.provider,
            model_name=self.config.model,
            api_key="",  # Will use env var
            temperature=self.config.temperature_subsequent
        )

        return Team(
            team_id="maker_default",
            name="MAKER Default Team",
            members=[model_config],
            description="Default team for MAKER execution"
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def solve_with_maker(
    task: str,
    mode: str = "recursive",
    k_ahead: int = 3,
    max_depth: int = 5,
    team: Optional[Team] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for quick MAKER solving.

    Args:
        task: Task description
        mode: Execution mode ("sequential", "recursive", "hybrid")
        k_ahead: Voting threshold
        max_depth: Max recursion depth
        team: Optional team
        context: Optional context

    Returns:
        Result dict with solution and metrics
    """
    config = create_maker_config(
        mode=mode,
        k_ahead=k_ahead,
        max_depth=max_depth
    )

    bridge = MAKERIntegrationBridge(config, team)

    return bridge.solve(task, context=context)


def solve_towers_of_hanoi(
    num_disks: int = 20,
    k_ahead: int = 3,
    team: Optional[Team] = None
) -> Dict[str, Any]:
    """
    Solve Towers of Hanoi using MAKER (as in the paper).

    This is the canonical example from the paper demonstrating
    zero-error solving of million-step tasks.

    Args:
        num_disks: Number of disks (20 = 1M+ steps)
        k_ahead: Voting threshold (3 was used in paper)
        team: Optional team

    Returns:
        Result dict with move sequence and metrics
    """
    # Initial state for Towers of Hanoi
    # Disks numbered 1 (smallest) to num_disks (largest)
    # Pegs: 0, 1, 2
    initial_state = [
        list(range(num_disks, 0, -1)),  # Peg 0: all disks
        [],  # Peg 1: empty
        []   # Peg 2: empty
    ]

    # Strategy prompt (from paper)
    system_prompt = """You are solving the Towers of Hanoi puzzle.
Rules:
- Only one disk can be moved at a time
- Only the top disk from any stack can be moved
- A larger disk may not be placed on top of a smaller disk

Strategy for even number of disks:
- If the previous move did NOT move disk 1, move disk 1 clockwise (0→1→2→0)
- If the previous move DID move disk 1, make the only legal move that does NOT involve disk 1

Respond with the next move in format:
move = [disk_id, from_peg, to_peg]
next_state = [[peg0], [peg1], [peg2]]"""

    def prompt_template(state):
        state_str = json.dumps(state, indent=2)
        return f"""Current state (pegs 0, 1, 2):
{state_str}

Find the next move according to the strategy."""

    def parse_response(raw_text: str):
        """Parse move and next_state from LLM response."""
        import re

        # Extract move
        move_match = re.search(r'move\s*=\s*\[(\d+),\s*(\d+),\s*(\d+)\]', raw_text)
        if move_match:
            move = [int(move_match.group(1)), int(move_match.group(2)), int(move_match.group(3))]
        else:
            raise ValueError("No move found in response")

        # Extract next_state
        state_match = re.search(r'next_state\s*=\s*(\[\[.*\]\])', raw_text, re.DOTALL)
        if state_match:
            next_state = json.loads(state_match.group(1))
        else:
            raise ValueError("No next_state found in response")

        return {"action": move, "next_state": next_state}, next_state

    def stop_condition(state):
        """Stop when all disks are on peg 2."""
        return len(state[2]) == num_disks and state[2] == list(range(num_disks, 0, -1))

    config = create_maker_config(
        mode="sequential",
        k_ahead=k_ahead,
        max_steps=(2 ** num_disks) - 1  # Optimal number of steps
    )

    bridge = MAKERIntegrationBridge(config, team)

    return bridge.solve(
        task=f"Solve Towers of Hanoi with {num_disks} disks",
        initial_state=initial_state,
        prompt_template=prompt_template,
        system_prompt=system_prompt,
        parser=parse_response,
        stop_condition=stop_condition
    )


def solve_multiplication(
    num1: int,
    num2: int,
    k_ahead: int = 3,
    team: Optional[Team] = None
) -> Dict[str, Any]:
    """
    Solve multi-digit multiplication using MAKER recursive decomposition.

    This is the second example from the paper (Appendix F) demonstrating
    general-purpose decomposition.

    Args:
        num1: First number
        num2: Second number
        k_ahead: Voting threshold
        team: Optional team

    Returns:
        Result dict with product and metrics
    """
    task = f"Calculate the product: {num1} × {num2}"

    context = {
        "operation": "multiplication",
        "num1": num1,
        "num2": num2,
        "expected_output": "integer product"
    }

    return solve_with_maker(
        task=task,
        mode="recursive",
        k_ahead=k_ahead,
        max_depth=4,
        team=team,
        context=context
    )


def get_integrated_status() -> Dict[str, Any]:
    """Get integrated system status."""
    base_status = maker_system_status()

    return {
        **base_status,
        "integration_version": "1.0.0",
        "supported_modes": ["sequential", "recursive", "hybrid"],
        "default_mode": "recursive",
        "examples": [
            "Towers of Hanoi (20 disks = 1M+ steps)",
            "Multi-digit multiplication",
            "General task decomposition"
        ]
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "MAKERIntegrationBridge",
    "MAKERIntegrationConfig",

    # Convenience functions
    "create_maker_config",
    "solve_with_maker",
    "solve_towers_of_hanoi",
    "solve_multiplication",
    "get_integrated_status",
]
