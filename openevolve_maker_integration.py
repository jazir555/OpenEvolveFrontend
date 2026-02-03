"""
OpenEvolve-MAKER Integration Module

This module integrates the MAKER framework (from arXiv:2511.09030) with the OpenEvolve
decomposition workflow, providing zero-error solving capabilities for long-horizon tasks.

Key Integrations:
1. MAKER ←→ OpenEvolve API: Uses OpenEvolveClient for LLM calls
2. MAKER ←→ Decomposition Workflow: Integrated into workflow_engine.py
3. MAKER ←→ MDAP: Works alongside existing MDAP implementation
4. MAKER ←→ Sovereign Teams: Uses Team configurations from workflow

Based on:
- Paper: "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
- OpenEvolve Decomposition Workflow (Decomposition_Workflow.md)
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

# Core imports
from workflow_structures import (
    ModelConfig, Team, WorkflowState, SubProblem, SolutionAttempt,
    GauntletDefinition, GauntletRoundRule
)

# Try to import OpenEvolve components
try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
    from openevolve_integration import OpenEvolveAPI
except ImportError:
    OpenEvolveClient = None
    OPENEVOLVE_AVAILABLE = False
    OpenEvolveAPI = None

# Import core MAKER implementation
from mdap_maker_complete import (
    MAKEREngine, RecursiveMAKERSolver, VotingEngine,
    VoteCollector, TaskDecomposition, MAKERRunMetrics
)

# Import existing MDAP
try:
    from mdap_engine import MDAPConfig, MDAPStep, MDAPTask
except ImportError:
    MDAPConfig = None
    MDAPStep = None
    MDAPTask = None

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class MAKERMode(Enum):
    """MAKER execution modes"""
    SEQUENTIAL = "sequential"  # Algorithm 1: generate_solution
    RECURSIVE = "recursive"    # Algorithm 4: Recursive decomposition
    HYBRID = "hybrid"          # ROMA + MAKER voting


@dataclass
class MAKERWorkflowConfig:
    """Configuration for MAKER within the decomposition workflow"""
    # Execution mode
    mode: MAKERMode = MAKERMode.RECURSIVE

    # Voting parameters
    k_ahead: int = 3
    num_candidates: int = 5
    enable_first_to_ahead: bool = True

    # Red-flagging
    enable_red_flagging: bool = True
    max_token_length: int = 750
    max_characters: int = 6000

    # Execution limits
    max_steps: int = 1000
    max_depth: int = 5
    timeout_seconds: int = 300

    # OpenEvolve integration
    use_openevolve_client: bool = True
    openevolve_base_url: Optional[str] = None
    openevolve_api_key: Optional[str] = None

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'MAKERWorkflowConfig':
        """Create a config from a predefined preset."""
        preset = MAKER_PRESETS.get(preset_name.upper(), MAKER_PRESETS["BALANCED"])
        config_dict = {**preset, **overrides}
        
        # Handle mode enum
        if isinstance(config_dict.get("mode"), str):
            config_dict["mode"] = MAKERMode(config_dict["mode"])
            
        return cls(**config_dict)


# Predefined MAKER Presets
MAKER_PRESETS = {
    "FAST": {
        "mode": MAKERMode.SEQUENTIAL,
        "k_ahead": 1,
        "num_candidates": 3,
        "max_depth": 2,
        "enable_red_flagging": False
    },
    "BALANCED": {
        "mode": MAKERMode.RECURSIVE,
        "k_ahead": 3,
        "num_candidates": 5,
        "max_depth": 5,
        "enable_red_flagging": True
    },
    "ZERO_ERROR": {
        "mode": MAKERMode.RECURSIVE,
        "k_ahead": 5,
        "num_candidates": 9,
        "max_depth": 10,
        "enable_red_flagging": True,
        "timeout_seconds": 600
    },
    "RESEARCH": {
        "mode": MAKERMode.HYBRID,
        "k_ahead": 2,
        "num_candidates": 4,
        "max_depth": 3,
        "enable_red_flagging": True
    }
}


# =============================================================================
# OPENEVOLVE-ADAPTED MAKER COMPONENTS
# =============================================================================

class OpenEvolveVoteCollector(VoteCollector):
    """
    Vote collector that uses OpenEvolveClient for LLM calls.

    Extends the core VoteCollector to work with OpenEvolve's API
    and client infrastructure.
    """

    def __init__(
        self,
        openevolve_client: Optional['OpenEvolveClient'] = None,
        openevolve_api: Optional['OpenEvolveAPI'] = None,
        max_token_length: int = 750,
        max_retries: int = 10,
        temperature_first: float = 0.0,
        temperature_subsequent: float = 0.1
    ):
        super().__init__(
            max_token_length=max_token_length,
            max_retries=max_retries,
            temperature_first=temperature_first,
            temperature_subsequent=temperature_subsequent
        )
        self.openevolve_client = openevolve_client
        self.openevolve_api = openevolve_api

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        agent: ModelConfig,
        temperature: float
    ) -> str:
        """
        Call LLM using OpenEvolveClient if available, otherwise fall back to direct API.

        This method provides three levels of fallback:
        1. OpenEvolveClient (preferred, uses evolution API)
        2. OpenEvolveAPI (HTTP-based)
        3. Direct LLM call (llm_utils._request_openai_compatible_chat)
        """
        # Try OpenEvolveClient first
        if self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                return self._call_via_openevolve_client(
                    prompt, system_prompt, agent, temperature
                )
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                logger.warning(f"OpenEvolveClient call failed: {e}, trying fallback")

        # Try OpenEvolveAPI (HTTP)
        if self.openevolve_api:
            try:
                return self._call_via_openevolve_api(
                    prompt, system_prompt, agent, temperature
                )
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                logger.warning(f"OpenEvolveAPI call failed: {e}, trying fallback")

        # Fall back to direct LLM call
        from llm_utils import _compose_messages, _request_openai_compatible_chat

        messages = _compose_messages(system_prompt, prompt)
        response = _request_openai_compatible_chat(
            api_key=agent.api_key,
            base_url=agent.api_base,
            model=agent.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_token_length
        )

        return response or ""

    def _call_via_openevolve_client(
        self,
        prompt: str,
        system_prompt: str,
        agent: ModelConfig,
        temperature: float
    ) -> str:
        """Call LLM via OpenEvolveClient evolution API."""
        # Build evolution config
        config = {
            "content": prompt,
            "system_prompt": system_prompt,
            "evolution_mode": "standard",
            "content_type": "analysis",
            "max_iterations": 1,
            "temperature": temperature,
            "max_tokens": self.max_token_length
        }

        # Call evolve
        result = self.openevolve_client.evolve(**config)

        if result.success and result.best_code:
            return result.best_code
        else:
            raise RuntimeError(f"OpenEvolveClient evolution failed: {result.error if hasattr(result, 'error') else 'Unknown error'}")

    def _call_via_openevolve_api(
        self,
        prompt: str,
        system_prompt: str,
        agent: ModelConfig,
        temperature: float
    ) -> str:
        """Call LLM via OpenEvolveAPI HTTP endpoint."""
        # This would use the OpenEvolveAPI's start_evolution method
        # For now, fall back to direct call
        raise NotImplementedError("OpenEvolveAPI HTTP calls not yet implemented for MAKER")


class OpenEvolveMAKEREngine(MAKEREngine):
    """
    MAKER engine that uses OpenEvolve client infrastructure.

    Extends the core MAKEREngine to work with OpenEvolve's
    client and API infrastructure.
    """

    def __init__(
        self,
        team: Team,
        k_ahead: int = 3,
        max_token_length: int = 750,
        max_steps: int = 1000,
        enable_first_to_ahead: bool = True,
        enable_red_flagging: bool = True,
        openevolve_client: Optional['OpenEvolveClient'] = None,
        openevolve_api: Optional['OpenEvolveAPI'] = None
    ):
        # Initialize with OpenEvolve-adapted vote collector
        vote_collector = OpenEvolveVoteCollector(
            openevolve_client=openevolve_client,
            openevolve_api=openevolve_api,
            max_token_length=max_token_length
        )

        voting_engine = VotingEngine(
            vote_collector=vote_collector,
            enable_first_to_ahead=enable_first_to_ahead
        )

        # Initialize parent class
        super().__init__(
            team=team,
            k_ahead=k_ahead,
            max_token_length=max_token_length,
            max_steps=max_steps,
            enable_first_to_ahead=enable_first_to_ahead,
            enable_red_flagging=enable_red_flagging
        )

        # Replace vote collector and voting engine with OpenEvolve versions
        self.vote_collector = vote_collector
        self.voting_engine = voting_engine
        self.openevolve_client = openevolve_client
        self.openevolve_api = openevolve_api


class OpenEvolveRecursiveMAKERSolver(RecursiveMAKERSolver):
    """
    Recursive MAKER solver that uses OpenEvolve client infrastructure.

    Extends the core RecursiveMAKERSolver to work with OpenEvolve's
    client and API infrastructure.
    """

    def __init__(
        self,
        team: Team,
        max_depth: int = 5,
        k_ahead: int = 3,
        num_candidates: int = 5,
        max_token_length: int = 750,
        openevolve_client: Optional['OpenEvolveClient'] = None,
        openevolve_api: Optional['OpenEvolveAPI'] = None
    ):
        # Initialize parent class
        super().__init__(
            team=team,
            max_depth=max_depth,
            k_ahead=k_ahead,
            num_candidates=num_candidates,
            max_token_length=max_token_length
        )

        # Replace vote collector with OpenEvolve version
        self.vote_collector = OpenEvolveVoteCollector(
            openevolve_client=openevolve_client,
            openevolve_api=openevolve_api,
            max_token_length=max_token_length
        )

        self.openevolve_client = openevolve_client
        self.openevolve_api = openevolve_api


# =============================================================================
# WORKFLOW INTEGRATION
# =============================================================================

class MAKERWorkflowIntegrator:
    """
    Integrates MAKER into the OpenEvolve decomposition workflow.

    This class provides the bridge between the workflow engine and MAKER,
    handling configuration, execution, and result processing.

    Integration Points (from Decomposition_Workflow.md):
    - Stage 0: Content Analysis (MAKER metadata in analyzed_context)
    - Stage 3: Sub-Problem Solving (MAKER execution via generate_solution_for_sub_problem)
    - Stage 4/5: Reassembly/Verification (MAKER outputs to gauntlet evaluation)
    """

    def __init__(
        self,
        config: MAKERWorkflowConfig,
        team: Optional[Team] = None,
        openevolve_client: Optional['OpenEvolveClient'] = None,
        openevolve_api: Optional['OpenEvolveAPI'] = None
    ):
        self.config = config
        self.team = team
        self.openevolve_client = openevolve_client
        self.openevolve_api = openevolve_api

        # Initialize appropriate engine
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the MAKER engine based on configuration mode."""
        if self.config.mode == MAKERMode.SEQUENTIAL:
            self.engine = OpenEvolveMAKEREngine(
                team=self.team or self._create_default_team(),
                k_ahead=self.config.k_ahead,
                max_token_length=self.config.max_token_length,
                max_steps=self.config.max_steps,
                enable_first_to_ahead=self.config.enable_first_to_ahead,
                enable_red_flagging=self.config.enable_red_flagging,
                openevolve_client=self.openevolve_client,
                openevolve_api=self.openevolve_api
            )
            self.solver = None

        elif self.config.mode == MAKERMode.RECURSIVE:
            self.engine = None
            self.solver = OpenEvolveRecursiveMAKERSolver(
                team=self.team or self._create_default_team(),
                max_depth=self.config.max_depth,
                k_ahead=self.config.k_ahead,
                num_candidates=self.config.num_candidates,
                max_token_length=self.config.max_token_length,
                openevolve_client=self.openevolve_client,
                openevolve_api=self.openevolve_api
            )

        elif self.config.mode == MAKERMode.HYBRID:
            # Hybrid mode: both sequential and recursive
            default_team = self.team or self._create_default_team()
            self.engine = OpenEvolveMAKEREngine(
                team=default_team,
                k_ahead=self.config.k_ahead,
                max_token_length=self.config.max_token_length,
                max_steps=self.config.max_steps,
                enable_first_to_ahead=self.config.enable_first_to_ahead,
                enable_red_flagging=self.config.enable_red_flagging,
                openevolve_client=self.openevolve_client,
                openevolve_api=self.openevolve_api
            )
            self.solver = OpenEvolveRecursiveMAKERSolver(
                team=default_team,
                max_depth=self.config.max_depth,
                k_ahead=self.config.k_ahead,
                num_candidates=self.config.num_candidates,
                max_token_length=self.config.max_token_length,
                openevolve_client=self.openevolve_client,
                openevolve_api=self.openevolve_api
            )
        else:
            raise ValueError(f"Unknown MAKER mode: {self.config.mode}")

        logger.info(f"Initialized MAKER in {self.config.mode.value} mode")

    def solve_subproblem(
        self,
        sub_problem: SubProblem,
        workflow_state: Optional[WorkflowState] = None
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using MAKER.

        This is the main integration point with the workflow engine,
        called from generate_solution_for_sub_problem().

        Args:
            sub_problem: The sub-problem to solve
            workflow_state: Current workflow state (optional)

        Returns:
            SolutionAttempt with the generated solution and metrics
        """
        logger.info(f"Solving sub-problem {sub_problem.id} with MAKER")

        start_time = time.time()

        try:
            # Build task and context from sub-problem
            task = self._build_task_from_subproblem(sub_problem)
            context = self._build_context_from_subproblem(sub_problem, workflow_state)

            # Execute based on mode
            if self.config.mode == MAKERMode.SEQUENTIAL:
                result = self._solve_sequential(sub_problem, task, context)
            elif self.config.mode == MAKERMode.RECURSIVE:
                result = self._solve_recursive(sub_problem, task, context)
            elif self.config.mode == MAKERMode.HYBRID:
                result = self._solve_hybrid(sub_problem, task, context)
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

            # Create solution attempt
            solution_attempt = self._create_solution_attempt(
                sub_problem=sub_problem,
                result=result,
                execution_time=time.time() - start_time
            )

            logger.info(f"Successfully solved sub-problem {sub_problem.id} with MAKER")
            return solution_attempt

        except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
            logger.error(f"MAKER failed for sub-problem {sub_problem.id}: {e}", exc_info=True)

            # Return failed solution attempt
            return SolutionAttempt(
                sub_problem_id=sub_problem.id,
                team_id=self.team.team_id if self.team else "maker_default",
                content="",
                metadata={
                    "error": str(e),
                    "mode": self.config.mode.value,
                    "execution_time": time.time() - start_time
                }
            )

    def _solve_sequential(
        self,
        sub_problem: SubProblem,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve using sequential MAKER (Algorithm 1)."""
        if not self.engine:
            raise RuntimeError("Sequential engine not initialized")

        # Build prompt template
        def prompt_template(state):
            return f"""Task: {task}

Context: {json.dumps(context, indent=2)}

Current state: {json.dumps(state, indent=2)}

Determine the next action."""

        # System prompt
        system_prompt = f"""You are solving a sub-problem: {sub_problem.title}

Description: {sub_problem.description}

Follow these steps:
1. Analyze the current state
2. Determine the next action
3. Respond in the specified format

Provide your response:"""

        # Execute
        action_list, final_state, metrics = self.engine.generate_solution(
            initial_state=context.get("initial_state", {}),
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            stop_condition=lambda s: s.get("done", False)
        )

        return {
            "actions": action_list,
            "final_state": final_state,
            "metrics": metrics,
            "mode": "sequential"
        }

    def _solve_recursive(
        self,
        sub_problem: SubProblem,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve using recursive MAKER (Algorithm 4)."""
        if not self.solver:
            raise RuntimeError("Recursive solver not initialized")

        # Execute
        solution, metrics = self.solver.solve(
            task=task,
            context=context,
            max_depth=self.config.max_depth
        )

        return {
            "solution": solution,
            "metrics": metrics,
            "mode": "recursive"
        }

    def _solve_hybrid(
        self,
        sub_problem: SubProblem,
        task: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve using hybrid MAKER (ROMA + MAKER)."""
        # Try ROMA decomposition first
        try:
            from roma_mcp_tools import analyze_with_roma

            roma_result = analyze_with_roma(
                task=task,
                max_depth=3,
                execution_mode="recursive"
            )

            if roma_result.get("decomposition"):
                # Apply MAKER voting to ROMA hierarchy
                if self.solver:
                    solution, metrics = self.solver.solve(
                        task=task,
                        context={**context, "roma_decomposition": roma_result["decomposition"]},
                        max_depth=self.config.max_depth
                    )

                    return {
                        "solution": solution,
                        "metrics": metrics,
                        "mode": "hybrid",
                        "roma_used": True
                    }
        except (ImportError, Exception) as e:
            logger.warning(f"ROMA analysis failed: {e}, falling back to recursive")

        # Fallback to recursive
        return self._solve_recursive(sub_problem, task, context)

    def _build_task_from_subproblem(self, sub_problem: SubProblem) -> str:
        """Build task description from sub-problem."""
        task = f"""{sub_problem.title}

{sub_problem.description}"""

        if sub_problem.success_criteria:
            criteria_str = "\n".join([f"- {c.description}" for c in sub_problem.success_criteria])
            task += f"\n\nSuccess Criteria:\n{criteria_str}"

        return task

    def _build_context_from_subproblem(
        self,
        sub_problem: SubProblem,
        workflow_state: Optional[WorkflowState]
    ) -> Dict[str, Any]:
        """Build context dict from sub-problem and workflow state."""
        context = {
            "sub_problem_id": sub_problem.id,
            "sub_problem_title": sub_problem.title,
            "priority": sub_problem.priority,
            "estimated_effort": sub_problem.estimated_effort,
            "type": sub_problem.type.value if hasattr(sub_problem.type, 'value') else str(sub_problem.type)
        }

        # Add workflow state context
        if workflow_state:
            context["workflow_id"] = workflow_state.workflow_id
            context["parent_problem"] = workflow_state.problem_title

            # Add MDAP config if present
            if hasattr(workflow_state, 'mdap_config') and workflow_state.mdap_config:
                context["mdap_enabled"] = True

        return context

    def _create_solution_attempt(
        self,
        sub_problem: SubProblem,
        result: Dict[str, Any],
        execution_time: float
    ) -> SolutionAttempt:
        """Create SolutionAttempt from MAKER result."""
        # Extract solution content
        if "solution" in result:
            content = json.dumps(result["solution"], indent=2)
        elif "actions" in result:
            content = json.dumps(result["actions"], indent=2)
        else:
            content = str(result)

        # Extract metrics
        metrics = result.get("metrics")
        if metrics and hasattr(metrics, '__dict__'):
            metrics_dict = metrics.__dict__.copy()
        else:
            metrics_dict = {}

        # Create solution attempt
        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            team_id=self.team.team_id if self.team else "maker_default",
            content=content,
            metadata={
                "maker_mode": result.get("mode", self.config.mode.value),
                "execution_time": execution_time,
                "k_ahead": self.config.k_ahead,
                "enable_red_flagging": self.config.enable_red_flagging,
                **metrics_dict
            }
        )

    def _create_default_team(self) -> Team:
        """Create default team if not provided."""
        from llm_utils import get_api_key

        model_config = ModelConfig(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            api_key=get_api_key("openai") or os.getenv("OPENAI_API_KEY", ""),
            temperature=0.1
        )

        return Team(
            team_id="maker_default",
            name="MAKER Default Team",
            members=[model_config],
            description="Default team for MAKER execution"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_maker_config_from_workflow(
    workflow_state: WorkflowState,
    sub_problem: Optional[SubProblem] = None
) -> MAKERWorkflowConfig:
    """
    Create MAKER configuration from workflow state.

    This extracts MAKER-specific configuration from the workflow state
    and sub-problem metadata.

    Args:
        workflow_state: Current workflow state
        sub_problem: Optional sub-problem for specific configuration

    Returns:
        MAKERWorkflowConfig object
    """
    # Extract MAKER config from workflow state metadata
    maker_config_dict = workflow_state.metadata.get("maker_config", {})

    # Override with sub-problem specific config
    if sub_problem and hasattr(sub_problem, 'metadata'):
        sub_problem_config = sub_problem.metadata.get("maker_config", {})
        maker_config_dict.update(sub_problem_config)

    # Create config object
    return MAKERWorkflowConfig(
        mode=MAKERMode(maker_config_dict.get("mode", "recursive")),
        k_ahead=maker_config_dict.get("k_ahead", 3),
        max_depth=maker_config_dict.get("max_depth", 5),
        enable_red_flagging=maker_config_dict.get("enable_red_flagging", True),
        max_token_length=maker_config_dict.get("max_token_length", 750)
    )


def create_maker_integrator(
    workflow_state: WorkflowState,
    team: Optional[Team] = None
) -> MAKERWorkflowIntegrator:
    """
    Create MAKER workflow integrator from workflow state.

    This is the main factory function used by the workflow engine.

    Args:
        workflow_state: Current workflow state
        team: Optional team (will use default if not provided)

    Returns:
        MAKERWorkflowIntegrator instance
    """
    # Get OpenEvolve client if available
    openevolve_client = None
    if OPENEVOLVE_AVAILABLE:
        try:
            openevolve_client = OpenEvolveClient()
        except (RuntimeError, ValueError, ConnectionError, ImportError) as e:
            logger.warning(f"Failed to create OpenEvolveClient: {e}")

    # Create config
    config = create_maker_config_from_workflow(workflow_state)

    # Create integrator
    return MAKERWorkflowIntegrator(
        config=config,
        team=team,
        openevolve_client=openevolve_client
    )


# =============================================================================
# WORKFLOW ENGINE INTEGRATION FUNCTIONS
# =============================================================================

def solve_subproblem_with_maker(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    team: Optional[Team] = None
) -> SolutionAttempt:
    """
    Main entry point for solving sub-problems with MAKER.

    This function is called from workflow_engine.py's generate_solution_for_sub_problem()
    when MAKER mode is enabled.

    Args:
        sub_problem: The sub-problem to solve
        workflow_state: Current workflow state
        team: Optional team (will use default if not provided)

    Returns:
        SolutionAttempt with the generated solution
    """
    # Create MAKER integrator
    integrator = create_maker_integrator(workflow_state, team)

    # Solve sub-problem
    return integrator.solve_subproblem(sub_problem, workflow_state)


def get_maker_status() -> Dict[str, Any]:
    """Get MAKER system status for workflow UI."""
    return {
        "maker_available": True,
        "openevolve_available": OPENEVOLVE_AVAILABLE,
        "supported_modes": [m.value for m in MAKERMode],
        "default_mode": MAKERMode.RECURSIVE.value,
        "algorithms": [
            "Algorithm 1: generate_solution",
            "Algorithm 2: do_voting",
            "Algorithm 3: get_vote",
            "Algorithm 4: Recursive solve"
        ],
        "features": [
            "Maximal Agentic Decomposition (MAD)",
            "First-to-ahead-by-k Error Correction",
            "Red-flagging",
            "OpenEvolve Client Integration",
            "Workflow State Integration"
        ]
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    "MAKERWorkflowIntegrator",
    "OpenEvolveMAKEREngine",
    "OpenEvolveRecursiveMAKERSolver",
    "OpenEvolveVoteCollector",

    # Configuration
    "MAKERWorkflowConfig",
    "MAKERMode",
    "MAKER_PRESETS",

    # Factory functions
    "create_maker_config_from_workflow",
    "create_maker_integrator",
    "solve_subproblem_with_maker",
    "get_maker_status",
]
