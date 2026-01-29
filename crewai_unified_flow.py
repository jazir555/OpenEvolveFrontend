"""
CrewAI Unified Flow - Complete Replacement for Hephaestus Unified Bridge

This module provides the unified flow for CrewAI that replaces the Hephaestus
service-based architecture with local event-driven workflow execution.

Architecture:
    User → CrewAI Flow → Execution Method (Traditional/ROMA/ROMA-MDAP-MAKER/Claudiomiro/DataPizza/Hybrid)

Execution Methods:
1. Traditional - AI-assisted decomposition with OpenEvolve
2. ROMA - Recursive meta-agent decomposition
3. ROMA-MDAP-MAKER - Recursive decomposition + Zero-error voting (NEW)
4. Claudiomiro - Autonomous development CLI
5. DataPizza - Multi-agent coordination
6. Hybrid - ROMA + Decomposition Workflow teams
7. Auto - Intelligent selection

License: MIT (replaces AGPL Hephaestus)
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

# CrewAI imports
try:
    from crewai import Flow, start, listen, router
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Mock decorators for standalone operation
    def start(func):
        return func
    def listen(func):
        return func
    def router(func):
        return func

# Import execution method configurations
from roma_config import (
    CrewAIROMAConfig,
    ROMAPhase1Config,
    ROMAPhase2Config,
    ROMAPhase3Config,
    ROMAPhase4Config,
    ROMAPhase5Config,
    ROMAPhase6Config,
    ROMAHybridConfig,
    ROMAConfigBuilder,
    ROMAConfigPresets,
)

from datapizza_config import (
    CrewAIDataPizzaConfig,
    DataPizzaPhase1Config,
    DataPizzaPhase2Config,
    DataPizzaPhase3Config,
    DataPizzaPhase4Config,
    DataPizzaMultiAgentConfig,
    DataPizzaConfigBuilder,
    DataPizzaConfigPresets,
)

from claudiomiro_config import (
    CrewAIClaudiomiroConfig,
    ClaudiomiroPhase1Config,
    ClaudiomiroPhase2Config,
    ClaudiomiroPhase3Config,
    ClaudiomiroPhase4Config,
    ClaudiomiroPhase5Config,
    ClaudiomiroPhase6Config,
    ClaudiomiroMultiRepoConfig,
    ClaudiomiroConfigBuilder,
    ClaudiomiroConfigPresets,
)

# Import bridge functions (will be ported from Hephaestus bridges)
try:
    from decomposition_crewai_bridge import (
        execute_phase_1_setup as decomposition_phase_1_setup,
        execute_phase_2_solve as decomposition_phase_2_solve,
        execute_phase_3_critique as decomposition_phase_3_critique,
        execute_phase_4_verify as decomposition_phase_4_verify,
        execute_phase_5_reassemble as decomposition_phase_5_reassemble,
        execute_phase_6_final_validation as decomposition_phase_6_final_validation,
    )
    DECOMPOSITION_BRIDGE_AVAILABLE = True
except ImportError:
    DECOMPOSITION_BRIDGE_AVAILABLE = False
    decomposition_phase_1_setup = None
    decomposition_phase_2_solve = None
    decomposition_phase_3_critique = None
    decomposition_phase_4_verify = None
    decomposition_phase_5_reassemble = None
    decomposition_phase_6_final_validation = None

try:
    from roma_crewai_bridge import (
        execute_phase_1_setup as roma_phase_1_setup,
        execute_phase_2_solve as roma_phase_2_solve,
        execute_phase_3_critique as roma_phase_3_critique,
        execute_phase_4_verify as roma_phase_4_verify,
        execute_phase_5_reassemble as roma_phase_5_reassemble,
        execute_phase_6_final_validation as roma_phase_6_final_validation,
        execute_full_workflow as roma_full_workflow,
    )
    ROMA_BRIDGE_AVAILABLE = True
except ImportError:
    ROMA_BRIDGE_AVAILABLE = False
    roma_phase_1_setup = None
    roma_phase_2_solve = None
    roma_phase_3_critique = None
    roma_phase_4_verify = None
    roma_phase_5_reassemble = None
    roma_phase_6_final_validation = None
    roma_full_workflow = None

try:
    from roma_mdap_maker_crewai_bridge import (
        execute_phase_1_setup as roma_mdap_maker_phase_1_setup,
        execute_phase_2_solve as roma_mdap_maker_phase_2_solve,
        execute_phase_3_critique as roma_mdap_maker_phase_3_critique,
        execute_phase_4_verify as roma_mdap_maker_phase_4_verify,
        execute_phase_5_reassemble as roma_mdap_maker_phase_5_reassemble,
        execute_phase_6_final_validation as roma_mdap_maker_phase_6_final_validation,
        execute_full_workflow as roma_mdap_maker_full_workflow,
        get_romamdapmaker_bridge_status,
    )
    ROMA_MDAP_MAKER_BRIDGE_AVAILABLE = True
except ImportError:
    ROMA_MDAP_MAKER_BRIDGE_AVAILABLE = False
    roma_mdap_maker_phase_1_setup = None
    roma_mdap_maker_phase_2_solve = None
    roma_mdap_maker_phase_3_critique = None
    roma_mdap_maker_phase_4_verify = None
    roma_mdap_maker_phase_5_reassemble = None
    roma_mdap_maker_phase_6_final_validation = None
    roma_mdap_maker_full_workflow = None

try:
    from datapizza_crewai_bridge import (
        execute_phase_1_setup as datapizza_phase_1_setup,
        execute_phase_2_solve as datapizza_phase_2_solve,
        execute_phase_3_critique as datapizza_phase_3_critique,
        execute_phase_4_verify as datapizza_phase_4_verify,
        execute_full_workflow as datapizza_full_workflow,
    )
    DATAPIZZA_BRIDGE_AVAILABLE = True
except ImportError:
    DATAPIZZA_BRIDGE_AVAILABLE = False
    datapizza_phase_1_setup = None
    datapizza_phase_2_solve = None
    datapizza_phase_3_critique = None
    datapizza_phase_4_verify = None
    datapizza_full_workflow = None

try:
    from claudiomiro_crewai_bridge import (
        ClaudiomiroCrewAIWorkflowBridge,
        CLAUDIOMIRO_AVAILABLE,
    )
    CLAUDIOMIRO_BRIDGE_AVAILABLE = True
except ImportError:
    CLAUDIOMIRO_BRIDGE_AVAILABLE = False
    CLAUDIOMIRO_AVAILABLE = False
    ClaudiomiroCrewAIWorkflowBridge = None

logger = logging.getLogger(__name__)


class ExecutionMethod(str, Enum):
    """Available execution methods"""
    TRADITIONAL = "traditional"
    ROMA = "roma"
    ROMA_MDAP_MAKER = "roma_mdap_maker"  # ZERO-ERROR
    CLAUDIOMIRO = "claudiomiro"
    DATAPIZZA = "datapizza"
    HYBRID = "hybrid"
    AUTO = "auto"


class CrewAIUnifiedFlow:
    """
    Unified CrewAI flow that replaces Hephaestus unified bridge.

    Provides:
    - Event-driven workflow execution
    - Intelligent execution method routing
    - State management with Pydantic models
    - Local execution (no external service dependencies)
    - Integration with all execution methods
    """

    def __init__(
        self,
        default_execution_method: ExecutionMethod = ExecutionMethod.AUTO,
        enable_persistence: bool = True,
        state_storage_dir: str = "./crewai_states"
    ):
        """
        Initialize CrewAI unified flow.

        Args:
            default_execution_method: Default execution method
            enable_persistence: Enable state persistence to disk
            state_storage_dir: Directory for state storage
        """
        self.default_execution_method = default_execution_method
        self.enable_persistence = enable_persistence
        self.state_storage_dir = state_storage_dir

        # Import state manager if needed
        if enable_persistence:
            from crewai_state_management import StateManager
            self.state_manager = StateManager(state_storage_dir)
        else:
            self.state_manager = None

        logger.info(f"CrewAIUnifiedFlow initialized with default_method={default_execution_method}")

    @start
    def phase_1_setup(
        self,
        problem_statement: str,
        execution_method: ExecutionMethod = ExecutionMethod.AUTO,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
        max_sub_problems: int = 15,
        decomposition_strategy: str = "semantic",
        use_evolution: bool = True,
        evolution_iterations: int = 50,
        # ROMA-specific parameters
        roma_max_depth: int = 3,
        roma_execution_mode: str = "recursive",
        roma_provider: Optional[str] = None,
        roma_model: Optional[str] = None,
        # ROMA-MDAP-MAKER parameters
        use_roma_mdap_maker: bool = False,
        reliability_preset: str = "standard",
        reliability_overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Phase 1: Problem Setup - Entry point for all workflows

        This is the @start decorated entry point that begins all workflows.
        """
        logger.info(f"Phase 1: CrewAI setup (method={execution_method})")
        logger.info(f"  Problem: {problem_statement[:100]}...")

        # Get reliability config if using ROMA-MDAP-MAKER
        reliability_config = None
        if execution_method == ExecutionMethod.ROMA_MDAP_MAKER:
            from roma_mdap_maker_reliability_ssot import get_reliability_config
            reliability_config = get_reliability_config(
                preset=reliability_preset,
                **(reliability_overrides or {})
            )

        # Auto-selection
        if execution_method == ExecutionMethod.AUTO:
            execution_method = self._select_execution_method(
                problem_statement,
                use_roma_mdap_maker
            )
            logger.info(f"  Auto-selected: {execution_method}")

        # Route to appropriate execution method
        if execution_method == ExecutionMethod.ROMA_MDAP_MAKER:
            if ROMA_MDAP_MAKER_BRIDGE_AVAILABLE and roma_mdap_maker_phase_1_setup:
                result = roma_mdap_maker_phase_1_setup(
                    problem_statement=problem_statement,
                    reliability_config=reliability_config,
                    **kwargs
                )
            else:
                logger.warning("ROMA-MDAP-MAKER bridge not available, falling back to ROMA")
                execution_method = ExecutionMethod.ROMA

        if execution_method == ExecutionMethod.ROMA:
            if ROMA_BRIDGE_AVAILABLE and roma_phase_1_setup:
                result = roma_phase_1_setup(
                    problem_statement=problem_statement,
                    max_depth=roma_max_depth,
                    execution_mode=roma_execution_mode,
                    provider=roma_provider,
                    model=roma_model,
                )
            else:
                logger.warning("ROMA bridge not available, falling back to traditional")
                execution_method = ExecutionMethod.TRADITIONAL

        if execution_method == ExecutionMethod.DATAPIZZA:
            if DATAPIZZA_BRIDGE_AVAILABLE and datapizza_phase_1_setup:
                result = datapizza_phase_1_setup(
                    problem_statement=problem_statement,
                    provider=kwargs.get("provider", "openai"),
                    api_key=kwargs.get("api_key"),
                    model=kwargs.get("model"),
                    enable_web_search=kwargs.get("enable_web_search", True),
                    planning_interval=kwargs.get("planning_interval", 3),
                    max_steps=kwargs.get("max_steps", 15),
                )
            else:
                logger.warning("DataPizza bridge not available, falling back to traditional")
                execution_method = ExecutionMethod.TRADITIONAL

        if execution_method == ExecutionMethod.CLAUDIOMIRO:
            if CLAUDIOMIRO_BRIDGE_AVAILABLE and CLAUDIOMIRO_AVAILABLE and ClaudiomiroCrewAIWorkflowBridge:
                bridge = self._get_claudiomiro_bridge(**kwargs)
                result = bridge.execute_phase_1_setup(
                    problem_statement=problem_statement,
                    problem_type=problem_type,
                    domain=domain,
                )
                result["status"] = "completed" if result.get("success") else "failed"
            else:
                logger.warning("Claudiomiro bridge not available, falling back to traditional")
                execution_method = ExecutionMethod.TRADITIONAL

        if execution_method == ExecutionMethod.HYBRID:
            execution_method = ExecutionMethod.TRADITIONAL

        if execution_method == ExecutionMethod.TRADITIONAL:
            if DECOMPOSITION_BRIDGE_AVAILABLE and decomposition_phase_1_setup:
                result = decomposition_phase_1_setup(
                    problem_statement=problem_statement,
                    problem_type=problem_type,
                    domain=domain,
                    max_sub_problems=max_sub_problems,
                    decomposition_strategy=decomposition_strategy,
                    use_evolution=use_evolution,
                    evolution_iterations=evolution_iterations,
                )
            else:
                raise NotImplementedError("Traditional decomposition bridge not available")

        # Store result if persistence enabled
        if self.enable_persistence and self.state_manager:
            workflow_id = result.get('workflow_id', f"workflow_{datetime.now().timestamp()}")
            from crewai_state_management import WorkflowState
            state = WorkflowState(
                phase=1,
                status="completed",
                execution_method=execution_method,
                problem_statement=problem_statement,
                metadata=result
            )
            self.state_manager.save_state(workflow_id, state)
            result['workflow_id'] = workflow_id

        result["execution_method"] = execution_method.value
        return result

    # Note: In a full CrewAI event-driven implementation, this would use @listen decorator
    # For now, this method should be called manually after phase_1_setup completes
    def phase_2_solve(
        self,
        phase_1_result: Dict[str, Any],
        team_name: Optional[str] = None,
        solve_subset: Optional[List[str]] = None,
        use_evolution: bool = True,
        evolution_iterations: int = 100,
        # Additional execution parameters...
        **kwargs
    ) -> Dict[str, Any]:
        """
        Phase 2: Solution Generation

        This @listen decorated method automatically receives phase 1 output.
        """
        logger.info("Phase 2: CrewAI solve")

        execution_method = self._normalize_execution_method(
            phase_1_result.get('execution_method')
        )
        decomposition_plan = self._normalize_decomposition_plan(
            phase_1_result.get('decomposition_plan')
        )

        # Route to appropriate execution method
        if execution_method == ExecutionMethod.ROMA_MDAP_MAKER:
            if ROMA_MDAP_MAKER_BRIDGE_AVAILABLE and roma_mdap_maker_phase_2_solve:
                # Solve each sub-problem with ROMA-MDAP-MAKER
                sub_problems = decomposition_plan.get('sub_problems', [])
                solutions = []
                for sub_problem in sub_problems:
                    solution = roma_mdap_maker_phase_2_solve(
                        sub_problem_id=sub_problem['id'],
                        sub_problem_description=sub_problem['description'],
                        context=phase_1_result,
                        **kwargs
                    )
                    solutions.append({
                        "id": sub_problem['id'],
                        "solution": solution.get("solution") or solution.get("result", ""),
                        "confidence": solution.get("confidence", 0.5),
                        "raw": solution,
                    })
                return {
                    "phase": 2,
                    "status": "completed",
                    "solutions": solutions,
                    "execution_method": execution_method.value,
                }

        elif execution_method == ExecutionMethod.ROMA:
            if ROMA_BRIDGE_AVAILABLE and roma_phase_2_solve:
                result = roma_phase_2_solve(
                    sub_problems=decomposition_plan.get("sub_problems", []),
                    **kwargs
                )
                if isinstance(result, dict):
                    result.setdefault("execution_method", execution_method.value)
                return result

        elif execution_method == ExecutionMethod.DATAPIZZA:
            if DATAPIZZA_BRIDGE_AVAILABLE and datapizza_phase_2_solve:
                sub_problems = phase_1_result.get("sub_problems") or decomposition_plan.get("sub_problems", [])
                if not sub_problems:
                    sub_problems = [{"id": "sp_1", "description": phase_1_result.get("problem_statement", "")}]
                result = datapizza_phase_2_solve(
                    sub_problems=sub_problems,
                    team_name=team_name,
                    solve_subset=solve_subset,
                    provider=kwargs.get("provider", "openai"),
                    api_key=kwargs.get("api_key"),
                    model=kwargs.get("model"),
                    working_directory=kwargs.get("working_directory"),
                    enable_filesystem=kwargs.get("enable_filesystem", True),
                    planning_interval=kwargs.get("planning_interval", 3),
                    max_steps=kwargs.get("max_steps", 20),
                )
                if isinstance(result, dict):
                    result.setdefault("execution_method", execution_method.value)
                return result

        elif execution_method == ExecutionMethod.CLAUDIOMIRO:
            if CLAUDIOMIRO_BRIDGE_AVAILABLE and CLAUDIOMIRO_AVAILABLE and ClaudiomiroCrewAIWorkflowBridge:
                bridge = self._get_claudiomiro_bridge(**kwargs)
                sub_tasks = phase_1_result.get("sub_tasks") or []
                result = bridge.execute_phase_2_solution(
                    problem_statement=phase_1_result.get("problem_statement", ""),
                    sub_problems=sub_tasks,
                    backend=kwargs.get("backend"),
                    frontend=kwargs.get("frontend"),
                    enable_parallel=kwargs.get("enable_parallel", True),
                )
                if isinstance(result, dict):
                    result.setdefault("execution_method", execution_method.value)
                return result

        # Fallback to traditional
        if DECOMPOSITION_BRIDGE_AVAILABLE and decomposition_phase_2_solve:
            result = decomposition_phase_2_solve(
                decomposition_plan=decomposition_plan,
                team_name=team_name,
                solve_subset=solve_subset,
                use_evolution=use_evolution,
                evolution_iterations=evolution_iterations,
                **kwargs
            )
            if isinstance(result, dict):
                result.setdefault("execution_method", ExecutionMethod.TRADITIONAL.value)
            return result

        raise NotImplementedError(f"Phase 2 not implemented for {execution_method}")

    def phase_3_critique(
        self,
        phase_2_result: Dict[str, Any],
        execution_method: Optional[ExecutionMethod] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Phase 3: Critique results from Phase 2."""
        solutions = self._extract_solutions(phase_2_result)
        method = execution_method or self._normalize_execution_method(
            phase_2_result.get("execution_method") or phase_2_result.get("execution_method_phase2")
        )

        if method == ExecutionMethod.ROMA_MDAP_MAKER and ROMA_MDAP_MAKER_BRIDGE_AVAILABLE:
            return roma_mdap_maker_phase_3_critique(solutions, **kwargs)
        if method == ExecutionMethod.ROMA and ROMA_BRIDGE_AVAILABLE:
            return roma_phase_3_critique(solutions, **kwargs)
        if method == ExecutionMethod.DATAPIZZA and DATAPIZZA_BRIDGE_AVAILABLE:
            return datapizza_phase_3_critique(solutions, **kwargs)
        if method == ExecutionMethod.CLAUDIOMIRO and CLAUDIOMIRO_BRIDGE_AVAILABLE and CLAUDIOMIRO_AVAILABLE:
            bridge = self._get_claudiomiro_bridge(**kwargs)
            return bridge.execute_phase_3_critique(solutions, **kwargs)
        if DECOMPOSITION_BRIDGE_AVAILABLE and decomposition_phase_3_critique:
            return decomposition_phase_3_critique(
                solutions=solutions,
                problem_statement=kwargs.get("problem_statement"),
                **kwargs
            )
        raise NotImplementedError(f"Phase 3 not implemented for {method}")

    def phase_4_verify(
        self,
        phase_2_result: Dict[str, Any],
        critiques: Optional[Dict[str, Any]] = None,
        execution_method: Optional[ExecutionMethod] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Phase 4: Verification."""
        solutions = self._extract_solutions(phase_2_result)
        method = execution_method or self._normalize_execution_method(
            phase_2_result.get("execution_method") or phase_2_result.get("execution_method_phase2")
        )

        if method == ExecutionMethod.ROMA_MDAP_MAKER and ROMA_MDAP_MAKER_BRIDGE_AVAILABLE:
            return roma_mdap_maker_phase_4_verify(solutions, **kwargs)
        if method == ExecutionMethod.ROMA and ROMA_BRIDGE_AVAILABLE:
            return roma_phase_4_verify(solutions, **kwargs)
        if method == ExecutionMethod.DATAPIZZA and DATAPIZZA_BRIDGE_AVAILABLE:
            return datapizza_phase_4_verify(
                solutions=solutions,
                critiques=critiques.get("critiques", []) if critiques else [],
                **kwargs
            )
        if method == ExecutionMethod.CLAUDIOMIRO and CLAUDIOMIRO_BRIDGE_AVAILABLE and CLAUDIOMIRO_AVAILABLE:
            bridge = self._get_claudiomiro_bridge(**kwargs)
            return bridge.execute_phase_4_verify(
                solutions=solutions,
                test_command=kwargs.get("test_command", "npm test"),
                **kwargs
            )
        if DECOMPOSITION_BRIDGE_AVAILABLE and decomposition_phase_4_verify:
            return decomposition_phase_4_verify(
                solutions=solutions,
                requirements=kwargs.get("requirements"),
                **kwargs
            )
        raise NotImplementedError(f"Phase 4 not implemented for {method}")

    def phase_5_reassemble(
        self,
        phase_2_result: Dict[str, Any],
        problem_statement: str,
        execution_method: Optional[ExecutionMethod] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Phase 5: Reassembly."""
        solutions = self._extract_solutions(phase_2_result)
        method = execution_method or self._normalize_execution_method(
            phase_2_result.get("execution_method") or phase_2_result.get("execution_method_phase2")
        )

        if method == ExecutionMethod.ROMA_MDAP_MAKER and ROMA_MDAP_MAKER_BRIDGE_AVAILABLE:
            return roma_mdap_maker_phase_5_reassemble(solutions, problem_statement, **kwargs)
        if method == ExecutionMethod.ROMA and ROMA_BRIDGE_AVAILABLE:
            return roma_phase_5_reassemble(solutions, problem_statement)
        if method == ExecutionMethod.CLAUDIOMIRO and CLAUDIOMIRO_BRIDGE_AVAILABLE and CLAUDIOMIRO_AVAILABLE:
            bridge = self._get_claudiomiro_bridge(**kwargs)
            return bridge.execute_phase_5_reassemble(
                sub_solutions=solutions,
                problem_statement=problem_statement,
                backend=kwargs.get("backend"),
                frontend=kwargs.get("frontend"),
            )
        if DECOMPOSITION_BRIDGE_AVAILABLE and decomposition_phase_5_reassemble:
            return decomposition_phase_5_reassemble(
                solutions=solutions,
                problem_statement=problem_statement,
                **kwargs
            )
        raise NotImplementedError(f"Phase 5 not implemented for {method}")

    def phase_6_final_validation(
        self,
        final_solution: str,
        problem_statement: str,
        execution_method: Optional[ExecutionMethod] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Phase 6: Final validation."""
        method = execution_method or self._normalize_execution_method(
            kwargs.get("execution_method")
        )

        if method == ExecutionMethod.ROMA_MDAP_MAKER and ROMA_MDAP_MAKER_BRIDGE_AVAILABLE:
            return roma_mdap_maker_phase_6_final_validation(final_solution, problem_statement, **kwargs)
        if method == ExecutionMethod.ROMA and ROMA_BRIDGE_AVAILABLE:
            return roma_phase_6_final_validation(final_solution, problem_statement)
        if method == ExecutionMethod.CLAUDIOMIRO and CLAUDIOMIRO_BRIDGE_AVAILABLE and CLAUDIOMIRO_AVAILABLE:
            bridge = self._get_claudiomiro_bridge(**kwargs)
            return bridge.execute_phase_6_final(
                final_solution=final_solution,
                problem_statement=problem_statement,
                target_branch=kwargs.get("target_branch", "main"),
                create_pr=kwargs.get("create_pr", True),
            )
        if DECOMPOSITION_BRIDGE_AVAILABLE and decomposition_phase_6_final_validation:
            return decomposition_phase_6_final_validation(
                final_solution=final_solution,
                problem_statement=problem_statement,
                **kwargs
            )
        raise NotImplementedError(f"Phase 6 not implemented for {method}")

    @router
    def _select_execution_method(
        self,
        problem_statement: str,
        use_roma_mdap_maker: bool
    ) -> ExecutionMethod:
        """
        Intelligent execution method selection based on problem analysis.

        This @router decorated method analyzes the problem and routes to the
        optimal execution method.
        """
        problem_lower = problem_statement.lower()

        # Priority 1: Zero-error critical tasks
        zero_error_keywords = [
            "critical", "zero error", "flawless", "perfect",
            "mission-critical", "safety-critical", "high-reliability",
            "life-critical", "medical", "financial", "legal compliance"
        ]
        if use_roma_mdap_maker and any(kw in problem_lower for kw in zero_error_keywords):
            logger.info("  Routing to ROMA-MDAP-MAKER (zero-error critical)")
            return ExecutionMethod.ROMA_MDAP_MAKER

        # Priority 2: Hierarchical decomposition
        decomposition_keywords = [
            "decompose", "break down", "hierarchical", "recursive",
            "complex structure", "nested", "multi-level"
        ]
        if any(kw in problem_lower for kw in decomposition_keywords):
            logger.info("  Routing to ROMA (hierarchical decomposition)")
            return ExecutionMethod.ROMA

        # Priority 3: Multi-agent coordination
        multi_agent_keywords = [
            "multi-agent", "coordination", "distributed system",
            "team collaboration", "swarm"
        ]
        if any(kw in problem_lower for kw in multi_agent_keywords):
            logger.info("  Routing to DataPizza (multi-agent)")
            return ExecutionMethod.DATAPIZZA

        # Priority 4: CLI/code generation
        cli_keywords = [
            "cli", "command line", "code generation", "autonomous",
            "development", "programming"
        ]
        if any(kw in problem_lower for kw in cli_keywords):
            logger.info("  Routing to Claudiomiro (CLI/development)")
            return ExecutionMethod.CLAUDIOMIRO

        # Default: Traditional
        logger.info("  Routing to Traditional (simple task)")
        return ExecutionMethod.TRADITIONAL

    def execute_full_workflow(
        self,
        problem_statement: str,
        execution_method: ExecutionMethod = ExecutionMethod.AUTO,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute complete 6-phase workflow.

        This is the main entry point that most applications will use.
        """
        logger.info(f"Starting full workflow: {problem_statement[:100]}...")

        # Phase 1: Setup
        phase1_result = self.phase_1_setup(
            problem_statement=problem_statement,
            execution_method=execution_method,
            **kwargs
        )
        selected_method = self._normalize_execution_method(
            phase1_result.get("execution_method") or execution_method
        )

        # Phase 2: Solve
        phase2_result = self.phase_2_solve(
            phase_1_result=phase1_result,
            **kwargs
        )

        # Phase 3: Critique
        phase3_result = self.phase_3_critique(
            phase_2_result=phase2_result,
            execution_method=selected_method,
            problem_statement=problem_statement,
            **kwargs
        )

        # Phase 4: Verify
        phase4_result = self.phase_4_verify(
            phase_2_result=phase2_result,
            critiques=phase3_result,
            execution_method=selected_method,
            **kwargs
        )

        # Phase 5: Reassemble
        phase5_result = self.phase_5_reassemble(
            phase_2_result=phase2_result,
            problem_statement=problem_statement,
            execution_method=selected_method,
            **kwargs
        )

        # Phase 6: Final validation
        phase6_result = self.phase_6_final_validation(
            final_solution=phase5_result.get("final_solution", ""),
            problem_statement=problem_statement,
            execution_method=selected_method,
            **kwargs
        )

        return {
            "workflow": "unified_crewai",
            "status": "completed",
            "phases": {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "phase4": phase4_result,
                "phase5": phase5_result,
                "phase6": phase6_result,
            },
            "final_solution": phase5_result.get("final_solution"),
            "message": "Workflow completed",
        }

    def _normalize_execution_method(self, value: Any) -> ExecutionMethod:
        """Normalize execution method values to enum."""
        if isinstance(value, ExecutionMethod):
            return value
        if isinstance(value, str):
            return ExecutionMethod(value.lower()) if value.lower() in ExecutionMethod._value2member_map_ else ExecutionMethod.AUTO
        return ExecutionMethod.AUTO

    def _normalize_decomposition_plan(self, plan: Any) -> Dict[str, Any]:
        """Normalize decomposition plan to dict format."""
        if plan is None:
            return {}
        if isinstance(plan, dict):
            return plan
        if hasattr(plan, "model_dump"):
            return plan.model_dump()
        if hasattr(plan, "dict"):
            return plan.dict()
        if hasattr(plan, "sub_problems"):
            return {"sub_problems": [
                {"id": sp.id, "description": sp.description, "title": getattr(sp, "title", sp.id),
                 "dependencies": getattr(sp, "dependencies", [])}
                for sp in getattr(plan, "sub_problems", [])
            ]}
        return {}

    def _extract_solutions(self, phase_2_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract solutions list from Phase 2 result."""
        solutions = phase_2_result.get("solutions", [])
        if isinstance(solutions, dict):
            return [
                {"id": key, "solution": value.get("solution") if isinstance(value, dict) else str(value)}
                for key, value in solutions.items()
            ]
        return solutions if isinstance(solutions, list) else []

    def _get_claudiomiro_bridge(self, **kwargs) -> Any:
        """Initialize Claudiomiro bridge with consistent defaults."""
        if not ClaudiomiroCrewAIWorkflowBridge:
            raise RuntimeError("Claudiomiro bridge not available")
        return ClaudiomiroCrewAIWorkflowBridge(
            working_dir=kwargs.get("working_dir", "."),
            ai_provider=kwargs.get("ai_provider", "claude"),
            enable_parallel=kwargs.get("enable_parallel", True),
            max_cycles=kwargs.get("max_cycles", 20),
            state_storage_dir=self.state_storage_dir,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get unified flow status"""
        return {
            "engine": "CrewAI",
            "version": "1.0.0",
            "total_execution_methods": 7,
            "execution_methods": [
                ExecutionMethod.TRADITIONAL,
                ExecutionMethod.ROMA,
                ExecutionMethod.ROMA_MDAP_MAKER,
                ExecutionMethod.CLAUDIOMIRO,
                ExecutionMethod.DATAPIZZA,
                ExecutionMethod.HYBRID,
                ExecutionMethod.AUTO
            ],
            "availability": {
                "crewai": CREWAI_AVAILABLE,
                "decomposition_bridge": DECOMPOSITION_BRIDGE_AVAILABLE,
                "roma_bridge": ROMA_BRIDGE_AVAILABLE,
                "roma_mdap_maker_bridge": ROMA_MDAP_MAKER_BRIDGE_AVAILABLE,
                "datapizza_bridge": DATAPIZZA_BRIDGE_AVAILABLE,
                "claudiomiro_bridge": CLAUDIOMIRO_BRIDGE_AVAILABLE,
            }
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_unified_flow(
    default_execution_method: ExecutionMethod = ExecutionMethod.AUTO,
    enable_persistence: bool = True
) -> CrewAIUnifiedFlow:
    """
    Factory function to create CrewAI unified flow.

    Args:
        default_execution_method: Default execution method
        enable_persistence: Enable state persistence

    Returns:
        Configured CrewAIUnifiedFlow instance
    """
    return CrewAIUnifiedFlow(
        default_execution_method=default_execution_method,
        enable_persistence=enable_persistence
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage
    flow = create_unified_flow()

    # Execute a workflow
    result = flow.execute_full_workflow(
        problem_statement="Design a zero-error distributed database system",
        execution_method=ExecutionMethod.ROMA_MDAP_MAKER
    )

    print(f"Workflow result: {result}")
