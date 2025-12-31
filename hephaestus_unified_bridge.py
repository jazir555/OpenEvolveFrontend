"""
Hephaestus Unified Bridge - Complete ROMA Integration with Full Configurability

This module provides the unified bridge between Hephaestus and all available
execution methods including ROMA and ROMA-Decomposition Hybrid.

This is the MAIN ENTRY POINT for Hephaestus agents to use ROMA and Hybrid modes.

Architecture:
    Hephaestus Agent → Unified Bridge → Execution Method (Traditional/Claudiomiro/DataPizza/ROMA/Hybrid)

Execution Methods:
1. Traditional - AI-assisted decomposition with OpenEvolve
2. Claudiomiro - Autonomous development CLI
3. DataPizza - Multi-agent coordination
4. ROMA - Recursive meta-agent decomposition
5. Hybrid - ROMA + Decomposition Workflow teams
6. Auto - Intelligent selection

Configurability:
    - Full configuration at every phase (Phase 1-6)
    - Per-phase ROMA settings (depth, mode, provider)
    - Per-phase Hybrid settings (depth, gauntlets, evolution)
    - Preset configurations for common use cases
"""

import logging
from typing import Dict, Any, List, Optional, Union

from roma_config import (
    HephaestusROMAConfig,
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
    HephaestusDataPizzaConfig,
    DataPizzaPhase1Config,
    DataPizzaPhase2Config,
    DataPizzaPhase3Config,
    DataPizzaPhase4Config,
    DataPizzaMultiAgentConfig,
    DataPizzaConfigBuilder,
    DataPizzaConfigPresets,
)

from claudiomiro_config import (
    HephaestusClaudiomiroConfig,
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

from decomposition_hephaestus_bridge import (
    execute_phase_1_setup as decomposition_phase_1_setup,
    execute_phase_2_solve as decomposition_phase_2_solve,
    execute_phase_3_critique as decomposition_phase_3_critique,
    execute_phase_4_verify as decomposition_phase_4_verify,
    execute_phase_5_reassemble as decomposition_phase_5_reassemble,
    execute_phase_6_final_validation as decomposition_phase_6_final_validation,
)

from roma_hephaestus_bridge import (
    execute_phase_1_setup as roma_phase_1_setup,
    execute_phase_2_solve as roma_phase_2_solve,
    execute_phase_3_critique as roma_phase_3_critique,
    execute_phase_4_verify as roma_phase_4_verify,
    execute_full_workflow as roma_full_workflow,
)

from roma_decomposition_hybrid import (
    ROMADecompositionHybrid,
    create_hybrid_config,
    get_hybrid_status,
)

from roma_mdap_maker_hephaestus_bridge import (
    execute_phase_1_setup as roma_mdap_maker_phase_1_setup,
    execute_phase_2_solve as roma_mdap_maker_phase_2_solve,
    execute_phase_3_critique as roma_mdap_maker_phase_3_critique,
    execute_phase_4_verify as roma_mdap_maker_phase_4_verify,
    execute_phase_5_reassemble as roma_mdap_maker_phase_5_reassemble,
    execute_phase_6_final_validation as roma_mdap_maker_phase_6_final_validation,
    execute_full_workflow as roma_mdap_maker_full_workflow,
    get_romamdapmaker_bridge_status,
)

logger = logging.getLogger(__name__)


# =============================================================================
# UNIFIED PHASE 1: PROBLEM SETUP
# =============================================================================

def execute_phase_1_setup(
    problem_statement: str,
    execution_method: str = "traditional",  # "traditional", "roma", "roma_mdap_maker", "auto"
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
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
    # ROMA-MDAP-MAKER parameters
    use_roma_mdap_maker: bool = False,
    roma_mdap_maker_max_depth: int = 3,
    roma_mdap_maker_provider: str = "openai",
    roma_mdap_maker_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup - Unified entry point

    Routes to appropriate execution method based on selection:
    - "traditional": Uses Decomposition Workflow (manual stages)
    - "roma": Uses ROMA automatic recursive decomposition
    - "roma_mdap_maker": Uses ROMA + MAKER zero-error voting (NEW)
    - "auto": Intelligently selects based on problem characteristics

    Args:
        problem_statement: The problem to solve
        execution_method: "traditional", "roma", "roma_mdap_maker", or "auto"
        problem_type: Type of problem
        domain: Problem domain
        max_sub_problems: Maximum sub-problems (traditional mode only)
        decomposition_strategy: Strategy for decomposition (traditional mode only)
        use_evolution: Use OpenEvolve (traditional mode only)
        evolution_iterations: Evolution iterations (traditional mode only)
        roma_max_depth: Max recursion depth for ROMA
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA
        roma_model: Model name for ROMA
        use_roma_mdap_maker: Enable ROMA-MDAP-MAKER for auto-selection
        roma_mdap_maker_max_depth: Max depth for ROMA-MDAP-MAKER analysis
        roma_mdap_maker_provider: AI provider for ROMA-MDAP-MAKER
        roma_mdap_maker_model: Model name for ROMA-MDAP-MAKER

    Returns:
        Dict with Phase 1 results
    """
    logger.info(f"Phase 1: Unified setup (method={execution_method})")

    # Auto-selection
    if execution_method == "auto":
        # Check for zero-error critical task keywords (highest priority)
        zero_error_keywords = ["critical", "zero error", "flawless", "perfect", "mission-critical", "safety-critical", "high-reliability"]
        if use_roma_mdap_maker and any(kw in problem_statement.lower() for kw in zero_error_keywords):
            logger.info("  Auto: Selected ROMA-MDAP-MAKER for zero-error critical task")
            execution_method = "roma_mdap_maker"
        # Check if problem keywords suggest ROMA
        elif any(kw in problem_statement.lower() for kw in ["decompose", "break down", "hierarchical", "recursive", "complex structure"]):
            logger.info("  Auto: Selected ROMA for decomposition problem")
            execution_method = "roma"
        else:
            logger.info("  Auto: Selected traditional method")
            execution_method = "traditional"

    # Route to appropriate method
    if execution_method == "roma_mdap_maker":
        return roma_mdap_maker_phase_1_setup(
            problem_statement=problem_statement,
            roma_max_depth_analysis=roma_mdap_maker_max_depth,
            provider=roma_mdap_maker_provider,
            model=roma_mdap_maker_model,
        )
    elif execution_method == "roma":
        return roma_phase_1_setup(
            problem_statement=problem_statement,
            max_depth=roma_max_depth,
            execution_mode=roma_execution_mode,
            provider=roma_provider,
            api_key=roma_api_key,
            model=roma_model,
        )
    else:  # "traditional"
        return decomposition_phase_1_setup(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
            max_sub_problems=max_sub_problems,
            decomposition_strategy=decomposition_strategy,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )


# =============================================================================
# UNIFIED PHASE 2: SOLUTION GENERATION
# =============================================================================

def execute_phase_2_solve(
    decomposition_plan: Dict[str, Any],
    execution_method: str = "traditional",  # "traditional", "claudiomiro", "datapizza", "roma", "hybrid", "roma_mdap_maker", "auto"
    team_name: Optional[str] = None,
    solve_subset: Optional[List[str]] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 100,
    # Claudiomiro parameters
    use_claudiomiro: bool = False,
    claudiomiro_provider: str = "claude",
    claudiomiro_backend: Optional[str] = None,
    claudiomiro_frontend: Optional[str] = None,
    working_dir: str = ".",
    max_cycles: int = 20,
    # DataPizza parameters
    use_datapizza: bool = False,
    datapizza_provider: str = "openai",
    datapizza_api_key: Optional[str] = None,
    datapizza_model: Optional[str] = None,
    datapizza_tools: Optional[List[str]] = None,
    datapizza_planning_interval: int = 3,
    datapizza_max_steps: int = 20,
    # ROMA parameters
    use_roma: bool = False,
    roma_max_depth: int = 2,
    roma_execution_mode: str = "recursive",
    roma_provider: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
    # ROMA-Decomposition Hybrid parameters
    use_hybrid: bool = False,
    hybrid_max_depth_analysis: int = 3,
    hybrid_max_depth_solving: int = 2,
    hybrid_execution_mode: str = "recursive",
    hybrid_provider: Optional[str] = None,
    hybrid_api_key: Optional[str] = None,
    hybrid_model: Optional[str] = None,
    hybrid_enable_gauntlets: bool = True,
    hybrid_enable_evolution: bool = True,
    hybrid_evolution_iterations: int = 50,
    # ROMA-MDAP-MAKER parameters
    use_roma_mdap_maker: bool = False,
    roma_mdap_maker_max_depth: int = 2,
    roma_mdap_maker_k_ahead: int = 3,
    roma_mdap_maker_enable_red_flagging: bool = True,
    roma_mdap_maker_max_samples: int = 100,
    roma_mdap_maker_enable_adaptive_k: bool = True,
    roma_mdap_maker_provider: str = "openai",
    roma_mdap_maker_api_key: Optional[str] = None,
    roma_mdap_maker_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation - Unified entry point

    Routes to appropriate execution method based on selection.

    Args:
        decomposition_plan: Complete decomposition plan from Phase 1
        execution_method: How to execute (all 7 methods available)
        team_name: Specific Blue Team to use
        solve_subset: List of sub-problem IDs to solve
        use_evolution: Use OpenEvolve (traditional mode)
        evolution_iterations: Evolution iterations
        use_claudiomiro: Enable Claudiomiro
        claudiomiro_provider: AI provider for Claudiomiro
        claudiomiro_backend: Backend directory
        claudiomiro_frontend: Frontend directory
        working_dir: Working directory
        max_cycles: Maximum Claudiomiro cycles
        use_datapizza: Enable DataPizza
        datapizza_provider: AI provider for DataPizza
        datapizza_api_key: API key for DataPizza
        datapizza_model: Model name for DataPizza
        datapizza_tools: List of tools to enable
        datapizza_planning_interval: Planning interval
        datapizza_max_steps: Maximum steps
        use_roma: Enable ROMA
        roma_max_depth: Maximum recursion depth for ROMA
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA
        roma_model: Model name for ROMA
        use_hybrid: Enable ROMA-Decomposition hybrid
        hybrid_max_depth_analysis: Max depth for ROMA analysis phase (hybrid mode)
        hybrid_max_depth_solving: Max depth for ROMA solving phase (hybrid mode)
        hybrid_execution_mode: ROMA execution mode for hybrid
        hybrid_provider: AI provider for hybrid mode
        hybrid_api_key: API key for hybrid mode provider
        hybrid_model: Model name for hybrid mode
        hybrid_enable_gauntlets: Enable Decomposition Workflow gauntlets in hybrid mode
        hybrid_enable_evolution: Enable evolution in hybrid mode
        hybrid_evolution_iterations: Evolution iterations for hybrid mode
        use_roma_mdap_maker: Enable ROMA-MDAP-MAKER (zero-error mode)
        roma_mdap_maker_max_depth: Max depth for ROMA-MDAP-MAKER
        roma_mdap_maker_k_ahead: K-ahead threshold for MAKER voting
        roma_mdap_maker_enable_red_flagging: Enable MAKER red-flagging
        roma_mdap_maker_max_samples: Max samples for MAKER voting
        roma_mdap_maker_enable_adaptive_k: Enable adaptive k-ahead selection
        roma_mdap_maker_provider: AI provider for ROMA-MDAP-MAKER
        roma_mdap_maker_api_key: API key for ROMA-MDAP-MAKER provider
        roma_mdap_maker_model: Model name for ROMA-MDAP-MAKER

    Returns:
        Dict with Phase 2 results
    """
    logger.info(f"Phase 2: Unified solution generation (method={execution_method})")

    # Route to decomposition bridge (it handles all execution methods)
    return decomposition_phase_2_solve(
        decomposition_plan=decomposition_plan,
        team_name=team_name,
        solve_subset=solve_subset,
        use_evolution=use_evolution,
        evolution_iterations=evolution_iterations,
        execution_method=execution_method,
        use_claudiomiro=use_claudiomiro,
        use_datapizza=use_datapizza,
        use_roma=use_roma,
        use_hybrid=use_hybrid,
        claudiomiro_provider=claudiomiro_provider,
        claudiomiro_backend=claudiomiro_backend,
        claudiomiro_frontend=claudiomiro_frontend,
        working_dir=working_dir,
        max_cycles=max_cycles,
        datapizza_provider=datapizza_provider,
        datapizza_api_key=datapizza_api_key,
        datapizza_model=datapizza_model,
        datapizza_tools=datapizza_tools,
        datapizza_planning_interval=datapizza_planning_interval,
        datapizza_max_steps=datapizza_max_steps,
        roma_max_depth=roma_max_depth,
        roma_execution_mode=roma_execution_mode,
        roma_provider=roma_provider,
        roma_api_key=roma_api_key,
        roma_model=roma_model,
        hybrid_max_depth_analysis=hybrid_max_depth_analysis,
        hybrid_max_depth_solving=hybrid_max_depth_solving,
        hybrid_execution_mode=hybrid_execution_mode,
        hybrid_provider=hybrid_provider,
        hybrid_api_key=hybrid_api_key,
        hybrid_model=hybrid_model,
        hybrid_enable_gauntlets=hybrid_enable_gauntlets,
        hybrid_enable_evolution=hybrid_enable_evolution,
        hybrid_evolution_iterations=hybrid_evolution_iterations,
        use_roma_mdap_maker=use_roma_mdap_maker,
        roma_mdap_maker_max_depth=roma_mdap_maker_max_depth,
        roma_mdap_maker_k_ahead=roma_mdap_maker_k_ahead,
        roma_mdap_maker_enable_red_flagging=roma_mdap_maker_enable_red_flagging,
        roma_mdap_maker_max_samples=roma_mdap_maker_max_samples,
        roma_mdap_maker_enable_adaptive_k=roma_mdap_maker_enable_adaptive_k,
        roma_mdap_maker_provider=roma_mdap_maker_provider,
        roma_mdap_maker_api_key=roma_mdap_maker_api_key,
        roma_mdap_maker_model=roma_mdap_maker_model,
    )


# =============================================================================
# UNIFIED FULL WORKFLOW
# =============================================================================

def execute_full_workflow(
    problem_statement: str,
    execution_method_phase2: str = "traditional",  # For Phase 2
    use_evolution: bool = True,
    # ROMA workflow parameters
    use_roma_workflow: bool = False,
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    roma_execution_mode: str = "recursive",
    roma_provider: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute full 6-phase Hephaestus workflow with ROMA integration

    This is the main entry point for Hephaestus to execute complete workflows.

    Args:
        problem_statement: The problem to solve
        execution_method_phase2: Execution method for Phase 2 solution generation
        use_evolution: Use OpenEvolve (traditional phases)
        use_roma_workflow: Use ROMA's full workflow instead of Decomposition Workflow
        roma_max_depth_analysis: Max depth for ROMA analysis phase
        roma_max_depth_solving: Max depth for ROMA solving phase
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA
        roma_model: Model name for ROMA

    Returns:
        Dict with complete workflow results
    """
    logger.info(f"Starting full unified workflow: {problem_statement[:50]}...")

    try:
        if use_roma_workflow:
            # Use ROMA's full native workflow
            logger.info("Using ROMA native workflow")
            return roma_full_workflow(
                problem_statement=problem_statement,
                max_depth_analysis=roma_max_depth_analysis,
                max_depth_solving=roma_max_depth_solving,
                execution_mode=roma_execution_mode,
                provider=roma_provider,
                api_key=roma_api_key,
                model=roma_model,
            )
        else:
            # Use Decomposition Workflow with ROMA/Hybrid available for Phase 2
            logger.info("Using Decomposition Workflow with ROMA/Hybrid options")

            # Phase 1: Setup (Traditional or ROMA)
            phase1_result = execute_phase_1_setup(
                problem_statement=problem_statement,
                execution_method="traditional",  # Use traditional for Phase 1
                use_evolution=use_evolution,
            )

            if phase1_result["status"] == "failed":
                return phase1_result

            # Phase 2: Solve (All execution methods available)
            phase2_result = execute_phase_2_solve(
                decomposition_plan=phase1_result,
                execution_method=execution_method_phase2,
                use_evolution=use_evolution,
            )

            if phase2_result["status"] == "failed":
                return phase2_result

            # Phase 3-6: Use Decomposition Workflow
            phase3_result = decomposition_phase_3_critique(
                solutions=phase2_result["solutions"],
                use_evolution=use_evolution,
            )

            phase4_result = decomposition_phase_4_verify(
                solutions=phase2_result["solutions"],
                use_evolution=use_evolution,
            )

            phase5_result = decomposition_phase_5_reassemble(
                solutions=phase2_result["solutions"],
                problem_statement=problem_statement,
                use_evolution=use_evolution,
            )

            phase6_result = decomposition_phase_6_final_validation(
                final_solution=phase5_result["final_solution"],
                problem_statement=problem_statement,
                use_evolution=use_evolution,
            )

            return {
                "workflow": "unified_decomposition_with_roma_options",
                "status": "completed",
                "phases": {
                    "phase1": phase1_result,
                    "phase2": phase2_result,
                    "phase3": phase3_result,
                    "phase4": phase4_result,
                    "phase5": phase5_result,
                    "phase6": phase6_result,
                },
                "message": "Full unified workflow completed successfully",
            }

    except Exception as e:
        logger.error(f"Full workflow failed: {e}")
        return {
            "workflow": "unified",
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# FULLY CONFIGURABLE PHASE EXECUTION (WITH CONFIG OBJECTS)
# =============================================================================

def execute_phase_1_with_config(
    problem_statement: str,
    config: Optional[ROMAPhase1Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 1: Problem Setup with full configuration

    This is the RECOMMENDED method for Phase 1 execution when using ROMA.
    Accepts a ROMAPhase1Config object for full control.

    Args:
        problem_statement: The problem to solve
        config: ROMAPhase1Config object (uses default if None)

    Returns:
        Dict with Phase 1 results
    """
    if config is None:
        config = ROMAPhase1Config()

    logger.info(f"Phase 1 with config: execution_mode={config.execution_mode}, max_depth={config.max_depth_analysis}")

    # Route to appropriate method based on config
    if config.provider == "roma":
        # Use ROMA for Phase 1
        return roma_phase_1_setup(
            problem_statement=problem_statement,
            max_depth=config.max_depth_analysis,
            execution_mode=config.execution_mode,
            provider=config.provider,
            api_key=config.api_key,
            model=config.model,
        )
    else:
        # Use traditional Decomposition Workflow
        return decomposition_phase_1_setup(
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )


def execute_phase_2_with_config(
    decomposition_plan: Dict[str, Any],
    config: Optional[ROMAPhase2Config] = None,
    hybrid_config: Optional[ROMAHybridConfig] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation with full configuration

    This is the RECOMMENDED method for Phase 2 execution when using ROMA or Hybrid.
    Accepts ROMAPhase2Config and optional ROMAHybridConfig objects.

    Args:
        decomposition_plan: Complete decomposition plan from Phase 1
        config: ROMAPhase2Config object (uses default if None)
        hybrid_config: ROMAHybridConfig object (only if using hybrid mode)

    Returns:
        Dict with Phase 2 results
    """
    if config is None:
        config = ROMAPhase2Config()

    logger.info(f"Phase 2 with config: execution_mode={config.execution_mode}, max_depth={config.max_depth_solving}")

    # Determine execution method from config
    if hybrid_config:
        # Use hybrid mode
        return decomposition_phase_2_solve(
            decomposition_plan=decomposition_plan,
            execution_method="hybrid",
            use_hybrid=True,
            hybrid_max_depth_analysis=hybrid_config.roma_max_depth_analysis,
            hybrid_max_depth_solving=hybrid_config.roma_max_depth_solving,
            hybrid_execution_mode=hybrid_config.roma_execution_mode,
            hybrid_provider=hybrid_config.roma_provider,
            hybrid_api_key=hybrid_config.roma_api_key,
            hybrid_model=hybrid_config.roma_model,
            hybrid_enable_gauntlets=hybrid_config.enable_gauntlets,
            hybrid_enable_evolution=hybrid_config.enable_evolution,
            hybrid_evolution_iterations=hybrid_config.evolution_iterations,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )
    else:
        # Use ROMA mode
        return decomposition_phase_2_solve(
            decomposition_plan=decomposition_plan,
            execution_method="roma",
            use_roma=True,
            roma_max_depth=config.max_depth_solving,
            roma_execution_mode=config.execution_mode,
            roma_provider=config.provider,
            roma_api_key=config.api_key,
            roma_model=config.model,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )


def execute_phase_3_with_config(
    solutions: List[Dict[str, Any]],
    config: Optional[ROMAPhase3Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique with full configuration

    This is the RECOMMENDED method for Phase 3 execution when using ROMA.
    Accepts a ROMAPhase3Config object for full control.

    Args:
        solutions: Solutions from Phase 2
        config: ROMAPhase3Config object (uses default if None)

    Returns:
        Dict with Phase 3 results
    """
    if config is None:
        config = ROMAPhase3Config()

    logger.info(f"Phase 3 with config: critique_focus={config.critique_focus}, intensity={config.critique_intensity}")

    # Use ROMA for critique
    return roma_phase_3_critique(
        solutions=solutions,
        critique_focus=config.critique_focus,
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
    )


def execute_phase_4_with_config(
    solutions: List[Dict[str, Any]],
    config: Optional[ROMAPhase4Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification with full configuration

    This is the RECOMMENDED method for Phase 4 execution when using ROMA.
    Accepts a ROMAPhase4Config object for full control.

    Args:
        solutions: Solutions from Phase 2
        config: ROMAPhase4Config object (uses default if None)

    Returns:
        Dict with Phase 4 results
    """
    if config is None:
        config = ROMAPhase4Config()

    logger.info(f"Phase 4 with config: strictness={config.verification_strictness}")

    # Use ROMA for verification
    return roma_phase_4_verify(
        solutions=solutions,
        verification_criteria=config.verification_criteria,
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
    )


def execute_full_workflow_with_config(
    problem_statement: str,
    config: Optional[HephaestusROMAConfig] = None,
) -> Dict[str, Any]:
    """
    Execute full 6-phase Hephaestus workflow with complete configuration

    This is the RECOMMENDED method for full workflow execution.
    Accepts a HephaestusROMAConfig object for complete control over all phases.

    Args:
        problem_statement: The problem to solve
        config: HephaestusROMAConfig object (uses default if None)

    Returns:
        Dict with complete workflow results
    """
    if config is None:
        config = HephaestusROMAConfig()

    logger.info(f"Starting full workflow with config: execution_method={config.execution_method}")

    # Validate configuration
    errors = config.validate()
    if any(errors.values()):
        error_msgs = [f"Phase {phase}: {errs}" for phase, errs in errors.items() if errs]
        logger.error(f"Configuration errors: {error_msgs}")
        return {
            "workflow": "full_with_config",
            "status": "failed",
            "error": "Invalid configuration",
            "validation_errors": errors,
        }

    try:
        # Phase 1: Setup
        logger.info("Phase 1: Problem Setup")
        phase1_result = execute_phase_1_with_config(
            problem_statement=problem_statement,
            config=config.phase1,
        )

        if phase1_result["status"] == "failed":
            return phase1_result

        # Phase 2: Solve
        logger.info("Phase 2: Solution Generation")
        phase2_result = execute_phase_2_with_config(
            decomposition_plan=phase1_result,
            config=config.phase2,
            hybrid_config=config.hybrid,
        )

        if phase2_result["status"] == "failed":
            return phase2_result

        # Phase 3: Critique
        logger.info("Phase 3: Adversarial Critique")
        phase3_result = execute_phase_3_with_config(
            solutions=phase2_result.get("solutions", []),
            config=config.phase3,
        )

        # Phase 4: Verify
        logger.info("Phase 4: Verification")
        phase4_result = execute_phase_4_with_config(
            solutions=phase2_result.get("solutions", []),
            config=config.phase4,
        )

        # Phase 5: Reassemble (use Decomposition Workflow)
        logger.info("Phase 5: Reassembly")
        phase5_result = decomposition_phase_5_reassemble(
            solutions=phase2_result.get("solutions", []),
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        # Phase 6: Final Validation (use Decomposition Workflow)
        logger.info("Phase 6: Final Validation")
        phase6_result = decomposition_phase_6_final_validation(
            final_solution=phase5_result.get("final_solution", ""),
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        return {
            "workflow": "full_with_config",
            "status": "completed",
            "phases": {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "phase4": phase4_result,
                "phase5": phase5_result,
                "phase6": phase6_result,
            },
            "config_used": config.to_dict(),
            "message": "Full workflow with configuration completed successfully",
        }

    except Exception as e:
        logger.error(f"Full workflow with config failed: {e}")
        return {
            "workflow": "full_with_config",
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# FULLY CONFIGURABLE PHASE EXECUTION: DATAPIZZA
# =============================================================================

def execute_phase_2_with_datapizza_config(
    decomposition_plan: Dict[str, Any],
    config: Optional[DataPizzaPhase2Config] = None,
    multi_agent_config: Optional[DataPizzaMultiAgentConfig] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation with DataPizza configuration

    This is the RECOMMENDED method for Phase 2 execution when using DataPizza.
    Accepts DataPizzaPhase2Config and optional DataPizzaMultiAgentConfig.

    Args:
        decomposition_plan: Complete decomposition plan from Phase 1
        config: DataPizzaPhase2Config object (uses default if None)
        multi_agent_config: DataPizzaMultiAgentConfig object (optional)

    Returns:
        Dict with Phase 2 results
    """
    if config is None:
        config = DataPizzaPhase2Config()

    logger.info(f"Phase 2 with DataPizza config: max_steps={config.max_steps}, planning_interval={config.planning_interval}")

    # Use decomposition bridge with DataPizza parameters
    return decomposition_phase_2_solve(
        decomposition_plan=decomposition_plan,
        execution_method="datapizza",
        use_datapizza=True,
        datapizza_provider=config.provider,
        datapizza_api_key=config.api_key,
        datapizza_model=config.model,
        datapizza_tools=config.tools,
        datapizza_planning_interval=config.planning_interval,
        datapizza_max_steps=config.max_steps,
        use_evolution=config.enable_evolution,
        evolution_iterations=config.evolution_iterations,
    )


def execute_phase_3_with_datapizza_config(
    solutions: List[Dict[str, Any]],
    config: Optional[DataPizzaPhase3Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique with DataPizza configuration

    This is the RECOMMENDED method for Phase 3 execution when using DataPizza.
    Accepts a DataPizzaPhase3Config object for full control.

    Args:
        solutions: Solutions from Phase 2
        config: DataPizzaPhase3Config object (uses default if None)

    Returns:
        Dict with Phase 3 results
    """
    if config is None:
        config = DataPizzaPhase3Config()

    logger.info(f"Phase 3 with DataPizza config: critique_focus={config.critique_focus}, intensity={config.critique_intensity}")

    # Use ROMA critique with DataPizza settings
    return roma_phase_3_critique(
        solutions=solutions,
        critique_focus=config.critique_focus[0] if config.critique_focus else "comprehensive",
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
    )


def execute_phase_4_with_datapizza_config(
    solutions: List[Dict[str, Any]],
    config: Optional[DataPizzaPhase4Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification with DataPizza configuration

    This is the RECOMMENDED method for Phase 4 execution when using DataPizza.
    Accepts a DataPizzaPhase4Config object for full control.

    Args:
        solutions: Solutions from Phase 2
        config: DataPizzaPhase4Config object (uses default if None)

    Returns:
        Dict with Phase 4 results
    """
    if config is None:
        config = DataPizzaPhase4Config()

    logger.info(f"Phase 4 with DataPizza config: strictness={config.verification_strictness}")

    # Use ROMA verification with DataPizza settings
    return roma_phase_4_verify(
        solutions=solutions,
        verification_criteria=config.verification_criteria,
        provider=config.provider,
        api_key=config.api_key,
        model=config.model,
    )


def execute_full_workflow_with_datapizza_config(
    problem_statement: str,
    config: Optional[HephaestusDataPizzaConfig] = None,
) -> Dict[str, Any]:
    """
    Execute full 6-phase Hephaestus workflow with DataPizza configuration

    This is the RECOMMENDED method for full workflow execution when using DataPizza.
    Accepts a HephaestusDataPizzaConfig object for complete control.

    Args:
        problem_statement: The problem to solve
        config: HephaestusDataPizzaConfig object (uses default if None)

    Returns:
        Dict with complete workflow results
    """
    if config is None:
        config = HephaestusDataPizzaConfig()

    logger.info(f"Starting full workflow with DataPizza config")

    # Validate configuration
    errors = config.validate()
    if any(errors.values()):
        logger.error(f"Configuration errors: {errors}")
        return {
            "workflow": "full_with_datapizza_config",
            "status": "failed",
            "error": "Invalid configuration",
            "validation_errors": errors,
        }

    try:
        # Phase 1: Setup (traditional with DataPizza options)
        logger.info("Phase 1: Problem Setup")
        phase1_result = decomposition_phase_1_setup(
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        if phase1_result["status"] == "failed":
            return phase1_result

        # Phase 2: Solve with DataPizza
        logger.info("Phase 2: Solution Generation with DataPizza")
        phase2_result = execute_phase_2_with_datapizza_config(
            decomposition_plan=phase1_result,
            config=config.phase2,
            multi_agent_config=config.multi_agent,
        )

        if phase2_result["status"] == "failed":
            return phase2_result

        # Phase 3: Critique
        logger.info("Phase 3: Adversarial Critique")
        phase3_result = execute_phase_3_with_datapizza_config(
            solutions=phase2_result.get("solutions", []),
            config=config.phase3,
        )

        # Phase 4: Verify
        logger.info("Phase 4: Verification")
        phase4_result = execute_phase_4_with_datapizza_config(
            solutions=phase2_result.get("solutions", []),
            config=config.phase4,
        )

        # Phase 5: Reassemble
        logger.info("Phase 5: Reassembly")
        phase5_result = decomposition_phase_5_reassemble(
            solutions=phase2_result.get("solutions", []),
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        # Phase 6: Final Validation
        logger.info("Phase 6: Final Validation")
        phase6_result = decomposition_phase_6_final_validation(
            final_solution=phase5_result.get("final_solution", ""),
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        return {
            "workflow": "full_with_datapizza_config",
            "status": "completed",
            "phases": {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "phase4": phase4_result,
                "phase5": phase5_result,
                "phase6": phase6_result,
            },
            "config_used": config.to_dict(),
            "message": "Full workflow with DataPizza configuration completed",
        }

    except Exception as e:
        logger.error(f"Full workflow with DataPizza config failed: {e}")
        return {
            "workflow": "full_with_datapizza_config",
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# FULLY CONFIGURABLE PHASE EXECUTION: CLAUDIOMIRO
# =============================================================================

def execute_phase_2_with_claudiomiro_config(
    decomposition_plan: Dict[str, Any],
    config: Optional[ClaudiomiroPhase2Config] = None,
    multi_repo_config: Optional[ClaudiomiroMultiRepoConfig] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 2: Solution Generation with Claudiomiro configuration

    This is the RECOMMENDED method for Phase 2 execution when using Claudiomiro.
    Accepts ClaudiomiroPhase2Config and optional ClaudiomiroMultiRepoConfig.

    Args:
        decomposition_plan: Complete decomposition plan from Phase 1
        config: ClaudiomiroPhase2Config object (uses default if None)
        multi_repo_config: ClaudiomiroMultiRepoConfig object (optional)

    Returns:
        Dict with Phase 2 results
    """
    if config is None:
        config = ClaudiomiroPhase2Config()

    logger.info(f"Phase 2 with Claudiomiro config: max_cycles={config.max_cycles}, provider={config.provider}")

    # Use decomposition bridge with Claudiomiro parameters
    return decomposition_phase_2_solve(
        decomposition_plan=decomposition_plan,
        execution_method="claudiomiro",
        use_claudiomiro=True,
        claudiomiro_provider=config.provider,
        claudiomiro_backend=config.backend,
        claudiomiro_frontend=config.frontend,
        working_dir=config.working_dir,
        max_cycles=config.max_cycles,
        use_evolution=config.enable_evolution,
        evolution_iterations=config.evolution_iterations,
    )


def execute_phase_3_with_claudiomiro_config(
    solutions: List[Dict[str, Any]],
    config: Optional[ClaudiomiroPhase3Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 3: Adversarial Critique with Claudiomiro configuration

    This is the RECOMMENDED method for Phase 3 execution when using Claudiomiro.
    Accepts a ClaudiomiroPhase3Config object for full control.

    Args:
        solutions: Solutions from Phase 2
        config: ClaudiomiroPhase3Config object (uses default if None)

    Returns:
        Dict with Phase 3 results
    """
    if config is None:
        config = ClaudiomiroPhase3Config()

    logger.info(f"Phase 3 with Claudiomiro config: critique_mode={config.critique_mode}, intensity={config.critique_intensity}")

    # Use traditional critique (Claudiomiro is primarily for Phase 2)
    return decomposition_phase_3_critique(
        solutions=solutions,
        use_evolution=False,
        evolution_iterations=30,
    )


def execute_phase_4_with_claudiomiro_config(
    solutions: List[Dict[str, Any]],
    config: Optional[ClaudiomiroPhase4Config] = None,
) -> Dict[str, Any]:
    """
    Execute Phase 4: Verification with Claudiomiro configuration

    This is the RECOMMENDED method for Phase 4 execution when using Claudiomiro.
    Accepts a ClaudiomiroPhase4Config object for full control.

    Args:
        solutions: Solutions from Phase 2
        config: ClaudiomiroPhase4Config object (uses default if None)

    Returns:
        Dict with Phase 4 results
    """
    if config is None:
        config = ClaudiomiroPhase4Config()

    logger.info(f"Phase 4 with Claudiomiro config: verification_strictness={config.verification_strictness}")

    # Use traditional verification (Claudiomiro is primarily for Phase 2)
    return decomposition_phase_4_verify(
        solutions=solutions,
        use_evolution=False,
        evolution_iterations=30,
    )


def execute_full_workflow_with_claudiomiro_config(
    problem_statement: str,
    config: Optional[HephaestusClaudiomiroConfig] = None,
) -> Dict[str, Any]:
    """
    Execute full 6-phase Hephaestus workflow with Claudiomiro configuration

    This is the RECOMMENDED method for full workflow execution when using Claudiomiro.
    Accepts a HephaestusClaudiomiroConfig object for complete control.

    Args:
        problem_statement: The problem to solve
        config: HephaestusClaudiomiroConfig object (uses default if None)

    Returns:
        Dict with complete workflow results
    """
    if config is None:
        config = HephaestusClaudiomiroConfig()

    logger.info(f"Starting full workflow with Claudiomiro config")

    # Validate configuration
    errors = config.validate()
    if any(errors.values()):
        logger.error(f"Configuration errors: {errors}")
        return {
            "workflow": "full_with_claudiomiro_config",
            "status": "failed",
            "error": "Invalid configuration",
            "validation_errors": errors,
        }

    try:
        # Phase 1: Setup
        logger.info("Phase 1: Problem Setup")
        phase1_result = decomposition_phase_1_setup(
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        if phase1_result["status"] == "failed":
            return phase1_result

        # Phase 2: Solve with Claudiomiro
        logger.info("Phase 2: Solution Generation with Claudiomiro")
        phase2_result = execute_phase_2_with_claudiomiro_config(
            decomposition_plan=phase1_result,
            config=config.phase2,
            multi_repo_config=config.multi_repo,
        )

        if phase2_result["status"] == "failed":
            return phase2_result

        # Phase 3: Critique
        logger.info("Phase 3: Adversarial Critique")
        phase3_result = execute_phase_3_with_claudiomiro_config(
            solutions=phase2_result.get("solutions", []),
            config=config.phase3,
        )

        # Phase 4: Verify
        logger.info("Phase 4: Verification")
        phase4_result = execute_phase_4_with_claudiomiro_config(
            solutions=phase2_result.get("solutions", []),
            config=config.phase4,
        )

        # Phase 5: Reassemble
        logger.info("Phase 5: Reassembly")
        phase5_result = decomposition_phase_5_reassemble(
            solutions=phase2_result.get("solutions", []),
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        # Phase 6: Final Validation
        logger.info("Phase 6: Final Validation")
        phase6_result = decomposition_phase_6_final_validation(
            final_solution=phase5_result.get("final_solution", ""),
            problem_statement=problem_statement,
            use_evolution=config.enable_evolution,
            evolution_iterations=config.evolution_iterations,
        )

        return {
            "workflow": "full_with_claudiomiro_config",
            "status": "completed",
            "phases": {
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "phase4": phase4_result,
                "phase5": phase5_result,
                "phase6": phase6_result,
            },
            "config_used": config.to_dict(),
            "message": "Full workflow with Claudiomiro configuration completed",
        }

    except Exception as e:
        logger.error(f"Full workflow with Claudiomiro config failed: {e}")
        return {
            "workflow": "full_with_claudiomiro_config",
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# HEPHAESTUS AGENT INTERFACE
# =============================================================================

class HephaestusUnifiedBridge:
    """
    Unified bridge class for Hephaestus agents.

    This is the main interface that Hephaestus agents should use to access
    all execution methods including ROMA and Hybrid.
    """

    def __init__(
        self,
        default_execution_method: str = "traditional",
        enable_roma: bool = True,
        enable_hybrid: bool = True,
        enable_claudiomiro: bool = True,
        enable_datapizza: bool = True,
    ):
        """
        Initialize the unified Hephaestus bridge.

        Args:
            default_execution_method: Default method for Phase 2
            enable_roma: Enable ROMA execution method
            enable_hybrid: Enable Hybrid execution method
            enable_claudiomiro: Enable Claudiomiro execution method
            enable_datapizza: Enable DataPizza execution method
        """
        self.default_execution_method = default_execution_method
        self.enable_roma = enable_roma
        self.enable_hybrid = enable_hybrid
        self.enable_claudiomiro = enable_claudiomiro
        self.enable_datapizza = enable_datapizza

        # Check availability
        status = get_hybrid_status()
        self.hybrid_available = status.get("available", False)

        logger.info(f"Initialized Hephaestus Unified Bridge")
        logger.info(f"  Default execution method: {default_execution_method}")
        logger.info(f"  ROMA enabled: {enable_roma}")
        logger.info(f"  Hybrid enabled: {enable_hybrid}")
        logger.info(f"  Claudiomiro enabled: {enable_claudiomiro}")
        logger.info(f"  DataPizza enabled: {enable_datapizza}")
        logger.info(f"  Hybrid available: {self.hybrid_available}")

    def execute_phase_1(self, problem_statement: str, **kwargs) -> Dict[str, Any]:
        """Execute Phase 1: Problem Setup"""
        return execute_phase_1_setup(problem_statement=problem_statement, **kwargs)

    def execute_phase_2(self, decomposition_plan: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute Phase 2: Solution Generation"""
        # Set default execution method if not specified
        if "execution_method" not in kwargs or kwargs["execution_method"] == "auto":
            kwargs["execution_method"] = self.default_execution_method

        # Enable/disable methods based on configuration
        if not self.enable_roma:
            kwargs["use_roma"] = False
        if not self.enable_hybrid:
            kwargs["use_hybrid"] = False
        if not self.enable_claudiomiro:
            kwargs["use_claudiomiro"] = False
        if not self.enable_datapizza:
            kwargs["use_datapizza"] = False

        return execute_phase_2_solve(decomposition_plan=decomposition_plan, **kwargs)

    def execute_full_workflow(self, problem_statement: str, **kwargs) -> Dict[str, Any]:
        """Execute full 6-phase workflow"""
        return execute_full_workflow(problem_statement=problem_statement, **kwargs)


# =============================================================================
# STATUS AND UTILITIES
# =============================================================================

def get_unified_bridge_status() -> Dict[str, Any]:
    """Get the status of all available execution methods"""
    from decomposition_mcp_tools import get_decomposition_status
    from roma_mcp_tools import get_roma_status
    from roma_mdap_maker_engine import get_roma_mdap_maker_status as get_romamdapmaker_status

    decomp_status = get_decomposition_status()
    roma_status = get_roma_status()
    hybrid_status = get_hybrid_status()
    romamdapmaker_status = get_romamdapmaker_status()
    romamdapmaker_bridge_status = get_romamdapmaker_bridge_status()

    return {
        "traditional_available": decomp_status.get("available", False),
        "claudiomiro_available": decomp_status.get("claudiomiro_available", False),
        "datapizza_available": decomp_status.get("datapizza_available", False),
        "roma_available": roma_status.get("available", False),
        "hybrid_available": hybrid_status.get("available", False),
        "roma_mdap_maker_available": romamdapmaker_status.get("available", False),
        "roma_mdap_maker_bridge_available": romamdapmaker_bridge_status.get("bridge_available", False),
        "total_execution_methods": 7,
        "execution_methods": [
            "traditional",
            "claudiomiro",
            "datapizza",
            "roma",
            "hybrid",
            "roma_mdap_maker",
            "auto",
        ],
    }
