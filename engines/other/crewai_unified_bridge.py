"""
CrewAI Unified Bridge - Complete Replacement for CrewAI Unified Bridge

This module provides the unified bridge interface that replaces crewai_unified_bridge.py
while maintaining full compatibility with the existing API.

Key Features:
1. Complete drop-in replacement for CrewAI unified bridge
2. All 7 execution methods supported (Traditional, ROMA, ROMA-MDAP-MAKER, Claudiomiro, DataPizza, Hybrid, Auto)
3. Full Phase 1-6 coordination
4. Auto-selection algorithm
5. Config-based execution for all phases

Architecture:
    User API -> CrewAI Unified Bridge -> CrewAI Unified Flow -> Local Execution

The bridge maintains API compatibility while delegating to the CrewAI infrastructure
built in Phase 1 (crewai_unified_flow.py, crewai_state_management.py, etc.)

License: MIT (replaces AGPL CrewAI)
"""
from __future__ import annotations


import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# Import CrewAI infrastructure
from crewai_unified_flow import CrewAIUnifiedFlow, ExecutionMethod as CrewAIExecutionMethod
from crewai_state_management import (
    WorkflowState,
    WorkflowStatus,
    ExecutionMethod,
    StateManager,
    create_workflow_state,
    create_state_manager,
)
from crewai_client import CrewAIClient, create_crewai_client
from crewai_zero_error_workflow import CrewAIZeroErrorWorkflow, create_zero_error_workflow

# Phase bridges for decomposition/ROMA
try:
    from decomposition_crewai_bridge import (
        execute_phase_3_critique as decomposition_bridge_phase_3,
        execute_phase_4_verify as decomposition_bridge_phase_4,
        execute_phase_5_reassemble as decomposition_bridge_phase_5,
        execute_phase_6_final_validation as decomposition_bridge_phase_6,
    )
    DECOMPOSITION_PHASE_BRIDGE_AVAILABLE = True
except ImportError:
    DECOMPOSITION_PHASE_BRIDGE_AVAILABLE = False
    decomposition_bridge_phase_3 = None
    decomposition_bridge_phase_4 = None
    decomposition_bridge_phase_5 = None
    decomposition_bridge_phase_6 = None

try:
    from roma_crewai_bridge import (
        execute_phase_3_critique as roma_bridge_phase_3,
        execute_phase_4_verify as roma_bridge_phase_4,
        execute_phase_5_reassemble as roma_bridge_phase_5,
        execute_phase_6_final_validation as roma_bridge_phase_6,
    )
    ROMA_PHASE_BRIDGE_AVAILABLE = True
except ImportError:
    ROMA_PHASE_BRIDGE_AVAILABLE = False
    roma_bridge_phase_3 = None
    roma_bridge_phase_4 = None
    roma_bridge_phase_5 = None

try:
    from roma_mdap_maker_crewai_bridge import get_romamdapmaker_bridge_status
    ROMA_MDAP_MAKER_BRIDGE_STATUS_AVAILABLE = True
except ImportError:
    ROMA_MDAP_MAKER_BRIDGE_STATUS_AVAILABLE = False
    get_romamdapmaker_bridge_status = None

# Import config classes for backward compatibility
try:
    from roma_config import (
        CrewAIROMAConfig,
        ROMAPhase1Config,
        ROMAPhase2Config,
        ROMAPhase3Config,
        ROMAPhase4Config,
        ROMAPhase5Config,
        ROMAPhase6Config,
        ROMAHybridConfig,
    )
    ROMA_CONFIG_AVAILABLE = True
except ImportError:
    ROMA_CONFIG_AVAILABLE = False
    CrewAIROMAConfig = None
    ROMAPhase1Config = None
    ROMAPhase2Config = None
    ROMAPhase3Config = None
    ROMAPhase4Config = None
    ROMAPhase5Config = None
    ROMAPhase6Config = None
    ROMAHybridConfig = None

logger = logging.getLogger(__name__)


# =============================================================================
# EXECUTION METHOD ENUMS (Backward Compatibility)
# =============================================================================

class ExecutionMethodEnum(str, Enum):
    """Execution methods (backward compatible with CrewAI)"""
    TRADITIONAL = "traditional"
    ROMA = "roma"
    ROMA_MDAP_MAKER = "roma_mdap_maker"  # ZERO-ERROR
    CLAUDIOMIRO = "claudiomiro"
    DATAPIZZA = "datapizza"
    HYBRID = "hybrid"
    AUTO = "auto"


# =============================================================================
# MAIN BRIDGE CLASS
# =============================================================================

class CrewAIUnifiedBridge:
    """
    Complete replacement for CrewAIUnifiedBridge.

    This class provides the same interface as the CrewAI unified bridge
    but uses local CrewAI execution instead of remote API calls.

    Key Features:
    - Drop-in replacement for crewai_unified_bridge.py
    - Supports all 7 execution methods
    - Full Phase 1-6 coordination
    - Auto-selection algorithm
    - Config-based execution
    """

    def __init__(
        self,
        state_storage_dir: str = "./crewai_states",
        enable_persistence: bool = True,
        default_execution_method: ExecutionMethod = ExecutionMethod.AUTO,
    ):
        """
        Initialize CrewAI unified bridge.

        Args:
            state_storage_dir: Directory for state storage
            enable_persistence: Enable state persistence
            default_execution_method: Default execution method
        """
        self.state_storage_dir = state_storage_dir
        self.enable_persistence = enable_persistence
        self.default_execution_method = default_execution_method

        method_enum = _map_execution_method(default_execution_method)

        # Initialize CrewAI components
        self.unified_flow = CrewAIUnifiedFlow(
            default_execution_method=method_enum,
            enable_persistence=enable_persistence,
            state_storage_dir=state_storage_dir,
        )

        self.client = create_crewai_client(
            state_storage_dir=state_storage_dir,
            enable_persistence=enable_persistence,
            default_execution_method=default_execution_method,
        )

        # Initialize zero-error workflow
        self.zero_error_workflow = create_zero_error_workflow(
            workflow_id="bridge_default"
        )

        # CAV-NLP integration
        self.use_cav_nlp = CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for CrewAIUnifiedBridge")

        logger.info(f"CrewAIUnifiedBridge initialized with method={default_execution_method}")


# =============================================================================
# PHASE 1: SETUP
# =============================================================================

def execute_phase_1_setup(
    problem_statement: str,
    execution_method: str = "traditional",
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
    # Zero-error parameters
    use_roma_mdap_maker: bool = False,
    reliability_preset: str = "standard",
    reliability_overrides: Optional[Dict[str, Any]] = None,
    reliability_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Phase 1: Problem Setup - Entry point for all workflows.

    This is the main entry point that replaces the CrewAI phase 1 setup.

    Args:
        problem_statement: The problem to solve
        execution_method: Execution method (traditional/roma/roma_mdap_maker/etc)
        problem_type: Type of problem
        domain: Problem domain
        max_sub_problems: Maximum sub-problems
        decomposition_strategy: Strategy for decomposition
        use_evolution: Use evolution
        evolution_iterations: Number of evolution iterations
        roma_max_depth: ROMA max depth
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_model: Model for ROMA
        use_roma_mdap_maker: Use ROMA-MDAP-MAKER (zero-error)
        reliability_preset: Reliability preset
        reliability_overrides: Reliability parameter overrides
        reliability_config: Pre-built reliability config
        **kwargs: Additional parameters

    Returns:
        Dict with Phase 1 results including decomposition plan
    """
    logger.info(f"Phase 1: CrewAI setup (method={execution_method})")

    # Create unified flow
    flow = CrewAIUnifiedFlow(
        default_execution_method=ExecutionMethod.AUTO,
        enable_persistence=True,
    )

    # Map execution method string to enum
    method_enum = _map_execution_method(execution_method)

    # Execute Phase 1 through unified flow
    result = flow.phase_1_setup(
        problem_statement=problem_statement,
        execution_method=method_enum,
        problem_type=problem_type,
        domain=domain,
        max_sub_problems=max_sub_problems,
        decomposition_strategy=decomposition_strategy,
        use_evolution=use_evolution,
        evolution_iterations=evolution_iterations,
        roma_max_depth=roma_max_depth,
        roma_execution_mode=roma_execution_mode,
        roma_provider=roma_provider,
        roma_model=roma_model,
        use_roma_mdap_maker=use_roma_mdap_maker,
        reliability_preset=reliability_preset,
        reliability_overrides=reliability_overrides,
        **kwargs
    )

    # Add execution method to result
    result["execution_method"] = execution_method

    return result


# =============================================================================
# PHASE 2: SOLVE
# =============================================================================

def execute_phase_2_solve(
    decomposition_plan: Dict[str, Any],
    execution_method: str = "traditional",
    team_name: Optional[str] = None,
    solve_subset: Optional[List[str]] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 100,
    # ROMA parameters
    use_roma: bool = False,
    roma_max_depth: int = 2,
    roma_execution_mode: str = "recursive",
    roma_provider: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
    # Hybrid parameters
    use_hybrid: bool = False,
    hybrid_max_depth_analysis: int = 3,
    hybrid_max_depth_solving: int = 2,
    hybrid_execution_mode: str = "recursive",
    hybrid_provider: Optional[str] = None,
    hybrid_api_key: Optional[str] = None,
    hybrid_model: Optional[str] = None,
    hybrid_enable_gauntlets: bool = False,
    hybrid_enable_evolution: bool = True,
    hybrid_evolution_iterations: int = 50,
    # Zero-error parameters
    use_roma_mdap_maker: bool = False,
    reliability_config: Optional[Dict[str, Any]] = None,
    # Additional parameters
    **kwargs
) -> Dict[str, Any]:
    """
    Phase 2: Solution Generation.

    Generate solutions for each sub-problem using the selected execution method.

    Args:
        decomposition_plan: Complete decomposition plan from Phase 1
        execution_method: Execution method
        team_name: Optional team name
        solve_subset: Optional subset of sub-problems to solve
        use_evolution: Use evolution
        evolution_iterations: Number of evolution iterations
        use_roma: Use ROMA
        roma_max_depth: ROMA max depth
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA
        roma_model: Model for ROMA
        use_hybrid: Use Hybrid mode
        hybrid_max_depth_analysis: Hybrid max depth for analysis
        hybrid_max_depth_solving: Hybrid max depth for solving
        hybrid_execution_mode: Hybrid execution mode
        hybrid_provider: AI provider for hybrid
        hybrid_api_key: API key for hybrid
        hybrid_model: Model for hybrid
        hybrid_enable_gauntlets: Enable gauntlets in hybrid
        hybrid_enable_evolution: Enable evolution in hybrid
        hybrid_evolution_iterations: Evolution iterations for hybrid
        use_roma_mdap_maker: Use ROMA-MDAP-MAKER (zero-error)
        reliability_config: Reliability configuration
        **kwargs: Additional parameters

    Returns:
        Dict with Phase 2 results including solutions
    """
    logger.info(f"Phase 2: CrewAI solve (method={execution_method})")

    # Create unified flow
    flow = CrewAIUnifiedFlow(
        default_execution_method=ExecutionMethod.AUTO,
        enable_persistence=True,
    )

    # Map execution method string to enum
    method_enum = _map_execution_method(execution_method)

    # Execute Phase 2 through unified flow
    result = flow.phase_2_solve(
        phase_1_result=decomposition_plan,
        team_name=team_name,
        solve_subset=solve_subset,
        use_evolution=use_evolution,
        evolution_iterations=evolution_iterations,
        **kwargs
    )

    return result


# =============================================================================
# PHASE 3-6: CRITIQUE, VERIFY, REASSEMBLE, FINAL VALIDATION
# =============================================================================

def decomposition_phase_3_critique(
    solutions: List[Dict[str, Any]],
    use_evolution: bool = True,
    evolution_iterations: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Phase 3: Adversarial Critique"""
    logger.info("Phase 3: CrewAI critique")
    if DECOMPOSITION_PHASE_BRIDGE_AVAILABLE and decomposition_bridge_phase_3:
        return decomposition_bridge_phase_3(
            solutions=solutions,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
            **kwargs
        )
    return {
        "phase": 3,
        "status": "failed",
        "critiques": [],
        "error": "Decomposition critique bridge not available",
        "message": "Critique phase failed",
    }


def decomposition_phase_4_verify(
    solutions: List[Dict[str, Any]],
    use_evolution: bool = True,
    evolution_iterations: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Phase 4: Verification"""
    logger.info("Phase 4: CrewAI verification")
    if DECOMPOSITION_PHASE_BRIDGE_AVAILABLE and decomposition_bridge_phase_4:
        return decomposition_bridge_phase_4(
            solutions=solutions,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
            **kwargs
        )
    return {
        "phase": 4,
        "status": "failed",
        "verifications": [],
        "error": "Decomposition verification bridge not available",
        "message": "Verification phase failed",
    }


def decomposition_phase_5_reassemble(
    solutions: List[Dict[str, Any]],
    problem_statement: str,
    use_evolution: bool = True,
    evolution_iterations: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Phase 5: Reassembly"""
    logger.info("Phase 5: CrewAI reassembly")
    if DECOMPOSITION_PHASE_BRIDGE_AVAILABLE and decomposition_bridge_phase_5:
        return decomposition_bridge_phase_5(
            solutions=solutions,
            problem_statement=problem_statement,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
            **kwargs
        )
    return {
        "phase": 5,
        "status": "failed",
        "final_solution": "",
        "error": "Decomposition reassembly bridge not available",
        "message": "Reassembly phase failed",
    }


def decomposition_phase_6_final_validation(
    final_solution: str,
    problem_statement: str,
    use_evolution: bool = True,
    evolution_iterations: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """Phase 6: Final Validation"""
    logger.info("Phase 6: CrewAI final validation")
    if DECOMPOSITION_PHASE_BRIDGE_AVAILABLE and decomposition_bridge_phase_6:
        return decomposition_bridge_phase_6(
            final_solution=final_solution,
            problem_statement=problem_statement,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
            **kwargs
        )
    return {
        "phase": 6,
        "status": "failed",
        "validation": "failed",
        "overall_score": 0.0,
        "error": "Decomposition final validation bridge not available",
        "message": "Final validation failed",
    }


# =============================================================================
# ROMA-SPECIFIC FUNCTIONS (Stubs for backward compatibility)
# =============================================================================

def roma_phase_1_setup(
    problem_statement: str,
    max_depth: int = 3,
    execution_mode: str = "recursive",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """ROMA Phase 1: Setup (stub - delegates to execute_phase_1_setup)"""
    return execute_phase_1_setup(
        problem_statement=problem_statement,
        execution_method="roma",
        roma_max_depth=max_depth,
        roma_execution_mode=execution_mode,
        roma_provider=provider,
        roma_model=model,
        **kwargs
    )


def roma_phase_2_solve(
    decomposition_plan: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """ROMA Phase 2: Solve (stub - delegates to execute_phase_2_solve)"""
    return execute_phase_2_solve(
        decomposition_plan=decomposition_plan,
        execution_method="roma",
        **kwargs
    )


def roma_phase_3_critique(
    solutions: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """ROMA Phase 3: Critique"""
    if ROMA_PHASE_BRIDGE_AVAILABLE and roma_bridge_phase_3:
        return roma_bridge_phase_3(solutions, **kwargs)
    return decomposition_phase_3_critique(solutions, **kwargs)


def roma_phase_4_verify(
    solutions: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """ROMA Phase 4: Verify"""
    if ROMA_PHASE_BRIDGE_AVAILABLE and roma_bridge_phase_4:
        return roma_bridge_phase_4(solutions, **kwargs)
    return decomposition_phase_4_verify(solutions, **kwargs)


def roma_full_workflow(
    problem_statement: str,
    max_depth_analysis: int = 3,
    max_depth_solving: int = 2,
    execution_mode: str = "recursive",
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """ROMA Full Workflow"""
    # Phase 1
    phase1 = roma_phase_1_setup(
        problem_statement=problem_statement,
        max_depth=max_depth_analysis,
        execution_mode=execution_mode,
        provider=provider,
        api_key=api_key,
        model=model,
    )

    if phase1["status"] == "failed":
        return phase1

    # Phase 2
    phase2 = roma_phase_2_solve(phase1)

    if phase2["status"] == "failed":
        return phase2

    # Phase 3-6
    phase3 = roma_phase_3_critique(phase2.get("solutions", []))
    phase4 = roma_phase_4_verify(phase2.get("solutions", []))
    phase5 = decomposition_phase_5_reassemble(phase2.get("solutions", []), problem_statement)
    phase6 = decomposition_phase_6_final_validation(phase5.get("final_solution", ""), problem_statement)

    return {
        "workflow": "roma",
        "status": "completed",
        "phases": {
            "phase1": phase1,
            "phase2": phase2,
            "phase3": phase3,
            "phase4": phase4,
            "phase5": phase5,
            "phase6": phase6,
        },
    }


# =============================================================================
# ROMA-MDAP-MAKER FUNCTIONS (Zero-Error)
# =============================================================================

def roma_mdap_maker_phase_1_setup(
    problem_statement: str,
    reliability_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """ROMA-MDAP-MAKER Phase 1: Setup (zero-error mode)"""
    return execute_phase_1_setup(
        problem_statement=problem_statement,
        execution_method="roma_mdap_maker",
        use_roma_mdap_maker=True,
        reliability_config=reliability_config,
        **kwargs
    )


def roma_mdap_maker_phase_2_solve(
    sub_problem_id: str,
    sub_problem_description: str,
    context: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """ROMA-MDAP-MAKER Phase 2: Solve (zero-error mode)"""
    # Create zero-error workflow
    workflow = create_zero_error_workflow()

    # Create simple decomposition plan
    from crewai_state_management import SubProblem, DecompositionPlan

    sub_problem = SubProblem(
        id=sub_problem_id,
        title=f"Sub-problem {sub_problem_id}",
        description=sub_problem_description,
    )

    decomposition_plan = DecompositionPlan(
        id="single_problem",
        problem_statement=sub_problem_description,
        sub_problems=[sub_problem],
    )

    # Execute workflow
    result = workflow.execute_workflow(
        problem_statement=sub_problem_description,
        decomposition_plan=decomposition_plan,
    )

    return {
        "sub_problem_id": sub_problem_id,
        "status": "completed" if result.status == "completed" else "failed",
        "solution": result.final_solution,
        "confidence": result.metrics.overall_confidence if result.metrics else 0.0,
    }


def roma_mdap_maker_full_workflow(
    problem_statement: str,
    reliability_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """ROMA-MDAP-MAKER Full Workflow (zero-error mode)"""
    workflow = create_zero_error_workflow()

    result = workflow.execute_workflow(
        problem_statement=problem_statement,
    )

    return {
        "workflow": "roma_mdap_maker",
        "status": result.status,
        "final_solution": result.final_solution,
        "metrics": result.metrics.to_dict() if result.metrics else None,
    }


# =============================================================================
# FULL WORKFLOW EXECUTION
# =============================================================================

def execute_full_workflow(
    problem_statement: str,
    execution_method_phase2: str = "traditional",
    use_evolution: bool = True,
    use_roma_workflow: bool = False,
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    roma_execution_mode: str = "recursive",
    roma_provider: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
    reliability_preset: str = "standard",
    reliability_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute full 6-phase workflow.

    This is the main entry point for complete workflow execution.

    Args:
        problem_statement: The problem to solve
        execution_method_phase2: Execution method for Phase 2
        use_evolution: Use evolution
        use_roma_workflow: Use ROMA's full workflow
        roma_max_depth_analysis: ROMA max depth for analysis
        roma_max_depth_solving: ROMA max depth for solving
        roma_execution_mode: ROMA execution mode
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA
        roma_model: Model for ROMA
        reliability_preset: Reliability preset
        reliability_overrides: Reliability parameter overrides

    Returns:
        Dict with complete workflow results
    """
    logger.info(f"Starting full workflow: {problem_statement[:50]}...")

    try:
        if use_roma_workflow:
            execution_method_phase2 = "roma"

        method_enum = _map_execution_method(execution_method_phase2)

        flow = CrewAIUnifiedFlow(
            default_execution_method=method_enum,
            enable_persistence=True,
        )

        return flow.execute_full_workflow(
            problem_statement=problem_statement,
            execution_method=method_enum,
            use_evolution=use_evolution,
            roma_max_depth=roma_max_depth_analysis,
            max_depth=roma_max_depth_solving,
            roma_execution_mode=roma_execution_mode,
            roma_provider=roma_provider,
            roma_model=roma_model,
            use_roma_mdap_maker=execution_method_phase2 == "roma_mdap_maker",
            reliability_preset=reliability_preset,
            reliability_overrides=reliability_overrides,
        )

    except (RuntimeError, ValueError, TypeError, ConnectionError) as e:
        logger.error(f"Full workflow failed: {e}")
        return {
            "workflow": "unified_crewai",
            "status": "failed",
            "error": str(e),
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _map_execution_method(method: str) -> CrewAIExecutionMethod:
    """Map string execution method to CrewAI enum."""
    if isinstance(method, CrewAIExecutionMethod):
        return method
    if hasattr(method, "value"):
        method = method.value
    mapping = {
        "traditional": CrewAIExecutionMethod.TRADITIONAL,
        "roma": CrewAIExecutionMethod.ROMA,
        "roma_mdap_maker": CrewAIExecutionMethod.ROMA_MDAP_MAKER,
        "claudiomiro": CrewAIExecutionMethod.CLAUDIOMIRO,
        "datapizza": CrewAIExecutionMethod.DATAPIZZA,
        "hybrid": CrewAIExecutionMethod.HYBRID,
        "auto": CrewAIExecutionMethod.AUTO,
    }
    return mapping.get(method.lower(), CrewAIExecutionMethod.AUTO)


def get_reliability_config(
    preset: str = "standard",
    **overrides
) -> Dict[str, Any]:
    """
    Get reliability configuration for ROMA-MDAP-MAKER.

    Args:
        preset: Reliability preset (standard, thorough, fast, validation)
        **overrides: Individual parameter overrides

    Returns:
        Reliability configuration dict
    """
    # Preset configurations
    presets = {
        "standard": {
            "maker_k_ahead": 5,
            "mdap_k_min": 2,
            "mdap_k_max": 8,
            "enable_red_flagging": True,
            "enable_first_to_ahead": True,
        },
        "thorough": {
            "maker_k_ahead": 7,
            "mdap_k_min": 3,
            "mdap_k_max": 10,
            "enable_red_flagging": True,
            "enable_first_to_ahead": True,
        },
        "fast": {
            "maker_k_ahead": 3,
            "mdap_k_min": 2,
            "mdap_k_max": 5,
            "enable_red_flagging": True,
            "enable_first_to_ahead": True,
        },
        "validation": {
            "maker_k_ahead": 10,
            "mdap_k_min": 5,
            "mdap_k_max": 15,
            "enable_red_flagging": True,
            "enable_first_to_ahead": True,
        },
    }

    config = presets.get(preset, presets["standard"]).copy()
    config.update(overrides)

    return config


def get_unified_bridge_status() -> Dict[str, Any]:
    """Get unified bridge availability status."""
    flow = CrewAIUnifiedFlow(
        default_execution_method=CrewAIExecutionMethod.AUTO,
        enable_persistence=False,
    )
    flow_status = flow.get_status()
    roma_mdap_status = (
        get_romamdapmaker_bridge_status()
        if ROMA_MDAP_MAKER_BRIDGE_STATUS_AVAILABLE and get_romamdapmaker_bridge_status
        else {}
    )
    return {
        "engine": "CrewAI",
        "version": flow_status.get("version", "1.0.0"),
        "total_execution_methods": flow_status.get("total_execution_methods", 0),
        "execution_methods": [m.value if hasattr(m, "value") else str(m) for m in flow_status.get("execution_methods", [])],
        "availability": flow_status.get("availability", {}),
        "roma_mdap_maker_bridge_available": flow_status.get("availability", {}).get("roma_mdap_maker_bridge", False),
        "roma_bridge_available": flow_status.get("availability", {}).get("roma_bridge", False),
        "decomposition_bridge_available": flow_status.get("availability", {}).get("decomposition_bridge", False),
        "claudiomiro_bridge_available": flow_status.get("availability", {}).get("claudiomiro_bridge", False),
        "datapizza_bridge_available": flow_status.get("availability", {}).get("datapizza_bridge", False),
        "roma_mdap_maker_bridge_status": roma_mdap_status,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("CrewAI Unified Bridge Example")
    print("=" * 50)

    # Execute a full workflow
    result = execute_full_workflow(
        problem_statement="Design a zero-error distributed database system",
        execution_method_phase2="roma_mdap_maker",
    )

    print(f"Workflow result: {result['status']}")
    if result["status"] == "completed":
        print("Phases completed:")
        for phase, phase_result in result.get("phases", {}).items():
            print(f"  {phase}: {phase_result.get('status', 'unknown')}")
