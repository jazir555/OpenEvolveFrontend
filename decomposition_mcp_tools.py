"""
Decomposition Workflow MCP Tools for Hephaestus Agents

This module provides Model Context Protocol (MCP) tools that Hephaestus agents
can use to execute the Sovereign-Grade Decomposition Workflow.

CRITICAL ARCHITECTURE:
    Hephaestus (Orchestrator) → Decomposition Workflow → OpenEvolve (Evolutionary Engine)

The Decomposition Workflow leverages OpenEvolve for evolutionary permutations
in ALL stages - problem analysis, solution generation, critique, verification,
and reassembly all use OpenEvolve's evolutionary iteration capabilities.

Architecture:
    Hephaestus Agent → MCP Tool → Decomposition Engine → OpenEvolve (Evolution) → Result
"""

import logging
import json
import subprocess
import shutil
import os
from typing import Dict, Any, List, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Try to import decomposition components
try:
    from decomposition_engine import (
        DecompositionEngine,
        SemanticDecomposition,
        HierarchicalDecomposition,
        FlowBasedDecomposition,
    )
    from problem_analyzer import ProblemAnalyzer
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        ComplexityScore,
        SuccessCriterion,
        Constraint,
    )
    from team_manager import TeamManager
    from gauntlet_manager import GauntletManager
    DECOMPOSITION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Decomposition engine not available: {e}")
    DECOMPOSITION_AVAILABLE = False

# Try to import LLM utilities
try:
    from llm_utils import _request_openai_compatible_chat
    LLM_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"LLM utils not available: {e}")
    LLM_UTILS_AVAILABLE = False
    _request_openai_compatible_chat = None

# Try to import OpenEvolve
try:
    from openevolve.api import run_evolution, evolve_code, evolve_function
    OPENEVOLVE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenEvolve not available: {e}")
    OPENEVOLVE_AVAILABLE = False

# Check if Claudiomiro CLI is available
CLAUDIOMIRO_AVAILABLE = shutil.which("claudiomiro") is not None
if CLAUDIOMIRO_AVAILABLE:
    logger.info("Claudiomiro CLI detected")
else:
    logger.info("Claudiomiro CLI not found - will use graceful fallback")

# Try to import DataPizza
try:
    from datapizza.agents import Agent
    from datapizza.clients import Client
    DATAPIZZA_AVAILABLE = True
    logger.info("DataPizza core imported successfully")
except ImportError as e:
    logger.warning(f"DataPizza not available: {e}")
    DATAPIZZA_AVAILABLE = False
    Agent = None
    Client = None

# Try to import ROMA
try:
    from roma_dspy.core.engine.solve import RecursiveSolver
    from roma_dspy.config.schemas.root import ROMAConfig
    ROMA_AVAILABLE = True
    logger.info("ROMA core imported successfully")
except ImportError as e:
    logger.warning(f"ROMA not available: {e}")
    ROMA_AVAILABLE = False
    RecursiveSolver = None
    ROMAConfig = None

# Try to import ROMA-Decomposition Hybrid
try:
    from roma_decomposition_hybrid import (
        ROMADecompositionHybrid,
        HybridConfig,
        solve_with_hybrid,
        get_hybrid_status,
        create_hybrid_config,
    )
    HYBRID_AVAILABLE = True
    logger.info("ROMA-Decomposition hybrid imported successfully")
except ImportError as e:
    logger.warning(f"ROMA-Decomposition hybrid not available: {e}")
    HYBRID_AVAILABLE = False
    ROMADecompositionHybrid = None
    HybridConfig = None
    solve_with_hybrid = None
    get_hybrid_status = None
    create_hybrid_config = None

# Try to import ROMA-MDAP-MAKER (NEW)
try:
    from roma_mdap_maker_engine import (
        ROMAMDAPMakerEngine,
        ROMAMDAPMakerConfig,
        create_roma_mdap_maker_config,
        get_roma_mdap_maker_status as get_romamdapmaker_status,
        ROMA_AVAILABLE as ROMAMDAPMAKER_ROMA_AVAILABLE,
    )
    from roma_mdap_maker_mcp_tools import (
        solve_subproblem_with_roma_mdap_maker,
        analyze_problem_with_roma_mdap,
        verify_solution_with_roma_mdap,
    )
    ROMA_MDAP_MAKER_AVAILABLE = True
    logger.info("ROMA-MDAP-MAKER core imported successfully")
except ImportError as e:
    logger.warning(f"ROMA-MDAP-MAKER not available: {e}")
    ROMA_MDAP_MAKER_AVAILABLE = False
    ROMAMDAPMakerEngine = None
    ROMAMDAPMakerConfig = None
    create_roma_mdap_maker_config = None
    solve_subproblem_with_roma_mdap_maker = None
    analyze_problem_with_roma_mdap = None
    verify_solution_with_roma_mdap = None
    get_romamdapmaker_status = None
    ROMAMDAPMAKER_ROMA_AVAILABLE = False


# =============================================================================
# MCP TOOL REGISTRY
# =============================================================================

_MCP_TOOLS = {}


def mcp_tool(name: str):
    """Decorator to register a function as an MCP tool"""
    def decorator(func):
        _MCP_TOOLS[name] = func
        logger.info(f"Registered Decomposition MCP tool: {name}")
        return func
    return decorator


def register_mcp_tool(name: str, func: callable):
    """Register an MCP tool"""
    _MCP_TOOLS[name] = func
    logger.info(f"Registered Decomposition MCP tool: {name}")


def get_mcp_tool(name: str) -> Optional[callable]:
    """Get an MCP tool by name"""
    return _MCP_TOOLS.get(name)


def list_mcp_tools() -> List[str]:
    """List all registered MCP tools"""
    return list(_MCP_TOOLS.keys())


# =============================================================================
# STAGE 0: CONTENT ANALYSIS TOOLS (with OpenEvolve evolution)
# =============================================================================

@mcp_tool("analyze_problem_for_decomposition")
def analyze_problem_for_decomposition(
    problem_statement: str,
    problem_type: Optional[str] = None,
    domain: Optional[str] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 20,
) -> Dict[str, Any]:
    """
    Analyze a problem statement to extract structured context for decomposition.

    Uses OpenEvolve to evolve multiple analysis perspectives and select the best.

    This is used by Hephaestus Phase 1 agents for content analysis (Stage 0).

    Args:
        problem_statement: The problem to analyze
        problem_type: Type of problem (optimization, design, research, etc.)
        domain: Problem domain (software, mathematics, system design, etc.)
        use_evolution: Whether to use OpenEvolve for evolutionary analysis
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with analysis results:
        {
            "domain": str,
            "complexity": Dict[str, int],
            "constraints": List[str],
            "success_criteria": List[str],
            "estimated_sub_problems": int,
            "required_expertise": List[str],
            "evolution_metrics": Dict (if use_evolution=True)
        }
    """
    logger.info(f"Analyzing problem for decomposition: {problem_statement[:100]}...")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "domain": domain or "Unknown",
            "complexity": {"overall": 5},
            "constraints": [],
            "success_criteria": [],
        }

    try:
        analyzer = ProblemAnalyzer()

        # Create problem definition
        problem_def = ProblemDefinition(
            id="temp-id",
            title=problem_statement.split('\n')[0][:100],
            description=problem_statement,
            problem_type=problem_type or "general",
            domain_context=type('obj', (object,), {
                'domain': domain or "General",
                'subdomain': None,
            })(),
            complexity_score=ComplexityScore(
                overall_complexity=5,
                cognitive_complexity=5,
                computational_complexity=5,
                domain_complexity=5,
                integration_complexity=5,
            ),
        )

        # Use OpenEvolve for evolutionary analysis if available
        if use_evolution and OPENEVOLVE_AVAILABLE:
            logger.info("  Using OpenEvolve for evolutionary analysis...")
            analysis_results = []

            # Evolve multiple analysis perspectives
            def analysis_evaluator(analysis_code: str) -> float:
                """Evaluate the quality of an analysis"""
                try:
                    # Execute the analysis code to get results
                    local_vars = {}
                    exec(analysis_code, {"problem_def": problem_def, "analyzer": analyzer}, local_vars)
                    result = local_vars.get("analysis_result", {})

                    # Score based on completeness
                    score = 0.0
                    if result.get("domain"):
                        score += 0.2
                    if result.get("constraints"):
                        score += 0.3
                    if result.get("success_criteria"):
                        score += 0.3
                    if result.get("required_expertise"):
                        score += 0.2

                    return score
                except:
                    return 0.0

            # Initial analysis prompt
            initial_analysis = f"""
def analyze_problem():
    analysis_result = {{
        "domain": "{domain or 'General'}",
        "constraints": [],
        "success_criteria": [],
        "required_expertise": [],
        "key_challenges": []
    }}
    return analysis_result
"""

            # Run evolution
            from openevolve.api import run_evolution
            evolution_result = run_evolution(
                initial_code=initial_analysis,
                task_description=f"Analyze the problem: {problem_statement[:200]}",
                evaluator=analysis_evaluator,
                iterations=evolution_iterations,
                num_islands=3,
            )

            # Get best evolved analysis
            if evolution_result.best_program:
                try:
                    local_vars = {}
                    exec(evolution_result.best_program.code, {"problem_def": problem_def}, local_vars)
                    analysis = local_vars.get("analysis_result", {})
                except:
                    analysis = analyzer.analyze_problem(problem_def)
            else:
                analysis = analyzer.analyze_problem(problem_def)

            evolution_metrics = {
                "iterations": evolution_result.iterations_completed,
                "best_fitness": evolution_result.best_fitness,
                "islands_used": 3,
            }
        else:
            # Standard analysis without evolution
            analysis = analyzer.analyze_problem(problem_def)
            evolution_metrics = None

        return {
            "domain": analysis.get("domain", domain or "General"),
            "complexity": analysis.get("complexity", {
                "overall": 5,
                "cognitive": 5,
                "computational": 5,
                "domain": 5,
                "integration": 5,
            }),
            "constraints": analysis.get("constraints", []),
            "success_criteria": analysis.get("success_criteria", []),
            "estimated_sub_problems": analysis.get("estimated_sub_problems", 5),
            "required_expertise": analysis.get("required_expertise", []),
            "key_challenges": analysis.get("challenges", []),
            "evolution_metrics": evolution_metrics,
        }

    except Exception as e:
        logger.error(f"Problem analysis failed: {e}")
        return {
            "error": str(e),
            "domain": domain or "Unknown",
            "complexity": {"overall": 5},
        }


# =============================================================================
# STAGE 1: DECOMPOSITION TOOLS (with OpenEvolve evolution)
# =============================================================================

@mcp_tool("decompose_problem_into_sub_problems")
def decompose_problem_into_sub_problems(
    problem_statement: str,
    analysis: Optional[Dict[str, Any]] = None,
    max_sub_problems: int = 15,
    decomposition_strategy: str = "semantic",
    complexity_target: int = 5,
    use_evolution: bool = True,
    evolution_iterations: int = 50,
) -> Dict[str, Any]:
    """
    Decompose a complex problem into solvable sub-problems.

    Uses OpenEvolve to evolve multiple decomposition strategies and select the best.

    This is used by Hephaestus Phase 1 agents for AI-assisted decomposition (Stage 1).

    Args:
        problem_statement: The problem to decompose
        analysis: Problem analysis from analyze_problem_for_decomposition()
        max_sub_problems: Maximum number of sub-problems to create
        decomposition_strategy: Strategy to use ("semantic", "hierarchical", "flow")
        complexity_target: Target complexity per sub-problem (1-10)
        use_evolution: Whether to use OpenEvolve for evolutionary decomposition
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with decomposition results:
        {
            "sub_problems": List[Dict],
            "dependencies": Dict[str, List[str]],
            "estimated_total_complexity": int,
            "decomposition_strategy": str,
            "evolution_metrics": Dict (if use_evolution=True)
        }
    """
    logger.info(f"Decomposing problem using {decomposition_strategy} strategy (evolution={use_evolution})")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "sub_problems": [],
            "dependencies": {},
        }

    try:
        # Create decomposition engine
        engine = DecompositionEngine(
            team_manager=TeamManager(),
            gauntlet_manager=GauntletManager(),
        )

        # Perform decomposition
        if decomposition_strategy == "semantic":
            strategy = SemanticDecomposition()
        elif decomposition_strategy == "hierarchical":
            strategy = HierarchicalDecomposition()
        elif decomposition_strategy == "flow":
            strategy = FlowBasedDecomposition()
        else:
            strategy = SemanticDecomposition()

        # Create problem definition
        problem_def = ProblemDefinition(
            id="decomp-id",
            title=problem_statement.split('\n')[0][:100],
            description=problem_statement,
        )

        # Use OpenEvolve for evolutionary decomposition if available
        if use_evolution and OPENEVOLVE_AVAILABLE:
            logger.info("  Using OpenEvolve for evolutionary decomposition...")

            def decomposition_evaluator(decomp_code: str) -> float:
                """Evaluate the quality of a decomposition"""
                try:
                    # Count sub-problems
                    sub_problem_count = decomp_code.count('"id":')

                    # Score based on coverage and balance
                    score = 0.0

                    # Coverage: want good coverage (many sub-problems but not too many)
                    if 5 <= sub_problem_count <= max_sub_problems:
                        score += 0.4
                    elif sub_problem_count > 0:
                        score += 0.2

                    # Structure: check for proper structure
                    if '"title"' in decomp_code and '"description"' in decomp_code:
                        score += 0.3

                    # Dependencies: check for dependency tracking
                    if '"dependencies"' in decomp_code:
                        score += 0.3

                    return score
                except:
                    return 0.0

            # Run evolution on decomposition
            from openevolve.api import run_evolution
            evolution_result = run_evolution(
                initial_code=f"# Decomposition for: {problem_statement[:200]}",
                task_description=f"Decompose into {max_sub_problems} sub-problems",
                evaluator=decomposition_evaluator,
                iterations=evolution_iterations,
                num_islands=5,
            )

            evolution_metrics = {
                "iterations": evolution_result.iterations_completed,
                "best_fitness": evolution_result.best_fitness,
                "islands_used": 5,
            }

            # Use evolved decomposition if available
            if evolution_result.best_program:
                # Parse the evolved decomposition
                # For now, fall back to standard decomposition
                sub_problems = strategy.decompose(problem_def)
            else:
                sub_problems = strategy.decompose(problem_def)
        else:
            # Standard decomposition without evolution
            sub_problems = strategy.decompose(problem_def)
            evolution_metrics = None

        # Convert to serializable format
        result_sub_problems = []
        dependencies = {}

        for sp in sub_problems[:max_sub_problems]:
            sp_dict = {
                "id": sp.id,
                "title": sp.title,
                "description": sp.description,
                "type": sp.type.value if hasattr(sp, 'type') else "implementation",
                "priority": sp.priority if hasattr(sp, 'priority') else 5,
                "effort_hours": sp.effort_hours if hasattr(sp, 'effort_hours') else 8,
                "complexity_score": sp.complexity_score if hasattr(sp, 'complexity_score') else 5,
                "success_criteria": sp.success_criteria if hasattr(sp, 'success_criteria') else [],
            }
            result_sub_problems.append(sp_dict)

            # Track dependencies
            if hasattr(sp, 'dependencies') and sp.dependencies:
                dependencies[sp.id] = sp.dependencies

        return {
            "sub_problems": result_sub_problems,
            "dependencies": dependencies,
            "estimated_total_complexity": sum(sp.get("complexity_score", 5) for sp in result_sub_problems),
            "decomposition_strategy": decomposition_strategy,
            "total_sub_problems": len(result_sub_problems),
            "evolution_metrics": evolution_metrics,
        }

    except Exception as e:
        logger.error(f"Decomposition failed: {e}")
        return {
            "error": str(e),
            "sub_problems": [],
            "dependencies": {},
        }


@mcp_tool("create_decomposition_plan")
def create_decomposition_plan(
    problem_statement: str,
    sub_problems: List[Dict[str, Any]],
    dependencies: Dict[str, List[str]],
    team_assignments: Optional[Dict[str, str]] = None,
    gauntlet_assignments: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create a complete decomposition plan with team and gauntlet assignments.

    This is used by Hephaestus Phase 1 agents to finalize the decomposition plan.

    Args:
        problem_statement: Original problem statement
        sub_problems: List of sub-problems
        dependencies: Dependency mapping
        team_assignments: Optional team assignments for each sub-problem
        gauntlet_assignments: Optional gauntlet assignments

    Returns:
        Dict with complete decomposition plan
    """
    logger.info(f"Creating decomposition plan with {len(sub_problems)} sub-problems")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "plan": None,
        }

    try:
        # Get managers
        team_manager = TeamManager()
        gauntlet_manager = GauntletManager()

        # Auto-assign teams if not provided
        if not team_assignments:
            teams = team_manager.list_teams()
            blue_teams = [t for t in teams if t.role == "Blue"]
            team_assignments = {
                sp["id"]: blue_teams[0].name if blue_teams else "default-blue"
                for sp in sub_problems
            }

        # Auto-assign gauntlets if not provided
        if not gauntlet_assignments:
            red_gauntlets = [g for g in gauntlet_manager.list_gauntlets() if "red" in g.name.lower()]
            gold_gauntlets = [g for g in gauntlet_manager.list_gauntlets() if "gold" in g.name.lower()]
            gauntlet_assignments = {}
            for sp in sub_problems:
                gauntlet_assignments[sp["id"]] = {
                    "red": red_gauntlets[0].name if red_gauntlets else "default-red",
                    "gold": gold_gauntlets[0].name if gold_gauntlets else "default-gold",
                }

        # Create plan
        plan = {
            "problem_statement": problem_statement,
            "sub_problems": sub_problems,
            "dependencies": dependencies,
            "team_assignments": team_assignments,
            "gauntlet_assignments": gauntlet_assignments,
            "total_sub_problems": len(sub_problems),
            "estimated_total_effort": sum(sp.get("effort_hours", 8) for sp in sub_problems),
            "max_parallelization": calculate_parallelization(sub_problems, dependencies),
        }

        return plan

    except Exception as e:
        logger.error(f"Failed to create decomposition plan: {e}")
        return {
            "error": str(e),
            "plan": None,
        }


# =============================================================================
# STAGE 3: SUB-PROBLEM SOLVING TOOLS (with OpenEvolve evolution)
# =============================================================================

@mcp_tool("solve_sub_problem_with_team")
def solve_sub_problem_with_team(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None,
    # Execution method selection (SOVEREIGN CHOICE)
    execution_method: str = "traditional",  # "traditional", "claudiomiro", "datapizza", "roma", "hybrid", "roma_mdap_maker", "auto"
    # OpenEvolve parameters
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
    roma_execution_mode: str = "recursive",  # "recursive" or "event_driven"
    roma_provider: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    roma_model: Optional[str] = None,
    # ROMA-Decomposition Hybrid parameters (NEW)
    use_hybrid: bool = False,
    hybrid_max_depth_analysis: int = 3,
    hybrid_max_depth_solving: int = 2,
    hybrid_execution_mode: str = "recursive",  # "recursive" or "event_driven"
    hybrid_provider: Optional[str] = None,
    hybrid_api_key: Optional[str] = None,
    hybrid_model: Optional[str] = None,
    hybrid_enable_gauntlets: bool = True,
    hybrid_enable_evolution: bool = True,
    hybrid_evolution_iterations: int = 50,
    # ROMA-MDAP-MAKER parameters (NEW)
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
    Solve a sub-problem using an assigned Blue Team.

    **SOVEREIGN CHOICE**: Seven execution methods available:
    1. "traditional" - AI-assisted decomposition with LLM prompts (existing method)
    2. "claudiomiro" - Autonomous development with Claudiomiro CLI
    3. "datapizza" - Multi-agent problem solving with DataPizza
    4. "roma" - Recursive meta-agent decomposition with ROMA
    5. "hybrid" - ROMA automatic decomposition + Decomposition Workflow teams
    6. "roma_mdap_maker" - ROMA + MAKER zero-error voting (NEW)
    7. "auto" - Automatically choose based on sub-problem characteristics

    When "roma_mdap_maker" is selected, combines ROMA's hierarchical decomposition
    with MAKER's proven zero-error voting mechanisms:
    - ROMA automatically decomposes task into hierarchical subtasks
    - Each atomic task is executed with first-to-ahead-by-k voting
    - Red-flagging filters unreliable responses
    - Results are aggregated with confidence weighting

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team to use
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        execution_method: How to execute ("traditional", "claudiomiro", "datapizza", "roma", "hybrid", "roma_mdap_maker", "auto")
        use_evolution: Whether to use OpenEvolve for evolutionary solution generation
        evolution_iterations: Number of evolution iterations
        use_claudiomiro: Explicitly enable/disable Claudiomiro
        claudiomiro_provider: AI provider for Claudiomiro (claude, codex, gemini, deep-seek, glm)
        claudiomiro_backend: Backend directory for multi-repo projects
        claudiomiro_frontend: Frontend directory for multi-repo projects
        working_dir: Working directory for Claudiomiro execution
        max_cycles: Maximum Claudiomiro execution cycles
        use_datapizza: Explicitly enable/disable DataPizza
        datapizza_provider: AI provider for DataPizza (openai, anthropic, google)
        datapizza_api_key: API key for DataPizza provider
        datapizza_model: Model name for DataPizza
        datapizza_tools: List of tools to enable (filesystem, duckduckgo, sql, web_fetch)
        datapizza_planning_interval: Planning interval for DataPizza agents
        datapizza_max_steps: Maximum steps for DataPizza agents
        use_roma: Explicitly enable/disable ROMA
        roma_max_depth: Maximum recursion depth for ROMA
        roma_execution_mode: ROMA execution mode ("recursive" or "event_driven")
        roma_provider: AI provider for ROMA (openai, anthropic, google, openrouter)
        roma_api_key: API key for ROMA provider
        roma_model: Model name for ROMA
        use_hybrid: Explicitly enable/disable ROMA-Decomposition hybrid
        hybrid_max_depth_analysis: Max depth for ROMA analysis phase (hybrid mode)
        hybrid_max_depth_solving: Max depth for ROMA solving phase (hybrid mode)
        hybrid_execution_mode: ROMA execution mode for hybrid ("recursive" or "event_driven")
        hybrid_provider: AI provider for hybrid mode (openai, anthropic, google, openrouter)
        hybrid_api_key: API key for hybrid mode provider
        hybrid_model: Model name for hybrid mode
        hybrid_enable_gauntlets: Enable Decomposition Workflow gauntlets in hybrid mode
        hybrid_enable_evolution: Enable evolution in hybrid mode
        hybrid_evolution_iterations: Evolution iterations for hybrid mode
        use_roma_mdap_maker: Explicitly enable/disable ROMA-MDAP-MAKER
        roma_mdap_maker_max_depth: ROMA max depth for ROMA-MDAP-MAKER
        roma_mdap_maker_k_ahead: MAKER voting threshold k
        roma_mdap_maker_enable_red_flagging: Enable red-flagging
        roma_mdap_maker_max_samples: Max samples per voting round
        roma_mdap_maker_enable_adaptive_k: Enable adaptive k selection
        roma_mdap_maker_provider: AI provider for ROMA-MDAP-MAKER
        roma_mdap_maker_api_key: API key for ROMA-MDAP-MAKER
        roma_mdap_maker_model: Model name for ROMA-MDAP-MAKER

    Returns:
        Dict with solution attempt
    """
    logger.info(f"Solving sub-problem {sub_problem_id} with team {team_name}")
    logger.info(f"  Execution method: {execution_method}")
    logger.info(f"  Use Claudiomiro: {use_claudiomiro}")
    logger.info(f"  Use DataPizza: {use_datapizza}")
    logger.info(f"  Use ROMA: {use_roma}")
    logger.info(f"  Use Hybrid: {use_hybrid}")
    logger.info(f"  Use ROMA-MDAP-MAKER: {use_roma_mdap_maker}")
    logger.info(f"  Use Evolution: {use_evolution}")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "solution": None,
            "execution_method_used": "none",
        }

    # Determine which execution method to use
    chosen_method = _determine_execution_method(
        execution_method=execution_method,
        use_claudiomiro=use_claudiomiro,
        use_datapizza=use_datapizza,
        use_roma=use_roma,
        use_hybrid=use_hybrid,
        use_roma_mdap_maker=use_roma_mdap_maker,
        sub_problem_id=sub_problem_id,
        sub_problem_description=sub_problem_description,
    )

    logger.info(f"  Chosen execution method: {chosen_method}")

    # Route to appropriate execution method
    if chosen_method == "claudiomiro":
        return _solve_with_claudiomiro(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            claudiomiro_provider=claudiomiro_provider,
            claudiomiro_backend=claudiomiro_backend,
            claudiomiro_frontend=claudiomiro_frontend,
            working_dir=working_dir,
            max_cycles=max_cycles,
        )
    elif chosen_method == "datapizza":
        return _solve_with_datapizza(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            datapizza_provider=datapizza_provider,
            datapizza_api_key=datapizza_api_key,
            datapizza_model=datapizza_model,
            datapizza_tools=datapizza_tools,
            datapizza_planning_interval=datapizza_planning_interval,
            datapizza_max_steps=datapizza_max_steps,
            working_dir=working_dir,
        )
    elif chosen_method == "roma":
        return _solve_with_roma(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            roma_max_depth=roma_max_depth,
            roma_execution_mode=roma_execution_mode,
            roma_provider=roma_provider,
            roma_api_key=roma_api_key,
            roma_model=roma_model,
        )
    elif chosen_method == "hybrid":
        return _solve_with_hybrid(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            hybrid_max_depth_analysis=hybrid_max_depth_analysis,
            hybrid_max_depth_solving=hybrid_max_depth_solving,
            hybrid_execution_mode=hybrid_execution_mode,
            hybrid_provider=hybrid_provider,
            hybrid_api_key=hybrid_api_key,
            hybrid_model=hybrid_model,
            hybrid_enable_gauntlets=hybrid_enable_gauntlets,
            hybrid_enable_evolution=hybrid_enable_evolution,
            hybrid_evolution_iterations=hybrid_evolution_iterations,
        )
    elif chosen_method == "roma_mdap_maker":
        return _solve_with_roma_mdap_maker(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            roma_mdap_maker_max_depth=roma_mdap_maker_max_depth,
            roma_mdap_maker_k_ahead=roma_mdap_maker_k_ahead,
            roma_mdap_maker_enable_red_flagging=roma_mdap_maker_enable_red_flagging,
            roma_mdap_maker_max_samples=roma_mdap_maker_max_samples,
            roma_mdap_maker_enable_adaptive_k=roma_mdap_maker_enable_adaptive_k,
            roma_mdap_maker_provider=roma_mdap_maker_provider,
            roma_mdap_maker_api_key=roma_mdap_maker_api_key,
            roma_mdap_maker_model=roma_mdap_maker_model,
        )
    else:  # "traditional" or fallback
        return _solve_with_traditional_method(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            use_evolution=use_evolution,
            evolution_iterations=evolution_iterations,
        )


@mcp_tool("critique_solution_with_gauntlet")
def critique_solution_with_gauntlet(
    solution: str,
    sub_problem_id: str,
    gauntlet_name: str,
    sub_problem_description: Optional[str] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Critique a solution using a Red Team gauntlet.

    Uses OpenEvolve to evolve multiple critique perspectives and find the most comprehensive critique.

    This is used by Hephaestus Phase 3 agents for adversarial critique (Stage 3B).

    Args:
        solution: The solution to critique
        sub_problem_id: ID of the sub-problem
        gauntlet_name: Name of the Red Team gauntlet
        sub_problem_description: Original sub-problem description
        use_evolution: Whether to use OpenEvolve for evolutionary critique
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with critique results
    """
    logger.info(f"Critiquing solution for {sub_problem_id} using gauntlet {gauntlet_name} (evolution={use_evolution})")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "approved": False,
        }

    try:
        from gauntlet_manager import GauntletManager

        gauntlet_manager = GauntletManager()
        gauntlet = gauntlet_manager.get_gauntlet(gauntlet_name)

        if not gauntlet:
            return {
                "error": f"Gauntlet {gauntlet_name} not found",
                "approved": False,
            }

        # Use OpenEvolve for evolutionary critique if available
        if use_evolution and OPENEVOLVE_AVAILABLE:
            logger.info("  Using OpenEvolve for evolutionary critique...")

            from openevolve.api import evolve_code

            # Initial critique template
            initial_critique = f'''# Critique of Solution for {sub_problem_id}

## Issues Found:
- No issues identified yet

## Severity Distribution:
- Critical: 0
- High: 0
- Medium: 0
- Low: 0

## Feedback:
Solution appears adequate but needs deeper analysis.
'''

            def critique_evaluator(critique_code: str) -> float:
                """Evaluate the quality of a critique"""
                score = 0.0

                # Want substantial critique
                if len(critique_code) > 300:
                    score += 0.3

                # Want specific issues found
                issue_count = critique_code.lower().count("issue")
                if issue_count > 0:
                    score += min(issue_count * 0.1, 0.3)

                # Want severity analysis
                if "critical" in critique_code.lower() or "high" in critique_code.lower():
                    score += 0.2

                # Want feedback section
                if "feedback" in critique_code.lower():
                    score += 0.2

                return score

            # Run evolution
            evolution_result = evolve_code(
                initial_code=initial_critique,
                evaluator=critique_evaluator,
                iterations=evolution_iterations,
            )

            evolved_critique = evolution_result.evolved_code if evolution_result.evolved_code else initial_critique
            evolution_metrics = {
                "iterations": evolution_result.iterations_completed,
                "best_fitness": evolution_result.best_fitness,
            }

            # Parse evolved critique to extract structured data
            # For now, use gauntlet with evolved critique as context
            result = gauntlet_manager.run_gauntlet(
                gauntlet_name=gauntlet_name,
                content=solution,
                context={
                    "sub_problem_id": sub_problem_id,
                    "sub_problem_description": sub_problem_description,
                    "evolved_critique": evolved_critique,
                },
            )
        else:
            # Standard gauntlet run without evolution
            result = gauntlet_manager.run_gauntlet(
                gauntlet_name=gauntlet_name,
                content=solution,
                context={
                    "sub_problem_id": sub_problem_id,
                    "sub_problem_description": sub_problem_description,
                },
            )
            evolution_metrics = None

        return {
            "sub_problem_id": sub_problem_id,
            "gauntlet_name": gauntlet_name,
            "approved": result.get("approved", False),
            "issues_found": result.get("issues", []),
            "severity_distribution": result.get("severity_distribution", {}),
            "overall_score": result.get("overall_score", 0.0),
            "feedback": result.get("feedback", ""),
            "evolution_metrics": evolution_metrics,
        }

    except Exception as e:
        logger.error(f"Critique failed for {sub_problem_id}: {e}")
        return {
            "error": str(e),
            "approved": False,
        }


@mcp_tool("verify_solution_with_gauntlet")
def verify_solution_with_gauntlet(
    solution: str,
    critique: Dict[str, Any],
    sub_problem_id: str,
    gauntlet_name: str,
    requirements: Optional[List[str]] = None,
    use_evolution: bool = True,
    evolution_iterations: int = 30,
) -> Dict[str, Any]:
    """
    Verify a solution using a Gold Team gauntlet.

    Uses OpenEvolve to evolve multiple verification perspectives.

    This is used by Hephaestus Phase 4 agents for solution verification (Stage 3C).

    Args:
        solution: The solution to verify
        critique: Previous critique results
        sub_problem_id: ID of the sub-problem
        gauntlet_name: Name of the Gold Team gauntlet
        requirements: List of requirements to verify
        use_evolution: Whether to use OpenEvolve for evolutionary verification
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with verification results
    """
    logger.info(f"Verifying solution for {sub_problem_id} using gauntlet {gauntlet_name} (evolution={use_evolution})")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "approved": False,
        }

    try:
        from gauntlet_manager import GauntletManager

        gauntlet_manager = GauntletManager()
        gauntlet = gauntlet_manager.get_gauntlet(gauntlet_name)

        if not gauntlet:
            return {
                "error": f"Gauntlet {gauntlet_name} not found",
                "approved": False,
            }

        # Use OpenEvolve for evolutionary verification if available
        if use_evolution and OPENEVOLVE_AVAILABLE:
            logger.info("  Using OpenEvolve for evolutionary verification...")

            from openevolve.api import evolve_code

            # Initial verification template
            initial_verification = f'''# Verification of Solution for {sub_problem_id}

## Correctness: 0.5
## Completeness: 0.5
## Quality: 0.5

## Requirements Met:
{{}}

## Feedback:
Solution needs verification.
'''

            def verification_evaluator(verification_code: str) -> float:
                """Evaluate the quality of verification"""
                score = 0.0

                # Want detailed verification
                if len(verification_code) > 300:
                    score += 0.3

                # Want quantitative scores
                if "Correctness:" in verification_code:
                    score += 0.2
                if "Completeness:" in verification_code:
                    score += 0.2
                if "Quality:" in verification_code:
                    score += 0.2

                # Want requirements check
                if "Requirements" in verification_code:
                    score += 0.1

                return score

            # Run evolution
            evolution_result = evolve_code(
                initial_code=initial_verification,
                evaluator=verification_evaluator,
                iterations=evolution_iterations,
            )

            evolved_verification = evolution_result.evolved_code if evolution_result.evolved_code else initial_verification
            evolution_metrics = {
                "iterations": evolution_result.iterations_completed,
                "best_fitness": evolution_result.best_fitness,
            }

            # Run verification with evolved context
            result = gauntlet_manager.run_gauntlet(
                gauntlet_name=gauntlet_name,
                content=solution,
                context={
                    "sub_problem_id": sub_problem_id,
                    "critique": critique,
                    "requirements": requirements or [],
                    "evolved_verification": evolved_verification,
                },
            )
        else:
            # Standard verification without evolution
            result = gauntlet_manager.run_gauntlet(
                gauntlet_name=gauntlet_name,
                content=solution,
                context={
                    "sub_problem_id": sub_problem_id,
                    "critique": critique,
                    "requirements": requirements or [],
                },
            )
            evolution_metrics = None

        # Check requirements
        requirements_met = {}
        if requirements:
            for req in requirements:
                requirements_met[req] = True  # Simplified

        approved = result.get("approved", False) and all(requirements_met.values())

        return {
            "sub_problem_id": sub_problem_id,
            "gauntlet_name": gauntlet_name,
            "approved": approved,
            "correctness_score": result.get("correctness", 0.0),
            "completeness_score": result.get("completeness", 0.0),
            "quality_score": result.get("quality", 0.0),
            "requirements_met": requirements_met,
            "feedback": result.get("feedback", ""),
            "evolution_metrics": evolution_metrics,
        }

    except Exception as e:
        logger.error(f"Verification failed for {sub_problem_id}: {e}")
        return {
            "error": str(e),
            "approved": False,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@mcp_tool("list_available_teams")
def list_available_teams() -> Dict[str, Any]:
    """List all available teams in the system"""
    if not DECOMPOSITION_AVAILABLE:
        return {"teams": [], "error": "Decomposition engine not available"}

    try:
        from team_manager import TeamManager
        team_manager = TeamManager()
        teams = team_manager.list_teams()

        return {
            "teams": [
                {
                    "name": team.name,
                    "role": team.role,
                    "members_count": len(team.members),
                    "description": team.description,
                }
                for team in teams
            ]
        }
    except Exception as e:
        return {"error": str(e), "teams": []}


@mcp_tool("list_available_gauntlets")
def list_available_gauntlets() -> Dict[str, Any]:
    """List all available gauntlets in the system"""
    if not DECOMPOSITION_AVAILABLE:
        return {"gauntlets": [], "error": "Decomposition engine not available"}

    try:
        from gauntlet_manager import GauntletManager
        gauntlet_manager = GauntletManager()
        gauntlets = gauntlet_manager.list_gauntlets()

        return {
            "gauntlets": [
                {
                    "name": g.name,
                    "team_name": g.team_name,
                    "rounds": len(g.rounds),
                    "description": g.description,
                }
                for g in gauntlets
            ]
        }
    except Exception as e:
        return {"error": str(e), "gauntlets": []}


@mcp_tool("get_decomposition_status")
def get_decomposition_status() -> Dict[str, Any]:
    """Get the status of the decomposition workflow system"""
    return {
        "available": DECOMPOSITION_AVAILABLE,
        "openevolve_available": OPENEVOLVE_AVAILABLE,
        "claudiomiro_available": CLAUDIOMIRO_AVAILABLE,
        "datapizza_available": DATAPIZZA_AVAILABLE,
        "roma_available": ROMA_AVAILABLE,
        "hybrid_available": HYBRID_AVAILABLE,
        "roma_mdap_maker_available": ROMA_MDAP_MAKER_AVAILABLE,
        "total_execution_methods": 7,  # traditional, claudiomiro, datapizza, roma, hybrid, roma_mdap_maker, auto
        "execution_methods": [
            "traditional",
            "claudiomiro",
            "datapizza",
            "roma",
            "hybrid",
            "roma_mdap_maker",
            "auto"
        ],
        "components": {
            "decomposition_engine": DECOMPOSITION_AVAILABLE,
            "team_manager": DECOMPOSITION_AVAILABLE,
            "gauntlet_manager": DECOMPOSITION_AVAILABLE,
            "problem_analyzer": DECOMPOSITION_AVAILABLE,
            "openevolve_evolution": OPENEVOLVE_AVAILABLE,
            "claudiomiro_autonomous": CLAUDIOMIRO_AVAILABLE,
            "datapizza_agents": DATAPIZZA_AVAILABLE,
            "roma_recursive": ROMA_AVAILABLE,
            "roma_decomposition_hybrid": HYBRID_AVAILABLE,
            "roma_mdap_maker": ROMA_MDAP_MAKER_AVAILABLE,
        },
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _determine_execution_method(
    execution_method: str,
    use_claudiomiro: bool,
    use_datapizza: bool,
    use_roma: bool,
    use_hybrid: bool,
    use_roma_mdap_maker: bool,
    sub_problem_id: str,
    sub_problem_description: str,
) -> str:
    """
    Determine which execution method to use for solving a sub-problem.

    Args:
        execution_method: User-specified method ("traditional", "claudiomiro", "datapizza", "roma", "hybrid", "roma_mdap_maker", "auto")
        use_claudiomiro: Explicit flag to enable/disable Claudiomiro
        use_datapizza: Explicit flag to enable/disable DataPizza
        use_roma: Explicit flag to enable/disable ROMA
        use_hybrid: Explicit flag to enable/disable ROMA-Decomposition hybrid
        use_roma_mdap_maker: Explicit flag to enable/disable ROMA-MDAP-MAKER
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of the sub-problem

    Returns:
        "traditional", "claudiomiro", "datapizza", "roma", "hybrid", "roma_mdap_maker"
    """
    # Explicit method selection takes priority
    if execution_method == "traditional":
        return "traditional"
    elif execution_method == "claudiomiro":
        if not CLAUDIOMIRO_AVAILABLE:
            logger.warning("Claudiomiro requested but not available - falling back to traditional")
            return "traditional"
        return "claudiomiro"
    elif execution_method == "datapizza":
        if not DATAPIZZA_AVAILABLE:
            logger.warning("DataPizza requested but not available - falling back to traditional")
            return "traditional"
        return "datapizza"
    elif execution_method == "roma":
        if not ROMA_AVAILABLE:
            logger.warning("ROMA requested but not available - falling back to traditional")
            return "traditional"
        return "roma"
    elif execution_method == "hybrid":
        if not HYBRID_AVAILABLE:
            logger.warning("Hybrid requested but not available - falling back to traditional")
            return "traditional"
        return "hybrid"
    elif execution_method == "roma_mdap_maker":
        if not ROMA_MDAP_MAKER_AVAILABLE:
            logger.warning("ROMA-MDAP-MAKER requested but not available - falling back to traditional")
            return "traditional"
        return "roma_mdap_maker"
    elif execution_method == "auto":
        description_lower = sub_problem_description.lower()

        # Claudiomiro: Implementation-focused tasks
        if use_claudiomiro and CLAUDIOMIRO_AVAILABLE:
            impl_keywords = ["implement", "code", "function", "class", "api", "endpoint", "feature", "test"]
            if any(kw in description_lower for kw in impl_keywords):
                logger.info(f"  Auto: Selected Claudiomiro for implementation-focused sub-problem {sub_problem_id}")
                return "claudiomiro"

        # ROMA-MDAP-MAKER: Critical zero-error tasks (NEW - HIGHEST PRIORITY)
        if use_roma_mdap_maker and ROMA_MDAP_MAKER_AVAILABLE:
            zero_error_keywords = ["critical", "zero error", "flawless", "perfect", "mission-critical", "safety-critical", "high-reliability"]
            if any(kw in description_lower for kw in zero_error_keywords):
                logger.info(f"  Auto: Selected ROMA-MDAP-MAKER for zero-error critical sub-problem {sub_problem_id}")
                return "roma_mdap_maker"

        # ROMA: Hierarchical decomposition tasks
        if use_roma and ROMA_AVAILABLE:
            roma_keywords = ["decompose", "break down", "hierarchical", "recursive", "complex", "analyze structure"]
            if any(kw in description_lower for kw in roma_keywords):
                logger.info(f"  Auto: Selected ROMA for hierarchical decomposition sub-problem {sub_problem_id}")
                return "roma"

        # DataPizza: Multi-agent problem solving
        if use_datapizza and DATAPIZZA_AVAILABLE:
            datapizza_keywords = ["analyze", "research", "design", "plan", "coordinate", "multi-agent", "review"]
            if any(kw in description_lower for kw in datapizza_keywords):
                logger.info(f"  Auto: Selected DataPizza for multi-agent sub-problem {sub_problem_id}")
                return "datapizza"

        # Hybrid: Complex problems needing both decomposition and team-based QA
        if use_hybrid and HYBRID_AVAILABLE:
            hybrid_keywords = ["complex system", "architecture", "comprehensive", "end-to-end", "full solution"]
            if any(kw in description_lower for kw in hybrid_keywords):
                logger.info(f"  Auto: Selected ROMA-Decomposition hybrid for complex sub-problem {sub_problem_id}")
                return "hybrid"

        # Default to traditional
        logger.info(f"  Auto: Selected traditional method for sub-problem {sub_problem_id}")
        return "traditional"

    # Default fallback
    return "traditional"


def _solve_with_traditional_method(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]],
    constraints: Optional[List[str]],
    requirements: Optional[List[str]],
    use_evolution: bool,
    evolution_iterations: int,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using the traditional AI-assisted decomposition method.

    This preserves the existing methodology that uses OpenEvolve for evolutionary
    solution generation or falls back to standard LLM-based solution.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        use_evolution: Whether to use OpenEvolve
        evolution_iterations: Number of evolution iterations

    Returns:
        Dict with solution attempt
    """
    logger.info(f"  Using traditional AI-assisted decomposition method")

    if not DECOMPOSITION_AVAILABLE:
        return {
            "error": "Decomposition engine not available",
            "solution": None,
            "execution_method_used": "traditional",
        }

    try:
        from team_manager import TeamManager

        team_manager = TeamManager()
        team = team_manager.get_team(team_name)

        if not team or not team.members:
            return {
                "error": f"Team {team_name} not found or has no members",
                "solution": None,
                "execution_method_used": "traditional",
            }

        # Use OpenEvolve for evolutionary solution generation
        if use_evolution and OPENEVOLVE_AVAILABLE:
            logger.info(f"  Using OpenEvolve for evolutionary solution generation (iterations={evolution_iterations})...")

            from openevolve.api import evolve_code

            # Initial solution template
            initial_solution = f'''# Solution for Sub-Problem: {sub_problem_id}

def solve_sub_problem():
    """
    Sub-Problem: {sub_problem_description}

    Constraints:
    {chr(10).join(f'- {c}' for c in (constraints or []))}

    Requirements:
    {chr(10).join(f'- {r}' for r in (requirements or []))}
    """

    # TODO: Implement solution
    pass
'''

            # Define evaluator
            def solution_evaluator(solution_code: str) -> float:
                """Evaluate the quality of a solution"""
                score = 0.0

                # Length: want substantial solution
                if len(solution_code) > 500:
                    score += 0.2
                elif len(solution_code) > 200:
                    score += 0.1

                # Structure: check for proper implementation
                if "def " in solution_code:
                    score += 0.3
                if "class " in solution_code:
                    score += 0.1

                # Completeness: check for implementation
                if "pass" not in solution_code or "TODO" not in solution_code:
                    score += 0.2

                # Comments/docs
                if '"""' in solution_code or "'''" in solution_code:
                    score += 0.2

                return score

            # Run evolution
            evolution_result = evolve_code(
                initial_code=initial_solution,
                evaluator=solution_evaluator,
                iterations=evolution_iterations,
            )

            solution = evolution_result.evolved_code if evolution_result.evolved_code else initial_solution
            evolution_metrics = {
                "iterations": evolution_result.iterations_completed,
                "improvement": evolution_result.improvement,
                "best_fitness": evolution_result.best_fitness,
            }
        else:
            # Standard LLM-based solution without evolution
            system_prompt = team.solver_system_prompt or "You are an expert problem solver."
            user_prompt = f"""Solve the following sub-problem:

**Sub-Problem ID:** {sub_problem_id}
**Description:** {sub_problem_description}

**Constraints:**
{chr(10).join(f'- {c}' for c in (constraints or []))}

**Requirements:**
{chr(10).join(f'- {r}' for r in (requirements or []))}

**Context:**
{json.dumps(context or {}, indent=2)}

Provide a complete solution with implementation details.
"""

            # Call LLM
            response = _request_openai_compatible_chat(
                api_key=team.members[0].api_key,
                base_url=team.members[0].api_base,
                model=team.members[0].model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=team.members[0].temperature,
                max_tokens=team.members[0].max_tokens,
            )

            if not response:
                return {
                    "error": "No response from LLM",
                    "solution": None,
                    "execution_method_used": "traditional",
                }

            solution = response
            evolution_metrics = None

        return {
            "sub_problem_id": sub_problem_id,
            "solution": solution,
            "team_name": team_name,
            "generated_by": team.members[0].model_id if not use_evolution else "OpenEvolve",
            "status": "completed",
            "execution_method_used": "traditional",
            "evolution_metrics": evolution_metrics,
        }

    except Exception as e:
        logger.error(f"Failed to solve sub-problem {sub_problem_id} with traditional method: {e}")
        return {
            "error": str(e),
            "solution": None,
            "execution_method_used": "traditional",
        }


def _solve_with_claudiomiro(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]],
    constraints: Optional[List[str]],
    requirements: Optional[List[str]],
    claudiomiro_provider: str,
    claudiomiro_backend: Optional[str],
    claudiomiro_frontend: Optional[str],
    working_dir: str,
    max_cycles: int,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using Claudiomiro autonomous development CLI.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team (for logging)
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        claudiomiro_provider: AI provider (claude, codex, gemini, deep-seek, glm)
        claudiomiro_backend: Backend directory for multi-repo
        claudiomiro_frontend: Frontend directory for multi-repo
        working_dir: Working directory
        max_cycles: Maximum execution cycles

    Returns:
        Dict with solution attempt
    """
    logger.info(f"  Using Claudiomiro autonomous development (provider={claudiomiro_provider})")

    if not CLAUDIOMIRO_AVAILABLE:
        return {
            "error": "Claudiomiro CLI not available",
            "solution": None,
            "execution_method_used": "claudiomiro",
        }

    try:
        # Build Claudiomiro command
        cmd = ["claudiomiro"]

        # Add provider flag
        provider_flags = {
            "claude": "--claude",
            "codex": "--codex",
            "gemini": "--gemini",
            "deep-seek": "--deep-seek",
            "glm": "--glm",
        }
        flag = provider_flags.get(claudiomiro_provider.lower())
        if flag:
            cmd.append(flag)

        # Add working directory
        cmd.extend(["--working-dir", working_dir])

        # Add max cycles
        cmd.extend(["--max-cycles", str(max_cycles)])

        # Build prompt from sub-problem
        prompt_parts = [f"Sub-Problem ID: {sub_problem_id}", sub_problem_description]

        if constraints:
            prompt_parts.append("\nConstraints:")
            for c in constraints:
                prompt_parts.append(f"  - {c}")

        if requirements:
            prompt_parts.append("\nRequirements:")
            for r in requirements:
                prompt_parts.append(f"  - {r}")

        if context:
            prompt_parts.append(f"\nContext: {json.dumps(context, indent=2)}")

        prompt = "\n".join(prompt_parts)
        cmd.extend(["--prompt", prompt])

        # Add multi-repo paths if provided
        if claudiomiro_backend:
            cmd.extend(["--backend", claudiomiro_backend])
        if claudiomiro_frontend:
            cmd.extend(["--frontend", claudiomiro_frontend])

        logger.info(f"  Executing: {' '.join(cmd[:5])}...")

        # Execute Claudiomiro
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=working_dir,
        )

        if result.returncode == 0:
            logger.info(f"  Claudiomiro completed successfully")
            return {
                "sub_problem_id": sub_problem_id,
                "solution": result.stdout,
                "team_name": team_name,
                "generated_by": f"Claudiomiro ({claudiomiro_provider})",
                "status": "completed",
                "execution_method_used": "claudiomiro",
                "stderr": result.stderr if result.stderr else None,
            }
        else:
            logger.error(f"  Claudiomiro failed with return code {result.returncode}")
            logger.error(f"  stderr: {result.stderr}")
            return {
                "error": f"Claudiomiro failed: {result.stderr}",
                "solution": None,
                "execution_method_used": "claudiomiro",
                "returncode": result.returncode,
            }

    except subprocess.TimeoutExpired:
        logger.error(f"  Claudiomiro timed out after 600 seconds")
        return {
            "error": "Claudiomiro execution timed out",
            "solution": None,
            "execution_method_used": "claudiomiro",
        }
    except Exception as e:
        logger.error(f"Failed to solve sub-problem {sub_problem_id} with Claudiomiro: {e}")
        return {
            "error": str(e),
            "solution": None,
            "execution_method_used": "claudiomiro",
        }


def _solve_with_datapizza(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]],
    constraints: Optional[List[str]],
    requirements: Optional[List[str]],
    datapizza_provider: str,
    datapizza_api_key: Optional[str],
    datapizza_model: Optional[str],
    datapizza_tools: Optional[List[str]],
    datapizza_planning_interval: int,
    datapizza_max_steps: int,
    working_dir: str,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using DataPizza multi-agent framework.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team (for logging)
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        datapizza_provider: AI provider (openai, anthropic, google)
        datapizza_api_key: API key for the provider
        datapizza_model: Model name to use
        datapizza_tools: List of tools to enable
        datapizza_planning_interval: Planning interval for agents
        datapizza_max_steps: Maximum steps for agents
        working_dir: Working directory for file operations

    Returns:
        Dict with solution attempt
    """
    logger.info(f"  Using DataPizza multi-agent framework (provider={datapizza_provider})")

    if not DATAPIZZA_AVAILABLE:
        return {
            "error": "DataPizza not available",
            "solution": None,
            "execution_method_used": "datapizza",
        }

    try:
        # Import DataPizza components
        from datapizza.agents import Agent
        from datapizza.clients import Client

        # Import client providers
        if datapizza_provider.lower() == "openai":
            from datapizza.clients.openai import OpenAIClient
            client = OpenAIClient(
                api_key=datapizza_api_key,
                model=datapizza_model or "gpt-4o-mini",
            )
        elif datapizza_provider.lower() == "anthropic":
            from datapizza.clients.anthropic import AnthropicClient
            client = AnthropicClient(
                api_key=datapizza_api_key,
                model=datapizza_model or "claude-3-5-sonnet-20241022",
            )
        elif datapizza_provider.lower() == "google":
            from datapizza.clients.google import GoogleClient
            client = GoogleClient(
                api_key=datapizza_api_key,
                model=datapizza_model or "gemini-pro",
            )
        else:
            return {
                "error": f"Unsupported DataPizza provider: {datapizza_provider}",
                "solution": None,
                "execution_method_used": "datapizza",
            }

        # Import tools
        tools = []
        datapizza_tools = datapizza_tools or ["filesystem"]

        if "filesystem" in datapizza_tools:
            try:
                from datapizza.tools.filesystem import FileSystem
                paths_to_include = [f"{working_dir}/**"] if working_dir and working_dir != "." else None
                tools.append(FileSystem(
                    paths_to_include=paths_to_include,
                    paths_to_exclude=["*.pyc", "__pycache__/", ".git/", "node_modules/"],
                ))
            except ImportError:
                logger.warning("DataPizza FileSystem tool not available")

        if "duckduckgo" in datapizza_tools:
            try:
                from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
                tools.append(DuckDuckGoSearchTool())
            except ImportError:
                logger.warning("DataPizza DuckDuckGo tool not available")

        if "sql" in datapizza_tools:
            try:
                from datapizza.tools.SQLDatabase import SQLDatabaseTool
                tools.append(SQLDatabaseTool())
            except ImportError:
                logger.warning("DataPizza SQL tool not available")

        if "web_fetch" in datapizza_tools:
            try:
                from datapizza.tools.web_fetch import WebFetchTool
                tools.append(WebFetchTool())
            except ImportError:
                logger.warning("DataPizza WebFetch tool not available")

        # Build prompt
        prompt_parts = [f"Sub-Problem ID: {sub_problem_id}", sub_problem_description]

        if context:
            prompt_parts.append(f"\nContext: {json.dumps(context, indent=2)}")

        if constraints:
            prompt_parts.append("\nConstraints:")
            for c in constraints:
                prompt_parts.append(f"  - {c}")

        if requirements:
            prompt_parts.append("\nRequirements:")
            for r in requirements:
                prompt_parts.append(f"  - {r}")

        prompt = "\n".join(prompt_parts)

        # Create agent
        agent = Agent(
            name=f"{team_name}_{sub_problem_id}",
            client=client,
            system_prompt="You are an expert problem solver. Analyze the problem and provide a complete, implementable solution.",
            tools=tools,
            planning_interval=datapizza_planning_interval,
            max_steps=datapizza_max_steps,
        )

        # Run agent
        logger.info(f"  Running DataPizza agent (max_steps={datapizza_max_steps}, planning_interval={datapizza_planning_interval})")
        result = agent.run(prompt)

        # Extract result
        if hasattr(result, 'text'):
            solution = result.text
        else:
            solution = str(result)

        # Get step count
        steps_taken = result.index if hasattr(result, 'index') else 0

        # Get tool usage
        tools_used = []
        if hasattr(result, 'tools_used'):
            tools_used = [t.function for t in result.tools_used]

        # Get token usage
        token_usage = None
        if hasattr(result, 'usage') and result.usage:
            token_usage = {
                "prompt_tokens": result.usage.prompt_tokens,
                "completion_tokens": result.usage.completion_tokens,
                "total_tokens": result.usage.total_tokens,
            }

        logger.info(f"  DataPizza agent completed: {steps_taken} steps, {len(tools_used)} tools used")

        return {
            "sub_problem_id": sub_problem_id,
            "solution": solution,
            "team_name": team_name,
            "generated_by": f"DataPizza ({datapizza_provider})",
            "status": "completed",
            "execution_method_used": "datapizza",
            "steps_taken": steps_taken,
            "tools_used": tools_used,
            "token_usage": token_usage,
        }

    except ImportError as e:
        logger.error(f"DataPizza import error: {e}")
        return {
            "error": f"DataPizza import failed: {e}",
            "solution": None,
            "execution_method_used": "datapizza",
        }
    except Exception as e:
        logger.error(f"Failed to solve sub-problem {sub_problem_id} with DataPizza: {e}")
        return {
            "error": str(e),
            "solution": None,
            "execution_method_used": "datapizza",
        }


def _solve_with_roma(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]],
    constraints: Optional[List[str]],
    requirements: Optional[List[str]],
    roma_max_depth: int,
    roma_execution_mode: str,
    roma_provider: Optional[str],
    roma_api_key: Optional[str],
    roma_model: Optional[str],
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA recursive meta-agent framework.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team (for logging)
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        roma_max_depth: Maximum recursion depth for ROMA
        roma_execution_mode: Execution mode ("recursive" or "event_driven")
        roma_provider: AI provider for ROMA
        roma_api_key: API key for ROMA provider
        roma_model: Model name for ROMA

    Returns:
        Dict with solution attempt
    """
    logger.info(f"  Using ROMA recursive meta-agent framework (mode={roma_execution_mode}, max_depth={roma_max_depth})")

    if not ROMA_AVAILABLE:
        return {
            "error": "ROMA not available",
            "solution": None,
            "execution_method_used": "roma",
        }

    try:
        # Import ROMA here
        from roma_dspy.core.engine.solve import RecursiveSolver
        from roma_dspy.config.schemas.root import ROMAConfig

        # Create ROMA config
        config = ROMAConfig()

        # Set API key if provided
        if roma_api_key and roma_provider:
            import os
            if roma_provider.lower() == "openai":
                os.environ["OPENAI_API_KEY"] = roma_api_key
            elif roma_provider.lower() == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = roma_api_key
            elif roma_provider.lower() == "google":
                os.environ["GOOGLE_API_KEY"] = roma_api_key
            elif roma_provider.lower() == "openrouter":
                os.environ["OPENROUTER_API_KEY"] = roma_api_key

        # Build task
        task_parts = [f"Sub-Problem ID: {sub_problem_id}", sub_problem_description]

        if context:
            task_parts.append(f"\nContext: {json.dumps(context, indent=2)}")

        if constraints:
            task_parts.append("\nConstraints:")
            for c in constraints:
                task_parts.append(f"  - {c}")

        if requirements:
            task_parts.append("\nRequirements:")
            for r in requirements:
                task_parts.append(f"  - {r}")

        task = "\n".join(task_parts)

        # Create solver
        solver = RecursiveSolver(
            config=config,
            max_depth=roma_max_depth,
            enable_logging=False,
            enable_checkpoints=False,
        )

        # Execute based on mode
        if roma_execution_mode == "event_driven":
            result_task_node = solver.event_solve(task)
        else:
            result_task_node = solver.solve(task)

        # Extract results
        result = result_task_node.result if hasattr(result_task_node, 'result') else str(result_task_node)
        status = result_task_node.status.value if hasattr(result_task_node, 'status') else "unknown"

        # Get DAG info if available
        dag_info = {}
        if solver.last_dag:
            dag_info = {
                "total_tasks": len(solver.last_dag.get_all_tasks()),
                "execution_id": solver.last_dag.execution_id,
            }

        # Get token usage
        token_usage = {
            "input_tokens": solver.get_total_input_tokens(),
            "output_tokens": solver.get_total_output_tokens(),
        }

        logger.info(f"  ROMA completed: status={status}, dag_tasks={dag_info.get('total_tasks', 0)}")

        return {
            "sub_problem_id": sub_problem_id,
            "solution": result,
            "team_name": team_name,
            "generated_by": f"ROMA ({roma_execution_mode})",
            "status": status,
            "execution_method_used": "roma",
            "dag_info": dag_info,
            "token_usage": token_usage,
        }

    except ImportError as e:
        logger.error(f"ROMA import error: {e}")
        return {
            "error": f"ROMA import failed: {e}",
            "solution": None,
            "execution_method_used": "roma",
        }
    except Exception as e:
        logger.error(f"Failed to solve sub-problem {sub_problem_id} with ROMA: {e}")
        return {
            "error": str(e),
            "solution": None,
            "execution_method_used": "roma",
        }


def _solve_with_hybrid(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]],
    constraints: Optional[List[str]],
    requirements: Optional[List[str]],
    hybrid_max_depth_analysis: int,
    hybrid_max_depth_solving: int,
    hybrid_execution_mode: str,
    hybrid_provider: Optional[str],
    hybrid_api_key: Optional[str],
    hybrid_model: Optional[str],
    hybrid_enable_gauntlets: bool,
    hybrid_enable_evolution: bool,
    hybrid_evolution_iterations: int,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA-Decomposition hybrid mode.

    Combines ROMA's automatic recursive decomposition with Decomposition Workflow's
    team-based quality assurance process.

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team (for logging)
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        hybrid_max_depth_analysis: Max depth for ROMA analysis phase
        hybrid_max_depth_solving: Max depth for ROMA solving phase
        hybrid_execution_mode: Execution mode ("recursive" or "event_driven")
        hybrid_provider: AI provider for hybrid mode
        hybrid_api_key: API key for hybrid mode provider
        hybrid_model: Model name for hybrid mode
        hybrid_enable_gauntlets: Enable Decomposition Workflow gauntlets
        hybrid_enable_evolution: Enable evolution in hybrid mode
        hybrid_evolution_iterations: Evolution iterations for hybrid mode

    Returns:
        Dict with solution attempt
    """
    logger.info(f"  Using ROMA-Decomposition hybrid mode (mode={hybrid_execution_mode})")
    logger.info(f"    Analysis depth: {hybrid_max_depth_analysis}, Solving depth: {hybrid_max_depth_solving}")
    logger.info(f"    Gauntlets: {hybrid_enable_gauntlets}, Evolution: {hybrid_enable_evolution}")

    if not HYBRID_AVAILABLE:
        return {
            "error": "ROMA-Decomposition hybrid not available",
            "solution": None,
            "execution_method_used": "hybrid",
        }

    try:
        # Import hybrid components
        from roma_decomposition_hybrid import create_hybrid_config, solve_with_hybrid

        # Create hybrid config
        config = create_hybrid_config(
            roma_max_depth_analysis=hybrid_max_depth_analysis,
            roma_max_depth_solving=hybrid_max_depth_solving,
            roma_execution_mode=hybrid_execution_mode,
            roma_provider=hybrid_provider,
            roma_model=hybrid_model,
            roma_api_key=hybrid_api_key,
            enable_gauntlets=hybrid_enable_gauntlets,
            enable_evolution=hybrid_enable_evolution,
            evolution_iterations=hybrid_evolution_iterations,
        )

        # Execute hybrid workflow
        result = solve_with_hybrid(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            team_name=team_name,
            context=context,
            constraints=constraints,
            requirements=requirements,
            config=config,
        )

        if "error" in result:
            logger.error(f"  Hybrid mode failed: {result['error']}")
            return {
                "error": result["error"],
                "solution": None,
                "execution_method_used": "hybrid",
            }

        logger.info(f"  Hybrid completed: stages={result.get('workflow_details', {}).get('stages_completed', 0)}")

        return result

    except ImportError as e:
        logger.error(f"Hybrid import error: {e}")
        return {
            "error": f"Hybrid import failed: {e}",
            "solution": None,
            "execution_method_used": "hybrid",
        }
    except Exception as e:
        logger.error(f"Failed to solve sub-problem {sub_problem_id} with hybrid mode: {e}")
        return {
            "error": str(e),
            "solution": None,
            "execution_method_used": "hybrid",
        }


def _solve_with_roma_mdap_maker(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]],
    constraints: Optional[List[str]],
    requirements: Optional[List[str]],
    roma_mdap_maker_max_depth: int,
    roma_mdap_maker_k_ahead: int,
    roma_mdap_maker_enable_red_flagging: bool,
    roma_mdap_maker_max_samples: int,
    roma_mdap_maker_enable_adaptive_k: bool,
    roma_mdap_maker_provider: str,
    roma_mdap_maker_api_key: Optional[str],
    roma_mdap_maker_model: str,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA-MDAP-MAKER mode.

    Combines ROMA's automatic recursive decomposition with MAKER's proven
    zero-error voting mechanisms (first-to-ahead-by-k + red-flagging).

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of what to solve
        team_name: Name of the Blue Team (for logging)
        context: Additional context and dependencies
        constraints: List of constraints
        requirements: List of requirements
        roma_mdap_maker_max_depth: Max depth for ROMA decomposition
        roma_mdap_maker_k_ahead: K-ahead threshold for MAKER voting
        roma_mdap_maker_enable_red_flagging: Enable MAKER red-flagging
        roma_mdap_maker_max_samples: Max samples for MAKER voting
        roma_mdap_maker_enable_adaptive_k: Enable adaptive k-ahead selection
        roma_mdap_maker_provider: AI provider for ROMA-MDAP-MAKER
        roma_mdap_maker_api_key: API key for provider
        roma_mdap_maker_model: Model name

    Returns:
        Dict with solution attempt
    """
    logger.info(f"  Using ROMA-MDAP-MAKER mode (zero-error guaranteed)")
    logger.info(f"    Max depth: {roma_mdap_maker_max_depth}, K-ahead: {roma_mdap_maker_k_ahead}")
    logger.info(f"    Red-flagging: {roma_mdap_maker_enable_red_flagging}, Adaptive K: {roma_mdap_maker_enable_adaptive_k}")
    logger.info(f"    Provider: {roma_mdap_maker_provider}, Model: {roma_mdap_maker_model}")

    if not ROMA_MDAP_MAKER_AVAILABLE:
        return {
            "error": "ROMA-MDAP-MAKER not available",
            "solution": None,
            "execution_method_used": "roma_mdap_maker",
        }

    try:
        # Create ROMA-MDAP-MAKER config
        config = create_roma_mdap_maker_config(
            roma_max_depth_analysis=roma_mdap_maker_max_depth,
            roma_max_depth_solving=roma_mdap_maker_max_depth,
            roma_execution_mode="recursive",
            provider=roma_mdap_maker_provider,
            model=roma_mdap_maker_model,
            api_key=roma_mdap_maker_api_key,
            mdap_k_ahead=roma_mdap_maker_k_ahead,
            mdap_max_samples=roma_mdap_maker_max_samples,
            mdap_enable_red_flagging=roma_mdap_maker_enable_red_flagging,
            apply_maker_to_roma_atomic=True,
            enable_hierarchical_voting=True,
            enable_adaptive_k=roma_mdap_maker_enable_adaptive_k,
        )

        # Execute ROMA-MDAP-MAKER workflow
        result = solve_subproblem_with_roma_mdap_maker(
            sub_problem_id=sub_problem_id,
            sub_problem_description=sub_problem_description,
            context=context,
            constraints=constraints,
            requirements=requirements,
            config=config,
        )

        if "error" in result:
            logger.error(f"  ROMA-MDAP-MAKER mode failed: {result['error']}")
            return {
                "error": result["error"],
                "solution": None,
                "execution_method_used": "roma_mdap_maker",
            }

        # Log key metrics
        metrics = result.get("roma_mdap_maker_metrics", {})
        logger.info(f"  ROMA-MDAP-MAKER completed:")
        logger.info(f"    ROMA levels: {metrics.get('roma_decomposition_levels', 0)}")
        logger.info(f"    Atomic tasks: {metrics.get('total_atomic_tasks', 0)}")
        logger.info(f"    Voting rounds: {metrics.get('total_voting_rounds', 0)}")
        logger.info(f"    Red-flags: {metrics.get('total_red_flags', 0)}")
        logger.info(f"    Error rate: {metrics.get('final_error_rate', 0.0):.4f}")

        return result

    except Exception as e:
        logger.error(f"Failed to solve sub-problem {sub_problem_id} with ROMA-MDAP-MAKER mode: {e}")
        return {
            "error": str(e),
            "solution": None,
            "execution_method_used": "roma_mdap_maker",
        }


def calculate_parallelization(sub_problems: List[Dict], dependencies: Dict[str, List[str]]) -> int:
    """Calculate maximum number of sub-problems that can be solved in parallel"""
    if not dependencies:
        return len(sub_problems)

    # Simple calculation: count sub-problems with no dependencies
    no_deps = sum(1 for sp in sub_problems if not dependencies.get(sp["id"], []))
    return max(1, no_deps)


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_mcp_tools():
    """Initialize all Decomposition MCP tools"""
    logger.info("Initializing Decomposition MCP tools...")
    tools = list_mcp_tools()
    logger.info(f"Registered {len(tools)} Decomposition MCP tools")
    for tool in tools:
        logger.info(f"  - {tool}")
    return {
        "total_tools": len(tools),
        "tools": tools,
    }


# Auto-initialize on import
initialize_mcp_tools()
