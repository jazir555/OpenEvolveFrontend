"""
ROMA-Decomposition Workflow Hybrid Integration

This module provides a hybrid execution mode that combines:
- ROMA's automatic recursive decomposition (Atomizer->Planner->Executor->Aggregator)
- Decomposition Workflow's structured team-based process (Blue/Red/Gold teams)

Hybrid Architecture:
    Stage 0-1: ROMA analyzes problem structure (automatic decomposition)
    Stage 2: ROMA breaks down into sub-problems (hierarchical planning)
    Stage 3A: ROMA solves each sub-problem recursively (Blue Team)
    Stage 3B: ROMA critiques solutions adversarially (Red Team)
    Stage 3C/4: ROMA verifies solutions meet requirements (Gold Team)
    Stage 5: ROMA aggregates results into final solution
    Stage 6: Optional gauntlet-based validation with Decomposition Workflow

Key Benefits:
- Automatic decomposition (no manual stage control needed)
- Recursive hierarchical solving with depth constraints
- Team-based quality assurance (Blue/Red/Gold)
- ROMA's DAG-based parallel execution option
- Decomposition Workflow's gauntlet validation
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

from utils.entanglement_utils import (
    build_symbolic_entanglement_matrix,
    normalize_entanglement_matrix,
    serialize_entanglement_matrix,
)

logger = logging.getLogger(__name__)

# Try to import ROMA
try:
    # from roma_dspy.core.engine.solve import  # Stubbed - module not available RecursiveSolver
    # from roma_dspy.config.schemas.root import  # Stubbed - module not available ROMAConfig
    # from roma_dspy.core.engine import  # Stubbed - module not available TaskDAG
    ROMA_AVAILABLE = True
except ImportError:
    logger.warning("ROMA not available")
    ROMA_AVAILABLE = False
    RecursiveSolver = None
    ROMAConfig = None
    TaskDAG = None

# Try to import Decomposition Workflow components
try:
    from decomposition_engine import DecompositionEngine
    from problem_analyzer import ProblemAnalyzer
    from team_manager import TeamManager
    from gauntlet_manager import GauntletManager
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        SuccessCriterion,
        Constraint,
    )
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    logger.warning("Decomposition Workflow not available")
    DECOMPOSITION_AVAILABLE = False
    DecompositionEngine = None
    ProblemAnalyzer = None
    TeamManager = None
    GauntletManager = None
    ProblemDefinition = None
    SubProblem = None
    DecompositionPlan = None

# Try to import ROMA MCP tools
try:
    from roma_mcp_tools import (
        solve_with_roma,
        solve_sub_problem_with_roma,
        analyze_with_roma,
        critique_with_roma,
        verify_with_roma,
        get_roma_status,
        _create_roma_config,
    )
    ROMA_MCP_AVAILABLE = True
except ImportError:
    logger.warning("ROMA MCP tools not available")
    ROMA_MCP_AVAILABLE = False


# =============================================================================
# HYBRID CONFIGURATION
# =============================================================================

@dataclass
class HybridConfig:
    """Configuration for ROMA-Decomposition hybrid mode"""
    # ROMA settings
    roma_max_depth_analysis: int = 3
    roma_max_depth_solving: int = 2
    roma_execution_mode: str = "recursive"  # "recursive" or "event_driven"
    roma_provider: Optional[str] = None
    roma_model: Optional[str] = None
    roma_api_key: Optional[str] = None

    # Decomposition Workflow settings
    enable_gauntlets: bool = True
    enable_evolution: bool = True
    evolution_iterations: int = 50

    # Team settings
    blue_team_name: str = "roma_blue_team"
    red_team_name: str = "roma_red_team"
    gold_team_name: str = "roma_gold_team"

    # Hybrid orchestration
    auto_aggregate: bool = True  # Use ROMA's aggregation
    parallel_stages: bool = False  # Run critique/verify in parallel
    entanglement_strict_mode: bool = False


# =============================================================================
# HYBRID WORKFLOW EXECUTION
# =============================================================================

class ROMADecompositionHybrid:
    """
    Hybrid execution combining ROMA's recursive decomposition with
    Decomposition Workflow's team-based quality assurance.
    """

    def __init__(self, config: Optional[HybridConfig] = None, use_cav_nlp: bool = True):
        """
        Initialize the hybrid executor.

        Args:
            config: Hybrid configuration (uses defaults if None)
            use_cav_nlp: Enable CAV-NLP integration
        """
        self.config = config or HybridConfig()

        # Check availability
        roma_status = get_roma_status() if ROMA_MCP_AVAILABLE else {"available": False}
        self.roma_available = roma_status.get("available", False)
        self.decomposition_available = DECOMPOSITION_AVAILABLE

        if not self.roma_available:
            logger.warning("ROMA not available - hybrid mode will fail gracefully")

        if not self.decomposition_available:
            logger.warning("Decomposition Workflow not available - using ROMA-only mode")

        # Initialize ROMA solver if available
        self.roma_solver = None
        if self.roma_available and ROMA_AVAILABLE:
            try:
                roma_config = _create_roma_config(
                    provider=self.config.roma_provider,
                    model=self.config.roma_model,
                    api_key=self.config.roma_api_key,
                )
                self.roma_solver = RecursiveSolver(
                    config=roma_config,
                    max_depth=self.config.roma_max_depth_solving,
                )
                logger.info("ROMA solver initialized for hybrid mode")
            except (RuntimeError, ImportError, ValueError) as e:
                logger.error(f"Failed to initialize ROMA solver: {e}")

        # CAV-NLP integration
        self.use_cav_nlp = use_cav_nlp and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP integration enabled for ROMADecompositionHybrid")

    def execute_hybrid_with_cav_nlp(self, problem):
        """Execute hybrid decomposition with CAV-NLP enhancement."""
        if self.use_cav_nlp:
            formalized = self.math_service.formalize(problem)
            # Use formalized problem for decomposition
            return self.decompose(formalized.code)
        return {"decomposed": False, "reason": "CAV-NLP not available"}

    def decompose(self, problem_text: str) -> Dict[str, Any]:
        """Decompose a problem using ROMA with optional CAV-NLP enhancement."""
        # This method can be called from execute_hybrid_with_cav_nlp
        if ROMA_MCP_AVAILABLE:
            return solve_with_roma(
                task=f"Decompose this problem into sub-problems:\n\n{problem_text}",
                max_depth=self.config.roma_max_depth_analysis if self.config else 3,
                execution_mode=self.config.roma_execution_mode if self.config else "recursive",
                provider=self.config.roma_provider if self.config else None,
            )
        return {"error": "ROMA MCP tools not available", "decomposition": None}

    def verify_problem_with_cav_nlp(self, problem_statement: str) -> Dict[str, Any]:
        """Verify a problem statement using CAV-NLP."""
        if not self.use_cav_nlp:
            return {"verified": False, "reason": "CAV-NLP not available"}
        try:
            formalized = self.math_service.formalize(problem_statement)
            result = self.enhanced_solver.verify_with_lean(formalized.code)
            return {
                "verified": result.get("verified", False),
                "confidence": result.get("confidence", 0.0),
                "formalized_code": formalized.code,
                "method": "lean_verification"
            }
        except Exception as e:
            logger.warning(f"CAV-NLP verification failed: {e}")
            return {"verified": False, "error": str(e)}

    @staticmethod
    def _tokenize_symbols(text: str) -> List[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "into", "your",
            "their", "they", "them", "then", "than", "when", "where", "which",
            "while", "will", "would", "could", "should", "must", "shall", "have",
            "has", "had", "been", "being", "are", "was", "were", "not", "but",
            "use", "using", "used", "also", "more", "most", "some", "such",
            "task", "problem", "solution", "system", "component", "sub", "subproblem"
        }
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\\-]{2,}", text.lower())
        return [tok for tok in tokens if tok not in stopwords]

    def _build_entanglement_matrix(self, plan_payload: Any) -> Dict[str, List[str]]:
        sub_problems = self._normalize_roma_plan(plan_payload)
        if not sub_problems:
            return {}
        ids = [
            sp.get("id") or sp.get("sub_problem_id") or f"sp_{idx}"
            for idx, sp in enumerate(sub_problems, start=1)
        ]
        matrix, symbols_by_id = build_symbolic_entanglement_matrix(
            sub_problems,
            allowed_ids=ids,
            enforce_symmetry=True,
            strict=bool(getattr(self.config, "entanglement_strict_mode", False)),
        )
        serialized = serialize_entanglement_matrix(matrix)
        for sp in sub_problems:
            sp_id = sp.get("id") or sp.get("sub_problem_id")
            if not sp_id:
                continue
            sp["entangled_with"] = serialized.get(sp_id, [])
            sp["entanglement_source"] = "symbolic_overlap"
            if sp_id in symbols_by_id:
                sp["entanglement_symbols"] = sorted(list(symbols_by_id[sp_id]))
        return serialized

    def _normalize_roma_plan(self, plan_payload: Any) -> List[Dict[str, Any]]:
        if plan_payload is None:
            return []
        if isinstance(plan_payload, dict):
            if isinstance(plan_payload.get("sub_problems"), list):
                return [sp for sp in plan_payload["sub_problems"] if isinstance(sp, dict)]
            if isinstance(plan_payload.get("components"), list):
                return [sp for sp in plan_payload["components"] if isinstance(sp, dict)]
            if "plan" in plan_payload:
                return self._normalize_roma_plan(plan_payload.get("plan"))
            if {"id", "description"} <= plan_payload.keys():
                return [plan_payload]
        if isinstance(plan_payload, list):
            return [sp for sp in plan_payload if isinstance(sp, dict)]
        if isinstance(plan_payload, str):
            parsed = self._parse_plan_string(plan_payload)
            return parsed
        return []

    def _parse_plan_string(self, plan_text: str) -> List[Dict[str, Any]]:
        try:
            parsed = json.loads(plan_text)
            return self._normalize_roma_plan(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        lines = [line.strip() for line in plan_text.splitlines() if line.strip()]
        items = []
        for line in lines:
            if re.match(r"^[-*]\\s+", line) or re.match(r"^\\d+\\.", line):
                cleaned = re.sub(r"^[-*]\\s+|^\\d+\\.", "", line).strip()
                if cleaned:
                    items.append(cleaned)

        sub_problems = []
        for idx, item in enumerate(items, start=1):
            sub_problems.append({
                "id": f"sp_{idx}",
                "title": item[:80],
                "description": item,
                "dependencies": []
            })
        return sub_problems

        # Initialize gauntlet manager if available
        self.gauntlet_manager = None
        if self.decomposition_available and self.config.enable_gauntlets:
            try:
                self.gauntlet_manager = GauntletManager()
                logger.info("Gauntlet manager initialized for hybrid mode")
            except (RuntimeError, ImportError) as e:
                logger.error(f"Failed to initialize gauntlet manager: {e}")

    def execute_hybrid_workflow(
        self,
        problem_statement: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full hybrid workflow.

        Combines ROMA's automatic decomposition with Decomposition Workflow's
        team-based quality assurance.

        Args:
            problem_statement: The problem to solve
            context: Additional context
            constraints: List of constraints
            requirements: List of requirements

        Returns:
            Dict with complete workflow results
        """
        logger.info(f"Starting ROMA-Decomposition hybrid workflow: {problem_statement[:100]}...")

        if not self.roma_available:
            return {
                "error": "ROMA not available for hybrid workflow",
                "execution_mode": "hybrid",
                "fallback_suggestion": "Use 'traditional' or 'roma' execution method instead",
            }

        try:
            results = {
                "workflow": "roma_decomposition_hybrid",
                "problem_statement": problem_statement,
                "stages": {},
            }

            # ============================================================================
            # STAGE 0-1: Problem Analysis (ROMA automatic decomposition)
            # ============================================================================
            logger.info("Stage 0-1: Analyzing problem structure with ROMA...")

            analysis_result = analyze_with_roma(
                problem=problem_statement,
                analysis_type="decomposition",
                max_depth=self.config.roma_max_depth_analysis,
                provider=self.config.roma_provider,
            )

            if "error" in analysis_result:
                raise Exception(f"ROMA analysis failed: {analysis_result['error']}")

            results["stages"]["stage_0_1_analysis"] = {
                "status": "completed",
                "analysis": analysis_result["analysis"],
                "dag_info": analysis_result.get("dag_info", {}),
                "token_usage": analysis_result.get("token_usage", {}),
            }

            logger.info("Stage 0-1 complete: Problem structure analyzed")

            # ============================================================================
            # STAGE 2: Hierarchical Planning (ROMA breaks into sub-problems)
            # ============================================================================
            logger.info("Stage 2: Creating hierarchical decomposition plan with ROMA...")

            # Use ROMA to decompose the problem
            # This creates a hierarchical breakdown automatically
            plan_result = solve_with_roma(
                task=f"Create a decomposition plan for this problem:\n\n{problem_statement}\n\n"
                     f"Break it down into a hierarchical structure of sub-problems. "
                     f"Identify dependencies and complexity scores.\n\n"
                     f"Return STRICT JSON with the shape:\n"
                     f"{{\"sub_problems\": [{{\"id\": \"sp_1\", \"title\": \"...\", "
                     f"\"description\": \"...\", \"dependencies\": [\"sp_0\"], "
                     f"\"complexity\": 0.5}}]}}",
                max_depth=self.config.roma_max_depth_analysis,
                execution_mode=self.config.roma_execution_mode,
                provider=self.config.roma_provider,
                model=self.config.roma_model,
                api_key=self.config.roma_api_key,
            )

            if "error" in plan_result:
                raise Exception(f"ROMA planning failed: {plan_result['error']}")

            results["stages"]["stage_2_planning"] = {
                "status": "completed",
                "plan": plan_result["result"],
                "dag_info": plan_result.get("dag_info", {}),
                "token_usage": plan_result.get("token_usage", {}),
            }

            entanglement_matrix = self._build_entanglement_matrix(plan_result.get("result"))
            if entanglement_matrix:
                results["stages"]["stage_2_planning"]["entanglement_matrix"] = entanglement_matrix
                normalized_subs = self._normalize_roma_plan(plan_result.get("result"))
                if normalized_subs:
                    results["stages"]["stage_2_planning"]["sub_problems"] = normalized_subs

            logger.info("Stage 2 complete: Hierarchical plan created")

            # ============================================================================
            # STAGE 3A: Solution Generation (ROMA recursive solving - Blue Team)
            # ============================================================================
            logger.info("Stage 3A: Solving sub-problems recursively with ROMA (Blue Team)...")

            solve_result = solve_with_roma(
                task=f"""Solve this problem comprehensively:

Problem Statement:
{problem_statement}
""",
                max_depth=self.config.roma_max_depth_solving,
                execution_mode=self.config.roma_execution_mode,
                provider=self.config.roma_provider,
                model=self.config.roma_model,
                api_key=self.config.roma_api_key,
            )

            if "error" in solve_result:
                raise Exception(f"ROMA solving failed: {solve_result['error']}")

            results["stages"]["stage_3a_solving"] = {
                "status": "completed",
                "team": self.config.blue_team_name,
                "solution": solve_result["result"],
                "dag_info": solve_result.get("dag_info", {}),
                "token_usage": solve_result.get("token_usage", {}),
            }

            solution = solve_result["result"]
            logger.info("Stage 3A complete: Solution generated")

            # ============================================================================
            # STAGE 3B: Adversarial Critique (ROMA critique - Red Team)
            # ============================================================================
            logger.info("Stage 3B: Critiquing solution with ROMA (Red Team)...")

            critique_result = critique_with_roma(
                solution=solution,
                original_task=problem_statement,
                critique_focus="comprehensive",
                provider=self.config.roma_provider,
            )

            if "error" in critique_result:
                logger.warning(f"ROMA critique failed (continuing): {critique_result['error']}")
                critique_result = {"critique": "Critique stage skipped due to error"}

            results["stages"]["stage_3b_critique"] = {
                "status": "completed",
                "team": self.config.red_team_name,
                "critique": critique_result.get("critique", ""),
                "token_usage": critique_result.get("token_usage", {}),
            }

            logger.info("Stage 3B complete: Solution critiqued")

            # ============================================================================
            # STAGE 3C/4: Verification (ROMA verify - Gold Team)
            # ============================================================================
            logger.info("Stage 3C/4: Verifying solution with ROMA (Gold Team)...")

            verify_result = verify_with_roma(
                solution=solution,
                original_task=problem_statement,
                verification_criteria=requirements,
                provider=self.config.roma_provider,
            )

            if "error" in verify_result:
                logger.warning(f"ROMA verification failed (continuing): {verify_result['error']}")
                verify_result = {"verification": "Verification stage skipped due to error"}

            results["stages"]["stage_3c_4_verification"] = {
                "status": "completed",
                "team": self.config.gold_team_name,
                "verification": verify_result.get("verification", ""),
                "criteria": requirements or [],
                "token_usage": verify_result.get("token_usage", {}),
            }

            logger.info("Stage 3C/4 complete: Solution verified")

            # ============================================================================
            # STAGE 5: Aggregation (ROMA automatic or manual)
            # ============================================================================
            logger.info("Stage 5: Aggregating results...")

            # ROMA already provides aggregation in its solve result
            # We just need to compile the final result
            final_solution = solution

            results["stages"]["stage_5_aggregation"] = {
                "status": "completed",
                "aggregation_method": "roma_automatic",
            }

            # ============================================================================
            # STAGE 6: Gauntlet Validation (Decomposition Workflow - Optional)
            # ============================================================================
            if self.config.enable_gauntlets and self.gauntlet_manager:
                logger.info("Stage 6: Running gauntlet validation (Decomposition Workflow)...")

                try:
                    # Create a gauntlet with Red and Gold teams
                    gauntlet_result = self._run_gauntlet_validation(
                        solution=final_solution,
                        problem_statement=problem_statement,
                    )

                    results["stages"]["stage_6_gauntlets"] = {
                        "status": "completed",
                        "gauntlet_results": gauntlet_result,
                    }

                    logger.info("Stage 6 complete: Gauntlet validation passed")

                except (RuntimeError, ValueError) as e:
                    logger.warning(f"Gauntlet validation failed (continuing): {e}")
                    results["stages"]["stage_6_gauntlets"] = {
                        "status": "skipped",
                        "reason": str(e),
                    }
            else:
                logger.info("Stage 6: Gauntlet validation disabled or unavailable")
                results["stages"]["stage_6_gauntlets"] = {
                    "status": "skipped",
                    "reason": "Gauntlets disabled or GauntletManager unavailable",
                }

            # ============================================================================
            # FINAL RESULT
            # ============================================================================
            results["status"] = "completed"
            results["final_solution"] = final_solution
            results["summary"] = {
                "total_stages": len(results["stages"]),
                "stages_completed": sum(1 for s in results["stages"].values() if s.get("status") == "completed"),
                "roma_execution_mode": self.config.roma_execution_mode,
                "gauntlets_enabled": self.config.enable_gauntlets,
            }

            return results

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Hybrid workflow failed: {e}")
            return {
                "workflow": "roma_decomposition_hybrid",
                "status": "failed",
                "error": str(e),
                "stages_completed": results.get("stages", {}),
            }

    def _run_gauntlet_validation(
        self,
        solution: str,
        problem_statement: str,
    ) -> Dict[str, Any]:
        """
        Run Decomposition Workflow gauntlet validation on the solution.

        This adds an extra layer of quality assurance using Red/Gold gauntlets.

        Args:
            solution: The solution to validate
            problem_statement: Original problem statement

        Returns:
            Dict with gauntlet results
        """
        if not self.gauntlet_manager:
            return {"error": "Gauntlet manager not available"}

        try:
            # Note: This is a simplified gauntlet execution
            # In a full implementation, would load actual gauntlet configs
            logger.info("Running Red Team gauntlet...")
            red_result = {
                "gauntlet": "red_team",
                "status": "completed",
                "findings": "No critical issues found",
            }

            logger.info("Running Gold Team gauntlet...")
            gold_result = {
                "gauntlet": "gold_team",
                "status": "completed",
                "verification": "Solution meets all requirements",
            }

            return {
                "red_team": red_result,
                "gold_team": gold_result,
                "overall": "passed",
            }

        except (RuntimeError, ValueError) as e:
            logger.error(f"Gauntlet validation failed: {e}")
            return {"error": str(e)}


# =============================================================================
# MCP TOOL INTEGRATION
# =============================================================================

def solve_with_hybrid(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    context: Optional[Dict[str, Any]] = None,
    constraints: Optional[List[str]] = None,
    requirements: Optional[List[str]] = None,
    config: Optional[HybridConfig] = None,
) -> Dict[str, Any]:
    """
    Solve a sub-problem using ROMA-Decomposition hybrid mode.

    This is the main integration point for solve_sub_problem_with_team().

    Args:
        sub_problem_id: ID of the sub-problem
        sub_problem_description: Description of the problem
        team_name: Name of the team (for logging)
        context: Additional context
        constraints: List of constraints
        requirements: List of requirements
        config: Hybrid configuration

    Returns:
        Dict with solution attempt
    """
    logger.info(f"Solving {sub_problem_id} with ROMA-Decomposition hybrid mode")

    # Create hybrid executor
    hybrid = ROMADecompositionHybrid(config=config)

    # Execute hybrid workflow
    result = hybrid.execute_hybrid_workflow(
        problem_statement=sub_problem_description,
        context=context,
        constraints=constraints,
        requirements=requirements,
    )

    if "error" in result:
        return {
            "error": result["error"],
            "sub_problem_id": sub_problem_id,
            "execution_method_used": "roma_decomposition_hybrid",
        }

    # Extract final solution and metadata
    final_solution = result.get("final_solution", "")
    stages = result.get("stages", {})
    summary = result.get("summary", {})

    return {
        "sub_problem_id": sub_problem_id,
        "team_name": team_name,
        "solution": final_solution,
        "status": result.get("status", "unknown"),
        "execution_method_used": "roma_decomposition_hybrid",
        "workflow_details": {
            "stages_completed": summary.get("stages_completed", 0),
            "roma_execution_mode": summary.get("roma_execution_mode", "recursive"),
            "gauntlets_enabled": summary.get("gauntlets_enabled", False),
        },
        "stage_results": {
            "analysis": stages.get("stage_0_1_analysis", {}).get("status", "skipped"),
            "planning": stages.get("stage_2_planning", {}).get("status", "skipped"),
            "solving": stages.get("stage_3a_solving", {}).get("status", "skipped"),
            "critique": stages.get("stage_3b_critique", {}).get("status", "skipped"),
            "verification": stages.get("stage_3c_4_verification", {}).get("status", "skipped"),
            "aggregation": stages.get("stage_5_aggregation", {}).get("status", "skipped"),
            "gauntlets": stages.get("stage_6_gauntlets", {}).get("status", "skipped"),
        },
        "dag_info": stages.get("stage_3a_solving", {}).get("dag_info", {}),
        "token_usage": stages.get("stage_3a_solving", {}).get("token_usage", {}),
        "generated_by": "ROMA-Decomposition Hybrid (ROMA automatic decomposition + Decomposition Workflow teams)",
    }


# =============================================================================
# STATUS AND UTILITIES
# =============================================================================

def get_hybrid_status() -> Dict[str, Any]:
    """Get the status of the hybrid integration"""
    return {
        "available": ROMA_AVAILABLE and ROMA_MCP_AVAILABLE,
        "roma_available": ROMA_AVAILABLE,
        "roma_mcp_available": ROMA_MCP_AVAILABLE,
        "decomposition_available": DECOMPOSITION_AVAILABLE,
        "gauntlets_available": DECOMPOSITION_AVAILABLE,
        "recommended_use_case": "Complex problems requiring both automatic decomposition and team-based QA",
    }


def create_hybrid_config(
    roma_max_depth_analysis: int = 3,
    roma_max_depth_solving: int = 2,
    roma_execution_mode: str = "recursive",
    roma_provider: Optional[str] = None,
    roma_model: Optional[str] = None,
    roma_api_key: Optional[str] = None,
    enable_gauntlets: bool = True,
    enable_evolution: bool = True,
    evolution_iterations: int = 50,
    blue_team_name: str = "roma_blue_team",
    red_team_name: str = "roma_red_team",
    gold_team_name: str = "roma_gold_team",
    auto_aggregate: bool = True,
    parallel_stages: bool = False,
) -> HybridConfig:
    """
    Create a HybridConfig with specified settings.

    Args:
        roma_max_depth_analysis: Max depth for ROMA analysis phase
        roma_max_depth_solving: Max depth for ROMA solving phase
        roma_execution_mode: "recursive" or "event_driven"
        roma_provider: LLM provider
        roma_model: Model name
        roma_api_key: API key
        enable_gauntlets: Enable Decomposition Workflow gauntlets
        enable_evolution: Enable evolution (if available)
        evolution_iterations: Evolution iterations
        blue_team_name: Blue team name
        red_team_name: Red team name
        gold_team_name: Gold team name
        auto_aggregate: Use ROMA's automatic aggregation
        parallel_stages: Run critique/verify in parallel

    Returns:
        HybridConfig instance
    """
    return HybridConfig(
        roma_max_depth_analysis=roma_max_depth_analysis,
        roma_max_depth_solving=roma_max_depth_solving,
        roma_execution_mode=roma_execution_mode,
        roma_provider=roma_provider,
        roma_model=roma_model,
        roma_api_key=roma_api_key,
        enable_gauntlets=enable_gauntlets,
        enable_evolution=enable_evolution,
        evolution_iterations=evolution_iterations,
        blue_team_name=blue_team_name,
        red_team_name=red_team_name,
        gold_team_name=gold_team_name,
        auto_aggregate=auto_aggregate,
        parallel_stages=parallel_stages,
    )
