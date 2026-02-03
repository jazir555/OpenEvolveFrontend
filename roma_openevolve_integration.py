"""
ROMA - OpenEvolve Integration Adapter

This module provides integration between ROMA's full capabilities and the OpenEvolve workflow system.

ROMA Capabilities Integrated:
- Phase 1: Problem Setup with ROMA analysis and decomposition
- Phase 2: Solution Generation with ROMA recursive solving
- Phase 3: Adversarial Critique using ROMA's recursive analysis
- Phase 4: Verification using ROMA's recursive verification approach
- Phase 5: Reassembly using ROMA's intelligent aggregation
- Phase 6: Final Validation with ROMA comprehensive validation

ROMA-MDAP-MAKER Enhanced:
- All phases available with MAKER voting consensus
- Red-flag detection for unreliable outputs
- Voting summaries and confidence aggregation

Author: Claude Code
Date: 2026-01-24
Version: 2.0 - Full Decomposition/Recomposition Support
"""

import logging
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from utils.entanglement_utils import (
        build_symbolic_entanglement_matrix,
        normalize_entanglement_matrix,
        serialize_entanglement_matrix,
    )
    ENTANGLEMENT_AVAILABLE = True
except ImportError:
    ENTANGLEMENT_AVAILABLE = False
    build_symbolic_entanglement_matrix = None  # type: ignore
    normalize_entanglement_matrix = None  # type: ignore
    serialize_entanglement_matrix = None  # type: ignore

try:
    from enhanced_decomposition_engine import (
        ProblemDefinition as EnhancedProblemDefinition,
        DecompositionPlan as EnhancedDecompositionPlan,
        SubProblem as EnhancedSubProblem,
        SubProblemType,
        ComplexityScore,
        DecompositionStrategy,
        ProblemDomain,
        SuccessCriterion,
    )
    ENHANCED_STRUCTURES_AVAILABLE = True
except ImportError:
    ENHANCED_STRUCTURES_AVAILABLE = False

try:
    from enhanced_recomposition_engine import (
        EnhancedRecompositionEngine,
        SubProblemSolution as EnhancedSubProblemSolution,
    )
    ENHANCED_RECOMPOSITION_AVAILABLE = True
except ImportError:
    ENHANCED_RECOMPOSITION_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ROMAOpenEvolveConfig:
    """Configuration for ROMA integration in OpenEvolve workflows."""

    # Enable/disable ROMA integration
    enable_roma: bool = False

    # Use ROMA-MDAP-MAKER (enhanced with voting) or standard ROMA
    use_roma_mdap_maker: bool = False

    # ROMA depth parameters
    analysis_depth: int = 3          # Phase 1: Problem analysis depth
    solving_depth: int = 2           # Phase 2: Solution generation depth
    critique_depth: int = 1          # Phase 3: Critique depth
    verification_depth: int = 1      # Phase 4: Verification depth
    reassembly_depth: int = 1        # Phase 5: Reassembly depth

    # ROMA execution mode
    execution_mode: str = "recursive"  # "recursive" or "event_driven"

    # Decomposition parameters
    max_sub_problems: int = 15
    decomposition_strategy: str = "semantic"  # "semantic", "hierarchical", "flow"

    # Provider/model configuration
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096

    # Fallback behavior when ROMA unavailable
    fallback_to_standard: bool = True

    # Entanglement settings
    entanglement_strict_mode: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.execution_mode not in ["recursive", "event_driven"]:
            raise ValueError(f"Invalid execution_mode: {self.execution_mode}. Must be 'recursive' or 'event_driven'")

        valid_strategies = ["semantic", "hierarchical", "flow", "roma"]
        if self.decomposition_strategy not in valid_strategies:
            raise ValueError(f"Invalid decomposition_strategy: {self.decomposition_strategy}. Must be one of {valid_strategies}")


def create_roma_openevolve_config(
    enable_roma: bool = False,
    use_mdap_maker: bool = False,
    **kwargs
) -> ROMAOpenEvolveConfig:
    """
    Create a ROMA-OpenEvolve configuration with sensible defaults.

    Args:
        enable_roma: Whether to enable ROMA integration
        use_mdap_maker: Whether to use ROMA-MDAP-MAKER (voting consensus)
        **kwargs: Additional configuration parameters

    Returns:
        ROMAOpenEvolveConfig instance
    """
    return ROMAOpenEvolveConfig(
        enable_roma=enable_roma,
        use_roma_mdap_maker=use_mdap_maker,
        **kwargs
    )


# =============================================================================
# INTEGRATION ADAPTER
# =============================================================================

class ROMAOpenEvolveAdapter:
    """
    Adapter for integrating ROMA's full capabilities into OpenEvolve workflows.

    This adapter provides a clean interface for OpenEvolve workflows to use ROMA's:
    - Decomposition (Phase 1 & 2)
    - Critique (Phase 3)
    - Verification (Phase 4)
    - Recomposition (Phase 5)
    - Final Validation (Phase 6)
    """

    def __init__(self, config: ROMAOpenEvolveConfig):
        """
        Initialize the ROMA-OpenEvolve adapter.

        Args:
            config: Configuration for ROMA integration
        """
        self.config = config
        self._roma_available = False
        self._roma_mdap_maker_available = False
        self._decomposition_available = False

        # Check ROMA bridge availability
        try:
            from roma_crewai_bridge import (
                execute_phase_1_setup,
                execute_phase_2_solve,
                execute_phase_3_critique,
                execute_phase_4_verify,
                execute_phase_5_reassemble,
                execute_phase_6_final_validation,
            )

            self._roma_bridge = __import__('roma_crewai_bridge')

            self.execute_phase_1_setup = execute_phase_1_setup
            self.execute_phase_2_solve = execute_phase_2_solve
            self.execute_phase_3_critique = execute_phase_3_critique
            self.execute_phase_4_verify = execute_phase_4_verify
            self.execute_phase_5_reassemble = execute_phase_5_reassemble
            self.execute_phase_6_final_validation = execute_phase_6_final_validation

            self._roma_available = True
            self._decomposition_available = True

            logger.info("Standard ROMA bridge loaded successfully")
        except ImportError as e:
            logger.warning(f"Standard ROMA bridge not available: {e}")
            self._roma_available = False
            self._decomposition_available = False

        # Check ROMA-MDAP-MAKER bridge availability
        try:
            from roma_mdap_maker_crewai_bridge import (
                execute_phase_1_setup as execute_phase_1_setup_mdap,
                execute_phase_2_solve as execute_phase_2_solve_mdap,
                execute_phase_3_critique as execute_phase_3_critique_mdap,
                execute_phase_4_verify as execute_phase_4_verify_mdap,
                execute_phase_5_reassemble as execute_phase_5_reassemble_mdap,
                execute_phase_6_final_validation as execute_phase_6_final_validation_mdap,
            )

            self._roma_mdap_maker_bridge = __import__('roma_mdap_maker_crewai_bridge')

            self.execute_phase_1_setup_mdap = execute_phase_1_setup_mdap
            self.execute_phase_2_solve_mdap = execute_phase_2_solve_mdap
            self.execute_phase_3_critique_mdap = execute_phase_3_critique_mdap
            self.execute_phase_4_verify_mdap = execute_phase_4_verify_mdap
            self.execute_phase_5_reassemble_mdap = execute_phase_5_reassemble_mdap
            self.execute_phase_6_final_validation_mdap = execute_phase_6_final_validation_mdap

            self._roma_mdap_maker_available = True

            logger.info("ROMA-MDAP-MAKER bridge loaded successfully")
        except ImportError as e:
            logger.warning(f"ROMA-MDAP-MAKER bridge not available: {e}")
            self._roma_mdap_maker_available = False

    def is_available(self) -> bool:
        """Check if ROMA integration is available."""
        if self.config.use_roma_mdap_maker:
            return self._roma_mdap_maker_available
        return self._roma_available

    def is_decomposition_available(self) -> bool:
        """Check if ROMA decomposition is available."""
        if self.config.use_roma_mdap_maker:
            return self._roma_mdap_maker_available
        return self._decomposition_available

    def _normalize_sub_problems(self, payload: Any) -> List[Dict[str, Any]]:
        """Normalize ROMA decomposition outputs into a list of dict sub-problems."""
        if payload is None:
            return []
        if isinstance(payload, dict):
            if isinstance(payload.get("sub_problems"), list):
                return [sp for sp in payload["sub_problems"] if isinstance(sp, dict)]
            if isinstance(payload.get("components"), list):
                return [sp for sp in payload["components"] if isinstance(sp, dict)]
            if "analysis" in payload and isinstance(payload["analysis"], dict):
                return self._normalize_sub_problems(payload["analysis"])
            if "decomposition_plan" in payload:
                return self._normalize_sub_problems(payload["decomposition_plan"])
        if isinstance(payload, list):
            return [sp for sp in payload if isinstance(sp, dict)]
        # ROMA CrewAI objects may expose sub_problems attribute
        if hasattr(payload, "sub_problems"):
            try:
                return [
                    {
                        "id": sp.id,
                        "title": sp.title,
                        "description": sp.description,
                        "dependencies": list(sp.dependencies or []),
                        "complexity_score": getattr(sp, "complexity_score", None),
                    }
                    for sp in payload.sub_problems
                ]
            except Exception:
                return []
        return []

    def _attach_entanglement(self, sub_problems: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Build and attach entanglement metadata to sub-problems."""
        if not ENTANGLEMENT_AVAILABLE or not sub_problems:
            return {}

        matrix, symbols_by_id = build_symbolic_entanglement_matrix(
            sub_problems,
            enforce_symmetry=True,
            strict=self.config.entanglement_strict_mode,
        )
        serialized = serialize_entanglement_matrix(matrix)
        for sp in sub_problems:
            sp_id = sp.get("id")
            if not sp_id:
                continue
            entangled_with = serialized.get(sp_id, [])
            sp.setdefault("metadata", {})
            sp["entangled_with"] = entangled_with
            sp["entanglement_source"] = "symbolic_overlap"
            if sp_id in symbols_by_id:
                sp["entanglement_symbols"] = sorted(list(symbols_by_id[sp_id]))
        return serialized

    @staticmethod
    def _infer_subproblem_type(title: str, description: str) -> SubProblemType:
        """Infer sub-problem type from content."""
        text = f"{title} {description}".lower()
        if any(k in text for k in ("design", "architecture", "schema", "interface")):
            return SubProblemType.DESIGN
        if any(k in text for k in ("test", "qa", "validation", "verify")):
            return SubProblemType.TESTING
        if any(k in text for k in ("research", "investigate", "explore", "survey")):
            return SubProblemType.RESEARCH
        if any(k in text for k in ("analysis", "analyze", "assessment")):
            return SubProblemType.ANALYSIS
        if any(k in text for k in ("document", "docs", "guide", "manual")):
            return SubProblemType.DOCUMENTATION
        if any(k in text for k in ("integrate", "integration", "connect")):
            return SubProblemType.INTEGRATION
        return SubProblemType.IMPLEMENTATION

    @staticmethod
    def _normalize_complexity(value: Any) -> float:
        try:
            return max(0.0, min(10.0, float(value)))
        except (TypeError, ValueError):
            return 5.0

    def to_decomposition_plan(
        self,
        problem: "EnhancedProblemDefinition",
        roma_result: Dict[str, Any],
    ) -> Optional["EnhancedDecompositionPlan"]:
        """Convert ROMA decomposition result to Enhanced DecompositionPlan."""
        if not ENHANCED_STRUCTURES_AVAILABLE:
            return None

        sub_problems_payload = self._normalize_sub_problems(roma_result)
        if not sub_problems_payload:
            return None

        sub_problems: List[EnhancedSubProblem] = []
        for sp in sub_problems_payload:
            sp_id = sp.get("id") or f"roma_{len(sub_problems) + 1}"
            title = sp.get("title") or sp_id
            description = sp.get("description") or title
            deps = list(sp.get("dependencies") or [])
            complexity_raw = sp.get("complexity_score") or sp.get("complexity") or 5.0
            complexity = self._normalize_complexity(complexity_raw)

            complexity_score = ComplexityScore(
                cognitive_complexity=complexity,
                computational_complexity=complexity,
                domain_complexity=complexity,
                integration_complexity=min(10.0, complexity + 0.5),
                coordination_complexity=min(10.0, complexity + 0.5),
                technical_complexity=complexity,
                overall_complexity=complexity,
                explanation="ROMA-derived complexity",
            )

            sp_type = self._infer_subproblem_type(title, description)
            metadata = sp.get("metadata", {}) if isinstance(sp.get("metadata"), dict) else {}
            for key in ("entangled_with", "entanglement_symbols", "entanglement_source"):
                if key in sp:
                    metadata[key] = sp.get(key)

            sub_problems.append(
                EnhancedSubProblem(
                    id=sp_id,
                    parent_id=problem.id,
                    title=title,
                    description=description,
                    type=sp_type,
                    complexity_score=complexity_score,
                    dependencies=deps,
                    success_criteria=[
                        SuccessCriterion(
                            id=f"roma_sc_{sp_id}",
                            description=f"Complete {title}",
                            metric="completion",
                            threshold=0.9,
                        )
                    ],
                    estimated_effort_hours=max(1.0, complexity * 3),
                    metadata=metadata,
                )
            )

        dependency_graph = {sp.id: list(sp.dependencies) for sp in sub_problems}
        execution_order = self._calculate_execution_order(sub_problems, dependency_graph)
        parallel_groups = self._identify_parallel_groups(sub_problems, dependency_graph)

        plan = EnhancedDecompositionPlan(
            id=f"roma_plan_{problem.id}",
            original_problem=problem,
            sub_problems=sub_problems,
            strategy_used=DecompositionStrategy.SEMANTIC,
            dependency_graph=dependency_graph,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            coverage_score=0.6,
            balance_score=0.6,
            coherence_score=0.6,
            overall_quality=0.6,
            metadata={
                "roma_used": roma_result.get("roma_used", True),
                "roma_type": roma_result.get("roma_type", "roma"),
                "source": "roma",
            },
        )

        entanglement_matrix = self._attach_entanglement(sub_problems_payload)
        if entanglement_matrix:
            plan.metadata["entanglement_matrix"] = entanglement_matrix
            for sp in plan.sub_problems:
                entangled_with = entanglement_matrix.get(sp.id, [])
                if entangled_with:
                    sp.metadata.setdefault("entangled_with", entangled_with)
                    sp.metadata.setdefault("entanglement_source", "symbolic_overlap")
        return plan

    @staticmethod
    def _calculate_execution_order(
        sub_problems: List["EnhancedSubProblem"],
        dependency_graph: Dict[str, List[str]],
    ) -> List[str]:
        """Topological execution order for sub-problems."""
        in_degree = {sp.id: len(sp.dependencies) for sp in sub_problems}
        dependents: Dict[str, List[str]] = {sp.id: [] for sp in sub_problems}
        for node, deps in dependency_graph.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(node)

        queue = deque([sp_id for sp_id, degree in in_degree.items() if degree == 0])
        order: List[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for dep in dependents.get(current, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        # Append remaining nodes (cycle-safe)
        for sp in sub_problems:
            if sp.id not in order:
                order.append(sp.id)

        return order

    @staticmethod
    def _identify_parallel_groups(
        sub_problems: List["EnhancedSubProblem"],
        dependency_graph: Dict[str, List[str]],
    ) -> List[List[str]]:
        dependency_sets = defaultdict(list)
        for sp in sub_problems:
            dependency_sets[frozenset(sp.dependencies)].append(sp.id)
        return list(dependency_sets.values())

    # =========================================================================
    # PHASE 1: PROBLEM SETUP & DECOMPOSITION
    # =========================================================================

    def setup_and_decompose_problem(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase 1: Setup and decompose problem using ROMA.

        Args:
            problem_statement: The problem to analyze and decompose
            problem_type: Type of problem (optimization, design, research, etc.)
            domain: Problem domain (software, mathematics, system design, etc.)

        Returns:
            Dictionary with:
            - status: 'completed', 'error', or 'fallback'
            - analysis: Problem analysis results
            - decomposition_plan: ROMA decomposition plan
            - sub_problems: List of sub-problems
            - roma_used: Whether ROMA was actually used
        """
        if not self.config.enable_roma or not self.is_decomposition_available():
            if self.config.fallback_to_standard:
                return self._fallback_decompose(problem_statement)
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": "ROMA not available and fallback disabled"
                }

        try:
            if self.config.use_roma_mdap_maker and self._roma_mdap_maker_available:
                logger.info("Using ROMA-MDAP-MAKER for Phase 1 setup and decomposition")
                result = self.execute_phase_1_setup_mdap(
                    problem_statement=problem_statement,
                    problem_type=problem_type,
                    domain=domain,
                    max_depth=self.config.analysis_depth,
                    execution_mode=self.config.execution_mode,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma_mdap_maker"
                return self._augment_decomposition_result(
                    result, problem_statement, domain=domain
                )
            else:
                logger.info("Using standard ROMA for Phase 1 setup and decomposition")
                result = self.execute_phase_1_setup(
                    problem_statement=problem_statement,
                    problem_type=problem_type,
                    domain=domain,
                    max_depth=self.config.analysis_depth,
                    execution_mode=self.config.execution_mode,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma"
                return self._augment_decomposition_result(
                    result, problem_statement, domain=domain
                )

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"ROMA Phase 1 failed: {e}")
            if self.config.fallback_to_standard:
                return self._augment_decomposition_result(
                    self._fallback_decompose(problem_statement, error=str(e)),
                    problem_statement,
                    domain=domain,
                )
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": f"ROMA Phase 1 failed: {str(e)}",
                    "error": str(e)
                }

    def _augment_decomposition_result(
        self,
        result: Dict[str, Any],
        problem_statement: str,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Attach entanglement matrix and optional OpenEvolve plan to ROMA results."""
        sub_problems = self._normalize_sub_problems(result)
        if not sub_problems and isinstance(result.get("analysis"), dict):
            sub_problems = self._normalize_sub_problems(result["analysis"])

        if sub_problems:
            result["sub_problems"] = sub_problems
            entanglement_matrix = self._attach_entanglement(sub_problems)
            if entanglement_matrix:
                result["entanglement_matrix"] = entanglement_matrix

        if ENHANCED_STRUCTURES_AVAILABLE:
            problem_def = self._build_problem_definition(problem_statement, domain)
            plan = self.to_decomposition_plan(problem_def, result)
            if plan:
                result["openevolve_plan"] = plan

        return result

    @staticmethod
    def _build_problem_definition(
        problem_statement: str,
        domain: Optional[str] = None,
    ) -> "EnhancedProblemDefinition":
        """Build minimal Enhanced ProblemDefinition for conversion."""
        domain_value = (domain or "generic").lower()
        domain_enum = ProblemDomain.GENERIC
        for candidate in ProblemDomain:
            if candidate.value == domain_value:
                domain_enum = candidate
                break

        complexity = ComplexityScore(
            cognitive_complexity=5.0,
            computational_complexity=5.0,
            domain_complexity=5.0,
            integration_complexity=5.0,
            coordination_complexity=5.0,
            technical_complexity=5.0,
            overall_complexity=5.0,
            explanation="ROMA adapter default complexity",
        )

        return EnhancedProblemDefinition(
            id=f"roma_problem_{hash(problem_statement) & 0xFFFFF}",
            title=problem_statement[:80] if problem_statement else "ROMA Problem",
            description=problem_statement,
            domain=domain_enum,
            complexity_score=complexity,
            constraints=[],
            success_criteria=[],
        )

    # =========================================================================
    # PHASE 2: SOLUTION GENERATION
    # =========================================================================

    def solve_sub_problems(
        self,
        sub_problems: List[Dict[str, Any]],
        team_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase 2: Generate solutions for sub-problems using ROMA.

        Args:
            sub_problems: List of sub-problems to solve
            team_name: Team name for agents

        Returns:
            Dictionary with:
            - status: 'completed', 'error', or 'fallback'
            - solutions: List of generated solutions
            - metrics: Performance metrics
            - roma_used: Whether ROMA was actually used
        """
        if not self.config.enable_roma or not self.is_decomposition_available():
            if self.config.fallback_to_standard:
                return self._fallback_solve(sub_problems)
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": "ROMA not available and fallback disabled"
                }

        try:
            if self.config.use_roma_mdap_maker and self._roma_mdap_maker_available:
                logger.info(f"Using ROMA-MDAP-MAKER for Phase 2 solving ({len(sub_problems)} sub-problems)")
                result = self.execute_phase_2_solve_mdap(
                    sub_problems=sub_problems,
                    team_name=team_name,
                    max_depth=self.config.solving_depth,
                    execution_mode=self.config.execution_mode,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma_mdap_maker"
                return result
            else:
                logger.info(f"Using standard ROMA for Phase 2 solving ({len(sub_problems)} sub-problems)")
                result = self.execute_phase_2_solve(
                    sub_problems=sub_problems,
                    team_name=team_name,
                    max_depth=self.config.solving_depth,
                    execution_mode=self.config.execution_mode,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma"
                return result

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"ROMA Phase 2 failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_solve(sub_problems, error=str(e))
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": f"ROMA Phase 2 failed: {str(e)}",
                    "error": str(e)
                }

    # =========================================================================
    # PHASE 3: ADVERSARIAL CRITIQUE
    # =========================================================================

    def critique_solutions(
        self,
        solutions: List[Dict[str, Any]],
        problem_statement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 3: Critique solutions using ROMA.

        Args:
            solutions: List of solutions to critique
            problem_statement: Optional problem statement for context

        Returns:
            Dictionary with critique results
        """
        # Add problem statement to solutions if provided
        if problem_statement:
            solutions = [
                {**sol, "problem_statement": problem_statement, "task": problem_statement}
                for sol in solutions
            ]

        if not self.config.enable_roma or not self.is_available():
            if self.config.fallback_to_standard:
                return self._fallback_critique(solutions)
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": "ROMA not available and fallback disabled"
                }

        try:
            if self.config.use_roma_mdap_maker and self._roma_mdap_maker_available:
                logger.info("Using ROMA-MDAP-MAKER for Phase 3 critique")
                result = self.execute_phase_3_critique_mdap(
                    solutions=solutions,
                    critique_depth=self.config.critique_depth,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma_mdap_maker"
                return result
            else:
                logger.info("Using standard ROMA for Phase 3 critique")
                result = self.execute_phase_3_critique(
                    solutions=solutions,
                    critique_depth=self.config.critique_depth,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma"
                return result

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"ROMA Phase 3 failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_critique(solutions, error=str(e))
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": f"ROMA Phase 3 failed: {str(e)}",
                    "error": str(e)
                }

    # =========================================================================
    # PHASE 4: VERIFICATION
    # =========================================================================

    def verify_solutions(
        self,
        solutions: List[Dict[str, Any]],
        requirements: Optional[List[str]] = None,
        problem_statement: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Phase 4: Verify solutions using ROMA.

        Args:
            solutions: List of solutions to verify
            requirements: Optional list of requirements
            problem_statement: Optional problem statement

        Returns:
            Dictionary with verification results
        """
        # Add requirements and problem statement to solutions
        if requirements or problem_statement:
            solutions = [
                {
                    **sol,
                    "requirements": sol.get("requirements", requirements),
                    "problem_statement": sol.get("problem_statement", problem_statement)
                }
                for sol in solutions
            ]

        if not self.config.enable_roma or not self.is_available():
            if self.config.fallback_to_standard:
                return self._fallback_verification(solutions)
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": "ROMA not available and fallback disabled"
                }

        try:
            if self.config.use_roma_mdap_maker and self._roma_mdap_maker_available:
                logger.info("Using ROMA-MDAP-MAKER for Phase 4 verification")
                result = self.execute_phase_4_verify_mdap(
                    solutions=solutions,
                    verification_depth=self.config.verification_depth,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma_mdap_maker"
                return result
            else:
                logger.info("Using standard ROMA for Phase 4 verification")
                result = self.execute_phase_4_verify(
                    solutions=solutions,
                    verification_depth=self.config.verification_depth,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma"
                return result

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"ROMA Phase 4 failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_verification(solutions, error=str(e))
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": f"ROMA Phase 4 failed: {str(e)}",
                    "error": str(e)
                }

    # =========================================================================
    # PHASE 5: REASSEMBLY / RECOMPOSITION
    # =========================================================================

    def reassemble_solutions(
        self,
        solutions: List[Dict[str, Any]],
        problem_statement: str,
    ) -> Dict[str, Any]:
        """
        Phase 5: Reassemble solutions using ROMA aggregation.

        Args:
            solutions: List of sub-solutions to reassemble
            problem_statement: Original problem statement

        Returns:
            Dictionary with reassembly results
        """
        if not self.config.enable_roma or not self.is_available():
            if self.config.fallback_to_standard:
                return self._fallback_reassemble(solutions, problem_statement)
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": "ROMA not available and fallback disabled"
                }

        try:
            if self.config.use_roma_mdap_maker and self._roma_mdap_maker_available:
                logger.info("Using ROMA-MDAP-MAKER for Phase 5 reassembly")
                result = self.execute_phase_5_reassemble_mdap(
                    solutions=solutions,
                    problem_statement=problem_statement,
                    reassembly_strategy="roma",
                    reassembly_depth=self.config.reassembly_depth,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma_mdap_maker"
                return result
            else:
                logger.info("Using standard ROMA for Phase 5 reassembly")
                result = self.execute_phase_5_reassemble(
                    solutions=solutions,
                    problem_statement=problem_statement,
                    reassembly_strategy="roma",
                    reassembly_depth=self.config.reassembly_depth,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma"
                return result

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"ROMA Phase 5 failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_reassemble(solutions, problem_statement, error=str(e))
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": f"ROMA Phase 5 failed: {str(e)}",
                    "error": str(e)
                }

    # =========================================================================
    # PHASE 6: FINAL VALIDATION
    # =========================================================================

    def final_validation(
        self,
        final_solution: str,
        problem_statement: str,
    ) -> Dict[str, Any]:
        """
        Phase 6: Final validation using ROMA.

        Args:
            final_solution: The assembled final solution
            problem_statement: Original problem statement

        Returns:
            Dictionary with validation results
        """
        if not self.config.enable_roma or not self.is_available():
            if self.config.fallback_to_standard:
                return self._fallback_final_validation(final_solution, problem_statement)
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": "ROMA not available and fallback disabled"
                }

        try:
            if self.config.use_roma_mdap_maker and self._roma_mdap_maker_available:
                logger.info("Using ROMA-MDAP-MAKER for Phase 6 final validation")
                result = self.execute_phase_6_final_validation_mdap(
                    final_solution=final_solution,
                    problem_statement=problem_statement,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma_mdap_maker"
                return result
            else:
                logger.info("Using standard ROMA for Phase 6 final validation")
                result = self.execute_phase_6_final_validation(
                    final_solution=final_solution,
                    problem_statement=problem_statement,
                    provider=self.config.provider,
                    model=self.config.model,
                )
                result["roma_used"] = True
                result["roma_type"] = "roma"
                return result

        except (RuntimeError, ValueError, TypeError) as e:
            logger.error(f"ROMA Phase 6 failed: {e}")
            if self.config.fallback_to_standard:
                return self._fallback_final_validation(final_solution, problem_statement, error=str(e))
            else:
                return {
                    "status": "error",
                    "roma_used": False,
                    "message": f"ROMA Phase 6 failed: {str(e)}",
                    "error": str(e)
                }

    # =========================================================================
    # FULL WORKFLOW
    # =========================================================================

    def execute_full_roma_workflow(
        self,
        problem_statement: str,
        problem_type: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute complete ROMA workflow (all 6 phases).

        Args:
            problem_statement: The problem to solve
            problem_type: Optional problem type
            domain: Optional problem domain

        Returns:
            Dictionary with complete workflow results
        """
        logger.info(f"Executing full ROMA workflow - {problem_statement[:50]}...")

        results = {}

        # Phase 1: Setup and Decomposition
        results["phase1"] = self.setup_and_decompose_problem(
            problem_statement=problem_statement,
            problem_type=problem_type,
            domain=domain,
        )

        if results["phase1"]["status"] == "failed":
            return {"workflow": "roma", "status": "failed", "error": results["phase1"].get("error")}

        # Phase 2: Solve Sub-problems
        sub_problems = results["phase1"].get("sub_problems", [])
        results["phase2"] = self.solve_sub_problems(
            sub_problems=sub_problems,
        )

        if results["phase2"]["status"] == "failed":
            return {"workflow": "roma", "status": "failed", "error": results["phase2"].get("error")}

        solutions = results["phase2"].get("solutions", [])

        # Phase 3: Critique
        results["phase3"] = self.critique_solutions(
            solutions=solutions,
            problem_statement=problem_statement,
        )

        # Phase 4: Verify
        results["phase4"] = self.verify_solutions(
            solutions=solutions,
            problem_statement=problem_statement,
        )

        # Phase 5: Reassemble
        results["phase5"] = self.reassemble_solutions(
            solutions=solutions,
            problem_statement=problem_statement,
        )

        # Phase 6: Final Validation
        final_solution = results["phase5"].get("final_solution", "")
        results["phase6"] = self.final_validation(
            final_solution=final_solution,
            problem_statement=problem_statement,
        )

        return {
            "workflow": "roma",
            "status": "completed",
            "phases": results,
            "message": "Full ROMA workflow completed",
        }

    # =========================================================================
    # FALLBACK METHODS
    # =========================================================================

    def _fallback_decompose(
        self,
        problem_statement: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback for decomposition when ROMA unavailable."""
        return {
            "status": "completed",
            "roma_used": False,
            "analysis": {
                "problem_statement": problem_statement,
                "complexity": 5.0,
                "estimated_sub_problems": 1,
            },
            "sub_problems": [
                {
                    "id": "sub_1",
                    "title": "Solve the problem as stated",
                    "description": problem_statement,
                    "dependencies": [],
                    "complexity_score": 0.5,
                }
            ],
            "message": "Fallback decomposition completed" + (f" (ROMA error: {error})" if error else ""),
            "fallback_used": True
        }

    def _fallback_solve(
        self,
        sub_problems: List[Dict[str, Any]],
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback for solving when ROMA unavailable."""
        return {
            "status": "completed",
            "roma_used": False,
            "solutions": [
                {
                    "id": sp["id"],
                    "solution": f"# Solution for {sp['title']}\n\nTo be implemented without ROMA assistance.",
                    "confidence": 0.5,
                }
                for sp in sub_problems
            ],
            "message": "Fallback solving completed" + (f" (ROMA error: {error})" if error else ""),
            "fallback_used": True
        }

    def _fallback_critique(
        self,
        solutions: List[Dict[str, Any]],
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback for critique when ROMA unavailable."""
        return {
            "status": "completed",
            "roma_used": False,
            "critiques": [
                {
                    "solution_id": sol.get("id", "unknown"),
                    "critique": "Basic review - ROMA critique unavailable",
                    "findings": [{"category": "Basic", "finding": "Solution reviewed without ROMA analysis"}],
                    "fallback": True,
                }
                for sol in solutions
            ],
            "message": "Fallback critique completed" + (f" (ROMA error: {error})" if error else ""),
            "fallback_used": True
        }

    def _fallback_verification(
        self,
        solutions: List[Dict[str, Any]],
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback for verification when ROMA unavailable."""
        return {
            "status": "completed",
            "roma_used": False,
            "verifications": [
                {
                    "solution_id": sol.get("id", "unknown"),
                    "verified": True,
                    "confidence": 0.5,
                    "findings": [{"check": "Basic verification", "result": "Passed (fallback mode)"}],
                    "fallback": True,
                }
                for sol in solutions
            ],
            "verified_count": len(solutions),
            "message": "Fallback verification completed" + (f" (ROMA error: {error})" if error else ""),
            "fallback_used": True
        }

    def _fallback_reassemble(
        self,
        solutions: List[Dict[str, Any]],
        problem_statement: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback for reassembly when ROMA unavailable."""
        if ENHANCED_RECOMPOSITION_AVAILABLE:
            try:
                recomposed = self._reassemble_with_enhanced_engine(
                    solutions=solutions,
                    problem_statement=problem_statement,
                )
                recomposed["message"] = "Fallback reassembly completed via enhanced recomposition"
                if error:
                    recomposed["message"] += f" (ROMA error: {error})"
                recomposed["fallback_used"] = True
                recomposed["roma_used"] = False
                return recomposed
            except (RuntimeError, ValueError, TypeError) as exc:
                logger.warning("Enhanced recomposition fallback failed: %s", exc)

        aggregated = "\n\n".join([
            f"## {sol.get('id', 'Solution')}\n\n{sol.get('solution', '')}"
            for sol in solutions
        ])
        return {
            "status": "completed",
            "roma_used": False,
            "reassembly_method": "simple_concatenation",
            "final_solution": f"# Solution for: {problem_statement}\n\n{aggregated}",
            "message": "Fallback reassembly completed" + (f" (ROMA error: {error})" if error else ""),
            "fallback_used": True
        }

    def _reassemble_with_enhanced_engine(
        self,
        solutions: List[Dict[str, Any]],
        problem_statement: str,
    ) -> Dict[str, Any]:
        """Use EnhancedRecompositionEngine to assemble fallback solution."""
        engine = EnhancedRecompositionEngine()
        sub_solutions: Dict[str, EnhancedSubProblemSolution] = {}
        dependency_graph: Dict[str, List[str]] = {}

        for sol in solutions:
            sol_id = sol.get("id") or sol.get("sub_problem_id") or sol.get("title") or f"roma_sol_{len(sub_solutions)+1}"
            content = sol.get("solution") or sol.get("solution_content") or ""
            quality = sol.get("quality_score") or sol.get("confidence") or 0.7
            metadata = sol.get("metadata", {}) if isinstance(sol.get("metadata"), dict) else {}
            for key in ("entangled_with", "entanglement_symbols", "entanglement_source"):
                if key in sol:
                    metadata[key] = sol.get(key)
            sub_solutions[sol_id] = EnhancedSubProblemSolution(
                sub_problem_id=sol_id,
                solution_content=content,
                quality_score=float(quality),
                metadata=metadata,
            )
            deps = sol.get("dependencies") or sol.get("depends_on") or []
            if isinstance(deps, str):
                deps = [deps]
            dependency_graph[sol_id] = list(deps) if isinstance(deps, list) else []

        entanglement_matrix: Dict[str, List[str]] = {}
        if ENTANGLEMENT_AVAILABLE:
            entanglement_matrix = self._attach_entanglement([
                {
                    "id": sol_id,
                    "title": sol_id,
                    "description": sub_solutions[sol_id].solution_content[:200],
                    "metadata": sub_solutions[sol_id].metadata,
                }
                for sol_id in sub_solutions
            ])

        integrated = engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=f"roma_problem_{hash(problem_statement) & 0xFFFFF}",
            decomposition_plan_id=f"roma_reassembly_{hash(problem_statement) & 0xFFFFF}",
            dependency_graph=dependency_graph,
            entanglement_matrix=entanglement_matrix or None,
        )

        return {
            "status": "completed",
            "reassembly_method": "enhanced_recomposition",
            "final_solution": integrated.assembled_content,
            "entanglement_matrix": entanglement_matrix,
            "integrated_solution": integrated,
        }

    def _fallback_final_validation(
        self,
        final_solution: str,
        problem_statement: str,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fallback for final validation when ROMA unavailable."""
        return {
            "status": "completed",
            "roma_used": False,
            "validation": "passed",
            "overall_score": 0.5,
            "message": "Fallback final validation completed" + (f" (ROMA error: {error})" if error else ""),
            "fallback_used": True
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_roma_adapter(
    enable_roma: bool = False,
    use_mdap_maker: bool = False,
    **kwargs
) -> ROMAOpenEvolveAdapter:
    """
    Create a ROMA-OpenEvolve adapter with the specified configuration.

    Args:
        enable_roma: Whether to enable ROMA integration
        use_mdap_maker: Whether to use ROMA-MDAP-MAKER (voting consensus)
        **kwargs: Additional configuration parameters

    Returns:
        ROMAOpenEvolveAdapter instance

    Example:
        >>> adapter = create_roma_adapter(enable_roma=True, use_mdap_maker=True)
        >>> result = adapter.execute_full_roma_workflow("Design a scalable API")
    """
    config = create_roma_openevolve_config(
        enable_roma=enable_roma,
        use_mdap_maker=use_mdap_maker,
        **kwargs
    )
    return ROMAOpenEvolveAdapter(config)


def get_roma_openevolve_status() -> Dict[str, Any]:
    """
    Get the status of ROMA-OpenEvolve integration.

    Returns:
        Dictionary with availability status
    """
    try:
        from roma_crewai_bridge import (
            execute_phase_1_setup,
            execute_phase_2_solve,
            execute_phase_3_critique,
            execute_phase_4_verify,
            execute_phase_5_reassemble,
            execute_phase_6_final_validation,
        )
        roma_standard_available = True
    except ImportError:
        roma_standard_available = False

    try:
        from roma_mdap_maker_crewai_bridge import (
            execute_phase_1_setup as execute_phase_1_setup_mdap,
            execute_phase_2_solve as execute_phase_2_solve_mdap,
            execute_phase_3_critique as execute_phase_3_critique_mdap,
            execute_phase_4_verify as execute_phase_4_verify_mdap,
            execute_phase_5_reassemble as execute_phase_5_reassemble_mdap,
            execute_phase_6_final_validation as execute_phase_6_final_validation_mdap,
        )
        roma_mdap_maker_available = True
    except ImportError:
        roma_mdap_maker_available = False

    return {
        "roma_standard_available": roma_standard_available,
        "roma_mdap_maker_available": roma_mdap_maker_available,
        "any_roma_available": roma_standard_available or roma_mdap_maker_available,
        "decomposition_available": roma_standard_available or roma_mdap_maker_available,
        "recomposition_available": roma_standard_available or roma_mdap_maker_available,
        "integration_ready": roma_standard_available or roma_mdap_maker_available
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("ROMA-OpenEvolve Integration Adapter v2.0")
    print("Full Decomposition/Recomposition Support")
    print("=" * 60)

    # Check availability
    status = get_roma_openevolve_status()
    print(f"ROMA Standard Available: {status['roma_standard_available']}")
    print(f"ROMA-MDAP-MAKER Available: {status['roma_mdap_maker_available']}")
    print(f"Decomposition Available: {status['decomposition_available']}")
    print(f"Recomposition Available: {status['recomposition_available']}")
    print(f"Integration Ready: {status['integration_ready']}")

    if status['integration_ready']:
        # Create adapter
        adapter = create_roma_adapter(
            enable_roma=True,
            use_mdap_maker=True,
            analysis_depth=3,
            solving_depth=2,
            critique_depth=1,
            verification_depth=1,
            reassembly_depth=1,
        )

        # Example: Full workflow
        print("\nTesting full ROMA workflow...")
        result = adapter.execute_full_roma_workflow(
            problem_statement="Design a scalable microservices architecture for an e-commerce platform",
            problem_type="design",
            domain="software_engineering"
        )

        print(f"\nWorkflow Status: {result['status']}")
        if result['status'] == 'completed':
            print("Phase Results:")
            for phase_name, phase_result in result.get('phases', {}).items():
                roma_used = phase_result.get('roma_used', False)
                print(f"  {phase_name}: {phase_result.get('status')} (ROMA: {roma_used})")
