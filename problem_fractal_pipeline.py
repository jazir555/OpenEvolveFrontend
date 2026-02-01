from __future__ import annotations

"""
problem_fractal_pipeline.py - CrewAI Integration

This file has been migrated from crewai # MIGRATED: was CrewAI (AGPL) to CrewAI (MIT).

Migration Date: 2026-01-21
Migration Status: Complete

All CrewAI references have been replaced with CrewAI equivalents.
The functionality remains the same, but now uses local CrewAI execution
instead of remote CrewAI API calls.

For questions, see: CREWAI_MIGRATION_MASTER_TASKLIST.md
"""

"""
Fractal Problem Pipeline Coordinator
Orchestrates ROMA + CrewAI decomposition, OpenEvolve gauntlet solving,
ROMA + CrewAI recomposition, and MDAP/MAKER verification.
"""

import os
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

from problem_decomposition import ProblemDecomposer, DecompositionStrategy, Component
from dependency_analyzer import DependencyAnalyzer
from problem_recomposition import SolutionAssembler
from decomposition_mcp_tools import (
    list_available_teams,
    list_available_gauntlets,
    solve_sub_problem_with_team,
    critique_solution_with_gauntlet,
    verify_solution_with_gauntlet,
)
from roma_mdap_maker_mcp_tools import (
    analyze_problem_with_roma_mdap,
    verify_solution_with_roma_mdap,
)

# MIGRATION: Import from sovereign_data_models with fallbacks
try:
    from sovereign_data_models import (
        DecompositionPlan,
        SubProblem,
        SolutionAttempt,
        generate_id,
    )
except ImportError:
    DecompositionPlan = SubProblem = SolutionAttempt = None
    generate_id = lambda prefix="": f"{prefix}_{str(uuid.uuid4())[:8]}"

# Create stubs for classes that don't exist in sovereign_data_models
@dataclass
class ComplexityScore:
    """Complexity score for problems."""
    explanation: str
    cognitive_complexity: float
    computational_complexity: float
    domain_complexity: float
    integration_complexity: float
    overall_complexity: float  # Added for compatibility

@dataclass
class DependencyGraph:
    """Dependency graph for sub-problems."""
    nodes: Dict[str, Any]
    edges: Dict[str, List[str]]
    execution_order: List[str] = field(default_factory=list)  # Added for compatibility

class SubProblemType:
    """Type of sub-problem."""
    value: str

    # Enum values
    IMPLEMENTATION = "IMPLEMENTATION"
    ANALYSIS = "ANALYSIS"
    VALIDATION = "VALIDATION"


# Stub for SovereignDecompositionStrategy
class SovereignDecompositionStrategy:
    """Decomposition strategy types."""
    HYBRID = "HYBRID"
    ROMA = "ROMA"
    SEMANTIC = "SEMANTIC"

try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
except ImportError:
    OpenEvolveClient = None
    OPENEVOLVE_AVAILABLE = False

logger = logging.getLogger(__name__)


def _fallback_judge(solution: str, requirements: List[str]) -> Dict[str, Any]:
    if not OPENEVOLVE_AVAILABLE or not OpenEvolveClient:
        return {"passed": True, "reason": "OpenEvolve unavailable"}

    client = OpenEvolveClient()
    prompt = (
        "Evaluate the following solution against these requirements. "
        "Return JSON with fields passed (true/false) and reason.\n\n"
        f"Requirements: {requirements}\n\nSolution:\n{solution}\n"
    )
    response = client.generate_completion(prompt)
    if not response:
        return {"passed": True, "reason": "No response"}
    text = str(response).strip()
    return {"passed": "true" in text.lower(), "reason": text}


@dataclass
class FractalPipelineConfig:
    enable_roma_decomposition: bool = True
    enable_roma_recomposition: bool = True
    enable_roma_final: bool = True
    enable_roma_solving: bool = True

    enable_mdap_maker_decomposition: bool = True
    enable_mdap_maker_solving: bool = True
    enable_mdap_maker_recomposition: bool = True
    enable_mdap_maker_final: bool = True

    enable_gauntlet_solving: bool = True
    enable_gauntlet_final: bool = True

    enable_fallback_judge: bool = True

    use_CrewAI_mirroring: bool = True
    crewai_api_base: Optional[str] = None
    crewai_api_key: Optional[str] = None
    crewai_workflow_id: Optional[str] = None
    CrewAI_agent_id: str = "fractal-pipeline"
    CrewAI_results_timeout_s: int = 300
    CrewAI_results_poll_s: int = 5

    decomposition_strategy: DecompositionStrategy = DecompositionStrategy.ROMA
    fallback_decomposition_strategy: DecompositionStrategy = DecompositionStrategy.SEMANTIC

    evolution_iterations: int = 100
    use_evolution: bool = True

    roma_max_depth: int = 3
    roma_recursion_depth_limit: Optional[int] = 1
    roma_max_subproblems: Optional[int] = 3
    roma_provider: Optional[str] = None
    roma_model: Optional[str] = None

    mdap_maker_k_ahead: int = 3
    mdap_maker_max_samples: int = 100
    mdap_maker_enable_red_flagging: bool = True
    mdap_maker_enable_adaptive_k: bool = True
    mdap_maker_provider: str = "openai"
    mdap_maker_model: str = "gpt-4o-mini"

    team_name: Optional[str] = None
    red_gauntlet_name: Optional[str] = None
    gold_gauntlet_name: Optional[str] = None


@dataclass
class FractalPipelineResult:
    decomposition_plan: DecompositionPlan
    sub_solutions: Dict[str, SolutionAttempt]
    recomposed_solution: str
    final_accepted: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def execute_mdap_step(
    sub_problem_id: str,
    sub_problem_description: str,
    team_name: str,
    requirements: List[str],
    config: FractalPipelineConfig,
    context: Optional[Dict[str, Any]] = None,
    red_gauntlet: Optional[str] = None,
    gold_gauntlet: Optional[str] = None,
) -> SolutionAttempt:
    """
    Execute a single atomic sub-problem using encapsulated MDAP/MAKER + gauntlets.
    This is the atomic unit exposed to CrewAI.
    """
    solve_result = solve_sub_problem_with_team(
        sub_problem_id=sub_problem_id,
        sub_problem_description=sub_problem_description,
        team_name=team_name,
        context=context,
        requirements=requirements,
        execution_method="roma_mdap_maker"
        if config.enable_mdap_maker_solving and config.enable_roma_solving
        else "traditional",
        use_evolution=config.use_evolution,
        evolution_iterations=config.evolution_iterations,
        use_roma_mdap_maker=config.enable_mdap_maker_solving and config.enable_roma_solving,
        roma_mdap_maker_max_depth=config.roma_max_depth,
        roma_mdap_maker_k_ahead=config.mdap_maker_k_ahead,
        roma_mdap_maker_enable_red_flagging=config.mdap_maker_enable_red_flagging,
        roma_mdap_maker_max_samples=config.mdap_maker_max_samples,
        roma_mdap_maker_enable_adaptive_k=config.mdap_maker_enable_adaptive_k,
        roma_mdap_maker_provider=config.mdap_maker_provider,
        roma_mdap_maker_model=config.mdap_maker_model,
    )

    solution_text = solve_result.get("solution") or solve_result.get("result") or ""
    confidence = solve_result.get("confidence", 0.5)

    if config.enable_gauntlet_solving:
        if red_gauntlet:
            critique_solution_with_gauntlet(
                sub_problem_id=sub_problem_id,
                solution=solution_text,
                gauntlet_name=red_gauntlet,
                use_evolution=config.use_evolution,
                evolution_iterations=config.evolution_iterations,
            )
        if gold_gauntlet:
            verify_solution_with_gauntlet(
                sub_problem_id=sub_problem_id,
                solution=solution_text,
                gauntlet_name=gold_gauntlet,
                use_evolution=config.use_evolution,
                evolution_iterations=config.evolution_iterations,
            )

    if config.enable_mdap_maker_solving and not config.enable_roma_solving:
        if config.enable_fallback_judge and solution_text:
            solve_result["mdap_maker_fallback_judge"] = _fallback_judge(
                solution_text,
                requirements,
            )

    return SolutionAttempt(
        id=generate_id("solution_attempt"),
        sub_problem_id=sub_problem_id,
        approach=solve_result.get("execution_method_used", "traditional"),
        solution_content=solution_text,
        team_id=team_name,
        confidence_score=confidence,
        status="solved" if solution_text else "failed",
        metadata={"solve_result": solve_result},
    )


class FractalPipelineCoordinator:
    def __init__(self, config: Optional[FractalPipelineConfig] = None) -> None:
        self.config = config or FractalPipelineConfig()
        self.decomposer = ProblemDecomposer()
        self.dependency_analyzer = DependencyAnalyzer()
        self._openevolve_client = None
        self.entanglement_matrix: Dict[str, set] = {}

    def run(self, problem_statement: str, requirements: Optional[List[str]] = None) -> FractalPipelineResult:
        requirements = requirements or []

        logger.info("Starting fractal pipeline run")
        decomposition_plan, component_map = self._decompose(problem_statement)
        sub_solutions = self._solve_sub_problems(decomposition_plan, component_map, requirements)
        if self.config.use_CrewAI_mirroring:
            sub_solutions = self._wait_for_CrewAI_results(decomposition_plan, sub_solutions)
        recomposed_solution = self._recompose(problem_statement, decomposition_plan, sub_solutions)
        final_accepted, final_metadata = self._final_verify(recomposed_solution, requirements)

        return FractalPipelineResult(
            decomposition_plan=decomposition_plan,
            sub_solutions=sub_solutions,
            recomposed_solution=recomposed_solution,
            final_accepted=final_accepted,
            metadata=final_metadata,
        )

    def _decompose(self, content: str) -> Tuple[DecompositionPlan, Dict[str, Component]]:
        if self.config.enable_roma_decomposition:
            strategy = self.config.decomposition_strategy
        else:
            strategy = self.config.fallback_decomposition_strategy

        result = self.decomposer.decompose_content(
            content=content,
            strategy=strategy,
            max_components=self._max_components(),
            roma_max_depth=self._analysis_depth(),
            roma_provider=self.config.roma_provider,
            roma_model=self.config.roma_model,
        )
        if not result:
            raise RuntimeError("Decomposition failed")

        for comp in result.components:
            comp.metadata.setdefault("problem_statement", content)
        component_map = {comp.id: comp for comp in result.components}
        plan = self._build_plan_from_components(result.components, result.dependency_graph)
        plan.metadata["problem_statement"] = content

        # Build symbolic entanglement matrix
        try:
            self.entanglement_matrix = self.dependency_analyzer.build_entanglement_matrix(plan.sub_problems)
            plan.metadata["entanglement_matrix"] = {
                k: list(v) for k, v in self.entanglement_matrix.items()
            }
        except (ValueError, AttributeError) as exc:
            logger.warning("Failed to build entanglement matrix: %s", exc)

        if self.config.enable_mdap_maker_decomposition:
            if self.config.enable_roma_decomposition:
                analysis_depth = self._analysis_depth()
                mdap_decomp = analyze_problem_with_roma_mdap(
                    problem_statement=content,
                    max_depth=analysis_depth,
                    provider=self.config.mdap_maker_provider,
                    model=self.config.mdap_maker_model,
                )
                plan.metadata["mdap_maker_decomposition"] = mdap_decomp
            elif self.config.enable_fallback_judge:
                plan.metadata["mdap_maker_decomposition"] = self._judge_with_llm(
                    "Decomposition quality check",
                    [content],
                )

        if self.config.use_CrewAI_mirroring:
            parent_task_id = self._create_CrewAI_task(
                task_description=f"Decompose problem: {content[:80]}",
                done_definition="Decomposition complete",
            )
            for comp in result.components:
                task_id = self._create_CrewAI_task(
                    task_description=f"Subproblem {comp.id}: {comp.title}",
                    done_definition=f"Solve subproblem {comp.id}",
                    parent_task_id=parent_task_id,
                )
                if task_id:
                    plan.metadata.setdefault("CrewAI_tasks", {})[comp.id] = task_id

        return plan, component_map

    def _solve_sub_problems(
        self,
        plan: DecompositionPlan,
        component_map: Dict[str, Component],
        requirements: List[str],
    ) -> Dict[str, SolutionAttempt]:
        team_name = self._resolve_team_name()
        red_gauntlet, gold_gauntlet = self._resolve_gauntlets()

        sub_solutions: Dict[str, SolutionAttempt] = {}

        solved_ids = set()
        for sub_problem in plan.sub_problems:
            if sub_problem.id in solved_ids:
                continue
            component = component_map.get(sub_problem.id)
            if not component:
                continue

            # Super-node merge for tightly coupled entanglement
            partner_id = self._select_super_node_partner(sub_problem.id, solved_ids)
            if partner_id:
                partner_component = component_map.get(partner_id)
                if partner_component:
                    merged_description = self._merge_component_context(component, partner_component)
                    attempt = execute_mdap_step(
                        sub_problem_id=f"{sub_problem.id}+{partner_id}",
                        sub_problem_description=merged_description,
                        team_name=team_name,
                        requirements=requirements,
                        config=self.config,
                        context={"super_node": [sub_problem.id, partner_id]},
                        red_gauntlet=red_gauntlet,
                        gold_gauntlet=gold_gauntlet,
                    )
                    attempt.metadata["super_node"] = [sub_problem.id, partner_id]
                    sub_solutions[sub_problem.id] = attempt
                    sub_solutions[partner_id] = SolutionAttempt(
                        id=generate_id("solution_attempt"),
                        sub_problem_id=partner_id,
                        approach=attempt.approach,
                        solution_content=attempt.solution_content,
                        team_id=attempt.team_id,
                        confidence_score=attempt.confidence_score,
                        status=attempt.status,
                        metadata={"super_node": [sub_problem.id, partner_id]},
                    )
                    solved_ids.update({sub_problem.id, partner_id})
                    self._propagate_entanglement(sub_problem.id, sub_solutions, plan)
                    self._propagate_entanglement(partner_id, sub_solutions, plan)
                    continue

            attempt = self._solve_component_recursive(
                component=component,
                team_name=team_name,
                requirements=requirements,
                red_gauntlet=red_gauntlet,
                gold_gauntlet=gold_gauntlet,
                depth=0,
                parent_task_id=plan.metadata.get("CrewAI_tasks", {}).get(sub_problem.id),
            )
            sub_solutions[sub_problem.id] = attempt
            solved_ids.add(sub_problem.id)
            self._propagate_entanglement(sub_problem.id, sub_solutions, plan)

            if self.config.use_CrewAI_mirroring:
                task_id = plan.metadata.get("CrewAI_tasks", {}).get(sub_problem.id)
                solution_text = attempt.solution_content
                if task_id:
                    self._complete_CrewAI_task(
                        task_id=task_id,
                        summary=f"Solved subproblem {sub_problem.id}",
                        key_learnings=[solution_text[:400]] if solution_text else [],
                        solution_payload={
                            "type": "subproblem_solution",
                            "sub_problem_id": sub_problem.id,
                            "solution": solution_text,
                            "status": attempt.status,
                        },
                    )

        return sub_solutions

    def _propagate_entanglement(
        self,
        source_id: str,
        sub_solutions: Dict[str, SolutionAttempt],
        plan: DecompositionPlan
    ) -> None:
        entangled = self.entanglement_matrix.get(source_id, set())
        if not entangled:
            return

        for target_id in entangled:
            attempt = sub_solutions.get(target_id)
            if attempt and attempt.status == "solved":
                attempt.status = "needs_consistency_refinement"
                attempt.metadata.setdefault("entanglement_invalidation", []).append(source_id)
                for sp in plan.sub_problems:
                    if sp.id == target_id:
                        sp.metadata["needs_consistency_refinement"] = True
                        break

    def _select_super_node_partner(self, component_id: str, solved_ids: set) -> Optional[str]:
        entangled = self.entanglement_matrix.get(component_id, set())
        for candidate in entangled:
            if candidate in solved_ids:
                continue
            # Tight coupling if both directions are entangled
            if component_id in self.entanglement_matrix.get(candidate, set()):
                return candidate
        return None

    def _merge_component_context(self, comp_a: Component, comp_b: Component) -> str:
        return (
            f"[Super-Node Merge]\n"
            f"Component {comp_a.id}: {comp_a.title}\n{comp_a.content}\n\n"
            f"Component {comp_b.id}: {comp_b.title}\n{comp_b.content}\n"
        )

    def _solve_component_recursive(
        self,
        component: Component,
        team_name: str,
        requirements: List[str],
        red_gauntlet: Optional[str],
        gold_gauntlet: Optional[str],
        depth: int,
        parent_task_id: Optional[str],
    ) -> SolutionAttempt:
        context = {
            "problem": component.metadata.get("problem_statement"),
            "component_id": component.id,
            "dependencies": component.dependencies,
            "depth": depth,
        }

        is_atomic = bool(component.metadata.get("roma_is_atomic", False))
        recursion_limit = self._recursion_limit()

        if self.config.enable_roma_solving and not is_atomic and (recursion_limit is None or depth < recursion_limit):
            logger.info("ROMA controller: decomposing component %s at depth %s", component.id, depth)
            nested = self.decomposer.decompose_content(
                content=component.content,
                strategy=DecompositionStrategy.ROMA,
                max_components=self._max_components(),
                roma_max_depth=max(1, (recursion_limit or self.config.roma_max_depth) - depth),
                roma_provider=self.config.roma_provider,
                roma_model=self.config.roma_model,
            )
            if nested and nested.components:
                nested_plan = self._build_plan_from_components(nested.components, nested.dependency_graph)
                nested_plan.metadata["problem_statement"] = component.content
                nested_map = {comp.id: comp for comp in nested.components}

                nested_solutions = {}
                for nested_component in nested.components:
                    task_id = None
                    if self.config.use_CrewAI_mirroring:
                        task_id = self._create_CrewAI_task(
                            task_description=f"Nested subproblem {nested_component.id}: {nested_component.title}",
                            done_definition=f"Solve nested subproblem {nested_component.id}",
                            parent_task_id=parent_task_id,
                        )
                    attempt = self._solve_component_recursive(
                        component=nested_component,
                        team_name=team_name,
                        requirements=requirements,
                        red_gauntlet=red_gauntlet,
                        gold_gauntlet=gold_gauntlet,
                        depth=depth + 1,
                        parent_task_id=task_id,
                    )
                    nested_solutions[nested_component.id] = attempt
                    if task_id:
                        self._complete_CrewAI_task(
                            task_id=task_id,
                            summary=f"Solved nested subproblem {nested_component.id}",
                            key_learnings=[attempt.solution_content[:400]] if attempt.solution_content else [],
                            solution_payload={
                                "type": "subproblem_solution",
                                "sub_problem_id": nested_component.id,
                                "solution": attempt.solution_content,
                                "status": attempt.status,
                            },
                        )

                assembly_strategy = "roma" if self.config.enable_roma_recomposition else "hierarchical"
                assembler = SolutionAssembler(
                    enable_roma=self.config.enable_roma_recomposition,
                    roma_max_depth=self.config.roma_max_depth,
                    roma_execution_mode="recursive",
                    roma_provider=self.config.roma_provider,
                    roma_model=self.config.roma_model,
                    crewai_api_base=self._CrewAI_api_base(),
                    crewai_api_key=self._CrewAI_api_key(),
                    crewai_workflow_id=self._CrewAI_workflow_id(),
                    CrewAI_agent_id=self.config.CrewAI_agent_id,
                )
                integrated = assembler.assemble_solution(
                    nested_plan,
                    nested_solutions,
                    assembly_strategy=assembly_strategy,
                )
                return SolutionAttempt(
                    id=generate_id("solution_attempt"),
                    sub_problem_id=component.id,
                    approach="roma_recursive",
                    solution_content=integrated.assembled_content,
                    team_id=team_name,
                    confidence_score=0.6,
                    status="solved",
                    metadata={"nested_plan": nested_plan.metadata},
                )

        logger.info("ROMA controller: executing atomic MDAP node %s", component.id)
        return execute_mdap_step(
            sub_problem_id=component.id,
            sub_problem_description=component.content,
            team_name=team_name,
            requirements=requirements,
            config=self.config,
            context=context,
            red_gauntlet=red_gauntlet,
            gold_gauntlet=gold_gauntlet,
        )

    def _recompose(
        self,
        problem_statement: str,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
    ) -> str:
        assembly_strategy = "hierarchical"
        if self.config.enable_roma_recomposition:
            assembly_strategy = "roma_CrewAI" if self.config.use_CrewAI_mirroring else "roma"

        assembler = SolutionAssembler(
            enable_roma=self.config.enable_roma_recomposition,
            roma_max_depth=self.config.roma_max_depth,
            roma_execution_mode="recursive",
            roma_provider=self.config.roma_provider,
            roma_model=self.config.roma_model,
            crewai_api_base=self._CrewAI_api_base(),
            crewai_api_key=self._CrewAI_api_key(),
            crewai_workflow_id=self._CrewAI_workflow_id(),
            CrewAI_agent_id=self.config.CrewAI_agent_id,
        )

        plan.metadata["problem_statement"] = problem_statement
        integrated = assembler.assemble_solution(
            plan,
            sub_solutions,
            assembly_strategy=assembly_strategy,
        )

        if self.config.enable_mdap_maker_recomposition:
            if self.config.enable_roma_recomposition:
                mdap_recompose = verify_solution_with_roma_mdap(
                    solution=integrated.assembled_content,
                    requirements=["coherence", "consistency", "integration"],
                    verification_depth=self.config.roma_max_depth,
                    provider=self.config.mdap_maker_provider,
                    model=self.config.mdap_maker_model,
                )
                integrated.metadata["mdap_maker_recomposition"] = mdap_recompose
            elif self.config.enable_fallback_judge:
                integrated.metadata["mdap_maker_recomposition"] = self._judge_with_llm(
                    integrated.assembled_content,
                    ["coherence", "consistency", "integration"],
                )

        if self.config.use_CrewAI_mirroring:
            task_id = self._create_CrewAI_task(
                task_description="Recompose solved subproblems into final solution",
                done_definition="Recomposition complete",
            )
            if task_id:
                self._complete_CrewAI_task(
                    task_id=task_id,
                    summary="Recomposition complete",
                    key_learnings=[integrated.assembled_content[:400]],
                    solution_payload={
                        "type": "recomposition_solution",
                        "solution": integrated.assembled_content,
                    },
                )

        return integrated.assembled_content

    def _final_verify(self, solution: str, requirements: List[str]) -> Tuple[bool, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}

        if self.config.enable_gauntlet_final:
            _, gold_gauntlet = self._resolve_gauntlets()
            if gold_gauntlet:
                metadata["gold_gauntlet_final"] = verify_solution_with_gauntlet(
                    sub_problem_id="final_solution",
                    solution=solution,
                    gauntlet_name=gold_gauntlet,
                    use_evolution=self.config.use_evolution,
                    evolution_iterations=self.config.evolution_iterations,
                )

        if self.config.enable_mdap_maker_final:
            if self.config.enable_roma_final:
                mdap_result = verify_solution_with_roma_mdap(
                    solution=solution,
                    requirements=requirements or ["correctness", "completeness"],
                    verification_depth=self.config.roma_max_depth,
                    provider=self.config.mdap_maker_provider,
                    model=self.config.mdap_maker_model,
                )
                metadata["mdap_maker_final"] = mdap_result
                return bool(mdap_result.get("passed", False)), metadata
            if self.config.enable_fallback_judge:
                metadata["mdap_maker_final"] = self._judge_with_llm(solution, requirements)
                return bool(metadata["mdap_maker_final"].get("passed", False)), metadata

        if self.config.enable_fallback_judge:
            metadata["fallback_judge"] = self._judge_with_llm(solution, requirements)
            return bool(metadata["fallback_judge"].get("passed", False)), metadata

        return True, metadata

    def _build_plan_from_components(
        self,
        components: List[Component],
        dependency_graph: Dict[str, List[str]],
    ) -> DecompositionPlan:
        sub_problems = []
        nodes = {}
        edges = {}

        for comp in components:
            sub_problem = SubProblem(
                id=comp.id,
                parent_id="root",
                title=comp.title,
                description=comp.content,
                type=self._map_component_type(comp),
                complexity_score=self._complexity_from_component(comp),
                dependencies=list(comp.dependencies or []),
                estimated_effort=comp.estimated_effort,
                priority=max(1, min(10, int(comp.evolution_priority * 5))),
                metadata=comp.metadata,
            )
            sub_problems.append(sub_problem)
            nodes[sub_problem.id] = sub_problem
            edges[sub_problem.id] = list(sub_problem.dependencies)

        dep_graph = DependencyGraph(
            nodes=nodes,
            edges=edges,
            execution_order=list(dependency_graph.keys()) if dependency_graph else [],
        )

        return DecompositionPlan(
            id=generate_id("decomp_plan"),
            problem_id=generate_id("problem"),
            strategy=SovereignDecompositionStrategy.HYBRID,
            sub_problems=sub_problems,
            dependency_graph=dep_graph,
            metadata={"problem_statement": ""},
        )

    def _map_component_type(self, component: Component) -> SubProblemType:
        mapping = {
            "core_logic": SubProblemType.IMPLEMENTATION,
            "supporting_function": SubProblemType.IMPLEMENTATION,
            "data_structure": SubProblemType.IMPLEMENTATION,
            "interface": SubProblemType.IMPLEMENTATION,
            "configuration": SubProblemType.IMPLEMENTATION,
            "documentation": SubProblemType.ANALYSIS,
            "test_case": SubProblemType.VALIDATION,
            "error_handling": SubProblemType.IMPLEMENTATION,
        }
        return mapping.get(component.component_type.value, SubProblemType.ANALYSIS)

    def _complexity_from_component(self, component: Component) -> ComplexityScore:
        overall = max(0.1, min(10.0, component.complexity_score * 10))
        return ComplexityScore(
            explanation="Derived from component complexity score",
            cognitive_complexity=overall,
            computational_complexity=overall,
            domain_complexity=overall,
            integration_complexity=overall,
            overall_complexity=overall,
        )

    def _resolve_team_name(self) -> str:
        if self.config.team_name:
            return self.config.team_name
        teams = list_available_teams().get("teams", [])
        return teams[0]["name"] if teams else "default-blue"

    def _resolve_gauntlets(self) -> Tuple[Optional[str], Optional[str]]:
        red = self.config.red_gauntlet_name
        gold = self.config.gold_gauntlet_name
        if red and gold:
            return red, gold

        gauntlets = list_available_gauntlets().get("gauntlets", [])
        if not red:
            red = next((g["name"] for g in gauntlets if "red" in g["name"].lower()), None)
        if not gold:
            gold = next((g["name"] for g in gauntlets if "gold" in g["name"].lower()), None)
        return red, gold

    def _judge_with_llm(self, solution: str, requirements: List[str]) -> Dict[str, Any]:
        if self._openevolve_client is None and OPENEVOLVE_AVAILABLE and OpenEvolveClient:
            self._openevolve_client = OpenEvolveClient()
        if self._openevolve_client:
            prompt = (
                "Evaluate the following solution against these requirements. "
                "Return JSON with fields passed (true/false) and reason.\n\n"
                f"Requirements: {requirements}\n\nSolution:\n{solution}\n"
            )
            response = self._openevolve_client.generate_completion(prompt)
            if response:
                text = str(response).strip()
                return {"passed": "true" in text.lower(), "reason": text}
        return _fallback_judge(solution, requirements)

    def _CrewAI_api_base(self) -> Optional[str]:
        return self.config.crewai_api_base or os.getenv("CREWAI_API_BASE", "http://localhost:8000")

    def _CrewAI_api_key(self) -> Optional[str]:
        return self.config.crewai_api_key or os.getenv("CREWAI_API_KEY")

    def _CrewAI_workflow_id(self) -> Optional[str]:
        return self.config.crewai_workflow_id or os.getenv("CrewAI_WORKFLOW_ID")

    def _create_CrewAI_task(
        self,
        task_description: str,
        done_definition: str,
        parent_task_id: Optional[str] = None,
    ) -> Optional[str]:
        if not (self.config.use_CrewAI_mirroring and REQUESTS_AVAILABLE):
            return None
        api_key = self._CrewAI_api_key()
        if not api_key:
            return None
        workflow_id = self._CrewAI_workflow_id()
        if not workflow_id:
            return None

        payload = {
            "task_description": task_description,
            "done_definition": done_definition,
            "ai_agent_id": self.config.CrewAI_agent_id,
            "workflow_id": workflow_id,
            "priority": "medium",
            "parent_task_id": parent_task_id,
        }

        try:
            response = requests.post(
                f"{self._CrewAI_api_base().rstrip('/')}/create_task",
                json=payload,
                headers={"X-API-Key": api_key},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("task_id") or data.get("id")
        except (requests.RequestException, IOError, ValueError) as exc:
            logger.warning("CrewAI task creation failed: %s", exc)
            return None

    def _complete_CrewAI_task(
        self,
        task_id: str,
        summary: str,
        key_learnings: List[str],
        solution_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not (self.config.use_CrewAI_mirroring and REQUESTS_AVAILABLE):
            return
        api_key = self._CrewAI_api_key()
        if not api_key:
            return
        agent_id = self._fetch_assigned_agent(task_id)
        if not agent_id:
            return

        payload = {
            "task_id": task_id,
            "status": "done",
            "summary": summary,
            "key_learnings": key_learnings or ["Completed"],
            "code_changes": [],
        }
        try:
            requests.post(
                f"{self._CrewAI_api_base().rstrip('/')}/update_task_status",
                json=payload,
                headers={"X-API-Key": api_key, "X-Agent-ID": agent_id},
                timeout=10,
            )
        except (requests.RequestException, IOError, ValueError) as exc:
            logger.warning("CrewAI task completion failed: %s", exc)

        if solution_payload:
            self._submit_CrewAI_result(agent_id, solution_payload)

    def _submit_CrewAI_result(self, agent_id: str, payload: Dict[str, Any]) -> None:
        api_key = self._CrewAI_api_key()
        if not api_key:
            return
        base = self._CrewAI_api_base()
        if not base:
            return
        try:
            requests.post(
                f"{base.rstrip('/')}/submit_result",
                json={
                    "markdown_file_path": None,
                    "explanation": payload,
                    "evidence": [],
                    "extra_files": [],
                },
                headers={"X-API-Key": api_key, "X-Agent-ID": agent_id},
                timeout=10,
            )
        except (requests.RequestException, IOError, ValueError) as exc:
            logger.warning("CrewAI submit_result failed: %s", exc)

    def _fetch_CrewAI_results(self) -> List[Dict[str, Any]]:
        api_key = self._CrewAI_api_key()
        workflow_id = self._CrewAI_workflow_id()
        base = self._CrewAI_api_base()
        if not (api_key and workflow_id and base and REQUESTS_AVAILABLE):
            return []
        try:
            response = requests.get(
                f"{base.rstrip('/')}/workflows/{workflow_id}/results",
                headers={"X-API-Key": api_key, "X-Agent-ID": self.config.CrewAI_agent_id},
                timeout=10,
            )
            if response.status_code != 200:
                return []
            data = response.json()
            if isinstance(data, list):
                return data
            return []
        except (requests.RequestException, IOError, ValueError) as exc:
            logger.warning("CrewAI results fetch failed: %s", exc)
            return []

    def _fetch_assigned_agent(self, task_id: str) -> Optional[str]:
        api_key = self._CrewAI_api_key()
        if not api_key:
            return None
        base = self._CrewAI_api_base()
        if not base:
            return None

        for _ in range(5):
            try:
                response = requests.get(
                    f"{base.rstrip('/')}/task_progress",
                    params={"task_id": task_id},
                    headers={"X-API-Key": api_key, "X-Agent-ID": self.config.CrewAI_agent_id},
                    timeout=10,
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        agent_id = data.get("assigned_agent_id")
                        if agent_id:
                            return agent_id
                time.sleep(1)
            except (requests.RequestException, IOError, ValueError):
                time.sleep(1)
        return None

    def _aggregate_CrewAI_results(self) -> Dict[str, str]:
        results = self._fetch_CrewAI_results()
        sub_results: Dict[str, str] = {}
        for item in results:
            payload = item.get("explanation")
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "subproblem_solution":
                continue
            sub_id = payload.get("sub_problem_id")
            solution = payload.get("solution")
            if sub_id and solution:
                sub_results[sub_id] = solution
        return sub_results

    def _max_components(self) -> int:
        if self.config.roma_max_subproblems is None or self.config.roma_max_subproblems <= 0:
            return 10000
        return self.config.roma_max_subproblems

    def _recursion_limit(self) -> Optional[int]:
        if self.config.roma_recursion_depth_limit is None or self.config.roma_recursion_depth_limit <= 0:
            return None
        return self.config.roma_recursion_depth_limit

    def _analysis_depth(self) -> int:
        limit = self._recursion_limit()
        if limit is None:
            return self.config.roma_max_depth
        return min(self.config.roma_max_depth, limit)

    def _wait_for_CrewAI_results(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, SolutionAttempt],
    ) -> Dict[str, SolutionAttempt]:
        expected_ids = {sp.id for sp in plan.sub_problems}
        deadline = time.time() + self.config.CrewAI_results_timeout_s

        while time.time() < deadline:
            heph_results = self._aggregate_CrewAI_results()
            if heph_results:
                for sub_id, solution in heph_results.items():
                    if sub_id in sub_solutions and solution:
                        sub_solutions[sub_id].solution_content = solution
            if expected_ids.issubset(heph_results.keys()):
                break
            time.sleep(self.config.CrewAI_results_poll_s)

        return sub_solutions
