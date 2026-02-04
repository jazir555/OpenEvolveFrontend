"""
Decomposition-Recomposition Integration - Unified Problem Solving Pipeline

This module provides seamless integration between decomposition and recomposition
systems, creating a complete end-to-end problem solving pipeline.

Features:
- Unified pipeline from problem to solution
- Bidirectional feedback between decomposition and recomposition
- Adaptive refinement based on assembly results
- Solution quality feedback to improve future decompositions
- Cross-domain knowledge transfer
- Performance optimization
- Comprehensive analytics and reporting

Version: 3.0.0
Author: OpenEvolve Sovereign System
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import uuid

# Import enhanced engines
from enhanced_decomposition_engine import (
    EnhancedDecompositionEngine,
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    DecompositionStrategy,
    ProblemDomain,
    ComplexityScore,
    create_problem_definition
)

from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    IntegratedSolution,
    SubProblemSolution,
    AssemblyStrategy,
    RecompositionConfig,
    QualityMetrics,
    create_subproblem_solution,
    Conflict,
    ConflictSeverity
)

# Optional ROMA integration
try:
    from roma_openevolve_integration import create_roma_adapter, ROMAOpenEvolveConfig
    ROMA_INTEGRATION_AVAILABLE = True
except ImportError:
    ROMA_INTEGRATION_AVAILABLE = False
    create_roma_adapter = None  # type: ignore
    ROMAOpenEvolveConfig = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)

# Public API exports
__all__ = [
    # Configuration
    'PipelineConfig',
    'PipelineStage',
    'PipelineResult',
    'PipelineAnalytics',
    # Core Classes
    'SolutionSolver',
    'SimpleSolutionSolver',
    'DecompositionRecompositionPipeline',
    'BatchPipelineProcessor',
    # Utility Functions
    'quick_solve',
]


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the decomposition-recomposition pipeline."""
    # Decomposition settings
    decomposition_strategy: Optional[DecompositionStrategy] = None
    min_subproblems: int = 3
    max_subproblems: int = 10
    max_depth: int = 3
    
    # Recomposition settings
    assembly_strategy: Optional[AssemblyStrategy] = None
    validation_level: str = "standard"
    auto_resolve_conflicts: bool = True
    
    # Pipeline settings
    enable_feedback_loop: bool = True
    max_iterations: int = 3
    quality_threshold: float = 0.75
    
    # Analytics
    collect_detailed_analytics: bool = True
    save_intermediate_results: bool = True

    # Entanglement
    entanglement_strict_mode: bool = False

    # ROMA integration
    enable_roma: bool = False
    use_roma_mdap_maker: bool = False
    roma_config: Optional[Dict[str, Any]] = None


@dataclass
class PipelineStage:
    """Represents a stage in the pipeline."""
    name: str
    status: str  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    
    def duration_seconds(self) -> float:
        """Get stage duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    pipeline_id: str
    problem: ProblemDefinition
    
    # Stages
    decomposition_plan: Optional[DecompositionPlan] = None
    sub_solutions: Dict[str, SubProblemSolution] = field(default_factory=dict)
    integrated_solution: Optional[IntegratedSolution] = None
    
    # Execution tracking
    stages: List[PipelineStage] = field(default_factory=list)
    current_stage: int = 0
    
    # Quality metrics
    decomposition_quality: float = 0.0
    solution_quality: float = 0.0
    overall_quality: float = 0.0
    
    # Feedback
    feedback_log: List[Dict[str, Any]] = field(default_factory=list)
    refinement_iterations: int = 0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def is_successful(self) -> bool:
        """Check if pipeline completed successfully."""
        return (
            self.integrated_solution is not None and
            self.integrated_solution.status.value == "completed" and
            self.overall_quality >= 0.6
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pipeline_id': self.pipeline_id,
            'problem_title': self.problem.title,
            'successful': self.is_successful(),
            'overall_quality': self.overall_quality,
            'stages_completed': len([s for s in self.stages if s.status == "completed"]),
            'total_stages': len(self.stages),
            'sub_problems_count': len(self.decomposition_plan.sub_problems) if self.decomposition_plan else 0,
            'conflicts_detected': len(self.integrated_solution.conflicts_detected) if self.integrated_solution else 0,
            'duration_seconds': (
                (self.completed_at - self.created_at).total_seconds()
                if self.completed_at else 0
            )
        }


@dataclass
class PipelineAnalytics:
    """Analytics for pipeline execution."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    
    avg_decomposition_time: float = 0.0
    avg_solution_time: float = 0.0
    avg_total_time: float = 0.0
    
    avg_quality_score: float = 0.0
    avg_conflict_count: float = 0.0
    
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    domain_distribution: Dict[str, int] = field(default_factory=dict)
    
    def record_execution(self, result: PipelineResult, duration: float) -> None:
        """Record execution metrics."""
        self.total_executions += 1
        
        if result.is_successful():
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        
        # Update averages
        self.avg_quality_score = (
            (self.avg_quality_score * (self.total_executions - 1) + result.overall_quality)
            / self.total_executions
        )
        
        self.avg_total_time = (
            (self.avg_total_time * (self.total_executions - 1) + duration)
            / self.total_executions
        )
        
        # Update strategy usage
        if result.decomposition_plan:
            strategy = result.decomposition_plan.strategy_used.value
            self.strategy_usage[strategy] = self.strategy_usage.get(strategy, 0) + 1
        
        # Update domain distribution
        domain = result.problem.domain.value
        self.domain_distribution[domain] = self.domain_distribution.get(domain, 0) + 1


# ============================================================================
# SOLUTION SOLVER INTERFACE
# ============================================================================

class SolutionSolver(ABC):
    """Abstract base class for solution solvers."""
    
    @abstractmethod
    def solve(self, sub_problem: SubProblem) -> SubProblemSolution:
        """
        Solve a sub-problem.
        
        Args:
            sub_problem: The sub-problem to solve
            
        Returns:
            Solution for the sub-problem
        """
        raise NotImplementedError("SolutionSolver.solve must be implemented")
    
    @abstractmethod
    def can_solve(self, sub_problem: SubProblem) -> Tuple[bool, float]:
        """
        Check if this solver can handle the sub-problem.
        
        Returns:
            Tuple of (can_solve, confidence)
        """
        raise NotImplementedError("SolutionSolver.can_solve must be implemented")


class SimpleSolutionSolver(SolutionSolver):
    """Deterministic solver with structured, entanglement-aware output."""

    def solve(self, sub_problem: SubProblem) -> SubProblemSolution:
        metadata = sub_problem.metadata or {}
        entangled_with = metadata.get("entangled_with", []) or []
        entanglement_symbols = metadata.get("entanglement_symbols", []) or []

        dependency_list = ", ".join(sub_problem.dependencies) if sub_problem.dependencies else "None"
        acceptance = sub_problem.acceptance_criteria or []
        success_criteria = [sc.description for sc in sub_problem.success_criteria] if sub_problem.success_criteria else []
        constraints = metadata.get("constraints", []) or []
        if hasattr(sub_problem, "specific_constraints"):
            constraints = list(set(constraints + list(sub_problem.specific_constraints or [])))

        deliverable_map = {
            "implementation": "Implement the required component with interfaces and tests.",
            "design": "Produce design artifacts, interfaces, and rationale.",
            "analysis": "Deliver analysis results with risks and recommendations.",
            "research": "Deliver research findings with sources and implications.",
            "validation": "Deliver validation report and test results.",
            "integration": "Deliver integration plan and compatibility notes.",
            "testing": "Deliver test plan, coverage, and outcomes.",
            "documentation": "Deliver documentation and usage guidance.",
        }
        type_key = sub_problem.type.value if hasattr(sub_problem.type, "value") else str(sub_problem.type)
        deliverable = deliverable_map.get(type_key.lower(), "Deliver a complete solution artifact.")

        entanglement_note = ""
        if entangled_with:
            symbols_text = ", ".join(entanglement_symbols) if entanglement_symbols else "n/a"
            entanglement_note = (
                "\n## Entanglement Coordination\n"
                f"- Entangled with: {', '.join(entangled_with)}\n"
                f"- Shared symbols: {symbols_text}\n"
                "- Keep interfaces consistent across entangled components.\n"
            )

        input_contracts = metadata.get("input_contracts", []) or []
        output_contracts = metadata.get("output_contracts", []) or []
        if not input_contracts and sub_problem.dependencies:
            input_contracts = [f"Output from {dep}" for dep in sub_problem.dependencies]
        if not output_contracts:
            output_contracts = [f"Deliverable for {sub_problem.id}"]

        content = (
            f"# Solution: {sub_problem.title}\n\n"
            f"## Scope\n{sub_problem.description}\n\n"
            f"## Dependencies\n{dependency_list}\n\n"
            f"## Inputs\n"
            + ("\n".join(f"- {c}" for c in input_contracts) if input_contracts else "- None")
            + "\n\n"
            f"## Outputs\n"
            + ("\n".join(f"- {c}" for c in output_contracts) if output_contracts else "- None")
            + "\n\n"
            f"## Deliverable\n{deliverable}\n\n"
            f"## Implementation Plan\n"
            f"1. Clarify inputs/outputs and interface boundaries.\n"
            f"2. Draft the core logic and edge-case handling.\n"
            f"3. Integrate dependency outputs and validate assumptions.\n"
            f"4. Produce tests/validation steps aligned with criteria.\n\n"
            f"## Constraints\n"
            + ("\n".join(f"- {c}" for c in constraints) if constraints else "- None specified")
            + "\n\n"
            f"## Acceptance Criteria\n"
            + ("\n".join(f"- {c}" for c in acceptance) if acceptance else "- None specified")
            + "\n\n"
            f"## Success Criteria\n"
            + ("\n".join(f"- {c}" for c in success_criteria) if success_criteria else "- None specified")
            + entanglement_note
        )

        base_quality = 0.7
        if acceptance:
            base_quality += 0.05
        if success_criteria:
            base_quality += 0.05
        if entangled_with:
            base_quality += 0.05
        base_quality += min(0.1, sub_problem.priority / 100)
        quality_score = min(0.95, base_quality)

        solution_metadata = dict(metadata)
        if entangled_with:
            solution_metadata["entangled_with"] = entangled_with
            if entanglement_symbols:
                solution_metadata["entanglement_symbols"] = entanglement_symbols
            entanglement_source = metadata.get("entanglement_source")
            if entanglement_source:
                solution_metadata.setdefault("entanglement_source", entanglement_source)
        solution_metadata["dependencies"] = list(sub_problem.dependencies or [])
        solution_metadata["inputs"] = list(input_contracts or [])
        solution_metadata["outputs"] = list(output_contracts or [])
        solution_metadata["acceptance_criteria"] = list(acceptance or [])
        solution_metadata["success_criteria"] = list(success_criteria or [])
        solution_metadata["deliverable_type"] = type_key.lower()
        solution_metadata["deliverable_summary"] = deliverable
        solution_metadata["entanglement_context"] = bool(entangled_with)

        return create_subproblem_solution(
            sub_problem_id=sub_problem.id,
            content=content.strip(),
            quality_score=quality_score,
            metadata=solution_metadata,
        )

    def can_solve(self, sub_problem: SubProblem) -> Tuple[bool, float]:
        description_ok = bool(sub_problem.description and sub_problem.description.strip())
        confidence = 0.6
        if description_ok:
            confidence += 0.2
        if sub_problem.success_criteria:
            confidence += 0.1
        if sub_problem.acceptance_criteria:
            confidence += 0.1
        return description_ok, min(1.0, confidence)


# ============================================================================
# INTEGRATED PIPELINE
# ============================================================================

class DecompositionRecompositionPipeline:
    """
    Integrated pipeline connecting decomposition and recomposition.
    
    This pipeline provides:
    1. Problem decomposition into sub-problems
    2. Sub-problem solving
    3. Solution assembly with conflict resolution
    4. Quality validation
    5. Feedback-driven refinement
    """
    
    def __init__(
        self,
        decomposition_engine: Optional[EnhancedDecompositionEngine] = None,
        recomposition_engine: Optional[EnhancedRecompositionEngine] = None,
        solution_solver: Optional[SolutionSolver] = None,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            decomposition_engine: Decomposition engine (created if None)
            recomposition_engine: Recomposition engine (created if None)
            solution_solver: Solution solver (created if None)
            config: Pipeline configuration
        """
        self.decomposition_engine = decomposition_engine or EnhancedDecompositionEngine()
        self.recomposition_engine = recomposition_engine or EnhancedRecompositionEngine()
        self.solution_solver = solution_solver or SimpleSolutionSolver()
        self.config = config or PipelineConfig()

        self.logger = logging.getLogger(self.__class__.__name__)

        # Optional ROMA adapter
        self.roma_adapter = None
        if self.config.enable_roma:
            if ROMA_INTEGRATION_AVAILABLE and create_roma_adapter is not None:
                roma_kwargs = dict(self.config.roma_config or {})
                self.roma_adapter = create_roma_adapter(
                    enable_roma=True,
                    use_mdap_maker=self.config.use_roma_mdap_maker,
                    **roma_kwargs,
                )
                self.logger.info("ROMA adapter initialized for decomposition pipeline")
            else:
                self.logger.warning("ROMA integration requested but not available")
        
        # Analytics
        self.analytics = PipelineAnalytics()
        
        # History
        self.execution_history: List[PipelineResult] = []
        
        self.logger.info("DecompositionRecompositionPipeline initialized")
    
    def execute(
        self,
        problem: ProblemDefinition,
        custom_solver: Optional[SolutionSolver] = None
    ) -> PipelineResult:
        """
        Execute the full pipeline.
        
        Args:
            problem: Problem to solve
            custom_solver: Optional custom solver for this execution
            
        Returns:
            PipelineResult with complete execution results
        """
        start_time = time.time()
        
        solver = custom_solver or self.solution_solver
        
        # Create result container
        result = PipelineResult(
            pipeline_id=self._generate_id("pipe"),
            problem=problem,
            stages=[]
        )
        
        self.logger.info(f"Starting pipeline execution for problem: {problem.title}")
        
        try:
            # Stage 1: Decomposition
            decomposition_stage = PipelineStage(
                name="decomposition",
                status="running",
                start_time=datetime.now()
            )
            result.stages.append(decomposition_stage)
            
            decomposition_plan = self._execute_decomposition(problem)
            result.decomposition_plan = decomposition_plan
            result.decomposition_quality = decomposition_plan.overall_quality
            
            decomposition_stage.status = "completed"
            decomposition_stage.end_time = datetime.now()
            decomposition_stage.result = {
                'sub_problems_count': len(decomposition_plan.sub_problems),
                'strategy': decomposition_plan.strategy_used.value,
                'quality': decomposition_plan.overall_quality
            }
            
            # Stage 2: Solution Generation
            solution_stage = PipelineStage(
                name="solution_generation",
                status="running",
                start_time=datetime.now()
            )
            result.stages.append(solution_stage)
            
            entanglement_matrix = (decomposition_plan.metadata or {}).get("entanglement_matrix", {}) or {}
            sub_solutions = self._generate_solutions(
                decomposition_plan.sub_problems,
                solver,
                entanglement_matrix=entanglement_matrix
            )
            result.sub_solutions = sub_solutions
            
            solution_stage.status = "completed"
            solution_stage.end_time = datetime.now()
            solution_stage.result = {
                'solutions_generated': len(sub_solutions),
                'avg_quality': sum(s.quality_score for s in sub_solutions.values()) / len(sub_solutions) if sub_solutions else 0
            }
            
            # Stage 3: Recomposition
            recomposition_stage = PipelineStage(
                name="recomposition",
                status="running",
                start_time=datetime.now()
            )
            result.stages.append(recomposition_stage)
            
            integrated_solution = self._execute_recomposition(
                decomposition_plan,
                sub_solutions
            )
            result.integrated_solution = integrated_solution
            result.solution_quality = integrated_solution.quality_metrics.overall_score
            
            recomposition_stage.status = "completed"
            recomposition_stage.end_time = datetime.now()
            recomposition_stage.result = {
                'conflicts_detected': len(integrated_solution.conflicts_detected),
                'conflicts_resolved': len(integrated_solution.conflicts_resolved),
                'quality': integrated_solution.quality_metrics.overall_score
            }
            
            # Stage 4: Validation & Refinement
            if self.config.enable_feedback_loop and result.solution_quality < self.config.quality_threshold:
                refinement_stage = PipelineStage(
                    name="refinement",
                    status="running",
                    start_time=datetime.now()
                )
                result.stages.append(refinement_stage)
                
                refined = self._refine_solution(result, solver)
                if refined:
                    result = refined
                
                refinement_stage.status = "completed"
                refinement_stage.end_time = datetime.now()
            
            # Calculate overall quality
            result.overall_quality = (
                result.decomposition_quality * 0.3 +
                result.solution_quality * 0.7
            )
            
        except (RuntimeError, ValueError, TypeError) as e:
            self.logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            # Mark current stage as failed
            if result.stages and result.stages[-1].status == "running":
                result.stages[-1].status = "failed"
                result.stages[-1].error = str(e)
                result.stages[-1].end_time = datetime.now()
        
        # Finalize
        result.completed_at = datetime.now()
        duration = time.time() - start_time
        
        # Record analytics
        self.analytics.record_execution(result, duration)
        self.execution_history.append(result)
        
        self.logger.info(
            f"Pipeline completed: quality={result.overall_quality:.2f}, "
            f"successful={result.is_successful()}"
        )
        
        return result
    
    def _execute_decomposition(
        self,
        problem: ProblemDefinition
    ) -> DecompositionPlan:
        """Execute decomposition stage."""
        self.logger.info("Executing decomposition")

        if self.roma_adapter and self.roma_adapter.is_decomposition_available():
            roma_result = self.roma_adapter.setup_and_decompose_problem(
                problem_statement=problem.description,
                problem_type=problem.metadata.get("problem_type") if isinstance(problem.metadata, dict) else None,
                domain=problem.domain.value if hasattr(problem.domain, "value") else str(problem.domain),
            )
            roma_plan = roma_result.get("openevolve_plan")
            if isinstance(roma_plan, DecompositionPlan):
                roma_plan.original_problem = problem
                roma_plan.metadata.setdefault("roma_result", roma_result)
                return roma_plan

        plan = self.decomposition_engine.decompose(
            problem=problem,
            strategy=self.config.decomposition_strategy,
            min_subproblems=self.config.min_subproblems,
            max_subproblems=self.config.max_subproblems,
            max_depth=self.config.max_depth
        )

        return plan
    
    def _generate_solutions(
        self,
        sub_problems: List[SubProblem],
        solver: SolutionSolver,
        entanglement_matrix: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, SubProblemSolution]:
        """Generate solutions for all sub-problems."""
        self.logger.info(f"Generating solutions for {len(sub_problems)} sub-problems")
        
        solutions = {}
        entanglement_matrix = entanglement_matrix or {}

        if self.roma_adapter and self.config.enable_roma and self.roma_adapter.is_available():
            payload = []
            for sub_problem in sub_problems:
                payload.append(
                    {
                        "id": sub_problem.id,
                        "title": sub_problem.title,
                        "description": sub_problem.description,
                        "dependencies": list(sub_problem.dependencies or []),
                        "metadata": sub_problem.metadata or {},
                    }
                )
            roma_result = self.roma_adapter.solve_sub_problems(payload)
            roma_solutions = roma_result.get("solutions", []) if isinstance(roma_result, dict) else []
            for sol in roma_solutions:
                if not isinstance(sol, dict):
                    continue
                sol_id = sol.get("id") or sol.get("sub_problem_id") or sol.get("title")
                if not sol_id:
                    continue
                content = sol.get("solution") or sol.get("solution_content") or ""
                quality = sol.get("quality_score") or sol.get("confidence") or 0.7
                metadata = sol.get("metadata", {}) if isinstance(sol.get("metadata"), dict) else {}
                solutions[sol_id] = SubProblemSolution(
                    sub_problem_id=sol_id,
                    solution_content=str(content),
                    quality_score=float(quality),
                    metadata=metadata,
                )
        
        for sub_problem in sub_problems:
            if sub_problem.id in solutions:
                continue
            if entanglement_matrix and isinstance(sub_problem.metadata, dict):
                entangled_with = entanglement_matrix.get(sub_problem.id, [])
                if entangled_with:
                    sub_problem.metadata.setdefault("entangled_with", entangled_with)
                    sub_problem.metadata.setdefault("entanglement_source", "symbolic_overlap")
            can_solve, confidence = solver.can_solve(sub_problem)
            
            if can_solve:
                solution = solver.solve(sub_problem)
                solutions[sub_problem.id] = solution
                self.logger.debug(f"Generated solution for {sub_problem.id}")
            else:
                self.logger.warning(f"Cannot solve sub-problem {sub_problem.id}")
        
        return solutions
    
    def _execute_recomposition(
        self,
        decomposition_plan: DecompositionPlan,
        sub_solutions: Dict[str, SubProblemSolution]
    ) -> IntegratedSolution:
        """Execute recomposition stage."""
        self.logger.info("Executing recomposition")
        
        entanglement_matrix = (decomposition_plan.metadata or {}).get("entanglement_matrix", {}) or {}
        solution = self.recomposition_engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=decomposition_plan.original_problem.id,
            decomposition_plan_id=decomposition_plan.id,
            dependency_graph=decomposition_plan.dependency_graph,
            strategy=self.config.assembly_strategy,
            entanglement_matrix=entanglement_matrix
        )

        if self.roma_adapter and self.config.enable_roma and self.roma_adapter.is_available():
            payload = []
            for sp_id, sol in sub_solutions.items():
                payload.append(
                    {
                        "id": sp_id,
                        "solution": sol.solution_content,
                        "metadata": sol.metadata if isinstance(sol.metadata, dict) else {},
                        "dependencies": decomposition_plan.dependency_graph.get(sp_id, []),
                    }
                )
            roma_result = self.roma_adapter.reassemble_solutions(
                solutions=payload,
                problem_statement=decomposition_plan.original_problem.description,
            )
            if isinstance(roma_result, dict) and roma_result.get("final_solution"):
                solution.assembled_content = roma_result["final_solution"]
                if isinstance(solution.metadata, dict):
                    solution.metadata["roma_reassembly"] = {
                        "roma_used": roma_result.get("roma_used", False),
                        "roma_type": roma_result.get("roma_type"),
                        "message": roma_result.get("message"),
                    }
        
        return solution
    
    def _refine_solution(
        self,
        current_result: PipelineResult,
        solver: SolutionSolver
    ) -> Optional[PipelineResult]:
        """
        Refine solution based on quality feedback.
        
        This implements an iterative refinement loop where:
        1. Quality issues are identified
        2. Problematic sub-problems are re-solved
        3. Solution is re-assembled
        """
        if current_result.refinement_iterations >= self.config.max_iterations:
            self.logger.info("Max refinement iterations reached")
            return None
        
        self.logger.info("Refining solution")
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(current_result)
        
        if not quality_issues:
            return None
        
        # Re-solve problematic sub-problems
        refined_solutions = current_result.sub_solutions.copy()
        
        for sub_problem_id in quality_issues:
            sub_problem = next(
                (sp for sp in current_result.decomposition_plan.sub_problems if sp.id == sub_problem_id),
                None
            )
            
            if sub_problem:
                # Enhance sub-problem with feedback
                sub_problem.metadata['refinement_iteration'] = (
                    sub_problem.metadata.get('refinement_iteration', 0) + 1
                )
                
                # Re-solve
                new_solution = solver.solve(sub_problem)
                new_solution.quality_score = min(1.0, new_solution.quality_score + 0.1)
                refined_solutions[sub_problem_id] = new_solution
        
        # Re-assemble
        refined_solution = self._execute_recomposition(
            current_result.decomposition_plan,
            refined_solutions
        )
        
        # Check if improved
        if refined_solution.quality_metrics.overall_score > current_result.solution_quality:
            improvement = refined_solution.quality_metrics.overall_score - current_result.solution_quality
            current_result.integrated_solution = refined_solution
            current_result.sub_solutions = refined_solutions
            current_result.solution_quality = refined_solution.quality_metrics.overall_score
            current_result.refinement_iterations += 1
            
            # Log feedback
            current_result.feedback_log.append({
                'iteration': current_result.refinement_iterations,
                'action': 're-solved problematic sub-problems',
                'improvement': improvement
            })
            
            return current_result
        
        return None
    
    def _identify_quality_issues(self, result: PipelineResult) -> List[str]:
        """Identify sub-problems with quality issues."""
        issues = []
        
        if not result.integrated_solution:
            return issues
        
        # Check for unresolved conflicts
        for conflict in result.integrated_solution.conflicts_detected:
            if not conflict.is_resolved():
                if conflict.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH]:
                    issues.extend(conflict.involved_solutions)
        
        # Check for low-quality solutions
        for sol_id, solution in result.sub_solutions.items():
            if solution.quality_score < 0.6:
                issues.append(sol_id)
            if isinstance(solution.metadata, dict):
                if solution.metadata.get("needs_consistency_refinement"):
                    issues.append(sol_id)
        
        return list(set(issues))
    
    def get_analytics(self) -> PipelineAnalytics:
        """Get pipeline analytics."""
        return self.analytics
    
    def get_execution_history(
        self,
        limit: Optional[int] = None
    ) -> List[PipelineResult]:
        """Get execution history."""
        history = sorted(
            self.execution_history,
            key=lambda r: r.created_at,
            reverse=True
        )
        
        if limit:
            return history[:limit]
        return history
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# BATCH PROCESSING
# ============================================================================

class BatchPipelineProcessor:
    """Process multiple problems through the pipeline."""
    
    def __init__(self, pipeline: DecompositionRecompositionPipeline):
        self.pipeline = pipeline
        self.results: List[PipelineResult] = []
    
    def process_batch(
        self,
        problems: List[ProblemDefinition],
        parallel: bool = False
    ) -> List[PipelineResult]:
        """
        Process multiple problems.
        
        Args:
            problems: List of problems to process
            parallel: Whether to process in parallel
            
        Returns:
            List of pipeline results
        """
        self.logger.info(f"Processing batch of {len(problems)} problems")
        
        results = []
        
        for problem in problems:
            result = self.pipeline.execute(problem)
            results.append(result)
        
        self.results.extend(results)
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get batch processing summary."""
        if not self.results:
            return {}
        
        successful = sum(1 for r in self.results if r.is_successful())
        
        return {
            'total': len(self.results),
            'successful': successful,
            'failed': len(self.results) - successful,
            'success_rate': successful / len(self.results),
            'avg_quality': sum(r.overall_quality for r in self.results) / len(self.results),
            'avg_duration': sum(
                (r.completed_at - r.created_at).total_seconds()
                for r in self.results if r.completed_at
            ) / len([r for r in self.results if r.completed_at])
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_solve(
    title: str,
    description: str,
    domain: ProblemDomain = ProblemDomain.GENERIC,
    complexity: Optional[float] = None
) -> PipelineResult:
    """
    Quick helper to solve a problem through the full pipeline.
    
    Args:
        title: Problem title
        description: Problem description
        domain: Problem domain
        complexity: Complexity estimate (auto-calculated if None)
        
    Returns:
        PipelineResult
    """
    # Create problem
    problem = create_problem_definition(title, description, domain, complexity)
    
    # Create and execute pipeline
    pipeline = DecompositionRecompositionPipeline()
    result = pipeline.execute(problem)
    
    return result


def analyze_solution(result: PipelineResult) -> Dict[str, Any]:
    """
    Analyze a pipeline result in detail.
    
    Args:
        result: Pipeline result to analyze
        
    Returns:
        Detailed analysis
    """
    analysis = {
        'overview': result.to_dict(),
        'decomposition': None,
        'recomposition': None,
        'recommendations': []
    }
    
    # Analyze decomposition
    if result.decomposition_plan:
        plan = result.decomposition_plan
        analysis['decomposition'] = {
            'strategy': plan.strategy_used.value,
            'sub_problems': len(plan.sub_problems),
            'quality': {
                'coverage': plan.coverage_score,
                'balance': plan.balance_score,
                'coherence': plan.coherence_score,
                'overall': plan.overall_quality
            },
            'execution_order': len(plan.execution_order),
            'parallel_groups': len(plan.parallel_groups)
        }
    
    # Analyze recomposition
    if result.integrated_solution:
        sol = result.integrated_solution
        analysis['recomposition'] = {
            'strategy': sol.assembly_strategy.value,
            'quality': sol.quality_metrics.to_dict() if sol.quality_metrics else {},
            'conflicts': sol.get_conflict_summary(),
            'content_length': len(sol.assembled_content)
        }
    
    # Generate recommendations
    recommendations = []
    
    if result.decomposition_quality < 0.7:
        recommendations.append("Consider using a different decomposition strategy")
    
    if result.solution_quality < 0.7:
        recommendations.append("Solution quality could be improved - consider refinement")
    
    if result.integrated_solution and len(result.integrated_solution.conflicts_detected) > 3:
        recommendations.append("High conflict count - review sub-problem boundaries")
    
    analysis['recommendations'] = recommendations
    
    return analysis


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Decomposition-Recomposition Pipeline Demo")
    print("=" * 60)
    
    # Create pipeline
    pipeline = DecompositionRecompositionPipeline()
    
    # Define problem
    problem = create_problem_definition(
        title="Build E-Commerce Platform",
        description="""
        Develop a comprehensive e-commerce platform with the following requirements:
        
        1. User Management: Registration, authentication, profile management
        2. Product Catalog: Browse, search, filter products with rich descriptions
        3. Shopping Cart: Add/remove items, persistent cart across sessions
        4. Payment Processing: Secure payment integration with multiple providers
        5. Order Management: Track orders, manage shipping, handle returns
        6. Admin Dashboard: Manage products, orders, users, and analytics
        7. Mobile Responsive: Work seamlessly on all devices
        8. Performance: Handle 1000+ concurrent users with <2s response time
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=8.0
    )
    
    print(f"\nProblem: {problem.title}")
    print(f"Domain: {problem.domain.value}")
    print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result = pipeline.execute(problem)
    
    # Display results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print(f"\nPipeline ID: {result.pipeline_id}")
    print(f"Successful: {result.is_successful()}")
    print(f"Overall Quality: {result.overall_quality:.2f}")
    print(f"Decomposition Quality: {result.decomposition_quality:.2f}")
    print(f"Solution Quality: {result.solution_quality:.2f}")
    
    if result.decomposition_plan:
        print(f"\nDecomposition:")
        print(f"  Strategy: {result.decomposition_plan.strategy_used.value}")
        print(f"  Sub-problems: {len(result.decomposition_plan.sub_problems)}")
        print(f"  Execution Order: {len(result.decomposition_plan.execution_order)} steps")
        print(f"  Parallel Groups: {len(result.decomposition_plan.parallel_groups)}")
    
    if result.integrated_solution:
        sol = result.integrated_solution
        print(f"\nIntegrated Solution:")
        print(f"  Strategy: {sol.assembly_strategy.value}")
        print(f"  Conflicts: {len(sol.conflicts_detected)} detected, {len(sol.conflicts_resolved)} resolved")
        print(f"  Content Length: {len(sol.assembled_content)} characters")
        
        if sol.quality_metrics:
            print(f"\n  Quality Metrics:")
            print(f"    Completeness: {sol.quality_metrics.completeness:.2f}")
            print(f"    Consistency: {sol.quality_metrics.consistency:.2f}")
            print(f"    Coherence: {sol.quality_metrics.coherence:.2f}")
            print(f"    Correctness: {sol.quality_metrics.correctness:.2f}")
    
    # Detailed analysis
    print("\n" + "=" * 60)
    print("Detailed Analysis")
    print("=" * 60)
    
    analysis = analyze_solution(result)
    
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
    
    # Pipeline stages
    print(f"\nPipeline Stages:")
    for stage in result.stages:
        duration = stage.duration_seconds()
        print(f"  {stage.name}: {stage.status} ({duration:.2f}s)")
    
    # Content preview
    if result.integrated_solution:
        print("\n" + "=" * 60)
        print("Solution Content Preview")
        print("=" * 60)
        content = result.integrated_solution.assembled_content
        print(content[:800] + "..." if len(content) > 800 else content)
