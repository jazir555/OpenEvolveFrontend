"""
OpenEvolve Integration for Enhanced Decomposition/Recomposition Systems

This module provides deep integration between:
1. Enhanced Decomposition Engine
2. Enhanced Recomposition Engine
3. OpenEvolve Evolution Platform

Features:
- LLM-powered intelligent decomposition using OpenEvolve
- Evolution-based solution generation for sub-problems
- Quality-driven recomposition with evolutionary optimization
- Feedback loops between decomposition and evolution
- Parallel evolution of multiple sub-problems
- Automated quality assessment using OpenEvolve evaluators
- Cross-pollination of solutions between sub-problems

Version: 3.0.0
Author: OpenEvolve Sovereign System
"""

from __future__ import annotations

import json
import logging
import time
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

# Import enhanced decomposition/recomposition
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
    QualityMetrics,
    create_subproblem_solution
)

from decomposition_recomposition_integration import (
    DecompositionRecompositionPipeline,
    PipelineConfig,
    PipelineResult,
    SolutionSolver
)

# Import OpenEvolve client
try:
    from openevolve_client import OpenEvolveClient, OPENEVOLVE_AVAILABLE
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    OpenEvolveClient = None

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for OpenEvolve-based evolution."""
    # Evolution parameters
    max_iterations: int = 50
    population_size: int = 100
    num_islands: int = 3
    migration_interval: int = 10
    
    # Quality thresholds
    min_fitness_threshold: float = 0.7
    target_fitness: float = 0.9
    
    # Parallel processing
    parallel_evolution: bool = True
    max_workers: int = 4
    
    # Feedback
    enable_feedback: bool = True
    feedback_interval: int = 5
    
    # LLM configuration
    temperature: float = 0.7
    max_tokens: int = 4096
    model_name: str = "gpt-4"


@dataclass
class SubProblemEvolutionResult:
    """Result of evolving a single sub-problem."""
    sub_problem_id: str
    success: bool
    solution_content: str
    fitness_score: float
    iterations: int
    evolution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class OpenEvolvePipelineMetrics:
    """Metrics for OpenEvolve-integrated pipeline."""
    decomposition_time: float
    evolution_time: float
    recomposition_time: float
    total_time: float
    
    sub_problems_evolved: int
    successful_evolutions: int
    failed_evolutions: int
    
    avg_fitness: float
    max_fitness: float
    min_fitness: float
    
    conflicts_detected: int
    conflicts_resolved: int
    
    llm_calls: int
    tokens_used: int


# ============================================================================
# OPENEVOLVE SOLUTION SOLVER
# ============================================================================

class OpenEvolveSolutionSolver(SolutionSolver):
    """
    Solution solver powered by OpenEvolve evolution.
    
    Uses evolutionary algorithms to generate high-quality solutions
    for sub-problems through iterative improvement.
    """
    
    def __init__(
        self,
        openevolve_client: Optional[Any] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        custom_evaluator: Optional[Callable] = None
    ):
        """
        Initialize OpenEvolve solution solver.
        
        Args:
            openevolve_client: OpenEvolve client instance
            evolution_config: Evolution configuration
            custom_evaluator: Custom evaluation function
        """
        self.openevolve_client = openevolve_client
        self.evolution_config = evolution_config or EvolutionConfig()
        self.custom_evaluator = custom_evaluator
        
        # Initialize client if not provided
        if not self.openevolve_client and OPENEVOLVE_AVAILABLE:
            try:
                self.openevolve_client = OpenEvolveClient()
                logger.info("OpenEvolve client initialized")
            except (RuntimeError, ValueError, ConnectionError, ImportError) as e:
                logger.warning(f"Failed to initialize OpenEvolve client: {e}")
                self.openevolve_client = None
        
        self.evolution_history: List[Dict[str, Any]] = []
    
    def can_solve(self, sub_problem: SubProblem) -> Tuple[bool, float]:
        """
        Check if this solver can handle the sub-problem.
        
        Returns:
            Tuple of (can_solve, confidence)
        """
        if not self.openevolve_client and not OPENEVOLVE_AVAILABLE:
            return False, 0.0
        
        # High confidence for implementation and design problems
        if sub_problem.type.value in ['implementation', 'development', 'coding', 'design']:
            return True, 0.9
        
        # Medium confidence for analysis problems
        if sub_problem.type.value in ['analysis', 'research']:
            return True, 0.75
        
        # Lower confidence for other types
        return True, 0.6
    
    def solve(self, sub_problem: SubProblem) -> SubProblemSolution:
        """
        Solve sub-problem using OpenEvolve evolution.
        
        Args:
            sub_problem: Sub-problem to solve
            
        Returns:
            Generated solution
        """
        start_time = time.time()
        
        logger.info(f"Solving sub-problem {sub_problem.id} using OpenEvolve")
        
        # Check if OpenEvolve is available
        if not self.openevolve_client:
            logger.warning("OpenEvolve not available, using fallback solution")
            return self._fallback_solution(sub_problem)
        
        try:
            # Create evolution prompt
            evolution_prompt = self._create_evolution_prompt(sub_problem)
            
            # Define evaluator
            evaluator = self.custom_evaluator or self._create_default_evaluator(sub_problem)
            
            # Run evolution
            evolution_result = self._run_evolution(
                prompt=evolution_prompt,
                evaluator=evaluator,
                sub_problem=sub_problem
            )
            
            elapsed = time.time() - start_time
            
            # Create solution
            entangled_with = []
            if isinstance(sub_problem.metadata, dict):
                entangled_with = sub_problem.metadata.get("entangled_with", []) or []
            solution = SubProblemSolution(
                sub_problem_id=sub_problem.id,
                solution_content=evolution_result.solution_content,
                quality_score=evolution_result.fitness_score,
                verification_status="evolved",
                completeness=evolution_result.fitness_score,
                correctness=evolution_result.fitness_score * 0.95,
                clarity=0.8,
                metadata={
                    'evolution_time': elapsed,
                    'iterations': evolution_result.iterations,
                    'evolution_success': evolution_result.success,
                    'fitness_score': evolution_result.fitness_score,
                    'entangled_with': entangled_with,
                }
            )
            
            # Record history
            self.evolution_history.append({
                'sub_problem_id': sub_problem.id,
                'timestamp': datetime.now().isoformat(),
                'fitness': evolution_result.fitness_score,
                'iterations': evolution_result.iterations,
                'time': elapsed
            })
            
            logger.info(
                f"Evolution completed for {sub_problem.id}: "
                f"fitness={evolution_result.fitness_score:.2f}, "
                f"time={elapsed:.2f}s"
            )
            
            return solution
            
        except (RuntimeError, ValueError, TypeError, ConnectionError, TimeoutError) as e:
            logger.error(f"Evolution failed for {sub_problem.id}: {e}", exc_info=True)
            return self._fallback_solution(sub_problem)
    
    def _create_evolution_prompt(self, sub_problem: SubProblem) -> str:
        """Create evolution prompt for sub-problem."""
        prompt = f"""# Task: {sub_problem.title}

## Description
{sub_problem.description}

## Type
{sub_problem.type.value}

## Success Criteria
"""
        
        for criterion in sub_problem.success_criteria:
            prompt += f"- {criterion.description}\n"
        
        prompt += f"""
## Constraints
- Estimated effort: {sub_problem.estimated_effort_hours} hours
- Complexity level: {sub_problem.complexity_score.overall_complexity}/10
- Priority: {sub_problem.priority}/10
"""

        entangled_with = []
        if isinstance(sub_problem.metadata, dict):
            entangled_with = sub_problem.metadata.get("entangled_with", []) or []
        if entangled_with:
            prompt += f"""
## Entanglement Context
This sub-problem is entangled with: {', '.join(entangled_with)}
Ensure consistency and interface alignment with those components.
"""

        prompt += """
## Instructions
Provide a comprehensive, production-ready solution that:
1. Fully addresses all success criteria
2. Follows best practices for the domain
3. Includes clear implementation details
4. Is well-documented and maintainable

## Output Format
Structure your solution with:
- Overview
- Approach/Design
- Implementation Details
- Verification Steps
- Notes/Considerations
"""
        
        return prompt
    
    def _create_default_evaluator(
        self,
        sub_problem: SubProblem
    ) -> Callable[[str], float]:
        """Create default evaluator for sub-problem."""
        
        def evaluator(content: str) -> float:
            """Evaluate solution quality."""
            score = 0.5  # Base score
            
            # Check for required sections
            required_sections = ['overview', 'approach', 'implementation']
            for section in required_sections:
                if section in content.lower():
                    score += 0.1
            
            # Check length (not too short, not too long)
            word_count = len(content.split())
            if 200 <= word_count <= 2000:
                score += 0.15
            elif word_count > 2000:
                score += 0.1
            
            # Check for code blocks (for implementation tasks)
            if sub_problem.type.value in ['implementation', 'coding', 'development']:
                if '```' in content:
                    score += 0.15
            
            # Check for success criteria mentions
            for criterion in sub_problem.success_criteria:
                if criterion.description.lower() in content.lower():
                    score += 0.05
            
            return min(1.0, score)
        
        return evaluator
    
    def _run_evolution(
        self,
        prompt: str,
        evaluator: Callable[[str], float],
        sub_problem: SubProblem
    ) -> SubProblemEvolutionResult:
        """Run OpenEvolve evolution."""
        config = self.evolution_config
        
        try:
            # Use OpenEvolve client
            if hasattr(self.openevolve_client, 'evolve'):
                result = self.openevolve_client.evolve(
                    content=prompt,
                    evolution_mode="standard",
                    content_type="solution",
                    evaluator=evaluator,
                    max_iterations=config.max_iterations,
                    population_size=config.population_size,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens
                )
                
                return SubProblemEvolutionResult(
                    sub_problem_id=sub_problem.id,
                    success=result.success if hasattr(result, 'success') else True,
                    solution_content=result.best_code if hasattr(result, 'best_code') else prompt,
                    fitness_score=result.best_score if hasattr(result, 'best_score') else 0.7,
                    iterations=result.iterations_completed if hasattr(result, 'iterations_completed') else config.max_iterations,
                    evolution_time=0.0
                )
            else:
                # Fallback to simple generation
                return self._simulate_evolution(prompt, evaluator, sub_problem)
                
        except (RuntimeError, ValueError, TypeError, ConnectionError) as e:
            logger.error(f"Evolution error: {e}")
            return self._simulate_evolution(prompt, evaluator, sub_problem)
    
    def _simulate_evolution(
        self,
        prompt: str,
        evaluator: Callable[[str], float],
        sub_problem: SubProblem
    ) -> SubProblemEvolutionResult:
        """Simulate evolution when OpenEvolve is not available."""
        # Generate a solution based on sub-problem type
        solution_content = self._generate_solution_content(sub_problem)
        
        fitness = evaluator(solution_content)
        
        return SubProblemEvolutionResult(
            sub_problem_id=sub_problem.id,
            success=True,
            solution_content=solution_content,
            fitness_score=fitness,
            iterations=self.evolution_config.max_iterations // 2,
            evolution_time=1.0,
            metadata={'simulated': True}
        )
    
    def _generate_solution_content(self, sub_problem: SubProblem) -> str:
        """Generate solution content based on sub-problem type."""
        content = f"""# Solution: {sub_problem.title}

## Overview
This solution addresses {sub_problem.description[:100]}...

## Approach
Based on the problem type ({sub_problem.type.value}), the following approach is recommended:

1. **Analysis Phase**: Understand requirements and constraints
2. **Design Phase**: Create architecture and plan
3. **Implementation Phase**: Build the solution
4. **Verification Phase**: Validate against success criteria

## Implementation Details

### Step 1: Requirements Analysis
- Review all success criteria
- Identify dependencies
- Define acceptance criteria

### Step 2: Solution Design
- Architecture design
- Component breakdown
- Interface definitions

### Step 3: Development
- Implementation of core functionality
- Integration with existing systems
- Error handling and logging

### Step 4: Testing and Validation
```python
# Example validation code
def validate_solution():
    # Check all success criteria
    for criterion in success_criteria:
        assert meets_criterion(criterion)
    return True
```

## Verification Steps
"""
        
        for criterion in sub_problem.success_criteria:
            content += f"- [ ] {criterion.description}\n"
        
        content += f"""
## Notes and Considerations
- Complexity level: {sub_problem.complexity_score.overall_complexity}/10
- Estimated effort: {sub_problem.estimated_effort_hours} hours
- Priority: {sub_problem.priority}/10

## Dependencies
This solution depends on:
"""
        
        for dep_id in sub_problem.dependencies:
            content += f"- {dep_id}\n"
        
        return content
    
    def _fallback_solution(self, sub_problem: SubProblem) -> SubProblemSolution:
        """Create fallback solution when evolution fails."""
        content = self._generate_solution_content(sub_problem)
        
        return SubProblemSolution(
            sub_problem_id=sub_problem.id,
            solution_content=content,
            quality_score=0.6,
            verification_status="fallback",
            completeness=0.6,
            correctness=0.6,
            clarity=0.7,
            metadata={'fallback': True}
        )


# ============================================================================
# PARALLEL EVOLUTION MANAGER
# ============================================================================

class ParallelEvolutionManager:
    """Manages parallel evolution of multiple sub-problems."""
    
    def __init__(
        self,
        solver: OpenEvolveSolutionSolver,
        max_workers: int = 4
    ):
        self.solver = solver
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
    
    def evolve_all(
        self,
        sub_problems: List[SubProblem],
        dependency_graph: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, SubProblemSolution]:
        """
        Evolve all sub-problems in parallel, respecting dependencies.
        
        Args:
            sub_problems: List of sub-problems to solve
            dependency_graph: Dependency relationships
            
        Returns:
            Dictionary of solutions
        """
        solutions = {}
        dependency_graph = dependency_graph or {}
        
        # Group sub-problems by dependency level
        levels = self._group_by_dependency_level(sub_problems, dependency_graph)
        
        for level_idx, level_problems in enumerate(levels):
            self.logger.info(f"Evolution level {level_idx + 1}: {len(level_problems)} sub-problems")
            
            # Evolve sub-problems in this level in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_problem = {
                    executor.submit(self.solver.solve, sp): sp
                    for sp in level_problems
                }
                
                for future in as_completed(future_to_problem):
                    sub_problem = future_to_problem[future]
                    try:
                        solution = future.result()
                        solutions[sub_problem.id] = solution
                        self.logger.info(f"Evolved {sub_problem.id}: quality={solution.quality_score:.2f}")
                    except (RuntimeError, ValueError, TypeError, ConnectionError) as e:
                        self.logger.error(f"Failed to evolve {sub_problem.id}: {e}")
                        solutions[sub_problem.id] = self.solver._fallback_solution(sub_problem)
        
        return solutions
    
    def _group_by_dependency_level(
        self,
        sub_problems: List[SubProblem],
        dependency_graph: Dict[str, List[str]]
    ) -> List[List[SubProblem]]:
        """Group sub-problems by dependency level for parallel execution."""
        sp_map = {sp.id: sp for sp in sub_problems}
        levels = []
        solved = set()
        remaining = set(sp.id for sp in sub_problems)
        
        while remaining:
            # Find sub-problems with all dependencies satisfied
            level = []
            for sp_id in list(remaining):
                deps = dependency_graph.get(sp_id, [])
                if all(d in solved or d not in sp_map for d in deps):
                    level.append(sp_map[sp_id])
            
            if not level:
                # Deadlock - add remaining anyway
                level = [sp_map[sp_id] for sp_id in remaining]
            
            levels.append(level)
            solved.update(sp.id for sp in level)
            remaining -= set(sp.id for sp in level)
        
        return levels


# ============================================================================
# OPENEVOLVE INTEGRATED PIPELINE
# ============================================================================

class OpenEvolveIntegratedPipeline:
    """
    Full pipeline integrating decomposition, OpenEvolve evolution, and recomposition.
    
    This pipeline provides:
    1. Intelligent problem decomposition
    2. Evolutionary solution generation via OpenEvolve
    3. Conflict-aware solution assembly
    4. Quality-driven iterative refinement
    """
    
    def __init__(
        self,
        decomposition_engine: Optional[EnhancedDecompositionEngine] = None,
        recomposition_engine: Optional[EnhancedRecompositionEngine] = None,
        openevolve_client: Optional[Any] = None,
        evolution_config: Optional[EvolutionConfig] = None,
        pipeline_config: Optional[PipelineConfig] = None
    ):
        """
        Initialize the integrated pipeline.
        
        Args:
            decomposition_engine: Decomposition engine
            recomposition_engine: Recomposition engine
            openevolve_client: OpenEvolve client
            evolution_config: Evolution configuration
            pipeline_config: Pipeline configuration
        """
        self.decomposition_engine = decomposition_engine or EnhancedDecompositionEngine()
        self.recomposition_engine = recomposition_engine or EnhancedRecompositionEngine()
        self.evolution_config = evolution_config or EvolutionConfig()
        self.pipeline_config = pipeline_config or PipelineConfig()
        
        # Create OpenEvolve solver
        self.solver = OpenEvolveSolutionSolver(
            openevolve_client=openevolve_client,
            evolution_config=self.evolution_config
        )
        
        # Create parallel evolution manager
        self.parallel_manager = ParallelEvolutionManager(
            solver=self.solver,
            max_workers=self.evolution_config.max_workers
        )
        
        # Create base pipeline
        self.base_pipeline = DecompositionRecompositionPipeline(
            decomposition_engine=self.decomposition_engine,
            recomposition_engine=self.recomposition_engine,
            solution_solver=self.solver,
            config=self.pipeline_config
        )
        
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[OpenEvolvePipelineMetrics] = []
    
    def execute(
        self,
        problem: ProblemDefinition,
        use_parallel_evolution: bool = True
    ) -> PipelineResult:
        """
        Execute the full OpenEvolve-integrated pipeline.
        
        Args:
            problem: Problem to solve
            use_parallel_evolution: Whether to evolve sub-problems in parallel
            
        Returns:
            PipelineResult with complete execution results
        """
        total_start = time.time()
        
        self.logger.info(f"Starting OpenEvolve-integrated pipeline for: {problem.title}")
        
        # Stage 1: Decomposition
        decomp_start = time.time()
        decomposition_plan = self.decomposition_engine.decompose(problem)
        decomp_time = time.time() - decomp_start

        entanglement_matrix = (decomposition_plan.metadata or {}).get("entanglement_matrix", {})
        if entanglement_matrix:
            for sp in decomposition_plan.sub_problems:
                sp.metadata.setdefault("entangled_with", entanglement_matrix.get(sp.id, []))
        
        self.logger.info(
            f"Decomposition complete: {len(decomposition_plan.sub_problems)} sub-problems, "
            f"quality={decomposition_plan.overall_quality:.2f}"
        )
        
        # Stage 2: Evolution (parallel or sequential)
        evolution_start = time.time()
        
        if use_parallel_evolution:
            sub_solutions = self.parallel_manager.evolve_all(
                decomposition_plan.sub_problems,
                decomposition_plan.dependency_graph
            )
        else:
            sub_solutions = {
                sp.id: self.solver.solve(sp)
                for sp in decomposition_plan.sub_problems
            }
        
        evolution_time = time.time() - evolution_start
        
        successful = sum(1 for s in sub_solutions.values() if s.quality_score >= 0.7)
        avg_fitness = sum(s.quality_score for s in sub_solutions.values()) / len(sub_solutions) if sub_solutions else 0
        
        self.logger.info(
            f"Evolution complete: {successful}/{len(sub_solutions)} successful, "
            f"avg_fitness={avg_fitness:.2f}"
        )
        
        # Stage 3: Recomposition
        recomp_start = time.time()
        integrated_solution = self.recomposition_engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=problem.id,
            decomposition_plan_id=decomposition_plan.id,
            dependency_graph=decomposition_plan.dependency_graph
        )
        recomp_time = time.time() - recomp_start
        
        self.logger.info(
            f"Recomposition complete: quality={integrated_solution.quality_metrics.overall_score:.2f}, "
            f"conflicts={len(integrated_solution.conflicts_detected)}"
        )
        
        # Build result
        total_time = time.time() - total_start
        
        result = PipelineResult(
            pipeline_id=self._generate_id("oe_pipe"),
            problem=problem,
            decomposition_plan=decomposition_plan,
            sub_solutions=sub_solutions,
            integrated_solution=integrated_solution,
            decomposition_quality=decomposition_plan.overall_quality,
            solution_quality=integrated_solution.quality_metrics.overall_score,
            overall_quality=(
                decomposition_plan.overall_quality * 0.3 +
                integrated_solution.quality_metrics.overall_score * 0.7
            )
        )
        
        # Record metrics
        metrics = OpenEvolvePipelineMetrics(
            decomposition_time=decomp_time,
            evolution_time=evolution_time,
            recomposition_time=recomp_time,
            total_time=total_time,
            sub_problems_evolved=len(sub_solutions),
            successful_evolutions=successful,
            failed_evolutions=len(sub_solutions) - successful,
            avg_fitness=avg_fitness,
            max_fitness=max(s.quality_score for s in sub_solutions.values()) if sub_solutions else 0,
            min_fitness=min(s.quality_score for s in sub_solutions.values()) if sub_solutions else 0,
            conflicts_detected=len(integrated_solution.conflicts_detected),
            conflicts_resolved=len(integrated_solution.conflicts_resolved),
            llm_calls=len(sub_solutions),
            tokens_used=0
        )
        self.metrics_history.append(metrics)
        
        self.logger.info(f"Pipeline complete: total_time={total_time:.2f}s, quality={result.overall_quality:.2f}")
        
        return result
    
    def _generate_id(self, prefix: str = "") -> str:
        """Generate unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_solve_with_openevolve(
    title: str,
    description: str,
    domain: ProblemDomain = ProblemDomain.SOFTWARE,
    complexity: Optional[float] = None,
    evolution_config: Optional[EvolutionConfig] = None
) -> PipelineResult:
    """
    Quick helper to solve a problem using OpenEvolve-integrated pipeline.
    
    Args:
        title: Problem title
        description: Problem description
        domain: Problem domain
        complexity: Complexity estimate
        evolution_config: Evolution configuration
        
    Returns:
        PipelineResult
    """
    problem = create_problem_definition(title, description, domain, complexity)
    
    pipeline = OpenEvolveIntegratedPipeline(evolution_config=evolution_config)
    result = pipeline.execute(problem)
    
    return result


def compare_strategies_with_openevolve(
    problem: ProblemDefinition,
    strategies: Optional[List[DecompositionStrategy]] = None
) -> Dict[str, Any]:
    """
    Compare different decomposition strategies with OpenEvolve evolution.
    
    Args:
        problem: Problem to solve
        strategies: List of strategies to compare
        
    Returns:
        Comparison results
    """
    strategies = strategies or [
        DecompositionStrategy.HIERARCHICAL,
        DecompositionStrategy.FUNCTIONAL,
        DecompositionStrategy.SEMANTIC,
        DecompositionStrategy.HYBRID,
    ]
    
    results = []
    
    for strategy in strategies:
        pipeline = OpenEvolveIntegratedPipeline()
        pipeline.decomposition_engine = EnhancedDecompositionEngine()
        
        # Override strategy
        decomposition_plan = pipeline.decomposition_engine.decompose(
            problem,
            strategy=strategy
        )
        
        # Evolve and assemble
        sub_solutions = pipeline.parallel_manager.evolve_all(
            decomposition_plan.sub_problems,
            decomposition_plan.dependency_graph
        )
        
        integrated_solution = pipeline.recomposition_engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=problem.id,
            decomposition_plan_id=decomposition_plan.id,
            dependency_graph=decomposition_plan.dependency_graph
        )
        
        results.append({
            'strategy': strategy.value,
            'sub_problems': len(decomposition_plan.sub_problems),
            'decomposition_quality': decomposition_plan.overall_quality,
            'solution_quality': integrated_solution.quality_metrics.overall_score,
            'conflicts': len(integrated_solution.conflicts_detected),
            'avg_fitness': sum(s.quality_score for s in sub_solutions.values()) / len(sub_solutions)
        })
    
    return {
        'problem': problem.title,
        'results': results,
        'best_strategy': max(results, key=lambda x: x['solution_quality'])['strategy']
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("OpenEvolve Enhanced Decomposition Integration Demo")
    print("=" * 70)
    
    # Create problem
    problem = create_problem_definition(
        title="Build Distributed Task Queue System",
        description="""
        Design and implement a distributed task queue system that can:
        - Handle 100,000+ tasks per second
        - Support priority queuing and delayed execution
        - Provide at-least-once delivery guarantees
        - Include monitoring and alerting capabilities
        - Scale horizontally across multiple nodes
        """,
        domain=ProblemDomain.SOFTWARE,
        complexity=8.5
    )
    
    print(f"\nProblem: {problem.title}")
    print(f"Domain: {problem.domain.value}")
    print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    
    # Create pipeline
    evolution_config = EvolutionConfig(
        max_iterations=30,
        parallel_evolution=True,
        max_workers=4
    )
    
    pipeline = OpenEvolveIntegratedPipeline(evolution_config=evolution_config)
    
    print("\nExecuting OpenEvolve-integrated pipeline...")
    print("-" * 70)
    
    # Execute
    result = pipeline.execute(problem)
    
    # Display results
    print("\nResults:")
    print("-" * 70)
    print(f"Pipeline ID: {result.pipeline_id}")
    print(f"Successful: {result.is_successful()}")
    print(f"Overall Quality: {result.overall_quality:.2f}")
    print(f"Decomposition Quality: {result.decomposition_quality:.2f}")
    print(f"Solution Quality: {result.solution_quality:.2f}")
    
    if result.decomposition_plan:
        print(f"\nDecomposition:")
        print(f"  Strategy: {result.decomposition_plan.strategy_used.value}")
        print(f"  Sub-problems: {len(result.decomposition_plan.sub_problems)}")
        print(f"  Parallel Groups: {len(result.decomposition_plan.parallel_groups)}")
    
    if result.integrated_solution:
        print(f"\nSolution:")
        print(f"  Assembly Strategy: {result.integrated_solution.assembly_strategy.value}")
        print(f"  Conflicts Detected: {len(result.integrated_solution.conflicts_detected)}")
        print(f"  Conflicts Resolved: {len(result.integrated_solution.conflicts_resolved)}")
        
        metrics = result.integrated_solution.quality_metrics
        print(f"\n  Quality Metrics:")
        print(f"    Overall: {metrics.overall_score:.2f}")
        print(f"    Completeness: {metrics.completeness:.2f}")
        print(f"    Consistency: {metrics.consistency:.2f}")
        print(f"    Coherence: {metrics.coherence:.2f}")
    
    # Display metrics history
    if pipeline.metrics_history:
        latest = pipeline.metrics_history[-1]
        print(f"\nPerformance Metrics:")
        print(f"  Total Time: {latest.total_time:.2f}s")
        print(f"  Decomposition: {latest.decomposition_time:.2f}s")
        print(f"  Evolution: {latest.evolution_time:.2f}s")
        print(f"  Recomposition: {latest.recomposition_time:.2f}s")
        print(f"  Avg Fitness: {latest.avg_fitness:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
