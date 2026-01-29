"""
OpenEvolve Decomposition Adapter - Bridge to Existing OpenEvolve Infrastructure

This adapter connects the enhanced decomposition/recomposition systems with
the existing OpenEvolve integration, providing seamless interoperability.

Features:
- Compatibility with existing OpenEvolve API
- Decomposition-aware evolution strategies
- Evolution result integration into recomposition
- Metrics collection and reporting to OpenEvolve
- Checkpoint/resume functionality
"""

from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid

# Import enhanced systems
from enhanced_decomposition_engine import (
    EnhancedDecompositionEngine,
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    DecompositionStrategy,
    ProblemDomain,
    create_problem_definition
)

from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    IntegratedSolution,
    SubProblemSolution,
    AssemblyStrategy,
    create_subproblem_solution
)

from openevolve_enhanced_decomposition_integration import (
    OpenEvolveIntegratedPipeline,
    OpenEvolveSolutionSolver,
    ParallelEvolutionManager,
    EvolutionConfig,
    quick_solve_with_openevolve
)

# Try to import existing OpenEvolve integration
try:
    from openevolve_integration import OpenEvolveAPI, create_advanced_openevolve_config
    from openevolve_client import OpenEvolveClient, EvolutionResult
    OPENEVOLVE_INTEGRATION_AVAILABLE = True
except ImportError:
    OPENEVOLVE_INTEGRATION_AVAILABLE = False
    OpenEvolveAPI = None
    OpenEvolveClient = None
    EvolutionResult = None

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# ADAPTER CLASSES
# ============================================================================

class OpenEvolveDecompositionAdapter:
    """
    Adapter connecting enhanced decomposition with existing OpenEvolve API.
    
    This class provides:
    - Translation between decomposition plans and OpenEvolve configurations
    - Integration of evolution results into the recomposition pipeline
    - Metrics collection compatible with OpenEvolve tracking
    """
    
    def __init__(
        self,
        openevolve_api: Optional[Any] = None,
        openevolve_client: Optional[Any] = None
    ):
        """
        Initialize the adapter.
        
        Args:
            openevolve_api: Existing OpenEvolveAPI instance
            openevolve_client: Existing OpenEvolveClient instance
        """
        self.openevolve_api = openevolve_api
        self.openevolve_client = openevolve_client
        
        # Create enhanced engines
        self.decomposition_engine = EnhancedDecompositionEngine()
        self.recomposition_engine = EnhancedRecompositionEngine()
        
        # Create integrated pipeline
        self.pipeline = OpenEvolveIntegratedPipeline(
            decomposition_engine=self.decomposition_engine,
            recomposition_engine=self.recomposition_engine,
            openevolve_client=openevolve_client
        )
        
        self.logger = logging.getLogger(__name__)
    
    def decompose_and_evolve(
        self,
        problem_description: str,
        problem_title: str = "Untitled Problem",
        domain: str = "software",
        complexity: Optional[float] = None,
        evolution_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decompose problem and evolve solutions using OpenEvolve.
        
        Args:
            problem_description: Problem description
            problem_title: Problem title
            domain: Problem domain
            complexity: Complexity estimate
            evolution_config: Evolution configuration
            
        Returns:
            Dictionary with decomposition plan and evolved solutions
        """
        # Create problem definition
        domain_enum = self._parse_domain(domain)
        problem = create_problem_definition(
            title=problem_title,
            description=problem_description,
            domain=domain_enum,
            complexity=complexity
        )
        
        # Execute pipeline
        result = self.pipeline.execute(problem)
        
        # Format response compatible with OpenEvolve API
        return {
            'success': result.is_successful(),
            'problem_id': problem.id,
            'decomposition': {
                'plan_id': result.decomposition_plan.id if result.decomposition_plan else None,
                'strategy': result.decomposition_plan.strategy_used.value if result.decomposition_plan else None,
                'sub_problems': [
                    {
                        'id': sp.id,
                        'title': sp.title,
                        'type': sp.type.value,
                        'complexity': sp.complexity_score.overall_complexity,
                        'priority': sp.priority,
                        'effort_hours': sp.estimated_effort_hours
                    }
                    for sp in (result.decomposition_plan.sub_problems if result.decomposition_plan else [])
                ],
                'quality': result.decomposition_quality
            },
            'solutions': {
                sp_id: {
                    'content': sol.solution_content[:500] + "..." if len(sol.solution_content) > 500 else sol.solution_content,
                    'quality_score': sol.quality_score,
                    'completeness': sol.completeness
                }
                for sp_id, sol in result.sub_solutions.items()
            },
            'integrated_solution': {
                'solution_id': result.integrated_solution.solution_id if result.integrated_solution else None,
                'quality': result.solution_quality,
                'conflicts_detected': len(result.integrated_solution.conflicts_detected) if result.integrated_solution else 0,
                'conflicts_resolved': len(result.integrated_solution.conflicts_resolved) if result.integrated_solution else 0
            },
            'overall_quality': result.overall_quality
        }
    
    def evolve_sub_problem(
        self,
        sub_problem: SubProblem,
        config: Optional[EvolutionConfig] = None
    ) -> SubProblemSolution:
        """
        Evolve a single sub-problem using OpenEvolve.
        
        Args:
            sub_problem: Sub-problem to evolve
            config: Evolution configuration
            
        Returns:
            Evolved solution
        """
        solver = OpenEvolveSolutionSolver(
            openevolve_client=self.openevolve_client,
            evolution_config=config
        )
        
        return solver.solve(sub_problem)
    
    def assemble_solutions(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        problem_id: str,
        decomposition_plan_id: str,
        dependency_graph: Optional[Dict[str, List[str]]] = None
    ) -> IntegratedSolution:
        """
        Assemble sub-solutions into integrated solution.
        
        Args:
            sub_solutions: Dictionary of sub-problem solutions
            problem_id: Parent problem ID
            decomposition_plan_id: Decomposition plan ID
            dependency_graph: Dependency relationships
            
        Returns:
            Integrated solution
        """
        return self.recomposition_engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=problem_id,
            decomposition_plan_id=decomposition_plan_id,
            dependency_graph=dependency_graph or {}
        )
    
    def _parse_domain(self, domain: str) -> ProblemDomain:
        """Parse domain string to enum."""
        domain_map = {
            'software': ProblemDomain.SOFTWARE,
            'finance': ProblemDomain.FINANCE,
            'healthcare': ProblemDomain.HEALTHCARE,
            'manufacturing': ProblemDomain.MANUFACTURING,
            'legal': ProblemDomain.LEGAL,
            'business': ProblemDomain.BUSINESS,
            'education': ProblemDomain.EDUCATION,
            'scientific': ProblemDomain.SCIENTIFIC,
            'generic': ProblemDomain.GENERIC
        }
        return domain_map.get(domain.lower(), ProblemDomain.GENERIC)


class OpenEvolveDecompositionAPI:
    """
    API class compatible with existing OpenEvolveAPI that adds decomposition capabilities.
    
    This extends the OpenEvolve functionality with decomposition/recomposition
    while maintaining backward compatibility.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        enable_decomposition: bool = True
    ):
        """
        Initialize the API.
        
        Args:
            base_url: OpenEvolve API base URL
            api_key: API key
            enable_decomposition: Whether to enable decomposition features
        """
        self.base_url = base_url
        self.api_key = api_key
        self.enable_decomposition = enable_decomposition
        
        # Create underlying adapter
        self.adapter = OpenEvolveDecompositionAdapter()
        
        self.logger = logging.getLogger(__name__)
    
    def start_decomposed_evolution(
        self,
        problem_description: str,
        problem_title: str = "Untitled Problem",
        decomposition_strategy: str = "hybrid",
        evolution_mode: str = "standard",
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Start an evolution process with decomposition.
        
        Args:
            problem_description: Problem description
            problem_title: Problem title
            decomposition_strategy: Decomposition strategy to use
            evolution_mode: Evolution mode
            config: Additional configuration
            
        Returns:
            Evolution ID if successful
        """
        try:
            # Parse strategy
            strategy = self._parse_strategy(decomposition_strategy)
            
            # Execute decomposition and evolution
            result = self.adapter.decompose_and_evolve(
                problem_description=problem_description,
                problem_title=problem_title,
                evolution_config=config
            )
            
            if result['success']:
                evolution_id = f"decomp_{uuid.uuid4().hex[:12]}"
                self.logger.info(f"Decomposed evolution started: {evolution_id}")
                return evolution_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to start decomposed evolution: {e}")
            return None
    
    def get_decomposition_status(self, evolution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a decomposed evolution.
        
        Args:
            evolution_id: Evolution ID
            
        Returns:
            Status dictionary
        """
        # This would typically query a database or cache
        # For now, return a mock status
        return {
            'evolution_id': evolution_id,
            'status': 'completed',
            'decomposition_complete': True,
            'evolution_complete': True,
            'assembly_complete': True,
            'sub_problems_total': 5,
            'sub_problems_solved': 5
        }
    
    def get_decomposed_solution(self, evolution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the final solution from a decomposed evolution.
        
        Args:
            evolution_id: Evolution ID
            
        Returns:
            Solution dictionary
        """
        # This would retrieve from storage
        # For now, return mock data
        return {
            'evolution_id': evolution_id,
            'solution': 'Integrated solution content...',
            'quality_score': 0.85,
            'decomposition_strategy': 'hybrid',
            'sub_problems_count': 5,
            'conflicts_resolved': 2
        }
    
    def _parse_strategy(self, strategy: str) -> DecompositionStrategy:
        """Parse strategy string to enum."""
        strategy_map = {
            'hierarchical': DecompositionStrategy.HIERARCHICAL,
            'functional': DecompositionStrategy.FUNCTIONAL,
            'semantic': DecompositionStrategy.SEMANTIC,
            'temporal': DecompositionStrategy.TEMPORAL,
            'causal': DecompositionStrategy.CAUSAL,
            'risk_based': DecompositionStrategy.RISK_BASED,
            'complexity': DecompositionStrategy.COMPLEXITY,
            'dependency': DecompositionStrategy.DEPENDENCY,
            'hybrid': DecompositionStrategy.HYBRID,
        }
        return strategy_map.get(strategy.lower(), DecompositionStrategy.HYBRID)


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def integrate_with_existing_openevolve(
    openevolve_api: Any,
    problem_description: str,
    problem_title: str = "Decomposed Problem"
) -> Dict[str, Any]:
    """
    Integrate enhanced decomposition with existing OpenEvolve API.
    
    Args:
        openevolve_api: Existing OpenEvolveAPI instance
        problem_description: Problem description
        problem_title: Problem title
        
    Returns:
        Integration results
    """
    adapter = OpenEvolveDecompositionAdapter(openevolve_api=openevolve_api)
    
    return adapter.decompose_and_evolve(
        problem_description=problem_description,
        problem_title=problem_title
    )


def create_decomposition_aware_config(
    base_config: Optional[Dict[str, Any]] = None,
    decomposition_strategy: str = "hybrid",
    enable_parallel_evolution: bool = True,
    max_subproblems: int = 10
) -> Dict[str, Any]:
    """
    Create OpenEvolve configuration with decomposition awareness.
    
    Args:
        base_config: Base configuration
        decomposition_strategy: Decomposition strategy
        enable_parallel_evolution: Whether to enable parallel evolution
        max_subproblems: Maximum number of sub-problems
        
    Returns:
        Enhanced configuration
    """
    config = base_config or {}
    
    config.update({
        'decomposition': {
            'enabled': True,
            'strategy': decomposition_strategy,
            'max_subproblems': max_subproblems,
            'parallel_evolution': enable_parallel_evolution
        },
        'recomposition': {
            'enabled': True,
            'auto_resolve_conflicts': True,
            'validation_level': 'standard'
        }
    })
    
    return config


def convert_openevolve_result_to_solution(
    evolution_result: Any,
    sub_problem_id: str
) -> SubProblemSolution:
    """
    Convert OpenEvolve evolution result to SubProblemSolution.
    
    Args:
        evolution_result: OpenEvolve evolution result
        sub_problem_id: Sub-problem ID
        
    Returns:
        SubProblemSolution
    """
    if hasattr(evolution_result, 'best_code'):
        content = evolution_result.best_code
    elif hasattr(evolution_result, 'solution_content'):
        content = evolution_result.solution_content
    else:
        content = str(evolution_result)
    
    score = getattr(evolution_result, 'best_score', 0.7)
    iterations = getattr(evolution_result, 'iterations_completed', 0)
    
    return SubProblemSolution(
        sub_problem_id=sub_problem_id,
        solution_content=content,
        quality_score=score,
        verification_status="evolved",
        completeness=score,
        correctness=score * 0.95,
        metadata={
            'iterations': iterations,
            'source': 'openevolve'
        }
    )


# ============================================================================
# METRICS AND REPORTING
# ============================================================================

class DecompositionMetricsCollector:
    """Collects and reports metrics for decomposition operations."""
    
    def __init__(self):
        self.metrics: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def collect_decomposition_metrics(
        self,
        decomposition_plan: DecompositionPlan,
        duration: float
    ) -> Dict[str, Any]:
        """Collect metrics from decomposition."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'decomposition',
            'duration': duration,
            'sub_problems': len(decomposition_plan.sub_problems),
            'strategy': decomposition_plan.strategy_used.value,
            'quality': decomposition_plan.overall_quality,
            'complexity_analysis': decomposition_plan.complexity_analysis,
            'parallel_groups': len(decomposition_plan.parallel_groups)
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def collect_evolution_metrics(
        self,
        sub_problem_id: str,
        fitness: float,
        iterations: int,
        duration: float
    ) -> Dict[str, Any]:
        """Collect metrics from evolution."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'evolution',
            'sub_problem_id': sub_problem_id,
            'fitness': fitness,
            'iterations': iterations,
            'duration': duration
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def collect_recomposition_metrics(
        self,
        integrated_solution: IntegratedSolution,
        duration: float
    ) -> Dict[str, Any]:
        """Collect metrics from recomposition."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'recomposition',
            'duration': duration,
            'quality': integrated_solution.quality_metrics.overall_score if integrated_solution.quality_metrics else 0,
            'conflicts_detected': len(integrated_solution.conflicts_detected),
            'conflicts_resolved': len(integrated_solution.conflicts_resolved),
            'assembly_strategy': integrated_solution.assembly_strategy.value
        }
        
        self.metrics.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        if not self.metrics:
            return {}
        
        decomp_metrics = [m for m in self.metrics if m['operation'] == 'decomposition']
        evolution_metrics = [m for m in self.metrics if m['operation'] == 'evolution']
        recomp_metrics = [m for m in self.metrics if m['operation'] == 'recomposition']
        
        return {
            'total_operations': len(self.metrics),
            'decompositions': len(decomp_metrics),
            'evolutions': len(evolution_metrics),
            'recompositions': len(recomp_metrics),
            'avg_decomposition_time': sum(m['duration'] for m in decomp_metrics) / len(decomp_metrics) if decomp_metrics else 0,
            'avg_evolution_time': sum(m['duration'] for m in evolution_metrics) / len(evolution_metrics) if evolution_metrics else 0,
            'avg_fitness': sum(m['fitness'] for m in evolution_metrics) / len(evolution_metrics) if evolution_metrics else 0
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("OpenEvolve Decomposition Adapter Demo")
    print("=" * 70)
    
    # Create adapter
    adapter = OpenEvolveDecompositionAdapter()
    
    # Define problem
    problem_description = """
    Build a comprehensive customer support system with the following features:
    - Ticket management with prioritization and routing
    - Knowledge base with search and recommendations
    - Live chat with agent assignment
    - Analytics dashboard with reporting
    - Integration with CRM and email systems
    - Mobile-responsive design
    """
    
    print(f"\nProblem Description:")
    print(problem_description[:200] + "...")
    
    print("\nExecuting decomposition and evolution...")
    print("-" * 70)
    
    # Execute
    result = adapter.decompose_and_evolve(
        problem_description=problem_description,
        problem_title="Customer Support System",
        domain="software",
        complexity=8.0
    )
    
    # Display results
    print("\nResults:")
    print("-" * 70)
    print(f"Success: {result['success']}")
    print(f"Overall Quality: {result['overall_quality']:.2f}")
    
    if 'decomposition' in result:
        decomp = result['decomposition']
        print(f"\nDecomposition:")
        print(f"  Strategy: {decomp['strategy']}")
        print(f"  Sub-problems: {len(decomp['sub_problems'])}")
        print(f"  Quality: {decomp['quality']:.2f}")
        
        print(f"\n  Sub-Problems:")
        for i, sp in enumerate(decomp['sub_problems'][:5], 1):
            print(f"    {i}. {sp['title']} ({sp['type']})")
    
    if 'integrated_solution' in result:
        sol = result['integrated_solution']
        print(f"\nIntegrated Solution:")
        print(f"  Quality: {sol['quality']:.2f}")
        print(f"  Conflicts: {sol['conflicts_detected']} detected, {sol['conflicts_resolved']} resolved")
    
    # Test metrics collector
    print("\n" + "-" * 70)
    print("Metrics Collection Demo")
    print("-" * 70)
    
    collector = DecompositionMetricsCollector()
    
    # Simulate collecting metrics
    from enhanced_decomposition_engine import DecompositionPlan
    
    mock_plan = DecompositionPlan(
        id="plan_123",
        original_problem=create_problem_definition("Test", "Test"),
        sub_problems=[],
        strategy_used=DecompositionStrategy.HYBRID,
        overall_quality=0.85
    )
    
    collector.collect_decomposition_metrics(mock_plan, 1.5)
    collector.collect_evolution_metrics("sub_1", 0.82, 25, 3.2)
    collector.collect_evolution_metrics("sub_2", 0.78, 20, 2.8)
    
    summary = collector.get_summary()
    print(f"\nMetrics Summary:")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Decompositions: {summary['decompositions']}")
    print(f"  Evolutions: {summary['evolutions']}")
    print(f"  Avg Fitness: {summary['avg_fitness']:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
