"""
Demo of OpenEvolve + LoongFlow PES Integration.

This demo shows how the integrated system uses PES planning
to guide OpenEvolve evolution with cost optimization.
"""

import asyncio
import logging
from dataclasses import dataclass

from .config import UnifiedEvolutionConfig, StrategySelectionMode
from .orchestrator import StrategyOrchestrator, EvolutionProblem


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PES-Integration-Demo")


@dataclass
class DemoResult:
    """Demo execution result."""
    strategy: str
    success: bool
    fitness: float
    iterations: int
    cost_usd: float
    time_seconds: float


async def demo_basic_integration():
    """Demo: Basic PES-enhanced evolution."""
    logger.info("=" * 60)
    logger.info("DEMO 1: Basic PES-Enhanced Evolution")
    logger.info("=" * 60)
    
    # Configure unified system
    config = UnifiedEvolutionConfig(
        strategy_selection_mode=StrategySelectionMode.PES_ONLY,
        enable_pes_planning=True,
        max_cost_usd=5.0,
        max_tokens=50000,
        max_iterations=50,
        population_size=30
    )
    
    # Create orchestrator
    orchestrator = StrategyOrchestrator(config)
    
    # Define problem
    problem = EvolutionProblem(
        description="Optimize a sorting algorithm for performance",
        code="def sort(arr): return sorted(arr)",
        language="python",
        exploration_focus=False
    )
    
    # Run evolution
    result = await orchestrator.evolve(problem)
    
    logger.info(f"Strategy used: {result.strategy_used.name}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Final fitness: {result.final_fitness:.2f}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Cost: ${result.cost_summary.get('cost_usd', 0):.2f}")
    logger.info(f"Time: {result.execution_time_seconds:.2f}s")
    
    return DemoResult(
        strategy=result.strategy_used.name,
        success=result.success,
        fitness=result.final_fitness,
        iterations=result.iterations,
        cost_usd=result.cost_summary.get("cost_usd", 0),
        time_seconds=result.execution_time_seconds
    )


async def demo_cost_optimization():
    """Demo: Cost-aware evolution with tight budget."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Cost-Aware Evolution (Tight Budget)")
    logger.info("=" * 60)
    
    # Configure with tight budget
    config = UnifiedEvolutionConfig(
        strategy_selection_mode=StrategySelectionMode.AUTO,
        enable_pes_planning=True,
        max_cost_usd=2.0,  # Tight budget
        max_tokens=20000,
        max_iterations=100,
        population_size=50
    )
    
    orchestrator = StrategyOrchestrator(config)
    
    problem = EvolutionProblem(
        description="Implement a function to calculate fibonacci numbers efficiently",
        code="def fib(n):\n    if n <= 1: return n\n    return fib(n-1) + fib(n-2)",
        language="python",
        constraints={"time_complexity": "O(n)"}
    )
    
    result = await orchestrator.evolve(problem)
    
    logger.info(f"Strategy used: {result.strategy_used.name}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Final fitness: {result.final_fitness:.2f}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Cost: ${result.cost_summary.get('cost_usd', 0):.2f}")
    logger.info(f"Budget: $2.00")
    logger.info(f"Budget used: {result.cost_summary.get('cost_usd', 0) / 2.0 * 100:.1f}%")
    
    return DemoResult(
        strategy=result.strategy_used.name,
        success=result.success,
        fitness=result.final_fitness,
        iterations=result.iterations,
        cost_usd=result.cost_summary.get("cost_usd", 0),
        time_seconds=result.execution_time_seconds
    )


async def demo_multi_objective():
    """Demo: Multi-objective evolution with PES guidance."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Multi-Objective PES-Guided Evolution")
    logger.info("=" * 60)
    
    config = UnifiedEvolutionConfig(
        strategy_selection_mode=StrategySelectionMode.AUTO,
        enable_pes_planning=True,
        max_cost_usd=10.0,
        max_iterations=80
    )
    
    orchestrator = StrategyOrchestrator(config)
    
    problem = EvolutionProblem(
        description="Optimize a trading algorithm",
        objectives=["profit", "risk", "liquidity"],
        constraints={"max_drawdown": 0.2}
    )
    
    result = await orchestrator.evolve(problem)
    
    logger.info(f"Strategy used: {result.strategy_used.name}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Final fitness: {result.final_fitness:.2f}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Cost: ${result.cost_summary.get('cost_usd', 0):.2f}")
    
    return DemoResult(
        strategy=result.strategy_used.name,
        success=result.success,
        fitness=result.final_fitness,
        iterations=result.iterations,
        cost_usd=result.cost_summary.get("cost_usd", 0),
        time_seconds=result.execution_time_seconds
    )


async def demo_exploration_focus():
    """Demo: Exploration-focused evolution with QD."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Exploration-Focused (QD Mode)")
    logger.info("=" * 60)
    
    config = UnifiedEvolutionConfig(
        strategy_selection_mode=StrategySelectionMode.AUTO,
        enable_pes_planning=True,
        max_cost_usd=8.0,
        max_iterations=100
    )
    
    orchestrator = StrategyOrchestrator(config)
    
    problem = EvolutionProblem(
        description="Discover diverse neural network architectures for image classification",
        exploration_focus=True,
        constraints={"max_params": 1000000}
    )
    
    result = await orchestrator.evolve(problem)
    
    logger.info(f"Strategy used: {result.strategy_used.name}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Final fitness: {result.final_fitness:.2f}")
    logger.info(f"Iterations: {result.iterations}")
    logger.info(f"Cost: ${result.cost_summary.get('cost_usd', 0):.2f}")
    
    return DemoResult(
        strategy=result.strategy_used.name,
        success=result.success,
        fitness=result.final_fitness,
        iterations=result.iterations,
        cost_usd=result.cost_summary.get("cost_usd", 0),
        time_seconds=result.execution_time_seconds
    )


async def demo_comparison():
    """Demo: Compare PES-enhanced vs Standard evolution."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: PES vs Standard Comparison")
    logger.info("=" * 60)
    
    problem = EvolutionProblem(
        description="Implement an efficient string matching algorithm",
        code="def find(text, pattern):\n    return text.find(pattern)",
        language="python"
    )
    
    # PES-enhanced
    logger.info("\n--- PES-Enhanced ---")
    pes_config = UnifiedEvolutionConfig(
        strategy_selection_mode=StrategySelectionMode.PES_ONLY,
        enable_pes_planning=True,
        max_cost_usd=5.0,
        max_iterations=50
    )
    pes_orchestrator = StrategyOrchestrator(pes_config)
    pes_result = await pes_orchestrator.evolve(problem)
    
    # Standard
    logger.info("\n--- Standard ---")
    std_config = UnifiedEvolutionConfig(
        strategy_selection_mode=StrategySelectionMode.MANUAL,
        evolution_mode="standard",
        enable_pes_planning=False,
        max_cost_usd=5.0,
        max_iterations=50
    )
    std_orchestrator = StrategyOrchestrator(std_config)
    std_result = await std_orchestrator.evolve(problem)
    
    # Compare
    logger.info("\n--- Comparison ---")
    logger.info(f"PES Fitness: {pes_result.final_fitness:.2f} vs Standard: {std_result.final_fitness:.2f}")
    logger.info(f"PES Iterations: {pes_result.iterations} vs Standard: {std_result.iterations}")
    logger.info(f"PES Cost: ${pes_result.cost_summary.get('cost_usd', 0):.2f} vs Standard: ${std_result.cost_summary.get('cost_usd', 0):.2f}")
    logger.info(f"PES Time: {pes_result.execution_time_seconds:.2f}s vs Standard: {std_result.execution_time_seconds:.2f}s")


async def run_all_demos():
    """Run all demos."""
    results = []
    
    try:
        results.append(await demo_basic_integration())
    except Exception as e:
        logger.error(f"Demo 1 failed: {e}")
    
    try:
        results.append(await demo_cost_optimization())
    except Exception as e:
        logger.error(f"Demo 2 failed: {e}")
    
    try:
        results.append(await demo_multi_objective())
    except Exception as e:
        logger.error(f"Demo 3 failed: {e}")
    
    try:
        results.append(await demo_exploration_focus())
    except Exception as e:
        logger.error(f"Demo 4 failed: {e}")
    
    try:
        await demo_comparison()
    except Exception as e:
        logger.error(f"Demo 5 failed: {e}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\nDemo {i}:")
        logger.info(f"  Strategy: {result.strategy}")
        logger.info(f"  Success: {result.success}")
        logger.info(f"  Fitness: {result.fitness:.2f}")
        logger.info(f"  Cost: ${result.cost_usd:.2f}")
        logger.info(f"  Time: {result.time_seconds:.2f}s")


if __name__ == "__main__":
    asyncio.run(run_all_demos())
