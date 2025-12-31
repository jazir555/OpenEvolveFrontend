"""
Demo: MDAP/MAKER + MCTS Unified Framework

This script demonstrates the unified MDAP/MAKER + MCTS framework with all three
hybrid approaches.

Usage:
    python demo_mdap_maker_mcts_unified.py

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import logging
import json
from typing import Dict, Any

# Import the unified framework
from mdap_maker_mcts_unified import (
    MDAPMAKERMCTSConfig,
    MDAPMAKERMCTSResult,
    MDAPMAKERMCTSEngine,
    MDAPMCTSCache,
    MDAPMCTSMonitor,
    MDAPAdaptiveSelector,
    MDAPCombinedSearch,
    MDAPMCTSBenchmark,
    MDAPMCTSWorkflowIntegrator,
    MDAPMCTSPresets,
    MCTSApproach,
    ProblemComplexity,
    SubProblem,
    SolutionAttempt,
    create_test_theorem,
    estimate_complexity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(title: str = ""):
    """Print a visual separator"""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def print_result(result: MDAPMAKERMCTSResult, approach_name: str = ""):
    """Pretty print a search result"""
    if approach_name:
        print(f"\n{approach_name} Results:")
        print("-" * 40)

    print(f"  Success: {result.success}")
    print(f"  Best Fitness: {result.best_fitness:.3f}")
    print(f"  Execution Time: {result.execution_time:.2f}s")

    if result.best_proof:
        preview = result.best_proof[:100] + "..." if len(result.best_proof) > 100 else result.best_proof
        print(f"  Proof Preview: {preview}")

    if result.consensus_score is not None:
        print(f"  Consensus Score: {result.consensus_score:.3f}")

    if result.agreement_level is not None:
        print(f"  Agreement Level: {result.agreement_level:.3f}")

    if result.agent_results:
        print(f"  Agent Evaluations: {len(result.agent_results)}")
        for agent_result in result.agent_results[:3]:  # Show first 3
            print(f"    - {agent_result.agent_id}: fitness={agent_result.fitness:.3f}, confidence={agent_result.confidence:.3f}")

    if result.verification_result:
        print(f"  Verification: {'Valid' if result.verification_result.is_valid else 'Failed'}")
        print(f"  Verification Time: {result.verification_result.verification_time:.2f}s")

    if result.error_message:
        print(f"  Error: {result.error_message}")

    if result.warnings:
        print(f"  Warnings: {len(result.warnings)}")
        for warning in result.warnings[:3]:
            print(f"    - {warning}")


async def demo_basic_usage():
    """Demonstrate basic usage of the unified framework"""
    print_separator("DEMO 1: Basic Usage")

    # Create a balanced configuration
    config = MDAPMCTSPresets.balanced()

    print(f"\nConfiguration:")
    print(f"  Approach: {config.approach.value}")
    print(f"  Num Agents: {config.num_agents}")
    print(f"  Voting Strategy: {config.voting_strategy}")
    print(f"  K-Ahead: {config.k_ahead}")
    print(f"  Enable Decomposition: {config.enable_decomposition}")
    print(f"  LeanAide Enabled: {config.leanaide_enabled}")

    # Create engine
    engine = MDAPMAKERMCTSEngine(config)

    # Create test theorem
    theorem = create_test_theorem("medium")
    complexity = estimate_complexity(theorem)
    print(f"\nTheorem Complexity: {complexity.value}")

    # Run search
    print(f"\nRunning search...")
    result = await engine.search(theorem)

    # Display results
    print_result(result, config.approach.value.upper())

    # Show cache statistics
    print(f"\nCache Statistics:")
    cache_stats = engine.cache.get_stats()
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")

    # Show monitor summary
    print(f"\nExecution Summary:")
    summary = engine.monitor.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


async def demo_all_approaches():
    """Demonstrate all three approaches"""
    print_separator("DEMO 2: All Three Approaches")

    theorem = create_test_theorem("medium")

    approaches = [
        MCTSApproach.EVOLVED_POLICIES,
        MCTSApproach.EVOLUTIONARY_NODES,
        MCTSApproach.COEVOLUTION
    ]

    results = {}

    for approach in approaches:
        print(f"\nTesting {approach.value}...")

        config = MDAPMAKERMCTSConfig(
            approach=approach,
            num_agents=5,
            simulations=50,  # Lower for demo
            max_depth=25
        )

        engine = MDAPMAKERMCTSEngine(config)
        result = await engine.search(theorem)
        results[approach.value] = result

        print_result(result, approach.value.upper())

    # Compare results
    print_separator("Approach Comparison")
    print(f"\n{'Approach':<25} {'Fitness':<12} {'Time (s)':<12} {'Success':<10}")
    print("-" * 60)

    for approach_name, result in results.items():
        print(f"{approach_name:<25} {result.best_fitness:<12.3f} {result.execution_time:<12.2f} {str(result.success):<10}")


async def demo_adaptive_selection():
    """Demonstrate adaptive approach selection"""
    print_separator("DEMO 3: Adaptive Approach Selection")

    selector = MDAPAdaptiveSelector()

    theorems = {
        "easy": "theorem easy (n : Nat) : n + 0 = n := by",
        "medium": "theorem medium (a b : Nat) : a * b = b * a := by",
        "hard": "theorem hard (a b c : Nat) : a * (b + c) = a * b + a * c := by"
    }

    print(f"\nAdaptive Selection for Different Complexities:")
    print(f"\n{'Theorem':<20} {'Complexity':<15} {'Selected Approach':<25}")
    print("-" * 65)

    for difficulty, theorem in theorems.items():
        approach = selector.select_approach(theorem, available_agents=5)
        complexity = estimate_complexity(theorem)
        print(f"{difficulty:<20} {complexity.value:<15} {approach.value:<25}")


async def demo_combined_search():
    """Demonstrate combined search using all approaches"""
    print_separator("DEMO 4: Combined Search")

    config = MDAPMAKERMCTSConfig(
        approach=MCTSApproach.COMBINED,
        num_agents=5,
        simulations=30,  # Lower for demo
        enable_decomposition=True
    )

    engine = MDAPMAKERMCTSEngine(config)

    theorem = create_test_theorem("medium")

    print(f"\nRunning combined search with all approaches...")
    result = await engine.search(theorem)

    print_result(result, "COMBINED")

    if result.metadata.get('approach_results'):
        print(f"\nIndividual Approach Results:")
        for approach, metrics in result.metadata['approach_results'].items():
            print(f"  {approach}: success={metrics['success']}, fitness={metrics['fitness']:.3f}")


async def demo_configuration_presets():
    """Demonstrate configuration presets"""
    print_separator("DEMO 5: Configuration Presets")

    presets = {
        "Fast": MDAPMCTSPresets.fast(),
        "Balanced": MDAPMCTSPresets.balanced(),
        "Thorough": MDAPMCTSPresets.thorough(),
        "Experimental": MDAPMCTSPresets.experimental()
    }

    print(f"\nAvailable Presets:")
    print(f"\n{'Preset':<15} {'Agents':<10} {'Simulations':<15} {'Decomposition':<15} {'LeanAide':<10}")
    print("-" * 80)

    for name, config in presets.items():
        print(f"{name:<15} {config.num_agents:<10} {config.simulations:<15} "
              f"{str(config.enable_decomposition):<15} {str(config.leanaide_enabled):<10}")


async def demo_workflow_integration():
    """Demonstrate workflow integration"""
    print_separator("DEMO 6: Workflow Integration")

    config = MDAPMAKERMCTSConfig(
        approach=MCTSApproach.EVOLVED_POLICIES,
        num_agents=5,
        simulations=40
    )

    integrator = MDAPMCTSWorkflowIntegrator(config)

    # Create a sub-problem
    subproblem = SubProblem(
        subproblem_id="sub_001",
        theorem=create_test_theorem("medium"),
        dependencies=[],
        priority=1
    )

    print(f"\nSub-problem ID: {subproblem.subproblem_id}")
    print(f"Theorem: {subproblem.theorem[:50]}...")
    print(f"Priority: {subproblem.priority}")

    print(f"\nSolving with MDAP/MCTS workflow...")
    solution = await integrator.solve_with_mdap_mcts(subproblem)

    print(f"\nSolution Attempt:")
    print(f"  Content: {solution.content[:100]}...")
    print(f"  Quality Metrics:")
    for key, value in solution.quality_metrics.items():
        print(f"    {key}: {value}")


async def demo_benchmarking():
    """Demonstrate benchmarking"""
    print_separator("DEMO 7: Benchmarking")

    config = MDAPMAKERMCTSConfig(
        num_agents=3,  # Lower for faster demo
        simulations=20,
        max_depth=15
    )

    benchmark = MDAPMCTSBenchmark(config)

    # Create test theorems
    test_theorems = [
        create_test_theorem("easy"),
        create_test_theorem("medium"),
        create_test_theorem("hard")
    ]

    print(f"\nBenchmarking {len(test_theorems)} theorems...")
    print(f"(This may take a moment...)")

    report = await benchmark.benchmark_all(
        test_theorems=test_theorems,
        approaches=[
            MCTSApproach.EVOLVED_POLICIES,
            MCTSApproach.EVOLUTIONARY_NODES
        ]
    )

    print(f"\nBenchmark Report:")
    print(f"  Timestamp: {report.timestamp}")
    print(f"  Test Theorem Count: {report.test_theorem_count}")

    print(f"\nApproach Results:")
    for approach_name, benchmark_result in report.approaches.items():
        print(f"\n  {approach_name}:")
        print(f"    Success Rate: {benchmark_result.success_rate:.1%}")
        print(f"    Avg Time: {benchmark_result.avg_time:.2f}s")
        print(f"    Avg Fitness: {benchmark_result.avg_fitness:.3f}")
        print(f"    Avg Consensus: {benchmark_result.avg_consensus:.3f}")

    print(f"\nComparison:")
    for key, value in report.comparison.items():
        print(f"  {key}: {value}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")


async def demo_serialization():
    """Demonstrate configuration and result serialization"""
    print_separator("DEMO 8: Serialization")

    # Create configuration
    config = MDAPMAKERMCTSConfig(
        approach=MCTSApproach.EVOLVED_POLICIES,
        num_agents=5,
        simulations=100,
        enable_decomposition=True,
        decomposition_depth=3
    )

    # Serialize to dict
    config_dict = config.to_dict()
    print(f"\nConfiguration serialized to dict:")
    print(f"  Keys: {list(config_dict.keys())[:10]}...")

    # Save to file
    with open('mdap_mcts_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved to: mdap_mcts_config.json")

    # Load from file
    with open('mdap_mcts_config.json', 'r') as f:
        loaded_dict = json.load(f)

    loaded_config = MDAPMAKERMCTSConfig.from_dict(loaded_dict)
    print(f"\nLoaded configuration:")
    print(f"  Approach: {loaded_config.approach.value}")
    print(f"  Num Agents: {loaded_config.num_agents}")
    print(f"  Simulations: {loaded_config.simulations}")

    # Demonstrate result serialization
    engine = MDAPMAKERMCTSEngine(loaded_config)
    result = await engine.search(create_test_theorem("easy"))

    result_dict = result.to_dict()
    print(f"\nResult serialized to dict:")
    print(f"  Keys: {list(result_dict.keys())[:10]}...")

    # Save result
    with open('mdap_mcts_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
    print(f"  Saved to: mdap_mcts_result.json")


async def demo_cache_management():
    """Demonstrate caching system"""
    print_separator("DEMO 9: Cache Management")

    cache = MDAPMCTSCache(max_size=100)

    # Set some values
    await cache.set('policy', 'key1', {'value': 1})
    await cache.set('node', 'key2', {'value': 2})
    await cache.set('tree', 'key3', {'value': 3})

    # Get values
    value1 = await cache.get('policy', 'key1')
    print(f"\nRetrieved value: {value1}")

    # Get or compute
    async def compute_fn():
        return {'computed': True}

    value4 = await cache.get_or_compute('policy', 'key4', compute_fn)
    print(f"Get or compute result: {value4}")

    # Get statistics
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


async def demo_validation():
    """Demonstrate configuration validation"""
    print_separator("DEMO 10: Configuration Validation")

    # Valid configuration
    valid_config = MDAPMAKERMCTSConfig(
        approach=MCTSApproach.EVOLVED_POLICIES,
        num_agents=5,
        simulations=100
    )

    errors = valid_config.validate()
    print(f"\nValid configuration errors: {errors if errors else 'None - Valid!'}")

    # Invalid configuration
    invalid_config = MDAPMAKERMCTSConfig(
        num_agents=0,  # Invalid: must be at least 1
        k_ahead=-1,  # Invalid: must be at least 1
        consensus_threshold=1.5  # Invalid: must be between 0 and 1
    )

    errors = invalid_config.validate()
    print(f"\nInvalid configuration errors:")
    for error in errors:
        print(f"  - {error}")


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("  MDAP/MAKER + MCTS UNIFIED FRAMEWORK - DEMONSTRATION")
    print("=" * 80)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("All Approaches", demo_all_approaches),
        ("Adaptive Selection", demo_adaptive_selection),
        ("Combined Search", demo_combined_search),
        ("Configuration Presets", demo_configuration_presets),
        ("Workflow Integration", demo_workflow_integration),
        ("Benchmarking", demo_benchmarking),
        ("Serialization", demo_serialization),
        ("Cache Management", demo_cache_management),
        ("Validation", demo_validation)
    ]

    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")

    print("\n" + "-" * 80)
    selection = input("\nSelect demo (1-10, or 'all' to run all): ").strip().lower()

    if selection == 'all':
        for name, demo_func in demos:
            try:
                await demo_func()
                await asyncio.sleep(0.5)  # Brief pause between demos
            except Exception as e:
                logger.error(f"Demo '{name}' failed: {e}", exc_info=True)
    elif selection.isdigit() and 1 <= int(selection) <= len(demos):
        index = int(selection) - 1
        name, demo_func = demos[index]
        try:
            await demo_func()
        except Exception as e:
            logger.error(f"Demo '{name}' failed: {e}", exc_info=True)
    else:
        print("Invalid selection. Running basic usage demo...")
        await demo_basic_usage()

    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
