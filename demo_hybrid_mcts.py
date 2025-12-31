#!/usr/bin/env python3
"""
Hybrid MCTS Framework - Demonstration Script

This script demonstrates all three hybrid MCTS approaches with examples
and showcases the unified framework's capabilities.

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import sys
from typing import List

from hybrid_mcts_framework import (
    HybridMCTSConfig,
    HybridMCTSEngine,
    HybridMCTSApproach,
    HybridMCTSPresets,
    HybridMCTSWorkflowIntegrator,
    AdaptiveHybridSelector,
    HybridBenchmark,
    print_result_summary,
    create_framework_from_preset,
    quick_search,
    thorough_search,
)


# Sample theorems for demonstration
SAMPLE_THEOREMS = [
    {
        "name": "Additive Identity",
        "statement": "theorem add_zero (a : nat) : a + 0 = a",
        "difficulty": "easy",
    },
    {
        "name": "Commutativity of Addition",
        "statement": "theorem add_comm (a b : nat) : a + b = b + a",
        "difficulty": "easy",
    },
    {
        "name": "Associativity of Addition",
        "statement": "theorem add_assoc (a b c : nat) : (a + b) + c = a + (b + c)",
        "difficulty": "medium",
    },
    {
        "name": "Multiplicative Identity",
        "statement": "theorem mul_one (a : nat) : a * 1 = a",
        "difficulty": "easy",
    },
    {
        "name": "Distributive Law",
        "statement": "theorem mul_add (a b c : nat) : a * (b + c) = a * b + a * c",
        "difficulty": "medium",
    },
]


def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_subheader(title: str):
    """Print a formatted subheader"""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70 + "\n")


async def demo_basic_usage():
    """Demonstrate basic usage of the framework"""
    print_header("DEMO 1: Basic Usage")

    print("Creating engine with balanced preset...")
    engine = create_framework_from_preset("balanced")

    theorem = SAMPLE_THEOREMS[0]["statement"]
    print(f"Theorem: {theorem}\n")

    print("Running hybrid MCTS search...")
    result = await engine.search(theorem)

    print_result_summary(result)

    # Show additional details
    print("Additional Details:")
    print(f"  Approach Used: {result.approach_used.value}")
    print(f"  Nodes Explored: {result.nodes_explored}")
    print(f"  Generations: {result.generations_completed}")

    if result.cache_hits > 0 or result.cache_misses > 0:
        print(f"  Cache Hits: {result.cache_hits}")
        print(f"  Cache Misses: {result.cache_misses}")

    return result


async def demo_evolved_policies():
    """Demonstrate evolved policies approach"""
    print_header("DEMO 2: Evolved Rollout Policies")

    print("Configuration for Evolved Policies:")
    config = HybridMCTSPresets.evolved_policies_only()
    print(f"  Policy Population: {config.policy_population_size}")
    print(f"  Policy Generations: {config.policy_generations}")
    print(f"  Adaptation Interval: {config.policy_adaptation_interval}")

    engine = HybridMCTSEngine(config)

    theorem = SAMPLE_THEOREMS[1]["statement"]
    print(f"\nTheorem: {theorem}\n")

    result = await engine.search(theorem)

    print("\nEvolved Policies Results:")
    print(f"  Best Fitness: {result.best_fitness:.4f}")
    print(f"  Execution Time: {result.execution_time:.2f}s")

    if result.policy_fitness_history:
        print(f"\n  Policy Evolution:")
        for i, fitness in enumerate(result.policy_fitness_history[::2]):  # Every other
            print(f"    Generation {i*2}: {fitness:.4f}")

    if result.final_policy:
        print(f"\n  Final Policy Structure: {type(result.final_policy).__name__}")

    print(f"\n  Policy Adaptations: {result.policy_adaptations}")
    print(f"  Policy Diversity: {result.policy_diversity:.4f}")

    return result


async def demo_evolutionary_nodes():
    """Demonstrate evolutionary nodes approach"""
    print_header("DEMO 3: Evolutionary MCTS Nodes")

    print("Configuration for Evolutionary Nodes:")
    config = HybridMCTSPresets.evolutionary_nodes_only()
    print(f"  Node Population: {config.node_population_size}")
    print(f"  Node Evolution Generations: {config.node_evolution_generations}")
    print(f"  Convergence Threshold: {config.node_convergence_threshold}")

    engine = HybridMCTSEngine(config)

    theorem = SAMPLE_THEOREMS[2]["statement"]
    print(f"\nTheorem: {theorem}\n")

    result = await engine.search(theorem)

    print("\nEvolutionary Nodes Results:")
    print(f"  Best Fitness: {result.best_fitness:.4f}")
    print(f"  Execution Time: {result.execution_time:.2f}")
    print(f"  Total Node Evaluations: {result.total_node_evaluations}")
    print(f"  Converged Nodes: {result.converged_nodes}")
    print(f"  Node Diversity: {result.node_diversity:.4f}")

    if result.node_convergence_history:
        print(f"\n  Node Convergence (first 3 nodes):")
        for node_name, history in list(result.node_convergence_history.items())[:3]:
            print(f"    {node_name}: {history[-1]:.4f}")

    return result


async def demo_coevolution():
    """Demonstrate coevolution approach"""
    print_header("DEMO 4: Coevolving Decision Trees")

    print("Configuration for Coevolution:")
    config = HybridMCTSPresets.coevolution_only()
    print(f"  Tree Population: {config.tree_population_size}")
    print(f"  Tree Generations: {config.tree_generations}")
    print(f"  Max Tree Depth: {config.tree_max_depth}")
    print(f"  Pareto Front Size: {config.pareto_front_size}")

    engine = HybridMCTSEngine(config)

    theorem = SAMPLE_THEOREMS[3]["statement"]
    print(f"\nTheorem: {theorem}\n")

    result = await engine.search(theorem)

    print("\nCoevolution Results:")
    print(f"  Best Fitness: {result.best_fitness:.4f}")
    print(f"  Execution Time: {result.execution_time:.2f}s")
    print(f"  Coevolution Cycles: {result.coevolution_cycles}")

    if result.tree_depth_stats:
        stats = result.tree_depth_stats
        print(f"\n  Tree Depth Statistics:")
        print(f"    Min: {stats.get('min', 'N/A')}")
        print(f"    Max: {stats.get('max', 'N/A')}")
        print(f"    Mean: {stats.get('mean', 'N/A')}")

    if result.pareto_front:
        print(f"\n  Pareto Front (showing first 5):")
        for i, individual in enumerate(result.pareto_front[:5]):
            complexity = individual.get('complexity', 'N/A')
            fitness = individual.get('fitness', 'N/A')
            print(f"    {i+1}. Complexity: {complexity}, Fitness: {fitness:.4f}")

    return result


async def demo_adaptive_selection():
    """Demonstrate adaptive approach selection"""
    print_header("DEMO 5: Adaptive Approach Selection")

    print("Configuration for Adaptive Selection:")
    config = HybridMCTSPresets.research()
    print(f"  Adaptive Enabled: {config.adaptive_enabled}")
    print(f"  Warmup Runs: {config.adaptive_warmup_runs}")

    selector = AdaptiveHybridSelector()
    engine = HybridMCTSEngine(config)
    engine.selector = selector

    print("\nRunning adaptive selection on multiple theorems...\n")

    results = []
    for i, thm in enumerate(SAMPLE_THEOREMS[:4], 1):
        print(f"Theorem {i}: {thm['name']}")
        print(f"  Statement: {thm['statement'][:60]}...")
        print(f"  Difficulty: {thm['difficulty']}")

        result = await engine.search(thm["statement"])
        results.append(result)

        print(f"  Selected Approach: {result.approach_used.value}")
        print(f"  Fitness: {result.best_fitness:.4f}")
        print(f"  Time: {result.execution_time:.2f}s\n")

    # Show selector statistics
    print("Adaptive Selector Statistics:")
    stats = selector.get_statistics()
    print(f"  Total Selections: {stats['total_selections']}")
    print(f"  Approach Runs: {stats['approach_runs']}")

    for key, value in stats.items():
        if 'performance' in key:
            approach = key.replace('_avg_performance', '').replace('_', ' ').title()
            print(f"  {approach} Avg Performance: {value:.4f}")

    return results


async def demo_combined_search():
    """Demonstrate combined search approach"""
    print_header("DEMO 6: Combined Hybrid Search")

    print("Configuration for Combined Search:")
    config = HybridMCTSPresets.thorough()
    print(f"  Combination Strategy: {config.combination_strategy}")
    print(f"  Approaches: Evolved Policies + Evolutionary Nodes + Coevolution")

    engine = HybridMCTSEngine(config)

    theorem = SAMPLE_THEOREMS[4]["statement"]
    print(f"\nTheorem: {theorem}\n")

    print("Running all three approaches in parallel...")
    result = await engine.search(theorem)

    print("\nCombined Search Results:")
    print(f"  Success: {result.success}")
    print(f"  Best Fitness: {result.best_fitness:.4f}")
    print(f"  Execution Time: {result.execution_time:.2f}s")
    print(f"  Combination Method: {result.metadata.get('combination_method', 'unknown')}")

    if result.approach_used == HybridMCTSApproach.COMBINED:
        print("\n  Combined approach integrated results from:")
        print("    - Evolved Rollout Policies")
        print("    - Evolutionary MCTS Nodes")
        print("    - Coevolving Decision Trees")

    return result


async def demo_workflow_integration():
    """Demonstrate workflow integration"""
    print_header("DEMO 7: Workflow Integration")

    config = HybridMCTSPresets.balanced()
    integrator = HybridMCTSWorkflowIntegrator(config)

    # Create sample subproblems
    subproblems = [
        {
            "id": f"subproblem_{i}",
            "statement": thm["statement"],
            "domain": "algebra",
            "difficulty": thm["difficulty"],
            "dependencies": [],
        }
        for i, thm in enumerate(SAMPLE_THEOREMS[:3])
    ]

    print(f"Created {len(subproblems)} subproblems\n")

    print("Solving subproblems sequentially...")
    solutions = []
    for sp in subproblems:
        print(f"\n  Solving: {sp['id']}")
        print(f"  Statement: {sp['statement'][:60]}...")

        solution = await integrator.solve_subproblem(sp)
        solutions.append(solution)

        print(f"  Success: {solution['success']}")
        print(f"  Approach: {solution['approach']}")
        print(f"  Fitness: {solution['fitness']:.4f}")

    print(f"\n\nBatch Processing:")
    print(f"  Total subproblems: {len(subproblems)}")
    print(f"  Successful: {sum(1 for s in solutions if s['success'])}")
    print(f"  Failed: {sum(1 for s in solutions if not s['success'])}")

    avg_fitness = sum(s['fitness'] for s in solutions) / len(solutions)
    print(f"  Average Fitness: {avg_fitness:.4f}")

    return solutions


async def demo_quick_utilities():
    """Demonstrate quick utility functions"""
    print_header("DEMO 8: Quick Utility Functions")

    print("1. Quick Search (Fast Preset):")
    result = await quick_search(SAMPLE_THEOREMS[0]["statement"])
    print(f"   Fitness: {result.best_fitness:.4f}")
    print(f"   Time: {result.execution_time:.2f}s")
    print(f"   Approach: {result.approach_used.value}")

    print("\n2. Thorough Search (Thorough Preset):")
    result = await thorough_search(SAMPLE_THEOREMS[1]["statement"])
    print(f"   Fitness: {result.best_fitness:.4f}")
    print(f"   Time: {result.execution_time:.2f}s")
    print(f"   Approach: {result.approach_used.value}")

    print("\n3. Preset Comparisons:")

    presets = ["fast", "balanced", "thorough"]
    theorem = SAMPLE_THEOREMS[2]["statement"]

    for preset in presets:
        engine = create_framework_from_preset(preset)
        result = await engine.search(theorem)
        print(f"\n   {preset.title()} Preset:")
        print(f"     Fitness: {result.best_fitness:.4f}")
        print(f"     Time: {result.execution_time:.2f}s")


async def demo_comparison():
    """Demonstrate approach comparison"""
    print_header("DEMO 9: Approach Comparison")

    config = HybridMCTSPresets.fast()
    benchmark = HybridBenchmark(config)

    test_theorems = [thm["statement"] for thm in SAMPLE_THEOREMS[:3]]

    print(f"Benchmarking {len(test_theorems)} theorems...\n")

    print("Running benchmark (this may take a while)...")
    comparison = await benchmark.benchmark_all(test_theorems)

    print("\nComparison Results:")
    print(f"  Best Overall Approach: {comparison.best_overall.value}")
    print(f"  Fastest Approach: {comparison.fastest.value}")
    print(f"  Most Reliable: {comparison.most_reliable.value}")

    print("\nApproach Performance:")
    for approach, bench in comparison.approaches.items():
        print(f"\n  {approach.value}:")
        print(f"    Success Rate: {bench.success_rate:.2%}")
        print(f"    Avg Fitness: {bench.avg_fitness:.4f}")
        print(f"    Avg Time: {bench.avg_time:.2f}s")
        print(f"    Best Fitness: {bench.best_fitness:.4f}")

    print("\nRecommendations:")
    for i, rec in enumerate(comparison.recommendations, 1):
        print(f"  {i}. {rec}")

    return comparison


async def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "HYBRID MCTS FRAMEWORK DEMONSTRATIONS" + " " * 15 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Evolved Policies", demo_evolved_policies),
        ("Evolutionary Nodes", demo_evolutionary_nodes),
        ("Coevolution", demo_coevolution),
        ("Adaptive Selection", demo_adaptive_selection),
        ("Combined Search", demo_combined_search),
        ("Workflow Integration", demo_workflow_integration),
        ("Quick Utilities", demo_quick_utilities),
        ("Approach Comparison", demo_comparison),
    ]

    print(f"\nRunning {len(demos)} demonstrations...\n")

    results = {}
    for name, demo_func in demos:
        try:
            result = await demo_func()
            results[name] = result
            await asyncio.sleep(0.5)  # Brief pause between demos
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print_header("DEMONSTRATION COMPLETE")

    print("All demonstrations completed successfully!")
    print(f"\nExecuted {len(results)} demonstrations:")
    for name in results.keys():
        print(f"  - {name}")

    print("\n" + "#" * 70)
    print("# Thank you for using the Hybrid MCTS Framework!")
    print("#" * 70 + "\n")


async def run_interactive_demo():
    """Interactive demo allowing user to select demonstrations"""
    print("\n" + "#" * 70)
    print("#" + " " * 18 + "HYBRID MCTS FRAMEWORK - INTERACTIVE" + " " * 18 + "#")
    print("#" * 70)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Evolved Policies", demo_evolved_policies),
        ("Evolutionary Nodes", demo_evolutionary_nodes),
        ("Coevolution", demo_coevolution),
        ("Adaptive Selection", demo_adaptive_selection),
        ("Combined Search", demo_combined_search),
        ("Workflow Integration", demo_workflow_integration),
        ("Quick Utilities", demo_quick_utilities),
        ("Approach Comparison", demo_comparison),
        ("Run All Demos", run_all_demos),
    ]

    while True:
        print("\nAvailable Demonstrations:")
        for i, (name, _) in enumerate(demos, 1):
            print(f"  {i}. {name}")
        print("  0. Exit")

        try:
            choice = input("\nSelect a demonstration (0-{}): ".format(len(demos)))
            choice = int(choice)

            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(demos):
                name, demo_func = demos[choice - 1]
                print(f"\nRunning: {name}")
                try:
                    await demo_func()
                except Exception as e:
                    print(f"\nERROR: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("Invalid choice. Please try again.")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            break


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hybrid MCTS Framework Demonstration"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--demo", "-d",
        type=str,
        choices=[
            "all", "basic", "evolved", "nodes", "coevolution",
            "adaptive", "combined", "workflow", "utilities", "comparison"
        ],
        default="all",
        help="Specific demonstration to run"
    )

    args = parser.parse_args()

    if args.interactive:
        asyncio.run(run_interactive_demo())
    else:
        demo_map = {
            "all": run_all_demos,
            "basic": demo_basic_usage,
            "evolved": demo_evolved_policies,
            "nodes": demo_evolutionary_nodes,
            "coevolution": demo_coevolution,
            "adaptive": demo_adaptive_selection,
            "combined": demo_combined_search,
            "workflow": demo_workflow_integration,
            "utilities": demo_quick_utilities,
            "comparison": demo_comparison,
        }

        demo_func = demo_map[args.demo]
        asyncio.run(demo_func())


if __name__ == "__main__":
    main()
