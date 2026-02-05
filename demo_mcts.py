#!/usr/bin/env python3
"""
LeanAide MCTS Demo

Demonstration script for MCTS proof search capabilities.
Shows various usage patterns and configurations.

Usage:
    python demo_mcts.py                    # Run all demos
    python demo_mcts.py --demo basic       # Run basic demo only
    python demo_mcts.py --demo custom      # Run custom rollout demo
    python demo_mcts.py --demo parallel    # Run parallel MCTS demo
"""

import sys
import argparse
from typing import List, Tuple
from leanaide_mcts import (
    LeanProofMCTS,
    ProofContext,
    Tactic,
    TacticAction,
    MCTS,
    MCTSNode
)


def print_header(title: str):
    """Print demo header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def demo_basic_mcts():
    """Demo 1: Basic MCTS search"""
    print_header("Demo 1: Basic MCTS Search")

    # Create a simple theorem
    context = ProofContext(
        goal="∀ n : Nat, n + 0 = n",
        hypotheses=[],
        available_lemmas=["Nat.add_zero"],
        depth=0
    )

    print(f"\nTheorem: {context.goal}")
    print(f"Available lemmas: {context.available_lemmas}")

    # Initialize MCTS
    mcts = LeanProofMCTS(
        exploration_constant=1.414,
        simulations=100,
        rollout_depth=5
    )

    print("\nRunning MCTS search...")
    best_sequence, root = mcts.search(context)

    # Display results
    print(f"\nBest proof sequence ({len(best_sequence)} tactics):")
    for i, action in enumerate(best_sequence, 1):
        print(f"  {i}. {action.tactic.name}")

    # Statistics
    stats = mcts.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Average time: {stats['average_time']:.2f}s")

    tree_stats = mcts.mcts.get_tree_statistics(root)
    print(f"\nTree statistics:")
    print(f"  Total nodes: {tree_stats['total_nodes']}")
    print(f"  Max depth: {tree_stats['max_depth']}")
    print(f"  Root value: {tree_stats['root_value']:.3f}")


def demo_custom_rollout():
    """Demo 2: Custom rollout policy"""
    print_header("Demo 2: Custom Rollout Policy")

    context = ProofContext(
        goal="∀ n m : Nat, n + m = m + n",
        hypotheses=[],
        available_lemmas=["Nat.add_zero", "Nat.add_succ"],
        depth=0
    )

    print(f"\nTheorem: {context.goal}")

    # Custom action generator with heuristic ordering
    def heuristic_action_generator(ctx):
        """Generate actions with heuristic ordering"""
        tactics = []

        # Safe tactics first
        safe_tactics = ["intro", "simp", "assumption"]
        for name in safe_tactics:
            tactics.append(TacticAction(
                tactic=Tactic(name=name, is_safe=True),
                context=ctx,
                estimated_value=0.8
            ))

        # Other tactics
        other_tactics = ["apply", "rw", "cases"]
        for name in other_tactics:
            tactics.append(TacticAction(
                tactic=Tactic(name=name),
                context=ctx,
                estimated_value=0.5
            ))

        return tactics

    # Custom evaluator
    def custom_evaluator(ctx):
        """Custom state evaluation"""
        if not ctx.goal:
            return 1.0

        value = 0.5
        value -= 0.01 * ctx.depth  # Depth penalty
        value += 0.02 * len(ctx.hypotheses)  # Hypothesis bonus
        return max(0.0, min(1.0, value))

    # Run MCTS with custom components
    mcts = MCTS(
        exploration_constant=1.414,
        rollout_depth=10
    )

    root = MCTSNode(state=context)

    print("\nRunning MCTS with custom rollout policy...")
    for i in range(50):
        node = mcts.select(root)
        actions = heuristic_action_generator(node.state)
        child = mcts.expand(node, actions)
        value = mcts.simulate(child, heuristic_action_generator, custom_evaluator)
        mcts.backpropagate(child, value)

        if i % 10 == 0:
            print(f"  Iteration {i}: root value = {root.average_value:.3f}")

    print(f"\nFinal root value: {root.average_value:.3f}")
    print(f"Total nodes visited: {mcts.total_simulations}")


def demo_configuration_presets():
    """Demo 3: Configuration presets"""
    print_header("Demo 3: Configuration Presets")

    context = ProofContext(
        goal="∀ n : Nat, n * n = n ^ 2",
        hypotheses=[],
        available_lemmas=["Nat.mul_zero", "Nat.pow_zero"],
        depth=0
    )

    print(f"\nTheorem: {context.goal}")

    presets = [
        ("Fast Mode", {"simulations": 50, "exploration_constant": 0.5}),
        ("Balanced Mode", {"simulations": 200, "exploration_constant": 1.414}),
        ("Thorough Mode", {"simulations": 500, "exploration_constant": 2.0})
    ]

    results = []

    for name, config in presets:
        print(f"\n{name}:")
        print(f"  Config: {config}")

        mcts = LeanProofMCTS(**config)
        start_time = __import__('time').time()

        best_sequence, root = mcts.search(context)
        elapsed = __import__('time').time() - start_time

        stats = {
            "name": name,
            "simulations": config["simulations"],
            "time": elapsed,
            "sequence_length": len(best_sequence),
            "root_value": root.average_value
        }

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Sequence length: {len(best_sequence)}")
        print(f"  Root value: {root.average_value:.3f}")

        results.append(stats)

    # Comparison
    print(f"\n{'=' * 80}")
    print("Comparison:")
    print(f"{'Mode':<15s} {'Time':>10s} {'Length':>10s} {'Value':>10s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15s} {r['time']:>10.2f} {r['sequence_length']:>10d} {r['root_value']:>10.3f}")


def demo_action_probabilities():
    """Demo 4: Action probabilities"""
    print_header("Demo 4: Action Probabilities Analysis")

    context = ProofContext(
        goal="∀ n : Nat, n + 0 = n",
        hypotheses=[],
        available_lemmas=["Nat.add_zero"],
        depth=0
    )

    mcts = LeanProofMCTS(
        simulations=200,
        exploration_constant=1.414
    )

    print(f"\nTheorem: {context.goal}")
    print("\nRunning MCTS search...")

    best_sequence, root = mcts.search(context)

    # Get action probabilities at different temperatures
    temperatures = [0.0, 0.5, 1.0, 2.0]

    print(f"\n{'=' * 80}")
    print("Action Probabilities at Different Temperatures:")
    print(f"{'=' * 80}")

    for temp in temperatures:
        probs = mcts.get_action_probabilities(root, temperature=temp)

        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])

        print(f"\nTemperature {temp}:")
        for action_id, prob in sorted_probs[:5]:
            print(f"  {action_id}: {prob:.3f}")


def demo_progressive_search():
    """Demo 5: Progressive search refinement"""
    print_header("Demo 5: Progressive Search Refinement")

    context = ProofContext(
        goal="∀ n m k : Nat, n + (m + k) = (n + m) + k",
        hypotheses=[],
        available_lemmas=["Nat.add_zero", "Nat.add_succ", "Nat.add_assoc"],
        depth=0
    )

    print(f"\nTheorem: {context.goal}")

    # Progressive search with increasing budgets
    budgets = [50, 100, 200, 500]

    print(f"\nProgressive search with increasing budgets:")
    print(f"{'Budget':<10s} {'Time':>10s} {'Value':>10s} {'Nodes':>10s}")
    print("-" * 80)

    previous_best = None

    for budget in budgets:
        mcts = LeanProofMCTS(simulations=budget)
        start_time = __import__('time').time()

        best_sequence, root = mcts.search(context)
        elapsed = __import__('time').time() - start_time

        tree_stats = mcts.mcts.get_tree_statistics(root)

        print(f"{budget:<10d} {elapsed:>10.2f} {root.average_value:>10.3f} {tree_stats['total_nodes']:>10d}")

        if best_sequence != previous_best:
            print(f"    -> New best sequence found (length: {len(best_sequence)})")
            previous_best = best_sequence


def demo_comparison_strategies():
    """Demo 6: Compare different MCTS strategies"""
    print_header("Demo 6: Strategy Comparison")

    context = ProofContext(
        goal="∀ n : Nat, n + 0 = n",
        hypotheses=[],
        available_lemmas=["Nat.add_zero"],
        depth=0
    )

    print(f"\nTheorem: {context.goal}")

    strategies = [
        ("Aggressive", 0.5, 100),
        ("Balanced", 1.414, 100),
        ("Conservative", 2.0, 100)
    ]

    results = []

    for name, exploration, simulations in strategies:
        mcts = LeanProofMCTS(
            exploration_constant=exploration,
            simulations=simulations
        )

        start_time = __import__('time').time()
        best_sequence, root = mcts.search(context)
        elapsed = __import__('time').time() - start_time

        tree_stats = mcts.mcts.get_tree_statistics(root)

        result = {
            "name": name,
            "exploration": exploration,
            "time": elapsed,
            "value": root.average_value,
            "nodes": tree_stats["total_nodes"],
            "max_depth": tree_stats["max_depth"]
        }

        results.append(result)

    # Print comparison
    print(f"\n{'=' * 80}")
    print("Strategy Comparison:")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<15s} {'Expl':>8s} {'Time':>10s} {'Value':>10s} {'Nodes':>10s} {'Depth':>8s}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<15s} {r['exploration']:>8.3f} {r['time']:>10.2f} {r['value']:>10.3f} {r['nodes']:>10d} {r['max_depth']:>8d}")


def run_all_demos():
    """Run all demos"""
    demos = [
        demo_basic_mcts,
        demo_custom_rollout,
        demo_configuration_presets,
        demo_action_probabilities,
        demo_progressive_search,
        demo_comparison_strategies
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n[FAIL] Demo failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="LeanAide MCTS Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_mcts.py                    # Run all demos
  python demo_mcts.py --demo basic       # Run basic demo only
  python demo_mcts.py --demo custom      # Run custom rollout demo
        """
    )

    parser.add_argument(
        "--demo",
        choices=["basic", "custom", "presets", "probabilities", "progressive", "comparison"],
        help="Run specific demo"
    )

    args = parser.parse_args()

    # Map demo names to functions
    demo_map = {
        "basic": demo_basic_mcts,
        "custom": demo_custom_rollout,
        "presets": demo_configuration_presets,
        "probabilities": demo_action_probabilities,
        "progressive": demo_progressive_search,
        "comparison": demo_comparison_strategies
    }

    # Run selected demo or all demos
    if args.demo:
        demo_map[args.demo]()
    else:
        run_all_demos()

    print("\n" + "=" * 80)
    print("Demo completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
