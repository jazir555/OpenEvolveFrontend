"""
MAKER/MDAP Hybrid Strategies - Demo

This script demonstrates the MAKER/MDAP integration with hybrid strategies,
combining zero-error voting and task decomposition with MCTS, Evolution, and
Adversarial testing.

Features demonstrated:
1. MCTS-Then-MAKER: MCTS exploration with MAKER voting refinement
2. MAKER-Then-Evolution: MAKER-generated population with evolution
3. MAKER-Adversarial: MAKER voting with red/blue team testing
4. Adaptive MAKER: Dynamic strategy switching
5. MAKER-MDAP Parallel: Parallel execution for efficiency
6. Full MAKER Hybrid: Complete integration of all components

Usage:
    python demo_hybrid_maker.py
"""

import asyncio
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


async def demo_1_mcts_then_maker():
    """Demo 1: MCTS-Then-MAKER"""
    print_section("DEMO 1: MCTS-Then-MAKER")

    from hybrid_maker_integration import MCTSThenMAKER

    theorem = "forall n m : nat, n + m = m + n"

    print(f"Theorem: {theorem}")
    print("Strategy: MCTS exploration + MAKER voting refinement")

    strategy = MCTSThenMAKER(
        mcts_simulations=50,
        maker_voting_threshold=3,
        population_size=10
    )

    result = await strategy.generate_proof(theorem)

    print("\n[Result]")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Time: {result.evolution_time:.2f}s")
    print(f"  Generations: {result.generations_completed}")

    if result.best_proof:
        print(f"\n  Best proof (first 200 chars):\n    {result.best_proof[:200]}...")

    return result


async def demo_2_maker_then_evolution():
    """Demo 2: MAKER-Then-Evolution"""
    print_section("DEMO 2: MAKER-Then-Evolution")

    from hybrid_maker_integration import MAKERThenEvolution

    theorem = "forall n : nat, n + 0 = n"

    print(f"Theorem: {theorem}")
    print("Strategy: MAKER voting generates population + Evolution refines")

    strategy = MAKERThenEvolution(
        maker_voting_threshold=3,
        evolution_generations=10,
        population_size=15,
        initial_candidates=30
    )

    result = await strategy.generate_proof(theorem)

    print("\n[Result]")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Time: {result.evolution_time:.2f}s")
    print(f"  Generations: {result.generations_completed}")

    if result.convergence_history:
        print(f"\n  Convergence (first 5 generations):")
        for i, fit in enumerate(result.convergence_history[:5]):
            print(f"    Gen {i}: {fit:.3f}")

    return result


async def demo_3_maker_adversarial():
    """Demo 3: MAKER-Adversarial Hybrid"""
    print_section("DEMO 3: MAKER-Adversarial Hybrid")

    from hybrid_maker_integration import MAKERAdversarialHybrid

    theorem = "forall a b c : nat, a + (b + c) = (a + b) + c"

    print(f"Theorem: {theorem}")
    print("Strategy: Red team attacks + Blue team defenses + MAKER voting")

    strategy = MAKERAdversarialHybrid(
        adversarial_rounds=3,
        maker_voting_threshold=3,
        red_team_size=2,
        blue_team_size=2
    )

    result = await strategy.generate_proof(theorem)

    print("\n[Result]")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Time: {result.evolution_time:.2f}s")
    print(f"  Adversarial rounds: {result.generations_completed}")

    if result.convergence_history:
        print(f"\n  Adversarial progression:")
        for i, fit in enumerate(result.convergence_history):
            print(f"    Round {i+1}: {fit:.3f}")

    return result


async def demo_4_adaptive_maker():
    """Demo 4: Adaptive MAKER Hybrid"""
    print_section("DEMO 4: Adaptive MAKER Hybrid")

    from hybrid_maker_integration import AdaptiveMAKERHybrid

    theorem = "forall n : nat, n * 1 = n"

    print(f"Theorem: {theorem}")
    print("Strategy: Dynamic switching between MAKER, MCTS, and Evolution")

    strategy = AdaptiveMAKERHybrid(
        diversity_threshold=0.3,
        convergence_threshold=0.95,
        max_generations=20
    )

    result = await strategy.generate_proof(theorem)

    print("\n[Result]")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Time: {result.evolution_time:.2f}s")
    print(f"  Generations: {result.generations_completed}")

    if result.convergence_history:
        print(f"\n  Convergence over generations:")
        print(f"    Start: {result.convergence_history[0]:.3f}")
        print(f"    Middle: {result.convergence_history[len(result.convergence_history)//2]:.3f}")
        print(f"    End: {result.convergence_history[-1]:.3f}")

    return result


async def demo_5_maker_mdap_parallel():
    """Demo 5: MAKER-MDAP Parallel"""
    print_section("DEMO 5: MAKER-MDAP Parallel")

    from hybrid_maker_integration import MAKERMDAPParallel

    theorem = "forall n m : nat, n + m = m + n"

    print(f"Theorem: {theorem}")
    print("Strategy: Parallel MAKER voting and MDAP decomposition")

    strategy = MAKERMDAPParallel(
        maker_voting_threshold=3,
        mdap_agents=4,
        combination_method="best_fitness"
    )

    result = await strategy.generate_proof(theorem)

    print("\n[Result]")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Time: {result.evolution_time:.2f}s")
    print(f"  Combination method: best_fitness")

    if result.best_proof:
        print(f"\n  Best proof (first 150 chars):\n    {result.best_proof[:150]}...")

    return result


async def demo_6_full_maker_hybrid():
    """Demo 6: Full MAKER Hybrid"""
    print_section("DEMO 6: Full MAKER Hybrid (All Components)")

    from hybrid_maker_integration import FullMAKERHybrid, MAKERHybridConfig

    theorem = "forall n m : nat, n + m = m + n"

    print(f"Theorem: {theorem}")
    print("Strategy: Complete MAKER framework with all components")
    print("  - MAKER voting")
    print("  - MDAP decomposition")
    print("  - MCTS exploration")
    print("  - Evolution optimization")
    print("  - Adversarial testing")
    print("  - Adaptive switching")
    print("  - Parallel execution")

    config = MAKERHybridConfig(
        enable_voting=True,
        voting_threshold=3,
        enable_decomposition=True,
        mcts_simulations=30,
        evolution_generations=10,
        adversarial_rounds=2,
        adaptive_switching=True
    )

    strategy = FullMAKERHybrid(config)

    result = await strategy.generate_proof(theorem)

    print("\n[Result]")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Time: {result.evolution_time:.2f}s")
    print(f"  Total generations: {result.generations_completed}")

    if result.best_proof:
        print(f"\n  Best proof (first 200 chars):\n    {result.best_proof[:200]}...")

    return result


async def demo_7_mode_comparison():
    """Demo 7: Compare all hybrid modes"""
    print_section("DEMO 7: Hybrid Mode Comparison")

    from hybrid_maker_integration import (
        run_maker_hybrid,
        MAKERHybridMode,
        MAKERHybridConfig
    )

    theorem = "forall n : nat, n + 0 = n"

    print(f"Theorem: {theorem}")
    print("Comparing all MAKER hybrid modes...\n")

    modes = [
        (MAKERHybridMode.MCTS_THEN_MAKER, "MCTS Then MAKER"),
        (MAKERHybridMode.MAKER_THEN_EVOLUTION, "MAKER Then Evolution"),
        (MAKERHybridMode.ADAPTIVE_MAKER, "Adaptive MAKER"),
    ]

    config = MAKERHybridConfig(
        voting_threshold=3,
        evolution_generations=5,
        mcts_simulations=30
    )

    results = []

    for mode, mode_name in modes:
        print(f"\n  Testing: {mode_name}...")

        try:
            result = await run_maker_hybrid(
                theorem=theorem,
                mode=mode,
                config=config
            )

            results.append({
                "mode": mode_name,
                "success": result.success,
                "fitness": result.best_fitness,
                "time": result.evolution_time
            })

            print(f"    Success: {result.success}")
            print(f"    Fitness: {result.best_fitness:.3f}")
            print(f"    Time: {result.evolution_time:.2f}s")

        except Exception as e:
            logger.error(f"Mode {mode_name} failed: {e}")
            results.append({
                "mode": mode_name,
                "success": False,
                "fitness": 0.0,
                "time": 0.0
            })

    # Summary table
    print("\n[Summary]")
    print("  Mode                  | Success | Fitness | Time")
    print("  ----------------------|---------|---------|------")
    for r in results:
        success_str = "[OK]" if r["success"] else "[FAIL]"
        print(f"  {r['mode']:20s} | {success_str:>7} | {r['fitness']:7.3f} | {r['time']:4.2f}s")

    return results


async def demo_8_capabilities():
    """Demo 8: Check MAKER hybrid capabilities"""
    print_section("DEMO 8: MAKER Hybrid Capabilities Check")

    from hybrid_maker_integration import get_maker_hybrid_capabilities

    capabilities = get_maker_hybrid_capabilities()

    print("MAKER Hybrid Integration Capabilities:")
    print(f"  - MAKER hybrid enabled: {capabilities.get('maker_hybrid_enabled', False)}")
    print(f"  - MAKER evolution: {capabilities.get('maker_evolution_available', False)}")
    print(f"  - MAKER adversarial: {capabilities.get('maker_adversarial_available', False)}")
    print(f"  - MAKER core: {capabilities.get('maker_core_available', False)}")
    print(f"  - MDAP: {capabilities.get('mdap_available', False)}")
    print(f"  - MCTS: {capabilities.get('mcts_available', False)}")
    print(f"  - Evolution: {capabilities.get('evolution_available', False)}")
    print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")

    print("\n  Available Modes:")
    for mode in capabilities.get('modes', []):
        print(f"    - {mode}")

    print("\n  Available Strategies:")
    for strategy in capabilities.get('strategies', []):
        print(f"    - {strategy}")

    if 'paper' in capabilities:
        paper = capabilities['paper']
        print(f"\n  Paper Reference:")
        print(f"    - Title: {paper.get('title', 'N/A')}")
        print(f"    - arXiv: {paper.get('arxiv', 'N/A')}")
        print(f"    - URL: {paper.get('url', 'N/A')}")

    return capabilities


async def main():
    """Run all demos."""
    print("\n")
    print("=" * 80)
    print("  MAKER/MDAP HYBRID STRATEGIES - DEMONSTRATION")
    print("  Paper: arXiv:2511.09030 (Solving a Million-Step LLM Task with Zero Errors)")
    print("=" * 80)
    print("")

    demos = [
        ("MCTS-Then-MAKER", demo_1_mcts_then_maker),
        ("MAKER-Then-Evolution", demo_2_maker_then_evolution),
        ("MAKER-Adversarial Hybrid", demo_3_maker_adversarial),
        ("Adaptive MAKER Hybrid", demo_4_adaptive_maker),
        ("MAKER-MDAP Parallel", demo_5_maker_mdap_parallel),
        ("Full MAKER Hybrid", demo_6_full_maker_hybrid),
        ("Mode Comparison", demo_7_mode_comparison),
        ("Capabilities Check", demo_8_capabilities),
    ]

    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all demos")
    print("")

    try:
        choice = input("Select demo (0-8, or press Enter for all): ").strip()
        if not choice:
            choice = "0"

        choice_num = int(choice)

        if choice_num == 0:
            # Run all demos
            for name, demo_func in demos:
                try:
                    if asyncio.iscoroutinefunction(demo_func):
                        await demo_func()
                    else:
                        demo_func()
                except Exception as e:
                    logger.error(f"Demo {name} failed: {e}", exc_info=True)
        elif 1 <= choice_num <= len(demos):
            # Run selected demo
            name, demo_func = demos[choice_num - 1]
            if asyncio.iscoroutinefunction(demo_func):
                await demo_func()
            else:
                demo_func()
        else:
            print("Invalid choice")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("  DEMO COMPLETED")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - hybrid_maker_integration.py")
    print("  - MAKER_HYBRID_INTEGRATION_GUIDE.md")
    print("  - Paper: https://arxiv.org/abs/2511.09030")
    print("")


if __name__ == "__main__":
    asyncio.run(main())
