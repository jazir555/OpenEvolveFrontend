#!/usr/bin/env python3
"""
MDAP-Enhanced Evolution Demo Script

Demonstrates MDAP-enhanced evolutionary computation for Lean 4 proof generation.

Usage:
    python demo_evolution_mdap.py                           # Run all demos
    python demo_evolution_mdap.py --demo basic              # Run basic demo only
    python demo_evolution_mdap.py --demo voting             # Run voting demo
    python demo_evolution_mdap.py --demo comparison         # Run comparison demo

Author: OpenEvolve Frontend Team
Date: 2025-12-30
"""

import sys
import random
import logging
from typing import Dict, List, Callable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    from evolution_maker_integration import (
        run_maker_evolution,
        MakerevolutionConfig,
        MakerevolutionMode,
        get_maker_evolution_capabilities,
        Individual,
        Population
    )
    EVOLUTION_MAKER_AVAILABLE = True
except ImportError as e:
    logger.error(f"Evolution-MAKER integration not available: {e}")
    EVOLUTION_MAKER_AVAILABLE = False


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def check_capabilities():
    """Check MAKER evolution capabilities"""
    print("\n" + "=" * 70)
    print("CHECKING CAPABILITIES")
    print("=" * 70)

    if not EVOLUTION_MAKER_AVAILABLE:
        print("[FAIL] Evolution-MAKER integration not available")
        print("   Please ensure evolution_maker_integration.py is installed")
        return False

    caps = get_maker_evolution_capabilities()

    print("\nComponent Availability:")
    for component, available in caps.items():
        status = "[OK]" if available else "[FAIL]"
        print(f"  {status} {component}")

    if not caps.get("full_integration", False):
        print("\n[WARN]  Full integration not available")
        print("   Some demos may not work correctly")
        return False

    print("\n[OK] All components available")
    return True


def demo_basic_mdap_evolution():
    """Demo 1: Basic MDAP-enhanced evolution"""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic MDAP-Enhanced Evolution")
    print("=" * 70)

    # Simple fitness evaluator
    def evaluator(genome: str) -> float:
        """Higher fitness is better"""
        score = 0.0
        if "intros" in genome:
            score += 3.0
        if "refl" in genome:
            score += 3.0
        if "simp" in genome:
            score += 1.0
        return score

    # Initial program
    initial_program = "intros n refl"

    # Configure evolution
    config = MakerevolutionConfig(
        mode=MakerevolutionMode.HYBRID,
        voting_threshold=3,
        population_size=20
    )

    print(f"\nInitial program: {initial_program}")
    print(f"Configuration: {config.mode.value}, k={config.voting_threshold}, pop={config.population_size}")

    # Run evolution
    print("\nRunning evolution...")
    result = run_maker_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        max_generations=20,
        config=config
    )

    # Display results
    print("\nResults:")
    print(f"  Best fitness: {result['best_fitness']:.3f}")
    print(f"  Generations completed: {result['generations_completed']}")
    print(f"  Converged: {result['converged']}")
    print(f"  Best program: {result['best_program']}")


def demo_voting_only():
    """Demo 2: Voting-only mode"""
    print("\n" + "=" * 70)
    print("DEMO 2: Voting-Only Mode")
    print("=" * 70)

    def evaluator(genome: str) -> float:
        return 5.0 if "intros refl" in genome else 2.0

    initial_program = "intros n sorry"

    config = MakerevolutionConfig(
        mode=MakerevolutionMode.VOTING_ONLY,
        voting_threshold=3,
        population_size=15
    )

    print(f"\nMode: Voting-only (no decomposition)")
    print(f"Voting threshold: k={config.voting_threshold}")

    result = run_maker_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        max_generations=15,
        config=config
    )

    print("\nResults:")
    print(f"  Best fitness: {result['best_fitness']:.3f}")
    print(f"  Best program: {result['best_program']}")


def demo_decomposition_only():
    """Demo 3: Decomposition-only mode"""
    print("\n" + "=" * 70)
    print("DEMO 3: Decomposition-Only Mode")
    print("=" * 70)

    def evaluator(genome: str) -> float:
        score = 0.0
        if "intros" in genome:
            score += 2.0
        if any(t in genome for t in ["refl", "simp", "induction"]):
            score += 2.0
        return score

    initial_program = "intros n induction n case n=0 case n=succ"

    config = MakerevolutionConfig(
        mode=MakerevolutionMode.DECOMPOSITION,
        enable_voting=False,
        enable_decomposition=True,
        decomposition_depth=3,
        population_size=20
    )

    print(f"\nMode: Decomposition-only (no voting)")
    print(f"Decomposition depth: {config.decomposition_depth}")

    result = run_maker_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        max_generations=15,
        config=config
    )

    print("\nResults:")
    print(f"  Best fitness: {result['best_fitness']:.3f}")
    print(f"  Best program: {result['best_program']}")


def demo_voting_threshold_comparison():
    """Demo 4: Compare different voting thresholds"""
    print("\n" + "=" * 70)
    print("DEMO 4: Voting Threshold Comparison")
    print("=" * 70)

    def evaluator(genome: str) -> float:
        return 5.0 if "intros refl" in genome else 2.0

    initial_program = "intros n sorry"

    thresholds = [2, 3, 5]
    results = {}

    print(f"\nTesting thresholds: {thresholds}")
    print(f"Initial program: {initial_program}")

    for k in thresholds:
        print(f"\nRunning with k={k}...")

        config = MakerevolutionConfig(
            voting_threshold=k,
            population_size=20
        )

        result = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=15,
            config=config
        )

        results[k] = result

        print(f"  Fitness: {result['best_fitness']:.3f}")
        print(f"  Generations: {result['generations_completed']}")

    # Comparison
    print("\n" + "-" * 70)
    print("Summary:")
    for k, result in results.items():
        print(f"  k={k}: fitness={result['best_fitness']:.3f}, gens={result['generations_completed']}")


def demo_pure_vs_mdap_comparison():
    """Demo 5: Compare pure evolution vs MDAP-enhanced"""
    print("\n" + "=" * 70)
    print("DEMO 5: Pure vs MDAP-Enhanced Evolution")
    print("=" * 70)

    def evaluator(genome: str) -> float:
        score = 0.0
        if "intros" in genome:
            score += 2.0
        if "refl" in genome:
            score += 3.0
        if "simp" in genome:
            score += 1.0
        return score

    initial_program = "intros n sorry"

    # Pure evolution
    print("\nRunning Pure Evolution...")
    pure_config = MakerevolutionConfig(
        enable_voting=False,
        enable_decomposition=False,
        population_size=20
    )

    pure_result = run_maker_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        max_generations=20,
        config=pure_config
    )

    print(f"  Fitness: {pure_result['best_fitness']:.3f}")
    print(f"  Generations: {pure_result['generations_completed']}")

    # MDAP-enhanced evolution
    print("\nRunning MDAP-Enhanced Evolution...")
    mdap_config = MakerevolutionConfig(
        mode=MakerevolutionMode.HYBRID,
        voting_threshold=3,
        population_size=20
    )

    mdap_result = run_maker_evolution(
        initial_program=initial_program,
        evaluator=evaluator,
        max_generations=20,
        config=mdap_config
    )

    print(f"  Fitness: {mdap_result['best_fitness']:.3f}")
    print(f"  Generations: {mdap_result['generations_completed']}")

    # Comparison
    print("\n" + "-" * 70)
    print("Comparison:")
    improvement = mdap_result['best_fitness'] - pure_result['best_fitness']
    print(f"  Fitness improvement: {improvement:+.3f}")
    print(f"  Pure best: {pure_result['best_program']}")
    print(f"  MDAP best: {mdap_result['best_program']}")


def demo_lean4_proof_evolution():
    """Demo 6: Lean 4 proof evolution"""
    print("\n" + "=" * 70)
    print("DEMO 6: Lean 4 Proof Evolution")
    print("=" * 70)

    def lean4_evaluator(proof: str) -> float:
        """Evaluate Lean 4 proof quality"""
        score = 0.0

        if "verified" in proof.lower():
            return 10.0

        if "sorry" in proof.lower():
            return 0.1

        if "intros" in proof:
            score += 2.0
        if "refl" in proof or "rfl" in proof:
            score += 3.0
        if "induction" in proof:
            score += 2.0

        # Prefer concise proofs
        score -= min(len(proof.split()) * 0.1, 2.0)

        return max(score, 0.0)

    theorem = "∀ n : Nat, n + 0 = n"
    initial_proof = f"theorem add_zero : {theorem} := intros sorry"

    print(f"\nTheorem: {theorem}")
    print(f"Initial proof sketch: {initial_proof}")

    config = MakerevolutionConfig(
        mode=MakerevolutionMode.HYBRID,
        voting_threshold=3,
        population_size=25,
        enable_decomposition=True
    )

    print("\nRunning MDAP-enhanced evolution...")
    result = run_maker_evolution(
        initial_program=initial_proof,
        evaluator=lean4_evaluator,
        max_generations=25,
        config=config
    )

    print("\nResults:")
    print(f"  Best fitness: {result['best_fitness']:.3f}")
    print(f"  Generations: {result['generations_completed']}")
    print(f"  Evolved proof:\n{result['best_program']}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

DEMO_FUNCTIONS = {
    "basic": demo_basic_mdap_evolution,
    "voting": demo_voting_only,
    "decomposition": demo_decomposition_only,
    "threshold": demo_voting_threshold_comparison,
    "comparison": demo_pure_vs_mdap_comparison,
    "lean4": demo_lean4_proof_evolution
}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MDAP-Enhanced Evolution Demo"
    )

    parser.add_argument(
        "--demo",
        "-d",
        choices=list(DEMO_FUNCTIONS.keys()) + ["all"],
        default="all",
        help="Demo to run (default: all)"
    )

    parser.add_argument(
        "--skip-capabilities-check",
        action="store_true",
        help="Skip capabilities check"
    )

    args = parser.parse_args()

    # Check capabilities
    if not args.skip_capabilities_check:
        if not check_capabilities():
            print("\n[WARN]  Cannot run demos without required components")
            sys.exit(1)

    # Run demos
    if args.demo == "all":
        print("\n" + "=" * 70)
        print("Running ALL Demos")
        print("=" * 70)

        for name, func in DEMO_FUNCTIONS.items():
            try:
                func()
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                logger.error(f"Demo '{name}' failed: {e}")
                import traceback
                traceback.print_exc()

    else:
        demo_func = DEMO_FUNCTIONS.get(args.demo)
        if demo_func:
            try:
                demo_func()
            except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                logger.error(f"Demo failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Unknown demo: {args.demo}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("Demos Complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
