"""
MAKER/MDAP-Enhanced Evolution - Demo

This script demonstrates the integration of MAKER (arXiv:2511.09030) and MDAP
into the OpenEvolve evolutionary computation workflow.

Features demonstrated:
1. MAKER-enhanced selection: Voting-based population selection
2. MDAP-enhanced decomposition: Task decomposition for evolution
3. Zero-error evolution: Statistical convergence guarantees
4. Hybrid modes: Combine MAKER with standard genetic operators

Usage:
    python demo_evolution_maker.py
"""

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


def demo_1_basic_maker_evolution():
    """Demo 1: Basic MAKER-enhanced evolution"""
    print_section("DEMO 1: Basic MAKER-Enhanced Evolution")

    from evolution import run_maker_enhanced_evolution

    # Sample program to evolve
    initial_program = """def factorial(n):
    # Calculate factorial
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

    print("Evolving factorial function...")
    print(f"Initial program length: {len(initial_program)} characters")

    # Define custom evaluator
    def evaluator(program: str) -> float:
        """Evaluate program quality (prefer longer, more complete programs)"""
        # Simple fitness: prefer programs with more documentation
        doc_lines = sum(1 for line in program.split('\n') if line.strip().startswith('#'))
        total_lines = len(program.split('\n'))
        return float(doc_lines * 10 + total_lines)

    # Run MAKER-enhanced evolution
    result = run_maker_enhanced_evolution(
        initial_program=initial_program,
        content_type="code",
        max_generations=10,
        enable_voting=True,
        enable_decomposition=True,
        voting_threshold=3,
        population_size=10,
        evaluator=evaluator
    )

    # Display results
    print("\n[OK] Evolution completed!")
    print(f"  - Method: {result.get('method', 'unknown')}")
    print(f"  - Best fitness: {result.get('best_fitness', 0):.2f}")
    print(f"  - Generations: {result.get('generations', 0)}")
    print(f"  - Evolution time: {result.get('evolution_time', 0):.2f}s")

    return result


def demo_2_voting_only():
    """Demo 2: MAKER voting without decomposition"""
    print_section("DEMO 2: MAKER Voting (Selection Only)")

    from evolution import run_maker_enhanced_evolution

    initial_program = """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

    print("Evolving bubble sort with MAKER voting...")
    print("MAKER voting: ENABLED")
    print("MDAP decomposition: DISABLED")

    def evaluator(program: str) -> float:
        """Evaluate program quality"""
        # Prefer programs with comments
        has_comments = '#' in program
        return float(len(program) + (100 if has_comments else 0))

    # Run with voting only
    result = run_maker_enhanced_evolution(
        initial_program=initial_program,
        content_type="code",
        max_generations=5,
        enable_voting=True,
        enable_decomposition=False,
        voting_threshold=3,
        population_size=8,
        evaluator=evaluator
    )

    print("\n[OK] Evolution completed!")
    print(f"  - Best fitness: {result.get('best_fitness', 0):.2f}")

    return result


def demo_3_decomposition_only():
    """Demo 3: MDAP decomposition without voting"""
    print_section("DEMO 3: MDAP Decomposition (Task Decomposition Only)")

    from evolution import run_maker_enhanced_evolution

    initial_program = """class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""

    print("Evolving calculator class...")
    print("MAKER voting: DISABLED")
    print("MDAP decomposition: ENABLED")

    def evaluator(program: str) -> float:
        """Evaluate program quality"""
        # Prefer programs with more methods
        num_methods = program.count('def ')
        return float(num_methods * 50 + len(program))

    # Run with decomposition only
    result = run_maker_enhanced_evolution(
        initial_program=initial_program,
        content_type="code",
        max_generations=5,
        enable_voting=False,
        enable_decomposition=True,
        population_size=8,
        evaluator=evaluator
    )

    print("\n[OK] Evolution completed!")
    print(f"  - Best fitness: {result.get('best_fitness', 0):.2f}")

    return result


def demo_4_varying_voting_thresholds():
    """Demo 4: Compare different voting thresholds"""
    print_section("DEMO 4: Voting Threshold Comparison")

    from evolution import run_maker_enhanced_evolution

    initial_program = """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

    def evaluator(program: str) -> float:
        """Evaluate program quality"""
        return float(len(program))

    k_values = [2, 3, 5]
    results = []

    print("Testing with different voting thresholds (k values)...")

    for k in k_values:
        print(f"\n  Testing with k={k}...")

        result = run_maker_enhanced_evolution(
            initial_program=initial_program,
            content_type="code",
            max_generations=5,
            enable_voting=True,
            voting_threshold=k,
            population_size=8,
            evaluator=evaluator
        )

        best_fitness = result.get('best_fitness', 0)
        generations = result.get('generations', 0)

        results.append({
            "k": k,
            "best_fitness": best_fitness,
            "generations": generations
        })

        print(f"    Best fitness: {best_fitness:.2f}, Generations: {generations}")

    print("\n[Summary]")
    print("  k   | Best Fitness | Generations")
    print("  ----|--------------|------------")
    for r in results:
        print(f"  {r['k']:>3} | {r['best_fitness']:>12.2f} | {r['generations']:>10}")

    return results


def demo_5_evolution_modes():
    """Demo 5: Compare different evolution modes"""
    print_section("DEMO 5: Evolution Mode Comparison")

    from evolution_maker_integration import MakerevolutionMode, run_maker_evolution, MakerevolutionConfig

    initial_program = """def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
"""

    def evaluator(program: str) -> float:
        """Evaluate program quality"""
        return float(len(program))

    modes = [
        (MakerevolutionMode.VOTING_ONLY, "Voting Only"),
        (MakerevolutionMode.DECOMPOSITION, "Decomposition Only"),
        (MakerevolutionMode.HYBRID, "Hybrid (Voting + Decomposition)")
    ]

    results = []

    for mode, mode_name in modes:
        print(f"\n  Testing mode: {mode_name}...")

        config = MakerevolutionConfig(
            mode=mode,
            enable_voting=(mode != MakerevolutionMode.DECOMPOSITION),
            enable_decomposition=(mode != MakerevolutionMode.VOTING_ONLY),
            population_size=8
        )

        result = run_maker_evolution(
            initial_program=initial_program,
            evaluator=evaluator,
            max_generations=5,
            config=config
        )

        best_fitness = result.get('best_fitness', 0)
        results.append({
            "mode": mode_name,
            "best_fitness": best_fitness
        })

        print(f"    Best fitness: {best_fitness:.2f}")

    print("\n[Summary]")
    print("  Mode              | Best Fitness")
    print("  ------------------|--------------")
    for r in results:
        print(f"  {r['mode']:16s} | {r['best_fitness']:>12.2f}")

    return results


def demo_6_capabilities():
    """Demo 6: Check MAKER/MDAP evolution capabilities"""
    print_section("DEMO 6: MAKER/MDAP Evolution Capabilities Check")

    from evolution import get_maker_evolution_capabilities

    capabilities = get_maker_evolution_capabilities()

    print("MAKER/MDAP Evolution Capabilities:")
    print(f"  - MAKER evolution enabled: {capabilities.get('maker_evolution_enabled', False)}")
    print(f"  - MDAP decomposition enabled: {capabilities.get('mdap_decomposition_enabled', False)}")
    print(f"  - Integration status: {capabilities.get('integration_status', 'unknown')}")

    print("\n  Evolution Modes:")
    for mode in capabilities.get('modes', []):
        print(f"    - {mode}")

    print("\n  Algorithms from Paper:")
    for algo in capabilities.get('algorithms', []):
        print(f"    - {algo}")

    if 'paper' in capabilities:
        paper = capabilities['paper']
        print(f"\n  Paper Reference:")
        print(f"    - Title: {paper.get('title', 'N/A')}")
        print(f"    - arXiv: {paper.get('arxiv', 'N/A')}")
        print(f"    - URL: {paper.get('url', 'N/A')}")

    return capabilities


def main():
    """Run all demos."""
    print("\n")
    print("=" * 80)
    print("  MAKER/MDAP-ENHANCED EVOLUTION - DEMONSTRATION")
    print("  Paper: arXiv:2511.09030 (Solving a Million-Step LLM Task with Zero Errors)")
    print("=" * 80)
    print("")

    demos = [
        ("Basic MAKER-Enhanced Evolution", demo_1_basic_maker_evolution),
        ("MAKER Voting Only", demo_2_voting_only),
        ("MDAP Decomposition Only", demo_3_decomposition_only),
        ("Voting Threshold Comparison", demo_4_varying_voting_thresholds),
        ("Evolution Mode Comparison", demo_5_evolution_modes),
        ("Capabilities Check", demo_6_capabilities),
    ]

    print("Available Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all demos")
    print("")

    try:
        choice = input("Select demo (0-6, or press Enter for all): ").strip()
        if not choice:
            choice = "0"

        choice_num = int(choice)

        if choice_num == 0:
            # Run all demos
            for name, demo_func in demos:
                try:
                    demo_func()
                except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
                    logger.error(f"Demo {name} failed: {e}", exc_info=True)
        elif 1 <= choice_num <= len(demos):
            # Run selected demo
            name, demo_func = demos[choice_num - 1]
            demo_func()
        else:
            print("Invalid choice")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except (RuntimeError, ValueError, ConnectionError, TimeoutError) as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("  DEMO COMPLETED")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - evolution_maker_integration.py")
    print("  - MAKER_EVOLUTION_INTEGRATION_GUIDE.md")
    print("  - Paper: https://arxiv.org/abs/2511.09030")
    print("")


if __name__ == "__main__":
    main()
