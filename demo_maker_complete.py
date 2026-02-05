"""
MAKER Complete Implementation Demo

This script demonstrates the complete MAKER implementation based on the paper:
"Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)

Examples:
1. Towers of Hanoi - The canonical million-step example from the paper
2. Multi-digit multiplication - Demonstrating general decomposition
3. Custom tasks - Demonstrating recursive solve

Usage:
    python demo_maker_complete.py [--mode MODE] [--example EXAMPLE]

Modes:
    - sequential: Algorithm 1 (generate_solution)
    - recursive: Algorithm 4 (recursive multi-agent solve)
    - hybrid: ROMA + MAKER integration

Examples:
    - hanoi: Towers of Hanoi (default)
    - multiplication: Multi-digit multiplication
    - custom: Custom task demonstration
"""

import argparse
import json
import logging
import sys
import time
from typing import Any, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_towers_of_hanoi(num_disks: int = 5, k_ahead: int = 3):
    """
    Demonstrate Towers of Hanoi solving with MAKER.

    This is the canonical example from the paper showing zero-error
    solving of long sequences. 20 disks = 1,048,575 steps.

    Args:
        num_disks: Number of disks (use small number for demo, 20 for paper result)
        k_ahead: Voting threshold (3 used in paper)
    """
    from maker_integration_bridge import solve_towers_of_hanoi, get_integrated_status

    logger.info("=" * 80)
    logger.info("DEMO: Towers of Hanoi with MAKER")
    logger.info("=" * 80)
    logger.info(f"Configuration: {num_disks} disks, k_ahead={k_ahead}")
    logger.info(f"Expected optimal steps: {2**num_disks - 1}")
    logger.info("")

    # Check system status
    status = get_integrated_status()
    logger.info(f"MAKER Status: {status['maker_available']}")
    logger.info(f"Algorithms: {', '.join(status['algorithms_implemented'])}")
    logger.info("")

    start_time = time.time()

    try:
        result = solve_towers_of_hanoi(
            num_disks=num_disks,
            k_ahead=k_ahead
        )

        elapsed = time.time() - start_time

        logger.info("")
        logger.info("=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Success: {result.get('success', False)}")
        logger.info(f"Mode: {result.get('mode', 'unknown')}")
        logger.info(f"Execution time: {elapsed:.2f}s")

        metrics = result.get('metrics', {})
        if metrics:
            logger.info("")
            logger.info("Metrics:")
            logger.info(f"  Total steps: {metrics.total_steps}")
            logger.info(f"  Total votes: {metrics.total_votes}")
            logger.info(f"  Red flags: {metrics.red_flags}")
            logger.info(f"  Decompositions: {metrics.decompositions}")
            logger.info(f"  Atomic solves: {metrics.atomic_solves}")
            logger.info(f"  Avg confidence: {metrics.avg_confidence:.2f}")

        result_data = result.get('result', {})
        if result_data and 'actions' in result_data:
            actions = result_data['actions']
            logger.info(f"")
            logger.info(f"Generated {len(actions)} moves")
            logger.info("")
            logger.info("First 5 moves:")
            for i, move in enumerate(actions[:5], 1):
                logger.info(f"  {i}. {move}")

            if len(actions) > 5:
                logger.info(f"  ... ({len(actions) - 5} more moves)")

        logger.info("")
        logger.info("[OK] Towers of Hanoi demo completed successfully!")

    except Exception as e:
        logger.error(f"Towers of Hanoi demo failed: {e}", exc_info=True)
        return False

    return True


def demo_multiplication(num1_digits: int = 3, num2_digits: int = 3, k_ahead: int = 3):
    """
    Demonstrate multi-digit multiplication with recursive decomposition.

    This is from Appendix F of the paper, showing general-purpose
    decomposition beyond predetermined sequences.

    Args:
        num1_digits: Number of digits in first number
        num2_digits: Number of digits in second number
        k_ahead: Voting threshold
    """
    from maker_integration_bridge import solve_multiplication

    # Generate random numbers
    num1 = int('9' * num1_digits)  # e.g., 999 for 3 digits
    num2 = int('9' * num2_digits)

    logger.info("=" * 80)
    logger.info("DEMO: Multi-Digit Multiplication with MAKER")
    logger.info("=" * 80)
    logger.info(f"Task: Calculate {num1} × {num2}")
    logger.info(f"Configuration: k_ahead={k_ahead}, mode=recursive")
    logger.info("")

    start_time = time.time()

    try:
        result = solve_multiplication(
            num1=num1,
            num2=num2,
            k_ahead=k_ahead
        )

        elapsed = time.time() - start_time

        logger.info("")
        logger.info("=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Success: {result.get('success', False)}")
        logger.info(f"Mode: {result.get('mode', 'unknown')}")
        logger.info(f"Execution time: {elapsed:.2f}s")

        metrics = result.get('metrics', {})
        if metrics:
            logger.info("")
            logger.info("Metrics:")
            logger.info(f"  Total steps: {metrics.total_steps}")
            logger.info(f"  Total votes: {metrics.total_votes}")
            logger.info(f"  Decompositions: {metrics.decompositions}")
            logger.info(f"  Atomic solves: {metrics.atomic_solves}")

        solution = result.get('result')
        if solution:
            logger.info("")
            logger.info("Solution:")
            logger.info(f"  {json.dumps(solution, indent=2)}")

        logger.info("")
        logger.info("[OK] Multiplication demo completed!")

    except Exception as e:
        logger.error(f"Multiplication demo failed: {e}", exc_info=True)
        return False

    return True


def demo_recursive_solve(task: str, k_ahead: int = 3, max_depth: int = 4):
    """
    Demonstrate recursive solve for a custom task.

    Shows how MAKER can decompose and solve arbitrary tasks.

    Args:
        task: Task description
        k_ahead: Voting threshold
        max_depth: Maximum recursion depth
    """
    from maker_integration_bridge import solve_with_maker

    logger.info("=" * 80)
    logger.info("DEMO: Recursive Solve with MAKER")
    logger.info("=" * 80)
    logger.info(f"Task: {task}")
    logger.info(f"Configuration: k_ahead={k_ahead}, max_depth={max_depth}")
    logger.info("")

    start_time = time.time()

    try:
        result = solve_with_maker(
            task=task,
            mode="recursive",
            k_ahead=k_ahead,
            max_depth=max_depth
        )

        elapsed = time.time() - start_time

        logger.info("")
        logger.info("=" * 80)
        logger.info("RESULTS")
        logger.info("=" * 80)
        logger.info(f"Success: {result.get('success', False)}")
        logger.info(f"Mode: {result.get('mode', 'unknown')}")
        logger.info(f"Execution time: {elapsed:.2f}s")

        metrics = result.get('metrics', {})
        if metrics:
            logger.info("")
            logger.info("Metrics:")
            logger.info(f"  Total steps: {metrics.total_steps}")
            logger.info(f"  Total votes: {metrics.total_votes}")
            logger.info(f"  Decompositions: {metrics.decompositions}")
            logger.info(f"  Atomic solves: {metrics.atomic_solves}")

        solution = result.get('result')
        if solution:
            logger.info("")
            logger.info("Solution:")
            logger.info(f"  {json.dumps(solution, indent=2)[:500]}...")
            if len(json.dumps(solution)) > 500:
                logger.info(f"  (truncated, total size: {len(json.dumps(solution))} chars)")

        logger.info("")
        logger.info("[OK] Recursive solve demo completed!")

    except Exception as e:
        logger.error(f"Recursive solve demo failed: {e}", exc_info=True)
        return False

    return True


def demo_voting_mechanisms():
    """
    Demonstrate the voting mechanisms from the paper.

    Shows first-to-k vs first-to-ahead-by-k voting.
    """
    from maker_integration_bridge import create_maker_config, MAKERIntegrationBridge
    from workflow_structures import ModelConfig, Team

    logger.info("=" * 80)
    logger.info("DEMO: Voting Mechanisms")
    logger.info("=" * 80)
    logger.info("Comparing first-to-k vs first-to-ahead-by-k voting")
    logger.info("")

    # Simple task for comparison
    task = "What is 7 × 8?"

    # Test both voting modes
    for mode in ["first_to_k", "first_to_ahead_by_k"]:
        logger.info(f"Testing mode: {mode}")
        logger.info("-" * 40)

        config = create_maker_config(
            mode="recursive",
            k_ahead=3,
            enable_first_to_ahead=(mode == "first_to_ahead_by_k")
        )

        bridge = MAKERIntegrationBridge(config)

        start = time.time()
        result = bridge.solve(task, context={"operation": "simple_math"})
        elapsed = time.time() - start

        logger.info(f"Result: {result.get('success', False)}")
        logger.info(f"Time: {elapsed:.2f}s")
        logger.info(f"Votes: {result.get('metrics', {}).total_votes}")
        logger.info("")


def demo_red_flagging():
    """
    Demonstrate red-flagging mechanism.

    Shows how overly long or malformed responses are filtered out.
    """
    from mdap_maker_complete import VoteCollector, MAKERRunMetrics
    from workflow_structures import ModelConfig

    logger.info("=" * 80)
    logger.info("DEMO: Red-Flagging Mechanism")
    logger.info("=" * 80)
    logger.info("Testing detection of unreliable responses")
    logger.info("")

    # Create vote collector with red-flagging enabled
    collector = VoteCollector(
        max_token_length=100,  # Very low threshold for demo
        max_retries=5
    )

    # Test cases
    test_cases = [
        ("Valid response", "The answer is 42", False),
        ("Too long", "x" * 500, True),
        ("Empty response", "", True),
        ("Malformed JSON", "{invalid json}", True),
    ]

    for name, response, should_flag in test_cases:
        is_flagged = collector._has_red_flags(response, None)
        status = "[OK]" if is_flagged == should_flag else "[FAIL]"
        logger.info(f"{status} {name}: flagged={is_flagged} (expected={should_flag})")

    logger.info("")
    logger.info("[OK] Red-flagging demo completed!")


def main():
    """Main demo runner."""
    parser = argparse.ArgumentParser(
        description="MAKER Complete Implementation Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Towers of Hanoi demo (default)
  python demo_maker_complete.py

  # Run with 10 disks
  python demo_maker_complete.py --example hanoi --num-disks 10

  # Run multiplication demo
  python demo_maker_complete.py --example multiplication

  # Run custom task demo
  python demo_maker_complete.py --example custom --task "Explain quantum computing"

  # Run all demos
  python demo_maker_complete.py --all

  # Test voting mechanisms
  python demo_maker_complete.py --test-voting
        """
    )

    parser.add_argument(
        '--mode',
        choices=['sequential', 'recursive', 'hybrid'],
        default='recursive',
        help='Execution mode (default: recursive)'
    )

    parser.add_argument(
        '--example',
        choices=['hanoi', 'multiplication', 'custom'],
        default='hanoi',
        help='Example to run (default: hanoi)'
    )

    parser.add_argument(
        '--num-disks',
        type=int,
        default=5,
        help='Number of disks for Towers of Hanoi (default: 5)'
    )

    parser.add_argument(
        '--k-ahead',
        type=int,
        default=3,
        help='Voting threshold k (default: 3)'
    )

    parser.add_argument(
        '--task',
        type=str,
        default="Explain the causes of the American Civil War",
        help='Custom task for demo (default: causes of American Civil War)'
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=4,
        help='Max recursion depth (default: 4)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all demos'
    )

    parser.add_argument(
        '--test-voting',
        action='store_true',
        help='Test voting mechanisms'
    )

    parser.add_argument(
        '--test-redflag',
        action='store_true',
        help='Test red-flagging mechanism'
    )

    args = parser.parse_args()

    # Print header
    logger.info("")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 15 + "MAKER COMPLETE IMPLEMENTATION DEMO" + " " * 34 + "║")
    logger.info("║" + " " * 20 + "Paper: arXiv:2511.09030" + " " * 38 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")

    results = []

    # Test voting mechanisms if requested
    if args.test_voting:
        demo_voting_mechanisms()
        logger.info("")
        return 0

    if args.test_redflag:
        demo_red_flagging()
        logger.info("")
        return 0

    # Run demos
    if args.all or args.example == 'hanoi':
        results.append(demo_towers_of_hanoi(args.num_disks, args.k_ahead))
        logger.info("")

    if args.all or args.example == 'multiplication':
        results.append(demo_multiplication(k_ahead=args.k_ahead))
        logger.info("")

    if args.all or args.example == 'custom':
        results.append(demo_recursive_solve(args.task, args.k_ahead, args.max_depth))
        logger.info("")

    # Summary
    if results:
        success_count = sum(1 for r in results if r)
        total_count = len(results)

        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Completed: {success_count}/{total_count} demos")

        if success_count == total_count:
            logger.info("")
            logger.info("[OK][OK][OK] ALL DEMOS PASSED [OK][OK][OK]")
            return 0
        else:
            logger.warning("")
            logger.warning(f"[FAIL] {total_count - success_count} demo(s) failed")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
