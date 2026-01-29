"""
ROMA-MDAP-MAKER + Associative Recomposition Demo

Demonstrates the complete integrated system with:
- ROMA hierarchical decomposition
- Associative recomposition (domain-agnostic LLM + algorithmic)
- MDAP multi-agent validation
- Ground truth verification

Usage:
    python roma_mdap_maker_associative_example.py
"""

import sys
import logging
from typing import Dict, Any

sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def demo_1_status_check():
    """Demo 1: System status check"""
    print("\n" + "="*80)
    print("DEMO 1: SYSTEM STATUS CHECK")
    print("="*80 + "\n")

    from roma_mdap_maker_associative_integration import (
        get_romamdapmaker_associative_status,
        ROMA_MDAP_MAKER_AVAILABLE,
        ASSOCIATIVE_AVAILABLE,
        GROUND_TRUTH_AVAILABLE
    )

    status = get_romamdapmaker_associative_status()

    print("Component Availability:")
    print(f"  ROMA-MDAP-MAKER: {'[OK]' if ROMA_MDAP_MAKER_AVAILABLE else '[X]'}")
    print(f"  Associative Recomposition: {'[OK]' if ASSOCIATIVE_AVAILABLE else '[X]'}")
    print(f"  Ground Truth Store: {'[OK]' if GROUND_TRUTH_AVAILABLE else '[X]'}")
    print(f"\n  Full System: {'[OK] READY' if status['full_system_available'] else '[X] PARTIAL'}")

    print(f"\nDescription: {status['description']}")


def demo_2_simple_problem():
    """Demo 2: Solve a simple problem"""
    print("\n" + "="*80)
    print("DEMO 2: SOLVE SIMPLE PROBLEM")
    print("="*80 + "\n")

    from roma_mdap_maker_associative_integration import (
        create_romamdapmaker_associative_config,
        ROMAMDAPMakerAssociativeEngine
    )
    from roma_mdap_maker_reliability_ssot import get_standard_config

    # Use SSOT for standardized high-reliability config
    config = get_standard_config(
        roma_max_depth_analysis=2,
        roma_max_depth_solving=2
    )

    # Create engine
    engine = ROMAMDAPMakerAssociativeEngine(config)

    # Define problem
    problem = """
    Build a simple user authentication system with:
    1. User registration with email verification
    2. Secure password storage (hashed)
    3. Login with JWT tokens
    4. Password reset functionality
    """

    print(f"Problem: {problem.strip()}\n")

    # Solve problem
    result = engine.solve_problem(
        problem=problem,
        context={
            "requirements": [
                "Secure password storage",
                "JWT-based authentication",
                "Email verification"
            ]
        }
    )

    # Display results
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80 + "\n")

    if result.get("error"):
        print(f"[X] Error: {result['error']}")
        print(f"Phase: {result.get('phase', 'unknown')}")
    else:
        print(f"[OK] Success: {result['success']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Total Time: {result['total_time']:.2f}s")
        print(f"\nBreakdown:")
        print(f"  Decomposition: {result['decomposition_time']:.2f}s")
        print(f"  Recomposition: {result['recomposition_time']:.2f}s")
        print(f"  Validation: {result['validation_time']:.2f}s")

        print(f"\nMetrics:")
        print(f"  Sub-solutions: {result['num_sub_solutions']}")
        print(f"  Atomic Tasks: {result['num_atomic_tasks']}")
        print(f"  ROMA Depth: {result['roma_depth']}")

        # Domain classification
        if result.get("domain_classification"):
            domain = result["domain_classification"]
            print(f"\nDomain Classification:")
            print(f"  Domain: {domain.get('domain', 'N/A')}")
            print(f"  Type: {domain.get('solution_type', 'N/A')}")
            print(f"  Field: {domain.get('field', 'N/A')}")
            print(f"  Complexity: {domain.get('complexity', 'N/A')}")

        # MDAP validation
        mdap = result.get("mdap_validation", {})
        if mdap.get("validation_details"):
            print(f"\nMDAP Validation:")
            print(f"  Validated: {mdap.get('validated', False)}")
            print(f"  Error Rate: {mdap.get('error_rate', 0):.2%}")
            print(f"  Red Flags: {mdap.get('red_flags', 0)}")

        # Solution preview
        solution = result.get("solution", "")
        if solution:
            preview_length = 500
            preview = solution[:preview_length]
            if len(solution) > preview_length:
                preview += "..."

            print(f"\nSolution Preview ({len(solution)} chars):")
            print("-"*80)
            print(preview)
            print("-"*80)

    print("\n")


def demo_3_complex_problem():
    """Demo 3: Solve a complex problem"""
    print("\n" + "="*80)
    print("DEMO 3: SOLVE COMPLEX PROBLEM")
    print("="*80 + "\n")

    from roma_mdap_maker_associative_integration import (
        solve_with_romamdapmaker_associative
    )

    # Define complex problem
    problem = """
    Design and implement a complete e-commerce product recommendation system with:
    1. User behavior tracking (page views, clicks, purchases)
    2. Collaborative filtering algorithm (user-based and item-based)
    3. Content-based filtering using product attributes
    4. Hybrid recommendation engine combining multiple approaches
    5. Real-time personalization with A/B testing framework
    6. Scalable architecture handling 1M+ products and 100K+ users
    """

    print(f"Problem: {problem.strip()}\n")

    # Solve with convenience function
    result = solve_with_romamdapmaker_associative(
        problem=problem,
        context={
            "constraints": [
                "Scalable to millions of products",
                "Real-time response time < 100ms",
                "A/B testing capabilities"
            ],
            "tech_stack": ["Python", "Redis", "PostgreSQL", "Kafka"]
        }
    )

    # Display results
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80 + "\n")

    if result.get("error"):
        print(f"[X] Error: {result['error']}")
    else:
        print(f"[OK] Success: {result['success']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Total Time: {result['total_time']:.2f}s")
        print(f"Error-Free: {result['error_free']}")

        print(f"\nDecomposition Details:")
        print(f"  Sub-solutions: {result['num_sub_solutions']}")
        print(f"  Atomic Tasks: {result['num_atomic_tasks']}")
        print(f"  ROMA Depth: {result['roma_depth']}")

        # Solution statistics
        solution = result.get("solution", "")
        if solution:
            print(f"\nSolution Statistics:")
            print(f"  Total Characters: {len(solution)}")
            print(f"  Estimated Lines: {len(solution.splitlines())}")

            # Count code blocks
            code_blocks = solution.count("```")
            print(f"  Code Blocks: {code_blocks // 2}")

    print("\n")


def demo_4_metrics():
    """Demo 4: View execution metrics"""
    print("\n" + "="*80)
    print("DEMO 4: EXECUTION METRICS")
    print("="*80 + "\n")

    from roma_mdap_maker_associative_integration import (
        create_romamdapmaker_associative_config,
        ROMAMDAPMakerAssociativeEngine
    )
    from roma_mdap_maker_reliability_ssot import get_standard_config

    # Create engine with standard SSOT config
    config = get_standard_config()
    engine = ROMAMDAPMakerAssociativeEngine(config)

    # Solve some test problems
    test_problems = [
        "Create a todo list application with add, edit, delete features",
        "Build a chat application with real-time messaging",
        "Design a REST API for a blog platform"
    ]

    print("Solving test problems to populate metrics...")
    for problem in test_problems:
        engine.solve_problem(problem=problem)

    # Get metrics
    metrics = engine.get_metrics()

    print("Execution Metrics:")
    print(f"  Total Problems Solved: {metrics['total_problems_solved']}")
    print(f"  Successful Recompositions: {metrics['successful_recompositions']}")
    print(f"  Failed Recompositions: {metrics['failed_recompositions']}")
    print(f"  Total Sub-Solutions: {metrics['total_sub_solutions']}")

    if metrics['total_problems_solved'] > 0:
        print(f"\nAverage Metrics:")
        print(f"  Confidence: {metrics['avg_confidence']:.2%}")
        print(f"  Decomposition Time: {metrics['total_decomposition_time']:.2f}s")
        print(f"  Recomposition Time: {metrics['total_recomposition_time']:.2f}s")
        print(f"  Validation Time: {metrics['total_validation_time']:.2f}s")
        print(f"  Total Time: {sum([metrics['total_decomposition_time'], metrics['total_recomposition_time'], metrics['total_validation_time']]):.2f}s")

    print("\n")


def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("ROMA-MDAP-MAKER + ASSOCIATIVE RECOMPOSITION DEMOS")
    print("="*80)

    # Run demos
    demo_1_status_check()

    # Only run problem-solving demos if system is available
    from roma_mdap_maker_associative_integration import get_romamdapmaker_associative_status
    status = get_romamdapmaker_associative_status()

    if status['roma_mdap_maker_available']:
        demo_2_simple_problem()
        demo_3_complex_problem()
        demo_4_metrics()
    else:
        print("\n[!] ROMA-MDAP-MAKER not available. Skipping problem-solving demos.")
        print("Install ROMA dependencies to run full demos.")

    print("\n" + "="*80)
    print("DEMOS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
