"""
Example: Business Process Decomposition

This example demonstrates flow-based decomposition of a business process.

Problem: Design order fulfillment process for e-commerce
"""

from decomposition_engine import FlowBasedDecomposition
from sovereign_data_models import (
    ProblemDefinition,
    ProblemType,
    DomainContext,
    ComplexityScore
)


def main():
    problem = ProblemDefinition(
        id="process-001",
        title="E-Commerce Order Fulfillment Process",
        description="""Design and implement an end-to-end order fulfillment process
        for an e-commerce platform:

        Process Requirements:
        - Order placement to delivery tracking
        - Inventory management
        - Payment processing
        - Shipping orchestration
        - Customer notifications
        - Returns and refunds

        Performance Requirements:
        - Process 1000 orders/hour
        - <5 minutes from order to shipping label
        - 99.9% order accuracy
        - Real-time tracking updates

        Integration Points:
        - Website frontend
        - Inventory system
        - Payment gateway
        - Shipping carriers (FedEx, UPS, USPS)
        - Email/SMS service
        - Analytics platform
        """,
        problem_type=ProblemType.IMPLEMENTATION,
        domain_context=DomainContext(
            domain="Business Process",
            subdomain="Order Fulfillment"
        ),
        complexity_score=ComplexityScore(
            overall_complexity=7,
            cognitive_complexity=6,
            computational_complexity=7,
            domain_complexity=7,
            integration_complexity=8
        )
    )

    print("=" * 80)
    print(f"BUSINESS PROCESS: {problem.title}")
    print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    print("=" * 80)

    # Use flow-based decomposition (best for processes)
    print("\nDecomposing with FLOW-BASED strategy...")
    strategy = FlowBasedDecomposition(
        preserve_order=True,
        allow_parallel=True,
        batch_processing=True
    )

    sub_problems = strategy.decompose(problem)

    print(f"\n[OK] Generated {len(sub_problems)} process stages\n")

    # Display as pipeline
    print("PROCESS FLOW:")
    print("=" * 80)

    for i, sp in enumerate(sub_problems, 1):
        # Check if parallelizable
        parallelizable = "⚡ (can parallelize)" if not sp.dependencies else ""

        print(f"\n[Stage {i}] {sp.title} {parallelizable}")
        print(f"   Type: {sp.type.value}")
        print(f"   Effort: {sp.estimated_effort}h")
        print(f"   Complexity: {sp.complexity_score.overall_complexity}/10")

        # Extract input/output from description
        desc_lines = sp.description.split('\n')
        for line in desc_lines:
            if 'input:' in line.lower():
                print(f"   Input: {line.split(':', 1)[1].strip()}")
            elif 'output:' in line.lower():
                print(f"   Output: {line.split(':', 1)[1].strip()}")

        # Show dependencies
        if sp.dependencies:
            dep_names = [f"Stage {sub_problems.index(sp) + 1}" for sp in sub_problems if sp.id in sp.dependencies]
            print(f"   Depends on: {', '.join(dep_names)}")

    # Calculate pipeline metrics
    print("\n" + "=" * 80)
    print("PIPELINE METRICS")
    print("=" * 80)

    # Count parallelizable stages
    parallelizable_count = sum(1 for sp in sub_problems if not sp.dependencies)

    # Calculate critical path length
    max_effort_path = max(
        sum(sp.estimated_effort for sp in sub_problems if sub_problems.index(sp) <= i)
        for i in range(len(sub_problems))
    )

    print(f"Total Stages: {len(sub_problems)}")
    print(f"Parallelizable Stages: {parallelizable_count}")
    print(f"Critical Path Effort: {max_effort_path}h")
    print(f"Potential Speedup: {len(sub_problems) / (len(sub_problems) - parallelizable_count):.1f}x")

    # Estimate throughput
    print(f"\nEstimated Throughput:")
    print(f"  Per stage (avg): {1 / (sum(sp.estimated_effort for sp in sub_problems) / len(sub_problems)):.3f} stages/hour")
    print(f"  Complete pipeline: {1 / max_effort_path:.3f} pipelines/hour")


if __name__ == "__main__":
    main()
