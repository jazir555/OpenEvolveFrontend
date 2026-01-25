"""
Example: Software Architecture Decomposition

This example demonstrates hierarchical decomposition of a complex
software system architecture.

Problem: Design a scalable microservices architecture for e-commerce
"""

from decomposition_engine import DecompositionEngine, HierarchicalDecomposition
from sovereign_data_models import (
    ProblemDefinition,
    ProblemType,
    DomainContext,
    ComplexityScore
)


def main():
    problem = ProblemDefinition(
        id="arch-001",
        title="E-Commerce Microservices Architecture",
        description="""Design and implement a scalable e-commerce platform using
        microservices architecture with the following requirements:

        Business Requirements:
        - Handle 10,000 concurrent users
        - Support 1M+ products
        - Process 1000 orders/minute
        - 99.99% uptime SLA
        - Global deployment (US, EU, APAC)

        Technical Requirements:
        - Microservices architecture
        - Event-driven communication
        - API Gateway layer
        - Service mesh (Istio)
        - Caching layer (Redis)
        - CDN for static content
        - Database replication (PostgreSQL)
        - Auto-scaling (Kubernetes)
        - Monitoring (Prometheus/Grafana)
        - Distributed tracing (Jaeger)
        """,
        problem_type=ProblemType.DESIGN,
        domain_context=DomainContext(
            domain="Software Architecture",
            subdomain="Distributed Systems"
        ),
        complexity_score=ComplexityScore(
            overall_complexity=9,
            cognitive_complexity=7,
            computational_complexity=8,
            domain_complexity=9,
            integration_complexity=10
        )
    )

    print("=" * 80)
    print(f"PROBLEM: {problem.title}")
    print(f"Complexity: {problem.complexity_score.overall_complexity}/10")
    print("=" * 80)

    # Use hierarchical decomposition
    print("\nDecomposing with HIERARCHICAL strategy...")
    engine = DecompositionEngine()
    result = engine.decompose(problem, strategy="hierarchical")

    print(f"\n✓ Generated {len(result.sub_problems)} sub-problems\n")

    # Display as hierarchy
    print("HIERARCHICAL STRUCTURE:")
    print("=" * 80)

    # Group by "level" (simulated based on dependencies)
    levels = {}
    for sp in result.sub_problems:
        level = len(sp.dependencies)
        if level not in levels:
            levels[level] = []
        levels[level].append(sp)

    for level in sorted(levels.keys()):
        indent = "  " * level
        print(f"\n{indent}LEVEL {level}:")

        for sp in levels[level]:
            icon = "🏗️" if level == 0 else "📦"
            print(f"{indent}{icon} {sp.title}")
            print(f"{indent}   Complexity: {sp.complexity_score.overall_complexity}/10")
            print(f"{indent}   Effort: {sp.estimated_effort}h")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total hierarchical levels: {len(levels)}")
    print(f"Total services/components: {len(result.sub_problems)}")


if __name__ == "__main__":
    main()
