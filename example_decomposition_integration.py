"""
Integration Example: decomposition_strategy.py with Sovereign System

This example demonstrates how to use the decomposition_strategy module
within the broader Sovereign system, including integration with:
- sovereign_data_models
- problem_fractal_pipeline
- Existing decomposition workflows

Author: Sovereign System
Date: 2026-01-21
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

# Import from decomposition_strategy
from decomposition_strategy import (
    SovereignDecompositionStrategy,
    DecompositionStrategyExecutor,
    StrategySelector,
    decompose_hybrid,
    decompose_roma,
    decompose_semantic,
    select_strategy,
    execute_strategy,
)

# Import from sovereign_data_models
try:
    from sovereign_data_models import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        ProblemStatus,
        generate_id
    )
    print("[OK] Successfully imported from sovereign_data_models")
except ImportError as e:
    print(f"[FAIL] Failed to import from sovereign_data_models: {e}")
    print("       Using fallback definitions from decomposition_strategy")
    from decomposition_strategy import (
        ProblemDefinition,
        SubProblem,
        DecompositionPlan,
        ProblemStatus,
        generate_id
    )

# Import from problem_fractal_pipeline (if available)
try:
    from problem_fractal_pipeline import (
        FractalPipelineCoordinator,
        FractalPipelineConfig,
        FractalPipelineResult
    )
    FRACTAL_PIPELINE_AVAILABLE = True
    print("[OK] Successfully imported from problem_fractal_pipeline")
except ImportError as e:
    print(f"[INFO] Failed to import from problem_fractal_pipeline: {e}")
    FRACTAL_PIPELINE_AVAILABLE = False


def example_1_basic_decomposition():
    """Example 1: Basic decomposition usage."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Decomposition")
    print("=" * 80)

    # Create a problem
    problem = ProblemDefinition(
        problem_id=generate_id("example"),
        title="Develop REST API Service",
        description=(
            "Design and implement a REST API service for user management. "
            "The API should support CRUD operations, authentication, "
            "and rate limiting. Use Python and FastAPI framework."
        ),
        domain="software_engineering",
        complexity="moderate",
        priority="high",
        estimated_effort="medium",
        requirements=[
            "User CRUD operations",
            "JWT authentication",
            "Rate limiting (100 req/min)",
            "Input validation",
            "API documentation"
        ],
        constraints=[
            "Use FastAPI",
            "PostgreSQL database",
            "Deploy to AWS"
        ],
        created_at=datetime.utcnow()
    )

    print(f"\nProblem: {problem.title}")
    print(f"Domain: {problem.domain}")
    print(f"Complexity: {problem.complexity}")
    print(f"Requirements: {len(problem.requirements)}")

    # Select strategy
    strategy = select_strategy(problem)
    print(f"\nRecommended Strategy: {strategy.value}")

    # Decompose
    plan = execute_strategy(strategy.value, problem)

    print(f"\nDecomposition Plan: {plan.plan_id}")
    print(f"Sub-problems created: {len(plan.sub_problems)}")
    print(f"Execution order: {len(plan.execution_order)} steps")

    print("\nSub-problems:")
    for i, sp in enumerate(plan.sub_problems, 1):
        print(f"  {i}. {sp.title}")
        print(f"     Confidence: {sp.confidence:.2f}")
        print(f"     Status: {sp.status.value}")

    return plan


def example_2_strategy_comparison():
    """Example 2: Compare all three strategies."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Strategy Comparison")
    print("=" * 80)

    problem = ProblemDefinition(
        problem_id=generate_id("comparison"),
        title="Build Microservices Architecture",
        description=(
            "Design and implement a microservices architecture for an e-commerce platform. "
            "Services include user management, product catalog, order processing, "
            "payment integration, and inventory management. Ensure high availability "
            "and scalability."
        ),
        domain="software_engineering",
        complexity="complex",
        priority="high",
        estimated_effort="large",
        requirements=[
            "Microservices architecture",
            "Service mesh (Istio)",
            "API Gateway",
            "Event-driven communication",
            "Distributed tracing"
        ],
        constraints=[
            "Cloud-native (Kubernetes)",
            "Budget: $10,000/month",
            "Team: 12 developers"
        ],
        created_at=datetime.utcnow()
    )

    print(f"\nProblem: {problem.title}")

    # Compare all strategies
    strategies = {
        "HYBRID": lambda: decompose_hybrid(problem, depth=3),
        "ROMA": lambda: decompose_roma(problem, max_depth=3),
        "SEMANTIC": lambda: decompose_semantic(problem, clusters=5)
    }

    results = {}
    for name, decompose_func in strategies.items():
        print(f"\n{'-' * 40}")
        print(f"Testing {name} Strategy")
        print(f"{'-' * 40}")

        try:
            plan = decompose_func()
            results[name] = plan

            print(f"Sub-problems: {len(plan.sub_problems)}")
            print(f"Dependencies: {len(plan.dependencies)}")
            avg_confidence = sum(sp.confidence for sp in plan.sub_problems) / len(plan.sub_problems)
            print(f"Avg Confidence: {avg_confidence:.2f}")

            # Show first 3 sub-problems
            print(f"\nSample sub-problems:")
            for i, sp in enumerate(plan.sub_problems[:3], 1):
                print(f"  {i}. {sp.title} (confidence: {sp.confidence:.2f})")

        except Exception as e:
            print(f"Error: {e}")
            results[name] = None

    return results


def example_3_executor_usage():
    """Example 3: Using DecompositionStrategyExecutor."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: DecompositionStrategyExecutor Usage")
    print("=" * 80)

    # Create executor with custom config
    config = {
        'max_depth': 4,
        'max_subproblems': 12,
        'num_clusters': 6
    }

    executor = DecompositionStrategyExecutor(config)

    problem = ProblemDefinition(
        problem_id=generate_id("executor"),
        title="Implement Data Pipeline",
        description=(
            "Create a data processing pipeline for real-time analytics. "
            "Ingest data from multiple sources, transform it, and store "
            "in a data warehouse. Include monitoring and alerting."
        ),
        domain="data_science",
        complexity="complex",
        priority="high",
        estimated_effort="large",
        requirements=[
            "Real-time data ingestion",
            "Data transformation",
            "Data quality validation",
            "Monitoring and alerting",
            "Scalable architecture"
        ],
        constraints=[
            "Use Apache Kafka",
            "Snowflake warehouse",
            "Latency < 5 seconds"
        ],
        created_at=datetime.utcnow()
    )

    print(f"\nProblem: {problem.title}")

    # Auto-select and execute
    print("\nAuto-selecting strategy...")
    plan = executor.execute_with_auto_selection(problem)

    print(f"\nSelected plan created: {plan.plan_id}")
    print(f"Sub-problems: {len(plan.sub_problems)}")

    # Analyze results
    print("\nAnalysis:")
    print(f"  Total sub-problems: {len(plan.sub_problems)}")
    print(f"  Dependency edges: {sum(len(v) for v in plan.dependencies.values())}")

    # Confidence distribution
    high_conf = len([sp for sp in plan.sub_problems if sp.confidence > 0.8])
    med_conf = len([sp for sp in plan.sub_problems if 0.6 <= sp.confidence <= 0.8])
    low_conf = len([sp for sp in plan.sub_problems if sp.confidence < 0.6])

    print(f"\nConfidence Distribution:")
    print(f"  High (>0.8): {high_conf}")
    print(f"  Medium (0.6-0.8): {med_conf}")
    print(f"  Low (<0.6): {low_conf}")

    return plan, executor


def example_4_dependency_analysis():
    """Example 4: Analyze dependencies in detail."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Dependency Analysis")
    print("=" * 80)

    problem = ProblemDefinition(
        problem_id=generate_id("deps"),
        title="Multi-Phase Software Project",
        description=(
            "Execute a multi-phase software project including requirements gathering, "
            "design, development, testing, and deployment. Each phase depends on "
            "the completion of previous phases."
        ),
        domain="software_engineering",
        complexity="moderate",
        priority="medium",
        estimated_effort="medium",
        requirements=[
            "Requirements gathering",
            "System design",
            "Development",
            "Testing",
            "Deployment"
        ],
        constraints=[
            "Agile methodology",
            "2-week sprints"
        ],
        created_at=datetime.utcnow()
    )

    # Use HYBRID for good phase detection
    plan = decompose_hybrid(problem, depth=2)

    print(f"\nProblem: {problem.title}")
    print(f"Strategy: HYBRID")

    # Analyze dependencies
    print(f"\nDependency Analysis:")
    print(f"Total dependencies: {len(plan.dependencies)}")

    for from_id, to_ids in plan.dependencies.items():
        # Find sub-problem titles
        from_sp = next((sp for sp in plan.sub_problems if sp.sub_problem_id == from_id), None)
        if from_sp:
            print(f"\n  {from_sp.title} ->")
            for to_id in to_ids:
                to_sp = next((sp for sp in plan.sub_problems if sp.sub_problem_id == to_id), None)
                if to_sp:
                    print(f"    - {to_sp.title}")

    # Show execution order
    print(f"\nExecution Order:")
    for i, sp_id in enumerate(plan.execution_order, 1):
        sp = next((s for s in plan.sub_problems if s.sub_problem_id == sp_id), None)
        if sp:
            print(f"  {i}. {sp.title}")

    return plan


def example_5_integration_with_fractal_pipeline():
    """Example 5: Integration with FractalPipelineCoordinator (if available)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Integration with Fractal Pipeline")
    print("=" * 80)

    if not FRACTAL_PIPELINE_AVAILABLE:
        print("\n[WARN] FractalPipelineCoordinator not available")
        print("       This example requires problem_fractal_pipeline.py")
        return None

    # Create problem
    problem = ProblemDefinition(
        problem_id=generate_id("fractal"),
        title="Complex System Integration",
        description=(
            "Integrate multiple systems into a unified platform. "
            "Includes authentication, data synchronization, "
            "API integration, and monitoring."
        ),
        domain="software_engineering",
        complexity="complex",
        priority="high",
        estimated_effort="large",
        requirements=[
            "System integration",
            "Data synchronization",
            "API integration",
            "Monitoring"
        ],
        constraints=[],
        created_at=datetime.utcnow()
    )

    print(f"\nProblem: {problem.title}")

    # Step 1: Decompose using decomposition_strategy
    print("\nStep 1: Decomposing problem...")
    plan = execute_strategy("HYBRID", problem, depth=2)
    print(f"  Created {len(plan.sub_problems)} sub-problems")

    # Step 2: Create fractal pipeline config
    print("\nStep 2: Setting up Fractal Pipeline...")
    config = FractalPipelineConfig(
        enable_roma_decomposition=True,
        enable_mdap_maker_solving=True,
        enable_gauntlet_solving=True,
        roma_max_depth=3
    )

    coordinator = FractalPipelineCoordinator(config)

    # Step 3: Execute through fractal pipeline
    print("\nStep 3: Executing through Fractal Pipeline...")
    try:
        result = coordinator.run(
            problem_statement=problem.description,
            requirements=problem.requirements
        )

        print(f"\n  [OK] Pipeline execution complete")
        print(f"       Recomposed solution length: {len(result.recomposed_solution)} chars")
        print(f"       Final accepted: {result.final_accepted}")

        return result

    except Exception as e:
        print(f"\n  [FAIL] Pipeline execution failed: {e}")
        return None


def example_6_strategy_selection_details():
    """Example 6: Detailed strategy selection analysis."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Strategy Selection Analysis")
    print("=" * 80)

    # Create different types of problems
    problems = [
        ProblemDefinition(
            problem_id=generate_id("simple"),
            title="Simple Task",
            description="A simple, straightforward task.",
            domain="general",
            complexity="simple",
            priority="low",
            estimated_effort="small",
            requirements=["One requirement"],
            constraints=[],
            created_at=datetime.utcnow()
        ),
        ProblemDefinition(
            problem_id=generate_id("structured"),
            title="Structured Project",
            description="First, design the system. Then, implement the components. " * 5,
            domain="software_engineering",
            complexity="moderate",
            priority="medium",
            estimated_effort="medium",
            requirements=[f"Phase {i}" for i in range(5)],
            constraints=["Budget limit"],
            created_at=datetime.utcnow()
        ),
        ProblemDefinition(
            problem_id=generate_id("complex"),
            title="Complex Enterprise System",
            description="Build a complex system with many components. " * 20,
            domain="software_engineering",
            complexity="complex",
            priority="high",
            estimated_effort="large",
            requirements=[f"Requirement {i}" for i in range(15)],
            constraints=[f"Constraint {i}" for i in range(8)],
            created_at=datetime.utcnow()
        ),
    ]

    selector = StrategySelector()

    for problem in problems:
        print(f"\n{'-' * 40}")
        print(f"Problem: {problem.title}")
        print(f"Complexity: {problem.complexity}")
        print(f"Requirements: {len(problem.requirements)}")

        # Get scores for all strategies
        print(f"\nStrategy Scores:")
        for strategy in SovereignDecompositionStrategy:
            score = selector._score_strategy(problem, strategy)
            print(f"  {strategy.value:12s}: {score:.2f}")

        # Get selected strategy
        selected = selector.select_strategy(problem)
        print(f"\nSelected: {selected.value} (*)")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DECOMPOSITION STRATEGY - INTEGRATION EXAMPLES")
    print("=" * 80)
    print("\nThis demonstration shows how to use decomposition_strategy.py")
    print("within the Sovereign system ecosystem.")

    try:
        # Run examples
        example_1_basic_decomposition()
        example_2_strategy_comparison()
        example_3_executor_usage()
        example_4_dependency_analysis()
        example_5_integration_with_fractal_pipeline()
        example_6_strategy_selection_details()

        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY [OK]")
        print("=" * 80)

    except Exception as e:
        print(f"\n[FAIL] Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
