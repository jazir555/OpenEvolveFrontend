"""
Example: Extract knowledge from LoongFlow PES run

This example demonstrates how to use the LoongFlowKnowledgeExtractor
to extract learning artifacts from LoongFlow Plan-Execute-Summarize (PES)
evolutionary runs and store them in the Knowledge Engine.

The extracted artifacts include:
1. Planning Strategy - Strategic approach from planning phase
2. Execution Pattern - Execution patterns and efficiency metrics
3. Reflection Insight - Learnings from summary/reflection
4. Evolutionary Lineage - Evolutionary tree and ancestry
5. Optimized Solution - Best solution found

Usage:
    python examples/loongflow_knowledge_extraction.py

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Import the LoongFlow integration
from knowledge_engine.integrations.loongflow_integration import (
    LoongFlowKnowledgeExtractor,
    create_loongflow_extractor,
)

# Optional: Import KnowledgeEngine if available
try:
    from knowledge_engine import KnowledgeEngine
    HAS_KE = True
except ImportError:
    HAS_KE = False
    print("Note: KnowledgeEngine not available, running without storage")


def create_mock_pes_run() -> Dict[str, Any]:
    """
    Create a mock LoongFlow PES run result for demonstration.

    In a real scenario, this would come from an actual LoongFlow PES execution.
    """
    return {
        "plan": {
            "strategy": "Use gradient descent with momentum and adaptive learning rate",
            "success_rate": 0.85,
            "iterations": 50,
            "approach": "hybrid_evolutionary",
            "planning_time": 2.5,
        },
        "execution": {
            "early_stops": [15, 25],
            "convergence_rate": 0.95,
            "iterations_to_best": 25,
            "total_evaluations": 30,
            "efficiency_gain": 0.60,
            "time_saved": 1200,
            "execution_time": 45.3,
        },
        "summary": {
            "insights": "Momentum helps escape local optima effectively. "
            "Adaptive learning rate prevents oscillation.",
            "what_worked": [
                "momentum",
                "adaptive_learning_rate",
                "early_stopping",
                "gradient_clipping",
            ],
            "what_failed": [
                "fixed_learning_rate",
                "large_batch_size",
                "vanilla_gradient_descent",
            ],
            "recommendations": [
                "Use momentum in future runs",
                "Implement adaptive learning rate scheduling",
                "Apply gradient clipping to prevent explosion",
            ],
        },
        "evolutionary_tree": {
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 45,
            "best_lineage": ["root", "gen1", "gen2", "gen3", "best"],
            "mutation_types": {
                "parameter_tweak": 20,
                "structure_change": 15,
                "hybridization": 10,
            },
        },
        "best_solution": {
            "code": """
def optimize_portfolio(weights, returns, risk_tolerance):
    '''
    Momentum-based portfolio optimization with adaptive learning rate.

    Args:
        weights: Initial portfolio weights
        returns: Historical returns matrix
        risk_tolerance: Risk tolerance parameter

    Returns:
        Optimized portfolio weights
    '''
    import numpy as np

    velocity = np.zeros_like(weights)
    momentum = 0.9
    learning_rate = 0.01
    beta1, beta2 = 0.9, 0.999
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    t = 0

    for i in range(100):
        # Compute gradient
        gradient = compute_portfolio_gradient(weights, returns, risk_tolerance)

        # Adam optimizer with momentum
        t += 1
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * (gradient ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update weights
        weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Apply momentum
        velocity = momentum * velocity - learning_rate * gradient
        weights = weights + velocity

        # Project to feasible region
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)

    return weights

def compute_portfolio_gradient(weights, returns, risk_tolerance):
    '''Compute gradient for portfolio optimization'''
    portfolio_return = np.dot(returns.mean(), weights)
    portfolio_variance = np.dot(weights.T, np.dot(returns.cov(), weights))
    gradient = returns.mean() - risk_tolerance * np.dot(returns.cov(), weights)
    return gradient
            """,
            "fitness": 0.95,
            "iteration": 25,
            "improvement": 0.40,
            "objective_value": 0.023,
        },
    }


async def example_basic_extraction():
    """
    Example 1: Basic artifact extraction without Knowledge Engine.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Artifact Extraction")
    print("=" * 80)

    # Create extractor (without Knowledge Engine)
    extractor = create_loongflow_extractor()

    # Create mock PES run results
    pes_run = create_mock_pes_run()

    # Extract artifacts
    print("\nExtracting artifacts from LoongFlow PES run...")
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_run,
        problem="Optimize portfolio allocation for maximum returns",
        problem_type="portfolio_optimization",
    )

    print(f"\n✓ Extracted {len(artifacts)} artifacts:")
    for i, artifact in enumerate(artifacts, 1):
        print(f"\n  {i}. {artifact['artifact_type'].upper()}")
        print(f"     ID: {artifact['id']}")
        print(f"     Confidence: {artifact['confidence']}")
        print(f"     Created: {artifact['created_at']}")

        # Show a preview of content
        content_preview = artifact['content'][:100] + "..." if len(artifact['content']) > 100 else artifact['content']
        print(f"     Content: {content_preview}")

        # Show key metadata
        metadata = artifact['metadata']
        if 'problem' in metadata:
            print(f"     Problem: {metadata['problem']}")
        if 'problem_type' in metadata:
            print(f"     Type: {metadata['problem_type']}")

    # Show extraction statistics
    stats = extractor.get_extraction_stats()
    print(f"\n✓ Extraction Statistics:")
    for artifact_type, count in stats.items():
        if count > 0:
            print(f"   - {artifact_type}: {count}")


async def example_with_knowledge_engine():
    """
    Example 2: Extraction with Knowledge Engine storage.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Artifact Extraction with Knowledge Engine")
    print("=" * 80)

    if not HAS_KE:
        print("\n⚠ Knowledge Engine not available. Skipping this example.")
        return

    # Initialize Knowledge Engine
    print("\nInitializing Knowledge Engine...")
    try:
        ke = KnowledgeEngine()
        await ke.initialize()
        print("✓ Knowledge Engine initialized")
    except Exception as e:
        print(f"⚠ Failed to initialize Knowledge Engine: {e}")
        print("  Running without storage...")
        ke = None

    # Create extractor with KE
    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=ke)

    # Create mock PES run
    pes_run = create_mock_pes_run()

    # Extract and store artifacts
    print("\nExtracting and storing artifacts...")
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_run,
        problem="Neural network hyperparameter optimization",
        problem_type="scientific",
    )

    print(f"\n✓ Extracted and stored {len(artifacts)} artifacts in Knowledge Engine")

    # Query for similar strategies
    print("\nQuerying for similar planning strategies...")
    strategies = await extractor.query_planning_strategies(
        problem_type="scientific",
        limit=5,
        min_success_rate=0.7,
    )

    if strategies:
        print(f"✓ Found {len(strategies)} similar strategies")
    else:
        print("  No strategies found (this is expected with mock data)")

    # Get efficiency metrics
    print("\nGetting efficiency metrics...")
    metrics = await extractor.get_efficiency_metrics(problem_type="scientific")

    if metrics:
        print("✓ Efficiency Metrics:")
        for key, value in metrics.items():
            print(f"   - {key}: {value}")
    else:
        print("  No metrics available (this is expected with mock data)")


async def example_multiple_runs():
    """
    Example 3: Extract knowledge from multiple PES runs.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multiple PES Runs")
    print("=" * 80)

    extractor = create_loongflow_extractor()
    extractor.reset_stats()

    # Simulate multiple runs
    problems = [
        ("Portfolio optimization", "finance"),
        ("Neural network training", "machine_learning"),
        ("Algorithm optimization", "scientific"),
        ("Resource allocation", "optimization"),
    ]

    print("\nProcessing multiple PES runs...")
    for i, (problem, problem_type) in enumerate(problems, 1):
        print(f"\n{i}. Processing: {problem}")

        # Create mock run with variations
        pes_run = create_mock_pes_run()
        pes_run["plan"]["success_rate"] = 0.7 + (i * 0.05)
        pes_run["best_solution"]["fitness"] = 0.85 + (i * 0.02)

        artifacts = await extractor.extract_from_pes_run(
            pes_run_results=pes_run,
            problem=problem,
            problem_type=problem_type,
        )

        print(f"   ✓ Extracted {len(artifacts)} artifacts")

    # Show aggregate statistics
    stats = extractor.get_extraction_stats()
    total_artifacts = sum(stats.values())

    print(f"\n✓ Total artifacts extracted: {total_artifacts}")
    print("\nBreakdown by type:")
    for artifact_type, count in stats.items():
        percentage = (count / total_artifacts * 100) if total_artifacts > 0 else 0
        print(f"   - {artifact_type}: {count} ({percentage:.1f}%)")


async def example_querying_knowledge():
    """
    Example 4: Querying extracted knowledge.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Querying Extracted Knowledge")
    print("=" * 80)

    if not HAS_KE:
        print("\n⚠ Knowledge Engine not available. Using mock data.")
        return

    extractor = create_loongflow_extractor()

    # First, populate with some data
    pes_run = create_mock_pes_run()
    await extractor.extract_from_pes_run(
        pes_run_results=pes_run,
        problem="Example problem",
        problem_type="finance",
    )

    print("\n1. Query Planning Strategies")
    print("   Searching for successful finance strategies...")

    strategies = await extractor.query_planning_strategies(
        problem_type="finance",
        limit=5,
        min_success_rate=0.7,
    )

    if strategies:
        print(f"   ✓ Found {len(strategies)} strategies")
        for i, strategy in enumerate(strategies[:3], 1):
            print(f"\n   Strategy {i}:")
            print(f"   - Success Rate: {strategy.get('metadata', {}).get('success_rate', 'N/A')}")
            print(f"   - Approach: {strategy.get('metadata', {}).get('planning_approach', 'N/A')}")
    else:
        print("   No strategies found (expected with mock data)")

    print("\n2. Get Efficiency Metrics")
    print("   Calculating average efficiency for finance problems...")

    metrics = await extractor.get_efficiency_metrics(problem_type="finance")

    if metrics:
        print("   ✓ Metrics retrieved:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   - {key}: {value:.3f}")
            else:
                print(f"   - {key}: {value}")
    else:
        print("   No metrics available (expected with mock data)")


async def example_save_to_file():
    """
    Example 5: Save extracted artifacts to JSON file.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Save Artifacts to File")
    print("=" * 80)

    extractor = create_loongflow_extractor()

    # Extract artifacts
    pes_run = create_mock_pes_run()
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_run,
        problem="Portfolio optimization",
        problem_type="finance",
    )

    # Save to file
    output_file = "loongflow_artifacts_example.json"
    with open(output_file, "w") as f:
        json.dump(artifacts, f, indent=2)

    print(f"\n✓ Saved {len(artifacts)} artifacts to {output_file}")

    # Show file info
    file_size = len(open(output_file, "r").read())
    print(f"  File size: {file_size} bytes")
    print(f"  Location: {output_file}")


async def main():
    """
    Run all examples.
    """
    print("\n" + "=" * 80)
    print("LoongFlow Knowledge Extraction Examples")
    print("=" * 80)
    print("\nThis example demonstrates how to extract knowledge artifacts")
    print("from LoongFlow PES runs and integrate them with the Knowledge Engine.")
    print("\nTimestamp:", datetime.now(timezone.utc).isoformat())

    try:
        # Run examples
        await example_basic_extraction()
        await example_with_knowledge_engine()
        await example_multiple_runs()
        await example_querying_knowledge()
        await example_save_to_file()

        print("\n" + "=" * 80)
        print("✓ All examples completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\n⚠ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
