"""
LoongFlow Knowledge Extraction - End-to-End Example

This example demonstrates the complete workflow of extracting knowledge
from LoongFlow PES runs and integrating with the Knowledge Engine.

Usage:
    python examples/loongflow_extraction_example.py

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
import logging
from datetime import datetime, timezone
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import LoongFlow integration
import sys
sys.path.insert(0, '..')

from knowledge_engine.integrations.loongflow_integration import (
    LoongFlowKnowledgeExtractor,
    PESRunResults,
    KnowledgeArtifact,
    ProblemDomain,
    ArtifactType,
)


def create_sample_pes_run() -> PESRunResults:
    """
    Create a sample PES run result for demonstration.

    In production, this would come from an actual LoongFlow PES execution.
    """
    return PESRunResults(
        plan={
            "strategy": "Use gradient descent with adaptive learning rate",
            "reasoning": "Convex optimization problem with local minima",
            "action_steps": [
                "Initialize weights with Xavier initialization",
                "Compute gradients using backpropagation",
                "Update weights with Adam optimizer",
                "Adapt learning rate based on validation loss",
            ],
            "success_criteria": {
                "convergence": 0.0001,
                "max_iterations": 1000,
            },
            "approach": "gradient_based",
            "success_rate": 0.87,
            "iterations": 50,
            "duration_ms": 1250,
        },
        execution={
            "early_stops": [15, 25, 35],
            "convergence_rate": 0.93,
            "iterations_to_best": 35,
            "total_evaluations": 40,
            "baseline_evaluations": 100,
            "time_saved": 75,
            "avg_iteration_time_ms": 85,
            "parameter_tuning": {
                "learning_rate_schedule": "exponential_decay",
                "batch_size": 32,
                "optimizer": "Adam",
            },
        },
        summary={
            "insights": "Adaptive learning rate was crucial for escaping local minima",
            "what_worked": [
                "Adam optimizer with beta1=0.9, beta2=0.999",
                "Exponential learning rate decay",
                "Xavier initialization",
                "Batch normalization",
            ],
            "what_failed": [
                "Fixed learning rate (0.001)",
                "SGD without momentum",
                "Random initialization",
            ],
            "recommendations": [
                "Always use Adam for this problem type",
                "Start with learning rate 0.001, decay exponentially",
                "Use Xavier initialization for deep networks",
            ],
            "adaptation_patterns": [
                "Learning rate decay when validation loss plateaus",
                "Batch normalization after each layer",
            ],
            "lessons_learned": [
                "Early stopping prevents overfitting",
                "Adaptive optimizers converge faster",
            ],
        },
        evolutionary_tree={
            "generations": 10,
            "avg_branching": 2.5,
            "total_mutations": 25,
            "best_path": ["gen_0", "gen_3", "gen_5", "gen_8", "gen_10"],
            "solutions": [
                {"gen": 0, "fitness": 0.45, "mutation": "initial"},
                {"gen": 3, "fitness": 0.67, "mutation": "add_momentum"},
                {"gen": 5, "fitness": 0.78, "mutation": "adaptive_lr"},
                {"gen": 8, "fitness": 0.89, "mutation": "batch_norm"},
                {"gen": 10, "fitness": 0.93, "mutation": "final_tuning"},
            ],
            "tree_structure": {
                "root": "gen_0",
                "branches": {
                    "gen_1": {"parent": "gen_0", "mutation": "increase_lr"},
                    "gen_2": {"parent": "gen_0", "mutation": "add_momentum"},
                    "gen_3": {"parent": "gen_2", "mutation": "tune_momentum"},
                }
            },
        },
        best_solution={
            "code": """
def optimize_neural_network(X, y, config):
    import numpy as np
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Learning rate scheduler
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        factor=0.5, patience=5, min_lr=1e-6
    )

    # Early stopping
    early_stop = keras.callbacks.EarlyStopping(
        patience=10, restore_best_weights=True
    )

    history = model.fit(
        X, y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[lr_schedule, early_stop],
        verbose=0
    )

    return model, history
            """,
            "fitness": 0.93,
            "iteration": 35,
            "improvement": 0.48,  # From 0.45 baseline to 0.93
            "params": {
                "learning_rate": 0.001,
                "epochs": 35,
                "batch_size": 32,
                "hidden_layers": [128, 64],
                "dropout_rate": 0.3,
            },
            "evaluation": {
                "accuracy": 0.93,
                "precision": 0.91,
                "recall": 0.89,
                "f1_score": 0.90,
            },
            "parents": ["gen_8"],
            "mutations": ["final_tuning"],
        },
        run_metadata={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 150,
            "total_llm_calls": 15,
            "total_evaluations": 40,
        },
    )


async def example_basic_extraction():
    """Example 1: Basic extraction without Knowledge Engine"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Extraction (No Storage)")
    print("="*70 + "\n")

    # Create extractor without Knowledge Engine
    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)

    # Create sample PES run
    pes_results = create_sample_pes_run()

    # Extract artifacts
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_results,
        problem="Optimize neural network architecture for binary classification",
        problem_type="neural_network_optimization",
    )

    print(f"Extracted {len(artifacts)} artifacts:\n")

    for i, artifact in enumerate(artifacts, 1):
        print(f"{i}. {artifact.artifact_type.upper()}")
        print(f"   Domain: {artifact.domain}")
        print(f"   Confidence: {artifact.confidence:.2f}")
        print(f"   Content keys: {list(artifact.content.keys())}")
        print(f"   Metadata keys: {list(artifact.metadata.keys())}")
        print()

    # Show extraction stats
    stats = extractor.get_extraction_stats()
    print(f"Extraction Statistics:")
    for artifact_type, count in stats.items():
        print(f"  {artifact_type}: {count}")
    print()


async def example_with_mock_storage():
    """Example 2: Extraction with mock storage backends"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Extraction with Mock Storage")
    print("="*70 + "\n")

    # Create mock Knowledge Engine
    from unittest.mock import Mock, AsyncMock

    mock_ke = Mock()
    mock_ke.graphiti_bridge = Mock()
    mock_ke.graphiti_bridge.add_episode = AsyncMock()
    mock_ke.qdrant_bridge = Mock()
    mock_ke.qdrant_bridge.upsert = AsyncMock()
    mock_ke.neo4j = Mock()
    mock_ke.neo4j.run = AsyncMock()
    mock_ke.mongodb = Mock()
    mock_ke.mongodb.insert_one = AsyncMock()

    # Create extractor with mock KE
    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)

    print(f"Storage Backends:")
    print(f"  Graphiti: {extractor.graphiti is not None}")
    print(f"  Qdrant: {extractor.qdrant is not None}")
    print(f"  Neo4j: {extractor.neo4j is not None}")
    print(f"  MongoDB: {extractor.mongodb is not None}")
    print()

    # Create sample PES run
    pes_results = create_sample_pes_run()

    # Extract artifacts (will be stored in mock backends)
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_results,
        problem="Optimize neural network architecture",
        problem_type="ml_optimization",
        domain="machine_learning",
        run_id="example_run_001",
    )

    print(f"Extracted and stored {len(artifacts)} artifacts\n")

    # Show storage calls
    print(f"Storage Operations:")
    print(f"  Graphiti episodes: {mock_ke.graphiti_bridge.add_episode.call_count}")
    print(f"  Neo4j queries: {mock_ke.neo4j.run.call_count}")
    print(f"  MongoDB inserts: {mock_ke.mongodb.insert_one.call_count}")
    print()


async def example_artifact_inspection():
    """Example 3: Inspect individual artifacts"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Artifact Inspection")
    print("="*70 + "\n")

    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)
    pes_results = create_sample_pes_run()

    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_results,
        problem="Optimize neural network",
        problem_type="ml_optimization",
    )

    # Find each artifact type
    planning = next((a for a in artifacts if a.artifact_type == "planning_strategy"), None)
    execution = next((a for a in artifacts if a.artifact_type == "execution_pattern"), None)
    reflection = next((a for a in artifacts if a.artifact_type == "reflection_insight"), None)
    lineage = next((a for a in artifacts if a.artifact_type == "evolutionary_lineage"), None)
    solution = next((a for a in artifacts if a.artifact_type == "optimized_solution"), None)

    # Planning Strategy
    if planning:
        print("PLANNING STRATEGY:")
        print(f"  Strategy: {planning.content['strategy']}")
        print(f"  Approach: {planning.content['planning_approach']}")
        print(f"  Success Rate: {planning.metadata['success_rate']}")
        print(f"  Action Steps: {len(planning.content['action_steps'])}")
        print()

    # Execution Pattern
    if execution:
        print("EXECUTION PATTERN:")
        print(f"  Early Stops: {execution.content['early_stopping_events']}")
        print(f"  Convergence Rate: {execution.content['convergence_rate']:.2f}")
        print(f"  Efficiency Gain: {execution.metadata['efficiency_gain']:.1%}")
        print(f"  Time Saved: {execution.metadata['time_saved_seconds']}s")
        print()

    # Reflection Insight
    if reflection:
        print("REFLECTION INSIGHT:")
        print(f"  Insights: {reflection.content['insights']}")
        print(f"  What Worked: {', '.join(reflection.content['what_worked'][:2])}...")
        print(f"  What Failed: {', '.join(reflection.content['what_failed'][:2])}...")
        print()

    # Evolutionary Lineage
    if lineage:
        print("EVOLUTIONARY LINEAGE:")
        print(f"  Generations: {lineage.content['generations']}")
        print(f"  Branching Factor: {lineage.content['branching_factor']}")
        print(f"  Best Path: {' → '.join(lineage.content['best_path'])}")
        print()

    # Optimized Solution
    if solution:
        print("OPTIMIZED SOLUTION:")
        print(f"  Fitness: {solution.content['fitness']:.3f}")
        print(f"  Iteration Found: {solution.content['iteration_found']}")
        print(f"  Improvement: {solution.content['improvement_over_baseline']:.1%}")
        print(f"  Code Size: {solution.metadata['solution_size']} chars")
        print()


async def example_domain_detection():
    """Example 4: Domain auto-detection"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Domain Auto-Detection")
    print("="*70 + "\n")

    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)

    # Test domain detection
    test_cases = [
        ("Optimize portfolio allocation for tech stocks", "financial", "finance"),
        ("Train neural network for image classification", "ml", "machine_learning"),
        ("Design experiment for chemical reaction analysis", "scientific", "science"),
        ("Prove mathematical theorem about prime numbers", "math", "mathematics"),
        ("Develop algorithmic trading strategy", "trading", "trading"),
        ("Optimize structural design for bridge", "engineering", "engineering"),
        ("Solve generic optimization problem", "general", "general"),
    ]

    print("Domain Detection Results:")
    for problem, problem_type, expected_domain in test_cases:
        detected = extractor._detect_domain(problem, problem_type)
        status = "✓" if detected == expected_domain else "✗"
        print(f"  {status} Problem: {problem[:50]}")
        print(f"     Detected: {detected}")
        print()


async def example_query_methods():
    """Example 5: Query methods (with mock)"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Query Methods")
    print("="*70 + "\n")

    # Create mock KE with query capability
    from unittest.mock import Mock, AsyncMock

    mock_ke = Mock()
    mock_ke.query = AsyncMock(return_value=[
        {
            "a.content": {"strategy": "Use Adam optimizer"},
            "a.metadata": {"success_rate": 0.87, "problem": "Neural network optimization"}
        },
        {
            "a.content": {"strategy": "Use genetic algorithm"},
            "a.metadata": {"success_rate": 0.82, "problem": "Portfolio optimization"}
        },
    ])

    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=mock_ke)

    # Query planning strategies
    strategies = await extractor.query_planning_strategies(
        problem_type="neural_network_optimization",
        domain="machine_learning",
        limit=10,
        min_success_rate=0.7,
    )

    print(f"Found {len(strategies)} planning strategies:\n")
    for strategy in strategies:
        print(f"  Strategy: {strategy.get('a.content', {}).get('strategy', 'N/A')}")
        print(f"  Success Rate: {strategy.get('a.metadata', {}).get('success_rate', 0):.2f}")
        print()

    # Get efficiency metrics
    mock_ke.query.return_value = [
        {
            "avg_efficiency": 0.62,
            "avg_evals": 38.0,
            "total_runs": 15,
        }
    ]

    metrics = await extractor.get_efficiency_metrics(
        problem_type="neural_network_optimization",
        domain="machine_learning",
    )

    print("Efficiency Metrics:")
    print(f"  Average Efficiency Gain: {metrics['avg_efficiency_gain']:.1%}")
    print(f"  Average Evaluations Saved: {metrics['avg_evaluations_saved']:.1f}")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
    print(f"  Total Runs: {metrics['total_runs']}")
    print()


async def example_serialization():
    """Example 6: Artifact serialization"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Artifact Serialization")
    print("="*70 + "\n")

    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)
    pes_results = create_sample_pes_run()

    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=pes_results,
        problem="Optimize neural network",
        problem_type="ml",
    )

    # Show serialization formats
    solution = next((a for a in artifacts if a.artifact_type == "optimized_solution"), None)

    if solution:
        # Dictionary format
        print("1. Dictionary Format:")
        artifact_dict = solution.to_dict()
        print(f"   Keys: {list(artifact_dict.keys())}")
        print()

        # Graphiti episode format
        print("2. Graphiti Episode Format:")
        episode = solution.to_graphiti_episode()
        print(f"   Preview: {episode[:200]}...")
        print()

        # Qdrant payload format
        print("3. Qdrant Payload Format:")
        payload = solution.to_qdrant_payload()
        print(f"   Keys: {list(payload.keys())}")
        print(f"   Timestamp: {payload['timestamp']}")
        print()


async def example_error_handling():
    """Example 7: Error handling"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Error Handling")
    print("="*70 + "\n")

    extractor = LoongFlowKnowledgeExtractor(knowledge_engine=None)

    # Test with invalid input
    print("1. Invalid Input Type:")
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results="invalid",  # Wrong type
        problem="Test",
        problem_type="test",
    )
    print(f"   Result: {len(artifacts)} artifacts (graceful degradation)")
    print()

    # Test with incomplete data
    print("2. Incomplete PES Data:")
    incomplete_pes = PESRunResults(
        plan={"strategy": "Test"},
        execution={},
        summary={},
        evolutionary_tree={},
        best_solution={},
    )
    artifacts = await extractor.extract_from_pes_run(
        pes_run_results=incomplete_pes,
        problem="Test",
        problem_type="test",
    )
    print(f"   Result: {len(artifacts)} artifacts extracted from available data")
    print()

    # Test domain detection with edge cases
    print("3. Edge Case Domain Detection:")
    domain = extractor._detect_domain("", "")
    print(f"   Empty input → Domain: {domain}")
    print()


async def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("LOONGFLOW KNOWLEDGE EXTRACTION - EXAMPLES")
    print("="*70)

    try:
        await example_basic_extraction()
        await example_with_mock_storage()
        await example_artifact_inspection()
        await example_domain_detection()
        await example_query_methods()
        await example_serialization()
        await example_error_handling()

        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")

    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())
