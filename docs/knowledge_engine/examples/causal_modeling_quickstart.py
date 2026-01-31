"""
Causal Modeling Quickstart Example

This example demonstrates how to use the knowledge engine's causal modeling
capabilities with the causal-learn integration.

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
import numpy as np
from datetime import datetime, UTC
from typing import Dict, Any, List

# Import knowledge engine components
from knowledge_engine.causal_modeling import CausalModelBuilder
from knowledge_engine.schemas.long_horizon import CausalModel


def generate_synthetic_outcomes(n_samples: int = 200) -> List[Dict[str, Any]]:
    """
    Generate synthetic agent outcomes with known causal structure.

    Causal structure:
        exploration_rate → fitness → diversity
        population_size → fitness
        temperature → diversity
    """
    np.random.seed(42)

    # Generate factors
    exploration_rate = np.random.beta(2, 5, n_samples)  # Mostly low
    population_size = np.random.randint(50, 200, n_samples)
    temperature = np.random.uniform(0.1, 2.0, n_samples)

    # Generate outcomes with causal relationships
    # fitness = 0.5 * exploration_rate + 0.3 * (population_size / 200) + noise
    fitness = (
        0.5 * exploration_rate +
        0.3 * (population_size / 200.0) +
        np.random.randn(n_samples) * 0.1
    )

    # diversity = 0.4 * fitness + 0.2 * temperature + noise
    diversity = (
        0.4 * fitness +
        0.2 * temperature +
        np.random.randn(n_samples) * 0.15
    )

    # convergence_speed = 0.6 * fitness + noise
    convergence_speed = (
        0.6 * fitness +
        np.random.randn(n_samples) * 0.2
    )

    # Build outcome list
    outcomes = []
    for i in range(n_samples):
        outcomes.append({
            "context": {
                "exploration_rate": float(exploration_rate[i]),
                "population_size": int(population_size[i]),
                "temperature": float(temperature[i])
            },
            "metrics": {
                "fitness": float(fitness[i]),
                "diversity": float(diversity[i]),
                "convergence_speed": float(convergence_speed[i])
            },
            "timestamp": datetime.now(UTC).isoformat()
        })

    return outcomes


async def example_basic_causal_discovery():
    """Example 1: Basic causal discovery"""
    print("\n" + "="*80)
    print("Example 1: Basic Causal Discovery")
    print("="*80)

    # Generate synthetic data
    outcomes = generate_synthetic_outcomes(n_samples=200)
    print(f"Generated {len(outcomes)} synthetic outcomes")

    # Initialize builder
    builder = CausalModelBuilder()

    # Build causal model
    print("\nBuilding causal model using PC algorithm...")
    model = await builder.build_model(
        domain="synthetic",
        outcomes=outcomes,
        method="pc",
        alpha=0.05,
        indep_test="fisherz"
    )

    print(f"\n✓ Built model: {model.model_id}")
    print(f"  Domain: {model.domain}")
    print(f"  Factors: {len(model.factors)}")
    print(f"  Outcomes: {len(model.outcomes)}")
    print(f"  Relationships: {len(model.relationships)}")
    print(f"  Graph nodes: {model.graph_data['num_nodes']}")
    print(f"  Graph edges: {model.graph_data['num_edges']}")

    # Display discovered relationships
    print("\nDiscovered Causal Relationships:")
    print("-" * 80)
    for i, rel in enumerate(model.relationships, 1):
        print(f"\n{i}. {rel.cause} → {rel.effect}")
        print(f"   Strength: {rel.strength:.3f}")
        print(f"   Confidence: {rel.confidence:.3f}")
        print(f"   Mechanism: {rel.mechanism}")
        if rel.evidence:
            print(f"   Evidence: {rel.evidence[0]}")

    return model, builder


async def example_intervention_prediction(model: CausalModel, builder: CausalModelBuilder):
    """Example 2: Predict intervention effects"""
    print("\n" + "="*80)
    print("Example 2: Intervention Prediction")
    print("="*80)

    # Get a factor to intervene on
    if model.factors:
        factor = model.factors[0]
        print(f"\nPredicting effect of intervening on: {factor}")

        # Predict effect of increasing the factor
        prediction = await builder.predict_intervention(
            model=model,
            cause=factor,
            value=0.8
        )

        print(f"\nIntervention: {prediction.intervention}")
        print(f"Predicted Effect: {prediction.predicted_effect:.3f}")
        print(f"Confidence: {prediction.confidence:.3f}")

        if prediction.alternative_outcomes:
            print("\nAlternative Scenarios:")
            for alt, effect in prediction.alternative_outcomes:
                print(f"  {alt}: {effect:.3f}")

        if prediction.risk_assessment:
            print("\nRisk Assessment:")
            for risk in prediction.risk_assessment:
                print(f"  ⚠ {risk}")


async def example_outcome_explanation(model: CausalModel, builder: CausalModelBuilder):
    """Example 3: Explain outcomes"""
    print("\n" + "="*80)
    print("Example 3: Outcome Explanation")
    print("="*80)

    if model.outcomes:
        outcome = model.outcomes[0]
        print(f"\nExplaining outcome: {outcome}")

        explanation = await builder.explain_outcome(
            model=model,
            outcome=outcome
        )

        print(f"\nCauses: {', '.join(explanation.causes)}")
        print(f"Overall Confidence: {explanation.confidence:.3f}")

        print("\nContributions:")
        for cause, contrib in sorted(
            explanation.contribution.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {cause}: {contrib:.3f}")

        if explanation.counterfactuals:
            print("\nCounterfactuals:")
            for cf in explanation.counterfactuals:
                print(f"  • {cf}")


async def example_model_persistence(model: CausalModel, builder: CausalModelBuilder):
    """Example 4: Model persistence"""
    print("\n" + "="*80)
    print("Example 4: Model Persistence")
    print("="*80)

    # Store model (will warn if no knowledge engine)
    print("\nStoring model...")
    model_id = await builder.store_model(model, version=1)
    print(f"✓ Stored model: {model_id}")

    # Try to load it back
    print("\nLoading model...")
    loaded_model = await builder.load_model(
        model_id=model_id,
        domain=model.domain
    )

    if loaded_model:
        print(f"✓ Loaded model: {loaded_model.model_id}")
        print(f"  Relationships: {len(loaded_model.relationships)}")
    else:
        print("ℹ Model not in persistent storage (expected without knowledge engine)")


async def example_cross_domain_learning(builder: CausalModelBuilder):
    """Example 5: Cross-domain learning"""
    print("\n" + "="*80)
    print("Example 5: Cross-Domain Learning")
    print("="*80)

    # Build model for source domain
    print("\nBuilding source domain model (finance)...")
    finance_outcomes = generate_synthetic_outcomes(n_samples=150)
    finance_model = await builder.build_model(
        domain="finance",
        outcomes=finance_outcomes,
        method="pc"
    )
    print(f"✓ Built finance model: {len(finance_model.relationships)} relationships")

    # Build model for target domain
    print("\nBuilding target domain model (trading)...")
    trading_outcomes = generate_synthetic_outcomes(n_samples=150)
    trading_model = await builder.build_model(
        domain="trading",
        outcomes=trading_outcomes,
        method="pc"
    )
    print(f"✓ Built trading model: {len(trading_model.relationships)} relationships")

    # Transfer knowledge
    print("\nTransferring causal knowledge...")
    suggested = await builder.transfer_causal_knowledge(
        source_domain="finance",
        target_domain="trading",
        min_similarity=0.5
    )

    if suggested:
        print(f"✓ Transferred {len(suggested)} suggested relationships:")
        for rel in suggested[:3]:  # Show top 3
            print(f"\n  {rel.cause} → {rel.effect}")
            print(f"    Confidence: {rel.confidence:.3f}")
            print(f"    Source: {rel.mechanism}")
    else:
        print("ℹ No similar models found (expected without Qdrant)")


async def example_algorithm_comparison():
    """Example 6: Compare different algorithms"""
    print("\n" + "="*80)
    print("Example 6: Algorithm Comparison")
    print("="*80)

    outcomes = generate_synthetic_outcomes(n_samples=200)
    algorithms = ["pc", "ges", "direct_lingam"]

    results = {}

    for algorithm in algorithms:
        print(f"\nTesting {algorithm.upper()} algorithm...")

        try:
            builder = CausalModelBuilder()
            model = await builder.build_model(
                domain=f"test_{algorithm}",
                outcomes=outcomes,
                method=algorithm
            )

            results[algorithm] = {
                "relationships": len(model.relationships),
                "nodes": model.graph_data["num_nodes"],
                "edges": model.graph_data["num_edges"]
            }

            print(f"  ✓ Relationships: {len(model.relationships)}")
            print(f"  ✓ Graph: {model.graph_data['num_nodes']} nodes, "
                  f"{model.graph_data['num_edges']} edges")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[algorithm] = None

    # Summary
    print("\n" + "-"*80)
    print("Algorithm Comparison Summary:")
    print("-"*80)

    for algo, result in results.items():
        if result:
            print(f"\n{algo.upper()}:")
            print(f"  Relationships: {result['relationships']}")
            print(f"  Graph Density: {result['edges'] / max(1, result['nodes']**2):.3f}")


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("CAUSAL MODELING QUICKSTART")
    print("="*80)
    print("\nThis example demonstrates causal discovery with causal-learn integration")
    print("on synthetic agent outcome data.")

    try:
        # Example 1: Basic discovery
        model, builder = await example_basic_causal_discovery()

        # Example 2: Intervention prediction
        await example_intervention_prediction(model, builder)

        # Example 3: Outcome explanation
        await example_outcome_explanation(model, builder)

        # Example 4: Persistence
        await example_model_persistence(model, builder)

        # Example 5: Cross-domain learning
        await example_cross_domain_learning(builder)

        # Example 6: Algorithm comparison
        await example_algorithm_comparison()

        print("\n" + "="*80)
        print("QUICKSTART COMPLETE")
        print("="*80)
        print("\nNext Steps:")
        print("  1. Try with your own agent outcomes")
        print("  2. Experiment with different algorithms")
        print("  3. Configure persistence (Neo4j, Qdrant)")
        print("  4. Explore cross-domain learning")
        print("\nFor more details, see: knowledge_engine/CAUSAL_MODELING.md")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
