"""
Long-Horizon Learning Quickstart

Demonstrates the complete long-horizon learning workflow for agentic systems.

This example shows how to:
1. Record streaming outcomes with online learning
2. Set up A/B tests for strategy comparison
3. Build causal models from outcomes
4. Extract meta-patterns across workflows
5. Transfer knowledge to new domains

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import asyncio
from datetime import datetime, UTC
import numpy as np

# Import knowledge engine components
from knowledge_engine import (
    OnlineLearner,
    ABTestFramework,
    CausalModelBuilder,
    MetaLearner,
    UnifiedEvolutionKnowledgeExtractor
)

# Import schemas
from knowledge_engine.schemas.long_horizon import (
    LearningOutcome,
    OutcomeType,
    ExplorationStrategy
)


# ========================================================================
# SIMULATED WORKFLOW EXECUTION
# ========================================================================

async def execute_workflow(strategy: str, problem: dict) -> dict:
    """
    Simulate workflow execution

    In a real system, this would:
    - Initialize the agent with the strategy
    - Execute the workflow
    - Return results

    For simulation, we generate synthetic results.
    """
    # Simulate execution time
    await asyncio.sleep(0.01)

    # Generate outcome based on strategy and problem
    if strategy == "pes":
        # PES performs well for expensive evaluations
        base_fitness = 0.85 if problem.get("evaluation_cost") == "high" else 0.65
    elif strategy == "qd":
        # QD performs well for diversity
        base_fitness = 0.80 if problem.get("need_diversity") else 0.60
    else:
        # Hybrid
        base_fitness = 0.75

    # Add noise
    fitness = base_fitness + np.random.randn() * 0.1
    fitness = max(0.0, min(1.0, fitness))

    # Determine success
    success = fitness > 0.7

    # Return outcome
    return {
        "workflow_id": f"wf_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}",
        "strategy_used": strategy,
        "outcome_type": OutcomeType.SUCCESS.value if success else OutcomeType.PARTIAL.value,
        "metrics": {
            "fitness": fitness,
            "convergence_time": 100 + np.random.randint(50),
            "llm_calls": 30 + np.random.randint(30)
        },
        "context": {
            "domain": problem.get("domain", "general"),
            "evaluation_cost": problem.get("evaluation_cost", "medium"),
            "exploration_rate": problem.get("exploration_rate", 0.5)
        }
    }


# ========================================================================
# ONLINE LEARNING DEMO
# ========================================================================

async def demo_online_learning():
    """Demonstrate online learning from streaming outcomes"""
    print("\n" + "="*70)
    print("ONLINE LEARNING DEMO")
    print("="*70)

    # Initialize learner
    learner = OnlineLearner(
        exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
        initial_epsilon=0.4,
        performance_window=20
    )

    print("\n1. Recording outcomes from workflow executions...")

    # Simulate 20 workflow executions
    for i in range(20):
        # Select strategy (explore or exploit)
        if await learner.should_explore():
            strategy = np.random.choice(["pes", "qd", "hybrid"])
            print(f"  Iteration {i+1}: Exploring with {strategy}")
        else:
            strategy = await learner.get_best_strategy("demo_workflow")
            if not strategy:
                strategy = "hybrid"
            print(f"  Iteration {i+1}: Exploiting with {strategy}")

        # Execute workflow
        problem = {
            "domain": "finance",
            "evaluation_cost": "high",
            "exploration_rate": 0.3
        }
        outcome_data = await execute_workflow(strategy, problem)

        # Create learning outcome
        outcome = LearningOutcome(
            workflow_id="demo_workflow",
            strategy_used=strategy,
            outcome_type=OutcomeType(outcome_data["outcome_type"]),
            metrics=outcome_data["metrics"],
            context=outcome_data["context"]
        )

        # Record outcome
        await learner.record_outcome(outcome)

    print("\n2. Analyzing strategy performance...")

    # Get performance for all strategies
    strategies = await learner.get_all_strategies("demo_workflow")
    for strat_id, perf in strategies.items():
        print(f"\n  Strategy: {strat_id}")
        print(f"    Total outcomes: {perf.total_outcomes}")
        print(f"    Moving average: {perf.moving_average:.3f}")
        print(f"    Success rate: {perf.success_rate:.1%}")

    # Get best strategy
    best = await learner.get_best_strategy("demo_workflow")
    print(f"\n  Best strategy: {best}")

    # Check for adaptation
    print("\n3. Checking for adaptation opportunities...")
    action = await learner.adapt_strategy("demo_workflow", 0.65)
    if action:
        print(f"  Recommendation: {action.description}")
        print(f"  Expected improvement: {action.expected_improvement:.1%}")
        print(f"  Confidence: {action.confidence:.1%}")
    else:
        print("  No adaptation needed")

    # Get statistics
    stats = await learner.get_statistics()
    print(f"\n4. Learning statistics:")
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Explore rate: {stats['explore_rate']:.1%}")
    print(f"  Current epsilon: {stats['current_epsilon']:.3f}")


# ========================================================================
# A/B TESTING DEMO
# ========================================================================

async def demo_ab_testing():
    """Demonstrate A/B testing for strategies"""
    print("\n" + "="*70)
    print("A/B TESTING DEMO")
    print("="*70)

    # Initialize framework
    framework = ABTestFramework(
        significance_level=0.05,
        min_sample_size=15,  # Lower for demo
        test_method="frequentist"
    )

    print("\n1. Creating A/B experiment...")

    # Create experiment
    experiment = await framework.create_experiment(
        name="PES vs QD for Finance",
        description="Compare PES and QD strategies for financial optimization",
        variants=["pes", "qd"]
    )

    print(f"  Experiment ID: {experiment.experiment_id}")
    print(f"  Variants: {list(experiment.variants.keys())}")

    print("\n2. Recording observations...")

    # Simulate observations (PES performs better)
    for i in range(20):
        # PES variant
        pes_outcome = await execute_workflow("pes", {"domain": "finance", "evaluation_cost": "high"})
        await framework.record_observation(
            experiment.experiment_id,
            "pes",
            outcome=pes_outcome["metrics"]["fitness"],
            is_success=pes_outcome["outcome_type"] == OutcomeType.SUCCESS.value
        )

        # QD variant
        qd_outcome = await execute_workflow("qd", {"domain": "finance", "need_diversity": False})
        await framework.record_observation(
            experiment.experiment_id,
            "qd",
            outcome=qd_outcome["metrics"]["fitness"],
            is_success=qd_outcome["outcome_type"] == OutcomeType.SUCCESS.value
        )

        if (i + 1) % 5 == 0:
            print(f"  Recorded {i + 1} observations per variant")

    print("\n3. Analyzing results...")

    # Get results
    results = await framework.get_results(experiment.experiment_id)

    print(f"\n  Statistical Analysis:")
    print(f"    Winner: {results.winner}")
    print(f"    Improvement: {results.improvement:.1%}")
    print(f"    Confidence: {results.confidence:.1%}")
    print(f"    Statistically significant: {results.significance}")
    print(f"    P-value: {results.p_value:.4f}")
    print(f"\n  Recommendation: {results.recommendation}")

    # Mark experiment complete
    await framework.complete_experiment(
        experiment.experiment_id,
        winner=results.winner,
        reason="Statistical significance reached"
    )


# ========================================================================
# CAUSAL MODELING DEMO
# ========================================================================

async def demo_causal_modeling():
    """Demonstrate causal modeling"""
    print("\n" + "="*70)
    print("CAUSAL MODELING DEMO")
    print("="*70)

    # Initialize builder
    builder = CausalModelBuilder(
        discovery_method="pc",
        min_confidence=0.5  # Lower for demo
    )

    print("\n1. Generating synthetic outcomes...")

    # Generate outcomes with known causal structure
    outcomes = []
    for exploration_rate in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for i in range(10):
            # Higher exploration -> higher diversity
            diversity = exploration_rate * 0.8 + np.random.randn() * 0.1

            # Higher diversity -> higher fitness
            fitness = diversity * 0.7 + np.random.randn() * 0.05

            outcome = {
                "context": {
                    "exploration_rate": exploration_rate,
                    "temperature": 1.0,
                    "population_size": 100
                },
                "metrics": {
                    "fitness": max(0.0, min(1.0, fitness)),
                    "diversity": max(0.0, min(1.0, diversity)),
                    "convergence_time": 120
                }
            }
            outcomes.append(outcome)

    print(f"  Generated {len(outcomes)} outcomes")

    print("\n2. Building causal model...")

    # Build model
    model = await builder.build_model(
        domain="demo",
        outcomes=outcomes
    )

    print(f"  Model ID: {model.model_id}")
    print(f"  Factors: {model.factors}")
    print(f"  Outcomes: {model.outcomes}")
    print(f"  Relationships: {len(model.relationships)}")

    print("\n3. Identified causal relationships:")
    for rel in model.relationships:
        print(f"  {rel.cause} -> {rel.effect}")
        print(f"    Strength: {rel.strength:.3f}")
        print(f"    Confidence: {rel.confidence:.1%}")
        if rel.mechanism:
            print(f"    Mechanism: {rel.mechanism}")

    print("\n4. Predicting intervention effects...")

    # Predict effect of changing exploration_rate
    prediction = await builder.predict_intervention(
        model=model,
        cause="exploration_rate",
        value=0.8
    )

    print(f"  Intervention: {prediction.intervention}")
    print(f"  Predicted effect: {prediction.predicted_effect:.3f}")
    print(f"  Confidence: {prediction.confidence:.1%}")

    if prediction.risk_assessment:
        print(f"  Risks:")
        for risk in prediction.risk_assessment:
            print(f"    - {risk}")

    print("\n5. Explaining outcomes...")

    # Explain fitness
    explanation = await builder.explain_outcome(model, "fitness")

    print(f"  Outcome: {explanation.outcome}")
    print(f"  Causes: {explanation.causes}")
    print(f"  Contribution: {explanation.contribution}")
    print(f"  Confidence: {explanation.confidence:.1%}")
    print(f"  Counterfactuals:")
    for cf in explanation.counterfactuals[:3]:
        print(f"    - {cf}")


# ========================================================================
# META-LEARNING DEMO
# ========================================================================

async def demo_meta_learning():
    """Demonstrate meta-learning across workflows"""
    print("\n" + "="*70)
    print("META-LEARNING DEMO")
    print("="*70)

    # Initialize meta-learner
    learner = MetaLearner(
        min_evidence=3,
        confidence_threshold=0.6
    )

    print("\n1. Generating workflow data...")

    # Generate workflow data
    workflows = []

    # Finance domain workflows (PES works well)
    for i in range(5):
        workflows.append({
            "workflow_id": f"finance_wf_{i}",
            "domain": "finance",
            "strategy": "pes",
            "outcome_type": "success",
            "fitness": 0.85 + np.random.random() * 0.1,
            "config": {
                "enable_planning": True,
                "max_iterations": 50
            },
            "context": {
                "evaluation_cost": "high",
                "exploration_rate": 0.3
            },
            "metrics": {
                "fitness": 0.87,
                "convergence_time": 120
            }
        })

    # Science domain workflows (QD works well)
    for i in range(5):
        workflows.append({
            "workflow_id": f"science_wf_{i}",
            "domain": "science",
            "strategy": "qd",
            "outcome_type": "success",
            "fitness": 0.80 + np.random.random() * 0.15,
            "config": {
                "feature_dimensions": ["complexity", "diversity"],
                "num_islands": 5
            },
            "context": {
                "need_diversity": True,
                "exploration_rate": 0.7
            },
            "metrics": {
                "fitness": 0.82,
                "diversity": 0.90
            }
        })

    print(f"  Generated {len(workflows)} workflows")

    print("\n2. Extracting meta-patterns...")

    # Extract patterns
    patterns = await learner.extract_patterns(workflows)

    print(f"  Extracted {len(patterns)} patterns\n")

    for i, pattern in enumerate(patterns[:5], 1):
        print(f"  Pattern {i}: {pattern.description}")
        print(f"    Confidence: {pattern.confidence:.1%}")
        print(f"    Expected benefit: {pattern.expected_benefit:.1%}")
        print(f"    Evidence: {len(pattern.evidence)} workflows")
        print()

    print("\n3. Recommending strategies for new problems...")

    # Recommend for finance problem
    print("\n  Finance problem:")
    recommendation = await learner.recommend_strategy({
        "problem_id": "finance_new",
        "domain": "finance",
        "num_variables": 75,
        "evaluation_cost": "high"
    })

    print(f"    Recommended: {recommendation.recommended_strategy}")
    print(f"    Confidence: {recommendation.confidence:.1%}")
    print(f"    Rationale: {recommendation.rationale}")
    print(f"    Expected performance: {recommendation.expected_performance:.2f}")

    # Recommend for science problem
    print("\n  Science problem:")
    recommendation = await learner.recommend_strategy({
        "problem_id": "science_new",
        "domain": "science",
        "num_variables": 50,
        "need_diversity": True
    })

    print(f"    Recommended: {recommendation.recommended_strategy}")
    print(f"    Confidence: {recommendation.confidence:.1%}")
    print(f"    Rationale: {recommendation.rationale}")

    print("\n4. Transferring knowledge...")

    # Transfer from finance to trading
    transferred = await learner.transfer_knowledge(
        source_domain="finance",
        target_domain="trading"
    )

    print(f"  Transferred {len(transferred)} patterns from finance to trading")

    # Recommend for trading problem
    print("\n  Trading problem (after transfer):")
    recommendation = await learner.recommend_strategy({
        "problem_id": "trading_new",
        "domain": "trading",
        "num_variables": 100
    })

    print(f"    Recommended: {recommendation.recommended_strategy}")
    print(f"    Transfer source: {recommendation.transfer_source}")


# ========================================================================
# INTEGRATED DEMO
# ========================================================================

async def demo_integration():
    """Demonstrate all components working together"""
    print("\n" + "="*70)
    print("INTEGRATED LONG-HORIZON LEARNING DEMO")
    print("="*70)

    # Initialize all components
    learner = OnlineLearner()
    framework = ABTestFramework(min_sample_size=10)
    causal_builder = CausalModelBuilder(min_confidence=0.5)
    meta_learner = MetaLearner(min_evidence=2)

    print("\n1. Running workflows and recording outcomes...")

    # Run workflows with different strategies
    workflows = []
    for i in range(15):
        strategy = "pes" if i % 2 == 0 else "qd"
        problem = {
            "domain": "finance",
            "evaluation_cost": "high",
            "exploration_rate": 0.3 + i * 0.05
        }

        outcome_data = await execute_workflow(strategy, problem)

        # Record for online learning
        outcome = LearningOutcome(
            workflow_id="integrated_workflow",
            strategy_used=strategy,
            outcome_type=OutcomeType(outcome_data["outcome_type"]),
            metrics=outcome_data["metrics"],
            context=outcome_data["context"]
        )
        await learner.record_outcome(outcome)

        # Store for causal modeling and meta-learning
        workflows.append(outcome_data)

    print(f"  Completed {len(workflows)} workflow executions")

    print("\n2. Online learning analysis...")

    best = await learner.get_best_strategy("integrated_workflow")
    print(f"  Best strategy: {best}")

    print("\n3. Building causal model...")

    causal_model = await causal_builder.build_model("finance", workflows)
    print(f"  Causal relationships: {len(causal_model.relationships)}")

    if causal_model.relationships:
        print(f"  Top relationship: {causal_model.relationships[0].cause} -> {causal_model.relationships[0].effect}")

    print("\n4. Extracting meta-patterns...")

    patterns = await meta_learner.extract_patterns(workflows)
    print(f"  Patterns extracted: {len(patterns)}")

    print("\n5. Complete analysis summary:")

    print(f"\n  Online Learning:")
    print(f"    Best strategy: {best}")
    stats = await learner.get_statistics()
    print(f"    Total outcomes: {stats['total_outcomes']}")

    print(f"\n  Causal Model:")
    print(f"    Factors: {len(causal_model.factors)}")
    print(f"    Relationships: {len(causal_model.relationships)}")

    print(f"\n  Meta-Learning:")
    print(f"    Patterns: {len(patterns)}")

    print("\n" + "="*70)
    print("Integrated demo complete!")
    print("="*70)


# ========================================================================
# MAIN
# ========================================================================

async def main():
    """Run all demos"""
    print("\n")
    print("#" * 70)
    print("# LONG-HORIZON LEARNING QUICKSTART")
    print("#" * 70)

    # Run individual demos
    await demo_online_learning()
    await demo_ab_testing()
    await demo_causal_modeling()
    await demo_meta_learning()

    # Run integrated demo
    await demo_integration()

    print("\n\nAll demos completed successfully!")
    print("\nNext steps:")
    print("  1. Integrate with your workflow execution system")
    print("  2. Configure exploration strategy for your domain")
    print("  3. Set up A/B tests for strategy validation")
    print("  4. Build causal models from historical data")
    print("  5. Extract meta-patterns across your workflows")


if __name__ == "__main__":
    asyncio.run(main())
