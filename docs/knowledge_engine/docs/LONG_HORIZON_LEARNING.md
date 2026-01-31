# Long-Horizon Learning for Agentic Workflows

## Overview

The Knowledge Engine now includes comprehensive long-horizon learning capabilities that enable continuous improvement and adaptation across extended workflow executions. This system learns from streaming outcomes, tests strategies systematically, builds causal models, and transfers knowledge across domains.

## Core Components

### 1. Online Learning Module

**Location**: `knowledge_engine/online_learning.py`

**Purpose**: Continuously learns from streaming workflow outcomes, tracking strategy performance over time and recommending adaptations.

**Key Features**:
- Streaming outcome recording (idempotent)
- Strategy performance tracking with moving averages
- Exploration vs exploitation strategies (ε-greedy, UCB, Thompson Sampling)
- Performance decay detection
- Adaptation recommendations

**Usage Example**:
```python
from knowledge_engine.online_learning import OnlineLearner
from knowledge_engine.schemas.long_horizon import LearningOutcome, OutcomeType

# Initialize learner
learner = OnlineLearner(
    exploration_strategy=ExplorationStrategy.EPSILON_GREEDY,
    initial_epsilon=0.3,
    performance_window=100
)

# Record outcomes as they happen
outcome = LearningOutcome(
    workflow_id="portfolio_optimization_001",
    strategy_used="pes",
    outcome_type=OutcomeType.SUCCESS,
    metrics={
        "fitness": 0.87,
        "convergence_time": 120,
        "llm_calls": 45
    },
    context={
        "domain": "finance",
        "exploration_rate": 0.3,
        "population_size": 100
    }
)
await learner.record_outcome(outcome)

# Get best strategy
best_strategy = await learner.get_best_strategy("portfolio_optimization_001")

# Check if we should explore
if await learner.should_explore():
    strategy = await learner.select_exploration_strategy("portfolio_optimization_001")

# Get adaptation recommendations
action = await learner.adapt_strategy("portfolio_optimization_001", current_performance=0.75)
if action:
    print(f"Recommended: {action.description}")
    print(f"Expected improvement: {action.expected_improvement:.1%}")
```

**Exploration Strategies**:

1. **ε-Greedy**: Explore with probability ε, exploit otherwise
   - Simple and effective
   - ε decays over time

2. **Upper Confidence Bound (UCB)**: Balance exploration and exploitation using confidence intervals
   - Optimistic strategy selection
   - Theoretically sound

3. **Thompson Sampling**: Bayesian approach using posterior distributions
   - Sample from Beta distribution
   - Naturally balances exploration

**Performance Decay Detection**:
- Tracks performance over sliding window
- Detects significant degradation
- Triggers adaptation when decay exceeds threshold

---

### 2. A/B Testing Framework

**Location**: `knowledge_engine/ab_testing.py`

**Purpose**: Statistically test different strategies, configurations, or approaches.

**Key Features**:
- Frequentist and Bayesian testing
- Early stopping based on sequential analysis
- Multiple comparison correction
- Statistical significance testing

**Usage Example**:
```python
from knowledge_engine.ab_testing import ABTestFramework

# Initialize framework
framework = ABTestFramework(
    significance_level=0.05,
    min_sample_size=100,
    test_method="frequentist",  # or "bayesian"
    enable_early_stopping=True
)

# Create experiment
experiment = await framework.create_experiment(
    name="PES vs QD for Trading",
    description="Compare PES and QD strategies for trading strategies",
    variants=["pes", "qd"],
    min_sample_size=50
)

# Record observations
await framework.record_observation(
    experiment_id=experiment.experiment_id,
    variant="pes",
    outcome=0.85,
    is_success=True
)

# Get results
results = await framework.get_results(experiment.experiment_id)

if results.significance:
    print(f"Winner: {results.winner}")
    print(f"Improvement: {results.improvement:.1%}")
    print(f"Confidence: {results.confidence:.1%}")
    print(f"Recommendation: {results.recommendation}")

# Select winner
winner = await framework.select_winner(experiment.experiment_id)
```

**Statistical Methods**:

1. **Frequentist**:
   - Two-sample t-test for continuous outcomes
   - Chi-squared test for binary outcomes
   - P-value based significance

2. **Bayesian**:
   - Beta-Bernoulli model for binary outcomes
   - Posterior probability of being best
   - More intuitive interpretation

**Early Stopping**:
- Sequential analysis
- Stops when overwhelming evidence
- Reduces required sample size

---

### 3. Causal Model Builder

**Location**: `knowledge_engine/causal_modeling.py`

**Purpose**: Build causal models from observational data to understand what affects what.

**Key Features**:
- Causal discovery from outcomes
- Intervention effect prediction
- Counterfactual reasoning
- Causal graph management

**Usage Example**:
```python
from knowledge_engine.causal_modeling import CausalModelBuilder

# Initialize builder
builder = CausalModelBuilder(
    discovery_method="pc",
    min_confidence=0.7
)

# Build model from outcomes
outcomes = [
    {
        "context": {
            "exploration_rate": 0.3,
            "temperature": 1.0,
            "population_size": 100
        },
        "metrics": {
            "fitness": 0.85,
            "diversity": 0.72,
            "convergence_time": 120
        }
    },
    # ... more outcomes
]

model = await builder.build_model(
    domain="finance",
    outcomes=outcomes
)

# Identify causes of fitness
causes = await builder.identify_causes(model, "fitness")
for cause in causes:
    print(f"{cause.cause} -> {cause.effect}")
    print(f"  Strength: {cause.strength:.2f}")
    print(f"  Mechanism: {cause.mechanism}")

# Predict intervention effect
prediction = await builder.predict_intervention(
    model=model,
    cause="exploration_rate",
    value=0.7
)
print(f"Predicted effect: {prediction.predicted_effect}")
print(f"Confidence: {prediction.confidence:.1%}")

# Explain an outcome
explanation = await builder.explain_outcome(model, "low_fitness")
print(f"Causes: {explanation.causes}")
print(f"Contribution: {explanation.contribution}")
print(f"Counterfactuals: {explanation.counterfactuals}")
```

**Causal Discovery**:
- Simplified PC algorithm
- Correlation-based with confidence intervals
- Handles both continuous and discrete variables

**Intervention Prediction**:
- "What if we change X?"
- Predicts effect magnitude
- Provides risk assessment

---

### 4. Meta-Learning System

**Location**: `knowledge_engine/meta_learning.py`

**Purpose**: Learn across workflow instances to extract transferable patterns.

**Key Features**:
- Pattern extraction across workflows
- Feature-based similarity
- Transfer learning across domains
- Strategy recommendation for new problems

**Usage Example**:
```python
from knowledge_engine.meta_learning import MetaLearner

# Initialize meta-learner
learner = MetaLearner(
    min_evidence=3,
    confidence_threshold=0.7
)

# Extract patterns from past workflows
workflows = [
    {
        "workflow_id": "wf_001",
        "domain": "finance",
        "strategy": "pes",
        "outcome_type": "success",
        "fitness": 0.87,
        "config": {"enable_planning": True},
        "context": {"num_variables": 50}
    },
    # ... more workflows
]

patterns = await learner.extract_patterns(workflows)

for pattern in patterns:
    print(f"Pattern: {pattern.description}")
    print(f"  Confidence: {pattern.confidence:.1%}")
    print(f"  Expected benefit: {pattern.expected_benefit:.1%}")
    print(f"  Evidence: {len(pattern.evidence)} workflows")

# Recommend strategy for new problem
recommendation = await learner.recommend_strategy({
    "problem_id": "new_portfolio_opt",
    "domain": "finance",
    "num_variables": 75,
    "evaluation_cost": "high"
})

print(f"Recommended: {recommendation.recommended_strategy}")
print(f"Confidence: {recommendation.confidence:.1%}")
print(f"Rationale: {recommendation.rationale}")
print(f"Expected performance: {recommendation.expected_performance:.2f}")

# Transfer knowledge to new domain
transferred = await learner.transfer_knowledge(
    source_domain="finance",
    target_domain="trading"
)
print(f"Transferred {len(transferred)} patterns")
```

**Pattern Types**:

1. **Strategy Patterns**: What strategies work for what problems
2. **Parameter Patterns**: Effective configuration patterns
3. **Feature Patterns**: Problem features that predict success

**Feature Extraction**:
- Domain, scale, complexity
- Evaluation cost
- Data availability
- Problem type

---

## Integration with Knowledge Engine

The long-horizon learning components integrate seamlessly with the existing Knowledge Engine:

```python
from knowledge_engine.integrations.unified_evolution_integration import (
    UnifiedEvolutionKnowledgeExtractor
)

# Initialize with knowledge engine
extractor = UnifiedEvolutionKnowledgeExtractor(knowledge_engine=ke)

# Record workflow outcome (online learning)
await extractor.record_workflow_outcome(
    workflow_id="portfolio_opt_001",
    strategy="pes",
    outcome={
        "success": True,
        "fitness": 0.87,
        "metrics": {"time": 120, "cost": 0.5}
    },
    timestamp=datetime.now(UTC)
)

# Get strategy performance
performance = await extractor.get_strategy_performance("portfolio_opt_001")

# Recommend adaptation
adaptation = await extractor.recommend_adaptation("portfolio_opt_001", 0.75)

# Build causal model
causal_model = await extractor.build_causal_model(
    domain="finance",
    outcomes=outcomes_list
)

# Predict intervention
effect = await extractor.predict_intervention_effect(
    domain="finance",
    cause="exploration_rate",
    value=0.7
)

# Extract meta-patterns
patterns = await extractor.extract_meta_patterns(workflows)

# Recommend strategy
recommendation = await extractor.recommend_strategy_for_problem({
    "domain": "trading",
    "num_variables": 100
})

# Create A/B experiment
experiment = await extractor.create_ab_experiment(
    name="Strategy Comparison",
    description="Compare PES vs QD",
    variants=["pes", "qd"]
)

# Record A/B observations
await extractor.record_ab_observation(
    experiment_id=experiment["experiment_id"],
    variant="pes",
    outcome=0.85,
    is_success=True
)

# Get A/B results
results = await extractor.get_ab_results(experiment["experiment_id"])
```

---

## Architectural Principles

### Law of Runtime Truth
All learning is based on actual execution outcomes, not assumptions or documentation.

### Law of Idempotency
Recording outcomes is safe to replay:
- Duplicate outcomes are detected and ignored
- UPSERT semantics for updates
- No side effects from repeated calls

### Law of UTC
All timestamps in UTC ISO-8601 format for consistency.

### Anti-Corruption Layer
Canonical schemas for all learning data prevent dependency leakage.

---

## Data Flow

```
Workflow Execution
       ↓
   Learning Outcome
       ↓
   Online Learner ─────→ Strategy Performance
       ↓                          ↓
   A/B Test Framework ←────── Adaptation Recommendation
       ↓
   Causal Model Builder ←──── Meta-Learning Patterns
       ↓                          ↓
   Knowledge Graph (Neo4j) ←── Vector DB (Qdrant)
       ↓
   Improved Strategy Selection
```

---

## Best Practices

### 1. Online Learning
- Start with ε-greedy for simplicity
- Set performance_window based on expected outcomes
- Monitor decay_rate for strategy degradation
- Use adaptation recommendations proactively

### 2. A/B Testing
- Use frequentist for simple comparisons
- Use Bayesian for complex multi-armed bandits
- Enable early stopping to reduce sample size
- Set min_sample_size based on effect size

### 3. Causal Modeling
- Collect diverse outcomes for robust models
- Focus on high-confidence relationships
- Use intervention predictions cautiously
- Validate causal findings with experiments

### 4. Meta-Learning
- Accumulate workflows across domains
- Use feature similarity for transfer
- Validate recommendations with A/B tests
- Update patterns regularly

---

## Testing

Run comprehensive tests:

```bash
# All long-horizon tests
pytest tests/test_long_horizon_learning.py -v

# Specific component
pytest tests/test_long_horizon_learning.py::TestOnlineLearning -v
pytest tests/test_long_horizon_learning.py::TestABTesting -v
pytest tests/test_long_horizon_learning.py::TestCausalModeling -v
pytest tests/test_long_horizon_learning.py::TestMetaLearning -v

# Integration tests
pytest tests/test_long_horizon_learning.py::TestLongHorizonIntegration -v
```

---

## API Reference

### OnlineLearner

**Methods**:
- `record_outcome(outcome: LearningOutcome)` - Record outcome (idempotent)
- `get_best_strategy(workflow_id: str)` - Get best performing strategy
- `should_explore() -> bool` - Exploration vs exploitation decision
- `select_exploration_strategy(workflow_id: str)` - Select strategy for exploration
- `adapt_strategy(workflow_id, current_performance)` - Recommend adaptation
- `get_strategy_performance(workflow_id, strategy_id)` - Get performance metrics
- `get_statistics()` - Overall learning statistics

### ABTestFramework

**Methods**:
- `create_experiment(name, description, variants)` - Create new experiment
- `record_observation(experiment_id, variant, outcome, is_success)` - Record observation (idempotent)
- `get_results(experiment_id)` - Get statistical analysis
- `select_winner(experiment_id)` - Get winning variant
- `complete_experiment(experiment_id, winner, reason)` - Mark complete
- `abandon_experiment(experiment_id, reason)` - Abandon experiment

### CausalModelBuilder

**Methods**:
- `build_model(domain, outcomes)` - Build causal model
- `update_model(model, new_data)` - Update with new data
- `identify_causes(model, outcome)` - Find causes of outcome
- `predict_intervention(model, cause, value)` - Predict effect
- `explain_outcome(model, outcome)` - Explain using causal model

### MetaLearner

**Methods**:
- `extract_patterns(workflows)` - Extract meta-patterns
- `recommend_strategy(problem)` - Recommend strategy for problem
- `transfer_knowledge(source_domain, target_domain)` - Transfer patterns
- `get_patterns(domain)` - Get patterns by domain

---

## Example: Complete Workflow

```python
import asyncio
from datetime import datetime, UTC

async def long_horizon_learning_demo():
    """Complete long-horizon learning workflow"""

    # 1. Initialize components
    from knowledge_engine.online_learning import OnlineLearner
    from knowledge_engine.ab_testing import ABTestFramework
    from knowledge_engine.causal_modeling import CausalModelBuilder
    from knowledge_engine.meta_learning import MetaLearner

    learner = OnlineLearner()
    framework = ABTestFramework()
    causal_builder = CausalModelBuilder()
    meta_learner = MetaLearner()

    # 2. Simulate workflow executions
    workflows = []
    for i in range(20):
        # Execute workflow with different strategies
        outcome = await execute_workflow(
            strategy="pes" if i % 2 == 0 else "qd",
            problem={"domain": "finance", "num_variables": 50}
        )

        # Record for online learning
        await learner.record_outcome(outcome)

        # Record for causal modeling
        workflows.append(outcome.to_dict())

    # 3. Get best strategy from online learning
    best = await learner.get_best_strategy("finance_workflow")
    print(f"Best strategy: {best}")

    # 4. Check for adaptation
    action = await learner.adapt_strategy("finance_workflow", 0.7)
    if action:
        print(f"Adaptation recommended: {action.description}")

    # 5. Build causal model
    causal_model = await causal_builder.build_model("finance", workflows)

    # 6. Identify what affects fitness
    causes = await causal_builder.identify_causes(causal_model, "fitness")
    print(f"Causal factors: {[c.cause for c in causes]}")

    # 7. Predict effect of changing exploration
    prediction = await causal_builder.predict_intervention(
        causal_model, "exploration_rate", 0.7
    )
    print(f"Predicted effect: {prediction.predicted_effect:.2f}")

    # 8. Extract meta-patterns
    patterns = await meta_learner.extract_patterns(workflows)
    print(f"Extracted {len(patterns)} patterns")

    # 9. Recommend strategy for new problem
    recommendation = await meta_learner.recommend_strategy({
        "domain": "trading",
        "num_variables": 75
    })
    print(f"Recommended: {recommendation.recommended_strategy}")
    print(f"Confidence: {recommendation.confidence:.1%}")

    # 10. Set up A/B test to validate
    experiment = await framework.create_experiment(
        name="Validate Recommendation",
        description=f"Test {recommendation.recommended_strategy} vs baseline",
        variants=[recommendation.recommended_strategy, "baseline"]
    )

    print(f"Created A/B test: {experiment.experiment_id}")

asyncio.run(long_horizon_learning_demo())
```

---

## Future Enhancements

1. **Deep Causal Discovery**: Integrate proper PC algorithm implementation
2. **Multi-Armed Bandits**: Contextual bandits for personalization
3. **Reinforcement Learning**: Deep RL for strategy selection
4. **Transfer Learning**: Domain adaptation techniques
5. **Causal Inference**: DoWhy integration for robust inference
6. **Meta-Reinforcement Learning**: Learn to learn faster

---

## References

- Online Learning: Cesa-Bianchi & Lugosi (2006)
- A/B Testing: Kohavi et al. (2009)
- Causal Discovery: Spirtes et al. (2000)
- Meta-Learning: Hospedales et al. (2020)
- Multi-Armed Bandits: Bubeck & Cesa-Bianchi (2012)

---

**Author**: Claude (Sonnet 4.5)
**Date**: January 30, 2026
**Version**: 1.0.0
