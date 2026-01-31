# Ensemble Strategy Selection

**Version:** 2.0
**Date:** 2026-01-30
**Component:** Knowledge Engine Core

## Overview

The Ensemble Strategy Selector is an intelligent system that combines multiple prediction methods to recommend optimal evolutionary strategies (OpenEvolve vs LoongFlow, and which mode) with confidence intervals and real-time learning.

### What It Does

1. **Ensemble Prediction** - Combines 4 prediction methods with weighted voting
2. **Confidence Intervals** - Provides statistical confidence in recommendations
3. **Real-Time Learning** - Adapt weights based on recent performance
4. **Cold Start Handling** - Good recommendations even without historical data
5. **Transparent Reasoning** - Explains all decisions in detail

### Prediction Methods

| Method | Description | When Used |
|--------|-------------|-----------|
| **Rule-Based** | Deterministic rules based on problem characteristics | Always available |
| **Similarity-Based** | Find similar historical problems | Requires ≥5 historical runs |
| **Trend-Based** | Analyze recent performance trends | Requires ≥10 historical runs |
| **ML-Based** | Machine learning model (Random Forest) | Requires ≥50 historical runs |

---

## Architecture

```
Problem Description
    ↓
Analyze Problem Characteristics
    ↓
Query Historical Performance
    ↓
┌─────────────────────────────────────────┐
│  ENSEMBLE PREDICTION                    │
│  ├─ Rule-Based Prediction               │
│  ├─ Similarity-Based Prediction         │
│  ├─ Trend-Based Prediction              │
│  └─ ML-Based Prediction (optional)      │
└─────────────────────────────────────────┘
    ↓
Weighted Voting (with confidence weighting)
    ↓
Confidence Interval Calculation (bootstrap)
    ↓
Final Recommendation
    ↓
Record for Learning
```

---

## Installation

```python
from knowledge_engine.core.strategy_recommender import (
    EnsembleStrategySelector,
    recommend_evolutionary_strategy
)
```

---

## Quick Start

### Basic Usage

```python
from knowledge_engine.core.strategy_recommender import recommend_evolutionary_strategy

# Get ensemble recommendation
prediction = await recommend_evolutionary_strategy(
    problem_description="Optimize portfolio allocation for max Sharpe ratio",
    domain="finance",
    constraints={
        "objectives": ["return", "risk"],
        "time_limit_seconds": 300
    },
    use_ensemble=True  # Use ensemble methods
)

# View recommendation
system, mode = prediction.strategy
print(f"System: {system}")
print(f"Mode: {mode}")
print(f"Point Estimate: {prediction.point_estimate:.2%}")
print(f"95% CI: [{prediction.confidence_interval[0]:.2%}, {prediction.confidence_interval[1]:.2%}]")
print(f"Method Agreement: {(1.0 - prediction.disagreement_ratio):.1%}")
```

### With Selector Class

```python
from knowledge_engine import KnowledgeEngine
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

# Initialize
ke = KnowledgeEngine()
selector = EnsembleStrategySelector(
    knowledge_engine=ke,
    learning_enabled=True,
    enable_ml=True  # Optional: enable ML predictions
)

# Get recommendation
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize experimental design",
    domain="science",
    constraints={"time_limit_seconds": 600},
    confidence_level=0.95
)

# Print explanation
print(selector.explain_ensemble_recommendation(
    prediction,
    prediction.problem_analysis
))
```

---

## API Reference

### EnsembleStrategySelector

Main class for ensemble-based strategy recommendations.

#### Constructor

```python
EnsembleStrategySelector(
    knowledge_engine=None,
    llm_client=None,
    use_ai_analysis: bool = True,
    learning_enabled: bool = True,
    enable_ml: bool = False
)
```

**Parameters:**
- `knowledge_engine`: Optional KnowledgeEngine instance for historical data
- `llm_client`: Optional LLM client for AI-powered analysis
- `use_ai_analysis`: Enable AI-powered problem analysis (default: True)
- `learning_enabled`: Enable learning from new runs (default: True)
- `enable_ml`: Enable ML-based prediction (default: False, requires scikit-learn)

#### Methods

##### recommend_with_ensemble()

```python
async def recommend_with_ensemble(
    problem_description: str,
    domain: str,
    constraints: Dict[str, Any],
    confidence_level: float = 0.95
) -> EnsemblePrediction
```

Generate ensemble-based strategy recommendation.

**Parameters:**
- `problem_description`: Text description of the problem
- `domain`: Problem domain ("finance", "trading", "science", "engineering", "pharma", "web", "general")
- `constraints`: Additional constraints and requirements
- `confidence_level`: Confidence level for intervals (0.90, 0.95, 0.99)

**Returns:** `EnsemblePrediction` object

**Example:**
```python
prediction = await selector.recommend_with_ensemble(
    problem_description="Optimize bridge design for safety and weight",
    domain="engineering",
    constraints={
        "objectives": ["weight", "safety"],
        "safety_critical": True,
        "time_limit_seconds": 600
    },
    confidence_level=0.95
)
```

##### explain_ensemble_recommendation()

```python
def explain_ensemble_recommendation(
    prediction: EnsemblePrediction,
    problem_chars: ProblemCharacteristics
) -> str
```

Generate detailed explanation of ensemble recommendation.

**Returns:** Formatted markdown text

##### handle_cold_start()

```python
async def handle_cold_start(
    problem_chars: ProblemCharacteristics,
    domain: str
) -> EnsemblePrediction
```

Generate recommendation when no historical data is available.

##### get_learning_metrics()

```python
def get_learning_metrics() -> Dict[str, Any]
```

Get current learning and accuracy metrics.

**Returns:** Dictionary with:
- `average_accuracy`: Overall prediction accuracy
- `recent_accuracy`: Recent 10 recommendations accuracy
- `total_recommendations`: Total number of recommendations made
- `recent_trend`: 'improving', 'declining', or 'stable'
- `method_weights`: Current ensemble weights

---

## Data Structures

### EnsemblePrediction

Complete ensemble prediction result.

```python
@dataclass
class EnsemblePrediction:
    strategy: Tuple[str, str]  # (system, mode)
    point_estimate: float  # Expected performance
    confidence_interval: Tuple[float, float]  # (lower, upper)
    confidence_level: float  # 0.90, 0.95, etc.
    prediction_methods: List[str]  # Which methods agreed
    disagreement_ratio: float  # 0.0 = unanimous, 1.0 = split
    reasoning: str
    method_weights: Dict[str, float]  # Weight of each method
    individual_predictions: Dict[str, Tuple[str, str, float]]
```

### MethodPrediction

Individual prediction from a method.

```python
@dataclass
class MethodPrediction:
    method: str  # 'rule_based', 'similarity', 'trend', 'ml'
    system: str  # 'openevolve', 'loongflow'
    mode: str  # 'pes', 'qd', 'mo', 'adversarial', 'standard'
    confidence: float  # 0.0 to 1.0
    reasoning: str
    evidence: Dict[str, Any]
```

### OnlineLearningTracker

Tracks recommendation accuracy and adapts weights.

```python
class OnlineLearningTracker:
    def record_recommendation(
        self,
        recommendation: EnsemblePrediction,
        problem_chars: ProblemCharacteristics
    ) -> str

    def record_actual_performance(
        self,
        recommendation_id: str,
        actual_performance: float,
        run_id: str = None
    ) -> Dict[str, float]

    def get_accuracy_metrics() -> Dict[str, Any]
```

---

## Prediction Methods

### 1. Rule-Based Prediction

Deterministic rules based on problem characteristics.

**Decision Tree:**
```
IF evaluation_cost in ["expensive", "very_expensive"]:
    → Recommend PES (60% fewer evaluations)
ELIF has_multiple_objectives:
    → Recommend MO (Pareto optimization)
ELIF requires_diversity:
    → Recommend QD (MAP-Elites exploration)
ELIF requires_robustness:
    → Recommend Adversarial (robustness testing)
ELSE:
    → Recommend PES (best general performance)
```

**Confidence:** 0.75 - 0.90

**Example:**
```python
prediction = await selector._rule_based_prediction(problem_chars, "finance")
# Returns: MethodPrediction with system='loongflow', mode='pes'
```

### 2. Similarity-Based Prediction

Find similar historical problems and use their best strategies.

**Algorithm:**
1. Calculate similarity scores (keyword overlap, domain match, complexity match)
2. Get top-k most similar runs
3. Aggregate performance by strategy
4. Return best performing strategy

**Confidence:** 0.5 - 0.95 (based on similarity scores)

**Example:**
```python
prediction = await selector._similarity_based_prediction(problem_chars, history)
# Returns: MethodPrediction with evidence={'similar_runs': 10, 'avg_similarity': 0.65}
```

### 3. Trend-Based Prediction

Analyze recent performance trends for each strategy.

**Algorithm:**
1. Get last N runs for domain
2. Calculate moving averages for each strategy
3. Compute trend (recent_avg - old_avg)
4. Return strategy with improving trend

**Confidence:** 0.5 - 0.90 (based on trend strength and sample count)

**Example:**
```python
prediction = await selector._trend_based_prediction(problem_chars, history, "finance")
# Returns: MethodPrediction with evidence={'trend': 0.05, 'recent_avg': 0.82}
```

### 4. ML-Based Prediction (Optional)

Train Random Forest classifier on historical data.

**Features:**
- Evaluation cost (encoded)
- Complexity (encoded)
- Has multiple objectives (boolean)
- Requires diversity (boolean)
- Requires robustness (boolean)

**Model:** Random Forest (50 estimators, max_depth=5)

**Confidence:** Model's probability for predicted class

**Example:**
```python
selector = EnsembleStrategySelector(enable_ml=True)
prediction = await selector._ml_based_prediction(problem_chars, history)
# Returns: MethodPrediction with evidence={'training_samples': 150, 'feature_importance': {...}}
```

---

## Confidence Intervals

### Bootstrap Method

Confidence intervals are calculated using bootstrap resampling:

1. Sample historical performance data with replacement (1000 samples)
2. Calculate mean score for each sample
3. Determine percentiles for confidence level

**Example:**
```python
point_estimate, ci = await selector._calculate_confidence_interval(
    strategy=('loongflow', 'pes'),
    problem_chars=problem_chars,
    history=historical_runs,
    confidence_level=0.95
)

# Returns: (0.82, (0.78, 0.86))
# Interpretation: We expect 82% performance, with 95% confidence between 78-86%
```

### Confidence Levels

| Level | Use Case |
|-------|----------|
| **0.90** | Quick decisions, wider tolerance |
| **0.95** | Standard analysis (default) |
| **0.99** | Critical decisions, narrow tolerance |

---

## Weighted Voting

### Voting Mechanism

Each prediction method contributes a weighted vote:

```python
votes[strategy] += weight[method] * confidence[method]
```

**Initial Weights:**
```python
weights = {
    'rule_based': 0.25,
    'similarity': 0.35,
    'trend': 0.25,
    'ml': 0.15
}
```

### Agreement Calculation

Agreement is calculated using entropy:

```python
# Normalized votes
normalized = {k: v / total for k, v in votes.items()}

# Entropy
entropy = -sum(p * log(p) for p in normalized.values())

# Agreement (1 - normalized_entropy)
agreement = 1.0 - (entropy / max_entropy)
```

**Interpretation:**
- **Agreement > 0.8**: High consensus among methods
- **Agreement 0.5-0.8**: Moderate consensus
- **Agreement < 0.5**: Low consensus, recommendation uncertain

---

## Real-Time Learning

### Learning Loop

```python
# 1. Make recommendation
prediction = await selector.recommend_with_ensemble(...)
rec_id = selector.learning_tracker.recommendations_made[-1]['id']

# 2. Execute evolutionary run
result = await run_evolution(...)

# 3. Record actual performance
metrics = selector.learning_tracker.record_actual_performance(
    recommendation_id=rec_id,
    actual_performance=result.final_score,
    run_id=result.run_id
)

# 4. Weights automatically adapt after 20+ recommendations
if metrics['weights_adapted']:
    print(f"New weights: {metrics['new_weights']}")
```

### Weight Adaptation

Weights adapt based on recent accuracy:

1. Calculate recent accuracy for each method (last 20 recommendations)
2. Normalize accuracies to sum to 1.0
3. Smooth transition with learning rate α = 0.3

```python
new_weight = (1 - α) * old_weight + α * accuracy_weight
```

### Accuracy Metrics

Track prediction accuracy over time:

```python
metrics = selector.get_learning_metrics()
print(f"Average accuracy: {metrics['average_accuracy']:.1%}")
print(f"Recent accuracy: {metrics['recent_accuracy']:.1%}")
print(f"Trend: {metrics['recent_trend']}")
print(f"Weights: {metrics['method_weights']}")
```

---

## Cold Start Handling

When no historical data is available:

```python
prediction = await selector.handle_cold_start(problem_chars, domain)
```

**Strategy:**
1. Use rule-based prediction
2. Lower confidence by 20%
3. Use domain-specific defaults
4. Add cold start explanation

**Example:**
```python
# Cold start for finance domain
prediction = await selector.handle_cold_start(problem_chars, "finance")
# Returns: EnsemblePrediction with confidence_level=0.80 (lowered from 1.0)
# Reasoning includes "[Cold start: using rule-based defaults]"
```

---

## Explanation System

### Generate Explanation

```python
explanation = selector.explain_ensemble_recommendation(
    prediction,
    problem_chars
)
```

**Output includes:**
- Selected strategy with confidence
- Expected performance with confidence interval
- Method agreement metrics
- Individual method predictions
- Problem analysis
- Learning metrics

**Example Output:**
```markdown
# Ensemble Strategy Recommendation

## Selected Strategy
**System:** LOONGFLOW
**Mode:** PES

## Expected Performance
**Point Estimate:** 82.00%
**95% Confidence Interval:** [78.00%, 86.00%]

## Method Agreement
**Agreement Level:** 85.0%
**Disagreement Ratio:** 15.0%

### Prediction Methods Used:
- **rule_based**: 30.0% weight
- **similarity**: 35.0% weight
- **trend**: 35.0% weight

### Individual Predictions:
- **rule_based**: loongflow/pes (confidence: 85.0%)
- **similarity**: loongflow/pes (confidence: 80.0%)
- **trend**: loongflow/pes (confidence: 78.0%)

## Detailed Reasoning
## Ensemble Strategy Selection

**Selected Strategy:** LOONGFLOW / PES
**Method Agreement:** 85.0%

### Individual Method Predictions:
- **rule_based**: loongflow/pes (confidence: 85.0%)
  - Reasoning: Expensive evaluations favor PES (60% fewer evaluations)
- **similarity**: loongflow/pes (confidence: 80.0%)
  - Reasoning: Found 10 similar runs (avg similarity: 0.65). Best strategy: pes with avg score 0.82
- **trend**: loongflow/pes (confidence: 78.0%)
  - Reasoning: Analyzing 15 recent runs. PES shows improving trend (+0.050, avg: 0.82)

### Ensemble Decision:
The weighted vote selected LOONGFLOW/PES based on 3 prediction methods.
High agreement among methods indicates strong consensus.

## Problem Analysis
- **Domain:** finance
- **Complexity:** high
- **Evaluation Cost:** expensive
- **Multiple Objectives:** True
- **Requires Diversity:** True
- **Requires Robustness:** True

## Learning Metrics
- **Average Accuracy:** 78.5%
- **Total Recommendations:** 25
- **Trend:** improving
- **Current Method Weights:**
  - rule_based: 28.0%
  - similarity: 38.0%
  - trend: 34.0%
```

---

## Domain-Specific Examples

### Finance

```python
prediction = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize portfolio allocation across 5 assets.
    Maximize return while minimizing risk (volatility).
    Ensure diversification and liquidity constraints.
    Use backtest on 3 years of daily data.
    """,
    domain="finance",
    constraints={
        "objectives": ["return", "risk"],
        "time_limit_seconds": 300  # 5 min backtest
    }
)

# Expected: LoongFlow PES (expensive evaluations)
assert prediction.strategy[1] == "pes"
assert prediction.point_estimate > 0.75
```

### Science

```python
prediction = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize chemical reaction conditions.
    Variables: temperature, pressure, catalyst amount.
    Each experiment requires molecular dynamics simulation.
    Goal: maximize yield while minimizing cost.
    """,
    domain="science",
    constraints={
        "objectives": ["yield", "cost"],
        "time_limit_seconds": 900  # 15 min simulation
    }
)

# Expected: PES or QD (expensive evaluations + diversity)
assert prediction.strategy[1] in ["pes", "qd"]
```

### Engineering

```python
prediction = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize truss bridge design.
    Minimize weight while supporting 1000kg load.
    Requires FEA simulation for each design.
    Must satisfy safety constraints.
    """,
    domain="engineering",
    constraints={
        "safety_critical": True,
        "time_limit_seconds": 600
    }
)

# Expected: PES or Adversarial (safety-critical)
assert prediction.strategy[1] in ["pes", "adversarial"]
```

---

## Performance Tracking

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Prediction Accuracy** | How well predictions match actual results | > 75% |
| **Confidence Calibration** | CI coverage (should match confidence level) | 95% |
| **Method Agreement** | Consensus among prediction methods | > 60% |
| **Weight Stability** | How much weights change over time | Low variance |

### Monitoring

```python
# Get current metrics
metrics = selector.get_learning_metrics()

# Check accuracy
if metrics['average_accuracy'] < 0.7:
    print("Warning: Low accuracy, consider gathering more data")

# Check trend
if metrics['recent_trend'] == 'declining':
    print("Warning: Accuracy declining, investigating...")

# Check weights
for method, weight in metrics['method_weights'].items():
    print(f"{method}: {weight:.1%}")
```

---

## Best Practices

### 1. Always Provide Domain

```python
# Good
prediction = await recommend_evolutionary_strategy(
    problem_description="...",
    domain="finance"  # Specific domain
)

# Less accurate
prediction = await recommend_evolutionary_strategy(
    problem_description="...",
    domain="general"  # Generic
)
```

### 2. Include Constraints

```python
prediction = await recommend_evolutionary_strategy(
    problem_description="...",
    domain="science",
    constraints={
        "objectives": ["yield", "cost"],  # Multiple objectives
        "time_limit_seconds": 600,  # Evaluation cost hint
        "safety_critical": False  # Robustness need
    }
)
```

### 3. Track Learning

```python
# After each evolutionary run
await selector.learn_from_run({
    "run_id": "run_001",
    "domain": domain,
    "strategy_used": prediction.strategy[1],
    "mode_used": prediction.strategy[1],
    "final_score": actual_score,
    "iterations": actual_iterations,
    "evaluations": actual_evaluations,
    "recommendation_id": rec_id  # Link to prediction
})
```

### 4. Check Confidence

```python
prediction = await selector.recommend_with_ensemble(...)

agreement = 1.0 - prediction.disagreement_ratio
if agreement < 0.5:
    print("Low agreement - consider manual strategy selection")
    # Or run multiple strategies in parallel
```

### 5. Review Individual Methods

```python
for method, (system, mode, conf) in prediction.individual_predictions.items():
    print(f"{method}: {system}/{mode} (confidence: {conf:.1%})")
```

---

## Troubleshooting

### Low Accuracy

**Problem:** Average accuracy < 0.7

**Solutions:**
1. Gather more historical data
2. Check problem characteristics are accurate
3. Verify domain specification
4. Review constraint definitions
5. Consider enabling ML predictions

### Low Method Agreement

**Problem:** Disagreement ratio > 0.5

**Solutions:**
1. Review individual method predictions
2. Check if problem has mixed characteristics
3. Consider running multiple strategies
4. Gather more domain-specific data

### Weights Not Adapting

**Problem:** Weights remain static

**Solutions:**
1. Ensure learning_enabled=True
2. Record actual performance for each recommendation
3. Need at least 20 recommendations for adaptation
4. Check weight adaptation logs

### Poor Cold Start Recommendations

**Problem:** Cold start recommendations are inaccurate

**Solutions:**
1. Provide more detailed problem description
2. Ensure correct domain specification
3. Include all relevant constraints
4. After first run, learning will improve

---

## Advanced Usage

### Custom Method Weights

```python
selector = EnsembleStrategySelector()

# Override default weights
selector.method_weights = {
    'rule_based': 0.40,  # More weight to rules
    'similarity': 0.30,
    'trend': 0.20,
    'ml': 0.10
}
```

### Minimum Sample Thresholds

```python
selector = EnsembleStrategySelector()

# Adjust minimum samples for each method
selector.min_samples_for_similarity = 10  # Default: 5
selector.min_samples_for_trend = 15  # Default: 10
selector.min_samples_for_ml = 100  # Default: 50
```

### Disable Learning

```python
selector = EnsembleStrategySelector(learning_enabled=False)
# Weights will not adapt
```

---

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/knowledge_engine/test_ensemble_strategy_selector.py -v

# Run specific test class
pytest tests/knowledge_engine/test_ensemble_strategy_selector.py::TestEnsemblePrediction -v

# Run with coverage
pytest tests/knowledge_engine/test_ensemble_strategy_selector.py --cov=knowledge_engine.core.strategy_recommender
```

### Test Coverage

The test suite includes:
- ✅ Ensemble prediction (3 tests)
- ✅ Individual prediction methods (6 tests)
- ✅ Confidence intervals (3 tests)
- ✅ Weighted voting (3 tests)
- ✅ Online learning (4 tests)
- ✅ Cold start (2 tests)
- ✅ Explanation generation (2 tests)
- ✅ Integration workflows (3 tests)

**Total: 26+ comprehensive tests**

---

## Migration from Basic Recommender

### Old API (Basic Recommender)

```python
from knowledge_engine.core.strategy_recommender import StrategyRecommender

recommender = StrategyRecommender(knowledge_engine=ke)
recommendation = await recommender.recommend_strategy(problem, domain, constraints)

print(f"System: {recommendation.recommended_system}")
print(f"Mode: {recommendation.recommended_mode}")
print(f"Confidence: {recommendation.confidence:.1%}")
```

### New API (Ensemble Selector)

```python
from knowledge_engine.core.strategy_recommender import EnsembleStrategySelector

selector = EnsembleStrategySelector(knowledge_engine=ke)
prediction = await selector.recommend_with_ensemble(problem, domain, constraints)

system, mode = prediction.strategy
print(f"System: {system}")
print(f"Mode: {mode}")
print(f"Point Estimate: {prediction.point_estimate:.2%}")
print(f"95% CI: {prediction.confidence_interval}")
print(f"Method Agreement: {(1.0 - prediction.disagreement_ratio):.1%}")
```

### Backward Compatibility

```python
# Use convenience function with ensemble flag
from knowledge_engine.core.strategy_recommender import recommend_evolutionary_strategy

# Old behavior (basic)
rec = await recommend_evolutionary_strategy(
    problem, domain, constraints, use_ensemble=False
)

# New behavior (ensemble)
pred = await recommend_evolutionary_strategy(
    problem, domain, constraints, use_ensemble=True
)
```

---

## References

- **Integration Roadmap:** `COMPREHENSIVE_INTEGRATION_ROADMAP.md` (Phase 4, Task 4.1)
- **Basic Recommender:** `STRATEGY_RECOMMENDER.md`
- **OpenEvolve Analysis:** `OPENEVOLVE_EVOLUTIONARY_ALGORITHM_FORENSIC_ANALYSIS.md`
- **LoongFlow PES Analysis:** `LOONGFLOW_PES_FORENSIC_ANALYSIS.md`

---

## License

MIT License - See LICENSE file for details
