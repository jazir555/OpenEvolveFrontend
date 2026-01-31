# AI-Powered Strategy Recommender

**Version:** 1.0
**Date:** 2026-01-30
**Component:** Knowledge Engine Core

## Overview

The Strategy Recommender is an AI-powered system that automatically selects the optimal evolutionary strategy (OpenEvolve vs LoongFlow, and which mode) based on problem characteristics and historical performance data.

### What It Does

1. **Analyzes Problem Characteristics** - Examines domain, complexity, evaluation cost, constraints
2. **Queries Historical Performance** - Retrieves similar past runs from knowledge graph
3. **Ranks Strategies** - Scores each strategy (PES, QD, MO, Adversarial, Standard)
4. **Recommends Optimal Approach** - Selects best strategy with confidence score
5. **Explains Reasoning** - Provides detailed explanation of recommendation
6. **Learns from Results** - Improves recommendations over time

### Available Strategies

| System | Mode | Best For |
|--------|------|----------|
| **LoongFlow** | PES (Plan-Execute-Summarize) | Expensive evaluations, complex reasoning |
| **OpenEvolve** | QD (Quality-Diversity) | Diversity exploration, behavioral space |
| **OpenEvolve** | MO (Multi-Objective) | Multiple competing objectives |
| **OpenEvolve** | Adversarial | Robustness testing, security |
| **OpenEvolve** | Standard | Simple optimization, cheap evaluations |

---

## Installation

The Strategy Recommender is part of the Knowledge Engine core:

```python
from knowledge_engine.core.strategy_recommender import (
    StrategyRecommender,
    recommend_evolutionary_strategy
)
```

---

## Quick Start

### Basic Usage

```python
from knowledge_engine.core.strategy_recommender import recommend_evolutionary_strategy

# Get recommendation
recommendation = await recommend_evolutionary_strategy(
    problem_description="Optimize portfolio allocation for max Sharpe ratio",
    domain="finance",
    constraints={
        "objectives": ["return", "risk"],
        "time_limit_seconds": 300  # 5 min per backtest
    }
)

# View recommendation
print(f"System: {recommendation.recommended_system}")
print(f"Mode: {recommendation.recommended_mode}")
print(f"Confidence: {recommendation.confidence:.1%}")
print(f"Expected iterations: {recommendation.expected_performance.expected_iterations}")
```

### With Knowledge Engine Integration

```python
from knowledge_engine import KnowledgeEngine
from knowledge_engine.core.strategy_recommender import StrategyRecommender

# Initialize
ke = KnowledgeEngine()
recommender = StrategyRecommender(knowledge_engine=ke)

# Get recommendation
recommendation = await recommender.recommend_strategy(
    problem_description="Optimize experimental design",
    domain="science",
    constraints={
        "time_limit_seconds": 600,
        "objectives": ["yield", "cost"]
    }
)

# Print explanation
print(recommender.explain_recommendation(recommendation))
```

---

## API Reference

### StrategyRecommender

Main class for AI-powered strategy recommendations.

#### Constructor

```python
StrategyRecommender(
    knowledge_engine=None,
    llm_client=None,
    use_ai_analysis: bool = True,
    learning_enabled: bool = True
)
```

**Parameters:**
- `knowledge_engine`: Optional KnowledgeEngine instance for historical data
- `llm_client`: Optional LLM client for AI-powered analysis
- `use_ai_analysis`: Enable AI-powered problem analysis (default: True)
- `learning_enabled`: Enable learning from new runs (default: True)

#### Methods

##### recommend_strategy()

```python
async def recommend_strategy(
    problem_description: str,
    domain: str,
    constraints: Dict[str, Any]
) -> StrategyRecommendation
```

Generate optimal strategy recommendation.

**Parameters:**
- `problem_description`: Text description of the problem
- `domain`: Problem domain ("finance", "trading", "science", "engineering", "pharma", "web", "general")
- `constraints`: Additional constraints and requirements

**Returns:** `StrategyRecommendation` object

**Example:**
```python
recommendation = await recommender.recommend_strategy(
    problem_description="Develop trading strategy with robustness testing",
    domain="trading",
    constraints={
        "objectives": ["return", "sharpe_ratio"],
        "safety_critical": True,
        "time_limit_seconds": 300
    }
)
```

##### analyze_problem_characteristics()

```python
async def analyze_problem_characteristics(
    problem: str,
    domain: str,
    constraints: Dict[str, Any]
) -> ProblemCharacteristics
```

Analyze problem to extract key characteristics.

**Returns:** `ProblemCharacteristics` object with:
- `domain`: Problem domain
- `complexity`: "low", "medium", or "high"
- `evaluation_cost`: "cheap", "moderate", "expensive", or "very_expensive"
- `has_multiple_objectives`: Boolean
- `requires_diversity`: Boolean
- `requires_robustness`: Boolean
- `constraint_count`: Number of constraints
- `estimated_iterations`: Estimated required iterations
- `keywords`: Extracted keywords
- `domain_specific_factors`: Domain-specific features

##### query_historical_performance()

```python
async def query_historical_performance(
    domain: str,
    problem_type: str
) -> List[HistoricalRun]
```

Query historical performance data from knowledge engine.

**Returns:** List of `HistoricalRun` objects

##### rank_strategies()

```python
async def rank_strategies(
    problem_chars: ProblemCharacteristics,
    history: List[HistoricalRun]
) -> List[RankedStrategy]
```

Rank all strategies based on problem and history.

**Returns:** List of `RankedStrategy` objects sorted by score

##### explain_recommendation()

```python
def explain_recommendation(
    recommendation: StrategyRecommendation
) -> str
```

Generate human-readable explanation.

**Returns:** Formatted markdown text

##### learn_from_run()

```python
async def learn_from_run(run_result: Dict[str, Any]) -> None
```

Learn from completed evolutionary run to improve future recommendations.

**Parameters:**
```python
run_result = {
    "run_id": "unique_id",
    "domain": "finance",
    "strategy_used": "pes",
    "mode_used": "pes",
    "complexity": "high",
    "final_score": 0.85,
    "iterations": 30,
    "evaluations": 30,
    "diversity_score": 0.7,
    "evaluation_cost": "expensive",
    "predicted_score": 0.80  # Optional
}
```

##### get_recommendation_confidence()

```python
def get_recommendation_confidence(
    recommendation: StrategyRecommendation
) -> float
```

Get calibrated confidence score (0.0 to 1.0).

---

## Data Structures

### StrategyRecommendation

Complete strategy recommendation.

```python
@dataclass
class StrategyRecommendation:
    recommended_system: str  # "openevolve", "loongflow", "hybrid"
    recommended_mode: str  # "pes", "qd", "mo", "adversarial", "standard"
    config_overrides: Dict[str, Any]  # Recommended parameter adjustments
    confidence: float  # 0.0 to 1.0
    reasoning: Explanation
    alternatives: List[AlternativeStrategy]
    expected_performance: PerformancePrediction
    problem_analysis: ProblemCharacteristics
    historical_context: List[HistoricalRun]
    ranking: List[RankedStrategy]
```

### ProblemCharacteristics

Analyzed problem characteristics.

```python
@dataclass
class ProblemCharacteristics:
    domain: str
    complexity: str  # "low", "medium", "high"
    evaluation_cost: str  # "cheap", "moderate", "expensive", "very_expensive"
    has_multiple_objectives: bool
    requires_diversity: bool
    requires_robustness: bool
    constraint_count: int
    estimated_iterations: int
    similar_problems: List[str]
    keywords: List[str]
    domain_specific_factors: Dict[str, Any]
```

### HistoricalRun

Historical evolutionary run data.

```python
@dataclass
class HistoricalRun:
    run_id: str
    domain: str
    strategy_used: str
    mode_used: str
    problem_complexity: str
    final_score: float
    convergence_speed: int  # iterations to convergence
    evaluation_count: int
    diversity_score: float
    timestamp: datetime
    metadata: Dict[str, Any]
    evaluation_cost: str
    sample_efficiency: float
```

### RankedStrategy

Strategy with score and analysis.

```python
@dataclass
class RankedStrategy:
    system: str
    mode: str
    score: float  # 0.0 to 100.0
    expected_performance: Dict[str, float]
    pros: List[str]
    cons: List[str]
    confidence: float
```

### PerformancePrediction

Predicted performance metrics.

```python
@dataclass
class PerformancePrediction:
    expected_iterations: int
    expected_time_seconds: float
    expected_score: float
    confidence_interval: Tuple[float, float]
    success_probability: float
```

---

## Domain-Specific Examples

### Finance

```python
recommendation = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize portfolio allocation across 5 assets.
    Maximize return while minimizing risk (volatility).
    Ensure diversification and liquidity constraints.
    Use backtest on 3 years of daily data.
    """,
    domain="finance",
    constraints={
        "objectives": ["return", "risk", "liquidity"],
        "constraints": ["no_short_selling", "max_allocation_0.4"],
        "time_limit_seconds": 300  # 5 min backtest
    }
)

# Expected: LoongFlow PES (expensive evaluations)
assert recommendation.recommended_mode == "pes"
assert recommendation.expected_performance.expected_iterations < 50
```

### Science

```python
recommendation = await recommend_evolutionary_strategy(
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

# Expected: LoongFlow PES or OpenEvolve QD
assert recommendation.recommended_mode in ["pes", "qd"]
```

### Engineering

```python
recommendation = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize truss bridge design.
    Minimize weight while supporting 1000kg load.
    Requires FEA simulation for each design.
    Must satisfy safety constraints.
    """,
    domain="engineering",
    constraints={
        "time_limit_seconds": 600,
        "safety_critical": True
    }
)

# Expected: PES or Adversarial (for safety)
modes = [recommendation.recommended_mode] + [a.mode for a in recommendation.alternatives]
assert "adversarial" in modes or "pes" in modes
```

### Trading

```python
recommendation = await recommend_evolutionary_strategy(
    problem_description="""
    Develop algorithmic trading strategy.
    Test on historical price data across market regimes.
    Need robustness to adverse market conditions.
    """,
    domain="trading",
    constraints={
        "safety_critical": True,
        "time_limit_seconds": 300
    }
)

# Expected: Adversarial or QD (robustness + diversity)
assert recommendation.recommended_mode in ["adversarial", "qd", "pes"]
```

### Pharma

```python
recommendation = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize molecular structure for drug target.
    Maximize binding affinity, minimize toxicity.
    Each evaluation requires molecular docking simulation.
    """,
    domain="pharma",
    constraints={
        "objectives": ["affinity", "toxicity"],
        "time_limit_seconds": 1200  # 20 min docking
    }
)

# Expected: QD (chemical space exploration) or PES
assert recommendation.recommended_mode in ["qd", "pes"]
```

### Web Design

```python
recommendation = await recommend_evolutionary_strategy(
    problem_description="""
    Optimize landing page design.
    Test button colors, placement, and copy.
    Use Lighthouse for performance scoring.
    """,
    domain="web",
    constraints={
        "time_limit_seconds": 5  # Very fast evaluation
    }
)

# Expected: Standard or QD (cheap evaluations)
assert recommendation.recommended_mode in ["standard", "qd"]
```

---

## Recommendation Logic

### Scoring Factors

The recommender scores each strategy based on multiple factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Evaluation Cost** | 30% | PES favored for expensive evals |
| **Multiple Objectives** | 25% | MO favored for multiple objectives |
| **Diversity Need** | 20% | QD favored for diversity |
| **Robustness Need** | 15% | Adversarial favored for robustness |
| **Historical Performance** | 10% | Past success in similar problems |

### Rules-Based Decision Tree

```
1. IF evaluation_cost == "very_expensive" OR "expensive":
   → Recommend LoongFlow PES (60% fewer evaluations)

2. ELSE IF has_multiple_objectives:
   → Recommend OpenEvolve MO (Pareto optimization)

3. ELSE IF requires_diversity:
   → Recommend OpenEvolve QD (MAP-Elites exploration)

4. ELSE IF requires_robustness:
   → Recommend OpenEvolve Adversarial (robustness testing)

5. ELSE:
   → Recommend OpenEvolve Standard (simple, effective)
```

### AI-Powered Enhancement

When `use_ai_analysis=True`, the recommender:
1. Uses LLM to analyze problem description
2. Extracts semantic features beyond keywords
3. Identifies domain-specific patterns
4. Generates nuanced reasoning

---

## Learning and Adaptation

### Learning Loop

```python
# Run 1: Get recommendation
rec1 = await recommender.recommend_strategy(problem, domain, constraints)

# Execute evolutionary run
result = await run_evolution(rec1.recommended_mode, rec1.config_overrides)

# Learn from result
await recommender.learn_from_run({
    "run_id": "run_001",
    "domain": domain,
    "strategy_used": rec1.recommended_mode,
    "mode_used": rec1.recommended_mode,
    "final_score": result.final_score,
    "iterations": result.iterations,
    "evaluations": result.evaluations,
    "diversity_score": result.diversity_score
})

# Run 2: Recommendation now informed by Run 1
rec2 = await recommender.recommend_strategy(problem, domain, constraints)
```

### Accuracy Tracking

The recommender tracks prediction accuracy:

```python
# Include predicted score in run result
await recommender.learn_from_run({
    "predicted_score": 0.85,  # From recommendation
    "final_score": 0.88,  # Actual result
    # ...
})

# Check accuracy
accuracy = sum(recommender.recommendation_accuracy) / len(recommender.recommendation_accuracy)
print(f"Average accuracy: {accuracy:.1%}")
```

---

## Performance Tracking

### Metrics to Track

1. **Recommendation Accuracy** - How well predictions match actual results
2. **Strategy Success Rate** - How often recommended strategies succeed
3. **Sample Efficiency** - Evaluations saved by following recommendations
4. **Time Efficiency** - Time saved by following recommendations

### Monitoring

```python
# Get confidence-adjusted recommendation
rec = await recommender.recommend_strategy(problem, domain, constraints)
adjusted_confidence = recommender.get_recommendation_confidence(rec)

print(f"Base confidence: {rec.confidence:.1%}")
print(f"Adjusted confidence: {adjusted_confidence:.1%}")

# Check if enough historical data
if rec.historical_context:
    print(f"Historical runs: {len(rec.historical_context)}")
else:
    print("No historical data - using rules-based recommendation")
```

---

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/knowledge_engine/test_strategy_recommender.py -v

# Run specific test class
pytest tests/knowledge_engine/test_strategy_recommender.py::TestFinanceScenarios -v

# Run with coverage
pytest tests/knowledge_engine/test_strategy_recommender.py --cov=knowledge_engine.core.strategy_recommender
```

### Test Coverage

The test suite includes:
- ✅ Problem characteristic extraction (6 tests)
- ✅ Historical performance queries (3 tests)
- ✅ Strategy ranking and scoring (5 tests)
- ✅ Recommendation generation (5 tests)
- ✅ Learning from runs (2 tests)
- ✅ Confidence calibration (3 tests)
- ✅ Domain-specific scenarios (6 tests)

**Total: 30+ comprehensive tests**

---

## Configuration

### Domain Heuristics

Customize domain-specific behavior:

```python
recommender = StrategyRecommender()

# Access domain heuristics
finance_heuristics = recommender.domain_heuristics["finance"]

print(finance_heuristics)
# {
#     "preferred_modes": ["pes", "mo", "standard"],
#     "evaluation_cost": "expensive",
#     "requires_diversity": True,
#     "requires_robustness": True,
#     "typical_iterations": 50
# }
```

### Customizing Heuristics

```python
# Add custom domain
recommender.domain_heuristics["custom_domain"] = {
    "preferred_modes": ["pes", "qd"],
    "evaluation_cost": "moderate",
    "requires_diversity": True,
    "requires_robustness": False,
    "typical_iterations": 100
}
```

---

## Integration with Unified Evolutionary Engine

```python
from openevolve.unified import UnifiedEvolutionaryEngine
from knowledge_engine.core.strategy_recommender import StrategyRecommender

# Initialize with knowledge engine
ke = KnowledgeEngine()
engine = UnifiedEvolutionaryEngine(knowledge_engine=ke)

# Run optimization with automatic strategy selection
result = await engine.evolve(
    problem="Optimize portfolio allocation",
    domain="finance",
    max_evaluations=50,
    enable_planning=True,
    enable_memory=True
)

# The engine uses StrategyRecommender internally to select PES mode
print(f"Strategy used: {result['strategy_used']}")
print(f"Confidence: {result['strategy_confidence']}")
```

---

## Best Practices

### 1. Always Provide Domain

```python
# Good
rec = await recommend_evolutionary_strategy(
    problem_description="...",
    domain="finance"  # Specific domain
)

# Less accurate
rec = await recommend_evolutionary_strategy(
    problem_description="...",
    domain="general"  # Generic
)
```

### 2. Include Constraints

```python
rec = await recommend_evolutionary_strategy(
    problem_description="...",
    domain="science",
    constraints={
        "objectives": ["yield", "cost"],  # Multiple objectives
        "time_limit_seconds": 600,  # Evaluation cost hint
        "safety_critical": False  # Robustness need
    }
)
```

### 3. Learn from Every Run

```python
# After evolutionary run completes
await recommender.learn_from_run(run_result)
```

### 4. Check Confidence

```python
rec = await recommender.recommend_strategy(...)

if rec.confidence < 0.5:
    print("Low confidence - consider manual strategy selection")
    # Or run multiple strategies in parallel
```

### 5. Review Alternatives

```python
rec = await recommender.recommend_strategy(...)

# Check top alternatives
for alt in rec.alternatives[:3]:
    print(f"{alt.system} ({alt.mode}): {alt.reason}")
    print(f"  When to use: {alt.when_to_use}")
```

---

## Troubleshooting

### Low Confidence Recommendations

**Problem:** Confidence < 0.5

**Solutions:**
1. Run the evolutionary process and learn from results
2. Provide more detailed problem description
3. Ensure correct domain specification
4. Check that constraints are specified

### Unexpected Recommendations

**Problem:** Recommends different strategy than expected

**Solutions:**
1. Review the explanation: `recommender.explain_recommendation(rec)`
2. Check problem characteristics: `rec.problem_analysis`
3. Verify constraints are correct
4. Review historical context: `rec.historical_context`

### Poor Performance

**Problem:** Recommended strategy underperforms

**Solutions:**
1. Learn from the run: `await recommender.learn_from_run(result)`
2. Check alternatives: `rec.alternatives`
3. Consider running multiple strategies in parallel
4. Review config overrides: `rec.config_overrides`

---

## Future Enhancements

Planned improvements:

1. **Deep Learning Integration** - Train models on historical data
2. **Multi-Armed Bandit** - Exploration-exploitation balance
3. **Transfer Learning** - Learn across domains
4. **Real-time Adaptation** - Adjust strategy during evolution
5. **Explainable AI** - More detailed reasoning

---

## References

- **Integration Roadmap:** `COMPREHENSIVE_INTEGRATION_ROADMAP.md` (Phase 2, Task 2.3)
- **OpenEvolve Analysis:** `OPENEVOLVE_EVOLUTIONARY_ALGORITHM_FORENSIC_ANALYSIS.md`
- **LoongFlow PES Analysis:** `LOONGFLOW_PES_FORENSIC_ANALYSIS.md`
- **Unified Config:** `openevolve/unified/config.py`

---

## License

MIT License - See LICENSE file for details
