# API Reference

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Table of Contents

- [Core API](#core-api)
- [Strategy Selector](#strategy-selector)
- [Domain Optimizers](#domain-optimizers)
- [Knowledge Engine](#knowledge-engine)
- [Gauntlet System](#gauntlet-system)
- [Configuration](#configuration)
- [Data Models](#data-models)

---

## Core API

### `evolve()`

Main entry point for evolutionary optimization.

#### Signature

```python
async def evolve(
    problem: str,
    domain: str = "general",
    max_evaluations: int = 100,
    max_iterations: int = None,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    enable_planning: bool = True,
    enable_memory: bool = True,
    enable_gauntlet: bool = True,
    enable_knowledge_engine: bool = True,
    evolution_mode: str = "auto",
    config: UnifiedEvolutionConfig = None,
    evaluation_function: Callable = None,
    mutation_operator: Any = None,
    crossover_operator: Any = None,
    selection_operator: Any = None,
    initial_solution: Dict[str, Any] = None,
    initial_population: List[Dict[str, Any]] = None,
    callbacks: List[Callable] = None,
    timeout: int = None,
    random_seed: int = None,
    verbose: bool = False,
    **kwargs
) -> EvolutionResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | str | **Required** | Problem description in natural language |
| `domain` | str | `"general"` | Application domain (finance, trading, science, engineering, pharma, web_design) |
| `max_evaluations` | int | `100` | Maximum number of evaluations allowed |
| `max_iterations` | int | `None` | Maximum iterations (overrides auto-calculation) |
| `objectives` | List[str] | `None` | List of optimization objectives |
| `constraints` | Dict | `None` | Problem constraints |
| `enable_planning` | bool | `True` | Enable PES planning phase (LoongFlow) |
| `enable_memory` | bool | `True` | Enable memory retrieval (LoongFlow) |
| `enable_gauntlet` | bool | `True` | Enable 3-round gauntlet evaluation |
| `enable_knowledge_engine` | bool | `True` | Enable knowledge extraction and learning |
| `evolution_mode` | str | `"auto"` | Evolution mode (auto, pes, qd, mo, adversarial, standard) |
| `config` | UnifiedEvolutionConfig | `None` | Custom configuration object |
| `evaluation_function` | Callable | `None` | Custom evaluation function |
| `mutation_operator` | Any | `None` | Custom mutation operator |
| `crossover_operator` | Any | `None` | Custom crossover operator |
| `selection_operator` | Any | `None` | Custom selection operator |
| `initial_solution` | Dict | `None` | Initial solution for warm start |
| `initial_population` | List[Dict] | `None` | Initial population for warm start |
| `callbacks` | List[Callable] | `None` | Callback functions for events |
| `timeout` | int | `None` | Maximum execution time (seconds) |
| `random_seed` | int | `None` | Random seed for reproducibility |
| `verbose` | bool | `False` | Enable verbose logging |
| `**kwargs` | Any | | Additional parameters passed to config |

#### Returns

`EvolutionResult` object with fields:

| Field | Type | Description |
|-------|------|-------------|
| `best_solution` | Dict[str, Any] | Best solution found |
| `fitness` | float | Primary fitness score |
| `objectives` | Dict[str, float] | Objective values (if multi-objective) |
| `strategy_used` | str | Evolution mode selected (pes, qd, mo, adversarial, standard) |
| `strategy_confidence` | float | Confidence in strategy selection (0-1) |
| `strategy_reason` | str | Reason for strategy selection |
| `evaluations` | int | Number of evaluations performed |
| `iterations` | int | Number of iterations/generations |
| `improvement` | str | Improvement over baseline (e.g., "60% fewer evaluations") |
| `gauntlet_results` | Dict | Gauntlet evaluation results |
| `pareto_front` | List[Dict] | Pareto-optimal solutions (if MO mode) |
| `archive` | Dict | MAP-Elites archive (if QD mode) |
| `evolutionary_tree` | Dict | Evolutionary tree structure (if PES mode) |
| `execution_time` | float | Total execution time (seconds) |
| `metadata` | Dict | Additional metadata |

#### Raises

| Exception | When |
|-----------|-------|
| `ValueError` | Invalid parameters or domain |
| `ConfigurationError` | Configuration validation fails |
| `EvaluationError` | Evaluation function fails |
| `TimeoutError` | Execution exceeds timeout |
| `KnowledgeEngineError` | Knowledge engine operations fail |

#### Examples

**Basic Usage:**
```python
from openevolve.unified import evolve

result = await evolve(
    problem="Optimize portfolio allocation for max return with min risk",
    domain="finance",
    max_evaluations=50
)

print(f"Best solution: {result['best_solution']}")
print(f"Fitness: {result['fitness']}")
print(f"Strategy: {result['strategy_used']}")
```

**Multi-Objective:**
```python
result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    evolution_mode="mo",
    objectives=["return", "risk", "liquidity"],
    max_evaluations=100
)

# Get Pareto front
for solution in result['pareto_front']:
    print(f"Return: {solution['objectives']['return']}")
    print(f"Risk: {solution['objectives']['risk']}")
```

**Custom Evaluation:**
```python
def my_evaluation(solution, problem):
    # Custom evaluation logic
    score = my_backtester(solution)
    return score

result = await evolve(
    problem="...",
    domain="finance",
    evaluation_function=my_evaluation
)
```

---

### `quick_evolve()`

Simplified API for quick experiments with time budget.

#### Signature

```python
async def quick_evolve(
    problem: str,
    domain: str = "general",
    max_minutes: int = 5,
    **kwargs
) -> EvolutionResult
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | str | **Required** | Problem description |
| `domain` | str | `"general"` | Application domain |
| `max_minutes` | int | `5` | Maximum execution time (minutes) |
| `**kwargs` | Any | | Additional parameters passed to `evolve()` |

#### Example

```python
result = await quick_evolve(
    problem="Optimize landing page for conversions",
    domain="web_design",
    max_minutes=5
)
```

---

### `evolve_batch()`

Run multiple evolutions in parallel.

#### Signature

```python
async def evolve_batch(
    problems: List[str],
    domain: str = "general",
    max_evaluations: int = 100,
    max_parallel: int = 4,
    **kwargs
) -> List[EvolutionResult]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problems` | List[str] | **Required** | List of problems to solve |
| `domain` | str | `"general"` | Application domain |
| `max_evaluations` | int | `100` | Evaluations per problem |
| `max_parallel` | int | `4` | Maximum parallel executions |
| `**kwargs` | Any | | Additional parameters |

#### Example

```python
problems = [
    "Optimize tech portfolio",
    "Optimize healthcare portfolio",
    "Optimize energy portfolio"
]

results = await evolve_batch(
    problems=problems,
    domain="finance",
    max_evaluations=50
)

for i, result in enumerate(results):
    print(f"Problem {i}: {result['fitness']}")
```

---

### `evolve_no_gauntlet()`

Evolution without gauntlet evaluation (faster, less quality assurance).

#### Signature

```python
async def evolve_no_gauntlet(
    problem: str,
    domain: str = "general",
    **kwargs
) -> EvolutionResult
```

#### Example

```python
result = await evolve_no_gauntlet(
    problem="Quick optimization",
    domain="finance",
    max_evaluations=30
)
```

---

## Strategy Selector

### `EnsembleStrategySelector`

AI-powered strategy selector that recommends optimal evolutionary modes.

#### Methods

##### `recommend_with_confidence()`

```python
async def recommend_with_confidence(
    problem: str,
    domain: str,
    constraints: Dict[str, Any] = None,
    objectives: List[str] = None,
    evaluation_cost: str = None
) -> StrategyRecommendation
```

**Returns:** `StrategyRecommendation`
```python
{
    "mode": "pes",  # Recommended mode
    "confidence": 0.9,  # Confidence (0-1)
    "reason": "Expensive evaluations, PES reduces cost by 60%",
    "expected_improvement": "60% fewer evaluations",
    "config": UnifiedEvolutionConfig  # Recommended config
}
```

**Example:**
```python
from openevolve.unified import EnsembleStrategySelector

selector = EnsembleStrategySelector()

recommendation = await selector.recommend_with_confidence(
    problem="Optimize portfolio allocation",
    domain="finance",
    constraints={"max_evaluations": 50}
)

print(f"Recommended: {recommendation['mode']}")
print(f"Confidence: {recommendation['confidence']}")
print(f"Reason: {recommendation['reason']}")
```

##### `learn_from_run()`

```python
async def learn_from_run(
    problem: str,
    domain: str,
    strategy_used: str,
    result: EvolutionResult
) -> None
```

**Example:**
```python
result = await evolve(problem=problem, domain="finance")
await selector.learn_from_run(
    problem=problem,
    domain="finance",
    strategy_used=result['strategy_used'],
    result=result
)
```

---

## Domain Optimizers

### `FinanceOptimizer`

Domain-specific optimizer for finance problems.

#### Methods

##### `optimize()`

```python
async def optimize(
    problem: str,
    max_evaluations: int = 50,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    **kwargs
) -> EvolutionResult
```

**Example:**
```python
from openevolve.unified.domain_optimizers import FinanceOptimizer

optimizer = FinanceOptimizer()

result = await optimizer.optimize(
    problem="Optimize portfolio allocation",
    max_evaluations=50,
    objectives=["return", "risk"],
    constraints={"max_position_size": 0.1}
)
```

---

### `TradingOptimizer`

Domain-specific optimizer for trading strategies.

#### Methods

##### `optimize()`

```python
async def optimize(
    problem: str,
    max_evaluations: int = 100,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    strategy_type: str = None,
    **kwargs
) -> EvolutionResult
```

**Example:**
```python
from openevolve.unified.domain_optimizers import TradingOptimizer

optimizer = TradingOptimizer()

result = await optimizer.optimize(
    problem="Develop momentum strategy",
    max_evaluations=100,
    objectives=["sharpe_ratio", "max_drawdown"],
    strategy_type="momentum"
)
```

---

### `ScienceOptimizer`

Domain-specific optimizer for scientific experiments.

#### Methods

##### `optimize()`

```python
async def optimize(
    problem: str,
    max_evaluations: int = 30,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    experiment_cost: float = None,
    **kwargs
) -> EvolutionResult
```

**Example:**
```python
from openevolve.unified.domain_optimizers import ScienceOptimizer

optimizer = ScienceOptimizer()

result = await optimizer.optimize(
    problem="Optimize chemical reaction conditions",
    max_evaluations=30,
    objectives=["yield", "purity"],
    experiment_cost=5000
)
```

---

### `EngineeringOptimizer`

Domain-specific optimizer for engineering design.

#### Methods

##### `optimize()`

```python
async def optimize(
    problem: str,
    max_evaluations: int = 100,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    safety_critical: bool = True,
    **kwargs
) -> EvolutionResult
```

**Example:**
```python
from openevolve.unified.domain_optimizers import EngineeringOptimizer

optimizer = EngineeringOptimizer()

result = await optimizer.optimize(
    problem="Design lightweight bridge",
    max_evaluations=100,
    objectives=["weight", "strength", "cost"],
    safety_critical=True
)
```

---

### `PharmaOptimizer`

Domain-specific optimizer for pharmaceutical discovery.

#### Methods

##### `optimize()`

```python
async def optimize(
    problem: str,
    max_evaluations: int = 200,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    **kwargs
) -> EvolutionResult
```

**Example:**
```python
from openevolve.unified.domain_optimizers import PharmaOptimizer

optimizer = PharmaOptimizer()

result = await optimizer.optimize(
    problem="Optimize drug candidate",
    max_evaluations=200,
    objectives=["binding_affinity", "toxicity", "solubility"]
)
```

---

### `WebDesignOptimizer`

Domain-specific optimizer for web design optimization.

#### Methods

##### `optimize()`

```python
async def optimize(
    problem: str,
    max_evaluations: int = 500,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,
    **kwargs
) -> EvolutionResult
```

**Example:**
```python
from openevolve.unified.domain_optimizers import WebDesignOptimizer

optimizer = WebDesignOptimizer()

result = await optimizer.optimize(
    problem="Optimize landing page",
    max_evaluations=500,
    objectives=["conversion_rate", "bounce_rate", "time_on_page"]
)
```

---

## Knowledge Engine

### `extract_knowledge()`

Extract knowledge artifacts from evolutionary run.

#### Signature

```python
async def extract_knowledge(
    run_id: str,
    results: EvolutionResult,
    system: str,  # "openevolve" or "loongflow"
    problem: str = None,
    domain: str = None,
    metadata: Dict[str, Any] = None
) -> KnowledgeArtifacts
```

**Returns:** `KnowledgeArtifacts`
```python
{
    "run_id": "run_123",
    "system": "loongflow",
    "timestamp": "2026-01-30T12:00:00Z",
    "solution_patterns": [...],
    "performance_metrics": {...},
    "evolutionary_tree": {...},
    "gauntlet_feedback": {...}
}
```

**Example:**
```python
from openevolve.unified.knowledge import extract_knowledge

artifacts = await extract_knowledge(
    run_id="run_123",
    results=result,
    system="loongflow",
    problem="Optimize portfolio",
    domain="finance"
)
```

---

### `query_knowledge()`

Query knowledge engine for similar runs and patterns.

#### Signature

```python
async def query_knowledge(
    query: str,
    domain: str = None,
    problem_type: str = None,
    limit: int = 10,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]
```

**Returns:** List of similar runs with their metadata.

**Example:**
```python
from openevolve.unified.knowledge import query_knowledge

similar_runs = await query_knowledge(
    query="Portfolio optimization with ESG constraints",
    domain="finance",
    limit=10
)

for run in similar_runs:
    print(f"Run: {run['run_id']}")
    print(f"Fitness: {run['fitness']}")
    print(f"Strategy: {run['strategy_used']}")
```

---

### `fuse_memories()`

Combine memories from OpenEvolve and LoongFlow.

#### Signature

```python
async def fuse_memories(
    openevolve_memory: Dict[str, Any],
    loongflow_memory: Dict[str, Any],
    fusion_strategy: str = "weighted_average"
) -> Dict[str, Any]
```

**Example:**
```python
from openevolve.unified.knowledge import fuse_memories

fused = await fuse_memories(
    openevolve_memory=oe_results,
    loongflow_memory=lf_results,
    fusion_strategy="weighted_average"
)
```

---

### `recommend_strategy()`

Get strategy recommendation from knowledge engine.

#### Signature

```python
async def recommend_strategy(
    problem_type: str,
    domain: str,
    constraints: Dict[str, Any] = None
) -> StrategyRecommendation
```

**Example:**
```python
from openevolve.unified.knowledge import recommend_strategy

recommendation = await recommend_strategy(
    problem_type="portfolio_optimization",
    domain="finance",
    constraints={"max_evaluations": 50}
)

print(f"Recommended: {recommendation['mode']}")
```

---

## Gauntlet System

### `ThreeRoundGauntletOrchestrator`

Orchestrates the 3-round gauntlet evaluation.

#### Methods

##### `run_full_gauntlet()`

```python
async def run_full_gauntlet(
    solution: Dict[str, Any],
    problem: str,
    domain: str,
    round_configs: List[GauntletRoundRule] = None
) -> GauntletResult
```

**Returns:** `GauntletResult`
```python
{
    "passed": bool,
    "round_scores": {
        "loongflow": 0.75,
        "red_team": 0.80,
        "gold_team": 0.92
    },
    "final_score": 0.87,
    "failed_round": str or None,
    "feedback": {...}
}
```

**Example:**
```python
from openevolve.unified.gauntlet import ThreeRoundGauntletOrchestrator

orchestrator = ThreeRoundGauntletOrchestrator()

result = await orchestrator.run_full_gauntlet(
    solution=best_solution,
    problem="Optimize portfolio",
    domain="finance"
)

if result['passed']:
    print("Solution passed all gauntlet rounds!")
else:
    print(f"Failed at: {result['failed_round']}")
```

---

### `LoongFlowGauntletEvaluator`

Evaluator adapter for LoongFlow AI evaluation (Round 1).

#### Methods

##### `evaluate()`

```python
async def evaluate(
    solution: Dict[str, Any],
    problem: str,
    context: Dict[str, Any] = None
) -> float
```

**Returns:** Score between 0 and 1.

---

### `RedTeamEvaluator`

Evaluator for adversarial attack (Round 2).

#### Methods

##### `attack()`

```python
async def attack(
    solution: Dict[str, Any],
    problem: str,
    domain: str,
    num_rounds: int = 5
) -> RedTeamResult
```

**Returns:** `RedTeamResult` with attack results.

---

### `GoldTeamEvaluator`

Evaluator for consensus verification (Round 3).

#### Methods

##### `verify()`

```python
async def verify(
    solution: Dict[str, Any],
    problem: str,
    domain: str,
    judges: List[str] = None
) -> GoldTeamResult
```

**Returns:** `GoldTeamResult` with consensus vote.

---

## Configuration

### `UnifiedEvolutionConfig`

Unified configuration for all evolutionary modes.

#### Constructor

```python
UnifiedEvolutionConfig(
    # Evolution parameters
    evolution_mode: str = "auto",
    max_evaluations: int = 100,
    max_iterations: int = None,
    convergence_threshold: float = 0.001,

    # Domain & problem
    domain: str = "general",
    problem: str = None,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None,

    # PES parameters
    enable_planning: bool = True,
    enable_memory: bool = True,
    early_stopping: bool = True,
    early_stop_threshold: float = 0.9,

    # OpenEvolve parameters
    population_size: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    elite_size: int = 10,

    # QD parameters
    grid_resolution: int = 10,
    feature_dimensions: List[str] = None,
    archive_size: int = 1000,

    # MO parameters
    pareto_front_size: int = 100,

    # Adversarial parameters
    adversarial_rounds: int = 20,
    red_team_models: List[str] = None,

    # Knowledge engine
    enable_knowledge_engine: bool = True,
    extract_knowledge: bool = True,

    # Gauntlet
    enable_gauntlet: bool = True,
    gauntlet_rounds: List[str] = None,

    # Evaluation
    evaluation_timeout: int = 300,
    evaluation_function: Callable = None,

    # Parallelization
    max_workers: int = 4,
    num_islands: int = 1,

    # Logging
    verbose: bool = False,
    log_level: str = "INFO",

    # Random seed
    random_seed: int = None
)
```

#### Methods

##### `validate()`

Validate configuration parameters.

```python
config.validate() -> bool
```

##### `to_dict()`

Convert configuration to dictionary.

```python
config.to_dict() -> Dict[str, Any]
```

##### `from_dict()`

Create configuration from dictionary.

```python
UnifiedEvolutionConfig.from_dict(dict) -> UnifiedEvolutionConfig
```

---

### `validate_config()`

Validate a configuration object.

```python
def validate_config(config: UnifiedEvolutionConfig) -> bool
```

**Raises:** `ConfigurationError` if invalid.

---

### `get_recommended_config()`

Get recommended configuration for domain and problem.

```python
async def get_recommended_config(
    domain: str,
    problem: str = None,
    objectives: List[str] = None,
    constraints: Dict[str, Any] = None
) -> UnifiedEvolutionConfig
```

**Example:**
```python
from openevolve.unified.config import get_recommended_config

config = await get_recommended_config(
    domain="finance",
    problem="Optimize portfolio",
    objectives=["return", "risk"]
)
```

---

## Data Models

### `EvolutionResult`

Result of evolutionary optimization.

```python
class EvolutionResult(TypedDict):
    best_solution: Dict[str, Any]
    fitness: float
    objectives: Dict[str, float]
    strategy_used: str
    strategy_confidence: float
    strategy_reason: str
    evaluations: int
    iterations: int
    improvement: str
    gauntlet_results: Dict[str, Any]
    pareto_front: List[Dict[str, Any]]
    archive: Dict[str, Any]
    evolutionary_tree: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]
```

---

### `StrategyRecommendation`

Strategy recommendation from selector.

```python
class StrategyRecommendation(TypedDict):
    mode: str
    confidence: float
    reason: str
    expected_improvement: str
    config: UnifiedEvolutionConfig
```

---

### `KnowledgeArtifacts`

Knowledge artifacts extracted from run.

```python
class KnowledgeArtifacts(TypedDict):
    run_id: str
    system: str
    timestamp: str
    solution_patterns: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    evolutionary_tree: Dict[str, Any]
    gauntlet_feedback: Dict[str, Any]
    successful_strategies: List[Dict[str, Any]]
```

---

### `GauntletResult`

Result of gauntlet evaluation.

```python
class GauntletResult(TypedDict):
    passed: bool
    round_scores: Dict[str, float]
    final_score: float
    failed_round: Optional[str]
    feedback: Dict[str, Any]
```

---

### `GauntletRoundRule`

Configuration for gauntlet round.

```python
class GauntletRoundRule(TypedDict):
    rule_id: str
    rule_type: str  # "automated" or "manual"
    min_score: float
    max_attempts: int
    evaluator: str
    timeout: int
```

---

**End of API Reference**

For usage examples, see:
- [Unified Evolution Engine Guide](UNIFIED_EVOLUTION_ENGINE_GUIDE.md)
- [Domain Guides](domains/)
- [Code Examples](../examples/)
