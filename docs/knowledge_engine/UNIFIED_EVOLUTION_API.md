# Unified Evolution API Documentation

## Overview

The **Unified Evolution API** provides a single entry point for all evolutionary optimization, automatically selecting and executing the optimal strategy for your problem.

**Before** (complex):
```python
# User had to know which system to use, which mode, how to configure...
if use_loongflow:
    result = loongflow.evolve(config=loongflow_config)
elif use_openevolve:
    if mode == "qd":
        result = openevolve.run_qd(config=qd_config)
    elif mode == "mo":
        result = openevolve.run_mo(config=mo_config)
    # ... lots of manual work
```

**After** (simple):
```python
# Just one function call, everything automatic
result = await evolve(
    problem="Optimize portfolio allocation",
    domain="finance"
)
# That's it!
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core API](#core-api)
3. [Convenience Functions](#convenience-functions)
4. [Domains](#domains)
5. [Strategy Selection](#strategy-selection)
6. [Configuration](#configuration)
7. [Progress Callbacks](#progress-callbacks)
8. [Knowledge Extraction](#knowledge-extraction)
9. [Gauntlet Evaluation](#gauntlet-evaluation)
10. [Result Handling](#result-handling)
11. [Advanced Usage](#advanced-usage)
12. [Examples](#examples)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install openevolve

# Optional: Install LoongFlow for PES mode
pip install loongflow

# Optional: Install Knowledge Engine dependencies
pip install neo4j qdrant-client graphiti
```

### Basic Usage

```python
from openevolve.unified import evolve

# Simple evolution
result = await evolve(
    problem="Maximize portfolio Sharpe ratio",
    domain="finance"
)

print(f"Best solution: {result.best_solution}")
print(f"Score: {result.final_score}")
print(f"Strategy used: {result.strategy_used.mode}")
```

---

## Core API

### `evolve()`

Main entry point for evolutionary optimization.

**Signature:**
```python
async def evolve(
    problem: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None,
    config: Optional[UnifiedEvolutionConfig] = None,
    run_gauntlet: bool = True,
    store_knowledge: bool = True,
    callback: Optional[Callable[[ProgressUpdate], None]] = None,
    knowledge_engine=None
) -> EvolutionResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | `str` | **required** | Problem description (natural language or code) |
| `domain` | `str` | `"general"` | Domain: finance, trading, science, engineering, pharma, web, general |
| `constraints` | `Dict` | `None` | Optional constraints (objectives, limits, time, etc.) |
| `config` | `UnifiedEvolutionConfig` | `None` | Optional configuration (auto-generated if not provided) |
| `run_gauntlet` | `bool` | `True` | Run 3-round gauntlet evaluation |
| `store_knowledge` | `bool` | `True` | Store results in Knowledge Engine |
| `callback` | `Callable` | `None` | Progress callback function |
| `knowledge_engine` | `KnowledgeEngine` | `None` | Optional knowledge engine instance |

**Returns:**
- `EvolutionResult` object with solution, score, metadata, and artifacts

**Example:**
```python
result = await evolve(
    problem="Optimize chemical reaction yield",
    domain="science",
    constraints={
        'objectives': ['yield', 'purity'],
        'time_limit_seconds': 300
    }
)
```

---

## Convenience Functions

### `quick_evolve()`

Fastest path to solution, returns just the solution string.

**Signature:**
```python
async def quick_evolve(problem: str, domain: str = "general") -> str
```

**Example:**
```python
solution = await quick_evolve(
    problem="Sort array efficiently",
    domain="general"
)
print(solution)
```

### `evolve_no_gauntlet()`

Evolution without gauntlet (faster, less quality assurance).

**Signature:**
```python
async def evolve_no_gauntlet(
    problem: str,
    domain: str = "general",
    constraints: Optional[Dict[str, Any]] = None
) -> EvolutionResult
```

**Example:**
```python
result = await evolve_no_gauntlet(
    problem="Quick optimization",
    domain="web"
)
```

### `evolve_batch()`

Evolve multiple problems in parallel.

**Signature:**
```python
async def evolve_batch(
    problems: List[str],
    domain: str = "general",
    max_concurrent: int = 3,
    constraints: Optional[Dict[str, Any]] = None
) -> List[EvolutionResult]
```

**Example:**
```python
problems = [
    "Optimize loading speed",
    "Improve accessibility",
    "Enhance SEO"
]

results = await evolve_batch(
    problems=problems,
    domain="web",
    max_concurrent=2
)

for i, result in enumerate(results):
    print(f"Problem {i}: {result.final_score}")
```

---

## Domains

The API supports 7 domains, each with optimized strategies:

### Finance
- **Best for:** Portfolio optimization, risk management, trading strategies
- **Evaluation cost:** Expensive (backtests)
- **Recommended mode:** PES or MO
- **Example:**
```python
result = await evolve(
    problem="Maximize portfolio return with minimum risk",
    domain="finance"
)
```

### Trading
- **Best for:** Trading signals, market making, arbitrage
- **Evaluation cost:** Expensive (backtests)
- **Recommended mode:** QD or PES
- **Example:**
```python
result = await evolve(
    problem="Develop mean-reversion trading strategy",
    domain="trading"
)
```

### Science
- **Best for:** Experimental design, simulation optimization
- **Evaluation cost:** Very expensive (experiments/simulations)
- **Recommended mode:** PES (60% fewer experiments)
- **Example:**
```python
result = await evolve(
    problem="Optimize catalyst composition for maximum yield",
    domain="science"
)
```

### Engineering
- **Best for:** Design optimization, structural analysis
- **Evaluation cost:** Expensive (FEA/CFD simulations)
- **Recommended mode:** PES or Adversarial
- **Example:**
```python
result = await evolve(
    problem="Minimize weight while maintaining structural integrity",
    domain="engineering"
)
```

### Pharma
- **Best for:** Drug discovery, dosage optimization
- **Evaluation cost:** Very expensive (docking/simulation)
- **Recommended mode:** QD or MO
- **Example:**
```python
result = await evolve(
    problem="Optimize drug binding affinity",
    domain="pharma"
)
```

### Web
- **Best for:** Performance, UX, A/B testing
- **Evaluation cost:** Cheap (Lighthouse tests)
- **Recommended mode:** Standard or QD
- **Example:**
```python
result = await evolve(
    problem="Maximize Core Web Vitals score",
    domain="web"
)
```

### General
- **Best for:** Generic optimization problems
- **Evaluation cost:** Moderate
- **Recommended mode:** PES or Standard
- **Example:**
```python
result = await evolve(
    problem="Traveling salesman problem",
    domain="general"
)
```

---

## Strategy Selection

The API automatically selects the optimal strategy based on:

### Selection Factors

1. **Evaluation Cost** (30 points)
   - Expensive evaluations → PES (60% fewer)
   - Cheap evaluations → Any mode

2. **Multiple Objectives** (25 points)
   - Multiple competing objectives → MO mode
   - Single objective → Any mode

3. **Diversity Need** (20 points)
   - Need diverse solutions → QD mode
   - Single best solution → Any mode

4. **Robustness Need** (15 points)
   - Safety-critical → Adversarial mode
   - Non-critical → Any mode

5. **Historical Performance** (10 points)
   - Past success → Weight toward winning mode

### Example Selection Logic

```python
# Finance domain + expensive evaluation + single objective
→ LoongFlow PES (60% fewer backtests)

# Trading + multiple objectives (return/risk/liquidity)
→ OpenEvolve MO (Pareto optimization)

# Science + need diverse experimental designs
→ OpenEvolve QD (MAP-Elites archive)

# Engineering + safety-critical
→ OpenEvolve Adversarial (robustness testing)
```

---

## Configuration

### Auto-Generated Configuration

By default, the API generates optimal configuration automatically:

```python
# Just specify problem, config is auto-generated
result = await evolve(
    problem="Optimize portfolio",
    domain="finance"
)
```

### Custom Configuration

Provide custom configuration for fine-grained control:

```python
from openevolve.unified.config import UnifiedEvolutionConfig, EvolutionMode

custom_config = UnifiedEvolutionConfig(
    max_iterations=50,
    evolution_mode=EvolutionMode.PES,
    pes=evolution_mode=PESConfig(
        enabled=True,
        enable_planning=True,
        enable_memory=True
    )
)

result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    config=custom_config
)
```

### Configuration Parameters

**Core Parameters:**
- `max_iterations`: Maximum iterations (default: 10000)
- `time_limit_seconds`: Maximum execution time
- `target_fitness`: Stop when fitness reached
- `random_seed`: For reproducibility

**PES Parameters:**
- `enable_planning`: Enable planning phase
- `enable_memory`: Use evolutionary memory
- `max_rounds`: Maximum PES rounds

**QD Parameters:**
- `grid_resolution`: MAP-Elites grid size
- `archive_size`: Elite archive size
- `feature_dimensions`: Behavioral features

**MO Parameters:**
- `objectives`: List of objectives
- `pareto_front_size`: Pareto front size
- `algorithm`: NSGA2, SPEA2, etc.

**Adversarial Parameters:**
- `adversarial_rounds`: Number of rounds
- `robustness_threshold`: Pass threshold

---

## Progress Callbacks

Track evolution progress with callbacks:

```python
from openevolve.unified import ProgressUpdate

def progress_callback(update: ProgressUpdate):
    print(f"[{update.stage}] {update.percent_complete}%: {update.message}")
    if update.stage == 'evolving':
        print(f"  Iteration {update.current_iteration}/{update.total_iterations}")
        print(f"  Current score: {update.current_score:.3f}")
        print(f"  Best score: {update.best_score_so_far:.3f}")

result = await evolve(
    problem="Optimize function",
    domain="general",
    callback=progress_callback
)
```

### Progress Stages

| Stage | Description |
|-------|-------------|
| `analyzing` | Analyzing problem characteristics |
| `selecting_strategy` | Selecting optimal evolutionary strategy |
| `generating_config` | Generating configuration |
| `evolving` | Running evolutionary optimization |
| `extracting_knowledge` | Extracting knowledge artifacts |
| `running_gauntlet` | Running gauntlet evaluation |
| `learning` | Updating strategy recommendations |
| `complete` | Evolution complete |

---

## Knowledge Extraction

The API automatically extracts learning from each run:

### Enable/Disable Extraction

```python
# Enable extraction (default)
result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    store_knowledge=True
)

# Disable extraction
result = await evolve(
    problem="Quick test",
    domain="general",
    store_knowledge=False
)
```

### Artifacts Extracted

1. **Solution Patterns**
   - Successful solution structures
   - Common optimization patterns

2. **Performance Metrics**
   - Convergence rate
   - Sample efficiency
   - Best fitness

3. **Strategy Effectiveness**
   - Which strategies worked
   - For which problem types

4. **Domain Insights**
   - Domain-specific patterns
   - Successful approaches

### Provide Knowledge Engine

```python
from knowledge_engine import UnifiedKnowledgeGraph

# Create knowledge engine
ke = UnifiedKnowledgeGraph()

# Use with evolution API
result = await evolve(
    problem="Optimize portfolio",
    domain="finance",
    knowledge_engine=ke,
    store_knowledge=True
)
```

---

## Gauntlet Evaluation

The 3-round gauntlet provides quality assurance:

### Rounds

1. **Round 1: LoongFlow AI** (20% weight)
   - Quick AI evaluation
   - Screens for obvious issues
   - Fast feedback

2. **Round 2: Red Team** (30% weight)
   - Adversarial testing
   - Attacks edge cases
   - Tests robustness

3. **Round 3: Gold Team** (50% weight)
   - Consensus verification
   - Multiple evaluators
   - Final approval

### Enable/Disable Gauntlet

```python
# Run gauntlet (default for quality)
result = await evolve(
    problem="Critical system",
    domain="engineering",
    run_gauntlet=True
)

print(f"Gauntlet passed: {result.gauntlet_result.passed}")
print(f"Gauntlet score: {result.gauntlet_result.final_score}")

# Skip gauntlet (faster)
result = await evolve(
    problem="Quick prototype",
    domain="web",
    run_gauntlet=False
)
```

### Gauntlet Results

```python
result = await evolve(
    problem="Safety-critical system",
    domain="engineering",
    run_gauntlet=True
)

if result.gauntlet_result:
    print(f"Rounds completed: {result.gauntlet_result.rounds_completed}")
    print(f"Round 1 (LoongFlow): {result.gauntlet_result.round1_result.score}")
    print(f"Round 2 (Red Team): {result.gauntlet_result.round2_result.score}")
    print(f"Round 3 (Gold Team): {result.gauntlet_result.round3_result.score}")
    print(f"Final score: {result.gauntlet_result.final_score}")
```

---

## Result Handling

### EvolutionResult Object

```python
result = await evolve(
    problem="Optimize portfolio",
    domain="finance"
)
```

**Attributes:**

```python
# Solution
result.best_solution      # Best solution found
result.final_score        # Fitness/score

# Strategy
result.strategy_used      # SystemMode object
result.config_used        # Configuration used

# Artifacts
result.evolution_artifacts  # List of artifacts
result.gauntlet_result      # Gauntlet result (if run)

# Performance
result.total_time         # Execution time (seconds)
result.iterations         # Iterations performed
result.evaluations        # Evaluations performed

# Metadata
result.metadata           # Additional metadata
result.error              # Error message if failed
```

### Convert to Dictionary

```python
result_dict = result.to_dict()
print(json.dumps(result_dict, indent=2))
```

### Save/Load Results

```python
# Save result
result.save("./results/portfolio_optimization.json")

# Load result
from openevolve.unified import EvolutionResult
loaded = EvolutionResult.load("./results/portfolio_optimization.json")
```

### Access Strategy Details

```python
result = await evolve(
    problem="Optimize portfolio",
    domain="finance"
)

print(f"System: {result.strategy_used.system}")
print(f"Mode: {result.strategy_used.mode}")
print(f"Confidence: {result.strategy_used.confidence:.2%}")
print(f"Reasoning: {result.strategy_used.reasoning}")
```

---

## Advanced Usage

### Custom Strategy Recommender

```python
from knowledge_engine.core.strategy_recommender import StrategyRecommender

# Create custom recommender
recommender = StrategyRecommender(
    knowledge_engine=ke,
    learning_enabled=True
)

# Use with API
api = UnifiedEvolutionAPI(
    strategy_recommender=recommender
)

result = await api.evolve(
    problem="Optimize portfolio",
    domain="finance"
)
```

### Custom Gauntlet Configuration

```python
from openevolve.gauntlets.three_round_orchestrator import ThreeRoundConfig

# Create strict gauntlet
strict_config = ThreeRoundConfig(
    round1_threshold=0.7,
    round2_threshold=0.8,
    round3_threshold=0.9,
    enable_early_termination=True
)

api = UnifiedEvolutionAPI(
    enable_gauntlets=True
)

# Use strict config (via custom gauntlet orchestrator)
```

### Domain-Specific Optimization

```python
# Finance: Maximize Sharpe ratio
result = await evolve(
    problem="Maximize portfolio Sharpe ratio with 20% max drawdown",
    domain="finance",
    constraints={
        'objectives': ['return', 'risk'],
        'max_drawdown': 0.20
    }
)

# Science: Minimize experiments
result = await evolve(
    problem="Optimize reaction conditions",
    domain="science",
    constraints={
        'experiment_cost': 5000,  # $5K per experiment
        'max_experiments': 20
    }
)

# Engineering: Safety-critical
result = await evolve(
    problem="Design bridge support structure",
    domain="engineering",
    constraints={
        'safety_factor': 3.0,
        'safety_critical': True
    },
    run_gauntlet=True  # Always run gauntlet for safety
)
```

---

## Examples

### Example 1: Portfolio Optimization

```python
from openevolve.unified import evolve

result = await evolve(
    problem="Maximize portfolio Sharpe ratio",
    domain="finance",
    constraints={
        'objectives': ['return', 'risk', 'liquidity'],
        'max_positions': 50,
        'min_diversification': 10
    }
)

print(f"Expected Sharpe: {result.final_score:.2f}")
print(f"Strategy: {result.strategy_used.mode}")
print(f"Evaluations: {result.evaluations}")  # ~60% fewer with PES
```

### Example 2: Experimental Design

```python
result = await evolve(
    problem="Optimize chemical reaction yield",
    domain="science",
    constraints={
        'experiment_cost': 5000,
        'max_budget': 100000  # $100K total
    }
)

print(f"Conditions: {result.best_solution}")
print(f"Predicted yield: {result.final_score:.1%}")
print(f"Experiments: {result.evaluations}")  # ~12 vs 30 baseline (60% reduction)
```

### Example 3: Batch Optimization

```python
problems = [
    "Optimize homepage load time",
    "Improve Time to Interactive",
    "Reduce First Contentful Paint"
]

results = await evolve_batch(
    problems=problems,
    domain="web",
    max_concurrent=3
)

for problem, result in zip(problems, results):
    print(f"{problem}: {result.final_score:.3f}")
```

### Example 4: With Progress Tracking

```python
import matplotlib.pyplot as plt

progress_data = []

def track_progress(update):
    if update.stage == 'evolving':
        progress_data.append({
            'iteration': update.current_iteration,
            'score': update.current_score,
            'best': update.best_score_so_far
        })

result = await evolve(
    problem="Optimize function",
    domain="general",
    callback=track_progress
)

# Plot progress
iters = [d['iteration'] for d in progress_data]
scores = [d['score'] for d in progress_data]
best = [d['best'] for d in progress_data]

plt.plot(iters, scores, label='Current')
plt.plot(iters, best, label='Best')
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.legend()
plt.show()
```

### Example 5: Multi-Objective Optimization

```python
result = await evolve(
    problem="Design efficient vehicle",
    domain="engineering",
    constraints={
        'objectives': ['speed', 'efficiency', 'safety'],
        'weights': {
            'speed': 0.3,
            'efficiency': 0.4,
            'safety': 0.3
        }
    }
)

# Access Pareto front
if result.strategy_used.mode == 'mo':
    print(f"Pareto solutions: {len(result.metadata.get('pareto_front', []))}")
```

---

## Performance Tips

1. **Use appropriate domain**
   - Correct domain enables optimal defaults
   - `finance`, `science`, `engineering` → use PES for expensive evals

2. **Adjust iteration count**
   - Expensive evaluations: fewer iterations (30-50)
   - Cheap evaluations: more iterations (200-500)

3. **Enable gauntlets for quality**
   - Critical systems: always run gauntlets
   - Prototypes: skip for speed

4. **Batch parallel problems**
   - Use `evolve_batch()` for multiple problems
   - Adjust `max_concurrent` based on resources

5. **Disable knowledge extraction for speed**
   - Production runs: enable learning
   - Development/testing: disable for speed

---

## Troubleshooting

### Problem: Strategy selection is wrong

**Solution:** Provide custom configuration
```python
custom_config = UnifiedEvolutionConfig(
    evolution_mode=EvolutionMode.QD  # Force QD mode
)

result = await evolve(
    problem="...",
    domain="...",
    config=custom_config
)
```

### Problem: Evolution is too slow

**Solution:** Reduce iterations or skip gauntlet
```python
result = await evolve(
    problem="...",
    domain="...",
    config=UnifiedEvolutionConfig(max_iterations=30),
    run_gauntlet=False
)
```

### Problem: Not enough diversity

**Solution:** Use QD mode explicitly
```python
result = await evolve(
    problem="...",
    domain="...",
    config=UnifiedEvolutionConfig(
        evolution_mode=EvolutionMode.QD,
        qd=QDConfig(
            enabled=True,
            archive_size=2000  # Larger archive
        )
    )
)
```

### Problem: Solution fails in production

**Solution:** Run gauntlet and adversarial testing
```python
result = await evolve(
    problem="...",
    domain="engineering",
    run_gauntlet=True  # Always test
)
```

---

## API Reference

See [API Reference](./api_reference.md) for complete API documentation.

---

## Support

For issues, questions, or contributions:
- GitHub: [openevolve/unified](https://github.com/openevolve/unified)
- Documentation: [docs.openevolve.ai](https://docs.openevolve.ai)
