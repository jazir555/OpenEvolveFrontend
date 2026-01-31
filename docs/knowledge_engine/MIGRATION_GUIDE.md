# Migration Guide

**Version:** 1.0
**Last Updated:** January 30, 2026

---

## Table of Contents

- [From Pure OpenEvolve](#from-pure-openevolve)
- [From Pure LoongFlow](#from-pure-loongflow)
- [Hybrid Migration](#hybrid-migration)
- [Compatibility Matrix](#compatibility-matrix)
- [Rollback Plan](#rollback-plan)

---

## From Pure OpenEvolve

### Before

```python
from openevolve import QDOptimizer, QDConfig

# Configure QD optimizer
config = QDConfig(
    grid_resolution=10,
    feature_dimensions=["risk", "return"],
    archive_size=1000
)

# Run optimization
optimizer = QDOptimizer(config=config)
result = optimizer.run(problem=problem)
```

### After

```python
from openevolve.unified import evolve

# Single function call
result = await evolve(
    problem=problem,
    domain="your_domain"
)

# Or specify mode explicitly
result = await evolve(
    problem=problem,
    domain="your_domain",
    evolution_mode="qd"
)
```

### Migration Steps

#### Step 1: Update Imports

**Before:**
```python
from openevolve import QDOptimizer, MOOptimizer, AdversarialOptimizer
from openevolve.config import QDConfig, MOConfig
```

**After:**
```python
from openevolve.unified import evolve, quick_evolve, evolve_batch
```

#### Step 2: Remove System Selection Logic

**Before:**
```python
# Manual system selection
if problem_type == "quality_diversity":
    optimizer = QDOptimizer(config=qd_config)
elif problem_type == "multi_objective":
    optimizer = MOOptimizer(config=mo_config)
elif problem_type == "adversarial":
    optimizer = AdversarialOptimizer(config=adv_config)
```

**After:**
```python
# Automatic selection
result = await evolve(
    problem=problem,
    domain=domain
)
# System automatically selected based on problem characteristics
```

#### Step 3: Remove Mode Selection Logic

**Before:**
```python
# Manual mode selection
if need_diversity:
    mode = "qd"
elif multiple_objectives:
    mode = "mo"
elif need_robustness:
    mode = "adversarial"
else:
    mode = "standard"

optimizer = create_optimizer(mode=mode)
```

**After:**
```python
# Automatic mode selection
result = await evolve(
    problem=problem,
    domain=domain
)
# Mode automatically selected
```

#### Step 4: Update Configuration

**Before:**
```python
config = QDConfig(
    grid_resolution=10,
    feature_dimensions=["risk", "return"],
    archive_size=1000,
    # ... 20 more QD-specific params
)
```

**After:**
```python
# Use unified config (or let auto-config work)
from openevolve.unified import UnifiedEvolutionConfig

config = UnifiedEvolutionConfig(
    domain="finance",
    evolution_mode="qd",
    grid_resolution=10,
    feature_dimensions=["risk", "return"],
    archive_size=1000
)

# Or even simpler - let auto-config handle it
result = await evolve(
    problem=problem,
    domain="finance"
)
```

#### Step 5: Test and Validate

```python
# Run old and new in parallel
old_result = old_optimizer.run(problem)

new_result = await evolve(
    problem=problem,
    domain=domain,
    max_evaluations=old_result.num_evaluations
)

# Compare results
assert new_result['fitness'] >= old_result.fitness
print(f"Improvement: {new_result['improvement']}")
```

#### Step 6: Monitor Performance

```python
# Log results for analysis
log_migration_result(
    old_system="openevolve",
    new_system="unified",
    problem=problem,
    old_fitness=old_result.fitness,
    new_fitness=new_result['fitness'],
    old_evals=old_result.num_evaluations,
    new_evals=new_result['evaluations']
)
```

### Example Migration: Portfolio Optimization

**Before (Pure OpenEvolve):**
```python
from openevolve import QDOptimizer, QDConfig

# Configure
config = QDConfig(
    grid_resolution=10,
    feature_dimensions=["risk", "return"],
    archive_size=1000,
    max_iterations=100
)

# Optimize
optimizer = QDOptimizer(config=config)
result = optimizer.run(
    problem="Optimize portfolio allocation",
    evaluation_function=portfolio_evaluator
)

# Access results
best_portfolio = result.best_solution
archive = result.archive
```

**After (Unified Engine):**
```python
from openevolve.unified import evolve

# Optimize (auto-selects QD mode)
result = await evolve(
    problem="Optimize portfolio allocation",
    domain="finance",
    evaluation_function=portfolio_evaluator
)

# Access results
best_portfolio = result['best_solution']
archive = result['archive']

# Bonus: Also get strategy recommendation
print(f"Mode used: {result['strategy_used']}")
print(f"Confidence: {result['strategy_confidence']}")
```

---

## From Pure LoongFlow

### Before

```python
from loongflow.agents.math_agent import MathPESAgent
from loongflow.framework.config import PESConfig

# Configure PES agent
config = PESConfig(
    max_iterations=50,
    enable_planning=True,
    enable_memory=True
)

# Run optimization
agent = MathPESAgent(config=config)
result = agent.run(problem=problem)
```

### After

```python
from openevolve.unified import evolve

# Single function call
result = await evolve(
    problem=problem,
    domain="your_domain"
)

# Or specify mode explicitly
result = await evolve(
    problem=problem,
    domain="your_domain",
    evolution_mode="pes"
)
```

### Migration Steps

#### Step 1: Update Imports

**Before:**
```python
from loongflow.agents.math_agent import MathPESAgent
from loongflow.agents.general_agent import GeneralPESAgent
from loongflow.framework.config import PESConfig
```

**After:**
```python
from openevolve.unified import evolve
```

#### Step 2: Remove Agent Selection Logic

**Before:**
```python
# Manual agent selection
if domain == "math":
    agent = MathPESAgent(config=config)
elif domain == "general":
    agent = GeneralPESAgent(config=config)
```

**After:**
```python
# Automatic selection
result = await evolve(
    problem=problem,
    domain=domain
)
```

#### Step 3: Update Configuration

**Before:**
```python
config = PESConfig(
    max_iterations=50,
    enable_planning=True,
    enable_memory=True,
    early_stopping=True,
    early_stop_threshold=0.9
)
```

**After:**
```python
# Use unified config
from openevolve.unified import UnifiedEvolutionConfig

config = UnifiedEvolutionConfig(
    domain="finance",
    evolution_mode="pes",
    max_iterations=50,
    enable_planning=True,
    enable_memory=True,
    early_stopping=True
)

# Or let auto-config handle it
result = await evolve(problem=problem, domain=domain)
```

#### Step 4: Handle Result Format Changes

**Before:**
```python
result = agent.run(problem)

best_solution = result.best_solution
best_fitness = result.best_fitness
total_evaluations = result.total_evaluations
```

**After:**
```python
result = await evolve(problem=problem, domain=domain)

best_solution = result['best_solution']
best_fitness = result['fitness']
total_evaluations = result['evaluations']

# Bonus: Additional information
strategy_used = result['strategy_used']
gauntlet_results = result['gauntlet_results']
```

#### Step 5: Test and Validate

```python
# Compare results
old_result = pes_agent.run(problem)
new_result = await evolve(problem=problem, domain=domain)

assert new_result['fitness'] >= old_result.best_fitness
assert new_result['evaluations'] <= old_result.total_evaluations
```

### Example Migration: Chemical Reaction Optimization

**Before (Pure LoongFlow):**
```python
from loongflow.agents.science_agent import SciencePESAgent

config = PESConfig(
    max_iterations=30,
    enable_planning=True,
    enable_memory=True
)

agent = SciencePESAgent(config=config)
result = agent.run(
    problem="Optimize chemical reaction conditions",
    context={"experiment_cost": 5000}
)

best_conditions = result.best_solution
```

**After (Unified Engine):**
```python
from openevolve.unified import evolve

result = await evolve(
    problem="Optimize chemical reaction conditions",
    domain="science",
    max_evaluations=30
)

best_conditions = result['best_solution']

# Bonus: Also get knowledge extraction
if result['knowledge_extracted']:
    similar_experiments = query_knowledge(
        "Similar chemical reactions",
        domain="science"
    )
```

---

## Hybrid Migration

### Migrating Gradually

#### Phase 1: Start with New Problems

```python
# New problems use unified API
new_result = await evolve(
    problem=new_problem,
    domain="finance"
)

# Old problems still work
old_result = old_optimizer.run(old_problem)
```

#### Phase 2: Migrate Non-Critical Problems

```python
# Low-stakes problems first
for problem in low_stakes_problems:
    result = await evolve(
        problem=problem,
        domain="finance"
    )

    # Validate results
    assert validate_result(result)

    # If passes, retire old code
    if result['fitness'] >= old_result['fitness']:
        retire_old_code(problem)
```

#### Phase 3: Migrate Critical Problems

```python
# Only after validation on non-critical
for problem in critical_problems:
    # Run both in parallel
    old_result = old_optimizer.run(problem)
    new_result = await evolve(
        problem=problem,
        domain="finance",
        config=conservative_config  # Careful tuning
    )

    # Validate thoroughly
    if validate_migration(old_result, new_result):
        switch_to_new(problem)
    else:
        investigate_failure()
```

#### Phase 4: Retire Old Code

```python
# After all problems migrated
# Deprecate old APIs
import warnings

def old_optimize(problem):
    warnings.warn(
        "old_optimize is deprecated, use evolve() instead",
        DeprecationWarning
    )
    return evolve(problem=problem)
```

### A/B Testing New vs Old

```python
# Run comparison
old_results = []
new_results = []

for problem in test_problems:
    old_result = old_optimizer.run(problem)
    new_result = await evolve(problem=problem, domain=domain)

    old_results.append(old_result)
    new_results.append(new_result)

# Analyze
old_avg = mean([r.fitness for r in old_results])
new_avg = mean([r['fitness'] for r in new_results])
improvement = (new_avg - old_avg) / old_avg

print(f"Average improvement: {improvement:.1%}")

# Statistical test
t_stat, p_value = ttest(old_results, new_results)
if p_value < 0.05:
    print("Significant improvement!")
```

---

## Compatibility Matrix

### OpenEvolve Feature Mapping

| OpenEvolve Feature | Unified Equivalent | Notes |
|-------------------|-------------------|-------|
| `QDOptimizer` | `evolve(mode="qd")` or `evolve(domain=X)` | Auto-selected for diversity needs |
| `MOOptimizer` | `evolve(mode="mo")` | Auto-selected for multiple objectives |
| `AdversarialOptimizer` | `evolve(mode="adversarial")` | Auto-selected for robustness needs |
| `StandardGA` | `evolve(mode="standard")` | Default fallback |
| `QDConfig` | `UnifiedEvolutionConfig(evolution_mode="qd")` | Merged into unified config |
| `MOConfig` | `UnifiedEvolutionConfig(evolution_mode="mo")` | Merged into unified config |
| `archive` | `result['archive']` | Same structure |
| `pareto_front` | `result['pareto_front']` | Same structure |
| `evolutionary_tree` | `result['evolutionary_tree']` | Same structure |

### LoongFlow Feature Mapping

| LoongFlow Feature | Unified Equivalent | Notes |
|------------------|-------------------|-------|
| `PESAgent` | `evolve(mode="pes")` | Auto-selected for expensive evaluations |
| `PESConfig` | `UnifiedEvolutionConfig(evolution_mode="pes")` | Merged into unified config |
| `planning_phase` | `enable_planning=True` | Same behavior |
| `memory_retrieval` | `enable_memory=True` | Same behavior |
| `early_stopping` | `early_stopping=True` | Same behavior |
| `plan` | `result['plan']` | Stored in evolutionary_tree |
| `summary` | `result['summary']` | Stored in evolutionary_tree |

### Parameter Mapping

#### OpenEvolve → Unified

| Old Parameter | New Parameter | Notes |
|--------------|--------------|-------|
| `grid_resolution` | `grid_resolution` | Same |
| `feature_dimensions` | `feature_dimensions` | Same |
| `archive_size` | `archive_size` | Same |
| `pareto_front_size` | `pareto_front_size` | Same |
| `adversarial_rounds` | `adversarial_rounds` | Same |
| `population_size` | `population_size` | Same |
| `mutation_rate` | `mutation_rate` | Same |
| `crossover_rate` | `crossover_rate` | Same |

#### LoongFlow → Unified

| Old Parameter | New Parameter | Notes |
|--------------|--------------|-------|
| `max_iterations` | `max_iterations` | Same |
| `enable_planning` | `enable_planning` | Same |
| `enable_memory` | `enable_memory` | Same |
| `early_stopping` | `early_stopping` | Same |
| `early_stop_threshold` | `early_stop_threshold` | Same |

---

## Rollback Plan

### Feature Flags

```python
# Enable/disable unified API
ENABLE_UNIFIED_API = os.getenv('ENABLE_UNIFIED_API', 'false')

if ENABLE_UNIFIED_API == 'true':
    result = await evolve(problem=problem, domain=domain)
else:
    result = old_optimizer.run(problem)
```

### Gradual Rollout

```python
# Percentage-based rollout
rollout_percentage = get_rollout_percentage()

if random.random() < rollout_percentage:
    result = await evolve(problem=problem, domain=domain)
else:
    result = old_optimizer.run(problem)
```

### Automatic Rollback

```python
# Monitor for issues
def monitor_performance(old_result, new_result):
    # Check for significant degradation
    if new_result['fitness'] < old_result.fitness * 0.9:
        alert_user("Performance degradation detected!")
        rollback_to_old()
        return False

    # Check for errors
    if new_result['errors'] > old_result.errors * 2:
        alert_user("Error rate increased!")
        rollback_to_old()
        return False

    return True
```

### Keeping Old Code Available

```python
# Keep old code as fallback
from openevolve.legacy import QDOptimizer, MOOptimizer

def evolve_with_fallback(problem, domain, **kwargs):
    try:
        # Try new unified API
        return await evolve(problem=problem, domain=domain, **kwargs)
    except Exception as e:
        logger.warning(f"Unified API failed: {e}, using fallback")
        # Fallback to old API
        if domain == "finance":
            return QDOptimizer().run(problem)
        elif domain == "trading":
            return MOOptimizer().run(problem)
        else:
            raise
```

### Validation Checks

```python
# Before rolling out to production
def validate_migration():
    checks = [
        check_fitness_improvement,
        check_evaluation_reduction,
        check_stability,
        check_errors,
        check_performance
    ]

    for check in checks:
        if not check():
            return False

    return True

# Only deploy if validation passes
if validate_migration():
    deploy_unified_api()
else:
    investigate_and_fix()
```

---

## Quick Reference

### Common Migration Patterns

#### Pattern 1: Simple Replacement

```python
# Before
result = optimizer.run(problem)

# After
result = await evolve(problem=problem, domain=domain)
```

#### Pattern 2: With Configuration

```python
# Before
config = QDConfig(grid_resolution=10)
optimizer = QDOptimizer(config=config)
result = optimizer.run(problem)

# After
result = await evolve(
    problem=problem,
    domain=domain,
    grid_resolution=10
)
```

#### Pattern 3: Multi-Objective

```python
# Before
config = MOConfig(objectives=["return", "risk"])
optimizer = MOOptimizer(config=config)
result = optimizer.run(problem)

# After
result = await evolve(
    problem=problem,
    domain=domain,
    objectives=["return", "risk"]
)
```

#### Pattern 4: Custom Evaluation

```python
# Before
result = optimizer.run(problem, evaluation_function=my_eval)

# After
result = await evolve(
    problem=problem,
    domain=domain,
    evaluation_function=my_eval
)
```

---

**End of Migration Guide**

For more information, see:
- [Unified Evolution Engine Guide](UNIFIED_EVOLUTION_ENGINE_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
