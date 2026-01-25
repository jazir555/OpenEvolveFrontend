# RESE Troubleshooting Guide

**Recursive Epistemic Solvability Engine**
**Version:** 1.0.0
**Last Updated:** 2025-12-31

---

## Table of Contents

1. [Common Issues](#common-issues)
2. [Debugging Procedures](#debugging-procedures)
3. [Performance Tuning](#performance-tuning)
4. [Error Messages](#error-messages)
5. [FAQ](#faq)

---

## Common Issues

### Installation Issues

#### Issue: ImportError when importing RESE

**Symptoms:**
```python
>>> import rese
ImportError: No module named 'rese'
```

**Diagnosis:**
- RESE not installed or not in PYTHONPATH

**Solutions:**

1. **Install in development mode:**
```bash
cd C:\Users\mmeadow\Documents\OpenEvolve\Frontend
pip install -e rese/
```

2. **Add to PYTHONPATH:**
```bash
export PYTHONPATH="${PYTHONPATH}:C:\Users\mmeadow\Documents\OpenEvolve\Frontend"
```

3. **Use absolute import:**
```python
import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')
import rese
```

---

#### Issue: Missing dependencies

**Symptoms:**
```python
>>> from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine
ImportError: No module named 'networkx'
```

**Diagnosis:**
- Required dependencies not installed

**Solutions:**

1. **Install all dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install specific missing package:**
```bash
pip install networkx
```

3. **Check installed packages:**
```bash
pip list | grep networkx
```

---

### Pipeline Issues

#### Issue: Pipeline fails immediately

**Symptoms:**
```python
result = pipeline.run(problem)
# PipelineError: Invalid input
```

**Diagnosis:**
- Input validation failed

**Solutions:**

1. **Check input format:**
```python
from rese.rese_pipeline import ProblemInput

# Ensure all required fields present
problem = ProblemInput(
    id="test_problem",          # Required
    description="Test problem",  # Required
    constraints=[...],           # Required
    variables={...}              # Required
)
```

2. **Validate constraints:**
```python
# Each constraint must have: id, type, description
valid_constraint = {
    "id": "c1",
    "type": "hard",  # or "soft", "preference"
    "description": "Constraint description",
    "formalization": "∀ x: x > 0"  # Optional
}
```

3. **Check variables:**
```python
# Variables must be a dictionary
variables = {
    "num_cities": 50,
    "coordinates": [...]
}
```

---

#### Issue: Phase I fails with contradiction errors

**Symptoms:**
```
[Phase I] Detected 5 contradictory constraint pairs
[Phase I] ✗ Failed
```

**Diagnosis:**
- Conflicting hard constraints

**Solutions:**

1. **Review contradictions:**
```python
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine

sce = SymbolicConstraintEngine()
for constraint_dict in problem.constraints:
    sce.add_constraint(constraint)

conflicts = sce.detect_conflicts()
for c1, c2 in conflicts:
    print(f"Conflict: {c1} vs {c2}")
```

2. **Convert some hard constraints to soft:**
```python
# Instead of:
constraint = {"id": "c1", "type": "hard", ...}

# Use:
constraint = {"id": "c1", "type": "soft", ...}
```

3. **Remove redundant constraints:**
```python
# Check if one constraint implies another
# Remove the weaker one
```

---

#### Issue: Phase II returns low isomorphism score

**Symptoms:**
```
[Phase II] Isomorphism score: 0.35
[Phase II] Knowledge transfer not recommended
```

**Diagnosis:**
- Domains too dissimilar for knowledge transfer

**Solutions:**

1. **Skip Phase II for unrelated domains:**
```python
# Run only Phase I and Phase III
result = pipeline.run(problem, phases=['phase1', 'phase3'])
```

2. **Use different source domains:**
```python
# Try multiple source domains
from rese.phase2.imech import IMechValidator, Domain

validator = IMechValidator()

best_score = 0
best_domain = None

for source in potential_source_domains:
    comparison = validator.compare_domains(source, target)

    if comparison.score > best_score:
        best_score = comparison.score
        best_domain = source

print(f"Best match: {best_domain.name} (score: {best_score:.2f})")
```

3. **Disable isomorphism requirement:**
```python
# In config.py
config.phase2.psi3_target_accuracy = 0.5  # Lower threshold
```

---

#### Issue: Phase III converges too slowly

**Symptoms:**
```
[Phase III] Iteration 1000/10000, ACI = 0.45
[Phase III] Iteration 2000/10000, ACI = 0.46
[Phase III] Taking too long...
```

**Diagnosis:**
- Poor ACI guidance or wrong parameters

**Solutions:**

1. **Reduce iterations:**
```python
from rese.config import Phase3Config, RESEConfig

config = RESEConfig(
    phase3=Phase3Config(
        gamma2_iterations=500,  # Reduce from 1000
        convergence_patience=50  # Stop if no improvement
    )
)
```

2. **Adjust exploration constant:**
```python
# Higher C = more exploration
search = MCTSSearch(
    exploration_constant=2.0,  # Default is 1.41
    iterations=1000
)
```

3. **Enable parallel agents:**
```python
config = RESEConfig(
    phase3=Phase3Config(
        gamma2_parallel_agents=8  # Use 8 parallel searches
    )
)
```

4. **Use better initialization:**
```python
# Warm start with heuristic solution
initial_state = heuristic_solution(problem)
result = search.search(initial_state)
```

---

#### Issue: Phase IV validation fails

**Symptoms:**
```
[Phase IV] Validation score: 0.52
[Phase IV] ✗ Validation failed (threshold: 0.70)
```

**Diagnosis:**
- Solution doesn't generalize

**Solutions:**

1. **Check for overfitting:**
```python
# Use larger holdout set
from rese.phase4.aci_reduction_validator import Delta3Validator

validator = Delta3Validator(
    holdout_ratio=0.3  # Increase from 0.2
)
```

2. **Review ACI reduction:**
```python
# If ACI reduction is small, solution isn't effective
if result.aci_reduction < 0.1:
    print("Warning: Minimal ACI reduction achieved")
    print("Solution may not be meaningful")
```

3. **Lower validation threshold:**
```python
validator = Delta3Validator(
    validation_threshold=0.6  # Lower from 0.7
)
```

---

### Performance Issues

#### Issue: Pipeline runs very slowly

**Symptoms:**
- Pipeline takes hours for small problems
- Memory usage grows excessively
- CPU utilization low

**Diagnosis:**
- Inefficient algorithms or poor configuration

**Solutions:**

1. **Enable caching:**
```python
from rese.config import RESEConfig, PipelineConfig

config = RESEConfig(
    pipeline=PipelineConfig(
        enable_caching=True,
        cache_ttl_seconds=3600
    )
)
```

2. **Profile execution:**
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

result = pipeline.run(problem)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime').print_stats(20)  # Top 20
```

3. **Use parallel processing:**
```python
config = RESEConfig(
    pipeline=PipelineConfig(
        max_parallel_tasks=4  # Run phases in parallel
    )
)
```

4. **Reduce problem size:**
```python
# Test with smaller instance first
small_problem = scale_down_problem(problem, factor=0.1)
result = pipeline.run(small_problem)
```

---

#### Issue: High memory usage

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Diagnosis:**
- Memory leak or excessive caching

**Solutions:**

1. **Clear cache:**
```python
from rese.rese_pipeline import CacheManager

cache = CacheManager(config)
cache.clear()
```

2. **Reduce cache size:**
```python
config = RESEConfig(
    pipeline=PipelineConfig(
        enable_caching=True,
        cache_ttl_seconds=1800  # Shorter TTL
    )
)
```

3. **Limit MCTS tree size:**
```python
from rese.phase3.mcts_search import MCTSSearch

search = MCTSSearch(
    max_tree_size=10000,  # Limit tree nodes
    iterations=1000
)
```

4. **Use generators instead of lists:**
```python
# Instead of:
results = [process(x) for x in large_list]

# Use:
def process_generator(data):
    for x in data:
        yield process(x)

results = process_generator(large_list)
```

---

### ACI Issues

#### Issue: ACI values not correlating with difficulty

**Symptoms:**
- Easy problems have low ACI
- Hard problems have high ACI
- ACI seems random

**Diagnosis:**
- ACI weights not calibrated for domain

**Solutions:**

1. **Recalibrate ACI weights:**
```python
from rese.gamma1.core.aci_calculator import ACICalculator

# Default: alpha=0.35, beta=0.35, gamma=0.30
# Try different weights for your domain
aci_calc = ACICalculator(
    alpha=0.5,  # More weight on entropy
    beta=0.3,   # Less on coherence
    gamma=0.2   # Less on solvability
)
```

2. **Collect ACI training data:**
```python
# For problems with known solve times
training_data = []

for problem in benchmark_problems:
    aci_result = aci_calc.calculate(problem)
    training_data.append({
        'ACI': aci_result.ACI,
        'solve_time': problem.solve_time,
        'components': aci_result.components
    })

# Find correlation
import numpy as np
aci_values = [d['ACI'] for d in training_data]
solve_times = [d['solve_time'] for d in training_data]

correlation = np.corrcoef(aci_values, solve_times)[0, 1]
print(f"Correlation: {correlation:.2f}")
```

3. **Use domain-specific ACI:**
```python
# Create custom ACI component for your domain
class CustomACIComponent:
    def calculate(self, csp_instance):
        # Domain-specific logic
        return custom_value
```

---

#### Issue: ACI calculation is slow

**Symptoms:**
- ACI calculation takes >1 second
- Becomes bottleneck in search

**Diagnosis:**
- Inefficient component calculations

**Solutions:**

1. **Enable ACI caching:**
```python
aci_calc = ACICalculator(
    use_cache=True  # Cache results
)
```

2. **Simplify components:**
```python
# Use approximate entropy instead of exact
from rese.gamma1.core.entropy_engine import DisorderEntropy

entropy_engine = DisorderEntropy(
    method='approximate'  # Faster but less accurate
)
```

3. **Batch calculations:**
```python
# Calculate ACI for multiple states at once
states = [state1, state2, state3, ...]
aci_results = aci_calc.calculate_batch(states)
```

---

### Constraint Issues

#### Issue: DITO contradiction detection is slow

**Symptoms:**
```
[DITO] Checking 10000 constraints...
[DITO] Taking >60 seconds...
```

**Diagnosis:**
- Large constraint set or inefficient algorithm

**Solutions:**

1. **Check if DITO is needed:**
```python
# For <1000 constraints, basic SCE is fine
if len(constraints) < 1000:
    # Use SCE's built-in detection
    conflicts = sce.detect_conflicts()
else:
    # Use DITO
    from rese.core.dito_optimizer import DITOOptimizer
    optimizer = DITOOptimizer()
```

2. **Pre-filter constraints:**
```python
# Only check potentially conflicting pairs
from rese.core.dito_optimizer import DITOOptimizer

optimizer = DITOOptimizer()
optimizer.enable_pre_filtering = True
contradictions = optimizer.detect_contradictions()
```

3. **Use probabilistic checking:**
```python
# Check random sample instead of all pairs
optimizer = DITOOptimizer()
optimizer.sample_rate = 0.1  # Check 10% of pairs
contradictions = optimizer.detect_contradictions()
```

---

#### Issue: Constraints not being enforced

**Symptoms:**
- Solution violates constraints
- No error messages

**Diagnosis:**
- Constraints marked as PREFERENCE or not validated

**Solutions:**

1. **Check constraint types:**
```python
# Ensure critical constraints are HARD
for constraint in problem.constraints:
    if is_critical(constraint):
        constraint['type'] = 'hard'
    else:
        constraint['type'] = 'soft'
```

2. **Enable constraint validation:**
```python
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine

sce = SymbolicConstraintEngine()
sce.enable_validation = True  # Validate all constraints
```

3. **Add penalty functions:**
```python
# For soft constraints, add penalty to objective
def objective_with_penalty(solution):
    base_objective = solution.value

    penalty = 0
    for constraint in violated_constraints:
        penalty += constraint.penalty_weight

    return base_objective - penalty
```

---

## Debugging Procedures

### Enable Debug Logging

```python
import logging

# Enable RESE debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Log to file
logging.basicConfig(
    level=logging.DEBUG,
    filename='rese_debug.log',
    filemode='w'
)
```

---

### Trace Execution Flow

```python
# Add tracing to see what's happening
import sys

# Enable tracing
def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        if 'rese' in code.co_filename:
            print(f"Calling: {code.co_name} in {code.co_filename}")
    return trace_calls

sys.settrace(trace_calls)

# Run pipeline
result = pipeline.run(problem)

# Disable tracing
sys.settrace(None)
```

---

### Inspect Intermediate Results

```python
# Add progress callback to inspect phases
def inspect_phase(result):
    print(f"\n=== Phase: {result.status} ===")
    print(f"Output: {result.output}")
    print(f"Metrics: {result.metrics}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

pipeline.add_progress_callback(inspect_phase)
result = pipeline.run(problem)
```

---

### Validate Individual Components

```python
# Test SCE separately
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine

sce = SymbolicConstraintEngine()
for constraint in problem.constraints:
    sce.add_constraint(constraint)

print(f"Constraints: {len(sce.get_all_constraints())}")
print(f"Conflicts: {len(sce.detect_conflicts())}")
print(f"Valid: {sce.validate()['is_valid']}")

# Test ACI calculator separately
from rese.gamma1.core.aci_calculator import ACICalculator

aci_calc = ACICalculator()
result = aci_calc.calculate(csp_instance)
print(f"ACI: {result.ACI}")
print(f"Components: {result.components}")
```

---

## Performance Tuning

### Phase-Specific Tuning

#### Phase I Tuning

```python
from rese.config import Phase1Config

phase1_config = Phase1Config(
    # SCE settings
    sce_max_constraints=10000,
    sce_enable_caching=True,

    # Φ₁.₅ settings
    phi15_assumption_threshold=0.6,  # Lower = more assumptions
    phi15_max_assumptions=100,

    # Φ₂ settings
    phi2_bias_threshold=0.5,         # Lower = more sensitive
    phi2_auto_debias=False,          # Manual review recommended

    # Φ₃ settings
    phi3_max_iterations=100          # Contradiction resolution iterations
)
```

#### Phase II Tuning

```python
from rese.config import Phase2Config

phase2_config = Phase2Config(
    # Ψ₁ settings
    psi1_complexity_reduction_target=0.1,  # Target 10% of original
    psi1_max_inversion_depth=5,

    # Ψ₂ settings
    psi2_similarity_threshold=0.7,         # Minimum for transfer
    psi2_embedding_model="text-embedding-ada-002",

    # Ψ₃/I_mech settings
    psi3_target_accuracy=0.80,
    psi3_use_lean4_proofs=True,
    psi3_parallel_isomorphism_check=True
)
```

#### Phase III Tuning

```python
from rese.config import Phase3Config

phase3_config = Phase3Config(
    # Γ₁ settings
    gamma1_signal_threshold=0.5,
    gamma1_use_entropy_engine=True,
    gamma1_use_coherence_engine=True,

    # Γ₂ settings
    gamma2_iterations=1000,
    gamma2_playout_depth=100,
    gamma2_exploration_constant=1.41,      # UCB constant
    gamma2_adaptive_c=True,                # Adaptive exploration
    gamma2_parallel_agents=4,              # Parallel MCTS
    gamma2_aci_guided=True,

    # Convergence settings
    convergence_enabled=True,
    convergence_patience=50,
    convergence_min_delta=0.001
)
```

#### Phase IV Tuning

```python
from rese.config import Phase4Config

phase4_config = Phase4Config(
    # Δ₁ settings
    delta1_max_components=50,
    delta1_integration_strategy="hierarchical",

    # Δ₂ settings
    delta2_prediction_horizon=10,
    delta2_model_type="ensemble",

    # Δ₃ settings
    delta3_validation_threshold=0.7,       # Minimum validation score
    delta3_min_aci_reduction=0.2,          # Minimum 20% reduction
    delta3_holdout_ratio=0.2,              # 20% for testing
    delta3_significance_level=0.05         # P-value threshold
)
```

---

### Memory Optimization

```python
from rese.config import RESEConfig, PipelineConfig

config = RESEConfig(
    pipeline=PipelineConfig(
        # Enable caching but limit size
        enable_caching=True,
        cache_ttl_seconds=1800,

        # Resource limits
        max_memory_gb=8.0,              # Limit memory
        max_time_seconds=3600,          # 1 hour timeout
        max_parallel_tasks=2,           # Reduce parallelism

        # Checkpointing
        checkpoint_interval=300         # Save every 5 minutes
    )
)
```

---

### Speed Optimization

```python
# 1. Use compiled algorithms
config = RESEConfig(
    pipeline=PipelineConfig(
        enable_caching=True,
        enable_monitoring=False  # Disable monitoring overhead
    )
)

# 2. Reduce problem size
# Test with subset first
small_problem = sample_problem(problem, fraction=0.1)

# 3. Use faster algorithms
from rese.phase2.imech import IMechValidator

validator = IMechValidator(
    algorithm='weisfeiler_lehman',  # Faster than VF2
    parallel=True,
    num_cores=4
)

# 4. Batch operations
# Instead of loop:
# for state in states:
#     aci = aci_calc.calculate(state)

# Use batch:
aci_results = aci_calc.calculate_batch(states)
```

---

## Error Messages

### Common Errors and Solutions

#### `ValueError: Constraint must have a non-empty ID`

**Cause:** Constraint missing ID field

**Solution:**
```python
# Add ID to constraint
constraint = {
    "id": "unique_constraint_1",  # Required
    "type": "hard",
    "description": "..."
}
```

---

#### `PipelineError: Phase execution failed`

**Cause:** Phase executor raised exception

**Solution:**
```python
# Check phase result for details
result = pipeline.run(problem)

for phase_name, phase_result in result.phase_results.items():
    if phase_result.status.value == 'failed':
        print(f"Failed phase: {phase_name}")
        print(f"Errors: {phase_result.errors}")
        print(f"Traceback:")
        for error in phase_result.errors:
            if 'Traceback' in error:
                print(error)
```

---

#### `ValidationError: Invalid CSP instance`

**Cause:** CSP instance structure invalid

**Solution:**
```python
from rese.gamma1.core.csp_models import CSPInstance

# Ensure required fields
csp = CSPInstance(
    variables=var_dict,       # Required
    domains=domain_dict,       # Required
    constraints=constraint_list,  # Required
    metadata={}               # Optional
)
```

---

#### `MemoryError: Unable to allocate array`

**Cause:** Insufficient memory

**Solution:**
```python
# 1. Reduce problem size
small_problem = scale_down(problem, 0.5)

# 2. Clear cache
from rese.rese_pipeline import CacheManager
cache = CacheManager(config)
cache.clear()

# 3. Reduce memory limit
config = RESEConfig(
    pipeline=PipelineConfig(
        max_memory_gb=4.0
    )
)
```

---

#### `ImportError: Cannot import name 'SymbolicConstraintEngine'`

**Cause:** Incorrect import path

**Solution:**
```python
# Use full path from rese
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine

# Or add parent to path
import sys
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')
from rese.core.symbolic_constraint_engine import SymbolicConstraintEngine
```

---

## FAQ

### Q: How do I know if RESE is working correctly?

**A:** Run the test suite:

```bash
pytest tests/ -v
```

Check that all tests pass. If tests fail, there may be environment issues.

---

### Q: Why is my ACI value always 0.5?

**A:** ACI weights may not be calibrated for your domain. Try:

1. Recalibrate weights using known problems
2. Check if components are being calculated correctly
3. Verify CSP instance structure

---

### Q: Can I run RESE without all phases?

**A:** Yes! Specify which phases to run:

```python
# Run only Phase I and Phase III
result = pipeline.run(problem, phases=['phase1', 'phase3'])
```

---

### Q: How do I get more detailed error messages?

**A:** Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

### Q: RESE is too slow for my use case. What can I do?

**A:** Try:

1. Enable caching
2. Reduce problem size
3. Use parallel processing
4. Skip unnecessary phases
5. Tune phase-specific parameters

---

### Q: Can I use RESE in production?

**A:** Yes! Use production configuration:

```python
from rese.config import Environment, RESEConfig

config = RESEConfig().for_environment(Environment.PRODUCTION)
```

---

### Q: How do I report a bug?

**A:** File an issue on GitHub with:

1. RESE version
2. Python version
3. Minimal reproducible example
4. Error messages and traceback
5. Expected vs actual behavior

---

## Getting Additional Help

### Documentation
- [User Guide](user_guide.md)
- [Developer Guide](developer_guide.md)
- [API Reference](api_reference.md)
- [Integration Guide](e2e_integration.md)

### Community
- **GitHub Issues:** https://github.com/your-org/rese/issues
- **Discussions:** https://github.com/your-org/rese/discussions

### Support
- **Email:** support@rese.example.com
- **Discord:** https://discord.gg/rese-community

---

## End of Troubleshooting Guide

Still stuck? Check the [FAQ](#faq) or [contact support](#getting-additional-help).
