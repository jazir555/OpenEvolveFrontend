# HYBRID MAKER USER GUIDE

Comprehensive user guide for Hybrid MAKER strategies.

**Version:** 1.0.0
**Paper:** arXiv:2511.09030
**Last Updated:** 2025-12-30

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [When to Use Each Strategy](#when-to-use-each-strategy)
4. [Configuration Guide](#configuration-guide)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)
8. [Migration Guide](#migration-guide)
9. [Best Practices](#best-practices)
10. [Advanced Topics](#advanced-topics)

---

## Introduction

### What is Hybrid MAKER?

The Hybrid MAKER system integrates the MAKER framework (Multi-Agent Voting with Escalation and Red-flagging) with multiple computational strategies including MCTS (Monte Carlo Tree Search), Evolutionary Algorithms, and Adversarial Testing.

**Key Benefits:**

- **Zero-Error Guarantees**: Statistical convergence through first-to-ahead-by-k voting
- **Adaptive Strategy Selection**: Dynamically choose optimal approach
- **Robustness**: Adversarial testing finds edge cases
- **Flexibility**: Multiple strategies for different problem types
- **Scalability**: Parallel execution and efficient caching

### When to Use Hybrid MAKER

Use Hybrid MAKER when:

- You need high-reliability solutions with zero errors
- Problems are complex and require multiple strategies
- You have computational resources for parallel execution
- Problems can be decomposed into subtasks
- You want statistical convergence guarantees

Consider alternatives when:

- Problems are simple and deterministic
- Computational resources are very limited
- Fast approximation is acceptable
- Single strategy is known to work well

---

## Getting Started

### Installation

Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `asyncio`: For async execution
- `dataclasses`: For configuration dataclasses
- `typing`: For type hints
- `logging`: For logging
- `json`: For serialization

### Quick Start

**Example 1: Basic Usage**

```python
import asyncio
from hybrid_maker_integration import run_maker_hybrid, MAKERHybridMode

async def main():
    theorem = "forall n : nat, n + 0 = n"

    # Run with default settings
    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MCTS_THEN_MAKER
    )

    if result.success:
        print(f"Success! Fitness: {result.best_fitness:.3f}")
        print(f"Proof:\n{result.best_proof}")
    else:
        print("Failed to find proof")

asyncio.run(main())
```

**Example 2: Custom Configuration**

```python
from hybrid_maker_integration import MAKERHybridConfig

config = MAKERHybridConfig(
    voting_threshold=4,
    mcts_simulations=200,
    evolution_generations=30,
    population_size=25,
    enable_red_flagging=True
)

result = await run_maker_hybrid(
    theorem="forall n m : nat, n + m = m + n",
    mode=MAKERHybridMode.FULL_MAKER_HYBRID,
    config=config
)
```

### Checking Capabilities

Check what's available in your installation:

```python
from hybrid_maker_integration import get_maker_hybrid_capabilities

caps = get_maker_hybrid_capabilities()

print(f"MAKER Hybrid Enabled: {caps['maker_hybrid_enabled']}")
print(f"Integration Status: {caps['integration_status']}")

print("\nAvailable Modes:")
for mode in caps['modes']:
    print(f"  - {mode}")
```

---

## When to Use Each Strategy

### Decision Tree

```
Is the problem time-critical?
├── Yes → MCTS_THEN_MAKER (fast exploration)
└── No
    ├── Is robustness critical?
    │   ├── Yes → MAKER_ADVERSARIAL
    │   └── No
    │       ├── Is complexity unknown?
    │       │   ├── Yes → ADAPTIVE_MAKER
    │       │   └── No
    │       │       ├── Is problem highly complex?
    │       │       │   ├── Yes → FULL_MAKER_HYBRID
    │       │       │   └── No → MAKER_THEN_EVOLUTION
    │       └── Can tasks be parallelized?
    │           ├── Yes → MAKER_MDAP_PARALLEL
    │           └── No → MAKER_THEN_EVOLUTION
```

### Strategy Comparison

| Strategy | Speed | Quality | Robustness | Complexity | Best For |
|----------|-------|---------|------------|------------|----------|
| MCTS_THEN_MAKER | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | Quick solutions, exploration |
| MAKER_THEN_EVOLUTION | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | High-quality refinement |
| MAKER_ADVERSARIAL | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★★★☆ | Edge cases, robustness |
| ADAPTIVE_MAKER | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | Unknown complexity |
| MAKER_MDAP_PARALLEL | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★☆☆ | Speed, parallel tasks |
| FULL_MAKER_HYBRID | ★☆☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ | Maximum quality |

### Detailed Strategy Guides

#### MCTS-Then-MAKER

**Best For:**
- Problems requiring exploration
- When you have time constraints
- When solution space is well-structured

**How It Works:**
1. MCTS explores solution space with multiple simulations
2. Generates diverse candidate solutions
3. MAKER voting selects best candidate with zero-error guarantee

**Example Use Cases:**
- Theorem proving with search space
- Code generation with multiple approaches
- Planning problems with clear states

**Configuration Tips:**
- `mcts_simulations`: 50-200 (more = better exploration, slower)
- `voting_threshold`: 2-4 (lower = faster, higher = more reliable)

#### MAKER-Then-Evolution

**Best For:**
- Problems requiring refinement
- When you have good initial candidates
- When gradual improvement is acceptable

**How It Works:**
1. MAKER voting generates high-quality initial population
2. Evolution refines through mutation and crossover
3. MAKER voting can be used in selection

**Example Use Cases:**
- Optimization problems
- Code refactoring
- Parameter tuning

**Configuration Tips:**
- `initial_candidates`: 30-100 (more = better population, slower)
- `evolution_generations`: 10-50 (more = better refinement, slower)
- `population_size`: 15-30 (balance diversity and speed)

#### MAKER-Adversarial

**Best For:**
- Security-critical applications
- Problems with many edge cases
- When robustness is paramount

**How It Works:**
1. Red team generates attacks
2. Blue team generates defenses
3. MAKER voting selects most robust solution
4. Repeat for multiple rounds

**Example Use Cases:**
- Adversarial example generation
- Robustness testing
- Security validation

**Configuration Tips:**
- `adversarial_rounds`: 3-10 (more = more robustness, slower)
- `red_team_size`: 2-5 (more = more diverse attacks)
- `blue_team_size`: 2-5 (more = more diverse defenses)

#### Adaptive MAKER

**Best For:**
- Problems with unknown complexity
- When you want automatic optimization
- When resource constraints vary

**How It Works:**
1. Monitor population diversity and convergence
2. Switch strategies based on metrics:
   - Low diversity → MAKER voting (explore)
   - High convergence → MDAP decomposition (refine)
   - Normal → Evolution (optimize)

**Example Use Cases:**
- General problem solving
- Unknown problem domains
- Dynamic environments

**Configuration Tips:**
- `diversity_threshold`: 0.2-0.5 (lower = more frequent switching)
- `convergence_threshold`: 0.9-0.99 (higher = stricter convergence)
- `max_generations`: 30-100 (timeout for adaptive search)

#### MAKER-MDAP Parallel

**Best For:**
- When speed is critical
- When tasks can be decomposed
- When you have parallel resources

**How It Works:**
1. MAKER voting runs in parallel with MDAP decomposition
2. Both processes work independently
3. Results combined at end (best or average)

**Example Use Cases:**
- Fast prototyping
- Multi-objective optimization
- Independent subtasks

**Configuration Tips:**
- `mdap_agents`: 4-10 (more = finer decomposition, more overhead)
- `combination_method`: "best_fitness" or "average"
- Ensure adequate parallel resources

#### Full MAKER Hybrid

**Best For:**
- Critical applications requiring maximum quality
- When resources are not constrained
- When you want comprehensive search

**How It Works:**
1. Executes all strategies sequentially
2. Tracks best result across all phases
3. Returns optimal solution

**Example Use Cases:**
- Production code generation
- Critical theorem proving
- Research applications

**Configuration Tips:**
- Enable all features for maximum quality
- Set higher thresholds for voting and convergence
- Allocate sufficient time for all phases

---

## Configuration Guide

### Configuration Parameters

#### MAKER Voting Parameters

**voting_threshold (k)**: 2-8
- **What**: First-to-ahead-by-k threshold
- **Impact**: Higher k = more reliable, slower
- **Default**: 3
- **Recommendations**:
  - Quick prototyping: k=2
  - Standard use: k=3-4
  - Critical applications: k=5-6
  - Maximum reliability: k=7-8

**enable_red_flagging**: True/False
- **What**: Enable quality control filtering
- **Impact**: Prevents low-quality solutions
- **Default**: True
- **Recommendation**: Always keep enabled unless debugging

#### MCTS Parameters

**mcts_simulations**: 10-500
- **What**: Number of MCTS simulations per exploration
- **Impact**: More simulations = better exploration, slower
- **Default**: 100
- **Recommendations**:
  - Quick exploration: 50
  - Standard: 100-150
  - Thorough: 200-300
  - Exhaustive: 400-500

**exploration_constant (C)**: 1.0-3.0
- **What**: MCTS UCB exploration constant
- **Impact**: Higher = more exploration, lower = more exploitation
- **Default**: 1.414 (sqrt(2))
- **Recommendations**:
  - Exploit known solutions: C=1.0
  - Balanced: C=1.414
  - Explore heavily: C=2.0-3.0

#### Evolution Parameters

**evolution_generations**: 1-100
- **What**: Number of evolution generations
- **Impact**: More generations = better refinement, slower
- **Default**: 20
- **Recommendations**:
  - Quick refinement: 5-10
  - Standard: 20-30
  - Thorough: 40-60
  - Exhaustive: 80-100

**population_size**: 5-100
- **What**: Size of evolution population
- **Impact**: Larger = more diversity, more memory/time
- **Default**: 20
- **Recommendations**:
  - Minimal: 5-10
  - Standard: 15-25
  - High diversity: 30-50
  - Maximum: 60-100

**mutation_rate**: 0.0-1.0
- **What**: Probability of mutation
- **Impact**: Higher = more exploration, lower = more stability
- **Default**: 0.1
- **Recommendations**:
  - Conservative: 0.05-0.1
  - Balanced: 0.1-0.2
  - Exploratory: 0.2-0.4

**crossover_rate**: 0.0-1.0
- **What**: Probability of crossover
- **Impact**: Higher = more recombination, lower = more mutation
- **Default**: 0.7
- **Recommendations**:
  - Mutation-focused: 0.3-0.5
  - Balanced: 0.6-0.8
  - Crossover-focused: 0.8-0.95

#### Adversarial Parameters

**adversarial_rounds**: 1-10
- **What**: Number of adversarial rounds
- **Impact**: More rounds = more robustness, slower
- **Default**: 3
- **Recommendations**:
  - Quick robustness check: 1-2
  - Standard: 3-5
  - Thorough: 6-8
  - Maximum: 9-10

**red_team_size**: 1-5
- **What**: Number of red team agents
- **Impact**: More agents = more diverse attacks
- **Default**: 2
- **Recommendations**:
  - Minimal: 1
  - Standard: 2-3
  - Diverse: 4-5

**blue_team_size**: 1-5
- **What**: Number of blue team agents
- **Impact**: More agents = more diverse defenses
- **Default**: 2
- **Recommendations**:
  - Minimal: 1
  - Standard: 2-3
  - Diverse: 4-5

#### Adaptive Parameters

**diversity_threshold**: 0.0-1.0
- **What**: Minimum population diversity before switching strategies
- **Impact**: Lower = more frequent switching
- **Default**: 0.3
- **Recommendations**:
  - Aggressive switching: 0.2-0.3
  - Balanced: 0.3-0.4
  - Conservative: 0.4-0.5

**convergence_threshold**: 0.0-1.0
- **What**: Fitness threshold for considering converged
- **Impact**: Higher = stricter convergence criteria
- **Default**: 0.95
- **Recommendations**:
  - Loose: 0.85-0.90
  - Standard: 0.93-0.97
  - Strict: 0.98-0.99

#### MDAP Parameters

**enable_decomposition**: True/False
- **What**: Enable task decomposition
- **Impact**: Breaks complex problems into subtasks
- **Default**: True
- **Recommendation**: Enable for complex problems

**decomposition_depth**: 1-5
- **What**: Maximum depth of task decomposition
- **Impact**: Deeper = more subtasks, more complex
- **Default**: 3
- **Recommendations**:
  - Shallow: 1-2
  - Standard: 3
  - Deep: 4-5

**max_subtasks**: 1-20
- **What**: Maximum number of subtasks to create
- **Impact**: More subtasks = finer granularity, more overhead
- **Default**: 10
- **Recommendations**:
  - Coarse: 3-5
  - Standard: 8-12
  - Fine: 15-20

### Configuration Presets

#### Quick Preset

```python
config = MAKERHybridConfig(
    voting_threshold=2,
    mcts_simulations=50,
    evolution_generations=10,
    population_size=15,
    adversarial_rounds=1
)
```

#### Balanced Preset

```python
config = MAKERHybridConfig(
    voting_threshold=3,
    mcts_simulations=100,
    evolution_generations=20,
    population_size=20,
    adversarial_rounds=3,
    diversity_threshold=0.3
)
```

#### Quality Preset

```python
config = MAKERHybridConfig(
    voting_threshold=5,
    mcts_simulations=200,
    evolution_generations=40,
    population_size=30,
    adversarial_rounds=5,
    enable_red_flagging=True,
    convergence_threshold=0.98
)
```

#### Maximum Preset

```python
config = MAKERHybridConfig(
    voting_threshold=6,
    mcts_simulations=400,
    evolution_generations=80,
    population_size=50,
    adversarial_rounds=8,
    enable_red_flagging=True,
    enable_decomposition=True,
    decomposition_depth=4,
    max_subtasks=15,
    convergence_threshold=0.99
)
```

---

## Performance Tuning

### Performance Profiling

Enable performance monitoring:

```python
import logging
import time

logging.basicConfig(level=logging.INFO)

# Time execution
start = time.time()
result = await run_maker_hybrid(theorem, mode=mode)
elapsed = time.time() - start

print(f"Execution time: {elapsed:.2f}s")
print(f"Generations: {result.generations_completed}")
print(f"Time per generation: {elapsed/result.generations_completed:.2f}s")
```

### Bottleneck Identification

Common bottlenecks:

1. **LLM API Latency** (Most common)
   - Symptom: Long execution time, low CPU usage
   - Solution: Parallel API calls, caching

2. **Vote Collection**
   - Symptom: Slow voting phase
   - Solution: Lower k, parallel collection

3. **Population Evaluation**
   - Symptom: Slow evolution phase
   - Solution: Reduce population size, parallel evaluation

### Optimization Strategies

#### Strategy 1: Reduce Voting Overhead

```python
# Lower k for faster (less reliable) voting
config = MAKERHybridConfig(voting_threshold=2)

# Or disable red flagging (not recommended)
config = MAKERHybridConfig(
    voting_threshold=3,
    enable_red_flagging=False
)
```

#### Strategy 2: Enable Caching

```python
from mdap_engine import MDAPConfig

# Enable MDAP caching
mdap_config = MDAPConfig(
    cache_ttl_seconds=3600,  # 1 hour
    cache_max_size=10000
)
```

#### Strategy 3: Parallel Execution

```python
# Use parallel mode
result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.MAKER_MDAP_PARALLEL
)
```

#### Strategy 4: Early Stopping

```python
# Set lower convergence threshold
config = MAKERHybridConfig(
    convergence_threshold=0.90,  # Stop earlier
    max_generations=30  # Limit generations
)
```

### Performance Benchmarks

Approximate execution times (theorem proving):

| Strategy | Small Theorem | Medium Theorem | Large Theorem |
|----------|---------------|----------------|---------------|
| MCTS_THEN_MAKER | 10-30s | 30-60s | 60-120s |
| MAKER_THEN_EVOLUTION | 20-60s | 60-120s | 120-300s |
| MAKER_ADVERSARIAL | 30-90s | 90-180s | 180-450s |
| ADAPTIVE_MAKER | 40-100s | 100-200s | 200-500s |
| MAKER_MDAP_PARALLEL | 15-40s | 40-80s | 80-160s |
| FULL_MAKER_HYBRID | 60-180s | 180-360s | 360-900s |

*Note: Times depend on LLM API latency, problem complexity, and configuration*

---

## Troubleshooting

### Common Issues

#### Issue 1: No Solution Found

**Symptoms:**
- `result.success == False`
- Low best fitness
- Many failed attempts

**Possible Causes:**
- Problem too complex
- Insufficient generations
- Wrong strategy selected
- Poor configuration

**Solutions:**
1. Increase generations/simulations
2. Try different strategy
3. Use FULL_MAKER_HYBRID for comprehensive search
4. Adjust configuration parameters
5. Check problem statement for errors

```python
# Try with more resources
config = MAKERHybridConfig(
    evolution_generations=50,
    mcts_simulations=200
)

# Or use full hybrid
result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.FULL_MAKER_HYBRID,
    config=config
)
```

#### Issue 2: Slow Execution

**Symptoms:**
- Long execution time
- High API latency

**Possible Causes:**
- High voting threshold
- Many generations/simulations
- Sequential execution
- No caching

**Solutions:**
1. Lower voting threshold
2. Reduce generations/simulations
3. Use parallel mode
4. Enable caching
5. Use faster strategy (MCTS_THEN_MAKER)

```python
# Faster configuration
config = MAKERHybridConfig(
    voting_threshold=2,
    mcts_simulations=50,
    evolution_generations=10
)

# Use parallel mode
result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.MAKER_MDAP_PARALLEL,
    config=config
)
```

#### Issue 3: Low Quality Solutions

**Symptoms:**
- Solution found but low fitness
- Poor convergence
- Many red flags

**Possible Causes:**
- Voting threshold too low
- Insufficient exploration
- Poor fitness function
- Red flagging too strict

**Solutions:**
1. Increase voting threshold
2. Increase simulations/generations
3. Improve fitness function
4. Adjust red flag rules

```python
# Higher quality configuration
config = MAKERHybridConfig(
    voting_threshold=5,
    mcts_simulations=200,
    evolution_generations=40
)

# Enable adversarial for robustness
result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.MAKER_ADVERSARIAL,
    config=config
)
```

#### Issue 4: Memory Issues

**Symptoms:**
- Out of memory errors
- Slow performance
- System crashes

**Possible Causes:**
- Population too large
- Many generations
- Large MCTS tree
- No memory cleanup

**Solutions:**
1. Reduce population size
2. Reduce generations
3. Enable checkpointing
4. Clear cache periodically

```python
# Memory-efficient configuration
config = MAKERHybridConfig(
    population_size=15,
    evolution_generations=20,
    mcts_simulations=50
)

# Enable checkpointing
from maker_engine import FileCheckpointStore
checkpoint_store = FileCheckpointStore("maker_state.json")
```

#### Issue 5: Import Errors

**Symptoms:**
- Module not found errors
- Import failures

**Possible Causes:**
- Missing dependencies
- Incorrect Python path
- Module not installed

**Solutions:**
1. Install dependencies: `pip install -r requirements.txt`
2. Check Python path
3. Verify module location
4. Reinstall packages

```bash
# Check installation
python -c "import hybrid_maker_integration; print('OK')"

# Reinstall if needed
pip install --upgrade -e .
```

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run with debug logging
result = await run_maker_hybrid(theorem, mode=mode)
```

### Getting Help

If issues persist:

1. Check capabilities:
   ```python
   caps = get_maker_hybrid_capabilities()
   print(caps)
   ```

2. Review logs for errors

3. Try simpler configuration

4. Consult documentation:
   - Architecture: `HYBRID_MAKER_ARCHITECTURE.md`
   - API: `HYBRID_MAKER_API.md`
   - Examples: `HYBRID_MAKER_EXAMPLES.md`

---

## FAQ

### General Questions

**Q: What is the MAKER framework?**
A: MAKER (Multi-Agent Voting with Escalation and Red-flagging) is a framework that uses first-to-ahead-by-k voting to achieve statistical zero-error guarantees in multi-agent systems.

**Q: What is arXiv:2511.09030?**
A: This is the paper "Solving a Million-Step LLM Task with Zero Errors" which introduces the MAKER framework.

**Q: How does first-to-ahead-by-k voting work?**
A: It requires a candidate to receive k more votes than any other candidate. For N = 2k - 1 votes with probability p > 0.5 of being correct, the probability of selecting the correct winner approaches 1 as N increases.

**Q: What are red flags?**
A: Red flags are quality control filters that reject low-quality candidates based on token limits, schema validation, confidence thresholds, and blocked patterns.

### Strategy Questions

**Q: Which strategy should I use?**
A: Start with MCTS_THEN_MAKER for simple problems. Use ADAPTIVE_MAKER if you're unsure. Use FULL_MAKER_HYBRID for critical applications.

**Q: Can I combine strategies?**
A: Yes! FULL_MAKER_HYBRID combines all strategies. You can also create custom combinations.

**Q: How do I know if a strategy is working?**
A: Monitor result.success, best_fitness, and convergence_history. If fitness is not improving, try a different strategy.

### Configuration Questions

**Q: What is the optimal voting threshold (k)?**
A: For most cases, k=3 is a good balance. Use k=2 for speed, k=4-5 for quality, k=6-8 for maximum reliability.

**Q: How many generations should I use?**
A: Start with 20 generations. Increase if convergence is slow, decrease if resources are limited.

**Q: What population size should I use?**
A: 20 is a good default. Use 10-15 for small problems, 25-30 for complex problems, 40-50 for maximum diversity.

### Performance Questions

**Q: Why is execution slow?**
A: MAKER requires multiple votes per step. Reduce k, enable caching, use parallel mode, or reduce generations/simulations.

**Q: How can I speed up execution?**
A: 1) Lower voting threshold, 2) Use MCTS_THEN_MAKER (fastest), 3) Enable caching, 4) Use parallel mode, 5) Reduce resources.

**Q: What is the time complexity?**
A: Depends on strategy. MCTS_THEN_MAKER is O(N × C × k), MAKER_THEN_EVOLUTION is O(P × G × k). See Architecture document for details.

### Technical Questions

**Q: Does MAKER guarantee zero errors?**
A: MAKER provides statistical zero-error guarantees. The probability of error approaches 0 as the number of votes increases.

**Q: Can MAKER handle multi-objective optimization?**
A: Yes, by defining fitness functions that combine multiple objectives.

**Q: How does MAKER handle constraints?**
A: Through red flagging (reject invalid candidates) and fitness function design (penalize constraint violations).

**Q: Can I use custom fitness functions?**
A: Yes! Provide a custom evaluator function that takes a program/genome and returns a fitness score.

**Q: How do I integrate with my existing code?**
A: Use the API functions (run_maker_hybrid, run_maker_evolution) which return standard result objects.

### Integration Questions

**Q: Can I use MAKER with my own LLM?**
A: Yes, configure the team with your ModelConfig instances.

**Q: Does MAKER work with other languages?**
A: The current implementation is Python. The concepts can be applied to other languages.

**Q: Can I use MAKER for non-theorem-proving tasks?**
A: Yes! MAKER is general-purpose. Examples include code generation, planning, optimization, and more.

---

## Migration Guide

### From Basic Evolution

**Before:**
```python
from evolution import run_evolution

result = run_evolution(
    initial_program="proof1",
    evaluator=fitness_fn,
    generations=50
)
```

**After:**
```python
from evolution_maker_integration import run_maker_evolution

result = run_maker_evolution(
    initial_program="proof1",
    evaluator=fitness_fn,
    max_generations=50,
    config=MakerevolutionConfig(
        mode=MakerevolutionMode.HYBRID,
        voting_threshold=3
    )
)
```

### From MCTS

**Before:**
```python
from leanaide_mcts import run_mcts_search

result = run_mcts_search(theorem, simulations=100)
```

**After:**
```python
from hybrid_maker_integration import run_maker_hybrid, MAKERHybridMode

result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.MCTS_THEN_MAKER,
    config=MAKERHybridConfig(mcts_simulations=100)
)
```

---

## Best Practices

### 1. Start Simple

Begin with MCTS_THEN_MAKER and basic configuration:

```python
config = MAKERHybridConfig(
    voting_threshold=3,
    mcts_simulations=100
)
```

### 2. Monitor Progress

Track convergence and metrics:

```python
result = await run_maker_hybrid(theorem, mode=mode)

print(f"Success: {result.success}")
print(f"Fitness: {result.best_fitness:.3f}")
print(f"Generations: {result.generations_completed}")

if result.convergence_history:
    print("Convergence:")
    for i, fit in enumerate(result.convergence_history[::10]):
        print(f"  Gen {i*10}: {fit:.3f}")
```

### 3. Handle Errors Gracefully

Always check results and handle failures:

```python
result = await run_maker_hybrid(theorem, mode=mode)

if not result.success:
    logger.error("Failed to find solution")
    for error in result.failed_attempts:
        logger.error(f"Error: {error}")

    # Try fallback
    result = await run_maker_hybrid(
        theorem,
        mode=MAKERHybridMode.MCTS_THEN_MAKER
    )
```

### 4. Use Checkpointing

For long-running tasks:

```python
from maker_engine import FileCheckpointStore

checkpoint_store = FileCheckpointStore("maker_state.json")

result = engine.solve(
    initial_state,
    step_builder,
    apply_action,
    checkpoint_store=checkpoint_store
)
```

### 5. Tune Incrementally

Start with defaults, adjust one parameter at a time:

```python
# Start with defaults
config = MAKERHybridConfig()

# Tune voting threshold
config.voting_threshold = 4
# Test...

# Then tune simulations
config.mcts_simulations = 150
# Test...

# Continue tuning...
```

---

## Advanced Topics

### Custom Fitness Functions

Define domain-specific evaluation:

```python
def custom_fitness(program: str) -> float:
    """Domain-specific fitness function"""
    score = 0.0

    # Check for required patterns
    if "induction n" in program:
        score += 0.4
    if "simp" in program:
        score += 0.3
    if "refl" in program:
        score += 0.2

    # Penalize issues
    if "sorry" in program:
        score -= 0.5
    if len(program) > 1000:
        score -= 0.2

    return max(0.0, min(1.0, score))
```

### Custom Red Flag Rules

Define custom quality filters:

```python
from mdap_engine import RedFlagRules

custom_rules = RedFlagRules(
    max_tokens=500,
    max_characters=4000,
    blocked_patterns=["unsafe", "eval", "exec"],
    min_confidence=0.3,
    require_schema_match=True
)

config = MakerConfig(red_flag_rules=custom_rules)
```

### Parallel Execution

Run multiple theorems in parallel:

```python
import asyncio

async def solve_multiple(theorems):
    tasks = []
    for theorem in theorems:
        task = run_maker_hybrid(
            theorem=theorem,
            mode=MAKERHybridMode.MCTS_THEN_MAKER
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results

theorems = [
    "forall n : nat, n + 0 = n",
    "forall n m : nat, n + m = m + n",
    "forall a b c : nat, a + (b + c) = (a + b) + c"
]

results = asyncio.run(solve_multiple(theorems))
```

### Distributed Execution

For very large problems, consider distributed MAKER:

```python
# Pseudocode for distributed execution
# Use Redis for shared state
# Use Celery for task distribution
# Use PostgreSQL for result storage

from celery import Celery

app = Celery('maker_tasks')

@app.task
def distributed_vote(candidates, agent_id):
    """Distributed vote collection"""
    # Implement voting logic
    pass

@app.task
def distributed_solve(problem):
    """Distributed MAKER solving"""
    # Decompose and distribute
    pass
```

---

**End of User Guide**

For more information, see:
- Architecture: `HYBRID_MAKER_ARCHITECTURE.md`
- API Reference: `HYBRID_MAKER_API.md`
- Examples: `HYBRID_MAKER_EXAMPLES.md`
- Integration: `HYBRID_MAKER_INTEGRATION.md`
