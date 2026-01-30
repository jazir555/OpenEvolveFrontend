# LeanAide MCTS Quick Reference

## Table of Contents

1. [Quick Start](#quick-start)
2. [Common Patterns](#common-patterns)
3. [Configuration Examples](#configuration-examples)
4. [API Quick Reference](#api-quick-reference)
5. [Performance Tips](#performance-tips)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
# MCTS is included in LeanAide
pip install leanaide
```

### Basic Usage

```python
from leanaide_mcts import LeanProofMCTS, ProofContext

# Create proof context
context = ProofContext(
    goal="∀ n : Nat, n + 0 = n",
    hypotheses=[],
    available_lemmas=["Nat.add_zero"]
)

# Run MCTS
mcts = LeanProofMCTS(simulations=1000)
best_sequence, root = mcts.search(context)

# View results
for action in best_sequence:
    print(action.tactic.name)
```

### Three-Line Usage

```python
# Minimum viable MCTS search
mcts = LeanProofMCTS()
proof, tree = mcts.search(ProofContext(goal="theorem"))
print([a.tactic.name for a in proof])
```

---

## Common Patterns

### Pattern 1: Simple Theorem (1-3 steps)

```python
mcts = LeanProofMCTS(
    simulations=100,
    exploration_constant=0.5,
    rollout_depth=3
)
```

### Pattern 2: Medium Theorem (4-7 steps)

```python
mcts = LeanProofMCTS(
    simulations=1000,
    exploration_constant=1.414,
    rollout_depth=7
)
```

### Pattern 3: Complex Theorem (8+ steps)

```python
mcts = LeanProofMCTS(
    simulations=5000,
    exploration_constant=2.0,
    rollout_depth=15,
    rollout_episodes=3
)
```

### Pattern 4: Unknown Domain

```python
mcts = LeanProofMCTS(
    simulations=2000,
    exploration_constant=3.0,
    dirichlet_alpha=0.5,
    dirichlet_epsilon=0.5
)
```

### Pattern 5: Proof Refinement

```python
mcts = LeanProofMCTS(
    simulations=2000,
    exploration_constant=1.0,  # Focused search
    rollout_depth=5,            # Local refinement
    temperature=0.5             # Mostly best actions
)
```

---

## Configuration Examples

### Fast Mode

```python
config = {
    "simulations": 100,
    "exploration_constant": 0.5,
    "rollout_depth": 3,
    "temperature": 0.0
}
```

### Balanced Mode (Default)

```python
config = {
    "simulations": 1000,
    "exploration_constant": 1.414,
    "rollout_depth": 7,
    "temperature": 1.0
}
```

### Thorough Mode

```python
config = {
    "simulations": 10000,
    "exploration_constant": 2.0,
    "rollout_depth": 15,
    "rollout_episodes": 3,
    "temperature": 0.5
}
```

### Exploratory Mode

```python
config = {
    "simulations": 5000,
    "exploration_constant": 3.0,
    "dirichlet_alpha": 0.5,
    "dirichlet_epsilon": 0.5,
    "temperature": 1.5
}
```

---

## API Quick Reference

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `LeanProofMCTS` | Main MCTS engine | `search()`, `get_statistics()` |
| `MCTS` | Core MCTS algorithm | `select()`, `expand()`, `simulate()`, `backpropagate()` |
| `MCTSNode` | Tree node | `get_ucb_score()`, `update()`, `add_child()` |
| `ProofContext` | Proof state | `to_dict()` |
| `Tactic` | Tactic metadata | `to_dict()` |
| `TacticAction` | Action in search | `to_dict()` |
| `MCTSResult` | Search result | `to_dict()` |

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exploration_constant` | float | 1.414 | UCB exploration (c) |
| `simulations` | int | 1000 | Number of iterations |
| `rollout_depth` | int | 5 | Max rollout depth |
| `rollout_episodes` | int | 1 | Rollouts per expansion |
| `temperature` | float | 1.0 | Action selection randomness |
| `dirichlet_alpha` | float | 0.3 | Dirichlet noise α |
| `dirichlet_epsilon` | float | 0.25 | Dirichlet noise ε |

---

## Performance Tips

### Speed Up Search

```python
# Reduce simulations
mcts = LeanProofMCTS(simulations=500)

# Shallow rollouts
mcts = LeanProofMCTS(rollout_depth=3)

# Lower temperature (faster convergence)
mcts = LeanProofMCTS(temperature=0.5)
```

### Improve Quality

```python
# Increase simulations
mcts = LeanProofMCTS(simulations=5000)

# Deeper rollouts
mcts = LeanProofMCTS(rollout_depth=15)

# Multiple rollout episodes
mcts = LeanProofMCTS(rollout_episodes=3)

# Higher exploration
mcts = LeanProofMCTS(exploration_constant=2.0)
```

### Reduce Memory

```python
# Enable transposition table
mcts = LeanProofMCTS(use_transposition_table=True)

# Reduce simulations
mcts = LeanProofMCTS(simulations=500)
```

---

## Troubleshooting

### Problem: Low-Quality Proofs

**Solutions**:
```python
# Increase simulations
mcts = LeanProofMCTS(simulations=5000)

# Adjust exploration
mcts = LeanProofMCTS(exploration_constant=1.414)

# Deeper rollouts
mcts = LeanProofMCTS(rollout_depth=10)
```

### Problem: Too Slow

**Solutions**:
```python
# Reduce simulations
mcts = LeanProofMCTS(simulations=500)

# Shallow rollouts
mcts = LeanProofMCTS(rollout_depth=3)
```

### Problem: Local Optima

**Solutions**:
```python
# Increase exploration
mcts = LeanProofMCTS(exploration_constant=2.0)

# Add Dirichlet noise
mcts = LeanProofMCTS(
    dirichlet_alpha=0.5,
    dirichlet_epsilon=0.5
)
```

---

## Best Practices

### DO ✅

```python
# 1. Start with defaults
mcts = LeanProofMCTS()

# 2. Adjust based on theorem complexity
if complexity == "simple":
    mcts = LeanProofMCTS(simulations=100)
elif complexity == "complex":
    mcts = LeanProofMCTS(simulations=5000)

# 3. Monitor progress
stats = mcts.get_statistics()
print(f"Average time: {stats['average_time']:.2f}s")
```

### DON'T ❌

```python
# 1. Don't use MCTS for trivial 1-step proofs
# Use direct search instead

# 2. Don't set exploration too high/low
# Bad:
mcts = LeanProofMCTS(exploration_constant=10.0)
# Good:
mcts = LeanProofMCTS(exploration_constant=1.414)

# 3. Don't use deep rollouts with many simulations
# Bad:
mcts = LeanProofMCTS(rollout_depth=20, simulations=10000)
# Good:
mcts = LeanProofMCTS(rollout_depth=10, simulations=1000)
```

---

## Command Line

### Run Tests

```bash
# All tests
python run_mcts_tests.py

# Specific category
python run_mcts_tests.py --category unit

# Performance benchmarks
python run_mcts_tests.py --benchmark
```

### Run Demos

```bash
# All demos
python demo_mcts.py

# Specific demo
python demo_mcts.py --demo basic
```

---

## Parameter Cheat Sheet

| Goal | Simulations | Exploration | Rollout Depth | Temperature |
|------|-------------|-------------|---------------|-------------|
| Fast | 100 | 0.5 | 3 | 0.0 |
| Default | 1000 | 1.414 | 7 | 1.0 |
| Thorough | 5000 | 2.0 | 15 | 0.5 |
| Explore | 2000 | 3.0 | 10 | 1.5 |
| Refine | 2000 | 1.0 | 5 | 0.5 |

---

*Last Updated: 2025-12-30*
*Version: 1.0.0*

For full documentation, see:
- `LEANAIDE_MCTS_GUIDE.md` - Comprehensive guide
- `LEANAIDE_MCTS_API.md` - Complete API reference
- `LEANAIDE_MCTS_EXAMPLES.md` - Usage examples
- `LEANAIDE_MCTS_ARCHITECTURE.md` - Architecture details
