# Adversarial Unified Framework - Quick Reference

## Quick Start

```python
import asyncio
from adversarial_unified import (
    AdversarialEngine,
    AdversarialPresets,
    MCTSApproach
)

async def main():
    # Create engine
    engine = AdversarialEngine(AdversarialPresets.balanced())

    # Run adversarial test
    result = await engine.adversarial_test(
        theorem="theorem example (n : Nat) : n + 0 = n := by",
        mcts_approach=MCTSApproach.EVOLVED_POLICIES
    )

    print(f"Robustness: {result.robustness_score:.2%}")
    print(f"Is Robust: {result.is_robust}")

asyncio.run(main())
```

## Configuration Presets

| Preset | Red | Blue | Generations | Use Case |
|--------|-----|------|-------------|----------|
| `fast()` | 2 | 3 | 3 | Quick testing |
| `balanced()` | 3 | 5 | 10 | General use |
| `thorough()` | 5 | 7 | 20 | Comprehensive |
| `self_play()` | 1 | 1 | 5 | Self-play training |

## Attack Strategies

```python
from adversarial_unified import AttackStrategy

# Available attacks
AttackStrategy.EDGES          # Edge case testing
AttackStrategy.ASSUMPTIONS    # Challenge assumptions
AttackStrategy.TACTICS        # Weak tactic applications
AttackStrategy.BOUNDARIES     # Test boundaries
AttackStrategy.LOGIC_GAPS     # Find logic gaps
AttackStrategy.COMPLEXITY     # Stress test
AttackStrategy.DECOMPOSITION  # Attack decomposition
AttackStrategy.CONSENSUS      # Attack consensus
```

## Defense Strategies

```python
from adversarial_unified import DefenseStrategyType

# Available defenses
DefenseStrategyType.REINFORCE  # Reinforce weak points
DefenseStrategyType.DIVERSIFY  # Add diverse paths
DefenseStrategyType.VERIFY     # Add verification
DefenseStrategyType.DECOMPOSE  # Further decompose
DefenseStrategyType.CONSENSUS  # Use MDAP consensus
DefenseStrategyType.FORMAL    # Formal verification
```

## MCTS Approaches

```python
from adversarial_unified import MCTSApproach

# Available approaches
MCTSApproach.EVOLVED_POLICIES    # Evolve rollout policies
MCTSApproach.EVOLUTIONARY_NODES  # Evolve MCTS nodes
MCTSApproach.COEVOLUTION         # Coevolve trees
MCTSApproach.ADAPTIVE            # Adaptive selection
MCTSApproach.COMBINED            # Combine all approaches
```

## Core Methods

### AdversarialEngine

```python
# Single adversarial test
result = await engine.adversarial_test(
    theorem="theorem ...",
    mcts_approach=MCTSApproach.EVOLVED_POLICIES
)

# Adversarial training
training_result = await engine.adversarial_training(
    theorem_corpus=["theorem1", "theorem2", ...],
    epochs=10
)

# Coevolution training
coevolution_result = await engine.coevolution_training(
    initial_theorems=["theorem1", "theorem2", ...]
)
```

### Result Objects

```python
# AdversarialTestResult
result.theorem               # Theorem statement
result.proof_generated       # Was proof generated?
result.best_proof           # Best proof found
result.robustness_score     # 0.0 to 1.0
result.is_robust            # Passes threshold?
result.total_attacks        # Total attacks
result.attacks_blocked      # Attacks defended
result.vulnerabilities_found # Successful attacks
result.fixes_applied        # Defenses applied

# AdversarialTrainingResult
training_result.final_success_rate  # Proof generation rate
training_result.final_robustness    # Final robustness
training_result.best_epoch          # Best epoch
training_result.convergence_curve   # Robustness over time

# CoevolutionResult
coevolution_result.generations_completed    # Total generations
coevolution_result.final_red_fitness        # Red team fitness
coevolution_result.final_blue_fitness       # Blue team fitness
coevolution_result.convergence_generation   # When converged
```

## Custom Configuration

```python
from adversarial_unified import AdversarialConfig

config = AdversarialConfig(
    red_team_size=5,
    blue_team_size=7,
    coevolution_generations=15,
    attack_strategies=[
        AttackStrategy.EDGES,
        AttackStrategy.ASSUMPTIONS,
        AttackStrategy.TACTICS
    ],
    defense_approaches=[
        MCTSApproach.EVOLVED_POLICIES,
        MCTSApproach.COEVOLUTION
    ],
    enable_mdap=True,
    num_mdap_agents=7,
    maker_voting_strategy="first_k_ahead",
    k_ahead=3,
    robustness_threshold=0.85,
    leanaide_enabled=True,
    enable_caching=True,
    enable_monitoring=True
)

engine = AdversarialEngine(config)
```

## Workflow Integration

```python
from adversarial_unified import AdversarialWorkflowIntegrator
from workflow_structures import SubProblem

# Create integrator
integrator = AdversarialWorkflowIntegrator(config)

# Create subproblem
subproblem = SubProblem(
    subproblem_id="sub_1",
    statement="theorem example (n : Nat) : n + 0 = n := by",
    dependencies=[],
    priority=1
)

# Solve with adversarial validation
solution = await integrator.solve_with_adversarial_validation(subproblem)

# Access results
robustness = solution.quality_metrics['robustness']
is_robust = solution.quality_metrics['is_robust']
attacks_blocked = solution.quality_metrics['attacks_blocked']
```

## Command-Line Usage

```bash
# Basic test
python adversarial_unified.py \
    "theorem example (n : Nat) : n + 0 = n := by" \
    --preset balanced \
    --approach evolved_policies

# With output
python adversarial_unified.py \
    "theorem example (n : Nat) : n + 0 = n := by" \
    --preset thorough \
    --output results.json

# Training
python adversarial_unified.py \
    "theorem example (n : Nat) : n + 0 = n := by" \
    --epochs 10 \
    --output training.json
```

## Result Summary

```python
from adversarial_unified import print_result_summary

# Print formatted summary
print_result_summary(result)

# Output:
# ============================================================
# Adversarial Testing Results
# ============================================================
# Theorem:      theorem example (n : Nat) : n + 0 = n := by
# Proof Generated: True
# Robustness:  85.50%
# Is Robust:   True
# MCTS Approach: evolved_policies
# Time:         45.23s
#
# Attack Summary:
#   Total Attacks:    8
#   Attacks Blocked:  6
#   Vulnerabilities:  2
#   Fixes Applied:    5
# ============================================================
```

## Utility Functions

```python
from adversarial_unified import (
    create_engine_from_preset,
    quick_adversarial_test,
    thorough_adversarial_test
)

# Create engine from preset
engine = create_engine_from_preset("balanced")

# Quick test
result = await quick_adversarial_test(
    theorem="theorem ...",
    approach="evolved_policies"
)

# Thorough test
result = await thorough_adversarial_test(
    theorem="theorem ...",
    approach="combined"
)
```

## Caching

```python
# Access cache stats
stats = engine.cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Cache sizes: {stats}")

# Clear cache
engine.cache.clear()
```

## Monitoring

```python
# Get monitoring summary
if engine.monitor:
    summary = engine.monitor.get_summary()
    print(f"Duration: {summary['duration_seconds']:.2f}s")
    print(f"Attack success rate: {summary['attack_success_rate']:.2%}")
    print(f"Defense success rate: {summary['defense_success_rate']:.2%}")
    print(f"Avg robustness: {summary['avg_robustness']:.2%}")
```

## Error Handling

```python
try:
    result = await engine.adversarial_test(theorem)
    if not result.proof_generated:
        print(f"Failed to generate proof: {result.metadata.get('error')}")
    elif not result.is_robust:
        print(f"Proof not robust: {result.robustness_score:.2%}")
        print("Weaknesses:")
        for weakness in result.metadata['robustness_report']['weaknesses']:
            print(f"  - {weakness}")
    else:
        print(f"Success! Robustness: {result.robustness_score:.2%}")
except Exception as e:
    print(f"Error: {e}")
```

## Best Practices

1. **Start with fast preset** for initial testing
2. **Use balanced preset** for development
3. **Use thorough preset** for production
4. **Enable caching** for repeated tests
5. **Enable monitoring** for long-running tasks
6. **Use self_play** for adversarial training
7. **Set appropriate thresholds** for your use case
8. **Review weakness reports** to improve proofs
9. **Use coevolution** for adaptive robustness
10. **Integrate LeanAide** for formal verification

## Common Patterns

### Pattern 1: Test Multiple Approaches

```python
approaches = [
    MCTSApproach.EVOLVED_POLICIES,
    MCTSApproach.EVOLUTIONARY_NODES,
    MCTSApproach.COEVOLUTION
]

results = {}
for approach in approaches:
    result = await engine.adversarial_test(theorem, approach)
    results[approach] = result

best = max(results.items(), key=lambda x: x[1].robustness_score)
print(f"Best approach: {best[0].value}")
```

### Pattern 2: Batch Testing

```python
theorems = [thm1, thm2, thm3, ...]

results = []
for theorem in theorems:
    result = await engine.adversarial_test(theorem)
    results.append(result)

avg_robustness = sum(r.robustness_score for r in results) / len(results)
print(f"Average robustness: {avg_robustness:.2%}")
```

### Pattern 3: Iterative Improvement

```python
robustness = 0.0
iteration = 0
theorem = "theorem ..."

while robustness < 0.9 and iteration < 5:
    result = await engine.adversarial_test(theorem)
    robustness = result.robustness_score

    if not result.is_robust:
        # Use feedback to improve theorem/proof
        theorem = improve_theorem(theorem, result)

    iteration += 1
    print(f"Iteration {iteration}: Robustness = {robustness:.2%}")
```

## Troubleshooting

### Issue: Low robustness scores

**Solutions:**
- Increase `coevolution_generations`
- Enable `ensemble_defense`
- Use `thorough` preset
- Add more attack strategies
- Enable LeanAide verification

### Issue: Slow performance

**Solutions:**
- Use `fast` preset
- Enable caching
- Reduce team sizes
- Decrease generations
- Disable monitoring

### Issue: Memory issues

**Solutions:**
- Reduce cache size
- Decrease team sizes
- Disable monitoring
- Process in batches

## Reference

**Full Documentation:** `ADVERSARIAL_UNIFIED_COMPLETE.md`
**Source Code:** `adversarial_unified.py` (2,151 lines)
**Dependencies:** `mdap_maker_mcts_unified.py`, `adversarial_maker_integration.py`
