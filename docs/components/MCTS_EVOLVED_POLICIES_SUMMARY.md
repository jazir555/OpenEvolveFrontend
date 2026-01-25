# MCTS Evolved Policies Implementation Summary

## Overview

This module implements **Example 1: Evolving better rollouts/policies for MCTS** from the LeanAide roadmap. It uses evolutionary algorithms to search for better "brains" (policies) to drive MCTS, rather than searching the game tree directly.

## Core Concept

Instead of using random rollouts or simple heuristics, we evolve sophisticated policies that:
- Learn which tactics work best in different contexts
- Adapt exploration based on proof depth
- Incorporate domain knowledge about mathematical proofs
- Improve over generations through natural selection

## Implementation Components

### 1. Policy Representation (`RolloutPolicyGenome`)

A sophisticated genome encoding:
- **Tactic weights**: Base preferences for each Lean tactic (intros, simp, rw, etc.)
- **Context modifiers**: Adjustments based on proof state (has equality, has implication, etc.)
- **Depth preferences**: Tactic selection varies by proof depth
- **Exploration parameters**: Control exploration vs exploitation
- **Goal proximity heuristics**: Adjust strategy based on how close we are to the goal
- **Lemma affinity**: Preferences for specific lemmas
- **Domain preferences**: Domain-specific tactic knowledge

### 2. Executable Policy (`TacticRolloutPolicy`)

Implements multiple selection strategies:
- **Epsilon-greedy**: Mostly best tactic, occasionally random exploration
- **Softmax (Boltzmann)**: Probabilistic selection based on scores
- **UCB (Upper Confidence Bound)**: Balance exploration/exploitation

Scoring factors:
```
score = base_weight + preference + context_modifier + depth_modifier
        + exploration_bonus + goal_proximity_modifier
```

### 3. Population Management (`PolicyPopulation`)

Evolutionary operations:
- **Tournament selection**: Best from random subset
- **Roulette wheel**: Fitness-proportionate selection
- **Rank selection**: Probability based on rank, not absolute fitness
- **Crossover**: Blend weights from two parents
- **Mutation**: Gaussian perturbations to parameters
- **Elitism**: Preserve best policies unchanged

### 4. Policy Evaluation (`PolicyEvaluator`)

Measures policy quality by:
1. Running MCTS with the policy on test theorems
2. Computing metrics:
   - Success rate (primary)
   - Average proof depth
   - Time to solution
   - Nodes explored
3. Combining into fitness score:
   ```
   fitness = success_rate * 10 + speed_bonus + efficiency_bonus - depth_penalty
   ```

### 5. Evolution Engine (`PolicyEvolutionEngine`)

Main evolutionary loop:
```
for generation in generations:
    1. Evaluate all policies
    2. Select best performers
    3. Create offspring via crossover/mutation
    4. Track best policy
    5. Save generation data
```

### 6. Enhanced MCTS (`EvolvedPolicyMCTS`)

MCTS that uses evolved policies for rollouts:
- Standard MCTS structure (Selection → Expansion → Simulation → Backpropagation)
- **Key difference**: Simulation phase uses evolved policy instead of random tactics
- Results in more intelligent search guided by learned preferences

### 7. Advanced Features

#### Adaptive Policy MCTS
- Adapts policy during search
- Every N iterations:
  - Analyzes tactic performance
  - Updates policy weights
  - Continues with improved policy

#### Co-Evolving MCTS
- Alternates policy evolution and search
- Phase 1: Evolve initial policies
- Phase 2: Search with best policy
- Phase 3: Evolve further based on search results
- Repeat for N phases

#### LeanAide-Guided Evolution
- Uses LeanAide for formal verification
- Bonus fitness for policies that produce verifiably correct proofs
- Penalizes policies that generate invalid Lean code

#### Multi-Objective Evolution (NSGA-II)
- Optimizes multiple objectives simultaneously:
  - Success rate
  - Speed
  - Elegance (short proofs)
  - Generality
- Returns Pareto front of non-dominated policies

#### Policy Transfer Learning
- Transfers policies between domains
- Fine-tunes with:
  - Small adaptation noise
  - Domain-specific mutations
  - Target domain training

## Key Innovations

1. **Evolution is searching for better brains**: Not searching proof space directly, but learning to search better

2. **Context-sensitive decision making**: Policies learn that "intros" is good early, "linarith" for equalities, etc.

3. **Adaptive exploration**: Exploration bonus decays with usage, automatically balancing explore/exploit

4. **Multi-scale learning**: Genome encodes preferences at tactic, context, and depth levels

5. **Transfer learning**: Reuse learned policies across different mathematical domains

## Usage Example

```python
# 1. Evolve policies on training theorems
best_policy = await evolve_mcts_rollout_policy(
    test_theorems=[
        "forall (a b : Nat), a + b = b + a",
        "forall (a b c : Nat), (a + b) + c = a + (b + c)",
        # ... more theorems
    ],
    generations=20,
    population_size=50,
    mcts_iterations=100
)

# 2. Use evolved policy for new theorems
result = await search_with_evolved_policy(
    theorem="forall (a b : Nat), a * b = b * a",
    policy=best_policy,
    max_iterations=1000
)

# 3. Transfer to new domain
transfer = PolicyTransfer(best_policy)
adapted_policy = transfer.transfer_policy("real_analysis")
fine_tuned = await transfer.fine_tune(adapted_policy, real_analysis_theorems)
```

## Benefits Compared to Baseline MCTS

1. **Better rollouts**: Intelligent tactics instead of random
2. **Faster convergence**: Fewer iterations needed
3. **Domain adaptation**: Policies learn domain-specific patterns
4. **Continuous improvement**: Evolution keeps finding better strategies
5. **Knowledge transfer**: Reuse learned policies across problems

## Integration Points

- **leanaide_mcts.py**: Uses base MCTS infrastructure
- **leanaide_evolution.py**: Integrates genetic operators
- **leanaide_client.py**: Optional Lean verification

## Future Enhancements

1. **Neural network policies**: Replace genome with learned network
2. **Meta-learning**: Learn how to evolve policies
3. **Hierarchical policies**: Different policies for different proof stages
4. **Ensemble policies**: Combine multiple policies
5. **Curriculum learning**: Start easy, progress to hard theorems

## Files Created

- `mcts_evolved_policies.py` (~2000 lines)
  - Complete implementation of all components
  - Production-ready with error handling
  - Comprehensive logging and statistics
  - Example usage in `main()` function
