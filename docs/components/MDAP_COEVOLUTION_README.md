# MDAP/MAKER Coevolving Decision Trees Integration

## Overview

This module integrates **MDAP** (Multi-Agent voting) and **MAKER** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) with **coevolving decision trees** for robust theorem proving with zero-error guarantees.

## Core Concept

Coevolve decision trees where each tree is evaluated by multiple agents, and MAKER voting determines the best candidates for evolution. This provides:

- **Multi-agent evaluation**: Each tree evaluated by multiple agents for robustness
- **MAKER voting**: First-to-ahead-by-k selection for consensus-driven evolution
- **Decomposition**: Trees can decompose complex problems into subtasks
- **Competitive coevolution**: Solvers and problems coevolve
- **Multi-objective optimization**: Pareto optimization across multiple criteria

## File Structure

```
mcts_coevolution_mdap.py         (1854 lines) - Main implementation
mdap_coevolution_examples.py     (400+ lines)  - Usage examples
MDAP_COEVOLUTION_README.md       (This file)   - Documentation
```

## Key Components

### 1. MDAP-Enhanced Decision Tree (`MDAPProofDecisionTree`)

Extends `ProofDecisionTree` with multi-agent evaluation capabilities:

```python
from mcts_coevolution_mdap import MDAPProofDecisionTree

tree = MDAPProofDecisionTree(
    root=root_node,
    num_agents=5,
    voting_strategy="first_k_ahead",
    k_ahead=3
)

# Tree maintains agent evaluations and consensus
tree.agent_evaluations      # Dict of agent_id -> AgentEvaluation
tree.consensus_score        # Consensus across agents
tree.agreement_level        # Agreement among agents
tree.agent_reliability      # Reliability scores for agents
```

**Key Methods:**
- `compute_consensus(evaluations)`: Compute weighted consensus score
- `compute_agreement(evaluations)`: Compute agreement level (1 - std_dev)
- `should_decompose(context)`: Decide if problem should be decomposed

### 2. Multi-Agent Tree Evaluator (`MDAPTreeEvaluator`)

Evaluates trees using multiple agents with Monte Carlo simulation:

```python
from mcts_coevolution_mdap import MDAPTreeEvaluator

evaluator = MDAPTreeEvaluator(
    num_agents=5,
    simulations=100,
    max_depth=50
)

evaluation = await evaluator.evaluate_tree_mdap(
    tree=mdap_tree,
    test_theorems=theorems,
    agent_configs=None  # Optional: list of ModelConfig
)

# Results include:
evaluation.consensus_score     # Overall consensus
evaluation.agreement_level     # Agent agreement
evaluation.agent_results       # Individual agent evaluations
evaluation.std_dev_success     # Variance in success rates
```

### 3. MAKER Voting for Trees (`TreeMAKERVoting`)

Implements first-to-ahead-by-k voting for tree selection:

```python
from mcts_coevolution_mdap import TreeMAKERVoting

voting = TreeMAKERVoting(
    k_ahead=3,
    voting_strategy="first_k_ahead"
)

# Select best trees using MAKER voting
selected_trees = voting.vote_on_best_trees(
    trees=population,
    evaluations=evaluations,
    count=10  # Select top 10
)

# Select parents for crossover
parents = voting.vote_on_crossover_parents(
    trees=population,
    evaluations=evaluations,
    num_parents=25,
    tournament_size=5
)
```

### 4. MDAP Tree Coevolution (`MDAPTreeCoevolution`)

Main coevolution engine with MDAP evaluation:

```python
from mcts_coevolution_mdap import MDAPTreeCoevolution

coevolution = MDAPTreeCoevolution(
    population_size=100,
    generations=50,
    crossover_rate=0.9,
    mutation_rate=0.1,
    elitism=5,
    max_depth=17,
    simulations=100,
    num_agents=5,
    k_ahead=3,
    voting_strategy="first_k_ahead"
)

# Run coevolution
best_tree = await coevolution.coevolve_mdap(
    test_theorems=theorems,
    leanaide_client=None  # Optional LeanAide client
)

# Access results
print(f"Best consensus: {best_tree.consensus_score:.4f}")
print(f"Agreement level: {best_tree.agreement_level:.4f}")
```

### 5. Decomposition-Enhanced Coevolution

Trees can decompose complex problems:

```python
from mcts_coevolution_mdap import DecompositionTreeCoevolution

decomp_coevolution = DecompositionTreeCoevolution(
    mdap_coevolution=base_coevolution,
    max_decomposition_depth=3,
    decomposition_threshold=0.7
)

best_tree = await decomp_coevolution.coevolve_with_decomposition(
    test_theorems=theorems
)
```

### 6. Competitive Coevolution

Solvers and problems coevolve:

```python
from mcts_coevolution_mdap import MDAPCompetitiveCoevolution

competitive = MDAPCompetitiveCoevolution(
    solver_pop_size=50,
    problem_pop_size=20,
    generations=100,
    num_agents=5,
    k_ahead=3
)

best_solver = await competitive.competitive_coevolve_mdap(
    initial_theorems=theorems
)
```

### 7. Multi-Objective Coevolution

Pareto optimization across multiple objectives:

```python
from mcts_coevolution_mdap import MDAPMultiObjectiveCoevolution

multi_obj = MDAPMultiObjectiveCoevolution(
    objectives=["success", "elegance", "simplicity"],
    population_size=100,
    generations=50,
    num_agents=5
)

pareto_front = await multi_obj.coevolve_multi_objective_mdap(
    test_theorems=theorems
)

# Access Pareto-optimal solutions
for tree in pareto_front:
    obj_fit = tree.objective_fitness
    print(f"success: {obj_fit['success']:.3f}, "
          f"elegance: {obj_fit['elegance']:.3f}")
```

### 8. Ensemble Methods

Combine multiple trees with MDAP voting:

```python
from mcts_coevolution_mdap import MDAPTreeEnsemble

ensemble = MDAPTreeEnsemble(
    trees=mdap_trees,
    voting_strategy="first_k_ahead",
    k_ahead=3
)

# Majority vote with consensus
result = await ensemble.majority_vote_mdap(context)

# Weighted vote by agent reliability
result = await ensemble.weighted_vote_mdap(context, weights=custom_weights)

# Cascade execution
result = await ensemble.cascade_mdap(context)
```

### 9. Performance Monitoring

Track coevolution progress:

```python
from mcts_coevolution_mdap import MDAPCoevolutionMonitor

monitor = MDAPCoevolutionMonitor()

# Track each generation
for gen in range(generations):
    # ... evaluate population ...
    monitor.track_generation(gen, population, evaluations)

# Generate reports
report = monitor.get_agent_reliability_report()
for agent_id, metrics in report.items():
    print(f"{agent_id}: avg={metrics['avg_score']:.3f}")

# Plot progress (requires matplotlib)
monitor.plot_progress()
```

## Usage Examples

### Quick Start

```python
import asyncio
from mcts_coevolution_mdap import run_mdap_coevolution_pipeline

async def main():
    theorems = [
        "∀ n: Nat, n + 0 = n",
        "∀ a b: Nat, a + b = b + a"
    ]

    best_tree = await run_mdap_coevolution_pipeline(theorems)

    print(f"Best consensus: {best_tree.consensus_score:.4f}")

asyncio.run(main())
```

### Custom Configuration

```python
from mcts_coevolution_mdap import (
    MDAPTreeCoevolution,
    create_mdap_config
)

# Create custom configuration
config = create_mdap_config(
    num_agents=7,
    k_ahead=5,
    voting_strategy="first_k_ahead"
)

# Create coevolution with custom settings
coevolution = MDAPTreeCoevolution(
    population_size=50,
    generations=30,
    num_agents=config["num_agents"],
    k_ahead=config["k_ahead"]
)

best_tree = await coevolution.coevolve_mdap(theorems)
```

### Run Examples

```bash
# Run all examples
python mdap_coevolution_examples.py

# Run specific example
python mdap_coevolution_examples.py
# Then select example number
```

## Voting Strategies

### First-to-Ahead-by-K (MAKER)

The MAKER voting mechanism from the paper "Solving a Million-Step LLM Task with Zero Errors":

```
Winner selected when: votes[winner] >= k + max(votes[other])
```

This ensures strong consensus before selection.

**Example:** With k=3, a candidate needs 3 more votes than the second-best to win.

### Simple First-to-K

```
Winner selected when: votes[winner] >= k
```

Less conservative, faster convergence.

## Agent Reliability

Agents develop reliability scores over time:

```python
# Tree maintains agent reliability
tree.agent_reliability = {
    "agent_0": 0.95,  # Highly reliable
    "agent_1": 0.82,  # Moderately reliable
    "agent_2": 0.76   # Less reliable
}

# Reliability updates based on prediction accuracy
tree.update_agent_reliability(
    agent_id="agent_0",
    predicted_performance=0.85,
    actual_performance=0.87
)
```

Reliability affects:
- Weight in consensus computation
- Weight in voting decisions
- Influence on tree selection

## Decomposition

Trees can decompose complex problems:

```python
# Tree decides whether to decompose
if tree.should_decompose(context):
    # Decompose into subtasks
    subtask1 = "Prove lemma for theorem"
    subtask2 = "Complete main proof using lemma"

    # Solve subtasks separately
    result1 = await solve_subtask(tree, subtask1)
    result2 = await solve_subtask(tree, subtask2)

    # Compose results
    final_result = compose(result1, result2)
```

Decomposition triggers when:
- Problem complexity is high (long theorem statement)
- Consensus score is below threshold
- Agent agreement is low

## Performance Metrics

### Tree-Level Metrics

- `consensus_score`: Weighted average of agent evaluations
- `agreement_level`: 1 - std_dev of agent success rates
- `fitness`: Overall fitness combining multiple factors

### Agent-Level Metrics

- `success_rate`: Proportion of successful proofs
- `avg_depth`: Average proof depth
- `avg_time`: Average proof time
- `elegance_score`: Proof elegance (0-1)
- `simplicity_score`: Proof simplicity (0-1)
- `robustness`: Consistency across simulations

### Population-Level Metrics

- Average consensus
- Average agreement
- Population diversity (std dev of fitness)
- Pareto front size (multi-objective)

## Integration with LeanAide

Optional LeanAide integration for formal verification:

```python
from leanaide_client import LeanAideClient

client = LeanAideClient()

# Coevolution with verification bonus
best_tree = await coevolution.coevolve_mdap(
    test_theorems=theorems,
    leanaide_client=client  # Adds verification bonus
)
```

Verification bonus applied to top-performing trees:
- Verified proofs receive fitness bonus
- Guides evolution toward formally verifiable solutions

## Configuration Reference

### MDAPTreeCoevolution Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | 100 | Number of trees in population |
| `generations` | 50 | Number of evolution generations |
| `crossover_rate` | 0.9 | Probability of crossover |
| `mutation_rate` | 0.1 | Probability of mutation |
| `elitism` | 5 | Number of elite individuals preserved |
| `max_depth` | 17 | Maximum tree depth |
| `simulations` | 100 | Monte Carlo simulations per evaluation |
| `num_agents` | 5 | Number of MDAP agents |
| `k_ahead` | 3 | K parameter for first-to-ahead-by-k |
| `voting_strategy` | "first_k_ahead" | Voting strategy to use |

### Voting Strategies

- `"first_k_ahead"`: MAKER first-to-ahead-by-k (recommended)
- `"first_to_k"`: Simple first-to-k
- `"majority"`: Simple majority
- `"weighted"`: Weighted by agent reliability

## Algorithm Flow

```
1. Initialize MDAP tree population
   └─ Convert base trees to MDAPProofDecisionTree

2. For each generation:
   ├─ Multi-agent evaluation
   │   ├─ Each agent evaluates each tree
   │   ├─ Compute consensus scores
   │   └─ Compute agreement levels
   │
   ├─ Optional: LeanAide verification
   │   └─ Apply verification bonus to top trees
   │
   ├─ Parent selection (MAKER voting)
   │   └─ Select trees using first-to-ahead-by-k
   │
   ├─ Create next generation
   │   ├─ Elitism (preserve best trees)
   │   ├─ Crossover (subtree exchange)
   │   └─ Mutation (subtree/point/etc.)
   │
   └─ Track metrics
       └─ Monitor consensus, agreement, diversity

3. Return best tree
   └─ Highest consensus score across all generations
```

## Key Features Summary

1. **Multi-Agent Tree Evaluation**: Each tree evaluated by multiple agents
2. **MAKER Voting**: First-to-ahead-by-k selection for robust consensus
3. **Agent Reliability Tracking**: Dynamic reliability scores per agent
4. **Decomposition**: Automatic problem decomposition for complex theorems
5. **Competitive Coevolution**: Solvers and problems coevolve
6. **Multi-Objective**: Pareto optimization with NSGA-II
7. **Ensemble Methods**: Combine multiple trees with consensus voting
8. **LeanAide Integration**: Optional formal verification
9. **Performance Monitoring**: Comprehensive tracking and visualization
10. **Zero-Error Guarantees**: Red-flagging from MAKER framework

## Theoretical Foundation

Based on two key frameworks:

### MDAP/MAKER
- **Paper**: "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)
- **Key contribution**: First-to-ahead-by-k voting with red-flagging
- **Benefit**: Zero-error guarantees through strong consensus requirements

### Coevolving Decision Trees
- **Framework**: Genetic programming with Monte Carlo evaluation
- **Key contribution**: Population-based search with stochastic evaluation
- **Benefit**: Efficient exploration of proof strategy space

**Integration**: Combines the robustness of MAKER voting with the exploration power of genetic programming.

## Advanced Usage

### Custom Agent Configurations

```python
from workflow_structures import ModelConfig, Team

# Define custom agents
agent_configs = [
    ModelConfig(
        model_id="gpt-4",
        api_key="...",
        api_base="https://api.openai.com/v1",
        temperature=0.0
    )
    for _ in range(5)
]

# Use custom agents in evaluation
evaluation = await evaluator.evaluate_tree_mdap(
    tree=tree,
    test_theorems=theorems,
    agent_configs=agent_configs
)
```

### Custom Fitness Functions

```python
def custom_fitness(tree: MDAPProofDecisionTree) -> float:
    """Custom fitness combining multiple metrics"""
    return (
        0.4 * tree.consensus_score +
        0.3 * tree.agreement_level +
        0.2 * (1.0 / (1.0 + tree.depth)) +  # Prefer shallow trees
        0.1 * tree.get_agent_reliability("agent_0")  # Trust primary agent
    )

# Apply during coevolution
for tree in population:
    tree.fitness = custom_fitness(tree)
```

### Parallel Evaluation

```python
from concurrent.futures import ThreadPoolExecutor

async def parallel_evaluation(population, theorems):
    """Evaluate population in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                evaluator.evaluate_tree_mdap,
                tree,
                theorems
            )
            for tree in population
        ]
        return await asyncio.gather(*tasks)
```

## Troubleshooting

### Low Consensus Scores

**Problem**: Trees have low consensus scores (< 0.5)

**Solutions**:
- Increase `num_agents` for more robust evaluation
- Increase `simulations` for more accurate evaluation
- Check if test theorems are too difficult
- Reduce `max_depth` if trees are too complex

### Low Agreement Levels

**Problem**: Agents disagree strongly (agreement < 0.6)

**Solutions**:
- Increase `agent_diversity` parameter
- Use different random seeds per agent
- Check if problem is underspecified
- Consider decomposition for complex problems

### Slow Convergence

**Problem**: Population doesn't improve over generations

**Solutions**:
- Increase `k_ahead` for stronger selection pressure
- Adjust `crossover_rate` and `mutation_rate`
- Use tournament selection instead of MAKER voting
- Check population diversity (may need to restart)

## References

1. **MAKER Paper**: "Solving a Million-Step LLM Task with Zero Errors"
   - arXiv:2511.09030
   - Introduces first-to-ahead-by-k voting and red-flagging

2. **Genetic Programming**: Koza, J.R. (1992)
   - Foundation for tree-based genetic programming

3. **NSGA-II**: Deb, K. et al. (2002)
   - Multi-objective optimization with Pareto fronts

4. **Lean Theorem Prover**: https://leanprover.github.io/
   - Formal verification backend

## License

This module is part of the OpenEvolve project.

## Contributing

To extend this module:

1. Add new voting strategies in `TreeMAKERVoting`
2. Implement new decomposition methods in `DecompositionTreeCoevolution`
3. Add custom mutation operators in `TreeMutation`
4. Extend monitoring in `MDAPCoevolutionMonitor`

## Contact

For questions or issues, please refer to the main OpenEvolve documentation.
