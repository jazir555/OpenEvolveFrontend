# MDAP Evolutionary MCTS - Quick Reference Guide

## Quick Start

```python
from mcts_evolutionary_nodes_mdap import create_mdap_evolutionary_mcts, ProofContext
import asyncio

async def main():
    # Create context
    context = ProofContext(
        theorem="forall (n : Nat), n + 0 = n",
        goals=["prove n + 0 = n"],
        hypotheses=[],
        available_tactics=["intros", "simp", "rw", "refl"]
    )

    # Create and run MDAP MCTS
    mdap_mcts = create_mdap_evolutionary_mcts(
        population_size=20,
        num_agents=5,
        voting_strategy="first_k_ahead",
        mcts_simulations=100
    )

    result = await mdap_mcts.search(context)
    print(f"Success: {result.success}")

asyncio.run(main())
```

## Core Classes

### MDAPEvolutionaryNode
**Purpose**: MCTS node with multi-agent evaluation

```python
node = create_mdap_node(
    state=proof_state,
    population_size=20,
    num_agents=5,
    voting_strategy="first_k_ahead",
    consensus_threshold=0.75,
    k_ahead=3,
    enable_decomposition=True
)

# Key methods
best = node.get_agent_consensus()
agreement = node.compute_agreement_level()
should_decompose = node.should_decompose()
node.initialize_mdap_populations(context)
```

### MDAPSequenceEvaluator
**Purpose**: Multi-agent sequence evaluation

```python
evaluator = MDAPSequenceEvaluator(num_agents=5)
evaluations = await evaluator.evaluate_mdap(sequences, node, context)

# Access results
for seq_id, eval in evaluations.items():
    print(f"Fitness: {eval.consensus_fitness:.4f}")
    print(f"Agreement: {eval.agreement_level:.4f}")
```

### SequenceMAKERVoting
**Purpose**: MAKER voting for selection

```python
voting = SequenceMAKERVoting(k_ahead=3, voting_strategy="first_k_ahead")
best = voting.vote_on_best_sequence(node, evaluations)

# Strategies: "first_k_ahead", "majority", "weighted"
```

### MDAPEvolutionaryMCTS
**Purpose**: Main MDAP MCTS algorithm

```python
mdap_mcts = create_mdap_evolutionary_mcts(
    population_size=20,
    evolution_generations=5,
    num_agents=5,
    voting_strategy="first_k_ahead",
    enable_decomposition=True,
    consensus_threshold=0.75,
    k_ahead=3,
    mcts_simulations=100
)

result = await mdap_mcts.search(context)
```

## Common Tasks

### Create MDAP Node

```python
from mcts_evolutionary_nodes_mdap import create_mdap_node, ProofState

state = ProofState(
    goals=["prove goal"],
    context=["hypothesis"],
    tactics_sequence=[],
    depth=0
)

node = create_mdap_node(state, num_agents=5)
```

### Evaluate Sequences

```python
evaluator = MDAPSequenceEvaluator(num_agents=5)
evaluations = await evaluator.evaluate_mdap(sequences, node, context)
```

### Vote on Best Sequence

```python
voting = SequenceMAKERVoting(k_ahead=3)
best = voting.vote_on_best_sequence(node, evaluations)
```

### Red-Flag Invalid Sequences

```python
from mcts_evolutionary_nodes_mdap import SequenceRedFlagger

flagger = SequenceRedFlagger()
is_flagged, reasons = flagger.check_sequence(sequence, context)
```

### Monitor Evolution

```python
from mcts_evolutionary_nodes_mdap import MDAPEvolutionMonitor

monitor = MDAPEvolutionMonitor()
monitor.track_generation(node_id, gen, metrics)
curve = monitor.get_convergence_curve(node_id)
reliability = monitor.get_agent_reliability(agent_id)
```

### Parallel Evolution

```python
from mcts_evolutionary_nodes_mdap import DistributedMDAPEvolution

distributed = DistributedMDAPEvolution(evolution, max_workers=4)
results = await distributed.evolve_nodes_parallel(nodes, context)
```

## Parameters

### MDAPEvolutionaryNode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 20 | Size of evolutionary population |
| `num_agents` | int | 5 | Number of agents for evaluation |
| `voting_strategy` | str | "first_k_ahead" | MAKER voting strategy |
| `consensus_threshold` | float | 0.75 | Threshold for agent agreement |
| `k_ahead` | int | 3 | K-ahead voting parameter |
| `enable_decomposition` | bool | True | Enable decomposition |

### MDAPSequenceEvaluator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_agents` | int | 5 | Number of agents |
| `leanaide_client` | LeanAideClient | None | Optional LeanAide client |

### SequenceMAKERVoting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k_ahead` | int | 3 | K-ahead parameter |
| `voting_strategy` | str | "first_k_ahead" | Voting strategy |

### MDAPEvolutionaryMCTS

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 20 | Population size at nodes |
| `evolution_generations` | int | 5 | Generations per simulation |
| `num_agents` | int | 5 | Number of agents |
| `voting_strategy` | str | "first_k_ahead" | Voting strategy |
| `enable_decomposition` | bool | True | Enable decomposition |
| `consensus_threshold` | float | 0.75 | Consensus threshold |
| `k_ahead` | int | 3 | K-ahead parameter |
| `mcts_simulations` | int | 100 | MCTS iterations |

## Voting Strategies

### first_k_ahead (default)
- **Description**: Sequence wins if k votes ahead
- **Use case**: Strong consensus needed
- **Speed**: Slower but reliable

```python
voting = SequenceMAKERVoting(k_ahead=3, voting_strategy="first_k_ahead")
```

### majority
- **Description**: Simple majority voting
- **Use case**: Fast decisions
- **Speed**: Fast

```python
voting = SequenceMAKERVoting(voting_strategy="majority")
```

### weighted
- **Description**: Weighted by confidence
- **Use case**: Balance fitness and confidence
- **Speed**: Medium

```python
voting = SequenceMAKERVoting(voting_strategy="weighted")
```

## Red-Flag Conditions

Sequences are flagged when:
- Invalid tactics for context
- Contains cycles (repeated patterns)
- Leads to dead end
- Exceeds depth limit
- Low agent agreement (< 0.3)

## Decomposition Triggers

Decomposition occurs when:
- Agent agreement < consensus_threshold
- Population diversity > 0.3
- Node depth < 15
- Not already decomposed

## Monitoring Metrics

### Generation Metrics
```python
metrics = {
    "avg_fitness": 0.75,
    "best_fitness": 0.90,
    "agent_fitness": {
        "agent_0": 0.80,
        "agent_1": 0.75,
        # ...
    }
}
```

### Monitor Methods
- `track_generation(node_id, gen, metrics)` - Track generation
- `get_convergence_curve(node_id)` - Get fitness over time
- `get_agent_reliability(agent_id)` - Get agent consistency
- `get_summary()` - Get overall statistics

## Factory Functions

### create_mdap_evolutionary_mcts
```python
mdap_mcts = create_mdap_evolutionary_mcts(
    population_size=20,
    evolution_generations=5,
    num_agents=5,
    voting_strategy="first_k_ahead",
    enable_decomposition=True,
    **kwargs
)
```

### create_mdap_node
```python
node = create_mdap_node(
    state=proof_state,
    population_size=20,
    num_agents=5,
    voting_strategy="first_k_ahead",
    consensus_threshold=0.75,
    **kwargs
)
```

## Integration Examples

### With LeanAide
```python
from mcts_evolutionary_nodes_mdap import MDAPEvolutionaryMCTSWithLeanAide
from leanaide_client import LeanAideClient

client = LeanAideClient()
mdap_mcts = MDAPEvolutionaryMCTSWithLeanAide(
    leanaide_client=client,
    num_agents=5
)
result = await mdap_mcts.search_with_verification(theorem)
```

### With Decomposition
```python
from mcts_evolutionary_nodes_mdap import DecompositionAwareEvolution

evolution = DecompositionAwareEvolution(
    node_evolution=node_evolution,
    mdap_evaluator=evaluator
)
solution = await evolution.evolve_with_decomposition(
    node, context, max_depth=3
)
```

## Performance Tips

1. **Population Size**: 20-50 for most problems
2. **Number of Agents**: 3-7 for good diversity
3. **MCTS Simulations**: 100-500 for thorough search
4. **Consensus Threshold**: 0.7-0.8 for balance
5. **K-Ahead**: 2-5 for voting strength

## Common Patterns

### Basic Search
```python
mdap_mcts = create_mdap_evolutionary_mcts(num_agents=5)
result = await mdap_mcts.search(context)
```

### With Verification
```python
mdap_mcts = MDAPEvolutionaryMCTSWithLeanAide(client)
result = await mdap_mcts.search_with_verification(theorem)
```

### Custom Evolution
```python
evolution = MDAPNodeEvolution(
    mdap_evaluator, crossover, mutator, selection, voting
)
best = await evolution.evolve_at_node_mdap(node, context, generations=5)
```

## Troubleshooting

### Low Agreement
- Increase `num_agents`
- Lower `consensus_threshold`
- Check diversity of initial population

### Slow Convergence
- Increase `evolution_generations`
- Adjust `mutation_rate`
- Enable decomposition

### Memory Issues
- Reduce `population_size`
- Reduce `mcts_simulations`
- Use red-flagging early

## See Also

- Full documentation: `MDAP_EVOLUTIONARY_MCTS_DOCUMENTATION.md`
- Test suite: `test_mdap_evolutionary_mcts.py`
- Base MCTS: `mcts_evolutionary_nodes.py`
- MDAP engine: `mdap_engine.py`
- MAKER engine: `maker_engine.py`
