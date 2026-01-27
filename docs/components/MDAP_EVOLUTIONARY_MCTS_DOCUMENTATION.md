# MDAP/MAKER Integration for Evolutionary MCTS Nodes

## Overview

This module integrates **MDAP** (Multi-Agent voting) and **MAKER** with the evolutionary MCTS nodes approach, creating rich exploration with multi-agent consensus and zero-error guarantees.

## Core Concept

Each MCTS node maintains populations that are evolved using:
1. **Multi-agent evaluation** - Multiple agents independently evaluate sequences
2. **MAKER voting** - Agent consensus is reached through k-ahead voting
3. **Decomposition** - Complex problems can be decomposed into subtasks
4. **Lean verification** - Formal verification ensures zero errors

## File Structure

- **mcts_evolutionary_nodes_mdap.py** (~1935 lines) - Main implementation
- **test_mdap_evolutionary_mcts.py** - Comprehensive test suite

## Implementation Components

### 1. MDAPEvolutionaryNode

Extended evolutionary node with MDAP capabilities:

```python
from mcts_evolutionary_nodes_mdap import MDAPEvolutionaryNode, create_mdap_node

# Create MDAP node
node = create_mdap_node(
    state=proof_state,
    population_size=20,
    num_agents=5,
    voting_strategy="first_k_ahead",
    consensus_threshold=0.75,
    k_ahead=3,
    enable_decomposition=True
)

# Get consensus sequence
best = node.get_agent_consensus()

# Compute agreement level
agreement = node.compute_agreement_level()

# Check if should decompose
should_decompose = node.should_decompose()
```

**Key Features:**
- Agent-specific populations (each agent has their own view)
- Multi-agent fitness tracking
- MAKER voting configuration
- Decomposition support
- Agreement level computation

### 2. MDAPSequenceEvaluator

Evaluate sequences using multiple agents:

```python
from mcts_evolutionary_nodes_mdap import MDAPSequenceEvaluator

# Create evaluator
evaluator = MDAPSequenceEvaluator(num_agents=5)

# Evaluate sequences
evaluations = await evaluator.evaluate_mdap(
    sequences=population,
    node=mdap_node,
    context=proof_context
)

# Access evaluation results
for seq_id, evaluation in evaluations.items():
    print(f"Consensus fitness: {evaluation.consensus_fitness}")
    print(f"Agreement level: {evaluation.agreement_level}")
    print(f"Red flags: {evaluation.red_flags}")
```

**Evaluation Process:**
1. Each agent independently evaluates the sequence
2. Agent-specific bias is added for diversity
3. Consensus is computed using weighted averaging
4. Agreement level is computed from variance
5. Low-confidence evaluations are flagged

### 3. SequenceMAKERVoting

MAKER voting for sequence selection:

```python
from mcts_evolutionary_nodes_mdap import SequenceMAKERVoting

# Create voting system
voting = SequenceMAKERVoting(
    k_ahead=3,
    voting_strategy="first_k_ahead"
)

# Vote on best sequence
best = voting.vote_on_best_sequence(node, evaluations)
```

**Voting Strategies:**

1. **first_k_ahead** (default)
   - Sequence wins if it's k votes ahead of all others
   - Provides strong consensus guarantee

2. **majority**
   - Simple majority voting
   - Faster but less conservative

3. **weighted**
   - Weighted sum of agent confidences
   - Balances fitness and confidence

### 4. MDAPNodeEvolution

Evolution at nodes with MDAP evaluation:

```python
from mcts_evolutionary_nodes_mdap import MDAPNodeEvolution

# Create evolution controller
evolution = MDAPNodeEvolution(
    mdap_evaluator=evaluator,
    sequence_crossover=crossover,
    sequence_mutator=mutator,
    sequence_selection=selection,
    maker_voting=voting
)

# Evolve at node
best = await evolution.evolve_at_node_mdap(
    node=mdap_node,
    context=proof_context,
    generations=5
)
```

**Evolution Loop:**
1. Multi-agent evaluation of population
2. Check for convergence (high agreement)
3. Select parents using MAKER voting
4. Crossover to create offspring
5. Mutate offspring
6. Survival selection with voting
7. Update agent populations

### 5. DecompositionAwareEvolution

Evolution with automatic decomposition:

```python
from mcts_evolutionary_nodes_mdap import DecompositionAwareEvolution

# Create decomposition-aware evolution
evolution = DecompositionAwareEvolution(
    node_evolution=node_evolution,
    mdap_evaluator=evaluator
)

# Evolve with decomposition
best = await evolution.evolve_with_decomposition(
    node=mdap_node,
    context=proof_context,
    max_depth=3
)
```

**Decomposition Triggers:**
- Low agent agreement (< consensus_threshold)
- High population diversity (> 0.3)
- Appropriate depth (< 15)
- Not already decomposed

### 6. MDAPEvolutionaryMCTS

Main MDAP evolutionary MCTS class:

```python
from mcts_evolutionary_nodes_mdap import create_mdap_evolutionary_mcts

# Create MDAP MCTS
mdap_mcts = create_mdap_evolutionary_mcts(
    population_size=20,
    evolution_generations=5,
    num_agents=5,
    voting_strategy="first_k_ahead",
    enable_decomposition=True,
    mcts_simulations=100
)

# Run search
result = await mdap_mcts.search(initial_context)
```

**Key Parameters:**
- `population_size`: Size of evolutionary population at each node
- `evolution_generations`: Generations per MCTS simulation
- `num_agents`: Number of agents for MDAP evaluation
- `voting_strategy`: MAKER voting strategy
- `enable_decomposition`: Enable automatic decomposition
- `consensus_threshold`: Threshold for agent agreement
- `k_ahead`: K-ahead parameter for voting

### 7. MDAPEvolutionaryMCTSWithLeanAide

Integration with Lean formal verification:

```python
from mcts_evolutionary_nodes_mdap import MDAPEvolutionaryMCTSWithLeanAide
from leanaide_client import LeanAideClient

# Create LeanAide client
client = LeanAideClient()

# Create MCTS with verification
mdap_mcts = MDAPEvolutionaryMCTSWithLeanAide(
    leanaide_client=client,
    population_size=20,
    num_agents=5
)

# Search with verification
result = await mdap_mcts.search_with_verification(
    theorem="forall (n : Nat), n + 0 = n"
)
```

**Verification Process:**
1. Run MDAP evolutionary MCTS
2. Collect top candidates
3. Verify each with Lean
4. Apply bonus for verified proofs
5. Return best verified proof

### 8. SequenceRedFlagger

Red-flag invalid sequences:

```python
from mcts_evolutionary_nodes_mdap import SequenceRedFlagger

# Create flagger
flagger = SequenceRedFlagger()

# Check sequence
is_flagged, reasons = flagger.check_sequence(sequence, context)

if is_flagged:
    print(f"Sequence flagged: {reasons}")
```

**Flag Conditions:**
- Invalid tactics for context
- Contains cycles
- Leads to dead end
- Exceeds depth limit
- Low agent agreement

### 9. DistributedMDAPEvolution

Parallel evolution at multiple nodes:

```python
from mcts_evolutionary_nodes_mdap import DistributedMDAPEvolution

# Create distributed evolution
distributed = DistributedMDAPEvolution(
    node_evolution=evolution,
    max_workers=4
)

# Evolve multiple nodes in parallel
results = await distributed.evolve_nodes_parallel(
    nodes=[node1, node2, node3],
    context=proof_context,
    max_workers=4
)
```

**Parallel Execution:**
- Semaphore limits concurrent workers
- Each node evolves independently
- Results collected into dictionary

### 10. MDAPEvolutionMonitor

Performance monitoring:

```python
from mcts_evolutionary_nodes_mdap import MDAPEvolutionMonitor

# Create monitor
monitor = MDAPEvolutionMonitor()

# Track generation
monitor.track_generation(
    node_id=node.node_id,
    generation=gen,
    metrics={
        "avg_fitness": avg_fitness,
        "best_fitness": best_fitness,
        "agent_fitness": agent_fitness_dict
    }
)

# Get convergence curve
curve = monitor.get_convergence_curve(node_id)

# Get agent reliability
reliability = monitor.get_agent_reliability("agent_0")

# Get summary
summary = monitor.get_summary()
```

## Usage Examples

### Example 1: Basic MDAP Evolutionary MCTS

```python
import asyncio
from mcts_evolutionary_nodes_mdap import (
    create_mdap_evolutionary_mcts,
    ProofContext
)

async def main():
    # Create proof context
    context = ProofContext(
        theorem="forall (a b : Nat), a + b = b + a",
        goals=["prove a + b = b + a"],
        hypotheses=[],
        available_tactics=[
            "intros", "simp", "rw", "apply", "exact",
            "induction", "cases", "linarith", "ring"
        ]
    )

    # Create MDAP MCTS
    mdap_mcts = create_mdap_evolutionary_mcts(
        population_size=20,
        evolution_generations=5,
        num_agents=5,
        voting_strategy="first_k_ahead",
        enable_decomposition=True,
        mcts_simulations=100
    )

    # Run search
    result = await mdap_mcts.search(context)

    # Print results
    print(f"Success: {result.success}")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Win rate: {result.win_rate:.4f}")

asyncio.run(main())
```

### Example 2: With Lean Verification

```python
from leanaide_client import LeanAideClient
from mcts_evolutionary_nodes_mdap import (
    MDAPEvolutionaryMCTSWithLeanAide,
    ProofContext
)

async def main():
    # Create LeanAide client
    client = LeanAideClient()

    # Create context
    context = ProofContext(
        theorem="forall (n : Nat), n + 0 = n",
        goals=["prove n + 0 = n"],
        hypotheses=[],
        available_tactics=["intros", "simp", "rw", "refl"]
    )

    # Create MCTS with verification
    mdap_mcts = MDAPEvolutionaryMCTSWithLeanAide(
        leanaide_client=client,
        population_size=20,
        num_agents=5
    )

    # Search with verification
    result = await mdap_mcts.search_with_verification(
        theorem="forall (n : Nat), n + 0 = n"
    )

    print(f"Verified proof: {result.success}")

asyncio.run(main())
```

### Example 3: Custom Evolution

```python
from mcts_evolutionary_nodes_mdap import (
    MDAPNodeEvolution,
    MDAPSequenceEvaluator,
    SequenceMAKERVoting,
    create_mdap_node,
    ProofContext,
    ProofState
)

async def main():
    # Create components
    evaluator = MDAPSequenceEvaluator(num_agents=5)
    voting = SequenceMAKERVoting(k_ahead=3)

    # Create node
    state = ProofState(
        goals=["prove goal"],
        context=[],
        tactics_sequence=[],
        depth=0
    )

    node = create_mdap_node(state, num_agents=5)

    # Create context
    context = ProofContext(
        theorem="test",
        goals=["prove goal"],
        hypotheses=[],
        available_tactics=["simp", "rw", "apply"]
    )

    # Initialize populations
    node.initialize_mdap_populations(context)

    # Evolve
    evolution = MDAPNodeEvolution(
        mdap_evaluator=evaluator,
        sequence_crossover=crossover,
        sequence_mutator=mutator,
        sequence_selection=selection,
        maker_voting=voting
    )

    best = await evolution.evolve_at_node_mdap(
        node, context, generations=5
    )

    print(f"Best fitness: {best.fitness:.4f}")

asyncio.run(main())
```

## Key Features

### 1. Multi-Agent Node Evolution
Each MCTS node maintains populations for multiple agents, enabling:
- Diverse evaluation perspectives
- Robust consensus through voting
- Better exploration of search space

### 2. MAKER Voting
Consensus-based sequence selection with:
- K-ahead voting for strong guarantees
- Multiple voting strategies
- Configurable consensus thresholds

### 3. Decomposition
Automatic problem decomposition when:
- Agent agreement is low
- Population diversity is high
- Problem complexity warrants it

### 4. LeanAide Verification
Formal verification provides:
- Zero-error guarantees
- Fitness bonuses for verified proofs
- Early pruning of invalid sequences

### 5. Red-Flagging
Filter invalid sequences early:
- Invalid tactics detection
- Cycle detection
- Dead end identification

### 6. Parallel Evolution
Distribute computation across workers:
- Semaphore-controlled parallelism
- Independent node evolution
- Efficient resource utilization

### 7. Performance Monitoring
Track convergence and quality:
- Generation-level metrics
- Agent reliability tracking
- Convergence curve analysis

## Performance Considerations

### Scalability
- **Population size**: 20-50 per node recommended
- **Number of agents**: 3-7 agents provides good diversity
- **MCTS simulations**: 100-500 for thorough search

### Memory Management
- Agent populations are stored separately
- Caching of evaluations reduces recomputation
- Red-flagging prevents wasteful computation

### Parallelization
- Use DistributedMDAPEvolution for multiple nodes
- Semaphore limits prevent resource exhaustion
- Independent tasks scale well

## Testing

Run the test suite:

```bash
python test_mdap_evolutionary_mcts.py
```

Tests include:
1. MDAP evolutionary node creation
2. MDAP sequence evaluation
3. MAKER voting
4. Sequence red-flagging
5. MDAP monitoring
6. Full MDAP MCTS (simplified)

## Integration Points

### With MCTS Evolutionary Nodes
```python
from mcts_evolutionary_nodes import EvolutionaryMCTS
from mcts_evolutionary_nodes_mdap import MDAPEvolutionaryMCTS

# MDAPEvolutionaryMCTS extends EvolutionaryMCTS
# All base MCTS functionality is preserved
```

### With MDAP Engine
```python
from mdap_engine import MDAPOrchestrator, MDAPConfig
from mcts_evolutionary_nodes_mdap import MDAPSequenceEvaluator

# Uses MDAP patterns for multi-agent evaluation
```

### With MAKER Engine
```python
from maker_engine import MakerEngine, MakerConfig
from mcts_evolutionary_nodes_mdap import SequenceMAKERVoting

# Uses MAKER voting patterns
```

### With Decomposition
```python
from decomposition_engine import DecompositionEngine
from mcts_evolutionary_nodes_mdap import DecompositionAwareEvolution

# Can use decomposition engine for subtask creation
```

## Future Enhancements

1. **Adaptive Agent Selection**
   - Choose agents dynamically based on problem type
   - Agent specialization for different tactic categories

2. **Hierarchical Decomposition**
   - Multi-level decomposition for complex problems
   - Subtask dependency resolution

3. **Learning from Verification**
   - Use Lean feedback to improve evaluation
   - Learn tactic success patterns

4. **Distributed MCTS**
   - Distribute MCTS search across machines
   - Shared population pools

5. **Adaptive Voting**
   - Dynamically adjust k-ahead parameter
   - Strategy selection based on convergence

## References

- **MCTS Evolutionary Nodes**: `mcts_evolutionary_nodes.py`
- **MDAP Engine**: `mdap_engine.py`
- **MAKER Engine**: `maker_engine.py`
- **Decomposition Engine**: `decomposition_engine.py`
- **LeanAide**: `leanaide_client.py`

## License

This module is part of the OpenEvolve project.

## Author

OpenEvolve Team

Created: 2025-12-30
