# MDAP/MAKER + MCTS Unified Framework

A comprehensive unified framework integrating MDAP (Multi-Agent voting) and MAKER (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) with three hybrid Monte Carlo Tree Search approaches for zero-error theorem proving.

## Overview

This framework provides a **single unified interface** to three powerful hybrid MCTS approaches, all enhanced with MDAP multi-agent evaluation and MAKER consensus-based voting:

1. **Evolved Policies**: Evolve rollout policies using MDAP evaluation
2. **Evolutionary Nodes**: Evolve action sequences at each MCTS node with MDAP
3. **Coevolution**: Coevolve decision trees with MDAP evaluation

## Key Features

### Core Features

- **Unified Configuration**: Single configuration class for all three approaches
- **Multi-Agent Evaluation**: All approaches use MDAP multi-agent consensus
- **MAKER Voting**: First-to-ahead-by-k voting for zero-error guarantees
- **Decomposition Support**: Automatic task decomposition for complex theorems
- **LeanAide Integration**: Formal verification with fitness bonuses
- **Adaptive Selection**: Automatically choose the best approach
- **Combined Search**: Run all approaches in parallel and combine results
- **Comprehensive Caching**: Avoid redundant computations
- **Workflow Integration**: OpenEvolve stages 3A/B/C support
- **Monitoring & Logging**: Track execution metrics

### Zero-Error Guarantees

The framework implements techniques from ["Solving a Million-Step LLM Task with Zero Errors"](https://arxiv.org/abs/2511.09030):

- **Multi-Agent Consensus**: Multiple agents evaluate each candidate
- **First-to-Ahead-by-K Voting**: Robust MAKER voting mechanism
- **Red-Flagging**: Detect and discard unreliable responses
- **Decomposition**: Break complex problems into atomic subtasks

## Installation

### Requirements

```bash
pip install asyncio numpy
```

### Optional Dependencies

```bash
# For Lean 4 formal verification
pip install leanaide-client

# For decomposition
pip install decomposition-engine

# For workflow integration
pip install openevolve-workflow
```

## Quick Start

### Basic Usage

```python
import asyncio
from mdap_maker_mcts_unified import (
    MDAPMAKERMCTSEngine,
    MDAPMAKERMCTSConfig,
    MCTSApproach
)

async def main():
    # Create configuration
    config = MDAPMAKERMCTSConfig(
        approach=MCTSApproach.EVOLVED_POLICIES,
        num_agents=5,
        simulations=100,
        enable_decomposition=True
    )

    # Create engine
    engine = MDAPMAKERMCTSEngine(config)

    # Search for proof
    theorem = "theorem example (n : Nat) : n + 0 = n := by"
    result = await engine.search(theorem)

    # Check results
    print(f"Success: {result.success}")
    print(f"Fitness: {result.best_fitness}")
    print(f"Proof: {result.best_proof}")

asyncio.run(main())
```

### Using Presets

```python
from mdap_maker_mcts_unified import MDAPMCTSPresets

# Fast execution
config = MDAPMCTSPresets.fast()

# Balanced (recommended)
config = MDAPMCTSPresets.balanced()

# Maximum quality
config = MDAPMCTSPresets.thorough()

# Experimental (try all approaches)
config = MDAPMCTSPresets.experimental()
```

### Adaptive Selection

```python
from mdap_maker_mcts_unified import MDAPAdaptiveSelector

selector = MDAPAdaptiveSelector()

# Automatically select best approach
approach = selector.select_approach(
    theorem="theorem complex (a b c : Nat) : a * (b + c) = a * b + a * c := by",
    domain="algebra",
    available_agents=5
)

config = MDAPMAKERMCTSConfig(approach=approach)
engine = MDAPMAKERMCTSEngine(config)
result = await engine.search(theorem)
```

### Combined Search

```python
# Run all three approaches and combine
config = MDAPMAKERMCTSConfig(
    approach=MCTSApproach.COMBINED,
    num_agents=5
)

engine = MDAPMAKERMCTSEngine(config)
result = await engine.search(theorem)

# Results include metrics from all approaches
print(result.metadata['approach_results'])
```

## Configuration

### Unified Configuration

```python
@dataclass
class MDAPMAKERMCTSConfig:
    # Base approach
    approach: MCTSApproach = MCTSApproach.EVOLVED_POLICIES

    # MDAP parameters
    num_agents: int = 5
    agent_reliability_threshold: float = 0.6
    enable_decomposition: bool = True
    decomposition_depth: int = 3

    # MAKER voting parameters
    voting_strategy: str = "first_k_ahead"
    k_ahead: int = 3
    consensus_threshold: float = 0.75

    # Common MCTS parameters
    exploration_constant: float = 1.414
    simulations: int = 100
    max_depth: int = 50

    # Approach-specific parameters
    evolved_policy: EvolvedPolicyConfig
    evolutionary_node: EvolutionaryNodeConfig
    coevolution: CoevolutionConfig

    # LeanAide integration
    leanaide_enabled: bool = True
    verification_bonus: float = 1.5

    # Performance
    parallel_evaluation: bool = True
    max_workers: int = 4
    enable_caching: bool = True
```

### Approach-Specific Configurations

#### Evolved Policies

```python
evolved_policy = EvolvedPolicyConfig(
    population_size=50,      # Number of policies in population
    generations=10,           # Number of evolution generations
    mutation_rate=0.1,        # Mutation probability
    crossover_rate=0.7,       # Crossover probability
    elite_fraction=0.1,       # Fraction of elite policies to keep
    tournament_size=3,        # Tournament selection size
    policy_depth=5,           # Maximum policy depth
    tactics_per_decision=10   # Number of tactics to consider
)
```

#### Evolutionary Nodes

```python
evolutionary_node = EvolutionaryNodeConfig(
    population_per_node=20,     # Population size at each node
    max_generations_per_node=5, # Max generations per node
    sequence_length=5,          # Length of action sequences
    mutation_rate=0.15,         # Mutation probability
    crossover_rate=0.6,         # Crossover probability
    selection_pressure=2.0,     # Selection pressure
    adaptation_interval=10      # Adaptation frequency
)
```

#### Coevolution

```python
coevolution = CoevolutionConfig(
    tree_population=30,         # Number of trees
    host_population=20,         # Number of host problems
    coevolution_generations=15, # Generations to coevolve
    tree_depth=4,               # Maximum tree depth
    branching_factor=3,         # Branching factor
    mutation_rate=0.2,          # Mutation probability
    crossover_rate=0.5,         # Crossover probability
    competitive_ratio=0.3       # Competitive ratio
)
```

## Results

### Result Structure

```python
@dataclass
class MDAPMAKERMCTSResult:
    # Basic results
    success: bool
    best_proof: Optional[str]
    best_fitness: float
    approach: MCTSApproach

    # MDAP metrics
    agent_results: List[AgentResult]
    consensus_score: float
    agreement_level: float
    voting_details: VotingDetails

    # Decomposition metrics
    decomposition_used: bool
    subtask_count: int
    decomposition_depth: int

    # Approach-specific metrics
    policy_metrics: PolicyMetrics      # For evolved policies
    node_metrics: NodeMetrics          # For evolutionary nodes
    tree_metrics: TreeMetrics          # For coevolution

    # Verification
    verification_result: VerificationResult

    # Performance
    execution_time: float
    total_evaluations: int
    generations_completed: int
    mcts_simulations: int
```

### Accessing Results

```python
result = await engine.search(theorem)

# Basic results
if result.success:
    print(f"Found proof: {result.best_proof}")
    print(f"Fitness: {result.best_fitness}")

# MDAP consensus
if result.consensus_score:
    print(f"Consensus: {result.consensus_score:.2%}")
    print(f"Agreement: {result.agreement_level:.2%}")

# Agent evaluations
if result.agent_results:
    for agent_result in result.agent_results:
        print(f"{agent_result.agent_id}: {agent_result.fitness:.3f}")

# Verification
if result.verification_result:
    if result.verification_result.is_valid:
        print("Proof verified by LeanAide!")
```

## Advanced Usage

### Benchmarking

```python
from mdap_maker_mcts_unified import MDAPMCTSBenchmark

config = MDAPMAKERMCTSConfig(num_agents=5)
benchmark = MDAPMCTSBenchmark(config)

# Benchmark all approaches
test_theorems = [
    "theorem thm1 (n : Nat) : n + 0 = n := by",
    "theorem thm2 (a b : Nat) : a + b = b + a := by",
    "theorem thm3 (a b c : Nat) : a * (b + c) = a * b + a * c := by"
]

report = await benchmark.benchmark_all(
    test_theorems=test_theorems,
    approaches=[
        MCTSApproach.EVOLVED_POLICIES,
        MCTSApproach.EVOLUTIONARY_NODES,
        MCTSApproach.COEVOLUTION
    ]
)

# View results
print(f"Best approach: {report.comparison['best_success_rate']['approach']}")
print(f"Success rate: {report.comparison['best_success_rate']['rate']:.1%}")
```

### Workflow Integration

```python
from mdap_maker_mcts_unified import (
    MDAPMCTSWorkflowIntegrator,
    SubProblem
)

integrator = MDAPMCTSWorkflowIntegrator(config)

subproblem = SubProblem(
    subproblem_id="sub_001",
    theorem="theorem example (n : Nat) : n + 0 = n := by",
    dependencies=[],
    priority=1
)

solution = await integrator.solve_with_mdap_mcts(subproblem)

print(f"Solution: {solution.content}")
print(f"Quality: {solution.quality_metrics}")
```

### Custom Caching

```python
from mdap_maker_mcts_unified import MDAPMCTSCache

cache = MDAPMCTSCache(max_size=10000)

# Use with engine
engine = MDAPMAKERMCTSEngine(config, cache=cache)

# Or directly
await cache.set('policy', 'my_key', {'data': 'value'})
value = await cache.get('policy', 'my_key')

# Compute with caching
result = await cache.get_or_compute(
    'policy',
    'my_key',
    lambda: expensive_computation()
)

# View statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

### Monitoring

```python
from mdap_maker_mcts_unified import MDAPMCTSMonitor

monitor = MDAPMCTSMonitor()
engine = MDAPMAKERMCTSEngine(config, monitor=monitor)

result = await engine.search(theorem)

# Get execution summary
summary = monitor.get_summary()
print(f"Duration: {summary['duration_seconds']:.2f}s")
print(f"Total evaluations: {summary['total_agent_evaluations']}")
print(f"Avg consensus: {summary.get('avg_consensus', 0):.3f}")
```

## Running the Demo

A comprehensive demo script is included:

```bash
python demo_mdap_maker_mcts_unified.py
```

The demo includes:
1. Basic usage
2. All three approaches
3. Adaptive selection
4. Combined search
5. Configuration presets
6. Workflow integration
7. Benchmarking
8. Serialization
9. Cache management
10. Validation

## Architecture

### Core Components

```
MDAP/MAKER + MCTS Unified Framework
│
├── Configuration Layer
│   ├── MDAPMAKERMCTSConfig (unified config)
│   ├── EvolvedPolicyConfig
│   ├── EvolutionaryNodeConfig
│   └── CoevolutionConfig
│
├── Engine Layer
│   ├── MDAPMAKERMCTSEngine (main engine)
│   ├── _search_evolved_policies()
│   ├── _search_evolutionary_nodes()
│   └── _search_coevolution()
│
├── MDAP Integration
│   ├── Multi-agent evaluation
│   ├── MAKER voting (first-to-ahead-by-k)
│   └── Consensus computation
│
├── Result Layer
│   ├── MDAPMAKERMCTSResult
│   ├── AgentResult
│   ├── VotingDetails
│   └── VerificationResult
│
├── Utility Layer
│   ├── MDAPMCTSCache (caching)
│   ├── MDAPMCTSMonitor (monitoring)
│   ├── MDAPAdaptiveSelector (adaptive selection)
│   └── MDAPCombinedSearch (combined search)
│
└── Integration Layer
    ├── MDAPMCTSBenchmark (benchmarking)
    ├── MDAPMCTSWorkflowIntegrator (workflow)
    └── MDAPMCTSPresets (presets)
```

### Data Flow

```
Theorem
    ↓
[Adaptive Selector]
    ↓
[MDAPMAKERMCTSEngine]
    ↓
[Approach-Specific Search]
    ├── Evolved Policies
    ├── Evolutionary Nodes
    └── Coevolution
    ↓
[MDAP Multi-Agent Evaluation]
    ├── Agent 1
    ├── Agent 2
    ├── ...
    └── Agent N
    ↓
[MAKER Voting]
    ├── First-to-Ahead-by-K
    └── Consensus
    ↓
[LeanAide Verification] (optional)
    ↓
[MDAPMAKERMCTSResult]
```

## API Reference

### Classes

#### `MDAPMAKERMCTSConfig`

Unified configuration for all approaches.

**Methods:**
- `to_dict() -> Dict`: Serialize to dictionary
- `from_dict(data: Dict) -> MDAPMAKERMCTSConfig`: Deserialize from dictionary
- `validate() -> List[str]`: Validate configuration, return errors

#### `MDAPMAKERMCTSEngine`

Main engine for MDAP/MAKER + MCTS.

**Methods:**
- `async search(theorem: str, approach: Optional[MCTSApproach]) -> MDAPMAKERMCTSResult`: Main search

#### `MDAPMAKERMCTSResult`

Unified result from all approaches.

**Methods:**
- `to_dict() -> Dict`: Convert to dictionary
- `from_dict(data: Dict) -> MDAPMAKERMCTSResult`: Create from dictionary

#### `MDAPMCTSCache`

Cache for avoiding redundant computations.

**Methods:**
- `async get(cache_type: str, key: str) -> Optional[Any]`: Get cached value
- `async set(cache_type: str, key: str, value: Any)`: Set cached value
- `async get_or_compute(cache_type: str, key: str, compute_fn: Callable) -> Any`: Get or compute
- `clear()`: Clear all caches
- `get_stats() -> Dict`: Get cache statistics

#### `MDAPAdaptiveSelector`

Select best approach based on problem features.

**Methods:**
- `select_approach(theorem: str, domain: str, available_agents: int) -> MCTSApproach`: Select approach
- `record_result(theorem: str, approach: MCTSApproach, success: bool, domain: str)`: Record for learning

#### `MDAPMCTSBenchmark`

Benchmark all approaches.

**Methods:**
- `async benchmark_all(test_theorems: List[str], approaches: List[MCTSApproach]) -> BenchmarkReport`: Run benchmarks

### Enums

#### `MCTSApproach`

Available MCTS approaches:
- `EVOLVED_POLICIES`: Evolve rollout policies
- `EVOLUTIONARY_NODES`: Evolve action sequences at nodes
- `COEVOLUTION`: Coevolve decision trees
- `ADAPTIVE`: Automatically select approach
- `COMBINED`: Run all and combine results

#### `VotingStrategy`

Voting strategies:
- `FIRST_K_AHEAD`: MAKER first-to-ahead-by-k (default)
- `FIRST_TO_K`: Simple first-to-k votes
- `MAJORITY`: Simple majority (>50%)
- `WEIGHTED`: Weighted by agent reliability
- `CONSENSUS`: High agreement threshold

## Performance Tips

1. **Use Presets**: Start with `MDAPMCTSPresets.balanced()` for good performance
2. **Enable Caching**: Set `enable_caching=True` for repeated searches
3. **Parallel Evaluation**: Set `parallel_evaluation=True` with `max_workers=4`
4. **Adjust Simulations**: Reduce `simulations` for faster results, increase for quality
5. **Use Adaptive Selection**: Let the framework choose the best approach automatically
6. **Decomposition**: Enable `enable_decomposition=True` for complex theorems
7. **LeanAide**: Enable `leanaide_enabled=True` for formal verification (slower but guaranteed correctness)

## Troubleshooting

### Import Errors

If you see import warnings, some optional dependencies are missing:

```python
WARNING: MDAP engine not available
WARNING: MAKER engine not available
```

The framework will gracefully degrade, but functionality will be limited.

Install missing dependencies:

```bash
pip install mdap-engine maker-engine
```

### Memory Issues

For large searches, reduce memory usage:

```python
config = MDAPMAKERMCTSConfig(
    cache_size=1000,  # Reduce cache
    max_workers=2,    # Reduce parallelism
    enable_decomposition=False  # Disable decomposition
)
```

### Slow Performance

For faster results:

```python
config = MDAPMAKERMCTSConfig(
    simulations=50,   # Reduce simulations
    max_depth=25,     # Reduce depth
    num_agents=3,     # Reduce agents
    leanaide_enabled=False  # Disable verification
)
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional MCTS approaches
- New voting strategies
- Enhanced decomposition algorithms
- Performance optimizations
- Better caching strategies
- Additional verification backends

## Citation

If you use this framework, please cite:

```bibtex
@misc{mdap_maker_mcts_2025,
  title={MDAP/MAKER + MCTS: A Unified Framework for Zero-Error Theorem Proving},
  author={OpenEvolve},
  year={2025},
  note={Integration of techniques from arXiv:2511.09030}
}
```

## References

1. [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030) - MDAP/MAKER paper
2. [Mastering the Game of Go with Deep Neural Networks](https://www.nature.com/articles/nature16961) - AlphaGo/MCTS
3. [Lean 4 Theorem Prover](https://leanprover.github.io/) - Formal verification

## License

MIT License - See LICENSE file for details

## Contact

For questions, issues, or contributions, please visit the OpenEvolve project repository.
