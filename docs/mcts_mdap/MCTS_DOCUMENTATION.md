# Monte Carlo Tree Search for Lean 4 Proof Search

## Overview

This module (`leanaide_mcts.py`) implements a comprehensive Monte Carlo Tree Search (MCTS) algorithm for automated Lean 4 proof search. Inspired by AlphaGo and AlphaZero architectures, it adapts MCTS to the domain of theorem proving.

## Architecture

### Core Components

```
MCTS (Main Orchestrator)
├── MCTSSelection (Selection Phase)
├── MCTSExpansion (Expansion Phase)
├── MCTSSimulation (Simulation Phase)
├── MCTSBackpropagation (Backpropagation Phase)
└── MCTSTree (Tree Management)
    └── MCTSNode (Tree Nodes)
        └── ProofState (Lean Proof State)
```

### Data Flow

```
Theorem Statement
    ↓
Initial Proof State (Root Node)
    ↓
┌─────────────────────────────────┐
│  MCTS Iteration Loop            │
│                                 │
│  1. Selection (UCT Policy)      │
│     ↓                            │
│  2. Expansion (Add Child)       │
│     ↓                            │
│  3. Simulation (Rollout)        │
│     ↓                            │
│  4. Backpropagation (Update)    │
│                                 │
└─────────────────────────────────┘
    ↓
Best Proof Path
    ↓
Lean 4 Proof Code
```

## Key Features

### 1. Four-Phase MCTS Algorithm

#### Selection Phase
- Uses **UCT (Upper Confidence Bound for Trees)** formula:
  ```
  UCT = W_i/N_i + c * sqrt(ln(N_parent) / N_i)
  ```
- Balances exploitation (high Q-value) vs exploration (low visits)
- Implements **progressive widening** for large action spaces

#### Expansion Phase
- Gets applicable tactics from LeanAide server
- Generates heuristic tactics as fallback
- Supports **action ranking** for intelligent selection
- Implements **transposition table** for state reuse

#### Simulation Phase
- Three rollout policies:
  - **Random**: Random tactic selection
  - **Heuristic**: Domain-guided tactic selection
  - **Learned**: Neural network policy (placeholder)
- Runs until terminal state or max depth

#### Backpropagation Phase
- Updates visit statistics (N, W, Q) up the tree
- Supports **AMAF (All-Moves-As-First)** updates:
  ```
  Q_AMAF = (1 - α) * Q_MCTS + α * Q_AMAF
  ```
- Faster convergence for proof search

### 2. Advanced Features

#### Transposition Table
- Hash-based state identification
- Reuses nodes for identical proof states
- Significant memory savings

#### Progressive Widening
- Gradually expands action space
- Formula: `k * N^α` actions explored
- Prevents premature action selection

#### Adaptive Exploration
- Temperature-based final selection
- Dirichlet noise for root exploration
- Dynamic c-param adjustment

#### Pruning
- Removes unpromising branches
- Based on visit count threshold
- Reduces memory footprint

### 3. LeanAide Integration

#### Proof State Management
```python
@dataclass
class ProofState:
    goals: List[str]           # Current unsolved goals
    context: List[str]         # Hypotheses and assumptions
    tactics_sequence: List     # Applied tactics
    depth: int                 # Proof depth
    is_complete: bool          # All goals solved?
    hash: str                  # State identifier
```

#### Tactic Application
- Integrates with LeanAide `elaborate` task
- Gets applicable tactics via API
- Applies tactics and retrieves new goals
- Simulates when server unavailable

## Usage Examples

### Basic Usage

```python
import asyncio
from leanaide_mcts import search_proof_with_mcts

async def main():
    theorem = "forall (a b : Nat), a + b = b + a"

    result = await search_proof_with_mcts(
        theorem=theorem,
        theorem_name="add_comm",
        max_iterations=1000,
        time_budget=60.0,
        rollout_policy="heuristic",
        enable_transposition_table=True
    )

    print(f"Success: {result.success}")
    print(f"Proof:\n{result.best_proof.lean_code}")

asyncio.run(main())
```

### Advanced Configuration

```python
from leanaide_mcts import MCTS, MCTSConfig

# Create custom configuration
config = MCTSConfig(
    max_iterations=5000,
    time_budget=300.0,
    c_param=1.414,              # UCT exploration constant
    rollout_depth=200,
    rollout_policy="heuristic",
    parallel_simulations=8,
    enable_transposition_table=True,
    enable_amaf=True,
    amaf_alpha=0.5,
    progressive_widening=True,
    early_termination=True,
    temperature=0.0,             # Greedy selection
    server_url="http://localhost:7654"
)

# Create MCTS instance
mcts = MCTS(config, theorem, theorem_name="my_theorem")

# Run search
result = await mcts.search()
```

### Accessing Statistics

```python
# Search statistics
print(f"Iterations: {result.search_iterations}")
print(f"Time: {result.time_elapsed:.2f}s")
print(f"Nodes visited: {result.nodes_visited}")
print(f"Win rate: {result.win_rate:.4f}")

# Tree statistics
stats = result.tree_statistics
print(f"Max depth: {stats['max_depth']}")
print(f"Branching factor: {stats['branching_factor']}")
print(f"Transposition hits: {stats['transposition_hits']}")
```

## Configuration Parameters

### MCTSConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | int | 1000 | Maximum MCTS iterations |
| `time_budget` | float | 60.0 | Time budget in seconds |
| `c_param` | float | 1.414 | UCT exploration constant (√2) |
| `rollout_depth` | int | 100 | Max rollout depth |
| `rollout_policy` | str | "heuristic" | Rollout policy type |
| `parallel_simulations` | int | 4 | Parallel simulation count |
| `enable_transposition_table` | bool | True | Enable state reuse |
| `enable_amaf` | bool | True | Enable AMAF updates |
| `amaf_alpha` | float | 0.5 | AMAF mixing parameter |
| `progressive_widening` | bool | True | Enable widening |
| `early_termination` | bool | True | Stop on proof found |
| `temperature` | float | 0.0 | Selection temperature |
| `max_tree_depth` | int | 50 | Maximum tree depth |
| `pruning_threshold` | float | 0.1 | Pruning threshold |

## Algorithm Details

### UCT Formula

```
UCT(i) = W_i/N_i + c * sqrt(ln(N_parent) / N_i)
```

Where:
- `W_i`: Total reward of node i
- `N_i`: Visit count of node i
- `c`: Exploration constant (default: √2)
- `N_parent`: Visit count of parent

### AMAF Update

```
Q_AMAF = (1 - α) * Q_MCTS + α * Q_AMAF
```

Where:
- `Q_MCTS`: Standard MCTS Q-value
- `Q_AMAF`: AMAF Q-value (from all rollouts)
- `α`: Mixing parameter (default: 0.5)

### Progressive Widening

```
k(N) = K * N^α
```

Where:
- `k(N)`: Number of actions to explore at visit N
- `K`: Total available actions
- `α`: Widening factor (default: 0.5)

## Performance Considerations

### Memory Management

1. **Transposition Table**: Reduces memory by reusing states
2. **Pruning**: Removes unpromising branches
3. **Cache Size Limit**: Configurable via `cache_size_mb`

### Speed Optimizations

1. **Parallel Simulations**: Run multiple rollouts concurrently
2. **Batch LeanAide Queries**: Reduce API calls
3. **Async Operations**: Non-blocking tactic application
4. **Early Termination**: Stop when proof found

### Tuning Tips

1. **Exploration vs Exploitation**:
   - Increase `c_param` for more exploration
   - Decrease for more exploitation

2. **Rollout Policy**:
   - "random": Fast but less accurate
   - "heuristic": Good balance
   - "learned": Best (requires training)

3. **Time Budget**:
   - Set based on theorem complexity
   - Simple theorems: 10-30 seconds
   - Complex theorems: 300+ seconds

## Integration with Evolutionary Framework

The MCTS implementation integrates seamlessly with the existing evolutionary LeanAide framework:

```python
from leanaide_evolution import LeanProofEvolutionEngine
from leanaide_mcts import MCTS

# Use MCTS for initial proof search
mcts = MCTS(config, theorem)
mcts_result = await mcts.search()

# Use evolutionary search for refinement
evo_engine = LeanProofEvolutionEngine(theorem)
evo_result = await evo_engine.evolve()

# Combine results
best_proof = select_best([mcts_result.best_proof, evo_result.best_proof])
```

## Testing

Run the test suite:

```bash
python test_leanaide_mcts.py
```

Tests cover:
- ProofState creation and hashing
- MCTSNode statistics and UCT calculation
- MCTSTree management
- Selection phase
- Simulation phase (random and heuristic)
- Backpropagation with AMAF
- Full MCTS search

## References

1. **AlphaGo Zero** (Silver et al., 2017)
   - Self-play reinforcement learning
   - Monte Carlo Tree Search with neural networks

2. **MCTS for Theorem Proving** (Irving et al., 2016)
   - DeepMath: Applying deep learning to theorem proving
   - MCTS for premise selection

3. **Upper Confidence Bounds** (Kocsis & Szepesvári, 2006)
   - UCB algorithm for bandit problems
   - UCT for MCTS

## License

This implementation is part of the OpenEvolve project.

## Author

OpenEvolve Team

## Version

1.0.0 (2025-12-30)
