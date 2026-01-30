# LeanAide MCTS-MDAP-MAKER Integration

## Quick Reference Guide

### Overview

This module provides a comprehensive integration of three powerful proof generation systems:

- **MCTS** (Monte Carlo Tree Search): Intelligent tree search with UCT exploration
- **MDAP** (Multi-Agent Pipeline): Multi-agent perspectives for reducing bias
- **MAKER** (Multi-Agent Knowledge Enhanced Reasoning): Voting consensus for error correction

### Key Components

#### 1. **MDAPMCTSNode**
Enhanced MCTS node with multi-agent voting support.

**Key Methods:**
```python
node.get_mdap_votes() -> List[ActionVote]
node.get_agent_performance(action: str) -> Dict
node.is_red_flagged() -> bool
node.add_agent_vote(agent_id, action, confidence, rationale, agent_type)
```

**Features:**
- Stores votes from multiple MDAP agents
- Tracks agent performance per action
- Red-flagging for quality control

#### 2. **MDAPMCTSExpansion**
Expansion phase enhanced with MDAP agent voting.

**Key Methods:**
```python
await expansion.expand_with_mdap(node, tree) -> MDAPMCTSNode
await expansion.collect_agent_votes(node) -> List[ActionVote]
expansion.aggregate_votes(votes, strategy) -> Optional[str]
expansion.red_flag_actions(votes) -> List[str]
```

**Features:**
- Multiple agents vote on best action during expansion
- MAKER strategies for vote aggregation (first-k-ahead, majority, weighted)
- Red-flagging filters invalid actions

#### 3. **MDAPMCTSSimulation**
Simulation phase enhanced with MAKER voting.

**Key Methods:**
```python
simulation.simulate_with_maker(state, voters) -> float
simulation.collect_tactic_votes(state, voters) -> List[TacticVote]
simulation.select_tactic_by_voting(votes) -> str
simulation.apply_tactic_with_verification(state, tactic) -> ProofState
```

**Features:**
- Multiple voters propose tactics during rollout
- Voting selects best tactic for each step
- Red-flagging filters invalid tactics

#### 4. **MDAPMCTS**
Main orchestrator combining all components.

**Key Methods:**
```python
await mcts.search_with_mdap(iterations, time_budget) -> MDAPMCTSResult
await mcts.run_iteration_mdap(root) -> None
mcts._select_with_agent_consensus(node) -> MDAPMCTSNode
mcts.backpropagate_with_agent_feedback(node, reward) -> None
```

**Features:**
- Combines intelligent tree search with multi-agent voting
- Red-flagging for pruning low-quality branches
- Agent performance tracking and adaptation

#### 5. **MDAPMCTSConfig**
Configuration combining MCTS, MDAP, and MAKER parameters.

**Parameters:**
```python
# MCTS settings
c_param: float = 1.414              # UCT exploration constant
max_iterations: int = 1000           # Maximum search iterations
rollout_depth: int = 100             # Maximum rollout depth
time_budget: float = 300.0           # Time limit in seconds

# MDAP settings
available_agents: List[str]          # Agent types to use
expansion_agents: int = 3            # Agents voting during expansion
parallel_agents: int = 4             # Parallel agent execution

# MAKER settings
simulation_voters: int = 5           # Voters during simulation
voting_strategy: str = "first_k_ahead"
k_ahead: int = 3                     # K value for first-k-ahead

# Red-flagging
enable_red_flagging: bool = True
prune_red_flagged: bool = True
red_flag_threshold: float = 0.3

# Agent selection
agent_selection_strategy: str = "adaptive"  # adaptive, random, performance_based
```

#### 6. **MDAPMCTSResult**
Result containing proof and comprehensive statistics.

**Fields:**
```python
best_proof: Optional[LeanProof]      # Best proof found
success: bool                        # Whether proof was completed
search_iterations: int               # Iterations performed
time_elapsed: float                  # Time taken
nodes_visited: int                   # Total nodes created
tree_depth: int                      # Maximum tree depth
win_rate: float                      # Estimated success rate
confidence: float                    # Confidence in result

# MDAP-specific
agent_statistics: Dict               # Per-agent performance
voting_statistics: Dict              # Voting metadata
red_flag_analysis: Dict              # Red-flag statistics
agent_performance_ranking: List      # Agents ranked by performance
```

### Algorithm Flow

```
MDAP-Enhanced MCTS Iteration:
┌─────────────────────────────────────────────────────────────┐
│  1. SELECTION (Standard MCTS)                                │
│     - Traverse tree from root using UCT                      │
│     - Balance exploration/exploitation                       │
│     - Select most promising leaf node                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. EXPANSION (MDAP-Enhanced)                                │
│     - Select N agents for voting                             │
│     - Collect votes from each agent                          │
│     - Aggregate votes using MAKER strategy                   │
│     - Red-flag invalid actions                               │
│     - Create child node with selected action                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. SIMULATION (MAKER-Enhanced)                              │
│     - Run rollout from new node                              │
│     - At each step:                                          │
│       * Collect tactic votes from N voters                   │
│       * Select tactic by voting                              │
│       * Apply tactic with verification                        │
│     - Estimate proof completion probability                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. BACKPROPAGATION (Agent Feedback)                          │
│     - Update statistics up the tree                          │
│     - Track agent performance per action                     │
│     - Update action preferences based on success             │
│     - Record which agents voted correctly                    │
└─────────────────────────────────────────────────────────────┘
```

### Usage Examples

#### Basic Usage

```python
import asyncio
from leanaide_mcts_mdap import search_with_mdap_mcts, MDAPMCTSConfig

async def main():
    # Configure the system
    config = MDAPMCTSConfig(
        # MCTS parameters
        c_param=1.414,
        max_iterations=500,
        time_budget=60.0,

        # MDAP parameters
        available_agents=["evolution", "mcts", "adversarial"],
        expansion_agents=3,

        # MAKER parameters
        simulation_voters=5,
        voting_strategy="first_k_ahead",
        k_ahead=3,

        # Red-flagging
        enable_red_flagging=True,
        prune_red_flagged=True
    )

    # Run search
    result = await search_with_mdap_mcts(
        theorem="forall (n m : Nat), n + m = m + n",
        theorem_name="add_comm",
        config=config
    )

    # Access results
    print(f"Success: {result.success}")
    print(f"Proof: {result.best_proof.lean_code}")
    print(f"Agent performance: {result.agent_performance_ranking}")

asyncio.run(main())
```

#### Advanced Configuration

```python
# Custom configuration with all parameters
config = MDAPMCTSConfig(
    # MCTS
    c_param=1.414,
    max_iterations=1000,
    rollout_depth=100,
    time_budget=300.0,

    # MDAP
    available_agents=["evolution", "mcts", "adversarial", "direct"],
    expansion_agents=4,
    parallel_agents=8,

    # MAKER
    simulation_voters=7,
    voting_strategy="weighted",  # or "first_k_ahead", "majority"
    k_ahead=3,

    # Red-flagging
    enable_red_flagging=True,
    prune_red_flagged=True,
    red_flag_threshold=0.3,

    # Agent selection
    agent_selection_strategy="performance_based",

    # LeanAide
    server_url="http://localhost:7654",
    enable_verification=True,

    # Performance
    enable_caching=True,
    cache_size=10000,

    # Logging
    log_level="INFO",
    enable_detailed_logging=False
)
```

### Vote Aggregation Strategies

#### First-K-Ahead (Recommended)
First action to be K votes ahead wins.
```python
config.voting_strategy = "first_k_ahead"
config.k_ahead = 3
```
**Advantages:**
- Fast convergence
- Clear winner
- Less sensitive to noise

#### Majority
Simple majority voting (>50%).
```python
config.voting_strategy = "majority"
```
**Advantages:**
- Democratic
- Robust to outliers
- Simple to understand

#### Weighted
Confidence-weighted voting.
```python
config.voting_strategy = "weighted"
```
**Advantages:**
- Considers agent confidence
- More nuanced decisions
- Better for uncertain situations

### Agent Selection Strategies

#### Adaptive (Recommended)
Selects agents based on node characteristics and diversity.
```python
config.agent_selection_strategy = "adaptive"
```
**Advantages:**
- Balances exploration and exploitation
- Diverse perspectives
- Context-aware

#### Random
Randomly selects agents for each operation.
```python
config.agent_selection_strategy = "random"
```
**Advantages:**
- Unbiased
- Exploratory
- Simple

#### Performance-Based
Selects top-performing agents based on history.
```python
config.agent_selection_strategy = "performance_based"
```
**Advantages:**
- Exploits successful agents
- Faster convergence
- Quality-focused

### Red-Flagging

Red-flagging identifies and filters low-quality actions:

**Triggers:**
- Low average confidence (< threshold)
- High disagreement among agents
- Invalid tactic syntax
- Tactic not applicable to current state

**Configuration:**
```python
config.enable_red_flagging = True
config.prune_red_flagged = True  # Remove red-flagged actions
config.red_flag_threshold = 0.3   # Confidence threshold
```

### Performance Analysis

#### Agent Statistics
```python
for agent_id, stats in result.agent_statistics.items():
    print(f"{agent_id}:")
    print(f"  Votes cast: {stats['votes_cast']}")
    print(f"  Votes accepted: {stats['votes_accepted']}")
    print(f"  Success rate: {stats['success_rate']:.3f}")
```

#### Performance Ranking
```python
for agent_id, success_rate in result.agent_performance_ranking:
    print(f"{agent_id}: {success_rate:.3f}")
```

#### Voting Statistics
```python
print(f"Total votes: {result.voting_statistics['total_agent_votes']}")
print(f"Accepted votes: {result.voting_statistics['accepted_votes']}")
print(f"Agents used: {result.voting_statistics['agents_used']}")
```

#### Red-Flag Analysis
```python
print(f"Red-flagged nodes: {result.red_flag_analysis['red_flagged_nodes']}")
print(f"Red-flag rate: {result.red_flag_analysis['red_flag_rate']:.3f}")
```

### Integration with Existing Systems

#### With MDAP
```python
from leanaide_mdap import LeanProofAgent, LeanMDAPOrchestrator

# MDAP agents can be used directly in MDAP-MCTS
agents = [LeanProofAgent(...) for _ in range(3)]
expansion = MDAPMCTSExpansion(config, agents=agents)
```

#### With MAKER
```python
from leanaide_maker import LeanTacticVoter, HeuristicVoter

# MAKER voters can be used in simulation
voters = [HeuristicVoter(...) for _ in range(5)]
reward = simulation.simulate_with_maker(state, voters)
```

### Best Practices

1. **Start with conservative settings:**
   ```python
   config = MDAPMCTSConfig(
       max_iterations=100,  # Start small
       expansion_agents=2,   # Few agents
       simulation_voters=3   # Few voters
   )
   ```

2. **Enable red-flagging for quality control:**
   ```python
   config.enable_red_flagging = True
   config.prune_red_flagged = True
   ```

3. **Use first-k-ahead for fast convergence:**
   ```python
   config.voting_strategy = "first_k_ahead"
   config.k_ahead = 3
   ```

4. **Monitor agent performance:**
   - Check `agent_performance_ranking` after search
   - Adjust `available_agents` based on performance
   - Use `performance_based` selection for exploitation

5. **Balance exploration/exploitation:**
   - Higher `c_param` = more exploration
   - Lower `c_param` = more exploitation
   - Recommended: `c_param = 1.414` (sqrt(2))

### Troubleshooting

**Problem: Slow convergence**
- Solution: Increase `k_ahead` for faster consensus
- Solution: Use `performance_based` agent selection
- Solution: Reduce `expansion_agents`

**Problem: Low success rate**
- Solution: Increase `simulation_voters` for better rollouts
- Solution: Enable red-flagging to filter bad actions
- Solution: Increase `rollout_depth` for deeper search

**Problem: High memory usage**
- Solution: Reduce `max_iterations`
- Solution: Enable `prune_red_flagged`
- Solution: Reduce cache size

**Problem: Red-flagging too aggressive**
- Solution: Lower `red_flag_threshold`
- Solution: Disable `prune_red_flagged`
- Solution: Increase agent confidence calibration

### File Structure

```
leanaide_mcts_mdap.py
├── Configuration (MDAPMCTSConfig)
├── Data Classes (ActionVote, MDAPMCTSResult)
├── MDAPMCTSNode (enhanced node with voting)
├── MDAPMCTSExpansion (MDAP-enhanced expansion)
├── MDAPMCTSSimulation (MAKER-enhanced simulation)
├── MDAPMCTSTree (tree management)
├── MDAPMCTS (main orchestrator)
└── Convenience Functions
```

### Dependencies

**Required:**
- Python 3.8+
- asyncio, dataclasses, typing, logging

**Optional (for full functionality):**
- `leanaide_mcts`: MCTS implementation
- `leanaide_mdap`: MDAP multi-agent system
- `leanaide_maker`: MAKER voting system
- `leanaide_evolution`: Evolutionary agents

**Fallback Mode:**
If optional dependencies are not available, the system will:
- Use simulated agent votes
- Use heuristic simulation
- Use basic node selection

### Performance Considerations

**Time Complexity:**
- Selection: O(tree_depth)
- Expansion: O(num_agents × agent_time)
- Simulation: O(rollout_depth × num_voters)
- Backpropagation: O(tree_depth)
- Total per iteration: O(tree_depth + num_agents × agent_time + rollout_depth × num_voters)

**Space Complexity:**
- Tree storage: O(num_nodes × state_size)
- Agent statistics: O(num_agents)
- Vote storage: O(num_nodes × num_votes)

**Parallelization:**
- Agent voting: Parallel by default
- Simulation voters: Can be parallelized
- Tree operations: Sequential (MCTS requirement)

**Optimization Tips:**
1. Use `parallel_agents` for faster voting
2. Enable `enable_caching` for state reuse
3. Limit `rollout_depth` for faster simulations
4. Use `adaptive` agent selection for balance

### Example Output

```
================================================================================
MDAP-MCTS Integration Example
================================================================================

Theorem: forall (n m : Nat), n + m = m + n


================================================================================
Results
================================================================================

Success: True
Iterations: 500
Time: 45.23s
Nodes visited: 1247
Tree depth: 23
Win rate: 0.8745
Confidence: 0.8234

================================================================================
Agent Statistics
================================================================================

evolution_agent:
  Votes cast: 342
  Votes accepted: 287
  Success rate: 0.839

mcts_agent:
  Votes cast: 298
  Votes accepted: 265
  Success rate: 0.889

adversarial_agent:
  Votes cast: 287
  Votes accepted: 234
  Success rate: 0.815

================================================================================
Voting Statistics
================================================================================

Total votes: 927
Accepted votes: 786
Agents used: 3

================================================================================
Red Flag Analysis
================================================================================

Red-flagged nodes: 45
Red-flag rate: 0.036

================================================================================
Best Proof
================================================================================

import Mathlib

theorem add_comm : forall (n m : Nat), n + m = m + n := by
  intros n m
  induction n
  . simp
  . intro n_ih
    rw [n_ih]
```

### API Reference

See inline documentation in `leanaide_mcts_mdap.py` for detailed API reference including:

- Class constructors and parameters
- Method signatures and return types
- Exception handling
- Usage examples for each component

### License

This code is part of the OpenEvolve project. See main project LICENSE file for details.

### Support

For issues, questions, or contributions:
- GitHub: [OpenEvolve repository]
- Documentation: [OpenEvolve docs]
- Examples: See `example_usage()` function in source file
