# BubbleLabs-LeanAide Integration Guide

## Overview

This integration connects **BubbleLabs** workflow visualization with **LeanAide** formal verification capabilities, enabling interactive theorem proving, MCTS visualization, and Lean4 code verification within the BubbleLabs UI.

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Component Reference](#component-reference)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## Features

### 🧮 LeanAide Task Execution
- **Theorem Translation**: Convert natural language theorems to Lean code
- **Proof Generation**: Generate formal proofs automatically
- **Code Verification**: Verify Lean code correctness
- **Math Queries**: Ask mathematical questions and get answers

### 🌳 MCTS Visualization
- **Interactive Tree Display**: Visualize Monte Carlo Tree Search proofs
- **Node Statistics**: View visits, values, and win rates for each node
- **Best Path Highlight**: See the highest-quality proof path
- **Agent Performance**: Track MDAP agent voting statistics

### ✅ Lean4 Verification
- **Step-by-Step Tracking**: Monitor each proof step
- **Goal Display**: See goals before and after each tactic
- **Error Reporting**: Get detailed error messages for failed proofs
- **Proof State**: View complete Lean4 proof state

### 🎯 MDAP Integration
- **Multi-Agent Voting**: See votes from different agent types
- **Decision Aggregation**: View how votes are combined
- **Red-Flag Analysis**: Identify low-confidence decisions
- **Performance Ranking**: Compare agent effectiveness

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BubbleLabs UI                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Workflow   │  │   MCTS       │  │   Lean4      │      │
│  │   Designer   │  │   Visualizer │  │   Verifier   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              LeanAide Integration Bridge                     │
│  - Task execution                                            │
│  - Visualization data generation                             │
│  - Thread-safe operations                                    │
│  - Error handling                                            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    LeanAide Components                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   MCTS-MDAP  │  │   MCP Tools  │  │   LeanAide   │      │
│  │   Engine     │  │              │  │   Client     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

```bash
# Python dependencies
pip install BubbleLab UI pandas asyncio aiohttp

# LeanAide components (optional but recommended)
pip install leanaide-client
pip install leanaide-mcts-mdap
```

### Setup

1. **Install Integration Files**:
   Place these files in your OpenEvolve Frontend directory:
   - `bubblelabs_leanaide_integration.py` - Core integration bridge
   - `bubblelabs_leanaide_ui.py` - UI components

2. **Configure Environment** (optional):
   ```bash
   export LEANAIDE_HOST="localhost"
   export LEANAIDE_PORT="7654"
   ```

3. **Initialize Integration**:
   ```python
   from bubblelabs_leanaide_integration import initialize_leanaide_integration

   status = initialize_leanaide_integration()
   print(status)
   ```

## Quick Start

### Basic Usage

```python
import BubbleLab UI as st
from bubblelabs_leanaide_ui import render_leanaide_in_bubblelabs

# In your BubbleLabs app
def main():
    st.title("My BubbleLabs Workflow")

    # Add LeanAide tab
    render_leanaide_in_bubblelabs()

if __name__ == "__main__":
    main()
```

### Using the Integration Bridge

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

# Get bridge instance
bridge = get_leanaide_bridge()

# Execute a theorem translation
result = bridge.execute_task(
    LeanAideTaskType.TRANSLATE_THEOREM,
    theorem_text="There are infinitely many primes",
    theorem_name="infinitely_many_primes"
)

if result.success:
    print(f"Lean code: {result.data['lean_code']}")
```

## Component Reference

### LeanAideIntegrationBridge

Main bridge class for LeanAide integration.

#### Methods

##### `execute_task(task_type, **kwargs)`
Execute a LeanAide task with BubbleLabs integration.

**Parameters:**
- `task_type` (LeanAideTaskType): Type of task to execute
- `**kwargs`: Task-specific parameters

**Returns:** `LeanAideExecutionResult`

**Example:**
```python
result = bridge.execute_task(
    LeanAideTaskType.MCTS_SEARCH,
    theorem="forall n m : Nat, n + m = m + n",
    max_iterations=1000,
    time_budget=60.0
)
```

##### `get_status()`
Get status of LeanAide integration.

**Returns:** `Dict[str, Any]` with status information

##### `get_tree(tree_id)`
Get MCTS tree visualization by ID.

##### `get_proof(proof_id)`
Get Lean4 proof visualization by ID.

##### `get_execution_history(limit=50)`
Get recent execution history.

### LeanAideUIComponent

BubbleLab UI UI component for LeanAide integration.

#### Methods

##### `render_leanaide_control_panel()`
Render main LeanAide control panel with all tabs.

##### `_render_theorem_proving_panel()`
Render theorem proving interface.

##### `_render_mcts_visualization()`
Render MCTS tree visualization.

##### `_render_lean4_verification()`
Render Lean4 code verification panel.

##### `_render_math_queries()`
Render mathematical query interface.

## Usage Examples

### Example 1: Translate and Prove a Theorem

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

bridge = get_leanaide_bridge()

# Step 1: Translate theorem
translation = bridge.execute_task(
    LeanAideTaskType.TRANSLATE_THEOREM,
    theorem_text="The square root of 2 is irrational",
    theorem_name="sqrt2_irrational"
)

print(f"Generated Lean: {translation.data['lean_code']}")

# Step 2: Generate proof
proof = bridge.execute_task(
    LeanAideTaskType.GENERATE_PROOF,
    theorem_text="The square root of 2 is irrational",
    theorem_code=translation.data['lean_code']
)

print(f"Proof: {proof.data['proof_document']}")
```

### Example 2: MCTS Proof Search

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

bridge = get_leanaide_bridge()

# Run MCTS search
result = bridge.execute_task(
    LeanAideTaskType.MCTS_SEARCH,
    theorem="forall (n m : Nat), n * m = m * n",
    theorem_name="mul_comm",
    max_iterations=500,
    time_budget=30.0,
    expansion_agents=3,
    simulation_voters=5
)

if result.success:
    # Get tree visualization
    tree_id = result.visualization_data['tree_id']
    tree = bridge.get_tree(tree_id)

    print(f"Tree has {len(tree.nodes)} nodes")
    print(f"Best path: {tree.best_path}")
    print(f"Win rate: {tree.statistics['win_rate']:.2%}")
```

### Example 3: Verify Lean Code

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

bridge = get_leanaide_bridge()

lean_code = """
theorem add_comm (a b : Nat) : a + b = b + a := by
  simp [Nat.add_comm]
"""

result = bridge.execute_task(
    LeanAideTaskType.VERIFY_SOLUTION,
    code=lean_code
)

if result.success:
    print(f"Valid: {result.data['is_valid']}")
    print(f"Unproven obligations: {result.data['unproven_count']}")
else:
    print(f"Verification failed: {result.error}")
```

### Example 4: Mathematical Query

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

bridge = get_leanaide_bridge()

result = bridge.execute_task(
    LeanAideTaskType.MATH_QUERY,
    query="What is the fundamental theorem of algebra?",
    n=3
)

if result.success:
    for i, answer in enumerate(result.data['answers'], 1):
        print(f"Answer {i}: {answer}")
```

### Example 5: BubbleLab UI UI Integration

```python
import BubbleLab UI as st
from bubblelabs_leanaide_ui import LeanAideUIComponent

def main():
    st.title("My Workflow Designer")

    # Add LeanAide control panel
    ui = LeanAideUIComponent()
    ui.render_leanaide_control_panel()

if __name__ == "__main__":
    main()
```

## API Reference

### LeanAideTaskType

Enumeration of available task types:

- `TRANSLATE_THEOREM` - Translate natural language to Lean
- `GENERATE_PROOF` - Generate a proof
- `VERIFY_SOLUTION` - Verify Lean code
- `MATH_QUERY` - Answer math questions
- `ELABORATE_CODE` - Elaborate and check Lean code
- `MCTS_SEARCH` - Run MCTS proof search

### Data Classes

#### LeanAideExecutionResult
```python
@dataclass
class LeanAideExecutionResult:
    task_type: LeanAideTaskType
    success: bool
    data: Optional[Dict[str, Any]]
    execution_time: float
    error: Optional[str]
    visualization_data: Optional[Dict[str, Any]]
    timestamp: str
```

#### MCTSNodeVisualization
```python
@dataclass
class MCTSNodeVisualization:
    node_id: str
    parent_id: Optional[str]
    action: str
    visits: int
    value: float
    win_rate: float
    depth: int
    is_terminal: bool
    children: List[str]
    agent_votes: List[Dict]
    red_flagged: bool
    hash: str
```

#### Lean4ProofStep
```python
@dataclass
class Lean4ProofStep:
    step_id: str
    step_number: int
    tactic: str
    goals_before: List[str]
    goals_after: List[str]
    proof_state: str
    is_valid: bool
    error_message: Optional[str]
    timestamp: str
```

## Troubleshooting

### LeanAide Server Not Responding

**Problem:** Tasks fail with connection errors

**Solutions:**
1. Check LeanAide server is running:
   ```bash
   curl http://localhost:7654
   ```

2. Verify server configuration:
   ```python
   status = bridge.get_status()
   print(status['server_status'])
   ```

3. Check firewall settings

### MCTS Not Available

**Problem:** MCTS search button disabled or fails

**Solutions:**
1. Install required dependencies:
   ```bash
   pip install leanaide-mcts-mdap
   ```

2. Check availability:
   ```python
   from bubblelabs_leanaide_integration import MCTS_AVAILABLE
   print(f"MCTS Available: {MCTS_AVAILABLE}")
   ```

### Import Errors

**Problem:** `ImportError: No module named 'leanaide_client'`

**Solutions:**
1. Install LeanAide client:
   ```bash
   pip install -e /path/to/leanaide/client
   ```

2. Add to Python path:
   ```python
   import sys
   sys.path.append('/path/to/leanaide')
   ```

### Thread Safety Issues

**Problem:** Concurrent task executions cause errors

**Solutions:**
The integration bridge is thread-safe. Use the singleton instance:
```python
from bubblelabs_leanaide_integration import get_leanaide_bridge

# Always use get_leanaide_bridge() for thread-safe access
bridge = get_leanaide_bridge()
```

### Performance Issues

**Problem:** Tasks take too long

**Solutions:**
1. Adjust timeout:
   ```python
   result = bridge.execute_task(
       LeanAideTaskType.MCTS_SEARCH,
       theorem="...",
       time_budget=30.0  # Reduce time budget
   )
   ```

2. Reduce iterations:
   ```python
   result = bridge.execute_task(
       LeanAideTaskType.MCTS_SEARCH,
       theorem="...",
       max_iterations=100  # Fewer iterations
   )
   ```

3. Use parallel execution:
   ```python
   import concurrent.futures

   with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
       futures = [
           executor.submit(bridge.execute_task, LeanAideTaskType.MATH_QUERY, query=q)
           for q in queries
       ]
       results = [f.result() for f in futures]
   ```

## Advanced Usage

### Custom MDAP Configuration

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

bridge = get_leanaide_bridge()

# Custom MCTS-MDAP configuration
result = bridge.execute_task(
    LeanAideTaskType.MCTS_SEARCH,
    theorem="...",
    c_param=1.5,           # UCB exploration parameter
    expansion_agents=5,    # More agents during expansion
    simulation_voters=10,  # More voters during simulation
    enable_red_flagging=True,
    red_flag_threshold=0.3
)
```

### Accessing Visualization Data

```python
# Get tree visualization
tree_id = result.visualization_data['tree_id']
tree = bridge.get_tree(tree_id)

# Export as JSON
import json
tree_json = tree.to_dict()
with open('mcts_tree.json', 'w') as f:
    json.dump(tree_json, f, indent=2)

# Access specific nodes
root_node = tree.nodes[tree.root_id]
print(f"Root visits: {root_node.visits}")
print(f"Root value: {root_node.value}")

# Iterate through best path
for node_id in tree.best_path:
    node = tree.nodes[node_id]
    print(f"Step {node.depth}: {node.action} (win_rate={node.win_rate:.2f})")
```

### Batch Operations

```python
from bubblelabs_leanaide_integration import get_leanaide_bridge, LeanAideTaskType

bridge = get_leanaide_bridge()

# Batch translate theorems
theorems = [
    ("There are infinitely many primes", "inf_primes"),
    ("sqrt(2) is irrational", "sqrt2_irrational"),
    ("Every natural number has a prime factorization", "prime_factor")
]

results = []
for theorem, name in theorems:
    result = bridge.execute_task(
        LeanAideTaskType.TRANSLATE_THEOREM,
        theorem_text=theorem,
        theorem_name=name
    )
    results.append(result)

# Check results
for i, result in enumerate(results):
    status = "✓" if result.success else "✗"
    print(f"{status} {theorems[i][0]}: {result.execution_time:.2f}s")
```

## Best Practices

1. **Always check availability** before using features:
   ```python
   from bubblelabs_leanaide_integration import MCTS_AVAILABLE, MDAP_AVAILABLE

   if MCTS_AVAILABLE:
       # Use MCTS features
   ```

2. **Handle errors gracefully**:
   ```python
   result = bridge.execute_task(...)
   if not result.success:
       print(f"Task failed: {result.error}")
       # Implement fallback logic
   ```

3. **Use appropriate timeouts**:
   - Simple tasks: 10-30 seconds
   - Proof generation: 60-120 seconds
   - MCTS search: 300+ seconds

4. **Clean up resources** when done:
   ```python
   bridge.cleanup()
   ```

5. **Monitor execution history** for debugging:
   ```python
   history = bridge.get_execution_history(limit=10)
   for result in history:
       print(f"{result.task_type}: {result.success}")
   ```

## Contributing

To extend this integration:

1. Add new task types to `LeanAideTaskType` enum
2. Implement handler in `LeanAideIntegrationBridge._execute_*`
3. Create visualization data class if needed
4. Add UI panel in `LeanAideUIComponent`
5. Update documentation

## License

This integration is part of the OpenEvolve project. See main project LICENSE for details.

## Support

For issues or questions:
- Check troubleshooting section above
- Review LeanAide documentation
- Open an issue on the OpenEvolve repository

