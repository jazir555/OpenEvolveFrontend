# LeanAide MCTS API Reference

> **STATUS: implemented** (see `integrations/leanaide/leanaide_mcts.py` — `MCTSConfig`, `MCTSResult`, `ProofState`, `MCTSNode`, `MCTSTree`, `MCTSSelection`, `MCTSExpansion`, `MCTSSimulation`, `MCTSBackpropagation`, `MCTS`; plus `integrations/leanaide/leanaide_mcts_strategies.py`).
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Structures](#data-structures)
3. [Enums](#enums)
4. [Configuration](#configuration)
5. [Methods](#methods)
6. [Utility Functions](#utility-functions)
7. [Type Signatures](#type-signatures)

---

## Core Classes

### MCTS

The main MCTS algorithm implementation.

```python
class MCTS:
    """Monte Carlo Tree Search for tactical proof search"""

    def __init__(
        self,
        exploration_constant: float = 1.414,
        rollout_depth: int = 10,
        rollout_episodes: int = 1,
        value_normalization: float = 1.0,
        discount_factor: float = 0.99
    )
```

**Parameters**:
- `exploration_constant` (float): UCB exploration constant (default: √2)
- `rollout_depth` (int): Maximum depth for rollout simulations (default: 10)
- `rollout_episodes` (int): Number of rollout episodes per expansion (default: 1)
- `value_normalization` (float): Normalize Q values (default: 1.0)
- `discount_factor` (float): Discount future values in rollouts (default: 0.99)

**Attributes**:
- `exploration_constant` (float): Current exploration constant
- `rollout_depth` (int): Current rollout depth limit
- `rollout_episodes` (int): Current rollout episodes
- `value_normalization` (float): Current value normalization
- `discount_factor` (float): Current discount factor
- `total_simulations` (int): Total simulations run
- `total_time` (float): Total time spent in simulations
- `tree_depth` (int): Maximum tree depth achieved

**Methods**:

#### select()

Select a leaf node using UCB traversal.

```python
def select(
    self,
    node: MCTSNode,
    value_normalization: Optional[float] = None
) -> MCTSNode
```

**Parameters**:
- `node` (MCTSNode): Root node to start selection from
- `value_normalization` (Optional[float]): Override for value normalization

**Returns**:
- (MCTSNode): Selected leaf node

**Example**:
```python
mcts = MCTS(exploration_constant=1.414)
leaf = mcts.select(root_node)
```

---

#### expand()

Expand a leaf node with available actions.

```python
def expand(
    self,
    node: MCTSNode,
    available_actions: List[TacticAction],
    policy_probs: Optional[Dict[str, float]] = None
) -> MCTSNode
```

**Parameters**:
- `node` (MCTSNode): Node to expand
- `available_actions` (List[TacticAction]): Actions to add as children
- `policy_probs` (Optional[Dict[str, float]]): Policy probabilities for actions

**Returns**:
- (MCTSNode): Expanded child node or original if not expandable

**Example**:
```python
actions = generate_actions(node.state)
child = mcts.expand(node, actions)
```

---

#### simulate()

Run rollout simulation from a node.

```python
def simulate(
    self,
    node: MCTSNode,
    action_generator: Callable[[ProofContext], List[TacticAction]],
    state_evaluator: Callable[[ProofContext], float]
) -> float
```

**Parameters**:
- `node` (MCTSNode): Node to run rollout from
- `action_generator` (Callable): Function to generate available actions
- `state_evaluator` (Callable): Function to evaluate state quality

**Returns**:
- (float): Simulated value (0.0 to 1.0)

**Example**:
```python
value = mcts.simulate(
    node,
    action_generator=lambda ctx: get_applicable_tactics(ctx),
    state_evaluator=lambda ctx: evaluate_proof_state(ctx)
)
```

---

#### backpropagate()

Backpropagate value from node to root.

```python
def backpropagate(
    self,
    node: MCTSNode,
    value: float
) -> None
```

**Parameters**:
- `node` (MCTSNode): Node to start backpropagation from
- `value` (float): Value to propagate

**Returns**:
- None

**Example**:
```python
mcts.backpropagate(node, reward=0.8)
```

---

#### run_simulation()

Run one complete MCTS simulation.

```python
def run_simulation(
    self,
    root: MCTSNode,
    action_generator: Callable[[ProofContext], List[TacticAction]],
    state_evaluator: Callable[[ProofContext], float],
    policy_network: Optional[Callable[[ProofContext], Dict[str, float]]] = None
) -> MCTSNode
```

**Parameters**:
- `root` (MCTSNode): Root node of the tree
- `action_generator` (Callable): Function to generate available actions
- `state_evaluator` (Callable): Function to evaluate state quality
- `policy_network` (Optional[Callable]): Policy network for prior probabilities

**Returns**:
- (MCTSNode): Final node of this simulation

**Example**:
```python
node = mcts.run_simulation(
    root,
    action_generator=my_action_generator,
    state_evaluator=my_evaluator,
    policy_network=None
)
```

---

#### get_best_child()

Get best child from a node.

```python
def get_best_child(
    self,
    node: MCTSNode,
    select_by_visit_count: bool = True
) -> Optional[MCTSNode]
```

**Parameters**:
- `node` (MCTSNode): Parent node
- `select_by_visit_count` (bool): If True, select most visited; otherwise select highest value

**Returns**:
- (Optional[MCTSNode]): Best child node or None

**Example**:
```python
best = mcts.get_best_child(root, select_by_visit_count=True)
```

---

#### get_tree_statistics()

Get statistics about the MCTS tree.

```python
def get_tree_statistics(self, root: MCTSNode) -> Dict[str, Any]
```

**Parameters**:
- `root` (MCTSNode): Root node of the tree

**Returns**:
- (Dict[str, Any]): Tree statistics dictionary with keys:
  - `total_nodes` (int): Total number of nodes
  - `max_depth` (int): Maximum tree depth
  - `average_depth` (float): Average node depth
  - `terminal_nodes` (int): Number of terminal nodes
  - `root_visits` (int): Root visit count
  - `root_value` (float): Root average value

**Example**:
```python
stats = mcts.get_tree_statistics(root)
print(f"Total nodes: {stats['total_nodes']}")
print(f"Max depth: {stats['max_depth']}")
```

---

### MCTSNode

Represents a node in the MCTS search tree.

```python
class MCTSNode:
    """A node in the MCTS search tree"""

    def __init__(
        self,
        state: ProofContext,
        parent: Optional['MCTSNode'] = None,
        action: Optional[TacticAction] = None,
        prior_probability: float = 0.0
    )
```

**Parameters**:
- `state` (ProofContext): Proof state this node represents
- `parent` (Optional[MCTSNode]): Parent node
- `action` (Optional[TacticAction]): Action that led to this node
- `prior_probability` (float): Prior probability from policy

**Attributes**:
- `state` (ProofContext): Associated proof state
- `parent` (Optional[MCTSNode]): Parent node reference
- `action` (Optional[TacticAction]): Action that created this node
- `visit_count` (int): Number of times node was visited
- `total_value` (float): Total value accumulated
- `prior_probability` (float): Prior probability
- `children` (Dict[str, MCTSNode]): Child nodes
- `unexplored_actions` (List[TacticAction]): Actions not yet explored
- `is_terminal` (bool): Whether node is terminal
- `is_fully_expanded` (bool): Whether all actions explored
- `value` (Optional[float]): Cached value for terminal nodes
- `node_id` (str): Unique node identifier
- `creation_time` (float): Node creation timestamp
- `depth` (int): Depth in tree

**Properties**:

#### is_leaf

Check if node is a leaf (no children).

```python
@property
def is_leaf(self) -> bool
```

**Returns**:
- (bool): True if node has no children

---

#### average_value

Get average value (Q-value) of this node.

```python
@property
def average_value(self) -> float
```

**Returns**:
- (float): Average value (total_value / visit_count)

---

#### is_root

Check if node is the root.

```python
@property
def is_root(self) -> bool
```

**Returns**:
- (bool): True if node has no parent

**Methods**:

#### get_ucb_score()

Calculate UCB score for node selection.

```python
def get_ucb_score(
    self,
    exploration_constant: float = 1.414,
    value_normalization: float = 1.0
) -> float
```

**Parameters**:
- `exploration_constant` (float): UCB exploration constant
- `value_normalization` (float): Value normalization factor

**Returns**:
- (float): UCB score

**Formula**:
```
UCB = Q + c * P * sqrt(N_parent) / (1 + N)
```

---

#### select_child()

Select best child using UCB score.

```python
def select_child(
    self,
    exploration_constant: float = 1.414,
    value_normalization: float = 1.0
) -> Optional['MCTSNode']
```

**Parameters**:
- `exploration_constant` (float): UCB exploration constant
- `value_normalization` (float): Value normalization factor

**Returns**:
- (Optional[MCTSNode]): Best child or None if no children

---

#### add_child()

Add a child node.

```python
def add_child(
    self,
    action: TacticAction,
    state: ProofContext,
    prior_probability: float = 0.0
) -> 'MCTSNode'
```

**Parameters**:
- `action` (TacticAction): Action that leads to child
- `state` (ProofContext): Resulting proof state
- `prior_probability` (float): Prior probability for child

**Returns**:
- (MCTSNode): Created child node

---

#### update()

Update node statistics with new value.

```python
def update(self, value: float) -> None
```

**Parameters**:
- `value` (float): Value to add (typically in [0, 1])

**Effects**:
- Increments `visit_count` by 1
- Adds `value` to `total_value`

---

#### get_path()

Get path from root to this node.

```python
def get_path(self) -> List['MCTSNode']
```

**Returns**:
- (List[MCTSNode]): List of nodes from root to this node (inclusive)

---

#### to_dict()

Convert node to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

**Returns**:
- (Dict[str, Any]): Dictionary representation with all node data

---

### LeanProofMCTS

Lean 4 specific MCTS implementation.

```python
class LeanProofMCTS:
    """Lean 4 specific MCTS for proof generation"""

    def __init__(
        self,
        exploration_constant: float = 1.414,
        simulations: int = 1000,
        rollout_depth: int = 5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    )
```

**Parameters**:
- `exploration_constant` (float): UCB exploration constant
- `simulations` (int): Number of MCTS simulations
- `rollout_depth` (int): Maximum rollout depth
- `temperature` (float): Temperature for action selection
- `dirichlet_alpha` (float): Dirichlet noise alpha parameter
- `dirichlet_epsilon` (float): Dirichlet noise mixing weight

**Attributes**:
- `mcts` (MCTS): Underlying MCTS algorithm
- `simulations` (int): Number of simulations to run
- `temperature` (float): Action selection temperature
- `dirichlet_alpha` (float): Dirichlet noise alpha
- `dirichlet_epsilon` (float): Dirichlet noise epsilon
- `tactic_library` (Dict[str, Tactic]): Available tactics
- `statistics` (Dict[str, Any]): Search statistics

**Class Attributes**:

#### LEAN_TACTICS

List of available Lean 4 tactics.

```python
LEAN_TACTICS: List[Tactic] = [
    Tactic(name="simp", category="simplification", is_safe=True),
    Tactic(name="rw", category="rewrite"),
    # ... more tactics
]
```

**Methods**:

#### search()

Run MCTS search for a proof.

```python
def search(
    self,
    initial_context: ProofContext,
    lean_client=None
) -> Tuple[List[TacticAction], MCTSNode]
```

**Parameters**:
- `initial_context` (ProofContext): Initial proof context
- `lean_client` (Optional): Lean 4 client for tactic application

**Returns**:
- (Tuple[List[TacticAction], MCTSNode]): Best action sequence and root node

**Example**:
```python
mcts = LeanProofMCTS(simulations=1000)
sequence, root = mcts.search(initial_context)
```

---

#### _generate_actions()

Generate available actions for a context.

```python
def _generate_actions(self, context: ProofContext) -> List[TacticAction]
```

**Parameters**:
- `context` (ProofContext): Current proof context

**Returns**:
- (List[TacticAction]): Applicable tactic actions

**Internal method**: Called automatically during search

---

#### _evaluate_state()

Evaluate proof state quality.

```python
def _evaluate_state(
    self,
    context: ProofContext,
    lean_client=None
) -> float
```

**Parameters**:
- `context` (ProofContext): Proof context to evaluate
- `lean_client` (Optional): Lean 4 client

**Returns**:
- (float): State value in [0, 1] where 1 is best

**Evaluation criteria**:
- Terminal state (empty goal): 1.0
- Depth penalty: -0.01 per depth level
- Hypothesis bonus: +0.02 per hypothesis
- Lemma bonus: +0.01 per available lemma

---

#### _add_dirichlet_noise()

Add Dirichlet noise for exploration.

```python
def _add_dirichlet_noise(self, node: MCTSNode) -> None
```

**Parameters**:
- `node` (MCTSNode): Node to add noise to

**Effects**:
- Modifies `prior_probability` of `unexplored_actions`

**Internal method**: Called during root initialization

---

#### _extract_best_sequence()

Extract best action sequence from tree.

```python
def _extract_best_sequence(self, root: MCTSNode) -> List[TacticAction]
```

**Parameters**:
- `root` (MCTSNode): Root node of MCTS tree

**Returns**:
- (List[TacticAction]): Best proof sequence found

**Selection**: Chooses most visited child at each level

---

#### get_action_probabilities()

Get action probabilities from MCTS.

```python
def get_action_probabilities(
    self,
    root: MCTSNode,
    temperature: float = 1.0
) -> Dict[str, float]
```

**Parameters**:
- `root` (MCTSNode): Root node
- `temperature` (float): Temperature for softmax

**Returns**:
- (Dict[str, float]): Action ID to probability mapping

**Formula**:
- If temperature == 0: Select most visited (deterministic)
- Otherwise: Softmax over visit counts

---

#### get_statistics()

Get MCTS search statistics.

```python
def get_statistics(self) -> Dict[str, Any]
```

**Returns**:
- (Dict[str, Any]): Statistics dictionary with keys:
  - `total_searches` (int): Total searches performed
  - `successful_proofs` (int): Number of proofs found
  - `average_depth` (float): Average proof depth
  - `average_time` (float): Average search time

---

## Data Structures

### ProofContext

Represents a proof state in Lean 4.

```python
@dataclass
class ProofContext:
    """Represents the current proof context"""

    goal: str  # Current goal to prove
    hypotheses: List[str]  # Available hypotheses
    available_lemmas: List[str]  # Available lemmas/theorems
    dependencies: List[str]  # Required dependencies
    depth: int = 0  # Proof depth
    parent_id: Optional[str] = None  # Parent state ID
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

**Methods**:

#### to_dict()

Convert to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

---

### Tactic

Represents a Lean 4 tactic.

```python
@dataclass
class Tactic:
    """Represents a Lean 4 tactic"""

    name: str  # Tactic name
    arguments: List[str]  # Tactic arguments
    category: str  # Tactic category
    applicability_score: float  # Applicability (0-1)
    success_rate: float  # Historical success rate
    avg_time: float  # Average execution time
    is_safe: bool  # Safe tactics don't fail
    metadata: Dict[str, Any]  # Additional metadata
```

**Methods**:

#### to_dict()

Convert to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

---

### TacticAction

Represents an action in the MCTS search space.

```python
@dataclass
class TacticAction:
    """Represents a tactic application"""

    tactic: Tactic  # The tactic to apply
    context: ProofContext  # Context to apply in
    action_id: str  # Unique action identifier
    estimated_value: float  # Estimated value
    prior_probability: float  # Prior probability
```

**Methods**:

#### to_dict()

Convert to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

---

### MCTSResult

Result of an MCTS search.

```python
@dataclass
class MCTSResult:
    """Result of MCTS search"""

    success: bool  # Whether search succeeded
    best_sequence: List[TacticAction]  # Best action sequence
    root_node: MCTSNode  # Root node of tree
    search_time: float  # Search time in seconds
    num_simulations: int  # Number of simulations
    tree_statistics: Dict[str, Any]  # Tree statistics
    proof_found: bool  # Whether proof was found
    proof_length: int  # Length of proof
```

**Methods**:

#### to_dict()

Convert to dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

**Returns**:
- Dictionary with all result fields plus `tactic_sequence` (list of tactic names)

---

## Enums

### TacticStatus

Status of a tactic application.

```python
class TacticStatus(Enum):
    PENDING = "pending"
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    SUCCESS = "success"
    FAILED = "failed"
```

### ProofState

State of proof search.

```python
class ProofState(Enum):
    IN_PROGRESS = "in_progress"
    PROVED = "proved"
    STUCK = "stuck"
    BRANCHED = "branched"
    CONTRADICTORY = "contradictory"
```

---

## Configuration

### Default Values

```python
DEFAULT_EXPLORATION_CONSTANT = 1.414  # sqrt(2)
DEFAULT_ROLLOUT_DEPTH = 10
DEFAULT_ROLLOUT_EPISODES = 1
DEFAULT_VALUE_NORMALIZATION = 1.0
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_SIMULATIONS = 1000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_DIRICHLET_EPSILON = 0.25
```

---

## Methods

### Utility Functions

#### run_mcts_search()

Convenience function to run MCTS search.

```python
def run_mcts_search(
    theorem_statement: str,
    context: ProofContext,
    simulations: int = 1000,
    exploration_constant: float = 1.414
) -> MCTSResult
```

**Parameters**:
- `theorem_statement` (str): Theorem to prove
- `context` (ProofContext): Initial proof context
- `simulations` (int): Number of MCTS simulations
- `exploration_constant` (float): UCB exploration constant

**Returns**:
- (MCTSResult): Search results

**Example**:
```python
result = run_mcts_search(
    theorem_statement="∀ n : Nat, n + 0 = n",
    context=proof_context,
    simulations=500
)
print(f"Proof found: {result.proof_found}")
print(f"Time: {result.search_time:.2f}s")
```

---

## Type Signatures

### Common Type Aliases

```python
ActionGenerator = Callable[[ProofContext], List[TacticAction]]
StateEvaluator = Callable[[ProofContext], float]
PolicyNetwork = Callable[[ProofContext], Dict[str, float]]
TacticTransition = Callable[[ProofContext, Tactic], ProofContext]
```

### Return Types

```python
SearchResult = Tuple[List[TacticAction], MCTSNode]
TreeStats = Dict[str, Any]
ActionProbabilities = Dict[str, float]
```

---

## Complete Example

```python
from leanaide_mcts import (
    LeanProofMCTS,
    ProofContext,
    Tactic,
    TacticAction
)

# Create proof context
context = ProofContext(
    goal="∀ n : Nat, n + 0 = n",
    hypotheses=[],
    available_lemmas=["Nat.add_zero"],
    depth=0
)

# Initialize MCTS
mcts = LeanProofMCTS(
    exploration_constant=1.414,
    simulations=1000,
    rollout_depth=7
)

# Run search
best_sequence, root = mcts.search(context)

# Get statistics
stats = mcts.get_statistics()
print(f"Searches: {stats['total_searches']}")
print(f"Proofs found: {stats['successful_proofs']}")

# Extract proof
for i, action in enumerate(best_sequence):
    print(f"{i+1}. {action.tactic.name}")

# Tree statistics
tree_stats = mcts.mcts.get_tree_statistics(root)
print(f"Total nodes: {tree_stats['total_nodes']}")
print(f"Max depth: {tree_stats['max_depth']}")
```

---

*Last Updated: 2025-12-30*
*Version: 1.0.0*
