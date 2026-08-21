# Hybrid MCTS-Evolution API Reference

> **STATUS: implemented** (see `engines/plugins/hybrid_mcts_framework.py` — `HybridMCTSConfig`, `HybridMCTSApproach`, `HybridMCTSResult`, `HybridMCTSEngine`, `AdaptiveHybridSelector`, `CombinedHybridMCTS`, `HybridMCTSPresets`, `HybridMCTSMonitor`, `HybridBenchmark`, `HybridMCTSWorkflowIntegrator`; evolved-policy API in `engines/mcts_mdap/mcts_evolved_policies.py` — `RolloutPolicyGenome`, `TacticRolloutPolicy`, `PolicyPopulation`, `PolicyEvolutionEngine`, `EvolvedPolicyMCTS`, `AdaptivePolicyMCTS`; evolutionary-node API in `engines/mcts_mdap/mcts_evolutionary_nodes.py` — `ActionSequence`, `EvolutionaryNode`, `SequenceCrossover`, `SequenceMutation`, `EvolutionaryMCTS`; coevolution API in `engines/mcts_mdap/mcts_coevolution.py` and `engines/mcts_mdap/mcts_coevolution_mdap.py`; unified framework in `engines/mcts_mdap/mdap_maker_mcts_unified.py`).
>
> **Note on the import path:** there is no top-level `hybrid_mcts` package in this distribution — the framework module is `engines/plugins/hybrid_mcts_framework.py` and the approach-specific classes live under `engines/mcts_mdap/`.
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

## Table of Contents

1. [Overview](#overview)
2. [Core Data Structures](#core-data-structures)
3. [Evolved Policies API](#evolved-policies-api)
4. [Evolutionary Nodes API](#evolutionary-nodes-api)
5. [Coevolution API](#coevolution-api)
6. [Unified Framework API](#unified-framework-api)
7. [Error Handling](#error-handling)
8. [Type Reference](#type-reference)

---

## Overview

### Module Structure

```python
from hybrid_mcts import (
    # Configuration
    HybridMCTSConfig,
    HybridMCTSApproach,
    HybridMCTSPresets,

    # Evolved Policies
    RolloutPolicyGenome,
    TacticRolloutPolicy,
    PolicyPopulation,
    PolicyEvolutionEngine,
    EvolvedPolicyMCTS,
    AdaptivePolicyMCTS,

    # Evolutionary Nodes
    EvolutionaryNode,
    ActionSequence,
    SequenceCrossover,
    SequenceMutation,
    EvolutionaryMCTS,

    # Coevolution
    ProofDecisionTree,
    DecisionNode,
    TreeCoevolution,
    MCTreeEvaluator,
    TreeEnsemble,

    # Unified Framework
    HybridMCTSEngine,
    HybridMCTSResult,
    AdaptiveHybridSelector,
    CombinedHybridMCTS,
)
```

### API Design Principles

1. **Consistency**: All approaches share common configuration pattern
2. **Composability**: Components can be mixed and matched
3. **Type Safety**: Full type annotations throughout
4. **Async-First**: All major operations are async
5. **Progress Tracking**: Built-in metrics and monitoring

---

## Core Data Structures

### HybridMCTSConfig

Configuration for all hybrid MCTS approaches.

```python
@dataclass
class HybridMCTSConfig:
    """Unified configuration for hybrid MCTS approaches."""

    # Approach Selection
    approach: HybridMCTSApproach = HybridMCTSApproach.EVOLVED_POLICIES

    # Evolution Parameters
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_count: int = 2
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3

    # MCTS Parameters (passed to MCTSConfig)
    mcts_simulations: int = 1000
    mcts_time_budget: float = 60.0
    mcts_exploration_constant: float = 1.414
    mcts_rollout_depth: int = 100
    mcts_parallel_simulations: int = 4

    # Evolved Policies Specific
    policy_training_generations: int = 50
    policy_population_size: int = 30
    policy_mutation_rate: float = 0.15

    # Evolutionary Nodes Specific
    node_population_size: int = 10
    node_evolution_frequency: int = 5
    sequence_length_range: Tuple[int, int] = (3, 10)

    # Coevolution Specific
    tree_population_size: int = 20
    evaluator_population_size: int = 15
    coevolution_generations: int = 30
    evaluation_simulations: int = 500

    # Adaptive Selection
    enable_adaptive_selection: bool = True
    adaptive_window_size: int = 10
    switch_threshold: float = 0.3

    # Performance
    enable_caching: bool = True
    cache_size: int = 1000
    max_workers: int = 4
    enable_progress_tracking: bool = True

    # Logging
    log_level: str = "INFO"
    log_metrics: bool = True
    save_tree: bool = False

    def validate(self) -> ValidationResult:
        """Validate configuration parameters."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HybridMCTSConfig':
        """Create from dictionary."""

    @classmethod
    def from_preset(cls, preset: HybridMCTSPreset) -> 'HybridMCTSConfig':
        """Load from preset configuration."""
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `approach` | `HybridMCTSApproach` | `EVOLVED_POLICIES` | Which hybrid approach to use |
| `population_size` | `int` | `50` | Size of evolution population |
| `generations` | `int` | `20` | Number of evolution generations |
| `mutation_rate` | `float` | `0.1` | Probability of mutation (0-1) |
| `crossover_rate` | `float` | `0.8` | Probability of crossover (0-1) |
| `elitism_count` | `int` | `2` | Number of elites to preserve |
| `selection_method` | `SelectionMethod` | `TOURNAMENT` | Selection method |
| `tournament_size` | `int` | `3` | Size of tournament |
| `mcts_simulations` | `int` | `1000` | MCTS iterations per search |
| `mcts_time_budget` | `float` | `60.0` | Max MCTS time in seconds |
| `mcts_exploration_constant` | `float` | `1.414` | UCT exploration parameter |
| `mcts_rollout_depth` | `int` | `100` | Max rollout depth |
| `mcts_parallel_simulations` | `int` | `4` | Parallel rollout count |
| `policy_training_generations` | `int` | `50` | Generations for policy training |
| `policy_population_size` | `int` | `30` | Size of policy population |
| `policy_mutation_rate` | `float` | `0.15` | Policy mutation rate |
| `node_population_size` | `int` | `10` | Population per MCTS node |
| `node_evolution_frequency` | `int` | `5` | Evolve every N node visits |
| `sequence_length_range` | `Tuple[int,int]` | `(3,10)` | Min/max action sequence length |
| `tree_population_size` | `int` | `20` | Size of tree population |
| `evaluator_population_size` | `int` | `15` | Size of evaluator population |
| `coevolution_generations` | `int` | `30` | Generations for coevolution |
| `evaluation_simulations` | `int` | `500` | MC simulations per evaluation |
| `enable_adaptive_selection` | `bool` | `True` | Enable adaptive approach selection |
| `adaptive_window_size` | `int` | `10` | Window for adaptive selection |
| `switch_threshold` | `float` | `0.3` | Performance diff to switch |
| `enable_caching` | `bool` | `True` | Enable result caching |
| `cache_size` | `int` | `1000` | Max cache entries |
| `max_workers` | `int` | `4` | Max parallel workers |
| `enable_progress_tracking` | `bool` | `True` | Track detailed metrics |
| `log_level` | `str` | `"INFO"` | Logging level |
| `log_metrics` | `bool` | `True` | Log performance metrics |
| `save_tree` | `bool` | `False` | Save MCTS tree to disk |

### HybridMCTSApproach

Enum defining available hybrid approaches.

```python
class HybridMCTSApproach(Enum):
    """Hybrid MCTS-Evolution approaches."""

    EVOLVED_POLICIES = "evolved_policies"
    """Evolve rollout policies for MCTS simulation."""

    EVOLUTIONARY_NODES = "evolutionary_nodes"
    """Maintain population of action sequences at each node."""

    COEVOLUTION = "coevolution"
    """Coevolve proof trees with evaluators."""

    ADAPTIVE = "adaptive"
    """Automatically select best approach."""

    COMBINED = "combined"
    """Combine multiple approaches."""
```

### HybridMCTSResult

Result returned by all hybrid MCTS approaches.

```python
@dataclass
class HybridMCTSResult:
    """Result from hybrid MCTS search."""

    # Core result
    success: bool
    best_proof: Optional[LeanProof]
    approach_used: HybridMCTSApproach

    # Performance metrics
    time_elapsed: float
    iterations_completed: int
    nodes_visited: int
    tree_depth: int

    # MCTS metrics
    mcts_win_rate: float
    mcts_confidence: float
    root_visits: int
    root_value: float

    # Evolution metrics
    evolution_metrics: EvolutionMetrics
    generations_completed: int
    best_fitness: float
    convergence_history: List[float]

    # Approach-specific
    policy_metrics: Optional[PolicyMetrics] = None
    node_metrics: Optional[NodeMetrics] = None
    coevolution_metrics: Optional[CoevolutionMetrics] = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config_snapshot: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""

    def get_proof_tactics(self) -> List[Tactic]:
        """Extract tactics from best proof."""

    def get_summary(self) -> str:
        """Get human-readable summary."""
```

### EvolutionMetrics

Metrics specific to evolution component.

```python
@dataclass
class EvolutionMetrics:
    """Metrics from evolution component."""

    generations_completed: int
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity: float
    convergence_rate: float
    stagnation_count: int
    mutation_count: int
    crossover_count: int
    elitism_count: int

    fitness_history: List[float] = field(default_factory=list)
    diversity_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""

    def get_convergence_summary(self) -> Dict[str, Any]:
        """Get convergence analysis."""
```

---

## Evolved Policies API

### RolloutPolicyGenome

Genome representation for evolved rollout policies.

```python
@dataclass
class RolloutPolicyGenome:
    """Genome encoding a rollout policy."""

    # Tactic selection weights
    tactic_weights: Dict[str, float] = field(default_factory=dict)

    # State feature weights
    goal_count_weight: float = 0.5
    depth_weight: float = 0.3
    context_size_weight: float = 0.2

    # Depth preferences
    max_rollout_depth: int = 100
    depth_penalty_factor: float = 0.01

    # Exploration parameters
    exploration_bonus: float = 0.1
    novelty_weight: float = 0.05

    # Meta
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)

    def copy(self) -> 'RolloutPolicyGenome':
        """Create deep copy."""

    def mutate(self, rate: float, rng: random.Random) -> 'RolloutPolicyGenome':
        """Apply mutation and return new genome."""

    def crossover(
        self,
        other: 'RolloutPolicyGenome',
        rng: random.Random
    ) -> Tuple['RolloutPolicyGenome', 'RolloutPolicyGenome']:
        """Crossover with another genome."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    def random(cls, num_tactics: int) -> 'RolloutPolicyGenome':
        """Generate random policy genome."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RolloutPolicyGenome':
        """Deserialize from dictionary."""

    def get_tactic_probability(self, tactic: str) -> float:
        """Get probability of selecting tactic."""

    def select_tactic(
        self,
        available_tactics: List[str],
        state: ProofState,
        rng: random.Random
    ) -> str:
        """Select tactic based on policy."""
```

#### Usage Example

```python
# Create random policy
policy = RolloutPolicyGenome.random(num_tactics=20)

# Mutate
mutated_policy = policy.mutate(rate=0.1, rng=random.Random(42))

# Crossover
child1, child2 = policy.crossover(other_policy, rng=random.Random(42))

# Use in simulation
tactic = policy.select_tactic(
    available_tactics=["intros", "simp", "rw"],
    state=current_state,
    rng=random.Random()
)
```

### TacticRolloutPolicy

Rollout policy that uses evolved genome.

```python
class TacticRolloutPolicy:
    """Rollout policy guided by evolved genome."""

    def __init__(self, genome: RolloutPolicyGenome):
        """Initialize with policy genome."""
        self.genome = genome
        self.rng = random.Random()

    def select_action(
        self,
        state: ProofState,
        available_actions: List[str]
    ) -> str:
        """Select action based on policy."""

    def set_seed(self, seed: int) -> None:
        """Set random seed."""

    def get_action_probabilities(
        self,
        state: ProofState,
        available_actions: List[str]
    ) -> Dict[str, float]:
        """Get probability distribution over actions."""

    def update_genome(self, genome: RolloutPolicyGenome) -> None:
        """Update policy with new genome."""

    def get_genome(self) -> RolloutPolicyGenome:
        """Get current genome."""
```

### PolicyPopulation

Population of policy genomes.

```python
class PolicyPopulation:
    """Population of policy genomes."""

    def __init__(
        self,
        size: int,
        num_tactics: int,
        genomes: Optional[List[RolloutPolicyGenome]] = None
    ):
        """Initialize population."""

    def initialize(self) -> None:
        """Initialize with random genomes."""

    def evaluate(
        self,
        test_theorems: List[str],
        mcts_config: MCTSConfig
    ) -> List[float]:
        """Evaluate all policies on test theorems."""

    def select(
        self,
        method: SelectionMethod,
        count: int,
        tournament_size: int = 3
    ) -> List[RolloutPolicyGenome]:
        """Select individuals using specified method."""

    def crossover(
        self,
        parents: List[RolloutPolicyGenome],
        rate: float,
        rng: random.Random
    ) -> List[RolloutPolicyGenome]:
        """Perform crossover on parent pairs."""

    def mutate(
        self,
        genomes: List[RolloutPolicyGenome],
        rate: float,
        rng: random.Random
    ) -> List[RolloutPolicyGenome]:
        """Mutate genomes."""

    def get_best(self, n: int = 1) -> List[RolloutPolicyGenome]:
        """Get n best genomes."""

    def get_statistics(self) -> PopulationStatistics:
        """Get population statistics."""

    def get_diversity(self) -> float:
        """Calculate population diversity."""

    def evolve(
        self,
        selection_method: SelectionMethod,
        crossover_rate: float,
        mutation_rate: float,
        elitism_count: int,
        rng: random.Random
    ) -> 'PolicyPopulation':
        """Create next generation."""
```

### PolicyEvolutionEngine

Engine for evolving rollout policies.

```python
class PolicyEvolutionEngine:
    """Engine for evolving MCTS rollout policies."""

    def __init__(self, config: HybridMCTSConfig):
        """Initialize evolution engine."""

    async def evolve_policies(
        self,
        test_theorems: List[str],
        initial_population: Optional[int] = None,
        generations: Optional[int] = None,
        mcts_config: Optional[MCTSConfig] = None,
        progress_callback: Optional[Callable[[int, EvolutionMetrics], None]] = None
    ) -> RolloutPolicyGenome:
        """
        Evolve policies on test theorems.

        Args:
            test_theorems: Theorems to evaluate on
            initial_population: Size of initial population
            generations: Number of generations to run
            mcts_config: MCTS config for evaluation
            progress_callback: Called after each generation

        Returns:
            Best policy genome found

        Example:
            engine = PolicyEvolutionEngine(config)
            best_policy = await engine.evolve_policies(
                test_theorems=[
                    "forall n, n + 0 = n",
                    "forall a b, a + b = b + a"
                ],
                generations=50
            )
        """

    def evaluate_policy(
        self,
        policy: RolloutPolicyGenome,
        test_theorems: List[str],
        mcts_config: MCTSConfig
    ) -> float:
        """Evaluate single policy on test theorems."""

    async def batch_evaluate(
        self,
        policies: List[RolloutPolicyGenome],
        test_theorems: List[str],
        mcts_config: MCTSConfig,
        max_workers: int = 4
    ) -> List[float]:
        """Evaluate multiple policies in parallel."""

    def save_policy(
        self,
        policy: RolloutPolicyGenome,
        filepath: str
    ) -> None:
        """Save policy to file."""

    @classmethod
    def load_policy(
        cls,
        filepath: str
    ) -> RolloutPolicyGenome:
        """Load policy from file."""

    def get_training_history(self) -> List[EvolutionMetrics]:
        """Get metrics from all generations."""

    def plot_convergence(self) -> Figure:
        """Plot convergence history."""
```

### EvolvedPolicyMCTS

MCTS that uses evolved rollout policy.

```python
class EvolvedPolicyMCTS:
    """MCTS with evolved rollout policy."""

    def __init__(
        self,
        policy: RolloutPolicyGenome,
        config: MCTSConfig,
        leanaide_client: Optional[LeanAideClient] = None
    ):
        """Initialize with evolved policy."""

    async def search(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        iterations: Optional[int] = None,
        time_budget: Optional[float] = None
    ) -> HybridMCTSResult:
        """
        Search for proof using evolved policy.

        Args:
            theorem: Theorem statement
            theorem_name: Optional theorem name
            iterations: Number of MCTS iterations
            time_budget: Max time in seconds

        Returns:
            HybridMCTSResult with best proof

        Example:
            policy = PolicyEvolutionEngine.load_policy("best_policy.json")
            mcts = EvolvedPolicyMCTS(policy, config)
            result = await mcts.search("forall n, n + 0 = n")
        """

    def set_policy(self, policy: RolloutPolicyGenome) -> None:
        """Update rollout policy."""

    def get_policy(self) -> RolloutPolicyGenome:
        """Get current policy."""

    def get_search_statistics(self) -> Dict[str, Any]:
        """Get detailed search statistics."""

    def export_tree(self) -> Dict[str, Any]:
        """Export MCTS tree structure."""
```

### AdaptivePolicyMCTS

MCTS that adapts policy during search.

```python
class AdaptivePolicyMCTS(EvolvedPolicyMCTS):
    """MCTS with online policy adaptation."""

    def __init__(
        self,
        initial_policy: RolloutPolicyGenome,
        config: MCTSConfig,
        adaptation_interval: int = 100,
        adaptation_window: int = 10
    ):
        """
        Initialize with adaptation parameters.

        Args:
            initial_policy: Starting policy
            config: MCTS configuration
            adaptation_interval: Adapt every N iterations
            adaptation_window: Use last N results for adaptation
        """

    async def search(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        **kwargs
    ) -> HybridMCTSResult:
        """Search with online policy adaptation."""

    def _adapt_policy(
        self,
        recent_results: List[Tuple[str, float]]
    ) -> RolloutPolicyGenome:
        """Adapt policy based on recent results."""

    def get_adaptation_history(self) -> List[Tuple[int, RolloutPolicyGenome]]:
        """Get policy adaptation history."""
```

---

## Evolutionary Nodes API

### ActionSequence

Sequence of tactics for evolutionary nodes.

```python
@dataclass
class ActionSequence:
    """Sequence of tactics (action sequence genome)."""

    actions: List[str]
    fitness: float = 0.0
    depth: int = 0

    # Parent tracking
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self):
        """Compute depth."""
        self.depth = len(self.actions)

    def copy(self) -> 'ActionSequence':
        """Create deep copy."""

    def truncate(self, max_length: int) -> 'ActionSequence':
        """Truncate to max length."""

    def append(self, action: str) -> 'ActionSequence':
        """Append action and return new sequence."""

    def is_valid(self, available_actions: Set[str]) -> bool:
        """Check if all actions are valid."""

    def to_string(self) -> str:
        """Convert to string representation."""

    @classmethod
    def random(
        cls,
        available_actions: List[str],
        length_range: Tuple[int, int],
        rng: random.Random
    ) -> 'ActionSequence':
        """Generate random action sequence."""

    def mutate(
        self,
        available_actions: List[str],
        rate: float,
        rng: random.Random
    ) -> 'ActionSequence':
        """Mutate sequence."""

    def crossover(
        self,
        other: 'ActionSequence',
        rng: random.Random
    ) -> Tuple['ActionSequence', 'ActionSequence']:
        """One-point crossover."""
```

### SequenceCrossover

Crossover operators for action sequences.

```python
class SequenceCrossover:
    """Crossover operators for action sequences."""

    @staticmethod
    def one_point(
        parent1: ActionSequence,
        parent2: ActionSequence,
        rng: random.Random
    ) -> Tuple[ActionSequence, ActionSequence]:
        """One-point crossover."""

    @staticmethod
    def two_point(
        parent1: ActionSequence,
        parent2: ActionSequence,
        rng: random.Random
    ) -> Tuple[ActionSequence, ActionSequence]:
        """Two-point crossover."""

    @staticmethod
    def uniform(
        parent1: ActionSequence,
        parent2: ActionSequence,
        rng: random.Random,
        mix_probability: float = 0.5
    ) -> Tuple[ActionSequence, ActionSequence]:
        """Uniform crossover."""

    @staticmethod
    def order(
        parent1: ActionSequence,
        parent2: ActionSequence,
        rng: random.Random
    ) -> Tuple[ActionSequence, ActionSequence]:
        """Order-preserving crossover."""
```

### SequenceMutation

Mutation operators for action sequences.

```python
class SequenceMutation:
    """Mutation operators for action sequences."""

    @staticmethod
    def add_action(
        sequence: ActionSequence,
        available_actions: List[str],
        rng: random.Random
    ) -> ActionSequence:
        """Add random action."""

    @staticmethod
    def remove_action(
        sequence: ActionSequence,
        rng: random.Random
    ) -> ActionSequence:
        """Remove random action."""

    @staticmethod
    def replace_action(
        sequence: ActionSequence,
        available_actions: List[str],
        rng: random.Random
    ) -> ActionSequence:
        """Replace random action."""

    @staticmethod
    def swap_actions(
        sequence: ActionSequence,
        rng: random.Random
    ) -> ActionSequence:
        """Swap two random actions."""

    @staticmethod
    def scramble(
        sequence: ActionSequence,
        rng: random.Random
    ) -> ActionSequence:
        """Scramble subsection of sequence."""

    @staticmethod
    def mutate(
        sequence: ActionSequence,
        available_actions: List[str],
        rate: float,
        rng: random.Random
    ) -> ActionSequence:
        """Apply mutation with given rate."""
```

### EvolutionaryNode

MCTS node with population of action sequences.

```python
class EvolutionaryNode:
    """MCTS node with action sequence population."""

    def __init__(
        self,
        state: ProofState,
        population_size: int,
        available_actions: List[str],
        parent: Optional['EvolutionaryNode'] = None
    ):
        """Initialize evolutionary node."""

    def initialize_population(
        self,
        size: int,
        length_range: Tuple[int, int]
    ) -> None:
        """Initialize random population."""

    def get_best_sequence(self) -> ActionSequence:
        """Get best action sequence."""

    def get_population(self) -> List[ActionSequence]:
        """Get current population."""

    def evolve_population(
        self,
        crossover_rate: float,
        mutation_rate: float,
        elitism_count: int,
        rng: random.Random
    ) -> None:
        """Evolve population at this node."""

    def evaluate_sequence(
        self,
        sequence: ActionSequence,
        evaluator: Callable[[ProofState, str], float]
    ) -> float:
        """Evaluate fitness of sequence."""

    def update_visit_count(self) -> None:
        """Increment visit count."""

    def should_evolve(self, frequency: int) -> bool:
        """Check if should evolve based on visit count."""

    def get_child(self, action: str) -> Optional['EvolutionaryNode']:
        """Get child node for action."""

    def add_child(self, action: str, child: 'EvolutionaryNode') -> None:
        """Add child node."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    def get_statistics(self) -> Dict[str, Any]:
        """Get node statistics."""
```

### EvolutionaryMCTS

MCTS with evolutionary nodes.

```python
class EvolutionaryMCTS:
    """MCTS with action sequence populations at each node."""

    def __init__(
        self,
        config: HybridMCTSConfig,
        leanaide_client: Optional[LeanAideClient] = None
    ):
        """Initialize evolutionary MCTS."""

    async def search(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        iterations: Optional[int] = None,
        time_budget: Optional[float] = None
    ) -> HybridMCTSResult:
        """
        Search for proof using evolutionary nodes.

        Args:
            theorem: Theorem statement
            theorem_name: Optional theorem name
            iterations: Number of MCTS iterations
            time_budget: Max time in seconds

        Returns:
            HybridMCTSResult with best proof

        Example:
            config = HybridMCTSPresets.evolutionary_nodes()
            mcts = EvolutionaryMCTS(config)
            result = await mcts.search(complex_theorem)
        """

    def _initialize_root(
        self,
        theorem: str
    ) -> EvolutionaryNode:
        """Initialize root node."""

    async def _select_node(
        self,
        root: EvolutionaryNode
    ) -> EvolutionaryNode:
        """Select leaf node using UCT."""

    async def _expand_node(
        self,
        node: EvolutionaryNode
    ) -> EvolutionaryNode:
        """Expand node with best sequence."""

    async def _simulate(
        self,
        node: EvolutionaryNode
    ) -> float:
        """Simulate using best sequence."""

    def _backpropagate(
        self,
        node: EvolutionaryNode,
        reward: float
    ) -> None:
        """Backpropagate reward."""

    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""

    def export_population_data(self) -> Dict[str, Any]:
        """Export all populations."""
```

---

## Coevolution API

### ProofDecisionTree

Decision tree representing proof strategy.

```python
class ProofDecisionTree:
    """Decision tree for proof search."""

    def __init__(
        self,
        root: Optional[DecisionNode] = None,
        fitness: float = 0.0
    ):
        """Initialize decision tree."""

    def generate_proof(
        self,
        theorem: str
    ) -> Optional[LeanProof]:
        """Generate proof by following tree."""

    def evaluate(
        self,
        test_theorems: List[str],
        evaluator: 'MCTreeEvaluator'
    ) -> float:
        """Evaluate tree on test theorems."""

    def mutate(
        self,
        mutation_rate: float,
        rng: random.Random
    ) -> 'ProofDecisionTree':
        """Mutate tree structure."""

    def crossover(
        self,
        other: 'ProofDecisionTree',
        rng: random.Random
    ) -> Tuple['ProofDecisionTree', 'ProofDecisionTree']:
        """Subtree crossover."""

    def get_depth(self) -> int:
        """Get maximum depth."""

    def get_size(self) -> int:
        """Get number of nodes."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    def random(
        cls,
        max_depth: int,
        available_tactics: List[str],
        rng: random.Random
    ) -> 'ProofDecisionTree':
        """Generate random tree."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProofDecisionTree':
        """Deserialize from dictionary."""

    def prune(self, max_depth: int) -> 'ProofDecisionTree':
        """Prune tree to max depth."""

    def get_subtree(self, node_id: str) -> Optional['ProofDecisionTree']:
        """Extract subtree."""

    def replace_subtree(
        self,
        node_id: str,
        new_subtree: 'ProofDecisionTree'
    ) -> bool:
        """Replace subtree."""
```

### DecisionNode

Node in proof decision tree.

```python
class DecisionNode:
    """Node in proof decision tree."""

    def __init__(
        self,
        node_type: str,
        tactic: Optional[str] = None,
        condition: Optional[Callable[[ProofState], bool]] = None,
        children: Optional[List['DecisionNode']] = None
    ):
        """
        Initialize decision node.

        Args:
            node_type: "action", "branch", or "leaf"
            tactic: Tactic to apply (for action nodes)
            condition: Branching condition
            children: Child nodes
        """

    def execute(
        self,
        state: ProofState
    ) -> Tuple[ProofState, Optional['DecisionNode']]:
        """Execute node and return (new_state, next_node)."""

    def is_leaf(self) -> bool:
        """Check if leaf node."""

    def get_child_count(self) -> int:
        """Get number of children."""

    def add_child(self, child: 'DecisionNode') -> None:
        """Add child node."""

    def copy(self) -> 'DecisionNode':
        """Create deep copy."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionNode':
        """Deserialize from dictionary."""

    @classmethod
    def action_node(
        cls,
        tactic: str
    ) -> 'DecisionNode':
        """Create action node."""

    @classmethod
    def branch_node(
        cls,
        condition: Callable[[ProofState], bool],
        true_branch: 'DecisionNode',
        false_branch: 'DecisionNode'
    ) -> 'DecisionNode':
        """Create branch node."""

    @classmethod
    def leaf_node(cls) -> 'DecisionNode':
        """Create leaf node."""
```

### TreeCoevolution

Coevolution of proof trees and evaluators.

```python
class TreeCoevolution:
    """Coevolution of proof trees and evaluators."""

    def __init__(
        self,
        config: HybridMCTSConfig,
        test_theorems: List[str]
    ):
        """Initialize coevolution."""

    async def coevolve(
        self,
        generations: Optional[int] = None,
        progress_callback: Optional[Callable[[int, CoevolutionMetrics], None]] = None
    ) -> Tuple[ProofDecisionTree, MCTreeEvaluator]:
        """
        Run coevolution.

        Args:
            generations: Number of generations
            progress_callback: Called each generation

        Returns:
            (best_tree, best_evaluator)

        Example:
            coevolution = TreeCoevolution(config, test_theorems)
            best_tree, best_evaluator = await coevolution.coevolve(
                generations=50
            )
        """

    def _initialize_populations(self) -> None:
        """Initialize tree and evaluator populations."""

    async def _evaluate_generation(
        self,
        generation: int
    ) -> Tuple[List[float], List[float]]:
        """Evaluate all tree-evaluator pairs."""

    def _evolve_trees(
        self,
        fitnesses: List[float]
    ) -> None:
        """Evolve tree population."""

    def _evolve_evaluators(
        self,
        fitnesses: List[float]
    ) -> None:
        """Evolve evaluator population."""

    def get_coevolution_history(self) -> List[CoevolutionMetrics]:
        """Get metrics from all generations."""

    def plot_arms_race(self) -> Figure:
        """Plot tree vs evaluator fitness over time."""

    def get_best_tree(self) -> ProofDecisionTree:
        """Get best tree from final population."""

    def get_best_evaluator(self) -> MCTreeEvaluator:
        """Get best evaluator from final population."""
```

### MCTreeEvaluator

Evaluator for proof trees.

```python
class MCTreeEvaluator:
    """Evaluator for proof decision trees."""

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        depth_weight: float = 0.1,
        tactic_diversity_weight: float = 0.2,
        success_weight: float = 0.7
    ):
        """Initialize evaluator."""

    def evaluate(
        self,
        tree: ProofDecisionTree,
        theorem: str,
        num_simulations: int = 100
    ) -> float:
        """
        Evaluate proof tree.

        Returns score in [0, 1].
        """

    def evaluate_batch(
        self,
        trees: List[ProofDecisionTree],
        theorems: List[str],
        num_simulations: int = 100
    ) -> List[float]:
        """Evaluate multiple tree-theorem pairs."""

    def _evaluate_success(
        self,
        tree: ProofDecisionTree,
        theorem: str
    ) -> float:
        """Evaluate proof success rate."""

    def _evaluate_efficiency(
        self,
        tree: ProofDecisionTree
    ) -> float:
        """Evaluate tree efficiency."""

    def _evaluate_diversity(
        self,
        tree: ProofDecisionTree
    ) -> float:
        """Evaluate tactic diversity."""

    def mutate(
        self,
        rate: float,
        rng: random.Random
    ) -> 'MCTreeEvaluator':
        """Mutate evaluator weights."""

    def crossover(
        self,
        other: 'MCTreeEvaluator',
        rng: random.Random
    ) -> Tuple['MCTreeEvaluator', 'MCTreeEvaluator']:
        """Crossover evaluators."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCTreeEvaluator':
        """Deserialize from dictionary."""

    @classmethod
    def random(cls, rng: random.Random) -> 'MCTreeEvaluator':
        """Generate random evaluator."""
```

### TreeEnsemble

Ensemble of proof trees.

```python
class TreeEnsemble:
    """Ensemble of proof decision trees."""

    def __init__(
        self,
        trees: List[ProofDecisionTree],
        weights: Optional[List[float]] = None
    ):
        """Initialize ensemble."""

    def add_tree(self, tree: ProofDecisionTree, weight: float = 1.0) -> None:
        """Add tree to ensemble."""

    def remove_tree(self, index: int) -> None:
        """Remove tree from ensemble."""

    def generate_proof(
        self,
        theorem: str,
        method: str = "voting"
    ) -> Optional[LeanProof]:
        """
        Generate proof using ensemble.

        Methods:
        - "voting": Majority vote
        - "weighted": Weighted vote
        - "best": Use best tree
        """

    def evaluate(
        self,
        test_theorems: List[str],
        evaluator: MCTreeEvaluator
    ) -> float:
        """Evaluate ensemble."""

    def prune(
        self,
        test_theorems: List[str],
        evaluator: MCTreeEvaluator,
        keep_ratio: float = 0.5
    ) -> 'TreeEnsemble':
        """Prune worst performing trees."""

    def get_diversity(self) -> float:
        """Calculate ensemble diversity."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TreeEnsemble':
        """Deserialize from dictionary."""
```

---

## Unified Framework API

### HybridMCTSEngine

Main engine for all hybrid approaches.

```python
class HybridMCTSEngine:
    """Unified engine for hybrid MCTS approaches."""

    def __init__(
        self,
        config: HybridMCTSConfig,
        leanaide_client: Optional[LeanAideClient] = None
    ):
        """Initialize hybrid MCTS engine."""

    async def search(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        iterations: Optional[int] = None,
        time_budget: Optional[float] = None
    ) -> HybridMCTSResult:
        """
        Search for proof using configured approach.

        Args:
            theorem: Theorem statement
            theorem_name: Optional theorem name
            iterations: Override config iterations
            time_budget: Override config time budget

        Returns:
            HybridMCTSResult

        Example:
            config = HybridMCTSPresets.balanced()
            engine = HybridMCTSEngine(config)
            result = await engine.search(
                "forall n, n + 0 = n"
            )
        """

    def set_approach(self, approach: HybridMCTSApproach) -> None:
        """Change hybrid approach."""

    def get_approach(self) -> HybridMCTSApproach:
        """Get current approach."""

    def get_config(self) -> HybridMCTSConfig:
        """Get current configuration."""

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics."""

    def reset(self) -> None:
        """Reset engine state."""

    async def batch_search(
        self,
        theorems: List[str],
        parallel: bool = True
    ) -> List[HybridMCTSResult]:
        """Search multiple theorems."""

    def save_state(self, filepath: str) -> None:
        """Save engine state to file."""

    @classmethod
    def load_state(cls, filepath: str) -> 'HybridMCTSEngine':
        """Load engine state from file."""
```

### AdaptiveHybridSelector

Automatic approach selection.

```python
class AdaptiveHybridSelector:
    """Automatically select best hybrid approach."""

    def __init__(
        self,
        window_size: int = 10,
        switch_threshold: float = 0.3
    ):
        """Initialize adaptive selector."""

    def select_approach(
        self,
        theorem: str,
        features: Optional[Dict[str, Any]] = None
    ) -> HybridMCTSApproach:
        """
        Select best approach for theorem.

        Args:
            theorem: Theorem statement
            features: Optional precomputed features

        Returns:
            Recommended approach

        Example:
            selector = AdaptiveHybridSelector()
            approach = selector.select_approach(complex_theorem)
            config.approach = approach
        """

    def extract_features(
        self,
        theorem: str
    ) -> Dict[str, Any]:
        """Extract features from theorem."""

    def update_performance(
        self,
        approach: HybridMCTSApproach,
        performance: float
    ) -> None:
        """Update performance tracking."""

    def get_performance_history(
        self
    ) -> Dict[HybridMCTSApproach, List[float]]:
        """Get performance history."""

    def get_recommendation_confidence(
        self,
        theorem: str
    ) -> float:
        """Get confidence in recommendation."""

    def reset_history(self) -> None:
        """Clear performance history."""

    def plot_performance(self) -> Figure:
        """Plot performance comparison."""
```

### CombinedHybridMCTS

Combine multiple approaches.

```python
class CombinedHybridMCTS:
    """Combine multiple hybrid approaches."""

    def __init__(
        self,
        approaches: List[HybridMCTSApproach],
        combination_method: str = "voting",
        config: Optional[HybridMCTSConfig] = None
    ):
        """
        Initialize combined hybrid.

        Args:
            approaches: List of approaches to combine
            combination_method: "voting", "weighted", or "sequential"
            config: Base configuration

        Combination Methods:
        - "voting": Majority vote on final proof
        - "weighted": Weighted combination by performance
        - "sequential": Try approaches in order
        """

    async def search_combined(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        time_budget: Optional[float] = None
    ) -> HybridMCTSResult:
        """
        Search using combined approaches.

        Returns result from best combination.

        Example:
            combined = CombinedHybridMCTS(
                approaches=[
                    HybridMCTSApproach.EVOLVED_POLICIES,
                    HybridMCTSApproach.EVOLUTIONARY_NODES
                ],
                combination_method="voting"
            )
            result = await combined.search_combined(theorem)
        """

    async def search_parallel(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        time_per_approach: Optional[float] = None
    ) -> List[HybridMCTSResult]:
        """Run all approaches in parallel."""

    def set_weights(
        self,
        weights: Dict[HybridMCTSApproach, float]
    ) -> None:
        """Set weights for weighted combination."""

    def get_weights(self) -> Dict[HybridMCTSApproach, float]:
        """Get current weights."""

    def learn_weights(
        self,
        training_theorems: List[str]
    ) -> None:
        """Learn optimal weights from training data."""
```

### HybridMCTSPresets

Predefined configurations.

```python
class HybridMCTSPresets:
    """Predefined configuration presets."""

    @staticmethod
    def fast() -> HybridMCTSConfig:
        """
        Fast configuration (speed over quality).

        - Evolved Policies approach
        - Small population
        - Few generations
        - Low simulation count

        Best for:
        - Quick prototypes
        - Time-critical applications
        - Simple theorems
        """

    @staticmethod
    def balanced() -> HybridMCTSConfig:
        """
        Balanced configuration.

        - Adaptive approach selection
        - Medium population
        - Medium generations
        - Medium simulation count

        Best for:
        - General use
        - Unknown problem complexity
        - Production systems
        """

    @staticmethod
    def thorough() -> HybridMCTSConfig:
        """
        Thorough configuration (quality over speed).

        - Coevolution approach
        - Large population
        - Many generations
        - High simulation count

        Best for:
        - Complex theorems
        - Research applications
        - Offline processing
        """

    @staticmethod
    def evolved_policies() -> HybridMCTSConfig:
        """Evolved Policies preset."""

    @staticmethod
    def evolutionary_nodes() -> HybridMCTSConfig:
        """Evolutionary Nodes preset."""

    @staticmethod
    def coevolution() -> HybridMCTSConfig:
        """Coevolution preset."""

    @staticmethod
    def custom(**kwargs) -> HybridMCTSConfig:
        """Create custom configuration from preset base."""
```

---

## Error Handling

### Exception Hierarchy

```python
class HybridMCTSError(Exception):
    """Base exception for hybrid MCTS."""

class ConfigurationError(HybridMCTSError):
    """Invalid configuration."""

class PolicyEvolutionError(HybridMCTSError):
    """Error during policy evolution."""

class NodeEvolutionError(HybridMCTSError):
    """Error during node evolution."""

class CoevolutionError(HybridMCTSError):
    """Error during coevolution."""

class SearchTimeout(HybridMCTSError):
    """Search exceeded time budget."""

class PopulationDiversityError(HybridMCTSError):
    """Population lost diversity."""
```

### Error Handling Patterns

```python
# Basic error handling
try:
    result = await engine.search(theorem)
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Fix configuration and retry
except SearchTimeout as e:
    logger.warning(f"Search timed out: {e}")
    # Handle partial result
except HybridMCTSError as e:
    logger.error(f"Hybrid MCTS error: {e}")
    # Fallback to other method

# Advanced error handling with recovery
try:
    result = await engine.search(theorem)
except PopulationDiversityError as e:
    logger.warning("Population lost diversity, reinitializing...")
    engine.reset()
    result = await engine.search(theorem)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Use fallback strategy
    result = await fallback_search(theorem)
```

---

## Type Reference

### Complete Type Definitions

```python
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Awaitable
)
from dataclasses import dataclass
from enum import Enum

# Core types
ProofState = ...  # From leanaide_mcts
Tactic = ...  # From leanaide_mcts
LeanProof = ...  # From leanaide_evolution
LeanAideClient = ...  # From leanaide_client

# Evolution types
SelectionMethod = Enum('SelectionMethod', ['TOURNAMENT', 'ROULETTE', 'RANK'])
CrossoverMethod = Enum('CrossoverMethod', ['ONE_POINT', 'TWO_POINT', 'UNIFORM'])
MutationMethod = Enum('MutationMethod', ['ADD', 'REMOVE', 'REPLACE', 'SWAP'])

# Result types
Fitness = float
Probability = float
Confidence = float

# Callback types
ProgressCallback = Callable[[int, EvolutionMetrics], Awaitable[None]]
ErrorCallback = Callable[[Exception], Awaitable[None]]

# Complex types
PopulationStatistics = Dict[str, Union[float, int, List[float]]]
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-30
**Author**: OpenEvolve Frontend Team
**Related Docs**:
- [HYBRID_MCTS_ARCHITECTURE.md](./HYBRID_MCTS_ARCHITECTURE.md)
- [HYBRID_MCTS_GUIDE.md](./HYBRID_MCTS_GUIDE.md)
- [HYBRID_MCTS_EXAMPLES.md](./HYBRID_MCTS_EXAMPLES.md)
