# HYBRID MAKER API REFERENCE

Complete API reference for Hybrid MAKER strategies integration.

**Version:** 1.0.0
**Paper:** arXiv:2511.09030
**Last Updated:** 2025-12-30

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration APIs](#configuration-apis)
3. [Core MAKER APIs](#core-maker-apis)
4. [Hybrid Strategy APIs](#hybrid-strategy-apis)
5. [Evolution Integration APIs](#evolution-integration-apis)
6. [Utility APIs](#utility-apis)
7. [Error Handling](#error-handling)
8. [Best Practices](#best-practices)
9. [Migration Guide](#migration-guide)
10. [Examples](#examples)

---

## Overview

The Hybrid MAKER API provides comprehensive access to the MAKER framework integrated with MCTS, Evolutionary Algorithms, and Adversarial Testing. The API is organized into logical modules:

- **Configuration**: Dataclasses for configuring strategies
- **Core MAKER**: Voting engine, red flagging, checkpointing
- **Hybrid Strategies**: Sequential, parallel, and adaptive approaches
- **Evolution Integration**: Population management, fitness evaluation
- **Utilities**: Capabilities, validation, metrics

---

## Configuration APIs

### MAKERHybridConfig

```python
@dataclass
class MAKERHybridConfig:
    """Configuration for MAKER-enhanced hybrid strategies"""

    # MAKER voting parameters
    enable_voting: bool = True
    voting_threshold: int = 3  # k for first-to-ahead-by-k
    enable_red_flagging: bool = True

    # MDAP decomposition parameters
    enable_decomposition: bool = True
    decomposition_depth: int = 3
    max_subtasks: int = 10

    # Hybrid strategy parameters
    mcts_simulations: int = 100
    evolution_generations: int = 20
    population_size: int = 20

    # Adversarial parameters
    adversarial_rounds: int = 3
    red_team_agents: int = 2
    blue_team_agents: int = 2

    # Adaptive parameters
    adaptive_switching: bool = True
    diversity_threshold: float = 0.3
    convergence_threshold: float = 0.95
```

**Parameters:**

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enable_voting` | bool | True | - | Enable MAKER voting |
| `voting_threshold` | int | 3 | 2-8 | k value for first-to-ahead-by-k |
| `enable_red_flagging` | bool | True | - | Enable red flagging |
| `enable_decomposition` | bool | True | - | Enable MDAP decomposition |
| `decomposition_depth` | int | 3 | 1-5 | Max decomposition depth |
| `max_subtasks` | int | 10 | 1-20 | Maximum subtasks |
| `mcts_simulations` | int | 100 | 10-500 | MCTS simulations per run |
| `evolution_generations` | int | 20 | 1-100 | Evolution generations |
| `population_size` | int | 20 | 5-100 | Population size |
| `adversarial_rounds` | int | 3 | 1-10 | Adversarial rounds |
| `red_team_agents` | int | 2 | 1-5 | Red team size |
| `blue_team_agents` | int | 2 | 1-5 | Blue team size |
| `adaptive_switching` | bool | True | - | Enable adaptive switching |
| `diversity_threshold` | float | 0.3 | 0.0-1.0 | Minimum diversity |
| `convergence_threshold` | float | 0.95 | 0.0-1.0 | Convergence threshold |

**Methods:**

```python
def to_dict(self) -> Dict[str, Any]:
    """Convert config to dictionary for serialization"""
    return {
        "enable_voting": self.enable_voting,
        "voting_threshold": self.voting_threshold,
        # ... all fields
    }
```

**Example:**

```python
from hybrid_maker_integration import MAKERHybridConfig

# Create config with custom parameters
config = MAKERHybridConfig(
    enable_voting=True,
    voting_threshold=4,
    mcts_simulations=200,
    evolution_generations=30,
    population_size=25
)

# Use with strategy
strategy = MCTSThenMAKER(
    mcts_simulations=config.mcts_simulations,
    maker_voting_threshold=config.voting_threshold
)

# Serialize for storage
config_dict = config.to_dict()
```

### MAKERHybridMode

```python
class MAKERHybridMode(Enum):
    """MAKER hybrid strategy modes"""
    MCTS_THEN_MAKER = "mcts_then_maker"
    MAKER_THEN_EVOLUTION = "maker_then_evolution"
    MAKER_ADVERSARIAL = "maker_adversarial"
    ADAPTIVE_MAKER = "adaptive_maker"
    MAKER_MDAP_PARALLEL = "maker_mdap_parallel"
    FULL_MAKER_HYBRID = "full_maker_hybrid"
```

**Values:**

- `MCTS_THEN_MAKER`: MCTS exploration followed by MAKER voting refinement
- `MAKER_THEN_EVOLUTION`: MAKER generates initial population, evolution refines
- `MAKER_ADVERSARIAL`: Red/blue team adversarial testing with MAKER voting
- `ADAPTIVE_MAKER`: Dynamic strategy switching based on metrics
- `MAKER_MDAP_PARALLEL`: Parallel MAKER voting and MDAP decomposition
- `FULL_MAKER_HYBRID`: Complete integration of all components

**Example:**

```python
from hybrid_maker_integration import MAKERHybridMode, run_maker_hybrid

# Use specific mode
result = await run_maker_hybrid(
    theorem="forall n m : nat, n + m = m + n",
    mode=MAKERHybridMode.MCTS_THEN_MAKER
)
```

### MakerevolutionConfig

```python
@dataclass
class MakerevolutionConfig:
    """Configuration for MAKER-enhanced evolutionary computation"""

    # Evolution mode
    mode: MakerevolutionMode = MakerevolutionMode.HYBRID

    # Population voting parameters
    enable_voting: bool = True
    voting_threshold: int = 3
    population_size: int = 20
    num_candidates: int = 5

    # MDAP decomposition parameters
    enable_decomposition: bool = True
    decomposition_depth: int = 3
    max_subtasks: int = 10

    # Zero-error parameters
    enable_red_flagging: bool = True
    convergence_threshold: float = 0.95
    max_iterations_without_improvement: int = 10

    # Adaptive parameters
    adaptive_voting: bool = True
    diversity_threshold: float = 0.3

    # MAKER-specific
    max_token_length: int = 750
    temperature: float = 0.7
```

**Modes:**

```python
class MakerevolutionMode(Enum):
    VOTING_ONLY = "voting_only"  # Use MAKER voting for selection only
    DECOMPOSITION = "decomposition"  # Use MDAP for task decomposition
    HYBRID = "hybrid"  # Combine MAKER voting + MDAP decomposition
    FULL_MAKER = "full_maker"  # Complete MAKER-based evolution
```

---

## Core MAKER APIs

### MakerConfig

```python
@dataclass
class MakerConfig:
    """Configuration for MAKER engine"""

    k_min: int = 2  # Minimum voting threshold
    k_max: int = 8  # Maximum voting threshold
    max_votes_per_step: int = 60  # Maximum votes to collect
    max_steps: int = 1000  # Maximum solving steps
    timeout_seconds: int = 90  # Timeout per step
    checkpoint_interval: int = 25  # Checkpoint frequency
    red_flag_rules: RedFlagRules = field(default_factory=RedFlagRules)
```

**Usage:**

```python
from maker_engine import MakerConfig, MakerEngine, Team

# Create team
team = Team(members=[model1, model2, model3])

# Configure MAKER
config = MakerConfig(
    k_min=2,
    k_max=5,
    max_votes_per_step=40,
    timeout_seconds=60,
    checkpoint_interval=20
)

# Create engine
engine = MakerEngine(team=team, config=config)
```

### MakerStep

```python
@dataclass
class MakerStep:
    """Single step in MAKER solving"""

    step_id: str
    prompt_template: str
    expected_schema: Optional[Dict[str, Any]] = None
    task_type: str = "general"
    priority: int = 0
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods:**

```python
def render_prompt(self, state: Any, history: List[Dict[str, Any]]) -> str:
    """Render prompt template with state and history"""
    state_payload = json.dumps(state, ensure_ascii=True)
    history_payload = json.dumps(history, ensure_ascii=True)
    return self.prompt_template.format(state=state_payload, history=history_payload)
```

**Example:**

```python
from maker_engine import MakerStep

step = MakerStep(
    step_id="step_1",
    prompt_template="Current state: {state}\nHistory: {history}\nNext action?",
    expected_schema={
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["action"]
    },
    task_type="solve",
    priority=1,
    system_prompt="You are a theorem prover."
)
```

### MakerEngine

```python
class MakerEngine:
    """Core MAKER voting engine"""

    def __init__(self, team: Team, config: MakerConfig):
        """
        Initialize MAKER engine.

        Args:
            team: Team of agents for voting
            config: MAKER configuration
        """
        self.team = team
        self.config = config
        self.red_flagger = RedFlagger(config.red_flag_rules)
        self.metrics = {
            "steps": 0,
            "votes_cast": 0,
            "red_flags": 0,
            "escalations": 0,
            "errors": 0
        }
```

**Methods:**

```python
def solve(
    self,
    initial_state: Any,
    step_builder: Callable[[Any, List[Dict[str, Any]]], MakerStep],
    apply_action: Callable[[Any, Any], Any],
    checkpoint_store: Optional[CheckpointStore] = None,
    stop_condition: Optional[Callable[[MakerState], bool]] = None
) -> MakerRunResult:
    """
    Execute MAKER solving with zero-error guarantees.

    Args:
        initial_state: Starting state
        step_builder: Function to build MakerStep from state and history
        apply_action: Function to apply action to state, producing next state
        checkpoint_store: Optional checkpoint store for fault tolerance
        stop_condition: Optional callable to check if solving should stop

    Returns:
        MakerRunResult containing final state, metrics, and termination reason

    Example:
        >>> def step_builder(state, history):
        ...     return MakerStep(
        ...         step_id="step_1",
        ...         prompt_template=f"Solve: {state}",
        ...         expected_schema={"type": "object"}
        ...     )
        >>>
        >>> def apply_action(state, action):
        ...     return state.update(action)
        >>>
        >>> result = engine.solve(
        ...     initial_state={"theorem": "n + 0 = n"},
        ...     step_builder=step_builder,
        ...     apply_action=apply_action
        ... )
    """
```

**Metrics Dictionary:**

```python
{
    "steps": int,           # Total steps executed
    "votes_cast": int,      # Total votes collected
    "red_flags": int,       # Total red flags raised
    "escalations": int,     # Total escalations to best-effort
    "errors": int           # Total errors encountered
}
```

### MakerRunResult

```python
@dataclass
class MakerRunResult:
    """Result of MAKER solving"""

    state: MakerState
    metrics: Dict[str, Any]
    terminated_reason: str
```

**Termination Reasons:**

- `"max_steps_reached"`: Executed maximum configured steps
- `"no_action_selected"`: No winner selected in voting
- `"stop_condition_met"`: User-defined stop condition triggered
- `"apply_action_failed:..."`: Action application failed

---

## Hybrid Strategy APIs

### MCTSThenMAKER

```python
class MCTSThenMAKER(HybridStrategy):
    """
    MCTS-Then-MAKER hybrid strategy.

    Two-phase approach:
    1. MCTS explores the search space to find candidate proofs
    2. MAKER voting refines candidates with zero-error guarantees
    """

    def __init__(
        self,
        mcts_simulations: int = 100,
        maker_voting_threshold: int = 3,
        population_size: int = 15
    ):
        """
        Initialize MCTS-Then-MAKER strategy.

        Args:
            mcts_simulations: Number of MCTS simulations per exploration
            maker_voting_threshold: k value for MAKER voting
            population_size: Number of candidates to generate

        Example:
            >>> strategy = MCTSThenMAKER(
            ...     mcts_simulations=200,
            ...     maker_voting_threshold=4,
            ...     population_size=20
            ... )
            >>> result = await strategy.generate_proof(
            ...     theorem="forall n : nat, n + 0 = n"
            ... )
        """
```

**Methods:**

```python
async def generate_proof(
    self,
    theorem: str,
    **kwargs
) -> EvolutionResult:
    """
    Generate proof using MCTS-Then-MAKER.

    Args:
        theorem: Theorem statement to prove
        **kwargs: Additional parameters (exploration constants, etc.)

    Returns:
        EvolutionResult containing:
            - success: bool
            - best_proof: Optional[str]
            - best_fitness: float
            - generations_completed: int
            - evolution_time: float
            - convergence_history: List[float]
            - failed_attempts: List[Dict]
    """
```

### MAKERThenEvolution

```python
class MAKERThenEvolution(HybridStrategy):
    """
    MAKER-Then-Evolution hybrid strategy.

    Two-phase approach:
    1. MAKER voting generates high-quality initial population
    2. Evolution refines population with genetic operators

    Benefits:
    - MAKER ensures zero-error initial population
    - Evolution explores variations around high-quality solutions
    - Combines statistical guarantees with evolutionary optimization
    """

    def __init__(
        self,
        maker_voting_threshold: int = 3,
        evolution_generations: int = 20,
        population_size: int = 20,
        initial_candidates: int = 50
    ):
        """
        Initialize MAKER-Then-Evolution strategy.

        Args:
            maker_voting_threshold: k value for MAKER voting
            evolution_generations: Number of evolution generations
            population_size: Size of evolution population
            initial_candidates: Number of initial candidates for voting

        Example:
            >>> strategy = MAKERThenEvolution(
            ...     maker_voting_threshold=3,
            ...     evolution_generations=30,
            ...     population_size=25,
            ...     initial_candidates=100
            ... )
            >>> result = await strategy.generate_proof(
            ...     theorem="forall n m : nat, n + m = m + n"
            ... )
        """
```

**Methods:**

```python
async def generate_proof(
    self,
    theorem: str,
    **kwargs
) -> EvolutionResult:
    """
    Generate proof using MAKER-Then-Evolution.

    Args:
        theorem: Theorem statement
        **kwargs: Additional parameters

    Returns:
        EvolutionResult with final proof and fitness history
    """
```

### MAKERAdversarialHybrid

```python
class MAKERAdversarialHybrid(HybridStrategy):
    """
    MAKER-Adversarial hybrid strategy.

    Combines MAKER voting with adversarial red/blue team testing:
    1. Red team generates attack scenarios
    2. Blue team generates defenses
    3. MAKER voting selects best solutions

    Benefits:
    - Adversarial testing finds edge cases
    - MAKER voting ensures robustness
    - Co-evolutionary improvement
    """

    def __init__(
        self,
        adversarial_rounds: int = 3,
        maker_voting_threshold: int = 3,
        red_team_size: int = 2,
        blue_team_size: int = 2
    ):
        """
        Initialize MAKER-Adversarial strategy.

        Args:
            adversarial_rounds: Number of adversarial rounds
            maker_voting_threshold: k value for MAKER voting
            red_team_size: Number of red team agents
            blue_team_size: Number of blue team agents

        Example:
            >>> strategy = MAKERAdversarialHybrid(
            ...     adversarial_rounds=5,
            ...     maker_voting_threshold=4,
            ...     red_team_size=3,
            ...     blue_team_size=3
            ... )
            >>> result = await strategy.generate_proof(
            ...     theorem="forall a b c : nat, a + (b + c) = (a + b) + c"
            ... )
        """
```

**Methods:**

```python
async def generate_proof(
    self,
    theorem: str,
    **kwargs
) -> EvolutionResult:
    """
    Generate proof using MAKER-Adversarial hybrid.

    Args:
        theorem: Theorem statement
        **kwargs: Additional parameters

    Returns:
        EvolutionResult with adversarial progression history
    """
```

### AdaptiveMAKERHybrid

```python
class AdaptiveMAKERHybrid(HybridStrategy):
    """
    Adaptive MAKER hybrid strategy.

    Dynamically switches between MAKER, MCTS, and Evolution based on
    population diversity and convergence metrics.

    Benefits:
    - Automatic strategy selection
    - Maintains population diversity
    - Prevents premature convergence
    - Optimizes computational resources
    """

    def __init__(
        self,
        diversity_threshold: float = 0.3,
        convergence_threshold: float = 0.95,
        max_generations: int = 50
    ):
        """
        Initialize Adaptive MAKER strategy.

        Args:
            diversity_threshold: Minimum diversity threshold (0.0-1.0)
            convergence_threshold: Convergence threshold (0.0-1.0)
            max_generations: Maximum generations to run

        Example:
            >>> strategy = AdaptiveMAKERHybrid(
            ...     diversity_threshold=0.4,
            ...     convergence_threshold=0.98,
            ...     max_generations=100
            ... )
            >>> result = await strategy.generate_proof(
            ...     theorem="forall n : nat, n * 1 = n"
            ... )
        """
```

**Methods:**

```python
async def generate_proof(
    self,
    theorem: str,
    **kwargs
) -> EvolutionResult:
    """
    Generate proof using adaptive MAKER hybrid.

    Args:
        theorem: Theorem statement
        **kwargs: Additional parameters

    Returns:
        EvolutionResult with convergence history and strategy switches
    """
```

### MAKERMDAPParallel

```python
class MAKERMDAPParallel(HybridStrategy):
    """
    MAKER-MDAP Parallel hybrid strategy.

    Runs MAKER voting and MDAP decomposition in parallel, then combines
    results for maximal efficiency.

    Benefits:
    - Parallel execution for speed
    - MAKER ensures selection quality
    - MDAP provides task decomposition
    - Combined results for optimal performance
    """

    def __init__(
        self,
        maker_voting_threshold: int = 3,
        mdap_agents: int = 4,
        combination_method: str = "best_fitness"
    ):
        """
        Initialize MAKER-MDAP Parallel strategy.

        Args:
            maker_voting_threshold: k value for MAKER voting
            mdap_agents: Number of MDAP agents
            combination_method: How to combine results ("best_fitness" or "average")

        Example:
            >>> strategy = MAKERMDAPParallel(
            ...     maker_voting_threshold=3,
            ...     mdap_agents=5,
            ...     combination_method="best_fitness"
            ... )
            >>> result = await strategy.generate_proof(
            ...     theorem="forall n m : nat, n + m = m + n"
            ... )
        """
```

**Methods:**

```python
async def generate_proof(
    self,
    theorem: str,
    **kwargs
) -> EvolutionResult:
    """
    Generate proof using MAKER-MDAP parallel.

    Args:
        theorem: Theorem statement
        **kwargs: Additional parameters

    Returns:
        EvolutionResult with combined parallel results
    """
```

### FullMAKERHybrid

```python
class FullMAKERHybrid(HybridStrategy):
    """
    Full MAKER hybrid strategy combining all components.

    Integrates:
    - MAKER voting for zero-error selection
    - MDAP decomposition for task breakdown
    - MCTS for exploration
    - Evolution for optimization
    - Adversarial for robustness

    Benefits:
    - Maximum reliability with zero-error guarantees
    - Comprehensive search of solution space
    - Adaptive strategy selection
    - Production-ready robustness
    """

    def __init__(self, config: MAKERHybridConfig = None):
        """
        Initialize Full MAKER Hybrid.

        Args:
            config: MAKER hybrid configuration (uses defaults if None)

        Example:
            >>> from hybrid_maker_integration import MAKERHybridConfig
            >>> config = MAKERHybridConfig(
            ...     enable_voting=True,
            ...     voting_threshold=4,
            ...     mcts_simulations=150,
            ...     evolution_generations=25,
            ...     adversarial_rounds=3
            ... )
            >>> strategy = FullMAKERHybrid(config)
            >>> result = await strategy.generate_proof(
            ...     theorem="forall n m : nat, n + m = m + n"
            ... )
        """
```

**Methods:**

```python
async def generate_proof(
    self,
    theorem: str,
    **kwargs
) -> EvolutionResult:
    """
    Generate proof using full MAKER hybrid.

    Executes all phases sequentially:
    1. MAKER voting
    2. MAKER + Evolution
    3. MAKER Adversarial
    4. Adaptive MAKER
    5. Parallel MAKER + MDAP

    Returns best result across all phases.

    Args:
        theorem: Theorem statement
        **kwargs: Additional parameters

    Returns:
        EvolutionResult with best solution across all phases
    """
```

---

## Evolution Integration APIs

### Individual

```python
@dataclass
class Individual:
    """Represents an individual in the evolution population"""

    genome: str  # The program/content
    fitness: float
    generation: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Compare individuals by fitness (for sorting)"""
        return self.fitness < other.fitness
```

**Example:**

```python
from evolution_maker_integration import Individual

# Create individual
individual = Individual(
    genome="theorem : n + 0 = n\nby\n  simp",
    fitness=0.85,
    generation=5,
    metadata={"parent_ids": [2, 3], "mutation_type": "insert_tactic"}
)
```

### Population

```python
@dataclass
class Population:
    """Represents a population of individuals"""

    individuals: List[Individual]
    generation: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_individual(self) -> Optional[Individual]:
        """Get the best individual in the population"""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def average_fitness(self) -> float:
        """Get average fitness of population"""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)

    @property
    def diversity(self) -> float:
        """
        Calculate population diversity (normalized hamming distance).
        Returns 0-1 where 1 = high diversity.
        """
        # Implementation details...
```

**Usage:**

```python
from evolution_maker_integration import Population, Individual

# Create population
individuals = [
    Individual(genome="proof1", fitness=0.7, generation=0),
    Individual(genome="proof2", fitness=0.85, generation=0),
    Individual(genome="proof3", fitness=0.6, generation=0)
]

population = Population(individuals=individuals, generation=0)

# Access properties
best = population.best_individual
avg_fitness = population.average_fitness
diversity = population.diversity

print(f"Best fitness: {best.fitness:.2f}")
print(f"Average fitness: {avg_fitness:.2f}")
print(f"Diversity: {diversity:.2f}")
```

### MAKEREvolutionEngine

```python
class MAKEREvolutionEngine:
    """
    Main evolution engine enhanced with MAKER/MDAP.

    Combines genetic algorithms with MAKER voting and MDAP decomposition
    for zero-error evolutionary computation.
    """

    def __init__(
        self,
        config: MakerevolutionConfig,
        evolution_config: Optional[EvolutionConfiguration] = None
    ):
        """
        Initialize MAKER evolution engine.

        Args:
            config: MAKER evolution configuration
            evolution_config: Optional standard evolution configuration

        Example:
            >>> config = MakerevolutionConfig(
            ...     mode=MakerevolutionMode.HYBRID,
            ...     voting_threshold=3,
            ...     population_size=20
            ... )
            >>> engine = MAKEREvolutionEngine(config)
        """
```

**Methods:**

```python
def run_evolution(
    self,
    initial_program: str,
    evaluator: Callable,
    max_generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7
) -> Dict[str, Any]:
    """
    Run MAKER-enhanced evolution.

    Args:
        initial_program: Starting program/content
        evaluator: Fitness evaluation function (program -> float)
        max_generations: Maximum generations to run
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover

    Returns:
        Dict with evolution results:
            - success: bool
            - best_program: str
            - best_fitness: float
            - generations: int
            - fitness_history: List[float]
            - final_population: Population
            - evolution_time: float
            - config: Dict
            - method: str

    Example:
        >>> def fitness_fn(program):
        ...     # Evaluate program quality
        ...     return score
        >>>
        >>> result = engine.run_evolution(
        ...     initial_program="my_code.py",
        ...     evaluator=fitness_fn,
        ...     max_generations=50
        ... )
        >>> print(f"Best fitness: {result['best_fitness']}")
        >>> print(f"Best program: {result['best_program']}")
    """
```

---

## Utility APIs

### run_maker_hybrid

```python
async def run_maker_hybrid(
    theorem: str,
    mode: MAKERHybridMode = MAKERHybridMode.FULL_MAKER_HYBRID,
    config: MAKERHybridConfig = None
) -> EvolutionResult:
    """
    Main entry point for MAKER hybrid strategies.

    This is the primary API for executing hybrid MAKER strategies.

    Args:
        theorem: Theorem statement to prove
        mode: Hybrid strategy mode (default: FULL_MAKER_HYBRID)
        config: MAKER hybrid configuration (uses defaults if None)

    Returns:
        EvolutionResult with final proof and metrics

    Example:
        >>> result = await run_maker_hybrid(
        ...     theorem="forall n m : nat, n + m = m + n",
        ...     mode=MAKERHybridMode.MCTS_THEN_MAKER,
        ...     config=MAKERHybridConfig(voting_threshold=4)
        ... )
        >>>
        >>> if result.success:
        ...     print(f"Proof found: {result.best_proof}")
        ...     print(f"Fitness: {result.best_fitness:.3f}")
        ...     print(f"Time: {result.evolution_time:.2f}s")
        ... else:
        ...     print("Failed to find proof")
        ...     for error in result.failed_attempts:
        ...         print(f"  Error: {error}")
    """
```

### run_maker_evolution

```python
def run_maker_evolution(
    initial_program: str,
    evaluator: Callable,
    max_generations: int = 100,
    config: Optional[MakerevolutionConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run MAKER-enhanced evolutionary computation.

    This is the main entry point for MAKER/MDAP-enhanced evolution.

    Args:
        initial_program: Starting program/content
        evaluator: Fitness evaluation function (takes program, returns float)
        max_generations: Maximum generations to evolve
        config: MAKER evolution configuration
        **kwargs: Additional parameters (mutation_rate, crossover_rate, etc.)

    Returns:
        Dict with evolution results

    Example:
        >>> def fitness_fn(program):
        ...     # Evaluate program quality
        ...     # Higher is better
        ...     return score
        >>>
        >>> result = run_maker_evolution(
        ...     initial_program="my_code.py",
        ...     evaluator=fitness_fn,
        ...     max_generations=50,
        ...     config=MakerevolutionConfig(
        ...         mode=MakerevolutionMode.HYBRID,
        ...         voting_threshold=3,
        ...         population_size=20
        ...     )
        ... )
        >>>
        >>> print(f"Best fitness: {result['best_fitness']:.3f}")
        >>> print(f"Generations: {result['generations']}")
    """
```

### get_maker_hybrid_capabilities

```python
def get_maker_hybrid_capabilities() -> Dict[str, Any]:
    """
    Get MAKER hybrid integration capabilities.

    Returns dictionary describing available components and their status.

    Returns:
        Dict with capability information:
            - maker_hybrid_enabled: bool
            - maker_evolution_available: bool
            - maker_adversarial_available: bool
            - maker_core_available: bool
            - mdap_available: bool
            - mcts_available: bool
            - evolution_available: bool
            - integration_status: str ("full", "partial", "none")
            - modes: List[str] - available modes
            - strategies: List[str] - available strategies
            - paper: Dict - paper reference

    Example:
        >>> capabilities = get_maker_hybrid_capabilities()
        >>>
        >>> print(f"MAKER Hybrid Enabled: {capabilities['maker_hybrid_enabled']}")
        >>> print(f"Integration Status: {capabilities['integration_status']}")
        >>>
        >>> print("Available Modes:")
        >>> for mode in capabilities['modes']:
        ...     print(f"  - {mode}")
        >>>
        >>> paper = capabilities['paper']
        >>> print(f"Paper: {paper['title']}")
        >>> print(f"arXiv: {paper['arxiv']}")
    """
```

### get_maker_evolution_capabilities

```python
def get_maker_evolution_capabilities() -> Dict[str, Any]:
    """
    Get capabilities of MAKER-enhanced evolution.

    Returns:
        Dict describing MAKER evolution capabilities:
            - maker_evolution_enabled: bool
            - mdap_decomposition_enabled: bool
            - modes: List[str]
            - algorithms: List[str]
            - features: Dict[str, str]
            - paper_reference: Dict

    Example:
        >>> caps = get_maker_evolution_capabilities()
        >>>
        >>> print("Available Algorithms:")
        >>> for algo in caps['algorithms']:
        ...     print(f"  {algo}")
        >>>
        >>> print("Features:")
        >>> for feature, desc in caps['features'].items():
        ...     print(f"  {feature}: {desc}")
    """
```

---

## Error Handling

### Error Types

```python
class MAKERError(Exception):
    """Base class for MAKER errors"""
    pass

class MAKERConfigurationError(MAKERError):
    """Raised when configuration is invalid"""
    pass

class MAKERVotingError(MAKERError):
    """Raised when voting fails"""
    pass

class MAKERTimeoutError(MAKERError):
    """Raised when operation times out"""
    pass

class MAKERDecompositionError(MAKERError):
    """Raised when decomposition fails"""
    pass
```

### Error Handling Patterns

**Pattern 1: Try-Except with Logging**

```python
import logging

logger = logging.getLogger(__name__)

try:
    result = await run_maker_hybrid(theorem, mode=mode)
except MAKERConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    # Use default configuration
    result = await run_maker_hybrid(theorem)
except MAKERVotingError as e:
    logger.error(f"Voting failed: {e}")
    # Fallback to standard evolution
    result = await run_evolution(theorem)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

**Pattern 2: Result Checking**

```python
result = await run_maker_hybrid(theorem, mode=mode)

if not result.success:
    logger.warning(f"Hybrid MAKER failed, checking errors")
    for error in result.failed_attempts:
        logger.error(f"Error: {error}")

    # Try fallback
    logger.info("Trying fallback mode")
    result = await run_maker_hybrid(theorem, mode=MAKERHybridMode.MCTS_THEN_MAKER)
```

**Pattern 3: Timeout Handling**

```python
import asyncio

async def run_with_timeout(theorem: str, timeout: float = 60.0):
    try:
        result = await asyncio.wait_for(
            run_maker_hybrid(theorem),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        logger.error(f"MAKER hybrid timed out after {timeout}s")
        return EvolutionResult(
            success=False,
            generations_completed=0,
            evolution_time=timeout,
            failed_attempts=[{"error": "timeout"}]
        )
```

---

## Best Practices

### 1. Configuration

**Do:**

```python
# Start with defaults, then tune
config = MAKERHybridConfig(
    voting_threshold=4,  # Higher for quality
    mcts_simulations=150  # More for exploration
)
```

**Don't:**

```python
# Don't set extreme values without testing
config = MAKERHybridConfig(
    voting_threshold=10,  # Too high, will be slow
    population_size=1000  # Too large, memory issues
)
```

### 2. Mode Selection

**Guidelines:**

- **Simple theorems**: `MCTS_THEN_MAKER`
- **Medium complexity**: `MAKER_THEN_EVOLUTION`
- **Need robustness**: `MAKER_ADVERSARIAL`
- **Unknown complexity**: `ADAPTIVE_MAKER`
- **Maximum quality**: `FULL_MAKER_HYBRID`

### 3. Error Handling

**Always:**

```python
result = await run_maker_hybrid(theorem, mode=mode)

if not result.success:
    # Log errors
    for error in result.failed_attempts:
        logger.error(f"Error: {error}")

    # Try fallback or notify user
```

### 4. Resource Management

**Use checkpoints for long runs:**

```python
from maker_engine import FileCheckpointStore

checkpoint_store = FileCheckpointStore("maker_state.json")

result = engine.solve(
    initial_state,
    step_builder,
    apply_action,
    checkpoint_store=checkpoint_store  # Auto-save
)
```

### 5. Performance

**Enable caching:**

```python
config = MDAPConfig(
    cache_ttl_seconds=3600,  # 1 hour
    cache_max_size=10000
)
```

---

## Migration Guide

### From Basic Evolution to MAKER Evolution

**Before:**

```python
from evolution import run_evolution

result = run_evolution(
    initial_program="proof1",
    evaluator=fitness_fn,
    generations=50
)
```

**After:**

```python
from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

result = run_maker_evolution(
    initial_program="proof1",
    evaluator=fitness_fn,
    max_generations=50,
    config=MakerevolutionConfig(
        mode=MakerevolutionMode.HYBRID,
        voting_threshold=3,
        population_size=20
    )
)
```

### From MCTS to MCTS-Then-MAKER

**Before:**

```python
from leanaide_mcts import run_mcts_search

result = run_mcts_search(theorem, simulations=100)
```

**After:**

```python
from hybrid_maker_integration import run_maker_hybrid, MAKERHybridMode

result = await run_maker_hybrid(
    theorem=theorem,
    mode=MAKERHybridMode.MCTS_THEN_MAKER,
    config=MAKERHybridConfig(mcts_simulations=100)
)
```

---

## Examples

### Example 1: Basic MCTS-Then-MAKER

```python
import asyncio
from hybrid_maker_integration import (
    run_maker_hybrid,
    MAKERHybridMode,
    MAKERHybridConfig
)

async def main():
    theorem = "forall n : nat, n + 0 = n"

    result = await run_maker_hybrid(
        theorem=theorem,
        mode=MAKERHybridMode.MCTS_THEN_MAKER,
        config=MAKERHybridConfig(
            voting_threshold=3,
            mcts_simulations=100
        )
    )

    if result.success:
        print(f"Proof found!")
        print(f"Fitness: {result.best_fitness:.3f}")
        print(f"Time: {result.evolution_time:.2f}s")
        print(f"Proof:\n{result.best_proof}")
    else:
        print("Failed to find proof")

asyncio.run(main())
```

### Example 2: MAKER-Then-Evolution with Custom Evaluator

```python
from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

def custom_evaluator(program: str) -> float:
    """Custom fitness function"""
    # Evaluate program quality
    score = 0.0

    # Check for required tactics
    if "induction" in program:
        score += 0.3
    if "simp" in program:
        score += 0.2
    if "refl" in program:
        score += 0.1

    # Penalize very long proofs
    if len(program) > 500:
        score -= 0.2

    return max(0.0, min(1.0, score))

result = run_maker_evolution(
    initial_program="theorem : n + 0 = n",
    evaluator=custom_evaluator,
    max_generations=30,
    config=MakerevolutionConfig(
        mode=MakerevolutionMode.HYBRID,
        voting_threshold=3,
        population_size=20
    )
)
```

### Example 3: Full MAKER Hybrid with Progress Tracking

```python
import asyncio
from hybrid_maker_integration import FullMAKERHybrid, MAKERHybridConfig

async def main():
    config = MAKERHybridConfig(
        enable_voting=True,
        voting_threshold=4,
        mcts_simulations=150,
        evolution_generations=25,
        adversarial_rounds=3,
        adaptive_switching=True
    )

    strategy = FullMAKERHybrid(config)

    theorem = "forall n m : nat, n + m = m + n"

    print(f"Starting Full MAKER Hybrid for: {theorem}")
    result = await strategy.generate_proof(theorem)

    print(f"\nResults:")
    print(f"  Success: {result.success}")
    print(f"  Best fitness: {result.best_fitness:.3f}")
    print(f"  Generations: {result.generations_completed}")
    print(f"  Time: {result.evolution_time:.2f}s")

    if result.convergence_history:
        print(f"\n  Convergence:")
        for i, fitness in enumerate(result.convergence_history[::5]):
            print(f"    Gen {i*5}: {fitness:.3f}")

asyncio.run(main())
```

---

**End of API Reference**

For more information, see:
- Architecture: `HYBRID_MAKER_ARCHITECTURE.md`
- User Guide: `HYBRID_MAKER_GUIDE.md`
- Examples: `HYBRID_MAKER_EXAMPLES.md`
- Integration: `HYBRID_MAKER_INTEGRATION.md`
