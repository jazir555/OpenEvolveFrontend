# LeanAide Evolutionary API Reference

> **STATUS: implemented** (all four modules exist: `integrations/leanaide/leanaide_evolution.py` — `LeanProofPopulation`, `LeanProofMutator`, `LeanProofCrossover`, `LeanProofEvaluator`; `integrations/leanaide/leanaide_adversarial.py` — `LeanBlueTeamAgent`, `LeanRedTeamAgent`, `LeanCounterexampleGenerator`, `LeanAdversarialArena`; `integrations/leanaide/leanaide_selfplay.py` — `SelfPlayResult`, `LeanProofExperienceBuffer`; `integrations/leanaide/leanaide_strategies.py` — `LeanTacticLibrary`, `LeanProofTemplate`).
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide Evolutionary Integration

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Genetic Evolution API (`leanaide_evolution.py`)](#2-genetic-evolution-api)
3. [Adversarial Evolution API (`leanaide_adversarial.py`)](#3-adversarial-evolution-api)
4. [Self-Play API (`leanaide_selfplay.py`)](#4-self-play-api)
5. [Strategy Library API (`leanaide_strategies.py`)](#5-strategy-library-api)
6. [Data Structures](#6-data-structures)
7. [Enums](#7-enums)
8. [Error Handling](#8-error-handling)

---

## 1. Module Overview

### 1.1 Module Files

```
leanaide_evolution.py          # Genetic evolution (GA-based proof search)
leanaide_adversarial.py        # Adversarial evolution (red team vs blue team)
leanaide_selfplay.py           # Self-play (AlphaZero-style learning)
leanaide_strategies.py         # Strategy library and tactics
```

### 1.2 Import Patterns

```python
# Import all evolutionary components
from leanaide_evolution import (
    LeanProofEvolutionEngine,
    LeanProofStrategy,
    evolve_proof
)

from leanaide_adversarial import (
    LeanAdversarialEvolution,
    LeanBlueTeamAgent,
    LeanRedTeamAgent,
    evolve_lean_proof
)

from leanaide_selfplay import (
    LeanSelfPlayEngine,
    LeanProofAgent,
    LeanProofExperience
)
```

---

## 2. Genetic Evolution API

### 2.1 Main Classes

#### `LeanProofEvolutionEngine`

Main evolutionary engine for genetic proof generation.

**Signature:**
```python
class LeanProofEvolutionEngine:
    def __init__(
        self,
        theorem: str,
        theorem_name: Optional[str] = None,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
        elitism_ratio: float = 0.1,
        server_url: str = "http://localhost:7654",
        convergence_threshold: float = 0.001,
        stagnation_limit: int = 10,
        target_fitness: float = 8.0,
        cache_enabled: bool = True,
        parallel_evaluation: bool = True
    )
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `theorem` | `str` | Required | Theorem statement (Lean syntax or natural language) |
| `theorem_name` | `Optional[str]` | `None` | Name for the theorem (auto-generated if None) |
| `population_size` | `int` | `20` | Number of proof strategies in population |
| `max_generations` | `int` | `50` | Maximum generations to evolve |
| `mutation_rate` | `float` | `0.1` | Probability of mutation per gene |
| `crossover_rate` | `float` | `0.8` | Probability of crossover between parents |
| `selection_method` | `SelectionMethod` | `TOURNAMENT` | Method for selecting parents |
| `crossover_method` | `CrossoverMethod` | `UNIFORM` | Method for combining parents |
| `elitism_ratio` | `float` | `0.1` | Fraction of elite strategies to preserve |
| `server_url` | `str` | `"http://localhost:7654"` | LeanAide server URL |
| `convergence_threshold` | `float` | `0.001` | Minimum improvement to avoid stagnation |
| `stagnation_limit` | `int` | `10` | Generations without improvement before stopping |
| `target_fitness` | `float` | `8.0` | Target fitness for early termination |
| `cache_enabled` | `bool` | `True` | Enable verification caching |
| `parallel_evaluation` | `bool` | `True` | Enable parallel fitness evaluation |

**Methods:**

##### `async evolve() -> EvolutionResult`

Run the evolutionary proof generation process.

**Returns:** `EvolutionResult` containing best proof and statistics

**Example:**
```python
engine = LeanProofEvolutionEngine(
    theorem="∀ n : Nat, n + 0 = n",
    population_size=30,
    max_generations=50
)

result = await engine.evolve()

if result.success:
    print(f"Found proof: {result.best_proof.lean_code}")
else:
    print(f"Best fitness: {result.best_strategy.fitness:.3f}")
```

**Raises:**
- `ConnectionError`: If LeanAide server unavailable
- `TimeoutError`: If verification timeout exceeded
- `RuntimeError`: If evolution fails critically

##### `async generate_initial_population() -> List[LeanProofStrategy]`

Generate initial population of proof strategies.

**Returns:** List of initial strategies

**Example:**
```python
strategies = await engine.generate_initial_population()
print(f"Generated {len(strategies)} strategies")
```

##### `async evaluate_population()`

Evaluate all strategies in current population.

**Example:**
```python
await engine.evaluate_population()
best = engine.population.get_best_strategy()
print(f"Best fitness: {best.fitness:.3f}")
```

##### `async create_next_generation()`

Create next generation through selection, crossover, and mutation.

**Example:**
```python
for generation in range(max_generations):
    await engine.evaluate_population()
    await engine.create_next_generation()
```

##### `get_best_proof() -> Optional[LeanProof]`

Get the best proof found so far.

**Returns:** Best proof or None if no population

**Example:**
```python
best = engine.get_best_proof()
if best and best.verification_result:
    print(f"Verification: {best.verification_result.is_valid}")
```

##### `async close()`

Clean up resources and close connections.

**Example:**
```python
try:
    result = await engine.evolve()
finally:
    await engine.close()
```

---

### 2.2 Supporting Classes

#### `LeanProofStrategy`

Represents an individual proof strategy in the population.

**Attributes:**
```python
@dataclass
class LeanProofStrategy:
    proof: LeanProof                    # The proof itself
    fitness: float = 0.0                # Quality score
    generation: int = 0                 # Generation created
    parents: List[str]                   # Parent strategy IDs
    mutation_history: List[MutationType] # Mutations applied
    birth_time: float                   # Creation timestamp
    evaluation_count: int = 0           # Times evaluated
    verified: bool = False              # Whether proof verified
    diversity_score: float = 0.0        # Diversity metric
    complexity_score: float = 0.0       # Proof complexity
    elegance_score: float = 0.0         # Elegance metric
    strategy_id: str                    # Unique identifier
```

**Methods:**

##### `get_tactics_sequence() -> str`

Get tactics as formatted string.

**Returns:** String representation of tactics

**Example:**
```python
sequence = strategy.get_tactics_sequence()
# "intros\nsimp\nnorm_num"
```

##### `calculate_complexity() -> float`

Calculate proof complexity (0-10 scale).

**Returns:** Complexity score

**Example:**
```python
complexity = strategy.calculate_complexity()
print(f"Complexity: {complexity:.2f}/10")
```

##### `calculate_elegance() -> float`

Calculate elegance score based on conciseness and simplicity.

**Returns:** Elegance score (0-1)

**Example:**
```python
elegance = strategy.calculate_elegance()
print(f"Elegance: {elegance:.2%}")
```

##### `to_dict() -> Dict[str, Any]`

Convert strategy to dictionary for serialization.

**Returns:** Dictionary representation

---

#### `LeanProofPopulation`

Manages a population of proof strategies.

**Signature:**
```python
class LeanProofPopulation:
    def __init__(
        self,
        strategies: List[LeanProofStrategy],
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        tournament_size: int = 3,
        elitism_ratio: float = 0.1
    )
```

**Methods:**

##### `get_best_strategy() -> Optional[LeanProofStrategy]`

Get strategy with highest fitness.

**Returns:** Best strategy or None

##### `get_worst_strategy() -> Optional[LeanProofStrategy]`

Get strategy with lowest fitness.

**Returns:** Worst strategy or None

##### `calculate_diversity() -> float`

Calculate population diversity using tactic sequence variation.

**Returns:** Diversity score (0-1, higher is more diverse)

##### `calculate_statistics() -> PopulationStatistics`

Calculate comprehensive population statistics.

**Returns:** `PopulationStatistics` object

##### `select_parents(num_parents: int) -> List[LeanProofStrategy]`

Select parent strategies using configured method.

**Parameters:**
- `num_parents`: Number of parents to select

**Returns:** List of selected parent strategies

##### `get_elites(num_elites: int) -> List[LeanProofStrategy]`

Get top N strategies (elitism).

**Parameters:**
- `num_elites`: Number of elites to return

**Returns:** List of elite strategies

---

#### `LeanProofMutator`

Applies mutations to proof strategies.

**Signature:**
```python
class LeanProofMutator:
    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.5,
        custom_tactics: Optional[List[str]] = None
    )
```

**Methods:**

##### `mutate(strategy: LeanProofStrategy) -> LeanProofStrategy`

Apply mutations to a strategy.

**Parameters:**
- `strategy`: Strategy to mutate

**Returns:** New mutated strategy (original not modified)

**Example:**
```python
mutator = LeanProofMutator(mutation_rate=0.15)
new_strategy = mutator.mutate(old_strategy)
```

**Mutation Types Applied:**
- `TACTIC_SUBSTITUTION`: Replace tactic with alternative
- `STEP_INSERTION`: Add new proof step
- `STEP_DELETION`: Remove proof step
- `GOAL_RESTRUCTURING`: Reorganize proof structure
- `LEMMA_INTRODUCTION`: Add helper lemma
- `LEMMA_REMOVAL`: Remove helper lemma
- `REORDERING`: Change tactic order
- `SIMPLIFICATION`: Simplify tactics

---

#### `LeanProofCrossover`

Performs crossover between two proof strategies.

**Signature:**
```python
class LeanProofCrossover:
    def __init__(self, crossover_rate: float = 0.8)
```

**Methods:**

##### `crossover(parent1, parent2, method) -> LeanProofStrategy`

Perform crossover between two parents.

**Parameters:**
- `parent1`: First parent strategy
- `parent2`: Second parent strategy
- `method`: Crossover method (see `CrossoverMethod` enum)

**Returns:** Child strategy combining parents

**Crossover Methods:**
- `SINGLE_POINT`: Split at random point and combine
- `TWO_POINT`: Select segment from one parent
- `UNIFORM`: Each tactic randomly selected from either parent
- `ORDERED`: Preserve relative order of tactics

**Example:**
```python
crossover = LeanProofCrossover(crossover_rate=0.8)
child = crossover.crossover(
    parent1,
    parent2,
    CrossoverMethod.UNIFORM
)
```

---

#### `LeanProofEvaluator`

Evaluates proof strategies using LeanAide verification.

**Signature:**
```python
class LeanProofEvaluator:
    def __init__(
        self,
        verification_engine: Optional[Lean4VerificationEngine] = None,
        server_url: str = "http://localhost:7654",
        cache_enabled: bool = True,
        parallel_evaluation: bool = True,
        max_concurrent: int = 5
    )
```

**Methods:**

##### `async evaluate(strategy, timeout=None) -> float`

Evaluate a single proof strategy.

**Parameters:**
- `strategy`: Strategy to evaluate
- `timeout`: Optional verification timeout in seconds

**Returns:** Fitness score (higher is better)

##### `async evaluate_population(strategies, timeout=None) -> Dict[str, float]`

Evaluate multiple strategies in parallel.

**Parameters:**
- `strategies`: List of strategies to evaluate
- `timeout`: Optional timeout per evaluation

**Returns:** Dictionary mapping strategy IDs to fitness scores

**Example:**
```python
evaluator = LeanProofEvaluator(parallel_evaluation=True)
fitnesses = await evaluator.evaluate_population(population.strategies)

for strategy_id, fitness in fitnesses.items():
    print(f"{strategy_id}: {fitness:.3f}")
```

##### `async close()`

Close the verification engine and release resources.

---

### 2.3 Data Classes

#### `EvolutionResult`

Result of an evolutionary proof generation run.

**Attributes:**
```python
@dataclass
class EvolutionResult:
    success: bool                              # Whether verified proof found
    best_proof: Optional[LeanProof]            # Best proof found
    best_strategy: Optional[LeanProofStrategy] # Best strategy
    generations_completed: int                 # Generations run
    total_evaluations: int                     # Total fitness evaluations
    evolution_time: float                      # Time in seconds
    statistics_history: List[PopulationStatistics]  # Stats per generation
    family_tree: Dict[str, List[str]]          # Parent -> children mapping
    failed_attempts: List[Dict[str, Any]]      # Failed attempts with errors
    convergence_history: List[float]           # Average fitness per generation
```

**Methods:**

##### `to_dict() -> Dict[str, Any]`

Convert result to dictionary.

**Example:**
```python
result = await engine.evolve()
result_dict = result.to_dict()

import json
print(json.dumps(result_dict, indent=2))
```

---

#### `PopulationStatistics`

Statistics about a population at a given generation.

**Attributes:**
```python
@dataclass
class PopulationStatistics:
    generation: int              # Current generation number
    population_size: int         # Number of strategies
    best_fitness: float          # Highest fitness in population
    worst_fitness: float         # Lowest fitness in population
    average_fitness: float       # Mean fitness
    fitness_std: float           # Standard deviation of fitness
    diversity_score: float       # Population diversity metric
    verified_count: int          # Number of verified proofs
    unique_strategies: int       # Number of unique tactic sequences
    average_complexity: float    # Mean complexity score
    average_elegance: float      # Mean elegance score
    convergence_rate: float      # Rate of fitness improvement
```

---

### 2.4 Convenience Functions

#### `evolve_proof()`

Convenience function for evolutionary proof generation.

**Signature:**
```python
async def evolve_proof(
    theorem: str,
    theorem_name: Optional[str] = None,
    max_generations: int = 50,
    population_size: int = 20,
    server_url: str = "http://localhost:7654",
    **kwargs
) -> EvolutionResult
```

**Parameters:** Same as `LeanProofEvolutionEngine.__init__`

**Returns:** `EvolutionResult`

**Example:**
```python
from leanaide_evolution import evolve_proof

result = await evolve_proof(
    theorem="∀ n : Nat, n + 0 = n",
    max_generations=30,
    population_size=30
)

print(f"Success: {result.success}")
print(f"Generations: {result.generations_completed}")
```

---

## 3. Adversarial Evolution API

### 3.1 Main Classes

#### `LeanAdversarialEvolution`

Main orchestrator for adversarial proof evolution.

**Signature:**
```python
class LeanAdversarialEvolution:
    def __init__(
        self,
        api_key: Optional[str] = None,
        lean_path: Optional[str] = None,
        knowledge_base_path: Optional[str] = None
    )
```

**Parameters:**
- `api_key`: OpenAI API key for LLM-based generation
- `lean_path`: Path to Lean 4 installation
- `knowledge_base_path`: Path to store learned patterns

**Methods:**

##### `async run_adversarial_evolution(theorem, context, rounds) -> Tuple[LeanProofStrategy, List[RoundResult], AdversarialStatistics]`

Run adversarial evolution on a theorem.

**Parameters:**
- `theorem`: Theorem statement to prove
- `context`: Proof context (creates empty if None)
- `rounds`: Number of adversarial rounds (default: 10)

**Returns:** Tuple of (final_proof, round_results, statistics)

**Example:**
```python
evolution = LeanAdversarialEvolution(api_key="sk-...")

final_proof, round_results, statistics = await evolution.run_adversarial_evolution(
    theorem="theorem injective {f : α → β} : ...",
    rounds=12
)

print(f"Blue win rate: {statistics.blue_success_rate:.1%}")
print(f"Most effective: {statistics.most_effective_approach.value}")
```

##### `blue_team_generate_proof(theorem, context) -> LeanProofStrategy`

Generate a proof using Blue Team.

**Parameters:**
- `theorem`: Theorem statement
- `context`: Proof context

**Returns:** Generated proof strategy

##### `red_team_critique(proof, theorem, context) -> List[ProofCritique]`

Critique a proof using Red Team.

**Parameters:**
- `proof`: Proof strategy to critique
- `theorem`: Original theorem
- `context`: Proof context

**Returns:** List of critiques identifying issues

##### `generate_counterexample(theorem, proof, critique) -> Optional[LeanCounterexample]`

Generate counterexample for failed proof.

**Parameters:**
- `theorem`: Theorem statement
- `proof`: Proof being critiqued
- `critique`: Critique suggesting counterexample

**Returns:** Counterexample if successful, None otherwise

##### `get_evolution_report() -> Dict[str, Any]`

Generate comprehensive evolution report.

**Returns:** Dictionary with evolution metrics

**Example:**
```python
report = evolution.get_evolution_report()
print(f"Total evolutions: {report['total_evolutions']}")
print(f"Leaderboard: {report['leaderboard']}")
```

---

### 3.2 Team Classes

#### `LeanBlueTeamAgent`

Blue Team agent that generates proof strategies.

**Signature:**
```python
class LeanBlueTeamAgent:
    def __init__(
        self,
        name: str = "BlueTeam",
        client: Optional['LeanAideClient'] = None,
        approaches: List[ProofApproach] = None
    )
```

**Methods:**

##### `generate_proof_strategy(theorem, context, previous_critiques, approach) -> LeanProofStrategy`

Generate a proof strategy for the given theorem.

**Parameters:**
- `theorem`: Theorem statement
- `context`: Proof context
- `previous_critiques`: Critiques from previous rounds to learn from
- `approach`: Specific approach to use (None to select adaptively)

**Returns:** `LeanProofStrategy` with tactics and Lean code

**Example:**
```python
blue_team = LeanBlueTeamAgent()

strategy = blue_team.generate_proof_strategy(
    theorem="∀ n : Nat, n + 0 = n",
    context=ProofContext(),
    previous_critiques=[],
    approach=ProofApproach.CONSTRUCTIVE
)

print(f"Approach: {strategy.approach.value}")
print(f"Tactics: {strategy.get_tactics_sequence()}")
```

##### `update_performance(approach, success)`

Update performance history for learning.

**Parameters:**
- `approach`: The approach used
- `success`: Success score (0-1)

##### `adapt_tactics(critiques)`

Adapt tactic selection based on critiques.

**Parameters:**
- `critiques`: List of critiques to learn from

---

#### `LeanRedTeamAgent`

Red Team agent that critiques proofs and finds counterexamples.

**Signature:**
```python
class LeanRedTeamAgent:
    def __init__(
        self,
        name: str = "RedTeam",
        client: Optional['LeanAideClient'] = None
    )
```

**Methods:**

##### `critique_proof(strategy, theorem, context) -> List[ProofCritique]`

Critique a proof strategy and identify flaws.

**Parameters:**
- `strategy`: The proof strategy to critique
- `theorem`: The theorem being proved
- `context`: Proof context

**Returns:** List of critiques identifying issues

**Example:**
```python
red_team = LeanRedTeamAgent()

critiques = red_team.critique_proof(
    strategy=blue_strategy,
    theorem="∀ n : Nat, n + 0 = n",
    context=ProofContext()
)

for critique in critiques:
    print(f"[{critique.severity.value}] {critique.issue_type}")
    print(f"  {critique.description}")
    if critique.fix_suggestion:
        print(f"  Fix: {critique.fix_suggestion}")
```

---

### 3.3 Supporting Classes

#### `LeanAdversarialArena`

Manages adversarial competition between Blue and Red teams.

**Signature:**
```python
class LeanAdversarialArena:
    def __init__(
        self,
        blue_team: LeanBlueTeamAgent,
        red_team: LeanRedTeamAgent,
        counterexample_generator: LeanCounterexampleGenerator
    )
```

**Methods:**

##### `run_adversarial_evolution(theorem, context, rounds) -> Tuple[LeanProofStrategy, List[RoundResult]]`

Run full adversarial evolution process.

**Parameters:**
- `theorem`: The theorem to prove
- `context`: Proof context
- `rounds`: Number of adversarial rounds

**Returns:** Tuple of (final_proof, round_results)

##### `run_adversarial_round(theorem, context, previous_critiques, round_number) -> RoundResult`

Run a single adversarial round.

**Parameters:**
- `theorem`: The theorem statement
- `context`: Proof context
- `previous_critiques`: Critiques from previous round
- `round_number`: Current round number

**Returns:** `RoundResult` with round statistics

##### `get_leaderboard() -> Dict[str, Any]`

Get current leaderboard statistics.

**Returns:** Dictionary with rankings and statistics

---

#### `LeanCounterexampleGenerator`

Generates and validates counterexamples for theorems.

**Signature:**
```python
class LeanCounterexampleGenerator:
    def __init__(
        self,
        client: Optional['LeanAideClient'] = None,
        knowledge_base_path: Optional[str] = None
    )
```

**Methods:**

##### `generate_counterexample(theorem, proof, critique) -> Optional[LeanCounterexample]`

Generate a counterexample based on a critique.

**Parameters:**
- `theorem`: The theorem statement
- `proof`: The proof strategy being critiqued
- `critique`: The critique suggesting a counterexample

**Returns:** `LeanCounterexample` if successful, None otherwise

##### `suggest_fix(theorem, counterexample) -> Optional[str]`

Suggest how to fix theorem based on counterexample.

**Parameters:**
- `theorem`: The theorem statement
- `counterexample`: Validated counterexample

**Returns:** Suggestion text or None

---

### 3.4 Data Classes

#### `RoundResult`

Result of an adversarial round.

**Attributes:**
```python
@dataclass
class RoundResult:
    round_number: int
    blue_strategy: LeanProofStrategy
    red_critique: List[ProofCritique]
    counterexamples: List[LeanCounterexample]
    blue_survived: bool              # Did proof survive attacks?
    red_score: float                  # 0-1, red team success
    blue_score: float                 # 0-1, blue team success
    improvements_made: List[str]      # Improvements suggested
    time_taken: float                 # Round duration in seconds
```

---

#### `ProofCritique`

A critique of a proof from the Red Team.

**Attributes:**
```python
@dataclass
class ProofCritique:
    issue_type: str                  # "logic_gap", "invalid_step", etc.
    description: str                 # Description of the issue
    severity: CritiqueSeverity        # CRITICAL, HIGH, MEDIUM, LOW, INFO
    location: Optional[str]          # Where in the proof
    counterexample_suggestion: Optional[str]  # Suggested counterexample
    confidence: float                # 0-1, confidence in critique
    fix_suggestion: Optional[str]     # Suggested fix
```

---

#### `LeanCounterexample`

A counterexample to a theorem or proof.

**Attributes:**
```python
@dataclass
class LeanCounterexample:
    theorem: str                     # The theorem being disproven
    counterexample_code: str         # Lean 4 counterexample code
    description: str                 # Human-readable description
    status: CounterexampleStatus     # VALID, INVALID, INCONCLUSIVE, PENDING
    verification_output: Optional[str]  # Verification results
    disproves_theorem: bool          # Whether it disproves the theorem
    confidence: float                # 0-1, confidence in counterexample
```

---

#### `AdversarialStatistics`

Statistics tracking adversarial performance.

**Attributes:**
```python
@dataclass
class AdversarialStatistics:
    total_rounds: int = 0
    blue_wins: int = 0
    red_wins: int = 0
    draws: int = 0
    blue_success_rate: float = 0.0
    red_success_rate: float = 0.0
    average_proof_complexity: float = 0.0
    unique_counterexamples_found: int = 0
    most_effective_approach: Optional[ProofApproach] = None
    approach_success_rates: Dict[ProofApproach, float]
```

---

### 3.5 Convenience Functions

#### `evolve_lean_proof()`

Convenience function for adversarial evolution.

**Signature:**
```python
def evolve_lean_proof(
    theorem: str,
    rounds: int = 10,
    api_key: Optional[str] = None
) -> Dict[str, Any]
```

**Returns:** Dictionary with proof, statistics, and history

**Example:**
```python
from leanaide_adversarial import evolve_lean_proof

result = evolve_lean_proof(
    theorem="∀ n : Nat, n + 0 = n",
    rounds=10
)

print(f"Strategy: {result['strategy']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Blue win rate: {result['statistics']['blue_win_rate']:.1%}")
```

---

## 4. Self-Play API

### 4.1 Main Classes

#### `LeanSelfPlayEngine`

Main self-play engine for Lean 4 proof improvement.

**Signature:**
```python
class LeanSelfPlayEngine:
    def __init__(
        self,
        leanaide_url: str = "http://localhost:7654",
        llm_config: Optional[Dict[str, Any]] = None,
        buffer_capacity: int = 10000,
        max_concurrent_games: int = 4
    )
```

**Parameters:**
- `leanaide_url`: LeanAide server URL
- `llm_config`: Configuration for LLM (provider, model, api_key)
- `buffer_capacity`: Maximum experiences in replay buffer
- `max_concurrent_games`: Maximum parallel self-play games

**Methods:**

##### `async run_self_play(theorem, games) -> LeanProof`

Run self-play for a specific theorem.

**Parameters:**
- `theorem`: Theorem statement or ID
- `games`: Number of self-play games to play (default: 10)

**Returns:** Best proof found

**Example:**
```python
engine = LeanSelfPlayEngine()

best_proof = await engine.run_self_play(
    theorem="∀ n : Nat, n + 0 = n",
    games=20
)

if best_proof.is_valid:
    print(f"Verified proof: {best_proof.lean_code}")
```

##### `async run_batch_self_play(theorems, games_per_theorem) -> Dict[str, LeanProof]`

Run self-play for multiple theorems.

**Parameters:**
- `theorems`: List of theorem statements
- `games_per_theorem`: Number of games per theorem

**Returns:** Dictionary mapping theorem ID to best proof

**Example:**
```python
theorems = [
    "∀ n : Nat, n + 0 = n",
    "∀ a b : Nat, a + b = b + a"
]

results = await engine.run_batch_self_play(
    theorems=theorems,
    games_per_theorem=10
)

for theorem, proof in results.items():
    print(f"{theorem}: valid={proof.is_valid}")
```

##### `async train_from_buffer(batch_size, iterations) -> TrainingMetrics`

Train agent from experience buffer.

**Parameters:**
- `batch_size`: Number of experiences per training batch
- `iterations`: Number of training iterations

**Returns:** `TrainingMetrics` with training statistics

**Example:**
```python
metrics = await engine.train_from_buffer(
    batch_size=32,
    iterations=100
)

print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Value loss: {metrics.value_loss:.4f}")
```

##### `get_training_progress() -> Dict[str, Any]`

Get training progress summary.

**Returns:** Dictionary with progress metrics

**Example:**
```python
progress = engine.get_training_progress()
print(f"Iteration: {progress['iteration']}")
print(f"Success rate: {progress['success_rate']:.1%}")
print(f"Avg reward: {progress['avg_reward']:.3f}")
print(f"Improvement: {progress['improvement']['relative']:.1%}")
```

##### `save_checkpoint(filepath)`

Save training checkpoint to disk.

**Parameters:**
- `filepath`: Path to save checkpoint

##### `load_checkpoint(filepath)`

Load training checkpoint from disk.

**Parameters:**
- `filepath`: Path to load checkpoint from

##### `async close()`

Clean up resources and close connections.

---

### 4.2 Supporting Classes

#### `LeanProofAgent`

Agent that generates and verifies Lean 4 proofs.

**Signature:**
```python
class LeanProofAgent:
    def __init__(
        self,
        agent_id: str,
        llm_config: Dict[str, Any],
        verifier: Lean4Verifier,
        exploration_rate: float = 0.3,
        temperature: float = 0.8
    )
```

**Methods:**

##### `async select_proof_strategy(theorem, training) -> LeanProofStrategy`

Select a proof strategy for the given theorem.

**Parameters:**
- `theorem`: Theorem to prove
- `training`: Whether in training mode (affects exploration)

**Returns:** Selected proof strategy

##### `async generate_proof(theorem, strategy) -> LeanProof`

Generate a proof for the theorem using the given strategy.

**Parameters:**
- `theorem`: Theorem to prove
- `strategy`: Proof strategy to use

**Returns:** Generated proof

##### `async evaluate_proof(proof) -> float`

Evaluate the quality of a proof.

**Parameters:**
- `proof`: Proof to evaluate

**Returns:** Value in [0, 1] where 1 is best

##### `update_performance(result)`

Update agent performance tracking.

**Parameters:**
- `result`: Result dictionary with performance metrics

---

#### `LeanSelfPlayGame`

Single self-play game episode.

**Signature:**
```python
class LeanSelfPlayGame:
    def __init__(
        self,
        theorem: LeanTheorem,
        agent: LeanProofAgent,
        verifier: Lean4Verifier
    )
```

**Methods:**

##### `async play() -> LeanProofExperience`

Execute a self-play game.

**Returns:** Experience tuple for training

**Example:**
```python
game = LeanSelfPlayGame(theorem, agent, verifier)
experience = await game.play()

print(f"Reward: {experience.reward:.3f}")
print(f"Value: {experience.value_estimate:.3f}")
print(f"Valid: {experience.proof.is_valid}")
```

---

#### `LeanProofExperienceBuffer`

Replay buffer for storing and sampling proof experiences.

**Signature:**
```python
class LeanProofExperienceBuffer:
    def __init__(
        self,
        capacity: int = 10000,
        prioritized: bool = True,
        priority_alpha: float = 0.6,
        priority_epsilon: float = 1e-6
    )
```

**Methods:**

##### `add(experience)`

Add experience to buffer.

##### `sample(batch_size, beta) -> List[LeanProofExperience]`

Sample a batch of experiences.

**Parameters:**
- `batch_size`: Number of experiences to sample
- `beta`: Importance sampling weight (0-1)

**Returns:** List of sampled experiences

##### `get_statistics() -> Dict[str, Any]`

Get buffer statistics.

**Returns:** Dictionary with size, success rate, avg reward, etc.

##### `save(filepath)`

Save buffer to disk.

##### `load(filepath)`

Load buffer from disk.

---

#### `Lean4Verifier`

Interface to Lean 4 theorem prover for proof verification.

**Signature:**
```python
class Lean4Verifier:
    def __init__(
        self,
        leanaide_url: str = "http://localhost:7654",
        timeout: int = 300
    )
```

**Methods:**

##### `async verify_proof(theorem, proof) -> Tuple[ProofStatus, str, str]`

Verify a Lean 4 proof using LeanAide server.

**Parameters:**
- `theorem`: Theorem to verify
- `proof`: Proof to verify

**Returns:** Tuple of (status, output, error_message)

**Example:**
```python
verifier = Lean4Verifier()

status, output, error = await verifier.verify_proof(theorem, proof)

if status == ProofStatus.VERIFIED:
    print("Proof verified successfully")
elif status == ProofStatus.FAILED:
    print(f"Proof failed: {error}")
```

##### `async close()`

Close the HTTP client.

---

### 4.3 Data Classes

#### `LeanTheorem`

A Lean 4 theorem to be proven.

**Attributes:**
```python
@dataclass
class LeanTheorem:
    id: str                          # Unique identifier
    statement: str                   # Theorem statement
    lean_code: str                   # Lean 4 code context
    difficulty: ProofDifficulty     # Difficulty level
    domain: str                      # Mathematical domain
    dependencies: List[str]          # Dependencies
    metadata: Dict[str, Any]         # Additional metadata
    created_at: float                # Creation timestamp
```

---

#### `LeanProof`

A complete Lean 4 proof.

**Attributes:**
```python
@dataclass
class LeanProof:
    theorem_id: str                  # Associated theorem
    tactics: List[LeanTactic]        # Proof tactics
    lean_code: str                   # Complete Lean code
    status: ProofStatus              # Verification status
    verification_output: str         # Verification output
    error_message: str               # Error message if failed
    confidence: float                # 0-1, confidence in proof
    generation_time: float           # Time to generate
    verification_time: float         # Time to verify
    metadata: Dict[str, Any]         # Additional metadata

    @property
    def tactic_count(self) -> int: ...

    @property
    def is_valid(self) -> bool: ...
```

---

#### `LeanTactic`

A single Lean 4 tactic.

**Attributes:**
```python
@dataclass
class LeanTactic:
    name: str                       # Tactic name (e.g., "simp")
    args: List[str]                 # Tactic arguments
    metadata: Dict[str, Any]         # Additional metadata

    def __str__(self) -> str: ...
```

---

#### `LeanProofExperience`

Experience from a self-play game.

**Attributes:**
```python
@dataclass
class LeanProofExperience:
    theorem: LeanTheorem             # Theorem being proven
    proof: LeanProof                 # Proof attempt
    reward: float                    # Reward signal
    strategy_used: str               # Strategy name
    value_estimate: float            # Estimated value
    policy_output: Dict[str, float]  # Strategy probabilities
    timestamp: float                 # Experience timestamp

    def to_training_dict(self) -> Dict[str, Any]: ...
```

---

#### `TrainingMetrics`

Metrics from training iteration.

**Attributes:**
```python
@dataclass
class TrainingMetrics:
    iteration: int                   # Training iteration
    total_games: int                 # Total games played
    success_rate: float              # Proportion of successful proofs
    avg_reward: float                # Average reward
    avg_proof_length: float          # Average proof length
    value_loss: float                # Value network loss
    policy_loss: float               # Policy network loss
    buffer_size: int                 # Current buffer size
    unique_theorems: int             # Number of unique theorems
    timestamp: float                 # Metrics timestamp
```

---

## 5. Strategy Library API

### 5.1 Main Classes

#### `LeanProofStrategy` (from Strategy Library)

A reusable proof strategy pattern.

**Attributes:**
```python
@dataclass
class LeanProofStrategy:
    name: str                       # Strategy name
    tactic_sequence: List[str]       # Default tactic sequence
    description: str                 # Strategy description
    applicable_domains: List[str]    # Applicable domains
    success_rate: float              # Historical success rate
    avg_proof_length: float          # Average proof length
    metadata: Dict[str, Any]         # Additional metadata
```

**Predefined Strategies:**

1. **Direct Proof** (`direct_proof`)
   - Tactics: `["intro", "apply", "exact"]`
   - Domains: Logic, Algebra
   - Description: Direct forward reasoning

2. **Proof by Contradiction** (`proof_by_contradiction`)
   - Tactics: `["intro", "by_contradiction", "contradiction"]`
   - Domains: Logic, Set Theory
   - Description: Assume negation and derive contradiction

3. **Induction** (`induction`)
   - Tactics: `["induction", "case", "simp"]`
   - Domains: Combinatorics, Algebra
   - Description: Proof by induction

4. **Calculation** (`calculation`)
   - Tactics: `["calc", "rw", "simp", "norm_num"]`
   - Domains: Algebra, Analysis
   - Description: Step-by-step calculation

---

### 5.2 Domain-Specific Tactics

**Logic Tactics:**
```python
LOGIC_TACTICS = [
    "intro",      # Introduce hypotheses
    "apply",      # Apply theorem
    "exact",      # Exact tactic
    "by",         # Structured proof
    "have",       # Intermediate claim
    "show"        # Show goal
]
```

**Algebra Tactics:**
```python
ALGEBRA_TACTICS = [
    "ring",       # Ring solver
    "linarith",   # Linear arithmetic
    "norm_num",   # Normalize numbers
    "field_simp", # Field simplification
    "abel"        # Abelian group tactics
]
```

**Analysis Tactics:**
```python
ANALYSIS_TACTICS = [
    "continuity",           # Continuity tactics
    "differentiability",    # Differentiation
    "integral"              # Integration
]
```

**Combinatorics Tactics:**
```python
COMBINATORICS_TACTICS = [
    "induction",   # Mathematical induction
    "cases",       # Case analysis
    "rcases"       # Robust case analysis
]
```

**General Tactics:**
```python
GENERAL_TACTICS = [
    "rw",          # Rewrite
    "simp",        # Simplify
    "assumption",  # Use assumption
    "contradiction"  # Find contradiction
]
```

---

## 6. Data Structures

### 6.1 Proof Context

```python
@dataclass
class ProofContext:
    available_lemmas: List[str]        # Available lemmas/theorems
    definitions: Dict[str, str]         # Type/term definitions
    imports: List[str]                  # Required imports
    tactics: Set[str]                    # Available tactics
    metadata: Dict[str, Any]            # Additional context
```

---

## 7. Enums

### 7.1 MutationType

```python
class MutationType(Enum):
    TACTIC_SUBSTITUTION = "tactic_substitution"
    STEP_INSERTION = "step_insertion"
    STEP_DELETION = "step_deletion"
    GOAL_RESTRUCTURING = "goal_restructuring"
    LEMMA_INTRODUCTION = "lemma_introduction"
    LEMMA_REMOVAL = "lemma_removal"
    REORDERING = "reordering"
    SIMPLIFICATION = "simplification"
```

### 7.2 SelectionMethod

```python
class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    STOCHASTIC_UNIVERSAL_SAMPLING = "stochastic_universal_sampling"
    TRUNCATION = "truncation"
```

### 7.3 CrossoverMethod

```python
class CrossoverMethod(Enum):
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ORDERED = "ordered"
    CYCLE = "cycle"
```

### 7.4 ProofApproach (Adversarial)

```python
class ProofApproach(Enum):
    CONSTRUCTIVE = "constructive"
    CLASSICAL = "classical"
    COMPUTATIONAL = "computational"
    INDIRECT = "indirect"
    STRUCTURAL = "structural"
    ALGEBRAIC = "algebraic"
```

### 7.5 CritiqueSeverity

```python
class CritiqueSeverity(Enum):
    CRITICAL = "critical"    # Proof is invalid
    HIGH = "high"           # Significant gap or flaw
    MEDIUM = "medium"       # Minor issue or weakness
    LOW = "low"            # Style or clarity issue
    INFO = "info"          # Suggestion for improvement
```

### 7.6 CounterexampleStatus

```python
class CounterexampleStatus(Enum):
    VALID = "valid"                  # Counterexample disproves theorem
    INVALID = "invalid"              # Counterexample doesn't work
    INCONCLUSIVE = "inconclusive"    # Cannot verify
    PENDING = "pending"              # Awaiting verification
```

### 7.7 ProofDifficulty (Self-Play)

```python
class ProofDifficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    RESEARCH = "research"
```

### 7.8 ProofStatus (Self-Play)

```python
class ProofStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
```

---

## 8. Error Handling

### 8.1 Exception Hierarchy

```python
# Base exception
class LeanAideEvolutionError(Exception):
    """Base exception for evolutionary LeanAide"""
    pass

# Connection errors
class LeanAideConnectionError(LeanAideEvolutionError):
    """Failed to connect to LeanAide server"""
    pass

# Verification errors
class LeanAideVerificationError(LeanAideEvolutionError):
    """Proof verification failed"""
    pass

# Evolution errors
class EvolutionConvergenceError(LeanAideEvolutionError):
    """Evolution failed to converge"""
    pass

class EvolutionStagnationError(LeanAideEvolutionError):
    """Evolution stagnated without improvement"""
    pass

# Configuration errors
class EvolutionConfigurationError(LeanAideEvolutionError):
    """Invalid evolutionary configuration"""
    pass
```

### 8.2 Error Handling Patterns

**Basic error handling:**
```python
try:
    result = await evolve_proof(theorem=theorem)
    if result.success:
        print(f"Proof: {result.best_proof.lean_code}")
    else:
        print(f"No proof found. Best fitness: {result.best_strategy.fitness:.3f}")

except LeanAideConnectionError as e:
    print(f"Cannot connect to LeanAide server: {e}")
    print("Check that LeanAide is running at http://localhost:7654")

except EvolutionConfigurationError as e:
    print(f"Invalid configuration: {e}")

except EvolutionStagnationError as e:
    print(f"Evolution stagnated: {e}")
    print("Try adjusting mutation_rate or increasing population_size")

except Exception as e:
    print(f"Unexpected error: {e}")
    logger.exception("Evolution failed")
```

**Advanced error handling with retries:**
```python
import asyncio

async def evolve_with_retry(
    theorem: str,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> EvolutionResult:
    """Evolve proof with automatic retry on connection errors"""

    for attempt in range(max_retries):
        try:
            engine = LeanProofEvolutionEngine(theorem=theorem)
            result = await engine.evolve()
            await engine.close()
            return result

        except LeanAideConnectionError as e:
            if attempt < max_retries - 1:
                print(f"Connection failed (attempt {attempt + 1}), retrying...")
                await asyncio.sleep(retry_delay)
            else:
                raise

        except Exception as e:
            # Don't retry on other errors
            raise

    raise LeanAideEvolutionError("All retries exhausted")
```

---

## Appendix A: Type Annotations

**Common Type Aliases:**
```python
from typing import (
    Any, Dict, List, Optional, Tuple, Union,
    Callable, Set, Coroutine
)

# Async function returning result
AsyncEvolutionResult = Coroutine[Any, Any, EvolutionResult]

# Strategy list
StrategyList = List[LeanProofStrategy]

# Fitness scores
FitnessScores = Dict[str, float]

# Round results list
RoundResults = List[RoundResult]

# Experience batch
ExperienceBatch = List[LeanProofExperience]
```

---

## Appendix B: Performance Tuning Parameters

**Quick tuning reference:**

```python
# For faster results (less thorough)
population_size=20
max_generations=30
parallel_evaluation=True
max_concurrent=10

# For better results (slower)
population_size=50
max_generations=100
elitism_ratio=0.15
verification_weight=12.0

# For difficult theorems
population_size=100
max_generations=150
mutation_rate=0.15
custom_tactics=["specialized_tactics"]

# For simple theorems
population_size=10
max_generations=10
mutation_rate=0.05

# For batch processing
cache_enabled=True
parallel_evaluation=True
max_concurrent=20
```

---

**Document End**

For usage examples, see `LEANAIDE_EVOLUTIONARY_EXAMPLES.md`
For integration guide, see `LEANAIDE_INTEGRATION_GUIDE.md`
