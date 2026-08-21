# LeanAide MDAP-Enhanced Evolution - API Reference

> **STATUS: implemented** (see `integrations/leanaide/leanaide_evolution_mdap.py` — `MDAPEvolutionConfig`, `MDAPResult`, `MDAPLeanPopulation`, `MDAPLeanSelector`, `MDAPLeanCrossover`, `MDAPLeanMutator`; plus `integrations/leanaide/leanaide_evolution_mdap_workflow.py` and `engines/other/evolution_maker_integration.py` for `MAKEREvolutionEngine`).
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide Evolution + MDAP Integration

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration Classes](#2-configuration-classes)
3. [Data Structures](#3-data-structures)
4. [Core Components](#4-core-components)
5. [Main Functions](#5-main-functions)
6. [Utility Functions](#6-utility-functions)
7. [Return Types](#7-return-types)
8. [Error Handling](#8-error-handling)
9. [Type Aliases](#9-type-aliases)

---

## 1. Overview

This document provides a complete API reference for the MDAP-enhanced evolutionary computation system, including all classes, functions, parameters, return types, and usage examples.

### Module Structure

```
evolution_maker_integration.py
├── Configuration
│   ├── MakerevolutionMode (enum)
│   └── MakerevolutionConfig (dataclass)
├── Data Structures
│   ├── Individual (dataclass)
│   └── Population (dataclass)
├── Core Components
│   ├── MAKERSelection (class)
│   ├── MDAPEvolutionDecomposer (class)
│   └── MAKEREvolutionEngine (class)
└── Functions
    ├── run_maker_evolution()
    └── get_maker_evolution_capabilities()
```

---

## 2. Configuration Classes

### 2.1 MakerevolutionMode

Enum defining evolution modes for MDAP-enhanced evolution.

```python
class MakerevolutionMode(Enum):
    """MAKER-enhanced evolution modes"""
    VOTING_ONLY = "voting_only"
    DECOMPOSITION = "decomposition"
    HYBRID = "hybrid"
    FULL_MAKER = "full_maker"
```

**Values**:
- `VOTING_ONLY`: Use MAKER voting for selection only (no decomposition)
- `DECOMPOSITION`: Use MDAP for task decomposition only (no voting)
- `HYBRID`: Combine MAKER voting + MDAP decomposition (recommended)
- `FULL_MAKER`: Complete MAKER-based evolution with maximum reliability

**Example**:
```python
from evolution_maker_integration import MakerevolutionMode

mode = MakerevolutionMode.HYBRID
print(mode.value)  # Output: "hybrid"
```

---

### 2.2 MakerevolutionConfig

Configuration dataclass for MAKER-enhanced evolutionary computation.

```python
@dataclass
class MakerevolutionConfig:
    """Configuration for MAKER-enhanced evolution"""

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

#### Parameters

**mode** (`MakerevolutionMode`):
- Evolution mode (voting, decomposition, hybrid, or full_maker)
- Default: `MakerevolutionMode.HYBRID`

**enable_voting** (`bool`):
- Enable voting-based parent selection
- Default: `True`

**voting_threshold** (`int`):
- K value for first-to-ahead-by-K voting
- Range: 2-8 (higher = more conservative)
- Default: `3`
- k=2: 95% success, fast
- k=3: 99% success, balanced
- k=5: 99.9% success, high-stakes
- k=8: 99.99% success, safety-critical

**population_size** (`int`):
- Number of individuals in population
- Range: 10-100
- Default: `20`

**num_candidates** (`int`):
- Number of candidates for voting (N = 2*k - 1)
- Default: `5` (automatically set based on k)

**enable_decomposition** (`bool`):
- Enable MDAP task decomposition
- Default: `True`

**decomposition_depth** (`int`):
- Maximum recursion depth for decomposition
- Range: 1-5
- Default: `3`

**max_subtasks** (`int`):
- Maximum number of subtasks to create
- Default: `10`

**enable_red_flagging** (`bool`):
- Enable red-flagging for quality control
- Default: `True`

**convergence_threshold** (`float`):
- Convergence threshold (0-1)
- Default: `0.95`

**max_iterations_without_improvement** (`int`):
- Maximum generations without improvement before stopping
- Default: `10`

**adaptive_voting** (`bool`):
- Enable adaptive voting threshold adjustment
- Default: `True`

**diversity_threshold** (`float`):
- Minimum diversity threshold (0-1)
- Default: `0.3`

**max_token_length** (`int`):
- Maximum token length for generated proofs
- Default: `750`

**temperature** (`float`):
- LLM temperature for agent generation
- Range: 0.0-1.0
- Default: `0.7`

#### Methods

**to_dict() -> Dict[str, Any]**
Convert configuration to dictionary.

```python
config = MakerevolutionConfig()
config_dict = config.to_dict()
print(config_dict["mode"])  # Output: "hybrid"
```

**Returns**: Dictionary representation of configuration

#### Example

```python
from evolution_maker_integration import MakerevolutionConfig, MakerevolutionMode

# Standard configuration
config = MakerevolutionConfig(
    mode=MakerevolutionMode.HYBRID,
    voting_threshold=3,
    population_size=30,
    enable_decomposition=True
)

# High-reliability configuration
high_reliability_config = MakerevolutionConfig(
    mode=MakerevolutionMode.FULL_MAKER,
    voting_threshold=5,
    population_size=50,
    enable_decomposition=True,
    decomposition_depth=5
)
```

---

## 3. Data Structures

### 3.1 Individual

Represents an individual in the evolution population.

```python
@dataclass
class Individual:
    """Represents an evolved individual"""
    genome: str
    fitness: float
    generation: int
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Attributes

**genome** (`str`):
- The program/content (e.g., Lean 4 proof code)
- Required

**fitness** (`float`):
- Fitness score (higher is better)
- Range: typically 0-10 for proofs
- Required

**generation** (`int`):
- Generation number when individual was created
- Required

**metadata** (`Dict[str, Any]`):
- Additional metadata (e.g., tactics, verification status)
- Default: Empty dict

#### Example

```python
from evolution_maker_integration import Individual

individual = Individual(
    genome="intros n refl",
    fitness=0.95,
    generation=5,
    metadata={
        "verified": True,
        "tactics": ["intros", "refl"],
        "proof_length": 2
    }
)

print(individual.fitness)  # Output: 0.95
print(individual.metadata["verified"])  # Output: True
```

---

### 3.2 Population

Represents a population of individuals.

```python
@dataclass
class Population:
    """Represents a population of individuals"""
    individuals: List[Individual]
    generation: int
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### Attributes

**individuals** (`List[Individual]`):
- List of individuals in the population
- Required

**generation** (`int`):
- Current generation number
- Required

**metadata** (`Dict[str, Any]`):
- Additional population metadata
- Default: Empty dict

#### Properties

**best_individual -> Optional[Individual]**
Get the best individual in the population (highest fitness).

```python
population = Population(individuals=[...], generation=0)
best = population.best_individual
print(best.fitness)
```

**Returns**: Best individual or None if population is empty

**average_fitness -> float**
Get average fitness of the population.

```python
avg = population.average_fitness
print(f"Average fitness: {avg:.3f}")
```

**Returns**: Average fitness (0.0 if population is empty)

**diversity -> float**
Calculate population diversity (normalized Hamming distance).

```python
diversity = population.diversity
print(f"Diversity: {diversity:.2f}")
```

**Returns**: Diversity score (0-1, where 1 = high diversity)

#### Example

```python
from evolution_maker_integration import Individual, Population

# Create population
individuals = [
    Individual(genome=f"proof_{i}", fitness=0.5 + i * 0.1, generation=0)
    for i in range(10)
]

population = Population(
    individuals=individuals,
    generation=0
)

# Access properties
best = population.best_individual
avg_fitness = population.average_fitness
diversity = population.diversity

print(f"Best: {best.genome} (fitness={best.fitness})")
print(f"Average fitness: {avg_fitness:.3f}")
print(f"Diversity: {diversity:.3f}")
```

---

## 4. Core Components

### 4.1 MAKERSelection

Voting-based selection operator using first-to-ahead-by-K consensus.

```python
class MAKERSelection:
    """Voting-based selection using MAKER framework"""

    def __init__(self, config: MakerevolutionConfig):
        """Initialize selector with configuration"""
        ...
```

#### Constructor

**MAKERSelection(config: MakerevolutionConfig)**
Initialize the selector with configuration.

```python
from evolution_maker_integration import MAKERSelection, MakerevolutionConfig

config = MakerevolutionConfig(voting_threshold=3)
selector = MAKERSelection(config)
```

**Parameters**:
- `config`: MAKER evolution configuration

#### Methods

**select(population: Population, num_parents: int = 2) -> List[Individual]**
Select parents using voting-based selection.

```python
parents = selector.select(population, num_parents=2)
print(f"Selected {len(parents)} parents")
```

**Parameters**:
- `population`: Population to select from
- `num_parents`: Number of parents to select (default: 2)

**Returns**: List of selected parent individuals

**_select_top_candidates(population: Population, n: int) -> List[Individual]**
Select top N candidates by fitness.

**Parameters**:
- `population`: Population to select from
- `n`: Number of candidates to select

**Returns**: List of top N candidates

**_voting_selection(candidates: List[Individual]) -> Individual**
Perform voting-based selection from candidates.

**Parameters**:
- `candidates`: List of candidates to vote on

**Returns**: Winning individual

#### Example

```python
from evolution_maker_integration import (
    MAKERSelection,
    MakerevolutionConfig,
    Individual,
    Population
)

# Create population
individuals = [
    Individual(genome=f"proof_{i}", fitness=0.5 + i * 0.1, generation=0)
    for i in range(10)
]
population = Population(individuals=individuals, generation=0)

# Create selector
config = MakerevolutionConfig(voting_threshold=3)
selector = MAKERSelection(config)

# Select parents
parents = selector.select(population, num_parents=2)

for i, parent in enumerate(parents):
    print(f"Parent {i+1}: {parent.genome} (fitness={parent.fitness})")
```

---

### 4.2 MDAPEvolutionDecomposer

Task decomposer for evolutionary computation using MDAP.

```python
class MDAPEvolutionDecomposer:
    """Decompose evolutionary tasks using MDAP"""

    def __init__(self, config: MakerevolutionConfig):
        """Initialize decomposer with configuration"""
        ...
```

#### Constructor

**MDAPEvolutionDecomposer(config: MakerevolutionConfig)**
Initialize the decomposer with configuration.

```python
from evolution_maker_integration import MDAPEvolutionDecomposer, MakerevolutionConfig

config = MakerevolutionConfig(
    enable_decomposition=True,
    decomposition_depth=3
)
decomposer = MDAPEvolutionDecomposer(config)
```

**Parameters**:
- `config`: MAKER evolution configuration

#### Methods

**decompose_task(task: str, evaluator: Callable[[str], float]) -> List[str]**
Decompose a task into subtasks.

```python
subtasks = decomposer.decompose_task(
    task="prove complex theorem",
    evaluator=my_evaluator
)
print(f"Decomposed into {len(subtasks)} subtasks")
```

**Parameters**:
- `task`: Task to decompose (e.g., theorem statement)
- `evaluator`: Fitness evaluator function

**Returns**: List of subtask descriptions

**analyze_landscape(population: Population) -> Dict[str, Any]**
Analyze fitness landscape of population.

**Parameters**:
- `population`: Population to analyze

**Returns**: Dictionary with landscape metrics (diversity, fitness variance, etc.)

#### Example

```python
from evolution_maker_integration import MDAPEvolutionDecomposer, MakerevolutionConfig

config = MakerevolutionConfig(
    enable_decomposition=True,
    decomposition_depth=3
)
decomposer = MDAPEvolutionDecomposer(config)

# Decompose complex theorem
theorem = "∀ (f : Nat → Nat), (∀ n, f n = 0) → f = (λ _, 0)"
subtasks = decomposer.decompose_task(theorem, evaluator=my_evaluator)

for i, subtask in enumerate(subtasks):
    print(f"Subtask {i+1}: {subtask}")
```

---

### 4.3 MAKEREvolutionEngine

Main evolution engine combining MAKER voting with evolutionary computation.

```python
class MAKEREvolutionEngine:
    """MAKER-enhanced evolution engine"""

    def __init__(self, config: MakerevolutionConfig):
        """Initialize engine with configuration"""
        ...
```

#### Constructor

**MAKEREvolutionEngine(config: MakerevolutionConfig)**
Initialize the evolution engine.

```python
from evolution_maker_integration import MAKEREvolutionEngine, MakerevolutionConfig

config = MakerevolutionConfig(
    mode=MakerevolutionMode.HYBRID,
    voting_threshold=3,
    population_size=30
)
engine = MAKEREvolutionEngine(config)
```

**Parameters**:
- `config`: MAKER evolution configuration

#### Methods

**initialize_population(initial_program: str, evaluator: Callable[[str], float]) -> Population**
Initialize population with variants of initial program.

```python
population = engine.initialize_population(
    initial_program="intros n refl",
    evaluator=my_evaluator
)
```

**Parameters**:
- `initial_program`: Starting program/proof
- `evaluator`: Fitness evaluator function

**Returns**: Initialized population

**evolve(population: Population, evaluator: Callable[[str], float], max_generations: int) -> Dict[str, Any]**
Run evolution for specified generations.

```python
result = engine.evolve(
    population=population,
    evaluator=my_evaluator,
    max_generations=30
)
```

**Parameters**:
- `population`: Initial population
- `evaluator`: Fitness evaluator function
- `max_generations`: Maximum generations to run

**Returns**: Dictionary with evolution results:
- `best_program`: Best evolved program
- `best_fitness`: Fitness of best program
- `generations_completed`: Number of generations run
- `converged`: Whether population converged
- `statistics`: Evolution statistics

**_create_next_generation(population: Population, evaluator: Callable[[str], float]) -> Population**
Create next generation through selection, crossover, mutation.

**Parameters**:
- `population`: Current population
- `evaluator`: Fitness evaluator function

**Returns**: Next generation population

#### Properties

**current_generation -> int**
Get current generation number.

**population -> Population**
Get current population.

**statistics -> Dict[str, Any]**
Get evolution statistics.

#### Example

```python
from evolution_maker_integration import (
    MAKEREvolutionEngine,
    MakerevolutionConfig,
    MakerevolutionMode
)

# Configure engine
config = MakerevolutionConfig(
    mode=MakerevolutionMode.HYBRID,
    voting_threshold=3,
    population_size=30,
    enable_decomposition=True
)

# Create engine
engine = MAKEREvolutionEngine(config)

# Initialize population
def evaluator(genome: str) -> float:
    if "verified" in genome:
        return 10.0
    elif "intros" in genome and "refl" in genome:
        return 5.0
    return 1.0

population = engine.initialize_population(
    initial_program="intros n refl",
    evaluator=evaluator
)

# Run evolution
result = engine.evolve(
    population=population,
    evaluator=evaluator,
    max_generations=30
)

print(f"Best fitness: {result['best_fitness']:.3f}")
print(f"Best program: {result['best_program']}")
print(f"Generations: {result['generations_completed']}")
print(f"Converged: {result['converged']}")
```

---

## 5. Main Functions

### 5.1 run_maker_evolution

Main entry point for MAKER-enhanced evolution.

```python
def run_maker_evolution(
    initial_program: str,
    evaluator: Callable[[str], float],
    max_generations: int = 50,
    config: Optional[MakerevolutionConfig] = None
) -> Dict[str, Any]
```

#### Parameters

**initial_program** (`str`):
- Initial program/proof to evolve
- Required

**evaluator** (`Callable[[str], float]`):
- Fitness evaluator function
- Takes genome as input, returns fitness (higher is better)
- Required

**max_generations** (`int`):
- Maximum generations to run
- Default: `50`

**config** (`Optional[MakerevolutionConfig]`):
- MAKER evolution configuration
- Default: `None` (uses default configuration)

#### Returns

Dictionary with evolution results:

```python
{
    "best_program": str,           # Best evolved program
    "best_fitness": float,         # Fitness of best program
    "generations_completed": int,  # Generations run
    "converged": bool,             # Whether converged
    "final_population_size": int,  # Final population size
    "statistics": {                # Evolution statistics
        "average_fitness": float,
        "best_fitness_history": List[float],
        "average_fitness_history": List[float],
        "diversity_history": List[float]
    }
}
```

#### Example

```python
from evolution_maker_integration import run_maker_evolution, MakerevolutionConfig

# Define evaluator
def evaluator(genome: str) -> float:
    """Higher fitness is better"""
    if "verified" in genome:
        return 10.0
    elif "intros" in genome and "refl" in genome:
        return 5.0
    return 1.0

# Run evolution
result = run_maker_evolution(
    initial_program="intros n refl",
    evaluator=evaluator,
    max_generations=30,
    config=MakerevolutionConfig(
        voting_threshold=3,
        population_size=20
    )
)

# Access results
print(f"Best program: {result['best_program']}")
print(f"Best fitness: {result['best_fitness']:.3f}")
print(f"Generations: {result['generations_completed']}")

# Access statistics
stats = result['statistics']
print(f"Final average fitness: {stats['average_fitness']:.3f}")
```

---

### 5.2 get_maker_evolution_capabilities

Check MAKER evolution capabilities and dependencies.

```python
def get_maker_evolution_capabilities() -> Dict[str, bool]
```

#### Returns

Dictionary indicating availability of components:

```python
{
    "mdap_available": bool,         # MDAP engine available
    "maker_available": bool,        # MAKER framework available
    "evolution_available": bool,    # Evolution module available
    "full_integration": bool        # Full integration available
}
```

#### Example

```python
from evolution_maker_integration import get_maker_evolution_capabilities

caps = get_maker_evolution_capabilities()

print("MAKER Evolution Capabilities:")
for component, available in caps.items():
    status = "✓" if available else "✗"
    print(f"  {status} {component}")

if not caps["full_integration"]:
    print("Warning: Full integration not available")
```

---

## 6. Utility Functions

### 6.1 create_random_individual

Create a random individual for testing.

```python
def create_random_individual(genome_length: int = 10) -> Individual:
    """Create random individual"""
    ...
```

#### Parameters

**genome_length** (`int`):
- Length of random genome to generate
- Default: `10`

#### Returns

Random individual

#### Example

```python
from evolution_maker_integration import create_random_individual

individual = create_random_individual(genome_length=5)
print(individual.genome)
```

---

### 6.2 calculate_population_diversity

Calculate diversity metric for population.

```python
def calculate_population_diversity(population: Population) -> float:
    """Calculate population diversity (0-1)"""
    ...
```

#### Parameters

**population** (`Population`):
- Population to analyze

#### Returns

Diversity score (0-1, where 1 = high diversity)

#### Example

```python
from evolution_maker_integration import calculate_population_diversity

diversity = calculate_population_diversity(population)
print(f"Population diversity: {diversity:.3f}")
```

---

## 7. Return Types

### 7.1 EvolutionResult

Dictionary returned by `run_maker_evolution()` and `MAKEREvolutionEngine.evolve()`.

```python
{
    "best_program": str,
    "best_fitness": float,
    "generations_completed": int,
    "converged": bool,
    "final_population_size": int,
    "statistics": {
        "average_fitness": float,
        "best_fitness_history": List[float],
        "average_fitness_history": List[float],
        "diversity_history": List[float],
        "verified_count": int,
        "total_evaluations": int
    }
}
```

#### Fields

**best_program** (`str`):
- Best evolved program/proof

**best_fitness** (`float`):
- Fitness of best program

**generations_completed** (`int`):
- Number of generations executed

**converged** (`bool`):
- Whether population converged before max_generations

**final_population_size** (`int`):
- Size of final population

**statistics** (`Dict[str, Any]`):
- Detailed evolution statistics

---

### 7.2 CapabilitiesResult

Dictionary returned by `get_maker_evolution_capabilities()`.

```python
{
    "mdap_available": bool,
    "maker_available": bool,
    "evolution_available": bool,
    "full_integration": bool
}
```

---

## 8. Error Handling

### 8.1 Exceptions

**MAKEREvolutionError**
Base exception for MAKER evolution errors.

```python
class MAKEREvolutionError(Exception):
    """Base exception for MAKER evolution errors"""
    pass
```

**ConfigurationError**
Raised when configuration is invalid.

```python
class ConfigurationError(MAKEREvolutionError):
    """Invalid configuration"""
    pass
```

**PopulationError**
Raised when population operations fail.

```python
class PopulationError(MAKEREvolutionError):
    """Population operation error"""
    pass
```

**VotingError**
Raised when voting fails.

```python
class VotingError(MAKEREvolutionError):
    """Voting operation error"""
    pass
```

### 8.2 Error Handling Examples

```python
from evolution_maker_integration import (
    run_maker_evolution,
    MAKEREvolutionError,
    ConfigurationError
)

try:
    result = run_maker_evolution(
        initial_program="test proof",
        evaluator=my_evaluator,
        max_generations=30
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Fix configuration and retry
except MAKEREvolutionError as e:
    print(f"Evolution error: {e}")
    # Handle error
except Exception as e:
    print(f"Unexpected error: {e}")
    # Fallback behavior
```

---

## 9. Type Aliases

```python
# Type aliases for common types
FitnessFunction = Callable[[str], float]
PopulationInitializer = Callable[[str, FitnessFunction], Population]
SelectionOperator = Callable[[Population, int], List[Individual]]
CrossoverOperator = Callable[[Individual, Individual], Tuple[Individual, Individual]]
MutationOperator = Callable[[Individual], Individual]
```

---

## Appendix A: Complete Example

```python
"""
Complete example of MDAP-enhanced evolution for Lean 4 proof generation
"""

from evolution_maker_integration import (
    run_maker_evolution,
    MakerevolutionConfig,
    MakerevolutionMode,
    get_maker_evolution_capabilities
)

# Check capabilities
caps = get_maker_evolution_capabilities()
if not caps["full_integration"]:
    print("Warning: Full integration not available")
    exit(1)

# Define fitness evaluator for Lean 4 proofs
def lean4_proof_evaluator(proof: str) -> float:
    """
    Evaluate Lean 4 proof quality.

    Returns:
        float: Fitness score (0-10, higher is better)
    """
    score = 0.0

    # Check for verification
    if "verified" in proof.lower():
        score += 10.0
    elif "sorry" in proof.lower():
        score += 0.0  # Admitted proof
        return score

    # Check for essential structure
    if "intros" in proof or "intro" in proof:
        score += 2.0

    if "refl" in proof or "rfl" in proof:
        score += 2.0

    if "simp" in proof:
        score += 1.0

    # Prefer shorter proofs
    tactic_count = len(proof.split())
    score -= min(tactic_count * 0.1, 2.0)

    return max(score, 0.0)

# Configure MDAP-enhanced evolution
config = MakerevolutionConfig(
    # Use hybrid mode (voting + decomposition)
    mode=MakerevolutionMode.HYBRID,

    # Voting parameters
    enable_voting=True,
    voting_threshold=3,  # 99% success rate
    population_size=30,

    # Decomposition parameters
    enable_decomposition=True,
    decomposition_depth=3,
    max_subtasks=10,

    # Convergence parameters
    enable_red_flagging=True,
    convergence_threshold=0.95,
    max_iterations_without_improvement=10,

    # Adaptive parameters
    adaptive_voting=True,
    diversity_threshold=0.3
)

# Theorem to prove
theorem = "∀ n : Nat, n + 0 = n"

# Initial proof sketch
initial_proof = f"""
theorem add_zero (n : Nat) : n + 0 = n :=
  intros n
  sorry
"""

# Run evolution
print(f"Evolving proof for: {theorem}")
print(f"Initial proof:\n{initial_proof}\n")

result = run_maker_evolution(
    initial_program=initial_proof,
    evaluator=lean4_proof_evaluator,
    max_generations=30,
    config=config
)

# Display results
print("=" * 70)
print("EVOLUTION COMPLETE")
print("=" * 70)
print(f"Best fitness: {result['best_fitness']:.3f}")
print(f"Generations: {result['generations_completed']}")
print(f"Converged: {result['converged']}")
print(f"\nBest proof:\n{result['best_program']}")

# Display statistics
stats = result['statistics']
print("\n" + "=" * 70)
print("STATISTICS")
print("=" * 70)
print(f"Final average fitness: {stats['average_fitness']:.3f}")
print(f"Total evaluations: {stats['total_evaluations']}")
print(f"Verified proofs: {stats['verified_count']}")
```

---

**Document End**

For more information, see:
- `LEANAIDE_EVOLUTION_MDAP_GUIDE.md` - User guide
- `LEANAIDE_EVOLUTION_MDAP_EXAMPLES.md` - Real-world examples
- `LEANAIDE_EVOLUTION_MDAP_ARCHITECTURE.md` - Architecture diagrams
