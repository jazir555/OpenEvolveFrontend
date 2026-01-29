"""
Type Safety for Hybrid MAKER Integration System

This module provides comprehensive type safety:
- Type aliases
- Type guards
- Type validation
- TypedDict definitions

Author: OpenEvolve Hybrid Type Safety Team
Created: 2025-01-07
Version: 1.0.0
"""

from __future__ import annotations

from typing import (
    Any, Dict, List, Optional, Tuple, Union, TypeVar, Type, Callable,
    TypedDict, Protocol, runtime_checkable, TypeGuard, get_type_hints,
    Literal
)
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Basic types
Theorem = str
Proof = str
Tactic = str
Genome = str
Fitness = float  # 0.0 to 1.0

# Strategy types
HybridStrategyName = Union[
    Literal["mcts_then_maker"],
    Literal["maker_then_evolution"],
    Literal["maker_adversarial"],
    Literal["adaptive_maker"],
    Literal["maker_mdap_parallel"],
    Literal["full_maker_hybrid"],
]

# Individual/Population
IndividualId = str
Generation = int


# =============================================================================
# TYPED DICTS
# =============================================================================

class EvolutionResultDict(TypedDict):
    """Type for evolution result"""
    success: bool
    generations_completed: int
    evolution_time: float
    best_proof: Optional[str]
    best_fitness: float
    convergence_history: List[float]
    failed_attempts: List[Dict[str, Any]]


class IndividualDict(TypedDict):
    """Type for individual"""
    id: IndividualId
    genome: Genome
    fitness: Fitness
    generation: Generation
    metadata: Dict[str, Any]


class PopulationDict(TypedDict):
    """Type for population"""
    individuals: List[IndividualDict]
    generation: Generation
    best_fitness: Fitness
    average_fitness: Fitness
    diversity: float


class HybridConfigDict(TypedDict):
    """Type for hybrid configuration"""
    enable_voting: bool
    voting_threshold: int
    enable_decomposition: bool
    mcts_simulations: int
    evolution_generations: int
    population_size: int
    adversarial_rounds: int
    adaptive_switching: bool


# =============================================================================
# PROTOCOLS (STRUCTURAL TYPING)
# =============================================================================

@runtime_checkable
class IndividualLike(Protocol):
    """Protocol for individual-like objects"""
    genome: str
    fitness: float
    generation: int


@runtime_checkable
class PopulationLike(Protocol):
    """Protocol for population-like objects"""
    individuals: List[IndividualLike]
    generation: int


@runtime_checkable
class StrategyLike(Protocol):
    """Protocol for strategy-like objects"""
    name: str
    description: str

    async def generate_proof(self, theorem: str, **kwargs) -> EvolutionResultDict:
        """Generate proof"""
        ...


# =============================================================================
# TYPE GUARDS
# =============================================================================

def is_individual_like(obj: Any) -> TypeGuard[IndividualLike]:
    """Check if object is individual-like"""
    return (
        hasattr(obj, 'genome')
        and hasattr(obj, 'fitness')
        and hasattr(obj, 'generation')
        and isinstance(obj.genome, str)
        and isinstance(obj.fitness, (int, float))
        and isinstance(obj.generation, int)
    )


def is_population_like(obj: Any) -> TypeGuard[PopulationLike]:
    """Check if object is population-like"""
    return (
        hasattr(obj, 'individuals')
        and hasattr(obj, 'generation')
        and isinstance(obj.individuals, list)
        and isinstance(obj.generation, int)
    )


def is_evolution_result(obj: Any) -> TypeGuard[EvolutionResultDict]:
    """Check if object is evolution result"""
    return (
        isinstance(obj, dict)
        and "success" in obj
        and "generations_completed" in obj
        and "evolution_time" in obj
        and "best_fitness" in obj
        and isinstance(obj["success"], bool)
        and isinstance(obj["generations_completed"], int)
        and isinstance(obj["evolution_time"], (int, float))
        and isinstance(obj["best_fitness"], (int, float))
    )


# =============================================================================
# TYPE VALIDATION
# =============================================================================

class HybridTypeError(Exception):
    """Type validation error"""

    def __init__(self, message: str, value: Any = None, expected_type: Type = None):
        self.message = message
        self.value = value
        self.expected_type = expected_type
        super().__init__(self.message)


def validate_fitness(value: Any, field_name: str = "fitness") -> float:
    """Validate fitness value"""
    if not isinstance(value, (int, float)):
        raise HybridTypeError(
            f"{field_name} must be a number",
            value=value,
            expected_type=float
        )

    if not 0.0 <= value <= 1.0:
        raise HybridTypeError(
            f"{field_name} must be between 0.0 and 1.0",
            value=value
        )

    return float(value)


def validate_genome(value: Any, field_name: str = "genome") -> str:
    """Validate genome value"""
    if not isinstance(value, str):
        raise HybridTypeError(
            f"{field_name} must be a string",
            value=value,
            expected_type=str
        )

    if not value:
        raise HybridTypeError(
            f"{field_name} cannot be empty",
            value=value
        )

    return value


def validate_theorem(value: Any, field_name: str = "theorem") -> str:
    """Validate theorem value"""
    if not isinstance(value, str):
        raise HybridTypeError(
            f"{field_name} must be a string",
            value=value,
            expected_type=str
        )

    if len(value.strip()) == 0:
        raise HybridTypeError(
            f"{field_name} cannot be empty",
            value=value
        )

    return value.strip()


# =============================================================================
# TYPED CONSTRUCTORS
# =============================================================================

@dataclass
class TypedIndividual:
    """Type-safe individual"""
    id: IndividualId
    genome: Genome
    fitness: Fitness
    generation: Generation
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate fields"""
        self.fitness = validate_fitness(self.fitness, "fitness")
        self.genome = validate_genome(self.genome, "genome")


@dataclass
class TypedEvolutionResult:
    """Type-safe evolution result"""
    success: bool
    generations_completed: int
    evolution_time: float
    best_proof: Optional[Proof]
    best_fitness: Fitness
    convergence_history: List[Fitness]

    def __post_init__(self):
        """Validate fields"""
        if self.best_proof is not None:
            self.best_proof = validate_genome(self.best_proof, "best_proof")

        self.best_fitness = validate_fitness(self.best_fitness, "best_fitness")

        for i, fitness in enumerate(self.convergence_history):
            self.convergence_history[i] = validate_fitness(fitness, f"convergence_history[{i}]")


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    print("Hybrid MAKER Type Safety")
    print("=" * 60)

    # Demo 1: Type guards
    print("\n1. Type Guards")
    print("-" * 40)

    individual = TypedIndividual(
        id="ind_1",
        genome="simp\nrw\nrefl",
        fitness=0.85,
        generation=5,
        metadata={}
    )

    print(f"Is individual-like: {is_individual_like(individual)}")
    print(f"Is population-like: {is_population_like({'individuals': [], 'generation': 0})}")

    # Demo 2: Type validation
    print("\n2. Type Validation")
    print("-" * 40)

    try:
        validate_fitness(0.85)
        print("✓ Valid fitness")
    except HybridTypeError as e:
        print(f"✗ {e.message}")

    try:
        validate_fitness(1.5)  # Invalid
        print("✗ Should have failed!")
    except HybridTypeError as e:
        print(f"✓ Caught type error: {e.message}")

    # Demo 3: Typed constructors
    print("\n3. Typed Constructors")
    print("-" * 40)

    result = TypedEvolutionResult(
        success=True,
        generations_completed=10,
        evolution_time=5.0,
        best_proof="simp",
        best_fitness=0.9,
        convergence_history=[0.5, 0.7, 0.9]
    )

    print(f"Result: success={result.success}, fitness={result.best_fitness}")

    # Demo 4: Evolution result validation
    print("\n4. Evolution Result Validation")
    print("-" * 40)

    valid_result = {
        "success": True,
        "generations_completed": 10,
        "evolution_time": 5.0,
        "best_proof": "simp",
        "best_fitness": 0.85,
        "convergence_history": [0.5, 0.7, 0.85],
        "failed_attempts": []
    }

    print(f"Is evolution result: {is_evolution_result(valid_result)}")

    print("\n" + "=" * 60)
    print("Type safety demo complete!")
