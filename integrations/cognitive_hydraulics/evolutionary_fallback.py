"""Evolutionary Solver - Genetic algorithm fallback.

Triggered when ACT-R fails or pressure >= 0.9.
Population-based code/solution generation.

Algorithm:
    1. Initialize population of candidate solutions
    2. Evaluate fitness of each individual
    3. Select parents using tournament selection
    4. Apply crossover and mutation
    5. Repeat until convergence or max generations

Fitness Criteria:
    - Syntax correctness (compilation)
    - Runtime success (execution without error)
    - Output correctness (expected result)
    - Efficiency (time/steps used)

Genetic Operators:
    - Selection: Tournament selection
    - Crossover: Single-point or uniform
    - Mutation: Random modification
"""

import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timezone
from enum import Enum, auto
from copy import deepcopy

import numpy as np

from .config import EvolutionaryConfig

logger = logging.getLogger(__name__)


class SolutionType(Enum):
    """Types of solutions that can be evolved."""
    CODE = auto()
    EXPRESSION = auto()
    SEQUENCE = auto()
    STRUCTURE = auto()


@dataclass
class Individual:
    """Single candidate solution."""
    individual_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    genome: Any = None
    fitness: float = 0.0
    
    # Evaluation metrics
    syntax_correct: bool = False
    runtime_success: bool = False
    output_correct: bool = False
    efficiency_score: float = 0.0
    
    # Metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def calculate_fitness(self, weights: Optional[Dict[str, float]] = None):
        """Calculate composite fitness from metrics."""
        w = weights or {
            "syntax": 0.2,
            "runtime": 0.3,
            "output": 0.4,
            "efficiency": 0.1
        }
        
        self.fitness = (
            w["syntax"] * (1.0 if self.syntax_correct else 0.0) +
            w["runtime"] * (1.0 if self.runtime_success else 0.0) +
            w["output"] * (1.0 if self.output_correct else 0.0) +
            w["efficiency"] * self.efficiency_score
        )
        
        return self.fitness


@dataclass
class Population:
    """Collection of candidate solutions."""
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    
    def get_best(self) -> Optional[Individual]:
        """Get individual with highest fitness."""
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda x: x.fitness)
    
    def get_average_fitness(self) -> float:
        """Get average fitness of population."""
        if not self.individuals:
            return 0.0
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)
    
    def get_diversity(self) -> float:
        """Measure genetic diversity as average pairwise distance."""
        if len(self.individuals) < 2:
            return 0.0
        
        distances = []
        for i, ind1 in enumerate(self.individuals):
            for ind2 in self.individuals[i+1:]:
                dist = self._genome_distance(ind1.genome, ind2.genome)
                distances.append(dist)
        
        return sum(distances) / len(distances) if distances else 0.0
    
    def _genome_distance(self, g1: Any, g2: Any) -> float:
        """Calculate distance between two genomes."""
        if isinstance(g1, str) and isinstance(g2, str):
            # String distance
            if len(g1) == 0 or len(g2) == 0:
                return 1.0
            # Simple normalized edit distance
            max_len = max(len(g1), len(g2))
            return sum(c1 != c2 for c1, c2 in zip(g1, g2)) / max_len
        
        if isinstance(g1, list) and isinstance(g2, list):
            # List distance
            max_len = max(len(g1), len(g2))
            if max_len == 0:
                return 0.0
            diff = sum(1 for i in range(min(len(g1), len(g2))) if g1[i] != g2[i])
            diff += abs(len(g1) - len(g2))
            return diff / max_len
        
        # Default: simple equality
        return 0.0 if g1 == g2 else 1.0
    
    def select_tournament(self, tournament_size: int = 3) -> Individual:
        """Select individual using tournament selection."""
        contestants = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        return max(contestants, key=lambda x: x.fitness)


class FitnessEvaluator:
    """Evaluate solution fitness."""
    
    def __init__(
        self,
        syntax_checker: Optional[Callable[[Any], bool]] = None,
        runtime_tester: Optional[Callable[[Any], Tuple[bool, Any]]] = None,
        output_validator: Optional[Callable[[Any, Any], bool]] = None
    ):
        self.syntax_checker = syntax_checker
        self.runtime_tester = runtime_tester
        self.output_validator = output_validator
    
    def evaluate(
        self,
        individual: Individual,
        expected_output: Optional[Any] = None,
        timeout_seconds: int = 5
    ) -> Individual:
        """Evaluate an individual's fitness."""
        # Check syntax
        if self.syntax_checker:
            individual.syntax_correct = self.syntax_checker(individual.genome)
        else:
            # Default: assume correct
            individual.syntax_correct = True
        
        # Check runtime
        if self.runtime_tester and individual.syntax_correct:
            try:
                success, output = self._run_with_timeout(
                    self.runtime_tester,
                    individual.genome,
                    timeout_seconds
                )
                individual.runtime_success = success
            except TimeoutError:
                individual.runtime_success = False
            except Exception as e:
                logger.debug(f"Runtime test error: {e}")
                individual.runtime_success = False
        
        # Check output
        if self.output_validator and individual.runtime_success:
            try:
                individual.output_correct = self.output_validator(
                    individual.genome, expected_output
                )
            except Exception as e:
                logger.debug(f"Output validation error: {e}")
                individual.output_correct = False
        
        # Calculate efficiency (simplified)
        if individual.runtime_success:
            # Fewer characters = more efficient
            if isinstance(individual.genome, str):
                individual.efficiency_score = max(0, 1.0 - (len(individual.genome) / 1000))
            else:
                individual.efficiency_score = 0.5
        
        # Calculate composite fitness
        individual.calculate_fitness()
        
        return individual
    
    def _run_with_timeout(
        self,
        func: Callable,
        arg: Any,
        timeout: int
    ) -> Any:
        """Run function with timeout."""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, arg)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Function timed out after {timeout} seconds")


class GeneticOperators:
    """Genetic algorithm operators."""
    
    def __init__(
        self,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7
    ):
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Character set for string mutations
        self.charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    
    def mutate(self, individual: Individual, solution_type: SolutionType) -> Individual:
        """Apply mutation to an individual."""
        if random.random() > self.mutation_rate:
            return individual
        
        genome = deepcopy(individual.genome)
        
        if solution_type == SolutionType.CODE or solution_type == SolutionType.EXPRESSION:
            genome = self._mutate_string(genome)
        elif solution_type == SolutionType.SEQUENCE:
            genome = self._mutate_list(genome)
        elif solution_type == SolutionType.STRUCTURE:
            genome = self._mutate_structure(genome)
        
        mutated = Individual(
            genome=genome,
            generation=individual.generation,
            parent_ids=[individual.individual_id]
        )
        
        return mutated
    
    def _mutate_string(self, genome: str) -> str:
        """Mutate a string genome."""
        if not genome:
            return genome
        
        genome_list = list(genome)
        
        # Choose mutation type
        mutation_type = random.choice(["substitute", "insert", "delete", "swap"])
        
        if mutation_type == "substitute":
            idx = random.randint(0, len(genome_list) - 1)
            genome_list[idx] = random.choice(self.charset)
        
        elif mutation_type == "insert":
            idx = random.randint(0, len(genome_list))
            genome_list.insert(idx, random.choice(self.charset))
        
        elif mutation_type == "delete" and len(genome_list) > 1:
            idx = random.randint(0, len(genome_list) - 1)
            del genome_list[idx]
        
        elif mutation_type == "swap" and len(genome_list) > 1:
            idx1, idx2 = random.sample(range(len(genome_list)), 2)
            genome_list[idx1], genome_list[idx2] = genome_list[idx2], genome_list[idx1]
        
        return "".join(genome_list)
    
    def _mutate_list(self, genome: List[Any]) -> List[Any]:
        """Mutate a list genome."""
        if not genome:
            return genome
        
        genome = list(genome)  # Copy
        
        mutation_type = random.choice(["modify", "insert", "delete", "swap"])
        
        if mutation_type == "modify":
            idx = random.randint(0, len(genome) - 1)
            genome[idx] = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        elif mutation_type == "insert":
            genome.append(random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
        
        elif mutation_type == "delete" and len(genome) > 1:
            idx = random.randint(0, len(genome) - 1)
            del genome[idx]
        
        elif mutation_type == "swap" and len(genome) > 1:
            idx1, idx2 = random.sample(range(len(genome)), 2)
            genome[idx1], genome[idx2] = genome[idx2], genome[idx1]
        
        return genome
    
    def _mutate_structure(self, genome: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate a structure genome."""
        genome = deepcopy(genome)
        
        if not genome:
            return {"value": random.random()}
        
        # Modify random value
        key = random.choice(list(genome.keys()))
        if isinstance(genome[key], (int, float)):
            genome[key] += random.gauss(0, 1)
        elif isinstance(genome[key], str):
            genome[key] = self._mutate_string(genome[key])
        
        return genome
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
        solution_type: SolutionType
    ) -> Tuple[Individual, Individual]:
        """Apply crossover between two parents."""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        g1 = parent1.genome
        g2 = parent2.genome
        
        if solution_type == SolutionType.CODE or solution_type == SolutionType.EXPRESSION:
            c1, c2 = self._crossover_string(g1, g2)
        elif solution_type == SolutionType.SEQUENCE:
            c1, c2 = self._crossover_list(g1, g2)
        else:
            c1, c2 = g1, g2  # No crossover for structures
        
        child1 = Individual(
            genome=c1,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.individual_id, parent2.individual_id]
        )
        
        child2 = Individual(
            genome=c2,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.individual_id, parent2.individual_id]
        )
        
        return child1, child2
    
    def _crossover_string(self, s1: str, s2: str) -> Tuple[str, str]:
        """Single-point crossover for strings."""
        if not s1 or not s2:
            return s1, s2
        
        # Find common crossover points
        min_len = min(len(s1), len(s2))
        if min_len < 2:
            return s1, s2
        
        point = random.randint(1, min_len - 1)
        
        c1 = s1[:point] + s2[point:]
        c2 = s2[:point] + s1[point:]
        
        return c1, c2
    
    def _crossover_list(self, l1: List, l2: List) -> Tuple[List, List]:
        """Single-point crossover for lists."""
        if not l1 or not l2:
            return l1, l2
        
        min_len = min(len(l1), len(l2))
        if min_len < 2:
            return l1, l2
        
        point = random.randint(1, min_len - 1)
        
        c1 = l1[:point] + l2[point:]
        c2 = l2[:point] + l1[point:]
        
        return c1, c2


class EvolutionarySolver:
    """
    Main Evolutionary Solver - Genetic Algorithm Engine.
    """
    
    def __init__(
        self,
        config: Optional[EvolutionaryConfig] = None,
        solution_type: SolutionType = SolutionType.CODE
    ):
        self.config = config or EvolutionaryConfig()
        self.solution_type = solution_type
        
        # Components
        self.population = Population()
        self.fitness_evaluator = FitnessEvaluator()
        self.operators = GeneticOperators(
            mutation_rate=self.config.mutation_rate,
            crossover_rate=self.config.crossover_rate
        )
        
        # State
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.converged = False
        self.stagnation_count = 0
    
    def initialize_population(
        self,
        size: int,
        problem: Dict[str, Any],
        seed_generator: Optional[Callable[[], Any]] = None
    ):
        """Create initial population."""
        self.population.individuals = []
        
        for i in range(size):
            if seed_generator:
                genome = seed_generator()
            else:
                genome = self._default_seed(problem)
            
            individual = Individual(
                genome=genome,
                generation=0
            )
            self.population.individuals.append(individual)
        
        logger.info(f"Initialized population with {size} individuals")
    
    def _default_seed(self, problem: Dict[str, Any]) -> Any:
        """Generate default seed based on problem."""
        if self.solution_type == SolutionType.CODE:
            # Return simple code template
            return "def solution():\n    pass"
        elif self.solution_type == SolutionType.EXPRESSION:
            return "x + y"
        elif self.solution_type == SolutionType.SEQUENCE:
            return [0, 1, 2, 3, 4]
        else:
            return {"value": 0}
    
    def evaluate_fitness(
        self,
        individual: Individual,
        expected_output: Optional[Any] = None
    ) -> Individual:
        """Score solution."""
        return self.fitness_evaluator.evaluate(
            individual,
            expected_output,
            self.config.timeout_seconds
        )
    
    def evolve(self, generations: int = 10) -> Individual:
        """Run GA iterations."""
        start_time = time.time()
        
        for gen in range(generations):
            self.generation = gen
            
            # Check timeout
            if time.time() - start_time > self.config.timeout_seconds:
                logger.info("Evolution timed out")
                break
            
            # Evaluate population
            for ind in self.population.individuals:
                self.evaluate_fitness(ind)
            
            # Record best fitness
            best = self.population.get_best()
            if best:
                self.best_fitness_history.append(best.fitness)
                logger.debug(f"Gen {gen}: Best fitness = {best.fitness:.3f}")
            
            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged at generation {gen}")
                break
            
            # Create next generation
            new_population = self._create_next_generation()
            self.population.individuals = new_population
            self.population.generation = gen + 1
        
        # Final evaluation
        for ind in self.population.individuals:
            self.evaluate_fitness(ind)
        
        return self.population.get_best()
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if len(self.best_fitness_history) < 2:
            return False
        
        # Check improvement
        if len(self.best_fitness_history) >= self.config.stagnation_generations:
            recent = self.best_fitness_history[-self.config.stagnation_generations:]
            improvement = max(recent) - min(recent)
            
            if improvement < self.config.convergence_threshold:
                self.stagnation_count += 1
                if self.stagnation_count >= 3:
                    self.converged = True
                    return True
            else:
                self.stagnation_count = 0
        
        # Check if perfect solution found
        if self.best_fitness_history[-1] >= 0.99:
            self.converged = True
            return True
        
        return False
    
    def _create_next_generation(self) -> List[Individual]:
        """Create next generation through selection, crossover, mutation."""
        new_population = []
        
        # Elitism: keep best individuals
        sorted_pop = sorted(
            self.population.individuals,
            key=lambda x: x.fitness,
            reverse=True
        )
        
        elites = sorted_pop[:self.config.elitism_count]
        new_population.extend(deepcopy(elites))
        
        # Fill rest with offspring
        while len(new_population) < len(self.population.individuals):
            # Select parents
            parent1 = self.population.select_tournament()
            parent2 = self.population.select_tournament()
            
            # Crossover
            child1, child2 = self.operators.crossover(
                parent1, parent2, self.solution_type
            )
            
            # Mutation
            child1 = self.operators.mutate(child1, self.solution_type)
            child2 = self.operators.mutate(child2, self.solution_type)
            
            new_population.extend([child1, child2])
        
        # Trim to population size
        return new_population[:len(self.population.individuals)]
    
    def mutate(self, individual: Individual) -> Individual:
        """Random modification."""
        return self.operators.mutate(individual, self.solution_type)
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """Combine solutions."""
        return self.operators.crossover(parent1, parent2, self.solution_type)
    
    def select_parents(self) -> Tuple[Individual, Individual]:
        """Tournament selection."""
        return (
            self.population.select_tournament(),
            self.population.select_tournament()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get solver statistics."""
        return {
            "generation": self.generation,
            "population_size": len(self.population.individuals),
            "best_fitness": self.best_fitness_history[-1] if self.best_fitness_history else 0.0,
            "average_fitness": self.population.get_average_fitness(),
            "diversity": self.population.get_diversity(),
            "converged": self.converged,
            "stagnation_count": self.stagnation_count
        }
