"""
Advanced Plugins for Hybrid MAKER Integration System

This module provides advanced plugins extending the hybrid MAKER framework:
- Tactic Generators (specialized proof tactics)
- Fitness Functions (advanced evaluation metrics)
- Selection Strategies (population selection methods)
- Decomposition Plugins (task breakdown strategies)
- Crossover Operators (genetic operators)
- Mutation Operators (variation operators)

Author: OpenEvolve Hybrid Plugins Team
Created: 2025-01-07
Version: 1.0.0
"""
from __future__ import annotations


import asyncio
import random
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# BASE PLUGIN CLASSES
# =============================================================================

class TacticGeneratorPlugin(ABC):
    """Base class for tactic generation plugins"""

    plugin_name: str = "base_tactic_generator"
    plugin_version: str = "1.0.0"

    @abstractmethod
    async def generate_tactics(
        self,
        theorem: str,
        context: Dict[str, Any],
        num_tactics: int
    ) -> List[str]:
        """
        Generate proof tactics

        Args:
            theorem: Theorem statement
            context: Proof context
            num_tactics: Number of tactics to generate

        Returns:
            List of tactic strings
        """
        pass


class FitnessFunctionPlugin(ABC):
    """Base class for fitness evaluation plugins"""

    plugin_name: str = "base_fitness_function"
    plugin_version: str = "1.0.0"

    @abstractmethod
    async def evaluate(
        self,
        proof: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate proof fitness

        Args:
            proof: Proof string
            theorem: Theorem statement
            context: Proof context

        Returns:
            Fitness score (0.0 to 1.0)
        """
        pass


class SelectionStrategyPlugin(ABC):
    """Base class for population selection plugins"""

    plugin_name: str = "base_selection_strategy"
    plugin_version: str = "1.0.0"

    @abstractmethod
    async def select(
        self,
        population: List[Any],
        num_select: int,
        context: Dict[str, Any]
    ) -> List[Any]:
        """
        Select individuals from population

        Args:
            population: List of individuals
            num_select: Number to select
            context: Selection context

        Returns:
            Selected individuals
        """
        pass


class DecompositionPlugin(ABC):
    """Base class for task decomposition plugins"""

    plugin_name: str = "base_decomposition"
    plugin_version: str = "1.0.0"

    @abstractmethod
    async def decompose(
        self,
        theorem: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Decompose theorem into subtasks

        Args:
            theorem: Theorem statement
            context: Decomposition context

        Returns:
            List of subtask strings
        """
        pass


class CrossoverPlugin(ABC):
    """Base class for crossover operators"""

    plugin_name: str = "base_crossover"
    plugin_version: str = "1.0.0"

    @abstractmethod
    async def crossover(
        self,
        parent1: Any,
        parent2: Any,
        context: Dict[str, Any]
    ) -> Any:
        """
        Perform crossover between two parents

        Args:
            parent1: First parent
            parent2: Second parent
            context: Crossover context

        Returns:
            Child individual
        """
        pass


class MutationPlugin(ABC):
    """Base class for mutation operators"""

    plugin_name: str = "base_mutation"
    plugin_version: str = "1.0.0"

    @abstractmethod
    async def mutate(
        self,
        individual: Any,
        context: Dict[str, Any]
    ) -> Any:
        """
        Mutate an individual

        Args:
            individual: Individual to mutate
            context: Mutation context

        Returns:
            Mutated individual
        """
        pass


# =============================================================================
# TACTIC GENERATOR PLUGINS
# =============================================================================

class AlgebraicTacticGenerator(TacticGeneratorPlugin):
    """
    Generate algebraic manipulation tactics

    Specializes in:
    - Ring/field properties
    - Polynomial manipulations
    - Inequality reasoning
    """

    plugin_name = "algebraic_tactics"
    plugin_version = "1.0.0"

    ALGEBRAIC_TACTICS = [
        "simp",           # Simplify
        "rw [add_comm]",  # Rewrite with commutativity
        "rw [mul_assoc]", # Rewrite with associativity
        "linarith",       # Linear arithmetic
        "nlinarith",      # Non-linear arithmetic
        "ring_nf",        # Ring normal form
        "field_simp",     # Field simplification
        "apply_funext",   # Function extensionality
    ]

    async def generate_tactics(
        self,
        theorem: str,
        context: Dict[str, Any],
        num_tactics: int
    ) -> List[str]:
        """Generate algebraic tactics"""
        tactics = []

        # Detect algebraic structures in theorem
        has_addition = "+" in theorem or "add" in theorem.lower()
        has_multiplication = "*" in theorem or "mul" in theorem.lower()
        has_inequality = any(op in theorem for op in ["<=", ">=", "<", ">"])

        if has_addition:
            tactics.append("rw [add_comm]")
            tactics.append("rw [add_assoc]")

        if has_multiplication:
            tactics.append("rw [mul_comm]")
            tactics.append("rw [mul_assoc]")

        if has_inequality:
            tactics.append("linarith")
            tactics.append("nlinarith")

        # Add general algebraic tactics
        tactics.extend(["simp", "ring_nf"])

        # Fill remaining with random algebraic tactics
        while len(tactics) < num_tactics:
            tactic = random.choice(self.ALGEBRAIC_TACTICS)
            if tactic not in tactics:
                tactics.append(tactic)

        return tactics[:num_tactics]


class InductionTacticGenerator(TacticGeneratorPlugin):
    """
    Generate induction-based tactics

    Specializes in:
    - Natural number induction
    - Structural induction
    - Recursion
    """

    plugin_name = "induction_tactics"
    plugin_version = "1.0.0"

    INDUCTION_TACTICS = [
        "induction n",
        "induction m",
        "induction k",
        "cases n",
        "case h : n = 0",
        "ring",
        "simp",
        "refl",
        "assumption",
    ]

    async def generate_tactics(
        self,
        theorem: str,
        context: Dict[str, Any],
        num_tactics: int
    ) -> List[str]:
        """Generate induction tactics"""
        tactics = []

        # Detect induction candidates
        if "nat" in theorem.lower() or "∀" in theorem:
            tactics.append("induction n")

        if "list" in theorem.lower():
            tactics.append("induction l")

        # Add base case tactics
        tactics.extend(["case h : n = 0", "simp", "refl"])

        # Add inductive step tactics
        tactics.extend(["rw [add_comm]", "assumption"])

        # Fill remaining
        while len(tactics) < num_tactics:
            tactic = random.choice(self.INDUCTION_TACTICS)
            if tactic not in tactics:
                tactics.append(tactic)

        return tactics[:num_tactics]


class LogicTacticGenerator(TacticGeneratorPlugin):
    """
    Generate logical reasoning tactics

    Specializes in:
    - Propositional logic
    - First-order logic
    - Quantifier reasoning
    """

    plugin_name = "logic_tactics"
    plugin_version = "1.0.0"

    LOGIC_TACTICS = [
        "intro",
        "apply",
        "exact",
        "refine",
        "assumption",
        "cases",
        "left",
        "right",
        "split",
        "constructor",
    ]

    async def generate_tactics(
        self,
        theorem: str,
        context: Dict[str, Any],
        num_tactics: int
    ) -> List[str]:
        """Generate logic tactics"""
        tactics = []

        # Detect logical structure
        has_forall = "forall" in theorem or "∀" in theorem
        has_exists = "exists" in theorem or "∃" in theorem
        has_implies = "->" in theorem or "->" in theorem or "implies" in theorem.lower()
        has_and = "∧" in theorem or "/\\" in theorem
        has_or = "∨" in theorem or "\\/" in theorem

        if has_forall:
            tactics.append("intro")

        if has_exists:
            tactics.append("existsi")

        if has_implies:
            tactics.append("intros")

        if has_and:
            tactics.append("split")

        if has_or:
            tactics.append("cases")

        # Add general logic tactics
        tactics.extend(["apply", "assumption", "exact"])

        # Fill remaining
        while len(tactics) < num_tactics:
            tactic = random.choice(self.LOGIC_TACTICS)
            if tactic not in tactics:
                tactics.append(tactic)

        return tactics[:num_tactics]


# =============================================================================
# FITNESS FUNCTION PLUGINS
# =============================================================================

class StructuralFitnessFunction(FitnessFunctionPlugin):
    """
    Evaluate proof based on structural properties

    Considers:
    - Proof length
    - Tactic diversity
    - Syntactic correctness
    """

    plugin_name = "structural_fitness"
    plugin_version = "1.0.0"

    async def evaluate(
        self,
        proof: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate structural fitness"""
        if not proof:
            return 0.0

        score = 0.0

        # Length component (prefer medium length)
        lines = proof.split('\n')
        length_score = min(1.0, len(lines) / 10.0)
        score += 0.3 * length_score

        # Tactic diversity
        unique_tactics = len(set(line.strip().split()[0] if line.strip() else "" for line in lines))
        diversity_score = min(1.0, unique_tactics / len(lines)) if lines else 0.0
        score += 0.4 * diversity_score

        # Syntactic correctness (basic)
        valid_chars = sum(1 for c in proof if c.isalnum() or c in ' \n\t[]{}().,;:_-+*/')
        correctness_score = valid_chars / max(len(proof), 1)
        score += 0.3 * correctness_score

        return min(score, 1.0)


class SemanticFitnessFunction(FitnessFunctionPlugin):
    """
    Evaluate proof based on semantic meaning

    Considers:
    - Goal relevance
    - Hypothesis usage
    - Logical coherence
    """

    plugin_name = "semantic_fitness"
    plugin_version = "1.0.0"

    async def evaluate(
        self,
        proof: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate semantic fitness"""
        if not proof:
            return 0.0

        score = 0.0

        # Check if proof contains relevant keywords
        theorem_words = set(theorem.lower().split())
        proof_words = set(proof.lower().split())

        # Word overlap
        overlap = len(theorem_words & proof_words)
        overlap_score = min(1.0, overlap / max(len(theorem_words), 1))
        score += 0.4 * overlap_score

        # Check for theorem-specific patterns
        if "forall" in theorem.lower() or "∀" in theorem:
            if "intro" in proof.lower() or "induction" in proof.lower():
                score += 0.2

        if "nat" in theorem.lower():
            if any(t in proof.lower() for t in ["induction", "cases", "ring"]):
                score += 0.2

        if "=" in theorem:
            if "refl" in proof or "rw" in proof:
                score += 0.2

        return min(score, 1.0)


class ProgressFitnessFunction(FitnessFunctionPlugin):
    """
    Evaluate proof based on progress toward goal

    Considers:
    - Goal proximity
    - Proof state improvements
    - Error reduction
    """

    plugin_name = "progress_fitness"
    plugin_version = "1.0.0"

    async def evaluate(
        self,
        proof: str,
        theorem: str,
        context: Dict[str, Any]
    ) -> float:
        """Evaluate progress fitness"""
        if not proof:
            return 0.0

        score = 0.0

        # Check for completion indicators
        completion_tactics = ["refl", "trivial", "assumption"]
        if any(tactic in proof.lower() for tactic in completion_tactics):
            score += 0.5

        # Check for simplification
        if "simp" in proof.lower():
            score += 0.2

        # Check for progress tactics
        progress_tactics = ["rw", "apply", "exact", "refine"]
        tactic_count = sum(1 for tactic in progress_tactics if tactic in proof.lower())
        score += 0.1 * tactic_count

        return min(score, 1.0)


# =============================================================================
# SELECTION STRATEGY PLUGINS
# =============================================================================

class TournamentSelection(SelectionStrategyPlugin):
    """
    Tournament selection strategy

    Selects best individuals from random tournaments
    """

    plugin_name = "tournament_selection"
    plugin_version = "1.0.0"

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    async def select(
        self,
        population: List[Any],
        num_select: int,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Select using tournament method"""
        selected = []

        for _ in range(num_select):
            # Random tournament participants
            if len(population) < self.tournament_size:
                tournament = population
            else:
                tournament = random.sample(population, self.tournament_size)

            # Select best from tournament
            best = max(tournament, key=lambda x: getattr(x, 'fitness', 0.0))
            selected.append(best)

        return selected


class RouletteWheelSelection(SelectionStrategyPlugin):
    """
    Roulette wheel (fitness proportionate) selection

    Selects individuals with probability proportional to fitness
    """

    plugin_name = "roulette_wheel_selection"
    plugin_version = "1.0.0"

    async def select(
        self,
        population: List[Any],
        num_select: int,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Select using roulette wheel method"""
        # Calculate total fitness
        total_fitness = sum(getattr(ind, 'fitness', 0.0) for ind in population)

        if total_fitness == 0:
            return random.sample(population, min(num_select, len(population)))

        # Calculate selection probabilities
        probabilities = [
            getattr(ind, 'fitness', 0.0) / total_fitness
            for ind in population
        ]

        # Select based on probabilities
        selected = []
        for _ in range(num_select):
            r = random.random()
            cumulative = 0.0

            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(population[i])
                    break

        return selected


class RankSelection(SelectionStrategyPlugin):
    """
    Rank-based selection

    Selects based on rank rather than raw fitness
    """

    plugin_name = "rank_selection"
    plugin_version = "1.0.0"

    async def select(
        self,
        population: List[Any],
        num_select: int,
        context: Dict[str, Any]
    ) -> List[Any]:
        """Select using rank-based method"""
        # Sort by fitness
        sorted_pop = sorted(
            population,
            key=lambda x: getattr(x, 'fitness', 0.0),
            reverse=True
        )

        # Assign ranks
        total_rank = len(sorted_pop) * (len(sorted_pop) + 1) // 2

        # Select based on rank probability
        selected = []
        for _ in range(num_select):
            r = random.randint(1, total_rank)
            cumulative = 0

            for i, ind in enumerate(sorted_pop):
                rank = len(sorted_pop) - i
                cumulative += rank

                if r <= cumulative:
                    selected.append(ind)
                    break

        return selected


# =============================================================================
# DECOMPOSITION PLUGINS
# =============================================================================

class QuantifierDecomposition(DecompositionPlugin):
    """
    Decompose theorems with quantifiers

    Handles:
    - Universal quantifiers
    - Existential quantifiers
    - Nested quantifiers
    """

    plugin_name = "quantifier_decomposition"
    plugin_version = "1.0.0"

    async def decompose(
        self,
        theorem: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Decompose quantified theorem"""
        subtasks = []

        # Extract quantified variables
        if "forall" in theorem.lower() or "∀" in theorem:
            # Get variables after forall
            parts = theorem.split(',')
            if len(parts) > 1:
                # Create subtasks for each variable
                for i, part in enumerate(parts[:-1]):
                    var = part.strip().split()[-1]
                    subtasks.append(f"Case {var} = 0")
                    subtasks.append(f"Case {var} = succ n")

        # Handle implication
        if "->" in theorem or "->" in theorem or "implies" in theorem.lower():
            subtasks.append("Prove antecedent")
            subtasks.append("Use antecedent to prove consequent")

        return subtasks


class ConjunctionDecomposition(DecompositionPlugin):
    """
    Decompose conjunctive theorems

    Handles:
    - Logical AND
    - Tuple structures
    - Product types
    """

    plugin_name = "conjunction_decomposition"
    plugin_version = "1.0.0"

    async def decompose(
        self,
        theorem: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Decompose conjunctive theorem"""
        subtasks = []

        # Split on AND
        if "∧" in theorem:
            parts = theorem.split('∧')
            for i, part in enumerate(parts):
                subtasks.append(f"Prove conjunct {i+1}: {part.strip()}")

        if "/\\" in theorem:
            parts = theorem.split('/\\')
            for i, part in enumerate(parts):
                subtasks.append(f"Prove conjunct {i+1}: {part.strip()}")

        # Default
        if not subtasks:
            subtasks.append(f"Prove: {theorem}")

        return subtasks


class DisjunctionDecomposition(DecompositionPlugin):
    """
    Decompose disjunctive theorems

    Handles:
    - Logical OR
    - Union types
    - Case analysis
    """

    plugin_name = "disjunction_decomposition"
    plugin_version = "1.0.0"

    async def decompose(
        self,
        theorem: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """Decompose disjunctive theorem"""
        subtasks = []

        # Split on OR
        if "∨" in theorem:
            parts = theorem.split('∨')
            for i, part in enumerate(parts):
                subtasks.append(f"Case {i+1}: {part.strip()}")

        if "\\/" in theorem:
            parts = theorem.split('\\/')
            for i, part in enumerate(parts):
                subtasks.append(f"Case {i+1}: {part.strip()}")

        # Default
        if not subtasks:
            subtasks.append(f"Prove: {theorem}")

        return subtasks


# =============================================================================
# CROSSOVER OPERATORS
# =============================================================================

class SinglePointCrossover(CrossoverPlugin):
    """
    Single-point crossover for proofs

    Splits parents at one point and recombines
    """

    plugin_name = "single_point_crossover"
    plugin_version = "1.0.0"

    async def crossover(
        self,
        parent1: Any,
        parent2: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Perform single-point crossover"""
        genome1 = getattr(parent1, 'genome', str(parent1))
        genome2 = getattr(parent2, 'genome', str(parent2))

        # Split genomes into lines
        lines1 = genome1.split('\n')
        lines2 = genome2.split('\n')

        # Choose crossover point
        point = random.randint(1, min(len(lines1), len(lines2)) - 1)

        # Create child
        child_lines = lines1[:point] + lines2[point:]
        child_genome = '\n'.join(child_lines)

        # Return child individual
        if hasattr(parent1, '__class__'):
            return parent1.__class__(
                genome=child_genome,
                fitness=0.0,
                generation=getattr(parent1, 'generation', 0) + 1,
                metadata={"crossover": "single_point", "point": point}
            )
        else:
            return child_genome


class UniformCrossover(CrossoverPlugin):
    """
    Uniform crossover for proofs

    Randomly selects genes from either parent
    """

    plugin_name = "uniform_crossover"
    plugin_version = "1.0.0"

    async def crossover(
        self,
        parent1: Any,
        parent2: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Perform uniform crossover"""
        genome1 = getattr(parent1, 'genome', str(parent1))
        genome2 = getattr(parent2, 'genome', str(parent2))

        # Split genomes into lines
        lines1 = genome1.split('\n')
        lines2 = genome2.split('\n')

        # Randomly select from each parent
        child_lines = []
        max_len = max(len(lines1), len(lines2))

        for i in range(max_len):
            if random.random() < 0.5 and i < len(lines1):
                child_lines.append(lines1[i])
            elif i < len(lines2):
                child_lines.append(lines2[i])

        child_genome = '\n'.join(child_lines)

        # Return child individual
        if hasattr(parent1, '__class__'):
            return parent1.__class__(
                genome=child_genome,
                fitness=0.0,
                generation=getattr(parent1, 'generation', 0) + 1,
                metadata={"crossover": "uniform"}
            )
        else:
            return child_genome


# =============================================================================
# MUTATION OPERATORS
# =============================================================================

class TacticInsertionMutation(MutationPlugin):
    """
    Insert random tactics into proof

    Adds new tactics at random positions
    """

    plugin_name = "tactic_insertion_mutation"
    plugin_version = "1.0.0"

    MUTATION_TACTICS = [
        "simp",
        "rw [add_comm]",
        "rw [mul_comm]",
        "assumption",
        "refl",
        "apply",
    ]

    async def mutate(
        self,
        individual: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Insert random tactic"""
        genome = getattr(individual, 'genome', str(individual))
        lines = genome.split('\n')

        # Choose insertion point
        point = random.randint(0, len(lines))

        # Choose random tactic
        tactic = random.choice(self.MUTATION_TACTICS)

        # Insert tactic
        lines.insert(point, f"  {tactic}")

        mutated_genome = '\n'.join(lines)

        # Return mutated individual
        if hasattr(individual, '__class__'):
            return individual.__class__(
                genome=mutated_genome,
                fitness=0.0,
                generation=getattr(individual, 'generation', 0) + 1,
                metadata=getattr(individual, 'metadata', {}).copy()
            )
        else:
            return mutated_genome


class TacticDeletionMutation(MutationPlugin):
    """
    Delete random tactics from proof

    Removes tactics to simplify proof
    """

    plugin_name = "tactic_deletion_mutation"
    plugin_version = "1.0.0"

    async def mutate(
        self,
        individual: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Delete random tactic"""
        genome = getattr(individual, 'genome', str(individual))
        lines = genome.split('\n')

        if len(lines) <= 1:
            return individual

        # Choose deletion point
        point = random.randint(0, len(lines) - 1)

        # Delete tactic
        lines.pop(point)

        mutated_genome = '\n'.join(lines)

        # Return mutated individual
        if hasattr(individual, '__class__'):
            return individual.__class__(
                genome=mutated_genome,
                fitness=0.0,
                generation=getattr(individual, 'generation', 0) + 1,
                metadata=getattr(individual, 'metadata', {}).copy()
            )
        else:
            return mutated_genome


class TacticReplacementMutation(MutationPlugin):
    """
    Replace random tactics in proof

    Substitutes tactics with alternatives
    """

    plugin_name = "tactic_replacement_mutation"
    plugin_version = "1.0.0"

    REPLACEMENTS = {
        "simp": ["ring_nf", "field_simp"],
        "rw [add_comm]": ["rw [add_assoc]", "rw [mul_comm]"],
        "induction n": ["induction m", "cases n"],
    }

    async def mutate(
        self,
        individual: Any,
        context: Dict[str, Any]
    ) -> Any:
        """Replace random tactic"""
        genome = getattr(individual, 'genome', str(individual))
        lines = genome.split('\n')

        if not lines:
            return individual

        # Choose replacement point
        point = random.randint(0, len(lines) - 1)

        # Get current tactic
        current = lines[point].strip()

        # Find replacement
        for base, replacements in self.REPLACEMENTS.items():
            if base in current:
                new_tactic = random.choice(replacements)
                lines[point] = f"  {new_tactic}"
                break

        mutated_genome = '\n'.join(lines)

        # Return mutated individual
        if hasattr(individual, '__class__'):
            return individual.__class__(
                genome=mutated_genome,
                fitness=0.0,
                generation=getattr(individual, 'generation', 0) + 1,
                metadata=getattr(individual, 'metadata', {}).copy()
            )
        else:
            return mutated_genome


# =============================================================================
# PLUGIN REGISTRY
# =============================================================================

class HybridPluginRegistry:
    """Registry for hybrid MAKER plugins"""

    def __init__(self):
        self.tactic_generators: Dict[str, TacticGeneratorPlugin] = {}
        self.fitness_functions: Dict[str, FitnessFunctionPlugin] = {}
        self.selection_strategies: Dict[str, SelectionStrategyPlugin] = {}
        self.decompositions: Dict[str, DecompositionPlugin] = {}
        self.crossovers: Dict[str, CrossoverPlugin] = {}
        self.mutations: Dict[str, MutationPlugin] = {}

        # Register built-in plugins
        self._register_builtin_plugins()

    def _register_builtin_plugins(self):
        """Register all built-in plugins"""
        # Tactic generators
        self.register_plugin(AlgebraicTacticGenerator())
        self.register_plugin(InductionTacticGenerator())
        self.register_plugin(LogicTacticGenerator())

        # Fitness functions
        self.register_plugin(StructuralFitnessFunction())
        self.register_plugin(SemanticFitnessFunction())
        self.register_plugin(ProgressFitnessFunction())

        # Selection strategies
        self.register_plugin(TournamentSelection())
        self.register_plugin(RouletteWheelSelection())
        self.register_plugin(RankSelection())

        # Decomposition plugins
        self.register_plugin(QuantifierDecomposition())
        self.register_plugin(ConjunctionDecomposition())
        self.register_plugin(DisjunctionDecomposition())

        # Crossover operators
        self.register_plugin(SinglePointCrossover())
        self.register_plugin(UniformCrossover())

        # Mutation operators
        self.register_plugin(TacticInsertionMutation())
        self.register_plugin(TacticDeletionMutation())
        self.register_plugin(TacticReplacementMutation())

    def register_plugin(self, plugin: Any):
        """Register a plugin"""
        if isinstance(plugin, TacticGeneratorPlugin):
            self.tactic_generators[plugin.plugin_name] = plugin
        elif isinstance(plugin, FitnessFunctionPlugin):
            self.fitness_functions[plugin.plugin_name] = plugin
        elif isinstance(plugin, SelectionStrategyPlugin):
            self.selection_strategies[plugin.plugin_name] = plugin
        elif isinstance(plugin, DecompositionPlugin):
            self.decompositions[plugin.plugin_name] = plugin
        elif isinstance(plugin, CrossoverPlugin):
            self.crossovers[plugin.plugin_name] = plugin
        elif isinstance(plugin, MutationPlugin):
            self.mutations[plugin.plugin_name] = plugin

    def get_plugin(self, plugin_type: str, plugin_name: str) -> Optional[Any]:
        """Get a plugin by type and name"""
        registries = {
            "tactic_generator": self.tactic_generators,
            "fitness_function": self.fitness_functions,
            "selection_strategy": self.selection_strategies,
            "decomposition": self.decompositions,
            "crossover": self.crossovers,
            "mutation": self.mutations,
        }

        registry = registries.get(plugin_type)
        return registry.get(plugin_name) if registry else None

    def list_plugins(self, plugin_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List all registered plugins"""
        if plugin_type:
            registries = {
                "tactic_generator": self.tactic_generators,
                "fitness_function": self.fitness_functions,
                "selection_strategy": self.selection_strategies,
                "decomposition": self.decompositions,
                "crossover": self.crossovers,
                "mutation": self.mutations,
            }
            registry = registries.get(plugin_type)
            return {plugin_type: list(registry.keys())} if registry else {}
        else:
            return {
                "tactic_generator": list(self.tactic_generators.keys()),
                "fitness_function": list(self.fitness_functions.keys()),
                "selection_strategy": list(self.selection_strategies.keys()),
                "decomposition": list(self.decompositions.keys()),
                "crossover": list(self.crossovers.keys()),
                "mutation": list(self.mutations.keys()),
            }


# =============================================================================
# DEMO / MAIN
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("Advanced Plugins for Hybrid MAKER Integration")
    print("=" * 60)

    # Create registry
    registry = HybridPluginRegistry()

    # List plugins
    print("\n1. Available Plugins")
    print("-" * 40)
    plugins = registry.list_plugins()
    for plugin_type, names in plugins.items():
        print(f"{plugin_type}: {', '.join(names)}")

    # Demo tactic generation
    print("\n2. Tactic Generation")
    print("-" * 40)

    theorem = "forall n m : nat, n + m = m + n"

    algebraic_gen = registry.get_plugin("tactic_generator", "algebraic_tactics")
    tactics = asyncio.run(algebraic_gen.generate_tactics(theorem, {}, 5))
    print(f"Algebraic tactics: {tactics}")

    induction_gen = registry.get_plugin("tactic_generator", "induction_tactics")
    tactics = asyncio.run(induction_gen.generate_tactics(theorem, {}, 5))
    print(f"Induction tactics: {tactics}")

    # Demo fitness evaluation
    print("\n3. Fitness Evaluation")
    print("-" * 40)

    proof = "simp\nrw [add_comm]\nrefl"

    structural_fn = registry.get_plugin("fitness_function", "structural_fitness")
    fitness = asyncio.run(structural_fn.evaluate(proof, theorem, {}))
    print(f"Structural fitness: {fitness:.3f}")

    semantic_fn = registry.get_plugin("fitness_function", "semantic_fitness")
    fitness = asyncio.run(semantic_fn.evaluate(proof, theorem, {}))
    print(f"Semantic fitness: {fitness:.3f}")

    # Demo decomposition
    print("\n4. Task Decomposition")
    print("-" * 40)

    quantifier_dec = registry.get_plugin("decomposition", "quantifier_decomposition")
    subtasks = asyncio.run(quantifier_dec.decompose(theorem, {}))
    print(f"Quantifier decomposition: {subtasks}")

    print("\n" + "=" * 60)
    print("Plugin demo complete!")
