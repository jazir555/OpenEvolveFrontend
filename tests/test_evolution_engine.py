"""
Comprehensive Unit Tests for Evolution Engine

Tests the core evolution engine including:
- Population management
- Selection mechanisms
- Crossover operations
- Mutation operations
- Fitness evaluation
- Evolution strategies

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPopulationManager:
    """Test population management"""

    @pytest.fixture
    def population_manager(self):
        """Create population manager for testing"""
        from evolution import PopulationManager
        return PopulationManager(
            population_size=50,
            max_population_size=100
        )

    def test_population_manager_creation(self, population_manager):
        """Test PopulationManager initialization"""
        from evolution import PopulationManager
        
        manager = PopulationManager(
            population_size=100,
            max_population_size=200
        )
        assert manager.population_size == 100
        assert manager.max_population_size == 200

    def test_create_individual(self, population_manager):
        """Test individual creation"""
        individual = population_manager.create_individual(
            genome={"weight": 0.5, "bias": 0.1}
        )
        
        assert individual is not None
        assert individual.genome is not None
        assert individual.fitness is None

    def test_add_individual(self, population_manager):
        """Test adding individual to population"""
        individual = population_manager.create_individual(
            genome={"x": 1.0}
        )
        population_manager.add_individual(individual)
        
        assert len(population_manager.population) == 1

    def test_get_fittest(self, population_manager):
        """Test getting fittest individual"""
        ind1 = population_manager.create_individual(genome={"val": 1.0})
        ind1.fitness = 0.5
        
        ind2 = population_manager.create_individual(genome={"val": 2.0})
        ind2.fitness = 0.8
        
        population_manager.add_individual(ind1)
        population_manager.add_individual(ind2)
        
        fittest = population_manager.get_fittest()
        assert fittest.fitness == 0.8


class TestSelectionMechanisms:
    """Test selection mechanisms"""

    def test_tournament_selection(self):
        """Test tournament selection"""
        from evolution import tournament_select
        
        individuals = [
            Mock(fitness=0.3),
            Mock(fitness=0.7),
            Mock(fitness=0.5),
            Mock(fitness=0.9)
        ]
        
        selected = tournament_select(individuals, tournament_size=2)
        
        assert selected in individuals
        assert selected.fitness >= 0.5  # Should favor higher fitness

    def test_roulette_wheel_selection(self):
        """Test roulette wheel selection"""
        from evolution import roulette_wheel_select
        
        individuals = [
            Mock(fitness=0.1),
            Mock(fitness=0.2),
            Mock(fitness=0.3),
            Mock(fitness=0.4)
        ]
        
        selected = roulette_wheel_select(individuals)
        
        assert selected in individuals

    def test_rank_selection(self):
        """Test rank-based selection"""
        from evolution import rank_select
        
        individuals = [
            Mock(fitness=0.1),
            Mock(fitness=0.2),
            Mock(fitness=0.3)
        ]
        
        selected = rank_select(individuals)
        
        assert selected in individuals


class TestCrossoverOperations:
    """Test crossover operations"""

    def test_single_point_crossover(self):
        """Test single-point crossover"""
        from evolution import single_point_crossover
        
        parent1 = {"genes": [1, 2, 3, 4, 5]}
        parent2 = {"genes": [5, 4, 3, 2, 1]}
        
        child1, child2 = single_point_crossover(parent1, parent2)
        
        assert len(child1["genes"]) == len(parent1["genes"])
        assert len(child2["genes"]) == len(parent2["genes"])

    def test_uniform_crossover(self):
        """Test uniform crossover"""
        from evolution import uniform_crossover
        
        parent1 = {"genes": [1, 1, 1, 1]}
        parent2 = {"genes": [0, 0, 0, 0]}
        
        child1, child2 = uniform_crossover(parent1, parent2, probability=0.5)
        
        assert len(child1["genes"]) == 4
        assert len(child2["genes"]) == 4

    def test_blend_crossover(self):
        """Test blend crossover for real-valued genes"""
        from evolution import blend_crossover
        
        parent1 = {"value": 2.0}
        parent2 = {"value": 4.0}
        
        child1, child2 = blend_crossover(parent1, parent2, alpha=0.5)
        
        assert child1["value"] != child2["value"]
        # Both children should be between parents
        assert min(parent1["value"], parent2["value"]) <= child1["value"] <= max(parent1["value"], parent2["value"])


class TestMutationOperations:
    """Test mutation operations"""

    def test_gaussian_mutation(self):
        """Test Gaussian mutation"""
        from evolution import gaussian_mutate
        
        genome = {"value": 1.0}
        mutated = gaussian_mutate(genome, mutation_rate=0.5, std=0.1)
        
        assert "value" in mutated
        # Should be different due to mutation

    def test_bit_flip_mutation(self):
        """Test bit-flip mutation"""
        from evolution import bit_flip_mutate
        
        genome = {"bits": [0, 1, 0, 1, 0]}
        mutated = bit_flip_mutate(genome, mutation_rate=0.5)
        
        assert len(mutated["bits"]) == 5
        assert isinstance(mutated["bits"][0], int)

    def test_swap_mutation(self):
        """Test swap mutation for permutations"""
        from evolution import swap_mutate
        
        genome = {"order": [1, 2, 3, 4, 5]}
        mutated = swap_mutate(genome, mutation_rate=0.5)
        
        assert len(mutated["order"]) == 5
        # May or may not be different

    def test_adaptive_mutation(self):
        """Test adaptive mutation rates"""
        from evolution import AdaptiveMutation
        
        mutator = AdaptiveMutation(
            initial_rate=0.5,
            min_rate=0.01,
            max_rate=0.5
        )
        
        # Should adapt based on fitness
        new_rate = mutator.adapt(current_fitness=0.9, best_fitness=1.0)
        assert new_rate <= mutator.initial_rate


class TestFitnessEvaluation:
    """Test fitness evaluation"""

    def test_fitness_calculator(self):
        """Test fitness calculation"""
        from evolution import FitnessCalculator
        
        calculator = FitnessCalculator()
        
        genome = {"x": 5.0}
        fitness = calculator.calculate(genome)
        
        assert isinstance(fitness, float)

    def test_multi_objective_fitness(self):
        """Test multi-objective fitness evaluation"""
        from evolution import MultiObjectiveFitness
        
        fitness = MultiObjectiveFitness(
            objectives={"accuracy": 0.9, "speed": 0.8, "memory": 0.7}
        )
        
        assert fitness.accuracy == 0.9
        assert fitness.speed == 0.8

    def test_pareto_dominance(self):
        """Test Pareto dominance comparison"""
        from evolution import MultiObjectiveFitness
        
        fitness1 = MultiObjectiveFitness(objectives={"accuracy": 0.9, "speed": 0.5})
        fitness2 = MultiObjectiveFitness(objectives={"accuracy": 0.8, "speed": 0.6})
        
        # fitness1 dominates if it's better in at least one objective
        # and not worse in any
        result = fitness1.dominates(fitness2)
        assert isinstance(result, bool)


class TestEvolutionStrategies:
    """Test evolution strategies"""

    def test_evolution_config(self):
        """Test evolution configuration"""
        from evolution import EvolutionConfig
        
        config = EvolutionConfig(
            population_size=100,
            max_generations=1000,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=5
        )
        
        assert config.population_size == 100
        assert config.max_generations == 1000

    def test_evolution_state(self):
        """Test evolution state tracking"""
        from evolution import EvolutionState
        
        state = EvolutionState(
            generation=0,
            best_fitness=0.0,
            average_fitness=0.0
        )
        
        assert state.generation == 0
        assert state.best_fitness == 0.0

    def test_evolution_run(self):
        """Test evolution run"""
        from evolution import EvolutionEngine
        
        engine = EvolutionEngine(
            population_size=20,
            max_generations=10
        )
        
        result = engine.run(
            fitness_function=lambda g: sum(g.values()) / len(g) if g else 0
        )
        
        assert result is not None
        assert hasattr(result, 'best_individual')
        assert hasattr(result, 'generation_count')


class TestEvolutionHistory:
    """Test evolution history tracking"""

    def test_history_recording(self):
        """Test recording evolution history"""
        from evolution import EvolutionHistory
        
        history = EvolutionHistory()
        
        history.record(
            generation=0,
            best_fitness=0.5,
            average_fitness=0.4,
            diversity=0.3
        )
        
        assert len(history.generations) == 1
        assert history.generations[0].best_fitness == 0.5

    def test_history_summary(self):
        """Test history summary generation"""
        from evolution import EvolutionHistory
        
        history = EvolutionHistory()
        history.record(generation=0, best_fitness=0.3, average_fitness=0.2, diversity=0.1)
        history.record(generation=1, best_fitness=0.5, average_fitness=0.4, diversity=0.2)
        
        summary = history.get_summary()
        
        assert summary["total_generations"] == 2
        assert summary["improvement"] > 0


class TestConvergenceDetection:
    """Test convergence detection"""

    def test_convergence_check(self):
        """Test convergence detection"""
        from evolution import ConvergenceDetector
        
        detector = ConvergenceDetector(
            window_size=10,
            threshold=0.01
        )
        
        # Should not be converged initially
        assert detector.is_converged() == False
        
        # Add similar fitness values
        for _ in range(10):
            detector.add_fitness(0.5)

    def test_diversity_check(self):
        """Test diversity tracking"""
        from evolution import DiversityTracker
        
        tracker = DiversityTracker()
        
        # Add individuals
        tracker.add_individual({"genome": [1, 2, 3]})
        tracker.add_individual({"genome": [4, 5, 6]})
        
        diversity = tracker.calculate_diversity()
        assert diversity >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
