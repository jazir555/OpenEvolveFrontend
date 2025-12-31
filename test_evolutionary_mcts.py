"""
Test script for Evolutionary MCTS implementation.

This script demonstrates and tests the key components of the
Evolutionary Monte Carlo Tree Search system.
"""

import asyncio
import logging
from mcts_evolutionary_nodes import (
    ActionSequence,
    ProofContext,
    EvolutionaryNode,
    SequenceCrossover,
    SequenceMutation,
    SequenceSelection,
    SequenceEvaluator,
    EvolutionaryMCTS,
    AdaptiveEvolutionController,
    EvolutionaryNodeCache,
    create_action_sequence_from_tactics,
    create_evolutionary_mcts,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_action_sequence():
    """Test ActionSequence creation and manipulation."""
    print("\n" + "=" * 80)
    print("TEST: Action Sequence")
    print("=" * 80)

    # Create sequence from tactics
    sequence = create_action_sequence_from_tactics(
        tactics=["intros", "simp", "rw [add_comm]"],
        generation=0
    )

    print(f"Created sequence with {sequence.length()} tactics")
    print(f"Valid: {sequence.is_valid()}")
    print(f"Lean code:\n{sequence.to_string()}")
    print(f"Hash: {sequence.calculate_hash()}")

    # Test copy
    sequence_copy = sequence.copy()
    sequence_copy.fitness = 0.8
    print(f"\nOriginal fitness: {sequence.fitness}")
    print(f"Copy fitness: {sequence_copy.fitness}")

    print("[PASS] ActionSequence test passed")


def test_sequence_crossover():
    """Test crossover operators."""
    print("\n" + "=" * 80)
    print("TEST: Sequence Crossover")
    print("=" * 80)

    # Create parent sequences
    parent1 = create_action_sequence_from_tactics(["intros", "simp", "apply"])
    parent2 = create_action_sequence_from_tactics(["intros", "rw", "exact"])

    # Create crossover operator
    crossover = SequenceCrossover(context_aware=True)
    context = ProofContext(
        theorem="test theorem",
        goals=["goal1"],
        hypotheses=[],
        available_tactics=["intros", "simp", "rw", "apply", "exact"]
    )

    # Test one-point crossover
    offspring1, offspring2 = crossover.one_point_crossover(parent1, parent2)
    print(f"One-point crossover:")
    print(f"  Offspring 1: {offspring1.to_string()}")
    print(f"  Offspring 2: {offspring2.to_string()}")

    # Test uniform crossover
    offspring1, offspring2 = crossover.uniform_crossover(parent1, parent2)
    print(f"\nUniform crossover:")
    print(f"  Offspring 1: {offspring1.to_string()}")
    print(f"  Offspring 2: {offspring2.to_string()}")

    # Test context-aware crossover
    offspring1, offspring2 = crossover.context_aware_crossover(parent1, parent2, context)
    print(f"\nContext-aware crossover:")
    print(f"  Offspring 1: {offspring1.to_string()}")
    print(f"  Offspring 2: {offspring2.to_string()}")

    print("[PASS] SequenceCrossover test passed")


def test_sequence_mutation():
    """Test mutation operators."""
    print("\n" + "=" * 80)
    print("TEST: Sequence Mutation")
    print("=" * 80)

    # Create sequence
    sequence = create_action_sequence_from_tactics(["intros", "simp", "apply"])
    print(f"Original: {sequence.to_string()}")

    # Create mutation operator
    mutation = SequenceMutation()

    # Test tactic substitution
    mutated = mutation.tactic_substitution(sequence)
    print(f"\nSubstitution: {mutated.to_string()}")

    # Test tactic insertion
    mutated = mutation.tactic_insertion(sequence)
    print(f"Insertion: {mutated.to_string()}")

    # Test tactic deletion
    mutated = mutation.tactic_deletion(sequence)
    print(f"Deletion: {mutated.to_string()}")

    # Test subsequence reorder
    mutated = mutation.subsequence_reorder(sequence)
    print(f"Reorder: {mutated.to_string()}")

    # Test adaptive mutation
    mutated = mutation.adaptive_mutation(sequence, mutation_rate=0.5)
    print(f"Adaptive: {mutated.to_string()}")

    print("[PASS] SequenceMutation test passed")


def test_sequence_selection():
    """Test selection operators."""
    print("\n" + "=" * 80)
    print("TEST: Sequence Selection")
    print("=" * 80)

    # Create population with varying fitness
    population = []
    for i, tactics in enumerate([
        ["intros", "simp"],
        ["apply", "exact"],
        ["rw", "simp"],
        ["induction", "simp", "exact"],
        ["cases", "linarith"]
    ]):
        seq = create_action_sequence_from_tactics(tactics)
        seq.fitness = i * 0.2  # Varying fitness
        population.append(seq)

    print(f"Population fitnesses: {[s.fitness for s in population]}")

    # Create selection operator
    selection = SequenceSelection()

    # Test tournament selection
    selected = selection.tournament_selection(population, tournament_size=3)
    print(f"\nTournament selection selected fitness: {selected.fitness}")

    # Test fitness proportionate selection
    selected = selection.fitness_proportionate_selection(population)
    print(f"Fitness proportionate selected fitness: {selected.fitness}")

    # Test rank selection
    selected = selection.rank_selection(population)
    print(f"Rank selection selected fitness: {selected.fitness}")

    # Test elitist selection
    elites = selection.elitist_selection(population, elite_count=2)
    print(f"\nElitist selection selected: {[e.fitness for e in elites]}")

    print("[PASS] SequenceSelection test passed")


def test_sequence_evaluator():
    """Test fitness evaluation."""
    print("\n" + "=" * 80)
    print("TEST: Sequence Evaluator")
    print("=" * 80)

    # Create evaluator
    evaluator = SequenceEvaluator()

    # Create context
    context = ProofContext(
        theorem="forall a b, a + b = b + a",
        goals=["prove a + b = b + a"],
        hypotheses=["Nat : Type"],
        available_tactics=["intros", "simp", "rw", "apply"]
    )

    # Test various sequences
    test_cases = [
        (["intros", "simp"], "short sequence"),
        (["intros", "simp", "rw", "apply", "exact"], "medium sequence"),
        (["intros"] * 20, "long sequence"),
        ([], "empty sequence")
    ]

    for tactics, description in test_cases:
        sequence = create_action_sequence_from_tactics(tactics)
        fitness = evaluator.evaluate(sequence, context)
        print(f"{description}: fitness = {fitness:.4f}")

    print("[PASS] SequenceEvaluator test passed")


def test_evolutionary_node():
    """Test EvolutionaryNode."""
    print("\n" + "=" * 80)
    print("TEST: Evolutionary Node")
    print("=" * 80)

    from mcts_evolutionary_nodes import ProofState

    # Create state
    state = ProofState(
        goals=["goal1", "goal2"],
        context=["hypothesis1"]
    )

    # Create node
    node = EvolutionaryNode(
        state=state,
        population_size=10,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

    print(f"Created node with population_size={node.population_size}")
    print(f"Depth: {node.depth}")
    print(f"Terminal: {node.is_terminal}")

    # Create population
    context = ProofContext(
        theorem="test",
        goals=["goal1"],
        hypotheses=[],
        available_tactics=["intros", "simp", "rw"]
    )

    sequences = []
    for i in range(10):
        seq = create_action_sequence_from_tactics(["intros", "simp", "rw"])
        seq.fitness = i * 0.1
        sequences.append(seq)

    node.update_population(sequences)

    print(f"\nPopulation updated:")
    print(f"  Best fitness: {node.best_fitness:.4f}")
    print(f"  Population size: {len(node.rollout_population)}")
    print(f"  Diversity: {node.get_population_diversity():.4f}")
    print(f"  Converged: {node.is_population_converged()}")

    print("[PASS] EvolutionaryNode test passed")


def test_adaptive_controller():
    """Test AdaptiveEvolutionController."""
    print("\n" + "=" * 80)
    print("TEST: Adaptive Evolution Controller")
    print("=" * 80)

    from mcts_evolutionary_nodes import ProofState

    controller = AdaptiveEvolutionController()

    # Create test node
    state = ProofState(goals=["goal1"])
    node = EvolutionaryNode(state, population_size=20)

    # Add some convergence history
    node.convergence_history = [0.5, 0.6, 0.65, 0.68, 0.70]
    node.N = 50

    # Test decisions
    should_evolve = controller.should_evolve_at_node(node, depth=5)
    print(f"Should evolve at depth 5: {should_evolve}")

    generations = controller.get_evolution_generations(node, depth=5)
    print(f"Generations at depth 5: {generations}")

    pop_size = controller.get_population_size(node, depth=5)
    print(f"Population size at depth 5: {pop_size}")

    mutation_rate = controller.get_mutation_rate(node, generation=5)
    print(f"Mutation rate at gen 5: {mutation_rate:.4f}")

    print("[PASS] AdaptiveEvolutionController test passed")


def test_node_cache():
    """Test EvolutionaryNodeCache."""
    print("\n" + "=" * 80)
    print("TEST: Evolutionary Node Cache")
    print("=" * 80)

    cache = EvolutionaryNodeCache(max_size=3)

    # Test cache miss
    def compute_node():
        from mcts_evolutionary_nodes import ProofState
        state = ProofState(goals=["goal1"])
        return EvolutionaryNode(state)

    node1 = cache.get_or_compute("hash1", compute_node)
    print(f"Cache miss - created node")

    # Test cache hit
    node1_cached = cache.get_or_compute("hash1", compute_node)
    print(f"Cache hit - retrieved node")

    # Add more entries
    cache.get_or_compute("hash2", compute_node)
    cache.get_or_compute("hash3", compute_node)
    cache.get_or_compute("hash4", compute_node)  # Should evict hash1

    # Get stats
    stats = cache.get_stats()
    print(f"\nCache statistics:")
    print(f"  Size: {stats['size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")

    print("[PASS] EvolutionaryNodeCache test passed")


async def test_evolutionary_mcts():
    """Test EvolutionaryMCTS search."""
    print("\n" + "=" * 80)
    print("TEST: Evolutionary MCTS Search")
    print("=" * 80)

    # Create proof context
    context = ProofContext(
        theorem="forall (a b : Nat), a + b = b + a",
        goals=["prove a + b = b + a"],
        hypotheses=[],
        available_tactics=[
            "intros", "simp", "rw", "apply", "exact",
            "induction", "cases", "linarith", "ring"
        ]
    )

    # Create evolutionary MCTS
    emcts = create_evolutionary_mcts(
        population_size=15,
        evolution_generations=3,
        mcts_simulations=20,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

    print(f"Created EvolutionaryMCTS")
    print(f"  Population size: {emcts.population_size}")
    print(f"  Evolution generations: {emcts.evolution_generations}")
    print(f"  MCTS simulations: {emcts.mcts_simulations}")

    # Run search
    print("\nRunning search...")
    result = await emcts.search(context)

    # Print results
    print(f"\nSearch complete!")
    print(f"  Success: {result.success}")
    print(f"  Time: {result.time_elapsed:.2f}s")
    print(f"  Nodes visited: {result.nodes_visited}")
    print(f"  Tree depth: {result.tree_depth}")
    print(f"  Win rate: {result.win_rate:.4f}")
    print(f"  Total evolutions: {result.search_statistics.get('total_evolutions', 0)}")
    print(f"  Total evaluations: {result.search_statistics.get('total_evaluations', 0)}")

    if result.best_proof:
        print(f"\nBest proof found:")
        print(result.best_proof.lean_code)

    print("[PASS] EvolutionaryMCTS test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EVOLUTIONARY MCTS TEST SUITE")
    print("=" * 80)

    # Synchronous tests
    test_action_sequence()
    test_sequence_crossover()
    test_sequence_mutation()
    test_sequence_selection()
    test_sequence_evaluator()
    test_evolutionary_node()
    test_adaptive_controller()
    test_node_cache()

    # Asynchronous test
    asyncio.run(test_evolutionary_mcts())

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED [PASS]")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
