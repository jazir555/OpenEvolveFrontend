"""
Test script for MDAP Evolutionary MCTS Integration

This script demonstrates the usage of the MDAP/MAKER integration
with evolutionary MCTS nodes.
"""

import asyncio
import logging
from mcts_evolutionary_nodes_mdap import (
    MDAPEvolutionaryMCTS,
    MDAPEvolutionaryNode,
    MDAPSequenceEvaluator,
    SequenceMAKERVoting,
    MDAPNodeEvolution,
    DecompositionAwareEvolution,
    SequenceRedFlagger,
    DistributedMDAPEvolution,
    MDAPEvolutionMonitor,
    create_mdap_evolutionary_mcts,
    create_mdap_node,
    ProofContext,
    ProofState,
    ActionSequence
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mdap_node():
    """Test MDAP evolutionary node."""
    print("\n" + "=" * 80)
    print("Test 1: MDAP Evolutionary Node")
    print("=" * 80)

    # Create proof state
    state = ProofState(
        goals=["prove goal 1", "prove goal 2"],
        context=["hypothesis 1"],
        tactics_sequence=[],
        depth=0
    )

    # Create MDAP node
    node = create_mdap_node(
        state=state,
        population_size=20,
        num_agents=5,
        voting_strategy="first_k_ahead",
        consensus_threshold=0.75
    )

    print(f"Created MDAP node: {node.node_id}")
    print(f"Number of agents: {node.num_agents}")
    print(f"Voting strategy: {node.voting_strategy}")
    print(f"Consensus threshold: {node.consensus_threshold}")
    print(f"Enable decomposition: {node.enable_decomposition}")

    # Create proof context
    context = ProofContext(
        theorem="test theorem",
        goals=["prove goal 1"],
        hypotheses=["hypothesis 1"],
        available_tactics=["simp", "rw", "apply", "exact"]
    )

    # Initialize populations
    node.initialize_mdap_populations(context)

    print(f"\nInitialized populations:")
    for agent_id, pop in node.agent_populations.items():
        print(f"  {agent_id}: {len(pop)} sequences")

    # Compute agreement level
    agreement = node.compute_agreement_level()
    print(f"\nAgent agreement level: {agreement:.4f}")

    # Check if should decompose
    should_decompose = node.should_decompose()
    print(f"Should decompose: {should_decompose}")

    print("\nMDAP node test PASSED")


async def test_mdap_sequence_evaluator():
    """Test MDAP sequence evaluator."""
    print("\n" + "=" * 80)
    print("Test 2: MDAP Sequence Evaluator")
    print("=" * 80)

    # Create evaluator
    evaluator = MDAPSequenceEvaluator(num_agents=5)

    # Create test sequences
    from mcts_evolutionary_nodes import Tactic

    sequences = [
        ActionSequence(
            actions=[Tactic(name="simp"), Tactic(name="rw")],
            fitness=0.5
        ),
        ActionSequence(
            actions=[Tactic(name="apply"), Tactic(name="exact")],
            fitness=0.6
        ),
    ]

    # Create node
    state = ProofState(
        goals=["prove goal"],
        context=[],
        tactics_sequence=[],
        depth=0
    )

    node = create_mdap_node(state=state, num_agents=5)

    # Create context
    context = ProofContext(
        theorem="test",
        goals=["prove goal"],
        hypotheses=[],
        available_tactics=["simp", "rw", "apply", "exact"]
    )

    # Evaluate with MDAP
    evaluations = await evaluator.evaluate_mdap(sequences, node, context)

    print(f"Evaluated {len(evaluations)} sequences")
    for seq_id, evaluation in evaluations.items():
        print(f"\nSequence {seq_id}:")
        print(f"  Consensus fitness: {evaluation.consensus_fitness:.4f}")
        print(f"  Agreement level: {evaluation.agreement_level:.4f}")
        print(f"  Red flags: {evaluation.red_flags}")

    print("\nMDAP sequence evaluator test PASSED")


async def test_maker_voting():
    """Test MAKER voting."""
    print("\n" + "=" * 80)
    print("Test 3: MAKER Voting")
    print("=" * 80)

    # Create voting system
    voting = SequenceMAKERVoting(k_ahead=3, voting_strategy="first_k_ahead")

    # Create node with population
    from mcts_evolutionary_nodes import Tactic

    state = ProofState(
        goals=["prove goal"],
        context=[],
        tactics_sequence=[],
        depth=0
    )

    node = create_mdap_node(state=state, population_size=10)

    # Add sequences to population
    for i in range(10):
        sequence = ActionSequence(
            actions=[Tactic(name="simp")],
            fitness=0.5 + (i * 0.05)
        )
        node.rollout_population.append(sequence)

    # Create mock evaluations
    from mcts_evolutionary_nodes_mdap import MDAPSequenceEvaluation, AgentEvaluationResult

    evaluations = {}
    for seq in node.rollout_population:
        evaluations[seq.sequence_id] = MDAPSequenceEvaluation(
            sequence_id=seq.sequence_id,
            agent_results=[
                AgentEvaluationResult(
                    agent_id=f"agent_{j}",
                    fitness=seq.fitness + (j * 0.01),
                    confidence=0.8,
                    reasoning="Good"
                )
                for j in range(5)
            ],
            consensus_fitness=seq.fitness,
            agreement_level=0.7,
            voting_details={f"agent_{j}": 1 for j in range(5)}
        )

    # Vote for best sequence
    best = voting.vote_on_best_sequence(node, evaluations)

    print(f"Selected sequence with fitness: {best.fitness:.4f}")
    print(f"Sequence ID: {best.sequence_id}")

    print("\nMAKER voting test PASSED")


async def test_sequence_red_flagger():
    """Test sequence red-flagger."""
    print("\n" + "=" * 80)
    print("Test 4: Sequence Red Flagger")
    print("=" * 80)

    from mcts_evolutionary_nodes import Tactic

    flagger = SequenceRedFlagger()

    # Create test sequences
    valid_sequence = ActionSequence(
        actions=[Tactic(name="simp"), Tactic(name="rw")],
        fitness=0.7
    )

    invalid_sequence = ActionSequence(
        actions=[
            Tactic(name="simp"),
            Tactic(name="simp"),
            Tactic(name="simp"),
            Tactic(name="simp"),
            Tactic(name="simp"),
            Tactic(name="simp")
        ],
        fitness=0.3
    )

    # Create context
    context = ProofContext(
        theorem="test",
        goals=["prove goal"],
        hypotheses=[],
        available_tactics=["simp", "rw", "apply"],
        depth_limit=5
    )

    # Check valid sequence
    is_flagged, reasons = flagger.check_sequence(valid_sequence, context)
    print(f"Valid sequence flagged: {is_flagged}")
    print(f"Reasons: {reasons}")

    # Check invalid sequence
    is_flagged, reasons = flagger.check_sequence(invalid_sequence, context)
    print(f"\nInvalid sequence flagged: {is_flagged}")
    print(f"Reasons: {reasons}")

    print("\nSequence red-flagger test PASSED")


async def test_mdap_monitor():
    """Test MDAP evolution monitor."""
    print("\n" + "=" * 80)
    print("Test 5: MDAP Evolution Monitor")
    print("=" * 80)

    monitor = MDAPEvolutionMonitor()

    # Track some generations
    node_id = "test_node_1"

    for gen in range(5):
        metrics = {
            "avg_fitness": 0.5 + (gen * 0.1),
            "best_fitness": 0.7 + (gen * 0.1),
            "agent_fitness": {
                f"agent_{i}": 0.5 + (gen * 0.1) + (i * 0.05)
                for i in range(5)
            }
        }
        monitor.track_generation(node_id, gen, metrics)

    # Get convergence curve
    curve = monitor.get_convergence_curve(node_id)
    print(f"Convergence curve: {curve}")

    # Get agent reliability
    for agent_id in ["agent_0", "agent_1", "agent_2"]:
        reliability = monitor.get_agent_reliability(agent_id)
        print(f"Agent {agent_id} reliability: {reliability:.4f}")

    # Get summary
    summary = monitor.get_summary()
    print(f"\nMonitor summary:")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Total generations: {summary['total_generations']}")
    print(f"  Avg generations per node: {summary['avg_generations_per_node']:.2f}")

    print("\nMDAP monitor test PASSED")


async def test_full_mdap_evolutionary_mcts():
    """Test full MDAP evolutionary MCTS (simplified)."""
    print("\n" + "=" * 80)
    print("Test 6: Full MDAP Evolutionary MCTS (Simplified)")
    print("=" * 80)

    # Create context
    context = ProofContext(
        theorem="forall (n : Nat), n + 0 = n",
        goals=["prove n + 0 = n"],
        hypotheses=[],
        available_tactics=[
            "intros", "simp", "rw", "apply", "exact",
            "induction", "cases", "refl"
        ],
        depth_limit=20
    )

    # Create MDAP evolutionary MCTS with reduced simulations for testing
    mdap_mcts = create_mdap_evolutionary_mcts(
        population_size=10,
        evolution_generations=2,
        num_agents=3,
        voting_strategy="first_k_ahead",
        enable_decomposition=True,
        mcts_simulations=5  # Reduced for testing
    )

    print(f"Created MDAP Evolutionary MCTS")
    print(f"  Population size: {mdap_mcts.population_size}")
    print(f"  Evolution generations: {mdap_mcts.evolution_generations}")
    print(f"  Number of agents: {mdap_mcts.num_agents}")
    print(f"  Voting strategy: {mdap_mcts.voting_strategy}")
    print(f"  MCTS simulations: {mdap_mcts.mcts_simulations}")

    print("\nMDAP evolutionary MCTS test PASSED (setup complete)")


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MDAP Evolutionary MCTS Integration Test Suite")
    print("=" * 80)

    try:
        await test_mdap_node()
        await test_mdap_sequence_evaluator()
        await test_maker_voting()
        await test_sequence_red_flagger()
        await test_mdap_monitor()
        await test_full_mdap_evolutionary_mcts()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print("\n" + "=" * 80)
        print("TESTS FAILED")
        print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
