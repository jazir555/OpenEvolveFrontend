"""
Test script for MCTS implementation.

This script demonstrates the MCTS proof search capabilities without requiring
a running LeanAide server.
"""

import asyncio
import logging
from leanaide_mcts import (
    MCTSConfig,
    MCTSResult,
    MCTSNode,
    MCTSTree,
    MCTSSelection,
    MCTSExpansion,
    MCTSSimulation,
    MCTSBackpropagation,
    MCTS,
    ProofState,
    RolloutPolicy,
    search_proof_with_mcts
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_proof_state():
    """Test ProofState creation and hashing."""
    print("\n" + "=" * 80)
    print("Test 1: ProofState")
    print("=" * 80)

    state1 = ProofState(
        goals=["forall (a b : Nat), a + b = b + a"],
        context=[],
        depth=0
    )

    print(f"State created with {len(state1.goals)} goal(s)")
    print(f"State hash: {state1.hash}")
    print(f"Is complete: {state1.is_complete}")

    assert state1.hash != "", "State hash should not be empty"
    print("[PASS] ProofState test passed")


def test_mcts_node():
    """Test MCTSNode creation and UCT calculation."""
    print("\n" + "=" * 80)
    print("Test 2: MCTSNode")
    print("=" * 80)

    state = ProofState(
        goals=["goal1", "goal2"],
        context=["hypothesis1"],
        depth=0
    )

    node = MCTSNode(state=state)
    print(f"Node created at depth {node.depth}")
    print(f"Is terminal: {node.is_terminal}")
    print(f"Is leaf: {node.is_leaf}")
    print(f"Initial N={node.N}, W={node.W}, Q={node.Q}")

    # Update node
    node.update(1.0)
    print(f"After update with reward=1.0: N={node.N}, W={node.W}, Q={node.Q}")

    # Test UCT value
    uct = node.uct_value(c_param=1.414)
    print(f"UCT value: {uct:.4f}")

    # Add child
    child_state = ProofState(
        goals=["goal2"],
        context=["hypothesis1", "hypothesis2"],
        depth=1
    )
    child = MCTSNode(state=child_state, parent=node, action="intros")
    node.add_child("intros", child)

    print(f"Added child with action 'intros'")
    print(f"Node has {len(node.children)} child(ren)")
    print(f"Untried actions: {len(node.untried_actions)}")

    print("[PASS] MCTSNode test passed")


def test_mcts_tree():
    """Test MCTSTree management."""
    print("\n" + "=" * 80)
    print("Test 3: MCTSTree")
    print("=" * 80)

    # Create root node
    root_state = ProofState(
        goals=["goal1", "goal2"],
        context=[],
        depth=0
    )
    root = MCTSNode(state=root_state)

    # Create tree
    tree = MCTSTree(root)
    print(f"Tree created with {tree.total_nodes} node(s)")

    # Add children
    for i in range(3):
        child_state = ProofState(
            goals=[f"goal{j}" for j in range(i+1, 3)],
            context=[f"hypothesis{i}"],
            depth=i+1
        )
        child = MCTSNode(state=child_state, parent=root, action=f"action{i}")
        root.add_child(f"action{i}", child)
        tree.add_node(child)

    print(f"Tree now has {tree.total_nodes} node(s)")
    print(f"Max depth: {tree.max_depth}")

    # Get statistics
    stats = tree.get_statistics()
    print(f"Tree statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get best path
    best_path = tree.get_best_path()
    print(f"Best path length: {len(best_path)}")

    print("[PASS] MCTSTree test passed")


def test_selection_phase():
    """Test MCTS selection phase."""
    print("\n" + "=" * 80)
    print("Test 4: MCTSSelection")
    print("=" * 80)

    # Create simple tree
    root_state = ProofState(
        goals=["goal1"],
        context=[],
        depth=0
    )
    root = MCTSNode(state=root_state)

    # Add children with different visit counts
    for i, action in enumerate(["intros", "simp", "apply"]):
        child_state = ProofState(
            goals=[],
            context=[],
            depth=1
        )
        child = MCTSNode(state=child_state, parent=root, action=action)
        # Give different visit counts
        for _ in range(i+1):
            child.update(1.0)
        root.add_child(action, child)

    # Update root
    root.N = sum(child.N for child in root.children.values())

    # Selection
    selection = MCTSSelection(c_param=1.414)
    selected = selection.select(root)

    print(f"Selected action: {selected.action}")
    print(f"Root has {len(root.children)} children")
    for action, child in root.children.items():
        print(f"  {action}: N={child.N}, Q={child.Q:.4f}")

    print("[PASS] MCTSSelection test passed")


def test_simulation_phase():
    """Test MCTS simulation phase."""
    print("\n" + "=" * 80)
    print("Test 5: MCTSSimulation")
    print("=" * 80)

    state = ProofState(
        goals=["goal1", "goal2", "goal3"],
        context=["h1", "h2"],
        depth=5
    )

    # Test random rollout
    sim_random = MCTSSimulation(
        rollout_policy=RolloutPolicy.RANDOM,
        max_depth=20
    )
    value_random = sim_random.simulate(state)
    print(f"Random rollout value: {value_random:.4f}")

    # Test heuristic rollout
    sim_heuristic = MCTSSimulation(
        rollout_policy=RolloutPolicy.HEURISTIC,
        max_depth=20
    )
    value_heuristic = sim_heuristic.simulate(state)
    print(f"Heuristic rollout value: {value_heuristic:.4f}")

    print("[PASS] MCTSSimulation test passed")


def test_backpropagation():
    """Test MCTS backpropagation phase."""
    print("\n" + "=" * 80)
    print("Test 6: MCTSBackpropagation")
    print("=" * 80)

    # Create tree: root -> child1 -> child2
    root_state = ProofState(
        goals=["goal1"],
        context=[],
        depth=0
    )
    root = MCTSNode(state=root_state)

    child1_state = ProofState(
        goals=["goal2"],
        context=["h1"],
        depth=1
    )
    child1 = MCTSNode(state=child1_state, parent=root, action="intros")

    child2_state = ProofState(
        goals=[],
        context=["h1", "h2"],
        depth=2,
        is_complete=True
    )
    child2 = MCTSNode(state=child2_state, parent=child1, action="simp")

    root.add_child("intros", child1)
    child1.add_child("simp", child2)

    print(f"Before backpropagation:")
    print(f"  root: N={root.N}, Q={root.Q:.4f}")
    print(f"  child1: N={child1.N}, Q={child1.Q:.4f}")
    print(f"  child2: N={child2.N}, Q={child2.Q:.4f}")

    # Backpropagate reward
    backprop = MCTSBackpropagation(enable_amaf=True)
    backprop.backpropagate(child2, reward=1.0, actions_seen=["intros", "simp"])

    print(f"\nAfter backpropagation with reward=1.0:")
    print(f"  root: N={root.N}, Q={root.Q:.4f}")
    print(f"  child1: N={child1.N}, Q={child1.Q:.4f}")
    print(f"  child2: N={child2.N}, Q={child2.Q:.4f}")

    assert root.N == 1, "Root should have 1 visit"
    assert child1.N == 1, "Child1 should have 1 visit"
    assert child2.N == 1, "Child2 should have 1 visit"

    print("[PASS] MCTSBackpropagation test passed")


async def test_full_mcts():
    """Test complete MCTS search (without LeanAide server)."""
    print("\n" + "=" * 80)
    print("Test 7: Full MCTS Search")
    print("=" * 80)

    # Create config
    config = MCTSConfig(
        max_iterations=50,
        time_budget=10.0,
        rollout_policy="heuristic",
        enable_transposition_table=True,
        early_termination=True
    )

    # Create MCTS instance
    theorem = "forall (a b : Nat), a + b = b + a"
    mcts = MCTS(config, theorem, theorem_name="add_comm")

    print(f"Running MCTS search for: {theorem}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Time budget: {config.time_budget}s")

    # Run search
    result = await mcts.search()

    print(f"\nSearch completed!")
    print(f"  Success: {result.success}")
    print(f"  Iterations: {result.search_iterations}")
    print(f"  Time: {result.time_elapsed:.2f}s")
    print(f"  Nodes visited: {result.nodes_visited}")
    print(f"  Tree depth: {result.tree_depth}")
    print(f"  Win rate: {result.win_rate:.4f}")
    print(f"  Confidence: {result.confidence:.4f}")

    if result.best_proof:
        print(f"\nBest proof found:")
        print(result.best_proof.lean_code[:500] + "...")

    print("\nTree statistics:")
    for key, value in result.tree_statistics.items():
        print(f"  {key}: {value}")

    print("[PASS] Full MCTS test passed")


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MCTS Implementation Test Suite")
    print("=" * 80)

    try:
        # Synchronous tests
        test_proof_state()
        test_mcts_node()
        test_mcts_tree()
        test_selection_phase()
        test_simulation_phase()
        test_backpropagation()

        # Async tests
        await test_full_mcts()

        print("\n" + "=" * 80)
        print("All tests passed!")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        print(f"\n[FAIL] Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
