"""
Test script for LeanAide MCTS Strategy Library

Demonstrates the key features without Unicode issues.
"""

import sys
import logging

# Import the MCTS strategies library
from leanaide_mcts_strategies import (
    # Enums
    RolloutPolicyType,
    SelectionStrategyType,
    ExpansionStrategyType,
    BackpropagationStrategyType,
    DomainType,

    # Classes
    MCTSNode,
    MCTSSearchResult,
    StrategyPerformance,

    # Rollout Policies
    RandomRolloutPolicy,
    HeuristicRolloutPolicy,
    LearnedRolloutPolicy,

    # Selection Strategies
    UCTSelection,
    AdaptiveUCTSelection,
    ThompsonSamplingSelection,

    # Expansion Strategies
    StandardExpansion,
    ProgressiveWidening,
    TreePolicyExpansion,

    # Backpropagation Strategies
    StandardBackpropagation,
    AMAFBackpropagation,

    # Domain Strategies
    InductionMCTS,
    AlgebraicMCTS,
    LogicalMCTS,

    # Factory
    MCTSStrategyFactory,
    MCTSPerformanceTracker,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_rollout_policies():
    """Test rollout policies"""
    print("=" * 60)
    print("Testing Rollout Policies")
    print("=" * 60)

    # Create test state
    test_state = {
        "goal": "forall n : Nat, n + 0 = n",
        "domain": "induction",
        "available_tactics": ["simp", "intro", "induction", "cases", "apply"],
        "context": [],
        "is_solved": False,
        "is_terminal": False,
    }

    # Test random rollout
    print("\n1. Random Rollout Policy:")
    random_policy = RandomRolloutPolicy()
    selected = random_policy.select_tactic(test_state["available_tactics"], test_state)
    print(f"   Selected tactic: {selected}")

    value = random_policy.rollout(test_state, max_depth=10)
    print(f"   Rollout value: {value:.3f}")

    # Test heuristic rollout
    print("\n2. Heuristic Rollout Policy:")
    heuristic_policy = HeuristicRolloutPolicy()
    selected = heuristic_policy.select_tactic(test_state["available_tactics"], test_state)
    print(f"   Selected tactic: {selected}")

    # Score tactics
    for tactic in test_state["available_tactics"]:
        score = heuristic_policy.score_tactic(tactic, test_state)
        print(f"   Score for '{tactic}': {score:.3f}")

    value = heuristic_policy.rollout(test_state, max_depth=10)
    print(f"   Rollout value: {value:.3f}")


def test_selection_strategies():
    """Test selection strategies"""
    print("\n" + "=" * 60)
    print("Testing Selection Strategies")
    print("=" * 60)

    # Create test nodes
    root = MCTSNode(state={"test": "root"}, visits=100)

    print("\nCreating child nodes:")
    for i in range(5):
        child = MCTSNode(
            state={"test": f"child_{i}"},
            parent=root,
            visits=random.randint(10, 50),
            value=random.uniform(5, 25)
        )
        child.mean_value = child.value / child.visits if child.visits > 0 else 0
        root.children.append(child)
        print(f"  Child {i}: visits={child.visits}, mean_value={child.mean_value:.3f}")

    # Test UCT selection
    print("\n1. UCT Selection:")
    uct = UCTSelection(c_param=1.414)
    selected = uct.select_child(root.children)
    print(f"   Selected: visits={selected.visits}, mean_value={selected.mean_value:.3f}")

    # Test Adaptive UCT
    print("\n2. Adaptive UCT Selection:")
    adaptive = AdaptiveUCTSelection(base_c=1.414)
    selected = adaptive.select_child(root.children, depth=5)
    print(f"   Selected: visits={selected.visits}, mean_value={selected.mean_value:.3f}")

    # Test Thompson Sampling
    print("\n3. Thompson Sampling Selection:")
    thompson = ThompsonSamplingSelection()
    selected = thompson.select_child(root.children)
    print(f"   Selected: visits={selected.visits}, mean_value={selected.mean_value:.3f}")


def test_expansion_strategies():
    """Test expansion strategies"""
    print("\n" + "=" * 60)
    print("Testing Expansion Strategies")
    print("=" * 60)

    # Create test node
    test_state = {
        "goal": "prove theorem",
        "available_tactics": ["simp", "intro", "apply", "cases", "induction"],
    }

    root = MCTSNode(
        state=test_state,
        visits=10,
        untried_actions=test_state["available_tactics"][:]
    )

    print(f"\nRoot node: {len(root.untried_actions)} untried actions")

    # Test standard expansion
    print("\n1. Standard Expansion:")
    standard = StandardExpansion()
    child = standard.expand(root)
    if child:
        print(f"   Expanded with action: {child.action}")
        print(f"   Remaining untried: {len(root.untried_actions)}")
    else:
        print("   No expansion performed")

    # Test progressive widening
    print("\n2. Progressive Widening:")
    root2 = MCTSNode(
        state=test_state,
        visits=5,
        untried_actions=test_state["available_tactics"][:]
    )
    widening = ProgressiveWidening(widening_param=3.0, widening_exponent=0.5)

    for i in range(3):
        should_expand = widening.should_expand_child(root2, len(root2.children))
        print(f"   Visit {i+1}: should_expand={should_expand}")

        if should_expand:
            child = widening.expand(root2)
            if child:
                print(f"     Expanded with: {child.action}")

    # Test tree policy expansion
    print("\n3. Tree Policy Expansion:")
    root3 = MCTSNode(
        state=test_state,
        visits=10,
        untried_actions=test_state["available_tactics"][:]
    )
    tree_policy = TreePolicyExpansion()
    child = tree_policy.expand(root3)
    if child:
        print(f"   Expanded with action: {child.action}")


def test_backpropagation_strategies():
    """Test backpropagation strategies"""
    print("\n" + "=" * 60)
    print("Testing Backpropagation Strategies")
    print("=" * 60)

    # Create a small tree
    test_state = {"goal": "test"}

    root = MCTSNode(state=test_state, visits=0)
    child1 = MCTSNode(state=test_state, parent=root, action="simp", visits=0)
    child2 = MCTSNode(state=test_state, parent=root, action="intro", visits=0)
    root.children = [child1, child2]

    # Test standard backpropagation
    print("\n1. Standard Backpropagation:")
    standard = StandardBackpropagation()

    print(f"   Before: root.visits={root.visits}, child1.visits={child1.visits}")

    standard.backpropagate(child1, reward=1.0)
    print(f"   After backprop with reward=1.0:")
    print(f"     child1: visits={child1.visits}, mean_value={child1.mean_value:.3f}")
    print(f"     root: visits={root.visits}, mean_value={root.mean_value:.3f}")

    # Test AMAF backpropagation
    print("\n2. AMAF Backpropagation:")
    amaf = AMAFBackpropagation(amaf_weight=1000.0)

    root2 = MCTSNode(state=test_state, visits=0)
    root2.untried_actions = ["simp", "intro", "apply"]
    child3 = MCTSNode(state=test_state, parent=root2, action="simp", visits=0)

    amaf.backpropagate(child3, reward=1.0, action="simp", visited_nodes=[child3, root2])

    print(f"   After AMAF backprop with action='simp':")
    print(f"     child3: visits={child3.visits}, mean_value={child3.mean_value:.3f}")
    print(f"     root2 AMAF stats for 'simp': visits={root2.amaf_visits.get('simp', 0)}")

    # Get combined value
    combined = amaf.get_combined_value(root2, "simp")
    print(f"     Combined value for 'simp': {combined:.3f}")


def test_domain_strategies():
    """Test domain-specific strategies"""
    print("\n" + "=" * 60)
    print("Testing Domain-Specific Strategies")
    print("=" * 60)

    # Test induction
    print("\n1. Induction Strategy:")
    induction = InductionMCTS()

    induction_state = {
        "goal": "forall n : Nat, n + 0 = n",
        "domain": "induction",
        "available_tactics": ["simp", "intro", "induction", "cases", "apply"],
        "depth": 0,
    }

    scores = induction.score_tactics(induction_state)
    print("   Tactic scores (sorted):")
    for tactic, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"     {tactic:12s}: {score:.3f}")

    bias = induction.rollout_bias(induction_state)
    print(f"   Rollout bias: {bias:.3f}")

    # Test algebraic
    print("\n2. Algebraic Strategy:")
    algebraic = AlgebraicMCTS()

    algebraic_state = {
        "goal": "forall a b : Real, (a + b)^2 = a^2 + 2*a*b + b^2",
        "domain": "algebraic",
        "available_tactics": ["ring", "simp", "calc", "norm_num", "linarith"],
        "depth": 0,
    }

    scores = algebraic.score_tactics(algebraic_state)
    print("   Tactic scores (sorted):")
    for tactic, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"     {tactic:12s}: {score:.3f}")

    bias = algebraic.rollout_bias(algebraic_state)
    print(f"   Rollout bias: {bias:.3f}")

    # Test logical
    print("\n3. Logical Strategy:")
    logical = LogicalMCTS()

    logical_state = {
        "goal": "forall P Q : Prop, P -> Q -> P",
        "domain": "logical",
        "available_tactics": ["intro", "intros", "apply", "exact", "constructor"],
        "depth": 0,
    }

    scores = logical.score_tactics(logical_state)
    print("   Tactic scores (sorted):")
    for tactic, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"     {tactic:12s}: {score:.3f}")

    bias = logical.rollout_bias(logical_state)
    print(f"   Rollout bias: {bias:.3f}")


def test_strategy_factory():
    """Test strategy factory"""
    print("\n" + "=" * 60)
    print("Testing Strategy Factory")
    print("=" * 60)

    # Create individual components
    print("\n1. Creating individual components:")

    rollout = MCTSStrategyFactory.create_rollout_policy(
        RolloutPolicyType.HEURISTIC
    )
    print(f"   Rollout: {rollout.name}")

    selection = MCTSStrategyFactory.create_selection_strategy(
        SelectionStrategyType.UCT,
        c_param=1.5
    )
    print(f"   Selection: {selection.name}")

    expansion = MCTSStrategyFactory.create_expansion_strategy(
        ExpansionStrategyType.PROGRESSIVE_WIDENING,
        widening_param=3.0
    )
    print(f"   Expansion: {expansion.name}")

    backprop = MCTSStrategyFactory.create_backpropagation_strategy(
        BackpropagationStrategyType.AMAF
    )
    print(f"   Backpropagation: {backprop.name}")

    # Create composite strategy
    print("\n2. Creating composite strategy:")
    composite = MCTSStrategyFactory.create_composite_strategy(
        rollout_policy=RolloutPolicyType.HEURISTIC,
        selection_strategy=SelectionStrategyType.ADAPTIVE_UCT,
        expansion_strategy=ExpansionStrategyType.TREE_POLICY,
        backpropagation_strategy=BackpropagationStrategyType.AMAF,
        base_c=1.3,
    )
    print(f"   Created with {len(composite)} components")
    for key, value in composite.items():
        if value is not None:
            print(f"     {key}: {value.name}")

    # Create preset strategies
    print("\n3. Creating preset strategies:")
    presets = ['balanced', 'fast', 'induction', 'algebraic']

    for preset in presets:
        strategy = MCTSStrategyFactory.create_preset_strategy(preset)
        domain = strategy.get('domain_strategy')
        domain_name = domain.name if domain else "None"
        print(f"   {preset.capitalize():12s}: domain={domain_name}")


def test_performance_tracker():
    """Test performance tracker"""
    print("\n" + "=" * 60)
    print("Testing Performance Tracker")
    print("=" * 60)

    tracker = MCTSPerformanceTracker()

    # Simulate searches
    strategies = ['uct_heuristic', 'adaptive_uct_amaf', 'thompson_tree']

    print("\nSimulating 10 searches...")
    for i in range(10):
        import random
        strategy = random.choice(strategies)

        result = MCTSSearchResult(
            success=random.random() > 0.6,
            search_time=random.uniform(1.0, 10.0),
            tree_depth=random.randint(5, 20),
            nodes_visited=random.randint(50, 200),
            value=random.uniform(0.0, 1.0),
        )

        quality = random.uniform(0.5, 1.0) if result.success else 0.0
        tracker.record_search(strategy, result, quality)

    # Display statistics
    print("\nStrategy performance:")
    for name, perf in tracker.strategy_performance.items():
        print(f"\n{name}:")
        print(f"  Success rate: {perf.success_rate:.2%}")
        print(f"  Avg time: {perf.avg_search_time:.2f}s")
        print(f"  Avg depth: {perf.avg_tree_depth:.1f}")
        print(f"  Total uses: {perf.total_uses}")

    # Compare strategies
    print("\nStrategy rankings:")
    rankings = tracker.compare_strategies(strategies)
    for rank, (strategy, score) in enumerate(rankings.items(), 1):
        print(f"  {rank}. {strategy:20s}: {score:.2f}")

    # Get best strategy
    best = tracker.get_best_strategy()
    print(f"\nBest overall strategy: {best}")

    best_induction = tracker.get_best_strategy(DomainType.INDUCTION)
    print(f"Best induction strategy: {best_induction}")


def run_all_tests():
    """Run all tests"""
    print("\n")
    print("*" * 60)
    print("LeanAide MCTS Strategy Library - Test Suite")
    print("*" * 60)

    try:
        test_rollout_policies()
        test_selection_strategies()
        test_expansion_strategies()
        test_backpropagation_strategies()
        test_domain_strategies()
        test_strategy_factory()
        test_performance_tracker()

        print("\n" + "*" * 60)
        print("All tests completed successfully!")
        print("*" * 60)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    import random
    run_all_tests()
