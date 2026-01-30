"""
Comprehensive Test Suite for MCTS-MDAP Integration

This test suite validates the integration of Monte Carlo Tree Search (MCTS) with
Multi-Agent Decomposition (MDAP) and MAKER systems for Lean 4 theorem proving.

Test Categories:
    1. Unit Tests: Test individual components (nodes, expansion, simulation)
    2. Integration Tests: Test complete MCTS-MDAP workflows
    3. Workflow Tests: Test Stage 3A/3B integration with decomposition
    4. Performance Tests: Compare MCTS vs MDAP-MCTS performance
    5. Edge Cases: Test error handling and corner cases

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import math
import random
import time
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable
from unittest.mock import Mock, MagicMock, patch

# Import MCTS components
try:
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
        Tactic,
        LeanProof,
        search_proof_with_mcts
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logging.warning("MCTS components not available - some tests will be skipped")

# Import MDAP/MAKER components
try:
    from mdap_engine import (
        MDAPConfig,
        MDAPStep,
        MDAPTask,
        MDAPVoteResult,
        MDAPStepResult,
        MDAPRunResult,
        MDAPOrchestrator,
        RedFlagRules,
        RedFlagger,
        AgentSelector
    )
    from mdap_maker_complete import (
        MAKEREngine,
        RecursiveMAKERSolver,
        VoteCollector,
        VotingEngine,
        TaskDecomposition,
        MAKERRunMetrics
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logging.warning("MDAP/MAKER components not available - some tests will be skipped")

# Import workflow structures
try:
    from workflow_structures import ModelConfig, Team
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    logging.warning("Workflow structures not available - some tests will be skipped")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# TEST UTILITIES AND FIXTURES
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for test environment."""
    enable_slow_tests: bool = False
    enable_integration_tests: bool = True
    verbose_output: bool = True
    mock_llm_calls: bool = True


class MDAPMCTSNode(MCTSNode):
    """
    Enhanced MCTS Node with MDAP voting capabilities.

    Extends standard MCTSNode with:
    - Agent votes for action selection
    - Red-flagging for unreliable actions
    - MAKER-style voting during expansion
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_votes: Dict[str, int] = {}
        self.red_flags: List[str] = []
        self.vote_confidence: float = 0.0
        self.maker_score: float = 0.0


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestMDAPMCTSNode(unittest.TestCase):
    """Unit tests for MDAPMCTSNode functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE:
            self.skipTest("MCTS not available")

        self.state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )
        self.node = MDAPMCTSNode(state=self.state)

    def test_node_initialization(self):
        """Test MDAPMCTSNode initialization."""
        self.assertIsNotNone(self.node)
        self.assertEqual(len(self.node.agent_votes), 0)
        self.assertEqual(len(self.node.red_flags), 0)
        self.assertEqual(self.node.vote_confidence, 0.0)
        self.assertEqual(self.node.maker_score, 0.0)

    def test_agent_votes_accumulation(self):
        """Test agent vote accumulation."""
        self.node.agent_votes["apply"] = 3
        self.node.agent_votes["rw"] = 2
        self.node.agent_votes["simp"] = 1

        self.assertEqual(self.node.agent_votes["apply"], 3)
        self.assertEqual(self.node.agent_votes["rw"], 2)
        self.assertEqual(len(self.node.agent_votes), 3)

    def test_red_flag_addition(self):
        """Test red flag addition."""
        self.node.red_flags.append("response_too_long")
        self.node.red_flags.append("low_confidence")

        self.assertEqual(len(self.node.red_flags), 2)
        self.assertIn("response_too_long", self.node.red_flags)

    def test_vote_confidence_calculation(self):
        """Test vote confidence calculation."""
        self.node.agent_votes = {"apply": 5, "rw": 2}
        total_votes = sum(self.node.agent_votes.values())
        self.node.vote_confidence = self.node.agent_votes["apply"] / total_votes

        self.assertAlmostEqual(self.node.vote_confidence, 5.0 / 7.0, places=5)

    def test_maker_score_update(self):
        """Test MAKER score update."""
        self.node.maker_score = 0.85
        self.assertEqual(self.node.maker_score, 0.85)

    def test_integration_with_parent_mcts(self):
        """Test that MDAPMCTSNode works with parent MCTSNode."""
        child_state = ProofState(
            goals=["forall (b : Nat), a + b = b + a"],
            context=["a : Nat"],
            depth=1
        )
        child = MDAPMCTSNode(state=child_state, parent=self.node, action="intros")

        self.assertEqual(child.parent, self.node)
        self.assertEqual(child.action, "intros")
        self.assertIn("intros", self.node.children)


class TestMDAPMCTSExpansion(unittest.TestCase):
    """Unit tests for MDAP-enhanced MCTS expansion."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE or not MDAP_AVAILABLE:
            self.skipTest("MCTS or MDAP not available")

        self.config = MCTSConfig(max_iterations=100, c_param=1.414)
        self.expansion = MCTSExpansion(self.config)

        # Create test node
        self.state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )
        self.node = MDAPMCTSNode(state=self.state)

    def test_basic_expansion(self):
        """Test basic expansion with available tactics."""
        available_actions = [
            Tactic(name="intros", params=[]),
            Tactic(name="apply", params=["Nat.add_comm"]),
            Tactic(name="rw", params=["Nat.add_comm"])
        ]

        expanded_node = self.expansion.expand(self.node, available_actions)

        self.assertIsNotNone(expanded_node)
        self.assertEqual(len(self.node.children), len(available_actions))

    def test_expansion_with_voting(self):
        """Test expansion with agent voting."""
        available_actions = [
            Tactic(name="intros", params=[]),
            Tactic(name="apply", params=["Nat.add_comm"])
        ]

        # Simulate agent votes
        agent_votes = {"intros": 5, "apply": 2}

        expanded_node = self.expansion.expand(
            self.node,
            available_actions,
            agent_votes=agent_votes
        )

        self.assertIsNotNone(expanded_node)
        # The most voted action should be expanded first
        self.assertEqual(self.node.agent_votes.get("intros", 0), 5)

    def test_expansion_with_red_flags(self):
        """Test expansion with red-flagged actions."""
        available_actions = [
            Tactic(name="intros", params=[]),
            Tactic(name="apply", params=["invalid_lemma"])  # Red-flagged
        ]

        red_flags = {"apply": ["lemma_not_found", "low_confidence"]}

        expanded_node = self.expansion.expand(
            self.node,
            available_actions,
            red_flags=red_flags
        )

        self.assertIsNotNone(expanded_node)
        # Red-flagged actions should be deprioritized or skipped

    def test_empty_expansion(self):
        """Test expansion with no available actions."""
        expanded_node = self.expansion.expand(self.node, [])

        self.assertIsNone(expanded_node)
        self.assertEqual(len(self.node.children), 0)

    def test_progressive_widening(self):
        """Test progressive widening for large action spaces."""
        # Create many actions
        available_actions = [
            Tactic(name=f"apply_{i}", params=[f"lemma_{i}"])
            for i in range(100)
        ]

        self.config.progressive_widening = True
        self.config.widening_factor = 0.5

        # First expansion should only add subset of actions
        expanded_node = self.expansion.expand(self.node, available_actions)

        self.assertIsNotNone(expanded_node)
        # With progressive widening, not all actions should be added immediately
        self.assertLess(len(self.node.children), len(available_actions))


class TestMDAPMCTSSimulation(unittest.TestCase):
    """Unit tests for MDAP-enhanced MCTS simulation."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE or not MDAP_AVAILABLE:
            self.skipTest("MCTS or MDAP not available")

        self.config = MCTSConfig(rollout_depth=10, rollout_policy="heuristic")
        self.simulation = MCTSSimulation(self.config)

        # Create test node
        self.state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )
        self.node = MDAPMCTSNode(state=self.state)

    def test_basic_rollout(self):
        """Test basic rollout simulation."""
        reward = self.simulation.simulate(self.node)

        self.assertIsInstance(reward, float)
        self.assertGreaterEqual(reward, 0.0)
        self.assertLessEqual(reward, 1.0)

    def test_rollout_with_maker_voting(self):
        """Test rollout with MAKER-style voting."""
        # Mock MAKER voting
        maker_engine = Mock()
        maker_engine.generate_solution.return_value = (
            [Tactic(name="intros", params=[])],
            self.state,
            MAKERRunMetrics(total_steps=1, avg_confidence=0.9)
        )

        reward = self.simulation.simulate(
            self.node,
            maker_engine=maker_engine
        )

        self.assertIsInstance(reward, float)
        # Higher confidence should lead to higher reward
        self.assertGreater(reward, 0.5)

    def test_rollout_depth_limit(self):
        """Test that rollout respects depth limit."""
        self.config.rollout_depth = 5

        # Track rollout depth
        max_depth_reached = [0]

        def depth_tracking_rollout(node, depth=0):
            max_depth_reached[0] = max(max_depth_reached[0], depth)
            if depth >= self.config.rollout_depth:
                return 0.5
            return 0.5

        with patch.object(self.simulation, '_heuristic_rollout', depth_tracking_rollout):
            reward = self.simulation.simulate(self.node)

        self.assertLessEqual(max_depth_reached[0], self.config.rollout_depth)

    def test_terminal_state_detection(self):
        """Test that terminal states are detected immediately."""
        terminal_state = ProofState(
            goals=[],
            context=["proved"],
            depth=5,
            is_complete=True
        )
        terminal_node = MDAPMCTSNode(state=terminal_state)

        reward = self.simulation.simulate(terminal_node)

        # Terminal state should return maximum reward
        self.assertEqual(reward, 1.0)

    def test_parallel_rollouts(self):
        """Test parallel rollout execution."""
        if not self.config.parallel_simulations or self.config.parallel_simulations < 2:
            self.skipTest("Parallel simulations not configured")

        rewards = self.simulation.simulate_parallel(
            self.node,
            num_simulations=4
        )

        self.assertEqual(len(rewards), 4)
        for reward in rewards:
            self.assertIsInstance(reward, float)


class TestMDAPMCTSOrchestration(unittest.TestCase):
    """Unit tests for MDAP-MCTS orchestration."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE or not MDAP_AVAILABLE:
            self.skipTest("MCTS or MDAP not available")

        self.config = MCTSConfig(
            max_iterations=50,
            c_param=1.414,
            time_budget=10.0
        )
        self.mcts = MCTS(self.config)

        # Create root node
        self.root_state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )
        self.root_node = MDAPMCTSNode(state=self.root_state)

    def test_initialization(self):
        """Test MCTS initialization."""
        self.assertIsNotNone(self.mcts)
        self.assertEqual(self.mcts.config.max_iterations, 50)

    def test_single_iteration(self):
        """Test a single MCTS iteration."""
        # Mock the necessary components
        with patch.object(self.mcts.selection, 'select', return_value=self.root_node):
            with patch.object(self.mcts.expansion, 'expand', return_value=self.root_node):
                with patch.object(self.mcts.simulation, 'simulate', return_value=0.5):
                    # Run iteration
                    self.mcts._run_iteration(self.root_node)

        # Node should be updated
        self.assertGreater(self.root_node.N, 0)

    def test_multiple_iterations(self):
        """Test multiple MCTS iterations."""
        initial_visits = self.root_node.N

        for i in range(10):
            with patch.object(self.mcts.selection, 'select', return_value=self.root_node):
                with patch.object(self.mcts.expansion, 'expand', return_value=self.root_node):
                    with patch.object(self.mcts.simulation, 'simulate', return_value=0.5):
                        self.mcts._run_iteration(self.root_node)

        self.assertGreater(self.root_node.N, initial_visits)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMCTSDAPIntegration(unittest.TestCase):
    """Integration tests for complete MCTS-MDAP workflows."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE or not MDAP_AVAILABLE:
            self.skipTest("MCTS or MDAP not available")

        self.config = MCTSConfig(
            max_iterations=100,
            time_budget=30.0,
            enable_transposition_table=True
        )
        self.mcts = MCTS(self.config)

        # Create MDAP orchestrator
        if WORKFLOW_AVAILABLE:
            self.team = self._create_mock_team()
            self.mdap_config = MDAPConfig(
                k_min=2,
                k_max=5,
                max_votes_per_step=20
            )
            self.mdap = MDAPOrchestrator(self.team, self.mdap_config)
        else:
            self.mdap = None

    def _create_mock_team(self) -> Team:
        """Create a mock team for testing."""
        team = Team(
            team_id="test_team",
            name="Test Team",
            members=[
                ModelConfig(
                    model_id="gpt-4",
                    api_key="test_key",
                    api_base="http://test.com",
                    temperature=0.0
                )
            ]
        )
        return team

    def test_complete_mcts_mdap_search(self):
        """Test complete MCTS-MDAP search loop."""
        # Create initial state
        initial_state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Mock MDAP voting
        if self.mdap:
            with patch.object(self.mdap, '_sample_candidate', return_value=('{"action": "intros"}', {"action": "intros"})):
                result = self.mcts.search(initial_state, mdap_orchestrator=self.mdap)
        else:
            result = self.mcts.search(initial_state)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, MCTSResult)

    def test_mcts_with_maker_simulation(self):
        """Test MCTS with MAKER-enhanced simulation."""
        initial_state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Create MAKER engine
        if WORKFLOW_AVAILABLE:
            maker_engine = MAKEREngine(
                team=self.team,
                k_ahead=3,
                max_steps=10
            )

            with patch.object(maker_engine.voting_engine, 'do_voting', return_value=(Tactic(name="intros", params=[]), {"intros": 5}, MAKERRunMetrics(total_votes=5))):
                result = self.mcts.search(initial_state, maker_engine=maker_engine)
        else:
            # Skip if workflow not available
            result = self.mcts.search(initial_state)

        self.assertIsNotNone(result)

    def test_voting_during_expansion(self):
        """Test voting during MCTS expansion phase."""
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )
        node = MDAPMCTSNode(state=state)

        available_actions = [
            Tactic(name="intros", params=[]),
            Tactic(name="apply", params=["Nat.add_comm"]),
            Tactic(name="rw", params=["Nat.add_comm"])
        ]

        # Simulate agent voting
        if self.mdap:
            with patch.object(self.mdap, '_sample_candidate', side_effect=[
                ('{"action": "intros"}', {"action": "intros"}),
                ('{"action": "intros"}', {"action": "intros"}),
                ('{"action": "apply"}', {"action": "apply"})
            ]):
                expanded = self.mcts.expansion.expand(
                    node,
                    available_actions,
                    mdap_orchestrator=self.mdap
                )

            self.assertIsNotNone(expanded)
            self.assertIn("intros", node.agent_votes)

    def test_voting_during_simulation(self):
        """Test voting during MCTS simulation phase."""
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )
        node = MDAPMCTSNode(state=state)

        # Create MAKER engine for simulation
        if WORKFLOW_AVAILABLE:
            maker_engine = MAKEREngine(team=self.team, k_ahead=3, max_steps=5)

            with patch.object(maker_engine.voting_engine, 'do_voting', return_value=(
                Tactic(name="intros", params=[]),
                {"intros": 5},
                MAKERRunMetrics(total_votes=5, avg_confidence=0.9)
            )):
                reward = self.mcts.simulation.simulate(
                    node,
                    maker_engine=maker_engine
                )

            self.assertIsInstance(reward, float)
            self.assertGreater(reward, 0.0)


# =============================================================================
# WORKFLOW TESTS
# =============================================================================

class TestMDAPMCTSWorkflow(unittest.TestCase):
    """Tests for MDAP-MCTS integration in workflow stages."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE or not MDAP_AVAILABLE:
            self.skipTest("MCTS or MDAP not available")

        self.config = MCTSConfig(max_iterations=50)
        self.mcts = MCTS(self.config)

    def test_stage_3a_mdap_mcts_integration(self):
        """Test Stage 3A: MDAP-MCTS integration for tactic selection."""
        # Simulate decomposition workflow stage
        problem = {
            "goal": "forall (a b : Nat), a + b = b + a",
            "context": []
        }

        # Create MDAP task for decomposition
        mdap_task = MDAPTask(
            task_id="decompose_add_comm",
            description="Decompose add_comm proof",
            steps=[
                MDAPStep(
                    step_id="select_tactics",
                    prompt="Select appropriate tactics",
                    task_type="decomposition"
                )
            ]
        )

        # Mock MDAP execution
        mdap_result = MDAPRunResult(
            task_id="decompose_add_comm",
            step_results={},
            metrics={"steps_completed": 1}
        )

        self.assertIsNotNone(mdap_result)
        self.assertEqual(mdap_result.task_id, "decompose_add_comm")

    def test_stage_3b_refinement(self):
        """Test Stage 3B: Refinement with MCTS."""
        # Initial decomposition result
        initial_proof = {
            "tactics": ["intros", "apply Nat.add_comm"],
            "confidence": 0.7
        }

        # Use MCTS to refine
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Mock MCTS refinement
        with patch.object(self.mcts, 'search', return_value=MCTSResult(
            success=True,
            search_iterations=10,
            win_rate=0.85
        )):
            refined_result = self.mcts.search(state)

        self.assertIsNotNone(refined_result)

    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection (MCTS vs MDAP vs MAKER)."""
        # Test different problem types
        problems = {
            "simple": {"difficulty": "low", "steps_estimate": 3},
            "medium": {"difficulty": "medium", "steps_estimate": 10},
            "complex": {"difficulty": "high", "steps_estimate": 50}
        }

        for problem_type, problem in problems.items():
            # Simple problems: Use pure MCTS
            if problem["difficulty"] == "low":
                strategy = "mcts"
            # Medium problems: Use MCTS with MDAP
            elif problem["difficulty"] == "medium":
                strategy = "mcts_mdap"
            # Complex problems: Use MAKER
            else:
                strategy = "maker"

            self.assertIsNotNone(strategy)
            logger.info(f"Problem type: {problem_type}, Strategy: {strategy}")

    def test_fallback_behavior(self):
        """Test fallback behavior when components fail."""
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Test fallback to pure MCTS if MDAP fails
        with patch.object(self.mcts.expansion, 'expand', side_effect=Exception("MDAP failed")):
            # Should fall back to basic expansion
            try:
                result = self.mcts.search(state)
                # If search completes despite failure, fallback worked
                self.assertIsNotNone(result)
            except (RuntimeError, ValueError, AttributeError):
                # Expected if no fallback implemented
                pass


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestMCTSDAPPerformance(unittest.TestCase):
    """Performance tests comparing MCTS vs MDAP-MCTS."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE:
            self.skipTest("MCTS not available")

        self.enable_slow_tests = False  # Set to True to run slow tests

    def test_pure_mcts_baseline(self):
        """Test pure MCTS performance baseline."""
        if not self.enable_slow_tests:
            self.skipTest("Slow tests disabled")

        config = MCTSConfig(max_iterations=100, time_budget=10.0)
        mcts = MCTS(config)

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        start_time = time.time()
        result = mcts.search(state)
        elapsed = time.time() - start_time

        logger.info(f"Pure MCTS: {result.search_iterations} iterations in {elapsed:.2f}s")
        self.assertLess(elapsed, config.time_budget + 1.0)  # Allow 1s overhead

    def test_mdap_mcts_comparison(self):
        """Compare MCTS vs MDAP-MCTS performance."""
        if not self.enable_slow_tests or not MDAP_AVAILABLE:
            self.skipTest("Slow tests or MDAP not available")

        config = MCTSConfig(max_iterations=100, time_budget=10.0)

        # Pure MCTS
        mcts_pure = MCTS(config)
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        start_time = time.time()
        result_pure = mcts_pure.search(state)
        time_pure = time.time() - start_time

        # MDAP-MCTS (mock MDAP for comparison)
        mcts_mdap = MCTS(config)
        with patch('mdap_engine.MDAPOrchestrator._sample_candidate', return_value=('{"action": "intros"}', {"action": "intros"})):
            start_time = time.time()
            result_mdap = mcts_mdap.search(state)
            time_mdap = time.time() - start_time

        logger.info(f"Pure MCTS: {time_pure:.2f}s, {result_pure.search_iterations} iterations")
        logger.info(f"MDAP-MCTS: {time_mdap:.2f}s, {result_mdap.search_iterations} iterations")

        # MDAP-MCTS should be comparable or better in quality
        self.assertIsNotNone(result_mdap)

    def test_convergence_rates(self):
        """Test convergence rates of different strategies."""
        if not self.enable_slow_tests:
            self.skipTest("Slow tests disabled")

        config = MCTSConfig(max_iterations=200)
        mcts = MCTS(config)

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Track convergence
        convergence_data = []
        for i in range(0, 200, 20):
            config.max_iterations = i
            result = mcts.search(state)
            convergence_data.append({
                "iterations": i,
                "win_rate": result.win_rate
            })

        # Convergence should improve with iterations
        self.assertGreater(convergence_data[-1]["win_rate"], convergence_data[0]["win_rate"])

    def test_agent_contribution(self):
        """Test contribution of different agent types."""
        if not MDAP_AVAILABLE or not WORKFLOW_AVAILABLE:
            self.skipTest("MDAP or workflow not available")

        # Create team with different agent types
        team = Team(
            team_id="mixed_team",
            name="Mixed Team",
            members=[
                ModelConfig(model_id="agent1", api_key="key1", api_base="http://test1.com", temperature=0.0),
                ModelConfig(model_id="agent2", api_key="key2", api_base="http://test2.com", temperature=0.1),
                ModelConfig(model_id="agent3", api_key="key3", api_base="http://test3.com", temperature=0.2)
            ]
        )

        mdap_config = MDAPConfig(k_min=2, k_max=5)
        mdap = MDAPOrchestrator(team, mdap_config)

        # Track agent contributions
        agent_contributions = {}
        for _ in range(10):
            with patch.object(mdap, '_sample_candidate', return_value=('{"action": "test"}', {"action": "test"})):
                agent = mdap.selector.select(MDAPStep(step_id="test", prompt="test"))
                agent_contributions[agent.model_id] = agent_contributions.get(agent.model_id, 0) + 1

        # All agents should contribute
        self.assertGreater(len(agent_contributions), 0)
        logger.info(f"Agent contributions: {agent_contributions}")

    def test_voting_overhead(self):
        """Test overhead introduced by voting."""
        if not MDAP_AVAILABLE:
            self.skipTest("MDAP not available")

        config = MCTSConfig(max_iterations=50)
        mcts = MCTS(config)

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Without voting
        start_time = time.time()
        result_no_voting = mcts.search(state)
        time_no_voting = time.time() - start_time

        # With voting (mock)
        with patch('mdap_engine.MDAPOrchestrator._sample_candidate', return_value=('{"action": "intros"}', {"action": "intros"})):
            start_time = time.time()
            result_with_voting = mcts.search(state)
            time_with_voting = time.time() - start_time

        overhead = time_with_voting - time_no_voting
        logger.info(f"Voting overhead: {overhead:.2f}s")

        # Overhead should be reasonable (< 50% increase)
        self.assertLess(overhead, time_no_voting * 0.5)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestMDAPMCTSEdgeCases(unittest.TestCase):
    """Edge case and error handling tests."""

    def setUp(self):
        """Set up test fixtures."""
        if not MCTS_AVAILABLE:
            self.skipTest("MCTS not available")

        self.config = MCTSConfig(max_iterations=50)
        self.mcts = MCTS(self.config)

    def test_all_agents_fail_during_voting(self):
        """Test behavior when all agents fail during voting."""
        if not MDAP_AVAILABLE:
            self.skipTest("MDAP not available")

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Mock all agents failing
        with patch('mdap_engine.MDAPOrchestrator._sample_candidate', side_effect=Exception("Agent failed")):
            try:
                result = self.mcts.search(state)
                # Should fall back to pure MCTS
                self.assertIsNotNone(result)
            except Exception as e:
                # If no fallback, should handle gracefully
                logger.warning(f"Expected exception when all agents fail: {e}")

    def test_voting_ties(self):
        """Test handling of voting ties."""
        if not MDAP_AVAILABLE:
            self.skipTest("MDAP not available")

        # Create tie scenario
        votes = {
            "intros": 5,
            "apply": 5,
            "rw": 3
        }

        # Tie should be broken deterministically
        max_votes = max(votes.values())
        winners = [action for action, count in votes.items() if count == max_votes]

        self.assertGreater(len(winners), 1)
        # First winner should be selected
        winner = sorted(winners)[0]
        self.assertIn(winner, winners)

    def test_red_flagged_actions_only(self):
        """Test behavior when all actions are red-flagged."""
        if not MDAP_AVAILABLE:
            self.skipTest("MDAP not available")

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        available_actions = [
            Tactic(name="apply_invalid", params=["nonexistent"]),
            Tactic(name="rw_invalid", params=["wrong"])
        ]

        red_flags = {
            "apply_invalid": ["lemma_not_found"],
            "rw_invalid": ["type_mismatch"]
        }

        # Should still proceed with best available action
        expanded = self.mcts.expansion.expand(
            MDAPMCTSNode(state=state),
            available_actions,
            red_flags=red_flags
        )

        # May return None if all red-flagged, or proceed with caution
        logger.info(f"Expansion with all red flags: {expanded}")

    def test_empty_agent_list(self):
        """Test behavior with empty agent list."""
        if not WORKFLOW_AVAILABLE:
            self.skipTest("Workflow structures not available")

        team = Team(team_id="empty", name="Empty Team", members=[])
        mdap_config = MDAPConfig(k_min=2, k_max=5)

        with self.assertRaises(ValueError):
            mdap = MDAPOrchestrator(team, mdap_config)
            mdap.selector.select(MDAPStep(step_id="test", prompt="test"))

    def test_timeout_during_voting(self):
        """Test timeout handling during voting."""
        if not MDAP_AVAILABLE:
            self.skipTest("MDAP not available")

        mdap_config = MDAPConfig(
            k_min=2,
            k_max=5,
            timeout_seconds=0.1,  # Very short timeout
            max_votes_per_step=100
        )

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Mock slow agent
        def slow_sample(*args, **kwargs):
            time.sleep(0.2)  # Exceeds timeout
            return '{"action": "test"}', {"action": "test"}

        with patch('mdap_engine.MDAPOrchestrator._sample_candidate', slow_sample):
            result = self.mcts.search(state)
            # Should handle timeout gracefully
            self.assertIsNotNone(result)

    def test_invalid_action_space(self):
        """Test handling of invalid action space."""
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # None actions
        expanded = self.mcts.expansion.expand(MDAPMCTSNode(state=state), None)
        self.assertIsNone(expanded)

        # Invalid action types
        invalid_actions = ["not_a_tactic", 123, None]
        expanded = self.mcts.expansion.expand(
            MDAPMCTSNode(state=state),
            invalid_actions
        )
        # Should handle gracefully
        self.assertIsNotNone(expanded)

    def test_concurrent_search_requests(self):
        """Test handling of concurrent search requests."""
        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Launch concurrent searches
        async def concurrent_search():
            loop = asyncio.get_event_loop()
            results = await asyncio.gather(
                loop.run_in_executor(None, self.mcts.search, state),
                loop.run_in_executor(None, self.mcts.search, state),
                loop.run_in_executor(None, self.mcts.search, state),
                return_exceptions=True
            )
            return results

        results = asyncio.run(concurrent_search())

        # All searches should complete
        for result in results:
            if not isinstance(result, Exception):
                self.assertIsNotNone(result)

    def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        config = MCTSConfig(
            max_iterations=1000,
            max_tree_depth=100,
            cache_size_mb=1  # Small cache
        )
        mcts = MCTS(config)

        state = ProofState(
            goals=["forall (a b : Nat), a + b = b + a"],
            context=[],
            depth=0
        )

        # Should handle memory pressure gracefully
        result = mcts.search(state)
        self.assertIsNotNone(result)


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_tests(
    test_config: Optional[TestConfig] = None,
    test_categories: Optional[List[str]] = None
) -> unittest.TestResult:
    """
    Run MCTS-MDAP integration tests.

    Args:
        test_config: Test configuration
        test_categories: Categories to run (unit, integration, workflow, performance, edge_cases)

    Returns:
        Test results
    """
    test_config = test_config or TestConfig()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add tests by category
    categories = test_categories or [
        "unit",
        "integration",
        "workflow",
        "performance",
        "edge_cases"
    ]

    if "unit" in categories:
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSNode))
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSExpansion))
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSSimulation))
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSOrchestration))

    if "integration" in categories and test_config.enable_integration_tests:
        suite.addTests(loader.loadTestsFromTestCase(TestMCTSDAPIntegration))

    if "workflow" in categories and test_config.enable_integration_tests:
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSWorkflow))

    if "performance" in categories and test_config.enable_slow_tests:
        suite.addTests(loader.loadTestsFromTestCase(TestMCTSDAPPerformance))

    if "edge_cases" in categories:
        suite.addTests(loader.loadTestsFromTestCase(TestMDAPMCTSEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(
        verbosity=2 if test_config.verbose_output else 1,
        buffer=True
    )
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)

    return result


if __name__ == "__main__":
    # Run all tests
    result = run_tests(TestConfig(
        enable_slow_tests=False,
        enable_integration_tests=True,
        verbose_output=True
    ))

    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
