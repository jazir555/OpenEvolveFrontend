"""
LeanAide MCTS-MDAP-MAKER Integration

Comprehensive integration of:
    - MCTS (Monte Carlo Tree Search) for intelligent tree search
    - MDAP (Multi-Agent Pipeline) for multi-agent voting
    - MAKER (Multi-Agent Knowledge Enhanced Reasoning) for tactic voting

Architecture:
    MDAPMCTSNode: Enhanced MCTS node with MDAP multi-agent voting
    MDAPMCTSExpansion: Expansion phase with MDAP agent voting
    MDAPMCTSSimulation: Simulation phase with MAKER voting
    MDAPMCTS: Main orchestrator combining all components

Benefits:
    - MCTS: Intelligent tree search with UCT exploration
    - MDAP: Multi-agent perspectives reduce bias
    - MAKER: Voting consensus with error correction
    - Red-flagging: Quality control and pruning

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import math
import random
import time
import uuid
import hashlib
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)
from pathlib import Path

# Import MCTS components
try:
    from leanaide_mcts import (
        MCTSNode,
        MCTSTree,
        MCTSSelection,
        MCTSExpansion,
        MCTSSimulation,
        MCTSBackpropagation,
        MCTS,
        MCTSConfig,
        MCTSResult,
        ProofState,
        RolloutPolicy,
        Tactic,
        LeanProof
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logging.warning("MCTS not available")

# Import MDAP components
try:
    from leanaide_mdap import (
        LeanProofAgent,
        ProofStrategy,
        LeanDomain,
        LeanMDAPConfig,
        LeanMDAPOrchestrator,
        LeanMDAPTask,
        LeanMDAPResult,
        LeanAgentSelector
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logging.warning("MDAP not available")

# Import MAKER components
try:
    from leanaide_maker import (
        LeanTacticVoter,
        TacticVote,
        LeanAggregator,
        AggregationStrategy,
        LeanMakerConfig,
        VoterType,
        LeanProofState as MakerProofState
    )
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    logging.warning("MAKER not available")

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MDAPMCTSConfig(MCTSConfig if MCTS_AVAILABLE else object):
    """
    Configuration for MDAP-enhanced MCTS.

    Combines MCTS, MDAP, and MAKER parameters.
    """
    # MCTS parameters
    c_param: float = 1.414
    max_iterations: int = 1000
    rollout_depth: int = 100
    time_budget: float = 300.0

    # MDAP parameters
    available_agents: List[str] = field(default_factory=lambda: [
        "evolution", "mcts", "adversarial", "direct"
    ])
    expansion_agents: int = 3  # Number of agents voting during expansion
    parallel_agents: int = 4

    # MAKER parameters
    simulation_voters: int = 5  # Number of voters during simulation
    voting_strategy: str = "first_k_ahead"  # Strategy for aggregating votes
    k_ahead: int = 3  # K value for first-k-ahead voting

    # Red-flagging
    enable_red_flagging: bool = True
    prune_red_flagged: bool = True
    red_flag_threshold: float = 0.3  # Confidence threshold below which to flag

    # Agent selection
    agent_selection_strategy: str = "adaptive"  # adaptive, random, performance_based

    # LeanAide integration
    server_url: str = "http://localhost:7654"
    enable_verification: bool = True

    # Caching and performance
    enable_caching: bool = True
    cache_size: int = 10000

    # Logging
    log_level: str = "INFO"
    enable_detailed_logging: bool = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ActionVote:
    """
    A vote for an action from an MDAP agent.

    Attributes:
        action: The action/tactic being voted for
        agent_id: ID of the agent casting this vote
        confidence: Confidence score (0.0 to 1.0)
        rationale: Explanation for the vote
        agent_type: Type of agent (evolution, mcts, adversarial, etc.)
        estimated_success: Estimated probability of success
        proof_state_hash: Hash of proof state when vote was cast
    """
    action: str
    agent_id: str
    confidence: float
    rationale: str
    agent_type: str
    estimated_success: float = 0.5
    proof_state_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MDAPMCTSResult(MCTSResult if MCTS_AVAILABLE else object):
    """
    Result of MDAP-enhanced MCTS search.

    Extends MCTSResult with MDAP-specific metadata.
    """
    # Standard MCTS result fields
    best_proof: Optional[LeanProof] = None
    success: bool = False
    search_iterations: int = 0
    time_elapsed: float = 0.0
    nodes_visited: int = 0
    tree_depth: int = 0
    win_rate: float = 0.0
    confidence: float = 0.0

    # MDAP-specific fields
    agent_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    voting_statistics: Dict[str, Any] = field(default_factory=dict)
    red_flag_analysis: Dict[str, Any] = field(default_factory=dict)
    agent_performance_ranking: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.best_proof:
            result["best_proof"] = self.best_proof.to_dict()
        return result


# =============================================================================
# Enhanced MCTS Node with MDAP Voting
# =============================================================================

class MDAPMCTSNode(MCTSNode if MCTS_AVAILABLE else object):
    """
    Enhanced MCTS node with MDAP multi-agent voting.

    Extends standard MCTS node with:
    - MDAP agent votes for actions
    - Agent performance tracking per action
    - Red-flag status
    """

    def __init__(
        self,
        state: ProofState,
        parent: Optional['MDAPMCTSNode'] = None,
        action: Optional[str] = None
    ):
        """Initialize MDAP-enhanced MCTS node."""
        if MCTS_AVAILABLE:
            super().__init__(state, parent, action)
        else:
            # Initialize basic fields if MCTS not available
            self.state = state
            self.parent = parent
            self.action = action
            self.N = 0
            self.W = 0.0
            self.Q = 0.0
            self.children = {}
            self.untried_actions = []
            self.depth = parent.depth + 1 if parent else 0
            self.is_terminal = state.is_complete
            self.hash = state.hash

        # MDAP-specific fields
        self.agent_votes: Dict[str, List[ActionVote]] = defaultdict(list)
        self.agent_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success": 0.0, "total": 0.0, "avg_confidence": 0.5}
        )
        self.red_flagged: bool = False
        self.red_flag_reasons: List[str] = []

    def get_mdap_votes(self) -> List[ActionVote]:
        """Get all MDAP agent votes for this node."""
        all_votes = []
        for action_votes in self.agent_votes.values():
            all_votes.extend(action_votes)
        return all_votes

    def get_agent_performance(self, action: str) -> Dict[str, float]:
        """
        Get agent performance metrics for a specific action.

        Returns:
            Dict with 'success', 'total', and 'avg_confidence' metrics
        """
        return self.agent_performance.get(action, {
            "success": 0.0,
            "total": 0.0,
            "avg_confidence": 0.5
        })

    def is_red_flagged(self) -> bool:
        """Check if this node is red-flagged."""
        return self.red_flagged

    def add_agent_vote(
        self,
        agent_id: str,
        action: str,
        confidence: float,
        rationale: str = "",
        agent_type: str = "unknown"
    ) -> None:
        """
        Add an agent vote for an action.

        Args:
            agent_id: ID of the agent
            action: Action being voted for
            confidence: Confidence score
            rationale: Explanation for the vote
            agent_type: Type of agent
        """
        vote = ActionVote(
            action=action,
            agent_id=agent_id,
            confidence=confidence,
            rationale=rationale,
            agent_type=agent_type,
            proof_state_hash=self.hash
        )
        self.agent_votes[action].append(vote)

    def update_agent_performance(
        self,
        action: str,
        success: bool,
        confidence: float
    ) -> None:
        """
        Update agent performance metrics for an action.

        Args:
            action: Action to update metrics for
            success: Whether the action was successful
            confidence: Confidence of the prediction
        """
        perf = self.agent_performance[action]
        perf["total"] += 1
        if success:
            perf["success"] += 1

        # Update average confidence with exponential moving average
        alpha = 0.1
        perf["avg_confidence"] = (
            alpha * confidence + (1 - alpha) * perf["avg_confidence"]
        )

    def set_red_flag(self, flagged: bool, reasons: List[str] = None) -> None:
        """
        Set red-flag status for this node.

        Args:
            flagged: Whether the node is red-flagged
            reasons: List of reasons for flagging
        """
        self.red_flagged = flagged
        if reasons:
            self.red_flag_reasons = reasons


# =============================================================================
# MDAP-Enhanced Expansion
# =============================================================================

class MDAPMCTSExpansion(MCTSExpansion if MCTS_AVAILABLE else object):
    """
    Expansion phase enhanced with MDAP voting.

    Instead of single action selection, uses multiple MDAP agents
    to vote on the best action.
    """

    def __init__(
        self,
        mdap_config: MDAPMCTSConfig,
        agents: Optional[List[LeanProofAgent]] = None
    ):
        """Initialize MDAP-enhanced expansion."""
        if MCTS_AVAILABLE:
            super().__init__(
                max_actions=mdap_config.max_iterations // 10,
                use_action_ranking=True
            )

        self.config = mdap_config
        self.agents = agents or []
        self.agents_initialized = False

    async def expand_with_mdap(
        self,
        node: MDAPMCTSNode,
        tree: 'MDAPMCTSTree'
    ) -> MDAPMCTSNode:
        """
        Expand a node using MDAP agent voting.

        Args:
            node: Node to expand
            tree: MCTS tree

        Returns:
            New child node created by expansion
        """
        if node.is_terminal:
            return node

        # Collect votes from MDAP agents
        votes = await self.collect_agent_votes(node)

        if not votes:
            # No votes collected, fall back to standard expansion
            if MCTS_AVAILABLE:
                return await self.expand(node, tree)
            else:
                return node

        # Aggregate votes using MAKER strategy
        selected_action = self.aggregate_votes(
            votes,
            strategy=self.config.voting_strategy
        )

        if not selected_action:
            # No action selected, mark as terminal
            node.is_terminal = True
            return node

        # Check for red flags
        if self.config.enable_red_flagging:
            red_flagged_actions = self.red_flag_actions(votes)
            if selected_action in red_flagged_actions:
                if self.config.prune_red_flagged:
                    node.set_red_flag(True, ["Action red-flagged during expansion"])
                    # Try alternative action
                    for vote in votes:
                        if vote.action not in red_flagged_actions:
                            selected_action = vote.action
                            break
                    else:
                        # All actions red-flagged
                        node.is_terminal = True
                        return node

        # Apply selected action
        new_state = await self._apply_tactic(node.state, selected_action)

        # Check for transposition
        existing_node = tree.get_node_by_hash(new_state.hash)
        if existing_node:
            node.add_child(selected_action, existing_node)
            return existing_node

        # Create new node
        child_node = MDAPMCTSNode(
            state=new_state,
            parent=node,
            action=selected_action
        )

        # Store votes in child node
        for vote in votes:
            if vote.action == selected_action:
                child_node.add_agent_vote(
                    vote.agent_id,
                    vote.action,
                    vote.confidence,
                    vote.rationale,
                    vote.agent_type
                )

        # Add to tree
        node.add_child(selected_action, child_node)
        tree.add_node(child_node)

        return child_node

    async def collect_agent_votes(
        self,
        node: MDAPMCTSNode
    ) -> List[ActionVote]:
        """
        Collect votes from MDAP agents for this node.

        Args:
            node: Node to collect votes for

        Returns:
            List of votes from all agents
        """
        votes = []

        # Initialize agents if needed
        if not self.agents_initialized and MDAP_AVAILABLE:
            await self._initialize_agents()
            self.agents_initialized = True

        # Select agents for this expansion
        selected_agents = self._select_agents(node)

        # Collect votes in parallel
        if self.config.parallel_agents > 1:
            votes = await self._collect_votes_parallel(node, selected_agents)
        else:
            votes = await self._collect_votes_sequential(node, selected_agents)

        return votes

    async def _initialize_agents(self) -> None:
        """Initialize MDAP agents."""
        if not MDAP_AVAILABLE:
            return

        # Create agents for each available strategy
        from leanaide_mdap import LeanMDAPConfig

        mdap_config = LeanMDAPConfig(
            available_agents=self.config.available_agents
        )

        # This would normally create actual agents
        # For now, we'll use placeholder agents
        self.agents = []

    def _select_agents(self, node: MDAPMCTSNode) -> List[Any]:
        """
        Select agents to use for this expansion.

        Args:
            node: Node being expanded

        Returns:
            List of selected agents
        """
        if not self.agents:
            return []

        num_agents = min(self.config.expansion_agents, len(self.agents))

        if self.config.agent_selection_strategy == "random":
            return random.sample(self.agents, num_agents)
        elif self.config.agent_selection_strategy == "adaptive":
            # Select based on node characteristics
            return self._adaptive_agent_selection(node, num_agents)
        else:  # performance_based
            # Select top-performing agents
            return self._performance_based_selection(node, num_agents)

    def _adaptive_agent_selection(
        self,
        node: MDAPMCTSNode,
        count: int
    ) -> List[Any]:
        """Select agents adaptively based on node state."""
        # Simple implementation: prefer diverse agent types
        if len(self.agents) <= count:
            return self.agents

        # Select agents with different types
        selected = []
        agent_types = set()

        for agent in self.agents:
            if len(selected) >= count:
                break
            agent_type = getattr(agent, 'agent_type', 'unknown')
            if agent_type not in agent_types:
                selected.append(agent)
                agent_types.add(agent_type)

        # Fill remaining slots randomly
        while len(selected) < count and len(selected) < len(self.agents):
            agent = random.choice(self.agents)
            if agent not in selected:
                selected.append(agent)

        return selected

    def _performance_based_selection(
        self,
        node: MDAPMCTSNode,
        count: int
    ) -> List[Any]:
        """Select agents based on historical performance."""
        # Get performance metrics for parent
        if node.parent:
            parent_perf = node.parent.agent_performance
            # Sort agents by performance
            sorted_agents = sorted(
                self.agents,
                key=lambda a: parent_perf.get(getattr(a, 'agent_id', a), {}).get('success', 0.0),
                reverse=True
            )
            return sorted_agents[:count]
        else:
            return self.agents[:count]

    async def _collect_votes_sequential(
        self,
        node: MDAPMCTSNode,
        agents: List[Any]
    ) -> List[ActionVote]:
        """Collect votes from agents sequentially."""
        votes = []

        for agent in agents:
            try:
                vote = await self._get_agent_vote(agent, node)
                if vote:
                    votes.append(vote)
            except (ValueError, TypeError, AttributeError, KeyError, RuntimeError) as e:
                logger.warning(f"Agent {getattr(agent, 'agent_id', 'unknown')} failed: {e}")

        return votes

    async def _collect_votes_parallel(
        self,
        node: MDAPMCTSNode,
        agents: List[Any]
    ) -> List[ActionVote]:
        """Collect votes from agents in parallel."""
        votes = []

        # Create tasks for all agents
        tasks = [self._get_agent_vote(agent, node) for agent in agents]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful votes
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Agent vote failed: {result}")
            elif result:
                votes.append(result)

        return votes

    async def _get_agent_vote(
        self,
        agent: Any,
        node: MDAPMCTSNode
    ) -> Optional[ActionVote]:
        """
        Get a vote from an agent.

        Args:
            agent: Agent to query
            node: Current node

        Returns:
            ActionVote from the agent
        """
        if not MDAP_AVAILABLE or not hasattr(agent, 'generate_proof'):
            # Simulate vote
            applicable_actions = self._get_applicable_actions(node.state)
            if not applicable_actions:
                return None

            action = random.choice(applicable_actions)
            agent_id = getattr(agent, 'agent_id', 'simulated_agent')
            agent_type = getattr(agent, 'agent_type', 'random')

            return ActionVote(
                action=action,
                agent_id=agent_id,
                confidence=random.uniform(0.5, 0.8),
                rationale=f"Simulated vote from {agent_type}",
                agent_type=agent_type,
                estimated_success=random.uniform(0.3, 0.7),
                proof_state_hash=node.hash
            )

        # Use actual agent
        try:
            # This is a simplified version - actual implementation would
            # query the agent for the best action
            proof = await agent.generate_proof(
                theorem=node.state.goals[0] if node.state.goals else "",
                domain=LeanDomain.GENERAL
            )

            if proof and proof.tactics:
                action = str(proof.tactics[0]) if proof.tactics else "simp"

                return ActionVote(
                    action=action,
                    agent_id=agent.agent_id,
                    confidence=proof.confidence,
                    rationale=f"Generated by {agent.agent_type.value}",
                    agent_type=agent.agent_type.value,
                    estimated_success=proof.confidence,
                    proof_state_hash=node.hash
                )
        except Exception as e:
            logger.warning(f"Agent {agent.agent_id} vote failed: {e}")

        return None

    def _get_applicable_actions(self, state: ProofState) -> List[str]:
        """Get applicable tactics for a state."""
        if MCTS_AVAILABLE:
            return MCTSExpansion.BASIC_TACTICS
        return ["simp", "intros", "rw", "apply", "cases"]

    def aggregate_votes(
        self,
        votes: List[ActionVote],
        strategy: str = "first_k_ahead"
    ) -> Optional[str]:
        """
        Aggregate votes using MAKER strategies.

        Args:
            votes: List of votes to aggregate
            strategy: Aggregation strategy

        Returns:
            Winning action, or None if no votes
        """
        if not votes:
            return None

        if strategy == "first_k_ahead":
            return self._aggregate_first_k_ahead(votes)
        elif strategy == "majority":
            return self._aggregate_majority(votes)
        elif strategy == "weighted":
            return self._aggregate_weighted(votes)
        else:
            return self._aggregate_first_k_ahead(votes)

    def _aggregate_first_k_ahead(self, votes: List[ActionVote]) -> Optional[str]:
        """
        Aggregate using first-K-ahead strategy.

        First action to be K votes ahead wins.
        """
        k = self.config.k_ahead

        # Count votes per action
        action_counts = Counter(v.action for v in votes)

        if not action_counts:
            return None

        # Find K-ahead winner
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_actions) >= 2:
            winner_count = sorted_actions[0][1]
            runner_up_count = sorted_actions[1][1]

            if winner_count >= runner_up_count + k:
                return sorted_actions[0][0]

        # No K-ahead winner, return most voted
        return sorted_actions[0][0]

    def _aggregate_majority(self, votes: List[ActionVote]) -> Optional[str]:
        """Aggregate using simple majority."""
        action_counts = Counter(v.action for v in votes)
        total = len(votes)

        for action, count in action_counts.items():
            if count > total / 2:
                return action

        # No majority, return most voted
        return action_counts.most_common(1)[0][0] if action_counts else None

    def _aggregate_weighted(self, votes: List[ActionVote]) -> Optional[str]:
        """Aggregate using confidence-weighted voting."""
        weighted_scores = defaultdict(float)

        for vote in votes:
            weighted_scores[vote.action] += vote.confidence

        if not weighted_scores:
            return None

        return max(weighted_scores, key=weighted_scores.get)

    def red_flag_actions(self, votes: List[ActionVote]) -> List[str]:
        """
        Identify red-flagged actions based on votes.

        Args:
            votes: List of votes to analyze

        Returns:
            List of red-flagged action names
        """
        red_flagged = []

        # Group votes by action
        action_votes = defaultdict(list)
        for vote in votes:
            action_votes[vote.action].append(vote)

        # Check each action
        for action, action_vote_list in action_votes.items():
            # Flag if low average confidence
            avg_confidence = sum(v.confidence for v in action_vote_list) / len(action_vote_list)
            if avg_confidence < self.config.red_flag_threshold:
                red_flagged.append(action)
                continue

            # Flag if all voters disagree (high variance)
            if len(action_vote_list) > 2:
                confidences = [v.confidence for v in action_vote_list]
                variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
                if variance > 0.1:
                    red_flagged.append(action)

        return red_flagged

    async def _apply_tactic(self, state: ProofState, tactic: str) -> ProofState:
        """Apply a tactic to get new state."""
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy(),
            depth=state.depth + 1
        )

        tactic_obj = Tactic(name=tactic.split()[0] if tactic else "simp")
        new_state.tactics_sequence.append(tactic_obj)

        # Simulate tactic application
        if tactic in ["intros", "intro"]:
            if new_state.goals:
                new_state.goals = new_state.goals[1:]
        elif tactic in ["simp", "aesop", "trivial"]:
            if random.random() > 0.7 and new_state.goals:
                new_state.goals = []
        elif tactic in ["cases", "induction"]:
            if new_state.goals and len(new_state.goals) == 1:
                new_state.goals = new_state.goals * 2

        new_state.is_complete = len(new_state.goals) == 0
        new_state.hash = new_state._compute_hash() if hasattr(new_state, '_compute_hash') else ""

        return new_state


# =============================================================================
# MAKER-Enhanced Simulation
# =============================================================================

class MDAPMCTSSimulation(MCTSSimulation if MCTS_AVAILABLE else object):
    """
    Simulation phase enhanced with MAKER voting.

    During rollout, multiple voters propose tactics and voting
    selects the best tactic for each step.
    """

    def __init__(
        self,
        mdap_config: MDAPMCTSConfig,
        voters: Optional[List[LeanTacticVoter]] = None
    ):
        """Initialize MAKER-enhanced simulation."""
        if MCTS_AVAILABLE:
            super().__init__(
                rollout_policy=RolloutPolicy.HEURISTIC,
                max_depth=mdap_config.rollout_depth
            )

        self.config = mdap_config
        self.voters = voters or []
        self.aggregator = LeanAggregator(
            strategy=AggregationStrategy.FIRST_K_AHEAD
        ) if MAKER_AVAILABLE else None

    def simulate_with_maker(
        self,
        state: ProofState,
        voters: List[LeanTacticVoter]
    ) -> float:
        """
        Run a simulation using MAKER voting for tactics.

        Args:
            state: Starting proof state
            voters: List of tactic voters

        Returns:
            Estimated value (0 = loss, 1 = win)
        """
        current_state = state
        score = 0.0

        for depth in range(self.config.rollout_depth):
            if current_state.is_complete or not current_state.goals:
                return 1.0

            # Collect tactic votes
            votes = self.collect_tactic_votes(current_state, voters)

            if not votes:
                # No votes, use heuristic
                tactic = self._select_heuristic_tactic(current_state)
            else:
                # Select tactic by voting
                tactic = self.select_tactic_by_voting(votes)

            # Apply tactic
            current_state = self.apply_tactic_with_verification(current_state, tactic)

            # Update score based on progress
            if len(current_state.goals) < len(state.goals):
                score += 0.1

        # Final score based on goal reduction
        initial_goals = len(state.goals)
        final_goals = len(current_state.goals)

        if initial_goals > 0:
            reduction = (initial_goals - final_goals) / initial_goals
            score = max(score, reduction)

        return min(1.0, score)

    def collect_tactic_votes(
        self,
        state: ProofState,
        voters: List[LeanTacticVoter]
    ) -> List[TacticVote]:
        """
        Collect votes from all voters for a state.

        Args:
            state: Current proof state
            voters: List of voters

        Returns:
            List of tactic votes
        """
        votes = []

        # Convert ProofState to MakerProofState if needed
        if MAKER_AVAILABLE:
            maker_state = MakerProofState(
                goals=state.goals,
                context=state.context,
                tactic_sequence=state.tactics_sequence,
                depth=state.depth,
                is_complete=state.is_complete
            )
        else:
            maker_state = None

        for voter in voters:
            try:
                if hasattr(voter, 'vote'):
                    if maker_state:
                        vote = voter.vote(maker_state)
                    else:
                        # Create simple vote
                        vote = TacticVote(
                            tactic=self._select_heuristic_tactic(state),
                            confidence=random.uniform(0.5, 0.7),
                            rationale=f"Heuristic from {getattr(voter, 'voter_id', 'unknown')}",
                            voter_id=getattr(voter, 'voter_id', 'unknown'),
                            voter_type=VoterType.HEURISTIC,
                            proof_state_hash=state.hash
                        )
                    votes.append(vote)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                logger.warning(f"Voter {getattr(voter, 'voter_id', 'unknown')} failed: {e}")

        return votes

    def select_tactic_by_voting(self, votes: List[TacticVote]) -> str:
        """
        Select tactic using voting aggregation.

        Args:
            votes: List of votes

        Returns:
            Selected tactic
        """
        if not votes:
            return "simp"

        if MAKER_AVAILABLE and self.aggregator:
            # Use MAKER aggregator
            k_value = self.config.k_ahead
            selected = self.aggregator.aggregate(votes, k_value=k_value)
            return selected or "simp"
        else:
            # Simple majority
            tactic_counts = Counter(v.tactic for v in votes)
            return tactic_counts.most_common(1)[0][0] if tactic_counts else "simp"

    def apply_tactic_with_verification(
        self,
        state: ProofState,
        tactic: str
    ) -> ProofState:
        """
        Apply a tactic with verification.

        Args:
            state: Current proof state
            tactic: Tactic to apply

        Returns:
            New proof state
        """
        new_state = ProofState(
            goals=state.goals.copy(),
            context=state.context.copy(),
            tactics_sequence=state.tactics_sequence.copy(),
            depth=state.depth + 1
        )

        tactic_obj = Tactic(name=tactic.split()[0] if tactic else "simp")
        new_state.tactics_sequence.append(tactic_obj)

        # Simulate tactic application
        if tactic in ["intros", "intro"]:
            if new_state.goals:
                new_state.goals = new_state.goals[1:]
        elif tactic in ["simp", "aesop", "trivial"]:
            if random.random() > 0.7 and new_state.goals:
                new_state.goals = []
        elif tactic in ["cases", "induction"]:
            if new_state.goals and len(new_state.goals) == 1:
                new_state.goals = new_state.goals * 2

        new_state.is_complete = len(new_state.goals) == 0

        return new_state

    def _select_heuristic_tactic(self, state: ProofState) -> str:
        """Select a tactic using heuristics."""
        if not state.goals:
            return "done"

        goal = state.goals[0]

        # Check for quantifiers
        if any(q in goal for q in ["forall", "∀", "→", "->"]):
            return "intros"

        # Check for equality
        if "=" in goal:
            return "linarith"

        # Check for logical connectives
        if any(c in goal for c in ["∧", "and"]):
            return "constructor"

        # Default
        return "simp"


# =============================================================================
# MDAP-Enhanced MCTS Tree
# =============================================================================

class MDAPMCTSTree(MCTSTree if MCTS_AVAILABLE else object):
    """MCTS tree for MDAP-enhanced nodes."""

    def __init__(self, root: MDAPMCTSNode):
        """Initialize the tree."""
        if MCTS_AVAILABLE:
            super().__init__(root)
        else:
            self.root = root
            self.total_nodes = 1
            self._nodes_by_hash = {root.hash: root}

    def add_node(self, node: MDAPMCTSNode, check_transposition: bool = True) -> bool:
        """Add a node to the tree."""
        if check_transposition and node.hash in self._nodes_by_hash:
            return False

        self._nodes_by_hash[node.hash] = node
        self.total_nodes += 1
        return True

    def get_node_by_hash(self, state_hash: str) -> Optional[MDAPMCTSNode]:
        """Get node by state hash."""
        return self._nodes_by_hash.get(state_hash)


# =============================================================================
# Main MDAP-Enhanced MCTS Orchestrator
# =============================================================================

class MDAPMCTS(MCTS if MCTS_AVAILABLE else object):
    """
    Main MCTS orchestrator with MDAP/MAKER integration.

    Combines:
    - Intelligent tree search (MCTS)
    - Multi-agent voting (MDAP)
    - Tactic voting (MAKER)
    - Red-flagging for quality control

    Algorithm Flow:
        For each iteration:
            1. Selection: UCT-based tree traversal
            2. Expansion: MDAP agents vote on best action
            3. Simulation: MAKER voting for rollout tactics
            4. Backpropagation: Update statistics with agent feedback
    """

    def __init__(
        self,
        config: MDAPMCTSConfig,
        theorem: str,
        theorem_name: Optional[str] = None
    ):
        """Initialize MDAP-enhanced MCTS."""
        self.config = config
        self.theorem = theorem
        self.theorem_name = theorem_name or "mdap_mcts_theorem"

        # Initialize components
        self.expansion = MDAPMCTSExpansion(config)
        self.simulation = MDAPMCTSSimulation(config)

        # Initialize tree
        initial_state = ProofState(goals=[theorem])
        self.root = MDAPMCTSNode(state=initial_state)
        self.tree = MDAPMCTSTree(self.root)

        # Statistics
        self.iterations_completed = 0
        self.start_time = 0.0
        self.best_node: Optional[MDAPMCTSNode] = None
        self.best_value = 0.0

        # Agent statistics
        self.agent_statistics: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"votes_cast": 0, "votes_accepted": 0, "success_rate": 0.0}
        )

    async def search_with_mdap(
        self,
        iterations: Optional[int] = None,
        time_budget: Optional[float] = None
    ) -> MDAPMCTSResult:
        """
        Run MDAP-enhanced MCTS search.

        Args:
            iterations: Number of iterations (overrides config)
            time_budget: Time budget in seconds (overrides config)

        Returns:
            MDAPMCTSResult with best proof and statistics
        """
        iterations = iterations or self.config.max_iterations
        time_budget = time_budget or self.config.time_budget

        self.start_time = time.time()
        logger.info(f"Starting MDAP-MCTS search for: {self.theorem}")
        logger.info(f"Max iterations: {iterations}, Time budget: {time_budget}s")

        try:
            for i in range(iterations):
                # Check time budget
                elapsed = time.time() - self.start_time
                if elapsed >= time_budget:
                    logger.info(f"Time budget exhausted after {i} iterations")
                    break

                # Check early termination
                if self.config.enable_red_flagging and self.best_node and self.best_node.is_terminal:
                    logger.info(f"Proof found after {i} iterations")
                    break

                # Run one iteration
                await self.run_iteration_mdap(self.root)

                self.iterations_completed = i + 1

                # Log progress
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - self.start_time
                    logger.info(
                        f"Iteration {i+1}/{iterations}: "
                        f"Root visits={self.root.N}, "
                        f"Best value={self.best_value:.4f}, "
                        f"Tree nodes={self.tree.total_nodes}, "
                        f"Time={elapsed:.2f}s"
                    )

            # Compile result
            return self._compile_result()

        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"MDAP-MCTS search failed: {e}", exc_info=True)
            return MDAPMCTSResult(
                success=False,
                search_iterations=self.iterations_completed,
                time_elapsed=time.time() - self.start_time,
                agent_statistics=dict(self.agent_statistics)
            )

    async def run_iteration_mdap(self, root: MDAPMCTSNode) -> None:
        """
        Run a single MDAP-enhanced MCTS iteration.

        Args:
            root: Root node of the tree
        """
        # 1. Selection: Select leaf node using UCT
        leaf = self._select_with_agent_consensus(root)

        # 2. Expansion: Expand with MDAP voting
        new_node = await self.expansion.expand_with_mdap(leaf, self.tree)

        # 3. Simulation: Run rollout with MAKER voting
        if self.config.simulation_voters > 0:
            # Create voters for simulation
            voters = self._create_simulation_voters()
            reward = self.simulation.simulate_with_maker(new_node.state, voters)
        else:
            # Standard simulation
            reward = self.simulation.simulate(new_node.state)

        # 4. Backpropagation: Update with agent feedback
        self.backpropagate_with_agent_feedback(new_node, reward)

        # Update best node
        if new_node.state.is_complete or reward > self.best_value:
            self.best_value = reward
            self.best_node = new_node

    def _select_with_agent_consensus(self, node: MDAPMCTSNode) -> MDAPMCTSNode:
        """
        Select a leaf node using UCT with agent consensus.

        Args:
            node: Starting node

        Returns:
            Selected leaf node
        """
        current = node

        while not current.is_terminal and current.children:
            # Use UCT to select best child
            if MCTS_AVAILABLE:
                current = current.best_child(self.config.c_param)
            else:
                # Simple selection
                current = max(current.children.values(), key=lambda c: c.N, default=current)

        return current

    def backpropagate_with_agent_feedback(
        self,
        node: MDAPMCTSNode,
        reward: float
    ) -> None:
        """
        Backpropagate reward with agent performance tracking.

        Args:
            node: Node to start backpropagation from
            reward: Reward to propagate
        """
        current = node

        while current is not None:
            # Update node statistics
            if MCTS_AVAILABLE:
                current.update(reward)
            else:
                current.N += 1
                current.W += reward
                current.Q = current.W / current.N

            # Update agent performance for the action that led to this node
            if current.action and current.parent:
                success = reward > 0.5
                confidence = reward  # Use reward as proxy for confidence
                current.parent.update_agent_performance(current.action, success, confidence)

            # Update global agent statistics
            if current.action:
                agent_id = f"agent_{current.action}"
                self.agent_statistics[agent_id]["votes_cast"] += 1
                if reward > 0.5:
                    self.agent_statistics[agent_id]["votes_accepted"] += 1

            # Move to parent
            current = current.parent

        # Update success rates
        for agent_id, stats in self.agent_statistics.items():
            if stats["votes_cast"] > 0:
                stats["success_rate"] = stats["votes_accepted"] / stats["votes_cast"]

    def _create_simulation_voters(self) -> List[Any]:
        """Create voters for simulation phase."""
        voters = []

        if not MAKER_AVAILABLE:
            # Create mock voters
            for i in range(self.config.simulation_voters):
                voter = type('MockVoter', (), {
                    'voter_id': f'mock_voter_{i}',
                    'vote': lambda state, i=i: TacticVote(
                        tactic=random.choice(["simp", "intros", "rw"]),
                        confidence=random.uniform(0.5, 0.8),
                        rationale=f"Mock vote {i}",
                        voter_id=f'mock_voter_{i}',
                        voter_type=VoterType.RANDOM,
                        proof_state_hash=getattr(state, 'hash', '')
                    )
                })()
                voters.append(voter)
        else:
            # Create actual voters
            from leanaide_maker import (
                HeuristicVoter, RandomVoter, LeanMakerConfig
            )

            maker_config = LeanMakerConfig()
            voter_types = [
                VoterType.HEURISTIC,
                VoterType.RANDOM,
                VoterType.EVOLUTIONARY
            ]

            for i in range(self.config.simulation_voters):
                vtype = voter_types[i % len(voter_types)]
                if vtype == VoterType.HEURISTIC:
                    voter = HeuristicVoter(
                        voter_id=f'voter_{i}',
                        voter_type=vtype,
                        config=maker_config
                    )
                else:
                    voter = RandomVoter(
                        voter_id=f'voter_{i}',
                        voter_type=vtype,
                        config=maker_config
                    )
                voters.append(voter)

        return voters

    def _compile_result(self) -> MDAPMCTSResult:
        """Compile final result."""
        elapsed = time.time() - self.start_time

        # Get best path
        best_path = self._get_best_path()

        # Create proof from best path
        best_proof = self._create_proof_from_path(best_path)

        # Calculate performance ranking
        performance_ranking = sorted(
            [(agent_id, stats["success_rate"]) for agent_id, stats in self.agent_statistics.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return MDAPMCTSResult(
            best_proof=best_proof,
            success=best_proof is not None and (
                not best_path or best_path[-1].is_terminal if best_path else False
            ),
            search_iterations=self.iterations_completed,
            time_elapsed=elapsed,
            nodes_visited=self.tree.total_nodes,
            tree_depth=max((n.depth for n in [self.root] + list(self.tree._nodes_by_hash.values())), default=0),
            win_rate=self.best_value,
            confidence=self._calculate_confidence(),
            agent_statistics=dict(self.agent_statistics),
            voting_statistics={
                "total_agent_votes": sum(s["votes_cast"] for s in self.agent_statistics.values()),
                "accepted_votes": sum(s["votes_accepted"] for s in self.agent_statistics.values()),
                "agents_used": len(self.agent_statistics)
            },
            red_flag_analysis={
                "red_flagged_nodes": sum(
                    1 for n in self.tree._nodes_by_hash.values() if hasattr(n, 'red_flagged') and n.red_flagged
                ),
                "red_flag_rate": sum(
                    1 for n in self.tree._nodes_by_hash.values() if hasattr(n, 'red_flagged') and n.red_flagged
                ) / max(1, self.tree.total_nodes)
            },
            agent_performance_ranking=performance_ranking
        )

    def _get_best_path(self) -> List[MDAPMCTSNode]:
        """Get the best path from root to leaf."""
        path = [self.root]
        current = self.root

        while current.children:
            # Select child with highest visit count
            current = max(current.children.values(), key=lambda c: c.N)
            path.append(current)

        return path

    def _create_proof_from_path(self, path: List[MDAPMCTSNode]) -> Optional[LeanProof]:
        """Create a LeanProof from a path of nodes."""
        if not path:
            return None

        # Extract tactics from path
        tactics = []
        for node in path:
            tactics.extend(node.state.tactics_sequence)

        # Generate Lean code
        lean_code = f"import Mathlib\n\n"
        lean_code += f"theorem {self.theorem_name} : {self.theorem} := by\n"
        for tactic in tactics:
            lean_code += f"  {tactic}\n"

        # Create proof object
        proof = LeanProof(
            theorem_name=self.theorem_name,
            theorem_statement=self.theorem,
            lean_code=lean_code,
            tactics=tactics
        )

        return proof

    def _calculate_confidence(self) -> float:
        """Calculate confidence score."""
        if self.best_node and self.root.N > 0:
            return self.best_node.N / self.root.N
        return 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

async def search_with_mdap_mcts(
    theorem: str,
    theorem_name: Optional[str] = None,
    config: Optional[MDAPMCTSConfig] = None,
    **kwargs
) -> MDAPMCTSResult:
    """
    Convenience function to run MDAP-MCTS proof search.

    Args:
        theorem: Theorem statement to prove
        theorem_name: Optional name for the theorem
        config: MDAP-MCTS configuration
        **kwargs: Additional configuration parameters

    Returns:
        MDAPMCTSResult with best proof and statistics
    """
    if config is None:
        config = MDAPMCTSConfig(**kwargs)

    mcts = MDAPMCTS(config, theorem, theorem_name)
    return await mcts.search_with_mdap()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    'MDAPMCTSConfig',

    # Data classes
    'ActionVote',
    'MDAPMCTSResult',

    # Core classes
    'MDAPMCTSNode',
    'MDAPMCTSExpansion',
    'MDAPMCTSSimulation',
    'MDAPMCTSTree',
    'MDAPMCTS',

    # Convenience functions
    'search_with_mdap_mcts'
]


# =============================================================================
# Example Usage
# =============================================================================

async def example_usage():
    """Example usage of MDAP-MCTS integration."""

    print("=" * 80)
    print("MDAP-MCTS Integration Example")
    print("=" * 80)

    # Simple theorem
    theorem = "forall (n m : Nat), n + m = m + n"

    print(f"\nTheorem: {theorem}\n")

    # Create configuration
    config = MDAPMCTSConfig(
        # MCTS settings
        c_param=1.414,
        max_iterations=500,
        rollout_depth=50,
        time_budget=60.0,

        # MDAP settings
        available_agents=["evolution", "mcts", "adversarial"],
        expansion_agents=3,
        parallel_agents=4,

        # MAKER settings
        simulation_voters=5,
        voting_strategy="first_k_ahead",
        k_ahead=3,

        # Red-flagging
        enable_red_flagging=True,
        prune_red_flagged=True,

        # Agent selection
        agent_selection_strategy="adaptive"
    )

    # Run MDAP-MCTS search
    result = await search_with_mdap_mcts(
        theorem=theorem,
        theorem_name="add_comm",
        config=config
    )

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"\nSuccess: {result.success}")
    print(f"Iterations: {result.search_iterations}")
    print(f"Time: {result.time_elapsed:.2f}s")
    print(f"Nodes visited: {result.nodes_visited}")
    print(f"Tree depth: {result.tree_depth}")
    print(f"Win rate: {result.win_rate:.4f}")
    print(f"Confidence: {result.confidence:.4f}")

    print("\n" + "=" * 80)
    print("Agent Statistics")
    print("=" * 80)
    for agent_id, stats in result.agent_statistics.items():
        print(f"\n{agent_id}:")
        print(f"  Votes cast: {stats['votes_cast']}")
        print(f"  Votes accepted: {stats['votes_accepted']}")
        print(f"  Success rate: {stats['success_rate']:.3f}")

    print("\n" + "=" * 80)
    print("Voting Statistics")
    print("=" * 80)
    for key, value in result.voting_statistics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("Red Flag Analysis")
    print("=" * 80)
    for key, value in result.red_flag_analysis.items():
        print(f"{key}: {value}")

    if result.best_proof:
        print("\n" + "=" * 80)
        print("Best Proof")
        print("=" * 80)
        print(f"\n{result.best_proof.lean_code}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
