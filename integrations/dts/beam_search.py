"""Parallel beam search for conversation optimization.

Maintains multiple conversation branches simultaneously for exploration.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .conversation_tree import ConversationTree, ConversationNode
from .trajectory_scorer import TrajectoryScorer, ScoreResult

logger = logging.getLogger(__name__)


@dataclass
class BeamState:
    """Search state for beam search.
    
    Attributes:
        active_branches: Currently active conversation branches
        round_number: Current search round
        budget_remaining: Remaining compute budget
        scores: Scores for active branches
        pruned_count: Number of branches pruned
        expanded_count: Number of branches expanded
    """
    active_branches: List[ConversationNode] = field(default_factory=list)
    round_number: int = 0
    budget_remaining: int = 1000  # Token/compute budget
    scores: Dict[str, float] = field(default_factory=dict)
    pruned_count: int = 0
    expanded_count: int = 0
    
    def get_best_branch(self) -> Optional[ConversationNode]:
        """Get the highest-scoring active branch."""
        if not self.active_branches:
            return None
        return max(self.active_branches, key=lambda n: self.scores.get(n.node_id, 0.0))
    
    def get_average_score(self) -> float:
        """Get average score of active branches."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "round_number": self.round_number,
            "active_branches": len(self.active_branches),
            "budget_remaining": self.budget_remaining,
            "average_score": self.get_average_score(),
            "pruned_count": self.pruned_count,
            "expanded_count": self.expanded_count,
        }


@dataclass
class BeamSearch:
    """Parallel beam search for conversation optimization.
    
    Maintains multiple conversation branches and prunes low-performing
    branches at each step to focus computation on promising paths.
    
    Attributes:
        beam_width: Number of branches to maintain
        max_depth: Maximum search depth
        scorer: Trajectory scorer for evaluation
        prune_threshold: Score threshold for pruning
    """
    beam_width: int = 5
    max_depth: int = 5
    scorer: Optional[TrajectoryScorer] = None
    prune_threshold: float = 5.0
    
    def __post_init__(self):
        """Initialize scorer if not provided."""
        if self.scorer is None:
            from .trajectory_scorer import TrajectoryScorer
            self.scorer = TrajectoryScorer()
    
    def search(
        self, 
        tree: ConversationTree,
        strategy_generator: Any,
        user_simulator: Any,
        initial_context: str = "",
        max_rounds: Optional[int] = None
    ) -> Tuple[ConversationTree, BeamState]:
        """Run beam search on conversation tree.
        
        Args:
            tree: Initial conversation tree with root
            strategy_generator: Generator for conversation strategies
            user_simulator: Simulator for user responses
            initial_context: Initial conversation context
            max_rounds: Override max depth
            
        Returns:
            Tuple of (final tree, final state)
        """
        max_rounds = max_rounds or self.max_depth
        
        # Initialize state
        state = BeamState(
            active_branches=[tree.root],
            budget_remaining=self.beam_width * max_rounds * 10,
        )
        
        logger.info(f"Starting beam search: width={self.beam_width}, max_depth={max_rounds}")
        
        # Run search rounds
        for round_num in range(max_rounds):
            state.round_number = round_num
            
            if not state.active_branches:
                logger.warning("No active branches remaining, stopping search")
                break
            
            # Expand each active branch
            new_branches = []
            for branch in state.active_branches:
                expanded = self.expand_node(
                    branch, 
                    tree,
                    strategy_generator, 
                    user_simulator,
                    initial_context
                )
                new_branches.extend(expanded)
                state.expanded_count += len(expanded)
            
            # Score all new branches
            for branch in new_branches:
                path = branch.get_path()
                score_result = self.scorer.score_trajectory(path)
                state.scores[branch.node_id] = score_result.overall_score
                branch.score = score_result.overall_score
            
            # Prune to maintain beam width
            pruned = self.prune_branches(new_branches, state.scores)
            state.pruned_count += len(new_branches) - len(pruned)
            state.active_branches = pruned
            
            logger.debug(f"Round {round_num}: {len(state.active_branches)} branches, "
                        f"avg_score={state.get_average_score():.2f}")
            
            # Check if we should stop early
            if state.get_average_score() > 9.0 and round_num >= 2:
                logger.info("High scores achieved, stopping early")
                break
        
        logger.info(f"Beam search complete: {state.expanded_count} expanded, "
                   f"{state.pruned_count} pruned")
        
        return tree, state
    
    def expand_node(
        self,
        node: ConversationNode,
        tree: ConversationTree,
        strategy_generator: Any,
        user_simulator: Any,
        context: str = ""
    ) -> List[ConversationNode]:
        """Expand a node by generating strategies and simulating responses.
        
        Args:
            node: Node to expand
            tree: Conversation tree
            strategy_generator: Strategy generator
            user_simulator: User simulator
            context: Conversation context
            
        Returns:
            List of new child nodes
        """
        if node.depth >= self.max_depth:
            return []
        
        new_nodes = []
        
        try:
            # Generate strategies for this node
            conversation_history = node.get_conversation_history()
            strategies = strategy_generator.generate_strategies(
                context=context,
                n=self.beam_width
            )
            
            # For each strategy, simulate user responses
            for strategy in strategies[:self.beam_width]:
                # System turn
                system_node = tree.add_node(
                    parent=node,
                    message=strategy,
                    speaker="system",
                    score=0.0,
                    metadata={
                        "type": "strategy",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                
                # Simulate user responses (multiple variants)
                try:
                    user_responses = user_simulator.generate_intent_variants(
                        strategy=strategy,
                        k=3  # Generate 3 user variants per strategy
                    )
                    
                    # For beam search, we typically keep one best user response
                    # But we score them all and keep the best
                    best_user_node = None
                    best_score = -1.0
                    
                    for user_response in user_responses[:1]:  # Keep first for beam efficiency
                        user_node = tree.add_node(
                            parent=system_node,
                            message=user_response,
                            speaker="user",
                            score=0.0,
                            metadata={
                                "type": "user_response",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )
                        
                        # Score this path
                        path = user_node.get_path()
                        score_result = self.scorer.score_trajectory(path)
                        user_node.score = score_result.overall_score
                        
                        if user_node.score > best_score:
                            best_score = user_node.score
                            best_user_node = user_node
                    
                    if best_user_node:
                        new_nodes.append(best_user_node)
                        
                except Exception as e:
                    logger.warning(f"User simulation failed: {e}")
                    # Add system node anyway
                    new_nodes.append(system_node)
                    
        except Exception as e:
            logger.warning(f"Node expansion failed: {e}")
        
        return new_nodes
    
    def prune_branches(
        self, 
        nodes: List[ConversationNode],
        scores: Dict[str, float]
    ) -> List[ConversationNode]:
        """Prune branches to maintain beam width.
        
        Args:
            nodes: Candidate nodes
            scores: Score dictionary
            
        Returns:
            Pruned list of nodes (at most beam_width)
        """
        if len(nodes) <= self.beam_width:
            return nodes
        
        # Filter by threshold
        above_threshold = [
            n for n in nodes 
            if scores.get(n.node_id, 0.0) >= self.prune_threshold
        ]
        
        # If too few above threshold, take top beam_width anyway
        candidates = above_threshold if len(above_threshold) >= self.beam_width else nodes
        
        # Sort by score and take top beam_width
        sorted_nodes = sorted(
            candidates,
            key=lambda n: scores.get(n.node_id, 0.0),
            reverse=True
        )
        
        return sorted_nodes[:self.beam_width]
    
    def get_best_path(self, tree: ConversationTree, state: BeamState) -> List[ConversationNode]:
        """Get the best path found during search.
        
        Args:
            tree: Conversation tree
            state: Final beam state
            
        Returns:
            Best conversation path
        """
        # First check active branches
        if state.active_branches:
            best_active = max(
                state.active_branches,
                key=lambda n: state.scores.get(n.node_id, 0.0)
            )
            return best_active.get_path()
        
        # Fall back to tree's best path
        return tree.get_best_path() or [tree.root]
    
    def explain_search(self, tree: ConversationTree, state: BeamState) -> str:
        """Generate human-readable explanation of search process.
        
        Args:
            tree: Conversation tree
            state: Beam state
            
        Returns:
            Explanation string
        """
        lines = [
            "Beam Search Summary",
            "=" * 50,
            f"Beam width: {self.beam_width}",
            f"Max depth: {self.max_depth}",
            f"Prune threshold: {self.prune_threshold}",
            "",
            "Search Statistics:",
            f"  Rounds completed: {state.round_number}",
            f"  Total nodes expanded: {state.expanded_count}",
            f"  Branches pruned: {state.pruned_count}",
            f"  Final active branches: {len(state.active_branches)}",
            "",
            "Score Statistics:",
            f"  Average score: {state.get_average_score():.2f}",
        ]
        
        if state.active_branches:
            best = state.get_best_branch()
            if best:
                lines.extend([
                    f"  Best branch score: {state.scores.get(best.node_id, 0.0):.2f}",
                    f"  Best branch depth: {best.depth}",
                ])
        
        tree_stats = tree.get_statistics()
        lines.extend([
            "",
            "Tree Statistics:",
            f"  Total nodes: {tree_stats.get('total_nodes', 0)}",
            f"  Total leaves: {tree_stats.get('total_leaves', 0)}",
            f"  Max depth: {tree_stats.get('max_depth', 0)}",
        ])
        
        return "\n".join(lines)


@dataclass
class ParallelBeamSearch(BeamSearch):
    """Parallel version of beam search.
    
    Expands multiple branches concurrently for faster search.
    """
    max_workers: int = 4
    
    def expand_node_parallel(
        self,
        nodes: List[ConversationNode],
        tree: ConversationTree,
        strategy_generator: Any,
        user_simulator: Any,
        context: str = ""
    ) -> List[ConversationNode]:
        """Expand multiple nodes in parallel.
        
        Args:
            nodes: Nodes to expand
            tree: Conversation tree
            strategy_generator: Strategy generator
            user_simulator: User simulator
            context: Conversation context
            
        Returns:
            List of new child nodes from all expansions
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_new_nodes = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit expansion tasks
            futures = {
                executor.submit(
                    self.expand_node,
                    node,
                    tree,
                    strategy_generator,
                    user_simulator,
                    context
                ): node 
                for node in nodes
            }
            
            # Collect results
            for future in as_completed(futures):
                try:
                    new_nodes = future.result()
                    all_new_nodes.extend(new_nodes)
                except Exception as e:
                    logger.warning(f"Parallel expansion failed: {e}")
        
        return all_new_nodes
    
    def search_parallel(
        self,
        tree: ConversationTree,
        strategy_generator: Any,
        user_simulator: Any,
        initial_context: str = "",
        max_rounds: Optional[int] = None
    ) -> Tuple[ConversationTree, BeamState]:
        """Run parallel beam search.
        
        Args:
            tree: Initial conversation tree
            strategy_generator: Strategy generator
            user_simulator: User simulator
            initial_context: Initial context
            max_rounds: Override max depth
            
        Returns:
            Tuple of (final tree, final state)
        """
        max_rounds = max_rounds or self.max_depth
        
        state = BeamState(
            active_branches=[tree.root],
            budget_remaining=self.beam_width * max_rounds * 10,
        )
        
        logger.info(f"Starting parallel beam search: width={self.beam_width}, "
                   f"max_depth={max_rounds}, workers={self.max_workers}")
        
        for round_num in range(max_rounds):
            state.round_number = round_num
            
            if not state.active_branches:
                break
            
            # Expand all branches in parallel
            new_branches = self.expand_node_parallel(
                state.active_branches,
                tree,
                strategy_generator,
                user_simulator,
                initial_context
            )
            
            state.expanded_count += len(new_branches)
            
            # Score all new branches
            for branch in new_branches:
                path = branch.get_path()
                score_result = self.scorer.score_trajectory(path)
                state.scores[branch.node_id] = score_result.overall_score
                branch.score = score_result.overall_score
            
            # Prune
            pruned = self.prune_branches(new_branches, state.scores)
            state.pruned_count += len(new_branches) - len(pruned)
            state.active_branches = pruned
        
        return tree, state


# Type hint imports
from typing import Any
