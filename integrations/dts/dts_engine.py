"""DTS Engine - Dialogue Tree Search orchestration.

Coordinates tree search, simulation, scoring, and pruning for
multi-turn conversation optimization.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field

from .conversation_tree import ConversationTree, ConversationNode, StrategyGenerator
from .user_simulator import UserSimulator, UserPersona, PREDEFINED_PERSONAS
from .trajectory_scorer import TrajectoryScorer, ScoreResult
from .beam_search import BeamSearch, BeamState

logger = logging.getLogger(__name__)


@dataclass
class DTSConfig:
    """Configuration for DTS engine.
    
    Attributes:
        beam_width: Number of conversation branches to maintain
        intent_variants: Number of user simulations per strategy
        judges: Number of independent scoring judges
        prune_threshold: Score threshold for pruning branches
        max_depth: Maximum conversation depth
        max_rounds: Maximum optimization rounds
        track_budget: Whether to track token/compute budget
        enable_parallel: Whether to use parallel execution
    """
    beam_width: int = 5
    intent_variants: int = 3
    judges: int = 3
    prune_threshold: float = 5.0
    max_depth: int = 5
    max_rounds: int = 3
    track_budget: bool = True
    enable_parallel: bool = False
    
    # Budget tracking
    max_tokens: int = 10000
    max_cost: float = 1.0  # USD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "beam_width": self.beam_width,
            "intent_variants": self.intent_variants,
            "judges": self.judges,
            "prune_threshold": self.prune_threshold,
            "max_depth": self.max_depth,
            "max_rounds": self.max_rounds,
            "enable_parallel": self.enable_parallel,
        }


@dataclass
class DTSResult:
    """Result from DTS optimization.
    
    Attributes:
        tree: Final conversation tree
        best_path: Highest-scoring conversation path
        best_score: Score of best path
        all_paths: All explored paths
        state: Final beam search state
        statistics: Search statistics
        metadata: Additional result data
    """
    tree: ConversationTree
    best_path: List[ConversationNode]
    best_score: float
    all_paths: List[List[ConversationNode]]
    state: BeamState
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_conversation_script(self) -> List[Dict[str, str]]:
        """Get best conversation as list of messages."""
        if not self.best_path:
            return []
        return [
            {"speaker": node.speaker, "message": node.message}
            for node in self.best_path
        ]
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "DTS Optimization Result",
            "=" * 50,
            f"Best score: {self.best_score:.2f}/10",
            f"Total paths explored: {len(self.all_paths)}",
            f"Best path length: {len(self.best_path)}",
            "",
            "Best Conversation:",
        ]
        
        for turn in self.get_conversation_script():
            speaker = turn["speaker"].upper()
            message = turn["message"][:60] + "..." if len(turn["message"]) > 60 else turn["message"]
            lines.append(f"  {speaker}: {message}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "best_score": self.best_score,
            "best_path": [
                {"speaker": n.speaker, "message": n.message, "score": n.score}
                for n in self.best_path
            ],
            "total_paths": len(self.all_paths),
            "statistics": self.statistics,
            "metadata": self.metadata,
        }


class DTSEngine:
    """Dialogue Tree Search Engine.
    
    Main orchestrator for conversation optimization. Coordinates:
    - Strategy generation
    - User simulation  
    - Multi-judge scoring
    - Beam search with pruning
    - Backpropagation
    
    Example:
        >>> engine = DTSEngine()
        >>> result = engine.optimize_conversation(
        ...     initial_context="Help user understand quantum computing",
        ...     goal="Explain quantum superposition",
        ...     rounds=3
        ... )
        >>> print(result.best_score)
    """
    
    def __init__(
        self,
        config: Optional[DTSConfig] = None,
        strategy_gen: Optional[StrategyGenerator] = None,
        user_sim: Optional[UserSimulator] = None,
        scorer: Optional[TrajectoryScorer] = None,
        beam_search: Optional[BeamSearch] = None,
        llm_client: Optional[Any] = None,
    ):
        """Initialize DTS engine.
        
        Args:
            config: DTS configuration
            strategy_gen: Strategy generator (created if None)
            user_sim: User simulator (created if None)
            scorer: Trajectory scorer (created if None)
            beam_search: Beam search instance (created if None)
            llm_client: LLM client for enhanced generation
        """
        self.config = config or DTSConfig()
        self.llm_client = llm_client
        
        # Initialize components
        self.strategy_gen = strategy_gen or StrategyGenerator(llm_client=llm_client)
        self.user_sim = user_sim or UserSimulator()
        self.scorer = scorer or TrajectoryScorer()
        self.beam_search = beam_search or BeamSearch(
            beam_width=self.config.beam_width,
            max_depth=self.config.max_depth,
            scorer=self.scorer,
            prune_threshold=self.config.prune_threshold,
        )
        
        # Tracking
        self._budget_used = 0
        self._tokens_used = 0
        self._start_time: Optional[datetime] = None
        
        logger.info(f"DTS Engine initialized: {self.config.to_dict()}")
    
    def optimize_conversation(
        self,
        initial_context: str,
        goal: str,
        rounds: Optional[int] = None,
        initial_message: Optional[str] = None,
    ) -> DTSResult:
        """Run full conversation optimization.
        
        Args:
            initial_context: Starting context for conversation
            goal: Target conversation goal
            rounds: Override max rounds from config
            initial_message: Optional initial user message
            
        Returns:
            DTSResult with optimization results
        """
        self._start_time = datetime.now(timezone.utc)
        rounds = rounds or self.config.max_rounds
        
        logger.info(f"Starting conversation optimization: goal='{goal}', rounds={rounds}")
        
        # Create initial tree
        root_message = initial_message or f"Goal: {goal}"
        root = ConversationNode(
            message=root_message,
            speaker="user" if initial_message else "system",
            depth=0,
            score=5.0,  # Neutral starting score
            metadata={
                "type": "root",
                "goal": goal,
                "timestamp": self._start_time.isoformat(),
            }
        )
        tree = ConversationTree(
            root=root,
            metadata={
                "goal": goal,
                "context": initial_context,
                "started_at": self._start_time.isoformat(),
            }
        )
        
        # Run beam search
        if self.config.enable_parallel:
            final_tree, state = self.beam_search.search_parallel(
                tree=tree,
                strategy_generator=self.strategy_gen,
                user_simulator=self.user_sim,
                initial_context=initial_context,
                max_rounds=rounds,
            )
        else:
            final_tree, state = self.beam_search.search(
                tree=tree,
                strategy_generator=self.strategy_gen,
                user_simulator=self.user_sim,
                initial_context=initial_context,
                max_rounds=rounds,
            )
        
        # Collect results
        best_path = self.beam_search.get_best_path(final_tree, state)
        all_paths = final_tree.get_branches()
        
        # Calculate statistics
        statistics = {
            **state.to_dict(),
            **final_tree.get_statistics(),
            "elapsed_time": self._get_elapsed_time(),
        }
        
        result = DTSResult(
            tree=final_tree,
            best_path=best_path or [root],
            best_score=state.scores.get(best_path[-1].node_id, 0.0) if best_path else 0.0,
            all_paths=all_paths,
            state=state,
            statistics=statistics,
            metadata={
                "config": self.config.to_dict(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        
        logger.info(f"Optimization complete: best_score={result.best_score:.2f}, "
                   f"paths={len(all_paths)}")
        
        return result
    
    def step(self, tree: ConversationTree, context: str = "") -> BeamState:
        """Execute a single optimization step.
        
        Args:
            tree: Current conversation tree
            context: Conversation context
            
        Returns:
            Updated beam state
        """
        # Get current leaves as active branches
        leaves = tree.get_leaves()
        
        state = BeamState(
            active_branches=leaves,
            round_number=0,
        )
        
        # Expand nodes
        new_branches = []
        for branch in state.active_branches:
            expanded = self.beam_search.expand_node(
                branch,
                tree,
                self.strategy_gen,
                self.user_sim,
                context
            )
            new_branches.extend(expanded)
        
        # Score and prune
        for branch in new_branches:
            path = branch.get_path()
            score_result = self.scorer.score_trajectory(path)
            state.scores[branch.node_id] = score_result.overall_score
            branch.score = score_result.overall_score
        
        state.active_branches = self.beam_search.prune_branches(
            new_branches, state.scores
        )
        
        return state
    
    def get_best_path(self, tree: ConversationTree) -> List[ConversationNode]:
        """Get the best path from a conversation tree.
        
        Args:
            tree: Conversation tree
            
        Returns:
            Best scoring path
        """
        return tree.get_best_path() or [tree.root]
    
    def explain_strategy(self, node: ConversationNode) -> str:
        """Explain the strategy for a conversation node.
        
        Args:
            node: Node to explain
            
        Returns:
            Explanation string
        """
        path = node.get_path()
        score_result = self.scorer.score_trajectory(path)
        
        lines = [
            f"Strategy Explanation (Node {node.node_id[:8]})",
            "=" * 50,
            f"Depth: {node.depth}",
            f"Score: {node.score:.2f}/10",
            f"Speaker: {node.speaker}",
            "",
            "Message:",
            f"  {node.message[:200]}..." if len(node.message) > 200 else f"  {node.message}",
            "",
            "Scoring Breakdown:",
        ]
        
        for criterion, score in score_result.criteria_scores.items():
            lines.append(f"  {criterion}: {score:.1f}")
        
        lines.extend([
            "",
            "Path Context:",
            f"  Total turns: {len(path)}",
        ])
        
        return "\n".join(lines)
    
    def add_persona(self, persona: UserPersona) -> None:
        """Add a user persona to the simulator."""
        self.user_sim.add_persona(persona)
    
    def set_personas(self, personas: List[UserPersona]) -> None:
        """Set the list of personas for simulation."""
        self.user_sim.personas = personas
    
    def get_budget_usage(self) -> Dict[str, Any]:
        """Get current budget usage statistics."""
        return {
            "tokens_used": self._tokens_used,
            "budget_used": self._budget_used,
            "max_tokens": self.config.max_tokens,
            "max_cost": self.config.max_cost,
            "token_percentage": (self._tokens_used / self.config.max_tokens * 100) if self.config.max_tokens > 0 else 0,
        }
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time since optimization started."""
        if self._start_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()
    
    def reset(self) -> None:
        """Reset engine state for new optimization."""
        self._budget_used = 0
        self._tokens_used = 0
        self._start_time = None
        logger.info("DTS Engine reset")


class DTSEngineBuilder:
    """Builder pattern for creating DTS engines.
    
    Provides a fluent interface for configuring the engine.
    
    Example:
        >>> engine = (DTSEngineBuilder()
        ...     .with_beam_width(10)
        ...     .with_max_depth(7)
        ...     .add_persona(custom_persona)
        ...     .build())
    """
    
    def __init__(self):
        """Initialize builder with defaults."""
        self.config = DTSConfig()
        self.strategy_gen = None
        self.user_sim = None
        self.scorer = None
        self.beam_search = None
        self.llm_client = None
        self.personas: List[UserPersona] = []
    
    def with_beam_width(self, width: int) -> 'DTSEngineBuilder':
        """Set beam width."""
        self.config.beam_width = width
        return self
    
    def with_max_depth(self, depth: int) -> 'DTSEngineBuilder':
        """Set max conversation depth."""
        self.config.max_depth = depth
        return self
    
    def with_max_rounds(self, rounds: int) -> 'DTSEngineBuilder':
        """Set max optimization rounds."""
        self.config.max_rounds = rounds
        return self
    
    def with_prune_threshold(self, threshold: float) -> 'DTSEngineBuilder':
        """Set pruning threshold."""
        self.config.prune_threshold = threshold
        return self
    
    def with_intent_variants(self, k: int) -> 'DTSEngineBuilder':
        """Set number of intent variants."""
        self.config.intent_variants = k
        return self
    
    def with_parallel(self, enabled: bool = True) -> 'DTSEngineBuilder':
        """Enable/disable parallel execution."""
        self.config.enable_parallel = enabled
        return self
    
    def with_llm_client(self, client: Any) -> 'DTSEngineBuilder':
        """Set LLM client."""
        self.llm_client = client
        return self
    
    def add_persona(self, persona: UserPersona) -> 'DTSEngineBuilder':
        """Add a user persona."""
        self.personas.append(persona)
        return self
    
    def with_strategy_generator(self, gen: StrategyGenerator) -> 'DTSEngineBuilder':
        """Set custom strategy generator."""
        self.strategy_gen = gen
        return self
    
    def with_scorer(self, scorer: TrajectoryScorer) -> 'DTSEngineBuilder':
        """Set custom trajectory scorer."""
        self.scorer = scorer
        return self
    
    def build(self) -> DTSEngine:
        """Build the DTS engine with configured settings."""
        engine = DTSEngine(
            config=self.config,
            strategy_gen=self.strategy_gen,
            user_sim=self.user_sim,
            scorer=self.scorer,
            beam_search=self.beam_search,
            llm_client=self.llm_client,
        )
        
        if self.personas:
            engine.set_personas(self.personas)
        
        return engine


# Type hint for Any
from typing import Any
