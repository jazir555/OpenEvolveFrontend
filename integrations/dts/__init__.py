"""DTS (Dialogue Tree Search) Integration.

LLM-powered tree search engine for multi-turn conversation optimization.
Explores conversation strategies in parallel, simulates diverse user reactions,
scores trajectories with multi-judge consensus, and prunes underperformers.

Core Algorithm:
    For each round:
        1. Generate N diverse conversation strategies
        2. For each strategy, simulate K user intent variants
        3. Roll out multi-turn conversations for each branch
        4. Score all trajectories with 3 independent judges
        5. Prune branches below threshold (median vote)
        6. Backpropagate scores up the tree
        7. Repeat with surviving branches

Example Usage:
    >>> from integrations.dts import DTSEngine, DTSConfig
    >>> 
    >>> # Create engine with default config
    >>> engine = DTSEngine()
    >>> 
    >>> # Or use builder for custom configuration
    >>> from integrations.dts import DTSEngineBuilder
    >>> engine = (DTSEngineBuilder()
    ...     .with_beam_width(10)
    ...     .with_max_depth(7)
    ...     .build())
    >>> 
    >>> # Run optimization
    >>> result = engine.optimize_conversation(
    ...     initial_context="Customer support scenario",
    ...     goal="Resolve customer's technical issue",
    ...     rounds=3
    ... )
    >>> 
    >>> # Access results
    >>> print(f"Best score: {result.best_score}")
    >>> for turn in result.get_conversation_script():
    ...     print(f"{turn['speaker']}: {turn['message']}")

Components:
    - ConversationNode: Individual conversation state
    - ConversationTree: Tree structure for exploration
    - StrategyGenerator: Generates diverse conversation strategies
    - UserSimulator: Simulates different user personas and intents
    - TrajectoryScorer: Multi-judge consensus scoring
    - BeamSearch: Parallel beam search algorithm
    - DTSEngine: Main orchestration engine
"""

__version__ = "1.0.0"
__author__ = "OpenEvolve"

# Core data structures
from .conversation_tree import (
    ConversationNode,
    ConversationTree,
    StrategyGenerator,
)

# User simulation
from .user_simulator import (
    UserPersona,
    UserSimulator,
    IntentModel,
    IntentType,
    PREDEFINED_PERSONAS,
)

# Scoring
from .trajectory_scorer import (
    ScoreResult,
    Judge,
    TrajectoryScorer,
    CriterionType,
)

# Search
from .beam_search import (
    BeamState,
    BeamSearch,
    ParallelBeamSearch,
)

# Main engine
from .dts_engine import (
    DTSConfig,
    DTSResult,
    DTSEngine,
    DTSEngineBuilder,
)

# Export all public APIs
__all__ = [
    # Core structures
    "ConversationNode",
    "ConversationTree",
    "StrategyGenerator",
    
    # User simulation
    "UserPersona",
    "UserSimulator",
    "IntentModel",
    "IntentType",
    "PREDEFINED_PERSONAS",
    
    # Scoring
    "ScoreResult",
    "Judge",
    "TrajectoryScorer",
    "CriterionType",
    
    # Search
    "BeamState",
    "BeamSearch",
    "ParallelBeamSearch",
    
    # Engine
    "DTSConfig",
    "DTSResult",
    "DTSEngine",
    "DTSEngineBuilder",
]
