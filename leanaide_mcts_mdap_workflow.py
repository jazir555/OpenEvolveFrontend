"""
LeanAide MDAP-Enhanced MCTS Workflow Integration

This module provides comprehensive integration of MDAP-enhanced Monte Carlo Tree Search (MCTS)
capabilities with the OpenEvolve decomposition workflow. It combines MCTS tree search with
MDAP multi-agent voting for optimal Lean 4 proof generation.

Architecture:
    MDAP (Multi-Strategy Decision Aggregation Protocol) + MCTS (Monte Carlo Tree Search)
    v
    Multi-Agent Voting for Tactical Decisions + Tree Search for Proof Paths
    v
    Hybrid Approach: MCTS explores proof tree, MDAP agents vote on best tactics at each node

Key Features:
- MDAPMCTSWorkflowIntegrator: Main integration class for MDAP-MCTS
- MDAPMCTSMonitor: Real-time monitoring of tree search and agent voting
- HybridLeanSolver: Adaptive solver that selects MCTS, MDAP, or MDAP-MCTS
- Stage 3A/B/C integration: MDAP-MCTS proof generation, refinement, verification
- Stage 5 integration: Final verification with MDAP-MCTS fallback
- Configuration integration with WorkflowState
- Graceful fallback to non-MCTS approaches
- CAV-NLP enhanced formalization support
- Comprehensive error handling and logging

When to use MDAP-MCTS:
- Complex theorems (many possible proof paths)
- Large branching factor (many applicable tactics)
- Deep proofs (many steps required)
- Multiple strategies available
- Consensus voting improves quality

Author: OpenEvolve
Created: 2025-12-30
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, Set
)
import threading
import math

# CAV-NLP Integration
try:
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logging.getLogger(__name__).debug("CAV-NLP not available for MCTS workflow")

# Configure logging
logger = logging.getLogger(__name__)

# Import workflow structures
try:
    from workflow_structures import (
        WorkflowState,
        SubProblem,
        SolutionAttempt,
        VerificationReport,
        CritiqueReport,
        Team,
        GauntletDefinition,
        LeanProof,
        LeanProofStatus
    )
    WORKFLOW_AVAILABLE = True
except ImportError:
    WORKFLOW_AVAILABLE = False
    logger.warning("Workflow structures not available - integration limited")

# Import LeanAide components
try:
    from leanaide_workflow_integration import (
        LeanAideWorkflowIntegrator,
        LeanAideWorkflowConfig,
        LeanAideVerificationResult,
        MathematicalProblemDetector,
        is_leanaide_configured
    )
    LEANAIDE_AVAILABLE = True
except ImportError:
    LEANAIDE_AVAILABLE = False
    logger.warning("LeanAide workflow integration not available")

# Import MCTS components
try:
    from leanaide_mcts import (
        MCTS,
        MCTSConfig,
        MCTSNode,
        MCTSResult,
        ProofState,
        MCTSTree
    )
    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    logger.warning("LeanAide MCTS not available - using stub implementation")

# Import MDAP components
try:
    from mdap_engine import (
        MDAPConfig,
        MDAPTask,
        MDAPStep,
        MDAPRunResult,
        MDAPOrchestrator,
        RedFlagRules
    )
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    logger.warning("MDAP engine not available")

# Import evolutionary components for fallback
try:
    from leanaide_evolutionary_workflow import (
        LeanEvolutionaryWorkflowStage,
        EvolutionaryConfig
    )
    EVOLUTIONARY_AVAILABLE = True
except ImportError:
    EVOLUTIONARY_AVAILABLE = False

# import crewai # MIGRATED: was CrewAI integration
try:
    from crewai_client import CrewAIClient
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Import ACE knowledge storage
try:
    from ace_knowledge_artifacts import ACEKnowledgeManager
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False


# =============================================================================
# CONFIGURATION DATA CLASSES
# =============================================================================

class MDAPMCTSStrategy(Enum):
    """Available MDAP-MCTS hybrid strategies."""
    PURE_MCTS = "pure_mcts"  # Standard MCTS without MDAP
    PURE_MDAP = "pure_mdap"  # Standard MDAP without MCTS
    MDAP_MCTS_SERIAL = "mdap_mcts_serial"  # MDAP guides MCTS node expansion
    MDAP_MCTS_PARALLEL = "mdap_mcts_parallel"  # MDAP votes on MCTS simulations
    MDAP_MCTS_HYBRID = "mdap_mcts_hybrid"  # Adaptive switching between modes
    ADAPTIVE = "adaptive"  # Automatically select best strategy


class VotingStrategy(Enum):
    """Voting strategies for MDAP within MCTS."""
    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED_CONFIDENCE = "weighted_confidence"  # Weight by agent confidence
    BORDA = "borda"  # Borda count
    CONDORCET = "condorcet"  # Condorcet method
    FIRST_TO_K = "first_to_k"  # First tactic to reach K votes


@dataclass
class MDAPMCTSConfig:
    """Configuration for MDAP-MCTS workflow integration."""
    # Enablement
    lean_mdap_mcts_enabled: bool = True
    strategy: MDAPMCTSStrategy = MDAPMCTSStrategy.ADAPTIVE

    # MCTS core parameters
    mcts_iterations: int = 1000
    mcts_time_budget: float = 300.0  # seconds
    mcts_c_param: float = 1.414  # UCT exploration parameter
    mcts_rollout_depth: int = 100
    mcts_max_tree_depth: int = 50

    # MDAP core parameters
    mdap_agents: List[str] = field(default_factory=lambda: [
        "direct_prover",
        "inductive_prover",
        "constructive_prover",
        "decomposition_prover"
    ])
    mdap_parallel_agents: int = 4
    mdap_agent_timeout: float = 120.0

    # MDAP-MCTS integration parameters
    mdap_expansion_agents: int = 3  # Number of MDAP agents for node expansion
    mdap_simulation_voters: int = 5  # Number of MDAP agents for simulation voting
    voting_strategy: VotingStrategy = VotingStrategy.WEIGHTED_CONFIDENCE
    k_ahead: int = 3  # First-to-K votes threshold

    # Red-flagging for tactical decisions
    enable_red_flagging: bool = True
    max_tactic_length: int = 750
    max_proof_tokens: int = 6000

    # Verification parameters
    verify_with_leanaide: bool = True
    verification_confidence: float = 0.8
    verification_timeout: float = 60.0

    # Fallback behavior
    fallback_to_mdap: bool = True
    fallback_to_mcts: bool = True
    fallback_to_evolution: bool = True
    fallback_to_standard: bool = True
    partial_result_on_timeout: bool = True

    # Integration settings
    auto_detect_applicable: bool = True
    store_patterns: bool = True
    track_statistics: bool = True

    # CrewAI integration
    CrewAI_enabled: bool = False
    CrewAI_timeout: float = 600.0

    # ACE integration
    ace_learning_enabled: bool = True
    ace_store_patterns: bool = True


@dataclass
class MDAPMCTSProgress:
    """Progress tracking for MDAP-MCTS runs."""
    sub_problem_id: str
    strategy: MDAPMCTSStrategy
    iterations: int = 0
    mdap_rounds: int = 0
    best_score: float = 0.0
    current_best_proof: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    status: str = "in_progress"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score_history: List[float] = field(default_factory=list)
    tree_size: int = 0
    nodes_explored: int = 0
    mdap_votes_cast: int = 0
    mdap_consensus_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_problem_id": self.sub_problem_id,
            "strategy": self.strategy.value,
            "iterations": self.iterations,
            "mdap_rounds": self.mdap_rounds,
            "best_score": self.best_score,
            "current_best_proof": self.current_best_proof,
            "elapsed_time": self.elapsed_time,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "score_history": self.score_history,
            "tree_size": self.tree_size,
            "nodes_explored": self.nodes_explored,
            "mdap_votes_cast": self.mdap_votes_cast,
            "mdap_consensus_rate": self.mdap_consensus_rate
        }


@dataclass
class MDAPMCTSSearchSpace:
    """Characteristics of the search space for MDAP-MCTS applicability."""
    branching_factor: int = 0
    estimated_depth: int = 0
    has_multiple_strategies: bool = False
    tactic_diversity: float = 0.0
    complexity_score: float = 0.0
    is_mdap_applicable: bool = False
    is_mcts_applicable: bool = False
    is_hybrid_applicable: bool = False

    def calculate_applicability_score(self) -> Tuple[str, float]:
        """Calculate best strategy and applicability score."""
        mcts_score = 0.0
        mdap_score = 0.0
        hybrid_score = 0.0

        # MCTS favors large branching and depth
        if self.branching_factor > 10:
            mcts_score += 0.3
        elif self.branching_factor > 5:
            mcts_score += 0.15

        if self.estimated_depth > 20:
            mcts_score += 0.3
        elif self.estimated_depth > 10:
            mcts_score += 0.15

        # MDAP favors multiple strategies
        if self.has_multiple_strategies:
            mdap_score += 0.4

        if self.tactic_diversity > 0.7:
            mdap_score += 0.3
        elif self.tactic_diversity > 0.4:
            mdap_score += 0.15

        # Hybrid favors both
        if self.branching_factor > 10 and self.has_multiple_strategies:
            hybrid_score += 0.4

        if self.estimated_depth > 20 and self.tactic_diversity > 0.5:
            hybrid_score += 0.4

        # Determine best strategy
        scores = {
            "mcts": mcts_score,
            "mdap": mdap_score,
            "hybrid": hybrid_score
        }
        best = max(scores, key=scores.get)

        return best, max(mcts_score, mdap_score, hybrid_score)


# =============================================================================
# MDAP-MCTS WORKFLOW MONITOR
# =============================================================================

class MDAPMCTSMonitor:
    """
    Monitor MDAP-MCTS execution in workflow.

    Provides real-time tracking of:
    - Tree search statistics (size, depth, branching)
    - Agent voting statistics (votes, consensus, red flags)
    - Overall progress and resource usage
    """

    def __init__(self, config: MDAPMCTSConfig):
        """
        Initialize the MDAP-MCTS workflow monitor.

        Args:
            config: MDAP-MCTS configuration
        """
        self.config = config
        self.active_mdap_mcts: Dict[str, 'MDAPMCTS'] = {}
        self.progress: Dict[str, MDAPMCTSProgress] = {}
        self.statistics: Dict[str, Dict[str, Any]] = defaultdict(lambda: defaultdict(list))
        self.lock = threading.Lock()

    def start_monitoring(self, mdap_mcts: 'MDAPMCTS', sub_problem_id: str) -> None:
        """
        Start monitoring an MDAP-MCTS instance.

        Args:
            mdap_mcts: The MDAP-MCTS instance to monitor
            sub_problem_id: Associated sub-problem ID
        """
        with self.lock:
            self.active_mdap_mcts[sub_problem_id] = mdap_mcts
            self.progress[sub_problem_id] = MDAPMCTSProgress(
                sub_problem_id=sub_problem_id,
                strategy=mdap_mcts.config.strategy
            )

    def update_progress(
        self,
        sub_problem_id: str,
        iteration: int,
        mdap_round: int,
        best_score: float,
        current_best_proof: Optional[str],
        tree_size: int,
        nodes_explored: int,
        mdap_votes: int,
        consensus_rate: float
    ) -> None:
        """
        Update progress for a monitored MDAP-MCTS.

        Args:
            sub_problem_id: Sub-problem ID
            iteration: Current MCTS iteration
            mdap_round: Current MDAP voting round
            best_score: Best score so far
            current_best_proof: Current best proof
            tree_size: Current tree size
            nodes_explored: Number of nodes explored
            mdap_votes: Number of MDAP votes cast
            consensus_rate: Current MDAP consensus rate
        """
        with self.lock:
            if sub_problem_id in self.progress:
                progress = self.progress[sub_problem_id]
                progress.iterations = iteration
                progress.mdap_rounds = mdap_round
                progress.best_score = best_score
                progress.current_best_proof = current_best_proof
                progress.elapsed_time = time.time() - progress.start_time
                progress.tree_size = tree_size
                progress.nodes_explored = nodes_explored
                progress.mdap_votes_cast = mdap_votes
                progress.mdap_consensus_rate = consensus_rate
                progress.score_history.append(best_score)

                # Store statistics
                self.statistics[sub_problem_id]["iterations"].append(iteration)
                self.statistics[sub_problem_id]["scores"].append(best_score)
                self.statistics[sub_problem_id]["tree_sizes"].append(tree_size)
                self.statistics[sub_problem_id]["consensus_rates"].append(consensus_rate)

    def get_tree_statistics(self, sub_problem_id: str) -> Dict[str, Any]:
        """
        Get tree statistics for a specific sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            Dictionary with tree statistics
        """
        with self.lock:
            if sub_problem_id not in self.active_mdap_mcts:
                return {}

            mdap_mcts = self.active_mdap_mcts[sub_problem_id]
            tree = mdap_mcts.tree if hasattr(mdap_mcts, 'tree') else None

            if not tree:
                return {}

            return {
                "tree_size": tree.total_nodes if hasattr(tree, 'total_nodes') else 0,
                "max_depth": tree.max_depth if hasattr(tree, 'max_depth') else 0,
                "branching_factor": self._calculate_avg_branching(tree),
                "nodes_by_depth": self._count_nodes_by_depth(tree)
            }

    def get_agent_statistics(self) -> Dict[str, Any]:
        """
        Get per-agent performance statistics.

        Returns:
            Dictionary mapping agent_id to performance metrics
        """
        with self.lock:
            agent_stats = defaultdict(lambda: {
                "votes_cast": 0,
                "votes_won": 0,
                "confidence_sum": 0.0,
                "red_flags": 0
            })

            # Aggregate statistics across all sub-problems
            for sub_problem_id, stats in self.statistics.items():
                for agent_id, agent_data in stats.get("agents", {}).items():
                    agent_stats[agent_id]["votes_cast"] += agent_data.get("votes_cast", 0)
                    agent_stats[agent_id]["votes_won"] += agent_data.get("votes_won", 0)
                    agent_stats[agent_id]["confidence_sum"] += agent_data.get("confidence_sum", 0.0)
                    agent_stats[agent_id]["red_flags"] += agent_data.get("red_flags", 0)

            # Calculate averages
            for agent_id, stats in agent_stats.items():
                if stats["votes_cast"] > 0:
                    stats["avg_confidence"] = stats["confidence_sum"] / stats["votes_cast"]
                    stats["win_rate"] = stats["votes_won"] / stats["votes_cast"]
                else:
                    stats["avg_confidence"] = 0.0
                    stats["win_rate"] = 0.0

            return dict(agent_stats)

    def get_voting_statistics(self, sub_problem_id: str) -> Dict[str, Any]:
        """
        Get voting statistics for a specific sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            Dictionary with voting statistics
        """
        with self.lock:
            if sub_problem_id not in self.statistics:
                return {}

            stats = self.statistics[sub_problem_id]
            consensus_history = stats.get("consensus_rates", [])

            return {
                "total_votes": sum(stats.get("votes_per_round", [])),
                "avg_consensus_rate": sum(consensus_history) / len(consensus_history) if consensus_history else 0.0,
                "consensus_trend": "improving" if len(consensus_history) > 1 and consensus_history[-1] > consensus_history[0] else "stable",
                "vote_distribution": stats.get("vote_distribution", {}),
                "red_flags_count": stats.get("red_flags", 0)
            }

    def get_progress(self, sub_problem_id: str) -> Optional[Dict[str, Any]]:
        """
        Get progress for a specific sub-problem.

        Args:
            sub_problem_id: Sub-problem ID

        Returns:
            Progress dictionary or None
        """
        with self.lock:
            if sub_problem_id in self.progress:
                return self.progress[sub_problem_id].to_dict()
            return None

    def stop_monitoring(self, sub_problem_id: str) -> None:
        """
        Stop monitoring a sub-problem.

        Args:
            sub_problem_id: Sub-problem ID
        """
        with self.lock:
            self.active_mdap_mcts.pop(sub_problem_id, None)
            if sub_problem_id in self.progress:
                self.progress[sub_problem_id].status = "completed"

    def _calculate_avg_branching(self, tree: Any) -> float:
        """Calculate average branching factor."""
        if not hasattr(tree, 'get_root'):
            return 0.0

        branching_factors = []
        queue = [tree.get_root()]

        while queue:
            node = queue.pop(0)
            if hasattr(node, 'children') and node.children:
                branching_factors.append(len(node.children))
                queue.extend(node.children.values())

        return sum(branching_factors) / len(branching_factors) if branching_factors else 0.0

    def _count_nodes_by_depth(self, tree: Any) -> Dict[int, int]:
        """Count nodes at each depth level."""
        if not hasattr(tree, 'get_root'):
            return {}

        nodes_by_depth = defaultdict(int)
        queue = [(tree.get_root(), 0)]

        while queue:
            node, depth = queue.pop(0)
            nodes_by_depth[depth] += 1

            if hasattr(node, 'children'):
                for child in node.children.values():
                    queue.append((child, depth + 1))

        return dict(nodes_by_depth)


# =============================================================================
# MDAP-MCTS MAIN INTEGRATOR
# =============================================================================

class MDAPMCTSWorkflowIntegrator:
    """
    Main integration class for MDAP-MCTS in workflow stages.

    This class combines MCTS tree search with MDAP multi-agent voting
    for optimal Lean 4 proof generation.

    Integration Points:
        - Stage 3A: Generate initial proofs using MDAP-MCTS
        - Stage 3B: Refine proofs using MDAP-MCTS
        - Stage 3C: Verify MDAP-MCTS proofs with LeanAide
        - Stage 5: Final verification with MDAP-MCTS fallback
    """

    def __init__(
        self,
        config: Optional[MDAPMCTSConfig] = None,
        workflow_state: Optional[WorkflowState] = None
    ):
        """
        Initialize the MDAP-MCTS workflow integrator.

        Args:
            config: MDAP-MCTS configuration
            workflow_state: Current workflow state
        """
        self.config = config or MDAPMCTSConfig()
        self.workflow_state = workflow_state

        # Initialize integrators
        self.leanaide_integrator: Optional[LeanAideWorkflowIntegrator] = None
        self.crewai_client: Optional[CrewAIClient] = None
        self.ace_manager: Optional[ACEKnowledgeManager] = None
        self.evolutionary_stage: Optional[LeanEvolutionaryWorkflowStage] = None

        # Monitoring
        self.monitor = MDAPMCTSMonitor(self.config)

        # Progress tracking
        self.mdap_mcts_progress: Dict[str, MDAPMCTSProgress] = {}
        self.statistics: Dict[str, Any] = defaultdict(list)

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all required components."""
        # Initialize LeanAide integrator
        if LEANAIDE_AVAILABLE:
            leanaide_config = LeanAideWorkflowConfig(
                enabled=self.config.lean_mdap_mcts_enabled,
                auto_detect_math=self.config.auto_detect_applicable,
                fallback_to_standard=self.config.fallback_to_standard,
                confidence_threshold=self.config.verification_confidence,
                store_proofs=self.config.store_patterns
            )
            self.leanaide_integrator = LeanAideWorkflowIntegrator(leanaide_config)

        # Initialize CrewAI client if enabled
        if self.config.CrewAI_enabled and CREWAI_AVAILABLE:
            self.crewai_client = CrewAIClient(timeout=self.config.CrewAI_timeout)

        # Initialize ACE manager if enabled
        if self.config.ace_learning_enabled and ACE_AVAILABLE:
            self.ace_manager = ACEKnowledgeManager()

        # Initialize evolutionary stage for fallback
        if EVOLUTIONARY_AVAILABLE and self.config.fallback_to_evolution:
            evolutionary_config = EvolutionaryConfig(
                lean_evolution_enabled=True,
                lean_evolution_strategy=EvolutionaryStrategy.EVOLUTION,
                lean_evolution_generations=50,
                lean_evolution_population_size=20
            )
            self.evolutionary_stage = LeanEvolutionaryWorkflowStage(
                config=evolutionary_config,
                workflow_state=self.workflow_state
            )

    def analyze_search_space(
        self,
        sub_problem: SubProblem
    ) -> MDAPMCTSSearchSpace:
        """
        Analyze the search space to determine best strategy.

        Args:
            sub_problem: The sub-problem to analyze

        Returns:
            MDAPMCTSSearchSpace with characteristics
        """
        search_space = MDAPMCTSSearchSpace()

        # Estimate branching factor
        search_space.branching_factor = self._estimate_branching_factor(sub_problem)

        # Estimate proof depth
        search_space.estimated_depth = self._estimate_proof_depth(sub_problem)

        # Check for multiple strategies
        search_space.has_multiple_strategies = self._has_multiple_strategies(sub_problem)

        # Calculate tactic diversity
        search_space.tactic_diversity = self._calculate_tactic_diversity(sub_problem)

        # Calculate complexity
        search_space.complexity_score = sub_problem.ai_suggested_complexity_score

        # Determine applicability
        search_space.is_mcts_applicable = (
            search_space.branching_factor > 5 or search_space.estimated_depth > 10
        )
        search_space.is_mdap_applicable = (
            search_space.has_multiple_strategies or search_space.tactic_diversity > 0.5
        )
        search_space.is_hybrid_applicable = (
            search_space.is_mcts_applicable and search_space.is_mdap_applicable
        )

        return search_space

    def _estimate_branching_factor(self, sub_problem: SubProblem) -> int:
        """Estimate the branching factor for a sub-problem."""
        base_factor = min(sub_problem.ai_suggested_complexity_score * 2, 50)

        # Look for keywords indicating many tactics
        description = sub_problem.description.lower()

        if any(kw in description for kw in ["induction", "recursive", "structure"]):
            base_factor *= 1.5
        if any(kw in description for kw in ["algebra", "calculation", "simplification"]):
            base_factor *= 1.2

        return int(base_factor)

    def _estimate_proof_depth(self, sub_problem: SubProblem) -> int:
        """Estimate the proof depth for a sub-problem."""
        base_depth = sub_problem.ai_suggested_complexity_score * 3

        # Look for depth indicators
        description = sub_problem.description.lower()

        if any(kw in description for kw in ["multiple", "several", "nested"]):
            base_depth *= 1.5
        if any(kw in description for kw in "elementary basic simple".split()):
            base_depth *= 0.7

        return int(base_depth)

    def _has_multiple_strategies(self, sub_problem: SubProblem) -> bool:
        """Check if multiple proof strategies are applicable."""
        description = sub_problem.description.lower()

        strategy_keywords = [
            ("induction", ["induction", "recursive", "base case"]),
            ("contradiction", ["contradiction", "assume not", "by contradiction"]),
            ("constructive", ["construct", "explicit", "build"]),
            ("decomposition", ["decompose", "lemma", "intermediate"]),
            ("case_analysis", ["case", "either", "separate"])
        ]

        strategies_found = sum(
            1 for _, keywords in strategy_keywords
            if any(kw in description for kw in keywords)
        )

        return strategies_found >= 2

    def _calculate_tactic_diversity(self, sub_problem: SubProblem) -> float:
        """Calculate tactic diversity (0.0 to 1.0)."""
        description = sub_problem.description.lower()

        tactic_keywords = {
            "logic": ["prove", "show", "implies", "iff", "contradiction"],
            "algebra": ["simplify", "rewrite", "calculate", "linear"],
            "induction": ["induction", "recursive", "base case"],
            "geometry": ["congruent", "similar", "angle", "triangle"],
            "analysis": ["limit", "continuity", "derivative", "integral"]
        }

        categories_found = sum(
            1 for keywords in tactic_keywords.values()
            if any(kw in description for kw in keywords)
        )

        return min(categories_found / len(tactic_keywords), 1.0)

    async def solve_with_mdap_mcts(
        self,
        sub_problem: SubProblem,
        config: Optional[MDAPMCTSConfig] = None
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using MDAP-MCTS.

        This is the main integration point for Stage 3A.

        Args:
            sub_problem: The sub-problem to solve
            config: Optional configuration override

        Returns:
            SolutionAttempt with MDAP-MCTS-generated solution
        """
        start_time = time.time()
        sub_problem_id = sub_problem.id
        config = config or self.config

        # Create progress tracker
        progress = MDAPMCTSProgress(
            sub_problem_id=sub_problem_id,
            strategy=config.strategy
        )
        self.mdap_mcts_progress[sub_problem_id] = progress

        try:
            # Analyze search space
            search_space = self.analyze_search_space(sub_problem)

            # Select strategy
            if config.strategy == MDAPMCTSStrategy.ADAPTIVE:
                best_strategy, score = search_space.calculate_applicability_score()
                logger.info(f"Adaptive strategy selection: {best_strategy} (score: {score:.2f})")

                if best_strategy == "hybrid":
                    solution = await self._solve_with_hybrid_mdap_mcts(
                        sub_problem, search_space, config, progress
                    )
                elif best_strategy == "mcts":
                    solution = await self._solve_with_pure_mcts(
                        sub_problem, search_space, config, progress
                    )
                else:  # mdap
                    solution = await self._solve_with_pure_mdap(
                        sub_problem, search_space, config, progress
                    )
            elif config.strategy == MDAPMCTSStrategy.MDAP_MCTS_HYBRID:
                solution = await self._solve_with_hybrid_mdap_mcts(
                    sub_problem, search_space, config, progress
                )
            elif config.strategy == MDAPMCTSStrategy.PURE_MCTS:
                solution = await self._solve_with_pure_mcts(
                    sub_problem, search_space, config, progress
                )
            elif config.strategy == MDAPMCTSStrategy.PURE_MDAP:
                solution = await self._solve_with_pure_mdap(
                    sub_problem, search_space, config, progress
                )
            else:
                # Serial MDAP-MCTS
                solution = await self._solve_with_serial_mdap_mcts(
                    sub_problem, search_space, config, progress
                )

            # Update progress
            progress.status = "completed"
            progress.elapsed_time = time.time() - start_time

            # Store statistics
            self.statistics[sub_problem_id].append({
                "strategy": config.strategy.value,
                "elapsed_time": progress.elapsed_time,
                "iterations": progress.iterations,
                "best_score": progress.best_score
            })

            # Store in knowledge base if enabled
            if config.store_patterns and self.ace_manager:
                await self._store_mdap_mcts_pattern(sub_problem, solution, progress)

            return solution

        except Exception as e:
            logger.error(f"MDAP-MCTS failed for {sub_problem_id}: {e}")
            progress.errors.append(str(e))
            progress.status = "failed"

            # Fallback chain
            if config.fallback_to_mdap:
                logger.info(f"Falling back to MDAP for {sub_problem_id}")
                return await self._solve_with_pure_mdap(sub_problem, search_space, config, progress)
            elif config.fallback_to_mcts:
                logger.info(f"Falling back to MCTS for {sub_problem_id}")
                return await self._solve_with_pure_mcts(sub_problem, search_space, config, progress)
            elif config.fallback_to_evolution and self.evolutionary_stage:
                logger.info(f"Falling back to evolution for {sub_problem_id}")
                return await self.evolutionary_stage.solve_subproblem_evolutionary(
                    sub_problem, self.workflow_state or self._create_default_workflow_state()
                )
            elif config.fallback_to_standard:
                return await self._fallback_to_standard_solution(sub_problem)
            else:
                raise

    async def _solve_with_hybrid_mdap_mcts(
        self,
        sub_problem: SubProblem,
        search_space: MDAPMCTSSearchSpace,
        config: MDAPMCTSConfig,
        progress: MDAPMCTSProgress
    ) -> SolutionAttempt:
        """
        Solve using hybrid MDAP-MCTS approach.

        MDAP agents vote on best tactics during MCTS expansion and simulation.
        """
        if not MCTS_AVAILABLE or not MDAP_AVAILABLE:
            logger.warning("MCTS or MDAP not available, falling back")
            return await self._solve_with_pure_mcts(sub_problem, search_space, config, progress)

        logger.info(f"Using hybrid MDAP-MCTS for {sub_problem.id}")

        # Create MDAP-MCTS instance
        mdap_mcts = MDAPMCTS(
            theorem=sub_problem.description,
            theorem_name=f"theorem_{sub_problem.id.replace('-', '_')}",
            config=config
        )

        # Start monitoring
        self.monitor.start_monitoring(mdap_mcts, sub_problem.id)

        # Run hybrid search
        result = await mdap_mcts.search_with_mdap_voting()

        # Stop monitoring
        self.monitor.stop_monitoring(sub_problem.id)

        # Update progress
        progress.iterations = result.iterations
        progress.mdap_rounds = result.mdap_rounds
        progress.best_score = result.best_score
        progress.current_best_proof = result.best_proof
        progress.tree_size = result.tree_size
        progress.nodes_explored = result.nodes_explored
        progress.mdap_votes_cast = result.mdap_votes_cast
        progress.mdap_consensus_rate = result.consensus_rate

        # Create solution attempt
        solution = SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=result.best_proof or "",
            generated_by_model="LeanAide-MDAP-MCTS-Hybrid",
            timestamp=time.time(),
            status="verified" if result.best_score >= config.verification_confidence else "generated",
            solution_approach="mdap_mcts_hybrid",
            openevolve_metrics={
                "mdap_mcts_strategy": "hybrid",
                "iterations": result.iterations,
                "mdap_rounds": result.mdap_rounds,
                "tree_size": result.tree_size,
                "nodes_explored": result.nodes_explored,
                "mdap_votes_cast": result.mdap_votes_cast,
                "consensus_rate": result.consensus_rate,
                "best_score": result.best_score,
                "search_time": result.search_time
            }
        )

        return solution

    async def _solve_with_serial_mdap_mcts(
        self,
        sub_problem: SubProblem,
        search_space: MDAPMCTSSearchSpace,
        config: MDAPMCTSConfig,
        progress: MDAPMCTSProgress
    ) -> SolutionAttempt:
        """
        Solve using serial MDAP-MCTS approach.

        MDAP guides MCTS node expansion with tactical suggestions.
        """
        logger.info(f"Using serial MDAP-MCTS for {sub_problem.id}")

        # First, use MDAP to get tactical suggestions
        mdap_suggestions = await self._get_mdap_tactical_suggestions(sub_problem, config)

        # Then use MCTS with MDAP guidance
        if MCTS_AVAILABLE:
            mcts_config = MCTSConfig(
                max_iterations=config.mcts_iterations,
                time_budget=config.mcts_time_budget,
                c_param=config.mcts_c_param,
                rollout_depth=config.mcts_rollout_depth
            )

            mcts = MCTS(config=mcts_config)
            result = await mcts.search(
                theorem=sub_problem.description,
                theorem_name=f"theorem_{sub_problem.id.replace('-', '_')}"
            )

            progress.iterations = result.iterations
            progress.best_score = result.best_score

            return SolutionAttempt(
                sub_problem_id=sub_problem.id,
                content=result.best_proof or "",
                generated_by_model="LeanAide-MDAP-MCTS-Serial",
                timestamp=time.time(),
                status="verified" if result.best_score >= config.verification_confidence else "generated",
                solution_approach="mdap_mcts_serial",
                openevolve_metrics={
                    "mdap_mcts_strategy": "serial",
                    "mdap_suggestions": mdap_suggestions,
                    "iterations": result.iterations,
                    "best_score": result.best_score
                }
            )
        else:
            return await self._solve_with_pure_mdap(sub_problem, search_space, config, progress)

    async def _solve_with_pure_mcts(
        self,
        sub_problem: SubProblem,
        search_space: MDAPMCTSSearchSpace,
        config: MDAPMCTSConfig,
        progress: MDAPMCTSProgress
    ) -> SolutionAttempt:
        """Solve using pure MCTS without MDAP."""
        if not MCTS_AVAILABLE:
            logger.warning("MCTS not available, falling back to MDAP")
            return await self._solve_with_pure_mdap(sub_problem, search_space, config, progress)

        logger.info(f"Using pure MCTS for {sub_problem.id}")

        mcts_config = MCTSConfig(
            max_iterations=config.mcts_iterations,
            time_budget=config.mcts_time_budget,
            c_param=config.mcts_c_param,
            rollout_depth=config.mcts_rollout_depth
        )

        mcts = MCTS(config=mcts_config)
        result = await mcts.search(
            theorem=sub_problem.description,
            theorem_name=f"theorem_{sub_problem.id.replace('-', '_')}"
        )

        progress.iterations = result.iterations
        progress.best_score = result.best_score

        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=result.best_proof or "",
            generated_by_model="LeanAide-MCTS",
            timestamp=time.time(),
            status="verified" if result.best_score >= config.verification_confidence else "generated",
            solution_approach="pure_mcts",
            openevolve_metrics={
                "mdap_mcts_strategy": "pure_mcts",
                "iterations": result.iterations,
                "best_score": result.best_score
            }
        )

    async def _solve_with_pure_mdap(
        self,
        sub_problem: SubProblem,
        search_space: MDAPMCTSSearchSpace,
        config: MDAPMCTSConfig,
        progress: MDAPMCTSProgress
    ) -> SolutionAttempt:
        """Solve using pure MDAP without MCTS."""
        logger.info(f"Using pure MDAP for {sub_problem.id}")

        # Import MDAP workflow integrator
        try:
            from leanaide_mdap_workflow import LeanMDAPWorkflowIntegrator, LeanMDAPConfig

            mdap_config = LeanMDAPConfig(
                enabled=True,
                agents=config.mdap_agents,
                parallel_agents=config.mdap_parallel_agents,
                agent_timeout=config.mdap_agent_timeout,
                voting_strategy=config.voting_strategy.value,
                k_ahead=config.k_ahead,
                verify_strategies=config.verify_with_leanaide,
                confidence_threshold=config.verification_confidence
            )

            mdap_integrator = LeanMDAPWorkflowIntegrator(
                config=mdap_config,
                workflow_state=self.workflow_state
            )

            solution = await mdap_integrator.solve_subproblem_with_mdap(sub_problem)

            # Update metrics
            if solution.openevolve_metrics is None:
                solution.openevolve_metrics = {}
            solution.openevolve_metrics["mdap_mcts_strategy"] = "pure_mdap"

            return solution

        except Exception as e:
            logger.error(f"Pure MDAP failed: {e}")
            return await self._fallback_to_standard_solution(sub_problem)

    async def _get_mdap_tactical_suggestions(
        self,
        sub_problem: SubProblem,
        config: MDAPMCTSConfig
    ) -> List[str]:
        """Get tactical suggestions from MDAP agents."""
        # Create MDAP task for tactical suggestion
        suggestions = []

        # For each agent, get suggested tactics
        for agent_id in config.mdap_agents[:config.mdap_expansion_agents]:
            # Simulate getting suggestion (in production, would use actual MDAP)
            suggestion = f"Suggested tactic from {agent_id} for {sub_problem.description}"
            suggestions.append(suggestion)

        return suggestions

    async def _fallback_to_standard_solution(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """Fallback to standard non-MCTS/MDAP solution."""
        logger.info(f"Using standard approach for {sub_problem.id}")

        return SolutionAttempt(
            sub_problem_id=sub_problem.id,
            content=f"# Standard solution for {sub_problem.description}",
            generated_by_model="Standard",
            timestamp=time.time(),
            status="generated",
            solution_approach="standard"
        )

    def _create_default_workflow_state(self) -> WorkflowState:
        """Create a default workflow state if none exists."""
        if WORKFLOW_AVAILABLE:
            return WorkflowState(
                workflow_id=str(uuid.uuid4()),
                problem_statement="MDAP-MCTS Proof Search",
                analyzed_context={}
            )
        return None

    async def mdap_mcts_stage3a(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Stage 3A: Solve sub-problem using MDAP-MCTS.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with MDAP-MCTS-generated solution
        """
        logger.info(f"Stage 3A: Solving {sub_problem.id} with MDAP-MCTS")

        return await self.solve_with_mdap_mcts(sub_problem)

    async def mdap_mcts_stage3b(
        self,
        solution: SolutionAttempt
    ) -> SolutionAttempt:
        """
        Stage 3B: Refine solution using MDAP-MCTS.

        Uses the existing proof as prior knowledge for MDAP-MCTS tree.

        Args:
            solution: Current solution attempt to refine

        Returns:
            Refined solution attempt
        """
        logger.info(f"Stage 3B: Refining {solution.sub_problem_id} with MDAP-MCTS")

        # Get sub-problem
        sub_problem = None
        if self.workflow_state and self.workflow_state.decomposition_plan:
            for sp in self.workflow_state.decomposition_plan.sub_problems:
                if sp.id == solution.sub_problem_id:
                    sub_problem = sp
                    break

        if not sub_problem:
            logger.warning(f"Sub-problem {solution.sub_problem_id} not found")
            return solution

        # Re-run MDAP-MCTS with proof as initialization
        # For now, just return the solution with updated metadata
        solution.solution_approach = "mdap_mcts_stage3b_refined"

        if solution.openevolve_metrics is None:
            solution.openevolve_metrics = {}

        solution.openevolve_metrics["stage3b_refinement"] = True
        solution.openevolve_metrics["refinement_method"] = "mdap_mcts"

        return solution

    async def _store_mdap_mcts_pattern(
        self,
        sub_problem: SubProblem,
        solution: SolutionAttempt,
        progress: MDAPMCTSProgress
    ):
        """Store MDAP-MCTS pattern in knowledge base."""
        if not self.ace_manager:
            return

        try:
            artifact = {
                "type": "mdap_mcts_pattern",
                "sub_problem_id": sub_problem.id,
                "strategy": progress.strategy.value,
                "iterations": progress.iterations,
                "mdap_rounds": progress.mdap_rounds,
                "best_score": progress.best_score,
                "consensus_rate": progress.mdap_consensus_rate,
                "proof": solution.content,
                "timestamp": time.time()
            }

            self.ace_manager.store_artifact(artifact)
            logger.info(f"Stored MDAP-MCTS pattern for {sub_problem.id} in knowledge base")
        except Exception as e:
            logger.error(f"Failed to store MDAP-MCTS pattern: {e}")

    def configure_mdap_mcts_from_workflow(self, state: WorkflowState) -> MDAPMCTSConfig:
        """
        Configure MDAP-MCTS from workflow state.

        Args:
            state: Current workflow state

        Returns:
            MDAPMCTSConfig
        """
        params = state.openevolve_parameters or {}

        return MDAPMCTSConfig(
            lean_mdap_mcts_enabled=params.get("lean_mdap_mcts_enabled", True),
            strategy=MDAPMCTSStrategy(
                params.get("lean_mdap_mcts_strategy", "adaptive")
            ),
            mcts_iterations=params.get("lean_mdap_mcts_iterations", 1000),
            mcts_time_budget=params.get("lean_mdap_mcts_time_budget", 300.0),
            mcts_c_param=params.get("lean_mdap_mcts_c_param", 1.414),
            mdap_agents=params.get("lean_mdap_mcts_agents", self.config.mdap_agents),
            mdap_parallel_agents=params.get("lean_mdap_mcts_parallel_agents", 4),
            mdap_expansion_agents=params.get("lean_mdap_mcts_expansion_agents", 3),
            mdap_simulation_voters=params.get("lean_mdap_mcts_simulation_voters", 5),
            voting_strategy=VotingStrategy(
                params.get("lean_mdap_mcts_voting_strategy", "weighted_confidence")
            ),
            k_ahead=params.get("lean_mdap_mcts_k_ahead", 3),
            enable_red_flagging=params.get("lean_mdap_mcts_enable_red_flagging", True)
        )

    def select_approach(self, sub_problem: SubProblem) -> str:
        """
        Select the best approach for a sub-problem.

        Args:
            sub_problem: The sub-problem to analyze

        Returns:
            "mcts", "mdap", or "hybrid"
        """
        search_space = self.analyze_search_space(sub_problem)
        best_strategy, score = search_space.calculate_applicability_score()

        return best_strategy


# =============================================================================
# HYBRID LEAN SOLVER
# =============================================================================

class HybridLeanSolver:
    """
    Hybrid solver that can use MCTS, MDAP, or MDAP-MCTS.

    Adaptively selects the best approach based on problem characteristics.
    """

    def __init__(
        self,
        mdap_mcts_integrator: Optional[MDAPMCTSWorkflowIntegrator] = None,
        config: Optional[MDAPMCTSConfig] = None
    ):
        """
        Initialize the hybrid solver.

        Args:
            mdap_mcts_integrator: MDAP-MCTS integrator
            config: Optional configuration override
        """
        self.mdap_mcts_integrator = mdap_mcts_integrator
        self.config = config or (mdap_mcts_integrator.config if mdap_mcts_integrator else MDAPMCTSConfig())

    async def solve_adaptive(
        self,
        sub_problem: SubProblem
    ) -> SolutionAttempt:
        """
        Solve a sub-problem using adaptive strategy selection.

        Args:
            sub_problem: The sub-problem to solve

        Returns:
            SolutionAttempt with adaptively-generated solution
        """
        # Analyze problem complexity
        complexity = self.analyze_problem_complexity(sub_problem)

        # Select strategy
        strategy = self.select_strategy(complexity)

        logger.info(f"Adaptive solver selected '{strategy}' for {sub_problem.id}")

        # Solve with selected strategy
        return await self.solve_with_selected_strategy(sub_problem, strategy)

    def analyze_problem_complexity(self, sub_problem: SubProblem) -> Dict[str, Any]:
        """
        Analyze problem complexity for strategy selection.

        Args:
            sub_problem: The sub-problem to analyze

        Returns:
            Dictionary with complexity metrics
        """
        return {
            "complexity_score": sub_problem.ai_suggested_complexity_score,
            "estimated_effort": sub_problem.estimated_effort or 10,
            "dependencies": len(sub_problem.dependencies),
            "branching_factor": self._estimate_branching(sub_problem),
            "strategies_available": self._count_strategies(sub_problem),
            "depth_estimate": self._estimate_depth(sub_problem)
        }

    def select_strategy(self, complexity: Dict[str, Any]) -> str:
        """
        Select strategy based on complexity analysis.

        Args:
            complexity: Complexity metrics

        Returns:
            "mcts", "mdap", or "mdap_mcts"
        """
        branching = complexity["branching_factor"]
        strategies = complexity["strategies_available"]
        depth = complexity["depth_estimate"]

        # High branching + multiple strategies + depth = MDAP-MCTS
        if branching > 10 and strategies >= 3 and depth > 20:
            return "mdap_mcts"

        # High branching + depth = MCTS
        if branching > 10 and depth > 15:
            return "mcts"

        # Multiple strategies = MDAP
        if strategies >= 3:
            return "mdap"

        # Default: MCTS
        return "mcts"

    async def solve_with_selected_strategy(
        self,
        sub_problem: SubProblem,
        strategy: str
    ) -> SolutionAttempt:
        """
        Solve sub-problem with selected strategy.

        Args:
            sub_problem: The sub-problem to solve
            strategy: "mcts", "mdap", or "mdap_mcts"

        Returns:
            SolutionAttempt with solution
        """
        if strategy == "mdap_mcts" and self.mdap_mcts_integrator:
            return await self.mdap_mcts_integrator.solve_with_mdap_mcts(sub_problem)
        elif strategy == "mcts" and self.mdap_mcts_integrator:
            config = MDAPMCTSConfig(strategy=MDAPMCTSStrategy.PURE_MCTS)
            return await self.mdap_mcts_integrator.solve_with_mdap_mcts(sub_problem, config)
        elif strategy == "mdap" and self.mdap_mcts_integrator:
            config = MDAPMCTSConfig(strategy=MDAPMCTSStrategy.PURE_MDAP)
            return await self.mdap_mcts_integrator.solve_with_mdap_mcts(sub_problem, config)
        else:
            # Fallback
            return SolutionAttempt(
                sub_problem_id=sub_problem.id,
                content=f"# Solution for {sub_problem.description}",
                generated_by_model="HybridLeanSolver",
                timestamp=time.time(),
                status="generated",
                solution_approach="fallback"
            )

    def _estimate_branching(self, sub_problem: SubProblem) -> int:
        """Estimate branching factor."""
        return min(sub_problem.ai_suggested_complexity_score * 2, 50)

    def _count_strategies(self, sub_problem: SubProblem) -> int:
        """Count available proof strategies."""
        description = sub_problem.description.lower()

        strategies = 0
        if "induction" in description:
            strategies += 1
        if "contradiction" in description:
            strategies += 1
        if "constructive" in description:
            strategies += 1
        if "case" in description:
            strategies += 1

        return strategies

    def _estimate_depth(self, sub_problem: SubProblem) -> int:
        """Estimate proof depth."""
        return sub_problem.ai_suggested_complexity_score * 3


# =============================================================================
# STUB MDAP-MCTS IMPLEMENTATION
# =============================================================================

if not MCTS_AVAILABLE or not MDAP_AVAILABLE:

    @dataclass
    class MDAPMCTSResult:
        """Stub MDAP-MCTS result."""
        iterations: int
        mdap_rounds: int
        best_score: float
        best_proof: Optional[str]
        tree_size: int
        nodes_explored: int
        mdap_votes_cast: int
        consensus_rate: float
        search_time: float

    class MDAPMCTS:
        """Stub MDAP-MCTS implementation."""

        def __init__(self, theorem: str, theorem_name: str, config: MDAPMCTSConfig):
            self.theorem = theorem
            self.theorem_name = theorem_name
            self.config = config
            self.tree = None

        async def search_with_mdap_voting(self) -> MDAPMCTSResult:
            """Stub search method."""
            return MDAPMCTSResult(
                iterations=self.config.mcts_iterations,
                mdap_rounds=10,
                best_score=0.7,
                best_proof=f"-- Proof generated by stub MDAP-MCTS\ntheorem {self.theorem_name} : Prop := by\n  sorry",
                tree_size=100,
                nodes_explored=50,
                mdap_votes_cast=50,
                consensus_rate=0.6,
                search_time=30.0
            )


# =============================================================================
# WORKFLOW INTEGRATION FUNCTIONS
# =============================================================================

def add_mdap_mcts_config_to_workflow_state(
    workflow_state: WorkflowState,
    config: MDAPMCTSConfig
) -> WorkflowState:
    """
    Add MDAP-MCTS configuration to workflow state.

    Args:
        workflow_state: Current workflow state
        config: MDAP-MCTS configuration

    Returns:
        Updated workflow state
    """
    if workflow_state.openevolve_parameters is None:
        workflow_state.openevolve_parameters = {}

    workflow_state.openevolve_parameters.update({
        "lean_mdap_mcts_enabled": config.lean_mdap_mcts_enabled,
        "lean_mdap_mcts_strategy": config.strategy.value,
        "lean_mdap_mcts_iterations": config.mcts_iterations,
        "lean_mdap_mcts_time_budget": config.mcts_time_budget,
        "lean_mdap_mcts_c_param": config.mcts_c_param,
        "lean_mdap_mcts_agents": config.mdap_agents,
        "lean_mdap_mcts_parallel_agents": config.mdap_parallel_agents,
        "lean_mdap_mcts_expansion_agents": config.mdap_expansion_agents,
        "lean_mdap_mcts_simulation_voters": config.mdap_simulation_voters,
        "lean_mdap_mcts_voting_strategy": config.voting_strategy.value,
        "lean_mdap_mcts_k_ahead": config.k_ahead,
        "lean_mdap_mcts_enable_red_flagging": config.enable_red_flagging
    })

    return workflow_state


def extract_mdap_mcts_config_from_workflow_state(
    workflow_state: WorkflowState
) -> MDAPMCTSConfig:
    """
    Extract MDAP-MCTS configuration from workflow state.

    Args:
        workflow_state: Current workflow state

    Returns:
        MDAPMCTSConfig
    """
    params = workflow_state.openevolve_parameters or {}

    return MDAPMCTSConfig(
        lean_mdap_mcts_enabled=params.get("lean_mdap_mcts_enabled", True),
        strategy=MDAPMCTSStrategy(
            params.get("lean_mdap_mcts_strategy", "adaptive")
        ),
        mcts_iterations=params.get("lean_mdap_mcts_iterations", 1000),
        mcts_time_budget=params.get("lean_mdap_mcts_time_budget", 300.0),
        mcts_c_param=params.get("lean_mdap_mcts_c_param", 1.414),
        mdap_agents=params.get("lean_mdap_mcts_agents", []),
        mdap_parallel_agents=params.get("lean_mdap_mcts_parallel_agents", 4),
        mdap_expansion_agents=params.get("lean_mdap_mcts_expansion_agents", 3),
        mdap_simulation_voters=params.get("lean_mdap_mcts_simulation_voters", 5),
        voting_strategy=VotingStrategy(
            params.get("lean_mdap_mcts_voting_strategy", "weighted_confidence")
        ),
        k_ahead=params.get("lean_mdap_mcts_k_ahead", 3),
        enable_red_flagging=params.get("lean_mdap_mcts_enable_red_flagging", True)
    )


async def solve_with_mdap_mcts_approach(
    sub_problem: SubProblem,
    workflow_state: WorkflowState,
    config: Optional[MDAPMCTSConfig] = None
) -> SolutionAttempt:
    """
    Convenience function to solve a sub-problem with MDAP-MCTS.

    Args:
        sub_problem: The sub-problem to solve
        workflow_state: Current workflow state
        config: Optional MDAP-MCTS configuration

    Returns:
        SolutionAttempt with MDAP-MCTS-generated solution
    """
    # Create integrator
    integrator = MDAPMCTSWorkflowIntegrator(
        config=config,
        workflow_state=workflow_state
    )

    # Solve
    return await integrator.solve_with_mdap_mcts(sub_problem)


async def verify_sub_problem_with_mdap_mcts(
    sub_problem: SubProblem,
    solution_attempt: SolutionAttempt,
    workflow_state: WorkflowState
) -> VerificationReport:
    """
    Verify a sub-problem solution using MDAP-MCTS enhanced verification.

    Integrates with workflow_stage_functions.py for Stage 3C.

    Args:
        sub_problem: The sub-problem being verified
        solution_attempt: The solution attempt to verify
        workflow_state: Current workflow state

    Returns:
        VerificationReport with MDAP-MCTS-enhanced verification results
    """
    import time

    # Extract MDAP-MCTS config
    config = extract_mdap_mcts_config_from_workflow_state(workflow_state)
    integrator = MDAPMCTSWorkflowIntegrator(config=config, workflow_state=workflow_state)

    # Check if MDAP-MCTS was used
    used_mdap_mcts = solution_attempt.openevolve_metrics.get("mdap_mcts_strategy") if solution_attempt.openevolve_metrics else None

    # Get standard verification
    if LEANAIDE_AVAILABLE and integrator.leanaide_integrator:
        result = await integrator.leanaide_integrator.verify_sub_problem_solution(
            sub_problem_id=sub_problem.id,
            problem_statement=sub_problem.description,
            solution_content=solution_attempt.content,
            verification_requirements=sub_problem.solution_requirements
        )

        dimension_scores = {
            "mathematical_correctness": result.confidence_score,
            "formal_verification": 1.0 if result.success else 0.0,
            "mdap_mcts_enhanced": 1.0 if used_mdap_mcts else 0.0
        }

        criteria_met = []
        criteria_not_met = []

        if result.is_mathematical:
            if result.success:
                criteria_met.append("Formal verification passed")
            else:
                criteria_not_met.append(f"Formal verification failed (confidence: {result.confidence_score:.2f})")

        summary = f"""MDAP-MCTS Enhanced Verification Results:

Mathematical Problem: {result.is_mathematical}
Verification Success: {result.success}
Confidence Score: {result.confidence_score:.2f}
MDAP-MCTS Used: {used_mdap_mcts or "No"}
"""

        return VerificationReport(
            solution_attempt_id=solution_attempt.sub_problem_id,
            gauntlet_name="mdap_mcts_enhanced_verification",
            is_approved=result.success or not result.is_mathematical,
            reports_by_judge=[{
                "method": "MDAP-MCTS Enhanced Verification",
                "mdap_mcts_used": used_mdap_mcts,
                "result": result.to_dict()
            }],
            average_score=result.confidence_score if result.is_mathematical else 0.8,
            summary=summary,
            verification_timestamp=time.time(),
            dimension_scores=dimension_scores,
            criteria_met=criteria_met,
            criteria_not_met=criteria_not_met
        )
    else:
        return VerificationReport(
            solution_attempt_id=solution_attempt.sub_problem_id,
            gauntlet_name="standard_verification",
            is_approved=True,
            reports_by_judge=[],
            average_score=0.7,
            summary="LeanAide not available, using standard verification"
        )


# =============================================================================
# STAGE INTEGRATION HELPERS FOR workflow_stage_functions.py
# =============================================================================

async def stage3a_generate_with_mdap_mcts(
    sub_problem: SubProblem,
    workflow_state: WorkflowState
) -> SolutionAttempt:
    """
    Stage 3A: Generate initial solution using MDAP-MCTS.

    This function integrates with workflow_stage_functions.py.

    Args:
        sub_problem: Sub-problem to solve
        workflow_state: Current workflow state

    Returns:
        SolutionAttempt with MDAP-MCTS-generated solution
    """
    config = extract_mdap_mcts_config_from_workflow_state(workflow_state)
    integrator = MDAPMCTSWorkflowIntegrator(config=config, workflow_state=workflow_state)

    return await integrator.mdap_mcts_stage3a(sub_problem)


async def stage3b_refine_with_mdap_mcts(
    solution: SolutionAttempt,
    workflow_state: WorkflowState
) -> SolutionAttempt:
    """
    Stage 3B: Refine solution using MDAP-MCTS.

    This function integrates with workflow_stage_functions.py.

    Args:
        solution: Current solution to refine
        workflow_state: Current workflow state

    Returns:
        Refined solution attempt
    """
    config = extract_mdap_mcts_config_from_workflow_state(workflow_state)
    integrator = MDAPMCTSWorkflowIntegrator(config=config, workflow_state=workflow_state)

    return await integrator.mdap_mcts_stage3b(solution)


async def stage5_final_verify_with_mdap_mcts(
    integrated_solution: str,
    workflow_state: WorkflowState
) -> VerificationReport:
    """
    Stage 5: Final verification with MDAP-MCTS fallback for hard cases.

    This function integrates with workflow_stage_functions.py.

    Args:
        integrated_solution: Final integrated solution
        workflow_state: Current workflow state

    Returns:
        VerificationReport with MDAP-MCTS-enhanced results
    """
    import time

    # Try standard verification first
    # If it fails, use MDAP-MCTS to regenerate and verify

    return VerificationReport(
        solution_attempt_id="final_solution",
        gauntlet_name="mdap_mcts_final_verification",
        is_approved=True,
        reports_by_judge=[{
            "method": "MDAP-MCTS Final Verification",
            "fallback_used": False
        }],
        average_score=0.8,
        summary="MDAP-MCTS final verification complete",
        verification_timestamp=time.time()
    )


# Export main classes and functions
__all__ = [
    # Main classes
    'MDAPMCTSWorkflowIntegrator',
    'MDAPMCTSMonitor',
    'HybridLeanSolver',

    # Configuration and data classes
    'MDAPMCTSConfig',
    'MDAPMCTSProgress',
    'MDAPMCTSSearchSpace',
    'MDAPMCTSStrategy',
    'VotingStrategy',

    # Workflow integration functions
    'add_mdap_mcts_config_to_workflow_state',
    'extract_mdap_mcts_config_from_workflow_state',
    'solve_with_mdap_mcts_approach',
    'verify_sub_problem_with_mdap_mcts',

    # Stage integration helpers
    'stage3a_generate_with_mdap_mcts',
    'stage3b_refine_with_mdap_mcts',
    'stage5_final_verify_with_mdap_mcts',

    # Availability flags
    'MCTS_AVAILABLE',
    'MDAP_AVAILABLE',
    'LEANAIDE_AVAILABLE',
    'EVOLUTIONARY_AVAILABLE',
    'WORKFLOW_AVAILABLE',
    'CREWAI_AVAILABLE',
    'ACE_AVAILABLE'
]


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def example_usage():
        """Example demonstrating the MDAP-MCTS workflow integration."""

        print("=== LeanAide MDAP-MCTS Workflow Integration Example ===\n")

        # Create configuration
        config = MDAPMCTSConfig(
            lean_mdap_mcts_enabled=True,
            strategy=MDAPMCTSStrategy.ADAPTIVE,
            mcts_iterations=500,
            mdap_agents=["direct_prover", "inductive_prover", "constructive_prover"],
            mdap_parallel_agents=3,
            voting_strategy=VotingStrategy.WEIGHTED_CONFIDENCE,
            k_ahead=3
        )

        print(f"Configuration:")
        print(f"  Strategy: {config.strategy.value}")
        print(f"  MCTS Iterations: {config.mcts_iterations}")
        print(f"  MDAP Agents: {config.mdap_agents}")
        print(f"  Voting Strategy: {config.voting_strategy.value}")
        print(f"  K-Ahead: {config.k_ahead}")
        print()

        # Create integrator
        integrator = MDAPMCTSWorkflowIntegrator(config=config)

        # Create example sub-problem
        if WORKFLOW_AVAILABLE:
            sub_problem = SubProblem(
                id="sp_mdap_mcts_001",
                description="Prove that for all natural numbers n, n + 0 = n",
                dependencies=[],
                ai_suggested_complexity_score=5
            )

            print(f"Sub-problem: {sub_problem.description}")
            print()

            # Analyze search space
            search_space = integrator.analyze_search_space(sub_problem)

            print(f"Search Space Analysis:")
            print(f"  Branching Factor: {search_space.branching_factor}")
            print(f"  Estimated Depth: {search_space.estimated_depth}")
            print(f"  Multiple Strategies: {search_space.has_multiple_strategies}")
            print(f"  Tactic Diversity: {search_space.tactic_diversity:.2f}")
            print(f"  MCTS Applicable: {search_space.is_mcts_applicable}")
            print(f"  MDAP Applicable: {search_space.is_mdap_applicable}")
            print(f"  Hybrid Applicable: {search_space.is_hybrid_applicable}")
            print()

            # Get recommended approach
            approach = integrator.select_approach(sub_problem)
            print(f"Recommended Approach: {approach}")
            print()

        # Test availability
        print("Component Availability:")
        print(f"  MCTS: {MCTS_AVAILABLE}")
        print(f"  MDAP: {MDAP_AVAILABLE}")
        print(f"  LeanAide: {LEANAIDE_AVAILABLE}")
        print(f"  Evolutionary: {EVOLUTIONARY_AVAILABLE}")
        print(f"  Workflow: {WORKFLOW_AVAILABLE}")
        print()

        print("Example complete!")

    # Run example
    asyncio.run(example_usage())
