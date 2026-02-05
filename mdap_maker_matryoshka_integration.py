#!/usr/bin/env python3
"""
================================================================================
MDAP/MAKER + MATRYOSHKA INTEGRATION (OPTIONAL)
================================================================================

This module provides optional Matryoshka integration for MDAP/MAKER systems.
If matryoshka_unified_memory_integration is not available, MDAP/MAKER
continues to work with standard functionality without any impact.

Key Features:
- Completely optional - MDAP/MAKER works without Matryoshka
- Graceful degradation - Falls back automatically if dependencies missing
- Configuration-driven - Enable/disable via MDAPMatryoshkaConfig
- Hybrid operation - Can use both MDAP and Matryoshka together
- CrewAI compatible - Works with CrewAI-based MAKER engines

Usage:
    # Basic usage (auto-detects Matryoshka availability)
    engine = MDAPMakerWithMatryoshka()
    result = engine.solve_with_document_analysis(problem, document_path)
    
    # Explicitly disable Matryoshka
    config = MDAPMatryoshkaConfig(enabled=False)
    engine = MDAPMakerWithMatryoshka(matryoshka_config=config)
    
    # With CrewAI
    crewai_engine = CrewAIMDAPMakerWithMatryoshka()

Dependencies:
    Required: None (MDAP/MAKER works standalone)
    Optional: matryoshka_unified_memory_integration, matryoshka_enhanced_client
              knowledge_unified_memory_system, crewai

Author: OpenEvolve Team
Version: 1.0.0
================================================================================
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from enum import Enum

# ================================================================================
# OPTIONAL DEPENDENCY HANDLING
# ================================================================================

logger = logging.getLogger(__name__)

# Matryoshka dependencies
try:
    from matryoshka_unified_memory_integration import (
        MatryoshkaMemoryBridge,
        MatryoshkaExplorationSession,
        UnifiedMatryoshkaClient,
        create_unified_matryoshka_client,
    )
    from matryoshka_enhanced_client import (
        EnhancedMatryoshkaClient,
        AnalysisOptions,
        create_enhanced_client,
    )
    MATRYOSHKA_AVAILABLE = True
    logger.debug("Matryoshka dependencies loaded successfully")
except ImportError as e:
    MATRYOSHKA_AVAILABLE = False
    logger.debug(f"Matryoshka not available: {e}")
    # Define None placeholders for type hints
    MatryoshkaMemoryBridge = None
    MatryoshkaExplorationSession = None
    UnifiedMatryoshkaClient = None
    EnhancedMatryoshkaClient = None
    AnalysisOptions = None

# MDAP/MAKER dependencies
try:
    from mdap_engine import MDAPConfig, MDAPRunResult, RedFlagRules, MDAPENGINE
    from maker_engine import MakerEngine, MakerConfig, MakerStep
    MDAP_AVAILABLE = True
    logger.debug("MDAP/MAKER dependencies loaded successfully")
except ImportError as e:
    MDAP_AVAILABLE = False
    logger.debug(f"MDAP/MAKER not available: {e}")
    MDAPConfig = None
    MDAPRunResult = None
    RedFlagRules = None
    MDAPENGINE = None
    MakerEngine = None
    MakerConfig = None
    MakerStep = None

# Unified Memory dependencies
try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        create_unified_system,
    )
    UNIFIED_MEMORY_AVAILABLE = True
    logger.debug("Unified Memory dependencies loaded successfully")
except ImportError as e:
    UNIFIED_MEMORY_AVAILABLE = False
    logger.debug(f"Unified Memory not available: {e}")
    UnifiedMemorySystem = None

# CrewAI dependencies
try:
    from crewai import Agent, Task, Crew
    from crewai_mdap_maker_engine import MAKEREngineCrewAI, MAKERConfig
    CREWAI_AVAILABLE = True
    logger.debug("CrewAI dependencies loaded successfully")
except ImportError as e:
    CREWAI_AVAILABLE = False
    logger.debug(f"CrewAI not available: {e}")
    Agent = None
    Task = None
    Crew = None
    MAKEREngineCrewAI = None
    MAKERConfig = None

# Decomposition dependencies
try:
    from decomposition_engine import DecompositionEngine, DecompositionResult
    from problem_decomposition import ProblemDefinition, SubProblem
    DECOMPOSITION_AVAILABLE = True
except ImportError:
    DECOMPOSITION_AVAILABLE = False
    DecompositionEngine = None
    DecompositionResult = None
    ProblemDefinition = None
    SubProblem = None

# Team dependencies
try:
    from team_manager import Team, TeamManager
    TEAM_AVAILABLE = True
except ImportError:
    TEAM_AVAILABLE = False
    Team = None
    TeamManager = None


# ================================================================================
# DATA CLASSES AND ENUMS
# ================================================================================

class ExplorationStrategy(Enum):
    """Available exploration strategies for Matryoshka."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


@dataclass
class MDAPMatryoshkaConfig:
    """
    Configuration for optional Matryoshka integration with MDAP/MAKER.
    
    All settings default to safe values that don't require Matryoshka.
    Set enabled=True to activate Matryoshka features.
    
    Attributes:
        enabled: Master switch for Matryoshka integration
        use_for_large_documents: Use Matryoshka for documents > 10MB
        use_for_deep_exploration: Use Matryoshka for complex codebases
        use_for_cross_session_learning: Learn from past analyses
        enable_unified_memory: Use 4-layer indexing system
        memory_storage_path: Path for memory storage
        matryoshka_max_turns: Maximum exploration turns
        memory_limit_per_context: Memory entries per context
        exploration_strategy: Exploration approach (breadth_first, depth_first, adaptive)
        mdap_for_structure: Use MDAP for decomposition structure
        matryoshka_for_exploration: Use Matryoshka for leaf exploration
        document_size_threshold_mb: Threshold for "large" documents
        fallback_on_error: Fall back to standard MDAP if Matryoshka fails
        cache_exploration_results: Cache Matryoshka exploration results
        exploration_timeout_seconds: Timeout for exploration operations
    """
    enabled: bool = False
    
    # When to use Matryoshka
    use_for_large_documents: bool = True
    use_for_deep_exploration: bool = True
    use_for_cross_session_learning: bool = True
    
    # Memory integration
    enable_unified_memory: bool = True
    memory_storage_path: Optional[str] = None
    
    # Analysis options
    matryoshka_max_turns: int = 20
    memory_limit_per_context: int = 15
    exploration_strategy: str = "adaptive"  # breadth_first, depth_first, adaptive
    
    # Hybrid mode: when both MDAP and Matryoshka are used
    mdap_for_structure: bool = True
    matryoshka_for_exploration: bool = True
    
    # Additional options
    document_size_threshold_mb: float = 10.0
    fallback_on_error: bool = True
    cache_exploration_results: bool = True
    exploration_timeout_seconds: int = 300
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.matryoshka_max_turns < 1:
            raise ValueError("matryoshka_max_turns must be >= 1")
        if self.memory_limit_per_context < 1:
            raise ValueError("memory_limit_per_context must be >= 1")
        if self.document_size_threshold_mb < 0:
            raise ValueError("document_size_threshold_mb must be >= 0")


@dataclass
class ExplorationResult:
    """Result from Matryoshka exploration."""
    content: str
    insights: List[str] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    confidence: float = 0.0
    exploration_depth: int = 0
    memory_references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDAPMatryoshkaResult:
    """
    Result from MDAP/MAKER with optional Matryoshka enhancement.
    
    Contains both standard MDAP results and optional Matryoshka enhancements.
    The matryoshka_enhanced flag indicates if Matryoshka was actually used.
    """
    # Core MDAP result (always present)
    mdap_result: Optional[Any] = None
    maker_result: Optional[Any] = None
    
    # Matryoshka enhancements (optional)
    matryoshka_enhanced: bool = False
    exploration_result: Optional[ExplorationResult] = None
    document_analysis: Optional[Dict[str, Any]] = None
    cross_session_insights: List[str] = field(default_factory=list)
    
    # Metadata
    execution_time_ms: float = 0.0
    fallback_used: bool = False
    error_message: Optional[str] = None
    
    def is_success(self) -> bool:
        """Check if the result represents a successful execution."""
        return self.mdap_result is not None or self.maker_result is not None
    
    def get_solution(self) -> Optional[str]:
        """Extract the solution text from the result."""
        if self.maker_result and hasattr(self.maker_result, 'solution'):
            return self.maker_result.solution
        if self.mdap_result and hasattr(self.mdap_result, 'solution'):
            return self.mdap_result.solution
        return None


@dataclass
class VotingResult:
    """Result from voting with optional context retrieval."""
    winner: Optional[Any] = None
    rankings: List[Tuple[Any, float]] = field(default_factory=list)
    context_used: bool = False
    retrieved_memories: List[str] = field(default_factory=list)
    voting_method: str = "standard"
    confidence: float = 0.0


@dataclass
class HybridDecompositionResult:
    """Result from hybrid MDAP + Matryoshka decomposition."""
    decomposition: Optional[Any] = None
    matryoshka_context: Optional[ExplorationResult] = None
    subproblems: List[Any] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    recommended_strategy: str = "standard"
    

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def _check_document_size(document_path: Optional[str], threshold_mb: float) -> bool:
    """Check if a document exceeds the size threshold."""
    if not document_path or not os.path.exists(document_path):
        return False
    try:
        size_bytes = os.path.getsize(document_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb > threshold_mb
    except (OSError, IOError):
        return False


def _estimate_complexity(problem: str) -> float:
    """Estimate problem complexity for Matryoshka decision."""
    # Simple heuristic based on length and keywords
    complexity = 0.0
    
    # Length factor
    complexity += min(len(problem) / 1000, 5.0)
    
    # Keyword factors
    complex_keywords = ['optimize', 'complex', 'distributed', 'architecture', 
                       'system', 'integration', 'multiple', 'dependencies']
    for keyword in complex_keywords:
        if keyword in problem.lower():
            complexity += 0.5
    
    return min(complexity, 10.0)


def _should_use_matryoshka(
    config: MDAPMatryoshkaConfig,
    document_path: Optional[str] = None,
    document_content: Optional[str] = None,
    problem: Optional[str] = None
) -> bool:
    """Determine if Matryoshka should be used based on configuration and inputs."""
    if not config.enabled or not MATRYOSHKA_AVAILABLE:
        return False
    
    # Check document size
    if document_path and config.use_for_large_documents:
        if _check_document_size(document_path, config.document_size_threshold_mb):
            return True
    
    # Check document content length
    if document_content and config.use_for_large_documents:
        if len(document_content) > config.document_size_threshold_mb * 1024 * 1024:
            return True
    
    # Check problem complexity
    if problem and config.use_for_deep_exploration:
        complexity = _estimate_complexity(problem)
        if complexity > 5.0:
            return True
    
    return False


# ================================================================================
# MAIN INTEGRATION CLASS
# ================================================================================

class MDAPMakerWithMatryoshka:
    """
    Enhanced MDAP/MAKER with optional Matryoshka integration.
    
    Falls back gracefully to standard MDAP/MAKER if:
    - Matryoshka not installed
    - Unified Memory not available  
    - Configured as disabled
    - Matryoshka binary not found
    - Any runtime error occurs (if fallback_on_error is True)
    
    Example:
        >>> config = MDAPMatryoshkaConfig(enabled=True)
        >>> engine = MDAPMakerWithMatryoshka(matryoshka_config=config)
        >>> result = engine.solve_with_document_analysis(
        ...     problem="Optimize this codebase",
        ...     document_path="/path/to/code"
        ... )
        >>> if result.matryoshka_enhanced:
        ...     print("Used Matryoshka for analysis")
    """
    
    def __init__(
        self,
        mdap_config: Optional[Any] = None,
        maker_config: Optional[Any] = None,
        matryoshka_config: Optional[MDAPMatryoshkaConfig] = None
    ):
        """
        Initialize MDAP/MAKER with optional Matryoshka.
        
        Args:
            mdap_config: Configuration for MDAP engine
            maker_config: Configuration for MAKER engine
            matryoshka_config: Configuration for Matryoshka integration
        """
        # Store configurations
        self.mdap_config = mdap_config or (MDAPConfig() if MDAP_AVAILABLE else None)
        self.maker_config = maker_config or (MakerConfig() if MDAP_AVAILABLE else None)
        self.matryoshka_config = matryoshka_config or MDAPMatryoshkaConfig()
        
        # Initialize standard MDAP/MAKER (always attempt)
        self.mdap_engine: Optional[Any] = None
        self.maker_engine: Optional[Any] = None
        self._init_mdap_maker()
        
        # Initialize Matryoshka only if enabled AND available
        self.matryoshka_client: Optional[Any] = None
        self.memory_bridge: Optional[Any] = None
        self.exploration_cache: Dict[str, ExplorationResult] = {}
        
        if self.matryoshka_config.enabled:
            self._init_matryoshka()
    
    def _init_mdap_maker(self):
        """Initialize standard MDAP/MAKER engines."""
        if not MDAP_AVAILABLE:
            logger.warning("MDAP/MAKER not available - limited functionality")
            return
        
        try:
            if self.mdap_config:
                self.mdap_engine = MDAPENGINE(self.mdap_config)
            
            if self.maker_config:
                team = Team() if TEAM_AVAILABLE else None
                self.maker_engine = MakerEngine(team, self.maker_config)
            
            logger.debug("MDAP/MAKER engines initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MDAP/MAKER: {e}")
    
    def _init_matryoshka(self):
        """Initialize Matryoshka if enabled and available."""
        if not self.matryoshka_config.enabled:
            logger.info("Matryoshka integration disabled by configuration")
            return
        
        if not MATRYOSHKA_AVAILABLE:
            logger.warning(
                "Matryoshka integration enabled but dependencies not available. "
                "Install with: pip install matryoshka"
            )
            return
        
        try:
            # Create enhanced Matryoshka client
            self.matryoshka_client = create_enhanced_client(
                storage_path=self.matryoshka_config.memory_storage_path,
                enable_unified_memory=self.matryoshka_config.enable_unified_memory
            )
            
            # Initialize memory bridge if Unified Memory is available
            if UNIFIED_MEMORY_AVAILABLE and MatryoshkaMemoryBridge is not None:
                if hasattr(self.matryoshka_client, 'unified_memory'):
                    self.memory_bridge = MatryoshkaMemoryBridge(
                        self.matryoshka_client.unified_memory
                    )
            
            logger.info("Matryoshka integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Matryoshka: {e}. Continuing without it.")
            self.matryoshka_client = None
            self.memory_bridge = None
    
    @property
    def has_matryoshka(self) -> bool:
        """Check if Matryoshka is available and working."""
        return (
            self.matryoshka_config.enabled 
            and MATRYOSHKA_AVAILABLE 
            and self.matryoshka_client is not None
        )
    
    @property
    def has_mdap(self) -> bool:
        """Check if MDAP is available."""
        return MDAP_AVAILABLE and self.mdap_engine is not None
    
    @property
    def has_maker(self) -> bool:
        """Check if MAKER is available."""
        return MDAP_AVAILABLE and self.maker_engine is not None
    
    # ================================================================================
    # MAIN METHODS
    # ================================================================================
    
    def solve_with_document_analysis(
        self,
        problem: str,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        use_matryoshka: Optional[bool] = None
    ) -> MDAPMatryoshkaResult:
        """
        Solve problem with optional Matryoshka document analysis.
        
        If use_matryoshka is None, auto-decide based on document size and config.
        If Matryoshka unavailable, gracefully falls back to standard MDAP.
        
        Args:
            problem: The problem statement to solve
            document_path: Path to document for analysis (optional)
            document_content: Direct document content (optional)
            use_matryoshka: Force Matryoshka use (None=auto, True=force, False=disable)
        
        Returns:
            MDAPMatryoshkaResult containing solution and metadata
        """
        import time
        start_time = time.time()
        
        result = MDAPMatryoshkaResult()
        
        try:
            # Determine if Matryoshka should be used
            if use_matryoshka is None:
                use_matryoshka = _should_use_matryoshka(
                    self.matryoshka_config,
                    document_path,
                    document_content,
                    problem
                )
            
            # Phase 1: Document Analysis with Matryoshka (if enabled)
            exploration_result = None
            if use_matryoshka and self.has_matryoshka:
                try:
                    exploration_result = self._explore_with_matryoshka(
                        problem, document_path, document_content
                    )
                    result.exploration_result = exploration_result
                    result.matryoshka_enhanced = True
                    logger.debug("Matryoshka exploration completed")
                except Exception as e:
                    logger.warning(f"Matryoshka exploration failed: {e}")
                    if not self.matryoshka_config.fallback_on_error:
                        raise
            
            # Phase 2: MDAP/MAKER Solution
            if self.has_maker:
                maker_result = self._run_maker(problem, exploration_result)
                result.maker_result = maker_result
            elif self.has_mdap:
                mdap_result = self._run_mdap(problem, exploration_result)
                result.mdap_result = mdap_result
            else:
                result.error_message = "No solver engine available"
                logger.error(result.error_message)
            
            # Phase 3: Cross-session learning (if enabled)
            if (self.matryoshka_config.use_for_cross_session_learning 
                and self.has_matryoshka 
                and exploration_result):
                insights = self._retrieve_cross_session_insights(problem)
                result.cross_session_insights = insights
            
        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Error in solve_with_document_analysis: {e}")
            
            if self.matryoshka_config.fallback_on_error and not result.is_success():
                logger.info("Attempting fallback to standard MDAP/MAKER")
                result.fallback_used = True
                try:
                    if self.has_maker:
                        result.maker_result = self._run_maker(problem, None)
                    elif self.has_mdap:
                        result.mdap_result = self._run_mdap(problem, None)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
        
        result.execution_time_ms = (time.time() - start_time) * 1000
        return result
    
    def decompose_with_memory(
        self,
        problem: str,
        context: Optional[str] = None
    ) -> HybridDecompositionResult:
        """
        Decompose problem using hybrid approach:
        - MDAP for structural decomposition
        - Matryoshka for context-aware exploration (if available)
        - Unified memory for cross-problem learning
        
        Args:
            problem: Problem to decompose
            context: Additional context for decomposition
        
        Returns:
            HybridDecompositionResult with decomposition and context
        """
        result = HybridDecompositionResult()
        
        # Get Matryoshka context if available
        matryoshka_context = None
        if self.has_matryoshka and self.matryoshka_config.matryoshka_for_exploration:
            try:
                matryoshka_context = self._explore_with_matryoshka(
                    problem, document_content=context
                )
                result.matryoshka_context = matryoshka_context
            except Exception as e:
                logger.warning(f"Matryoshka context retrieval failed: {e}")
        
        # Perform MDAP decomposition
        if self.has_mdap and self.matryoshka_config.mdap_for_structure:
            try:
                # Combine problem with Matryoshka insights
                enhanced_problem = problem
                if matryoshka_context:
                    enhanced_problem = self._enhance_problem_with_context(
                        problem, matryoshka_context
                    )
                
                # TODO: Implement actual decomposition call when available
                # For now, create a placeholder result
                result.decomposition = {
                    'problem': enhanced_problem,
                    'original_problem': problem,
                    'method': 'mdap_with_matryoshka' if matryoshka_context else 'mdap_only'
                }
                
            except Exception as e:
                logger.error(f"MDAP decomposition failed: {e}")
        
        return result
    
    def vote_with_context_retrieval(
        self,
        candidates: List[Any],
        context_query: str,
        voting_method: str = "standard"
    ) -> VotingResult:
        """
        Voting with enhanced context retrieval:
        - Standard voting if no Matryoshka
        - Hybrid retrieval of relevant memories if Matryoshka available
        
        Args:
            candidates: List of candidates to vote on
            context_query: Query for context retrieval
            voting_method: Voting method to use
        
        Returns:
            VotingResult with rankings and context
        """
        result = VotingResult()
        result.voting_method = voting_method
        
        # Retrieve context from Matryoshka if available
        retrieved_memories = []
        if self.has_matryoshka and self.memory_bridge:
            try:
                memories = self.memory_bridge.retrieve_relevant_memories(
                    context_query,
                    limit=self.matryoshka_config.memory_limit_per_context
                )
                retrieved_memories = memories
                result.retrieved_memories = memories
                result.context_used = True
            except Exception as e:
                logger.warning(f"Memory retrieval failed: {e}")
        
        # Perform voting (simplified implementation)
        if candidates:
            # Score candidates based on context if available
            rankings = []
            for i, candidate in enumerate(candidates):
                # Simple scoring - in practice would use more sophisticated methods
                score = 1.0 - (i * 0.1)  # Default ranking
                if retrieved_memories:
                    # Boost score based on context relevance
                    score += 0.1 * len(retrieved_memories)
                rankings.append((candidate, score))
            
            # Sort by score
            rankings.sort(key=lambda x: x[1], reverse=True)
            result.rankings = rankings
            result.winner = rankings[0][0] if rankings else None
            result.confidence = rankings[0][1] if rankings else 0.0
        
        return result
    
    # ================================================================================
    # INTERNAL HELPER METHODS
    # ================================================================================
    
    def _explore_with_matryoshka(
        self,
        problem: str,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None
    ) -> ExplorationResult:
        """Explore using Matryoshka."""
        if not self.has_matryoshka or not self.matryoshka_client:
            raise RuntimeError("Matryoshka not available")
        
        # Check cache
        cache_key = f"{problem}:{document_path or ''}:{hash(document_content or '')}"
        if (self.matryoshka_config.cache_exploration_results 
            and cache_key in self.exploration_cache):
            logger.debug("Returning cached exploration result")
            return self.exploration_cache[cache_key]
        
        # Perform exploration
        # Note: Actual implementation depends on Matryoshka API
        exploration = ExplorationResult(
            content=document_content or "",
            insights=[f"Explored: {problem}"],
            confidence=0.8,
            exploration_depth=self.matryoshka_config.matryoshka_max_turns
        )
        
        # Cache result
        if self.matryoshka_config.cache_exploration_results:
            self.exploration_cache[cache_key] = exploration
        
        return exploration
    
    def _run_maker(
        self,
        problem: str,
        exploration_result: Optional[ExplorationResult] = None
    ) -> Any:
        """Run MAKER engine with optional exploration context."""
        if not self.has_maker or not self.maker_engine:
            raise RuntimeError("MAKER not available")
        
        # Enhance problem with exploration insights
        enhanced_problem = problem
        if exploration_result:
            context = f"\n\nContext from document analysis:\n{exploration_result.content}"
            enhanced_problem = problem + context
        
        # Run MAKER (actual implementation depends on MAKER API)
        return self.maker_engine.solve(enhanced_problem)
    
    def _run_mdap(
        self,
        problem: str,
        exploration_result: Optional[ExplorationResult] = None
    ) -> Any:
        """Run MDAP engine with optional exploration context."""
        if not self.has_mdap or not self.mdap_engine:
            raise RuntimeError("MDAP not available")
        
        # Enhance problem with exploration insights
        enhanced_problem = problem
        if exploration_result:
            context = f"\n\nContext from document analysis:\n{exploration_result.content}"
            enhanced_problem = problem + context
        
        # Run MDAP (actual implementation depends on MDAP API)
        return self.mdap_engine.run(enhanced_problem)
    
    def _retrieve_cross_session_insights(self, problem: str) -> List[str]:
        """Retrieve insights from past sessions."""
        if not self.memory_bridge:
            return []
        
        try:
            memories = self.memory_bridge.retrieve_relevant_memories(problem, limit=5)
            return [str(m) for m in memories]
        except Exception as e:
            logger.warning(f"Failed to retrieve cross-session insights: {e}")
            return []
    
    def _enhance_problem_with_context(
        self,
        problem: str,
        context: ExplorationResult
    ) -> str:
        """Enhance problem statement with Matryoshka context."""
        enhanced = problem
        
        if context.content:
            enhanced += f"\n\nDocument Context:\n{context.content[:2000]}"
        
        if context.insights:
            enhanced += f"\n\nKey Insights:\n" + "\n".join(f"- {i}" for i in context.insights[:5])
        
        if context.key_concepts:
            enhanced += f"\n\nKey Concepts: {', '.join(context.key_concepts[:10])}"
        
        return enhanced
    
    def get_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'matryoshka_available': MATRYOSHKA_AVAILABLE,
            'matryoshka_enabled': self.matryoshka_config.enabled,
            'matryoshka_active': self.has_matryoshka,
            'mdap_available': self.has_mdap,
            'maker_available': self.has_maker,
            'unified_memory_available': UNIFIED_MEMORY_AVAILABLE,
            'memory_bridge_active': self.memory_bridge is not None,
            'exploration_cache_size': len(self.exploration_cache),
        }


# ================================================================================
# CREWAI INTEGRATION (also optional)
# ================================================================================

class CrewAIMDAPMakerWithMatryoshka:
    """
    CrewAI-based MDAP/MAKER with optional Matryoshka.
    Extends crewai_mdap_maker_engine with document exploration.
    
    This class provides a CrewAI-compatible wrapper around MDAPMakerWithMatryoshka,
    allowing CrewAI agents to leverage Matryoshka document analysis.
    
    Example:
        >>> config = MDAPMatryoshkaConfig(enabled=True)
        >>> engine = CrewAIMDAPMakerWithMatryoshka(matryoshka_config=config)
        >>> # Use with CrewAI crew
        >>> crew = Crew(agents=[...], tasks=[...])
    """
    
    def __init__(
        self,
        maker_config: Optional[Any] = None,
        matryoshka_config: Optional[MDAPMatryoshkaConfig] = None
    ):
        """
        Initialize CrewAI MAKER with optional Matryoshka.
        
        Args:
            maker_config: Configuration for MAKER engine
            matryoshka_config: Configuration for Matryoshka integration
        """
        self.matryoshka_config = matryoshka_config or MDAPMatryoshkaConfig()
        
        # Initialize base CrewAI MAKER if available
        self.base_maker: Optional[Any] = None
        if CREWAI_AVAILABLE:
            try:
                config = maker_config or (MAKERConfig() if MAKERConfig else None)
                self.base_maker = MAKEREngineCrewAI(config)
            except Exception as e:
                logger.warning(f"Failed to initialize CrewAI MAKER: {e}")
        
        # Initialize optional Matryoshka integration
        self.matryoshka = MDAPMakerWithMatryoshka(
            maker_config=maker_config,
            matryoshka_config=matryoshka_config
        )
    
    @property
    def has_crewai(self) -> bool:
        """Check if CrewAI is available."""
        return CREWAI_AVAILABLE and self.base_maker is not None
    
    @property
    def has_matryoshka(self) -> bool:
        """Check if Matryoshka is available."""
        return self.matryoshka.has_matryoshka
    
    def solve(
        self,
        problem: str,
        document_path: Optional[str] = None,
        document_content: Optional[str] = None,
        use_matryoshka: Optional[bool] = None
    ) -> MDAPMatryoshkaResult:
        """
        Solve problem using CrewAI MAKER with optional Matryoshka.
        
        Args:
            problem: Problem statement
            document_path: Path to document for analysis
            document_content: Direct document content
            use_matryoshka: Force Matryoshka use
        
        Returns:
            MDAPMatryoshkaResult with solution
        """
        return self.matryoshka.solve_with_document_analysis(
            problem=problem,
            document_path=document_path,
            document_content=document_content,
            use_matryoshka=use_matryoshka
        )
    
    def create_analysis_task(
        self,
        problem: str,
        document_path: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create a CrewAI task for document analysis.
        
        Args:
            problem: Problem statement
            document_path: Path to document
        
        Returns:
            CrewAI Task or None if CrewAI not available
        """
        if not CREWAI_AVAILABLE or not Agent or not Task:
            logger.warning("CrewAI not available for task creation")
            return None
        
        # Create analysis agent
        analyst = Agent(
            role='Document Analyst',
            goal='Analyze documents for relevant information',
            backstory='Expert at extracting insights from large documents'
        )
        
        # Create analysis task
        task = Task(
            description=f"Analyze document for problem: {problem}",
            agent=analyst,
            context={'document_path': document_path}
        )
        
        return task
    
    def get_status(self) -> Dict[str, Any]:
        """Get current integration status."""
        return {
            'crewai_available': CREWAI_AVAILABLE,
            'crewai_active': self.has_crewai,
            **self.matryoshka.get_status()
        }


# ================================================================================
# FACTORY FUNCTIONS
# ================================================================================

def create_mdap_maker_with_matryoshka(
    enabled: bool = False,
    mdap_config: Optional[Any] = None,
    maker_config: Optional[Any] = None,
    **kwargs
) -> MDAPMakerWithMatryoshka:
    """
    Factory function to create MDAPMakerWithMatryoshka.
    
    Args:
        enabled: Enable Matryoshka integration
        mdap_config: MDAP configuration
        maker_config: MAKER configuration
        **kwargs: Additional config options for MDAPMatryoshkaConfig
    
    Returns:
        Configured MDAPMakerWithMatryoshka instance
    
    Example:
        >>> engine = create_mdap_maker_with_matryoshka(
        ...     enabled=True,
        ...     use_for_large_documents=True
        ... )
    """
    config = MDAPMatryoshkaConfig(enabled=enabled, **kwargs)
    return MDAPMakerWithMatryoshka(
        mdap_config=mdap_config,
        maker_config=maker_config,
        matryoshka_config=config
    )


def create_crewai_maker_with_matryoshka(
    enabled: bool = False,
    maker_config: Optional[Any] = None,
    **kwargs
) -> CrewAIMDAPMakerWithMatryoshka:
    """
    Factory function to create CrewAIMDAPMakerWithMatryoshka.
    
    Args:
        enabled: Enable Matryoshka integration
        maker_config: MAKER configuration
        **kwargs: Additional config options
    
    Returns:
        Configured CrewAIMDAPMakerWithMatryoshka instance
    """
    config = MDAPMatryoshkaConfig(enabled=enabled, **kwargs)
    return CrewAIMDAPMakerWithMatryoshka(
        maker_config=maker_config,
        matryoshka_config=config
    )


def create_auto_configured_engine(
    problem: Optional[str] = None,
    document_path: Optional[str] = None,
    document_content: Optional[str] = None
) -> MDAPMakerWithMatryoshka:
    """
    Create auto-configured engine based on inputs.
    
    Automatically enables Matryoshka if:
    - Dependencies are available
    - Document is large
    - Problem appears complex
    
    Args:
        problem: Problem statement
        document_path: Document path
        document_content: Document content
    
    Returns:
        Auto-configured MDAPMakerWithMatryoshka
    """
    enabled = MATRYOSHKA_AVAILABLE and (
        _check_document_size(document_path, 10.0) if document_path else False
        or (len(document_content) > 10 * 1024 * 1024 if document_content else False)
        or (_estimate_complexity(problem or "") > 5.0 if problem else False)
    )
    
    return create_mdap_maker_with_matryoshka(
        enabled=enabled,
        use_for_large_documents=True,
        use_for_deep_exploration=True
    )


# ================================================================================
# HELPER UTILITIES
# ================================================================================

class MatryoshkaDecisionHelper:
    """
    Helper class for deciding when to use Matryoshka.
    
    Provides analysis of problems and documents to recommend
    whether Matryoshka would be beneficial.
    """
    
    @staticmethod
    def analyze_document(document_path: str) -> Dict[str, Any]:
        """Analyze a document and return recommendations."""
        result = {
            'path': document_path,
            'exists': os.path.exists(document_path),
            'size_mb': 0.0,
            'is_large': False,
            'recommend_matryoshka': False,
        }
        
        if result['exists']:
            try:
                size_bytes = os.path.getsize(document_path)
                size_mb = size_bytes / (1024 * 1024)
                result['size_mb'] = round(size_mb, 2)
                result['is_large'] = size_mb > 10.0
                result['recommend_matryoshka'] = size_mb > 5.0
            except (OSError, IOError) as e:
                result['error'] = str(e)
        
        return result
    
    @staticmethod
    def analyze_problem(problem: str) -> Dict[str, Any]:
        """Analyze a problem and return recommendations."""
        complexity = _estimate_complexity(problem)
        
        return {
            'length': len(problem),
            'complexity_score': round(complexity, 2),
            'complexity_level': 'high' if complexity > 7 else 'medium' if complexity > 3 else 'low',
            'recommend_matryoshka': complexity > 5.0,
            'keywords_found': [
                kw for kw in ['optimize', 'complex', 'distributed', 'architecture']
                if kw in problem.lower()
            ]
        }
    
    @staticmethod
    def get_recommendation(
        problem: Optional[str] = None,
        document_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive recommendation for Matryoshka use."""
        doc_analysis = MatryoshkaDecisionHelper.analyze_document(document_path) if document_path else {}
        prob_analysis = MatryoshkaDecisionHelper.analyze_problem(problem) if problem else {}
        
        recommend = (
            doc_analysis.get('recommend_matryoshka', False) or
            prob_analysis.get('recommend_matryoshka', False)
        )
        
        return {
            'recommend_matryoshka': recommend and MATRYOSHKA_AVAILABLE,
            'matryoshka_available': MATRYOSHKA_AVAILABLE,
            'reason': 'Document is large or problem is complex' if recommend else 'Standard processing sufficient',
            'document_analysis': doc_analysis,
            'problem_analysis': prob_analysis,
        }


def check_integration_health() -> Dict[str, Any]:
    """
    Check health of all integration components.
    
    Returns:
        Dict with health status of all components
    """
    return {
        'matryoshka': {
            'available': MATRYOSHKA_AVAILABLE,
            'components': {
                'unified_client': UnifiedMatryoshkaClient is not None,
                'enhanced_client': EnhancedMatryoshkaClient is not None,
                'memory_bridge': MatryoshkaMemoryBridge is not None,
            }
        },
        'mdap_maker': {
            'available': MDAP_AVAILABLE,
            'components': {
                'mdap_engine': MDAPENGINE is not None,
                'maker_engine': MakerEngine is not None,
            }
        },
        'unified_memory': {
            'available': UNIFIED_MEMORY_AVAILABLE,
            'system': UnifiedMemorySystem is not None,
        },
        'crewai': {
            'available': CREWAI_AVAILABLE,
            'components': {
                'base': Crew is not None,
                'maker_engine': MAKEREngineCrewAI is not None,
            }
        },
        'decomposition': {
            'available': DECOMPOSITION_AVAILABLE,
        },
        'team': {
            'available': TEAM_AVAILABLE,
        }
    }


def get_integration_info() -> str:
    """Get formatted information about integration status."""
    health = check_integration_health()
    
    lines = [
        "MDAP/MAKER + Matryoshka Integration Status",
        "=" * 50,
        "",
        f"Matryoshka Available: {'[OK]' if health['matryoshka']['available'] else '[FAIL]'}",
        f"  - Unified Client: {'[OK]' if health['matryoshka']['components']['unified_client'] else '[FAIL]'}",
        f"  - Enhanced Client: {'[OK]' if health['matryoshka']['components']['enhanced_client'] else '[FAIL]'}",
        f"  - Memory Bridge: {'[OK]' if health['matryoshka']['components']['memory_bridge'] else '[FAIL]'}",
        "",
        f"MDAP/MAKER Available: {'[OK]' if health['mdap_maker']['available'] else '[FAIL]'}",
        f"  - MDAP Engine: {'[OK]' if health['mdap_maker']['components']['mdap_engine'] else '[FAIL]'}",
        f"  - MAKER Engine: {'[OK]' if health['mdap_maker']['components']['maker_engine'] else '[FAIL]'}",
        "",
        f"Unified Memory Available: {'[OK]' if health['unified_memory']['available'] else '[FAIL]'}",
        f"CrewAI Available: {'[OK]' if health['crewai']['available'] else '[FAIL]'}",
        f"Decomposition Available: {'[OK]' if health['decomposition']['available'] else '[FAIL]'}",
        f"Team System Available: {'[OK]' if health['team']['available'] else '[FAIL]'}",
        "",
        "Usage:",
        "  engine = create_mdap_maker_with_matryoshka(enabled=True)",
        "  result = engine.solve_with_document_analysis(problem, doc_path)",
    ]
    
    return "\n".join(lines)


# ================================================================================
# MODULE ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    # Print integration status when run directly
    print(get_integration_info())
    
    # Example usage demonstration
    print("\n" + "=" * 50)
    print("Example Usage:")
    print("=" * 50)
    
    # Create engine (works with or without Matryoshka)
    engine = create_mdap_maker_with_matryoshka(enabled=False)
    status = engine.get_status()
    
    print(f"\nEngine Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Show recommendation
    helper = MatryoshkaDecisionHelper()
    rec = helper.get_recommendation(
        problem="Optimize distributed system architecture",
        document_path="./large_codebase.py"
    )
    
    print(f"\nRecommendation for sample problem:")
    print(f"  Use Matryoshka: {rec['recommend_matryoshka']}")
    print(f"  Reason: {rec['reason']}")
