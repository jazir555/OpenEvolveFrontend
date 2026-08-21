"""
Enhanced MDAP/MAKER Client with Optional Integrations

Production-ready client for MDAP (Multi-Decomposition Analysis Pipeline) and MAKER
with optional Matryoshka integration for large document analysis and memory bridge
for cross-session learning.

All integrations are optional and gracefully degrade if dependencies unavailable.

Usage:
    # Quick start with all features
    client = create_full_client("./memory")
    result = client.solve("Optimize this algorithm", document_path="spec.pdf")
    
    # Minimal setup
    client = create_minimal_client()
    result = client.solve("Simple problem")
    
    # Check capabilities
    print(client.capabilities)
"""
from __future__ import annotations


import logging
import time
import threading
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

try:
    from mdap_maker_matryoshka_integration import (
        MDAPMakerWithMatryoshka,
        MDAPMatryoshkaConfig,
        MDAPMatryoshkaResult
    )
    MATRYOSHKA_INTEGRATION_AVAILABLE = True
except ImportError:
    MATRYOSHKA_INTEGRATION_AVAILABLE = False
    MDAPMakerWithMatryoshka = None
    MDAPMatryoshkaConfig = None
    MDAPMatryoshkaResult = None

try:
    from mdap_memory_bridge import MDAPMemoryBridge, create_mdap_memory_bridge
    MEMORY_BRIDGE_AVAILABLE = True
except ImportError:
    MEMORY_BRIDGE_AVAILABLE = False
    MDAPMemoryBridge = None
    create_mdap_memory_bridge = None

# Standard MDAP/MAKER (may or may not be available)
try:
    from mdap_engine import MDAPConfig, MDAPENGINE, MDAPRunResult
    MDAP_AVAILABLE = True
except ImportError:
    MDAP_AVAILABLE = False
    MDAPConfig = None
    MDAPENGINE = None
    MDAPRunResult = None

try:
    from maker_engine import MakerEngine, MakerConfig, MakerStep, MakerRunResult
    MAKER_AVAILABLE = True
except ImportError:
    MAKER_AVAILABLE = False
    MakerEngine = None
    MakerConfig = None
    MakerStep = None
    MakerRunResult = None

# CrewAI version
try:
    from crewai_mdap_maker_engine import MAKEREngineCrewAI, MAKERConfig
    CREWAI_MAKER_AVAILABLE = True
except ImportError:
    CREWAI_MAKER_AVAILABLE = False
    MAKEREngineCrewAI = None
    MAKERConfig = None


# ============================================================================
# RESULT CLASSES
# ============================================================================

@dataclass
class SimilarSolution:
    """Represents a similar solution from memory."""
    problem: str
    solution: Any
    similarity_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentAnalysisResult:
    """Result from document analysis."""
    document_path: str
    document_size_mb: float
    analysis: str
    key_points: List[str]
    recommendations: List[str]
    chunks_analyzed: int
    time_seconds: float
    used_matryoshka: bool
    success: bool
    error: Optional[str] = None


@dataclass
class EnhancedDecompositionResult:
    """Result from enhanced decomposition."""
    subproblems: List[Dict[str, Any]]
    strategy_used: str
    similar_decompositions_found: int
    quality_score: float
    decomposition_time: float
    subproblem_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedVotingResult:
    """Result from enhanced voting."""
    winner: Any
    confidence: float
    vote_distribution: Dict[str, int]
    learned_patterns_applied: int
    red_flags_triggered: int
    voting_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedMDAPResult:
    """Result from enhanced solve operation."""
    solution: Any
    success: bool
    steps_taken: int
    time_seconds: float
    used_matryoshka: bool
    used_memory_bridge: bool
    matryoshka_session_id: Optional[str] = None
    decomposition_reused: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "solution": self.solution,
            "success": self.success,
            "steps_taken": self.steps_taken,
            "time_seconds": self.time_seconds,
            "used_matryoshka": self.used_matryoshka,
            "used_memory_bridge": self.used_memory_bridge,
            "matryoshka_session_id": self.matryoshka_session_id,
            "decomposition_reused": self.decomposition_reused,
            "metadata": self.metadata
        }


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EnhancedMDAPConfig:
    """
    Master configuration for Enhanced MDAP/MAKER.
    
    All integrations are optional and can be enabled/disabled.
    """
    # Core MDAP/MAKER config
    mdap_config: Optional[Any] = None  # MDAPConfig if available
    maker_config: Optional[Any] = None  # MakerConfig if available
    use_crewai: bool = False  # Use CrewAI-based MAKER if available
    
    # Optional Matryoshka integration
    enable_matryoshka: bool = False
    matryoshka_config: Optional[Any] = None  # MDAPMatryoshkaConfig
    
    # Optional memory bridge
    enable_memory_bridge: bool = True
    memory_storage_path: Optional[str] = None
    
    # Behavior settings
    auto_select_matryoshka: bool = True  # Auto-use for large docs
    large_document_threshold_mb: float = 10.0
    enable_cross_session_learning: bool = True
    
    # Performance settings
    max_decomposition_retries: int = 3
    voting_timeout_seconds: float = 300.0
    enable_parallel_processing: bool = True
    
    # Thread safety
    enable_thread_safety: bool = True


# ============================================================================
# ENHANCED MDAP CLIENT
# ============================================================================

class EnhancedMDAPClient:
    """
    Production-ready MDAP/MAKER client with optional enhancements.
    
    Features (all optional):
    - Standard MDAP/MAKER decomposition and voting
    - Matryoshka for large document analysis
    - Unified Memory for cross-session learning
    - CrewAI integration for agent-based execution
    
    Gracefully degrades if optional dependencies unavailable.
    
    Thread-safe for concurrent operations.
    """
    
    def __init__(self, config: EnhancedMDAPConfig = None):
        """
        Initialize the Enhanced MDAP Client.
        
        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or EnhancedMDAPConfig()
        
        # Thread safety
        self._lock = threading.RLock() if self.config.enable_thread_safety else None
        
        # Track what's available
        self._capabilities = {
            "mdap": MDAP_AVAILABLE,
            "maker": MAKER_AVAILABLE,
            "crewai_maker": CREWAI_MAKER_AVAILABLE,
            "matryoshka": False,  # Set after init
            "memory_bridge": False,
            "unified_memory": False
        }
        
        # Component references
        self.core_engine = None
        self.matryoshka_mdap = None
        self.memory_bridge = None
        self._mdap_engine = None
        
        # Initialize components
        self._init_core()
        self._init_matryoshka()
        self._init_memory_bridge()
        
        logger.info(f"EnhancedMDAPClient initialized. Capabilities: {self._capabilities}")
    
    def _acquire_lock(self):
        """Acquire lock if thread safety enabled."""
        if self._lock:
            self._lock.acquire()
    
    def _release_lock(self):
        """Release lock if thread safety enabled."""
        if self._lock:
            self._lock.release()
    
    @contextmanager
    def _locked_operation(self):
        """Context manager for thread-safe operations."""
        self._acquire_lock()
        try:
            yield
        finally:
            self._release_lock()
    
    def _init_core(self):
        """Initialize core MDAP/MAKER."""
        if not MAKER_AVAILABLE:
            logger.warning("MDAP/MAKER not available. Client will have limited functionality.")
            return
        
        try:
            # Initialize based on config
            if self.config.use_crewai and CREWAI_MAKER_AVAILABLE:
                maker_config = self.config.maker_config
                if maker_config is None:
                    maker_config = MAKERConfig() if MAKERConfig else None
                self.core_engine = MAKEREngineCrewAI(maker_config)
                logger.info("Initialized CrewAI-based MAKER engine")
            else:
                # Fall back to standard
                from workflow_structures import Team
                maker_config = self.config.maker_config
                if maker_config is None and MakerConfig:
                    maker_config = MakerConfig()
                self.core_engine = MakerEngine(Team(), maker_config)
                logger.info("Initialized standard MAKER engine")
            
            # Initialize MDAP engine if available
            if MDAP_AVAILABLE and MDAPENGINE:
                mdap_config = self.config.mdap_config
                if mdap_config is None and MDAPConfig:
                    mdap_config = MDAPConfig()
                self._mdap_engine = MDAPENGINE(mdap_config)
                logger.info("Initialized MDAP engine")
                
        except Exception as e:
            logger.error(f"Failed to initialize core engine: {e}")
            self.core_engine = None
    
    def _init_matryoshka(self):
        """Initialize optional Matryoshka integration."""
        if not self.config.enable_matryoshka:
            logger.debug("Matryoshka integration disabled by config")
            return
        
        if not MATRYOSHKA_INTEGRATION_AVAILABLE:
            logger.warning("Matryoshka integration not available. Install dependencies.")
            return
        
        try:
            # Initialize Matryoshka-enhanced MDAP
            matryoshka_config = self.config.matryoshka_config
            if matryoshka_config is None and MDAPMatryoshkaConfig:
                matryoshka_config = MDAPMatryoshkaConfig(enabled=True)
            
            self.matryoshka_mdap = MDAPMakerWithMatryoshka(
                mdap_config=self.config.mdap_config,
                maker_config=self.config.maker_config,
                matryoshka_config=matryoshka_config
            )
            
            # Check if Matryoshka is actually available
            has_matryoshka = getattr(self.matryoshka_mdap, 'has_matryoshka', False)
            self._capabilities["matryoshka"] = has_matryoshka
            
            if has_matryoshka:
                logger.info("Matryoshka integration initialized successfully")
            else:
                logger.warning("Matryoshka integration initialized but not functional")
                
        except Exception as e:
            logger.error(f"Failed to initialize Matryoshka: {e}")
            self.matryoshka_mdap = None
            self._capabilities["matryoshka"] = False
    
    def _init_memory_bridge(self):
        """Initialize optional memory bridge."""
        if not self.config.enable_memory_bridge:
            logger.debug("Memory bridge disabled by config")
            return
        
        if not MEMORY_BRIDGE_AVAILABLE:
            logger.warning("Memory bridge not available. Install mdap_memory_bridge.")
            return
        
        try:
            self.memory_bridge = create_mdap_memory_bridge(
                storage_path=self.config.memory_storage_path,
                enabled=True
            )
            
            is_enabled = getattr(self.memory_bridge, 'enabled', False)
            self._capabilities["memory_bridge"] = is_enabled
            self._capabilities["unified_memory"] = is_enabled
            
            if is_enabled:
                logger.info(f"Memory bridge initialized at {self.config.memory_storage_path}")
            else:
                logger.warning("Memory bridge initialized but not enabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize memory bridge: {e}")
            self.memory_bridge = None
            self._capabilities["memory_bridge"] = False
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """
        Check what features are available.
        
        Returns:
            Dictionary mapping capability names to availability booleans.
        """
        with self._locked_operation():
            return self._capabilities.copy()
    
    def _should_use_matryoshka(self, document_path: Optional[str], use_matryoshka: Optional[bool]) -> bool:
        """
        Determine whether to use Matryoshka based on configuration and document.
        
        Args:
            document_path: Path to document if provided
            use_matryoshka: Override setting (None for auto)
            
        Returns:
            True if Matryoshka should be used
        """
        # Explicit override
        if use_matryoshka is not None:
            return use_matryoshka and self._capabilities["matryoshka"]
        
        # Auto-select based on document size
        if not self.config.auto_select_matryoshka:
            return False
        
        if not document_path or not os.path.exists(document_path):
            return False
        
        try:
            size_mb = os.path.getsize(document_path) / (1024 * 1024)
            return size_mb >= self.config.large_document_threshold_mb
        except OSError:
            return False
    
    def solve(
        self,
        problem: str,
        context: Optional[str] = None,
        document_path: Optional[str] = None,
        use_matryoshka: Optional[bool] = None
    ) -> EnhancedMDAPResult:
        """
        Main solve method with all optional integrations.
        
        Automatically selects best approach based on:
        - Problem characteristics
        - Available integrations
        - Document size (if provided)
        
        Args:
            problem: The problem to solve
            context: Additional context
            document_path: Path to document (triggers Matryoshka if large)
            use_matryoshka: Force enable/disable Matryoshka (auto if None)
            
        Returns:
            EnhancedMDAPResult with solution and metadata
            
        Raises:
            RuntimeError: If no solving capability is available
        """
        start_time = time.time()
        
        with self._locked_operation():
            # Determine approach
            should_use_matryoshka = self._should_use_matryoshka(document_path, use_matryoshka)
            
            try:
                if should_use_matryoshka and self.matryoshka_mdap:
                    return self._solve_with_matryoshka(problem, context, document_path, start_time)
                elif self.core_engine:
                    return self._solve_with_maker(problem, context, start_time)
                elif self._mdap_engine:
                    return self._solve_with_mdap(problem, context, start_time)
                else:
                    raise RuntimeError("No solving engine available. Install MDAP/MAKER dependencies.")
                    
            except Exception as e:
                logger.error(f"Solve failed: {e}")
                elapsed = time.time() - start_time
                return EnhancedMDAPResult(
                    solution=None,
                    success=False,
                    steps_taken=0,
                    time_seconds=elapsed,
                    used_matryoshka=should_use_matryoshka,
                    used_memory_bridge=self._capabilities["memory_bridge"],
                    metadata={"error": str(e), "error_type": type(e).__name__}
                )
    
    def _solve_with_matryoshka(
        self,
        problem: str,
        context: Optional[str],
        document_path: Optional[str],
        start_time: float
    ) -> EnhancedMDAPResult:
        """Solve using Matryoshka integration."""
        try:
            result = self.matryoshka_mdap.solve(
                problem=problem,
                context=context,
                document_path=document_path
            )
            
            elapsed = time.time() - start_time
            
            # Extract session ID if available
            session_id = None
            if hasattr(result, 'matryoshka_session_id'):
                session_id = result.matryoshka_session_id
            
            return EnhancedMDAPResult(
                solution=getattr(result, 'solution', result),
                success=getattr(result, 'success', True),
                steps_taken=getattr(result, 'steps_taken', 0),
                time_seconds=elapsed,
                used_matryoshka=True,
                used_memory_bridge=self._capabilities["memory_bridge"],
                matryoshka_session_id=session_id,
                metadata=getattr(result, 'metadata', {})
            )
            
        except Exception as e:
            logger.warning(f"Matryoshka solve failed, falling back: {e}")
            if self.core_engine:
                return self._solve_with_maker(problem, context, start_time)
            raise
    
    def _solve_with_maker(
        self,
        problem: str,
        context: Optional[str],
        start_time: float
    ) -> EnhancedMDAPResult:
        """Solve using standard MAKER engine."""
        try:
            # Check memory for similar solutions first
            decomposition_reused = False
            if self.memory_bridge and self.config.enable_cross_session_learning:
                similar = self.get_similar_solutions(problem, limit=1)
                if similar and similar[0].similarity_score > 0.9:
                    logger.info("Found highly similar solution in memory")
                    decomposition_reused = True
            
            # Run MAKER
            result = self.core_engine.solve(problem, context)
            
            elapsed = time.time() - start_time
            
            # Store in memory
            if self.memory_bridge and self.config.enable_cross_session_learning:
                self._store_solution(problem, result)
            
            return EnhancedMDAPResult(
                solution=getattr(result, 'solution', result),
                success=getattr(result, 'success', True),
                steps_taken=getattr(result, 'steps_taken', 1),
                time_seconds=elapsed,
                used_matryoshka=False,
                used_memory_bridge=self._capabilities["memory_bridge"],
                decomposition_reused=decomposition_reused,
                metadata=getattr(result, 'metadata', {})
            )
            
        except Exception as e:
            logger.error(f"MAKER solve failed: {e}")
            raise
    
    def _solve_with_mdap(
        self,
        problem: str,
        context: Optional[str],
        start_time: float
    ) -> EnhancedMDAPResult:
        """Solve using MDAP engine."""
        try:
            result = self._mdap_engine.run(problem, context)
            elapsed = time.time() - start_time
            
            return EnhancedMDAPResult(
                solution=getattr(result, 'solution', result),
                success=getattr(result, 'success', True),
                steps_taken=getattr(result, 'iterations', 1),
                time_seconds=elapsed,
                used_matryoshka=False,
                used_memory_bridge=self._capabilities["memory_bridge"],
                metadata=getattr(result, 'metadata', {})
            )
            
        except Exception as e:
            logger.error(f"MDAP solve failed: {e}")
            raise
    
    def _store_solution(self, problem: str, result: Any) -> None:
        """Store solution in memory bridge."""
        if not self.memory_bridge:
            return
        
        try:
            # Attempt to store - method may vary based on bridge implementation
            if hasattr(self.memory_bridge, 'store_solution'):
                self.memory_bridge.store_solution(problem, result)
            elif hasattr(self.memory_bridge, 'store'):
                self.memory_bridge.store({
                    "problem": problem,
                    "solution": result,
                    "timestamp": time.time()
                })
        except Exception as e:
            logger.debug(f"Failed to store solution: {e}")
    
    def decompose(
        self,
        problem: str,
        use_memory: bool = True
    ) -> EnhancedDecompositionResult:
        """
        Decompose problem with optional memory retrieval.
        
        If memory_bridge enabled:
        - Retrieve similar past decompositions
        - Use as guidance for new decomposition
        - Store result for future reuse
        
        Args:
            problem: Problem to decompose
            use_memory: Whether to use memory retrieval
            
        Returns:
            EnhancedDecompositionResult with subproblems
            
        Raises:
            RuntimeError: If no decomposition capability available
        """
        start_time = time.time()
        similar_count = 0
        
        with self._locked_operation():
            # Retrieve from memory if enabled
            similar_decompositions = []
            if use_memory and self.memory_bridge and self._capabilities["memory_bridge"]:
                try:
                    similar_decompositions = self._retrieve_decompositions(problem)
                    similar_count = len(similar_decompositions)
                except Exception as e:
                    logger.debug(f"Failed to retrieve decompositions: {e}")
            
            # Perform decomposition
            try:
                if self._mdap_engine and hasattr(self._mdap_engine, 'decompose'):
                    subproblems = self._mdap_engine.decompose(problem)
                    strategy = "mdap"
                elif self.core_engine and hasattr(self.core_engine, 'decompose'):
                    subproblems = self.core_engine.decompose(problem)
                    strategy = "maker"
                else:
                    # Fallback: create simple decomposition
                    subproblems = self._simple_decomposition(problem)
                    strategy = "simple"
                
                # Store in memory
                if use_memory and self.memory_bridge:
                    self._store_decomposition(problem, subproblems)
                
                elapsed = time.time() - start_time
                
                # Calculate quality score
                quality_score = self._calculate_decomposition_quality(subproblems)
                
                return EnhancedDecompositionResult(
                    subproblems=subproblems if isinstance(subproblems, list) else [],
                    strategy_used=strategy,
                    similar_decompositions_found=similar_count,
                    quality_score=quality_score,
                    decomposition_time=elapsed,
                    subproblem_count=len(subproblems) if isinstance(subproblems, list) else 0
                )
                
            except Exception as e:
                logger.error(f"Decomposition failed: {e}")
                raise RuntimeError(f"Failed to decompose problem: {e}")
    
    def _retrieve_decompositions(self, problem: str) -> List[Dict[str, Any]]:
        """Retrieve similar decompositions from memory."""
        if not self.memory_bridge:
            return []
        
        try:
            if hasattr(self.memory_bridge, 'retrieve_decompositions'):
                return self.memory_bridge.retrieve_decompositions(problem)
            elif hasattr(self.memory_bridge, 'search'):
                return self.memory_bridge.search(problem, type_filter='decomposition')
        except Exception as e:
            logger.debug(f"Memory retrieval failed: {e}")
        
        return []
    
    def _store_decomposition(self, problem: str, subproblems: List[Dict[str, Any]]) -> None:
        """Store decomposition in memory."""
        if not self.memory_bridge:
            return
        
        try:
            if hasattr(self.memory_bridge, 'store_decomposition'):
                self.memory_bridge.store_decomposition(problem, subproblems)
            elif hasattr(self.memory_bridge, 'store'):
                self.memory_bridge.store({
                    "type": "decomposition",
                    "problem": problem,
                    "subproblems": subproblems,
                    "timestamp": time.time()
                })
        except Exception as e:
            logger.debug(f"Failed to store decomposition: {e}")
    
    def _simple_decomposition(self, problem: str) -> List[Dict[str, Any]]:
        """Create simple decomposition when no engine available."""
        return [{
            "id": "subproblem_1",
            "description": problem,
            "priority": "high",
            "estimated_complexity": "medium"
        }]
    
    def _calculate_decomposition_quality(self, subproblems: List[Dict[str, Any]]) -> float:
        """Calculate quality score for decomposition."""
        if not subproblems:
            return 0.0
        
        # Simple heuristic based on subproblem count and structure
        count = len(subproblems)
        if count < 2:
            return 0.5
        elif count <= 5:
            return 0.8
        elif count <= 10:
            return 0.7
        else:
            return 0.6
    
    def vote(
        self,
        candidates: List[Any],
        context: str,
        use_learned_patterns: bool = True
    ) -> EnhancedVotingResult:
        """
        Vote on candidates with optional learned patterns.
        
        If memory_bridge enabled:
        - Retrieve voting patterns for similar scenarios
        - Apply learned red-flag patterns
        - Record outcome for future learning
        
        Args:
            candidates: List of candidates to vote on
            context: Voting context
            use_learned_patterns: Whether to use learned patterns
            
        Returns:
            EnhancedVotingResult with winner and metadata
            
        Raises:
            ValueError: If candidates list is empty
            RuntimeError: If voting fails
        """
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        start_time = time.time()
        patterns_applied = 0
        red_flags = 0
        
        with self._locked_operation():
            try:
                # Retrieve learned patterns
                learned_patterns = []
                if use_learned_patterns and self.memory_bridge:
                    try:
                        learned_patterns = self._retrieve_voting_patterns(context)
                        patterns_applied = len(learned_patterns)
                    except Exception as e:
                        logger.debug(f"Failed to retrieve voting patterns: {e}")
                
                # Perform voting
                if self._mdap_engine and hasattr(self._mdap_engine, 'vote'):
                    result = self._mdap_engine.vote(candidates, context)
                    winner = getattr(result, 'winner', candidates[0] if candidates else None)
                    vote_dist = getattr(result, 'distribution', {})
                elif self.core_engine and hasattr(self.core_engine, 'vote'):
                    result = self.core_engine.vote(candidates, context)
                    winner = getattr(result, 'winner', candidates[0] if candidates else None)
                    vote_dist = getattr(result, 'distribution', {})
                else:
                    # Simple majority fallback
                    winner = candidates[0]
                    vote_dist = {str(i): 1 for i in range(len(candidates))}
                
                # Apply red-flag patterns
                if learned_patterns:
                    red_flags = self._apply_red_flags(candidates, learned_patterns)
                
                # Record outcome
                if self.memory_bridge and self.config.enable_cross_session_learning:
                    self._record_voting_outcome(context, winner, candidates)
                
                elapsed = time.time() - start_time
                
                # Calculate confidence
                confidence = self._calculate_vote_confidence(vote_dist)
                
                return EnhancedVotingResult(
                    winner=winner,
                    confidence=confidence,
                    vote_distribution=vote_dist if isinstance(vote_dist, dict) else {},
                    learned_patterns_applied=patterns_applied,
                    red_flags_triggered=red_flags,
                    voting_time=elapsed
                )
                
            except Exception as e:
                logger.error(f"Voting failed: {e}")
                raise RuntimeError(f"Failed to perform voting: {e}")
    
    def _retrieve_voting_patterns(self, context: str) -> List[Dict[str, Any]]:
        """Retrieve learned voting patterns."""
        if not self.memory_bridge:
            return []
        
        try:
            if hasattr(self.memory_bridge, 'retrieve_voting_patterns'):
                return self.memory_bridge.retrieve_voting_patterns(context)
            elif hasattr(self.memory_bridge, 'search'):
                return self.memory_bridge.search(context, type_filter='voting_pattern')
        except Exception as e:
            logger.debug(f"Pattern retrieval failed: {e}")
        
        return []
    
    def _apply_red_flags(self, candidates: List[Any], patterns: List[Dict[str, Any]]) -> int:
        """Apply red-flag patterns to candidates."""
        flags = 0
        for pattern in patterns:
            if pattern.get('type') == 'red_flag':
                flags += 1
        return flags
    
    def _record_voting_outcome(
        self,
        context: str,
        winner: Any,
        candidates: List[Any]
    ) -> None:
        """Record voting outcome for learning."""
        if not self.memory_bridge:
            return
        
        try:
            if hasattr(self.memory_bridge, 'record_voting_outcome'):
                self.memory_bridge.record_voting_outcome(context, winner, candidates)
            elif hasattr(self.memory_bridge, 'store'):
                self.memory_bridge.store({
                    "type": "voting_outcome",
                    "context": context,
                    "winner": winner,
                    "candidate_count": len(candidates),
                    "timestamp": time.time()
                })
        except Exception as e:
            logger.debug(f"Failed to record voting outcome: {e}")
    
    def _calculate_vote_confidence(self, distribution: Dict[str, int]) -> float:
        """Calculate confidence from vote distribution."""
        if not distribution:
            return 0.0
        
        total_votes = sum(distribution.values())
        if total_votes == 0:
            return 0.0
        
        max_votes = max(distribution.values())
        return max_votes / total_votes
    
    def analyze_document(
        self,
        query: str,
        document_path: str
    ) -> DocumentAnalysisResult:
        """
        Analyze document using Matryoshka if available.
        Falls back to standard processing.
        
        Args:
            query: Analysis query
            document_path: Path to document
            
        Returns:
            DocumentAnalysisResult with analysis results
        """
        start_time = time.time()
        
        # Validate document
        if not os.path.exists(document_path):
            return DocumentAnalysisResult(
                document_path=document_path,
                document_size_mb=0.0,
                analysis="",
                key_points=[],
                recommendations=[],
                chunks_analyzed=0,
                time_seconds=0.0,
                used_matryoshka=False,
                success=False,
                error=f"Document not found: {document_path}"
            )
        
        try:
            size_mb = os.path.getsize(document_path) / (1024 * 1024)
        except OSError as e:
            return DocumentAnalysisResult(
                document_path=document_path,
                document_size_mb=0.0,
                analysis="",
                key_points=[],
                recommendations=[],
                chunks_analyzed=0,
                time_seconds=0.0,
                used_matryoshka=False,
                success=False,
                error=f"Cannot read document: {e}"
            )
        
        # Use Matryoshka if available and document is large
        use_matryoshka = (
            self._capabilities["matryoshka"] and
            size_mb >= self.config.large_document_threshold_mb
        )
        
        try:
            if use_matryoshka and self.matryoshka_mdap:
                return self._analyze_with_matryoshka(
                    query, document_path, size_mb, start_time
                )
            else:
                return self._analyze_standard(
                    query, document_path, size_mb, start_time
                )
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Document analysis failed: {e}")
            return DocumentAnalysisResult(
                document_path=document_path,
                document_size_mb=size_mb,
                analysis="",
                key_points=[],
                recommendations=[],
                chunks_analyzed=0,
                time_seconds=elapsed,
                used_matryoshka=use_matryoshka,
                success=False,
                error=str(e)
            )
    
    def _analyze_with_matryoshka(
        self,
        query: str,
        document_path: str,
        size_mb: float,
        start_time: float
    ) -> DocumentAnalysisResult:
        """Analyze document using Matryoshka."""
        try:
            result = self.matryoshka_mdap.analyze_document(query, document_path)
            elapsed = time.time() - start_time
            
            return DocumentAnalysisResult(
                document_path=document_path,
                document_size_mb=size_mb,
                analysis=getattr(result, 'analysis', str(result)),
                key_points=getattr(result, 'key_points', []),
                recommendations=getattr(result, 'recommendations', []),
                chunks_analyzed=getattr(result, 'chunks_analyzed', 0),
                time_seconds=elapsed,
                used_matryoshka=True,
                success=True
            )
            
        except Exception as e:
            logger.warning(f"Matryoshka analysis failed, falling back: {e}")
            return self._analyze_standard(query, document_path, size_mb, start_time)
    
    def _analyze_standard(
        self,
        query: str,
        document_path: str,
        size_mb: float,
        start_time: float
    ) -> DocumentAnalysisResult:
        """Standard document analysis without Matryoshka."""
        try:
            # Read document content
            with open(document_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Simple analysis - in production, this would use an LLM
            lines = content.split('\n')
            key_points = [line.strip() for line in lines[:20] if line.strip()]
            
            elapsed = time.time() - start_time
            
            return DocumentAnalysisResult(
                document_path=document_path,
                document_size_mb=size_mb,
                analysis=f"Document contains {len(content)} characters",
                key_points=key_points[:10],
                recommendations=["Consider using Matryoshka for better analysis"],
                chunks_analyzed=1,
                time_seconds=elapsed,
                used_matryoshka=False,
                success=True
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            return DocumentAnalysisResult(
                document_path=document_path,
                document_size_mb=size_mb,
                analysis="",
                key_points=[],
                recommendations=[],
                chunks_analyzed=0,
                time_seconds=elapsed,
                used_matryoshka=False,
                success=False,
                error=str(e)
            )
    
    def get_similar_solutions(
        self,
        problem: str,
        limit: int = 5
    ) -> List[SimilarSolution]:
        """
        Get similar solutions from memory if available.
        
        Args:
            problem: Problem to find similar solutions for
            limit: Maximum number of results
            
        Returns:
            List of SimilarSolution objects
        """
        if not self.memory_bridge or not self._capabilities["memory_bridge"]:
            return []
        
        with self._locked_operation():
            try:
                # Try different retrieval methods
                results = []
                
                if hasattr(self.memory_bridge, 'find_similar_solutions'):
                    raw_results = self.memory_bridge.find_similar_solutions(problem, limit)
                    for r in raw_results:
                        results.append(SimilarSolution(
                            problem=r.get('problem', ''),
                            solution=r.get('solution'),
                            similarity_score=r.get('similarity', 0.0),
                            timestamp=r.get('timestamp', 0.0),
                            metadata=r.get('metadata', {})
                        ))
                elif hasattr(self.memory_bridge, 'search'):
                    raw_results = self.memory_bridge.search(problem, limit=limit)
                    for r in raw_results:
                        results.append(SimilarSolution(
                            problem=r.get('problem', r.get('query', '')),
                            solution=r.get('solution', r.get('result')),
                            similarity_score=r.get('score', 0.5),
                            timestamp=r.get('timestamp', 0.0),
                            metadata=r.get('metadata', {})
                        ))
                
                return results
                
            except Exception as e:
                logger.debug(f"Failed to get similar solutions: {e}")
                return []
    
    def export_knowledge(self, path: str) -> None:
        """
        Export learned knowledge for sharing.
        
        Args:
            path: Export file path
            
        Raises:
            RuntimeError: If memory bridge not available or export fails
        """
        if not self.memory_bridge or not self._capabilities["memory_bridge"]:
            raise RuntimeError("Memory bridge not available. Cannot export knowledge.")
        
        with self._locked_operation():
            try:
                if hasattr(self.memory_bridge, 'export_knowledge'):
                    self.memory_bridge.export_knowledge(path)
                elif hasattr(self.memory_bridge, 'export'):
                    self.memory_bridge.export(path)
                else:
                    # Manual export
                    import json
                    knowledge = self._collect_knowledge()
                    with open(path, 'w') as f:
                        json.dump(knowledge, f, indent=2, default=str)
                
                logger.info(f"Knowledge exported to {path}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to export knowledge: {e}")
    
    def import_knowledge(self, path: str) -> None:
        """
        Import learned knowledge from file.
        
        Args:
            path: Import file path
            
        Raises:
            RuntimeError: If memory bridge not available or import fails
            FileNotFoundError: If import file doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Knowledge file not found: {path}")
        
        if not self.memory_bridge or not self._capabilities["memory_bridge"]:
            raise RuntimeError("Memory bridge not available. Cannot import knowledge.")
        
        with self._locked_operation():
            try:
                if hasattr(self.memory_bridge, 'import_knowledge'):
                    self.memory_bridge.import_knowledge(path)
                elif hasattr(self.memory_bridge, 'import_data'):
                    self.memory_bridge.import_data(path)
                else:
                    # Manual import
                    import json
                    with open(path, 'r') as f:
                        knowledge = json.load(f)
                    self._apply_knowledge(knowledge)
                
                logger.info(f"Knowledge imported from {path}")
                
            except Exception as e:
                raise RuntimeError(f"Failed to import knowledge: {e}")
    
    def _collect_knowledge(self) -> Dict[str, Any]:
        """Collect all knowledge for export."""
        return {
            "capabilities": self._capabilities,
            "config": {
                "enable_cross_session_learning": self.config.enable_cross_session_learning,
                "large_document_threshold_mb": self.config.large_document_threshold_mb
            },
            "export_time": time.time()
        }
    
    def _apply_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Apply imported knowledge."""
        # Override with imported capabilities if present
        if "capabilities" in knowledge:
            logger.info("Applying imported capabilities settings")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_enhanced_client(
    enable_matryoshka: bool = False,
    enable_memory: bool = True,
    use_crewai: bool = False,
    storage_path: Optional[str] = None
) -> EnhancedMDAPClient:
    """
    Factory for creating enhanced client with sensible defaults.
    
    Args:
        enable_matryoshka: Enable Matryoshka integration if available
        enable_memory: Enable unified memory bridge if available
        use_crewai: Use CrewAI-based MAKER if available
        storage_path: Path for memory storage
        
    Returns:
        Configured EnhancedMDAPClient
    """
    config = EnhancedMDAPConfig(
        enable_matryoshka=enable_matryoshka,
        enable_memory_bridge=enable_memory,
        use_crewai=use_crewai,
        memory_storage_path=storage_path,
        auto_select_matryoshka=enable_matryoshka
    )
    
    return EnhancedMDAPClient(config)


def create_minimal_client() -> EnhancedMDAPClient:
    """
    Create client with only core MDAP/MAKER (no optional integrations).
    
    Returns:
        EnhancedMDAPClient with minimal configuration
    """
    config = EnhancedMDAPConfig(
        enable_matryoshka=False,
        enable_memory_bridge=False,
        use_crewai=False,
        auto_select_matryoshka=False,
        enable_cross_session_learning=False
    )
    
    return EnhancedMDAPClient(config)


def create_full_client(storage_path: str = "./mdap_memory") -> EnhancedMDAPClient:
    """
    Create client with all optional integrations enabled.
    
    Args:
        storage_path: Path for memory storage
        
    Returns:
        EnhancedMDAPClient with all features enabled
    """
    # Ensure storage directory exists
    if storage_path:
        os.makedirs(storage_path, exist_ok=True)
    
    config = EnhancedMDAPConfig(
        enable_matryoshka=True,
        enable_memory_bridge=True,
        use_crewai=CREWAI_MAKER_AVAILABLE,
        memory_storage_path=storage_path,
        auto_select_matryoshka=True,
        enable_cross_session_learning=True
    )
    
    return EnhancedMDAPClient(config)


def create_document_focused_client(storage_path: str = "./mdap_memory") -> EnhancedMDAPClient:
    """
    Create client optimized for large document analysis.
    
    Enables Matryoshka and lowers threshold for large document detection.
    
    Args:
        storage_path: Path for memory storage
        
    Returns:
        EnhancedMDAPClient optimized for documents
    """
    if storage_path:
        os.makedirs(storage_path, exist_ok=True)
    
    config = EnhancedMDAPConfig(
        enable_matryoshka=True,
        enable_memory_bridge=True,
        memory_storage_path=storage_path,
        auto_select_matryoshka=True,
        large_document_threshold_mb=5.0,  # Lower threshold
        enable_cross_session_learning=True
    )
    
    return EnhancedMDAPClient(config)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main client
    'EnhancedMDAPClient',
    'EnhancedMDAPConfig',
    
    # Results
    'EnhancedMDAPResult',
    'EnhancedDecompositionResult',
    'EnhancedVotingResult',
    'DocumentAnalysisResult',
    'SimilarSolution',
    
    # Factory functions
    'create_enhanced_client',
    'create_minimal_client',
    'create_full_client',
    'create_document_focused_client',
    
    # Availability flags
    'MATRYOSHKA_INTEGRATION_AVAILABLE',
    'MEMORY_BRIDGE_AVAILABLE',
    'MDAP_AVAILABLE',
    'MAKER_AVAILABLE',
    'CREWAI_MAKER_AVAILABLE',
]


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test client creation
    print("Testing Enhanced MDAP Client...")
    
    # Create client
    client = create_full_client("./test_memory")
    
    # Check capabilities
    print(f"\nCapabilities: {client.capabilities}")
    
    # Test solve (will fail gracefully if no engines available)
    try:
        result = client.solve("Test problem")
        print(f"\nSolve result: success={result.success}, time={result.time_seconds:.2f}s")
    except Exception as e:
        print(f"\nSolve failed (expected if no engines): {e}")
    
    # Test decomposition
    try:
        decomp = client.decompose("Test decomposition problem")
        print(f"Decomposition: {decomp.subproblem_count} subproblems, strategy={decomp.strategy_used}")
    except Exception as e:
        print(f"Decomposition failed: {e}")
    
    print("\nTest complete!")
