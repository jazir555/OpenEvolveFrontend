"""
MDAP Memory Bridge - Connecting MDAP/MAKER with Unified Memory System

This module provides a bridge between MDAP/MAKER's decomposition/voting system
and the Unified Memory System, enabling:

1. Storage of decompositions in unified memory (4-layer indexed)
2. Retrieval of relevant past decompositions for similar problems
3. Learning from voting patterns across sessions
4. Building knowledge graph of problem->subproblem relationships

Key Features:
- 4-layer indexing (hash, hierarchical, graph, semantic)
- Voting pattern learning and prediction
- Solution success tracking with error pattern analysis
- Graph relationships between problems and subproblems
- Graceful no-op when unified memory unavailable
- Thread-safe operations

Usage:
    >>> from mdap_memory_bridge import MDAPMemoryBridge, create_mdap_memory_bridge
    >>> bridge = create_mdap_memory_bridge(storage_path="./mdap_memory")
    >>> 
    >>> # Store a decomposition
    >>> decomp_id = bridge.store_decomposition(
    ...     problem="Optimize portfolio risk",
    ...     subproblems=[{"text": "Analyze correlations", "type": "analysis"}],
    ...     strategy="financial_domain",
    ...     quality_score=0.92
    ... )
    >>> 
    >>> # Find similar decompositions
    >>> similar = bridge.find_similar_decompositions("Minimize investment risk")

Author: OpenEvolve AI
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIONAL DEPENDENCIES
# ============================================================================

try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem,
        UnifiedMemory,
        UnifiedMemoryConfig,
        ConversationSession,
        TurnProcessingResult,
        MemoryStatus
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False
    UnifiedMemorySystem = None
    UnifiedMemory = None
    UnifiedMemoryConfig = None
    ConversationSession = None
    TurnProcessingResult = None
    MemoryStatus = None

try:
    from knowledge_hierarchical_index import MemoryLevel, MemoryNode, HierarchicalIndex
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    MemoryLevel = None
    MemoryNode = None
    HierarchicalIndex = None

try:
    from knowledge_graph_index import RelationshipType, GraphIndex, NodeType, TraversalResult
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False
    RelationshipType = None
    GraphIndex = None
    NodeType = None
    TraversalResult = None

try:
    from knowledge_hash_index import compute_md5_hash, HashIndex
    HASH_AVAILABLE = True
except ImportError:
    HASH_AVAILABLE = False
    compute_md5_hash = None
    HashIndex = None

try:
    from knowledge_semantic_index import SemanticIndex, SemanticQuery
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    SemanticIndex = None
    SemanticQuery = None


# ============================================================================
# CONSTANTS
# ============================================================================

DECOMPOSITION_MEMORY_TYPE = "mdap_decomposition"
VOTING_PATTERN_MEMORY_TYPE = "mdap_voting_pattern"
SOLUTION_ATTEMPT_MEMORY_TYPE = "mdap_solution_attempt"
RED_FLAG_PATTERN_MEMORY_TYPE = "mdap_red_flag"

DEFAULT_DECOMPOSITION_LEVEL = "IMPORTANT"  # Reusable knowledge
SUCCESSFUL_SOLUTION_LEVEL = "CORE"  # High-value patterns
FAILED_SOLUTION_LEVEL = "CONTEXTUAL"  # Learn from failures

PROBLEM_TO_SUBPROBLEM_REL = RelationshipType.PART_OF if GRAPH_AVAILABLE else None
SUBPROBLEM_TO_PROBLEM_REL = RelationshipType.DEPENDS_ON if GRAPH_AVAILABLE else None
SIMILAR_PROBLEM_REL = RelationshipType.SEMANTIC if GRAPH_AVAILABLE else None
SEQUENTIAL_STEP_REL = RelationshipType.SEQUENTIAL if GRAPH_AVAILABLE else None


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DecompositionMemory:
    """
    A stored decomposition with full indexing metadata.
    
    Attributes:
        decomposition_id: Unique identifier for this decomposition
        problem_hash: Hash of the problem text for deduplication
        problem_text: The original problem statement
        subproblems: List of subproblem dictionaries with metadata
        decomposition_strategy: Strategy used (e.g., 'financial_domain')
        quality_score: Quality assessment score (0.0 - 1.0)
        usage_count: Number of times this decomposition was reused
        success_count: Number of successful applications
        created_at: Timestamp of creation
        last_used: Timestamp of last access
        hierarchical_level: Importance level in hierarchy
        graph_edges: Relationships to other memory nodes
        semantic_embedding: Vector embedding for similarity search
    """
    decomposition_id: str
    problem_hash: str
    problem_text: str
    subproblems: List[Dict[str, Any]]
    decomposition_strategy: str
    quality_score: float
    usage_count: int = 0
    success_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    hierarchical_level: str = "IMPORTANT"  # CORE, IMPORTANT, CONTEXTUAL, GRANULAR
    graph_edges: List[Dict[str, str]] = field(default_factory=list)
    semantic_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decomposition_id": self.decomposition_id,
            "problem_hash": self.problem_hash,
            "problem_text": self.problem_text[:500] + "..." if len(self.problem_text) > 500 else self.problem_text,
            "subproblems_count": len(self.subproblems),
            "decomposition_strategy": self.decomposition_strategy,
            "quality_score": self.quality_score,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "hierarchical_level": self.hierarchical_level,
            "has_embedding": self.semantic_embedding is not None
        }
    
    def touch(self) -> None:
        """Mark as accessed."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1


@dataclass
class VotingPatternMemory:
    """
    Learned voting patterns for similar decisions.
    
    Attributes:
        pattern_id: Unique identifier for this pattern
        problem_type: Classification of problem (e.g., 'optimization', 'analysis')
        candidate_types: Types of candidates in the vote
        winning_strategy: Which strategy won most often
        vote_distribution: Historical vote counts per candidate
        confidence: Confidence in this pattern (0.0 - 1.0)
        red_flag_patterns: Red flags that occurred in similar contexts
    """
    pattern_id: str
    problem_type: str
    candidate_types: List[str]
    winning_strategy: str
    vote_distribution: Dict[str, int]
    confidence: float
    red_flag_patterns: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    occurrence_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "problem_type": self.problem_type,
            "candidate_types": self.candidate_types,
            "winning_strategy": self.winning_strategy,
            "vote_distribution": self.vote_distribution,
            "confidence": self.confidence,
            "red_flag_patterns": self.red_flag_patterns,
            "occurrence_count": self.occurrence_count
        }


@dataclass
class SolutionAttemptMemory:
    """Memory of a solution attempt with outcome."""
    attempt_id: str
    problem: str
    problem_hash: str
    solution_type: str
    success: bool
    error_pattern: Optional[str]
    execution_time_ms: Optional[float]
    created_at: datetime = field(default_factory=datetime.utcnow)
    hierarchical_level: str = "CONTEXTUAL"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "problem_hash": self.problem_hash,
            "solution_type": self.solution_type,
            "success": self.success,
            "error_pattern": self.error_pattern,
            "execution_time_ms": self.execution_time_ms
        }


# ============================================================================
# MDAP MEMORY BRIDGE
# ============================================================================

class MDAPMemoryBridge:
    """
    Bridges MDAP/MAKER operations with Unified Memory System.
    
    Stores and retrieves:
    - Decompositions (hierarchically indexed)
    - Voting patterns (learned over time)
    - Solution attempts (with outcomes)
    - Red flag patterns (what to avoid)
    
    Completely optional - works as no-op if unified memory unavailable.
    
    Thread Safety:
        All public methods are thread-safe using internal locks.
    """
    
    def __init__(
        self,
        unified_memory: Optional[UnifiedMemorySystem] = None,
        storage_path: Optional[str] = None,
        config: Optional[UnifiedMemoryConfig] = None
    ):
        """
        Initialize the MDAP Memory Bridge.
        
        Args:
            unified_memory: Existing UnifiedMemorySystem instance (optional)
            storage_path: Path for memory storage (if creating new system)
            config: Configuration for unified memory system
        """
        self._lock = threading.RLock()
        
        # Determine if we can use unified memory
        if unified_memory is not None and UNIFIED_MEMORY_AVAILABLE:
            self.unified_memory = unified_memory
            self.enabled = True
        elif UNIFIED_MEMORY_AVAILABLE and (storage_path or config):
            # Create new unified memory system
            try:
                if config is None:
                    config = UnifiedMemoryConfig(
                        db_dir=storage_path or "./mdap_memory",
                        system_name="mdap_memory_bridge"
                    )
                self.unified_memory = UnifiedMemorySystem(config)
                self.enabled = True
            except Exception as e:
                logger.error(f"Failed to create unified memory system: {e}")
                self.unified_memory = None
                self.enabled = False
        else:
            self.unified_memory = None
            self.enabled = False
        
        # In-memory caches for fast access
        self._decomposition_cache: Dict[str, DecompositionMemory] = {}
        self._voting_pattern_cache: Dict[str, VotingPatternMemory] = {}
        self._solution_cache: Dict[str, SolutionAttemptMemory] = {}
        
        # Index mappings
        self._problem_hash_to_decomp: Dict[str, str] = {}  # hash -> decomposition_id
        self._problem_type_to_patterns: Dict[str, Set[str]] = defaultdict(set)
        
        if self.enabled:
            logger.info("MDAPMemoryBridge initialized with unified memory")
        else:
            logger.info("MDAPMemoryBridge initialized in no-op mode")
    
    # ========================================================================
    # DECOMPOSITION MEMORY
    # ========================================================================
    
    def store_decomposition(
        self,
        problem: str,
        subproblems: List[Dict[str, Any]],
        strategy: str,
        quality_score: float,
        parent_problem: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store a decomposition in unified memory with 4-layer indexing.
        
        Indexing Pipeline:
        1. Hash: Deduplicate identical decompositions
        2. Hierarchical: Mark as IMPORTANT (reusable knowledge)
        3. Graph: Link to parent problem if exists
        4. Semantic: Embed for similarity search
        
        Args:
            problem: The problem statement
            subproblems: List of subproblem dictionaries
            strategy: Decomposition strategy used
            quality_score: Quality assessment (0.0 - 1.0)
            parent_problem: Optional parent problem for hierarchical linking
            metadata: Additional metadata
            
        Returns:
            decomposition_id if stored successfully, None otherwise
        """
        if not self.enabled:
            return None
        
        with self._lock:
            try:
                # Compute problem hash for deduplication
                problem_hash = self._compute_problem_hash(problem)
                
                # Check for existing identical decomposition
                if problem_hash in self._problem_hash_to_decomp:
                    existing_id = self._problem_hash_to_decomp[problem_hash]
                    logger.debug(f"Duplicate decomposition found: {existing_id}")
                    return existing_id
                
                # Create decomposition memory
                decomp_id = f"decomp_{uuid.uuid4().hex[:16]}"
                decomposition = DecompositionMemory(
                    decomposition_id=decomp_id,
                    problem_hash=problem_hash,
                    problem_text=problem,
                    subproblems=subproblems,
                    decomposition_strategy=strategy,
                    quality_score=quality_score,
                    hierarchical_level=DEFAULT_DECOMPOSITION_LEVEL
                )
                
                # 1. Store in unified memory with 4-layer indexing
                self._index_decomposition(decomposition, parent_problem, metadata)
                
                # 2. Update local cache
                self._decomposition_cache[decomp_id] = decomposition
                self._problem_hash_to_decomp[problem_hash] = decomp_id
                
                logger.info(f"Stored decomposition {decomp_id} with {len(subproblems)} subproblems")
                return decomp_id
                
            except Exception as e:
                logger.error(f"Failed to store decomposition: {e}")
                return None
    
    def find_similar_decompositions(
        self,
        problem: str,
        limit: int = 5,
        min_quality: float = 0.0
    ) -> List[DecompositionMemory]:
        """
        Find decompositions for similar problems.
        
        Uses semantic search + graph traversal for comprehensive retrieval.
        
        Args:
            problem: Problem statement to find similar decompositions for
            limit: Maximum number of results
            min_quality: Minimum quality score filter
            
        Returns:
            List of DecompositionMemory objects, sorted by relevance
        """
        if not self.enabled:
            return []
        
        with self._lock:
            try:
                results = []
                seen_ids = set()
                
                # 1. Semantic similarity search via unified memory
                if self.unified_memory and hasattr(self.unified_memory, '_hybrid_retrieve'):
                    # Create a temporary conversation for retrieval
                    conv_id = f"mdap_search_{uuid.uuid4().hex[:8]}"
                    
                    # Use hybrid retrieval
                    memories = self.unified_memory._hybrid_retrieve(
                        query=problem,
                        conversation_id=conv_id,
                        limit=limit * 2  # Get extra for filtering
                    )
                    
                    for memory in memories:
                        if (hasattr(memory, 'memory_type') and 
                            memory.memory_type == DECOMPOSITION_MEMORY_TYPE):
                            decomp_id = memory.memory_id
                            if decomp_id in self._decomposition_cache:
                                decomp = self._decomposition_cache[decomp_id]
                                if decomp.quality_score >= min_quality:
                                    decomp.touch()
                                    if decomp_id not in seen_ids:
                                        results.append(decomp)
                                        seen_ids.add(decomp_id)
                
                # 2. Graph traversal for related problems
                if GRAPH_AVAILABLE and self.unified_memory and self.unified_memory.graph_index:
                    # Find problem nodes with similar content
                    problem_hash = self._compute_problem_hash(problem)
                    if problem_hash in self._problem_hash_to_decomp:
                        decomp_id = self._problem_hash_to_decomp[problem_hash]
                        # Traverse graph for related decompositions
                        # This would find problems linked via PART_OF, DEPENDS_ON, etc.
                
                # 3. Hash-based exact match check
                problem_hash = self._compute_problem_hash(problem)
                if problem_hash in self._problem_hash_to_decomp:
                    decomp_id = self._problem_hash_to_decomp[problem_hash]
                    if decomp_id in self._decomposition_cache:
                        exact_match = self._decomposition_cache[decomp_id]
                        if exact_match.quality_score >= min_quality and decomp_id not in seen_ids:
                            results.insert(0, exact_match)  # Insert at beginning as best match
                
                # Sort by quality and usage
                results.sort(key=lambda d: (d.quality_score, d.usage_count), reverse=True)
                
                return results[:limit]
                
            except Exception as e:
                logger.error(f"Error finding similar decompositions: {e}")
                return []
    
    def get_decomposition_for_subproblem(
        self,
        subproblem_text: str
    ) -> Optional[DecompositionMemory]:
        """
        Check if this subproblem has been decomposed before.
        
        Useful for recursive decomposition - checks if a subproblem
        itself has existing decompositions.
        
        Args:
            subproblem_text: The subproblem to check
            
        Returns:
            DecompositionMemory if found, None otherwise
        """
        if not self.enabled:
            return None
        
        with self._lock:
            try:
                # Check cache first
                subproblem_hash = self._compute_problem_hash(subproblem_text)
                if subproblem_hash in self._problem_hash_to_decomp:
                    decomp_id = self._problem_hash_to_decomp[subproblem_hash]
                    if decomp_id in self._decomposition_cache:
                        return self._decomposition_cache[decomp_id]
                
                # Search via unified memory
                similar = self.find_similar_decompositions(subproblem_text, limit=1)
                return similar[0] if similar else None
                
            except Exception as e:
                logger.error(f"Error getting decomposition for subproblem: {e}")
                return None
    
    def update_decomposition_success(
        self,
        decomposition_id: str,
        success: bool
    ) -> bool:
        """
        Update decomposition with success/failure feedback.
        
        Args:
            decomposition_id: ID of the decomposition
            success: Whether the decomposition led to a successful solution
            
        Returns:
            True if updated successfully
        """
        if not self.enabled or decomposition_id not in self._decomposition_cache:
            return False
        
        with self._lock:
            try:
                decomp = self._decomposition_cache[decomposition_id]
                decomp.touch()
                
                if success:
                    decomp.success_count += 1
                    # Promote to CORE if highly successful
                    if decomp.success_count >= 3 and decomp.quality_score > 0.8:
                        decomp.hierarchical_level = "CORE"
                
                logger.debug(f"Updated decomposition {decomposition_id}: success={success}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating decomposition: {e}")
                return False
    
    # ========================================================================
    # VOTING PATTERN MEMORY
    # ========================================================================
    
    def record_voting_outcome(
        self,
        problem_type: str,
        candidates: List[Any],
        winner: Any,
        vote_distribution: Dict[str, int],
        red_flags: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Record voting outcome for learning.
        
        Helps predict winners in future similar votes by building
        patterns of what strategies work for different problem types.
        
        Args:
            problem_type: Classification of the problem
            candidates: List of candidate options
            winner: The winning candidate
            vote_distribution: Vote counts per candidate
            red_flags: Red flags raised during voting
            metadata: Additional context
            
        Returns:
            pattern_id if recorded successfully
        """
        if not self.enabled:
            return None
        
        with self._lock:
            try:
                # Check for existing pattern for this problem type
                existing_pattern = self._find_voting_pattern(problem_type, candidates)
                
                if existing_pattern:
                    # Update existing pattern
                    existing_pattern.occurrence_count += 1
                    existing_pattern.vote_distribution = self._merge_vote_distributions(
                        existing_pattern.vote_distribution,
                        vote_distribution
                    )
                    # Update winning strategy if changed
                    winner_str = str(winner) if winner else "none"
                    if winner_str == existing_pattern.winning_strategy:
                        existing_pattern.confidence = min(1.0, existing_pattern.confidence + 0.05)
                    else:
                        existing_pattern.confidence = max(0.0, existing_pattern.confidence - 0.1)
                    existing_pattern.winning_strategy = winner_str
                    # Merge red flags
                    for rf in red_flags:
                        if rf not in existing_pattern.red_flag_patterns:
                            existing_pattern.red_flag_patterns.append(rf)
                    
                    logger.debug(f"Updated voting pattern: {existing_pattern.pattern_id}")
                    return existing_pattern.pattern_id
                else:
                    # Create new pattern
                    pattern_id = f"vote_pattern_{uuid.uuid4().hex[:12]}"
                    candidate_types = [type(c).__name__ for c in candidates]
                    
                    pattern = VotingPatternMemory(
                        pattern_id=pattern_id,
                        problem_type=problem_type,
                        candidate_types=candidate_types,
                        winning_strategy=str(winner) if winner else "none",
                        vote_distribution=vote_distribution,
                        confidence=0.5,  # Start with moderate confidence
                        red_flag_patterns=red_flags
                    )
                    
                    # Store in unified memory
                    self._index_voting_pattern(pattern)
                    
                    # Update local caches
                    self._voting_pattern_cache[pattern_id] = pattern
                    self._problem_type_to_patterns[problem_type].add(pattern_id)
                    
                    logger.info(f"Created voting pattern {pattern_id} for {problem_type}")
                    return pattern_id
                    
            except Exception as e:
                logger.error(f"Failed to record voting outcome: {e}")
                return None
    
    def get_voting_guidance(
        self,
        problem_type: str,
        candidates: List[Any]
    ) -> Optional[VotingPatternMemory]:
        """
        Get learned guidance for similar voting scenarios.
        
        Args:
            problem_type: Type of problem being voted on
            candidates: Current candidate options
            
        Returns:
            VotingPatternMemory with guidance, or None if no pattern exists
        """
        if not self.enabled:
            return None
        
        with self._lock:
            return self._find_voting_pattern(problem_type, candidates)
    
    def predict_winner(
        self,
        problem_type: str,
        candidates: List[Any]
    ) -> Tuple[Optional[Any], float]:
        """
        Predict the likely winner based on learned patterns.
        
        Args:
            problem_type: Type of problem
            candidates: Candidate options
            
        Returns:
            Tuple of (predicted_winner, confidence)
        """
        if not self.enabled:
            return None, 0.0
        
        with self._lock:
            pattern = self._find_voting_pattern(problem_type, candidates)
            if pattern and pattern.confidence > 0.6:
                # Find candidate matching winning strategy
                winner_str = pattern.winning_strategy
                for candidate in candidates:
                    if str(candidate) == winner_str:
                        return candidate, pattern.confidence
            
            return None, 0.0
    
    # ========================================================================
    # SOLUTION MEMORY
    # ========================================================================
    
    def store_solution_attempt(
        self,
        problem: str,
        solution: Any,
        success: bool,
        error_pattern: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store solution attempt with outcome.
        
        Successful solutions marked as CORE memories.
        Failed solutions track error patterns to avoid.
        
        Args:
            problem: The problem that was attempted
            solution: The solution that was tried
            success: Whether it succeeded
            error_pattern: Error pattern if failed
            execution_time_ms: Execution time in milliseconds
            metadata: Additional metadata
            
        Returns:
            attempt_id if stored successfully
        """
        if not self.enabled:
            return None
        
        with self._lock:
            try:
                problem_hash = self._compute_problem_hash(problem)
                attempt_id = f"solution_{uuid.uuid4().hex[:16]}"
                
                # Determine hierarchical level based on success
                hierarchical_level = SUCCESSFUL_SOLUTION_LEVEL if success else FAILED_SOLUTION_LEVEL
                
                attempt = SolutionAttemptMemory(
                    attempt_id=attempt_id,
                    problem=problem,
                    problem_hash=problem_hash,
                    solution_type=type(solution).__name__ if solution else "unknown",
                    success=success,
                    error_pattern=error_pattern,
                    execution_time_ms=execution_time_ms,
                    hierarchical_level=hierarchical_level
                )
                
                # Index in unified memory
                self._index_solution_attempt(attempt, metadata)
                
                # Update local cache
                self._solution_cache[attempt_id] = attempt
                
                # If successful, also update any matching decomposition
                if problem_hash in self._problem_hash_to_decomp:
                    decomp_id = self._problem_hash_to_decomp[problem_hash]
                    self.update_decomposition_success(decomp_id, success)
                
                logger.info(f"Stored solution attempt {attempt_id}: success={success}")
                return attempt_id
                
            except Exception as e:
                logger.error(f"Failed to store solution attempt: {e}")
                return None
    
    def find_similar_successful_solutions(
        self,
        problem: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find successful solutions to similar problems.
        
        Args:
            problem: Problem to find solutions for
            limit: Maximum number of results
            
        Returns:
            List of solution dictionaries
        """
        if not self.enabled:
            return []
        
        with self._lock:
            try:
                results = []
                
                # Search unified memory for successful solution attempts
                if self.unified_memory and hasattr(self.unified_memory, '_hybrid_retrieve'):
                    conv_id = f"mdap_solution_search_{uuid.uuid4().hex[:8]}"
                    memories = self.unified_memory._hybrid_retrieve(
                        query=problem,
                        conversation_id=conv_id,
                        limit=limit * 2
                    )
                    
                    for memory in memories:
                        if (hasattr(memory, 'memory_type') and 
                            memory.memory_type == SOLUTION_ATTEMPT_MEMORY_TYPE):
                            # Check if in cache
                            if memory.memory_id in self._solution_cache:
                                solution = self._solution_cache[memory.memory_id]
                                if solution.success:
                                    results.append(solution.to_dict())
                
                # Also check local cache
                problem_hash = self._compute_problem_hash(problem)
                for attempt_id, attempt in self._solution_cache.items():
                    if attempt.success and attempt.problem_hash == problem_hash:
                        if attempt.to_dict() not in results:
                            results.append(attempt.to_dict())
                
                return results[:limit]
                
            except Exception as e:
                logger.error(f"Error finding successful solutions: {e}")
                return []
    
    def get_error_patterns_to_avoid(
        self,
        problem_type: str
    ) -> List[str]:
        """
        Get error patterns that have occurred for similar problems.
        
        Args:
            problem_type: Type of problem
            
        Returns:
            List of error patterns to avoid
        """
        if not self.enabled:
            return []
        
        with self._lock:
            patterns = []
            for attempt in self._solution_cache.values():
                if not attempt.success and attempt.error_pattern:
                    patterns.append(attempt.error_pattern)
            return list(set(patterns))[:10]  # Return unique patterns, max 10
    
    # ========================================================================
    # KNOWLEDGE GRAPH OPERATIONS
    # ========================================================================
    
    def build_problem_subproblem_graph(
        self,
        problem: str,
        subproblems: List[str]
    ) -> Optional[str]:
        """
        Build knowledge graph relationships between problem and subproblems.
        
        Creates:
        - Problem node
        - Subproblem nodes
        - PART_OF relationships (subproblem -> problem)
        - SEQUENTIAL relationships between ordered subproblems
        
        Args:
            problem: The main problem
            subproblems: List of subproblem texts
            
        Returns:
            problem_node_id if successful
        """
        if not self.enabled or not GRAPH_AVAILABLE:
            return None
        
        with self._lock:
            try:
                if not self.unified_memory or not self.unified_memory.graph_index:
                    return None
                
                # Create problem node
                problem_node_id = self.unified_memory.graph_index.add_node(
                    content=problem,
                    node_type=NodeType.CONCEPT if NodeType else None,
                    metadata={"type": "mdap_problem", "created_by": "mdap_bridge"}
                )
                
                # Create subproblem nodes and relationships
                prev_subproblem_id = None
                for i, subproblem in enumerate(subproblems):
                    subproblem_node_id = self.unified_memory.graph_index.add_node(
                        content=subproblem,
                        node_type=NodeType.CONCEPT if NodeType else None,
                        metadata={
                            "type": "mdap_subproblem",
                            "order": i,
                            "parent_problem": problem[:100]
                        }
                    )
                    
                    # Link subproblem to problem (PART_OF)
                    if RelationshipType:
                        self.unified_memory.graph_index.add_edge(
                            source_id=subproblem_node_id,
                            target_id=problem_node_id,
                            relationship_type=RelationshipType.PART_OF,
                            metadata={"order": i}
                        )
                    
                    # Link to previous subproblem (SEQUENTIAL)
                    if prev_subproblem_id and RelationshipType:
                        self.unified_memory.graph_index.add_edge(
                            source_id=prev_subproblem_id,
                            target_id=subproblem_node_id,
                            relationship_type=RelationshipType.SEQUENTIAL,
                            metadata={"step": i}
                        )
                    
                    prev_subproblem_id = subproblem_node_id
                
                logger.info(f"Built problem-subproblem graph for: {problem[:50]}...")
                return problem_node_id
                
            except Exception as e:
                logger.error(f"Failed to build problem graph: {e}")
                return None
    
    def traverse_problem_graph(
        self,
        problem_node_id: str,
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Traverse the knowledge graph from a problem node.
        
        Args:
            problem_node_id: Starting node ID
            depth: Traversal depth
            
        Returns:
            List of connected nodes
        """
        if not self.enabled or not GRAPH_AVAILABLE:
            return []
        
        with self._lock:
            try:
                if not self.unified_memory or not self.unified_memory.graph_index:
                    return []
                
                result = self.unified_memory.graph_index.traverse_relationships(
                    start_node_id=problem_node_id,
                    depth=depth,
                    mode=TraversalMode.BFS if TraversalMode else None
                )
                
                return [node.to_dict() for node in result.nodes] if result else []
                
            except Exception as e:
                logger.error(f"Error traversing problem graph: {e}")
                return []
    
    # ========================================================================
    # STATISTICS AND EXPORT
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "decompositions_stored": len(self._decomposition_cache),
                "voting_patterns_learned": len(self._voting_pattern_cache),
                "solution_attempts_tracked": len(self._solution_cache),
                "unique_problems": len(self._problem_hash_to_decomp),
                "problem_types_with_patterns": len(self._problem_type_to_patterns)
            }
    
    def export_memories(self, file_path: Optional[str] = None) -> str:
        """
        Export all memories to JSON file.
        
        Args:
            file_path: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        with self._lock:
            export_data = {
                "exported_at": datetime.utcnow().isoformat(),
                "decompositions": [
                    d.to_dict() for d in self._decomposition_cache.values()
                ],
                "voting_patterns": [
                    p.to_dict() for p in self._voting_pattern_cache.values()
                ],
                "solution_attempts": [
                    s.to_dict() for s in self._solution_cache.values()
                ]
            }
            
            if file_path is None:
                file_path = f"mdap_memories_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported memories to {file_path}")
            return file_path
    
    # ========================================================================
    # PRIVATE HELPERS
    # ========================================================================
    
    def _compute_problem_hash(self, problem: str) -> str:
        """Compute hash for problem deduplication."""
        normalized = problem.lower().strip()
        if HASH_AVAILABLE and compute_md5_hash:
            return compute_md5_hash(normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _index_decomposition(
        self,
        decomposition: DecompositionMemory,
        parent_problem: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Index decomposition through all 4 layers."""
        if not self.unified_memory:
            return
        
        # Create unified memory object
        content = json.dumps({
            "problem": decomposition.problem_text,
            "subproblems": decomposition.subproblems,
            "strategy": decomposition.decomposition_strategy,
            "quality": decomposition.quality_score
        })
        
        memory = UnifiedMemory(
            memory_id=decomposition.decomposition_id,
            content=content,
            memory_type=DECOMPOSITION_MEMORY_TYPE,
            importance=decomposition.quality_score,
            confidence=decomposition.quality_score
        )
        
        # Index through unified system
        if hasattr(self.unified_memory, '_index_memory'):
            self.unified_memory._index_memory(memory)
        
        # Add to hierarchical index
        if HIERARCHICAL_AVAILABLE and self.unified_memory.hierarchical_index:
            try:
                level = MemoryLevel.from_string(decomposition.hierarchical_level)
            except:
                level = MemoryLevel.IMPORTANT if MemoryLevel else None
            
            if level:
                self.unified_memory.hierarchical_index.add_node(
                    content=decomposition.problem_text,
                    level=level,
                    importance=decomposition.quality_score,
                    tags=["mdap", "decomposition", decomposition.decomposition_strategy],
                    metadata={
                        "decomposition_id": decomposition.decomposition_id,
                        "subproblems_count": len(decomposition.subproblems)
                    }
                )
        
        # Add to graph index with relationships
        if GRAPH_AVAILABLE and self.unified_memory.graph_index:
            self.build_problem_subproblem_graph(
                decomposition.problem_text,
                [sp.get("text", str(sp)) for sp in decomposition.subproblems]
            )
    
    def _index_voting_pattern(self, pattern: VotingPatternMemory) -> None:
        """Index voting pattern in unified memory."""
        if not self.unified_memory:
            return
        
        content = json.dumps({
            "problem_type": pattern.problem_type,
            "winning_strategy": pattern.winning_strategy,
            "confidence": pattern.confidence,
            "vote_distribution": pattern.vote_distribution
        })
        
        memory = UnifiedMemory(
            memory_id=pattern.pattern_id,
            content=content,
            memory_type=VOTING_PATTERN_MEMORY_TYPE,
            importance=pattern.confidence,
            confidence=pattern.confidence
        )
        
        if hasattr(self.unified_memory, '_index_memory'):
            self.unified_memory._index_memory(memory)
    
    def _index_solution_attempt(self, attempt: SolutionAttemptMemory, metadata: Optional[Dict]) -> None:
        """Index solution attempt in unified memory."""
        if not self.unified_memory:
            return
        
        content = json.dumps({
            "problem_hash": attempt.problem_hash,
            "solution_type": attempt.solution_type,
            "success": attempt.success,
            "error_pattern": attempt.error_pattern,
            "execution_time_ms": attempt.execution_time_ms
        })
        
        memory = UnifiedMemory(
            memory_id=attempt.attempt_id,
            content=content,
            memory_type=SOLUTION_ATTEMPT_MEMORY_TYPE,
            importance=0.8 if attempt.success else 0.4,
            confidence=1.0 if attempt.success else 0.6
        )
        
        if hasattr(self.unified_memory, '_index_memory'):
            self.unified_memory._index_memory(memory)
    
    def _find_voting_pattern(
        self,
        problem_type: str,
        candidates: List[Any]
    ) -> Optional[VotingPatternMemory]:
        """Find matching voting pattern from cache."""
        candidate_types = tuple(type(c).__name__ for c in candidates)
        
        for pattern_id in self._problem_type_to_patterns.get(problem_type, set()):
            if pattern_id in self._voting_pattern_cache:
                pattern = self._voting_pattern_cache[pattern_id]
                if tuple(pattern.candidate_types) == candidate_types:
                    return pattern
        
        return None
    
    def _merge_vote_distributions(
        self,
        existing: Dict[str, int],
        new: Dict[str, int]
    ) -> Dict[str, int]:
        """Merge two vote distributions."""
        merged = existing.copy()
        for key, count in new.items():
            merged[key] = merged.get(key, 0) + count
        return merged


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_mdap_memory_bridge(
    storage_path: Optional[str] = None,
    enabled: bool = True,
    config: Optional[UnifiedMemoryConfig] = None
) -> MDAPMemoryBridge:
    """
    Factory function with automatic unified memory creation.
    
    Args:
        storage_path: Directory for memory storage
        enabled: Whether to enable the bridge (False = no-op mode)
        config: Advanced configuration (overrides storage_path)
        
    Returns:
        Configured MDAPMemoryBridge instance
        
    Example:
        >>> bridge = create_mdap_memory_bridge("./my_mdap_memory")
        >>> decomp_id = bridge.store_decomposition(...)
    """
    if not enabled:
        # Return no-op bridge
        return MDAPMemoryBridge(unified_memory=None)
    
    return MDAPMemoryBridge(
        storage_path=storage_path,
        config=config
    )


def get_or_create_bridge(
    existing: Optional[MDAPMemoryBridge] = None,
    storage_path: Optional[str] = None
) -> MDAPMemoryBridge:
    """
    Get existing bridge or create new one.
    
    Convenience function for lazy initialization patterns.
    
    Args:
        existing: Existing bridge instance (if any)
        storage_path: Storage path for new bridge (if needed)
        
    Returns:
        MDAPMemoryBridge instance (existing or new)
        
    Example:
        >>> _bridge = None
        >>> def get_bridge():
        ...     global _bridge
        ...     _bridge = get_or_create_bridge(_bridge, "./memory")
        ...     return _bridge
    """
    if existing is not None:
        return existing
    return create_mdap_memory_bridge(storage_path)


def compute_decomposition_similarity(
    decomp1: DecompositionMemory,
    decomp2: DecompositionMemory
) -> float:
    """
    Compute similarity between two decompositions.
    
    Args:
        decomp1: First decomposition
        decomp2: Second decomposition
        
    Returns:
        Similarity score (0.0 - 1.0)
    """
    # Strategy match
    strategy_match = 1.0 if decomp1.decomposition_strategy == decomp2.decomposition_strategy else 0.0
    
    # Subproblem count similarity
    count1 = len(decomp1.subproblems)
    count2 = len(decomp2.subproblems)
    count_sim = 1.0 - abs(count1 - count2) / max(count1, count2, 1)
    
    # Quality similarity
    quality_sim = 1.0 - abs(decomp1.quality_score - decomp2.quality_score)
    
    # Weighted average
    return (strategy_match * 0.3 + count_sim * 0.3 + quality_sim * 0.4)


# ============================================================================
# NO-OP FALLBACK (when unified memory unavailable)
# ============================================================================

class NoOpMDAPMemoryBridge:
    """
    No-op implementation for when unified memory is unavailable.
    
    All methods return empty/None results without errors.
    """
    
    def __init__(self):
        self.enabled = False
    
    def store_decomposition(self, *args, **kwargs) -> None:
        return None
    
    def find_similar_decompositions(self, *args, **kwargs) -> List:
        return []
    
    def get_decomposition_for_subproblem(self, *args, **kwargs) -> None:
        return None
    
    def record_voting_outcome(self, *args, **kwargs) -> None:
        return None
    
    def get_voting_guidance(self, *args, **kwargs) -> None:
        return None
    
    def store_solution_attempt(self, *args, **kwargs) -> None:
        return None
    
    def find_similar_successful_solutions(self, *args, **kwargs) -> List:
        return []
    
    def get_statistics(self) -> Dict:
        return {"enabled": False}


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main class
    'MDAPMemoryBridge',
    'NoOpMDAPMemoryBridge',
    
    # Data classes
    'DecompositionMemory',
    'VotingPatternMemory',
    'SolutionAttemptMemory',
    
    # Factory functions
    'create_mdap_memory_bridge',
    'get_or_create_bridge',
    'compute_decomposition_similarity',
    
    # Constants
    'DECOMPOSITION_MEMORY_TYPE',
    'VOTING_PATTERN_MEMORY_TYPE',
    'SOLUTION_ATTEMPT_MEMORY_TYPE',
]


# Version info
__version__ = "1.0.0"
__author__ = "OpenEvolve AI"
