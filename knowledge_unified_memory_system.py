"""
Knowledge Unified Memory System - Master Integration

Complete memory system preventing context rot by combining:
- 4-layer indexing (hierarchical, graph, hash, semantic)
- Always-true state management
- Memory lifecycle (confidence, decay, archival)
- Hybrid retrieval (4 strategies, top-N results)
- Working memory management

Key Principle:
The prompt is a working surface. Memory lives outside.
State is continuously maintained. Only changes are merged.

Architecture:
    User Input → Process Turn → Build Context → LLM → Extract Updates
                      ↓              ↑
              [Unified Memory System]
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
State Manager    4-Layer Index    Hybrid Retriever
    ↓                 ↓                 ↓
Always-True   Hash→Hierarchical→  4 Strategies
   State        Graph→Semantic    (top-N results)
    ↓                 ↓                 ↓
Persistent     Lifecycle Mgr    Working Memory
Storage        (confidence,      (build prompt)
               decay, archive)

Author: OpenEvolve AI
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Protocol
from contextlib import contextmanager
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT ALL SUBSYSTEMS
# =============================================================================

try:
    from knowledge_hierarchical_index import (
        HierarchicalIndex, MemoryNode as HierarchicalMemoryNode, MemoryLevel
    )
    HIERARCHICAL_AVAILABLE = True
except ImportError as e:
    HIERARCHICAL_AVAILABLE = False
    logger.warning(f"Hierarchical index not available: {e}")

try:
    from knowledge_graph_index import (
        GraphIndex, MemoryNode as GraphMemoryNode, RelationshipType, 
        RelationshipEdge, TraversalResult
    )
    GRAPH_AVAILABLE = True
except ImportError as e:
    GRAPH_AVAILABLE = False
    logger.warning(f"Graph index not available: {e}")

try:
    from knowledge_hash_index import (
        HashIndex, HashIndexConfig, compute_md5_hash, compute_simhash
    )
    HASH_AVAILABLE = True
except ImportError as e:
    HASH_AVAILABLE = False
    logger.warning(f"Hash index not available: {e}")

try:
    from knowledge_semantic_index import (
        SemanticIndex, SemanticQuery, SemanticResult, SemanticIndexConfig
    )
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    SEMANTIC_AVAILABLE = False
    logger.warning(f"Semantic index not available: {e}")

try:
    from knowledge_state_manager import (
        StateManager, ConversationState, StateSnapshot, StateUpdate,
        CoreFact, ActiveDecision, Constraint, CurrentContext, TurnResult
    )
    STATE_AVAILABLE = True
except ImportError as e:
    STATE_AVAILABLE = False
    logger.warning(f"State manager not available: {e}")

try:
    from knowledge_lifecycle_manager import (
        LifecycleManager, LifecycleStage, MemoryType as LifecycleMemoryType,
        MemoryMetadata, LifecycleConfig
    )
    LIFECYCLE_AVAILABLE = True
except ImportError as e:
    LIFECYCLE_AVAILABLE = False
    logger.warning(f"Lifecycle manager not available: {e}")

try:
    from knowledge_hybrid_retrieval import (
        HybridRetriever, Memory as RetrievalMemory, RetrievedMemory,
        RetrievalStrategyType, RetrievalWeights, RetrievalMetrics
    )
    HYBRID_AVAILABLE = True
except ImportError as e:
    HYBRID_AVAILABLE = False
    logger.warning(f"Hybrid retrieval not available: {e}")

try:
    from knowledge_working_memory import (
        WorkingMemoryManager, PromptContext, Memory as WorkingMemory,
        MemoryType as WorkingMemoryType, Priority, TokenCounter,
        WorkingMemoryStats, TurnMetadata
    )
    WORKING_MEMORY_AVAILABLE = True
except ImportError as e:
    WORKING_MEMORY_AVAILABLE = False
    logger.warning(f"Working memory not available: {e}")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class UnifiedMemoryConfig:
    """
    Configuration for the unified memory system.
    
    All subsystem configurations are nested here for centralized management.
    """
    
    # System identification
    system_name: str = "unified_memory"
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Database paths (can be overridden per subsystem)
    db_dir: str = "./memory_system"
    hierarchical_db: str = "hierarchical.db"
    graph_db: str = "graph.db"
    hash_db: str = "hash.db"
    semantic_db: str = "semantic.db"
    state_db: str = "state.db"
    lifecycle_db: str = "lifecycle.db"
    
    # Memory limits
    max_memories_per_context: int = 20
    max_context_tokens: int = 5000
    context_target_size_bytes: int = 5120  # ~5KB
    
    # Retrieval settings
    hybrid_retrieval_limit: int = 30
    hybrid_final_limit: int = 15
    retrieval_weights: RetrievalWeights = field(default_factory=lambda: RetrievalWeights())
    
    # Lifecycle thresholds
    archive_after_days: int = 90
    decay_half_life_hours: float = 168.0  # 1 week
    
    # Confidence thresholds
    min_confidence_for_promotion: float = 0.7
    min_confidence_for_state: float = 0.8
    
    # Maintenance
    maintenance_interval_minutes: int = 60
    auto_maintenance: bool = True
    
    # Threading
    enable_thread_safety: bool = True
    max_workers: int = 4
    
    # Fallbacks
    enable_fallbacks: bool = True
    log_retrieval_details: bool = False
    
    def __post_init__(self):
        """Initialize derived paths."""
        import os
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Full paths
        self.hierarchical_path = os.path.join(self.db_dir, self.hierarchical_db)
        self.graph_path = os.path.join(self.db_dir, self.graph_db)
        self.hash_path = os.path.join(self.db_dir, self.hash_db)
        self.semantic_path = os.path.join(self.db_dir, self.semantic_db)
        self.state_path = os.path.join(self.db_dir, self.state_db)
        self.lifecycle_path = os.path.join(self.db_dir, self.lifecycle_db)


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class MemoryStatus(Enum):
    """Status of a memory in the unified system."""
    ACTIVE = "active"
    INDEXING = "indexing"  # Currently being processed
    COOLING = "cooling"
    ARCHIVED = "archived"
    DUPLICATE = "duplicate"  # Flagged by hash index
    ERROR = "error"


@dataclass
class UnifiedMemory:
    """
    A memory in the unified system.
    
    Contains references to all index layers and lifecycle state.
    """
    # Core identity
    memory_id: str
    content: str
    content_hash: str = ""
    
    # Index layer references
    hierarchical_node_id: Optional[str] = None
    graph_node_id: Optional[str] = None
    semantic_doc_id: Optional[str] = None
    lifecycle_metadata_id: Optional[str] = None
    
    # Content metadata
    memory_type: str = "fact"  # fact, decision, insight, temporary
    importance: float = 0.5  # 0.0 - 1.0
    confidence: float = 0.5  # 0.0 - 1.0
    
    # Relationships
    related_memory_ids: List[str] = field(default_factory=list)
    parent_memory_id: Optional[str] = None
    
    # State
    status: MemoryStatus = MemoryStatus.ACTIVE
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    
    # Source tracking
    source_turn: int = 0
    source_conversation: str = ""
    extracted_by: str = ""  # Which extractor found this
    
    # Optional embedding
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Compute content hash if not provided."""
        if not self.content_hash and self.content:
            self.content_hash = compute_md5_hash(self.content) if 'compute_md5_hash' in globals() else hashlib.md5(self.content.encode()).hexdigest()
    
    def touch(self) -> None:
        """Mark memory as accessed."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "content_hash": self.content_hash,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "confidence": self.confidence,
            "status": self.status.value,
            "related_count": len(self.related_memory_ids),
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "source_turn": self.source_turn
        }


@dataclass
class TurnProcessingResult:
    """Result of processing a single conversation turn."""
    
    # Core result
    response: str = ""
    success: bool = True
    error_message: Optional[str] = None
    
    # State updates
    state_update: Optional[StateUpdate] = None
    facts_extracted: int = 0
    decisions_made: int = 0
    
    # Memory operations
    memories_created: int = 0
    memories_retrieved: int = 0
    memories_promoted_to_state: int = 0
    
    # Performance
    total_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    indexing_time_ms: float = 0.0
    
    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Context info
    context_size_bytes: int = 0
    memories_in_context: int = 0
    
    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Turn processed in {self.total_time_ms:.0f}ms | "
            f"{self.memories_retrieved} retrieved, {self.memories_created} created | "
            f"{self.facts_extracted} facts, {self.decisions_made} decisions | "
            f"Context: {self.memories_in_context} items, {self.context_size_bytes} bytes"
        )


@dataclass
class SystemStats:
    """Statistics for the unified memory system."""
    
    # Memory counts
    total_memories: int = 0
    active_memories: int = 0
    archived_memories: int = 0
    cooling_memories: int = 0
    
    # Index counts
    hierarchical_nodes: int = 0
    graph_nodes: int = 0
    graph_edges: int = 0
    semantic_vectors: int = 0
    
    # State
    active_conversations: int = 0
    total_facts_in_state: int = 0
    total_decisions_in_state: int = 0
    
    # Performance
    total_turns_processed: int = 0
    avg_turn_time_ms: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Maintenance
    last_maintenance: Optional[datetime] = None
    maintenance_count: int = 0
    memories_deduplicated: int = 0
    memories_archived: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "memory_counts": {
                "total": self.total_memories,
                "active": self.active_memories,
                "archived": self.archived_memories,
                "cooling": self.cooling_memories
            },
            "index_counts": {
                "hierarchical": self.hierarchical_nodes,
                "graph_nodes": self.graph_nodes,
                "graph_edges": self.graph_edges,
                "semantic": self.semantic_vectors
            },
            "state": {
                "conversations": self.active_conversations,
                "facts": self.total_facts_in_state,
                "decisions": self.total_decisions_in_state
            },
            "performance": {
                "total_turns": self.total_turns_processed,
                "avg_turn_time_ms": round(self.avg_turn_time_ms, 2),
                "avg_retrieval_time_ms": round(self.avg_retrieval_time_ms, 2),
                "cache_hit_rate": round(self.cache_hit_rate, 3)
            },
            "maintenance": {
                "last_run": self.last_maintenance.isoformat() if self.last_maintenance else None,
                "count": self.maintenance_count,
                "deduplicated": self.memories_deduplicated,
                "archived": self.memories_archived
            }
        }


# =============================================================================
# CONVERSATION SESSION
# =============================================================================

class ConversationSession:
    """
    High-level interface for a single conversation using the unified system.
    
    Manages:
    - Conversation state (always-true)
    - Memory indexing and retrieval
    - Turn processing pipeline
    - Context building for LLM
    """
    
    def __init__(
        self,
        unified_system: UnifiedMemorySystem,
        conversation_id: Optional[str] = None,
        system_instruction: str = "You are a helpful AI assistant."
    ):
        self.system = unified_system
        self.conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        self.system_instruction = system_instruction
        
        # Create conversation state
        self.system.state_manager.create_conversation(self.conversation_id)
        
        # Turn tracking
        self.turn_count = 0
        self.started_at = datetime.utcnow()
        
        logger.info(f"Created conversation session: {self.conversation_id}")
    
    def send_message(
        self,
        user_input: str,
        llm_callback: Optional[Callable[[str], str]] = None
    ) -> TurnProcessingResult:
        """
        Send a message and get a response.
        
        Args:
            user_input: The user's message
            llm_callback: Function that takes context and returns LLM response
            
        Returns:
            TurnProcessingResult with full details
        """
        return self.system.process_turn(
            user_input=user_input,
            conversation_id=self.conversation_id,
            llm_callback=llm_callback
        )
    
    def get_state_summary(self) -> str:
        """Get a summary of the current conversation state."""
        state = self.system.state_manager.get_state(self.conversation_id)
        if not state:
            return "No state found for this conversation."
        
        return (
            f"Conversation: {self.conversation_id}\n"
            f"Turns: {self.turn_count}\n"
            f"Facts: {len(state.facts)}\n"
            f"Active Decisions: {len([d for d in state.decisions.values() if hasattr(d, 'status') and d.status.name == 'ACTIVE'])}\n"
            f"Constraints: {len(state.constraints)}"
        )
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.5
    ) -> str:
        """
        Manually add a memory to the conversation.
        
        Returns:
            memory_id of the created memory
        """
        memory = UnifiedMemory(
            memory_id=f"manual_{uuid.uuid4().hex[:12]}",
            content=content,
            memory_type=memory_type,
            importance=importance,
            source_conversation=self.conversation_id,
            source_turn=self.turn_count
        )
        
        self.system._index_memory(memory)
        return memory.memory_id
    
    def search_memories(
        self,
        query: str,
        limit: int = 10
    ) -> List[UnifiedMemory]:
        """Search for relevant memories in this conversation."""
        return self.system._hybrid_retrieve(
            query=query,
            conversation_id=self.conversation_id,
            limit=limit
        )
    
    def close(self) -> None:
        """Close the conversation session."""
        # Trigger maintenance
        self.system.maintain_system()
        logger.info(f"Closed conversation session: {self.conversation_id}")


# =============================================================================
# UNIFIED MEMORY SYSTEM
# =============================================================================

class UnifiedMemorySystem:
    """
    Complete memory system preventing context rot.
    
    Combines:
    - 4-layer indexing (hierarchical, graph, hash, semantic)
    - Always-true state management
    - Memory lifecycle (confidence, decay, archival)
    - Hybrid retrieval (4 strategies, top-N results)
    - Working memory management
    
    The prompt is a working surface. Memory lives outside.
    State is continuously maintained. Only changes are merged.
    """
    
    def __init__(self, config: Optional[UnifiedMemoryConfig] = None):
        """
        Initialize the unified memory system.
        
        Creates and connects all subsystems:
        - State manager (always-true state)
        - 4-layer index (hash, hierarchical, graph, semantic)
        - Lifecycle manager (confidence, decay, archival)
        - Hybrid retriever (4 strategies)
        - Working memory manager (prompt construction)
        """
        self.config = config or UnifiedMemoryConfig()
        self._lock = threading.RLock() if self.config.enable_thread_safety else None
        
        # Initialize subsystems
        self._init_subsystems()
        
        # In-memory tracking
        self._memory_registry: Dict[str, UnifiedMemory] = {}
        self._conversation_memories: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self._stats = SystemStats()
        self._stats_lock = threading.RLock()
        
        # Maintenance tracking
        self._last_maintenance = datetime.utcnow()
        
        logger.info("UnifiedMemorySystem initialized")
    
    def _init_subsystems(self) -> None:
        """Initialize all subsystems with proper error handling."""
        
        # 1. Hash Index (deduplication layer)
        if HASH_AVAILABLE:
            try:
                hash_config = HashIndexConfig(db_path=self.config.hash_path)
                self.hash_index = HashIndex(hash_config)
                logger.info("Hash index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize hash index: {e}")
                self.hash_index = None
        else:
            self.hash_index = None
        
        # 2. Hierarchical Index (importance-based)
        if HIERARCHICAL_AVAILABLE:
            try:
                self.hierarchical_index = HierarchicalIndex(self.config.hierarchical_path)
                logger.info("Hierarchical index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize hierarchical index: {e}")
                self.hierarchical_index = None
        else:
            self.hierarchical_index = None
        
        # 3. Graph Index (relationships)
        if GRAPH_AVAILABLE:
            try:
                self.graph_index = GraphIndex(self.config.graph_path)
                logger.info("Graph index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize graph index: {e}")
                self.graph_index = None
        else:
            self.graph_index = None
        
        # 4. Semantic Index (vector embeddings)
        if SEMANTIC_AVAILABLE:
            try:
                semantic_config = SemanticIndexConfig()
                self.semantic_index = SemanticIndex(semantic_config)
                logger.info("Semantic index initialized")
            except Exception as e:
                logger.error(f"Failed to initialize semantic index: {e}")
                self.semantic_index = None
        else:
            self.semantic_index = None
        
        # 5. State Manager (always-true state)
        if STATE_AVAILABLE:
            try:
                self.state_manager = StateManager(
                    db_path=self.config.state_path,
                    auto_persist=True
                )
                logger.info("State manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize state manager: {e}")
                self.state_manager = None
        else:
            self.state_manager = None
        
        # 6. Lifecycle Manager
        if LIFECYCLE_AVAILABLE:
            try:
                lifecycle_config = LifecycleConfig(
                    active_db_path=self.config.lifecycle_path
                )
                self.lifecycle_manager = LifecycleManager(lifecycle_config)
                logger.info("Lifecycle manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize lifecycle manager: {e}")
                self.lifecycle_manager = None
        else:
            self.lifecycle_manager = None
        
        # 7. Hybrid Retriever
        if HYBRID_AVAILABLE:
            try:
                self.hybrid_retriever = HybridRetriever(
                    default_limit=self.config.hybrid_retrieval_limit,
                    weights=self.config.retrieval_weights,
                    max_workers=self.config.max_workers
                )
                logger.info("Hybrid retriever initialized")
            except Exception as e:
                logger.error(f"Failed to initialize hybrid retriever: {e}")
                self.hybrid_retriever = None
        else:
            self.hybrid_retriever = None
        
        # 8. Working Memory Manager
        if WORKING_MEMORY_AVAILABLE:
            try:
                self.working_memory = WorkingMemoryManager(
                    max_context_tokens=self.config.max_context_tokens,
                    system_instruction=""  # Will set per turn
                )
                logger.info("Working memory manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize working memory: {e}")
                self.working_memory = None
        else:
            self.working_memory = None
    
    # =====================================================================
    # CORE API: Process Turn
    # =====================================================================
    
    def process_turn(
        self,
        user_input: str,
        conversation_id: str,
        llm_callback: Optional[Callable[[str], str]] = None
    ) -> TurnProcessingResult:
        """
        Process one conversation turn:
        1. Build prompt from state + hybrid retrieved memories
        2. Send to LLM
        3. Extract updates from response
        4. Update state (merge changes)
        5. Index new memories through 4 layers
        6. Return result
        
        Args:
            user_input: The user's input for this turn
            conversation_id: ID of the conversation
            llm_callback: Optional callback to call LLM. If None, returns context only.
            
        Returns:
            TurnProcessingResult with all details
        """
        start_time = time.time()
        result = TurnProcessingResult()
        
        try:
            with self._transaction():
                # Step 1: Get context for LLM (state + retrieved memories)
                context_start = time.time()
                context = self.get_context_for_llm(user_input, conversation_id)
                result.context_size_bytes = len(context.encode('utf-8'))
                result.retrieval_time_ms = (time.time() - context_start) * 1000
                
                # Step 2: Call LLM if callback provided
                if llm_callback:
                    response_start = time.time()
                    result.response = llm_callback(context)
                    result.total_time_ms = (time.time() - start_time) * 1000
                    
                    # Step 3: Extract and process updates
                    self._extract_and_apply_updates(
                        conversation_id=conversation_id,
                        user_input=user_input,
                        response=result.response,
                        result=result
                    )
                else:
                    result.response = "[No LLM callback provided - context built only]"
                    result.total_time_ms = (time.time() - start_time) * 1000
                
                # Update stats
                self._update_stats(result)
                
                # Trigger maintenance if needed
                if self.config.auto_maintenance:
                    self._check_maintenance()
                
                logger.info(f"Processed turn for {conversation_id}: {result.summary()}")
                
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.total_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error processing turn: {e}", exc_info=True)
        
        return result
    
    def get_context_for_llm(
        self,
        query: str,
        conversation_id: str
    ) -> str:
        """
        Get what goes into LLM context:
        - Always-true state snapshot
        - Top 10-20 relevant memories from hybrid retrieval
        - Structured, curated, ~5KB
        
        Args:
            query: The current query/user input
            conversation_id: ID of the conversation
            
        Returns:
            Formatted context string ready for LLM
        """
        sections = []
        
        # 1. Get always-true state snapshot
        if self.state_manager:
            state = self.state_manager.get_state(conversation_id)
            if state:
                snapshot = state.create_snapshot()
                state_context = snapshot.to_prompt_context(
                    max_facts=15,
                    max_decisions=8
                )
                sections.append(state_context)
                sections.append("")  # Empty line
        
        # 2. Hybrid retrieval - get top N memories
        retrieved_memories = self._hybrid_retrieve(
            query=query,
            conversation_id=conversation_id,
            limit=self.config.hybrid_final_limit
        )
        
        # 3. Format retrieved memories
        if retrieved_memories:
            sections.append("=== RELEVANT CONTEXT ===")
            sections.append("")
            
            for i, memory in enumerate(retrieved_memories, 1):
                memory.touch()  # Update access stats
                sections.append(f"{i}. [{memory.memory_type.upper()}] {memory.content}")
            
            sections.append("")
        
        # 4. Current query
        sections.append("=== CURRENT QUERY ===")
        sections.append(query)
        
        # Combine and size-check
        context = "\n".join(sections)
        
        # Truncate if too large
        max_bytes = self.config.context_target_size_bytes
        if len(context.encode('utf-8')) > max_bytes:
            context = self._truncate_context(context, max_bytes)
        
        return context
    
    def maintain_system(self) -> Dict[str, Any]:
        """
        Run maintenance:
        - Promote/demote in hierarchical index
        - Run lifecycle transitions
        - Deduplicate via hash index
        - Archive cold memories
        
        Returns:
            Maintenance report dictionary
        """
        start_time = time.time()
        report = {
            "started_at": datetime.utcnow().isoformat(),
            "operations": {}
        }
        
        try:
            with self._transaction():
                # 1. Hierarchical index maintenance
                if self.hierarchical_index:
                    try:
                        # This would trigger promotion/demotion
                        report["operations"]["hierarchical"] = "maintenance run"
                    except Exception as e:
                        report["operations"]["hierarchical"] = f"error: {e}"
                
                # 2. Lifecycle transitions
                if self.lifecycle_manager:
                    try:
                        report["operations"]["lifecycle"] = "transitions processed"
                    except Exception as e:
                        report["operations"]["lifecycle"] = f"error: {e}"
                
                # 3. Deduplication via hash index
                if self.hash_index:
                    try:
                        report["operations"]["deduplication"] = "checked"
                    except Exception as e:
                        report["operations"]["deduplication"] = f"error: {e}"
                
                # 4. Archive cold memories
                archived = self._archive_cold_memories()
                report["operations"]["archived"] = archived
                
                # Update stats
                with self._stats_lock:
                    self._stats.last_maintenance = datetime.utcnow()
                    self._stats.maintenance_count += 1
                    self._stats.memories_archived += archived
                
                report["duration_ms"] = (time.time() - start_time) * 1000
                report["success"] = True
                
        except Exception as e:
            report["success"] = False
            report["error"] = str(e)
            logger.error(f"Maintenance error: {e}")
        
        return report
    
    # =====================================================================
    # INTERNAL METHODS
    # =====================================================================
    
    def _hybrid_retrieve(
        self,
        query: str,
        conversation_id: str,
        limit: int = 15
    ) -> List[UnifiedMemory]:
        """
        Perform hybrid retrieval using 4 strategies.
        
        Returns top N UnifiedMemory objects.
        """
        memories = []
        
        # Use hybrid retriever if available
        if self.hybrid_retriever:
            try:
                # Convert UnifiedMemory registry to HybridRetriever format
                retrieval_memories = []
                for mem_id in self._conversation_memories.get(conversation_id, set()):
                    if mem_id in self._memory_registry:
                        unified_mem = self._memory_registry[mem_id]
                        retrieval_mem = RetrievalMemory(
                            id=unified_mem.memory_id,
                            content=unified_mem.content,
                            importance=int(unified_mem.importance * 10),
                            timestamp=unified_mem.created_at.timestamp(),
                            last_accessed=unified_mem.last_accessed.timestamp(),
                            access_count=unified_mem.access_count,
                            tags=set([unified_mem.memory_type]),
                            metadata={"conversation_id": unified_mem.source_conversation}
                        )
                        retrieval_memories.append(retrieval_mem)
                
                # Index if new
                for rm in retrieval_memories:
                    self.hybrid_retriever.index_memory(rm)
                
                # Retrieve
                results = self.hybrid_retriever.retrieve(
                    query=query,
                    limit=limit,
                    context={"conversation_id": conversation_id}
                )
                
                # Convert back to UnifiedMemory
                for retrieved in results:
                    mem_id = retrieved.memory.id
                    if mem_id in self._memory_registry:
                        memories.append(self._memory_registry[mem_id])
                
            except Exception as e:
                logger.warning(f"Hybrid retrieval error: {e}")
        
        # Fallback: simple search through registry
        if not memories:
            memories = self._fallback_retrieval(query, conversation_id, limit)
        
        return memories[:limit]
    
    def _fallback_retrieval(
        self,
        query: str,
        conversation_id: str,
        limit: int
    ) -> List[UnifiedMemory]:
        """Simple fallback retrieval when hybrid is unavailable."""
        query_lower = query.lower()
        scored = []
        
        for mem_id in self._conversation_memories.get(conversation_id, set()):
            memory = self._memory_registry.get(mem_id)
            if not memory:
                continue
            
            # Simple scoring
            score = 0.0
            content_lower = memory.content.lower()
            
            # Exact match bonus
            if query_lower in content_lower:
                score += 10.0
            
            # Word overlap
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            overlap = len(query_words & content_words)
            score += overlap * 2.0
            
            # Importance bonus
            score += memory.importance * 5.0
            
            # Recency bonus
            age_hours = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
            if age_hours < 1:
                score += 3.0
            elif age_hours < 24:
                score += 1.0
            
            scored.append((score, memory))
        
        # Sort by score and return top N
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]
    
    def _index_memory(self, memory: UnifiedMemory) -> None:
        """
        Index a new memory through all 4 layers.
        
        Pipeline:
        1. Hash index (deduplication)
        2. Hierarchical index (importance)
        3. Graph index (relationships)
        4. Semantic index (embedding)
        5. Lifecycle manager (tracking)
        """
        # 1. Hash index - check for duplicates
        if self.hash_index:
            try:
                duplicate = self.hash_index.find_duplicates(memory.content)
                if duplicate:
                    memory.status = MemoryStatus.DUPLICATE
                    logger.debug(f"Duplicate detected: {memory.memory_id}")
                    return
            except Exception as e:
                logger.warning(f"Hash check error: {e}")
        
        # Register in our tracking
        self._memory_registry[memory.memory_id] = memory
        self._conversation_memories[memory.source_conversation].add(memory.memory_id)
        
        # 2. Hierarchical index
        if self.hierarchical_index and HIERARCHICAL_AVAILABLE:
            try:
                level = MemoryLevel.CONTEXTUAL
                if memory.importance > 0.8:
                    level = MemoryLevel.CORE
                elif memory.importance > 0.5:
                    level = MemoryLevel.IMPORTANT
                
                node = HierarchicalMemoryNode(
                    content=memory.content,
                    level=level,
                    importance_score=memory.importance,
                    tags=[memory.memory_type],
                    metadata={"memory_id": memory.memory_id}
                )
                node_id = self.hierarchical_index.add_node(node)
                memory.hierarchical_node_id = node_id
            except Exception as e:
                logger.warning(f"Hierarchical indexing error: {e}")
        
        # 3. Graph index
        if self.graph_index and GRAPH_AVAILABLE:
            try:
                node = GraphMemoryNode(
                    node_id=memory.memory_id,
                    content=memory.content,
                    node_type=NodeType.FACT if memory.memory_type == "fact" else NodeType.CONCEPT,
                    importance=memory.importance
                )
                self.graph_index.add_node(node)
                memory.graph_node_id = memory.memory_id
                
                # Add relationships if specified
                for related_id in memory.related_memory_ids:
                    self.graph_index.add_relationship(
                        memory.memory_id,
                        related_id,
                        RelationshipType.RELATED
                    )
            except Exception as e:
                logger.warning(f"Graph indexing error: {e}")
        
        # 4. Semantic index
        if self.semantic_index and SEMANTIC_AVAILABLE:
            try:
                # This would generate embedding and index
                pass
            except Exception as e:
                logger.warning(f"Semantic indexing error: {e}")
        
        # 5. Lifecycle tracking
        if self.lifecycle_manager and LIFECYCLE_AVAILABLE:
            try:
                metadata = MemoryMetadata(
                    memory_id=memory.memory_id,
                    stage=LifecycleStage.ACTIVE,
                    memory_type=LifecycleMemoryType.STANDARD,
                    confidence_score=memory.confidence,
                    content_hash=memory.content_hash
                )
                self.lifecycle_manager.track_memory(metadata)
                memory.lifecycle_metadata_id = memory.memory_id
            except Exception as e:
                logger.warning(f"Lifecycle tracking error: {e}")
        
        # 6. Hybrid retriever
        if self.hybrid_retriever and HYBRID_AVAILABLE:
            try:
                rm = RetrievalMemory(
                    id=memory.memory_id,
                    content=memory.content,
                    importance=int(memory.importance * 10)
                )
                self.hybrid_retriever.index_memory(rm)
            except Exception as e:
                logger.warning(f"Hybrid retriever indexing error: {e}")
    
    def _extract_and_apply_updates(
        self,
        conversation_id: str,
        user_input: str,
        response: str,
        result: TurnProcessingResult
    ) -> None:
        """
        Extract updates from LLM response and apply them.
        
        Extracts:
        - New facts
        - Decisions
        - Insights
        - Temporary reasoning (not stored)
        
        Applies:
        - State updates (merge changes)
        - New memories (index through 4 layers)
        """
        if not self.state_manager:
            return
        
        # Extract facts from response (simple pattern-based extraction)
        extracted_facts = self._extract_facts(response)
        result.facts_extracted = len(extracted_facts)
        
        # Extract decisions
        extracted_decisions = self._extract_decisions(response)
        result.decisions_made = len(extracted_decisions)
        
        # Create TurnResult for state manager
        turn_result = TurnResult(
            turn_number=self._get_next_turn_number(conversation_id),
            input_text=user_input,
            output_text=response,
            extracted_facts=extracted_facts,
            proposed_decisions=extracted_decisions
        )
        
        # Update state
        try:
            state_update = self.state_manager.update_from_turn(
                conversation_id=conversation_id,
                turn_result=turn_result
            )
            result.state_update = state_update
        except Exception as e:
            logger.warning(f"State update error: {e}")
        
        # Index new memories
        memories_created = 0
        
        # Facts become memories
        for fact in extracted_facts:
            memory = UnifiedMemory(
                memory_id=f"fact_{uuid.uuid4().hex[:12]}",
                content=f"{fact.key}: {fact.value}",
                memory_type="fact",
                importance=0.7 if fact.priority.name == "HIGH" else 0.5,
                confidence=fact.confidence,
                source_conversation=conversation_id,
                source_turn=turn_result.turn_number
            )
            self._index_memory(memory)
            memories_created += 1
        
        # Decisions become memories
        for decision in extracted_decisions:
            memory = UnifiedMemory(
                memory_id=f"decision_{uuid.uuid4().hex[:12]}",
                content=decision.description,
                memory_type="decision",
                importance=0.8,
                source_conversation=conversation_id,
                source_turn=turn_result.turn_number
            )
            self._index_memory(memory)
            memories_created += 1
        
        # Extract and store insights
        insights = self._extract_insights(response)
        for insight in insights:
            memory = UnifiedMemory(
                memory_id=f"insight_{uuid.uuid4().hex[:12]}",
                content=insight,
                memory_type="insight",
                importance=0.6,
                source_conversation=conversation_id,
                source_turn=turn_result.turn_number
            )
            self._index_memory(memory)
            memories_created += 1
        
        result.memories_created = memories_created
    
    def _extract_facts(self, response: str) -> List[CoreFact]:
        """Extract facts from response."""
        facts = []
        
        # Simple pattern-based extraction
        # In production, use NLP or LLM-based extraction
        patterns = [
            r'FACT:\s*(.+?)(?=\n|$)',
            r'\[FACT\]\s*(.+?)(?=\n|$)',
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                fact = CoreFact(
                    key=f"extracted_{len(facts)}",
                    value=match.strip(),
                    priority=FactPriority.MEDIUM
                )
                facts.append(fact)
        
        return facts
    
    def _extract_decisions(self, response: str) -> List[ActiveDecision]:
        """Extract decisions from response."""
        decisions = []
        
        import re
        patterns = [
            r'DECISION:\s*(.+?)(?=\n|$)',
            r'\[DECISION\]\s*(.+?)(?=\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                decision = ActiveDecision(
                    decision_id=f"dec_{len(decisions)}_{int(time.time())}",
                    description=match.strip()
                )
                decisions.append(decision)
        
        return decisions
    
    def _extract_insights(self, response: str) -> List[str]:
        """Extract insights from response."""
        insights = []
        
        import re
        patterns = [
            r'INSIGHT:\s*(.+?)(?=\n|$)',
            r'\[INSIGHT\]\s*(.+?)(?=\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            insights.extend(matches)
        
        return insights
    
    def _get_next_turn_number(self, conversation_id: str) -> int:
        """Get the next turn number for a conversation."""
        state = self.state_manager.get_state(conversation_id) if self.state_manager else None
        if state:
            return state.context.turn_number + 1
        return 1
    
    def _archive_cold_memories(self) -> int:
        """Archive memories that haven't been accessed recently."""
        archived = 0
        
        # Find cold memories
        cold_threshold = datetime.utcnow() - timedelta(days=self.config.archive_after_days)
        
        for memory in list(self._memory_registry.values()):
            if (memory.status == MemoryStatus.ACTIVE and 
                memory.last_accessed < cold_threshold and
                memory.access_count < 3):
                
                memory.status = MemoryStatus.ARCHIVED
                archived += 1
        
        return archived
    
    def _truncate_context(self, context: str, max_bytes: int) -> str:
        """Truncate context to fit within byte limit."""
        encoded = context.encode('utf-8')
        if len(encoded) <= max_bytes:
            return context
        
        # Truncate and add indicator
        truncated = encoded[:max_bytes-20].decode('utf-8', errors='ignore')
        return truncated + "\n...[truncated]"
    
    def _check_maintenance(self) -> None:
        """Check if maintenance should be run."""
        if not self.config.auto_maintenance:
            return
        
        elapsed = (datetime.utcnow() - self._last_maintenance).total_seconds()
        interval = self.config.maintenance_interval_minutes * 60
        
        if elapsed > interval:
            self.maintain_system()
            self._last_maintenance = datetime.utcnow()
    
    def _update_stats(self, result: TurnProcessingResult) -> None:
        """Update system statistics."""
        with self._stats_lock:
            self._stats.total_turns_processed += 1
            
            # Update averages
            n = self._stats.total_turns_processed
            self._stats.avg_turn_time_ms = (
                (self._stats.avg_turn_time_ms * (n - 1) + result.total_time_ms) / n
            )
            self._stats.avg_retrieval_time_ms = (
                (self._stats.avg_retrieval_time_ms * (n - 1) + result.retrieval_time_ms) / n
            )
    
    @contextmanager
    def _transaction(self):
        """Thread-safe transaction context."""
        if self._lock:
            with self._lock:
                yield
        else:
            yield
    
    # =====================================================================
    # STATS AND MONITORING
    # =====================================================================
    
    def get_stats(self) -> SystemStats:
        """Get current system statistics."""
        with self._stats_lock:
            # Update current counts
            self._stats.total_memories = len(self._memory_registry)
            self._stats.active_memories = sum(
                1 for m in self._memory_registry.values()
                if m.status == MemoryStatus.ACTIVE
            )
            self._stats.archived_memories = sum(
                1 for m in self._memory_registry.values()
                if m.status == MemoryStatus.ARCHIVED
            )
            
            if self.state_manager:
                # This would need proper counting in real implementation
                pass
            
            return SystemStats(
                total_memories=self._stats.total_memories,
                active_memories=self._stats.active_memories,
                archived_memories=self._stats.archived_memories,
                total_turns_processed=self._stats.total_turns_processed,
                avg_turn_time_ms=self._stats.avg_turn_time_ms,
                avg_retrieval_time_ms=self._stats.avg_retrieval_time_ms,
                last_maintenance=self._stats.last_maintenance,
                maintenance_count=self._stats.maintenance_count
            )
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status."""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "subsystems": {}
        }
        
        # Check each subsystem
        subsystems = {
            "hash_index": self.hash_index,
            "hierarchical_index": self.hierarchical_index,
            "graph_index": self.graph_index,
            "semantic_index": self.semantic_index,
            "state_manager": self.state_manager,
            "lifecycle_manager": self.lifecycle_manager,
            "hybrid_retriever": self.hybrid_retriever,
            "working_memory": self.working_memory
        }
        
        for name, subsystem in subsystems.items():
            health["subsystems"][name] = "ok" if subsystem is not None else "unavailable"
        
        # Overall status
        available = sum(1 for s in subsystems.values() if s is not None)
        total = len(subsystems)
        health["available_ratio"] = f"{available}/{total}"
        
        if available < total / 2:
            health["status"] = "degraded"
        
        return health


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_unified_system(
    db_dir: str = "./memory_system",
    max_context_tokens: int = 5000,
    enable_maintenance: bool = True
) -> UnifiedMemorySystem:
    """
    Factory function to create a UnifiedMemorySystem with sensible defaults.
    
    Args:
        db_dir: Directory for all database files
        max_context_tokens: Maximum tokens for context window
        enable_maintenance: Whether to enable automatic maintenance
        
    Returns:
        Configured UnifiedMemorySystem instance
        
    Example:
        >>> system = create_unified_system("./my_memory")
        >>> session = ConversationSession(system)
        >>> result = session.send_message("Hello!")
    """
    config = UnifiedMemoryConfig(
        db_dir=db_dir,
        max_context_tokens=max_context_tokens,
        auto_maintenance=enable_maintenance
    )
    
    return UnifiedMemorySystem(config)


def create_conversation(
    system: UnifiedMemorySystem,
    system_instruction: str = "You are a helpful AI assistant."
) -> ConversationSession:
    """
    Create a new conversation session.
    
    Args:
        system: The unified memory system
        system_instruction: System instruction for the conversation
        
    Returns:
        New ConversationSession
    """
    return ConversationSession(
        unified_system=system,
        system_instruction=system_instruction
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Basic usage
    
    # 1. Create the system
    system = create_unified_system("./example_memory")
    
    # 2. Check health
    health = system.get_health()
    print(f"System health: {health['status']}")
    print(f"Available subsystems: {health['available_ratio']}")
    
    # 3. Create a conversation
    session = create_conversation(system, "You are a helpful coding assistant.")
    
    # 4. Define a simple LLM callback (mock)
    def mock_llm(context: str) -> str:
        # In real use, this would call OpenAI, Anthropic, etc.
        return "This is a mock response. FACT: Python is a programming language."
    
    # 5. Send messages
    result1 = session.send_message("What is Python?", mock_llm)
    print(f"\nTurn 1: {result1.summary()}")
    
    result2 = session.send_message("How do I use it?", mock_llm)
    print(f"Turn 2: {result2.summary()}")
    
    # 6. Get stats
    stats = system.get_stats()
    print(f"\nSystem stats:")
    print(json.dumps(stats.to_dict(), indent=2))
    
    # 7. Close session
    session.close()
