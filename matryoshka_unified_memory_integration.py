"""
Matryoshka Unified Memory Integration

Integrates the Unified Memory System with Matryoshka RLM to prevent context rot
in long-running Matryoshka sessions using the 4-layer memory system.

Purpose:
--------
Matryoshka analyzes documents 100x larger than context windows through iterative
exploration. This integration prevents context rot in long-running Matryoshka 
sessions using the 4-layer memory system (hierarchical, graph, hash, semantic).

Key Integration Points:
-----------------------
1. Matryoshka Exploration Memory: Each exploration step (code execution, result 
   observation) is stored in the unified memory system with proper indexing.
   
2. Cross-Session Learning: Insights from one Matryoshka analysis are available 
   to future analyses via the semantic/graph indexes.
   
3. Stateful Exploration: Instead of simple summary strings, use full state 
   management for exploration state.
   
4. Hybrid Retrieval for Context: When Matryoshka needs context, use hybrid 
   retrieval (not just the immediate previous step).

Architecture:
-------------
    Matryoshka RLM <--> MatryoshkaMemoryBridge <--> UnifiedMemorySystem
           v                                    v
    ExplorationSession                    4-Layer Index
           v                                    v
    Stateful Turns                    Hash->Hierarchical->
           v                                    Graph->Semantic
    SynthesisResult                        v
                                    Always-True State

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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Iterator
from contextlib import contextmanager
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# IMPORT UNIFIED MEMORY SYSTEM
# =============================================================================

try:
    from knowledge_unified_memory_system import (
        UnifiedMemorySystem, UnifiedMemory, UnifiedMemoryConfig,
        MemoryStatus, TurnProcessingResult, SystemStats,
        ConversationSession, create_unified_system
    )
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError as e:
    UNIFIED_MEMORY_AVAILABLE = False
    logger.warning(f"Unified memory system not available: {e}")

try:
    from knowledge_state_manager import (
        CoreFact, ActiveDecision, Constraint, CurrentContext,
        ConversationState, StateUpdate, TurnResult
    )
    STATE_MANAGER_AVAILABLE = True
except ImportError:
    STATE_MANAGER_AVAILABLE = False

try:
    from knowledge_hybrid_retrieval import RetrievedMemory, RetrievalStrategyType
    HYBRID_RETRIEVAL_AVAILABLE = True
except ImportError:
    HYBRID_RETRIEVAL_AVAILABLE = False

try:
    from glue.adapters.matryoshka_adapter import (
        MatryoshkaClient, StatefulMatryoshkaClient
    )
    MATRYOSHKA_ADAPTER_AVAILABLE = True
except ImportError:
    MATRYOSHKA_ADAPTER_AVAILABLE = False
    logger.warning("Matryoshka adapter not available")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ExplorationStepType(Enum):
    """Types of exploration steps in Matryoshka analysis."""
    INITIALIZATION = "initialization"      # Document loading, first setup
    CODE_GENERATION = "code_generation"    # Generated Python code to explore
    CODE_EXECUTION = "code_execution"      # Executed code
    OBSERVATION = "observation"            # Raw observation from execution
    INSIGHT = "insight"                    # Derived insight/understanding
    HYPOTHESIS = "hypothesis"              # Formed hypothesis
    VERIFICATION = "verification"          # Verified or refuted hypothesis
    SYNTHESIS = "synthesis"                # Final synthesis step
    ERROR = "error"                        # Error during exploration


@dataclass
class ExplorationStep:
    """
    A single step in the Matryoshka exploration process.
    
    Each step represents one iteration of the exploration loop:
    - Query -> Code -> Execute -> Observe -> Insight
    """
    step_id: str
    session_id: str
    turn_number: int
    step_type: ExplorationStepType
    
    # Content
    query: str = ""                          # The question/task for this step
    code_executed: Optional[str] = None      # Python code generated/executed
    observation: Optional[str] = None        # Raw observation result
    insight: Optional[str] = None            # Derived insight
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    execution_time_ms: float = 0.0
    tokens_used: int = 0
    
    # Relationships
    previous_step_id: Optional[str] = None   # Link to previous step
    related_step_ids: List[str] = field(default_factory=list)
    
    # Importance scoring
    importance: float = 0.5                  # 0.0 - 1.0
    confidence: float = 0.5                  # 0.0 - 1.0
    
    # Source tracking
    document_path: Optional[str] = None
    document_section: Optional[str] = None   # Which part of document
    
    def to_memory_content(self) -> str:
        """Convert step to unified memory content format."""
        sections = [
            f"[Turn {self.turn_number}] {self.step_type.value.upper()}",
            f"Query: {self.query[:200]}..." if len(self.query) > 200 else f"Query: {self.query}",
        ]
        
        if self.code_executed:
            code_snippet = self.code_executed[:300] + "..." if len(self.code_executed) > 300 else self.code_executed
            sections.append(f"Code:\n{code_snippet}")
        
        if self.observation:
            obs_snippet = self.observation[:300] + "..." if len(self.observation) > 300 else self.observation
            sections.append(f"Observation:\n{obs_snippet}")
        
        if self.insight:
            sections.append(f"Insight: {self.insight}")
        
        return "\n\n".join(sections)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "step_type": self.step_type.value,
            "query": self.query,
            "code_executed": self.code_executed,
            "observation": self.observation,
            "insight": self.insight,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms,
            "tokens_used": self.tokens_used,
            "previous_step_id": self.previous_step_id,
            "related_step_ids": self.related_step_ids,
            "importance": self.importance,
            "confidence": self.confidence,
            "document_path": self.document_path,
            "document_section": self.document_section,
        }


@dataclass
class DocumentState:
    """
    Always-true state for a document being analyzed.
    
    Maintains:
    - Document metadata (type, size, structure)
    - Key findings discovered so far
    - Exploration progress (what's been checked)
    - Current hypothesis/goal
    """
    session_id: str
    document_path: str
    
    # Document metadata
    document_type: Optional[str] = None          # e.g., "python", "markdown", "json"
    document_size_bytes: int = 0
    document_structure: Optional[str] = None     # Summary of structure
    
    # Exploration progress
    sections_explored: Set[str] = field(default_factory=set)
    sections_remaining: Set[str] = field(default_factory=set)
    total_turns: int = 0
    
    # Key findings (accumulated insights)
    key_findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Current state
    current_hypothesis: Optional[str] = None
    current_goal: Optional[str] = None
    exploration_complete: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_finding(self, finding: str, confidence: float = 0.5, 
                    source_step_id: Optional[str] = None) -> None:
        """Add a key finding to the document state."""
        self.key_findings.append({
            "finding": finding,
            "confidence": confidence,
            "source_step_id": source_step_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_updated = datetime.utcnow()
    
    def mark_section_explored(self, section: str) -> None:
        """Mark a document section as explored."""
        self.sections_explored.add(section)
        self.sections_remaining.discard(section)
        self.last_updated = datetime.utcnow()
    
    def to_state_facts(self) -> List[CoreFact]:
        """Convert document state to core facts for state manager."""
        if not STATE_MANAGER_AVAILABLE:
            return []
        
        facts = [
            CoreFact(
                key="document_path",
                value=self.document_path,
                priority="HIGH",
                confidence=1.0
            ),
            CoreFact(
                key="document_type",
                value=self.document_type or "unknown",
                priority="MEDIUM",
                confidence=0.9
            ),
            CoreFact(
                key="exploration_progress",
                value=f"{len(self.sections_explored)}/{len(self.sections_explored) + len(self.sections_remaining)} sections",
                priority="MEDIUM",
                confidence=0.95
            ),
            CoreFact(
                key="current_goal",
                value=self.current_goal or "explore document",
                priority="HIGH",
                confidence=0.8
            ),
        ]
        
        # Add key findings as facts
        for i, finding in enumerate(self.key_findings[-5:], 1):  # Last 5 findings
            facts.append(CoreFact(
                key=f"finding_{i}",
                value=finding["finding"][:200],  # Truncate long findings
                priority="HIGH" if finding["confidence"] > 0.8 else "MEDIUM",
                confidence=finding["confidence"]
            ))
        
        return facts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "document_path": self.document_path,
            "document_type": self.document_type,
            "document_size_bytes": self.document_size_bytes,
            "document_structure": self.document_structure,
            "sections_explored": list(self.sections_explored),
            "sections_remaining": list(self.sections_remaining),
            "total_turns": self.total_turns,
            "key_findings": self.key_findings,
            "current_hypothesis": self.current_hypothesis,
            "current_goal": self.current_goal,
            "exploration_complete": self.exploration_complete,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class ExplorationContext:
    """
    Context retrieved for a Matryoshka exploration step.
    
    Combines state snapshot with relevant memories for decision-making.
    """
    # Document state
    document_state: Optional[DocumentState] = None
    
    # Relevant memories from previous steps
    relevant_memories: List[UnifiedMemory] = field(default_factory=list)
    
    # Previous steps in chain (for continuity)
    step_chain: List[ExplorationStep] = field(default_factory=list)
    
    # Context statistics
    total_memories_available: int = 0
    memories_in_context: int = 0
    context_size_bytes: int = 0
    
    def to_prompt_context(self, max_bytes: int = 5120) -> str:
        """
        Format context as a prompt string for LLM.
        
        Returns structured context suitable for Matryoshka code generation.
        """
        sections = []
        
        # Document state summary
        if self.document_state:
            sections.append("=== DOCUMENT STATE ===")
            sections.append(f"Path: {self.document_state.document_path}")
            sections.append(f"Type: {self.document_state.document_type or 'unknown'}")
            sections.append(f"Progress: {len(self.document_state.sections_explored)} sections explored")
            if self.document_state.current_goal:
                sections.append(f"Current Goal: {self.document_state.current_goal}")
            if self.document_state.current_hypothesis:
                sections.append(f"Current Hypothesis: {self.document_state.current_hypothesis}")
            sections.append("")
        
        # Key findings
        if self.document_state and self.document_state.key_findings:
            sections.append("=== KEY FINDINGS SO FAR ===")
            for finding in self.document_state.key_findings[-5:]:  # Last 5
                confidence_str = f"[{finding['confidence']:.0%} confidence]"
                sections.append(f"- {finding['finding'][:150]} {confidence_str}")
            sections.append("")
        
        # Relevant memories from previous steps
        if self.relevant_memories:
            sections.append("=== RELEVANT PREVIOUS STEPS ===")
            for i, memory in enumerate(self.relevant_memories[:10], 1):
                memory_type = memory.memory_type.upper()
                content_preview = memory.content[:200].replace('\n', ' ')
                sections.append(f"{i}. [{memory_type}] {content_preview}...")
            sections.append("")
        
        # Previous step chain (recent context)
        if self.step_chain:
            sections.append("=== RECENT EXPLORATION CHAIN ===")
            for step in self.step_chain[-3:]:  # Last 3 steps
                sections.append(f"Turn {step.turn_number}: {step.step_type.value}")
                if step.insight:
                    sections.append(f"  -> Insight: {step.insight[:100]}...")
            sections.append("")
        
        context = "\n".join(sections)
        
        # Truncate if needed
        if len(context.encode('utf-8')) > max_bytes:
            context = context[:max_bytes - 50].rsplit('\n', 1)[0]
            context += "\n\n...[additional context truncated]"
        
        return context


@dataclass
class ExplorationResult:
    """Result of a Matryoshka exploration session."""
    session_id: str
    success: bool
    document_path: str
    original_query: str
    
    # Exploration data
    steps: List[ExplorationStep] = field(default_factory=list)
    final_synthesis: Optional[str] = None
    key_findings: List[str] = field(default_factory=list)
    
    # Statistics
    total_turns: int = 0
    total_execution_time_ms: float = 0.0
    total_tokens_used: int = 0
    memories_created: int = 0
    
    # Timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # Error info
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "success": self.success,
            "document_path": self.document_path,
            "original_query": self.original_query,
            "steps": [s.to_dict() for s in self.steps],
            "final_synthesis": self.final_synthesis,
            "key_findings": self.key_findings,
            "total_turns": self.total_turns,
            "total_execution_time_ms": self.total_execution_time_ms,
            "total_tokens_used": self.total_tokens_used,
            "memories_created": self.memories_created,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
        }


@dataclass
class SynthesisResult:
    """Result of synthesizing findings from a Matryoshka session."""
    session_id: str
    synthesis: str
    
    # Derived from
    steps_used: int = 0
    memories_considered: int = 0
    
    # Quality metrics
    confidence_score: float = 0.0
    coverage_score: float = 0.0  # How much of document was covered
    
    # Structured findings
    key_findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Source tracking
    source_memory_ids: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "synthesis": self.synthesis,
            "steps_used": self.steps_used,
            "memories_considered": self.memories_considered,
            "confidence_score": self.confidence_score,
            "coverage_score": self.coverage_score,
            "key_findings": self.key_findings,
            "recommendations": self.recommendations,
            "source_memory_ids": self.source_memory_ids,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AnalysisResult:
    """High-level result from analyzing a document with memory-backed Matryoshka."""
    session_id: str
    success: bool
    document_path: str
    query: str
    
    # Results
    answer: Optional[str] = None
    findings: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)
    
    # Context
    exploration_summary: Optional[str] = None
    relevant_memories_accessed: int = 0
    
    # Timing
    processing_time_ms: float = 0.0
    
    # Error
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "success": self.success,
            "document_path": self.document_path,
            "query": self.query,
            "answer": self.answer,
            "findings": self.findings,
            "code_examples": self.code_examples,
            "exploration_summary": self.exploration_summary,
            "relevant_memories_accessed": self.relevant_memories_accessed,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
        }


# =============================================================================
# MATRYOSHKA MEMORY BRIDGE
# =============================================================================

class MatryoshkaMemoryBridge:
    """
    Bridges Matryoshka RLM with Unified Memory System.
    
    Each Matryoshka exploration turn is:
    - Indexed through 4 layers (hierarchical, graph, hash, semantic)
    - Added to always-true state
    - Available via hybrid retrieval for future turns
    
    This prevents context rot by maintaining full exploration history
    outside the LLM context window, with intelligent retrieval.
    """
    
    def __init__(self, unified_memory: Optional[UnifiedMemorySystem] = None):
        """
        Initialize the memory bridge.
        
        Args:
            unified_memory: Existing UnifiedMemorySystem instance, or None to create new
        """
        self._lock = threading.RLock()
        
        # Initialize or use provided unified memory
        if unified_memory:
            self.unified_memory = unified_memory
        elif UNIFIED_MEMORY_AVAILABLE:
            self.unified_memory = create_unified_system(
                db_dir="./matryoshka_memory",
                max_context_tokens=8000,
                enable_maintenance=True
            )
        else:
            self.unified_memory = None
            logger.warning("No unified memory system available - bridge will operate in fallback mode")
        
        # Session tracking
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._document_states: Dict[str, DocumentState] = {}
        self._exploration_steps: Dict[str, List[ExplorationStep]] = defaultdict(list)
        
        # Step ID to memory ID mapping
        self._step_memory_map: Dict[str, str] = {}
        
        logger.info("MatryoshkaMemoryBridge initialized")
    
    def record_exploration_step(
        self,
        session_id: str,
        turn_number: int,
        query: str,
        code_executed: str,
        observation: str,
        insight: str,
        step_type: ExplorationStepType = ExplorationStepType.OBSERVATION,
        document_path: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 0.5
    ) -> Optional[UnifiedMemory]:
        """
        Record one Matryoshka exploration step.
        Indexes through all 4 layers automatically.
        
        Args:
            session_id: Unique session identifier
            turn_number: Current turn number in exploration
            query: The query/task for this step
            code_executed: Python code that was executed
            observation: Raw observation from execution
            insight: Derived insight
            step_type: Type of exploration step
            document_path: Path to document being analyzed
            importance: Importance score (0.0 - 1.0)
            confidence: Confidence score (0.0 - 1.0)
            
        Returns:
            UnifiedMemory entry if successful, None otherwise
        """
        if not self.unified_memory:
            logger.warning("Cannot record step - no unified memory available")
            return None
        
        try:
            with self._lock:
                # Create exploration step
                step_id = f"step_{session_id}_{turn_number}_{uuid.uuid4().hex[:8]}"
                
                # Find previous step for chaining
                previous_steps = self._exploration_steps.get(session_id, [])
                previous_step_id = previous_steps[-1].step_id if previous_steps else None
                
                step = ExplorationStep(
                    step_id=step_id,
                    session_id=session_id,
                    turn_number=turn_number,
                    step_type=step_type,
                    query=query,
                    code_executed=code_executed,
                    observation=observation,
                    insight=insight,
                    previous_step_id=previous_step_id,
                    importance=importance,
                    confidence=confidence,
                    document_path=document_path
                )
                
                # Store step locally
                self._exploration_steps[session_id].append(step)
                
                # Create unified memory entry
                memory = UnifiedMemory(
                    memory_id=step_id,
                    content=step.to_memory_content(),
                    memory_type=step_type.value,
                    importance=importance,
                    confidence=confidence,
                    source_conversation=session_id,
                    source_turn=turn_number,
                    related_memory_ids=[previous_step_id] if previous_step_id else []
                )
                
                # Index through unified memory system (4 layers)
                self._index_memory(memory, session_id)
                
                # Update document state if available
                if session_id in self._document_states:
                    doc_state = self._document_states[session_id]
                    doc_state.total_turns = max(doc_state.total_turns, turn_number)
                    if insight:
                        doc_state.add_finding(insight, confidence, step_id)
                    self._update_document_state_in_memory(session_id, doc_state)
                
                # Map step to memory
                self._step_memory_map[step_id] = memory.memory_id
                
                logger.debug(f"Recorded exploration step {step_id} for session {session_id}")
                return memory
                
        except Exception as e:
            logger.error(f"Error recording exploration step: {e}", exc_info=True)
            return None
    
    def _index_memory(self, memory: UnifiedMemory, session_id: str) -> None:
        """Index a memory through the unified memory system."""
        if not self.unified_memory:
            return
        
        try:
            # Use the unified memory system's internal indexing
            self.unified_memory._index_memory(memory)
            
            # Also add to conversation tracking
            self.unified_memory._conversation_memories[session_id].add(memory.memory_id)
            self.unified_memory._memory_registry[memory.memory_id] = memory
            
        except Exception as e:
            logger.warning(f"Error indexing memory: {e}")
    
    def _update_document_state_in_memory(self, session_id: str, 
                                          doc_state: DocumentState) -> None:
        """Update document state in unified memory."""
        if not self.unified_memory or not STATE_MANAGER_AVAILABLE:
            return
        
        try:
            # Convert to facts and update state
            facts = doc_state.to_state_facts()
            
            # Create a turn result for state update
            turn_result = TurnResult(
                turn_number=doc_state.total_turns,
                input_text=f"Exploration turn {doc_state.total_turns}",
                output_text=f"Updated document state with {len(facts)} facts",
                extracted_facts=facts,
                proposed_decisions=[]
            )
            
            # Update state in state manager
            self.unified_memory.state_manager.update_from_turn(
                conversation_id=session_id,
                turn_result=turn_result
            )
            
        except Exception as e:
            logger.warning(f"Error updating document state: {e}")
    
    def get_exploration_context(
        self,
        session_id: str,
        current_query: str,
        max_memories: int = 15
    ) -> ExplorationContext:
        """
        Get relevant context for current exploration step.
        Uses hybrid retrieval across all previous steps.
        
        Args:
            session_id: Session identifier
            current_query: Current query to use for relevance scoring
            max_memories: Maximum memories to retrieve (default 15)
            
        Returns:
            ExplorationContext with document state and relevant memories
        """
        context = ExplorationContext()
        
        try:
            with self._lock:
                # Get document state
                if session_id in self._document_states:
                    context.document_state = self._document_states[session_id]
                
                # Get step chain for continuity
                if session_id in self._exploration_steps:
                    context.step_chain = self._exploration_steps[session_id][-5:]  # Last 5
                
                # Hybrid retrieval from unified memory
                if self.unified_memory:
                    retrieved = self._hybrid_retrieve_for_session(
                        session_id, current_query, max_memories
                    )
                    context.relevant_memories = retrieved
                    context.memories_in_context = len(retrieved)
                    context.total_memories_available = len(
                        self._exploration_steps.get(session_id, [])
                    )
                
                # Calculate context size
                prompt_context = context.to_prompt_context()
                context.context_size_bytes = len(prompt_context.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error getting exploration context: {e}")
        
        return context
    
    def _hybrid_retrieve_for_session(
        self,
        session_id: str,
        query: str,
        limit: int = 15
    ) -> List[UnifiedMemory]:
        """
        Perform hybrid retrieval for a session.
        
        Uses multiple strategies:
        1. Semantic similarity to query
        2. Recency (recent steps more relevant)
        3. Importance (high-importance steps)
        4. Graph traversal (related steps)
        """
        if not self.unified_memory:
            return []
        
        memories = []
        
        try:
            # Use unified memory's hybrid retrieval if available
            retrieved = self.unified_memory._hybrid_retrieve(
                query=query,
                conversation_id=session_id,
                limit=limit
            )
            
            if retrieved:
                memories.extend(retrieved)
            else:
                # Fallback: simple scoring from local steps
                memories = self._fallback_retrieval(session_id, query, limit)
                
        except Exception as e:
            logger.warning(f"Hybrid retrieval error: {e}")
            memories = self._fallback_retrieval(session_id, query, limit)
        
        return memories[:limit]
    
    def _fallback_retrieval(
        self,
        session_id: str,
        query: str,
        limit: int
    ) -> List[UnifiedMemory]:
        """Fallback retrieval when hybrid retriever is unavailable."""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored = []
        steps = self._exploration_steps.get(session_id, [])
        
        for step in steps:
            score = 0.0
            
            # Content similarity
            content = step.to_memory_content().lower()
            
            # Word overlap
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            score += overlap * 2.0
            
            # Exact phrase match
            if query_lower in content:
                score += 10.0
            
            # Importance bonus
            score += step.importance * 5.0
            
            # Recency bonus
            if steps:
                recency = step.turn_number / max(steps[-1].turn_number, 1)
                score += recency * 3.0
            
            # Create unified memory from step
            memory = UnifiedMemory(
                memory_id=step.step_id,
                content=step.to_memory_content(),
                memory_type=step.step_type.value,
                importance=step.importance,
                confidence=step.confidence,
                source_conversation=session_id,
                source_turn=step.turn_number
            )
            
            scored.append((score, memory))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:limit]]
    
    def synthesize_findings(self, session_id: str) -> SynthesisResult:
        """
        Synthesize final findings using state + all indexed memories.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SynthesisResult with comprehensive findings
        """
        result = SynthesisResult(session_id=session_id, synthesis="")
        
        try:
            with self._lock:
                steps = self._exploration_steps.get(session_id, [])
                doc_state = self._document_states.get(session_id)
                
                if not steps:
                    result.synthesis = "No exploration steps found for this session."
                    return result
                
                # Build synthesis from accumulated findings
                synthesis_parts = []
                
                # 1. Document overview
                if doc_state:
                    synthesis_parts.append(f"# Analysis of {doc_state.document_path}")
                    synthesis_parts.append(f"\nDocument Type: {doc_state.document_type or 'Unknown'}")
                    synthesis_parts.append(f"Total Turns: {doc_state.total_turns}")
                    synthesis_parts.append(f"Sections Explored: {len(doc_state.sections_explored)}")
                
                # 2. Key findings
                if doc_state and doc_state.key_findings:
                    synthesis_parts.append("\n## Key Findings")
                    for finding in sorted(
                        doc_state.key_findings, 
                        key=lambda f: f.get('confidence', 0), 
                        reverse=True
                    )[:10]:  # Top 10 by confidence
                        confidence = finding.get('confidence', 0)
                        finding_text = finding.get('finding', '')
                        synthesis_parts.append(f"\n- ({confidence:.0%} confidence) {finding_text}")
                
                # 3. Exploration insights
                synthesis_parts.append("\n## Exploration Summary")
                
                # Group steps by type
                steps_by_type: Dict[ExplorationStepType, List[ExplorationStep]] = defaultdict(list)
                for step in steps:
                    steps_by_type[step.step_type].append(step)
                
                for step_type, type_steps in sorted(steps_by_type.items(), 
                                                      key=lambda x: len(x[1]), 
                                                      reverse=True):
                    synthesis_parts.append(f"\n{step_type.value.replace('_', ' ').title()}: {len(type_steps)} steps")
                    
                    # Add top insights for this type
                    insights = [s.insight for s in type_steps if s.insight]
                    for insight in insights[:3]:
                        synthesis_parts.append(f"  - {insight[:100]}...")
                
                # 4. Retrieve any additional relevant memories
                if self.unified_memory:
                    all_memories = self._hybrid_retrieve_for_session(
                        session_id, 
                        "final synthesis findings summary", 
                        limit=20
                    )
                    result.memories_considered = len(all_memories)
                    result.source_memory_ids = [m.memory_id for m in all_memories]
                
                # Combine synthesis
                result.synthesis = "\n".join(synthesis_parts)
                result.steps_used = len(steps)
                result.key_findings = doc_state.key_findings if doc_state else []
                
                # Calculate confidence based on exploration coverage
                if doc_state:
                    total_sections = len(doc_state.sections_explored) + len(doc_state.sections_remaining)
                    if total_sections > 0:
                        result.coverage_score = len(doc_state.sections_explored) / total_sections
                    
                    # Average confidence of findings
                    if doc_state.key_findings:
                        avg_confidence = sum(
                            f.get('confidence', 0.5) for f in doc_state.key_findings
                        ) / len(doc_state.key_findings)
                        result.confidence_score = avg_confidence
                
                logger.info(f"Synthesized findings for session {session_id}: "
                           f"{result.steps_used} steps, "
                           f"{len(result.key_findings)} findings")
                
        except Exception as e:
            logger.error(f"Error synthesizing findings: {e}")
            result.synthesis = f"Error during synthesis: {str(e)}"
        
        return result
    
    def initialize_document_state(
        self,
        session_id: str,
        document_path: str,
        document_type: Optional[str] = None,
        document_size: int = 0,
        initial_goal: Optional[str] = None
    ) -> DocumentState:
        """
        Initialize state for a new document analysis.
        
        Args:
            session_id: Unique session identifier
            document_path: Path to document being analyzed
            document_type: Type of document (python, markdown, etc.)
            document_size: Size in bytes
            initial_goal: Initial exploration goal/query
            
        Returns:
            Initialized DocumentState
        """
        with self._lock:
            doc_state = DocumentState(
                session_id=session_id,
                document_path=document_path,
                document_type=document_type,
                document_size_bytes=document_size,
                current_goal=initial_goal
            )
            
            self._document_states[session_id] = doc_state
            
            # Create conversation in unified memory
            if self.unified_memory and self.unified_memory.state_manager:
                try:
                    self.unified_memory.state_manager.create_conversation(session_id)
                except Exception as e:
                    logger.warning(f"Could not create conversation state: {e}")
            
            logger.info(f"Initialized document state for session {session_id}: {document_path}")
            return doc_state
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        with self._lock:
            steps = self._exploration_steps.get(session_id, [])
            doc_state = self._document_states.get(session_id)
            
            return {
                "session_id": session_id,
                "total_steps": len(steps),
                "document_path": doc_state.document_path if doc_state else None,
                "document_type": doc_state.document_type if doc_state else None,
                "total_turns": doc_state.total_turns if doc_state else len(steps),
                "findings_count": len(doc_state.key_findings) if doc_state else 0,
                "sections_explored": len(doc_state.sections_explored) if doc_state else 0,
                "exploration_complete": doc_state.exploration_complete if doc_state else False,
            }
    
    def cleanup_session(self, session_id: str) -> None:
        """Clean up session data."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
            if session_id in self._document_states:
                del self._document_states[session_id]
            if session_id in self._exploration_steps:
                del self._exploration_steps[session_id]
            
            logger.info(f"Cleaned up session {session_id}")


# =============================================================================
# MATRYOSHKA EXPLORATION SESSION
# =============================================================================

class MatryoshkaExplorationSession:
    """
    A Matryoshka session backed by unified memory.
    Replaces the simple StatefulMatryoshkaClient.sessions dict.
    
    Features:
    - Full 4-layer memory indexing for each exploration step
    - Hybrid retrieval for context building
    - Document state management
    - Cross-session learning via semantic indexing
    """
    
    def __init__(
        self,
        session_id: str,
        document_path: str,
        query: str,
        memory_bridge: Optional[MatryoshkaMemoryBridge] = None,
        unified_memory: Optional[UnifiedMemorySystem] = None,
        matryoshka_client: Optional[MatryoshkaClient] = None
    ):
        """
        Initialize exploration session.
        
        Args:
            session_id: Unique session identifier
            document_path: Path to document being analyzed
            query: Initial exploration query
            memory_bridge: Existing memory bridge or None to create new
            unified_memory: Unified memory system instance
            matryoshka_client: Matryoshka client for code execution
        """
        self.session_id = session_id
        self.document_path = document_path
        self.original_query = query
        
        # Initialize or use provided memory bridge
        if memory_bridge:
            self.memory_bridge = memory_bridge
        else:
            self.memory_bridge = MatryoshkaMemoryBridge(unified_memory)
        
        # Matryoshka client for actual code generation/execution
        if matryoshka_client:
            self.matryoshka = matryoshka_client
        elif MATRYOSHKA_ADAPTER_AVAILABLE:
            self.matryoshka = MatryoshkaClient()
        else:
            self.matryoshka = None
        
        # Initialize document state
        doc_type = self._detect_document_type(document_path)
        doc_size = self._get_document_size(document_path)
        
        self.document_state = self.memory_bridge.initialize_document_state(
            session_id=session_id,
            document_path=document_path,
            document_type=doc_type,
            document_size=doc_size,
            initial_goal=query
        )
        
        # Session state
        self.current_turn = 0
        self.max_turns = 10
        self.is_complete = False
        self._lock = threading.RLock()
        
        # Record initialization step
        self._record_initialization()
        
        logger.info(f"Created MatryoshkaExplorationSession {session_id} for {document_path}")
    
    def _detect_document_type(self, document_path: str) -> Optional[str]:
        """Detect document type from file extension."""
        ext = document_path.split('.')[-1].lower() if '.' in document_path else ''
        type_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'md': 'markdown',
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'txt': 'text',
            'csv': 'csv',
            'html': 'html',
            'css': 'css',
        }
        return type_map.get(ext)
    
    def _get_document_size(self, document_path: str) -> int:
        """Get document size in bytes."""
        try:
            import os
            return os.path.getsize(document_path)
        except Exception:
            return 0
    
    def _record_initialization(self) -> None:
        """Record initialization step in memory."""
        self.memory_bridge.record_exploration_step(
            session_id=self.session_id,
            turn_number=0,
            query=self.original_query,
            code_executed="",
            observation=f"Initialized analysis of {self.document_path}",
            insight=f"Starting exploration with goal: {self.original_query}",
            step_type=ExplorationStepType.INITIALIZATION,
            document_path=self.document_path,
            importance=0.8,
            confidence=1.0
        )
    
    def explore(
        self,
        max_turns: int = 10,
        llm_code_callback: Optional[Callable[[str], str]] = None
    ) -> ExplorationResult:
        """
        Run Matryoshka exploration with unified memory backing.
        
        Each turn:
        1. Get context from unified memory (hybrid retrieval)
        2. Generate code to explore
        3. Execute and observe
        4. Record step in unified memory (4-layer indexing)
        5. Update state
        
        Args:
            max_turns: Maximum exploration turns
            llm_code_callback: Callback function(query, context) -> code
            
        Returns:
            ExplorationResult with full exploration data
        """
        with self._lock:
            self.max_turns = max_turns
            result = ExplorationResult(
                session_id=self.session_id,
                success=False,
                document_path=self.document_path,
                original_query=self.original_query
            )
            
            start_time = time.time()
            
            try:
                for turn in range(1, max_turns + 1):
                    self.current_turn = turn
                    
                    # 1. Get context from unified memory
                    context = self._build_exploration_context()
                    
                    # 2. Generate exploration code
                    code, query = self._generate_exploration_code(context, llm_code_callback)
                    
                    # 3. Execute code and get observation
                    observation, execution_time = self._execute_exploration_code(code)
                    
                    # 4. Derive insight from observation
                    insight = self._derive_insight(observation, query)
                    
                    # 5. Record step in unified memory
                    self.memory_bridge.record_exploration_step(
                        session_id=self.session_id,
                        turn_number=turn,
                        query=query,
                        code_executed=code,
                        observation=observation,
                        insight=insight,
                        step_type=ExplorationStepType.OBSERVATION,
                        document_path=self.document_path,
                        importance=self._calculate_importance(observation, insight),
                        confidence=self._estimate_confidence(observation)
                    )
                    
                    # Check for completion
                    if self._check_exploration_complete(observation, insight):
                        self.is_complete = True
                        break
                
                # Synthesize final findings
                synthesis = self.memory_bridge.synthesize_findings(self.session_id)
                result.final_synthesis = synthesis.synthesis
                result.key_findings = [f["finding"] for f in synthesis.key_findings]
                
                result.success = True
                result.total_turns = self.current_turn
                result.total_execution_time_ms = (time.time() - start_time) * 1000
                result.steps = self.memory_bridge._exploration_steps.get(self.session_id, [])
                result.memories_created = len(result.steps)
                result.completed_at = datetime.utcnow()
                
            except Exception as e:
                logger.error(f"Error during exploration: {e}", exc_info=True)
                result.error_message = str(e)
                result.success = False
            
            return result
    
    def _build_exploration_context(self) -> str:
        """Build context for current exploration step."""
        # Get exploration context from memory bridge
        exploration_context = self.memory_bridge.get_exploration_context(
            session_id=self.session_id,
            current_query=self.document_state.current_goal or self.original_query,
            max_memories=15
        )
        
        return exploration_context.to_prompt_context(max_bytes=5120)
    
    def _generate_exploration_code(
        self,
        context: str,
        llm_callback: Optional[Callable[[str], str]] = None
    ) -> Tuple[str, str]:
        """
        Generate code for next exploration step.
        
        Args:
            context: Context from memory system
            llm_callback: Optional callback for code generation
            
        Returns:
            (generated_code, query_used)
        """
        query = self._build_exploration_query(context)
        
        if llm_callback:
            code = llm_callback(query)
        elif self.matryoshka:
            # Use Matryoshka for code generation
            try:
                code = self.matryoshka.analyze(query, self.document_path, max_turns=1)
            except Exception as e:
                logger.warning(f"Matryoshka code generation failed: {e}")
                code = self._fallback_code_generation(context)
        else:
            code = self._fallback_code_generation(context)
        
        return code, query
    
    def _build_exploration_query(self, context: str) -> str:
        """Build the query for the next exploration step."""
        query_parts = [
            f"Goal: {self.document_state.current_goal or self.original_query}",
            "\nContext:",
            context[:2000],  # Limit context size
            "\nGenerate Python code to explore the document and make progress toward the goal.",
            "The code should read from the document and extract relevant information."
        ]
        return "\n".join(query_parts)
    
    def _fallback_code_generation(self, context: str) -> str:
        """Fallback code generation when LLM is unavailable."""
        doc_type = self.document_state.document_type or "text"
        
        if doc_type == "python":
            return f'''
# Fallback exploration code for Python file
with open("{self.document_path}", "r") as f:
    content = f.read()
    
# Analyze structure
lines = content.split("\\n")
print(f"File has {{len(lines)}} lines")

# Look for classes and functions
import re
classes = re.findall(r"class \\w+", content)
functions = re.findall(r"def \\w+", content)
print(f"Found {{len(classes)}} classes, {{len(functions)}} functions")
'''
        else:
            return f'''
# Fallback exploration code
with open("{self.document_path}", "r") as f:
    content = f.read()
    
lines = content.split("\\n")
print(f"Document has {{len(lines)}} lines, {{len(content)}} characters")
print("First 500 characters:")
print(content[:500])
'''
    
    def _execute_exploration_code(self, code: str) -> Tuple[str, float]:
        """
        Execute exploration code and return observation.
        
        Returns:
            (observation_string, execution_time_ms)
        """
        import subprocess
        import tempfile
        import os
        
        start_time = time.time()
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                tmp_path = f.name
            
            # Execute in subprocess for safety
            result = subprocess.run(
                ['python', tmp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Build observation
            observation_parts = []
            if result.stdout:
                observation_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                observation_parts.append(f"STDERR:\n{result.stderr}")
            if result.returncode != 0:
                observation_parts.append(f"Return code: {result.returncode}")
            
            observation = "\n\n".join(observation_parts) or "No output"
            
        except subprocess.TimeoutExpired:
            observation = "Execution timed out after 30 seconds"
            execution_time = 30000
        except Exception as e:
            observation = f"Execution error: {str(e)}"
            execution_time = (time.time() - start_time) * 1000
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        return observation, execution_time
    
    def _derive_insight(self, observation: str, query: str) -> str:
        """Derive insight from observation."""
        # Simple insight extraction - in production, use LLM
        lines = observation.split('\n')
        
        # Look for key findings in output
        insights = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                if any(keyword in line.lower() for keyword in 
                       ['found', 'total', 'count', 'error', 'warning', 'class', 'function']):
                    insights.append(line)
        
        return insights[0] if insights else f"Observed: {observation[:100]}..."
    
    def _calculate_importance(self, observation: str, insight: str) -> float:
        """Calculate importance score for a step."""
        importance = 0.5  # Base importance
        
        # Increase for key findings
        if any(kw in observation.lower() for kw in ['error', 'exception', 'critical']):
            importance += 0.3
        
        if any(kw in insight.lower() for kw in ['key', 'critical', 'important', 'finding']):
            importance += 0.2
        
        # Decrease for empty/short observations
        if len(observation) < 50:
            importance -= 0.2
        
        return max(0.0, min(1.0, importance))
    
    def _estimate_confidence(self, observation: str) -> float:
        """Estimate confidence in observation."""
        confidence = 0.7  # Base confidence
        
        # Increase for clear output
        if 'STDOUT:' in observation and len(observation) > 100:
            confidence += 0.1
        
        # Decrease for errors
        if 'STDERR:' in observation or 'error' in observation.lower():
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _check_exploration_complete(self, observation: str, insight: str) -> bool:
        """Check if exploration should terminate."""
        # Check for explicit completion markers
        completion_markers = [
            'exploration complete',
            'analysis complete',
            'done',
            'finished',
            'no more data'
        ]
        
        combined = (observation + " " + insight).lower()
        return any(marker in combined for marker in completion_markers)
    
    def get_current_context(self) -> ExplorationContext:
        """Get current exploration context."""
        return self.memory_bridge.get_exploration_context(
            session_id=self.session_id,
            current_query=self.document_state.current_goal or self.original_query
        )
    
    def add_finding(self, finding: str, confidence: float = 0.7) -> None:
        """Manually add a finding to the document state."""
        self.document_state.add_finding(finding, confidence)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self.memory_bridge.get_session_stats(self.session_id)


# =============================================================================
# UNIFIED MATRYOSHKA CLIENT
# =============================================================================

class UnifiedMatryoshkaClient:
    """
    Enhanced Matryoshka client using unified memory system.
    Drop-in replacement for StatefulMatryoshkaClient.
    
    Features:
    - Full 4-layer memory indexing for each exploration step
    - Cross-session learning via semantic/graph indexes
    - Stateful exploration with document state management
    - Hybrid retrieval for context building
    - Thread-safe session management
    """
    
    def __init__(
        self,
        unified_memory: Optional[UnifiedMemorySystem] = None,
        executable_path: Optional[str] = None
    ):
        """
        Initialize unified Matryoshka client.
        
        Args:
            unified_memory: Existing UnifiedMemorySystem or None to create new
            executable_path: Path to Matryoshka executable
        """
        self._lock = threading.RLock()
        
        # Initialize memory bridge
        self.memory_bridge = MatryoshkaMemoryBridge(unified_memory)
        
        # Initialize base Matryoshka client
        if MATRYOSHKA_ADAPTER_AVAILABLE:
            self.base_client = MatryoshkaClient(executable_path=executable_path)
        else:
            self.base_client = None
        
        # Session tracking
        self._active_sessions: Dict[str, MatryoshkaExplorationSession] = {}
        self._analysis_results: Dict[str, AnalysisResult] = {}
        
        # Configuration
        self.default_max_turns = 10
        self.context_retrieval_limit = 15
        
        logger.info("UnifiedMatryoshkaClient initialized")
    
    def analyze_with_memory(
        self,
        query: str,
        file_path: str,
        session_id: Optional[str] = None,
        max_turns: int = 10,
        llm_code_callback: Optional[Callable[[str], str]] = None
    ) -> AnalysisResult:
        """
        Analyze document using Matryoshka with full memory system.
        
        - Creates session with unified memory
        - Each exploration step indexed through 4 layers
        - State maintained across turns
        - Final synthesis uses all indexed memories
        
        Args:
            query: Analysis query/question
            file_path: Path to document to analyze
            session_id: Optional session ID (generated if not provided)
            max_turns: Maximum exploration turns
            llm_code_callback: Optional callback for code generation
            
        Returns:
            AnalysisResult with findings and context
        """
        start_time = time.time()
        
        # Generate session ID if not provided
        if not session_id:
            session_id = f"matryoshka_{uuid.uuid4().hex[:16]}"
        
        result = AnalysisResult(
            session_id=session_id,
            success=False,
            document_path=file_path,
            query=query
        )
        
        try:
            with self._lock:
                # Check if file exists
                import os
                if not os.path.exists(file_path):
                    result.error = f"File not found: {file_path}"
                    return result
                
                # Create exploration session
                session = MatryoshkaExplorationSession(
                    session_id=session_id,
                    document_path=file_path,
                    query=query,
                    memory_bridge=self.memory_bridge,
                    matryoshka_client=self.base_client
                )
                
                self._active_sessions[session_id] = session
            
            # Run exploration
            exploration_result = session.explore(
                max_turns=max_turns,
                llm_code_callback=llm_code_callback
            )
            
            # Build analysis result
            result.success = exploration_result.success
            result.answer = exploration_result.final_synthesis
            result.findings = exploration_result.key_findings
            result.exploration_summary = (
                f"Completed {exploration_result.total_turns} turns of exploration. "
                f"Created {exploration_result.memories_created} indexed memories."
            )
            result.relevant_memories_accessed = exploration_result.memories_created
            result.processing_time_ms = exploration_result.total_execution_time_ms
            
            if exploration_result.error_message:
                result.error = exploration_result.error_message
            
            # Store result
            self._analysis_results[session_id] = result
            
            logger.info(f"Completed analysis for session {session_id}: "
                       f"{len(result.findings)} findings")
            
        except Exception as e:
            logger.error(f"Error in analyze_with_memory: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def continue_analysis(
        self,
        session_id: str,
        follow_up_query: str,
        max_turns: int = 5,
        llm_code_callback: Optional[Callable[[str], str]] = None
    ) -> AnalysisResult:
        """
        Continue a previous analysis session.
        Uses unified memory to recall previous exploration.
        
        Args:
            session_id: Session ID from previous analysis
            follow_up_query: New query to explore
            max_turns: Additional exploration turns
            llm_code_callback: Optional callback for code generation
            
        Returns:
            AnalysisResult with updated findings
        """
        start_time = time.time()
        
        result = AnalysisResult(
            session_id=session_id,
            success=False,
            document_path="",
            query=follow_up_query
        )
        
        try:
            with self._lock:
                # Check for existing session
                if session_id not in self._active_sessions:
                    result.error = f"Session {session_id} not found"
                    return result
                
                session = self._active_sessions[session_id]
                
                # Update query and reset completion status
                session.document_state.current_goal = follow_up_query
                session.is_complete = False
                session.current_turn = session.document_state.total_turns
            
            # Continue exploration
            exploration_result = session.explore(
                max_turns=session.current_turn + max_turns,
                llm_code_callback=llm_code_callback
            )
            
            # Build result
            result.success = exploration_result.success
            result.document_path = session.document_path
            result.answer = exploration_result.final_synthesis
            result.findings = exploration_result.key_findings
            result.exploration_summary = (
                f"Continued analysis with {max_turns} additional turns. "
                f"Total: {exploration_result.total_turns} turns."
            )
            result.relevant_memories_accessed = exploration_result.memories_created
            result.processing_time_ms = exploration_result.total_execution_time_ms
            
            # Update stored result
            self._analysis_results[session_id] = result
            
            logger.info(f"Continued analysis for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error in continue_analysis: {e}", exc_info=True)
            result.success = False
            result.error = str(e)
            result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def get_session_memory(self, session_id: str) -> Optional[ExplorationContext]:
        """
        Get the memory context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            ExplorationContext if session exists
        """
        if session_id not in self._active_sessions:
            return None
        
        session = self._active_sessions[session_id]
        return session.get_current_context()
    
    def search_across_sessions(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for insights across all analysis sessions.
        Enables cross-document learning.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of relevant findings from all sessions
        """
        results = []
        
        try:
            # Use unified memory's hybrid retrieval
            for session_id in self._active_sessions:
                context = self.memory_bridge.get_exploration_context(
                    session_id=session_id,
                    current_query=query,
                    max_memories=limit
                )
                
                for memory in context.relevant_memories:
                    results.append({
                        "session_id": session_id,
                        "memory_id": memory.memory_id,
                        "content": memory.content,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance,
                        "confidence": memory.confidence
                    })
            
            # Sort by importance and confidence
            results.sort(
                key=lambda x: (x["importance"] * x["confidence"]),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error searching across sessions: {e}")
        
        return results[:limit]
    
    def get_session_synthesis(self, session_id: str) -> Optional[SynthesisResult]:
        """
        Get synthesized findings for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            SynthesisResult if session exists
        """
        if session_id not in self._active_sessions:
            return None
        
        return self.memory_bridge.synthesize_findings(session_id)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active analysis sessions."""
        with self._lock:
            sessions = []
            for session_id, session in self._active_sessions.items():
                stats = session.get_stats()
                sessions.append(stats)
            return sessions
    
    def close_session(self, session_id: str) -> bool:
        """
        Close an analysis session and clean up resources.
        
        Args:
            session_id: Session to close
            
        Returns:
            True if session was closed
        """
        with self._lock:
            if session_id not in self._active_sessions:
                return False
            
            # Clean up
            self.memory_bridge.cleanup_session(session_id)
            del self._active_sessions[session_id]
            
            logger.info(f"Closed session {session_id}")
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._lock:
            total_sessions = len(self._active_sessions)
            total_memories = sum(
                s.get_stats().get("total_steps", 0)
                for s in self._active_sessions.values()
            )
            
            return {
                "active_sessions": total_sessions,
                "total_indexed_memories": total_memories,
                "memory_bridge_healthy": self.memory_bridge.unified_memory is not None,
                "matryoshka_available": self.base_client is not None
            }
    
    def is_available(self) -> bool:
        """Check if the client is available."""
        return self.memory_bridge.unified_memory is not None


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_unified_matryoshka_client(
    db_dir: str = "./matryoshka_memory",
    executable_path: Optional[str] = None
) -> UnifiedMatryoshkaClient:
    """
    Factory function to create a UnifiedMatryoshkaClient.
    
    Args:
        db_dir: Directory for memory system databases
        executable_path: Path to Matryoshka executable
        
    Returns:
        Configured UnifiedMatryoshkaClient
        
    Example:
        >>> client = create_unified_matryoshka_client("./my_memory")
        >>> result = client.analyze_with_memory(
        ...     query="Find all functions in this file",
        ...     file_path="./my_code.py"
        ... )
        >>> print(result.answer)
    """
    # Create unified memory system
    if UNIFIED_MEMORY_AVAILABLE:
        unified_memory = create_unified_system(
            db_dir=db_dir,
            max_context_tokens=8000,
            enable_maintenance=True
        )
    else:
        unified_memory = None
    
    return UnifiedMatryoshkaClient(
        unified_memory=unified_memory,
        executable_path=executable_path
    )


def create_memory_backed_session(
    document_path: str,
    query: str,
    unified_memory: Optional[UnifiedMemorySystem] = None
) -> MatryoshkaExplorationSession:
    """
    Create a memory-backed exploration session.
    
    Args:
        document_path: Path to document to analyze
        query: Analysis query
        unified_memory: Optional existing unified memory system
        
    Returns:
        MatryoshkaExplorationSession ready to explore
    """
    session_id = f"exploration_{uuid.uuid4().hex[:16]}"
    
    return MatryoshkaExplorationSession(
        session_id=session_id,
        document_path=document_path,
        query=query,
        unified_memory=unified_memory
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Basic usage of unified Matryoshka client
    
    print("=" * 60)
    print("Matryoshka Unified Memory Integration Demo")
    print("=" * 60)
    
    # 1. Create client
    client = create_unified_matryoshka_client("./demo_memory")
    
    print(f"\nClient created:")
    print(f"  Available: {client.is_available()}")
    print(f"  Stats: {client.get_stats()}")
    
    # 2. Example: Create a sample file to analyze
    import tempfile
    import os
    
    sample_code = '''
def calculate_sum(numbers):
    """Calculate sum of a list of numbers."""
    return sum(numbers)

def find_max(numbers):
    """Find maximum value in a list."""
    return max(numbers) if numbers else None

class DataProcessor:
    def __init__(self):
        self.data = []
    
    def add_value(self, value):
        self.data.append(value)
    
    def process(self):
        return calculate_sum(self.data)
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(sample_code)
        temp_path = f.name
    
    try:
        # 3. Analyze with memory
        print(f"\nAnalyzing: {temp_path}")
        result = client.analyze_with_memory(
            query="Find all classes and functions in this file",
            file_path=temp_path,
            max_turns=3
        )
        
        print(f"\nResult:")
        print(f"  Success: {result.success}")
        print(f"  Session ID: {result.session_id}")
        print(f"  Findings: {len(result.findings)}")
        if result.findings:
            print(f"  First finding: {result.findings[0][:100]}...")
        print(f"  Processing time: {result.processing_time_ms:.0f}ms")
        
        # 4. Get synthesis
        synthesis = client.get_session_synthesis(result.session_id)
        if synthesis:
            print(f"\nSynthesis preview:")
            print(f"  {synthesis.synthesis[:300]}...")
        
        # 5. Continue analysis
        print("\nContinuing analysis...")
        continue_result = client.continue_analysis(
            session_id=result.session_id,
            follow_up_query="What are the methods in the DataProcessor class?",
            max_turns=2
        )
        
        print(f"  Continue success: {continue_result.success}")
        print(f"  Total findings now: {len(continue_result.findings)}")
        
        # 6. List sessions
        sessions = client.list_sessions()
        print(f"\nActive sessions: {len(sessions)}")
        for session in sessions:
            print(f"  - {session['session_id']}: {session['total_steps']} steps")
        
        # 7. Clean up
        client.close_session(result.session_id)
        print(f"\nSession closed.")
        
    finally:
        os.unlink(temp_path)
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
