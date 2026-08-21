"""
Knowledge Working Memory System

Implements the distinction between:
- Working Memory: What goes into the LLM prompt (small, curated, temporary)
- Long-term Memory: Persistent storage (large, indexed, survives turns)

Key Principle: The prompt is a working surface, not memory.
Memory lives outside and gets updated every turn.
"""
from __future__ import annotations


import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Set, TypeVar
from collections import deque


logger = logging.getLogger(__name__)


# ============================================================================
# Token Counting
# ============================================================================

class TokenCounter:
    """
    Estimates token counts for text.
    Uses a simple approximation (4 chars ≈ 1 token for English).
    Can be replaced with tiktoken or similar for production use.
    """
    
    def __init__(self, chars_per_token: float = 4.0):
        self.chars_per_token = chars_per_token
    
    def count(self, text: str) -> int:
        """Estimate token count for text."""
        if not text:
            return 0
        return max(1, int(len(text) / self.chars_per_token))
    
    def count_messages(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for a list of messages."""
        total = 0
        for msg in messages:
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
            # Add overhead per message
            total += 4
        return total


# ============================================================================
# Data Structures
# ============================================================================

class MemoryType(Enum):
    """Types of memories with different persistence characteristics."""
    STATE = auto()           # Always-true system state
    FACT = auto()            # Verified facts
    DECISION = auto()        # Made decisions
    INSIGHT = auto()         # Derived insights
    TEMPORARY = auto()       # Temporary reasoning (not persisted)
    QUERY = auto()           # User queries
    RESPONSE = auto()        # Assistant responses


class Priority(Enum):
    """Priority levels for context inclusion."""
    CRITICAL = 1      # Always include
    HIGH = 2          # Include if space
    MEDIUM = 3        # Include if room
    LOW = 4           # Only if lots of room
    DISCARDABLE = 5   # First to drop


@dataclass
class Memory:
    """
    A single memory unit.
    """
    id: str
    content: str
    memory_type: MemoryType
    priority: Priority = Priority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def touch(self):
        """Mark as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_prompt_text(self) -> str:
        """Convert to text suitable for prompt inclusion."""
        prefix = self.memory_type.name.lower()
        return f"[{prefix}] {self.content}"


@dataclass 
class TurnMetadata:
    """Metadata about a conversation turn."""
    turn_id: str
    timestamp: float
    query: str
    response: str
    duration_ms: float
    token_count: int
    memory_ids_accessed: List[str] = field(default_factory=list)
    new_memories_created: List[str] = field(default_factory=list)


@dataclass
class WorkingMemoryStats:
    """Statistics about working memory usage."""
    total_tokens_in_context: int
    max_tokens_allowed: int
    token_utilization_pct: float
    memories_in_context: int
    state_items: int
    buffer_items: int
    retrieval_hits: int
    retrieval_misses: int
    hit_rate: float
    avg_retrieval_time_ms: float
    turn_count: int


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    memories: List[Memory]
    total_candidates: int
    retrieval_time_ms: float
    query_embedding: Optional[List[float]] = None


# ============================================================================
# State Management
# ============================================================================

@dataclass
class StateSnapshot:
    """
    Always-true state that persists across turns.
    This is the 'program state' - not execution history.
    """
    facts: Dict[str, Any] = field(default_factory=dict)
    decisions: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    context_variables: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def update_fact(self, key: str, value: Any):
        """Update a fact in state."""
        self.facts[key] = value
        self.version += 1
        self.last_updated = time.time()
    
    def record_decision(self, key: str, decision: Any, rationale: str = ""):
        """Record a decision."""
        self.decisions[key] = {
            "decision": decision,
            "rationale": rationale,
            "timestamp": time.time()
        }
        self.version += 1
        self.last_updated = time.time()
    
    def set_preference(self, key: str, value: Any):
        """Set a user preference."""
        self.preferences[key] = value
        self.version += 1
        self.last_updated = time.time()
    
    def to_prompt_section(self) -> str:
        """Convert state to a prompt section."""
        lines = ["=== Current State ==="]
        
        if self.facts:
            lines.append("Facts:")
            for k, v in self.facts.items():
                lines.append(f"  - {k}: {v}")
        
        if self.decisions:
            lines.append("Decisions:")
            for k, v in self.decisions.items():
                if isinstance(v, dict):
                    lines.append(f"  - {k}: {v.get('decision', v)}")
                else:
                    lines.append(f"  - {k}: {v}")
        
        if self.preferences:
            lines.append("Preferences:")
            for k, v in self.preferences.items():
                lines.append(f"  - {k}: {v}")
        
        if self.context_variables:
            lines.append("Context:")
            for k, v in self.context_variables.items():
                lines.append(f"  - {k}: {v}")
        
        return "\n".join(lines) if len(lines) > 1 else ""


# ============================================================================
# Prompt Context
# ============================================================================

@dataclass
class PromptContext:
    """
    What actually goes into the LLM prompt.
    Structured for clean, effective prompting.
    """
    system_instruction: str = ""
    state_section: str = ""
    relevant_memories: List[Memory] = field(default_factory=list)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    current_query: str = ""
    working_buffer: List[str] = field(default_factory=list)
    
    def to_messages(self) -> List[Dict[str, str]]:
        """
        Convert to message format for LLM APIs.
        Returns: List of {role, content} dicts
        """
        messages = []
        
        # System message with instructions and state
        system_content = self.system_instruction
        if self.state_section:
            system_content += "\n\n" + self.state_section
        
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content.strip()
            })
        
        # Add relevant memories as context
        if self.relevant_memories:
            memory_content = "\n".join([
                m.to_prompt_text() for m in self.relevant_memories
            ])
            messages.append({
                "role": "system",
                "content": f"Relevant context:\n{memory_content}"
            })
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add working buffer (temporary reasoning)
        if self.working_buffer:
            buffer_text = "\n".join(self.working_buffer)
            messages.append({
                "role": "system",
                "content": f"Working notes:\n{buffer_text}"
            })
        
        # Current query
        if self.current_query:
            messages.append({
                "role": "user",
                "content": self.current_query
            })
        
        return messages
    
    def to_plain_text(self) -> str:
        """Convert to a single plain text string."""
        lines = []
        
        if self.system_instruction:
            lines.append(f"Instructions: {self.system_instruction}")
            lines.append("")
        
        if self.state_section:
            lines.append(self.state_section)
            lines.append("")
        
        if self.relevant_memories:
            lines.append("Relevant memories:")
            for mem in self.relevant_memories:
                lines.append(f"  - {mem.content}")
            lines.append("")
        
        if self.conversation_history:
            lines.append("Conversation:")
            for msg in self.conversation_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            lines.append("")
        
        if self.working_buffer:
            lines.append("Working notes:")
            for note in self.working_buffer:
                lines.append(f"  - {note}")
            lines.append("")
        
        if self.current_query:
            lines.append(f"User: {self.current_query}")
        
        return "\n".join(lines)


# ============================================================================
# Working Memory Buffer
# ============================================================================

class WorkingMemoryBuffer:
    """
    Very short-term buffer for current turn only.
    Holds temporary reasoning that doesn't get stored.
    Cleared each turn.
    """
    
    def __init__(self, max_items: int = 10, max_item_length: int = 500):
        self.items: deque = deque(maxlen=max_items)
        self.max_item_length = max_item_length
        self._lock = threading.RLock()
    
    def add(self, content: str, item_type: str = "reasoning") -> None:
        """Add an item to the buffer."""
        with self._lock:
            # Truncate if too long
            if len(content) > self.max_item_length:
                content = content[:self.max_item_length - 3] + "..."
            
            self.items.append({
                "content": content,
                "type": item_type,
                "timestamp": time.time()
            })
    
    def add_reasoning_step(self, step: str) -> None:
        """Add a temporary reasoning step."""
        self.add(step, "reasoning")
    
    def add_intermediate_result(self, result: str) -> None:
        """Add an intermediate calculation result."""
        self.add(result, "intermediate")
    
    def get_contents(self) -> List[str]:
        """Get all buffer contents as strings."""
        with self._lock:
            return [item["content"] for item in self.items]
    
    def clear(self) -> None:
        """Clear the buffer (call at end of turn)."""
        with self._lock:
            self.items.clear()
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return len(self.items) == 0
    
    def size(self) -> int:
        """Get number of items in buffer."""
        with self._lock:
            return len(self.items)


# ============================================================================
# Context Window Optimizer
# ============================================================================

class ContextWindowOptimizer:
    """
    Optimizes what fits in context window.
    Uses priority-based inclusion with fallbacks.
    """
    
    def __init__(
        self,
        token_counter: Optional[TokenCounter] = None,
        reserve_tokens: int = 500
    ):
        self.token_counter = token_counter or TokenCounter()
        self.reserve_tokens = reserve_tokens  # Reserve for response
        self.stats = {
            "optimizations_run": 0,
            "items_excluded": 0,
            "tokens_saved": 0
        }
    
    def optimize(
        self,
        context: PromptContext,
        max_tokens: int
    ) -> PromptContext:
        """
        Optimize context to fit within token limit.
        Returns a new, potentially reduced context.
        """
        available_tokens = max_tokens - self.reserve_tokens
        
        # Start with highest priority items
        optimized = PromptContext(
            system_instruction=context.system_instruction,
            state_section=context.state_section,
            current_query=context.current_query
        )
        
        current_tokens = self._count_context_tokens(optimized)
        
        # Add memories by priority
        sorted_memories = sorted(
            context.relevant_memories,
            key=lambda m: (m.priority.value, -m.last_accessed)
        )
        
        for memory in sorted_memories:
            mem_tokens = self.token_counter.count(memory.to_prompt_text())
            if current_tokens + mem_tokens <= available_tokens:
                optimized.relevant_memories.append(memory)
                current_tokens += mem_tokens
                memory.touch()
            else:
                self.stats["items_excluded"] += 1
                self.stats["tokens_saved"] += mem_tokens
        
        # Add working buffer if room
        for note in context.working_buffer:
            note_tokens = self.token_counter.count(note)
            if current_tokens + note_tokens <= available_tokens:
                optimized.working_buffer.append(note)
                current_tokens += note_tokens
        
        # Add recent conversation history
        for msg in reversed(context.conversation_history):
            msg_tokens = self.token_counter.count(msg.get("content", ""))
            if current_tokens + msg_tokens <= available_tokens:
                optimized.conversation_history.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        self.stats["optimizations_run"] += 1
        return optimized
    
    def _count_context_tokens(self, context: PromptContext) -> int:
        """Count tokens in a context."""
        total = 0
        total += self.token_counter.count(context.system_instruction)
        total += self.token_counter.count(context.state_section)
        total += self.token_counter.count(context.current_query)
        
        for mem in context.relevant_memories:
            total += self.token_counter.count(mem.to_prompt_text())
        
        for note in context.working_buffer:
            total += self.token_counter.count(note)
        
        for msg in context.conversation_history:
            total += self.token_counter.count(msg.get("content", ""))
        
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return self.stats.copy()


# ============================================================================
# Memory Retrieval Interface
# ============================================================================

class MemoryRetriever(ABC):
    """Abstract interface for memory retrieval."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Retrieve relevant memories."""
        pass
    
    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        top_k: int = 10
    ) -> RetrievalResult:
        """Hybrid semantic + keyword search."""
        pass


class SimpleMemoryRetriever(MemoryRetriever):
    """
    Simple in-memory retriever for demonstration.
    In production, this would use vector DB + keyword search.
    """
    
    def __init__(self, memories: Optional[List[Memory]] = None):
        self.memories: Dict[str, Memory] = {}
        self._lock = threading.RLock()
        
        if memories:
            for mem in memories:
                self.memories[mem.id] = mem
    
    def add_memory(self, memory: Memory) -> None:
        """Add a memory to the retrievable set."""
        with self._lock:
            self.memories[memory.id] = memory
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> RetrievalResult:
        """Simple keyword-based retrieval."""
        start_time = time.time()
        
        with self._lock:
            query_lower = query.lower()
            scored = []
            
            for mem in self.memories.values():
                # Apply filters
                if filters:
                    skip = False
                    for key, value in filters.items():
                        if mem.metadata.get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue
                
                # Simple keyword scoring
                score = 0
                content_lower = mem.content.lower()
                
                # Exact match bonus
                if query_lower in content_lower:
                    score += 10
                
                # Word overlap
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                score += overlap
                
                # Priority bonus (lower value = higher priority)
                score += (6 - mem.priority.value) * 2
                
                # Recency bonus
                age_hours = (time.time() - mem.last_accessed) / 3600
                if age_hours < 1:
                    score += 5
                elif age_hours < 24:
                    score += 2
                
                scored.append((score, mem))
            
            # Sort by score descending
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [mem for _, mem in scored[:top_k]]
            
            retrieval_time = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                memories=results,
                total_candidates=len(scored),
                retrieval_time_ms=retrieval_time
            )
    
    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        top_k: int = 10
    ) -> RetrievalResult:
        """
        Hybrid search (in this simple version, just keyword search).
        Production: Would combine vector similarity + BM25.
        """
        return self.retrieve(query, top_k)


# ============================================================================
# Response Processor
# ============================================================================

class ResponseProcessor:
    """
    Processes LLM responses to extract updates for long-term memory.
    """
    
    def __init__(self):
        self.extraction_patterns = {
            "fact": re.compile(
                r'(?:FACT|Fact|fact)[\s:]*(.+?)(?:\n|$)',
                re.MULTILINE
            ),
            "decision": re.compile(
                r'(?:DECISION|Decision|decision)[\s:]*(.+?)(?:\n|$)',
                re.MULTILINE
            ),
            "preference": re.compile(
                r'(?:PREFERENCE|Preference|preference)[\s:]*(.+?)(?:\n|$)',
                re.MULTILINE
            ),
            "insight": re.compile(
                r'(?:INSIGHT|Insight|insight)[\s:]*(.+?)(?:\n|$)',
                re.MULTILINE
            ),
        }
    
    def extract_facts(self, response: str) -> List[str]:
        """Extract stated facts from response."""
        matches = self.extraction_patterns["fact"].findall(response)
        return [m.strip() for m in matches if m.strip()]
    
    def extract_decisions(self, response: str) -> List[Dict[str, str]]:
        """Extract decisions from response."""
        matches = self.extraction_patterns["decision"].findall(response)
        decisions = []
        for match in matches:
            # Try to parse "key: value" format
            if ":" in match:
                parts = match.split(":", 1)
                decisions.append({
                    "key": parts[0].strip(),
                    "value": parts[1].strip()
                })
            else:
                decisions.append({"key": "decision", "value": match.strip()})
        return decisions
    
    def extract_preferences(self, response: str) -> List[Dict[str, str]]:
        """Extract preferences from response."""
        matches = self.extraction_patterns["preference"].findall(response)
        prefs = []
        for match in matches:
            if ":" in match:
                parts = match.split(":", 1)
                prefs.append({"key": parts[0].strip(), "value": parts[1].strip()})
            else:
                prefs.append({"key": "preference", "value": match.strip()})
        return prefs
    
    def extract_insights(self, response: str) -> List[str]:
        """Extract insights from response."""
        matches = self.extraction_patterns["insight"].findall(response)
        return [m.strip() for m in matches if m.strip()]
    
    def identify_temporary_content(self, response: str) -> List[str]:
        """
        Identify content that should NOT be persisted.
        This includes:
        - Reasoning chains ("Let me think...", "First... then...")
        - Intermediate calculations
        - Self-corrections
        """
        temporary_patterns = [
            r'(?:Let me think|Thinking|Let me consider)[^.]*\.',
            r'(?:First|Second|Third)[,\s][^.]*\.',
            r'(?:Step \d+)[:\s][^.]*\.',
            r'(?:Wait|Actually|Hmm)[,\s][^.]*\.',
        ]
        
        temporary = []
        for pattern in temporary_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            temporary.extend(matches)
        
        return temporary


# ============================================================================
# Working Memory Manager
# ============================================================================

class WorkingMemoryManager:
    """
    Manages the working memory (what goes into LLM prompt) vs
    long-term memory (persistent storage).
    
    Working memory = small, curated, temporary
    Long-term memory = large, persistent, indexed
    """
    
    def __init__(
        self,
        retriever: Optional[MemoryRetriever] = None,
        token_counter: Optional[TokenCounter] = None,
        max_context_tokens: int = 4000,
        system_instruction: str = ""
    ):
        # Components
        self.retriever = retriever or SimpleMemoryRetriever()
        self.token_counter = token_counter or TokenCounter()
        self.optimizer = ContextWindowOptimizer(self.token_counter)
        self.processor = ResponseProcessor()
        self.buffer = WorkingMemoryBuffer()
        
        # State
        self.state = StateSnapshot()
        self.system_instruction = system_instruction
        
        # Configuration
        self.max_context_tokens = max_context_tokens
        self.max_memories_per_query = 20
        
        # Tracking
        self.turn_count = 0
        self.retrieval_hits = 0
        self.retrieval_misses = 0
        self.total_retrieval_time_ms = 0.0
        self._lock = threading.RLock()
        
        # Turn history (limited)
        self.recent_turns: deque = deque(maxlen=10)
    
    def build_prompt_context(
        self,
        query: str,
        max_tokens: Optional[int] = None
    ) -> PromptContext:
        """
        Build what goes into the prompt for this turn:
        1. Get always-true state snapshot
        2. Retrieve top-N relevant memories via hybrid search
        3. Add current query
        4. Format for LLM
        """
        with self._lock:
            max_tokens = max_tokens or self.max_context_tokens
            
            # Start timing
            start_time = time.time()
            
            # Step 1: Retrieve relevant memories
            retrieval_result = self.retriever.hybrid_search(
                query=query,
                top_k=self.max_memories_per_query
            )
            
            # Update stats
            self.total_retrieval_time_ms += retrieval_result.retrieval_time_ms
            if retrieval_result.memories:
                self.retrieval_hits += 1
            else:
                self.retrieval_misses += 1
            
            # Step 2: Mark retrieved memories as accessed
            for mem in retrieval_result.memories:
                mem.touch()
            
            # Step 3: Build initial context
            context = PromptContext(
                system_instruction=self.system_instruction,
                state_section=self.state.to_prompt_section(),
                relevant_memories=retrieval_result.memories,
                current_query=query,
                working_buffer=self.buffer.get_contents(),
                conversation_history=self._build_history()
            )
            
            # Step 4: Optimize to fit token budget
            optimized = self.optimizer.optimize(context, max_tokens)
            
            logger.debug(
                f"Built prompt context: {len(optimized.relevant_memories)} memories, "
                f"retrieval took {retrieval_result.retrieval_time_ms:.1f}ms"
            )
            
            return optimized
    
    def update_from_response(
        self,
        response: str,
        turn_metadata: Optional[TurnMetadata] = None
    ) -> Dict[str, Any]:
        """
        After LLM responds, extract and update long-term memory:
        - Promote decisions/facts to state
        - Add new memories to indexes
        - Discard temporary reasoning steps
        
        Returns summary of what was updated.
        """
        with self._lock:
            updates = {
                "facts_added": 0,
                "decisions_recorded": 0,
                "preferences_set": 0,
                "insights_stored": 0,
                "temporary_items_discarded": 0
            }
            
            # Step 1: Extract and update facts
            facts = self.processor.extract_facts(response)
            for i, fact in enumerate(facts):
                key = f"fact_{self.turn_count}_{i}"
                self.state.update_fact(key, fact)
                updates["facts_added"] += 1
                
                # Also store as memory
                if isinstance(self.retriever, SimpleMemoryRetriever):
                    mem = Memory(
                        id=f"fact_mem_{key}",
                        content=fact,
                        memory_type=MemoryType.FACT,
                        priority=Priority.HIGH,
                        metadata={"source": "llm_response", "turn": self.turn_count}
                    )
                    self.retriever.add_memory(mem)
            
            # Step 2: Extract and record decisions
            decisions = self.processor.extract_decisions(response)
            for decision in decisions:
                self.state.record_decision(
                    decision["key"],
                    decision["value"],
                    rationale="Extracted from LLM response"
                )
                updates["decisions_recorded"] += 1
            
            # Step 3: Extract preferences
            preferences = self.processor.extract_preferences(response)
            for pref in preferences:
                self.state.set_preference(pref["key"], pref["value"])
                updates["preferences_set"] += 1
            
            # Step 4: Extract and store insights
            insights = self.processor.extract_insights(response)
            for i, insight in enumerate(insights):
                if isinstance(self.retriever, SimpleMemoryRetriever):
                    mem = Memory(
                        id=f"insight_{self.turn_count}_{i}",
                        content=insight,
                        memory_type=MemoryType.INSIGHT,
                        priority=Priority.MEDIUM,
                        metadata={"source": "llm_response", "turn": self.turn_count}
                    )
                    self.retriever.add_memory(mem)
                    updates["insights_stored"] += 1
            
            # Step 5: Identify temporary content (not persisted)
            temporary = self.processor.identify_temporary_content(response)
            updates["temporary_items_discarded"] = len(temporary)
            
            # Step 6: Clear working buffer (temporary reasoning)
            buffer_size = self.buffer.size()
            self.buffer.clear()
            updates["buffer_cleared"] = buffer_size
            
            # Step 7: Increment turn counter
            self.turn_count += 1
            
            # Step 8: Store turn metadata
            if turn_metadata:
                self.recent_turns.append(turn_metadata)
            
            logger.debug(
                f"Updated memory from response: {updates['facts_added']} facts, "
                f"{updates['decisions_recorded']} decisions, "
                f"{updates['temporary_items_discarded']} temporary items discarded"
            )
            
            return updates
    
    def get_working_memory_stats(self) -> WorkingMemoryStats:
        """Get stats: token count, memory count, hit rates."""
        with self._lock:
            # Build a sample context to count tokens
            sample_context = PromptContext(
                system_instruction=self.system_instruction,
                state_section=self.state.to_prompt_section(),
                relevant_memories=[],  # Empty for base count
                current_query="sample query"
            )
            
            base_tokens = self.token_counter.count(sample_context.to_plain_text())
            
            # Calculate hit rate
            total_retrievals = self.retrieval_hits + self.retrieval_misses
            hit_rate = self.retrieval_hits / total_retrievals if total_retrievals > 0 else 0.0
            
            # Average retrieval time
            avg_retrieval_time = (
                self.total_retrieval_time_ms / total_retrievals
                if total_retrievals > 0 else 0.0
            )
            
            return WorkingMemoryStats(
                total_tokens_in_context=base_tokens,
                max_tokens_allowed=self.max_context_tokens,
                token_utilization_pct=(base_tokens / self.max_context_tokens) * 100,
                memories_in_context=0,  # Would need actual context
                state_items=len(self.state.facts) + len(self.state.decisions),
                buffer_items=self.buffer.size(),
                retrieval_hits=self.retrieval_hits,
                retrieval_misses=self.retrieval_misses,
                hit_rate=hit_rate,
                avg_retrieval_time_ms=avg_retrieval_time,
                turn_count=self.turn_count
            )
    
    def add_to_buffer(self, content: str, item_type: str = "reasoning") -> None:
        """Add content to the working buffer."""
        self.buffer.add(content, item_type)
    
    def update_state_fact(self, key: str, value: Any) -> None:
        """Update a fact in the persistent state."""
        with self._lock:
            self.state.update_fact(key, value)
    
    def record_decision(self, key: str, decision: Any, rationale: str = "") -> None:
        """Record a decision in persistent state."""
        with self._lock:
            self.state.record_decision(key, decision, rationale)
    
    def add_long_term_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a memory to long-term storage."""
        with self._lock:
            mem_id = f"mem_{self.turn_count}_{int(time.time() * 1000)}"
            memory = Memory(
                id=mem_id,
                content=content,
                memory_type=memory_type,
                priority=priority,
                metadata=metadata or {}
            )
            
            if isinstance(self.retriever, SimpleMemoryRetriever):
                self.retriever.add_memory(memory)
            
            return mem_id
    
    def _build_history(self) -> List[Dict[str, str]]:
        """Build recent conversation history."""
        history = []
        for turn in list(self.recent_turns)[-3:]:  # Last 3 turns
            history.append({"role": "user", "content": turn.query})
            history.append({"role": "assistant", "content": turn.response})
        return history
    
    def get_state(self) -> StateSnapshot:
        """Get current state snapshot."""
        with self._lock:
            return StateSnapshot(
                facts=self.state.facts.copy(),
                decisions=self.state.decisions.copy(),
                preferences=self.state.preferences.copy(),
                context_variables=self.state.context_variables.copy(),
                version=self.state.version,
                last_updated=self.state.last_updated
            )
    
    def reset(self) -> None:
        """Reset all working memory (but not long-term)."""
        with self._lock:
            self.buffer.clear()
            self.turn_count = 0
            self.recent_turns.clear()
            logger.info("Working memory reset")


# ============================================================================
# Factory and Helper Functions
# ============================================================================

def create_working_memory_manager(
    memories: Optional[List[Memory]] = None,
    max_context_tokens: int = 4000,
    system_instruction: str = "You are a helpful AI assistant."
) -> WorkingMemoryManager:
    """
    Factory function to create a WorkingMemoryManager with simple retriever.
    
    Args:
        memories: Initial set of memories
        max_context_tokens: Maximum tokens for context
        system_instruction: System prompt instruction
    
    Returns:
        Configured WorkingMemoryManager
    """
    retriever = SimpleMemoryRetriever(memories)
    return WorkingMemoryManager(
        retriever=retriever,
        max_context_tokens=max_context_tokens,
        system_instruction=system_instruction
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Create manager
    manager = create_working_memory_manager(
        max_context_tokens=4000,
        system_instruction="You are a helpful coding assistant."
    )
    
    # Add some long-term memories
    manager.add_long_term_memory(
        content="Python uses indentation for code blocks",
        memory_type=MemoryType.FACT,
        priority=Priority.HIGH
    )
    manager.add_long_term_memory(
        content="List comprehensions are more efficient than for loops for simple operations",
        memory_type=MemoryType.INSIGHT,
        priority=Priority.MEDIUM
    )
    
    # Update state
    manager.update_state_fact("language", "Python")
    manager.update_state_fact("version", "3.11")
    manager.record_decision("style_guide", "PEP8", "Standard Python style")
    
    # Simulate a turn
    print("=" * 60)
    print("TURN 1: Building prompt context")
    print("=" * 60)
    
    query = "How should I format my Python code?"
    context = manager.build_prompt_context(query)
    
    print("\nPrompt Context:")
    print(context.to_plain_text())
    
    # Simulate LLM response
    response = """
    DECISION: formatter: black
    FACT: Black is the standard Python formatter
    PREFERENCE: line_length: 88
    INSIGHT: Consistent formatting improves readability
    
    Let me think about this... Black is a great choice because it has minimal configuration.
    """
    
    print("\n" + "=" * 60)
    print("Processing LLM response")
    print("=" * 60)
    
    updates = manager.update_from_response(response)
    print(f"\nUpdates made: {json.dumps(updates, indent=2)}")
    
    # Show state after update
    print("\n" + "=" * 60)
    print("State after update:")
    print("=" * 60)
    print(manager.get_state().to_prompt_section())
    
    # Show stats
    print("\n" + "=" * 60)
    print("Working Memory Stats:")
    print("=" * 60)
    stats = manager.get_working_memory_stats()
    print(f"Turn count: {stats.turn_count}")
    print(f"State items: {stats.state_items}")
    print(f"Retrieval hit rate: {stats.hit_rate:.1%}")
    print(f"Avg retrieval time: {stats.avg_retrieval_time_ms:.1f}ms")
