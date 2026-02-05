"""
Knowledge State Manager - Always-True State Object Pattern

Instead of treating conversation as a scroll, maintain a structured state
that gets updated incrementally. The state never leaves - it's continuously
maintained outside the prompt.

Key Concepts:
- Always-true state: Maintained continuously, never dropped from context
- Incremental updates: Each turn contributes changes, not full transcript
- State is source of truth: Transcript is just input to update state
- Merge, don't append: Update existing facts if they change
- Promote decisions: When reasoning completes, promote result to state
"""

import json
import sqlite3
import threading
import hashlib
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum, auto
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class FactPriority(Enum):
    """Priority levels for facts in the state."""
    CRITICAL = auto()      # Core requirements, must never be violated
    HIGH = auto()          # Important constraints and decisions
    MEDIUM = auto()        # Supporting information
    LOW = auto()           # Background context, can be evicted


class DecisionStatus(Enum):
    """Status of decisions in the state."""
    PENDING = auto()       # Decision proposed but not confirmed
    ACTIVE = auto()        # Currently active decision
    SUPERSEDED = auto()    # Replaced by newer decision
    REVOKED = auto()       # Explicitly cancelled


@dataclass
class CoreFact:
    """
    A core fact in the conversation state.
    Facts are the fundamental truths that ground the conversation.
    """
    key: str                           # Unique identifier for the fact
    value: Any                         # The fact value
    priority: FactPriority = FactPriority.MEDIUM
    source_turn: int = 0               # Which turn introduced this fact
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0            # Confidence in the fact (0.0-1.0)
    dependencies: Set[str] = field(default_factory=set)  # Keys of dependent facts
    version: int = 1                   # Version for tracking updates
    
    def __hash__(self) -> int:
        return hash(self.key)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoreFact):
            return NotImplemented
        return self.key == other.key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'priority': self.priority.name,
            'source_turn': self.source_turn,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'dependencies': list(self.dependencies),
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoreFact':
        """Create from dictionary."""
        return cls(
            key=data['key'],
            value=data['value'],
            priority=FactPriority[data['priority']],
            source_turn=data['source_turn'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            confidence=data['confidence'],
            dependencies=set(data.get('dependencies', [])),
            version=data.get('version', 1)
        )
    
    def update_value(self, new_value: Any, turn: int) -> 'CoreFact':
        """Create updated version of this fact."""
        return CoreFact(
            key=self.key,
            value=new_value,
            priority=self.priority,
            source_turn=turn,
            timestamp=datetime.utcnow(),
            confidence=self.confidence,
            dependencies=self.dependencies,
            version=self.version + 1
        )


@dataclass
class ActiveDecision:
    """
    A decision that has been made and is currently active.
    Decisions represent commitments that constrain future actions.
    """
    decision_id: str
    description: str
    rationale: str = ""                 # Why this decision was made
    status: DecisionStatus = DecisionStatus.ACTIVE
    source_turn: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    supersedes: Optional[str] = None    # ID of decision this replaces
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'decision_id': self.decision_id,
            'description': self.description,
            'rationale': self.rationale,
            'status': self.status.name,
            'source_turn': self.source_turn,
            'timestamp': self.timestamp.isoformat(),
            'supersedes': self.supersedes,
            'constraints': self.constraints,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActiveDecision':
        """Create from dictionary."""
        return cls(
            decision_id=data['decision_id'],
            description=data['description'],
            rationale=data.get('rationale', ''),
            status=DecisionStatus[data['status']],
            source_turn=data['source_turn'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            supersedes=data.get('supersedes'),
            constraints=data.get('constraints', []),
            metadata=data.get('metadata', {})
        )


@dataclass
class Constraint:
    """
    A constraint that limits the solution space.
    Constraints are inviolable requirements.
    """
    constraint_id: str
    description: str
    constraint_type: str = "hard"       # "hard" or "soft"
    source_decision: Optional[str] = None
    source_turn: int = 0
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'constraint_id': self.constraint_id,
            'description': self.description,
            'constraint_type': self.constraint_type,
            'source_decision': self.source_decision,
            'source_turn': self.source_turn,
            'active': self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Constraint':
        return cls(
            constraint_id=data['constraint_id'],
            description=data['description'],
            constraint_type=data.get('constraint_type', 'hard'),
            source_decision=data.get('source_decision'),
            source_turn=data.get('source_turn', 0),
            active=data.get('active', True)
        )


@dataclass
class CurrentContext:
    """
    The current context for the conversation.
    This is the working memory - what we're currently focusing on.
    """
    topic: str = ""                     # Current topic of discussion
    active_goals: List[str] = field(default_factory=list)
    pending_questions: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    turn_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'active_goals': self.active_goals,
            'pending_questions': self.pending_questions,
            'working_memory': self.working_memory,
            'turn_number': self.turn_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurrentContext':
        return cls(
            topic=data.get('topic', ''),
            active_goals=data.get('active_goals', []),
            pending_questions=data.get('pending_questions', []),
            working_memory=data.get('working_memory', {}),
            turn_number=data.get('turn_number', 0)
        )


@dataclass
class TurnResult:
    """
    Result from a single conversation turn.
    This is the input to update the state.
    """
    turn_number: int
    input_text: str = ""                # User input (for reference)
    output_text: str = ""               # System output (for reference)
    extracted_facts: List[CoreFact] = field(default_factory=list)
    proposed_decisions: List[ActiveDecision] = field(default_factory=list)
    new_constraints: List[Constraint] = field(default_factory=list)
    resolved_items: List[str] = field(default_factory=list)  # IDs of resolved facts/decisions
    context_update: Optional[CurrentContext] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateSnapshot:
    """
    Curated snapshot for LLM prompt.
    Small, structured, clean - only what's needed for current turn.
    """
    turn_number: int
    core_facts: List[CoreFact] = field(default_factory=list)
    active_decisions: List[ActiveDecision] = field(default_factory=list)
    active_constraints: List[Constraint] = field(default_factory=list)
    current_context: Optional[CurrentContext] = None
    relevant_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(self, max_facts: int = 20, max_decisions: int = 10) -> str:
        """
        Convert snapshot to a clean prompt context string.
        Prioritizes by importance and recency.
        """
        lines = ["=== CONVERSATION STATE ===", ""]
        
        # Core facts (sorted by priority, then recency)
        priority_order = {
            FactPriority.CRITICAL: 0,
            FactPriority.HIGH: 1,
            FactPriority.MEDIUM: 2,
            FactPriority.LOW: 3
        }
        sorted_facts = sorted(
            self.core_facts,
            key=lambda f: (priority_order.get(f.priority, 2), -f.source_turn)
        )[:max_facts]
        
        if sorted_facts:
            lines.append("CORE FACTS:")
            for fact in sorted_facts:
                priority_marker = ""
                if fact.priority == FactPriority.CRITICAL:
                    priority_marker = " [CRITICAL]"
                elif fact.priority == FactPriority.HIGH:
                    priority_marker = " [HIGH]"
                lines.append(f"  * {fact.key}: {fact.value}{priority_marker}")
            lines.append("")
        
        # Active decisions
        active = [d for d in self.active_decisions if d.status == DecisionStatus.ACTIVE]
        active = sorted(active, key=lambda d: d.source_turn, reverse=True)[:max_decisions]
        
        if active:
            lines.append("ACTIVE DECISIONS:")
            for decision in active:
                lines.append(f"  * {decision.description}")
                if decision.rationale:
                    lines.append(f"    (Rationale: {decision.rationale})")
            lines.append("")
        
        # Active constraints
        hard_constraints = [c for c in self.active_constraints if c.active and c.constraint_type == "hard"]
        if hard_constraints:
            lines.append("CONSTRAINTS:")
            for constraint in hard_constraints:
                lines.append(f"  * {constraint.description}")
            lines.append("")
        
        # Current context
        if self.current_context:
            lines.append("CURRENT CONTEXT:")
            if self.current_context.topic:
                lines.append(f"  Topic: {self.current_context.topic}")
            if self.current_context.active_goals:
                lines.append(f"  Goals: {', '.join(self.current_context.active_goals)}")
            if self.current_context.pending_questions:
                lines.append(f"  Pending: {', '.join(self.current_context.pending_questions)}")
            lines.append("")
        
        lines.append("=== END STATE ===")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'turn_number': self.turn_number,
            'core_facts': [f.to_dict() for f in self.core_facts],
            'active_decisions': [d.to_dict() for d in self.active_decisions],
            'active_constraints': [c.to_dict() for c in self.active_constraints],
            'current_context': self.current_context.to_dict() if self.current_context else None,
            'relevant_details': self.relevant_details
        }


@dataclass
class StateUpdate:
    """
    Incremental update from a single turn.
    Contains only what changed - not the full state.
    """
    turn_number: int
    new_facts: List[CoreFact] = field(default_factory=list)
    modified_facts: List[Tuple[CoreFact, CoreFact]] = field(default_factory=list)  # (old, new)
    removed_fact_keys: Set[str] = field(default_factory=set)
    new_decisions: List[ActiveDecision] = field(default_factory=list)
    modified_decisions: List[ActiveDecision] = field(default_factory=list)
    resolved_decision_ids: Set[str] = field(default_factory=set)
    new_constraints: List[Constraint] = field(default_factory=list)
    removed_constraint_ids: Set[str] = field(default_factory=set)
    context_changes: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get a summary of the update."""
        parts = []
        if self.new_facts:
            parts.append(f"+{len(self.new_facts)} facts")
        if self.modified_facts:
            parts.append(f"~{len(self.modified_facts)} facts")
        if self.removed_fact_keys:
            parts.append(f"-{len(self.removed_fact_keys)} facts")
        if self.new_decisions:
            parts.append(f"+{len(self.new_decisions)} decisions")
        if self.resolved_decision_ids:
            parts.append(f"[OK]{len(self.resolved_decision_ids)} resolved")
        if self.new_constraints:
            parts.append(f"+{len(self.new_constraints)} constraints")
        return ", ".join(parts) if parts else "no changes"


@dataclass
class StateVersion:
    """
    A version of the state for history tracking.
    """
    version_id: str
    turn_number: int
    timestamp: datetime
    state_hash: str                      # Hash of the serialized state
    update_summary: str
    parent_version: Optional[str] = None  # Previous version


class ConversationState:
    """
    The "always true" state object that persists across turns.
    Never truncated, continuously updated.
    
    This is the source of truth for the conversation. The transcript
    is merely input to update this state.
    """
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id or self._generate_id()
        self.facts: Dict[str, CoreFact] = {}
        self.decisions: Dict[str, ActiveDecision] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.context: CurrentContext = CurrentContext()
        self.version_history: List[StateVersion] = []
        self.created_at: datetime = datetime.utcnow()
        self.last_updated: datetime = datetime.utcnow()
        self._state_hash: Optional[str] = None
    
    def _generate_id(self) -> str:
        """Generate unique conversation ID."""
        return f"conv_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(self) % 10000:04d}"
    
    def _compute_hash(self) -> str:
        """Compute hash of current state for change detection."""
        state_data = {
            'facts': {k: v.to_dict() for k, v in sorted(self.facts.items())},
            'decisions': {k: v.to_dict() for k, v in sorted(self.decisions.items())},
            'constraints': {k: v.to_dict() for k, v in sorted(self.constraints.items())},
            'context': self.context.to_dict()
        }
        return hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def has_changed(self) -> bool:
        """Check if state has changed since last hash computation."""
        current_hash = self._compute_hash()
        return current_hash != self._state_hash
    
    def update_hash(self) -> None:
        """Update the stored hash to current state."""
        self._state_hash = self._compute_hash()
    
    def add_fact(self, fact: CoreFact) -> None:
        """Add or update a fact."""
        self.facts[fact.key] = fact
        self.last_updated = datetime.utcnow()
    
    def get_fact(self, key: str) -> Optional[CoreFact]:
        """Get a fact by key."""
        return self.facts.get(key)
    
    def remove_fact(self, key: str) -> Optional[CoreFact]:
        """Remove a fact by key."""
        fact = self.facts.pop(key, None)
        if fact:
            self.last_updated = datetime.utcnow()
        return fact
    
    def add_decision(self, decision: ActiveDecision) -> None:
        """Add a decision."""
        # If this supersedes another decision, mark the old one
        if decision.supersedes and decision.supersedes in self.decisions:
            old_decision = self.decisions[decision.supersedes]
            old_decision.status = DecisionStatus.SUPERSEDED
        
        self.decisions[decision.decision_id] = decision
        self.last_updated = datetime.utcnow()
    
    def get_decision(self, decision_id: str) -> Optional[ActiveDecision]:
        """Get a decision by ID."""
        return self.decisions.get(decision_id)
    
    def resolve_decision(self, decision_id: str) -> bool:
        """Mark a decision as resolved."""
        if decision_id in self.decisions:
            self.decisions[decision_id].status = DecisionStatus.REVOKED
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint."""
        self.constraints[constraint.constraint_id] = constraint
        self.last_updated = datetime.utcnow()
    
    def remove_constraint(self, constraint_id: str) -> bool:
        """Remove a constraint."""
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
            self.last_updated = datetime.utcnow()
            return True
        return False
    
    def get_active_facts(self) -> List[CoreFact]:
        """Get all active facts."""
        return list(self.facts.values())
    
    def get_active_decisions(self) -> List[ActiveDecision]:
        """Get all active (non-revoked, non-superseded) decisions."""
        return [
            d for d in self.decisions.values()
            if d.status == DecisionStatus.ACTIVE
        ]
    
    def get_active_constraints(self) -> List[Constraint]:
        """Get all active constraints."""
        return [
            c for c in self.constraints.values()
            if c.active
        ]
    
    def update_context(self, context: CurrentContext) -> None:
        """Update the current context."""
        self.context = context
        self.last_updated = datetime.utcnow()
    
    def create_snapshot(self, turn_number: Optional[int] = None) -> StateSnapshot:
        """Create a snapshot of the current state."""
        return StateSnapshot(
            turn_number=turn_number or self.context.turn_number,
            core_facts=self.get_active_facts(),
            active_decisions=self.get_active_decisions(),
            active_constraints=self.get_active_constraints(),
            current_context=self.context,
            relevant_details={
                'total_facts': len(self.facts),
                'total_decisions': len(self.decisions),
                'total_constraints': len(self.constraints),
                'conversation_id': self.conversation_id
            }
        )
    
    def record_version(self, update_summary: str = "") -> StateVersion:
        """Record current state as a version."""
        version = StateVersion(
            version_id=f"v_{len(self.version_history)}_{datetime.utcnow().strftime('%H%M%S')}",
            turn_number=self.context.turn_number,
            timestamp=datetime.utcnow(),
            state_hash=self._compute_hash(),
            update_summary=update_summary,
            parent_version=self.version_history[-1].version_id if self.version_history else None
        )
        self.version_history.append(version)
        self.update_hash()
        return version
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire state to dictionary."""
        return {
            'conversation_id': self.conversation_id,
            'facts': {k: v.to_dict() for k, v in self.facts.items()},
            'decisions': {k: v.to_dict() for k, v in self.decisions.items()},
            'constraints': {k: v.to_dict() for k, v in self.constraints.items()},
            'context': self.context.to_dict(),
            'version_history': [
                {
                    'version_id': v.version_id,
                    'turn_number': v.turn_number,
                    'timestamp': v.timestamp.isoformat(),
                    'state_hash': v.state_hash,
                    'update_summary': v.update_summary,
                    'parent_version': v.parent_version
                }
                for v in self.version_history
            ],
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create state from dictionary."""
        state = cls(conversation_id=data.get('conversation_id'))
        
        state.facts = {
            k: CoreFact.from_dict(v)
            for k, v in data.get('facts', {}).items()
        }
        state.decisions = {
            k: ActiveDecision.from_dict(v)
            for k, v in data.get('decisions', {}).items()
        }
        state.constraints = {
            k: Constraint.from_dict(v)
            for k, v in data.get('constraints', {}).items()
        }
        state.context = CurrentContext.from_dict(data.get('context', {}))
        state.created_at = datetime.fromisoformat(data.get('created_at', datetime.utcnow().isoformat()))
        state.last_updated = datetime.fromisoformat(data.get('last_updated', datetime.utcnow().isoformat()))
        
        # Restore version history
        for v_data in data.get('version_history', []):
            version = StateVersion(
                version_id=v_data['version_id'],
                turn_number=v_data['turn_number'],
                timestamp=datetime.fromisoformat(v_data['timestamp']),
                state_hash=v_data['state_hash'],
                update_summary=v_data.get('update_summary', ''),
                parent_version=v_data.get('parent_version')
            )
            state.version_history.append(version)
        
        state.update_hash()
        return state
    
    def copy(self) -> 'ConversationState':
        """Create a deep copy of the state."""
        return ConversationState.from_dict(self.to_dict())


class StateManager:
    """
    Maintains the conversation state outside the prompt.
    
    Each turn contributes changes (new facts, decisions, constraints)
    which get merged into existing memory rather than appended.
    
    Key principle: State is the source of truth. The transcript is just
    input to update the state.
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        auto_persist: bool = True,
        max_versions: int = 100
    ):
        """
        Initialize the state manager.
        
        Args:
            db_path: Path to SQLite database for persistence (None for JSON only)
            auto_persist: Whether to auto-save state changes
            max_versions: Maximum number of versions to keep per conversation
        """
        self._states: Dict[str, ConversationState] = {}
        self._db_path = db_path
        self._auto_persist = auto_persist
        self._max_versions = max_versions
        self._lock = threading.RLock()
        
        if db_path:
            self._init_database()
    
    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_states (
                    conversation_id TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS state_versions (
                    version_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    update_summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversation_states(conversation_id)
                )
            """)
            conn.commit()
    
    @contextmanager
    def _transaction(self):
        """Thread-safe transaction context."""
        with self._lock:
            yield
    
    def create_conversation(self, conversation_id: Optional[str] = None) -> ConversationState:
        """Create a new conversation state."""
        with self._transaction():
            state = ConversationState(conversation_id)
            self._states[state.conversation_id] = state
            
            if self._auto_persist and self._db_path:
                self._persist_state(state)
            
            logger.info(f"Created conversation {state.conversation_id}")
            return state
    
    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Get state for a conversation."""
        with self._transaction():
            if conversation_id in self._states:
                return self._states[conversation_id]
            
            # Try to load from database
            if self._db_path:
                return self._load_state(conversation_id)
            
            return None
    
    def update_from_turn(self, conversation_id: str, turn_result: TurnResult) -> StateUpdate:
        """
        Update state from a conversation turn.
        
        Only contributes CHANGES, not raw text. Extracts structured
        updates from the turn result and merges them into state.
        
        Args:
            conversation_id: ID of the conversation
            turn_result: Result from the turn processing
            
        Returns:
            StateUpdate describing what changed
        """
        with self._transaction():
            state = self.get_state(conversation_id)
            if not state:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            update = StateUpdate(turn_number=turn_result.turn_number)
            
            # Process extracted facts
            for fact in turn_result.extracted_facts:
                if fact.key in state.facts:
                    old_fact = state.facts[fact.key]
                    if old_fact.value != fact.value:
                        # Fact modified - create updated version
                        new_fact = fact.update_value(fact.value, turn_result.turn_number)
                        update.modified_facts.append((old_fact, new_fact))
                        state.add_fact(new_fact)
                else:
                    # New fact
                    fact.source_turn = turn_result.turn_number
                    update.new_facts.append(fact)
                    state.add_fact(fact)
            
            # Process proposed decisions
            for decision in turn_result.proposed_decisions:
                decision.source_turn = turn_result.turn_number
                if decision.decision_id in state.decisions:
                    update.modified_decisions.append(decision)
                else:
                    update.new_decisions.append(decision)
                state.add_decision(decision)
            
            # Process new constraints
            for constraint in turn_result.new_constraints:
                constraint.source_turn = turn_result.turn_number
                update.new_constraints.append(constraint)
                state.add_constraint(constraint)
            
            # Process resolved items
            for item_id in turn_result.resolved_items:
                if item_id in state.decisions:
                    state.resolve_decision(item_id)
                    update.resolved_decision_ids.add(item_id)
                # Could also resolve facts, constraints, etc.
            
            # Update context if provided
            if turn_result.context_update:
                state.update_context(turn_result.context_update)
                update.context_changes = turn_result.context_update.to_dict()
            else:
                # Update turn number in context
                state.context.turn_number = turn_result.turn_number
            
            # Record version
            state.record_version(update.summary())
            
            # Prune old versions if needed
            if len(state.version_history) > self._max_versions:
                state.version_history = state.version_history[-self._max_versions:]
            
            # Persist if enabled
            if self._auto_persist and self._db_path:
                self._persist_state(state)
                self._persist_version(conversation_id, state.version_history[-1], state)
            
            logger.info(f"Updated conversation {conversation_id}: {update.summary()}")
            return update
    
    def merge_incremental_update(self, conversation_id: str, update: StateUpdate) -> None:
        """
        Merge incremental changes into existing state.
        
        Not append - MERGE. Updates existing facts if they changed.
        Handles conflicts intelligently.
        
        Args:
            conversation_id: ID of the conversation
            update: Incremental update to merge
        """
        with self._transaction():
            state = self.get_state(conversation_id)
            if not state:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Merge new facts
            for fact in update.new_facts:
                if fact.key not in state.facts:
                    state.add_fact(fact)
                elif state.facts[fact.key].version < fact.version:
                    # Incoming fact is newer version
                    state.add_fact(fact)
            
            # Merge modified facts
            for old_fact, new_fact in update.modified_facts:
                current = state.facts.get(new_fact.key)
                if current and current.version == old_fact.version:
                    # No conflict - apply update
                    state.add_fact(new_fact)
                elif current and current.version < new_fact.version:
                    # Incoming is newer - apply
                    state.add_fact(new_fact)
                # else: conflict - keep current (could implement merge strategy)
            
            # Remove facts
            for key in update.removed_fact_keys:
                state.remove_fact(key)
            
            # Merge decisions
            for decision in update.new_decisions:
                state.add_decision(decision)
            
            for decision in update.modified_decisions:
                if decision.decision_id in state.decisions:
                    state.decisions[decision.decision_id] = decision
            
            # Resolve decisions
            for decision_id in update.resolved_decision_ids:
                state.resolve_decision(decision_id)
            
            # Merge constraints
            for constraint in update.new_constraints:
                state.add_constraint(constraint)
            
            for constraint_id in update.removed_constraint_ids:
                state.remove_constraint(constraint_id)
            
            # Update context
            if update.context_changes:
                state.context = CurrentContext.from_dict(update.context_changes)
            
            state.record_version("merged external update")
            
            if self._auto_persist and self._db_path:
                self._persist_state(state)
    
    def get_state_snapshot(
        self,
        conversation_id: str,
        include_facts: bool = True,
        include_decisions: bool = True,
        include_constraints: bool = True,
        fact_filter: Optional[callable] = None,
        decision_filter: Optional[callable] = None
    ) -> Optional[StateSnapshot]:
        """
        Get curated snapshot for current turn.
        
        Returns a clean snapshot containing:
        - Current core facts
        - Active decisions
        - Relevant supporting detail
        - What has been solved/decided
        
        Args:
            conversation_id: ID of the conversation
            include_facts: Whether to include facts
            include_decisions: Whether to include decisions
            include_constraints: Whether to include constraints
            fact_filter: Optional filter function for facts
            decision_filter: Optional filter function for decisions
            
        Returns:
            StateSnapshot or None if conversation not found
        """
        with self._transaction():
            state = self.get_state(conversation_id)
            if not state:
                return None
            
            facts = []
            decisions = []
            constraints = []
            
            if include_facts:
                facts = state.get_active_facts()
                if fact_filter:
                    facts = [f for f in facts if fact_filter(f)]
            
            if include_decisions:
                decisions = state.get_active_decisions()
                if decision_filter:
                    decisions = [d for d in decisions if decision_filter(d)]
            
            if include_constraints:
                constraints = state.get_active_constraints()
            
            return StateSnapshot(
                turn_number=state.context.turn_number,
                core_facts=facts,
                active_decisions=decisions,
                active_constraints=constraints,
                current_context=state.context,
                relevant_details={
                    'total_facts': len(state.facts),
                    'total_decisions': len(state.decisions),
                    'total_constraints': len(state.constraints)
                }
            )
    
    def compute_state_diff(
        self,
        conversation_id: str,
        from_turn: int,
        to_turn: int
    ) -> Optional[StateUpdate]:
        """
        Compute what changed between two turns.
        
        Args:
            conversation_id: ID of the conversation
            from_turn: Starting turn number
            to_turn: Ending turn number
            
        Returns:
            StateUpdate describing the diff, or None if conversation not found
        """
        with self._transaction():
            state = self.get_state(conversation_id)
            if not state:
                return None
            
            # Find versions at the specified turns
            from_version = None
            to_version = None
            
            for version in state.version_history:
                if version.turn_number == from_turn:
                    from_version = version
                if version.turn_number == to_turn:
                    to_version = version
            
            if not from_version or not to_version:
                logger.warning(f"Could not find versions for turns {from_turn} to {to_turn}")
                return None
            
            # Compare states (simplified - could be more sophisticated)
            # For now, return facts/decisions from the turns in range
            diff = StateUpdate(turn_number=to_turn)
            
            for fact in state.facts.values():
                if from_turn < fact.source_turn <= to_turn:
                    if fact.version == 1:
                        diff.new_facts.append(fact)
                    else:
                        diff.modified_facts.append((fact, fact))  # Simplified
            
            for decision in state.decisions.values():
                if from_turn < decision.source_turn <= to_turn:
                    diff.new_decisions.append(decision)
            
            return diff
    
    def _persist_state(self, state: ConversationState) -> None:
        """Persist state to database."""
        if not self._db_path:
            return
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO conversation_states 
                    (conversation_id, state_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (state.conversation_id, json.dumps(state.to_dict()))
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to persist state: {e}")
    
    def _persist_version(
        self,
        conversation_id: str,
        version: StateVersion,
        state: ConversationState
    ) -> None:
        """Persist a state version to database."""
        if not self._db_path:
            return
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO state_versions
                    (version_id, conversation_id, turn_number, state_json, update_summary)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        version.version_id,
                        conversation_id,
                        version.turn_number,
                        json.dumps(state.to_dict()),
                        version.update_summary
                    )
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to persist version: {e}")
    
    def _load_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Load state from database."""
        if not self._db_path:
            return None
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT state_json FROM conversation_states WHERE conversation_id = ?",
                    (conversation_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    state_data = json.loads(row[0])
                    state = ConversationState.from_dict(state_data)
                    self._states[conversation_id] = state
                    return state
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Failed to load state: {e}")
        
        return None
    
    def list_conversations(self) -> List[str]:
        """List all conversation IDs."""
        with self._transaction():
            ids = list(self._states.keys())
            
            if self._db_path:
                try:
                    with sqlite3.connect(self._db_path) as conn:
                        cursor = conn.execute("SELECT conversation_id FROM conversation_states")
                        db_ids = [row[0] for row in cursor.fetchall()]
                        # Merge without duplicates
                        ids = list(set(ids + db_ids))
                except sqlite3.Error as e:
                    logger.error(f"Failed to list conversations: {e}")
            
            return ids
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its history."""
        with self._transaction():
            if conversation_id in self._states:
                del self._states[conversation_id]
            
            if self._db_path:
                try:
                    with sqlite3.connect(self._db_path) as conn:
                        conn.execute(
                            "DELETE FROM state_versions WHERE conversation_id = ?",
                            (conversation_id,)
                        )
                        conn.execute(
                            "DELETE FROM conversation_states WHERE conversation_id = ?",
                            (conversation_id,)
                        )
                        conn.commit()
                        return True
                except sqlite3.Error as e:
                    logger.error(f"Failed to delete conversation: {e}")
                    return False
            
            return True
    
    def export_to_json(self, conversation_id: str, file_path: str) -> bool:
        """Export conversation state to JSON file."""
        with self._transaction():
            state = self.get_state(conversation_id)
            if not state:
                return False
            
            try:
                with open(file_path, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
                return True
            except (IOError, json.JSONEncodeError) as e:
                logger.error(f"Failed to export state: {e}")
                return False
    
    def import_from_json(self, file_path: str) -> Optional[ConversationState]:
        """Import conversation state from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            state = ConversationState.from_dict(data)
            
            with self._transaction():
                self._states[state.conversation_id] = state
                
                if self._auto_persist and self._db_path:
                    self._persist_state(state)
                    for version in state.version_history:
                        self._persist_version(state.conversation_id, version, state)
                
                return state
        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to import state: {e}")
            return None
    
    def get_state_at_version(
        self,
        conversation_id: str,
        version_id: str
    ) -> Optional[ConversationState]:
        """Get state at a specific version."""
        if not self._db_path:
            return None
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT state_json FROM state_versions 
                    WHERE conversation_id = ? AND version_id = ?
                    """,
                    (conversation_id, version_id)
                )
                row = cursor.fetchone()
                
                if row:
                    return ConversationState.from_dict(json.loads(row[0]))
        except (sqlite3.Error, json.JSONDecodeError) as e:
            logger.error(f"Failed to load version: {e}")
        
        return None
    
    def get_version_history(self, conversation_id: str) -> List[StateVersion]:
        """Get version history for a conversation."""
        with self._transaction():
            state = self.get_state(conversation_id)
            if state:
                return state.version_history.copy()
            return []


class StateConflictResolver:
    """
    Handles conflicts when merging state updates.
    
    Provides strategies for resolving conflicts between concurrent
    updates or between local and remote state.
    """
    
    @staticmethod
    def last_write_wins(
        local: CoreFact,
        remote: CoreFact
    ) -> CoreFact:
        """Use the most recently updated fact."""
        if remote.timestamp > local.timestamp:
            return remote
        return local
    
    @staticmethod
    def highest_version_wins(
        local: CoreFact,
        remote: CoreFact
    ) -> CoreFact:
        """Use the fact with the highest version number."""
        if remote.version > local.version:
            return remote
        return local
    
    @staticmethod
    def highest_priority_wins(
        local: CoreFact,
        remote: CoreFact
    ) -> CoreFact:
        """Use the fact with the highest priority."""
        priority_rank = {
            FactPriority.CRITICAL: 4,
            FactPriority.HIGH: 3,
            FactPriority.MEDIUM: 2,
            FactPriority.LOW: 1
        }
        
        local_rank = priority_rank.get(local.priority, 0)
        remote_rank = priority_rank.get(remote.priority, 0)
        
        if remote_rank > local_rank:
            return remote
        elif local_rank > remote_rank:
            return local
        else:
            # Same priority - use last write
            return StateConflictResolver.last_write_wins(local, remote)
    
    @staticmethod
    def merge_values(
        local: CoreFact,
        remote: CoreFact
    ) -> CoreFact:
        """Attempt to merge fact values if they're mergeable types."""
        if isinstance(local.value, dict) and isinstance(remote.value, dict):
            merged = {**local.value, **remote.value}
            return local.update_value(merged, max(local.source_turn, remote.source_turn))
        elif isinstance(local.value, list) and isinstance(remote.value, list):
            merged = local.value + [x for x in remote.value if x not in local.value]
            return local.update_value(merged, max(local.source_turn, remote.source_turn))
        else:
            # Can't merge - use last write
            return StateConflictResolver.last_write_wins(local, remote)


# Convenience functions for common operations

def create_manager(db_path: Optional[str] = None) -> StateManager:
    """Create a new state manager with optional persistence."""
    return StateManager(db_path=db_path)


def quick_fact(key: str, value: Any, priority: FactPriority = FactPriority.MEDIUM) -> CoreFact:
    """Create a CoreFact with minimal boilerplate."""
    return CoreFact(key=key, value=value, priority=priority)


def quick_decision(
    description: str,
    rationale: str = "",
    supersedes: Optional[str] = None
) -> ActiveDecision:
    """Create an ActiveDecision with minimal boilerplate."""
    decision_id = f"dec_{datetime.utcnow().strftime('%H%M%S')}_{hash(description) % 10000:04d}"
    return ActiveDecision(
        decision_id=decision_id,
        description=description,
        rationale=rationale,
        supersedes=supersedes
    )


def quick_constraint(description: str, constraint_type: str = "hard") -> Constraint:
    """Create a Constraint with minimal boilerplate."""
    constraint_id = f"con_{datetime.utcnow().strftime('%H%M%S')}_{hash(description) % 10000:04d}"
    return Constraint(
        constraint_id=constraint_id,
        description=description,
        constraint_type=constraint_type
    )


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a state manager with in-memory storage (no db_path)
    manager = StateManager(auto_persist=False)
    
    # Create a conversation
    state = manager.create_conversation()
    print(f"Created conversation: {state.conversation_id}")
    
    # Simulate conversation turns
    for turn in range(1, 4):
        print(f"\n--- Turn {turn} ---")
        
        # Create turn result with extracted information
        turn_result = TurnResult(
            turn_number=turn,
            input_text=f"User input for turn {turn}",
            output_text=f"System response for turn {turn}",
            extracted_facts=[
                quick_fact(f"fact_{turn}", f"value_{turn}", FactPriority.HIGH),
                quick_fact("persistent_fact", f"updated_value_{turn}", FactPriority.CRITICAL)
            ],
            proposed_decisions=[
                quick_decision(f"Decision from turn {turn}", f"Rationale for turn {turn}")
            ],
            new_constraints=[
                quick_constraint(f"Constraint from turn {turn}")
            ],
            context_update=CurrentContext(
                topic=f"Topic {turn}",
                active_goals=[f"Goal {turn}"],
                turn_number=turn
            )
        )
        
        # Update state from turn
        update = manager.update_from_turn(state.conversation_id, turn_result)
        print(f"Update: {update.summary()}")
        
        # Get snapshot for next turn
        snapshot = manager.get_state_snapshot(state.conversation_id)
        if snapshot:
            print(f"\nSnapshot for prompt:\n{snapshot.to_prompt_context()}")
    
    # Show final state stats
    final_state = manager.get_state(state.conversation_id)
    print(f"\n=== Final State ===")
    print(f"Facts: {len(final_state.facts)}")
    print(f"Decisions: {len(final_state.decisions)}")
    print(f"Constraints: {len(final_state.constraints)}")
    print(f"Versions: {len(final_state.version_history)}")
    
    # Demonstrate state diff
    diff = manager.compute_state_diff(state.conversation_id, 1, 3)
    if diff:
        print(f"\nChanges from turn 1 to 3: {diff.summary()}")
    
    print("\n[OK] Knowledge State Manager demo complete!")
