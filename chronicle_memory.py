"""
Chronicle Memory - Temporal Episodic Memory System

Implements an event-sourced approach to agent memory. Unlike Knowledge Graphs
(which store facts), the Chronicle stores experiences and narratives:
"First we tried A, then B failed, so we did C"

Key Features:
- Event-sourced memory architecture
- Temporal sequencing of agent actions
- Experience replay for learning
- Loop detection and prevention
- Narrative reconstruction
- Integration with Graphiti for hybrid memory
"""


import os
import json
import time
import uuid
import asyncio
import hashlib
import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Chronicle Memory
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


# **ACTUAL INTEGRATION HELPER METHODS**: Chronicle Memory
def _trigger_chronicle_alerts(operation, success, event_id=None, error=None, metadata=None):
    """Trigger alerts for chronicle memory operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        severity = AlertSeverity.HIGH if operation in ["record_event", "create_narrative"] else AlertSeverity.MEDIUM
        alert_mgr.trigger_alert(
            title=f"Chronicle Memory {operation} Failed",
            message=f"Chronicle memory operation '{operation}' failed: {error}",
            severity=severity,
            source="ChronicleMemory",
            metadata=metadata or {"event_id": event_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger chronicle alert: {e}")


def _extract_chronicle_knowledge(operation, event_id, event_type, result):
    """Extract knowledge from chronicle memory operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"chronicle_{operation}_{event_id}",
            artifact_type="chronicle_memory_operation",
            source_component="ChronicleMemory",
            content={
                "operation": operation,
                "event_id": event_id,
                "event_type": event_type,
                "outcome": getattr(result, 'outcome', None) if result else None,
                "success": result is not None,
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract chronicle knowledge: {e}")


def _track_chronicle_performance(operation, success, duration_seconds, event_type, events_count=0):
    """Track performance of chronicle memory operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name=f"chronicle_{event_type}",
            component_name="ChronicleMemory",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "event_type": event_type,
                "events_count": events_count
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track chronicle performance: {e}")


class EventType(Enum):
    """Types of events in the chronicle"""
    # Action events
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    
    # Decision events
    DECISION_MADE = "decision_made"
    STRATEGY_SELECTED = "strategy_selected"
    MODEL_ROUTED = "model_routed"
    
    # Agent events
    AGENT_SPAWNED = "agent_spawned"
    AGENT_COMPLETED = "agent_completed"
    AGENT_ERROR = "agent_error"
    
    # Workflow events
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_STAGE_CHANGED = "workflow_stage_changed"
    WORKFLOW_COMPLETED = "workflow_completed"
    
    # Learning events
    ATTEMPT_MADE = "attempt_made"
    RETRY_TRIGGERED = "retry_triggered"
    STRATEGY_PIVOTED = "strategy_pivoted"
    LESSON_LEARNED = "lesson_learned"
    
    # System events
    FIX_APPLIED = "fix_applied"
    TEST_EXECUTED = "test_executed"
    VERIFICATION_DONE = "verification_done"
    SANDBOX_CREATED = "sandbox_created"


class Outcome(Enum):
    """Outcome of an event"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ChronicleEvent:
    """A single event in the chronicle"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    agent_id: str
    session_id: str
    
    # Event content
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    outcome: Outcome = Outcome.UNKNOWN
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    parent_event_id: Optional[str] = None
    related_events: List[str] = field(default_factory=list)
    
    # Narrative
    narrative: str = ""  # Human-readable description
    lesson: Optional[str] = None  # What was learned
    
    # Metrics
    duration_ms: Optional[float] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "action": self.action,
            "parameters": self.parameters,
            "outcome": self.outcome.value,
            "context": self.context,
            "parent_event_id": self.parent_event_id,
            "related_events": self.related_events,
            "narrative": self.narrative,
            "lesson": self.lesson,
            "duration_ms": self.duration_ms,
            "retry_count": self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChronicleEvent":
        """Create from dictionary"""
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            action=data["action"],
            parameters=data.get("parameters", {}),
            outcome=Outcome(data.get("outcome", "unknown")),
            context=data.get("context", {}),
            parent_event_id=data.get("parent_event_id"),
            related_events=data.get("related_events", []),
            narrative=data.get("narrative", ""),
            lesson=data.get("lesson"),
            duration_ms=data.get("duration_ms"),
            retry_count=data.get("retry_count", 0)
        )


@dataclass
class ExperiencePattern:
    """A pattern learned from multiple events"""
    pattern_id: str
    description: str
    event_sequence: List[EventType]
    typical_outcome: Outcome
    success_rate: float
    first_seen: datetime
    last_seen: datetime
    occurrence_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "event_sequence": [e.value for e in self.event_sequence],
            "typical_outcome": self.typical_outcome.value,
            "success_rate": self.success_rate,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "occurrence_count": self.occurrence_count
        }


@dataclass
class Narrative:
    """A reconstructed narrative from a sequence of events"""
    narrative_id: str
    title: str
    events: List[ChronicleEvent]
    summary: str
    lessons_learned: List[str] = field(default_factory=list)
    outcome: Outcome = Outcome.UNKNOWN
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate narrative duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_story(self) -> str:
        """Convert to human-readable story"""
        story = f"# {self.title}\n\n"
        story += f"{self.summary}\n\n"
        
        story += "## Timeline:\n"
        for event in self.events:
            time_str = event.timestamp.strftime("%H:%M:%S")
            icon = "[OK]" if event.outcome == Outcome.SUCCESS else "[FAIL]"
            story += f"- [{time_str}] {icon} {event.narrative}\n"
        
        if self.lessons_learned:
            story += "\n## Lessons Learned:\n"
            for lesson in self.lessons_learned:
                story += f"- {lesson}\n"
        
        return story


class ChronicleStore:
    """Storage backend for chronicle events"""
    
    def __init__(self, storage_path: str = "./chronicle_store"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._events: Dict[str, ChronicleEvent] = {}
        self._session_events: Dict[str, List[str]] = defaultdict(list)
        self._agent_events: Dict[str, List[str]] = defaultdict(list)
    
    async def append(self, event: ChronicleEvent):
        """Append an event to the store"""
        self._events[event.event_id] = event
        self._session_events[event.session_id].append(event.event_id)
        self._agent_events[event.agent_id].append(event.event_id)
        
        # Persist to disk
        await self._persist_event(event)
    
    async def _persist_event(self, event: ChronicleEvent):
        """Persist event to disk"""
        # Organize by date for easier querying
        date_path = self.storage_path / event.timestamp.strftime("%Y/%m/%d")
        date_path.mkdir(parents=True, exist_ok=True)
        
        file_path = date_path / f"{event.event_id}.json"
        
        with open(file_path, "w") as f:
            json.dump(event.to_dict(), f, indent=2)
    
    async def get_event(self, event_id: str) -> Optional[ChronicleEvent]:
        """Get event by ID"""
        if event_id in self._events:
            return self._events[event_id]
        
        # Try to load from disk
        # Search in date directories
        for date_dir in self.storage_path.rglob("*.json"):
            if date_dir.stem == event_id:
                with open(date_dir) as f:
                    return ChronicleEvent.from_dict(json.load(f))
        
        return None
    
    async def get_session_events(
        self,
        session_id: str,
        since: Optional[datetime] = None
    ) -> List[ChronicleEvent]:
        """Get all events for a session"""
        event_ids = self._session_events.get(session_id, [])
        events = [self._events.get(eid) for eid in event_ids]
        events = [e for e in events if e is not None]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return sorted(events, key=lambda e: e.timestamp)
    
    async def get_agent_events(
        self,
        agent_id: str,
        event_types: Optional[List[EventType]] = None
    ) -> List[ChronicleEvent]:
        """Get events for an agent"""
        event_ids = self._agent_events.get(agent_id, [])
        events = [self._events.get(eid) for eid in event_ids]
        events = [e for e in events if e is not None]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return sorted(events, key=lambda e: e.timestamp)
    
    async def query(
        self,
        agent_id: Optional[str] = None,
        event_types: Optional[List[EventType]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        outcome: Optional[Outcome] = None,
        limit: int = 100
    ) -> List[ChronicleEvent]:
        """Query events with filters"""
        events = list(self._events.values())
        
        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        if until:
            events = [e for e in events if e.timestamp <= until]
        
        if outcome:
            events = [e for e in events if e.outcome == outcome]
        
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    async def get_recent(self, minutes: int = 60) -> List[ChronicleEvent]:
        """Get recent events"""
        since = datetime.utcnow() - timedelta(minutes=minutes)
        return await self.query(since=since)


class LoopDetector:
    """Detects and prevents repetitive behavior loops"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self._attempt_history: Dict[str, List[Dict]] = {}
    
    def is_similar_attempt(
        self,
        action: str,
        parameters: Dict[str, Any],
        history: List[ChronicleEvent]
    ) -> Tuple[bool, Optional[ChronicleEvent]]:
        """
        Check if this action has been tried before
        
        Returns:
            Tuple of (is_similar, previous_event)
        """
        for event in reversed(history):
            if event.action == action:
                # Compare parameters
                similarity = self._calculate_similarity(parameters, event.parameters)
                if similarity >= self.similarity_threshold:
                    return True, event
        
        return False, None
    
    def _calculate_similarity(
        self,
        params1: Dict[str, Any],
        params2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two parameter sets"""
        if not params1 and not params2:
            return 1.0
        
        if not params1 or not params2:
            return 0.0
        
        # Simple Jaccard similarity on keys
        keys1 = set(params1.keys())
        keys2 = set(params2.keys())
        
        intersection = keys1 & keys2
        union = keys1 | keys2
        
        if not union:
            return 1.0
        
        key_similarity = len(intersection) / len(union)
        
        # Check values for common keys
        value_matches = sum(
            1 for k in intersection if params1.get(k) == params2.get(k)
        )
        value_similarity = value_matches / len(intersection) if intersection else 0
        
        return (key_similarity + value_similarity) / 2
    
    def detect_loop_pattern(
        self,
        events: List[ChronicleEvent],
        window_size: int = 5
    ) -> Optional[List[ChronicleEvent]]:
        """Detect if recent events form a loop pattern"""
        if len(events) < window_size * 2:
            return None
        
        recent = events[-window_size:]
        previous = events[-window_size * 2:-window_size]
        
        # Check if action sequences match
        recent_actions = [e.action for e in recent]
        previous_actions = [e.action for e in previous]
        
        if recent_actions == previous_actions:
            return recent
        
        return None


class ChronicleMemory:
    """
    Main Chronicle Memory System - Temporal Episodic Memory
    
    Stores the narrative history of agent actions, enabling:
    - Experience replay and learning
    - Loop detection
    - Narrative reconstruction
    - Strategy memory
    """
    
    def __init__(
        self,
        storage_path: str = "./chronicle_store",
        session_id: Optional[str] = None
    ):
        self.store = ChronicleStore(storage_path)
        self.session_id = session_id or str(uuid.uuid4())
        self.loop_detector = LoopDetector()
        self._current_agent: Optional[str] = None
        
        # Pattern learning
        self._patterns: Dict[str, ExperiencePattern] = {}
        
        # Event stack for parent-child relationships
        self._event_stack: List[str] = []
    
    def set_agent(self, agent_id: str):
        """Set the current agent ID"""
        self._current_agent = agent_id

    async def synthesize_adr(
        self,
        title: str,
        decision: str,
        rationale: str,
        consequences: str,
        alternatives_rejected: Optional[List[str]] = None,
        entangled_components: Optional[List[str]] = None,
        convergence_trace_event_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synthesize an Architecture Decision Record (ADR) using MADR template.

        Returns:
            Dictionary with ADR metadata and content.
        """
        decision_id = str(uuid.uuid4())[:12]
        alternatives_rejected = alternatives_rejected or []
        entangled_components = entangled_components or []

        # Build convergence trace if not provided
        if convergence_trace_event_ids is None:
            recent_events = await self.store.get_session_events(self.session_id)
            convergence_trace_event_ids = [e.event_id for e in recent_events[-10:]]

        adr_content = (
            f"# ADR-{decision_id}: {title}\n"
            "## Context\n"
            f"{rationale}\n"
            "## Decision\n"
            f"{decision}\n"
            "## Rationale\n"
            f"{rationale}\n"
            "## Consequences\n"
            f"{consequences}\n"
            f"## Alternatives Rejected\n"
            f"{chr(10).join(['- ' + a for a in alternatives_rejected]) if alternatives_rejected else '- None'}\n"
            f"## Entangled Components\n"
            f"{chr(10).join(['- ' + e for e in entangled_components]) if entangled_components else '- None'}\n"
        )

        adr_dir = self.store.storage_path / "adrs"
        adr_dir.mkdir(parents=True, exist_ok=True)
        adr_path = adr_dir / f"ADR-{decision_id}.md"
        with open(adr_path, "w", encoding="utf-8") as f:
            f.write(adr_content)

        await self.record_event(
            event_type=EventType.DECISION_MADE,
            action="adr_synthesized",
            parameters={
                "decision_id": decision_id,
                "adr_path": str(adr_path),
                "alternatives_rejected": alternatives_rejected,
                "convergence_trace": convergence_trace_event_ids
            },
            outcome=Outcome.SUCCESS,
            narrative=f"ADR synthesized: {title}"
        )

        return {
            "decision_id": decision_id,
            "title": title,
            "adr_path": str(adr_path),
            "alternatives_rejected": alternatives_rejected,
            "convergence_trace": convergence_trace_event_ids,
            "entangled_components": entangled_components,
            "content": adr_content
        }

    async def extract_reasoning_path(self, limit: int = 25) -> List[str]:
        """
        Extract a reasoning path (sequence of actions) from recent chronicle events.
        """
        events = await self.store.get_session_events(self.session_id)
        path = [e.action for e in events[-limit:]]
        return [p for p in path if p]
    
    async def record_event(
        self,
        event_type: EventType,
        action: str,
        parameters: Dict[str, Any] = None,
        outcome: Outcome = Outcome.UNKNOWN,
        narrative: str = None,
        context: Dict[str, Any] = None
    ) -> ChronicleEvent:
        """
        Record an event in the chronicle

        Args:
            event_type: Type of event
            action: Action description
            parameters: Action parameters
            outcome: Outcome of the action
            narrative: Human-readable description
            context: Additional context

        Returns:
            The recorded event
        """
        start_time = time.time()
        event_id = str(uuid.uuid4())[:12]
        success = False

        try:
            # Get parent from stack
            parent_id = self._event_stack[-1] if self._event_stack else None

            event = ChronicleEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                agent_id=self._current_agent or "unknown",
                session_id=self.session_id,
                action=action,
                parameters=parameters or {},
                outcome=outcome,
                context=context or {},
                parent_event_id=parent_id,
                narrative=narrative or action
            )

            await self.store.append(event)

            logger.debug(f"Recorded event: {event_id} - {action}")

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance
            success = True
            duration = time.time() - start_time
            _extract_chronicle_knowledge("record_event", event_id, event_type.value, event)
            _track_chronicle_performance("record_event", True, duration, event_type.value)

            return event

        except Exception as e:
            duration = time.time() - start_time
            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            _trigger_chronicle_alerts("record_event", False, event_id, str(e))
            _track_chronicle_performance("record_event", False, duration, event_type.value)
            raise
    
    async def start_action(
        self,
        action: str,
        parameters: Dict[str, Any] = None,
        narrative: str = None
    ) -> ChronicleEvent:
        """Record the start of an action"""
        event = await self.record_event(
            event_type=EventType.ACTION_STARTED,
            action=action,
            parameters=parameters,
            narrative=narrative or f"Starting: {action}"
        )
        
        # Push to stack
        self._event_stack.append(event.event_id)
        
        return event
    
    async def complete_action(
        self,
        outcome: Outcome = Outcome.SUCCESS,
        result: Any = None,
        lesson: str = None,
        duration_ms: float = None
    ):
        """Complete the current action"""
        if not self._event_stack:
            logger.warning("No active action to complete")
            return
        
        parent_id = self._event_stack.pop()
        parent_event = await self.store.get_event(parent_id)
        
        if not parent_event:
            return
        
        # Record completion
        await self.record_event(
            event_type=EventType.ACTION_COMPLETED,
            action=f"{parent_event.action}_completed",
            parameters={"result": result},
            outcome=outcome,
            narrative=f"Completed: {parent_event.action} ({outcome.value})",
            parent_event_id=parent_id
        )
        
        # Update parent with outcome
        parent_event.outcome = outcome
        parent_event.lesson = lesson
        parent_event.duration_ms = duration_ms
        
        await self.store.append(parent_event)
    
    async def check_for_loops(
        self,
        action: str,
        parameters: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if we're about to repeat a failed strategy
        
        Returns:
            Tuple of (should_prevent, warning_message)
        """
        # Get recent events
        recent = await self.store.get_recent(minutes=30)
        
        # Check for similar attempts
        is_similar, prev_event = self.loop_detector.is_similar_attempt(
            action, parameters, recent
        )
        
        if is_similar and prev_event:
            if prev_event.outcome in [Outcome.FAILURE, Outcome.ERROR]:
                return True, (
                    f"This strategy was already attempted {prev_event.timestamp} "
                    f"and failed. Consider a different approach."
                )
            elif prev_event.retry_count >= 3:
                return True, (
                    f"This strategy has been retried {prev_event.retry_count} times. "
                    f"Consider escalating or pivoting."
                )
        
        # Check for loop patterns
        loop = self.loop_detector.detect_loop_pattern(recent)
        if loop:
            return True, (
                f"Detected potential loop pattern: actions are repeating. "
                f"Last attempt: {loop[-1].action}"
            )
        
        return False, None
    
    async def get_experience_summary(
        self,
        action_type: Optional[str] = None,
        timeframe_minutes: int = 60
    ) -> Dict[str, Any]:
        """Get summary of recent experiences"""
        events = await self.store.get_recent(timeframe_minutes)
        
        if action_type:
            events = [e for e in events if action_type in e.action]
        
        # Calculate statistics
        total = len(events)
        successes = len([e for e in events if e.outcome == Outcome.SUCCESS])
        failures = len([e for e in events if e.outcome == Outcome.FAILURE])
        
        # Group by action
        action_counts = defaultdict(int)
        for event in events:
            action_counts[event.action] += 1
        
        # Get lessons learned
        lessons = [
            e.lesson for e in events 
            if e.lesson and e.outcome == Outcome.SUCCESS
        ]
        
        return {
            "total_events": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0,
            "most_common_actions": sorted(
                action_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "lessons_learned": lessons[:10]
        }
    
    async def reconstruct_narrative(
        self,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> Narrative:
        """Reconstruct a narrative from events"""
        sid = session_id or self.session_id
        events = await self.store.get_session_events(sid, since)
        
        # Build narrative
        summary_parts = []
        lessons = []
        
        for event in events:
            if event.event_type == EventType.ACTION_STARTED:
                summary_parts.append(event.narrative)
            if event.lesson:
                lessons.append(event.lesson)
        
        summary = " -> ".join(summary_parts) if summary_parts else "No actions recorded"
        
        # Determine overall outcome
        if events:
            final_outcome = events[-1].outcome
        else:
            final_outcome = Outcome.UNKNOWN
        
        return Narrative(
            narrative_id=str(uuid.uuid4())[:12],
            title=f"Session {sid[:8]}",
            events=events,
            summary=summary,
            lessons_learned=lessons,
            outcome=final_outcome,
            start_time=events[0].timestamp if events else datetime.utcnow(),
            end_time=events[-1].timestamp if events else None
        )
    
    async def suggest_strategy(
        self,
        current_action: str,
        current_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Suggest a strategy based on past experiences
        
        This is called when the Self-Healing loop kicks in for the 5th time.
        The agent needs to know "I already tried Strategy X and it failed".
        """
        # Get similar past actions
        recent = await self.store.get_recent(minutes=120)
        
        similar_events = [
            e for e in recent 
            if current_action in e.action
        ]
        
        if not similar_events:
            return None
        
        # Find successful variants
        successes = [e for e in similar_events if e.outcome == Outcome.SUCCESS]
        failures = [e for e in similar_events if e.outcome == Outcome.FAILURE]
        
        if successes:
            # Suggest the successful approach
            best = max(successes, key=lambda e: e.timestamp)
            return {
                "suggestion": "try_previous_success",
                "previous_event": best.event_id,
                "parameters": best.parameters,
                "confidence": len(successes) / len(similar_events),
                "reasoning": f"This approach succeeded {len(successes)} times before"
            }
        
        if len(failures) >= 3:
            # Suggest pivoting
            return {
                "suggestion": "pivot_strategy",
                "failed_attempts": len(failures),
                "confidence": 1.0,
                "reasoning": f"Strategy has failed {len(failures)} times. Try a different approach."
            }
        
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get chronicle statistics"""
        recent = await self.store.get_recent(minutes=60 * 24)  # Last 24 hours
        
        return {
            "total_events_24h": len(recent),
            "session_id": self.session_id,
            "current_agent": self._current_agent,
            "active_action_stack": len(self._event_stack)
        }


# Convenience functions for quick usage
async def create_chronicle(
    session_id: Optional[str] = None,
    storage_path: str = "./chronicle_store"
) -> ChronicleMemory:
    """Create a new chronicle memory instance"""
    return ChronicleMemory(storage_path, session_id)


async def record_attempt(
    action: str,
    parameters: Dict[str, Any] = None,
    chronicle: ChronicleMemory = None
) -> ChronicleMemory:
    """Quick function to record an attempt"""
    if chronicle is None:
        chronicle = await create_chronicle()
    
    await chronicle.start_action(action, parameters)
    return chronicle


# Example usage
if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("CHRONICLE MEMORY DEMO - Temporal Episodic Memory")
        print("=" * 60)
        
        # Create chronicle
        chronicle = await create_chronicle()
        chronicle.set_agent("blue-team-agent-1")
        
        print("\n[OK] Chronicle Memory initialized")
        print(f"  Session: {chronicle.session_id[:8]}...")
        print(f"  Agent: {chronicle._current_agent}")
        
        # Simulate self-healing loop scenario
        print("\n" + "=" * 60)
        print("Scenario: Self-Healing Loop")
        print("=" * 60)
        
        strategies = [
            ("strategy_A", {"approach": "quick_fix"}, Outcome.FAILURE),
            ("strategy_A", {"approach": "quick_fix"}, Outcome.FAILURE),
            ("strategy_A", {"approach": "quick_fix"}, Outcome.FAILURE),
        ]
        
        for i, (action, params, outcome) in enumerate(strategies, 1):
            # Check for loops before attempting
            should_prevent, warning = await chronicle.check_for_loops(action, params)
            
            print(f"\nAttempt {i}: {action}")
            
            if should_prevent:
                print(f"  [WARN]  LOOP DETECTED: {warning}")
                print("  -> Suggesting strategy pivot")
                break
            
            # Record the attempt
            await chronicle.start_action(action, params, f"Trying {action}")
            await chronicle.complete_action(outcome=outcome)
            
            print(f"  Outcome: {outcome.value}")
        
        # Get experience summary
        print("\n" + "=" * 60)
        print("Experience Summary:")
        print("=" * 60)
        
        summary = await chronicle.get_experience_summary()
        print(f"  Total events: {summary['total_events']}")
        print(f"  Successes: {summary['successes']}")
        print(f"  Failures: {summary['failures']}")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        
        # Reconstruct narrative
        print("\n" + "=" * 60)
        print("Narrative Reconstruction:")
        print("=" * 60)
        
        narrative = await chronicle.reconstruct_narrative()
        print(narrative.to_story())
        
        # Test strategy suggestion
        print("=" * 60)
        print("Strategy Suggestion:")
        print("=" * 60)
        
        suggestion = await chronicle.suggest_strategy("strategy_A", {})
        if suggestion:
            print(f"Suggestion: {suggestion['suggestion']}")
            print(f"Reasoning: {suggestion['reasoning']}")
        
        print("\n[OK] Demo complete")
    
    asyncio.run(demo())
