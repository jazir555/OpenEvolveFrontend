"""
RESE Event Bus Implementation

Following CLAUDE.md principles:
- Law of Idempotency: Event deduplication by event_id
- Law of Configuration Explicitness: All config via env vars
- Structured Logging: JSON with correlation_id
- Timeout: All operations bounded
"""

import os
import sys
import json
import uuid
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timezone
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schemas"))

try:
    from orchestration.config import PipelineConfig
except ImportError:
    from glue.orchestration.config import PipelineConfig


# ============================================================================
# EVENT SCHEMAS
# ============================================================================

@dataclass
class Event:
    """
    RESE Event schema.

    All events include correlation_id for tracing across phases.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_service: str = ""
    target_service: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "correlation_id": self.correlation_id,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "source_service": self.source_service,
            "target_service": self.target_service,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create from dictionary."""
        return cls(**data)


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class EventBusLogger:
    """Structured logger for Event Bus operations."""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.logger = logging.getLogger("rese_event_bus")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            self.logger.addHandler(handler)

    def _log(self, level: str, msg: str, **kwargs):
        """Log in JSON Lines format."""
        log_entry = {
            "msg": msg,
            "level": level,
            "correlation_id": self.correlation_id,
            "source_service": "rese_event_bus",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        log_json = json.dumps(log_entry)
        self.logger.log(getattr(logging, level.upper()), log_json)

    def info(self, msg: str, **kwargs):
        self._log("INFO", msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log("WARNING", msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log("ERROR", msg, **kwargs)

    def debug(self, msg: str, **kwargs):
        self._log("DEBUG", msg, **kwargs)


# ============================================================================
# EVENT BUS
# ============================================================================

class EventBus:
    """
    Event bus for inter-phase communication.

    Features:
    - Publish/Subscribe pattern
    - Event deduplication (idempotency)
    - Event persistence (optional)
    - Correlation tracking
    - Dead Letter Queue integration
    """

    # Event types
    PHASE_I_STARTED = "phase_i.started"
    PHASE_I_COMPLETED = "phase_i.completed"
    PHASE_I_FAILED = "phase_i.failed"

    PHASE_II_STARTED = "phase_ii.started"
    PHASE_II_COMPLETED = "phase_ii.completed"
    PHASE_II_FAILED = "phase_ii.failed"

    PHASE_III_STARTED = "phase_iii.started"
    PHASE_III_COMPLETED = "phase_iii.completed"
    PHASE_III_FAILED = "phase_iii.failed"

    PHASE_IV_STARTED = "phase_iv.started"
    PHASE_IV_COMPLETED = "phase_iv.completed"
    PHASE_IV_FAILED = "phase_iv.failed"

    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"

    HYPOTHESIS_GENERATED = "hypothesis.generated"
    HYPOTHESIS_VALIDATED = "hypothesis.validated"
    PATTERN_RECOGNIZED = "pattern.recognized"
    ISOMORPHISM_FOUND = "isomorphism.found"
    CONTRADICTION_DETECTED = "contradiction.detected"

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize Event Bus.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig.from_env()
        self.logger = EventBusLogger()

        # Subscriptions: event_type -> list of handlers
        self.subscriptions: Dict[str, List[Callable]] = defaultdict(list)

        # Event history for deduplication
        self.processed_events: Set[str] = set()

        # Event buffer (circular buffer)
        self.max_events = self.config.event_bus_max_events
        self.event_buffer: deque = deque(maxlen=self.max_events)

        # Lock for thread safety
        self.lock = threading.Lock()

        # Event persistence
        self.persist_enabled = self.config.event_bus_persist_events
        self.persist_path = self.config.event_bus_persist_path

        if self.persist_enabled and self.persist_path:
            self._load_persisted_events()

        self.logger.info(
            "Event Bus initialized",
            max_events=self.max_events,
            persist_enabled=self.persist_enabled
        )

    def subscribe(self, event_type: str, handler: Callable[[Event], None]):
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function that takes Event
        """
        with self.lock:
            self.subscriptions[event_type].append(handler)

        self.logger.debug(
            "Subscribed to event",
            event_type=event_type,
            handler=handler.__name__
        )

    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]):
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Callback function to remove
        """
        with self.lock:
            if event_type in self.subscriptions:
                try:
                    self.subscriptions[event_type].remove(handler)
                except ValueError:
                    pass

        self.logger.debug(
            "Unsubscribed from event",
            event_type=event_type,
            handler=handler.__name__
        )

    def publish(self, event: Event) -> bool:
        """
        Publish an event.

        Idempotent: Duplicate events (by event_id) are ignored.

        Args:
            event: Event to publish

        Returns:
            True if event was published (not duplicate)
        """
        # Check for duplicate (Law of Idempotency)
        if event.event_id in self.processed_events:
            self.logger.debug(
                "Duplicate event ignored",
                event_id=event.event_id,
                event_type=event.event_type
            )
            return False

        with self.lock:
            # Mark as processed
            self.processed_events.add(event.event_id)

            # Add to buffer
            self.event_buffer.append(event)

            # Persist if enabled
            if self.persist_enabled and self.persist_path:
                self._persist_event(event)

        # Get subscribers for this event type
        subscribers = self.subscriptions.get(event.event_type, [])

        # Publish to all subscribers
        for handler in subscribers:
            try:
                handler(event)
            except Exception as e:
                self.logger.error(
                    "Event handler failed",
                    event_type=event.event_type,
                    handler=handler.__name__,
                    error=str(e)
                )

        self.logger.debug(
            "Event published",
            event_id=event.event_id,
            event_type=event.event_type,
            subscriber_count=len(subscribers)
        )

        return True

    def publish_sync(self, event_type: str, data: Dict[str, Any],
                     correlation_id: Optional[str] = None,
                     source_service: str = "",
                     target_service: Optional[str] = None) -> Event:
        """
        Publish an event synchronously (helper method).

        Args:
            event_type: Type of event
            data: Event data
            correlation_id: Correlation ID for tracing
            source_service: Source service
            target_service: Target service (optional)

        Returns:
            Created Event
        """
        event = Event(
            event_type=event_type,
            data=data,
            correlation_id=correlation_id or str(uuid.uuid4()),
            source_service=source_service,
            target_service=target_service
        )
        self.publish(event)
        return event

    async def publish_async(self, event: Event) -> bool:
        """
        Publish an event asynchronously.

        Args:
            event: Event to publish

        Returns:
            True if event was published
        """
        # For now, just call sync publish
        # TODO: Implement true async with asyncio
        return self.publish(event)

    def get_history(self, event_type: Optional[str] = None,
                    correlation_id: Optional[str] = None,
                    limit: int = 100) -> List[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (optional)
            correlation_id: Filter by correlation ID (optional)
            limit: Maximum number of events to return

        Returns:
            List of events
        """
        with self.lock:
            events = list(self.event_buffer)

        # Filter
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]

        # Limit
        events = events[-limit:]

        return events

    def clear_history(self):
        """Clear event history (except processed IDs for deduplication)."""
        with self.lock:
            self.event_buffer.clear()

        self.logger.info("Event history cleared")

    def _persist_event(self, event: Event):
        """
        Persist event to disk.

        Args:
            event: Event to persist
        """
        try:
            persist_path = Path(self.persist_path)
            persist_path.mkdir(parents=True, exist_ok=True)

            event_file = persist_path / f"{event.event_id}.json"
            with open(event_file, 'w') as f:
                json.dump(event.to_dict(), f, indent=2)

        except Exception as e:
            self.logger.error(
                "Failed to persist event",
                event_id=event.event_id,
                error=str(e)
            )

    def _load_persisted_events(self):
        """Load persisted events from disk."""
        try:
            persist_path = Path(self.persist_path)
            if not persist_path.exists():
                return

            for event_file in persist_path.glob("*.json"):
                with open(event_file, 'r') as f:
                    event_data = json.load(f)
                    event = Event.from_dict(event_data)

                    # Add to processed set (don't replay)
                    self.processed_events.add(event.event_id)

            self.logger.info(
                "Loaded persisted events",
                count=len(self.processed_events)
            )

        except Exception as e:
            self.logger.error(
                "Failed to load persisted events",
                error=str(e)
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.

        Returns:
            Statistics dictionary
        """
        with self.lock:
            subscription_counts = {
                event_type: len(handlers)
                for event_type, handlers in self.subscriptions.items()
            }

        return {
            "total_subscriptions": sum(subscription_counts.values()),
            "subscription_counts": subscription_counts,
            "processed_events": len(self.processed_events),
            "buffer_size": len(self.event_buffer),
            "max_buffer_size": self.max_events,
            "persist_enabled": self.persist_enabled,
            "persist_path": self.persist_path,
        }


# ============================================================================
# CORRELATION MANAGER
# ============================================================================

class CorrelationManager:
    """
    Manages correlation IDs across all phases.

    Ensures traceability across the entire pipeline.
    """

    def __init__(self):
        self.logger = EventBusLogger()
        self.active_correlations: Dict[str, Dict[str, Any]] = {}

    def create_correlation(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new correlation ID.

        Args:
            metadata: Optional metadata to attach

        Returns:
            Correlation ID
        """
        correlation_id = str(uuid.uuid4())

        self.active_correlations[correlation_id] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "phases_completed": [],
            "phases_failed": [],
        }

        self.logger.debug(
            "Created correlation ID",
            correlation_id=correlation_id
        )

        return correlation_id

    def update_correlation(self, correlation_id: str, phase: str,
                          status: str, **kwargs):
        """
        Update correlation metadata.

        Args:
            correlation_id: Correlation ID
            phase: Phase name
            status: Status (completed, failed, etc.)
            **kwargs: Additional metadata
        """
        if correlation_id not in self.active_correlations:
            self.logger.warning(
                "Unknown correlation ID",
                correlation_id=correlation_id
            )
            return

        correlation_data = self.active_correlations[correlation_id]

        if status == "completed":
            correlation_data["phases_completed"].append(phase)
        elif status == "failed":
            correlation_data["phases_failed"].append(phase)

        correlation_data.update(kwargs)

        self.logger.debug(
            "Updated correlation",
            correlation_id=correlation_id,
            phase=phase,
            status=status
        )

    def get_correlation(self, correlation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get correlation metadata.

        Args:
            correlation_id: Correlation ID

        Returns:
            Correlation metadata or None
        """
        return self.active_correlations.get(correlation_id)

    def close_correlation(self, correlation_id: str):
        """
        Close a correlation (cleanup).

        Args:
            correlation_id: Correlation ID
        """
        if correlation_id in self.active_correlations:
            del self.active_correlations[correlation_id]

            self.logger.debug(
                "Closed correlation",
                correlation_id=correlation_id
            )
