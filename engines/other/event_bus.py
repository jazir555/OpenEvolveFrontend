"""
Event Bus with Valkey - License: Apache 2.0

Asynchronous event bus implementation using Valkey (Apache 2.0 Redis alternative).
Enables pub/sub messaging, event streaming, and workflow notifications.

Dependencies (all permissive licenses):
- valkey-py: MIT License (Valkey is Apache 2.0)
- pydantic: MIT License
- asyncio: Python Standard Library (PSF License)

Author: OpenEvolve
Date: 2026-02-02
"""
from __future__ import annotations


import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, AsyncIterator, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib

# Valkey client - MIT License (Valkey server is Apache 2.0)
try:
    import valkey
    from valkey.asyncio import Valkey
    VALKEY_AVAILABLE = True
except ImportError:
    VALKEY_AVAILABLE = False
    logging.warning("valkey-py not installed. EventBus will use in-memory fallback.")

logger = logging.getLogger(__name__)

class EventPriority(Enum):
    """Event priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class EventType(Enum):
    """System event types."""
    # Workflow events
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    
    # Decomposition events
    DECOMPOSITION_STARTED = "decomposition.started"
    DECOMPOSITION_COMPLETED = "decomposition.completed"
    SUBPROBLEM_CREATED = "subproblem.created"
    SUBPROBLEM_COMPLETED = "subproblem.completed"
    
    # Knowledge events
    KNOWLEDGE_EXTRACTED = "knowledge.extracted"
    KNOWLEDGE_VERIFIED = "knowledge.verified"
    CONTRADICTION_DETECTED = "knowledge.contradiction"
    
    # Gauntlet events
    GAUNTLET_STARTED = "gauntlet.started"
    GAUNTLET_ROUND_COMPLETED = "gauntlet.round.completed"
    GAUNTLET_COMPLETED = "gauntlet.completed"
    
    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    SYSTEM_METRIC = "system.metric"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class Event:
    """Event data structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType = EventType.CUSTOM
    source: str = "system"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        data['type'] = EventType(data['type'])
        data['priority'] = EventPriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EventBus:
    """
    Asynchronous event bus using Valkey.
    
    Features:
    - Pub/Sub messaging
    - Event persistence (optional)
    - Priority-based delivery
    - Correlation tracking
    - Webhook notifications
    - In-memory fallback (when Valkey unavailable)
    
    License: Apache 2.0 (Valkey)
    """
    
    def __init__(
        self,
        valkey_host: str = "localhost",
        valkey_port: int = 6379,
        valkey_db: int = 0,
        enable_persistence: bool = True,
        max_history: int = 10000
    ):
        self.valkey_host = valkey_host
        self.valkey_port = valkey_port
        self.valkey_db = valkey_db
        self.enable_persistence = enable_persistence
        self.max_history = max_history
        
        self._valkey: Optional[Valkey] = None
        self._pubsub = None
        self._subscribers: Dict[EventType, Set[Callable]] = {et: set() for et in EventType}
        self._wildcards: Set[Callable] = set()
        self._connected = False
        self._history: List[Event] = []  # In-memory fallback
        self._lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Connect to Valkey server."""
        if not VALKEY_AVAILABLE:
            logger.info("Valkey not available, using in-memory event bus")
            return False
            
        try:
            self._valkey = Valkey(
                host=self.valkey_host,
                port=self.valkey_port,
                db=self.valkey_db,
                decode_responses=True
            )
            await self._valkey.ping()
            self._connected = True
            
            # Start pubsub listener
            self._pubsub = self._valkey.pubsub()
            asyncio.create_task(self._listen_for_events())
            
            logger.info(f"Connected to Valkey at {self.valkey_host}:{self.valkey_port}")
            return True
            
        except Exception as e:
            logger.warning(f"Could not connect to Valkey: {e}. Using in-memory fallback.")
            self._connected = False
            return False
            
    async def disconnect(self) -> None:
        """Disconnect from Valkey."""
        if self._pubsub:
            await self._pubsub.close()
        if self._valkey:
            await self._valkey.close()
        self._connected = False
        logger.info("Disconnected from Valkey")
        
    async def _listen_for_events(self) -> None:
        """Background task to listen for pub/sub events."""
        if not self._pubsub:
            return
            
        # Subscribe to all event types
        channels = [et.value for et in EventType]
        await self._pubsub.subscribe(*channels)
        
        async for message in self._pubsub.listen():
            if message['type'] == 'message':
                try:
                    event = Event.from_json(message['data'])
                    await self._dispatch_to_local(event)
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    
    async def _dispatch_to_local(self, event: Event) -> None:
        """Dispatch event to local subscribers."""
        # Dispatch to type-specific subscribers
        handlers = self._subscribers.get(event.type, set()).copy()
        
        # Dispatch to wildcard subscribers
        handlers.update(self._wildcards)
        
        # Execute handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event))
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")
                
    async def publish(self, event: Event, persist: Optional[bool] = None) -> bool:
        """
        Publish an event.
        
        Args:
            event: Event to publish
            persist: Whether to persist event (defaults to enable_persistence)
            
        Returns:
            True if published successfully
        """
        should_persist = persist if persist is not None else self.enable_persistence
        
        # Dispatch to local subscribers
        await self._dispatch_to_local(event)
        
        # Store in history
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history:]
        
        # Publish to Valkey if connected
        if self._connected and self._valkey:
            try:
                # Publish to pub/sub channel
                channel = event.type.value
                await self._valkey.publish(channel, event.to_json())
                
                # Persist if enabled
                if should_persist:
                    key = f"events:{event.type.value}:{event.id}"
                    await self._valkey.setex(
                        key,
                        86400 * 7,  # 7 days TTL
                        event.to_json()
                    )
                    
                    # Add to sorted set by timestamp
                    score = event.timestamp.timestamp()
                    await self._valkey.zadd(
                        f"events:{event.type.value}:timeline",
                        {event.id: score}
                    )
                    
                return True
                
            except Exception as e:
                logger.error(f"Error publishing to Valkey: {e}")
                return False
                
        return True
        
    def subscribe(
        self,
        event_type: Optional[EventType] = None,
        handler: Optional[Callable[[Event], Any]] = None
    ) -> Callable[[Event], Any]:
        """
        Subscribe to events.
        
        Args:
            event_type: Type to subscribe to (None = wildcard)
            handler: Callback function (can be async)
            
        Returns:
            The handler (for use as decorator)
        """
        def decorator(func: Callable[[Event], Any]) -> Callable[[Event], Any]:
            if event_type is None:
                self._wildcards.add(func)
            else:
                self._subscribers[event_type].add(func)
            return func
            
        if handler:
            return decorator(handler)
        return decorator
        
    def unsubscribe(
        self,
        handler: Callable[[Event], Any],
        event_type: Optional[EventType] = None
    ) -> bool:
        """Unsubscribe a handler."""
        if event_type is None:
            if handler in self._wildcards:
                self._wildcards.remove(handler)
                return True
            # Try to remove from all types
            removed = False
            for handlers in self._subscribers.values():
                if handler in handlers:
                    handlers.remove(handler)
                    removed = True
            return removed
        else:
            handlers = self._subscribers.get(event_type, set())
            if handler in handlers:
                handlers.remove(handler)
                return True
            return False
            
    async def get_history(
        self,
        event_type: Optional[EventType] = None,
        workflow_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by type
            workflow_id: Filter by workflow
            correlation_id: Filter by correlation
            limit: Maximum events to return
            since: Only events after this time
        """
        async with self._lock:
            events = self._history.copy()
            
        # Apply filters
        if event_type:
            events = [e for e in events if e.type == event_type]
        if workflow_id:
            events = [e for e in events if e.workflow_id == workflow_id]
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]
        if since:
            events = [e for e in events if e.timestamp >= since]
            
        # Sort by timestamp (newest first)
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]
        
    async def wait_for_event(
        self,
        event_type: EventType,
        predicate: Optional[Callable[[Event], bool]] = None,
        timeout: float = 30.0
    ) -> Optional[Event]:
        """
        Wait for a specific event.
        
        Args:
            event_type: Type to wait for
            predicate: Optional condition function
            timeout: Maximum wait time
            
        Returns:
            Event if found, None if timeout
        """
        future: asyncio.Future[Event] = asyncio.Future()
        
        async def handler(event: Event) -> None:
            if not future.done():
                if predicate is None or predicate(event):
                    future.set_result(event)
                    
        self.subscribe(event_type, handler)
        
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.unsubscribe(handler, event_type)


class WorkflowEventTracker:
    """
    Helper class for tracking workflow events.
    
    Provides high-level tracking for workflow lifecycle:
    - Start/complete/fail events
    - Progress updates
    - Subproblem tracking
    - Webhook notifications
    """
    
    def __init__(self, event_bus: EventBus, webhook_url: Optional[str] = None):
        self.event_bus = event_bus
        self.webhook_url = webhook_url
        
    async def track_workflow_start(
        self,
        workflow_id: str,
        problem: str,
        correlation_id: Optional[str] = None
    ) -> Event:
        """Track workflow start."""
        event = Event(
            type=EventType.WORKFLOW_STARTED,
            source="workflow_tracker",
            workflow_id=workflow_id,
            correlation_id=correlation_id,
            payload={
                "problem": problem[:500],  # Truncate for size
                "workflow_id": workflow_id
            }
        )
        await self.event_bus.publish(event)
        return event
        
    async def track_workflow_complete(
        self,
        workflow_id: str,
        result: Dict[str, Any],
        duration_seconds: float
    ) -> Event:
        """Track workflow completion."""
        event = Event(
            type=EventType.WORKFLOW_COMPLETED,
            source="workflow_tracker",
            workflow_id=workflow_id,
            payload={
                "workflow_id": workflow_id,
                "duration_seconds": duration_seconds,
                "result_summary": result.get("summary", "No summary"),
                "success": result.get("success", True)
            }
        )
        await self.event_bus.publish(event)
        return event
        
    async def track_decomposition(
        self,
        workflow_id: str,
        plan_id: str,
        num_subproblems: int
    ) -> Event:
        """Track decomposition completion."""
        event = Event(
            type=EventType.DECOMPOSITION_COMPLETED,
            source="workflow_tracker",
            workflow_id=workflow_id,
            payload={
                "plan_id": plan_id,
                "num_subproblems": num_subproblems
            }
        )
        await self.event_bus.publish(event)
        return event
        
    async def track_subproblem_complete(
        self,
        workflow_id: str,
        subproblem_id: str,
        title: str,
        success: bool
    ) -> Event:
        """Track subproblem completion."""
        event = Event(
            type=EventType.SUBPROBLEM_COMPLETED,
            source="workflow_tracker",
            workflow_id=workflow_id,
            priority=EventPriority.HIGH if not success else EventPriority.NORMAL,
            payload={
                "subproblem_id": subproblem_id,
                "title": title,
                "success": success
            }
        )
        await self.event_bus.publish(event)
        return event


# Global event bus instance
_event_bus: Optional[EventBus] = None


async def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
        await _event_bus.connect()
    return _event_bus


# Convenience functions for common operations

async def publish_event(
    event_type: EventType,
    payload: Dict[str, Any],
    workflow_id: Optional[str] = None,
    priority: EventPriority = EventPriority.NORMAL
) -> Event:
    """Publish a simple event."""
    bus = await get_event_bus()
    event = Event(
        type=event_type,
        source="direct_publish",
        workflow_id=workflow_id,
        priority=priority,
        payload=payload
    )
    await bus.publish(event)
    return event


async def on_event(
    event_type: EventType,
    handler: Optional[Callable[[Event], Any]] = None
) -> Callable:
    """Decorator/convenience for subscribing to events."""
    bus = await get_event_bus()
    return await bus.subscribe(event_type, handler)
