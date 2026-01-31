"""
Real-time Collaboration Layer

WebSocket-based real-time collaboration for the knowledge engine.

Features:
- WebSocket server for client connections
- Real-time event broadcasting
- Operational Transformation for concurrent edits
- Presence tracking (who's online, what they're viewing)
- Cursor tracking and user awareness
- Conflict resolution
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

# Note: In a real implementation, you'd import websockets or similar
# For this example, we'll create interfaces that can be implemented

logger = logging.getLogger(__name__)


class CollaborationEventType(Enum):
    """Types of collaboration events."""
    # Knowledge events
    KNOWLEDGE_CREATED = "knowledge_created"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    KNOWLEDGE_DELETED = "knowledge_deleted"
    
    # User events
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_PRESENCE = "user_presence"
    CURSOR_POSITION = "cursor_position"
    SELECTION_CHANGE = "selection_change"
    
    # Lock events
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    LOCK_CONFLICT = "lock_conflict"
    
    # Operation events
    OPERATION_APPLIED = "operation_applied"
    OPERATION_REJECTED = "operation_rejected"


@dataclass
class UserPresence:
    """Information about a user's presence."""
    user_id: str
    user_name: str
    session_id: str
    status: str  # "active", "idle", "away"
    current_view: Optional[str] = None  # ID of item being viewed
    cursor_position: Optional[Dict[str, Any]] = None
    selection: Optional[Dict[str, Any]] = None
    joined_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "session_id": self.session_id,
            "status": self.status,
            "current_view": self.current_view,
            "cursor_position": self.cursor_position,
            "selection": self.selection,
            "joined_at": self.joined_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


@dataclass
class CollaborationEvent:
    """A real-time collaboration event."""
    event_id: str
    event_type: CollaborationEventType
    user_id: str
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }
    
    @classmethod
    def create(
        cls,
        event_type: CollaborationEventType,
        user_id: str,
        session_id: str,
        data: Dict[str, Any]
    ) -> CollaborationEvent:
        return cls(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.utcnow(),
            data=data
        )


@dataclass
class Operation:
    """
    Operational Transformation operation.
    Represents a change to a document that can be transformed.
    """
    operation_id: str
    user_id: str
    item_id: str
    operation_type: str  # "insert", "delete", "retain"
    position: int
    content: Optional[str] = None
    length: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    revision: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "user_id": self.user_id,
            "item_id": self.item_id,
            "operation_type": self.operation_type,
            "position": self.position,
            "content": self.content,
            "length": self.length,
            "timestamp": self.timestamp.isoformat(),
            "revision": self.revision
        }


class ItemLock:
    """Lock for exclusive editing of an item."""
    
    def __init__(self, item_id: str, user_id: str, session_id: str, ttl: int = 60):
        self.item_id = item_id
        self.user_id = user_id
        self.session_id = session_id
        self.acquired_at = datetime.utcnow()
        self.expires_at = self.acquired_at + timedelta(seconds=ttl)
        self.ttl = ttl
        
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at
    
    def extend(self, additional_seconds: int = 60):
        self.expires_at = datetime.utcnow() + timedelta(seconds=additional_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }


class PresenceManager:
    """Manages user presence and awareness."""
    
    def __init__(self, idle_timeout: int = 300):
        self.idle_timeout = idle_timeout
        self._presence: Dict[str, UserPresence] = {}  # session_id -> presence
        self._user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        self._item_viewers: Dict[str, Set[str]] = defaultdict(set)  # item_id -> session_ids
        self._lock = asyncio.Lock()
        
        # Start cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the presence manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the presence manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def user_joined(
        self, 
        user_id: str, 
        user_name: str, 
        session_id: str
    ) -> UserPresence:
        """Record user joining."""
        async with self._lock:
            presence = UserPresence(
                user_id=user_id,
                user_name=user_name,
                session_id=session_id,
                status="active"
            )
            
            self._presence[session_id] = presence
            self._user_sessions[user_id].add(session_id)
            
            logger.info(f"User joined: {user_name} ({user_id}) session {session_id}")
            
            return presence
    
    async def user_left(self, session_id: str) -> Optional[UserPresence]:
        """Record user leaving."""
        async with self._lock:
            presence = self._presence.pop(session_id, None)
            
            if presence:
                self._user_sessions[presence.user_id].discard(session_id)
                if presence.current_view:
                    self._item_viewers[presence.current_view].discard(session_id)
                
                logger.info(f"User left: {presence.user_name} session {session_id}")
                
            return presence
    
    async def update_activity(self, session_id: str):
        """Update last activity timestamp."""
        async with self._lock:
            presence = self._presence.get(session_id)
            if presence:
                presence.last_activity = datetime.utcnow()
                presence.status = "active"
    
    async def set_current_view(self, session_id: str, item_id: Optional[str]):
        """Set what item the user is currently viewing."""
        async with self._lock:
            presence = self._presence.get(session_id)
            if presence:
                # Remove from old view
                if presence.current_view:
                    self._item_viewers[presence.current_view].discard(session_id)
                
                # Add to new view
                presence.current_view = item_id
                if item_id:
                    self._item_viewers[item_id].add(session_id)
                
                presence.last_activity = datetime.utcnow()
    
    async def update_cursor(
        self, 
        session_id: str, 
        cursor_position: Dict[str, Any]
    ):
        """Update user's cursor position."""
        async with self._lock:
            presence = self._presence.get(session_id)
            if presence:
                presence.cursor_position = cursor_position
                presence.last_activity = datetime.utcnow()
    
    async def get_presence(self, session_id: str) -> Optional[UserPresence]:
        """Get presence info for a session."""
        async with self._lock:
            return self._presence.get(session_id)
    
    async def get_all_presence(self) -> List[UserPresence]:
        """Get all active presence info."""
        async with self._lock:
            return list(self._presence.values())
    
    async def get_item_viewers(self, item_id: str) -> List[UserPresence]:
        """Get all users viewing a specific item."""
        async with self._lock:
            session_ids = self._item_viewers.get(item_id, set())
            return [self._presence[sid] for sid in session_ids if sid in self._presence]
    
    async def _cleanup_loop(self):
        """Periodically clean up stale presence entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_stale_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Presence cleanup error: {e}")
    
    async def _cleanup_stale_entries(self):
        """Remove stale entries."""
        async with self._lock:
            now = datetime.utcnow()
            stale_threshold = now - timedelta(seconds=self.idle_timeout)
            
            stale_sessions = [
                sid for sid, presence in self._presence.items()
                if presence.last_activity < stale_threshold
            ]
            
            for sid in stale_sessions:
                presence = self._presence.pop(sid, None)
                if presence:
                    presence.status = "offline"
                    self._user_sessions[presence.user_id].discard(sid)
                    if presence.current_view:
                        self._item_viewers[presence.current_view].discard(sid)


class LockManager:
    """Manages item locks for exclusive editing."""
    
    def __init__(self, default_ttl: int = 60):
        self.default_ttl = default_ttl
        self._locks: Dict[str, ItemLock] = {}  # item_id -> lock
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the lock manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the lock manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def acquire_lock(
        self, 
        item_id: str, 
        user_id: str, 
        session_id: str,
        ttl: Optional[int] = None
    ) -> Tuple[bool, Optional[ItemLock]]:
        """
        Try to acquire a lock on an item.
        
        Returns:
            (success, lock) tuple
        """
        async with self._lock:
            existing_lock = self._locks.get(item_id)
            
            if existing_lock:
                if existing_lock.is_expired():
                    # Lock expired, can acquire
                    del self._locks[item_id]
                elif existing_lock.user_id == user_id:
                    # Same user, extend the lock
                    existing_lock.extend()
                    return True, existing_lock
                else:
                    # Someone else has the lock
                    return False, existing_lock
            
            # Acquire new lock
            new_lock = ItemLock(item_id, user_id, session_id, ttl or self.default_ttl)
            self._locks[item_id] = new_lock
            
            return True, new_lock
    
    async def release_lock(self, item_id: str, session_id: str) -> bool:
        """Release a lock."""
        async with self._lock:
            lock = self._locks.get(item_id)
            if lock and lock.session_id == session_id:
                del self._locks[item_id]
                return True
            return False
    
    async def get_lock(self, item_id: str) -> Optional[ItemLock]:
        """Get lock info for an item."""
        async with self._lock:
            lock = self._locks.get(item_id)
            if lock and lock.is_expired():
                del self._locks[item_id]
                return None
            return lock
    
    async def _cleanup_loop(self):
        """Periodically clean up expired locks."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._cleanup_expired_locks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lock cleanup error: {e}")
    
    async def _cleanup_expired_locks(self):
        """Remove expired locks."""
        async with self._lock:
            expired = [
                item_id for item_id, lock in self._locks.items()
                if lock.is_expired()
            ]
            for item_id in expired:
                del self._locks[item_id]
                logger.debug(f"Expired lock removed for {item_id}")


class OperationalTransformation:
    """
    Operational Transformation engine for collaborative editing.
    
    Implements OT algorithms to handle concurrent edits.
    """
    
    def __init__(self):
        self._operations: Dict[str, List[Operation]] = defaultdict(list)  # item_id -> operations
        self._revisions: Dict[str, int] = defaultdict(int)  # item_id -> current revision
        
    def transform(
        self, 
        op1: Operation, 
        op2: Operation
    ) -> Tuple[Operation, Operation]:
        """
        Transform two operations to maintain consistency.
        
        This implements the core OT algorithm.
        """
        # Simple transformation for insert-insert conflicts
        if op1.operation_type == "insert" and op2.operation_type == "insert":
            if op1.position < op2.position:
                # op1 comes first, op2 shifts
                new_op2 = Operation(
                    operation_id=op2.operation_id,
                    user_id=op2.user_id,
                    item_id=op2.item_id,
                    operation_type=op2.operation_type,
                    position=op2.position + (len(op1.content) if op1.content else 0),
                    content=op2.content,
                    length=op2.length,
                    revision=op2.revision
                )
                return op1, new_op2
            elif op1.position > op2.position:
                # op2 comes first, op1 shifts
                new_op1 = Operation(
                    operation_id=op1.operation_id,
                    user_id=op1.user_id,
                    item_id=op1.item_id,
                    operation_type=op1.operation_type,
                    position=op1.position + (len(op2.content) if op2.content else 0),
                    content=op1.content,
                    length=op1.length,
                    revision=op1.revision
                )
                return new_op1, op2
            else:
                # Same position, use user_id as tiebreaker
                if op1.user_id < op2.user_id:
                    new_op2 = Operation(
                        operation_id=op2.operation_id,
                        user_id=op2.user_id,
                        item_id=op2.item_id,
                        operation_type=op2.operation_type,
                        position=op2.position + (len(op1.content) if op1.content else 0),
                        content=op2.content,
                        length=op2.length,
                        revision=op2.revision
                    )
                    return op1, new_op2
                else:
                    new_op1 = Operation(
                        operation_id=op1.operation_id,
                        user_id=op1.user_id,
                        item_id=op1.item_id,
                        operation_type=op1.operation_type,
                        position=op1.position + (len(op2.content) if op2.content else 0),
                        content=op1.content,
                        length=op1.length,
                        revision=op1.revision
                    )
                    return new_op1, op2
        
        # For other combinations, return unchanged (simplified)
        return op1, op2
    
    def apply_operation(self, operation: Operation, content: str) -> str:
        """Apply an operation to content."""
        if operation.operation_type == "insert":
            return content[:operation.position] + (operation.content or "") + content[operation.position:]
        elif operation.operation_type == "delete":
            return content[:operation.position] + content[operation.position + operation.length:]
        elif operation.operation_type == "retain":
            return content  # No change
        return content
    
    def add_operation(self, operation: Operation):
        """Add an operation to the history."""
        self._operations[operation.item_id].append(operation)
        self._revisions[operation.item_id] = max(
            self._revisions[operation.item_id],
            operation.revision
        )
    
    def get_operations_since(self, item_id: str, revision: int) -> List[Operation]:
        """Get all operations since a given revision."""
        return [op for op in self._operations[item_id] if op.revision > revision]


class RealtimeCollaborationServer:
    """
    Main collaboration server that ties everything together.
    """
    
    def __init__(self):
        self.presence_manager = PresenceManager()
        self.lock_manager = LockManager()
        self.ot_engine = OperationalTransformation()
        
        # Connected clients
        self._clients: Dict[str, Any] = {}  # session_id -> client connection
        
        # Event callbacks
        self._event_callbacks: List[Callable[[CollaborationEvent], None]] = []
        
        self._running = False
        
    async def start(self):
        """Start the collaboration server."""
        self._running = True
        await self.presence_manager.start()
        await self.lock_manager.start()
        logger.info("RealtimeCollaborationServer started")
        
    async def stop(self):
        """Stop the collaboration server."""
        self._running = False
        await self.presence_manager.stop()
        await self.lock_manager.stop()
        logger.info("RealtimeCollaborationServer stopped")
        
    async def client_connected(
        self, 
        session_id: str, 
        user_id: str, 
        user_name: str,
        connection: Any
    ):
        """Handle a new client connection."""
        self._clients[session_id] = connection
        
        # Record presence
        presence = await self.presence_manager.user_joined(user_id, user_name, session_id)
        
        # Broadcast join event
        event = CollaborationEvent.create(
            CollaborationEventType.USER_JOINED,
            user_id,
            session_id,
            {"user_name": user_name, "presence": presence.to_dict()}
        )
        await self._broadcast_event(event, exclude_session=session_id)
        
        # Send current presence to new client
        all_presence = await self.presence_manager.get_all_presence()
        await self._send_to_client(session_id, {
            "type": "presence_list",
            "data": [p.to_dict() for p in all_presence if p.session_id != session_id]
        })
        
        logger.info(f"Client connected: {user_name} ({session_id})")
        
    async def client_disconnected(self, session_id: str):
        """Handle client disconnection."""
        presence = await self.presence_manager.user_left(session_id)
        self._clients.pop(session_id, None)
        
        if presence:
            # Release any locks held by this session
            # Note: In a real implementation, you'd track which locks are held
            
            # Broadcast leave event
            event = CollaborationEvent.create(
                CollaborationEventType.USER_LEFT,
                presence.user_id,
                session_id,
                {"user_name": presence.user_name}
            )
            await self._broadcast_event(event)
            
            logger.info(f"Client disconnected: {presence.user_name} ({session_id})")
        
    async def handle_message(self, session_id: str, message: Dict[str, Any]):
        """Handle a message from a client."""
        msg_type = message.get("type")
        
        if msg_type == "view_item":
            await self._handle_view_item(session_id, message)
        elif msg_type == "cursor_move":
            await self._handle_cursor_move(session_id, message)
        elif msg_type == "acquire_lock":
            await self._handle_acquire_lock(session_id, message)
        elif msg_type == "release_lock":
            await self._handle_release_lock(session_id, message)
        elif msg_type == "operation":
            await self._handle_operation(session_id, message)
        elif msg_type == "heartbeat":
            await self.presence_manager.update_activity(session_id)
        
    async def _handle_view_item(self, session_id: str, message: Dict[str, Any]):
        """Handle view item message."""
        item_id = message.get("item_id")
        await self.presence_manager.set_current_view(session_id, item_id)
        
        # Notify other viewers
        presence = await self.presence_manager.get_presence(session_id)
        if presence:
            viewers = await self.presence_manager.get_item_viewers(item_id)
            for viewer in viewers:
                if viewer.session_id != session_id:
                    await self._send_to_client(viewer.session_id, {
                        "type": "viewer_joined",
                        "data": {
                            "item_id": item_id,
                            "user": presence.to_dict()
                        }
                    })
        
    async def _handle_cursor_move(self, session_id: str, message: Dict[str, Any]):
        """Handle cursor movement."""
        position = message.get("position")
        await self.presence_manager.update_cursor(session_id, position)
        
        presence = await self.presence_manager.get_presence(session_id)
        if presence and presence.current_view:
            # Broadcast to other viewers of the same item
            viewers = await self.presence_manager.get_item_viewers(presence.current_view)
            event = CollaborationEvent.create(
                CollaborationEventType.CURSOR_POSITION,
                presence.user_id,
                session_id,
                {
                    "item_id": presence.current_view,
                    "position": position,
                    "user_name": presence.user_name
                }
            )
            for viewer in viewers:
                if viewer.session_id != session_id:
                    await self._send_to_client(viewer.session_id, event.to_dict())
        
    async def _handle_acquire_lock(self, session_id: str, message: Dict[str, Any]):
        """Handle lock acquisition request."""
        item_id = message.get("item_id")
        user_id = message.get("user_id")
        
        success, lock = await self.lock_manager.acquire_lock(item_id, user_id, session_id)
        
        await self._send_to_client(session_id, {
            "type": "lock_response",
            "data": {
                "item_id": item_id,
                "acquired": success,
                "lock": lock.to_dict() if lock else None
            }
        })
        
        if success:
            # Broadcast lock acquired
            event = CollaborationEvent.create(
                CollaborationEventType.LOCK_ACQUIRED,
                user_id,
                session_id,
                {"item_id": item_id, "lock": lock.to_dict()}
            )
            await self._broadcast_event(event)
        
    async def _handle_release_lock(self, session_id: str, message: Dict[str, Any]):
        """Handle lock release."""
        item_id = message.get("item_id")
        success = await self.lock_manager.release_lock(item_id, session_id)
        
        if success:
            presence = await self.presence_manager.get_presence(session_id)
            event = CollaborationEvent.create(
                CollaborationEventType.LOCK_RELEASED,
                presence.user_id if presence else "unknown",
                session_id,
                {"item_id": item_id}
            )
            await self._broadcast_event(event)
        
    async def _handle_operation(self, session_id: str, message: Dict[str, Any]):
        """Handle collaborative editing operation."""
        operation_data = message.get("operation", {})
        operation = Operation(
            operation_id=operation_data.get("operation_id", str(uuid.uuid4())),
            user_id=operation_data.get("user_id"),
            item_id=operation_data.get("item_id"),
            operation_type=operation_data.get("operation_type"),
            position=operation_data.get("position", 0),
            content=operation_data.get("content"),
            length=operation_data.get("length", 0)
        )
        
        # Check if user has lock for this item
        lock = await self.lock_manager.get_lock(operation.item_id)
        if lock and lock.session_id != session_id:
            # Reject operation - someone else has the lock
            await self._send_to_client(session_id, {
                "type": "operation_rejected",
                "data": {
                    "operation_id": operation.operation_id,
                    "reason": "item_locked",
                    "locked_by": lock.user_id
                }
            })
            return
        
        # Get operations since this client's revision
        client_revision = message.get("revision", 0)
        concurrent_ops = self.ot_engine.get_operations_since(
            operation.item_id, 
            client_revision
        )
        
        # Transform operation against concurrent operations
        transformed_op = operation
        for other_op in concurrent_ops:
            transformed_op, _ = self.ot_engine.transform(transformed_op, other_op)
        
        # Assign new revision
        transformed_op.revision = self.ot_engine._revisions[operation.item_id] + 1
        
        # Add to history
        self.ot_engine.add_operation(transformed_op)
        
        # Broadcast to other clients viewing the same item
        presence = await self.presence_manager.get_presence(session_id)
        if presence and presence.current_view:
            viewers = await self.presence_manager.get_item_viewers(presence.current_view)
            event = CollaborationEvent.create(
                CollaborationEventType.OPERATION_APPLIED,
                presence.user_id,
                session_id,
                {"operation": transformed_op.to_dict()}
            )
            for viewer in viewers:
                if viewer.session_id != session_id:
                    await self._send_to_client(viewer.session_id, event.to_dict())
        
        # Acknowledge to sender
        await self._send_to_client(session_id, {
            "type": "operation_ack",
            "data": {
                "operation_id": operation.operation_id,
                "revision": transformed_op.revision
            }
        })
        
    async def _send_to_client(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        # In a real implementation, this would use WebSocket
        # For now, we just log it
        logger.debug(f"Sending to {session_id}: {message}")
        
    async def _broadcast_event(
        self, 
        event: CollaborationEvent, 
        exclude_session: Optional[str] = None
    ):
        """Broadcast an event to all clients."""
        for session_id in self._clients:
            if session_id != exclude_session:
                await self._send_to_client(session_id, event.to_dict())
        
        # Notify registered callbacks
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
        
    def on_event(self, callback: Callable[[CollaborationEvent], None]):
        """Register an event callback."""
        self._event_callbacks.append(callback)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "connected_clients": len(self._clients),
            "presence": len(self.presence_manager._presence),
            "active_locks": len(self.lock_manager._locks)
        }


__all__ = [
    "CollaborationEventType",
    "CollaborationEvent",
    "UserPresence",
    "Operation",
    "ItemLock",
    "PresenceManager",
    "LockManager",
    "OperationalTransformation",
    "RealtimeCollaborationServer"
]
