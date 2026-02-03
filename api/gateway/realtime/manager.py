"""
WebSocket Connection Manager
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Manager
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    WebSocket connection manager for real-time updates
    """

    def __init__(self):
        # Store active connections by room
        self.active_connections: Dict[str, Set[WebSocket]] = {}

        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, room: str, user_id: Optional[str] = None):
        """
        Connect a WebSocket to a room

        Args:
            websocket: WebSocket connection
            room: Room identifier (e.g., evolution_id, test_id)
            user_id: Optional user identifier
        """
        await websocket.accept()

        # Add room if it doesn't exist
        if room not in self.active_connections:
            self.active_connections[room] = set()

        # Add connection to room
        self.active_connections[room].add(websocket)

        # Store metadata
        self.connection_metadata[websocket] = {
            "room": room,
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
        }

        logger.info(f"WebSocket connected to room: {room}, user: {user_id}")

        # Send welcome message
        await websocket.send_json({
            "type": "connected",
            "data": {
                "room": room,
                "timestamp": datetime.utcnow().isoformat(),
            }
        })

    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket from its room

        Args:
            websocket: WebSocket connection
        """
        # Get metadata
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            room = metadata["room"]
            user_id = metadata.get("user_id")

            # Remove from room
            if room in self.active_connections:
                self.active_connections[room].discard(websocket)

                # Clean up empty rooms
                if not self.active_connections[room]:
                    del self.active_connections[room]
                    logger.info(f"Room {room} deleted (no connections)")

            # Remove metadata
            del self.connection_metadata[websocket]

            logger.info(f"WebSocket disconnected from room: {room}, user: {user_id}")

    async def broadcast(self, room: str, message: Dict[str, Any]):
        """
        Broadcast a message to all connections in a room

        Args:
            room: Room identifier
            message: Message to broadcast (will be converted to JSON)
        """
        if room not in self.active_connections:
            logger.warning(f"Cannot broadcast to room {room}: room not found")
            return

        # Prepare message with timestamp
        message["timestamp"] = datetime.utcnow().isoformat()

        # Send to all connections in room
        disconnected = set()
        for connection in self.active_connections[room]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to connection: {e}")
                disconnected.add(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            await self.disconnect(connection)

    async def send_personal(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send a message to a specific WebSocket connection

        Args:
            websocket: WebSocket connection
            message: Message to send
        """
        message["timestamp"] = datetime.utcnow().isoformat()
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)

    def get_room_connections(self, room: str) -> int:
        """Get the number of active connections in a room"""
        return len(self.active_connections.get(room, set()))

    def get_total_connections(self) -> int:
        """Get the total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())

    def get_room_users(self, room: str) -> Set[Optional[str]]:
        """Get all user IDs in a room"""
        users = set()
        if room in self.active_connections:
            for connection in self.active_connections[room]:
                metadata = self.connection_metadata.get(connection, {})
                user_id = metadata.get("user_id")
                if user_id:
                    users.add(user_id)
        return users

    def is_user_in_room(self, room: str, user_id: str) -> bool:
        """Check if a user is in a room"""
        return user_id in self.get_room_users(room)


# Global connection manager instance
manager = ConnectionManager()


class RoomManager:
    """
    Room-specific manager for different types of real-time updates
    """

    def __init__(self, room_type: str):
        self.room_type = room_type
        self.connection_manager = manager

    def get_room_id(self, resource_id: str) -> str:
        """Get the full room identifier"""
        return f"{self.room_type}:{resource_id}"

    async def broadcast_update(self, resource_id: str, update_type: str, data: Dict[str, Any]):
        """Broadcast an update to a resource's room"""
        room = self.get_room_id(resource_id)
        message = {
            "type": update_type,
            "data": data,
            "room": room,
        }
        await self.connection_manager.broadcast(room, message)

    async def broadcast_progress(self, resource_id: str, progress: float, status: str):
        """Broadcast progress update"""
        await self.broadcast_update(resource_id, "progress_update", {
            "resource_id": resource_id,
            "progress": progress,
            "status": status,
        })

    async def broadcast_complete(self, resource_id: str, result: Dict[str, Any]):
        """Broadcast completion status"""
        await self.broadcast_update(resource_id, "complete", {
            "resource_id": resource_id,
            "result": result,
        })

    async def broadcast_error(self, resource_id: str, error: str):
        """Broadcast error"""
        await self.broadcast_update(resource_id, "error", {
            "resource_id": resource_id,
            "error": error,
        })


class EvolutionRoomManager(RoomManager):
    """Manager for evolution progress rooms"""

    def __init__(self):
        super().__init__("evolution")

    async def broadcast_generation(self, evolution_id: str, generation: int, fitness: float):
        """Broadcast new generation data"""
        await self.broadcast_update(evolution_id, "generation_complete", {      
            "evolution_id": evolution_id,
            "generation": generation,
            "fitness": fitness,
        })

    async def broadcast_descendant_created(self, evolution_id: str, payload: Dict[str, Any]):
        """Broadcast a newly created descendant node"""
        await self.broadcast_update(evolution_id, "descendant_created", payload)

    async def broadcast_descendant_status(self, evolution_id: str, payload: Dict[str, Any]):
        """Broadcast a descendant status update (survived/killed)"""
        await self.broadcast_update(evolution_id, "descendant_status", payload)


class AdversarialRoomManager(RoomManager):
    """Manager for adversarial testing rooms"""

    def __init__(self):
        super().__init__("adversarial")

    async def broadcast_attack(self, test_id: str, round_num: int, attack: Dict[str, Any]):
        """Broadcast red team attack"""
        await self.broadcast_update(test_id, "attack_generated", {
            "test_id": test_id,
            "round": round_num,
            "attack": attack,
        })

    async def broadcast_patch(self, test_id: str, round_num: int, patch: Dict[str, Any]):
        """Broadcast blue team patch"""
        await self.broadcast_update(test_id, "patch_generated", {
            "test_id": test_id,
            "round": round_num,
            "patch": patch,
        })


class CollaborationRoomManager(RoomManager):
    """Manager for collaboration rooms"""

    def __init__(self):
        super().__init__("collaboration")

    async def broadcast_user_joined(self, room_id: str, user_id: str, username: str):
        """Broadcast user joined event"""
        await self.broadcast_update(room_id, "user_joined", {
            "room_id": room_id,
            "user_id": user_id,
            "username": username,
        })

    async def broadcast_user_left(self, room_id: str, user_id: str):
        """Broadcast user left event"""
        await self.broadcast_update(room_id, "user_left", {
            "room_id": room_id,
            "user_id": user_id,
        })

    async def broadcast_content_update(self, room_id: str, user_id: str, content: str):
        """Broadcast content update"""
        await self.broadcast_update(room_id, "content_update", {
            "room_id": room_id,
            "user_id": user_id,
            "content": content,
        })

    async def broadcast_cursor_update(self, room_id: str, user_id: str, position: Dict[str, int]):
        """Broadcast cursor position update"""
        await self.broadcast_update(room_id, "cursor_update", {
            "room_id": room_id,
            "user_id": user_id,
            "position": position,
        })
