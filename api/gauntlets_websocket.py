"""
Gauntlet WebSocket API

WebSocket endpoint for real-time gauntlet execution updates and bidirectional communication.

Features:
- Real-time execution updates
- Bidirectional communication for interactive gauntlets
- Connection management and authentication
- Event streaming for progress updates
- Error handling and reconnection logic
- Integration with gauntlet server

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class EventType(Enum):
    """WebSocket event types"""
    EXECUTION_STARTED = "execution_started"
    ROUND_STARTED = "round_started"
    ROUND_COMPLETED = "round_completed"
    PROGRESS_UPDATE = "progress_update"
    EXECUTION_COMPLETED = "execution_completed"
    ERROR = "error"
    CONNECTION_ACK = "connection_ack"
    PING = "ping"
    PONG = "pong"


@dataclass
class WebSocketEvent:
    """
    Event sent over WebSocket connection.

    Attributes:
        event_type: Type of event
        data: Event data
        timestamp: When event was created
        execution_id: Associated execution ID (if any)
    """
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: time.time())
    execution_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON for transmission"""
        return json.dumps({
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "execution_id": self.execution_id
        })

    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketEvent":
        """Create from JSON"""
        data = json.loads(json_str)
        return cls(
            event_type=EventType(data["event_type"]),
            data=data["data"],
            timestamp=data.get("timestamp", time.time()),
            execution_id=data.get("execution_id")
        )


class ConnectionManager:
    """
    Manages WebSocket connections.

    Handles:
    - Connection registration/removal
    - Broadcasting to multiple clients
    - Sending to specific clients
    - Connection authentication
    """

    def __init__(self):
        """Initialize connection manager"""
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}
        self.connection_auth: Dict[str, str] = {}  # connection_id -> auth_token
        self.execution_subscriptions: Dict[str, Set[str]] = {}  # execution_id -> set of connection_ids

    async def connect(self, websocket: WebSocketServerProtocol, auth_token: Optional[str] = None) -> str:
        """
        Accept and register a new connection.

        Args:
            websocket: WebSocket connection
            auth_token: Optional authentication token

        Returns:
            Connection ID
        """
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket

        if auth_token:
            self.connection_auth[connection_id] = auth_token

        logger.info(f"WebSocket connection established: {connection_id}")

        # Send acknowledgment
        ack_event = WebSocketEvent(
            event_type=EventType.CONNECTION_ACK,
            data={"connection_id": connection_id}
        )
        await self.send_event(connection_id, ack_event)

        return connection_id

    def disconnect(self, connection_id: str):
        """Remove a connection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        if connection_id in self.connection_auth:
            del self.connection_auth[connection_id]

        # Remove from execution subscriptions
        for execution_id, subscribers in self.execution_subscriptions.items():
            subscribers.discard(connection_id)

        logger.info(f"WebSocket connection closed: {connection_id}")

    async def send_event(self, connection_id: str, event: WebSocketEvent):
        """
        Send event to specific connection.

        Args:
            connection_id: Target connection ID
            event: Event to send
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Connection not found: {connection_id}")
            return

        try:
            websocket = self.active_connections[connection_id]
            await websocket.send(event.to_json())
        except Exception as e:
            logger.error(f"Error sending event to {connection_id}: {e}")
            self.disconnect(connection_id)

    async def broadcast(self, event: WebSocketEvent):
        """Broadcast event to all active connections"""
        if not self.active_connections:
            return

        # Create tasks for all connections
        tasks = [
            self.send_event(conn_id, event)
            for conn_id in self.active_connections.keys()
        ]

        # Execute all sends concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def broadcast_to_execution(
        self,
        execution_id: str,
        event: WebSocketEvent
    ):
        """
        Broadcast event to all connections subscribed to an execution.

        Args:
            execution_id: Execution ID
            event: Event to broadcast
        """
        if execution_id not in self.execution_subscriptions:
            return

        subscribers = self.execution_subscriptions[execution_id]

        tasks = [
            self.send_event(conn_id, event)
            for conn_id in subscribers
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    def subscribe_to_execution(self, connection_id: str, execution_id: str):
        """Subscribe connection to execution updates"""
        if execution_id not in self.execution_subscriptions:
            self.execution_subscriptions[execution_id] = set()

        self.execution_subscriptions[execution_id].add(connection_id)
        logger.debug(f"Connection {connection_id} subscribed to execution {execution_id}")

    def unsubscribe_from_execution(self, connection_id: str, execution_id: str):
        """Unsubscribe connection from execution updates"""
        if execution_id in self.execution_subscriptions:
            self.execution_subscriptions[execution_id].discard(connection_id)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


class GauntletWebSocketServer:
    """
    WebSocket server for real-time gauntlet updates.

    Provides real-time streaming of gauntlet execution progress,
    results, and enables interactive gauntlet evaluation.

    Example:
        >>> server = GauntletWebSocketServer(host="localhost", port=8765)
        >>> await server.start()
        >>>
        >>> # Client connects and receives real-time updates
        >>> # Server broadcasts events during execution
        >>> await server.broadcast_execution_progress(execution_id, progress_data)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0
    ):
        """
        Initialize WebSocket server.

        Args:
            host: Host to bind to
            port: Port to bind to
            ping_interval: Seconds between ping frames
            ping_timeout: Seconds to wait for pong response
        """
        self.host = host
        self.port = port
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout

        self.manager = ConnectionManager()
        self.server = None

        logger.info(f"Gauntlet WebSocket Server configured: {host}:{port}")

    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")

        self.server = await websockets.serve(
            self.handle_connection,
            self.host,
            self.port,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout
        )

        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle new WebSocket connection.

        Args:
            websocket: WebSocket connection
            path: URL path
        """
        connection_id = await self.manager.connect(websocket)

        try:
            async for message in websocket:
                await self.handle_message(connection_id, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed by client: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling connection {connection_id}: {e}")
        finally:
            self.manager.disconnect(connection_id)

    async def handle_message(self, connection_id: str, message: str):
        """
        Handle incoming message from client.

        Args:
            connection_id: Connection ID
            message: Message JSON string
        """
        try:
            event = WebSocketEvent.from_json(message)

            # Handle different event types
            if event.event_type == EventType.PING:
                # Respond with pong
                pong_event = WebSocketEvent(
                    event_type=EventType.PONG,
                    data={"timestamp": time.time()}
                )
                await self.manager.send_event(connection_id, pong_event)

            elif event.event_type == EventType.EXECUTION_STARTED:
                # Subscribe to execution updates
                execution_id = event.execution_id or event.data.get("execution_id")
                if execution_id:
                    self.manager.subscribe_to_execution(connection_id, execution_id)

                    # Acknowledge subscription
                    ack_event = WebSocketEvent(
                        event_type=EventType.EXECUTION_STARTED,
                        data={"status": "subscribed"},
                        execution_id=execution_id
                    )
                    await self.manager.send_event(connection_id, ack_event)

        except Exception as e:
            logger.error(f"Error handling message from {connection_id}: {e}")

            # Send error event
            error_event = WebSocketEvent(
                event_type=EventType.ERROR,
                data={"error": str(e)}
            )
            await self.manager.send_event(connection_id, error_event)

    async def broadcast_execution_progress(
        self,
        execution_id: str,
        round_number: int,
        progress: float,
        status: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast execution progress to subscribed clients.

        Args:
            execution_id: Execution ID
            round_number: Current round number
            progress: Progress (0.0 to 1.0)
            status: Status message
            data: Additional data
        """
        event = WebSocketEvent(
            event_type=EventType.PROGRESS_UPDATE,
            data={
                "round_number": round_number,
                "progress": progress,
                "status": status,
                **(data or {})
            },
            execution_id=execution_id
        )

        await self.manager.broadcast_to_execution(execution_id, event)

    async def broadcast_round_completed(
        self,
        execution_id: str,
        round_number: int,
        passed: bool,
        score: float,
        feedback: str
    ):
        """
        Broadcast round completion to subscribed clients.

        Args:
            execution_id: Execution ID
            round_number: Completed round number
            passed: Whether round was passed
            score: Score achieved
            feedback: Feedback message
        """
        event = WebSocketEvent(
            event_type=EventType.ROUND_COMPLETED,
            data={
                "round_number": round_number,
                "passed": passed,
                "score": score,
                "feedback": feedback
            },
            execution_id=execution_id
        )

        await self.manager.broadcast_to_execution(execution_id, event)

    async def broadcast_execution_completed(
        self,
        execution_id: str,
        passed: bool,
        final_score: float,
        rounds_completed: int,
        total_time: float
    ):
        """
        Broadcast execution completion to subscribed clients.

        Args:
            execution_id: Execution ID
            passed: Whether gauntlet was passed
            final_score: Final score
            rounds_completed: Number of rounds completed
            total_time: Total execution time
        """
        event = WebSocketEvent(
            event_type=EventType.EXECUTION_COMPLETED,
            data={
                "passed": passed,
                "final_score": final_score,
                "rounds_completed": rounds_completed,
                "total_time": total_time
            },
            execution_id=execution_id
        )

        await self.manager.broadcast_to_execution(execution_id, event)

    async def broadcast_error(self, execution_id: str, error: str):
        """
        Broadcast error to subscribed clients.

        Args:
            execution_id: Execution ID
            error: Error message
        """
        event = WebSocketEvent(
            event_type=EventType.ERROR,
            data={"error": error},
            execution_id=execution_id
        )

        await self.manager.broadcast_to_execution(execution_id, event)


class GauntletWebSocketClient:
    """
    WebSocket client for connecting to gauntlet WebSocket server.

    Example:
        >>> client = GauntletWebSocketClient("ws://localhost:8765")
        >>> await client.connect()
        >>>
        >>> # Subscribe to execution updates
        >>> await client.subscribe_to_execution(execution_id)
        >>>
        >>> # Receive real-time updates
        >>> async for event in client.events():
        ...     print(f"Event: {event.event_type} - {event.data}")
    """

    def __init__(
        self,
        uri: str,
        reconnect: bool = True,
        reconnect_delay: float = 5.0
    ):
        """
        Initialize WebSocket client.

        Args:
            uri: WebSocket server URI
            reconnect: Whether to auto-reconnect
            reconnect_delay: Delay between reconnection attempts
        """
        self.uri = uri
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay

        self.websocket = None
        self.connection_id = None
        self.event_queue = asyncio.Queue()

    async def connect(self):
        """Connect to WebSocket server"""
        logger.info(f"Connecting to WebSocket server: {self.uri}")

        try:
            self.websocket = await websockets.connect(self.uri)
            logger.info("WebSocket connection established")

            # Start message handler
            asyncio.create_task(self._handle_messages())

            # Wait for connection acknowledgment
            ack_event = await self.event_queue.get()
            if ack_event.event_type == EventType.CONNECTION_ACK:
                self.connection_id = ack_event.data["connection_id"]
                logger.info(f"Connection acknowledged: {self.connection_id}")

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def disconnect(self):
        """Disconnect from WebSocket server"""
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")

    async def _handle_messages(self):
        """Handle incoming messages from server"""
        try:
            async for message in self.websocket:
                event = WebSocketEvent.from_json(message)
                await self.event_queue.put(event)

        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self.reconnect:
                logger.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                await asyncio.sleep(self.reconnect_delay)
                await self.connect()

    async def events(self):
        """Async generator for receiving events"""
        while True:
            event = await self.event_queue.get()
            yield event

    async def subscribe_to_execution(self, execution_id: str):
        """
        Subscribe to execution updates.

        Args:
            execution_id: Execution ID to subscribe to
        """
        event = WebSocketEvent(
            event_type=EventType.EXECUTION_STARTED,
            data={"execution_id": execution_id},
            execution_id=execution_id
        )

        await self.websocket.send(event.to_json())
        logger.info(f"Subscribed to execution: {execution_id}")

    async def send_ping(self):
        """Send ping to server"""
        event = WebSocketEvent(
            event_type=EventType.PING,
            data={}
        )

        await self.websocket.send(event.to_json())
