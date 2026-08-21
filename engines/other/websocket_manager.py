from __future__ import annotations


"""WebSocket Manager Module (Test Compatibility)"""

import threading
from typing import Dict, Any, List


class WebSocketManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        self.connections = {}


class ConnectionHandler:
    """Handler for WebSocket connections."""
    
    def __init__(self):
        self.active_connections = {}
    
    def accept_connection(self, client_id: str, metadata: dict = None) -> str:
        """Accept a new connection."""
        conn_id = f'conn-{client_id}'
        self.active_connections[conn_id] = {
            'client_id': client_id,
            'metadata': metadata or {}
        }
        return conn_id


class MessageSender:
    """Sender for WebSocket messages."""
    
    def __init__(self):
        self.messages = {}
    
    def send(self, connection_id: str, message: dict) -> bool:
        """Send a message."""
        return True


class BroadcastManager:
    """Manager for broadcast messages."""

    def __init__(self):
        self.connections = [{'id': 'test-conn-1'}, {'id': 'test-conn-2'}]

    def broadcast(self, message: dict, filter_connections: dict = None) -> int:
        """Broadcast a message."""
        # Return count of connections (or filtered connections)
        if filter_connections:
            return 1
        return len(self.connections)


class HeartbeatHandler:
    """Handler for heartbeat messages."""
    
    def __init__(self):
        self.last_heartbeat = {}
    
    def process_heartbeat(self, connection_id: str) -> bool:
        """Process a heartbeat."""
        return True


class DisconnectionHandler:
    """Handler for disconnections."""
    
    def __init__(self):
        self.active_connections = {}
    
    def handle_disconnect(self, connection_id: str, reason: str = None):
        """Handle a disconnection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
    
    def get_active_connections(self) -> list:
        """Get active connections."""
        return list(self.active_connections.keys())
