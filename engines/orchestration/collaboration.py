from __future__ import annotations

import asyncio
from ui_shim import ui as st
from typing import Set, Dict, Any
import socket  # Added this import
import threading # Added this import


# Optional imports with fallbacks
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False

import json


class CollaborationServer:
    def __init__(self, host="localhost", port=8765):
        if not WEBSOCKETS_AVAILABLE:
            self.host = host
            self.port = port
            self.server = None
            print("WebSockets not available - collaboration features disabled")
            return
            
        self.host = host
        self.port = port
        self.server = None
        self.users: Set[websockets.WebSocketServerProtocol] = set()
        self.user_info: Dict[websockets.WebSocketServerProtocol, Dict[str, Any]] = {}

    async def handler(self, websocket, path: str):
        if not WEBSOCKETS_AVAILABLE:
            return
            
        """
        Handle incoming websocket connections.
        """
        self.users.add(websocket)
        self.user_info[websocket] = {"id": str(id(websocket))}
        try:
            await self.broadcast_presence()
            async for message in websocket:
                data = json.loads(message)
                # Use .get() for safe dictionary access
                msg_type = data.get("type")
                if msg_type == "update_presence":
                    self.user_info[websocket].update(data.get("payload", {}))
                    await self.broadcast_presence()
                elif msg_type == "share_config":
                    await self.broadcast_config(data.get("payload", {}))
                elif msg_type == "share_results":
                    await self.broadcast_results(data.get("payload", {}))
                elif msg_type == "cursor_update":
                    await self.broadcast_cursor(websocket, data.get("payload", {}))
                elif msg_type == "text_update":
                    await self.broadcast_text(websocket, data.get("payload", {}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.users.discard(websocket)
            if websocket in self.user_info:
                del self.user_info[websocket]
            await self.broadcast_presence()

    async def broadcast_presence(self):
        """Broadcast user presence updates to all connected clients."""
        if not WEBSOCKETS_AVAILABLE:
            return
        presence_data = {
            "type": "presence",
            "payload": {
                "users": [
                    {"id": info["id"]} 
                    for info in self.user_info.values()
                ]
            }
        }
        await self._broadcast(presence_data)

    async def broadcast_config(self, config: Dict[str, Any]):
        """Broadcast configuration updates."""
        if not WEBSOCKETS_AVAILABLE:
            return
        await self._broadcast({"type": "config", "payload": config})

    async def broadcast_results(self, results: Dict[str, Any]):
        """Broadcast results to all users."""
        if not WEBSOCKETS_AVAILABLE:
            return
        await self._broadcast({"type": "results", "payload": results})

    async def broadcast_cursor(self, sender, cursor_data: Dict[str, Any]):
        """Broadcast cursor position to other users."""
        if not WEBSOCKETS_AVAILABLE:
            return
        message = {
            "type": "cursor",
            "payload": {**cursor_data, "user_id": self.user_info.get(sender, {}).get("id")}
        }
        await self._broadcast(message, exclude=sender)

    async def broadcast_text(self, sender, text_data: Dict[str, Any]):
        """Broadcast text updates to other users."""
        if not WEBSOCKETS_AVAILABLE:
            return
        message = {
            "type": "text",
            "payload": {**text_data, "user_id": self.user_info.get(sender, {}).get("id")}
        }
        await self._broadcast(message, exclude=sender)

    async def _broadcast(self, message: Dict[str, Any], exclude=None):
        """Send message to all connected users."""
        if not WEBSOCKETS_AVAILABLE:
            return
        disconnected = []
        for user in self.users:
            if user != exclude:
                try:
                    await user.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(user)
        
        # Clean up disconnected users
        for user in disconnected:
            self.users.discard(user)
            if user in self.user_info:
                del self.user_info[user]

    async def start(self):
        """Start the collaboration server."""
        if not WEBSOCKETS_AVAILABLE:
            print("WebSockets not available - cannot start collaboration server")
            return
            
        print(f"Starting collaboration server on {self.host}:{self.port}")
        self.server = await websockets.serve(self.handler, self.host, self.port)
        print(f"Collaboration server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the collaboration server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            print("Collaboration server stopped")

    def run_in_thread(self):
        """Run the server in a background thread."""
        if not WEBSOCKETS_AVAILABLE:
            print("WebSockets not available - cannot run collaboration server")
            return
            
        def run_server():
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.start())
            loop.run_forever()
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print(f"Collaboration server started in background thread")


def get_local_ip():
    """Get the local IP address for display."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def display_collaboration_panel():
    """Display the collaboration panel in UI."""
    st.subheader("Real-time Collaboration")
    
    if not WEBSOCKETS_AVAILABLE:
        st.warning("WebSockets not available. Install with: pip install websockets")
        return
    
    local_ip = get_local_ip()
    
    st.info(f"""
    **Collaboration Server**
    - Local: ws://localhost:8765
    - Network: ws://{local_ip}:8765
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Server"):
            st.session_state.collaboration_server = CollaborationServer()
            st.session_state.collaboration_server.run_in_thread()
            st.success("Collaboration server started!")
    
    with col2:
        if st.button("Stop Server"):
            if hasattr(st.session_state, 'collaboration_server'):
                # Note: Proper async cleanup would require more complexity
                st.info("Server stopping... (restart app for clean state)")
            else:
                st.warning("No server running")


# Alias for tests
Collaboration = CollaborationServer

