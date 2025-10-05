import asyncio
import streamlit as st
import websockets
import json
import threading
from typing import Set, Dict, Any


class CollaborationServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.server = None
        self.users: Set[websockets.WebSocketServerProtocol] = set()
        self.user_info: Dict[websockets.WebSocketServerProtocol, Dict[str, Any]] = {}

    async def handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """
        Handle incoming websocket connections.
        """
        self.users.add(websocket)
        self.user_info[websocket] = {"id": str(id(websocket))}
        try:
            await self.broadcast_presence()
            async for message in websocket:
                data = json.loads(message)
                if data["type"] == "update_presence":
                    self.user_info[websocket].update(data["payload"])
                    await self.broadcast_presence()
                elif data["type"] == "share_config":
                    await self.broadcast_config(data["payload"])
                elif data["type"] == "share_results":
                    await self.broadcast_results(data["payload"])
                elif data["type"] == "cursor_update":
                    await self.broadcast_cursor(websocket, data["payload"])
                elif data["type"] == "text_update":
                    await self.broadcast_text(websocket, data["payload"])
        finally:
            self.users.remove(websocket)
            del self.user_info[websocket]
            await self.broadcast_presence()

    async def broadcast_presence(self):
        """
        Broadcast presence information to all connected users.
        """
        presence_data = {
            "type": "presence_update",
            "payload": list(self.user_info.values()),
        }
        message = json.dumps(presence_data)
        if self.users:
            await asyncio.wait([user.send(message) for user in self.users])

    async def broadcast_notification(self, payload: Dict[str, Any]):
        """
        Broadcast a notification to all connected users.
        """
        notification_data = {"type": "notification", "payload": payload}
        message = json.dumps(notification_data)
        if self.users:
            await asyncio.wait([user.send(message) for user in self.users])

    async def broadcast_config(self, payload: Dict[str, Any]):
        """
        Broadcast the evolution configuration to all connected users.
        """
        config_data = {"type": "config_update", "payload": payload}
        message = json.dumps(config_data)
        if self.users:
            await asyncio.wait([user.send(message) for user in self.users])

    async def broadcast_results(self, payload: Dict[str, Any]):
        """
        Broadcast the evolution results to all connected users.
        """
        results_data = {"type": "results_update", "payload": payload}
        message = json.dumps(results_data)
        if self.users:
            await asyncio.wait([user.send(message) for user in self.users])

    async def broadcast_comment(self, payload: Dict[str, Any]):
        """
        Broadcast a new comment to all connected users.
        """
        comment_data = {"type": "comment_added", "payload": payload}
        message = json.dumps(comment_data)
        if self.users:
            await asyncio.wait([user.send(message) for user in self.users])

    async def broadcast_cursor(
        self, sender: websockets.WebSocketServerProtocol, payload: Dict[str, Any]
    ):
        """
        Broadcast cursor position to other users.
        """
        cursor_data = {
            "type": "cursor_update",
            "payload": payload,
            "sender": self.user_info[sender]["id"],
        }
        message = json.dumps(cursor_data)
        for user in self.users:
            if user != sender:
                await user.send(message)

    async def broadcast_text(
        self, sender: websockets.WebSocketServerProtocol, payload: Dict[str, Any]
    ):
        """
        Broadcast text updates to other users.
        """
        text_data = {
            "type": "text_update",
            "payload": payload,
            "sender": self.user_info[sender]["id"],
        }
        message = json.dumps(text_data)
        for user in self.users:
            if user != sender:
                await user.send(message)

    def start(self):
        """
        Start the websocket server in a separate thread.
        """

        async def run_server():
            self.server = await websockets.serve(self.handler, self.host, self.port)
            await self.server.wait_closed()

        def run_loop():
            asyncio.run(run_server())

        thread = threading.Thread(target=run_loop)
        thread.daemon = True
        thread.start()
        print(f"Collaboration server started on ws://{self.host}:{self.port}")


def start_collaboration_server():
    if "collaboration_server_instance" not in st.session_state:
        # Check if the port is available before creating the server
        import socket
        port = 8765  # Default collaboration port
        
        # Test if port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(('localhost', port))
            if result == 0:  # Port is in use
                st.warning(f"Port {port} is already in use. Collaboration server may not start properly.")
        
        st.session_state.collaboration_server_instance = CollaborationServer(host="localhost", port=port)
        st.session_state.collaboration_server_started = False

    if not st.session_state.collaboration_server_started:
        try:
            st.session_state.collaboration_server_instance.start()
            st.session_state.collaboration_server_started = True
            print(f"Collaboration server started successfully on ws://localhost:{8765}")
        except Exception as e:
            st.error(f"Failed to start collaboration server: {e}")
            st.session_state.collaboration_server_started = False
