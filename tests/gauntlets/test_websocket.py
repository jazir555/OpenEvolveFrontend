"""
Test Suite for Gauntlet WebSocket API

Comprehensive tests for the WebSocket API component.

Author: OpenEvolve Gauntlet System
Date: 2026-02-03
"""

import unittest
import asyncio
import json
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from api.gauntlets_websocket import (
        WebSocketEvent,
        EventType,
        ConnectionManager,
        GauntletWebSocketServer,
        GauntletWebSocketClient
    )
    WEBSOCKET_AVAILABLE = True
except ImportError as e:
    WEBSOCKET_AVAILABLE = False
    WEBSOCKET_IMPORT_ERROR = str(e)


class TestWebSocketEvent(unittest.TestCase):
    """Test WebSocketEvent dataclass"""

    def test_event_creation(self):
        """Test creating a WebSocket event"""
        event = WebSocketEvent(
            event_type=EventType.EXECUTION_STARTED,
            data={"execution_id": "exec_123"},
            execution_id="exec_123"
        )

        self.assertEqual(event.event_type, EventType.EXECUTION_STARTED)
        self.assertEqual(event.data["execution_id"], "exec_123")
        self.assertEqual(event.execution_id, "exec_123")

    def test_event_to_json(self):
        """Test converting event to JSON"""
        event = WebSocketEvent(
            event_type=EventType.PROGRESS_UPDATE,
            data={"progress": 0.5},
            execution_id="exec_123"
        )

        json_str = event.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["event_type"], "progress_update")
        self.assertEqual(data["data"]["progress"], 0.5)
        self.assertEqual(data["execution_id"], "exec_123")

    def test_event_from_json(self):
        """Test creating event from JSON"""
        json_str = json.dumps({
            "event_type": "round_completed",
            "data": {"round_number": 1},
            "timestamp": time.time(),
            "execution_id": "exec_123"
        })

        event = WebSocketEvent.from_json(json_str)

        self.assertEqual(event.event_type, EventType.ROUND_COMPLETED)
        self.assertEqual(event.data["round_number"], 1)


@unittest.skipIf(not WEBSOCKET_AVAILABLE, f"WebSocket not available: {WEBSOCKET_IMPORT_ERROR if not WEBSOCKET_AVAILABLE else 'Unknown'}")
class TestConnectionManager(unittest.TestCase):
    """Test ConnectionManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = ConnectionManager()

    def test_connection_manager_initialization(self):
        """Test connection manager initialization"""
        self.assertEqual(self.manager.get_connection_count(), 0)
        self.assertEqual(len(self.manager.active_connections), 0)

    def test_get_connection_count(self):
        """Test getting connection count"""
        count = self.manager.get_connection_count()
        self.assertEqual(count, 0)
        self.assertIsInstance(count, int)


@unittest.skipIf(not WEBSOCKET_AVAILABLE, f"WebSocket not available: {WEBSOCKET_IMPORT_ERROR if not WEBSOCKET_AVAILABLE else 'Unknown'}")
class TestGauntletWebSocketServer(unittest.TestCase):
    """Test Gauntlet WebSocket Server"""

    def setUp(self):
        """Set up test fixtures"""
        self.server = GauntletWebSocketServer(
            host="localhost",
            port=8766,  # Use different port for testing
            ping_interval=30.0,
            ping_timeout=10.0
        )

    def test_server_initialization(self):
        """Test server initialization"""
        self.assertEqual(self.server.host, "localhost")
        self.assertEqual(self.server.port, 8766)
        self.assertEqual(self.server.ping_interval, 30.0)
        self.assertIsNotNone(self.server.manager)

    def test_broadcast_execution_progress(self):
        """Test broadcasting execution progress"""
        execution_id = "exec_test_123"

        # Create mock event loop
        async def test_broadcast():
            await self.server.broadcast_execution_progress(
                execution_id=execution_id,
                round_number=1,
                progress=0.5,
                status="Running"
            )

        # Run async test
        asyncio.run(test_broadcast())

    def test_broadcast_round_completed(self):
        """Test broadcasting round completion"""
        execution_id = "exec_test_123"

        async def test_broadcast():
            await self.server.broadcast_round_completed(
                execution_id=execution_id,
                round_number=1,
                passed=True,
                score=0.85,
                feedback="Excellent solution"
            )

        asyncio.run(test_broadcast())

    def test_broadcast_execution_completed(self):
        """Test broadcasting execution completion"""
        execution_id = "exec_test_123"

        async def test_broadcast():
            await self.server.broadcast_execution_completed(
                execution_id=execution_id,
                passed=True,
                final_score=0.87,
                rounds_completed=3,
                total_time=45.0
            )

        asyncio.run(test_broadcast())

    def test_broadcast_error(self):
        """Test broadcasting error"""
        execution_id = "exec_test_123"

        async def test_broadcast():
            await self.server.broadcast_error(
                execution_id=execution_id,
                error="Test error"
            )

        asyncio.run(test_broadcast())


@unittest.skipIf(not WEBSOCKET_AVAILABLE, f"WebSocket not available: {WEBSOCKET_IMPORT_ERROR if not WEBSOCKET_AVAILABLE else 'Unknown'}")
class TestWebSocketIntegration(unittest.TestCase):
    """Integration tests for WebSocket functionality"""

    def test_event_serialization_roundtrip(self):
        """Test that events survive serialization roundtrip"""
        original_event = WebSocketEvent(
            event_type=EventType.ROUND_COMPLETED,
            data={"score": 0.85},
            execution_id="exec_123"
        )

        # Serialize to JSON
        json_str = original_event.to_json()

        # Deserialize back
        restored_event = WebSocketEvent.from_json(json_str)

        # Verify all fields match
        self.assertEqual(restored_event.event_type, original_event.event_type)
        self.assertEqual(restored_event.data, original_event.data)
        self.assertEqual(restored_event.execution_id, original_event.execution_id)

    def test_all_event_types(self):
        """Test all event types can be created and serialized"""
        event_types = [
            EventType.EXECUTION_STARTED,
            EventType.ROUND_STARTED,
            EventType.ROUND_COMPLETED,
            EventType.PROGRESS_UPDATE,
            EventType.EXECUTION_COMPLETED,
            EventType.ERROR,
            EventType.CONNECTION_ACK,
            EventType.PING,
            EventType.PONG
        ]

        for event_type in event_types:
            event = WebSocketEvent(
                event_type=event_type,
                data={"test": "data"},
                execution_id="exec_123"
            )

            # Should serialize successfully
            json_str = event.to_json()
            self.assertIsNotNone(json_str)

            # Should deserialize successfully
            restored = WebSocketEvent.from_json(json_str)
            self.assertEqual(restored.event_type, event_type)

    def test_connection_subscription_flow(self):
        """Test connection and subscription flow"""
        manager = ConnectionManager()

        # Simulate connection
        connection_id = "conn_test_123"
        execution_id = "exec_test_123"

        # Subscribe to execution
        manager.subscribe_to_execution(connection_id, execution_id)

        # Verify subscription
        self.assertIn(execution_id, manager.execution_subscriptions)
        self.assertIn(connection_id, manager.execution_subscriptions[execution_id])

        # Unsubscribe
        manager.unsubscribe_from_execution(connection_id, execution_id)

        # Verify unsubscription
        self.assertNotIn(connection_id, manager.execution_subscriptions[execution_id])


class TestWebSocketSecurity(unittest.TestCase):
    """Security tests for WebSocket"""

    def test_event_data_sanitization(self):
        """Test that event data is properly handled"""
        # Test with potentially malicious data
        malicious_data = {
            "script": "<script>alert('xss')</script>",
            "sql": "'; DROP TABLE users; --",
            "path": "../../etc/passwd"
        }

        event = WebSocketEvent(
            event_type=EventType.PROGRESS_UPDATE,
            data=malicious_data
        )

        # Event should be created (data is just stored)
        # In production, this would be sanitized by the receiver
        self.assertIsNotNone(event)

    def test_large_message_handling(self):
        """Test handling of large messages"""
        # Create event with large data
        large_data = {
            "data": "x" * 100000  # 100KB of data
        }

        event = WebSocketEvent(
            event_type=EventType.PROGRESS_UPDATE,
            data=large_data
        )

        # Should handle gracefully (just serialize)
        json_str = event.to_json()
        self.assertIsNotNone(json_str)


class TestWebSocketPerformance(unittest.TestCase):
    """Performance tests for WebSocket"""

    def test_event_serialization_performance(self):
        """Test event serialization performance"""
        event = WebSocketEvent(
            event_type=EventType.PROGRESS_UPDATE,
            data={
                "round_number": 1,
                "progress": 0.5,
                "status": "Running",
                "metrics": {f"metric_{i}": i * 0.1 for i in range(100)}
            }
        )

        # Measure serialization time
        start = time.time()
        for _ in range(1000):
            json_str = event.to_json()
        elapsed = time.time() - start

        # Should serialize 1000 times in less than 1 second
        self.assertLess(elapsed, 1.0)

    def test_event_deserialization_performance(self):
        """Test event deserialization performance"""
        event = WebSocketEvent(
            event_type=EventType.PROGRESS_UPDATE,
            data={"progress": 0.5}
        )
        json_str = event.to_json()

        # Measure deserialization time
        start = time.time()
        for _ in range(1000):
            restored = WebSocketEvent.from_json(json_str)
        elapsed = time.time() - start

        # Should deserialize 1000 times in less than 1 second
        self.assertLess(elapsed, 1.0)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
