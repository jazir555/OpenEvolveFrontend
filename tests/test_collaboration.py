import unittest
import pytest
from collaboration import CollaborationServer, WEBSOCKETS_AVAILABLE

pytestmark = pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets not available")


class TestCollaboration(unittest.TestCase):
    def setUp(self):
        if not WEBSOCKETS_AVAILABLE:
            self.skipTest("websockets not available")
        self.server = CollaborationServer()

    def tearDown(self):
        if self.server and self.server.server:
            self.server.server.close()

    @pytest.mark.asyncio
    async def test_broadcast(self):
        import websockets
        import json

        await self.server.start()

        try:
            async with websockets.connect(
                f"ws://{self.server.host}:{self.server.port}"
            ) as websocket1:
                async with websockets.connect(
                    f"ws://{self.server.host}:{self.server.port}"
                ) as websocket2:
                    test_message = {"type": "test", "payload": "hello"}
                    await websocket1.send(json.dumps(test_message))
                    response = await websocket2.recv()
                    self.assertEqual(json.loads(response), test_message)
        finally:
            await self.server.stop()


if __name__ == "__main__":
    unittest.main()
