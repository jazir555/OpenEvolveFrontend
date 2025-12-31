import unittest
import websockets
import json
from collaboration import CollaborationServer


class TestCollaboration(unittest.TestCase):
    def setUp(self):
        self.server = CollaborationServer()
        self.server.start()

    def tearDown(self):
        self.server.server.close()

    async def test_broadcast(self):
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


if __name__ == "__main__":
    unittest.main()
