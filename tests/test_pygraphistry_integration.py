import asyncio
import unittest
import os
import sys
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations import IntegrationFactory
from integrations.pygraphistry.adapter import PygraphistryAdapter

class TestPygraphistryIntegration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.factory = IntegrationFactory()
        
    async def test_factory_get_visualization(self):
        """Test getting pygraphistry via factory."""
        viz = await self.factory.get_visualization("pygraphistry")
        self.assertIsNotNone(viz)
        self.assertIsInstance(viz, PygraphistryAdapter)
        
    async def test_adapter_initialization(self):
        """Test adapter initialization."""
        viz = await self.factory.get_visualization("pygraphistry")
        if viz:
            # Should be initialized by the factory
            self.assertTrue(viz.is_initialized)
            
    async def test_bridge_integration(self):
        """Test the high-level bridge."""
        from integrations.pygraphistry.bridge import PygraphistryBridge
        
        bridge = PygraphistryBridge()
        connected = await bridge.connect()
        # Even if pygraphistry is not installed, the bridge should handle it gracefully
        # but let's check if it attempts to connect
        self.assertIsNotNone(bridge.adapter)

if __name__ == "__main__":
    unittest.main()
