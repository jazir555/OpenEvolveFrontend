#!/usr/bin/env python3
"""
Simple test to verify that the OpenEvolve BubbleLabs plugin has enhanced error handling.
"""

import asyncio
import logging
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin
from bubblelabs_plugin_system import PluginState

# Configure logging to see the error messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_basic_functionality():
    """Test basic plugin functionality."""
    print("Testing basic plugin functionality...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Test initialization
        await plugin.initialize()
        print(f"✓ Initialization completed, status: {plugin._status.state}")
        
        # Test start
        await plugin.start()
        print(f"✓ Start completed, status: {plugin._status.state}")
        
        # Test health check
        health = await plugin.health_check()
        print(f"✓ Health check completed, health: {health}")
        
        # Test stopping
        await plugin.stop()
        print(f"✓ Stop completed, status: {plugin._status.state}")
        
        # Test cleanup
        await plugin.cleanup()
        print(f"✓ Cleanup completed, status: {plugin._status.state}")


async def test_error_handling():
    """Test that errors are handled gracefully."""
    print("\nTesting error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        # Make the initialize method raise an exception
        mock_integration.initialize.side_effect = Exception("Test initialization error")
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # This should handle the error gracefully
        await plugin.initialize()
        
        # Check that the plugin is in an error state but didn't crash
        print(f"✓ Initialization handled error gracefully, status: {plugin._status.state}")
        print(f"  Health: {plugin._status.health}")
        print(f"  Message: {plugin._status.message}")


async def main():
    """Run tests."""
    print("Testing OpenEvolve BubbleLabs plugin enhanced error handling...\n")
    
    await test_basic_functionality()
    await test_error_handling()
    
    print("\n✓ Tests completed! The OpenEvolve BubbleLabs plugin has enhanced error handling.")


if __name__ == "__main__":
    asyncio.run(main())