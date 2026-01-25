#!/usr/bin/env python3
"""
Test script to verify that the OpenEvolve BubbleLabs plugin fails gracefully.
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

async def test_plugin_initialization_error():
    """Test that plugin handles initialization errors gracefully."""
    print("Testing plugin initialization error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    # Create a mock integration that raises an exception during initialization
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize.side_effect = Exception("Simulated initialization failure")
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"error": "Not implemented"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # This should handle the error gracefully and not crash
        await plugin.initialize()
        
        # Check that the plugin is in an error state
        assert plugin._status.state == PluginState.ERROR
        assert plugin._status.health == "unhealthy"
        assert "initialization failure" in plugin._status.message.lower()
        
        print("✓ Initialization error handled gracefully")
        print(f"  Status: {plugin._status.state}")
        print(f"  Health: {plugin._status.health}")
        print(f"  Message: {plugin._status.message}")


async def test_plugin_start_error():
    """Test that plugin handles start errors gracefully."""
    print("\nTesting plugin start error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    # Create a mock integration that works during init but fails during start
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"error": "Not implemented"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Initialize successfully first
        await plugin.initialize()
        assert plugin._status.state == PluginState.INITIALIZED
        
        # Now simulate an error during start
        with patch.object(plugin, '_auto_cleanup_loop', side_effect=Exception("Simulated start failure")):
            await plugin.start()
            
            # Check that the plugin is in an error state
            assert plugin._status.state == PluginState.ERROR
            assert plugin._status.health == "unhealthy"
            assert "start failed" in plugin._status.message.lower()
            
            print("✓ Start error handled gracefully")
            print(f"  Status: {plugin._status.state}")
            print(f"  Health: {plugin._status.health}")
            print(f"  Message: {plugin._status.message}")


async def test_plugin_stop_error():
    """Test that plugin handles stop errors gracefully."""
    print("\nTesting plugin stop error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"error": "Not implemented"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Initialize and start successfully first
        await plugin.initialize()
        await plugin.start()
        assert plugin._status.state == PluginState.STARTED
        
        # Now simulate an error during stop
        with patch.object(plugin, '_cancel_all_workflows', side_effect=Exception("Simulated stop failure")):
            await plugin.stop()
            
            # Check that the plugin is in an error state
            assert plugin._status.state == PluginState.ERROR
            assert plugin._status.health == "unhealthy"
            assert "stop failed" in plugin._status.message.lower()
            
            print("✓ Stop error handled gracefully")
            print(f"  Status: {plugin._status.state}")
            print(f"  Health: {plugin._status.health}")
            print(f"  Message: {plugin._status.message}")


async def test_plugin_cleanup_error():
    """Test that plugin handles cleanup errors gracefully."""
    print("\nTesting plugin cleanup error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.side_effect = Exception("Simulated cleanup failure")
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"error": "Not implemented"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Initialize successfully first
        await plugin.initialize()
        assert plugin._status.state == PluginState.INITIALIZED
        
        # Now cleanup should handle the error gracefully
        await plugin.cleanup()
        
        # Check that the plugin is in an error state due to the cleanup failure
        assert plugin._status.state == PluginState.ERROR
        assert plugin._status.health == "unhealthy"
        assert "cleanup failed" in plugin._status.message.lower()
        
        print("✓ Cleanup error handled gracefully")
        print(f"  Status: {plugin._status.state}")
        print(f"  Health: {plugin._status.health}")
        print(f"  Message: {plugin._status.message}")


async def test_workflow_control_error():
    """Test that workflow control handles errors gracefully."""
    print("\nTesting workflow control error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        # Simulate an error in the control method
        mock_integration.control_workflow_local.side_effect = Exception("Simulated control failure")
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Initialize successfully
        await plugin.initialize()
        await plugin.start()
        
        # Control workflow should handle the error gracefully and return an error response
        result = await plugin.control_workflow("test-id", "start")
        
        assert "error" in result
        assert "control workflow" in result["error"].lower()
        assert result["status"] == "error"
        
        print("✓ Workflow control error handled gracefully")
        print(f"  Result: {result}")


async def main():
    """Run all tests."""
    print("Testing OpenEvolve BubbleLabs plugin graceful failure mechanisms...\n")
    
    await test_plugin_initialization_error()
    await test_plugin_start_error()
    await test_plugin_stop_error()
    await test_plugin_cleanup_error()
    await test_workflow_control_error()
    
    print("\n✓ All tests passed! The OpenEvolve BubbleLabs plugin handles errors gracefully.")


if __name__ == "__main__":
    asyncio.run(main())