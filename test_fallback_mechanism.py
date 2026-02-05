#!/usr/bin/env python3
"""
Test to verify the fallback integration mechanism works properly.
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

async def test_fallback_integration():
    """Test that the plugin creates a fallback integration when the real one fails."""
    print("Testing fallback integration mechanism...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    # Patch the BubbleLabsIntegration constructor to raise an exception
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration_class.side_effect = Exception("Simulated integration failure")
        
        # Create the plugin - this should trigger fallback mechanism
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Check that fallback integration was created
        print(f"[OK] Plugin created successfully despite integration failure")
        print(f"  Status: {plugin._status.state}")
        print(f"  Health: {plugin._status.health}")
        print(f"  Message: {plugin._status.message}")
        
        # Verify the plugin is in error state but still functional
        assert plugin._status.state == PluginState.ERROR
        assert plugin._status.health == "degraded"
        
        # Test that basic operations work with fallback integration
        try:
            defs = await plugin.list_workflow_definitions()
            print(f"[OK] list_workflow_definitions worked with fallback: {len(defs)} definitions")
            
            instances = await plugin.list_workflow_instances()
            print(f"[OK] list_workflow_instances worked with fallback: {len(instances)} instances")
            
            result = await plugin.control_workflow("test", "start")
            print(f"[OK] control_workflow worked with fallback: {result}")
            
            def_result = await plugin.get_workflow_definition("test")
            print(f"[OK] get_workflow_definition worked with fallback: {def_result}")
            
        except (RuntimeError, AttributeError, TypeError) as e:
            print(f"[FAIL] Error during fallback operations: {e}")
            raise
    
    print("[OK] Fallback integration mechanism works correctly")


async def test_normal_integration():
    """Test that the plugin works normally when integration succeeds."""
    print("\nTesting normal integration operation...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        mock_integration.get_workflow_definition.return_value = None
        mock_integration.create_workflow_definition_from_openevolve.return_value = None
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Check that normal integration was created
        print(f"[OK] Normal plugin created successfully")
        print(f"  Status: {plugin._status.state}")
        
        # Verify the plugin is in loaded state
        assert plugin._status.state != PluginState.ERROR
        
        # Test that basic operations work
        try:
            defs = await plugin.list_workflow_definitions()
            print(f"[OK] list_workflow_definitions worked: {len(defs)} definitions")
            
            instances = await plugin.list_workflow_instances()
            print(f"[OK] list_workflow_instances worked: {len(instances)} instances")
            
            result = await plugin.control_workflow("test", "start")
            print(f"[OK] control_workflow worked: {result}")
            
        except (RuntimeError, AttributeError, TypeError) as e:
            print(f"[FAIL] Error during normal operations: {e}")
            raise
    
    print("[OK] Normal integration operation works correctly")


async def main():
    """Run fallback mechanism tests."""
    print("Testing fallback integration mechanism...\n")
    
    await test_fallback_integration()
    await test_normal_integration()
    
    print("\n[OK] All fallback mechanism tests passed! The OpenEvolve BubbleLabs plugin handles both normal and fallback scenarios gracefully.")


if __name__ == "__main__":
    asyncio.run(main())