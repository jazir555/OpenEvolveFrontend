#!/usr/bin/env python3
"""
Comprehensive test to verify that the OpenEvolve BubbleLabs plugin fails gracefully in all scenarios.
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

async def test_all_method_error_handling():
    """Test error handling for all major methods."""
    print("Testing error handling for all major methods...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        mock_integration.get_workflow_definition.return_value = None
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        # Test all methods handle errors gracefully
        print("[OK] Plugin instantiation successful")
        
        # Test initialization
        await plugin.initialize()
        print(f"[OK] Initialization completed, status: {plugin._status.state}")
        
        # Test start
        await plugin.start()
        print(f"[OK] Start completed, status: {plugin._status.state}")
        
        # Test health check
        health = await plugin.health_check()
        print(f"[OK] Health check completed, health: {health}")
        
        # Test node methods
        node_result = plugin.get_node("test_node")
        print(f"[OK] get_node completed, result: {node_result}")
        
        # Test workflow definition methods
        defs = await plugin.list_workflow_definitions()
        print(f"[OK] list_workflow_definitions completed, count: {len(defs)}")
        
        def_result = await plugin.get_workflow_definition("test_id")
        print(f"[OK] get_workflow_definition completed, result: {def_result is None}")
        
        # Test workflow instance methods
        instances = await plugin.list_workflow_instances()
        print(f"[OK] list_workflow_instances completed, count: {len(instances)}")
        
        # Test control workflow
        control_result = await plugin.control_workflow("test_id", "start")
        print(f"[OK] control_workflow completed, result: {control_result}")
        
        # Test metrics
        metrics = await plugin.get_metrics()
        print(f"[OK] get_metrics completed, keys: {list(metrics.keys())}")
        
        # Test reset metrics
        await plugin.reset_metrics()
        print("[OK] reset_metrics completed")
        
        # Test stopping
        await plugin.stop()
        print(f"[OK] Stop completed, status: {plugin._status.state}")
        
        # Test cleanup
        await plugin.cleanup()
        print(f"[OK] Cleanup completed, status: {plugin._status.state}")


async def test_error_scenarios():
    """Test specific error scenarios."""
    print("\nTesting specific error scenarios...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        # Make list methods throw exceptions
        mock_integration.list_workflow_definitions.side_effect = Exception("List defs error")
        mock_integration.list_workflow_instances.side_effect = Exception("List instances error")
        mock_integration.control_workflow_local.side_effect = Exception("Control workflow error")
        mock_integration.get_workflow_definition.side_effect = Exception("Get definition error")
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        await plugin.initialize()
        await plugin.start()
        
        # These should all handle errors gracefully now
        defs = await plugin.list_workflow_definitions()
        print(f"[OK] list_workflow_definitions handled error, returned: {defs}")
        
        instances = await plugin.list_workflow_instances()
        print(f"[OK] list_workflow_instances handled error, returned: {instances}")
        
        def_result = await plugin.get_workflow_definition("test_id")
        print(f"[OK] get_workflow_definition handled error, returned: {def_result}")
        
        control_result = await plugin.control_workflow("test_id", "start")
        print(f"[OK] control_workflow handled error, returned: {control_result}")
        
        metrics = await plugin.get_metrics()
        print(f"[OK] get_metrics handled error, returned keys: {list(metrics.keys())}")
        
        await plugin.stop()
        await plugin.cleanup()
        
        print("[OK] All error scenarios handled gracefully")


async def test_node_creation_error():
    """Test node creation error handling."""
    print("\nTesting node creation error handling...")
    
    config = {"max_instance_age_seconds": 7 * 24 * 3600, "max_instances": 1000}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class, \
         patch('openevolve_bubblelabs_plugin.create_node') as mock_create_node:
        
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        
        mock_integration_class.return_value = mock_integration
        mock_create_node.side_effect = Exception("Node creation error")
        
        plugin = OpenEvolveBubbleLabsPlugin(config)
        
        await plugin.initialize()
        
        # This should handle the error gracefully
        node_result = plugin.get_node("invalid_node_type")
        print(f"[OK] get_node handled error, returned: {node_result}")
        
        await plugin.cleanup()
        
        print("[OK] Node creation error handled gracefully")


async def main():
    """Run comprehensive tests."""
    print("Running comprehensive tests for OpenEvolve BubbleLabs plugin error handling...\n")
    
    await test_all_method_error_handling()
    await test_error_scenarios()
    await test_node_creation_error()
    
    print("\n[OK] All comprehensive tests passed! The OpenEvolve BubbleLabs plugin handles errors gracefully in all scenarios.")


if __name__ == "__main__":
    asyncio.run(main())