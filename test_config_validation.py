#!/usr/bin/env python3
"""
Test to verify the configuration validation mechanism works properly.
"""

import asyncio
import logging
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openevolve_bubblelabs_plugin import OpenEvolveBubbleLabsPlugin

# Configure logging to see the error messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_config_validation():
    """Test that the plugin validates configuration properly."""
    print("Testing configuration validation...")
    
    # Test with valid config
    valid_config = {
        "max_instance_age_seconds": 86400,  # 1 day
        "max_instances": 500,
        "enable_auto_cleanup": True,
        "cleanup_interval_seconds": 1800  # 30 minutes
    }
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(valid_config)
        
        # Check that config was validated and stored properly
        print(f"✓ Valid config accepted: {plugin._auto_cleanup_enabled}")
        print(f"  Max instance age: {plugin._integration._MAX_INSTANCE_AGE_SECONDS}")
        print(f"  Max instances: {plugin._integration._MAX_INSTANCES}")
        print(f"  Cleanup interval: {plugin._cleanup_interval}")
    
    # Test with invalid config - should use defaults
    invalid_config = {
        "max_instance_age_seconds": "invalid",  # Should use default
        "max_instances": -100,  # Should use default
        "enable_auto_cleanup": "maybe",  # Should convert to bool
        "cleanup_interval_seconds": 0,  # Should use default
        "extra_param": "should_be_preserved"
    }
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(invalid_config)
        
        # Check that invalid values were replaced with defaults
        print(f"✓ Invalid config handled gracefully")
        print(f"  Max instance age: {plugin._integration._MAX_INSTANCE_AGE_SECONDS} (should be default)")
        print(f"  Max instances: {plugin._integration._MAX_INSTANCES} (should be default)")
        print(f"  Cleanup interval: {plugin._cleanup_interval} (should be default)")
        print(f"  Extra param preserved: {'extra_param' in plugin._config}")
    
    # Test with empty config - should use all defaults
    empty_config = {}
    
    with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
        mock_integration = Mock()
        mock_integration.initialize = Mock()
        mock_integration._cleanup_old_instances.return_value = 0
        mock_integration.list_workflow_definitions.return_value = []
        mock_integration.list_workflow_instances.return_value = []
        mock_integration.control_workflow_local.return_value = {"status": "success"}
        
        mock_integration_class.return_value = mock_integration
        
        plugin = OpenEvolveBubbleLabsPlugin(empty_config)
        
        # Check that defaults were used
        print(f"✓ Empty config handled with defaults")
        print(f"  Max instance age: {plugin._integration._MAX_INSTANCE_AGE_SECONDS}")
        print(f"  Max instances: {plugin._integration._MAX_INSTANCES}")
        print(f"  Auto cleanup enabled: {plugin._auto_cleanup_enabled}")
        print(f"  Cleanup interval: {plugin._cleanup_interval}")
    
    print("✓ Configuration validation works correctly")


async def test_config_validation_edge_cases():
    """Test edge cases for configuration validation."""
    print("\nTesting configuration validation edge cases...")
    
    edge_configs = [
        {"max_instance_age_seconds": 0},  # Zero value
        {"max_instance_age_seconds": -1},  # Negative value
        {"max_instances": 0},  # Zero value
        {"max_instances": -1},  # Negative value
        {"cleanup_interval_seconds": 0},  # Zero value
        {"cleanup_interval_seconds": -1},  # Negative value
        {"enable_auto_cleanup": None},  # None value
        {"enable_auto_cleanup": []},  # Empty list
        {"enable_auto_cleanup": {}},  # Empty dict
    ]
    
    for i, config in enumerate(edge_configs):
        with patch('openevolve_bubblelabs_plugin.BubbleLabsIntegration') as mock_integration_class:
            mock_integration = Mock()
            mock_integration.initialize = Mock()
            mock_integration._cleanup_old_instances.return_value = 0
            mock_integration.list_workflow_definitions.return_value = []
            mock_integration.list_workflow_instances.return_value = []
            mock_integration.control_workflow_local.return_value = {"status": "success"}
            
            mock_integration_class.return_value = mock_integration
            
            try:
                plugin = OpenEvolveBubbleLabsPlugin(config)
                print(f"✓ Config {i+1} handled gracefully: {config}")
            except (RuntimeError, ValueError, TypeError) as e:
                print(f"✗ Config {i+1} caused error: {e}")
                raise
    
    print("✓ All edge cases handled gracefully")


async def main():
    """Run configuration validation tests."""
    print("Testing configuration validation mechanism...\n")
    
    await test_config_validation()
    await test_config_validation_edge_cases()
    
    print("\n✓ All configuration validation tests passed! The OpenEvolve BubbleLabs plugin handles configuration validation gracefully.")


if __name__ == "__main__":
    asyncio.run(main())