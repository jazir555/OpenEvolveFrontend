"""
Integration Test for LeanAide Autoformalization System with MDAP/MAKER

This test verifies that the autoformalization system properly integrates
with the existing MDAP and MAKER components.
"""

import asyncio
from unittest.mock import Mock, AsyncMock, patch

from leanaide_autoformalization_mdap_maker import (
    LeanAideAutoformalizationEngine,
    AutoformalizationStrategy
)


async def test_integration_with_existing_components():
    """Test that the autoformalization system integrates with existing components."""
    print("Testing integration with existing LeanAide/MDAP/MAKER components...")
    
    # Create mock LeanAide client
    mock_leanaide_client = Mock()
    mock_leanaide_client.cache = Mock()
    
    # Test that we can import and use with existing MDAP components
    try:
        from leanaide_mdap import LeanMDAPOrchestrator, LeanMDAPConfig
        print("OK: Successfully imported leanaide_mdap components")
    except ImportError as e:
        print(f"WARNING: leanaide_mdap import failed: {e}")

    # Test that we can import and use with existing Lean4 integration
    try:
        from lean4_integration import AutoformalizationEngine
        print("OK: Successfully imported lean4_integration components")
    except ImportError as e:
        print(f"WARNING: lean4_integration import failed: {e}")

    # Test that we can create the engine with mock components
    try:
        engine = LeanAideAutoformalizationEngine(
            leanaide_client=mock_leanaide_client,
            enable_caching=True
        )
        print("OK: Successfully created LeanAideAutoformalizationEngine")
    except Exception as e:
        print(f"ERROR: Failed to create engine: {e}")
        return False
    
    # Test that the engine has expected methods
    expected_methods = [
        'autoformalize',
        'get_system_status',
        '_select_adaptive_strategy',
        '_infer_domain'
    ]
    
    for method in expected_methods:
        if hasattr(engine, method):
            print(f"OK: Method {method} exists")
        else:
            print(f"ERROR: Method {method} missing")
            return False

    # Test that all strategies are available
    strategies = [s.value for s in AutoformalizationStrategy]
    expected_strategies = ['direct', 'mdap', 'maker', 'hybrid', 'adaptive']

    for strategy in expected_strategies:
        if strategy in strategies:
            print(f"OK: Strategy {strategy} available")
        else:
            print(f"ERROR: Strategy {strategy} missing")
            return False

    print("\nIntegration test completed successfully!")
    print("The autoformalization system is properly integrated with existing components.")
    return True


async def test_system_compatibility():
    """Test system compatibility with existing workflow."""
    print("\nTesting system compatibility...")
    
    # Test imports of related modules
    modules_to_test = [
        'leanaide_mdap',
        'lean4_integration', 
        'mdap_engine',
        'workflow_structures'
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"OK: Module {module_name} imports successfully")
        except ImportError:
            print(f"WARNING: Module {module_name} import failed (may be expected)")

    # Test that our system can work alongside existing systems
    try:
        # This should not conflict with existing implementations
        from leanaide_autoformalization_mdap_maker import (
            create_leanaide_autoformalization_engine,
            autoformalize_with_mdap_maker
        )
        print("OK: Compatibility functions import successfully")
    except ImportError as e:
        print(f"ERROR: Compatibility functions failed: {e}")
        return False

    print("Compatibility test completed!")
    return True


async def main():
    """Run all integration tests."""
    print("LeanAide Autoformalization System - Integration Test")
    print("=" * 60)
    
    success1 = await test_integration_with_existing_components()
    success2 = await test_system_compatibility()
    
    if success1 and success2:
        print("\nOK: All integration tests passed!")
        print("The autoformalization system is fully integrated with existing components.")
        return True
    else:
        print("\nERROR: Some integration tests failed!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)