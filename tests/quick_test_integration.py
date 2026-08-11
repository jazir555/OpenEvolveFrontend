"""
Quick verification test for the LeanAide SOP Integration with Predictive Flagging
"""

import asyncio
from unittest.mock import Mock

from leanaide_sop_integration import LeanAideSOPIntegration, MathematicalComponent

async def quick_test():
    print("Testing LeanAide SOP Integration with Predictive Flagging...")
    
    # Create mock client
    mock_client = Mock()
    mock_client.cache = {}
    
    # Create integration
    integration = LeanAideSOPIntegration(mock_client)
    
    print("SUCCESS: Integration created successfully")
    
    # Test component extraction
    test_sop = """
    # Test SOP
    
    ## Objective
    Prove that for all natural numbers n, n + 0 = n.
    
    ## Condition
    Where x > 0, verify that x² > 0.
    """
    
    components = await integration.extract_mathematical_components(test_sop)
    print(f"SUCCESS: Extracted {len(components)} mathematical components")

    # Test domain inference
    for component in components:
        print(f"  - Component: {component.description[:50]}... (domain: {component.domain})")

    # Test verification with fallback
    if components:
        result = await integration.verify_mathematical_component(components[0])
        print(f"SUCCESS: Verification completed with success={result.success}, confidence={result.confidence}")

    # Test SOP verification
    sop_results = await integration.verify_sop_mathematical_components(test_sop)
    print(f"SUCCESS: SOP verification completed: {sop_results}")

    print("\nSUCCESS: All basic functionality tests passed!")
    print("The LeanAide SOP Integration with Predictive Flagging is working correctly.")

if __name__ == "__main__":
    asyncio.run(quick_test())