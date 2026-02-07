"""
Simple test to verify CrewAI integration changes work correctly
"""
import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

def test_basic_imports():
    """Test that basic imports work without errors"""
    try:
        from crewai_unified_flow import CrewAIUnifiedFlow, ExecutionMethod
        print("✓ CrewAIUnifiedFlow import successful")
        
        # Test creating an instance
        flow = CrewAIUnifiedFlow()
        print("✓ CrewAIUnifiedFlow instantiation successful")
        
        return True
    except Exception as e:
        print(f"✗ Import/instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_async_methods():
    """Test that async methods exist and are callable"""
    try:
        from crewai_unified_flow import CrewAIUnifiedFlow
        flow = CrewAIUnifiedFlow()
        
        # Check that phase_1_setup is async
        import inspect
        assert inspect.iscoroutinefunction(flow.phase_1_setup), "phase_1_setup should be async"
        print("✓ phase_1_setup is async")
        
        assert inspect.iscoroutinefunction(flow.phase_2_solve), "phase_2_solve should be async"
        print("✓ phase_2_solve is async")
        
        assert inspect.iscoroutinefunction(flow.phase_3_critique), "phase_3_critique should be async"
        print("✓ phase_3_critique is async")
        
        assert inspect.iscoroutinefunction(flow.phase_4_verify), "phase_4_verify should be async"
        print("✓ phase_4_verify is async")
        
        assert inspect.iscoroutinefunction(flow.phase_5_reassemble), "phase_5_reassemble should be async"
        print("✓ phase_5_reassemble is async")
        
        assert inspect.iscoroutinefunction(flow.phase_6_final_validation), "phase_6_final_validation should be async"
        print("✓ phase_6_final_validation is async")
        
        return True
    except Exception as e:
        print(f"✗ Async method check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_execution():
    """Test async execution of phase_1_setup"""
    try:
        from crewai_unified_flow import CrewAIUnifiedFlow, ExecutionMethod
        flow = CrewAIUnifiedFlow()
        
        # Test calling the async method
        result = await flow.phase_1_setup(
            problem_statement="Test async execution",
            execution_method=ExecutionMethod.TRADITIONAL
        )
        
        print(f"✓ Async execution successful, result type: {type(result)}")
        return True
    except Exception as e:
        print(f"✗ Async execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("Testing CrewAI Integration Changes...")
    print("=" * 50)
    
    success = True
    
    # Test basic imports
    success &= test_basic_imports()
    print()
    
    # Test async methods
    success &= test_async_methods()
    print()
    
    # Test async execution
    success &= await test_async_execution()
    print()
    
    if success:
        print("🎉 All tests passed! CrewAI integration changes are working correctly.")
    else:
        print("❌ Some tests failed.")
    
    return success

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)