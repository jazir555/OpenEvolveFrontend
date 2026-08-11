"""
Simple test to verify async patterns in CrewAI integration
"""
import sys
import os

# Add the project root to the path
sys.path.insert(0, r'C:\Users\mmeadow\Documents\OpenEvolve\Frontend')

def test_async_patterns():
    """Test that async patterns are properly implemented"""
    print("Testing async patterns in CrewAI integration...")
    
    success = True
    
    # Test unified flow - check that execute_full_workflow awaits phase_1_setup
    with open(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\crewai_unified_flow.py", "r", encoding="utf-8") as f:
        content = f.read()
        
        # Check that execute_full_workflow awaits phase_1_setup
        if "await self.phase_1_setup(" in content:
            print("  [PASS] execute_full_workflow properly awaits phase_1_setup")
        else:
            print("  [FAIL] execute_full_workflow does NOT await phase_1_setup")
            success = False
    
    # Test client - check that execute_phase is async
    with open(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\crewai_client.py", "r", encoding="utf-8") as f:
        content = f.read()
        
        # Check that execute_phase is async
        if "async def execute_phase" in content:
            print("  [PASS] execute_phase is async")
        else:
            print("  [FAIL] execute_phase is NOT async")
            success = False
    
    # Test API routes are async
    with open(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\crewai_api_routes.py", "r", encoding="utf-8") as f:
        content = f.read()
        
        if "async def execute_crewai_task_endpoint" in content:
            print("  [PASS] API execute endpoint is async")
        else:
            print("  [WARN] API execute endpoint async status not found")
    
    # Test that decomposition bridge methods are async
    with open(r"C:\Users\mmeadow\Documents\OpenEvolve\Frontend\decomposition_crewai_bridge.py", "r", encoding="utf-8") as f:
        content = f.read()
        
        if "async def execute_phase_4_verification" in content:
            print("  [PASS] execute_phase_4_verification is async")
        else:
            print("  [FAIL] execute_phase_4_verification is NOT async")
            success = False
        
        if "async def execute_phase_5_reassembly" in content:
            print("  [PASS] execute_phase_5_reassembly is async")
        else:
            print("  [FAIL] execute_phase_5_reassembly is NOT async")
            success = False
        
        if "async def execute_phase_6_validation" in content:
            print("  [PASS] execute_phase_6_validation is async")
        else:
            print("  [FAIL] execute_phase_6_validation is NOT async")
            success = False
    
    return success

def main():
    print("Running Async Pattern Verification for CrewAI Integration...")
    print("=" * 60)
    
    success = test_async_patterns()
    
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] All async patterns verified successfully!")
        print("CrewAI integration has proper async/await implementation")
    else:
        print("[FAILURE] Some async patterns are incorrect")
    
    return success

if __name__ == "__main__":
    result = main()
    exit(0 if result else 1)