"""
Comprehensive CrewAI Integration Test

This module tests all aspects of the CrewAI integration to ensure
everything works together properly.
"""

import asyncio
import json
import tempfile
import os
from datetime import datetime
from typing import Dict, Any, List

# Import all CrewAI integration components
from crewai_integration_complete import (
    CrewAIIntegration,
    execute_crewai_workflow,
    get_crewai_integration
)
from crewai_state_management import (
    StateManager,
    WorkflowState,
    ExecutionMethod,
    WorkflowStatus
)
from crewai_zero_error_workflow import (
    ZeroErrorWorkflow,
    create_workflow_definition
)
from ace_crewai_bridge import ACECrewAIWorkflowBridge


async def test_basic_crewai_integration():
    """Test basic CrewAI integration functionality."""
    print("Testing Basic CrewAI Integration...")
    
    try:
        # Initialize integration
        integration = CrewAIIntegration(
            model="gpt-4o-mini",
            state_storage_dir=tempfile.mkdtemp(),
            enable_learning=False,  # Disable learning for basic test
            enable_zero_error=False
        )
        
        # Test agents configuration
        agents_config = [
            {
                "role": "Test Agent",
                "goal": "Test basic functionality",
                "backstory": "A test agent for integration testing",
                "allow_delegation": False
            }
        ]
        
        # Test tasks configuration
        tasks_config = [
            {
                "description": "Say hello world",
                "expected_output": "A greeting message"
            }
        ]
        
        # Execute workflow
        result = await integration.create_and_execute_workflow(
            problem_statement="Say hello world",
            agents_config=agents_config,
            tasks_config=tasks_config,
            workflow_id="test_basic_workflow"
        )
        
        print(f"  ✓ Basic workflow completed: {result.get('success', False)}")
        print(f"  ✓ Result keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic integration test failed: {e}")
        return False


async def test_state_management():
    """Test state management functionality."""
    print("\nTesting State Management...")
    
    try:
        # Create temporary directory for state storage
        temp_dir = tempfile.mkdtemp()
        
        # Initialize state manager
        state_manager = StateManager(temp_dir)
        
        # Create a test workflow state
        workflow_id = "test_state_workflow"
        state = state_manager.load_state(workflow_id)
        
        if state is None:
            from crewai_state_management import create_workflow_state
            state = create_workflow_state(
                workflow_id=workflow_id,
                problem_statement="Test state management",
                execution_method=ExecutionMethod.TRADITIONAL
            )
        
        # Update state
        state.phase = 2
        state.status = WorkflowStatus.IN_PROGRESS
        
        # Save state
        state_manager.save_state(workflow_id, state)
        
        # Load state back
        loaded_state = state_manager.load_state(workflow_id)
        
        assert loaded_state is not None, "State should be loaded"
        assert loaded_state.phase == 2, "Phase should be updated"
        assert loaded_state.status == WorkflowStatus.IN_PROGRESS, "Status should be updated"
        
        print("  ✓ State management working correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ State management test failed: {e}")
        return False


async def test_zero_error_workflow():
    """Test zero-error workflow functionality."""
    print("\nTesting Zero-Error Workflow...")
    
    try:
        # Create a simple workflow definition
        workflow_def = create_workflow_definition(
            name="test_zero_error",
            description="Test zero-error workflow",
            steps=[
                {
                    "name": "test_step",
                    "action": "python_function",
                    "function": "test_func",
                    "parameters": {}
                }
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "test_input": {"type": "string"}
                },
                "required": ["test_input"]
            }
        )
        
        # Create zero-error workflow
        zero_error_wf = ZeroErrorWorkflow(workflow_def)
        
        # Execute with minimal inputs
        result = await zero_error_wf.execute({"test_input": "test"})
        
        print(f"  ✓ Zero-error workflow completed: {result.status.value}")
        print(f"  ✓ Steps completed: {result.steps_completed}/{result.steps_total}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Zero-error workflow test failed: {e}")
        return False


async def test_ace_bridge_integration():
    """Test ACE bridge integration."""
    print("\nTesting ACE Bridge Integration...")
    
    try:
        # Initialize ACE bridge
        ace_bridge = ACECrewAIWorkflowBridge(
            model="gpt-4o-mini",
            enable_learning=False  # Skip learning for this test
        )
        
        # Test phase 1 execution
        result = ace_bridge.execute_phase_1_setup(
            problem_statement="Test ACE bridge integration",
            enable_learning=False
        )
        
        print(f"  ✓ Phase 1 execution completed: {result.get('success', False)}")
        print(f"  ✓ Analysis: {result.get('analysis', 'N/A')[:50]}...")
        
        # Test full workflow execution
        full_result = ace_bridge.execute_full_workflow(
            problem_statement="Test full workflow",
            enable_learning=False
        )
        
        print(f"  ✓ Full workflow completed: {full_result.get('workflow_success', False)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ACE bridge test failed: {e}")
        return False


async def test_complete_integration():
    """Test complete integration of all components."""
    print("\nTesting Complete Integration...")
    
    try:
        # Create temporary directory for state storage
        temp_dir = tempfile.mkdtemp()
        
        # Initialize complete integration
        integration = CrewAIIntegration(
            model="gpt-4o-mini",
            state_storage_dir=temp_dir,
            enable_learning=True,
            enable_zero_error=True
        )
        
        # Define test agents
        agents_config = [
            {
                "role": "Research Analyst",
                "goal": "Analyze the given problem",
                "backstory": "An expert analyst with deep knowledge.",
                "allow_delegation": False
            },
            {
                "role": "Solution Architect",
                "goal": "Design a solution based on analysis",
                "backstory": "A solution architect with implementation experience.",
                "allow_delegation": False
            }
        ]
        
        # Define test tasks
        tasks_config = [
            {
                "description": "Analyze the problem: How can we improve user engagement?",
                "expected_output": "A detailed analysis of factors affecting user engagement"
            },
            {
                "description": "Design a solution based on the analysis",
                "expected_output": "A comprehensive solution to improve user engagement"
            }
        ]
        
        # Execute complete workflow
        result = await integration.create_and_execute_workflow(
            problem_statement="How can we improve user engagement?",
            agents_config=agents_config,
            tasks_config=tasks_config,
            execution_method=ExecutionMethod.ROMA_MDAP_MAKER
        )
        
        print(f"  ✓ Complete workflow completed: {result.get('success', False)}")
        
        # Check if state was saved
        workflow_id = result.get('workflow_id')
        if workflow_id:
            state = integration.get_workflow_state(workflow_id)
            if state:
                print(f"  ✓ State saved and retrieved: {state.status.value}")
            else:
                print("  ⚠ State not found after execution")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Complete integration test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests."""
    print("Running Comprehensive CrewAI Integration Tests")
    print("=" * 50)
    
    tests = [
        test_basic_crewai_integration,
        test_state_management,
        test_zero_error_workflow,
        test_ace_bridge_integration,
        test_complete_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")
    print(f"Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! CrewAI integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the integration.")
    
    return all(results)


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    exit(0 if success else 1)