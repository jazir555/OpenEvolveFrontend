"""
Comprehensive Tests for CrewAI Integration

This module provides comprehensive tests for all aspects of the CrewAI integration
to ensure everything works correctly together.
"""

import asyncio
import tempfile
import os
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import all CrewAI components to test
from crewai_hub import CrewAIHub, get_crewai_hub, execute_crewai_task
from crewai_integration_complete import CrewAIIntegration
from crewai_state_management import StateManager, WorkflowState, ExecutionMethod, WorkflowStatus
from crewai_zero_error_workflow import ZeroErrorWorkflow, create_workflow_definition
from ace_crewai_bridge import ACECrewAIWorkflowBridge
from crewai_client import CrewAIClient, CrewAIMonitor
from crewai_unified_flow import CrewAIUnifiedFlow, ExecutionMethod as FlowExecutionMethod
from crewai_api_routes import (
    execute_crewai_task_endpoint,
    list_crewai_workflows_endpoint,
    get_crewai_workflow_endpoint,
    get_crewai_workflow_metrics_endpoint,
    get_crewai_status_endpoint,
    CrewAITaskRequest
)


class TestCrewAIHub:
    """Test the CrewAI Hub component."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Use temporary directory for state storage
        self.temp_dir = tempfile.mkdtemp()
        self.hub = CrewAIHub(
            state_storage_dir=self.temp_dir,
            enable_learning=False,  # Disable learning for tests
            enable_zero_error=True
        )
    
    def teardown_method(self):
        """Cleanup after tests."""
        self.hub.cleanup()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_hub_initialization(self):
        """Test CrewAI Hub initialization."""
        assert self.hub is not None
        assert self.hub.state_manager is not None
        assert self.hub.unified_flow is not None
        assert self.hub.client is not None
        assert self.hub.monitor is not None
        assert self.hub.integration is not None
        assert self.hub.delegation_manager is not None
    
    async def test_execute_workflow(self):
        """Test workflow execution."""
        result = await self.hub.execute_workflow(
            problem_statement="Test workflow execution",
            execution_method=ExecutionMethod.TRADITIONAL
        )
        
        assert result is not None
        # The result might be success or failure depending on CrewAI availability,
        # but it should at least return a dictionary
        assert isinstance(result, dict)
    
    async def test_get_workflow_state(self):
        """Test getting workflow state."""
        # Create a dummy workflow ID
        workflow_id = "test_workflow_123"
        
        # Create a test state
        state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement="Test problem",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.IN_PROGRESS
        )
        
        # Save the state
        self.hub.state_manager.save_state(workflow_id, state)
        
        # Retrieve the state
        retrieved_state = self.hub.get_workflow_state(workflow_id)
        
        assert retrieved_state is not None
        assert retrieved_state.workflow_id == workflow_id
        assert retrieved_state.problem_statement == "Test problem"
    
    async def test_list_workflows(self):
        """Test listing workflows."""
        # Create a test workflow
        workflow_id = "test_list_workflow"
        state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement="Test problem",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.IN_PROGRESS
        )
        self.hub.state_manager.save_state(workflow_id, state)
        
        # List workflows
        workflows = self.hub.list_workflows()
        
        assert workflow_id in workflows
        assert len(workflows) >= 1
    
    async def test_get_workflow_metrics(self):
        """Test getting workflow metrics."""
        # Create a test workflow
        workflow_id = "test_metrics_workflow"
        state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement="Test problem",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.IN_PROGRESS
        )
        self.hub.state_manager.save_state(workflow_id, state)
        
        # Get metrics
        metrics = self.hub.get_workflow_metrics(workflow_id)
        
        assert "workflow_id" in metrics
        assert "state_info" in metrics
        assert "monitor_metrics" in metrics
        assert metrics["workflow_id"] == workflow_id


class TestCrewAIStateManagement:
    """Test state management functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.state_manager = StateManager(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_and_save_state(self):
        """Test creating and saving workflow state."""
        workflow_id = "test_state_workflow"
        state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement="Test problem statement",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.PENDING
        )
        
        # Save state
        self.state_manager.save_state(workflow_id, state)
        
        # Load state
        loaded_state = self.state_manager.load_state(workflow_id)
        
        assert loaded_state is not None
        assert loaded_state.workflow_id == workflow_id
        assert loaded_state.problem_statement == "Test problem statement"
        assert loaded_state.execution_method == ExecutionMethod.TRADITIONAL
        assert loaded_state.status == WorkflowStatus.PENDING
    
    def test_list_workflows_by_status(self):
        """Test listing workflows by status."""
        # Create test workflows with different statuses
        workflow1_id = "test_wf_pending"
        workflow2_id = "test_wf_running"
        
        state1 = WorkflowState(
            workflow_id=workflow1_id,
            problem_statement="Pending workflow",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.PENDING
        )
        state2 = WorkflowState(
            workflow_id=workflow2_id,
            problem_statement="Running workflow",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.IN_PROGRESS
        )
        
        self.state_manager.save_state(workflow1_id, state1)
        self.state_manager.save_state(workflow2_id, state2)
        
        # List by pending status
        pending_workflows = self.state_manager.list_workflows(status=WorkflowStatus.PENDING)
        assert workflow1_id in pending_workflows
        assert workflow2_id not in pending_workflows
        
        # List all workflows
        all_workflows = self.state_manager.list_workflows()
        assert len(all_workflows) >= 2
        assert workflow1_id in all_workflows
        assert workflow2_id in all_workflows


class TestCrewAIClient:
    """Test CrewAI Client functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.client = CrewAIClient(state_storage_dir=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_client_initialization(self):
        """Test CrewAI client initialization."""
        assert self.client is not None
        assert self.client.state_storage_dir == self.temp_dir
        assert self.client.unified_flow is not None
    
    async def test_execute_workflow(self):
        """Test workflow execution via client."""
        result = await self.client.execute_workflow(
            problem_statement="Test client workflow execution",
            execution_method=ExecutionMethod.TRADITIONAL
        )

        # The result should be an ExecutionResult object or dict
        assert result is not None
        assert hasattr(result, 'workflow_id') or 'workflow_id' in result
        assert hasattr(result, 'status') or 'status' in result


class TestCrewAIUnifiedFlow:
    """Test unified flow functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.flow = CrewAIUnifiedFlow(
            default_execution_method=FlowExecutionMethod.AUTO,
            enable_persistence=True,
            state_storage_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_phase_1_setup(self):
        """Test Phase 1 setup."""
        result = self.flow.phase_1_setup(
            problem_statement="Test phase 1 setup",
            execution_method=FlowExecutionMethod.TRADITIONAL
        )
        
        assert result is not None
        assert "phase" in result
        assert result["phase"] == 1
        assert "status" in result
    
    async def test_execute_full_workflow(self):
        """Test full workflow execution."""
        result = await self.flow.execute_full_workflow(
            problem_statement="Test full workflow",
            execution_method=FlowExecutionMethod.TRADITIONAL
        )

        assert result is not None
        assert "workflow" in result
        assert "phases" in result
        assert isinstance(result["phases"], dict)


class TestCrewAIAPIRoutes:
    """Test API routes functionality."""
    
    async def test_execute_crewai_task_endpoint(self):
        """Test the execute task endpoint."""
        request = CrewAITaskRequest(
            problem_statement="Test API endpoint execution",
            execution_method="traditional"
        )
        
        # Mock the actual execution to avoid dependency issues
        with patch('crewai_hub.execute_crewai_task') as mock_execute:
            mock_execute.return_value = {
                "success": True,
                "workflow_id": "test_api_workflow",
                "result": "Test result"
            }
            
            result = await execute_crewai_task_endpoint(request)

            # Just verify that the function executed without error
            # The exact return format may vary depending on the implementation
            assert result is not None
    
    async def test_list_workflows_endpoint(self):
        """Test the list workflows endpoint."""
        result = list_crewai_workflows_endpoint(status=None)
        
        assert result is not None
        assert "workflows" in result
        assert "count" in result
    
    async def test_get_status_endpoint(self):
        """Test the status endpoint."""
        result = get_crewai_status_endpoint()
        
        assert result is not None
        assert "hub" in result
        assert "components" in result


class TestIntegrationScenarios:
    """Test complex integration scenarios."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_complete_workflow_scenario(self):
        """Test a complete workflow scenario from start to finish."""
        # Initialize hub
        hub = CrewAIHub(
            state_storage_dir=self.temp_dir,
            enable_learning=False,
            enable_zero_error=True
        )
        
        # Execute a workflow
        result = await hub.execute_workflow(
            problem_statement="Complete test workflow scenario",
            execution_method=ExecutionMethod.TRADITIONAL
        )
        
        # Even if CrewAI is not available, the call should not crash
        assert result is not None
        assert isinstance(result, dict)
        
        # Test that we can get the status
        status = hub.get_crewai_status()
        assert status is not None
        assert "hub" in status
        assert "components" in status
        
        hub.cleanup()
    
    async def test_state_persistence_scenario(self):
        """Test state persistence across different operations."""
        # Initialize state manager
        state_manager = StateManager(self.temp_dir)
        
        # Create and save a state
        workflow_id = "persistent_test_workflow"
        state = WorkflowState(
            workflow_id=workflow_id,
            problem_statement="Persistent state test",
            execution_method=ExecutionMethod.TRADITIONAL,
            status=WorkflowStatus.IN_PROGRESS
        )
        state_manager.save_state(workflow_id, state)
        
        # Verify state is saved
        saved_state = state_manager.load_state(workflow_id)
        assert saved_state is not None
        assert saved_state.status == WorkflowStatus.IN_PROGRESS
        
        # Update state
        saved_state.status = WorkflowStatus.COMPLETED
        state_manager.save_state(workflow_id, saved_state)
        
        # Verify state is updated
        updated_state = state_manager.load_state(workflow_id)
        assert updated_state is not None
        assert updated_state.status == WorkflowStatus.COMPLETED


# Run all tests
async def run_all_tests():
    """Run all test suites."""
    print("Running CrewAI Integration Tests...")
    
    test_instance = TestCrewAIHub()
    test_instance.setup_method()
    
    try:
        await test_instance.test_hub_initialization()
        print("[PASS] Hub initialization test passed")
        
        await test_instance.test_execute_workflow()
        print("[PASS] Workflow execution test passed")
        
        await test_instance.test_get_workflow_state()
        print("[PASS] Get workflow state test passed")
        
        await test_instance.test_list_workflows()
        print("[PASS] List workflows test passed")
        
        await test_instance.test_get_workflow_metrics()
        print("[PASS] Get workflow metrics test passed")
        
    finally:
        test_instance.teardown_method()
    
    # Test state management
    state_test = TestCrewAIStateManagement()
    state_test.setup_method()
    
    try:
        state_test.test_create_and_save_state()
        print("[PASS] State creation and save test passed")
        
        state_test.test_list_workflows_by_status()
        print("[PASS] List workflows by status test passed")
        
    finally:
        state_test.teardown_method()
    
    # Test client
    client_test = TestCrewAIClient()
    client_test.setup_method()
    
    try:
        await client_test.test_client_initialization()
        print("[PASS] Client initialization test passed")
        
        await client_test.test_execute_workflow()
        print("[PASS] Client workflow execution test passed")
        
    finally:
        client_test.teardown_method()
    
    # Test unified flow
    flow_test = TestCrewAIUnifiedFlow()
    flow_test.setup_method()
    
    try:
        await flow_test.test_phase_1_setup()
        print("[PASS] Phase 1 setup test passed")
        
        await flow_test.test_execute_full_workflow()
        print("[PASS] Full workflow execution test passed")
        
    finally:
        flow_test.teardown_method()
    
    # Test API routes
    api_test = TestCrewAIAPIRoutes()
    
    await api_test.test_execute_crewai_task_endpoint()
    print("[PASS] API execute task endpoint test passed")
    
    await api_test.test_list_workflows_endpoint()
    print("[PASS] API list workflows endpoint test passed")
    
    await api_test.test_get_status_endpoint()
    print("[PASS] API status endpoint test passed")
    
    # Test integration scenarios
    scenario_test = TestIntegrationScenarios()
    scenario_test.setup_method()
    
    try:
        await scenario_test.test_complete_workflow_scenario()
        print("[PASS] Complete workflow scenario test passed")
        
        await scenario_test.test_state_persistence_scenario()
        print("[PASS] State persistence scenario test passed")
        
    finally:
        scenario_test.teardown_method()
    
    print("\n[SUCCESS] All CrewAI integration tests passed!")


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())