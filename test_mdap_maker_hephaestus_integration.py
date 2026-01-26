<<<<<<< HEAD
"""
Test Suite for MDAP/MAKER-Hephaestus Integration

This module provides comprehensive tests for the integration between MDAP/MAKER
and the Hephaestus project management system.

Tests cover:
1. MDAP task and step synchronization
2. MAKER run and step synchronization
3. Voting result tracking
4. Red-flag tracking
5. Combined MDAP/MAKER workflows
6. Bidirectional sync
7. Error handling and edge cases
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# Import components to test
from hephaestus_integration import (
    HephaestusClient,
    HephaestusIntegrationManager,
    MDAPTaskSync,
    MAKERRunSync,
    TicketStatus,
    TicketType,
    MDAP_AVAILABLE,
    MAKER_AVAILABLE,
    setup_hephaestus_integration
)

# Import workflow structures
from workflow_structures import WorkflowState, DecompositionPlan, SubProblem

# Import MDAP and MAKER components
try:
    from mdap_engine import (
        MDAPTask, MDAPStep, MDAPConfig, MDAPRunResult,
        MDAPStepResult, MDAPVoteResult, RedFlagRules
    )
    from maker_engine import (
        MakerConfig, MakerRunResult, MakerState, MakerStep
    )
    from mdap_maker_complete import MAKERRunMetrics
    MDAP_LIBS_AVAILABLE = True
except ImportError:
    MDAP_LIBS_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_hephaestus_client():
    """Create a mock Hephaestus client"""
    client = Mock(spec=HephaestusClient)
    client.create_ticket = Mock(return_value="test-ticket-123")
    client.update_ticket = Mock(return_value=True)
    client.get_ticket = Mock(return_value={
        'id': 'test-ticket-123',
        'status': 'todo',
        'title': 'Test Ticket'
    })
    client.get_tickets_by_label = Mock(return_value=[])
    client.session = Mock()
    return client


@pytest.fixture
def workflow_state():
    """Create a test workflow state"""
    workflow = WorkflowState(
        problem_statement="Test problem for MDAP/MAKER integration",
        workflow_id="test-workflow-001",
        start_time=time.time()
    )
    workflow.ace_enabled = True
    workflow.ace_agent_id = "test-ace-agent"
    return workflow


@pytest.fixture
def mdap_task():
    """Create a test MDAP task"""
    if not MDAP_LIBS_AVAILABLE:
        pytest.skip("MDAP libraries not available")

    steps = [
        MDAPStep(
            step_id="step-1",
            prompt="Analyze the problem",
            task_type="decomposition",
            priority=1,
            expected_schema={"type": "object"}
        ),
        MDAPStep(
            step_id="step-2",
            prompt="Generate solution",
            task_type="solve",
            priority=2,
            expected_schema={"type": "object"}
        )
    ]

    return MDAPTask(
        task_id="mdap-task-001",
        description="Test MDAP task",
        steps=steps,
        max_retries=2,
        target_success_rate=0.95,
        metadata={"test": "data"}
    )


@pytest.fixture
def maker_config():
    """Create a test MAKER configuration"""
    if not MDAP_LIBS_AVAILABLE:
        pytest.skip("MDAP/MAKER libraries not available")

    return MakerConfig(
        k_min=2,
        k_max=8,
        max_votes_per_step=50,
        max_steps=100,
        timeout_seconds=60,
        checkpoint_interval=10
    )


@pytest.fixture
def maker_initial_state():
    """Create a test MAKER initial state"""
    return {
        "problem": "Test problem",
        "current_step": 0,
        "status": "initialized"
    }


# =============================================================================
# MDAP Integration Tests
# =============================================================================

class TestMDAPTaskSync:
    """Test suite for MDAP task synchronization"""

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_create_mdap_task_ticket(self, mock_hephaestus_client, mdap_task):
        """Test creating an MDAP task ticket"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)

        ticket_id = mdap_sync.create_mdap_task_ticket(mdap_task)

        assert ticket_id == "test-ticket-123"
        assert mock_hephaestus_client.create_ticket.called
        assert "mdap-task-001" in mdap_sync.task_id_to_ticket_map

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_create_mdap_step_tickets(self, mock_hephaestus_client, mdap_task):
        """Test creating step tickets for MDAP task"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)

        ticket_id = mdap_sync.create_mdap_task_ticket(mdap_task)

        # Verify step tickets were created
        assert len(mdap_sync.step_id_to_ticket_map) == len(mdap_task.steps)
        assert "step-1" in mdap_sync.step_id_to_ticket_map
        assert "step-2" in mdap_sync.step_id_to_ticket_map

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_sync_mdap_step_result(self, mock_hephaestus_client, mdap_task):
        """Test syncing MDAP step results"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)
        mdap_sync.create_mdap_task_ticket(mdap_task)

        # Create mock results
        vote_result = MDAPVoteResult(
            winner={"solution": "test"},
            votes={"{\"solution\": \"test\"}": 5},
            red_flags=0,
            confidence=0.95,
            attempts=5,
            duration_seconds=10.0
        )

        step_result = MDAPStepResult(
            step_id="step-1",
            vote_result=vote_result,
            status="success",
            retries=0
        )

        success = mdap_sync.sync_mdap_step_result("step-1", step_result, vote_result)

        assert success
        assert mock_hephaestus_client.update_ticket.called

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_sync_mdap_task_completion(self, mock_hephaestus_client, mdap_task):
        """Test syncing MDAP task completion"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)
        mdap_sync.create_mdap_task_ticket(mdap_task)

        # Create mock run result
        run_result = MDAPRunResult(
            task_id="mdap-task-001",
            step_results={},
            metrics={
                "steps_completed": 2,
                "steps_failed": 0,
                "votes_cast": 10,
                "red_flags": 0
            }
        )

        success = mdap_sync.sync_mdap_task_completion("mdap-task-001", run_result)

        assert success
        assert mock_hephaestus_client.update_ticket.called


# =============================================================================
# MAKER Integration Tests
# =============================================================================

class TestMAKERRunSync:
    """Test suite for MAKER run synchronization"""

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_create_maker_run_ticket(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test creating a MAKER run ticket"""
        maker_sync = MAKERRunSync(mock_hephaestus_client)

        ticket_id = maker_sync.create_maker_run_ticket(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )

        assert ticket_id == "test-ticket-123"
        assert mock_hephaestus_client.create_ticket.called
        assert "maker-run-001" in maker_sync.run_id_to_ticket_map

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_sync_maker_step(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test syncing MAKER step execution"""
        maker_sync = MAKERRunSync(mock_hephaestus_client)
        maker_sync.create_maker_run_ticket(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )

        state = MakerState(
            step_index=1,
            current_state=maker_initial_state,
            history=[],
            last_action=None
        )

        action = {"type": "move", "direction": "forward"}

        success = maker_sync.sync_maker_step("maker-run-001", 1, state, action)

        assert success
        assert mock_hephaestus_client.create_ticket.called

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_sync_maker_run_completion(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test syncing MAKER run completion"""
        maker_sync = MAKERRunSync(mock_hephaestus_client)
        maker_sync.create_maker_run_ticket(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )

        state = MakerState(
            step_index=10,
            current_state=maker_initial_state,
            history=[{"step": i} for i in range(10)],
            last_action={"type": "complete"}
        )

        run_result = MakerRunResult(
            state=state,
            metrics={
                "steps": 10,
                "votes_cast": 50,
                "red_flags": 2,
                "escalations": 1,
                "errors": 0
            },
            terminated_reason="stop_condition_met"
        )

        success = maker_sync.sync_maker_run_completion("maker-run-001", run_result)

        assert success
        assert mock_hephaestus_client.update_ticket.called


# =============================================================================
# Integration Manager Tests
# =============================================================================

class TestHephaestusIntegrationManager:
    """Test suite for HephaestusIntegrationManager with MDAP/MAKER"""

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_initialize_mdap_maker_workflow(
        self,
        mock_hephaestus_client,
        workflow_state,
        mdap_task,
        maker_config,
        maker_initial_state
    ):
        """Test initializing a combined MDAP/MAKER workflow"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        ticket_ids = manager.initialize_mdap_maker_workflow(
            workflow_state=workflow_state,
            mdap_task=mdap_task,
            maker_run_id="maker-run-001",
            maker_config=maker_config,
            maker_initial_state=maker_initial_state
        )

        assert ticket_ids["workflow_epic"] is not None
        assert ticket_ids["mdap_task"] is not None
        assert ticket_ids["maker_run"] is not None

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_get_mdap_maker_sync_status(self, mock_hephaestus_client):
        """Test getting MDAP/MAKER sync status"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        status = manager.get_mdap_maker_sync_status()

        assert "mdap_available" in status
        assert "maker_available" in status
        assert "mdap_sync_enabled" in status
        assert "maker_sync_enabled" in status


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end tests for MDAP/MAKER-Hephaestus integration"""

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_full_mdap_workflow(
        self,
        mock_hephaestus_client,
        mdap_task
    ):
        """Test a complete MDAP workflow with Hephaestus sync"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        # Sync MDAP task
        ticket_id = manager.sync_mdap_task(mdap_task)
        assert ticket_id is not None

        # Simulate step execution
        vote_result = MDAPVoteResult(
            winner={"result": "success"},
            votes={"{\"result\": \"success\"}": 3},
            red_flags=0,
            confidence=1.0,
            attempts=3,
            duration_seconds=5.0
        )

        step_result = MDAPStepResult(
            step_id="step-1",
            vote_result=vote_result,
            status="success",
            retries=0
        )

        manager.sync_mdap_step_result("step-1", step_result, vote_result)

        # Complete task
        run_result = MDAPRunResult(
            task_id="mdap-task-001",
            step_results={},
            metrics={
                "steps_completed": 1,
                "steps_failed": 0,
                "votes_cast": 3,
                "red_flags": 0
            }
        )

        manager.sync_mdap_task_completion("mdap-task-001", run_result)

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_full_maker_workflow(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test a complete MAKER workflow with Hephaestus sync"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        # Sync MAKER run
        ticket_id = manager.sync_maker_run(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )
        assert ticket_id is not None

        # Simulate step execution
        state = MakerState(
            step_index=1,
            current_state=maker_initial_state,
            history=[],
            last_action=None
        )

        action = {"type": "test_action"}

        manager.sync_maker_step("maker-run-001", 1, state, action)

        # Complete run
        run_result = MakerRunResult(
            state=state,
            metrics={
                "steps": 1,
                "votes_cast": 5,
                "red_flags": 0,
                "escalations": 0,
                "errors": 0
            },
            terminated_reason="completed"
        )

        manager.sync_maker_run_completion("maker-run-001", run_result)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in MDAP/MAKER-Hephaestus integration"""

    def test_mdap_sync_when_unavailable(self, mock_hephaestus_client):
        """Test MDAP sync behavior when MDAP is unavailable"""
        # Temporarily disable MDAP
        global MDAP_AVAILABLE
        original_value = MDAP_AVAILABLE
        MDAP_AVAILABLE = False

        try:
            mdap_sync = MDAPTaskSync(mock_hephaestus_client)
            assert mdap_sync.create_mdap_task_ticket(None) is None
        finally:
            MDAP_AVAILABLE = original_value

    def test_maker_sync_when_unavailable(self, mock_hephaestus_client):
        """Test MAKER sync behavior when MAKER is unavailable"""
        # Temporarily disable MAKER
        global MAKER_AVAILABLE
        original_value = MAKER_AVAILABLE
        MAKER_AVAILABLE = False

        try:
            maker_sync = MAKERRunSync(mock_hephaestus_client)
            assert maker_sync.create_maker_run_ticket(None, None, None) is None
        finally:
            MAKER_AVAILABLE = original_value

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_sync_step_result_without_ticket(self, mock_hephaestus_client, mdap_task):
        """Test syncing step result when no ticket exists"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)

        vote_result = MDAPVoteResult(
            winner=None,
            votes={},
            red_flags=1,
            confidence=0.0,
            attempts=10,
            duration_seconds=60.0
        )

        step_result = MDAPStepResult(
            step_id="non-existent-step",
            vote_result=vote_result,
            status="failure",
            retries=2
        )

        success = mdap_sync.sync_mdap_step_result("non-existent-step", step_result, vote_result)

        # Should not crash, but return False
        assert success is False


# =============================================================================
# Running Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
=======
"""
Test Suite for MDAP/MAKER-Hephaestus Integration

This module provides comprehensive tests for the integration between MDAP/MAKER
and the Hephaestus project management system.

Tests cover:
1. MDAP task and step synchronization
2. MAKER run and step synchronization
3. Voting result tracking
4. Red-flag tracking
5. Combined MDAP/MAKER workflows
6. Bidirectional sync
7. Error handling and edge cases
"""

import pytest
import json
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List

# Import components to test
from hephaestus_integration import (
    HephaestusClient,
    HephaestusIntegrationManager,
    MDAPTaskSync,
    MAKERRunSync,
    TicketStatus,
    TicketType,
    MDAP_AVAILABLE,
    MAKER_AVAILABLE,
    setup_hephaestus_integration
)

# Import workflow structures
from workflow_structures import WorkflowState, DecompositionPlan, SubProblem

# Import MDAP and MAKER components
try:
    from mdap_engine import (
        MDAPTask, MDAPStep, MDAPConfig, MDAPRunResult,
        MDAPStepResult, MDAPVoteResult, RedFlagRules
    )
    from maker_engine import (
        MakerConfig, MakerRunResult, MakerState, MakerStep
    )
    from mdap_maker_complete import MAKERRunMetrics
    MDAP_LIBS_AVAILABLE = True
except ImportError:
    MDAP_LIBS_AVAILABLE = False


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_hephaestus_client():
    """Create a mock Hephaestus client"""
    client = Mock(spec=HephaestusClient)
    client.create_ticket = Mock(return_value="test-ticket-123")
    client.update_ticket = Mock(return_value=True)
    client.get_ticket = Mock(return_value={
        'id': 'test-ticket-123',
        'status': 'todo',
        'title': 'Test Ticket'
    })
    client.get_tickets_by_label = Mock(return_value=[])
    client.session = Mock()
    return client


@pytest.fixture
def workflow_state():
    """Create a test workflow state"""
    workflow = WorkflowState(
        problem_statement="Test problem for MDAP/MAKER integration",
        workflow_id="test-workflow-001",
        start_time=time.time()
    )
    workflow.ace_enabled = True
    workflow.ace_agent_id = "test-ace-agent"
    return workflow


@pytest.fixture
def mdap_task():
    """Create a test MDAP task"""
    if not MDAP_LIBS_AVAILABLE:
        pytest.skip("MDAP libraries not available")

    steps = [
        MDAPStep(
            step_id="step-1",
            prompt="Analyze the problem",
            task_type="decomposition",
            priority=1,
            expected_schema={"type": "object"}
        ),
        MDAPStep(
            step_id="step-2",
            prompt="Generate solution",
            task_type="solve",
            priority=2,
            expected_schema={"type": "object"}
        )
    ]

    return MDAPTask(
        task_id="mdap-task-001",
        description="Test MDAP task",
        steps=steps,
        max_retries=2,
        target_success_rate=0.95,
        metadata={"test": "data"}
    )


@pytest.fixture
def maker_config():
    """Create a test MAKER configuration"""
    if not MDAP_LIBS_AVAILABLE:
        pytest.skip("MDAP/MAKER libraries not available")

    return MakerConfig(
        k_min=2,
        k_max=8,
        max_votes_per_step=50,
        max_steps=100,
        timeout_seconds=60,
        checkpoint_interval=10
    )


@pytest.fixture
def maker_initial_state():
    """Create a test MAKER initial state"""
    return {
        "problem": "Test problem",
        "current_step": 0,
        "status": "initialized"
    }


# =============================================================================
# MDAP Integration Tests
# =============================================================================

class TestMDAPTaskSync:
    """Test suite for MDAP task synchronization"""

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_create_mdap_task_ticket(self, mock_hephaestus_client, mdap_task):
        """Test creating an MDAP task ticket"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)

        ticket_id = mdap_sync.create_mdap_task_ticket(mdap_task)

        assert ticket_id == "test-ticket-123"
        assert mock_hephaestus_client.create_ticket.called
        assert "mdap-task-001" in mdap_sync.task_id_to_ticket_map

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_create_mdap_step_tickets(self, mock_hephaestus_client, mdap_task):
        """Test creating step tickets for MDAP task"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)

        ticket_id = mdap_sync.create_mdap_task_ticket(mdap_task)

        # Verify step tickets were created
        assert len(mdap_sync.step_id_to_ticket_map) == len(mdap_task.steps)
        assert "step-1" in mdap_sync.step_id_to_ticket_map
        assert "step-2" in mdap_sync.step_id_to_ticket_map

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_sync_mdap_step_result(self, mock_hephaestus_client, mdap_task):
        """Test syncing MDAP step results"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)
        mdap_sync.create_mdap_task_ticket(mdap_task)

        # Create mock results
        vote_result = MDAPVoteResult(
            winner={"solution": "test"},
            votes={"{\"solution\": \"test\"}": 5},
            red_flags=0,
            confidence=0.95,
            attempts=5,
            duration_seconds=10.0
        )

        step_result = MDAPStepResult(
            step_id="step-1",
            vote_result=vote_result,
            status="success",
            retries=0
        )

        success = mdap_sync.sync_mdap_step_result("step-1", step_result, vote_result)

        assert success
        assert mock_hephaestus_client.update_ticket.called

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_sync_mdap_task_completion(self, mock_hephaestus_client, mdap_task):
        """Test syncing MDAP task completion"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)
        mdap_sync.create_mdap_task_ticket(mdap_task)

        # Create mock run result
        run_result = MDAPRunResult(
            task_id="mdap-task-001",
            step_results={},
            metrics={
                "steps_completed": 2,
                "steps_failed": 0,
                "votes_cast": 10,
                "red_flags": 0
            }
        )

        success = mdap_sync.sync_mdap_task_completion("mdap-task-001", run_result)

        assert success
        assert mock_hephaestus_client.update_ticket.called


# =============================================================================
# MAKER Integration Tests
# =============================================================================

class TestMAKERRunSync:
    """Test suite for MAKER run synchronization"""

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_create_maker_run_ticket(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test creating a MAKER run ticket"""
        maker_sync = MAKERRunSync(mock_hephaestus_client)

        ticket_id = maker_sync.create_maker_run_ticket(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )

        assert ticket_id == "test-ticket-123"
        assert mock_hephaestus_client.create_ticket.called
        assert "maker-run-001" in maker_sync.run_id_to_ticket_map

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_sync_maker_step(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test syncing MAKER step execution"""
        maker_sync = MAKERRunSync(mock_hephaestus_client)
        maker_sync.create_maker_run_ticket(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )

        state = MakerState(
            step_index=1,
            current_state=maker_initial_state,
            history=[],
            last_action=None
        )

        action = {"type": "move", "direction": "forward"}

        success = maker_sync.sync_maker_step("maker-run-001", 1, state, action)

        assert success
        assert mock_hephaestus_client.create_ticket.called

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_sync_maker_run_completion(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test syncing MAKER run completion"""
        maker_sync = MAKERRunSync(mock_hephaestus_client)
        maker_sync.create_maker_run_ticket(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )

        state = MakerState(
            step_index=10,
            current_state=maker_initial_state,
            history=[{"step": i} for i in range(10)],
            last_action={"type": "complete"}
        )

        run_result = MakerRunResult(
            state=state,
            metrics={
                "steps": 10,
                "votes_cast": 50,
                "red_flags": 2,
                "escalations": 1,
                "errors": 0
            },
            terminated_reason="stop_condition_met"
        )

        success = maker_sync.sync_maker_run_completion("maker-run-001", run_result)

        assert success
        assert mock_hephaestus_client.update_ticket.called


# =============================================================================
# Integration Manager Tests
# =============================================================================

class TestHephaestusIntegrationManager:
    """Test suite for HephaestusIntegrationManager with MDAP/MAKER"""

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_initialize_mdap_maker_workflow(
        self,
        mock_hephaestus_client,
        workflow_state,
        mdap_task,
        maker_config,
        maker_initial_state
    ):
        """Test initializing a combined MDAP/MAKER workflow"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        ticket_ids = manager.initialize_mdap_maker_workflow(
            workflow_state=workflow_state,
            mdap_task=mdap_task,
            maker_run_id="maker-run-001",
            maker_config=maker_config,
            maker_initial_state=maker_initial_state
        )

        assert ticket_ids["workflow_epic"] is not None
        assert ticket_ids["mdap_task"] is not None
        assert ticket_ids["maker_run"] is not None

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_get_mdap_maker_sync_status(self, mock_hephaestus_client):
        """Test getting MDAP/MAKER sync status"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        status = manager.get_mdap_maker_sync_status()

        assert "mdap_available" in status
        assert "maker_available" in status
        assert "mdap_sync_enabled" in status
        assert "maker_sync_enabled" in status


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end tests for MDAP/MAKER-Hephaestus integration"""

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_full_mdap_workflow(
        self,
        mock_hephaestus_client,
        mdap_task
    ):
        """Test a complete MDAP workflow with Hephaestus sync"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        # Sync MDAP task
        ticket_id = manager.sync_mdap_task(mdap_task)
        assert ticket_id is not None

        # Simulate step execution
        vote_result = MDAPVoteResult(
            winner={"result": "success"},
            votes={"{\"result\": \"success\"}": 3},
            red_flags=0,
            confidence=1.0,
            attempts=3,
            duration_seconds=5.0
        )

        step_result = MDAPStepResult(
            step_id="step-1",
            vote_result=vote_result,
            status="success",
            retries=0
        )

        manager.sync_mdap_step_result("step-1", step_result, vote_result)

        # Complete task
        run_result = MDAPRunResult(
            task_id="mdap-task-001",
            step_results={},
            metrics={
                "steps_completed": 1,
                "steps_failed": 0,
                "votes_cast": 3,
                "red_flags": 0
            }
        )

        manager.sync_mdap_task_completion("mdap-task-001", run_result)

    @pytest.mark.skipif(not MAKER_AVAILABLE, reason="MAKER not available")
    def test_full_maker_workflow(
        self,
        mock_hephaestus_client,
        maker_config,
        maker_initial_state
    ):
        """Test a complete MAKER workflow with Hephaestus sync"""
        manager = HephaestusIntegrationManager(
            api_base="http://test.com",
            api_key="test-key",
            project_id="test-project"
        )
        manager.client = mock_hephaestus_client

        # Sync MAKER run
        ticket_id = manager.sync_maker_run(
            run_id="maker-run-001",
            initial_state=maker_initial_state,
            config=maker_config
        )
        assert ticket_id is not None

        # Simulate step execution
        state = MakerState(
            step_index=1,
            current_state=maker_initial_state,
            history=[],
            last_action=None
        )

        action = {"type": "test_action"}

        manager.sync_maker_step("maker-run-001", 1, state, action)

        # Complete run
        run_result = MakerRunResult(
            state=state,
            metrics={
                "steps": 1,
                "votes_cast": 5,
                "red_flags": 0,
                "escalations": 0,
                "errors": 0
            },
            terminated_reason="completed"
        )

        manager.sync_maker_run_completion("maker-run-001", run_result)


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling in MDAP/MAKER-Hephaestus integration"""

    def test_mdap_sync_when_unavailable(self, mock_hephaestus_client):
        """Test MDAP sync behavior when MDAP is unavailable"""
        # Temporarily disable MDAP
        global MDAP_AVAILABLE
        original_value = MDAP_AVAILABLE
        MDAP_AVAILABLE = False

        try:
            mdap_sync = MDAPTaskSync(mock_hephaestus_client)
            assert mdap_sync.create_mdap_task_ticket(None) is None
        finally:
            MDAP_AVAILABLE = original_value

    def test_maker_sync_when_unavailable(self, mock_hephaestus_client):
        """Test MAKER sync behavior when MAKER is unavailable"""
        # Temporarily disable MAKER
        global MAKER_AVAILABLE
        original_value = MAKER_AVAILABLE
        MAKER_AVAILABLE = False

        try:
            maker_sync = MAKERRunSync(mock_hephaestus_client)
            assert maker_sync.create_maker_run_ticket(None, None, None) is None
        finally:
            MAKER_AVAILABLE = original_value

    @pytest.mark.skipif(not MDAP_AVAILABLE, reason="MDAP not available")
    def test_sync_step_result_without_ticket(self, mock_hephaestus_client, mdap_task):
        """Test syncing step result when no ticket exists"""
        mdap_sync = MDAPTaskSync(mock_hephaestus_client)

        vote_result = MDAPVoteResult(
            winner=None,
            votes={},
            red_flags=1,
            confidence=0.0,
            attempts=10,
            duration_seconds=60.0
        )

        step_result = MDAPStepResult(
            step_id="non-existent-step",
            vote_result=vote_result,
            status="failure",
            retries=2
        )

        success = mdap_sync.sync_mdap_step_result("non-existent-step", step_result, vote_result)

        # Should not crash, but return False
        assert success is False


# =============================================================================
# Running Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
>>>>>>> 1cb9c5e35 (update)
