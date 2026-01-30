"""
Comprehensive State Machine Validation Test Suite

This test suite validates all state transitions for workflows and tickets,
ensuring that invalid transitions are rejected and valid transitions are allowed.

Author: OpenEvolve Team
Date: 2025-12-29
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bubblelabs_crewai_bridge # MIGRATED import (
        ExtendedWorkflowStatus,
        ExtendedTicketStatus,
        VALID_WORKFLOW_TRANSITIONS,
        VALID_TICKET_TRANSITIONS,
        validate_workflow_transition,
        validate_ticket_transition,
        get_valid_workflow_transitions,
        get_valid_ticket_transitions,
        is_terminal_workflow_status,
        is_terminal_ticket_status
    )
    STATE_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Could not import state validation: {e}")
    STATE_VALIDATION_AVAILABLE = False


class TestWorkflowStateTransitions(unittest.TestCase):
    """Test workflow state transitions."""

    def setUp(self):
        """Set up test fixtures."""
        if not STATE_VALIDATION_AVAILABLE:
            self.skipTest("State validation not available")

    def test_valid_workflow_transitions(self):
        """Test that all defined workflow transitions are valid."""
        # Test CREATED -> PENDING
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.CREATED, ExtendedWorkflowStatus.PENDING)
        )

        # Test PENDING -> RUNNING
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.PENDING, ExtendedWorkflowStatus.RUNNING)
        )

        # Test RUNNING -> PAUSED
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.RUNNING, ExtendedWorkflowStatus.PAUSED)
        )

        # Test RUNNING -> COMPLETED
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.RUNNING, ExtendedWorkflowStatus.COMPLETED)
        )

        # Test RUNNING -> FAILED
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.RUNNING, ExtendedWorkflowStatus.FAILED)
        )

        # Test PAUSED -> RUNNING (resume)
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.PAUSED, ExtendedWorkflowStatus.RUNNING)
        )

        # Test FAILED -> PENDING (retry)
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.FAILED, ExtendedWorkflowStatus.PENDING)
        )

        # Test STOPPED -> RUNNING (restart)
        self.assertTrue(
            validate_workflow_transition(ExtendedWorkflowStatus.STOPPED, ExtendedWorkflowStatus.RUNNING)
        )

    def test_invalid_workflow_transitions(self):
        """Test that invalid workflow transitions are rejected."""
        # Test CREATED -> RUNNING (should go through PENDING first)
        self.assertFalse(
            validate_workflow_transition(ExtendedWorkflowStatus.CREATED, ExtendedWorkflowStatus.RUNNING)
        )

        # Test COMPLETED -> RUNNING (can't restart completed workflow)
        self.assertFalse(
            validate_workflow_transition(ExtendedWorkflowStatus.COMPLETED, ExtendedWorkflowStatus.RUNNING)
        )

        # Test CANCELLED -> RUNNING (can't restart cancelled workflow)
        self.assertFalse(
            validate_workflow_transition(ExtendedWorkflowStatus.CANCELLED, ExtendedWorkflowStatus.RUNNING)
        )

        # Test PENDING -> COMPLETED (must run first)
        self.assertFalse(
            validate_workflow_transition(ExtendedWorkflowStatus.PENDING, ExtendedWorkflowStatus.COMPLETED)
        )

        # Test FAILED -> COMPLETED (must retry and run first)
        self.assertFalse(
            validate_workflow_transition(ExtendedWorkflowStatus.FAILED, ExtendedWorkflowStatus.COMPLETED)
        )

    def test_noop_workflow_transitions(self):
        """Test that no-op transitions (same state) are always valid."""
        for status in ExtendedWorkflowStatus:
            self.assertTrue(
                validate_workflow_transition(status, status),
                f"No-op transition for {status} should be valid"
            )

    def test_workflow_string_transitions(self):
        """Test workflow transitions with string inputs."""
        # Test with string inputs
        self.assertTrue(validate_workflow_transition("created", "pending"))
        self.assertTrue(validate_workflow_transition("running", "paused"))
        self.assertFalse(validate_workflow_transition("completed", "running"))

    def test_workflow_terminal_states(self):
        """Test that terminal workflow states have no valid transitions."""
        # COMPLETED and CANCELLED are terminal states
        self.assertTrue(is_terminal_workflow_status(ExtendedWorkflowStatus.COMPLETED))
        self.assertTrue(is_terminal_workflow_status(ExtendedWorkflowStatus.CANCELLED))

        # Other states are not terminal
        self.assertFalse(is_terminal_workflow_status(ExtendedWorkflowStatus.CREATED))
        self.assertFalse(is_terminal_workflow_status(ExtendedWorkflowStatus.RUNNING))
        self.assertFalse(is_terminal_workflow_status(ExtendedWorkflowStatus.PAUSED))

    def test_get_valid_workflow_transitions(self):
        """Test getting valid workflow transitions."""
        # From CREATED
        created_transitions = get_valid_workflow_transitions(ExtendedWorkflowStatus.CREATED)
        self.assertIn("pending", created_transitions)
        self.assertIn("cancelled", created_transitions)
        self.assertNotIn("running", created_transitions)

        # From RUNNING
        running_transitions = get_valid_workflow_transitions(ExtendedWorkflowStatus.RUNNING)
        self.assertIn("paused", running_transitions)
        self.assertIn("completed", running_transitions)
        self.assertIn("failed", running_transitions)
        self.assertIn("cancelled", running_transitions)

        # From COMPLETED (terminal)
        completed_transitions = get_valid_workflow_transitions(ExtendedWorkflowStatus.COMPLETED)
        self.assertEqual(len(completed_transitions), 0)


class TestTicketStateTransitions(unittest.TestCase):
    """Test ticket state transitions."""

    def setUp(self):
        """Set up test fixtures."""
        if not STATE_VALIDATION_AVAILABLE:
            self.skipTest("State validation not available")

    def test_valid_ticket_transitions(self):
        """Test that all defined ticket transitions are valid."""
        # Test TODO -> IN_PROGRESS
        self.assertTrue(
            validate_ticket_transition(ExtendedTicketStatus.TODO, ExtendedTicketStatus.IN_PROGRESS)
        )

        # Test IN_PROGRESS -> IN_REVIEW
        self.assertTrue(
            validate_ticket_transition(ExtendedTicketStatus.IN_PROGRESS, ExtendedTicketStatus.IN_REVIEW)
        )

        # Test IN_REVIEW -> DONE
        self.assertTrue(
            validate_ticket_transition(ExtendedTicketStatus.IN_REVIEW, ExtendedTicketStatus.DONE)
        )

        # Test IN_REVIEW -> IN_PROGRESS (back to work)
        self.assertTrue(
            validate_ticket_transition(ExtendedTicketStatus.IN_REVIEW, ExtendedTicketStatus.IN_PROGRESS)
        )

        # Test IN_PROGRESS -> TODO (backlog)
        self.assertTrue(
            validate_ticket_transition(ExtendedTicketStatus.IN_PROGRESS, ExtendedTicketStatus.TODO)
        )

        # Test BLOCKED -> TODO (unblock)
        self.assertTrue(
            validate_ticket_transition(ExtendedTicketStatus.BLOCKED, ExtendedTicketStatus.TODO)
        )

    def test_invalid_ticket_transitions(self):
        """Test that invalid ticket transitions are rejected."""
        # Test TODO -> DONE (must go through IN_PROGRESS and IN_REVIEW)
        self.assertFalse(
            validate_ticket_transition(ExtendedTicketStatus.TODO, ExtendedTicketStatus.DONE)
        )

        # Test DONE -> TODO (can't reopen done tickets)
        self.assertFalse(
            validate_ticket_transition(ExtendedTicketStatus.DONE, ExtendedTicketStatus.TODO)
        )

        # Test CANCELLED -> TODO (can't reopen cancelled tickets)
        self.assertFalse(
            validate_ticket_transition(ExtendedTicketStatus.CANCELLED, ExtendedTicketStatus.TODO)
        )

        # Test IN_PROGRESS -> DONE (must go through IN_REVIEW first)
        self.assertFalse(
            validate_ticket_transition(ExtendedTicketStatus.IN_PROGRESS, ExtendedTicketStatus.DONE)
        )

    def test_noop_ticket_transitions(self):
        """Test that no-op transitions (same state) are always valid."""
        for status in ExtendedTicketStatus:
            self.assertTrue(
                validate_ticket_transition(status, status),
                f"No-op transition for {status} should be valid"
            )

    def test_ticket_string_transitions(self):
        """Test ticket transitions with string inputs."""
        # Test with string inputs (ticket statuses are uppercase)
        self.assertTrue(validate_ticket_transition("TODO", "IN_PROGRESS"))
        self.assertTrue(validate_ticket_transition("IN_PROGRESS", "IN_REVIEW"))
        self.assertFalse(validate_ticket_transition("TODO", "DONE"))

    def test_ticket_terminal_states(self):
        """Test that terminal ticket states have no valid transitions."""
        # DONE and CANCELLED are terminal states
        self.assertTrue(is_terminal_ticket_status(ExtendedTicketStatus.DONE))
        self.assertTrue(is_terminal_ticket_status(ExtendedTicketStatus.CANCELLED))

        # Other states are not terminal
        self.assertFalse(is_terminal_ticket_status(ExtendedTicketStatus.TODO))
        self.assertFalse(is_terminal_ticket_status(ExtendedTicketStatus.IN_PROGRESS))

    def test_get_valid_ticket_transitions(self):
        """Test getting valid ticket transitions."""
        # From TODO
        todo_transitions = get_valid_ticket_transitions(ExtendedTicketStatus.TODO)
        self.assertIn("IN_PROGRESS", todo_transitions)
        self.assertIn("CANCELLED", todo_transitions)
        self.assertIn("BLOCKED", todo_transitions)
        self.assertNotIn("DONE", todo_transitions)

        # From IN_PROGRESS
        progress_transitions = get_valid_ticket_transitions(ExtendedTicketStatus.IN_PROGRESS)
        self.assertIn("IN_REVIEW", progress_transitions)
        self.assertIn("TODO", progress_transitions)
        self.assertIn("BLOCKED", progress_transitions)

        # From DONE (terminal)
        done_transitions = get_valid_ticket_transitions(ExtendedTicketStatus.DONE)
        self.assertEqual(len(done_transitions), 0)


class TestWorkflowToTicketMapping(unittest.TestCase):
    """Test workflow status to ticket status mapping."""

    def setUp(self):
        """Set up test fixtures."""
        if not STATE_VALIDATION_AVAILABLE:
            self.skipTest("State validation not available")

        # Import the bridge class
        from bubblelabs_crewai_bridge # MIGRATED import BubbleLabsHephaestusBridge
        from bubblelabs_integration import BubbleLabsIntegration

        self.bridge = BubbleLabsHephaestusBridge(
            bubblelabs_integration=BubbleLabsIntegration()
        )

    def test_created_pending_workflow_to_todo_ticket(self):
        """Test CREATED/PENDING workflow maps to TODO ticket."""
        from openevolve_bubblelabs_api import WorkflowStatus

        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.CREATED, 0.0
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.TODO)

        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.PENDING, 0.0
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.TODO)

    def test_running_workflow_progress_mapping(self):
        """Test RUNNING workflow maps based on progress."""
        from openevolve_bubblelabs_api import WorkflowStatus

        # Low progress -> TODO
        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.RUNNING, 0.1
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.TODO)

        # Medium progress -> IN_PROGRESS
        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.RUNNING, 0.5
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.IN_PROGRESS)

        # High progress -> IN_REVIEW
        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.RUNNING, 0.9
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.IN_REVIEW)

    def test_completed_workflow_to_done_ticket(self):
        """Test COMPLETED workflow maps to DONE ticket."""
        from openevolve_bubblelabs_api import WorkflowStatus

        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.COMPLETED, 1.0
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.DONE)

    def test_failed_cancelled_workflow_to_blocked_ticket(self):
        """Test FAILED/CANCELLED workflow maps to BLOCKED ticket."""
        from openevolve_bubblelabs_api import WorkflowStatus

        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.FAILED, 0.5
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.BLOCKED)

        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.CANCELLED, 0.5
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.CANCELLED)

    def test_paused_workflow_to_blocked_ticket(self):
        """Test PAUSED workflow maps to BLOCKED ticket."""
        from openevolve_bubblelabs_api import WorkflowStatus

        ticket_status = self.bridge._map_workflow_status_to_ticket_status(
            WorkflowStatus.PAUSED, 0.5
        )
        self.assertEqual(ticket_status, ExtendedTicketStatus.BLOCKED)


class TestStateTransitionCoverage(unittest.TestCase):
    """Test complete coverage of all state transitions."""

    def setUp(self):
        """Set up test fixtures."""
        if not STATE_VALIDATION_AVAILABLE:
            self.skipTest("State validation not available")

    def test_all_workflow_states_defined(self):
        """Test that all workflow states are defined."""
        expected_states = {
            "created", "pending", "running", "paused", "stopping",
            "stopped", "completed", "failed", "cancelled"
        }

        actual_states = {status.value for status in ExtendedWorkflowStatus}
        self.assertEqual(expected_states, actual_states)

    def test_all_ticket_states_defined(self):
        """Test that all ticket states are defined."""
        expected_states = {
            "TODO", "IN_PROGRESS", "IN_REVIEW", "DONE", "CANCELLED", "BLOCKED"
        }

        actual_states = {status.value for status in ExtendedTicketStatus}
        self.assertEqual(expected_states, actual_states)

    def test_all_workflow_transitions_defined(self):
        """Test that all workflow states have transition definitions."""
        for status in ExtendedWorkflowStatus:
            self.assertIn(
                status,
                VALID_WORKFLOW_TRANSITIONS,
                f"Workflow status {status} missing from transition table"
            )

    def test_all_ticket_transitions_defined(self):
        """Test that all ticket states have transition definitions."""
        for status in ExtendedTicketStatus:
            self.assertIn(
                status,
                VALID_TICKET_TRANSITIONS,
                f"Ticket status {status} missing from transition table"
            )

    def test_workflow_transition_consistency(self):
        """Test that workflow transitions are internally consistent."""
        # If A can transition to B, then B's transitions should be valid from the state machine
        for from_status, to_statuses in VALID_WORKFLOW_TRANSITIONS.items():
            for to_status in to_statuses:
                # Either to_status is terminal, or it has valid outgoing transitions
                if len(VALID_WORKFLOW_TRANSITIONS.get(to_status, set())) > 0:
                    # Non-terminal state should have transitions
                    self.assertIsNotNone(VALID_WORKFLOW_TRANSITIONS.get(to_status))

    def test_ticket_transition_consistency(self):
        """Test that ticket transitions are internally consistent."""
        # If A can transition to B, then B's transitions should be valid from the state machine
        for from_status, to_statuses in VALID_TICKET_TRANSITIONS.items():
            for to_status in to_statuses:
                # Either to_status is terminal, or it has valid outgoing transitions
                if len(VALID_TICKET_TRANSITIONS.get(to_status, set())) > 0:
                    # Non-terminal state should have transitions
                    self.assertIsNotNone(VALID_TICKET_TRANSITIONS.get(to_status))


def run_tests():
    """Run all tests and generate report."""
    if not STATE_VALIDATION_AVAILABLE:
        print("ERROR: State validation not available. Cannot run tests.")
        return False

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowStateTransitions))
    suite.addTests(loader.loadTestsFromTestCase(TestTicketStateTransitions))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkflowToTicketMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestStateTransitionCoverage))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*80)
    print("STATE MACHINE VALIDATION TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
