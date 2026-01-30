"""
Test workflow-to-ticket mappings persistence in BubbleLabs-CrewAI Bridge.

This test verifies that:
1. Mappings are saved to database when created
2. Mappings are restored on application restart
3. LRU cache and database stay synchronized
4. Old mappings are cleaned up properly
5. All CRUD operations persist correctly

Author: OpenEvolve Team
Date: 2025-12-29
"""

import os
import sys
import time
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the bridge and related classes
try:
    from bubblelabs_crewai_bridge import (
        BubbleLabsCrewAIBridge,
        WorkflowTicketMapping,
        BubbleLabsTicketConfig
    )
    from bubblelabs_integration import BubbleLabsIntegration, BubbleWorkflowDefinition
    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import bridge modules: {e}")
    BRIDGE_AVAILABLE = False


class TestBubbleLabsPersistence(unittest.TestCase):
    """Test suite for workflow-to-ticket mapping persistence."""

    def setUp(self):
        """Set up test environment before each test."""
        if not BRIDGE_AVAILABLE:
            self.skipTest("BubbleLabs bridge not available")

        # Use temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_mappings.db")

        # Create mock BubbleLabs integration
        self.mock_bubblelabs = Mock(spec=BubbleLabsIntegration)
        self.mock_bubblelabs.workflow_definitions = {}

        # Create mock CrewAI client
        self.mock_CrewAI = Mock()
        self.mock_CrewAI.create_ticket = Mock(return_value="TEST-123")
        self.mock_CrewAI.update_ticket = Mock(return_value=True)

    def tearDown(self):
        """Clean up after each test."""
        # Remove test database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except:
                pass

    def _create_bridge(self) -> BubbleLabsCrewAIBridge:
        """Create a bridge instance for testing."""
        bridge = BubbleLabsCrewAIBridge(
            bubblelabs_integration=self.mock_bubblelabs,
            crewai_client=self.mock_CrewAI,
            batch_size=10,
            mappings_db_path=self.test_db_path  # Use test database path
        )
        return bridge

    def _create_mock_workflow(self, workflow_id: str, name: str) -> Mock:
        """Create a mock workflow definition."""
        workflow = Mock(spec=BubbleWorkflowDefinition)
        workflow.id = workflow_id
        workflow.name = name
        workflow.description = f"Test workflow {name}"
        workflow.metadata = {"created_at": time.time()}
        workflow.nodes = [{"id": "node1", "type": "test", "data": {"label": "Test Node"}}]
        workflow.edges = []
        return workflow

    def test_database_initialization(self):
        """Test that database is properly initialized on bridge creation."""
        bridge = self._create_bridge()

        # Verify database file exists
        self.assertTrue(os.path.exists(self.test_db_path), "Database file should be created")

        # Verify table structure
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='workflow_ticket_mappings'
        """)
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Table should exist")

        # Check if indexes exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name='idx_mappings_ticket_status'
        """)
        result = cursor.fetchone()
        self.assertIsNotNone(result, "Status index should exist")

        conn.close()

    def test_mapping_saved_on_create(self):
        """Test that mapping is saved to database when ticket is created."""
        bridge = self._create_bridge()

        # Create a workflow and ticket
        workflow = self._create_mock_workflow("wf-001", "Test Workflow")
        self.mock_bubblelabs.workflow_definitions["wf-001"] = workflow

        ticket_id = bridge.create_ticket_from_workflow(workflow)

        # Verify ticket was created
        self.assertIsNotNone(ticket_id, "Ticket ID should be returned")
        self.assertEqual(ticket_id, "TEST-123", "Correct ticket ID should be returned")

        # Verify mapping is in memory cache
        self.assertIn("wf-001", bridge._mappings, "Mapping should be in cache")
        self.assertEqual(bridge._mappings["wf-001"].ticket_id, "TEST-123", "Cache should have correct ticket ID")

        # Verify mapping is in database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ticket_id, ticket_status FROM workflow_ticket_mappings WHERE workflow_id = ?", ("wf-001",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Mapping should be in database")
        self.assertEqual(row[0], "TEST-123", "Database should have correct ticket ID")
        self.assertEqual(row[1], "TODO", "Database should have correct status")

    def test_mappings_loaded_on_restart(self):
        """Test that mappings are restored from database on bridge restart."""
        # Create first bridge instance and add mapping
        bridge1 = self._create_bridge()
        workflow = self._create_mock_workflow("wf-002", "Persistent Workflow")
        self.mock_bubblelabs.workflow_definitions["wf-002"] = workflow

        ticket_id = bridge1.create_ticket_from_workflow(workflow)
        self.assertIsNotNone(ticket_id, "First bridge should create ticket")

        # Simulate application restart by creating new bridge instance
        bridge2 = self._create_bridge()

        # Verify mapping was loaded from database
        self.assertIn("wf-002", bridge2._mappings, "Mapping should be loaded on restart")
        self.assertEqual(bridge2._mappings["wf-002"].ticket_id, ticket_id, "Loaded mapping should have correct ticket ID")

    def test_update_persisted_to_database(self):
        """Test that mapping updates are persisted to database."""
        bridge = self._create_bridge()
        workflow = self._create_mock_workflow("wf-003", "Update Test")
        self.mock_bubblelabs.workflow_definitions["wf-003"] = workflow

        # Create initial mapping
        ticket_id = bridge.create_ticket_from_workflow(workflow)
        self.assertIsNotNone(ticket_id, "Ticket should be created")

        # Update ticket progress
        from openevolve_bubblelabs_api import WorkflowStatus, WorkflowMetrics
        metrics = WorkflowMetrics(
            execution_time=10.0,
            tokens_used=1000,
            iterations_completed=5,
            total_iterations=10
        )

        success = bridge.update_ticket_progress(
            "wf-003",  # Using workflow_id as instance_id
            progress=0.5,
            status=WorkflowStatus.RUNNING,
            metrics=metrics
        )

        self.assertTrue(success, "Update should succeed")

        # Verify update in database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ticket_status FROM workflow_ticket_mappings WHERE workflow_id = ?", ("wf-003",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Mapping should exist")
        # Status should be updated based on progress
        self.assertIn(row[0], ["TODO", "IN_PROGRESS", "IN_REVIEW"], "Status should reflect progress")

    def test_close_persisted_to_database(self):
        """Test that ticket close is persisted to database."""
        bridge = self._create_bridge()
        workflow = self._create_mock_workflow("wf-004", "Close Test")
        self.mock_bubblelabs.workflow_definitions["wf-004"] = workflow

        # Create ticket
        ticket_id = bridge.create_ticket_from_workflow(workflow)
        self.assertIsNotNone(ticket_id, "Ticket should be created")

        # Close ticket
        success = bridge.close_ticket_on_completion("wf-004", success=True)
        self.assertTrue(success, "Close should succeed")

        # Verify close in database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT ticket_status FROM workflow_ticket_mappings WHERE workflow_id = ?", ("wf-004",))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Mapping should exist")
        self.assertEqual(row[0], "DONE", "Status should be DONE")

    def test_get_all_mappings(self):
        """Test retrieving all mappings from database."""
        bridge = self._create_bridge()

        # Create multiple workflows and tickets
        for i in range(5):
            workflow_id = f"wf-{i:03d}"
            workflow = self._create_mock_workflow(workflow_id, f"Workflow {i}")
            self.mock_bubblelabs.workflow_definitions[workflow_id] = workflow
            bridge.create_ticket_from_workflow(workflow)

        # Get all mappings
        all_mappings = bridge.get_all_mappings()

        # Verify all mappings are retrieved
        self.assertEqual(len(all_mappings), 5, "Should retrieve all 5 mappings")

        # Verify mappings have correct structure
        for workflow_id, mapping in all_mappings.items():
            self.assertIsInstance(mapping, WorkflowTicketMapping, "Should be WorkflowTicketMapping object")
            self.assertIsNotNone(mapping.ticket_id, "Should have ticket ID")
            self.assertIsNotNone(mapping.ticket_status, "Should have status")

    def test_cleanup_old_mappings(self):
        """Test cleanup of old mappings."""
        bridge = self._create_bridge()

        # Create some workflows and tickets
        for i in range(3):
            workflow_id = f"wf-old-{i}"
            workflow = self._create_mock_workflow(workflow_id, f"Old Workflow {i}")
            self.mock_bubblelabs.workflow_definitions[workflow_id] = workflow
            bridge.create_ticket_from_workflow(workflow)

        # Manually update timestamps to simulate old mappings
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        old_time = time.time() - (100 * 86400)  # 100 days ago
        cursor.execute("""
            UPDATE workflow_ticket_mappings
            SET created_at = ?, updated_at = ?, ticket_status = 'DONE'
            WHERE workflow_id LIKE 'wf-old-%'
        """, (old_time, old_time))
        conn.commit()
        conn.close()

        # Reload to get updated data
        bridge._load_mappings_from_db()

        # Run cleanup
        deleted_count = bridge.cleanup_old_mappings(max_age_days=90)

        # Verify old mappings were deleted
        self.assertEqual(deleted_count, 3, "Should delete 3 old mappings")

        # Verify they're gone from database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings WHERE workflow_id LIKE 'wf-old-%'")
        count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(count, 0, "Old mappings should be deleted from database")

    def test_get_mapping_stats(self):
        """Test retrieving mapping statistics."""
        bridge = self._create_bridge()

        # Create some mappings
        for i in range(3):
            workflow_id = f"wf-stats-{i}"
            workflow = self._create_mock_workflow(workflow_id, f"Stats Workflow {i}")
            self.mock_bubblelabs.workflow_definitions[workflow_id] = workflow
            bridge.create_ticket_from_workflow(workflow)

        # Get stats
        stats = bridge.get_mapping_stats()

        # Verify stats
        self.assertIn("total_mappings", stats, "Should have total count")
        self.assertEqual(stats["total_mappings"], 3, "Should have 3 mappings")
        self.assertIn("by_status", stats, "Should have status breakdown")
        self.assertIn("cache_size", stats, "Should have cache size")
        self.assertIn("database_path", stats, "Should have database path")

    def test_lru_cache_sync_with_database(self):
        """Test that LRU cache stays synchronized with database."""
        bridge = self._create_bridge()

        # Create more mappings than cache can hold
        for i in range(bridge._MAX_MAPPINGS + 10):
            workflow_id = f"wf-lru-{i:04d}"
            workflow = self._create_mock_workflow(workflow_id, f"LRU Test {i}")
            self.mock_bubblelabs.workflow_definitions[workflow_id] = workflow
            bridge.create_ticket_from_workflow(workflow)

        # Verify cache respects max size
        self.assertLessEqual(len(bridge._mappings), bridge._MAX_MAPPINGS, "Cache should respect max size")

        # Verify all mappings are in database
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
        db_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(db_count, bridge._MAX_MAPPINGS + 10, "Database should have all mappings")

    def test_concurrent_access_safety(self):
        """Test that database operations are thread-safe."""
        import threading

        bridge = self._create_bridge()
        errors = []
        created_count = [0]

        def create_workflow(index):
            try:
                workflow_id = f"wf-concurrent-{index}"
                workflow = self._create_mock_workflow(workflow_id, f"Concurrent {index}")
                bridge.create_ticket_from_workflow(workflow)
                created_count[0] += 1
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=create_workflow, args=(i,))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Verify no errors occurred
        self.assertEqual(len(errors), 0, "No concurrent access errors should occur")
        self.assertEqual(created_count[0], 10, "All workflows should be created")


def run_persistence_tests():
    """Run persistence tests and return results."""
    if not BRIDGE_AVAILABLE:
        return {
            "status": "skipped",
            "reason": "BubbleLabs bridge not available",
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0
        }

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBubbleLabsPersistence)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return {
        "status": "completed",
        "tests_run": result.testsRun,
        "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
        "tests_failed": len(result.failures) + len(result.errors),
        "failures": [str(f) for f in result.failures],
        "errors": [str(e) for e in result.errors]
    }


if __name__ == "__main__":
    print("=" * 80)
    print("BubbleLabs-CrewAI Bridge Persistence Tests")
    print("=" * 80)
    print()

    results = run_persistence_tests()

    print()
    print("=" * 80)
    print("Test Results Summary")
    print("=" * 80)
    print(f"Status: {results['status']}")
    if results['status'] == 'completed':
        print(f"Tests Run: {results['tests_run']}")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")

        if results['tests_failed'] > 0:
            print()
            print("Failures:")
            for failure in results['failures']:
                print(f"  - {failure}")
            print()
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error}")
    else:
        print(f"Reason: {results.get('reason', 'Unknown')}")

    print("=" * 80)

    # Exit with appropriate code
    sys.exit(0 if results['status'] == 'completed' and results['tests_failed'] == 0 else 1)
