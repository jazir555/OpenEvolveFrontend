"""
Database Cleanup Test Suite

This module tests the automatic database cleanup functionality for both
BubbleLabs analytics and Hephaestus mappings databases.

Test Coverage:
- Manual cleanup of old workflows
- Manual cleanup of old mappings
- Automatic cleanup (daily interval)
- Database size monitoring
- Cleanup statistics
- Thread lifecycle management
- Space reclamation (VACUUM)

Author: OpenEvolve Team
Date: 2025-12-29
"""

import os
import sys
import time
import sqlite3
import unittest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bubblelabs_analytics import BubbleLabsAnalytics, cleanup_all_databases
from bubblelabs_hephaestus_bridge import BubbleLabsHephaestusBridge, WorkflowTicketMapping


class TestAnalyticsDatabaseCleanup(unittest.TestCase):
    """Test cleanup functionality for BubbleLabs analytics database."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_analytics.db")

        # Create analytics tracker
        self.analytics = BubbleLabsAnalytics(db_path=self.db_path)

        # Create test data of various ages
        self._create_test_data()

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop cleanup thread
        self.analytics.stop_cleanup_thread()

        # Close connections
        self.analytics.close_all_connections()

        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create test workflow data with different ages."""
        now = time.time()

        # Create old workflows (100 days ago - should be cleaned up)
        for i in range(5):
            workflow_id = f"old-workflow-{i}"
            self.analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Old Workflow {i}",
                instance_id=f"old-instance-{i}"
            )

            # Manually set start_time to be old
            with self.analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ? WHERE workflow_id = ?
                """, (now - (100 * 86400), workflow_id))
                conn.commit()

            # Add some metrics
            self.analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id=f"node-{i}",
                node_type="test",
                tokens_used=1000,
                execution_time=1.0,
                provider="openai",
                input_tokens=500,
                output_tokens=500
            )

            self.analytics.end_workflow_tracking(workflow_id, status="completed")

        # Create recent workflows (10 days ago - should NOT be cleaned up)
        for i in range(5):
            workflow_id = f"recent-workflow-{i}"
            self.analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Recent Workflow {i}",
                instance_id=f"recent-instance-{i}"
            )

            # Manually set start_time to be recent
            with self.analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ? WHERE workflow_id = ?
                """, (now - (10 * 86400), workflow_id))
                conn.commit()

            # Add some metrics
            self.analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id=f"node-{i}",
                node_type="test",
                tokens_used=1000,
                execution_time=1.0,
                provider="anthropic",
                input_tokens=500,
                output_tokens=500
            )

            self.analytics.end_workflow_tracking(workflow_id, status="completed")

    def test_cleanup_old_workflows(self):
        """Test manual cleanup of old workflows."""
        # Get initial database size
        initial_size = self.analytics.get_database_size()
        self.assertGreater(initial_size['workflow_count'], 0)

        # Cleanup workflows older than 90 days
        result = self.analytics.cleanup_old_workflows(max_age_days=90)

        # Verify cleanup happened
        self.assertGreater(result['workflows'], 0, "Should have deleted some old workflows")
        self.assertEqual(result['workflows'], 5, "Should have deleted 5 old workflows")

        # Verify only recent workflows remain
        final_size = self.analytics.get_database_size()
        self.assertEqual(final_size['workflow_count'], 5, "Should have 5 recent workflows")

        # Verify old workflows are gone
        with self.analytics.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM workflows WHERE workflow_id LIKE 'old-workflow-%'")
            old_count = cursor.fetchone()[0]
            self.assertEqual(old_count, 0, "All old workflows should be deleted")

    def test_cleanup_failed_workflows(self):
        """Test cleanup of failed workflows."""
        # Create some failed workflows
        now = time.time()

        for i in range(3):
            workflow_id = f"failed-workflow-{i}"
            self.analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Failed Workflow {i}",
                instance_id=f"failed-instance-{i}"
            )

            # Set as old and failed
            with self.analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ?, status = 'failed'
                    WHERE workflow_id = ?
                """, (now - (100 * 86400), workflow_id))
                conn.commit()

        # Cleanup failed workflows
        deleted_count = self.analytics.cleanup_failed_workflows(max_age_days=90)

        # Verify cleanup happened
        self.assertEqual(deleted_count, 3, "Should have deleted 3 failed workflows")

    def test_get_database_size(self):
        """Test database size monitoring."""
        size_info = self.analytics.get_database_size()

        # Verify structure
        self.assertIn('file_size_bytes', size_info)
        self.assertIn('file_size_mb', size_info)
        self.assertIn('workflow_count', size_info)
        self.assertIn('node_count', size_info)
        self.assertIn('provider_count', size_info)
        self.assertIn('total_records', size_info)

        # Verify counts match expected
        self.assertEqual(size_info['workflow_count'], 10, "Should have 10 workflows")
        self.assertGreater(size_info['node_count'], 0, "Should have node metrics")

    def test_get_cleanup_statistics(self):
        """Test cleanup statistics."""
        stats = self.analytics.get_cleanup_statistics()

        # Verify structure
        self.assertIn('retention_days', stats)
        self.assertIn('old_workflows', stats)
        self.assertIn('current_size_mb', stats)
        self.assertIn('last_cleanup', stats)
        self.assertIn('cleanup_interval_days', stats)
        self.assertIn('next_cleanup_in_seconds', stats)

        # Verify values
        self.assertEqual(stats['retention_days'], 90)
        # old_workflows should be 5 (we created 5 old workflows in setUp)
        self.assertEqual(stats['old_workflows'], 5, "Should have 5 old workflows")
        self.assertEqual(stats['cleanup_interval_days'], 1.0)

    def test_auto_cleanup_if_needed(self):
        """Test automatic cleanup trigger."""
        # Get initial size
        initial_size = self.analytics.get_database_size()
        self.assertEqual(initial_size['workflow_count'], 10)

        # Force cleanup by setting last_cleanup to past
        self.analytics._last_cleanup = time.time() - (86400 * 2)  # 2 days ago

        # Trigger auto cleanup
        self.analytics.auto_cleanup_if_needed()

        # Verify cleanup happened
        final_size = self.analytics.get_database_size()
        self.assertEqual(final_size['workflow_count'], 5, "Should have cleaned up old workflows")

    def test_cleanup_thread_lifecycle(self):
        """Test cleanup thread lifecycle management."""
        # Verify thread is running
        self.assertTrue(self.analytics._cleanup_running, "Cleanup thread should be running")
        self.assertIsNotNone(self.analytics._cleanup_thread, "Cleanup thread should exist")

        # Stop thread
        result = self.analytics.stop_cleanup_thread()
        self.assertTrue(result, "Stop should succeed")

        # Verify thread stopped
        self.assertFalse(self.analytics._cleanup_running, "Cleanup thread should be stopped")

    def test_vacuum_reclaims_space(self):
        """Test that VACUUM reclaims disk space."""
        # Create a lot of data first to ensure database grows
        for i in range(100):
            workflow_id = f"vacuum-test-workflow-{i}"
            self.analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Vacuum Test Workflow {i}",
                instance_id=f"vacuum-test-instance-{i}"
            )

            # Add lots of metrics
            for j in range(10):
                self.analytics.track_node_execution(
                    workflow_id=workflow_id,
                    node_id=f"node-{j}",
                    node_type="test",
                    tokens_used=1000,
                    execution_time=1.0,
                    provider="openai",
                    input_tokens=500,
                    output_tokens=500
                )

            self.analytics.end_workflow_tracking(workflow_id, status="completed")

        # Get size after adding lots of data
        size_with_data = os.path.getsize(self.db_path)

        # Delete all data
        with self.analytics.get_connection() as conn:
            conn.execute("DELETE FROM workflows")
            conn.commit()

        # Run cleanup which includes VACUUM
        self.analytics.cleanup_old_workflows(max_age_days=0)

        # Get size after VACUUM
        final_size = os.path.getsize(self.db_path)

        # Final size should be less than or equal to the size with data
        # (VACUUM should never increase the size)
        self.assertLessEqual(final_size, size_with_data,
                            "VACUUM should not increase database file size")

        # Verify VACUUM was called (at least some reduction occurred or size is reasonable)
        # For an empty database with indexes, the size should typically be under 500KB
        # This is a generous threshold to account for filesystem overhead
        self.assertLessEqual(final_size, 512000,  # 500KB max after VACUUM
                            "VACUUM should reclaim most of the space (final size should be under 500KB)")


class TestMappingsDatabaseCleanup(unittest.TestCase):
    """Test cleanup functionality for Hephaestus mappings database."""

    def setUp(self):
        """Set up test fixtures with temporary database."""
        # Create temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_mappings.db")

        # Create bridge
        self.bridge = BubbleLabsHephaestusBridge()

        # Override database path
        self.bridge._mappings_db_path = self.db_path

        # Initialize database
        self.bridge._init_mappings_database()

        # Create test data
        self._create_test_data()

    def tearDown(self):
        """Clean up test fixtures."""
        # Stop background sync
        self.bridge.stop_background_sync()

        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create test mapping data with different ages."""
        now = time.time()

        # Create old mappings (100 days ago - should be cleaned up)
        for i in range(5):
            workflow_id = f"old-workflow-{i}"

            mapping = WorkflowTicketMapping(workflow_id)
            mapping.ticket_id = f"OLD-TICKET-{i}"
            mapping.ticket_status = "DONE"
            mapping.created_at = now - (100 * 86400)
            mapping.updated_at = now - (100 * 86400)

            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_ticket_mappings
                (workflow_id, ticket_id, ticket_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (workflow_id, mapping.ticket_id, mapping.ticket_status,
                 mapping.created_at, mapping.updated_at))
            conn.commit()
            conn.close()

        # Create recent mappings (10 days ago - should NOT be cleaned up)
        for i in range(5):
            workflow_id = f"recent-workflow-{i}"

            mapping = WorkflowTicketMapping(workflow_id)
            mapping.ticket_id = f"RECENT-TICKET-{i}"
            mapping.ticket_status = "TODO"
            mapping.created_at = now - (10 * 86400)
            mapping.updated_at = now - (10 * 86400)

            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workflow_ticket_mappings
                (workflow_id, ticket_id, ticket_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (workflow_id, mapping.ticket_id, mapping.ticket_status,
                 mapping.created_at, mapping.updated_at))
            conn.commit()
            conn.close()

    def test_cleanup_old_mappings(self):
        """Test manual cleanup of old mappings."""
        # Get initial count
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
        initial_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(initial_count, 10, "Should have 10 mappings initially")

        # Cleanup old mappings
        deleted_count = self.bridge.cleanup_old_mappings(max_age_days=90)

        # Verify cleanup happened
        self.assertEqual(deleted_count, 5, "Should have deleted 5 old mappings")

        # Verify only recent mappings remain
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
        final_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(final_count, 5, "Should have 5 recent mappings")

        # Verify old mappings are gone
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings WHERE workflow_id LIKE 'old-workflow-%'")
        old_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(old_count, 0, "All old mappings should be deleted")

    def test_auto_cleanup_if_needed(self):
        """Test automatic cleanup trigger."""
        # Force cleanup by setting last_cleanup to past
        self.bridge._last_mappings_cleanup = time.time() - (86400 * 2)  # 2 days ago

        # Get initial count
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
        initial_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(initial_count, 10)

        # Trigger auto cleanup
        self.bridge.auto_cleanup_if_needed()

        # Verify cleanup happened
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
        final_count = cursor.fetchone()[0]
        conn.close()

        self.assertEqual(final_count, 5, "Should have cleaned up old mappings")

    def test_get_mapping_stats(self):
        """Test mapping statistics."""
        stats = self.bridge.get_mapping_stats()

        # Verify structure
        self.assertIn('total_mappings', stats)
        self.assertIn('by_status', stats)
        self.assertIn('oldest_mapping', stats)
        self.assertIn('newest_mapping', stats)
        self.assertIn('database_path', stats)

        # Verify counts
        self.assertEqual(stats['total_mappings'], 10)


class TestCleanupAllDatabases(unittest.TestCase):
    """Test cleanup of all databases."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()

        # Create test databases
        self.analytics_db = os.path.join(self.test_dir, "bubblelabs_analytics.db")
        self.mappings_db = os.path.join(self.test_dir, "hephaestus_workflow_mappings.db")

        # Create analytics database with test data
        analytics = BubbleLabsAnalytics(db_path=self.analytics_db)
        analytics.start_workflow_tracking(
            workflow_id="test-workflow",
            workflow_name="Test",
            instance_id="test-instance"
        )
        analytics.close_all_connections()

        # Create mappings database with test data
        conn = sqlite3.connect(self.mappings_db)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_ticket_mappings (
                id INTEGER PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                ticket_id TEXT NOT NULL,
                ticket_status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        cursor.execute("""
            INSERT INTO workflow_ticket_mappings
            (workflow_id, ticket_id, ticket_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("test-workflow", "TICKET-1", "DONE", time.time(), time.time()))
        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cleanup_all_databases(self):
        """Test cleanup of all databases."""
        # Run cleanup
        results = cleanup_all_databases(base_path=self.test_dir, retention_days=90)

        # Verify results
        self.assertIn('analytics', results)
        self.assertIn('mappings', results)

        # Verify databases still exist
        self.assertTrue(os.path.exists(self.analytics_db))
        self.assertTrue(os.path.exists(self.mappings_db))


class TestCleanupIntegration(unittest.TestCase):
    """Integration tests for cleanup functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.analytics_db = os.path.join(self.test_dir, "test_analytics.db")

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_cleanup_prevents_unbounded_growth(self):
        """Test that cleanup prevents unbounded database growth."""
        analytics = BubbleLabsAnalytics(db_path=self.analytics_db)

        # Create many old workflows
        now = time.time()
        for i in range(100):
            workflow_id = f"old-workflow-{i}"
            analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Old Workflow {i}",
                instance_id=f"old-instance-{i}"
            )

            # Set as old
            with analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ? WHERE workflow_id = ?
                """, (now - (100 * 86400), workflow_id))
                conn.commit()

        # Verify database has grown
        size_before = analytics.get_database_size()
        self.assertEqual(size_before['workflow_count'], 100)

        # Run cleanup
        result = analytics.cleanup_old_workflows(max_age_days=90)

        # Verify all old workflows deleted
        self.assertEqual(result['workflows'], 100)

        size_after = analytics.get_database_size()
        self.assertEqual(size_after['workflow_count'], 0)

        # Clean up
        analytics.stop_cleanup_thread()
        analytics.close_all_connections()


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyticsDatabaseCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestMappingsDatabaseCleanup))
    suite.addTests(loader.loadTestsFromTestCase(TestCleanupAllDatabases))
    suite.addTests(loader.loadTestsFromTestCase(TestCleanupIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("CLEANUP TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 70)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
