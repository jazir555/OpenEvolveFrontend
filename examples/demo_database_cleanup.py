"""
Database Cleanup Demonstration

This script demonstrates the automatic database cleanup functionality for
BubbleLabs analytics and CREWAI mappings databases.

Features demonstrated:
1. Creating test data of various ages
2. Manual cleanup of old data
3. Automatic cleanup (daily interval)
4. Database size monitoring
5. Cleanup statistics
6. Space reclamation (VACUUM)

Usage:
    python demo_database_cleanup.py

Author: OpenEvolve Team
Date: 2025-12-29
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from bubblelabs_analytics import BubbleLabsAnalytics, cleanup_all_databases
from bubblelabs_crewai_bridge import BubbleLabsCREWAIBridge


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def demo_analytics_cleanup():
    """Demonstrate analytics database cleanup."""
    print_section("ANALYTICS DATABASE CLEANUP DEMONSTRATION")

    # Create temporary database
    db_path = "demo_analytics.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    analytics = BubbleLabsAnalytics(db_path=db_path)

    try:
        # Create test data
        print_subsection("Creating Test Data")
        now = time.time()

        # Old workflows (should be cleaned up)
        print("\nCreating 10 old workflows (100 days old)...")
        for i in range(10):
            workflow_id = f"old-workflow-{i}"
            analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Old Workflow {i}",
                instance_id=f"old-instance-{i}"
            )

            # Manually set start_time to be old
            with analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ? WHERE workflow_id = ?
                """, (now - (100 * 86400), workflow_id))
                conn.commit()

            # Add metrics
            analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id=f"node-{i}",
                node_type="test",
                tokens_used=1000,
                execution_time=1.0,
                provider="openai",
                input_tokens=500,
                output_tokens=500
            )

            analytics.end_workflow_tracking(workflow_id, status="completed")

        # Recent workflows (should NOT be cleaned up)
        print("\nCreating 10 recent workflows (10 days old)...")
        for i in range(10):
            workflow_id = f"recent-workflow-{i}"
            analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Recent Workflow {i}",
                instance_id=f"recent-instance-{i}"
            )

            # Manually set start_time to be recent
            with analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ? WHERE workflow_id = ?
                """, (now - (10 * 86400), workflow_id))
                conn.commit()

            # Add metrics
            analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id=f"node-{i}",
                node_type="test",
                tokens_used=1000,
                execution_time=1.0,
                provider="anthropic",
                input_tokens=500,
                output_tokens=500
            )

            analytics.end_workflow_tracking(workflow_id, status="completed")

        # Show initial statistics
        print_subsection("Initial Database Statistics")
        initial_size = analytics.get_database_size()
        print(f"\nTotal workflows: {initial_size['workflow_count']}")
        print(f"Total node metrics: {initial_size['node_count']}")
        print(f"Total provider metrics: {initial_size['provider_count']}")
        print(f"Total records: {initial_size['total_records']}")
        print(f"Database file size: {initial_size['file_size_mb']:.2f} MB")

        # Show cleanup statistics
        print_subsection("Cleanup Statistics")
        cleanup_stats = analytics.get_cleanup_statistics()
        print(f"\nRetention policy: {cleanup_stats['retention_days']} days")
        print(f"Old workflows (eligible for cleanup): {cleanup_stats['old_workflows']}")
        print(f"Current database size: {cleanup_stats['current_size_mb']:.2f} MB")
        print(f"Last cleanup: {datetime.fromtimestamp(cleanup_stats['last_cleanup']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Next cleanup in: {cleanup_stats['next_cleanup_in_seconds']:.0f} seconds")

        # Run manual cleanup
        print_subsection("Running Manual Cleanup")
        print("\nCleaning up workflows older than 90 days...")
        result = analytics.cleanup_old_workflows(max_age_days=90)

        print(f"\nDeleted workflows: {result['workflows']}")
        print(f"Deleted node metrics: {result['node_metrics']}")
        print(f"Deleted provider metrics: {result['provider_metrics']}")
        print(f"Total records deleted: {result['total']}")

        # Show final statistics
        print_subsection("Final Database Statistics")
        final_size = analytics.get_database_size()
        print(f"\nTotal workflows: {final_size['workflow_count']}")
        print(f"Total node metrics: {final_size['node_count']}")
        print(f"Total provider metrics: {final_size['provider_count']}")
        print(f"Total records: {final_size['total_records']}")
        print(f"Database file size: {final_size['file_size_mb']:.2f} MB")

        # Calculate space savings
        workflows_deleted = initial_size['workflow_count'] - final_size['workflow_count']
        records_deleted = initial_size['total_records'] - final_size['total_records']
        space_saved = initial_size['file_size_mb'] - final_size['file_size_mb']

        print(f"\nSpace reclaimed: {space_saved:.2f} MB")
        print(f"Records removed: {records_deleted}")

        # Demonstrate failed workflow cleanup
        print_subsection("Failed Workflow Cleanup")
        print("\nCreating 5 failed workflows (100 days old)...")

        for i in range(5):
            workflow_id = f"failed-workflow-{i}"
            analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Failed Workflow {i}",
                instance_id=f"failed-instance-{i}"
            )

            # Set as old and failed
            with analytics.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE workflows SET start_time = ?, status = 'failed'
                    WHERE workflow_id = ?
                """, (now - (100 * 86400), workflow_id))
                conn.commit()

        print("\nCleaning up failed workflows...")
        failed_deleted = analytics.cleanup_failed_workflows(max_age_days=90)
        print(f"Deleted failed workflows: {failed_deleted}")

    finally:
        # Cleanup
        analytics.stop_cleanup_thread()
        analytics.close_all_connections()

        # Remove demo database
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"\nRemoved demo database: {db_path}")


def demo_mappings_cleanup():
    """Demonstrate mappings database cleanup."""
    print_section("MAPPINGS DATABASE CLEANUP DEMONSTRATION")

    # Create temporary database
    db_path = "demo_mappings.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    import sqlite3

    # Initialize database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflow_ticket_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_id TEXT NOT NULL,
            ticket_id TEXT NOT NULL,
            ticket_status TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(workflow_id)
        )
    """)
    conn.commit()
    conn.close()

    # Create bridge
    bridge = BubbleLabsCREWAIBridge()
    bridge._mappings_db_path = db_path

    try:
        # Create test data
        print_subsection("Creating Test Data")
        now = time.time()

        # Old mappings (should be cleaned up)
        print("\nCreating 10 old mappings (100 days old)...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for i in range(10):
            cursor.execute("""
                INSERT INTO workflow_ticket_mappings
                (workflow_id, ticket_id, ticket_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (f"old-workflow-{i}", f"OLD-TICKET-{i}", "DONE",
                 now - (100 * 86400), now - (100 * 86400)))

        conn.commit()
        conn.close()

        # Recent mappings (should NOT be cleaned up)
        print("\nCreating 10 recent mappings (10 days old)...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for i in range(10):
            cursor.execute("""
                INSERT INTO workflow_ticket_mappings
                (workflow_id, ticket_id, ticket_status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (f"recent-workflow-{i}", f"RECENT-TICKET-{i}", "TODO",
                 now - (10 * 86400), now - (10 * 86400)))

        conn.commit()
        conn.close()

        # Show initial statistics
        print_subsection("Initial Mapping Statistics")
        initial_stats = bridge.get_mapping_stats()
        print(f"\nTotal mappings: {initial_stats['total_mappings']}")
        print(f"By status: {json.dumps(initial_stats['by_status'], indent=2)}")
        print(f"Oldest mapping: {initial_stats['oldest_mapping']}")
        print(f"Newest mapping: {initial_stats['newest_mapping']}")

        # Run manual cleanup
        print_subsection("Running Manual Cleanup")
        print("\nCleaning up mappings older than 90 days...")
        deleted_count = bridge.cleanup_old_mappings(max_age_days=90)

        print(f"\nDeleted mappings: {deleted_count}")

        # Show final statistics
        print_subsection("Final Mapping Statistics")
        final_stats = bridge.get_mapping_stats()
        print(f"\nTotal mappings: {final_stats['total_mappings']}")
        print(f"By status: {json.dumps(final_stats['by_status'], indent=2)}")
        print(f"Oldest mapping: {final_stats['oldest_mapping']}")
        print(f"Newest mapping: {final_stats['newest_mapping']}")

        print(f"\nMappings removed: {initial_stats['total_mappings'] - final_stats['total_mappings']}")

    finally:
        # Cleanup
        bridge.stop_background_sync()

        # Remove demo database
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"\nRemoved demo database: {db_path}")


def demo_auto_cleanup():
    """Demonstrate automatic cleanup."""
    print_section("AUTOMATIC CLEANUP DEMONSTRATION")

    # Create temporary database
    db_path = "demo_auto_cleanup.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    analytics = BubbleLabsAnalytics(db_path=db_path)

    try:
        print_subsection("Creating Old Test Data")
        now = time.time()

        # Create old workflow
        analytics.start_workflow_tracking(
            workflow_id="old-workflow",
            workflow_name="Old Workflow",
            instance_id="old-instance"
        )

        # Set as old
        with analytics.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE workflows SET start_time = ? WHERE workflow_id = ?
            """, (now - (100 * 86400), "old-workflow"))
            conn.commit()

        analytics.end_workflow_tracking("old-workflow", status="completed")

        print(f"\nTotal workflows: 1")

        # Force auto cleanup by setting last_cleanup to past
        print_subsection("Triggering Automatic Cleanup")
        print("\nSetting last_cleanup to 2 days ago...")
        analytics._last_cleanup = time.time() - (86400 * 2)

        print("\nCalling auto_cleanup_if_needed()...")
        analytics.auto_cleanup_if_needed()

        # Verify cleanup happened
        print_subsection("After Automatic Cleanup")
        final_size = analytics.get_database_size()
        print(f"\nTotal workflows: {final_size['workflow_count']}")
        print(f"Cleanup ran successfully: {'YES' if final_size['workflow_count'] == 0 else 'NO'}")

    finally:
        # Cleanup
        analytics.stop_cleanup_thread()
        analytics.close_all_connections()

        # Remove demo database
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"\nRemoved demo database: {db_path}")


def demo_cleanup_all_databases():
    """Demonstrate cleanup of all databases."""
    print_section("CLEANUP ALL DATABASES DEMONSTRATION")

    # Create temporary directory for demo
    demo_dir = "demo_cleanup_all"
    if os.path.exists(demo_dir):
        import shutil
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)

    try:
        # Create test databases
        print_subsection("Creating Test Databases")

        # Create analytics database
        analytics = BubbleLabsAnalytics(db_path=os.path.join(demo_dir, "bubblelabs_analytics.db"))
        analytics.start_workflow_tracking(
            workflow_id="test-workflow",
            workflow_name="Test",
            instance_id="test-instance"
        )
        analytics.close_all_connections()
        print("\nCreated analytics database")

        # Create mappings database
        import sqlite3
        conn = sqlite3.connect(os.path.join(demo_dir, "crewai_workflow_mappings.db"))
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
        print("Created mappings database")

        # Run cleanup
        print_subsection("Running Cleanup on All Databases")
        print("\nCalling cleanup_all_databases()...")
        results = cleanup_all_databases(base_path=demo_dir, retention_days=90)

        # Show results
        print_subsection("Cleanup Results")
        print(f"\nAnalytics cleanup: {json.dumps(results.get('analytics', {}), indent=2)}")
        print(f"\nMappings cleanup: {json.dumps(results.get('mappings', {}), indent=2)}")

    finally:
        # Remove demo directory
        import shutil
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
            print(f"\nRemoved demo directory: {demo_dir}")


def main():
    """Run all demonstrations."""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  DATABASE CLEANUP DEMONSTRATION".center(68) + "  *")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    print("\nThis demonstration shows the automatic database cleanup functionality")
    print("that prevents unbounded growth of BubbleLabs databases.")

    # Run demonstrations
    demo_analytics_cleanup()
    demo_mappings_cleanup()
    demo_auto_cleanup()
    demo_cleanup_all_databases()

    # Summary
    print_section("SUMMARY")
    print("\nDatabase cleanup features demonstrated:")
    print("  [OK] Manual cleanup of old workflows")
    print("  [OK] Manual cleanup of old mappings")
    print("  [OK] Automatic cleanup (daily interval)")
    print("  [OK] Database size monitoring")
    print("  [OK] Cleanup statistics")
    print("  [OK] Space reclamation (VACUUM)")
    print("  [OK] Cleanup of all databases")

    print("\nKey benefits:")
    print("  * Prevents unbounded database growth")
    print("  * 90-day retention policy for data")
    print("  * Automatic daily cleanup")
    print("  * Manual cleanup on-demand")
    print("  * Comprehensive monitoring and statistics")

    print("\n" + "=" * 70)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
