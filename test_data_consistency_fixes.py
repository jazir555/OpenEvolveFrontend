"""
Test Data Consistency Fixes

Tests for CRITICAL data consistency fixes:
- Issue 1: Foreign Key Constraints Enforced
- Issue 2: Bridge Mappings Persistence (NOT IMPLEMENTED)

Run this test to verify the fixes are working correctly.
"""

import sqlite3
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add frontend to path
sys.path.insert(0, str(Path(__file__).parent))

from bubblelabs_analytics import BubbleLabsAnalytics


def test_foreign_keys_enabled():
    """Test 1: Verify foreign keys are enabled in analytics database."""
    print("\n" + "="*70)
    print("TEST 1: Verify Foreign Keys Enabled")
    print("="*70)

    try:
        # Create analytics instance
        analytics = BubbleLabsAnalytics(db_path="test_analytics.db")

        # Check if foreign keys are enabled
        with analytics.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys")
            result = cursor.fetchone()

            if result[0] == 1:
                print("[OK] PASS: Foreign keys are ENABLED")
                print(f"   PRAGMA foreign_keys = {result[0]}")
                return True
            else:
                print("[FAIL] FAIL: Foreign keys are NOT enabled")
                print(f"   PRAGMA foreign_keys = {result[0]}")
                return False

    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False
    finally:
        # Cleanup
        try:
            analytics.close_all_connections()
            if os.path.exists("test_analytics.db"):
                os.remove("test_analytics.db")
        except:
            pass


def test_foreign_key_enforcement():
    """Test 2: Verify foreign key constraints prevent orphaned records."""
    print("\n" + "="*70)
    print("TEST 2: Verify Foreign Key Constraint Enforcement")
    print("="*70)

    try:
        # Create analytics instance
        analytics = BubbleLabsAnalytics(db_path="test_analytics.db")

        # Try to insert node_metrics without parent workflow (should fail)
        print("   Attempting to insert orphaned record...")
        with analytics.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO node_metrics (workflow_id, node_id, node_type)
                    VALUES ('fake-workflow-id', 'node1', 'test')
                """)
                conn.commit()
                print("[FAIL] FAIL: Foreign key constraint NOT enforced")
                print("   Orphaned record was created (this should not happen)")
                return False
            except sqlite3.IntegrityError as e:
                print("[OK] PASS: Foreign key constraint enforced")
                print(f"   Error: {e}")
                return True

    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False
    finally:
        # Cleanup
        try:
            analytics.close_all_connections()
            if os.path.exists("test_analytics.db"):
                os.remove("test_analytics.db")
        except:
            pass


def test_cascade_delete():
    """Test 3: Verify CASCADE delete removes child records."""
    print("\n" + "="*70)
    print("TEST 3: Verify CASCADE DELETE")
    print("="*70)

    try:
        # Create analytics instance
        analytics = BubbleLabsAnalytics(db_path="test_analytics.db")

        # Create a workflow
        workflow_id = "test-cascade-workflow"
        print(f"   Creating workflow: {workflow_id}")
        analytics.start_workflow_tracking(
            workflow_id=workflow_id,
            workflow_name="Test Workflow",
            instance_id="test-instance"
        )

        # Add node metrics
        print("   Adding node metrics...")
        analytics.track_node_execution(
            workflow_id=workflow_id,
            node_id="node1",
            node_type="test",
            tokens_used=100,
            execution_time=1.0
        )

        # Verify node_metrics exist
        with analytics.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM node_metrics WHERE workflow_id = ?", (workflow_id,))
            before_count = cursor.fetchone()[0]
            print(f"   Node metrics before delete: {before_count}")

            # Delete workflow (should cascade to node_metrics)
            print(f"   Deleting workflow: {workflow_id}")
            cursor.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            conn.commit()

            # Verify node_metrics deleted
            cursor.execute("SELECT COUNT(*) FROM node_metrics WHERE workflow_id = ?", (workflow_id,))
            after_count = cursor.fetchone()[0]
            print(f"   Node metrics after delete: {after_count}")

            if before_count > 0 and after_count == 0:
                print("[OK] PASS: CASCADE delete working correctly")
                print("   Child records were automatically deleted")
                return True
            else:
                print("[FAIL] FAIL: CASCADE delete not working")
                print(f"   Before: {before_count}, After: {after_count}")
                return False

    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            analytics.close_all_connections()
            if os.path.exists("test_analytics.db"):
                os.remove("test_analytics.db")
        except:
            pass


def test_referential_integrity():
    """Test 4: Comprehensive referential integrity test."""
    print("\n" + "="*70)
    print("TEST 4: Comprehensive Referential Integrity Test")
    print("="*70)

    try:
        # Create analytics instance
        analytics = BubbleLabsAnalytics(db_path="test_analytics.db")

        # Create workflow
        workflow_id = "test-integrity-workflow"
        print(f"   Creating workflow: {workflow_id}")
        analytics.start_workflow_tracking(
            workflow_id=workflow_id,
            workflow_name="Integrity Test Workflow",
            instance_id="test-instance"
        )

        # Add multiple node metrics
        print("   Adding node metrics...")
        for i in range(3):
            analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id=f"node{i}",
                node_type="test",
                tokens_used=100 * (i + 1),
                execution_time=1.0 * (i + 1)
            )

        # Verify records exist
        with analytics.get_connection() as conn:
            cursor = conn.cursor()

            # Check workflow
            cursor.execute("SELECT * FROM workflows WHERE workflow_id = ?", (workflow_id,))
            workflow = cursor.fetchone()
            print(f"   Workflow exists: {workflow is not None}")

            # Check node metrics
            cursor.execute("SELECT COUNT(*) FROM node_metrics WHERE workflow_id = ?", (workflow_id,))
            node_count = cursor.fetchone()[0]
            print(f"   Node metrics count: {node_count}")

            # Check provider metrics
            cursor.execute("SELECT COUNT(*) FROM provider_metrics WHERE workflow_id = ?", (workflow_id,))
            provider_count = cursor.fetchone()[0]
            print(f"   Provider metrics count: {provider_count}")

            # Delete workflow
            print(f"   Deleting workflow: {workflow_id}")
            cursor.execute("DELETE FROM workflows WHERE workflow_id = ?", (workflow_id,))
            conn.commit()

            # Verify all child records deleted
            cursor.execute("SELECT COUNT(*) FROM node_metrics WHERE workflow_id = ?", (workflow_id,))
            after_nodes = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM provider_metrics WHERE workflow_id = ?", (workflow_id,))
            after_providers = cursor.fetchone()[0]

            if after_nodes == 0 and after_providers == 0:
                print("[OK] PASS: All child records deleted via CASCADE")
                print(f"   Node metrics deleted: {node_count} -> {after_nodes}")
                print(f"   Provider metrics deleted: {provider_count} -> {after_providers}")
                return True
            else:
                print("[FAIL] FAIL: Some child records not deleted")
                print(f"   Node metrics: {node_count} -> {after_nodes}")
                print(f"   Provider metrics: {provider_count} -> {after_providers}")
                return False

    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            analytics.close_all_connections()
            if os.path.exists("test_analytics.db"):
                os.remove("test_analytics.db")
        except:
            pass


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DATA CONSISTENCY FIXES - VERIFICATION TESTS")
    print("="*70)
    print("\nTesting CRITICAL data consistency fixes in BubbleLabs Analytics")
    print("Issue 1: Foreign Key Constraints Enforced")
    print("Issue 2: Bridge Mappings Persistence (NOT TESTED - Not Implemented)")

    results = []

    # Run tests
    results.append(("Foreign Keys Enabled", test_foreign_keys_enabled()))
    results.append(("Foreign Key Enforcement", test_foreign_key_enforcement()))
    results.append(("CASCADE DELETE", test_cascade_delete()))
    results.append(("Referential Integrity", test_referential_integrity()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "-"*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Data consistency fixes are working correctly.")
        return 0
    else:
        print(f"\n[WARN]  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
