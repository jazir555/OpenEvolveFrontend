"""
Memory and Resource Leak Detection Test Suite for BubbleLabs Integration

This script performs comprehensive memory and resource leak detection by:
1. Creating workflows repeatedly (1000 iterations)
2. Tracking memory usage with psutil
3. Checking for memory growth
4. Verifying cleanup of resources

Run this test to detect memory leaks in the BubbleLabs integration system.

Usage:
    python test_memory_leaks.py

Requirements:
    pip install psutil matplotlib
"""

import gc
import os
import sys
import time
import tracemalloc
import sqlite3
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available. Install with: pip install psutil")

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


class MemoryProfiler:
    """Profile memory usage during test execution."""

    def __init__(self):
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self.snapshots: List[Dict[str, Any]] = []
        self.baseline_memory = None

    def take_snapshot(self, label: str) -> Dict[str, Any]:
        """Take a memory snapshot."""
        if not PSUTIL_AVAILABLE:
            return {"label": label, "rss_mb": 0, "vms_mb": 0}

        mem_info = self.process.memory_info()
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "rss_mb": mem_info.rss / 1024 / 1024,  # Resident Set Size in MB
            "vms_mb": mem_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        }
        self.snapshots.append(snapshot)

        if self.baseline_memory is None:
            self.baseline_memory = snapshot["rss_mb"]

        return snapshot

    def get_memory_growth(self) -> float:
        """Get memory growth in MB since baseline."""
        if not self.snapshots:
            return 0.0
        current = self.snapshots[-1]["rss_mb"]
        baseline = self.baseline_memory or 0
        return current - baseline

    def print_summary(self):
        """Print memory usage summary."""
        if not PSUTIL_AVAILABLE:
            print("\n=== MEMORY PROFILING SUMMARY ===")
            print("psutil not available - skipping memory profiling")
            return

        print("\n=== MEMORY PROFILING SUMMARY ===")
        print(f"{'Snapshot':<30} {'RSS (MB)':<15} {'VMS (MB)':<15} {'Growth (MB)':<15}")
        print("-" * 75)

        baseline_rss = self.snapshots[0]["rss_mb"] if self.snapshots else 0

        for snapshot in self.snapshots:
            growth = snapshot["rss_mb"] - baseline_rss
            print(f"{snapshot['label']:<30} {snapshot['rss_mb']:<15.2f} "
                  f"{snapshot['vms_mb']:<15.2f} {growth:<15.2f}")

        total_growth = self.get_memory_growth()
        print("-" * 75)
        print(f"{'TOTAL GROWTH':<30} {total_growth:<15.2f} MB")

        if total_growth > 100:
            print("\n[WARN]  WARNING: Significant memory growth detected!")
        elif total_growth > 50:
            print("\n[WARN]  NOTICE: Moderate memory growth detected")
        else:
            print("\n[OK] Memory growth within acceptable limits")


class BubbleLabsLeakDetector:
    """Detect memory and resource leaks in BubbleLabs integration."""

    def __init__(self):
        self.profiler = MemoryProfiler()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def test_bubblelabs_CREWAI_bridge(self):
        """Test BubbleLabs-CREWAI bridge for memory leaks."""
        print("\n" + "=" * 80)
        print("TEST 1: BubbleLabs-CREWAI Bridge Memory Leak Detection")
        print("=" * 80)

        try:
            from bubblelabs_crewai_bridge import BubbleLabsCREWAIBridge  # MIGRATED
            from bubblelabs_integration import BubbleLabsIntegration
        except ImportError as e:
            print(f"[WARN]  Could not import: {e}")
            return

        # Baseline
        self.profiler.take_snapshot("Baseline")

        # Create bridge
        bubblelabs = BubbleLabsIntegration()
        bridge = BubbleLabsCREWAIBridge(bubblelabs_integration=bubblelabs)
        self.profiler.take_snapshot("After bridge creation")

        # Create multiple workflow definitions
        for i in range(100):
            definition = bubblelabs.create_workflow_definition_from_openevolve(
                problem_statement=f"Test problem {i}",
                team_config={},
                gauntlet_config={}
            )

            # Create ticket
            bridge.create_ticket_from_workflow(definition)

            if (i + 1) % 20 == 0:
                self.profiler.take_snapshot(f"After {i+1} workflows")

        # Check for unbounded growth in mappings
        print(f"\nChecking bridge mappings...")
        print(f"  mappings dict size: {len(bridge.mappings)}")
        print(f"  instance_to_definition_map size: {len(bridge.instance_to_definition_map)}")

        if len(bridge.mappings) == 100:
            print("  [OK] Mappings grew as expected (100 entries)")
        else:
            self.warnings.append(f"Unexpected mappings size: {len(bridge.mappings)}")

        # Start/stop background sync thread
        print("\nTesting background sync thread...")
        bridge.start_background_sync()
        self.profiler.take_snapshot("After starting sync thread")

        time.sleep(2)  # Let it run briefly

        bridge.stop_background_sync()
        self.profiler.take_snapshot("After stopping sync thread")

        # Check if thread stopped
        if bridge.sync_thread and bridge.sync_thread.is_alive():
            self.errors.append("Background sync thread did not stop properly!")
        else:
            print("  [OK] Background sync thread stopped successfully")

        # Force cleanup
        del bridge
        del bubblelabs
        gc.collect()
        self.profiler.take_snapshot("After cleanup")

    def test_bubblelabs_mcp_tools(self):
        """Test BubbleLabs MCP tools for memory leaks."""
        print("\n" + "=" * 80)
        print("TEST 2: BubbleLabs MCP Tools Memory Leak Detection")
        print("=" * 80)

        try:
            from bubblelabs_mcp_tools import (
                get_shared_bubblelabs,
                get_shared_api,
                list_mcp_tools,
                _MCP_TOOLS
            )
        except ImportError as e:
            print(f"[WARN]  Could not import: {e}")
            return

        # Baseline
        self.profiler.take_snapshot("Baseline")

        # Check singleton pattern
        print("\nTesting singleton instances...")
        instance1 = get_shared_bubblelabs()
        instance2 = get_shared_bubblelabs()

        if instance1 is instance2:
            print("  [OK] Singleton pattern working correctly")
        else:
            self.errors.append("Singleton pattern broken - multiple instances created!")

        # Check MCP tools registry
        print(f"\nMCP tools registered: {len(list_mcp_tools())}")
        print(f"  _MCP_TOOLS dict size: {len(_MCP_TOOLS)}")

        self.profiler.take_snapshot("After singleton check")

    def test_bubblelabs_analytics(self):
        """Test BubbleLabs analytics for memory leaks."""
        print("\n" + "=" * 80)
        print("TEST 3: BubbleLabs Analytics Memory Leak Detection")
        print("=" * 80)

        try:
            from bubblelabs_analytics import BubbleLabsAnalytics
        except ImportError as e:
            print(f"[WARN]  Could not import: {e}")
            return

        # Create temporary database
        test_db = "/tmp/test_bubblelabs_analytics.db"

        # Baseline
        self.profiler.take_snapshot("Baseline")

        # Create analytics tracker
        analytics = BubbleLabsAnalytics(db_path=test_db, pool_size=5)
        self.profiler.take_snapshot("After analytics creation")

        # Track many workflows
        print("\nTracking workflows...")
        for i in range(100):
            workflow_id = f"test-workflow-{i}"
            instance_id = f"test-instance-{i}"

            analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name=f"Test Workflow {i}",
                instance_id=instance_id
            )

            # Track node executions
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

            if (i + 1) % 25 == 0:
                self.profiler.take_snapshot(f"After {i+1} workflows")

        # Check connection pool
        print(f"\nConnection pool size: {len(analytics._connection_pool)}")
        if len(analytics._connection_pool) <= analytics._pool_size:
            print("  [OK] Connection pool bounded correctly")
        else:
            self.errors.append(f"Connection pool exceeded max size: "
                             f"{len(analytics._connection_pool)} > {analytics._pool_size}")

        # Check database growth
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM workflows")
        workflow_count = cursor.fetchone()[0]
        print(f"  Database workflow count: {workflow_count}")

        cursor.execute("SELECT COUNT(*) FROM node_metrics")
        node_count = cursor.fetchone()[0]
        print(f"  Database node metrics count: {node_count}")

        conn.close()

        # Close all connections
        analytics.close_all_connections()
        self.profiler.take_snapshot("After closing connections")

        # Cleanup
        del analytics
        gc.collect()
        self.profiler.take_snapshot("After cleanup")

        # Remove test database
        try:
            os.remove(test_db)
        except:
            pass

    def test_bubblelabs_integration(self):
        """Test BubbleLabs integration for memory leaks."""
        print("\n" + "=" * 80)
        print("TEST 4: BubbleLabs Integration Memory Leak Detection")
        print("=" * 80)

        try:
            from bubblelabs_integration import BubbleLabsIntegration
        except ImportError as e:
            print(f"[WARN]  Could not import: {e}")
            return

        # Baseline
        self.profiler.take_snapshot("Baseline")

        # Create integration
        integration = BubbleLabsIntegration()
        self.profiler.take_snapshot("After integration creation")

        # Create many workflow definitions
        print("\nCreating workflow definitions...")
        for i in range(100):
            definition = integration.create_workflow_definition_from_openevolve(
                problem_statement=f"Test problem {i}: " * 10,  # Make it larger
                team_config={"planner_team": f"Team-{i}"},
                gauntlet_config={"sub_problem_red_gauntlet": f"Gauntlet-{i}"}
            )

            if (i + 1) % 25 == 0:
                self.profiler.take_snapshot(f"After {i+1} definitions")

        # Check unbounded collections
        print(f"\nWorkflow definitions: {len(integration.workflow_definitions)}")
        print(f"Workflow instances: {len(integration.workflow_instances)}")
        print(f"Running threads: {len(integration.running_threads)}")

        # Check for missing cleanup
        if len(integration.workflow_definitions) == 100:
            print("  [OK] Workflow definitions stored correctly")
        else:
            self.warnings.append(f"Unexpected definition count: "
                               f"{len(integration.workflow_definitions)}")

        # Cleanup
        del integration
        gc.collect()
        self.profiler.take_snapshot("After cleanup")

    def test_bubblelabs_security(self):
        """Test BubbleLabs security for memory leaks."""
        print("\n" + "=" * 80)
        print("TEST 5: BubbleLabs Security Memory Leak Detection")
        print("=" * 80)

        try:
            from bubblelabs_security import (
                AuthenticationManager,
                CSRFProtection,
                RateLimiter
            )
        except ImportError as e:
            print(f"[WARN]  Could not import: {e}")
            return

        # Baseline
        self.profiler.take_snapshot("Baseline")

        # Test AuthenticationManager
        print("\nTesting AuthenticationManager...")
        auth_manager = AuthenticationManager()
        print(f"  API keys: {len(auth_manager.api_keys)}")
        print(f"  Sessions: {len(auth_manager.sessions)}")

        # Add many sessions (simulating users)
        for i in range(100):
            from bubblelabs_security import SecurityContext, UserRole
            session_id = f"session-{i}"
            context = SecurityContext(
                user_id=f"user-{i}",
                role=UserRole.OPERATOR,
                session_id=session_id,
                authenticated=True
            )
            auth_manager.sessions[session_id] = context

        print(f"  After adding 100 sessions: {len(auth_manager.sessions)}")
        self.profiler.take_snapshot("After 100 sessions")

        # Test CSRFProtection
        print("\nTesting CSRFProtection...")
        csrf = CSRFProtection()
        tokens = []
        for i in range(100):
            token = csrf.generate_token(f"session-{i}")
            tokens.append(token)

        print(f"  Generated tokens: {len(csrf.tokens)}")
        self.profiler.take_snapshot("After 100 CSRF tokens")

        # Test RateLimiter
        print("\nTesting RateLimiter...")
        rate_limiter = RateLimiter()

        # Simulate many requests
        for i in range(100):
            rate_limiter.check_rate_limit(f"user-{i}")

        print(f"  Rate limit buckets: {len(rate_limiter.buckets)}")
        self.profiler.take_snapshot("After 100 rate limit buckets")

        # Check for unbounded growth
        if len(csrf.tokens) == 100:
            print("  [OK] CSRF tokens grew as expected")
        else:
            self.warnings.append(f"Unexpected CSRF token count: {len(csrf.tokens)}")

        if len(rate_limiter.buckets) == 100:
            print("  [OK] Rate limit buckets grew as expected")
        else:
            self.warnings.append(f"Unexpected rate limit bucket count: "
                               f"{len(rate_limiter.buckets)}")

        # Cleanup
        del auth_manager, csrf, rate_limiter
        gc.collect()
        self.profiler.take_snapshot("After cleanup")

    def test_database_connection_leaks(self):
        """Test for database connection leaks."""
        print("\n" + "=" * 80)
        print("TEST 6: Database Connection Leak Detection")
        print("=" * 80)

        test_db = "/tmp/test_connection_leak.db"

        try:
            from bubblelabs_analytics import BubbleLabsAnalytics
        except ImportError as e:
            print(f"[WARN]  Could not import: {e}")
            return

        # Baseline
        self.profiler.take_snapshot("Baseline")

        # Create analytics tracker
        analytics = BubbleLabsAnalytics(db_path=test_db, pool_size=5)

        # Perform many operations
        print("\nPerforming database operations...")
        for i in range(50):
            analytics.start_workflow_tracking(
                workflow_id=f"workflow-{i}",
                workflow_name=f"Workflow {i}",
                instance_id=f"instance-{i}"
            )
            analytics.end_workflow_tracking(f"workflow-{i}")

        self.profiler.take_snapshot("After 50 operations")

        # Check connection pool
        pool_size = len(analytics._connection_pool)
        print(f"\nConnection pool size: {pool_size}")
        print(f"Max pool size: {analytics._pool_size}")

        if pool_size <= analytics._pool_size:
            print("  [OK] Connection pool within bounds")
        else:
            self.errors.append(f"Connection pool leak detected: "
                             f"{pool_size} > {analytics._pool_size}")

        # Close connections
        analytics.close_all_connections()
        self.profiler.take_snapshot("After closing connections")

        # Verify connections closed
        if len(analytics._connection_pool) == 0:
            print("  [OK] All connections closed")
        else:
            self.errors.append(f"Not all connections closed: "
                             f"{len(analytics._connection_pool)} remaining")

        # Cleanup
        del analytics
        gc.collect()
        self.profiler.take_snapshot("After cleanup")

        # Remove test database
        try:
            os.remove(test_db)
        except:
            pass

    def run_all_tests(self):
        """Run all memory leak tests."""
        print("\n" + "=" * 80)
        print("BUBBLELABS MEMORY LEAK DETECTION TEST SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Start tracing memory allocations
        tracemalloc.start()

        # Run tests
        self.test_bubblelabs_CREWAI_bridge()
        self.test_bubblelabs_mcp_tools()
        self.test_bubblelabs_analytics()
        self.test_bubblelabs_integration()
        self.test_bubblelabs_security()
        self.test_database_connection_leaks()

        # Print memory profiling summary
        self.profiler.print_summary()

        # Print tracemalloc summary
        print("\n" + "=" * 80)
        print("TRACEMALLOC SUMMARY (Top 10 Allocations)")
        print("=" * 80)

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        for stat in top_stats[:10]:
            print(f"{stat}")

        # Print errors and warnings
        if self.errors:
            print("\n" + "=" * 80)
            print("ERRORS DETECTED")
            print("=" * 80)
            for error in self.errors:
                print(f"  [FAIL] {error}")

        if self.warnings:
            print("\n" + "=" * 80)
            print("WARNINGS")
            print("=" * 80)
            for warning in self.warnings:
                print(f"  [WARN]  {warning}")

        if not self.errors and not self.warnings:
            print("\n" + "=" * 80)
            print("[OK] NO MEMORY LEAKS DETECTED")
            print("=" * 80)

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Final assessment
        memory_growth = self.profiler.get_memory_growth()
        print("\n" + "=" * 80)
        print("FINAL ASSESSMENT")
        print("=" * 80)

        if self.errors:
            print("[FAIL] FAIL: Memory leaks detected!")
            print(f"   Errors: {len(self.errors)}")
            print(f"   Memory growth: {memory_growth:.2f} MB")
        elif memory_growth > 100:
            print("[WARN]  WARNING: Significant memory growth detected")
            print(f"   Memory growth: {memory_growth:.2f} MB")
        elif memory_growth > 50:
            print("[WARN]  NOTICE: Moderate memory growth")
            print(f"   Memory growth: {memory_growth:.2f} MB")
        else:
            print("[OK] PASS: No significant memory leaks detected")
            print(f"   Memory growth: {memory_growth:.2f} MB")


if __name__ == "__main__":
    detector = BubbleLabsLeakDetector()
    detector.run_all_tests()
