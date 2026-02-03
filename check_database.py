"""Quick check of the mappings database."""

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Check Database
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import enterprise_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

import sqlite3
import os
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# **ACTUAL INTEGRATION HELPER METHODS**: Check Database
def _trigger_db_check_alerts(operation, success, check_id=None, error=None, metadata=None):
    """Trigger alerts for database check operations"""
    if not ALERTING_AVAILABLE:
        return

    try:
        alert_mgr = get_alert_manager()
        if success:
            return  # No alerts for successful operations

        alert_mgr.trigger_alert(
            title=f"DB Check {operation} Failed",
            message=f"Database check operation '{operation}' failed: {error}",
            severity=AlertSeverity.MEDIUM,
            source="CheckDatabase",
            metadata=metadata or {"check_id": check_id, "operation": operation}
        )
    except Exception as e:
        logger.warning(f"Failed to trigger DB check alert: {e}")


def _extract_db_check_knowledge(operation, check_id, result):
    """Extract knowledge from database check operations"""
    if not KNOWLEDGE_AVAILABLE:
        return

    try:
        artifact = KnowledgeArtifact(
            artifact_id=f"db_check_{operation}_{check_id}",
            artifact_type="database_check",
            source_component="CheckDatabase",
            content={
                "operation": operation,
                "check_id": check_id,
                "database_exists": result.get("database_exists", False),
                "total_mappings": result.get("total_mappings", 0),
                "success": result.get("success", False),
            },
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        enterprise_knowledge_engine.store_artifact(artifact)
    except Exception as e:
        logger.warning(f"Failed to extract DB check knowledge: {e}")


def _track_db_check_performance(operation, success, duration_seconds, table_count=0):
    """Track performance of database check operations"""
    if not ADAPTIVE_AVAILABLE:
        return

    try:
        tracker = StrategyPerformanceTracker.get_instance()
        data = StrategyPerformanceData(
            strategy_name="database_check",
            component_name="CheckDatabase",
            operation_name=operation,
            success=success,
            duration_seconds=duration_seconds,
            metadata={
                "table_count": table_count
            }
        )
        tracker.record_execution(data)
    except Exception as e:
        logger.warning(f"Failed to track DB check performance: {e}")

db_path = "crewai_workflow_mappings.db"

start_time = time.time()
success = False
check_id = f"db_check_{int(time.time()) % 10000:04d}"

try:
    if os.path.exists(db_path):
        print(f"Database file exists: {db_path}")
        print(f"File size: {os.path.getsize(db_path)} bytes")
        print()

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        print("Database Schema:")
        print("=" * 80)
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        table_count = len(tables)
        for table in tables:
            print(table[0])
            print()

        print("Indexes:")
        print("=" * 80)
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        indexes = cursor.fetchall()
        for idx in indexes:
            print(idx[0])
            print()

        print("Current Data:")
        print("=" * 80)
        cursor.execute("SELECT COUNT(*) FROM workflow_ticket_mappings")
        count = cursor.fetchone()[0]
        print(f"Total mappings: {count}")

        if count > 0:
            cursor.execute("SELECT workflow_id, ticket_id, ticket_status, created_at, updated_at FROM workflow_ticket_mappings LIMIT 5")
            rows = cursor.fetchall()
            print("\nRecent mappings:")
            for row in rows:
                workflow_id, ticket_id, ticket_status, created_at, updated_at = row
                print(f"  {workflow_id} -> {ticket_id} ({ticket_status})")

        conn.close()

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance
        success = True
        duration = time.time() - start_time
        result = {
            "database_exists": True,
            "total_mappings": count,
            "success": True
        }
        _extract_db_check_knowledge("check_database", check_id, result)
        _track_db_check_performance("check_database", True, duration, table_count)

    else:
        print(f"Database file not found: {db_path}")
        success = True
        duration = time.time() - start_time
        result = {
            "database_exists": False,
            "total_mappings": 0,
            "success": True
        }
        _extract_db_check_knowledge("check_database", check_id, result)
        _track_db_check_performance("check_database", True, duration, 0)

except Exception as e:
    duration = time.time() - start_time
    # **ACTUAL INTEGRATION**: Trigger alert and track failure
    _trigger_db_check_alerts("check_database", False, check_id, str(e))
    _track_db_check_performance("check_database", False, duration, 0)
    print(f"\nError during database check: {e}")
