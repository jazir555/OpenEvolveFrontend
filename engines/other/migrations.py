"""
Sovereign-Grade Problem Decomposition System - Database Migrations

Maps schema version numbers to the list of SQL statements required to reach
that version. The base schema (version 0) is created by
``SovereignDatabase.init_database``; additional migrations can be appended here
as the schema evolves.
"""

from typing import Dict, List

MIGRATIONS: Dict[int, List[str]] = {
    1: [
        """
        CREATE TABLE IF NOT EXISTS workflow_runs (
            workflow_id TEXT PRIMARY KEY,
            tenant_id TEXT,
            status TEXT,
            current_stage TEXT,
            problem_statement TEXT,
            workflow_type TEXT,
            start_time REAL,
            end_time REAL,
            progress REAL,
            state_blob BLOB,
            created_at TEXT,
            updated_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS workflow_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user TEXT,
            role TEXT,
            operation TEXT,
            resource TEXT,
            resource_id TEXT,
            success INTEGER,
            details_json TEXT
        )
        """,
    ],
}
