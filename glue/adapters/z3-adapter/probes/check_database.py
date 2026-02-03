#!/usr/bin/env python3
"""
Z3 Database Probe Script (Python Version)

Purpose: Verify Z3 database connectivity and data integrity
Compliance: Law of Runtime Truth - verify before implementation

Environment Variables Required:
  DATABASE_URL        - Path to SQLite database file (default: ./z3_knowledge.db)
  TIMEOUT_MS          - Query timeout in milliseconds (default: 5000)

Exit Codes:
  0 - All database checks passed
  1 - Required environment variable missing
  2 - Database file not found
  3 - Database not readable
  5 - Database schema invalid
  6 - Query execution failed

Author: OpenEvolve Federation
Created: 2026-02-03
"""

import json
import os
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


# =============================================================================
# Configuration (from environment variables)
# =============================================================================

DATABASE_URL = os.environ.get('DATABASE_URL', './z3_knowledge.db')
TIMEOUT_MS = int(os.environ.get('TIMEOUT_MS', '5000'))
TIMEOUT_SEC = TIMEOUT_MS / 1000.0


# =============================================================================
# Utility Functions
# =============================================================================

def log_json(level: str, msg: str, **kwargs):
    """Log JSON Lines output."""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    log_entry = {
        'level': level,
        'msg': msg,
        'timestamp': timestamp,
        'probe': 'check_database.py',
        **kwargs
    }
    print(json.dumps(log_entry))


def sql_query(query: str, db_path: str) -> tuple[bool, str | list]:
    """Execute SQL query with timeout."""
    try:
        conn = sqlite3.connect(db_path, timeout=TIMEOUT_SEC)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        return True, result
    except sqlite3.Error as e:
        return False, str(e)


# =============================================================================
# Probe Functions
# =============================================================================

def probe_file_exists() -> bool:
    """Probe 1: Database File Exists."""
    log_json('info', f'Checking database file: {DATABASE_URL}')

    db_path = Path(DATABASE_URL)
    if not db_path.exists():
        log_json('error', f'Database file not found: {DATABASE_URL}')
        return False

    log_json('info', 'Database file exists')

    # Check file size
    size = db_path.stat().st_size
    log_json('info', f'Database file size: {size} bytes')

    return True


def probe_readable() -> bool:
    """Probe 2: Database Is Readable."""
    log_json('info', 'Testing database readability')

    db_path = Path(DATABASE_URL)
    if not os.access(db_path, os.R_OK):
        log_json('error', 'Database file is not readable')
        return False

    # Try to open database
    success, result = sql_query("PRAGMA integrity_check;", DATABASE_URL)

    if not success:
        log_json('error', f'Database integrity check failed: {result}')
        return False

    if result and result[0][0] != 'ok':
        log_json('error', f'Database integrity check returned: {result[0][0]}')
        return False

    log_json('info', 'Database is readable and passes integrity check')
    return True


def probe_schema() -> bool:
    """Probe 3: Schema Validation."""
    log_json('info', 'Validating database schema')

    # Check for expected tables
    success, result = sql_query(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;",
        DATABASE_URL
    )

    if not success:
        log_json('error', f'Failed to query schema: {result}')
        return False

    tables = [row[0] for row in result]
    log_json('info', f'Found tables: {", ".join(tables)}')

    # Check for solver_results table
    has_solver_results = any('solver_results' in t.lower() for t in tables)
    if has_solver_results:
        log_json('info', 'Table solver_results exists')

        # Get table structure
        success, columns = sql_query("PRAGMA table_info(solver_results);", DATABASE_URL)
        if success:
            col_names = [col[1] for col in columns]
            log_json('info', f'solver_results columns: {", ".join(col_names)}')
    else:
        log_json('warn', 'Expected table solver_results not found (database may be empty)')

    # Check for theorem_proofs table
    has_theorem_proofs = any('theorem_proofs' in t.lower() for t in tables)
    if has_theorem_proofs:
        log_json('info', 'Table theorem_proofs exists')

        # Get table structure
        success, columns = sql_query("PRAGMA table_info(theorem_proofs);", DATABASE_URL)
        if success:
            col_names = [col[1] for col in columns]
            log_json('info', f'theorem_proofs columns: {", ".join(col_names)}')
    else:
        log_json('warn', 'Expected table theorem_proofs not found (database may be empty)')

    return True


def probe_query_test() -> bool:
    """Probe 4: Data Query Test."""
    log_json('info', 'Testing database query operations')

    # Test basic SELECT query
    success, result = sql_query(
        "SELECT COUNT(*) as count FROM sqlite_master WHERE type='table';",
        DATABASE_URL
    )

    if not success:
        log_json('error', f'Query test failed: {result}')
        return False

    table_count = result[0][0] if result else 0
    log_json('info', f'Database contains {table_count} tables')

    # If solver_results table exists, check row count
    success, result = sql_query(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='solver_results';",
        DATABASE_URL
    )

    if success and result and result[0][0] == 1:
        success, row_count = sql_query("SELECT COUNT(*) FROM solver_results;", DATABASE_URL)
        if success:
            log_json('info', f'solver_results table contains {row_count[0][0]} rows')

    return True


# =============================================================================
# Main Execution
# =============================================================================

def main():
    log_json('info', 'Starting Z3 database probe')
    log_json('info', f'Database URL: {DATABASE_URL}')
    log_json('info', f'Timeout: {TIMEOUT_MS}ms')

    # Validate environment
    if not DATABASE_URL:
        log_json('error', 'DATABASE_URL environment variable is not set')
        sys.exit(1)

    # Run probes sequentially (fail fast on first error)
    if not probe_file_exists():
        log_json('error', 'File existence probe failed')
        sys.exit(2)

    if not probe_readable():
        log_json('error', 'Readability probe failed')
        sys.exit(3)

    if not probe_schema():
        log_json('error', 'Schema validation probe failed')
        sys.exit(5)

    if not probe_query_test():
        log_json('error', 'Query test probe failed')
        sys.exit(6)

    # All probes passed
    log_json('info', 'All Z3 database probes passed successfully')
    sys.exit(0)


if __name__ == '__main__':
    main()
