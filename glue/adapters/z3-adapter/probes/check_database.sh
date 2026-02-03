#!/bin/bash
##############################################################################
# Z3 Database Probe Script
#
# Purpose: Verify Z3 database connectivity and data integrity
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   DATABASE_URL        - Path to SQLite database file (default: ./z3_knowledge.db)
#   TIMEOUT_MS          - Query timeout in milliseconds (default: 5000)
#
# Exit Codes:
#   0 - All database checks passed
#   1 - Required environment variable missing
#   2 - Database file not found
#   3 - Database not readable
#   4 - sqlite3 not available
#   5 - Database schema invalid
#   6 - Query execution failed
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

DATABASE_URL="${DATABASE_URL:-./z3_knowledge.db}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# =============================================================================
# Utility Functions
# =============================================================================

# Log JSON Lines output
log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_database.sh\"}"
}

# Check if sqlite3 is available
check_sqlite3() {
    if ! command -v sqlite3 &> /dev/null; then
        log_json "error" "sqlite3 is not installed or not in PATH"
        exit 4
    fi
}

# Execute SQL query with timeout
sql_query() {
    local query="$1"

    # Run sqlite3 with timeout
    timeout "$TIMEOUT_SEC" sqlite3 "$DATABASE_URL" "$query" 2>&1 || {
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            log_json "error" "Query timeout after ${TIMEOUT_MS}ms"
            return 1
        fi
        return $exit_code
    }
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Database File Exists
probe_file_exists() {
    log_json "info" "Checking database file: $DATABASE_URL"

    if [ ! -f "$DATABASE_URL" ]; then
        log_json "error" "Database file not found: $DATABASE_URL"
        return 1
    fi

    log_json "info" "Database file exists"

    # Check file size
    local size
    size=$(stat -f%z "$DATABASE_URL" 2>/dev/null || stat -c%s "$DATABASE_URL" 2>/dev/null || echo "0")
    log_json "info" "Database file size: ${size} bytes"

    return 0
}

# Probe 2: Database Is Readable
probe_readable() {
    log_json "info" "Testing database readability"

    if [ ! -r "$DATABASE_URL" ]; then
        log_json "error" "Database file is not readable"
        return 1
    fi

    # Try to open database
    local result
    result=$(sql_query "PRAGMA integrity_check;" 2>&1)

    if [ $? -ne 0 ]; then
        log_json "error" "Database integrity check failed: $result"
        return 1
    fi

    if [ "$result" != "ok" ]; then
        log_json "error" "Database integrity check returned: $result"
        return 1
    fi

    log_json "info" "Database is readable and passes integrity check"
    return 0
}

# Probe 3: Schema Validation
probe_schema() {
    log_json "info" "Validating database schema"

    # Check for expected tables
    local tables
    tables=$(sql_query "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;" 2>&1)

    if [ $? -ne 0 ]; then
        log_json "error" "Failed to query schema: $tables"
        return 1
    fi

    log_json "info" "Found tables: $tables"

    # Check for solver_results table (expected in Z3 integration)
    local has_solver_results
    has_solver_results=$(echo "$tables" | grep -c "solver_results" || echo "0")

    if [ "$has_solver_results" -eq "0" ]; then
        log_json "warn" "Expected table 'solver_results' not found (database may be empty)"
    else
        log_json "info" "Table 'solver_results' exists"

        # Check table structure
        local columns
        columns=$(sql_query "PRAGMA table_info(solver_results);" 2>&1)
        log_json "info" "solver_results columns: $columns"
    fi

    # Check for theorem_proofs table (expected in Z3 integration)
    local has_theorem_proofs
    has_theorem_proofs=$(echo "$tables" | grep -c "theorem_proofs" || echo "0")

    if [ "$has_theorem_proofs" -eq "0" ]; then
        log_json "warn" "Expected table 'theorem_proofs' not found (database may be empty)"
    else
        log_json "info" "Table 'theorem_proofs' exists"

        # Check table structure
        local columns
        columns=$(sql_query "PRAGMA table_info(theorem_proofs);" 2>&1)
        log_json "info" "theorem_proofs columns: $columns"
    fi

    return 0
}

# Probe 4: Data Query Test
probe_query_test() {
    log_json "info" "Testing database query operations"

    # Test basic SELECT query
    local result
    result=$(sql_query "SELECT COUNT(*) as count FROM sqlite_master WHERE type='table';" 2>&1)

    if [ $? -ne 0 ]; then
        log_json "error" "Query test failed: $result"
        return 1
    fi

    local table_count
    table_count=$(echo "$result" | tr -d '[:space:]')
    log_json "info" "Database contains $table_count tables"

    # If solver_results table exists, check row count
    local has_solver_results
    has_solver_results=$(sql_query "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='solver_results';" 2>&1 | tr -d '[:space:]')

    if [ "$has_solver_results" = "1" ]; then
        local row_count
        row_count=$(sql_query "SELECT COUNT(*) FROM solver_results;" 2>&1 | tr -d '[:space:]')
        log_json "info" "solver_results table contains $row_count rows"
    fi

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Z3 database probe"
    log_json "info" "Database URL: $DATABASE_URL"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check prerequisites
    check_sqlite3

    # Validate environment
    if [ -z "$DATABASE_URL" ]; then
        log_json "error" "DATABASE_URL environment variable is not set"
        exit 1
    fi

    # Run probes sequentially (fail fast on first error)
    if ! probe_file_exists; then
        log_json "error" "File existence probe failed"
        exit 2
    fi

    if ! probe_readable; then
        log_json "error" "Readability probe failed"
        exit 3
    fi

    if ! probe_schema; then
        log_json "error" "Schema validation probe failed"
        exit 5
    fi

    if ! probe_query_test; then
        log_json "error" "Query test probe failed"
        exit 6
    fi

    # All probes passed
    log_json "info" "All Z3 database probes passed successfully"
    exit 0
}

# Run main function
main "$@"
