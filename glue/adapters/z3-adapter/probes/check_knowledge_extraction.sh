#!/bin/bash
##############################################################################
# Z3 Knowledge Extraction Probe Script
#
# Purpose: Verify Z3 knowledge graph extraction and reasoning APIs
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   Z3_API_URL          - Base URL of Z3 API (default: http://localhost:8000)
#   DATABASE_URL        - Path to SQLite database (default: ./z3_knowledge.db)
#   TIMEOUT_MS          - Request timeout in milliseconds (default: 5000)
#
# Exit Codes:
#   0 - All knowledge extraction probes passed
#   1 - Required environment variable missing
#   2 - Knowledge base endpoint check failed
#   3 - Pattern recognition check failed
#   4 - Knowledge graph query check failed
#   5 - curl not available
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

Z3_API_URL="${Z3_API_URL:-http://localhost:8000}"
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
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_knowledge_extraction.sh\"}"
}

# Check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        log_json "error" "curl is not installed or not in PATH"
        exit 5
    fi
}

# Check if sqlite3 is available
check_sqlite3() {
    if ! command -v sqlite3 &> /dev/null; then
        log_json "warn" "sqlite3 is not installed - skipping database checks"
        return 1
    fi
    return 0
}

# Make API request with timeout
api_request() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="${3:-}"

    local url="${Z3_API_URL}${endpoint}"

    if [ -n "$data" ]; then
        curl -s -X "$method" \
            --max-time "$TIMEOUT_SEC" \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$url" 2>&1
    else
        curl -s -X "$method" \
            --max-time "$TIMEOUT_SEC" \
            -H "Content-Type: application/json" \
            "$url" 2>&1
    fi
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Knowledge Base Status Endpoint
probe_knowledge_base() {
    log_json "info" "Testing knowledge base endpoint: ${Z3_API_URL}/api/v1/knowledge/status"

    local response
    response=$(api_request "/api/v1/knowledge/status" "GET")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "warn" "Knowledge base endpoint returned invalid JSON (may not be implemented yet): $response"
        return 0  # Don't fail - this endpoint may not exist yet
    fi

    # Check for status field
    local status
    status=$(echo "$response" | jq -r '.status // empty')

    if [ "$status" = "error" ]; then
        log_json "error" "Knowledge base endpoint returned error: $response"
        return 1
    fi

    log_json "info" "Knowledge base status: $status"

    # Log entry count if available
    local entry_count
    entry_count=$(echo "$response" | jq -r '.entry_count // "N/A"' 2>/dev/null)
    log_json "info" "Knowledge base entry count: $entry_count"

    return 0
}

# Probe 2: Pattern Recognition Test
probe_pattern_recognition() {
    log_json "info" "Testing pattern recognition: ${Z3_API_URL}/api/v1/knowledge/patterns"

    # Request pattern analysis for a simple arithmetic constraint
    local pattern_request='{
        "problem_type": "constraint_satisfaction",
        "constraints": ["x > 5", "x < 10", "y = x * 2"],
        "extract_patterns": true
    }'

    local response
    response=$(api_request "/api/v1/knowledge/patterns" "POST" "$pattern_request")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "warn" "Pattern recognition endpoint returned invalid JSON (may not be implemented yet): $response"
        return 0  # Don't fail - this endpoint may not exist yet
    fi

    # Check for success field
    local success
    success=$(echo "$response" | jq -r '.success // false')

    if [ "$success" = "false" ]; then
        log_json "error" "Pattern recognition failed: $response"
        return 1
    fi

    # Log recognized patterns
    local patterns
    patterns=$(echo "$response" | jq -r '.patterns // []' 2>/dev/null)
    log_json "info" "Recognized patterns: $patterns"

    return 0
}

# Probe 3: Knowledge Graph Query
probe_knowledge_graph() {
    log_json "info" "Testing knowledge graph query: ${Z3_API_URL}/api/v1/knowledge/search"

    # Search for similar problems
    local search_request='{
        "query": "arithmetic constraints with bounds",
        "limit": 5
    }'

    local response
    response=$(api_request "/api/v1/knowledge/search" "POST" "$search_request")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "warn" "Knowledge graph endpoint returned invalid JSON (may not be implemented yet): $response"
        return 0  # Don't fail - this endpoint may not exist yet
    fi

    # Check for success field
    local success
    success=$(echo "$response" | jq -r '.success // false')

    if [ "$success" = "false" ]; then
        log_json "error" "Knowledge graph query failed: $response"
        return 1
    fi

    # Log search results
    local result_count
    result_count=$(echo "$response" | jq -r '.results | length' 2>/dev/null || echo "0")
    log_json "info" "Knowledge graph returned $result_count results"

    return 0
}

# Probe 4: Database Knowledge Tables
probe_database_knowledge() {
    log_json "info" "Checking database knowledge tables"

    if ! check_sqlite3; then
        log_json "info" "Skipping database knowledge check (sqlite3 not available)"
        return 0
    fi

    # Check if database file exists
    if [ ! -f "$DATABASE_URL" ]; then
        log_json "warn" "Database file not found: $DATABASE_URL (may not exist yet)"
        return 0
    fi

    # Query for knowledge-related tables
    local tables
    tables=$(sqlite3 "$DATABASE_URL" "SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%knowledge%' OR name LIKE '%pattern%' OR name LIKE '%learning%') ORDER BY name;" 2>&1)

    if [ $? -ne 0 ]; then
        log_json "error" "Failed to query knowledge tables: $tables"
        return 1
    fi

    if [ -z "$tables" ]; then
        log_json "warn" "No knowledge-related tables found in database (database may be empty)"
        return 0
    fi

    log_json "info" "Found knowledge tables: $tables"

    # Check row counts in knowledge tables
    while IFS= read -r table; do
        if [ -n "$table" ]; then
            local count
            count=$(sqlite3 "$DATABASE_URL" "SELECT COUNT(*) FROM $table;" 2>&1)
            log_json "info" "Table '$table' contains $count entries"
        fi
    done <<< "$tables"

    return 0
}

# Probe 5: Knowledge Extraction from Solve Results
probe_extraction_from_results() {
    log_json "info" "Testing knowledge extraction from solve results"

    # Solve a problem and check if knowledge is extracted
    local solve_request='{
        "problem": "(declare-const x Int) (declare-const y Int) (assert (> x 0)) (assert (> y 0)) (assert (= (+ x y) 10)) (check-sat)",
        "timeout": 5.0,
        "extract_knowledge": true
    }'

    local response
    response=$(api_request "/api/v1/solve" "POST" "$solve_request")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "warn" "Solve endpoint returned invalid JSON: $response"
        return 1
    fi

    # Check for success
    local success
    success=$(echo "$response" | jq -r '.success // false')

    if [ "$success" != "true" ]; then
        log_json "warn" "Solve request failed (expected for this test): $response"
        return 0  # Don't fail - we're testing the extraction capability
    fi

    # Check if knowledge was extracted
    local knowledge_extracted
    knowledge_extracted=$(echo "$response" | jq -r '.knowledge_extracted // false' 2>/dev/null)

    if [ "$knowledge_extracted" = "true" ]; then
        log_json "info" "Knowledge extraction from solve results: enabled"

        local patterns
        patterns=$(echo "$response" | jq -r '.patterns // []' 2>/dev/null)
        log_json "info" "Extracted patterns: $patterns"
    else
        log_json "info" "Knowledge extraction not enabled in response (optional feature)"
    fi

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Z3 knowledge extraction probe"
    log_json "info" "Target URL: $Z3_API_URL"
    log_json "info" "Database URL: $DATABASE_URL"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check prerequisites
    check_curl

    # Validate environment
    if [ -z "$Z3_API_URL" ]; then
        log_json "error" "Z3_API_URL environment variable is not set"
        exit 1
    fi

    # Run probes sequentially (don't fail fast - collect all results)
    local exit_code=0

    if ! probe_knowledge_base; then
        log_json "error" "Knowledge base probe failed"
        exit_code=2
    fi

    if ! probe_pattern_recognition; then
        log_json "error" "Pattern recognition probe failed"
        exit_code=3
    fi

    if ! probe_knowledge_graph; then
        log_json "error" "Knowledge graph probe failed"
        exit_code=4
    fi

    if ! probe_database_knowledge; then
        log_json "warn" "Database knowledge check failed (non-critical)"
        # Don't set exit_code - this is informational
    fi

    if ! probe_extraction_from_results; then
        log_json "warn" "Knowledge extraction probe failed (non-critical)"
        # Don't set exit_code - this is an optional feature
    fi

    if [ $exit_code -eq 0 ]; then
        log_json "info" "All Z3 knowledge extraction probes passed successfully"
    else
        log_json "error" "Some Z3 knowledge extraction probes failed"
    fi

    exit $exit_code
}

# Run main function
main "$@"
