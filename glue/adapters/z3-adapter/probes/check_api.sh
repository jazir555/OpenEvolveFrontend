#!/bin/bash
##############################################################################
# Z3 API Probe Script
#
# Purpose: Verify Z3 API endpoints are accessible and responding correctly
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   Z3_API_URL          - Base URL of Z3 API (default: http://localhost:8000)
#   TIMEOUT_MS          - Request timeout in milliseconds (default: 5000)
#
# Exit Codes:
#   0 - All probes passed
#   1 - Required environment variable missing
#   2 - API health check failed
#   3 - Solve endpoint check failed
#   4 - Prove endpoint check failed
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
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_api.sh\"}"
}

# Check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        log_json "error" "curl is not installed or not in PATH"
        exit 5
    fi
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

# Probe 1: Health Check Endpoint
probe_health() {
    log_json "info" "Testing health endpoint: ${Z3_API_URL}/health"

    local response
    response=$(api_request "/health" "GET")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "Health endpoint returned invalid JSON: $response"
        return 1
    fi

    # Check for status field
    local status
    status=$(echo "$response" | jq -r '.status // empty')

    if [ "$status" != "ok" ] && [ "$status" != "healthy" ]; then
        log_json "error" "Health check failed with status: $status"
        return 1
    fi

    log_json "info" "Health check passed: $status"

    # Log component availability
    local components
    components=$(echo "$response" | jq -r '.components // {}' 2>/dev/null || echo "{}")
    log_json "info" "Components status: $components"

    return 0
}

# Probe 2: Solve Endpoint (with simple problem)
probe_solve() {
    log_json "info" "Testing solve endpoint: ${Z3_API_URL}/api/v1/solve"

    # Simple SMT problem: x > 5 AND x < 10
    local solve_request='{
        "problem": "(declare-const x Int) (assert (> x 5)) (assert (< x 10)) (check-sat)",
        "timeout": 5.0
    }'

    local response
    response=$(api_request "/api/v1/solve" "POST" "$solve_request")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "Solve endpoint returned invalid JSON: $response"
        return 1
    fi

    # Check for success field
    local success
    success=$(echo "$response" | jq -r '.success // false')

    if [ "$success" != "true" ]; then
        log_json "error" "Solve endpoint failed: $response"
        return 1
    fi

    # Check for satisfiable result
    local satisfiable
    satisfiable=$(echo "$response" | jq -r '.satisfiable // null')

    if [ "$satisfiable" != "true" ]; then
        log_json "warn" "Solve endpoint returned unexpected satisfiability: $satisfiable"
    fi

    log_json "info" "Solve endpoint test passed"

    return 0
}

# Probe 3: Prove Endpoint
probe_prove() {
    log_json "info" "Testing prove endpoint: ${Z3_API_URL}/api/v1/prove"

    # Simple theorem: x > 5 implies x > 3
    local prove_request='{
        "theorem": "(declare-const x Int) (assert (> x 5)) (assert (not (> x 3))) (check-sat)",
        "extract_proof": false
    }'

    local response
    response=$(api_request "/api/v1/prove" "POST" "$prove_request")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "Prove endpoint returned invalid JSON: $response"
        return 1
    fi

    # Check for success field
    local success
    success=$(echo "$response" | jq -r '.success // false')

    if [ "$success" != "true" ]; then
        log_json "error" "Prove endpoint failed: $response"
        return 1
    fi

    log_json "info" "Prove endpoint test passed"

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Z3 API probe"
    log_json "info" "Target URL: $Z3_API_URL"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check prerequisites
    check_curl

    # Validate environment
    if [ -z "$Z3_API_URL" ]; then
        log_json "error" "Z3_API_URL environment variable is not set"
        exit 1
    fi

    # Run probes sequentially (fail fast on first error)
    if ! probe_health; then
        log_json "error" "Health check probe failed"
        exit 2
    fi

    if ! probe_solve; then
        log_json "error" "Solve endpoint probe failed"
        exit 3
    fi

    if ! probe_prove; then
        log_json "error" "Prove endpoint probe failed"
        exit 4
    fi

    # All probes passed
    log_json "info" "All Z3 API probes passed successfully"
    exit 0
}

# Run main function
main "$@"
