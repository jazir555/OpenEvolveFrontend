#!/bin/bash
##############################################################################
# BubbleLab API Probe Script
#
# Purpose: Verify BubbleLab API endpoints are accessible and responding correctly
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   BUBBLELAB_API_URL   - Base URL of BubbleLab API (default: http://localhost:3000)
#   TIMEOUT_MS          - Request timeout in milliseconds (default: 5000)
#
# Exit Codes:
#   0 - All probes passed
#   1 - Required environment variable missing
#   2 - Health check failed
#   3 - BubbleFlow list endpoint failed
#   4 - BubbleFlow creation endpoint failed
#   5 - curl not available
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

BUBBLELAB_API_URL="${BUBBLELAB_API_URL:-http://localhost:3000}"
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
    local auth_header="${4:-}"

    local url="${BUBBLELAB_API_URL}${endpoint}"

    if [ -n "$data" ]; then
        if [ -n "$auth_header" ]; then
            curl -s -X "$method" \
                --max-time "$TIMEOUT_SEC" \
                -H "Content-Type: application/json" \
                -H "Authorization: $auth_header" \
                -d "$data" \
                "$url" 2>&1
        else
            curl -s -X "$method" \
                --max-time "$TIMEOUT_SEC" \
                -H "Content-Type: application/json" \
                -d "$data" \
                "$url" 2>&1
        fi
    else
        if [ -n "$auth_header" ]; then
            curl -s -X "$method" \
                --max-time "$TIMEOUT_SEC" \
                -H "Content-Type: application/json" \
                -H "Authorization: $auth_header" \
                "$url" 2>&1
        else
            curl -s -X "$method" \
                --max-time "$TIMEOUT_SEC" \
                -H "Content-Type: application/json" \
                "$url" 2>&1
        fi
    fi
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Health Check Endpoint
probe_health() {
    log_json "info" "Testing health endpoint: ${BUBBLELAB_API_URL}/health"

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

    # Log additional health info if available
    local version
    version=$(echo "$response" | jq -r '.version // empty' 2>/dev/null || echo "")
    if [ -n "$version" ]; then
        log_json "info" "API version: $version"
    fi

    return 0
}

# Probe 2: List BubbleFlows Endpoint
probe_list_flows() {
    log_json "info" "Testing BubbleFlow list endpoint: ${BUBBLELAB_API_URL}/bubble-flow"

    local response
    response=$(api_request "/bubble-flow" "GET")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "BubbleFlow list endpoint returned invalid JSON: $response"
        return 1
    fi

    # Check for success/data field (response structure may vary)
    local has_data
    has_data=$(echo "$response" | jq -e 'has("data") or has("bubbleFlows") or has("flows")' 2>/dev/null || echo "false")

    if [ "$has_data" != "true" ]; then
        # Might be empty array which is valid
        local is_array
        is_array=$(echo "$response" | jq -e 'type == "array"' 2>/dev/null || echo "false")
        if [ "$is_array" != "true" ]; then
            log_json "warn" "BubbleFlow list returned unexpected structure: $response"
        fi
    fi

    log_json "info" "BubbleFlow list endpoint test passed"

    return 0
}

# Probe 3: BubbleFlow Code Validation Endpoint
probe_validate_code() {
    log_json "info" "Testing code validation endpoint: ${BUBBLELAB_API_URL}/bubble-flow/validate"

    # Minimal valid BubbleFlow code for testing
    local validate_request='{
        "code": "import { BubbleFlow } from '\''@bubblelab/bubble-core'\''; export class TestFlow extends BubbleFlow '\''webhook/http'\'' { async handle(payload) { return { message: '\''test'\'' }; } }"
    }'

    local response
    response=$(api_request "/bubble-flow/validate" "POST" "$validate_request")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "Validation endpoint returned invalid JSON: $response"
        return 1
    fi

    # Check for validation response structure
    local has_validation
    has_validation=$(echo "$response" | jq -e 'has("valid") or has("errors") or has("syntaxErrors")' 2>/dev/null || echo "false")

    if [ "$has_validation" != "true" ]; then
        log_json "warn" "Validation endpoint returned unexpected structure: $response"
    else
        log_json "info" "Code validation endpoint test passed"
    fi

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting BubbleLab API probe"
    log_json "info" "Target URL: $BUBBLELAB_API_URL"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check prerequisites
    check_curl

    # Validate environment
    if [ -z "$BUBBLELAB_API_URL" ]; then
        log_json "error" "BUBBLELAB_API_URL environment variable is not set"
        exit 1
    fi

    # Run probes sequentially (fail fast on first error)
    if ! probe_health; then
        log_json "error" "Health check probe failed"
        exit 2
    fi

    if ! probe_list_flows; then
        log_json "error" "BubbleFlow list probe failed"
        exit 3
    fi

    if ! probe_validate_code; then
        log_json "error" "Code validation probe failed"
        exit 4
    fi

    # All probes passed
    log_json "info" "All BubbleLab API probes passed successfully"
    exit 0
}

# Run main function
main "$@"
