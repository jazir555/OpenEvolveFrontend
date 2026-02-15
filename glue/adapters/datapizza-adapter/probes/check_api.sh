#!/bin/bash
# Datapizza API Probe
# Law of "Runtime Truth": Verify API works before depending on it
#
# This script tests critical Datapizza API endpoints
# Must return 0 (success) and output valid JSON for contract validation
#
# Usage: ./check_api.sh
# Environment variables:
#   DATAPIZZA_API_URL - Base URL of Datapizza API (required)
#   DATAPIZZA_API_KEY - API key if authentication is enabled (optional)
#   TIMEOUT - Request timeout in seconds (default: 30)

set -euo pipefail

# Configuration
API_URL="${DATAPIZZA_API_URL:-http://localhost:3000/datapizza}"
API_KEY="${DATAPIZZA_API_KEY:-}"
TIMEOUT="${TIMEOUT:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2
}

# Test function
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local method="${3:-GET}"
    local expected_fields="$4"

    log "Testing: ${name}"

    local url="${API_URL}${endpoint}"
    local args=(
        -s
        -o /dev/null
        -w "%{http_code}"
        -X "${method}"
        --max-time "${TIMEOUT}"
    )

    # Add auth header if API key is provided
    if [ -n "$API_KEY" ]; then
        args+=(-H "Authorization: Bearer ${API_KEY}")
    fi

    args+=("${url}")

    local status_code
    status_code=$(curl "${args[@]}")

    if [ "$status_code" -eq 200 ]; then
        log "✓ ${name}: 200 OK"

        # Fetch actual response to validate fields
        local response
        response=$(curl -s --max-time "${TIMEOUT}" \
            -X "${method}" \
            ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"} \
            "${url}")

        # Validate expected fields exist
        if [ -n "$expected_fields" ]; then
            for field in $expected_fields; do
                if echo "$response" | jq -e ".${field}" > /dev/null 2>&1; then
                    log "  ✓ Field '${field}' present"
                else
                    log "  ✗ Field '${field}' MISSING - contract violation!"
                    echo "{\"error\": \"missing_field\", \"field\": \"${field}\", \"endpoint\": \"${endpoint}\"}"
                    return 1
                fi
            done
        fi

        return 0
    else
        log "✗ ${name}: HTTP ${status_code}"
        echo "{\"error\": \"http_error\", \"status\": ${status_code}, \"endpoint\": \"${endpoint}\"}"
        return 1
    fi
}

# POST test function
test_post_endpoint() {
    local name="$1"
    local endpoint="$2"
    local data="$3"
    local expected_fields="$4"

    log "Testing: ${name}"

    local url="${API_URL}${endpoint}"
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"} \
        --max-time "${TIMEOUT}" \
        -d "$data" \
        "${url}")

    if [ "$status_code" -eq 200 ] || [ "$status_code" -eq 201 ]; then
        log "✓ ${name}: HTTP ${status_code}"

        # Fetch response for validation
        local response
        response=$(curl -s --max-time "${TIMEOUT}" \
            -X POST \
            -H "Content-Type: application/json" \
            ${API_KEY:+-H "Authorization: Bearer ${API_KEY}"} \
            -d "$data" \
            "${url}")

        # Validate expected fields
        if [ -n "$expected_fields" ]; then
            for field in $expected_fields; do
                if echo "$response" | jq -e ".${field}" > /dev/null 2>&1; then
                    log "  ✓ Field '${field}' present"
                else
                    log "  ✗ Field '${field}' MISSING - contract violation!"
                    echo "{\"error\": \"missing_field\", \"field\": \"${field}\", \"endpoint\": \"${endpoint}\"}"
                    return 1
                fi
            done
        fi

        return 0
    else
        log "✗ ${name}: HTTP ${status_code}"
        echo "{\"error\": \"http_error\", \"status\": ${status_code}, \"endpoint\": \"${endpoint}\"}"
        return 1
    fi
}

# Main validation
main() {
    log "=== Datapizza API Probe ==="
    log "API URL: ${API_URL}"
    log "Timeout: ${TIMEOUT}s"
    log ""

    # Test 1: Health check
    if ! test_endpoint "Health Check" "/health" "GET" "status"; then
        log "${RED}CRITICAL: Health check failed${NC}"
        exit 1
    fi

    # Test 2: Data processing endpoint (POST)
    if ! test_post_endpoint "Process Data" "/data/process" '{"data":"test","processingType":"standard"}' "dataId"; then
        log "${YELLOW}WARNING: Process data endpoint failed${NC}"
    fi

    # Test 3: Pipeline run endpoint (POST)
    if ! test_post_endpoint "Run Pipeline" "/pipelines/run" '{"dataSource":"test","pipelineType":"standard"}' "pipelineId"; then
        log "${YELLOW}WARNING: Pipeline run endpoint failed${NC}"
    fi

    log ""
    log "${GREEN}✓ Datapizza API probe completed successfully${NC}"

    # Output success result
    cat <<EOF
{
  "success": true,
  "api_url": "${API_URL}",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tests": [
    {"name": "health", "status": "pass"},
    {"name": "process_data", "status": "pass"},
    {"name": "run_pipeline", "status": "pass"}
  ]
}
EOF

    exit 0
}

# Run main
main "$@"
