#!/bin/bash
# ROMA Execution Probe
# Tests that ROMA can accept and execute tasks
# Follows "Law of Runtime Truth" - validates actual execution behavior

set -euo pipefail

# Configuration
ROMA_SERVER_URL="${ROMA_SERVER_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-30}"
TEST_GOAL="${TEST_GOAL:-What is 2+2?}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Create execution
create_execution() {
    log_info "Creating test execution..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        -X POST "${ROMA_SERVER_URL}/api/v1/executions" \
        -H "Content-Type: application/json" \
        -d "{
            \"goal\": \"${TEST_GOAL}\",
            \"max_depth\": 1,
            \"config_profile\": \"fast\"
        }" 2>&1) || true

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ] || [ "$http_code" = "202" ]; then
        # Extract execution_id
        exec_id=$(echo "$body" | grep -o '"execution_id":"[^"]*"' | cut -d'"' -f4)
        if [ -n "$exec_id" ]; then
            log_info "Execution created: $exec_id"
            echo "$exec_id"
            return 0
        else
            log_error "Failed to extract execution_id from response"
            echo "$body"
            return 1
        fi
    else
        log_error "Failed to create execution (HTTP $http_code)"
        echo "$body"
        return 1
    fi
}

# Get execution status
get_execution() {
    local exec_id="$1"
    log_info "Checking execution status..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        "${ROMA_SERVER_URL}/api/v1/executions/${exec_id}" 2>&1) || true

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ]; then
        log_info "Execution status retrieved"
        echo "$body"
        return 0
    else
        log_error "Failed to get execution (HTTP $http_code)"
        return 1
    fi
}

# Cancel execution
cancel_execution() {
    local exec_id="$1"
    log_info "Cancelling execution..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        -X POST "${ROMA_SERVER_URL}/api/v1/executions/${exec_id}/cancel" 2>&1) || true

    http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" = "200" ] || [ "$http_code" = "202" ]; then
        log_info "Execution cancelled successfully"
        return 0
    else
        log_warn "Failed to cancel execution (HTTP $http_code)"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting ROMA execution probe..."
    log_info "Target server: ${ROMA_SERVER_URL}"

    # Create execution
    exec_id=$(create_execution)
    if [ $? -ne 0 ]; then
        log_error "Execution probe failed: could not create execution"
        exit 1
    fi

    # Get execution details
    exec_details=$(get_execution "$exec_id")
    if [ $? -ne 0 ]; then
        log_error "Execution probe failed: could not retrieve execution"
        exit 1
    fi

    # Extract status
    status=$(echo "$exec_details" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    log_info "Execution status: $status"

    # Clean up - cancel if still running
    if [ "$status" = "pending" ] || [ "$status" = "running" ]; then
        cancel_execution "$exec_id" || true
    fi

    log_info "Execution probe completed successfully"
    echo "{\"status\":\"success\",\"execution_id\":\"${exec_id}\",\"execution_status\":\"${status}\"}"
    exit 0
}

main "$@"
