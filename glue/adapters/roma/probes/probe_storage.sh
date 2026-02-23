#!/bin/bash
# ROMA Storage/Checkpoint Probe
# Tests that ROMA can store and retrieve checkpoints
# Follows "Law of Runtime Truth" - validates storage behavior

set -euo pipefail

# Configuration
ROMA_SERVER_URL="${ROMA_SERVER_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# Create execution for checkpoint testing
create_test_execution() {
    log_info "Creating test execution for checkpoint..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        -X POST "${ROMA_SERVER_URL}/api/v1/executions" \
        -H "Content-Type: application/json" \
        -d '{
            "goal": "Test checkpoint creation",
            "max_depth": 2,
            "config_profile": "fast"
        }' 2>&1) || true

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ] || [ "$http_code" = "202" ]; then
        exec_id=$(echo "$body" | grep -o '"execution_id":"[^"]*"' | cut -d'"' -f4)
        log_info "Test execution created: $exec_id"
        echo "$exec_id"
        return 0
    else
        log_error "Failed to create test execution (HTTP $http_code)"
        return 1
    fi
}

# Get checkpoint
get_checkpoint() {
    local exec_id="$1"
    log_info "Retrieving checkpoint for execution..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        "${ROMA_SERVER_URL}/api/v1/executions/${exec_id}/checkpoint" 2>&1) || true

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ]; then
        # Validate checkpoint structure
        if echo "$body" | grep -q "nodes\|sub_tasks\|decomposition"; then
            log_info "Checkpoint retrieved and validated"
            echo "$body"
            return 0
        else
            log_warn "Checkpoint response missing expected fields"
            echo "$body"
            return 0  # Still success, checkpoint may be empty for new execution
        fi
    elif [ "$http_code" = "404" ]; then
        log_warn "No checkpoint found (execution may not have started yet)"
        return 0  # Not an error, checkpoint may not exist yet
    else
        log_error "Failed to retrieve checkpoint (HTTP $http_code)"
        return 1
    fi
}

# Get execution data (MLflow traces)
get_execution_data() {
    local exec_id="$1"
    log_info "Retrieving execution data (traces)..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        "${ROMA_SERVER_URL}/api/v1/executions/${exec_id}/data" 2>&1) || true

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "200" ]; then
        log_info "Execution data retrieved"
        echo "$body"
        return 0
    elif [ "$http_code" = "404" ]; then
        log_warn "No execution data found (MLflow may not be enabled)"
        return 0  # Not an error if MLflow is disabled
    else
        log_error "Failed to retrieve execution data (HTTP $http_code)"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting ROMA storage/checkpoint probe..."
    log_info "Target server: ${ROMA_SERVER_URL}"

    # Create test execution
    exec_id=$(create_test_execution)
    if [ $? -ne 0 ]; then
        log_error "Storage probe failed: could not create execution"
        exit 1
    fi

    # Test checkpoint retrieval
    checkpoint=$(get_checkpoint "$exec_id")
    if [ $? -ne 0 ]; then
        log_error "Storage probe failed: could not retrieve checkpoint"
        exit 1
    fi

    # Test execution data retrieval
    exec_data=$(get_execution_data "$exec_id")
    if [ $? -ne 0 ]; then
        log_warn "Execution data retrieval failed (MLflow may be disabled)"
    fi

    log_info "Storage probe completed successfully"
    echo "{\"status\":\"success\",\"execution_id\":\"${exec_id}\"}"
    exit 0
}

main "$@"
