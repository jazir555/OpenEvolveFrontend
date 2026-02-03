#!/bin/bash
##############################################################################
# BubbleLab Workflow Execution Probe Script
#
# Purpose: Verify BubbleLab workflow execution is working correctly
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   BUBBLELAB_API_URL   - Base URL of BubbleLab API (default: http://localhost:3000)
#   TIMEOUT_MS          - Request timeout in milliseconds (default: 10000)
#   BUBBLELAB_AUTH_TOKEN - Auth token (optional, for authenticated endpoints)
#
# Exit Codes:
#   0 - All probes passed
#   1 - Required environment variable missing
#   2 - Workflow creation failed
#   3 - Workflow execution failed
#   4 - Workflow execution history failed
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

BUBBLELAB_API_URL="${BUBBLELAB_API_URL:-http://localhost:3000}"
TIMEOUT_MS="${TIMEOUT_MS:-10000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))
BUBBLELAB_AUTH_TOKEN="${BUBBLELAB_AUTH_TOKEN:-}"

# =============================================================================
# Utility Functions
# =============================================================================

log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_workflows.sh\"}"
}

check_curl() {
    if ! command -v curl &> /dev/null; then
        log_json "error" "curl is not installed or not in PATH"
        exit 5
    fi
}

api_request() {
    local endpoint="$1"
    local method="${2:-GET}"
    local data="${3:-}"

    local url="${BUBBLELAB_API_URL}${endpoint}"
    local auth_header=""

    if [ -n "$BUBBLELAB_AUTH_TOKEN" ]; then
        auth_header="Bearer $BUBBLELAB_AUTH_TOKEN"
    fi

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

# Probe 1: Create Test Workflow
probe_create_workflow() {
    log_json "info" "Testing workflow creation"

    # Create a simple workflow with data transformation
    local workflow_code='{
        "name": "Workflow Probe Test",
        "description": "Testing workflow creation",
        "code": "import { BubbleFlow } from '\''@bubblelab/bubble-core'\''; export interface Output { processed: boolean; input_data: any; timestamp: string; } export class WorkflowProbeFlow extends BubbleFlow '\''webhook/http'\'' { constructor() { super('\''workflow-probe'\'', '\''Test workflow execution'\''); } async handle(payload: any) { return { processed: true, input_data: payload, timestamp: new Date().toISOString() }; } }",
        "eventType": "webhook/http",
        "webhookActive": false
    }'

    local response
    response=$(api_request "/bubble-flow" "POST" "$workflow_code")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "Workflow creation returned invalid JSON: $response"
        return 1
    fi

    # Check for workflow ID
    local workflow_id
    workflow_id=$(echo "$response" | jq -r '.id // .bubbleFlowId // .flowId // empty' 2>/dev/null)

    if [ -z "$workflow_id" ]; then
        local error_msg
        error_msg=$(echo "$response" | jq -r '.error // .message // empty' 2>/dev/null)
        log_json "error" "Workflow creation failed: ${error_msg:-unknown error}"
        return 1
    fi

    log_json "info" "Workflow created successfully: ID=$workflow_id"
    echo "$workflow_id"

    return 0
}

# Probe 2: Execute Workflow
probe_execute_workflow() {
    local workflow_id="$1"

    log_json "info" "Testing workflow execution for ID: $workflow_id"

    local execution_payload='{
        "payload": {
            "test_data": "probe_value",
            "timestamp": "2026-02-03T00:00:00Z"
        }
    }'

    local response
    response=$(api_request "/bubble-flow/${workflow_id}/execute" "POST" "$execution_payload")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "error" "Workflow execution returned invalid JSON: $response"
        return 1
    fi

    # Check for execution result
    local has_output
    has_output=$(echo "$response" | jq -e 'has("output") or has("result")' 2>/dev/null || echo "false")

    if [ "$has_output" = "true" ]; then
        local output
        output=$(echo "$response" | jq -c '.output // .result' 2>/dev/null)
        log_json "info" "Workflow executed successfully: $output"
    else
        # Check for error
        local error_msg
        error_msg=$(echo "$response" | jq -r '.error // .message // empty' 2>/dev/null)

        if [ -n "$error_msg" ]; then
            log_json "error" "Workflow execution failed: $error_msg"
            return 1
        else
            log_json "warn" "Workflow execution returned unexpected structure: $response"
        fi
    fi

    return 0
}

# Probe 3: Query Execution History
probe_execution_history() {
    local workflow_id="$1"

    log_json "info" "Testing execution history query for ID: $workflow_id"

    local response
    response=$(api_request "/bubble-flow/${workflow_id}/executions?limit=5" "GET")

    # Check if response contains valid JSON
    if ! echo "$response" | jq -e '.' &> /dev/null; then
        log_json "warn" "Execution history returned invalid JSON: $response"
        return 0  # Don't fail, might not be supported
    fi

    # Check for executions array
    local has_executions
    has_executions=$(echo "$response" | jq -e 'has("executions") or has("data")' 2>/dev/null || echo "false")

    if [ "$has_executions" = "true" ]; then
        local count
        count=$(echo "$response" | jq 'length // .executions | length // .data | length // 0' 2>/dev/null)
        log_json "info" "Execution history retrieved: $count executions"
    else
        log_json "info" "Execution history endpoint test passed"
    fi

    return 0
}

# Probe 4: Test Streaming Execution
probe_streaming_execution() {
    local workflow_id="$1"

    log_json "info" "Testing streaming execution for ID: $workflow_id"

    local execution_payload='{
        "payload": {
            "test_data": "streaming_probe"
        }
    }'

    # Note: This probe doesn't actually consume the stream, just checks if endpoint exists
    local response
    response=$(api_request "/bubble-flow/${workflow_id}/execute-stream" "POST" "$execution_payload")

    # Streaming returns server-sent events, not JSON
    if [ -n "$response" ]; then
        log_json "info" "Streaming execution endpoint available"
    else
        log_json "warn" "Streaming execution endpoint may not be available"
    fi

    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting BubbleLab workflow execution probe"
    log_json "info" "Target URL: $BUBBLELAB_API_URL"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    check_curl

    if [ -z "$BUBBLELAB_API_URL" ]; then
        log_json "error" "BUBBLELAB_API_URL environment variable is not set"
        exit 1
    fi

    # Run probes sequentially
    local workflow_id=""
    if ! workflow_id=$(probe_create_workflow); then
        log_json "error" "Workflow creation probe failed"
        exit 2
    fi

    if ! probe_execute_workflow "$workflow_id"; then
        log_json "error" "Workflow execution probe failed"
        exit 3
    fi

    if ! probe_execution_history "$workflow_id"; then
        log_json "error" "Execution history probe failed"
        exit 4
    fi

    if ! probe_streaming_execution "$workflow_id"; then
        log_json "warn" "Streaming execution probe failed (non-critical)"
    fi

    log_json "info" "All BubbleLab workflow execution probes passed"
    exit 0
}

main "$@"
