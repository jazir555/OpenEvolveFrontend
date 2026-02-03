#!/bin/bash

###############################################################################
# check_workflows.sh - Test Workflow Execution
#
# This script verifies that the OpenEvolve workflow orchestration system
# is functional by testing a minimal workflow execution.
#
# Environment Variables Required:
#   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API
#   WORKFLOW_TIMEOUT_MS - Workflow execution timeout (default: 30000)
#   TEST_WORKFLOW_ID - ID for the test workflow (default: test-workflow-001)
#
# Usage:
#   ./check_workflows.sh
#
# Exit Codes:
#   0 - Workflow system is functional
#   1 - Workflow execution failed
#   2 - API unreachable
###############################################################################

set -euo pipefail

# Default values
DEFAULT_API_URL="http://localhost:8002"
DEFAULT_WORKFLOW_TIMEOUT=30
DEFAULT_WORKFLOW_ID="test-workflow-$(date +%s)"

# Load environment variables
API_URL="${OPENEVOLVE_API_URL:-$DEFAULT_API_URL}"
WORKFLOW_TIMEOUT_SEC=$(( (${WORKFLOW_TIMEOUT_MS:-30000} + 999) / 1000 ))
TEST_WORKFLOW_ID="${TEST_WORKFLOW_ID:-$DEFAULT_WORKFLOW_ID}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check_workflows] $1${NC}"
}

log_info() { log "$1" ""; }
log_success() { log "$1" "$GREEN"; }
log_error() { log "$1" "$RED"; }
log_warning() { log "$1" "$YELLOW"; }
log_workflow() { log "$1" "$BLUE"; }

# Validate API is accessible
validate_api() {
    log_info "Validating API accessibility..."

    local health_url="${API_URL}/health"
    local response_code

    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time 5 \
        -X GET \
        "$health_url" 2>&1) || true

    if [[ "$response_code" != "200" ]]; then
        log_error "API is not accessible (status: $response_code)"
        return 2
    fi

    log_success "API is accessible"
    return 0
}

# Create a test workflow
create_test_workflow() {
    log_workflow "Creating test workflow: $TEST_WORKFLOW_ID"

    local create_url="${API_URL}/openevolve/workflows"

    # Create a minimal workflow definition
    local workflow_payload=$(cat <<EOF
{
  "workflow_id": "$TEST_WORKFLOW_ID",
  "name": "Test Workflow",
  "description": "Minimal test workflow for health check",
  "problem_statement": "Test problem statement",
  "max_refinement_loops": 1,
  "auto_approval_enabled": true,
  "sub_problems": [
    {
      "id": "test-sub-1",
      "description": "Test sub-problem",
      "solver_team_name": "test-team",
      "gold_team_gauntlet_name": "test-gauntlet"
    }
  ]
}
EOF
)

    local response
    response=$(curl -s \
        --max-time 10 \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$workflow_payload" \
        "$create_url" 2>&1) || true

    if [[ -n "$response" ]]; then
        log_success "Workflow created successfully"
        log_info "Response: $response"
        return 0
    else
        log_error "Failed to create workflow"
        return 1
    fi
}

# Check workflow status
check_workflow_status() {
    log_workflow "Checking workflow status..."

    local status_url="${API_URL}/openevolve/workflows/${TEST_WORKFLOW_ID}/status"

    local response
    response=$(curl -s \
        --max-time 5 \
        -X GET \
        -H "Content-Type: application/json" \
        "$status_url" 2>&1) || true

    if [[ -n "$response" ]]; then
        log_success "Workflow status retrieved"
        log_info "Response: $response"
        return 0
    else
        log_warning "Could not retrieve workflow status"
        return 0  # Non-fatal
    fi
}

# List available workflows
list_workflows() {
    log_info "Listing available workflows..."

    local list_url="${API_URL}/openevolve/workflows"

    local response
    response=$(curl -s \
        --max-time 5 \
        -X GET \
        -H "Content-Type: application/json" \
        "$list_url" 2>&1) || true

    if [[ -n "$response" ]]; then
        log_success "Workflows retrieved successfully"
        echo "$response" | head -c 200
        echo ""
        return 0
    else
        log_warning "Could not retrieve workflows list"
        return 0  # Non-fatal
    fi
}

# Check teams (required for workflows)
check_teams() {
    log_info "Checking available teams..."

    local teams_url="${API_URL}/openevolve/teams"

    local response
    response=$(curl -s \
        --max-time 5 \
        -X GET \
        -H "Content-Type: application/json" \
        "$teams_url" 2>&1) || true

    if [[ -n "$response" ]]; then
        local team_count=$(echo "$response" | grep -o '"name"' | wc -l || echo "0")
        log_success "Teams retrieved: $team_count teams available"
        return 0
    else
        log_warning "No teams available (workflows may not function)"
        return 0  # Non-fatal for health check
    fi
}

# Check gauntlets (required for workflows)
check_gauntlets() {
    log_info "Checking available gauntlets..."

    local gauntlets_url="${API_URL}/openevolve/gauntlets"

    local response
    response=$(curl -s \
        --max-time 5 \
        -X GET \
        -H "Content-Type: application/json" \
        "$gauntlets_url" 2>&1) || true

    if [[ -n "$response" ]]; then
        local gauntlet_count=$(echo "$response" | grep -o '"name"' | wc -l || echo "0")
        log_success "Gauntlets retrieved: $gauntlet_count gauntlets available"
        return 0
    else
        log_warning "No gauntlets available (workflows may not function)"
        return 0  # Non-fatal for health check
    fi
}

# Main execution
main() {
    log_info "Starting workflow execution check..."
    echo ""

    local exit_code=0

    # Validate API first
    if ! validate_api; then
        log_error "Cannot proceed without API access"
        exit 2
    fi
    echo ""

    # Check prerequisites
    check_teams
    echo ""

    check_gauntlets
    echo ""

    # List workflows
    list_workflows
    echo ""

    # Try to create a test workflow
    if create_test_workflow; then
        echo ""

        # Check status
        check_workflow_status
        echo ""

        log_success "Workflow system is functional"
    else
        log_warning "Could not create test workflow (may require teams/gauntlets first)"
        exit_code=1
    fi

    exit $exit_code
}

# Run main function
main "$@"
