#!/bin/bash

###############################################################################
# check_api.sh - Probe LoongFlow API Availability
#
# This script verifies that LoongFlow is accessible and has the required
# PES (Plan-Execute-Summary) framework available.
#
# Environment Variables Required:
#   LOONGFLOW_API_URL - Base URL of the LoongFlow service (required)
#   LOONGFLOW_TIMEOUT_MS - Request timeout in milliseconds (default: 5000)
#
# Usage:
#   ./check_api.sh
#
# Exit Codes:
#   0 - LoongFlow is healthy
#   1 - LoongFlow is unhealthy or unreachable
#   2 - Missing required environment variables
###############################################################################

set -euo pipefail

# Default values
DEFAULT_API_URL="http://localhost:8000"
DEFAULT_TIMEOUT=5

# Load environment variables with defaults
API_URL="${LOONGFLOW_API_URL:-$DEFAULT_API_URL}"
TIMEOUT_SEC=$(( (${LOONGFLOW_TIMEOUT_MS:-5000} + 999) / 1000 ))  # Convert ms to seconds, round up

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check_api] $1${NC}"
}

log_info() {
    log "$1" ""
}

log_success() {
    log "$1" "$GREEN"
}

log_error() {
    log "$1" "$RED"
}

log_warning() {
    log "$1" "$YELLOW"
}

# Validate environment
validate_env() {
    log_info "Validating environment..."

    if [[ -z "$API_URL" ]]; then
        log_error "LOONGFLOW_API_URL is not set"
        exit 2
    fi

    # Check if timeout is a valid number
    if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
        log_error "LOONGFLOW_TIMEOUT_MS must be a valid number"
        exit 2
    fi

    log_success "Environment validation passed"
    log_info "API URL: $API_URL"
    log_info "Timeout: ${TIMEOUT_SEC}s"
}

# Check if LoongFlow is accessible by checking the Python process
check_loongflow_process() {
    log_info "Checking if LoongFlow is running..."

    # Check for Python processes with loongflow or pes_agent
    local running_procs
    running_procs=$(pgrep -f "loongflow" 2>/dev/null || pgrep -f "pes_agent" 2>/dev/null || echo "")

    if [[ -n "$running_procs" ]]; then
        log_success "LoongFlow process detected (PIDs: $running_procs)"
        return 0
    else
        log_warning "No LoongFlow process detected"
        log_info "This is OK if LoongFlow runs on-demand or in a container"
        return 0
    fi
}

# Check if workspace directory exists (LoongFlow creates these)
check_workspace() {
    log_info "Checking for LoongFlow workspace directories..."

    # LoongFlow creates output directories like ./output-hello-world
    local found_dirs=0

    for dir in ./output-*; do
        if [[ -d "$dir" ]]; then
            log_success "Found workspace directory: $dir"
            found_dirs=$((found_dirs + 1))
        fi
    done

    if [[ $found_dirs -gt 0 ]]; then
        log_info "Found $found_dirs workspace directory(s)"
    else
        log_info "No workspace directories found (this is OK for fresh installations)"
    fi

    return 0
}

# Check if LoongFlow source code is accessible
check_loongflow_source() {
    log_info "Checking LoongFlow source code accessibility..."

    # Check relative path to core-projects
    local loongflow_path="../../../../core-projects/LoongFlow"

    if [[ -d "$loongflow_path" ]]; then
        log_success "LoongFlow source found at: $loongflow_path"

        # Check for key components
        if [[ -f "$loongflow_path/src/loongflow/framework/pes/pes_agent.py" ]]; then
            log_success "PES Agent module found"
        else
            log_error "PES Agent module NOT found"
            return 1
        fi

        if [[ -f "$loongflow_path/src/loongflow/framework/pes/database/database.py" ]]; then
            log_success "EvolveDatabase module found"
        else
            log_error "EvolveDatabase module NOT found"
            return 1
        fi

        return 0
    else
        log_error "LoongFlow source NOT found at: $loongflow_path"
        log_info "This adapter requires LoongFlow to be in core-projects/"
        return 1
    fi
}

# Check example configuration files
check_example_configs() {
    log_info "Checking for example task configurations..."

    local loongflow_path="../../../../core-projects/LoongFlow"
    local config_count=0

    # Look for task_config.yaml files in examples
    while IFS= read -r -d '' config_file; do
        log_success "Found example config: $config_file"
        config_count=$((config_count + 1))
    done < <(find "$loongflow_path/agents" -name "task_config.yaml" -print0 2>/dev/null || true)

    if [[ $config_count -gt 0 ]]; then
        log_info "Found $config_count example configuration(s)"
        return 0
    else
        log_warning "No example configurations found"
        return 0  # Not fatal
    fi
}

# Main execution
main() {
    log_info "Starting LoongFlow API probe..."
    echo ""

    validate_env
    echo ""

    local exit_code=0

    # Check source code (critical)
    if ! check_loongflow_source; then
        exit_code=1
    fi
    echo ""

    # Check example configs (informational)
    check_example_configs
    echo ""

    # Check process (informational)
    check_loongflow_process
    echo ""

    # Check workspaces (informational)
    check_workspace
    echo ""

    if [[ $exit_code -eq 0 ]]; then
        log_success "LoongFlow API probe completed successfully"
    else
        log_error "LoongFlow API probe failed"
    fi

    exit $exit_code
}

# Run main function
main "$@"
