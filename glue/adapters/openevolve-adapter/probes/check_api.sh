#!/bin/bash

###############################################################################
# check_api.sh - Probe OpenEvolve API Availability
#
# This script verifies that the OpenEvolve API is accessible and responsive.
# It tests the main API endpoint and returns the result.
#
# Environment Variables Required:
#   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API (default: http://localhost:8002)
#   TIMEOUT_MS - Request timeout in milliseconds (default: 5000)
#
# Usage:
#   ./check_api.sh
#
# Exit Codes:
#   0 - API is healthy
#   1 - API is unhealthy or unreachable
#   2 - Missing required environment variables
###############################################################################

set -euo pipefail

# Default values
DEFAULT_API_URL="http://localhost:8002"
DEFAULT_TIMEOUT=5

# Load environment variables with defaults
API_URL="${OPENEVOLVE_API_URL:-$DEFAULT_API_URL}"
TIMEOUT_SEC=$(( (${TIMEOUT_MS:-5000} + 999) / 1000 ))  # Convert ms to seconds, round up

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
        log_error "OPENEVOLVE_API_URL is not set"
        exit 2
    fi

    # Check if timeout is a valid number
    if ! [[ "$TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
        log_error "TIMEOUT_MS must be a valid number"
        exit 2
    fi

    log_success "Environment validation passed"
    log_info "API URL: $API_URL"
    log_info "Timeout: ${TIMEOUT_SEC}s"
}

# Check API health endpoint
check_health_endpoint() {
    log_info "Checking health endpoint..."

    local health_url="${API_URL}/health"
    local start_time=$(date +%s%3N)

    # Make the request with timeout
    local response_code
    local response_body

    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X GET \
        -H "Content-Type: application/json" \
        "$health_url" 2>&1) || true

    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    if [[ "$response_code" == "200" ]]; then
        log_success "Health check returned 200 OK (${duration}ms)"
        return 0
    else
        log_error "Health check failed with status code: $response_code (${duration}ms)"
        return 1
    fi
}

# Check root endpoint
check_root_endpoint() {
    log_info "Checking root endpoint..."

    local root_url="${API_URL}/"
    local start_time=$(date +%s%3N)

    local response_body
    response_body=$(curl -s \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X GET \
        -H "Content-Type: application/json" \
        "$root_url" 2>&1) || true

    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    if [[ -n "$response_body" ]]; then
        log_success "Root endpoint returned response (${duration}ms)"
        log_info "Response: $response_body"
        return 0
    else
        log_error "Root endpoint returned empty response (${duration}ms)"
        return 1
    fi
}

# Check teams endpoint
check_teams_endpoint() {
    log_info "Checking teams endpoint..."

    local teams_url="${API_URL}/openevolve/teams"
    local start_time=$(date +%s%3N)

    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X GET \
        -H "Content-Type: application/json" \
        "$teams_url" 2>&1) || true

    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    if [[ "$response_code" == "200" ]]; then
        log_success "Teams endpoint returned 200 OK (${duration}ms)"
        return 0
    else
        log_warning "Teams endpoint returned status code: $response_code (${duration}ms)"
        return 0  # Non-fatal, API might be up but no teams configured
    fi
}

# Main execution
main() {
    log_info "Starting OpenEvolve API probe..."
    echo ""

    validate_env
    echo ""

    local exit_code=0

    if ! check_health_endpoint; then
        exit_code=1
    fi
    echo ""

    if ! check_root_endpoint; then
        exit_code=1
    fi
    echo ""

    # Teams endpoint is informational, don't fail on it
    check_teams_endpoint
    echo ""

    if [[ $exit_code -eq 0 ]]; then
        log_success "OpenEvolve API probe completed successfully"
    else
        log_error "OpenEvolve API probe failed"
    fi

    exit $exit_code
}

# Run main function
main "$@"
