#!/bin/bash

###############################################################################
# check-plugin-api.sh - Probe OpenEvolve React Plugin API Availability
#
# This script verifies that the OpenEvolve React Plugin can communicate
# with the OpenEvolve backend API. It tests plugin functionality endpoints.
#
# Environment Variables Required:
#   OPENEVOLVE_API_URL - Base URL of the OpenEvolve API (default: http://localhost:8002)
#   PLUGIN_TIMEOUT_MS - Request timeout in milliseconds (default: 10000)
#
# Usage:
#   ./check-plugin-api.sh
#
# Exit Codes:
#   0 - Plugin API is healthy
#   1 - Plugin API is unhealthy or unreachable
#   2 - Missing required environment variables
###############################################################################

set -euo pipefail

# Default values
DEFAULT_API_URL="http://localhost:8002"
DEFAULT_TIMEOUT=10

# Load environment variables with defaults
API_URL="${OPENEVOLVE_API_URL:-$DEFAULT_API_URL}"
TIMEOUT_SEC=$(( (${PLUGIN_TIMEOUT_MS:-10000} + 999) / 1000 ))  # Convert ms to seconds, round up

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check-plugin-api] $1${NC}"
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
        log_error "PLUGIN_TIMEOUT_MS must be a valid number"
        exit 2
    fi

    log_success "Environment validation passed"
    log_info "API URL: $API_URL"
    log_info "Timeout: ${TIMEOUT_SEC}s"
}

# Check plugin health endpoint
check_plugin_health() {
    log_info "Checking plugin health endpoint..."

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
        log_success "Health endpoint returned 200 OK (${duration}ms)"
        return 0
    else
        log_error "Health endpoint returned $response_code (expected 200)"
        return 1
    fi
}

# Check teams endpoint (plugin uses this for team management)
check_teams_endpoint() {
    log_info "Checking teams endpoint..."

    local teams_url="${API_URL}/teams"
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

    if [[ "$response_code" =~ ^(200|404)$ ]]; then
        log_success "Teams endpoint accessible (${duration}ms)"
        return 0
    else
        log_error "Teams endpoint returned $response_code"
        return 1
    fi
}

# Check gauntlets endpoint (plugin uses this for gauntlet management)
check_gauntlets_endpoint() {
    log_info "Checking gauntlets endpoint..."

    local gauntlets_url="${API_URL}/gauntlets"
    local start_time=$(date +%s%3N)

    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X GET \
        -H "Content-Type: application/json" \
        "$gauntlets_url" 2>&1) || true

    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    if [[ "$response_code" =~ ^(200|404)$ ]]; then
        log_success "Gauntlets endpoint accessible (${duration}ms)"
        return 0
    else
        log_error "Gauntlets endpoint returned $response_code"
        return 1
    fi
}

# Check workflows endpoint (plugin uses this for workflow execution)
check_workflows_endpoint() {
    log_info "Checking workflows endpoint..."

    local workflows_url="${API_URL}/workflows"
    local start_time=$(date +%s%3N)

    local response_code
    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X GET \
        -H "Content-Type: application/json" \
        "$workflows_url" 2>&1) || true

    local end_time=$(date +%s%3N)
    local duration=$((end_time - start_time))

    if [[ "$response_code" =~ ^(200|404)$ ]]; then
        log_success "Workflows endpoint accessible (${duration}ms)"
        return 0
    else
        log_error "Workflows endpoint returned $response_code"
        return 1
    fi
}

# Check CORS headers (required for plugin from browser)
check_cors_headers() {
    log_info "Checking CORS headers..."

    local cors_check=$(curl -s -I \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X OPTIONS \
        -H "Origin: http://localhost:3000" \
        -H "Access-Control-Request-Method: POST" \
        "$API_URL/health" 2>&1 | grep -i "access-control-allow-origin" || true)

    if [[ -n "$cors_check" ]]; then
        log_success "CORS headers present"
        return 0
    else
        log_warning "CORS headers not detected (plugin may not work from browser)"
        return 0  # Not fatal, just a warning
    fi
}

# Main execution
main() {
    echo ""
    echo "========================================"
    echo "OpenEvolve React Plugin - API Probe"
    echo "========================================"
    echo ""

    validate_env
    echo ""

    local overall_status=0

    # Run all checks
    check_plugin_health || overall_status=1
    check_teams_endpoint || overall_status=1
    check_gauntlets_endpoint || overall_status=1
    check_workflows_endpoint || overall_status=1
    check_cors_headers || true

    echo ""
    echo "========================================"

    if [[ $overall_status -eq 0 ]]; then
        log_success "✅ All plugin API checks passed!"
        echo "========================================"
        echo ""
        exit 0
    else
        log_error "❌ Some plugin API checks failed"
        echo "========================================"
        echo ""
        exit 1
    fi
}

# Run main function
main "$@"
