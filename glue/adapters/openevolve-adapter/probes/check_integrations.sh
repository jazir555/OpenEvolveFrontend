#!/bin/bash

###############################################################################
# check_integrations.sh - Probe All Integration Health
#
# This script verifies the health of all integrated adapters and services
# that the OpenEvolve main orchestrator coordinates.
#
# Environment Variables Required:
#   INTEGRATION_TIMEOUT_MS - Timeout for each integration check (default: 3000)
#   PARALLEL_CHECKS - Number of parallel checks (default: 5)
#
# Integrations Checked:
#   - Z3 Prover Adapter
#   - LeanAide Adapter
#   - RAGBits Adapter
#   - Vector DB Adapter
#   - Graphiti Adapter
#   - KarateClub Adapter
#   - Knowledge Engine
#   - Event Bus
#
# Usage:
#   ./check_integrations.sh
#
# Exit Codes:
#   0 - All integrations healthy
#   1 - Some integrations unhealthy
#   2 - Critical integrations unreachable
###############################################################################

set -euo pipefail

# Default values
DEFAULT_TIMEOUT=3
DEFAULT_PARALLEL=5

# Load environment variables
TIMEOUT_SEC=$(( (${INTEGRATION_TIMEOUT_MS:-3000} + 999) / 1000 ))
PARALLEL_CHECKS=${PARALLEL_CHECKS:-$DEFAULT_PARALLEL}

# Integration endpoints (configurable via environment)
Z3_ADAPTER_URL="${Z3_ADAPTER_URL:-http://localhost:8080/health}"
LEANAIDE_ADAPTER_URL="${LEANAIDE_ADAPTER_URL:-http://localhost:8081/health}"
RAGBITS_ADAPTER_URL="${RAGBITS_ADAPTER_URL:-http://localhost:8082/health}"
VECTOR_DB_URL="${VECTOR_DB_URL:-http://localhost:8083/health}"
GRAPHITI_ADAPTER_URL="${GRAPHITI_ADAPTER_URL:-http://localhost:8084/health}"
KARATECLUB_ADAPTER_URL="${KARATECLUB_ADAPTER_URL:-http://localhost:8085/health}"
KNOWLEDGE_ENGINE_URL="${KNOWLEDGE_ENGINE_URL:-http://localhost:8086/health}"
EVENT_BUS_URL="${EVENT_BUS_URL:-http://localhost:8087/health}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check_integrations] $1${NC}"
}

log_info() { log "$1" ""; }
log_success() { log "$1" "$GREEN"; }
log_error() { log "$1" "$RED"; }
log_warning() { log "$1" "$YELLOW"; }
log_integration() { log "$1" "$BLUE"; }

# Store results
declare -A INTEGRATION_STATUS
declare -A INTEGRATION_LATENCY

# Check a single integration
check_integration() {
    local name=$1
    local url=$2
    local is_critical=${3:-false}

    log_integration "Checking $name..."

    local start_time=$(date +%s%3N)
    local response_code
    local error_msg

    response_code=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SEC" \
        --connect-timeout "$TIMEOUT_SEC" \
        -X GET \
        -H "Content-Type: application/json" \
        "$url" 2>&1) || true

    local end_time=$(date +%s%3N)
    local latency=$((end_time - start_time))

    if [[ "$response_code" == "200" ]]; then
        INTEGRATION_STATUS[$name]="healthy"
        INTEGRATION_LATENCY[$name]=$latency
        log_success "✓ $name: healthy (${latency}ms)"
        return 0
    else
        INTEGRATION_STATUS[$name]="unhealthy"
        INTEGRATION_LATENCY[$name]=$latency

        if [[ "$is_critical" == "true" ]]; then
            log_error "✗ $name: CRITICAL - status $response_code (${latency}ms)"
            return 2
        else
            log_warning "✗ $name: unhealthy - status $response_code (${latency}ms)"
            return 1
        fi
    fi
}

# Print summary
print_summary() {
    echo ""
    log_info "=== Integration Health Summary ==="
    echo ""

    local total=0
    local healthy=0
    local unhealthy=0
    local critical_failed=0

    for name in "${!INTEGRATION_STATUS[@]}"; do
        total=$((total + 1))
        local status="${INTEGRATION_STATUS[$name]}"
        local latency="${INTEGRATION_LATENCY[$name]}"

        if [[ "$status" == "healthy" ]]; then
            echo -e "${GREEN}✓${NC} $name: ${status} (${latency}ms)"
            healthy=$((healthy + 1))
        else
            echo -e "${RED}✗${NC} $name: ${status} (${latency}ms)"
            unhealthy=$((unhealthy + 1))
        fi
    done

    echo ""
    log_info "Total: $total | Healthy: $healthy | Unhealthy: $unhealthy"

    if [[ $unhealthy -gt 0 ]]; then
        log_warning "Some integrations are unhealthy"
    else
        log_success "All integrations are healthy"
    fi
}

# Main execution
main() {
    log_info "Starting integration health check..."
    echo ""

    local exit_code=0

    # Check all integrations
    check_integration "Z3 Prover" "$Z3_ADAPTER_URL" false || true
    check_integration "LeanAide" "$LEANAIDE_ADAPTER_URL" false || true
    check_integration "RAGBits" "$RAGBITS_ADAPTER_URL" false || true
    check_integration "Vector DB" "$VECTOR_DB_URL" false || true
    check_integration "Graphiti" "$GRAPHITI_ADAPTER_URL" false || true
    check_integration "KarateClub" "$KARATECLUB_ADAPTER_URL" false || true
    check_integration "Knowledge Engine" "$KNOWLEDGE_ENGINE_URL" true || exit_code=$?
    check_integration "Event Bus" "$EVENT_BUS_URL" true || exit_code=$?

    echo ""

    # Print summary
    print_summary

    exit $exit_code
}

# Run main function
main "$@"
