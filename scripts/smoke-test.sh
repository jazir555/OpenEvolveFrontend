#!/bin/bash
# ============================================================================
# OpenEvolve Frontend - Smoke Test Script
# ============================================================================
# This script performs smoke tests after deployment to verify basic
# functionality of the deployed system.
#
# Operating Mode: ZERO TRUST - Verify deployment success
# ============================================================================

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="staging"
BASE_URL=""
TIMEOUT=30
VERBOSE=false

# Environment URLs
declare -A ENV_URLS=(
    ["staging"]="https://staging.openevolve.io"
    ["production"]="https://openevolve.io"
    ["local"]="http://localhost:8080"
)

# ============================================================================
# Logging Functions
# ============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# ============================================================================
# Test Functions
# ============================================================================
test_health_endpoint() {
    local url="$1/health"

    log_info "Testing health endpoint: $url"

    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)

    log_debug "Health endpoint response: $response"

    if [[ "$response" == "200" ]]; then
        log_success "Health endpoint is accessible"
        return 0
    else
        log_error "Health endpoint returned: $response"
        return 1
    fi
}

test_api_readiness() {
    local url="$1/api/v1/ready"

    log_info "Testing API readiness: $url"

    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)

    log_debug "Readiness endpoint response: $response"

    if [[ "$response" == "200" ]]; then
        log_success "API is ready"
        return 0
    else
        log_warning "API not ready (response: $response)"
        return 1
    fi
}

test_event_bus_connection() {
    local url="$1/api/v1/status/eventbus"

    log_info "Testing event bus connection: $url"

    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)

    log_debug "Event bus status response: $response"

    if [[ "$response" == "200" ]]; then
        log_success "Event bus is connected"
        return 0
    else
        log_warning "Event bus not accessible (response: $response)"
        return 1
    fi
}

test_workflow_engine() {
    local url="$1/api/v1/workflows/health"

    log_info "Testing workflow engine: $url"

    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)

    log_debug "Workflow engine response: $response"

    if [[ "$response" == "200" ]]; then
        log_success "Workflow engine is healthy"
        return 0
    else
        log_warning "Workflow engine not healthy (response: $response)"
        return 1
    fi
}

test_adapter_status() {
    local url="$1/api/v1/adapters/status"

    log_info "Testing adapter status: $url"

    local response
    response=$(curl -s --max-time "$TIMEOUT" "$url" 2>&1)

    log_debug "Adapter status response: $response"

    if echo "$response" | jq -e '.status == "operational"' >/dev/null 2>&1; then
        local adapter_count
        adapter_count=$(echo "$response" | jq -r '.adapters | length')
        log_success "Adapters operational ($adapter_count adapters)"
        return 0
    else
        log_warning "Adapters not fully operational"
        return 1
    fi
}

test_metrics_endpoint() {
    local url="$1/metrics"

    log_info "Testing metrics endpoint: $url"

    local response
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1)

    log_debug "Metrics endpoint response: $response"

    if [[ "$response" == "200" ]]; then
        log_success "Metrics endpoint is accessible"
        return 0
    else
        log_warning "Metrics endpoint not accessible (response: $response)"
        return 1
    fi
}

# ============================================================================
# Kubernetes-specific Tests
# ============================================================================
test_kubernetes_deployment() {
    local namespace=$1
    local deployment="openevolve-frontend"

    log_info "Testing Kubernetes deployment in namespace: $namespace"

    # Check if deployment exists
    if ! kubectl get deployment "$deployment" -n "$namespace" &>/dev/null; then
        log_error "Deployment not found: $deployment"
        return 1
    fi

    # Check deployment status
    local ready_replicas
    ready_replicas=$(kubectl get deployment "$deployment" -n "$namespace" \
        -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")

    local desired_replicas
    desired_replicas=$(kubectl get deployment "$deployment" -n "$namespace" \
        -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")

    log_debug "Replicas: $ready_replicas/$desired_replicas ready"

    if [[ "$ready_replicas" == "$desired_replicas" ]] && [[ "$ready_replicas" -gt 0 ]]; then
        log_success "Deployment is healthy ($ready_replicas/$desired_replicas replicas ready)"
        return 0
    else
        log_error "Deployment not ready ($ready_replicas/$desired_replicas replicas ready)"
        return 1
    fi
}

test_kubernetes_pods() {
    local namespace=$1
    local label="app=openevolve-frontend"

    log_info "Testing Kubernetes pods in namespace: $namespace"

    # Get pod status
    local pod_status
    pod_status=$(kubectl get pods -n "$namespace" -l "$label" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\t"}{.status.containerStatuses[0].ready}{"\n"}{end}' 2>&1)

    log_debug "Pod status:\n$pod_status"

    # Count pods
    local total_pods
    total_pods=$(kubectl get pods -n "$namespace" -l "$label" --no-headers 2>/dev/null | wc -l)

    local ready_pods
    ready_pods=$(kubectl get pods -n "$namespace" -l "$label" --no-headers 2>/dev/null | awk '{if ($4=="Running" && $3=="1/1") print $0}' | wc -l)

    log_debug "Pods: $ready_pods/$total_pods ready"

    if [[ "$ready_pods" == "$total_pods" ]] && [[ "$total_pods" -gt 0 ]]; then
        log_success "All pods are healthy ($ready_pods/$total_pods ready)"
        return 0
    else
        log_error "Not all pods are healthy ($ready_pods/$total_pods ready)"
        return 1
    fi
}

# ============================================================================
# Summary and Reporting
# ============================================================================
print_summary() {
    local total=$1
    local passed=$2
    local failed=$((total - passed))

    echo ""
    echo "========================================"
    echo "Smoke Test Summary"
    echo "========================================"
    echo "Total Tests:  $total"
    echo "Passed:       ${GREEN}$passed${NC}"
    echo "Failed:       ${RED}$failed${NC}"
    echo "========================================"

    if [[ $failed -eq 0 ]]; then
        log_success "All smoke tests passed!"
        return 0
    else
        log_error "Some smoke tests failed!"
        return 1
    fi
}

# ============================================================================
# Usage Function
# ============================================================================
usage() {
    cat <<EOF
Usage: $0 [options]

Options:
    -e, --environment <env>    Environment to test (staging|production|local)
    -u, --url <url>           Custom base URL (overrides environment)
    -t, --timeout <seconds>   Request timeout (default: 30)
    -k, --kubectl             Run Kubernetes-specific tests
    -n, --namespace <ns>      Kubernetes namespace (default: openevolve-<env>)
    -v, --verbose             Enable verbose output
    -h, --help                Show this help message

Examples:
    $0                                          # Test staging environment
    $0 -e production                            # Test production environment
    $0 -u http://localhost:8080                 # Test custom URL
    $0 -e production -k                         # Test production with kubectl checks
    $0 -e staging -t 60 -v                      # Test staging with 60s timeout, verbose

Environment Variables:
    ENVIRONMENT         Target environment
    BASE_URL           Custom base URL
    TIMEOUT            Request timeout
    NAMESPACE          Kubernetes namespace

EOF
}

# ============================================================================
# Main Test Execution
# ============================================================================
main() {
    local total_tests=0
    local passed_tests=0
    local run_k8s_tests=false
    local namespace=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -u|--url)
                BASE_URL="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -k|--kubectl)
                run_k8s_tests=true
                shift
                ;;
            -n|--namespace)
                namespace="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Use environment variables if set
    BASE_URL="${BASE_URL:-${ENV_URLS[$ENVIRONMENT]}}"
    namespace="${namespace:-openevolve-${ENVIRONMENT}}"

    # Validate base URL
    if [[ -z "$BASE_URL" ]]; then
        log_error "Base URL not specified. Use --url or --environment"
        usage
        exit 1
    fi

    echo "========================================"
    echo "OpenEvolve Frontend - Smoke Tests"
    echo "========================================"
    echo "Environment: $ENVIRONMENT"
    echo "Base URL: $BASE_URL"
    echo "Timeout: ${TIMEOUT}s"
    echo "========================================"
    echo ""

    # Run HTTP-based tests
    log_info "Running HTTP-based smoke tests..."
    echo ""

    # Health endpoint
    ((total_tests++))
    if test_health_endpoint "$BASE_URL"; then
        ((passed_tests++))
    fi

    # API readiness
    ((total_tests++))
    if test_api_readiness "$BASE_URL"; then
        ((passed_tests++))
    fi

    # Event bus connection
    ((total_tests++))
    if test_event_bus_connection "$BASE_URL"; then
        ((passed_tests++))
    fi

    # Workflow engine
    ((total_tests++))
    if test_workflow_engine "$BASE_URL"; then
        ((passed_tests++))
    fi

    # Adapter status
    ((total_tests++))
    if test_adapter_status "$BASE_URL"; then
        ((passed_tests++))
    fi

    # Metrics endpoint
    ((total_tests++))
    if test_metrics_endpoint "$BASE_URL"; then
        ((passed_tests++))
    fi

    echo ""

    # Run Kubernetes-specific tests if requested
    if [[ "$run_k8s_tests" == true ]]; then
        log_info "Running Kubernetes-specific tests..."
        echo ""

        # Check if kubectl is available
        if ! command -v kubectl &>/dev/null; then
            log_warning "kubectl not found. Skipping Kubernetes tests."
        else
            # Deployment status
            ((total_tests++))
            if test_kubernetes_deployment "$namespace"; then
                ((passed_tests++))
            fi

            # Pod status
            ((total_tests++))
            if test_kubernetes_pods "$namespace"; then
                ((passed_tests++))
            fi
        fi

        echo ""
    fi

    # Print summary and exit
    print_summary "$total_tests" "$passed_tests"
    exit $?
}

# ============================================================================
# Script Entry Point
# ============================================================================
main "$@"
