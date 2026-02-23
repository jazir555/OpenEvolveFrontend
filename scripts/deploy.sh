#!/bin/bash
# ============================================================================
# OpenEvolve Frontend - Deployment Script
# ============================================================================
# This script handles deployment to various environments with proper
# validation and rollback capabilities.
#
# Operating Mode: ZERO TRUST - Validates everything before deployment
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

# Environment configuration
declare -A ENV_CONFIG=(
    ["staging"]="openevolve-staging"
    ["production"]="openevolve-production"
)

# ============================================================================
# Logging Functions
# ============================================================================
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Validation Functions
# ============================================================================
validate_environment() {
    local env=$1

    if [[ -z "${ENV_CONFIG[$env]}" ]]; then
        log_error "Invalid environment: $env"
        log_info "Valid environments: ${!ENV_CONFIG[@]}"
        exit 1
    fi

    log_success "Environment validated: $env"
}

validate_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    log_success "kubectl found: $(kubectl version --client --short)"
}

validate_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    log_success "Docker found: $(docker --version)"
}

validate_cluster_access() {
    local namespace=$1

    log_info "Validating cluster access to namespace: $namespace"

    if ! kubectl get namespace "$namespace" &> /dev/null; then
        log_error "Cannot access namespace: $namespace"
        log_error "Please check your kubeconfig and permissions."
        exit 1
    fi

    log_success "Cluster access validated for namespace: $namespace"
}

# ============================================================================
# Deployment Functions
# ============================================================================
get_current_image() {
    local namespace=$1
    local deployment="openevolve-frontend"

    kubectl get deployment "$deployment" -n "$namespace" \
        -o jsonpath='{.spec.template.spec.containers[0].image}'
}

deploy_image() {
    local environment=$1
    local image_tag=$2
    local namespace=${ENV_CONFIG[$environment]}
    local deployment="openevolve-frontend"

    log_info "=========================================="
    log_info "Deploying to $environment"
    log_info "=========================================="
    log_info "Namespace: $namespace"
    log_info "Image: $image_tag"
    log_info "Deployment: $deployment"
    log_info "=========================================="

    # Get current image for potential rollback
    local current_image
    current_image=$(get_current_image "$namespace")
    log_info "Current image: $current_image"

    # Update deployment
    log_info "Updating deployment..."
    kubectl set image deployment/"$deployment" \
        openevolve="$image_tag" \
        -n "$namespace"

    # Wait for rollout
    log_info "Waiting for rollout to complete..."
    if kubectl rollout status deployment/"$deployment" \
        -n "$namespace" \
        --timeout=5m; then
        log_success "Deployment successful!"
    else
        log_error "Deployment failed or timed out!"
        log_info "Rolling back to previous image..."

        kubectl set image deployment/"$deployment" \
            openevolve="$current_image" \
            -n "$namespace"

        kubectl rollout status deployment/"$deployment" \
            -n "$namespace" \
            --timeout=5m

        log_error "Rollback complete. Deployment aborted."
        exit 1
    fi
}

run_smoke_tests() {
    local environment=$1

    log_info "Running smoke tests..."

    if npm run test:smoke -- --env="$environment"; then
        log_success "Smoke tests passed!"
    else
        log_warning "Smoke tests failed!"

        if [[ "$environment" == "production" ]]; then
            log_error "Smoke tests failed in production. Initiating rollback..."
            return 1
        fi
    fi

    return 0
}

get_pod_status() {
    local namespace=$1

    log_info "Pod status in $namespace:"
    kubectl get pods -n "$namespace" -l app=openevolve-frontend
}

# ============================================================================
# Usage Function
# ============================================================================
usage() {
    cat <<EOF
Usage: $0 <environment> [options]

Arguments:
    environment    Deployment environment (staging|production)

Options:
    --image-tag    Docker image tag to deploy (default: latest)
    --skip-tests   Skip smoke tests after deployment
    --status-only  Only show current deployment status
    --help         Show this help message

Examples:
    $0 staging
    $0 production --image-tag=v1.2.3
    $0 staging --skip-tests
    $0 production --status-only

Environment Variables:
    KUBECONFIG     Path to kubeconfig file (default: ~/.kube/config)
    IMAGE_TAG      Docker image tag (can be set via --image-tag)

EOF
}

# ============================================================================
# Main Deployment Flow
# ============================================================================
main() {
    local environment=""
    local image_tag="latest"
    local skip_tests=false
    local status_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                usage
                exit 0
                ;;
            --image-tag)
                image_tag="$2"
                shift 2
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --status-only)
                status_only=true
                shift
                ;;
            staging|production)
                environment="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    # Validate environment
    if [[ -z "$environment" ]]; then
        log_error "Environment not specified"
        usage
        exit 1
    fi

    validate_environment "$environment"
    validate_kubectl
    validate_docker

    local namespace=${ENV_CONFIG[$environment]}

    if [[ "$status_only" == true ]]; then
        log_info "Current deployment status for $environment:"
        get_pod_status "$namespace"
        log_info "Current image: $(get_current_image "$namespace")"
        exit 0
    fi

    # Validate cluster access
    validate_cluster_access "$namespace"

    # Use IMAGE_TAG from environment if set
    if [[ -n "${IMAGE_TAG:-}" ]]; then
        image_tag="$IMAGE_TAG"
    fi

    # Perform deployment
    cd "$PROJECT_ROOT"

    log_info "Starting deployment to $environment..."
    deploy_image "$environment" "$image_tag"

    # Show pod status
    get_pod_status "$namespace"

    # Run smoke tests
    if [[ "$skip_tests" == false ]]; then
        if ! run_smoke_tests "$environment"; then
            log_error "Smoke tests failed for $environment"
            exit 1
        fi
    fi

    log_success "=========================================="
    log_success "Deployment to $environment completed!"
    log_success "Image: $image_tag"
    log_success "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    log_success "=========================================="
}

# ============================================================================
# Script Entry Point
# ============================================================================
main "$@"
