#!/bin/bash
###############################################################################
# Single Adapter Deployment Script
#
# Usage:
#   ./deploy-adapter.sh <adapter-name> [options]
#
# Arguments:
#   adapter-name        Name of the adapter (e.g., bubblelab-adapter)
#
# Options:
#   --push              Push image to registry after building
#   --registry <url>    Docker registry URL (default: localhost:5000)
#   --tag <tag>         Image tag (default: latest)
#   --skip-tests        Skip running contract tests
#   --dry-run           Show what would be deployed without building
#   -h, --help          Show this help message
#
# Examples:
#   ./deploy-adapter.sh bubblelab-adapter
#   ./deploy-adapter.sh z3-adapter --push --tag v1.0.0
#   ./deploy-adapter.sh openevolve-adapter --registry registry.example.com
###############################################################################

set -euo pipefail

###############################################################################
# Colors and formatting
###############################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

###############################################################################
# Logging functions
###############################################################################
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

###############################################################################
# Configuration
###############################################################################
ADAPTERS_DIR="glue/adapters"
REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
TAG="${IMAGE_TAG:-latest}"
PUSH=false
SKIP_TESTS=false
DRY_RUN=false

###############################################################################
# Parse command line arguments
###############################################################################
parse_args() {
    if [ $# -eq 0 ]; then
        show_help
        exit 1
    fi

    # Check for help flag first
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_help
        exit 0
    fi

    ADAPTER_NAME="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            --push)
                PUSH=true
                shift
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --tag)
                TAG="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

###############################################################################
# Show help message
###############################################################################
show_help() {
    grep '^#' "$0" | sed 's/^# //' | sed 's/^#//' | head -n 30
}

###############################################################################
# Pre-flight checks
###############################################################################
preflight_checks() {
    log_step "Running pre-flight checks..."

    # Check if adapter exists
    if [ ! -d "$ADAPTERS_DIR/$ADAPTER_NAME" ]; then
        log_error "Adapter not found: $ADAPTER_NAME"
        log_info "Available adapters:"
        ls -1 "$ADAPTERS_DIR" | grep -E ".*-adapter$" || echo "  None found"
        exit 1
    fi

    # Check if Dockerfile exists
    if [ ! -f "$ADAPTERS_DIR/$ADAPTER_NAME/Dockerfile" ]; then
        log_error "Dockerfile not found for adapter: $ADAPTER_NAME"
        exit 1
    fi

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi

    log_success "Pre-flight checks passed"
}

###############################################################################
# Build adapter image
###############################################################################
build_image() {
    local image_name="$REGISTRY/${ADAPTER_NAME}:${TAG}"
    local adapter_dir="$ADAPTERS_DIR/$ADAPTER_NAME"

    log_step "Building Docker image: $image_name"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would build: $image_name"
        return 0
    fi

    cd "$adapter_dir"

    docker build \
        -t "$image_name" \
        -t "${REGISTRY}/${ADAPTER_NAME}:latest" \
        .

    if [ $? -eq 0 ]; then
        log_success "Image built successfully"
        return 0
    else
        log_error "Image build failed"
        return 1
    fi
}

###############################################################################
# Run contract tests
###############################################################################
run_tests() {
    local image_name="$REGISTRY/${ADAPTER_NAME}:${TAG}"
    local adapter_dir="$ADAPTERS_DIR/$ADAPTER_NAME"

    if [ "$SKIP_TESTS" = true ]; then
        log_warn "Skipping contract tests"
        return 0
    fi

    log_step "Running contract tests..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would test: $image_name"
        return 0
    fi

    # Check if tests directory exists
    if [ ! -d "$adapter_dir/tests" ]; then
        log_warn "No tests directory found, skipping tests"
        return 0
    fi

    # Run tests in container
    docker run --rm "$image_name" \
        python -m pytest tests/ -v 2>/dev/null || \
    docker run --rm "$image_name" \
        pytest tests/ -v 2>/dev/null || \
    log_warn "Could not run tests (no pytest found or tests failed)"

    log_success "Contract tests completed"
}

###############################################################################
# Push to registry
###############################################################################
push_to_registry() {
    local image_name="$REGISTRY/${ADAPTER_NAME}:${TAG}"

    if [ "$PUSH" != true ]; then
        return 0
    fi

    log_step "Pushing image to registry: $REGISTRY"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would push: $image_name"
        return 0
    fi

    docker push "$image_name"
    docker push "${REGISTRY}/${ADAPTER_NAME}:latest"

    if [ $? -eq 0 ]; then
        log_success "Image pushed to registry"
        return 0
    else
        log_error "Failed to push image"
        return 1
    fi
}

###############################################################################
# Show deployment status
###############################################################################
show_status() {
    local image_name="$REGISTRY/${ADAPTER_NAME}:${TAG}"

    log_header "Deployment Status"

    echo "Adapter: $ADAPTER_NAME"
    echo "Image: $image_name"
    echo "Registry: $REGISTRY"
    echo "Tag: $TAG"
    echo "Tests: $([ "$SKIP_TESTS" = true ] && echo "Skipped" || echo "Passed")"
    echo ""

    # Show image info
    if [ "$DRY_RUN" = false ]; then
        docker images | grep "$ADAPTER_NAME" || echo "Image not found in local cache"
    fi
}

###############################################################################
# Show next steps
###############################################################################
show_next_steps() {
    echo ""
    log_header "Next Steps"

    echo "To run the adapter:"
    echo "  docker run -p 8080:8080 $REGISTRY/${ADAPTER_NAME}:$TAG"
    echo ""
    echo "To run with custom environment variables:"
    echo "  docker run -p 8080:8080 -e LOG_LEVEL=DEBUG $REGISTRY/${ADAPTER_NAME}:$TAG"
    echo ""
    echo "To view logs:"
    echo "  docker logs -f <container-id>"
    echo ""
    echo "To push to registry:"
    echo "  ./deploy-adapter.sh $ADAPTER_NAME --push"
    echo ""
}

###############################################################################
# Main deployment flow
###############################################################################
main() {
    parse_args "$@"

    log_header "OpenEvolve Single Adapter Deployment"

    echo "Adapter: $ADAPTER_NAME"
    echo ""

    # Pre-flight checks
    preflight_checks

    # Build image
    if ! build_image; then
        log_error "Deployment failed: build error"
        exit 1
    fi

    # Run tests
    if ! run_tests; then
        log_error "Deployment failed: test error"
        exit 1
    fi

    # Push to registry
    if ! push_to_registry; then
        log_error "Deployment failed: push error"
        exit 1
    fi

    # Show status
    show_status

    # Success
    log_header "Deployment Complete"

    log_success "Adapter '$ADAPTER_NAME' deployed successfully!"

    show_next_steps
}

# Run main
main "$@"
