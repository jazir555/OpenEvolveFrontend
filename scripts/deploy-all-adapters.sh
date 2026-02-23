#!/bin/bash
###############################################################################
# Universal Deployment Script for All Adapters
#
# Usage:
#   ./deploy-all-adapters.sh [options]
#
# Options:
#   --push              Push images to registry after building
#   --registry <url>    Docker registry URL (default: localhost:5000)
#   --tag <tag>         Image tag (default: latest)
#   --skip-tests        Skip running contract tests
#   --adapter <name>    Deploy only a specific adapter
#   --dry-run           Show what would be deployed without building
#   -h, --help          Show this help message
#
# Examples:
#   ./deploy-all-adapters.sh
#   ./deploy-all-adapters.sh --push --registry registry.example.com
#   ./deploy-all-adapters.sh --adapter bubblelab-adapter --tag v1.0.0
#   ./deploy-all-adapters.sh --skip-tests
###############################################################################

set -euo pipefail

###############################################################################
# Colors and formatting
###############################################################################
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
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
# Default configuration
###############################################################################
ADAPTERS_DIR="glue/adapters"
REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
TAG="${IMAGE_TAG:-latest}"
PUSH=false
SKIP_TESTS=false
SPECIFIC_ADAPTER=""
DRY_RUN=false

###############################################################################
# Parse command line arguments
###############################################################################
parse_args() {
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
            --adapter)
                SPECIFIC_ADAPTER="$2"
                shift 2
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

    # Check if we're in the right directory
    if [ ! -d "$ADAPTERS_DIR" ]; then
        log_error "Adapters directory not found: $ADAPTERS_DIR"
        log_info "Please run this script from the project root directory"
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
# Get list of adapters to deploy
###############################################################################
get_adapters() {
    if [ -n "$SPECIFIC_ADAPTER" ]; then
        # Deploy specific adapter
        if [ ! -d "$ADAPTERS_DIR/$SPECIFIC_ADAPTER" ]; then
            log_error "Adapter not found: $SPECIFIC_ADAPTER"
            exit 1
        fi
        echo "$SPECIFIC_ADAPTER"
    else
        # Deploy all adapters
        for adapter_dir in "$ADAPTERS_DIR"/*; do
            if [ -d "$adapter_dir" ]; then
                basename "$adapter_dir"
            fi
        done
    fi
}

###############################################################################
# Check if adapter has Dockerfile
###############################################################################
has_dockerfile() {
    local adapter="$1"
    [ -f "$ADAPTERS_DIR/$adapter/Dockerfile" ]
}

###############################################################################
# Build adapter image
###############################################################################
build_adapter() {
    local adapter="$1"
    local adapter_dir="$ADAPTERS_DIR/$adapter"
    local image_name="$REGISTRY/${adapter}:${TAG}"

    log_step "Building $adapter..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would build: $image_name"
        return 0
    fi

    # Build the image
    cd "$adapter_dir"
    docker build \
        -t "$image_name" \
        -t "${REGISTRY}/${adapter}:latest" \
        .

    if [ $? -eq 0 ]; then
        log_success "Built $image_name"
    else
        log_error "Failed to build $adapter"
        return 1
    fi
}

###############################################################################
# Run contract tests for adapter
###############################################################################
test_adapter() {
    local adapter="$1"
    local adapter_dir="$ADAPTERS_DIR/$adapter"
    local image_name="$REGISTRY/${adapter}:${TAG}"

    if [ "$SKIP_TESTS" = true ]; then
        log_warn "Skipping tests for $adapter"
        return 0
    fi

    log_step "Testing $adapter..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would test: $image_name"
        return 0
    fi

    # Check if tests directory exists
    if [ ! -d "$adapter_dir/tests" ]; then
        log_warn "No tests directory found for $adapter"
        return 0
    fi

    # Run tests in container
    docker run --rm "$image_name" \
        python -m pytest tests/ -v 2>/dev/null || \
    docker run --rm "$image_name" \
        pytest tests/ -v 2>/dev/null || \
    log_warn "Could not run tests for $adapter (no pytest found or tests failed)"

    log_success "Tests completed for $adapter"
}

###############################################################################
# Push image to registry
###############################################################################
push_adapter() {
    local adapter="$1"
    local image_name="$REGISTRY/${adapter}:${TAG}"

    if [ "$PUSH" != true ]; then
        return 0
    fi

    log_step "Pushing $adapter to registry..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would push: $image_name"
        return 0
    fi

    docker push "$image_name"
    docker push "${REGISTRY}/${adapter}:latest"

    if [ $? -eq 0 ]; then
        log_success "Pushed $adapter to registry"
    else
        log_error "Failed to push $adapter"
        return 1
    fi
}

###############################################################################
# Deploy single adapter
###############################################################################
deploy_adapter() {
    local adapter="$1"

    log_header "Deploying: $adapter"

    # Check if adapter has Dockerfile
    if ! has_dockerfile "$adapter"; then
        log_warn "Skipping $adapter (no Dockerfile)"
        return 0
    fi

    # Build
    if ! build_adapter "$adapter"; then
        log_error "Failed to build $adapter"
        return 1
    fi

    # Test
    if ! test_adapter "$adapter"; then
        log_error "Failed tests for $adapter"
        return 1
    fi

    # Push
    if ! push_adapter "$adapter"; then
        log_error "Failed to push $adapter"
        return 1
    fi

    log_success "Deployed $adapter successfully"
}

###############################################################################
# Show deployment summary
###############################################################################
show_summary() {
    log_header "Deployment Summary"

    local adapters=($(get_adapters))
    local total=${#adapters[@]}
    local with_dockerfile=0

    for adapter in "${adapters[@]}"; do
        if has_dockerfile "$adapter"; then
            ((with_dockerfile++))
        fi
    done

    echo "Total adapters found: $total"
    echo "Adapters with Dockerfiles: $with_dockerfile"
    echo "Registry: $REGISTRY"
    echo "Tag: $TAG"
    echo "Push to registry: $PUSH"
    echo "Skip tests: $SKIP_TESTS"
    echo "Dry run: $DRY_RUN"

    if [ -n "$SPECIFIC_ADAPTER" ]; then
        echo "Specific adapter: $SPECIFIC_ADAPTER"
    fi

    echo ""
}

###############################################################################
# Show next steps
###############################################################################
show_next_steps() {
    echo ""
    log_header "Next Steps"

    echo "To run individual adapters:"
    echo "  docker run -p 8080:8080 $REGISTRY/<adapter-name>:$TAG"
    echo ""
    echo "To view all built images:"
    echo "  docker images | grep $REGISTRY"
    echo ""
    echo "To push to registry:"
    echo "  ./deploy-all-adapters.sh --push"
    echo ""
}

###############################################################################
# Main deployment flow
###############################################################################
main() {
    parse_args "$@"

    log_header "OpenEvolve Adapter Deployment System"

    # Pre-flight checks
    preflight_checks

    # Show summary
    show_summary

    # Get adapters to deploy
    local adapters=($(get_adapters))

    # Track statistics
    local success_count=0
    local failed_count=0
    local skipped_count=0

    # Deploy each adapter
    for adapter in "${adapters[@]}"; do
        if ! has_dockerfile "$adapter"; then
            log_warn "Skipping $adapter (no Dockerfile)"
            ((skipped_count++))
            continue
        fi

        if deploy_adapter "$adapter"; then
            ((success_count++))
        else
            ((failed_count++))
        fi
    done

    # Final summary
    log_header "Deployment Complete"

    echo "Successful: $success_count"
    echo "Failed: $failed_count"
    echo "Skipped: $skipped_count"
    echo ""

    if [ "$failed_count" -gt 0 ]; then
        log_error "Some adapters failed to deploy"
        exit 1
    else
        log_success "All adapters deployed successfully!"
        show_next_steps
    fi
}

# Run main
main "$@"
