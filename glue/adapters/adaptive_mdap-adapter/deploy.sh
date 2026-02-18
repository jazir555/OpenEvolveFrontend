#!/bin/bash
###############################################################################
# Deployment Script for Adaptive MDAP/MAKER Adapter
#
# Usage:
#   ./deploy.sh [environment]
#
# Environments:
#   - local: Docker Compose (default)
#   - k8s: Kubernetes
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default environment
ENVIRONMENT="${1:-local}"

###############################################################################
# Pre-flight checks
###############################################################################

preflight_checks() {
    log_info "Running pre-flight checks..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check if kubectl is installed (for k8s deployment)
    if [ "$ENVIRONMENT" = "k8s" ]; then
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl is not installed"
            exit 1
        fi
    fi

    # Check if adapter directory exists
    if [ ! -d "$SCRIPT_DIR" ]; then
        log_error "Adapter directory not found: $SCRIPT_DIR"
        exit 1
    fi

    # Check if required files exist
    if [ ! -f "$SCRIPT_DIR/Dockerfile" ]; then
        log_error "Dockerfile not found"
        exit 1
    fi

    if [ ! -f "$SCRIPT_DIR/requirements.txt" ]; then
        log_error "requirements.txt not found"
        exit 1
    fi

    log_info "Pre-flight checks passed"
}

###############################################################################
# Build Docker image
###############################################################################

build_image() {
    log_info "Building Docker image..."

    cd "$SCRIPT_DIR"

    # Build image
    docker build \
        -t adaptive-mdap-adapter:latest \
        -t adaptive-mdap-adapter:1.0.0 \
        .

    if [ $? -eq 0 ]; then
        log_info "✓ Docker image built successfully"
    else
        log_error "✗ Docker image build failed"
        exit 1
    fi
}

###############################################################################
# Run contract tests
###############################################################################

run_tests() {
    log_info "Running contract tests..."

    cd "$SCRIPT_DIR"

    # Run tests in container
    docker run --rm \
        -e ADAPTIVE_MDAP_TIMEOUT_MS=10000 \
        -e ADAPTIVE_MDAP_LOG_LEVEL=DEBUG \
        adaptive-mdap-adapter:1.0.0 \
        python -m pytest tests/contract.test.py -v

    if [ $? -eq 0 ]; then
        log_info "✓ Contract tests passed"
    else
        log_error "✗ Contract tests failed"
        exit 1
    fi
}

###############################################################################
# Deploy to local environment (Docker Compose)
###############################################################################

deploy_local() {
    log_info "Deploying to local environment (Docker Compose)..."

    # Create docker-compose.yml if it doesn't exist
    cat > "$SCRIPT_DIR/docker-compose.yml" <<EOF
version: '3.8'

services:
  adapter:
    image: adaptive-mdap-adapter:1.0.0
    container_name: adaptive-mdap-adapter
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - ADAPTIVE_MDAP_TIMEOUT_MS=5000
      - ADAPTIVE_MDAP_LOG_LEVEL=INFO
      - PROMETHEUS_PORT=9090
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.path.insert(0, '/app/src'); from adaptive_mdap_adapter import get_adapter; adapter = get_adapter(); health = adapter.health_check(); sys.exit(0 if health['status'] == 'healthy' else 1)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s
EOF

    # Start container
    docker-compose up -d

    if [ $? -eq 0 ]; then
        log_info "✓ Adapter started in Docker"
        log_info "  Health check: curl http://localhost:8080/health"
        log_info "  Metrics: curl http://localhost:9090/metrics"
    else
        log_error "✗ Failed to start adapter"
        exit 1
    fi
}

###############################################################################
# Deploy to Kubernetes
###############################################################################

deploy_k8s() {
    log_info "Deploying to Kubernetes..."

    # Check if namespace exists
    if ! kubectl get namespace openevolve &> /dev/null; then
        log_info "Creating namespace: openevolve"
        kubectl create namespace openevolve
    fi

    # Apply ConfigMap
    log_info "Applying ConfigMap..."
    kubectl apply -f "$SCRIPT_DIR/k8s-deployment.yaml"

    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/adaptive-mdap-adapter -n openevolve --timeout=5m

    # Get service info
    log_info "Getting service information..."
    kubectl get svc adaptive-mdap-adapter -n openevolve

    # Get pods
    log_info "Getting pods..."
    kubectl get pods -n openevolve -l app=adaptive-mdap-adapter
}

###############################################################################
# Run probes
###############################################################################

run_probes() {
    log_info "Running runtime verification probes..."

    cd "$SCRIPT_DIR"

    # Make sure probes are executable
    chmod +x probes/*.sh

    # Run master probe
    ./probes/check_api.sh

    if [ $? -eq 0 ]; then
        log_info "✓ All probes passed"
    else
        log_error "✗ Some probes failed"
        exit 1
    fi
}

###############################################################################
# Show status
###############################################################################

show_status() {
    log_info "Adapter Status"
    echo "----------------------------------------"

    if [ "$ENVIRONMENT" = "local" ]; then
        if command -v docker-compose &> /dev/null; then
            docker-compose ps
        fi

        # Health check
        if command -v curl &> /dev/null; then
            echo ""
            log_info "Health check:"
            curl -s http://localhost:8080/health | python -m json.tool 2>/dev/null || echo "  Adapter not running"
        fi
    elif [ "$ENVIRONMENT" = "k8s" ]; then
        kubectl get pods -n openevolve -l app=adaptive-mdap-adapter
        kubectl get svc adaptive-mdap-adapter -n openevolve
    fi
}

###############################################################################
# Main deployment flow
###############################################################################

main() {
    echo "=========================================="
    echo "Adaptive MDAP/MAKER Adapter Deployment"
    echo "=========================================="
    echo "Environment: $ENVIRONMENT"
    echo ""

    # Pre-flight checks
    preflight_checks

    # Build image
    build_image

    # Run tests
    run_tests

    # Run probes
    run_probes

    # Deploy based on environment
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        k8s)
            deploy_k8s
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Valid environments: local, k8s"
            exit 1
            ;;
    esac

    # Show status
    show_status

    echo ""
    echo "=========================================="
    log_info "✓ Deployment complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    if [ "$ENVIRONMENT" = "local" ]; then
        echo "  1. Check health: curl http://localhost:8080/health"
        echo "  2. View metrics: curl http://localhost:9090/metrics"
        echo "  3. Run CLI: python adapter-cli.py health"
        echo "  4. View logs: docker-compose logs -f"
        echo "  5. Stop: docker-compose down"
    elif [ "$ENVIRONMENT" = "k8s" ]; then
        echo "  1. Check pods: kubectl get pods -n openevolve"
        echo "  2. Port forward: kubectl port-forward svc/adaptive-mdap-adapter 8080:8080 -n openevolve"
        echo "  3. Check logs: kubectl logs -f deployment/adaptive-mdap-adapter -n openevolve"
        echo "  4. Run CLI: kubectl exec -it deployment/adaptive-mdap-adapter -n openevolve -- python adapter-cli.py health"
    fi
    echo ""
}

# Run main
main "$@"
