#!/bin/bash
# =============================================================================
# OpenEvolve Development Environment Startup Script
# =============================================================================
# This script starts all infrastructure services in the correct order with
# health checks to ensure everything is ready before proceeding.
#
# Usage:
#   ./scripts/dev-start.sh [--with-tools] [--skip-health-check]
#
# Options:
#   --with-tools    Start optional management UIs (pgAdmin, Redis Commander)
#   --skip-health-check  Skip health checks (not recommended)
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.infrastructure.yml"
ENV_FILE=".env.infrastructure"
COMPOSE_PROJECT_NAME="openevolve"

# Parse arguments
WITH_TOOLS=false
SKIP_HEALTH_CHECK=false

for arg in "$@"; do
    case $arg in
        --with-tools)
            WITH_TOOLS=true
            shift
            ;;
        --skip-health-check)
            SKIP_HEALTH_CHECK=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Functions
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

check_docker() {
    log_info "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    log_success "Docker is installed"
}

check_env_file() {
    log_info "Checking environment configuration..."

    if [ ! -f "$ENV_FILE" ]; then
        log_warning "Environment file not found: $ENV_FILE"
        log_info "Creating from template..."

        if [ -f ".env.infrastructure.example" ]; then
            cp .env.infrastructure.example "$ENV_FILE"
            log_warning "Please edit $ENV_FILE with your configuration before running again"
            exit 1
        else
            log_error "Template file not found: .env.infrastructure.example"
            exit 1
        fi
    fi

    log_success "Environment file found"
}

start_services() {
    log_info "Starting infrastructure services..."

    COMPOSE_CMD="docker compose"
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    fi

    # Build compose command
    COMPOSE_ARGS="-f $COMPOSE_FILE -p $COMPOSE_PROJECT_NAME --env-file $ENV_FILE"

    if [ "$WITH_TOOLS" = true ]; then
        COMPOSE_ARGS="$COMPOSE_ARGS --profile tools"
        log_info "Including management UI tools (pgAdmin, Redis Commander)"
    fi

    # Start services
    $COMPOSE_CMD $COMPOSE_ARGS up -d

    log_success "Services started"
}

wait_for_service() {
    SERVICE_NAME=$1
    SERVICE_HOST=$2
    SERVICE_PORT=$3
    MAX_ATTEMPTS=30
    ATTEMPT=0

    log_info "Waiting for $SERVICE_NAME to be ready..."

    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        if nc -z "$SERVICE_HOST" "$SERVICE_PORT" 2>/dev/null; then
            log_success "$SERVICE_NAME is ready"
            return 0
        fi

        ATTEMPT=$((ATTEMPT + 1))
        echo -n "."
        sleep 2
    done

    echo
    log_error "$SERVICE_NAME failed to start within expected time"
    return 1
}

run_health_checks() {
    if [ "$SKIP_HEALTH_CHECK" = true ]; then
        log_warning "Skipping health checks"
        return 0
    fi

    log_info "Running health checks..."

    # Check PostgreSQL
    wait_for_service "PostgreSQL" "localhost" "5432"

    # Check Qdrant
    wait_for_service "Qdrant" "localhost" "6333"

    # Check Redis
    wait_for_service "Redis" "localhost" "6379"

    log_success "All services are healthy"
}

print_service_info() {
    echo ""
    echo "============================================================================"
    echo " OpenEvolve Infrastructure Services"
    echo "============================================================================"
    echo ""
    echo "PostgreSQL:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  Database: openevolve"
    echo "  User: openevolve"
    echo "  Connection String: postgresql://openevolve:changeme@localhost:5432/openevolve"
    echo ""
    echo "Qdrant Vector Database:"
    echo "  HTTP API: http://localhost:6333"
    echo "  gRPC API: localhost:6334"
    echo "  Dashboard: http://localhost:6333/dashboard"
    echo ""
    echo "Redis:"
    echo "  Host: localhost"
    echo "  Port: 6379"
    echo "  URL: redis://localhost:6379"
    echo ""

    if [ "$WITH_TOOLS" = true ]; then
        echo "Management Tools:"
        echo "  pgAdmin: http://localhost:5050"
        echo "    Email: admin@openevolve.local"
        echo "    Password: (see .env.infrastructure)"
        echo "  Redis Commander: http://localhost:8081"
        echo ""
    fi

    echo "============================================================================"
    echo ""
    echo "Useful Commands:"
    echo "  View logs: docker compose -f $COMPOSE_FILE -p $COMPOSE_PROJECT_NAME logs -f"
    echo "  Stop services: docker compose -f $COMPOSE_FILE -p $COMPOSE_PROJECT_NAME down"
    echo "  Restart services: docker compose -f $COMPOSE_FILE -p $COMPOSE_PROJECT_NAME restart"
    echo ""
}

# Main execution
main() {
    echo ""
    log_info "OpenEvolve Infrastructure Setup"
    echo ""

    check_docker
    check_env_file
    start_services

    if [ "$SKIP_HEALTH_CHECK" = false ]; then
        sleep 5  # Give services a moment to start
        run_health_checks
    fi

    print_service_info

    log_success "Infrastructure setup complete!"
}

# Run main function
main "$@"
