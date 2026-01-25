#!/bin/bash
# =============================================================================
# OpenEvolve Development Environment Stop Script
# =============================================================================
# This script stops all infrastructure services gracefully.
#
# Usage:
#   ./scripts/dev-stop.sh [--volumes]
#
# Options:
#   --volumes    Also remove named volumes (WARNING: This deletes data!)
# =============================================================================

set -e

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
REMOVE_VOLUMES=false

for arg in "$@"; do
    case $arg in
        --volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

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

# Main execution
main() {
    echo ""
    log_info "Stopping OpenEvolve Infrastructure Services"
    echo ""

    COMPOSE_CMD="docker compose"
    if command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
    fi

    COMPOSE_ARGS="-f $COMPOSE_FILE -p $COMPOSE_PROJECT_NAME --env-file $ENV_FILE"

    if [ "$REMOVE_VOLUMES" = true ]; then
        log_warning "This will delete all data in the volumes!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Aborted"
            exit 0
        fi
        COMPOSE_ARGS="$COMPOSE_ARGS -v"
    fi

    $COMPOSE_CMD $COMPOSE_ARGS down

    if [ "$REMOVE_VOLUMES" = true ]; then
        log_success "Services stopped and volumes removed"
    else
        log_success "Services stopped (volumes preserved)"
    fi

    echo ""
    log_info "To start services again, run: ./scripts/dev-start.sh"
    echo ""
}

main "$@"
