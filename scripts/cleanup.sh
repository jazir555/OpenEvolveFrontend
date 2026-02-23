#!/bin/bash
# =============================================================================
# OpenEvolve Cleanup Script
# License: Apache 2.0
# Description: Stop services and clean up containers, volumes, and artifacts
# Usage: ./cleanup.sh [--volumes] [--all] [--dry-run]
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CLEAN_VOLUMES=false
CLEAN_ALL=false
DRY_RUN=false

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OpenEvolve Cleanup Script - Stop services and clean up

OPTIONS:
    --volumes           Remove Docker volumes (WARNING: Deletes data!)
    --all               Remove everything including images and build cache
    --dry-run           Show what would be done without executing
    -h, --help          Show this help message

EXAMPLES:
    $0                              # Stop services only
    $0 --volumes                    # Stop services and remove volumes
    $0 --all                        # Complete cleanup including images

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --volumes)
            CLEAN_VOLUMES=true
            shift
            ;;
        --all)
            CLEAN_ALL=true
            CLEAN_VOLUMES=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     OpenEvolve Frontend - Cleanup Script                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

cd "${PROJECT_ROOT}"

# Step 1: Stop all services
log_info "Step 1: Stopping all services..."
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would stop services: docker compose down"
else
    docker compose down
    log_success "Services stopped"
fi

# Step 2: Remove containers
log_info "Step 2: Removing containers..."
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would remove containers"
else
    docker compose rm -f -v 2>/dev/null || true
    log_success "Containers removed"
fi

# Step 3: Remove volumes if requested
if [ "$CLEAN_VOLUMES" = true ]; then
    log_warning "Step 3: Removing Docker volumes (WARNING: This deletes data!)..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would remove volumes"
    else
        read -p "Are you sure you want to delete all data? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            docker volume rm openevolve_frontend_valkey_data 2>/dev/null || true
            docker volume rm openevolve_frontend_prometheus_data 2>/dev/null || true
            docker volume rm openevolve_frontend_grafana_data 2>/dev/null || true
            docker volume rm openevolve_frontend_openevolve_data 2>/dev/null || true
            log_success "Volumes removed"
        else
            log_warning "Volume removal cancelled"
        fi
    fi
fi

# Step 4: Clean build artifacts
log_info "Step 4: Cleaning build artifacts..."
if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would clean build artifacts"
else
    # Clean node_modules from glue layer
    find "${PROJECT_ROOT}/glue" -name "node_modules" -type d -exec rm -rf {} + 2>/dev/null || true

    # Clean dist directories
    find "${PROJECT_ROOT}/glue" -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true

    # Clean log files older than 7 days
    find "${PROJECT_ROOT}/logs" -name "*.log" -mtime +7 -delete 2>/dev/null || true

    log_success "Build artifacts cleaned"
fi

# Step 5: Remove images and cache if --all
if [ "$CLEAN_ALL" = true ]; then
    log_info "Step 5: Removing Docker images and build cache..."
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY-RUN] Would remove Docker images and build cache"
    else
        docker image prune -af
        docker builder prune -af
        log_success "Docker images and cache removed"
    fi
fi

echo ""
log_success "Cleanup completed successfully!"
echo ""
log_info "To restart services, run: ./scripts/quick-start.sh"
