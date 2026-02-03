#!/bin/bash
# Docker Deployment Script - License: Apache 2.0
# Deploys OpenEvolve using Docker Compose

set -e

echo "=========================================="
echo "OpenEvolve Docker Deployment"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

cd "$PROJECT_DIR"

# Function to print status
print_status() {
    echo -e "${BLUE}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker found"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose found"
    
    # Check compose file
    if [ ! -f "$COMPOSE_FILE" ]; then
        print_error "docker-compose.yml not found"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Pull latest images
pull_images() {
    print_status "Pulling latest images..."
    docker-compose pull
    print_success "Images pulled"
}

# Build images
build_images() {
    print_status "Building images..."
    docker-compose build --no-cache
    print_success "Images built"
}

# Start services
start_services() {
    print_status "Starting services..."
    docker-compose up -d
    
    print_status "Waiting for services to be healthy..."
    sleep 10
    
    # Check service health
    if docker-compose ps | grep -q "Up"; then
        print_success "Services started successfully"
        echo
        echo "Service URLs:"
        echo "  REST API:   http://localhost:8000"
        echo "  GraphQL:    http://localhost:8001/graphql"
        echo "  Gateway:    http://localhost:8080"
        echo "  Grafana:    http://localhost:3000"
        echo "  Jaeger:     http://localhost:16686"
        echo "  Prometheus: http://localhost:9090"
    else
        print_error "Services failed to start"
        docker-compose logs
        exit 1
    fi
}

# Stop services
stop_services() {
    print_status "Stopping services..."
    docker-compose down
    print_success "Services stopped"
}

# View logs
view_logs() {
    docker-compose logs -f "$@"
}

# Check status
check_status() {
    print_status "Service Status:"
    docker-compose ps
    echo
    print_status "Resource Usage:"
    docker stats --no-stream
}

# Update deployment
update() {
    print_status "Updating deployment..."
    docker-compose pull
    docker-compose up -d
    print_success "Update complete"
}

# Clean up
cleanup() {
    print_warning "This will remove all containers, networks, and volumes!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup complete"
    else
        print_status "Cleanup cancelled"
    fi
}

# Backup data
backup() {
    print_status "Creating backup..."
    
    BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup volumes
    docker run --rm -v openevolve_valkey_data:/data -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/valkey.tar.gz -C /data .
    docker run --rm -v openevolve_openevolve_data:/data -v "$(pwd)/$BACKUP_DIR:/backup" alpine tar czf /backup/openevolve.tar.gz -C /data .
    
    print_success "Backup created in $BACKUP_DIR"
}

# Print usage
print_usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  up          - Start all services"
    echo "  down        - Stop all services"
    echo "  restart     - Restart services"
    echo "  build       - Build images"
    echo "  pull        - Pull latest images"
    echo "  update      - Pull and restart"
    echo "  logs        - View logs [service]"
    echo "  status      - Check status"
    echo "  ps          - List containers"
    echo "  exec        - Execute command in container"
    echo "  backup      - Backup data"
    echo "  cleanup     - Clean up containers and volumes"
    echo "  full        - Full deployment (build + up)"
    echo "  help        - Show this help"
}

# Main function
main() {
    case "${1:-up}" in
        up|start)
            check_prerequisites
            start_services
            ;;
        down|stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            start_services
            ;;
        build)
            check_prerequisites
            build_images
            ;;
        pull)
            check_prerequisites
            pull_images
            ;;
        update)
            check_prerequisites
            update
            ;;
        logs)
            shift
            view_logs "$@"
            ;;
        status)
            check_status
            ;;
        ps)
            docker-compose ps
            ;;
        exec)
            shift
            docker-compose exec "$@"
            ;;
        backup)
            backup
            ;;
        cleanup)
            cleanup
            ;;
        full)
            check_prerequisites
            build_images
            start_services
            echo
            print_success "Full deployment complete!"
            ;;
        help|--help|-h)
            print_usage
            ;;
        *)
            print_error "Unknown command: $1"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
