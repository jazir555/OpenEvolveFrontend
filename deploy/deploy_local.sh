#!/bin/bash
# Local Deployment Script - License: Apache 2.0
# Deploys OpenEvolve integration system locally

set -e

echo "=========================================="
echo "OpenEvolve Local Deployment"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
LOGS_DIR="$PROJECT_DIR/logs"

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
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_success "Python $PYTHON_VERSION found"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed"
        exit 1
    fi
    print_success "pip3 found"
    
    # Check if running from correct directory
    if [ ! -f "$PROJECT_DIR/unified_mcp_server.py" ]; then
        print_error "Please run this script from the OpenEvolve project directory"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        print_success "Created virtual environment"
    else
        print_warning "Virtual environment already exists"
    fi
    
    source "$VENV_DIR/bin/activate"
    print_success "Activated virtual environment"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    pip install --upgrade pip
    
    if [ -f "$PROJECT_DIR/requirements_integration.txt" ]; then
        pip install -r "$PROJECT_DIR/requirements_integration.txt"
    fi
    
    print_success "Dependencies installed"
}

# Create required directories
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p "$LOGS_DIR"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/plugins"
    mkdir -p "$PROJECT_DIR/backups"
    mkdir -p "$PROJECT_DIR/knowledge_extraction"
    
    print_success "Directories created"
}

# Setup configuration
setup_config() {
    print_status "Setting up configuration..."
    
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        if [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            print_warning "Created .env from example. Please review and update!"
        fi
    fi
    
    print_success "Configuration ready"
}

# Run verification
run_verification() {
    print_status "Running verification..."
    
    if python "$PROJECT_DIR/verify_integration.py"; then
        print_success "Verification passed"
    else
        print_error "Verification failed"
        exit 1
    fi
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Check if services are already running
    if pgrep -f "openevolve" > /dev/null; then
        print_warning "OpenEvolve services may already be running"
        read -p "Stop existing services and restart? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            pkill -f "openevolve" || true
            sleep 2
        else
            print_status "Skipping service start"
            return
        fi
    fi
    
    # Start services in background
    nohup python -m openevolve_cli services start --all > "$LOGS_DIR/openevolve.log" 2>&1 &
    
    print_status "Waiting for services to start..."
    sleep 5
    
    # Check if services started
    if pgrep -f "openevolve" > /dev/null; then
        print_success "Services started successfully"
        echo
        echo "Service URLs:"
        echo "  REST API:   http://localhost:8000"
        echo "  GraphQL:    http://localhost:8001/graphql"
        echo "  Gateway:    http://localhost:8080"
        echo "  Logs:       $LOGS_DIR/openevolve.log"
    else
        print_error "Failed to start services"
        print_error "Check logs: $LOGS_DIR/openevolve.log"
        exit 1
    fi
}

# Print usage
print_usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  setup       - Setup environment and install dependencies"
    echo "  start       - Start all services"
    echo "  stop        - Stop all services"
    echo "  restart     - Restart services"
    echo "  status      - Check service status"
    echo "  logs        - View service logs"
    echo "  verify      - Run verification"
    echo "  full        - Full deployment (setup + start)"
    echo "  help        - Show this help"
}

# Main function
main() {
    case "${1:-full}" in
        setup)
            check_prerequisites
            setup_venv
            install_dependencies
            setup_directories
            setup_config
            run_verification
            ;;
        start)
            setup_venv
            start_services
            ;;
        stop)
            print_status "Stopping services..."
            pkill -f "openevolve" || true
            print_success "Services stopped"
            ;;
        restart)
            $0 stop
            sleep 2
            $0 start
            ;;
        status)
            if pgrep -f "openevolve" > /dev/null; then
                print_success "Services are running"
                pgrep -la "openevolve"
            else
                print_warning "Services are not running"
            fi
            ;;
        logs)
            if [ -f "$LOGS_DIR/openevolve.log" ]; then
                tail -f "$LOGS_DIR/openevolve.log"
            else
                print_error "Log file not found"
            fi
            ;;
        verify)
            setup_venv
            run_verification
            ;;
        full)
            check_prerequisites
            setup_venv
            install_dependencies
            setup_directories
            setup_config
            run_verification
            start_services
            echo
            print_success "Deployment complete!"
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
