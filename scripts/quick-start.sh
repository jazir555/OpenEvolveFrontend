#!/bin/bash
# =============================================================================
# OpenEvolve Quick Start Script
# License: Apache 2.0
#
# Description: Automated setup and deployment script for OpenEvolve Frontend
# Usage: ./quick-start.sh [--dry-run] [--skip-tests] [--env-file FILE]
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/quick-start-$(date +%Y%m%d-%H%M%S).log"
DRY_RUN=false
SKIP_TESTS=false
ENV_FILE="${PROJECT_ROOT}/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Service health check endpoints (adjust as needed)
declare -A HEALTH_ENDPOINTS=(
    ["openevolve-app"]="http://localhost:8080/health"
    ["openevolve-valkey"]="http://localhost:6379"
    ["openevolve-prometheus"]="http://localhost:9090/-/healthy"
    ["openevolve-grafana"]="http://localhost:3000/api/health"
)

# =============================================================================
# Utility Functions
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "${LOG_FILE}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "${LOG_FILE}"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "${LOG_FILE}"
}

print_step() {
    local step="$1"
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  STEP ${step}${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OpenEvolve Quick Start Script - Automated setup and deployment

OPTIONS:
    --dry-run           Show what would be done without executing
    --skip-tests        Skip running tests
    --env-file FILE     Use specific environment file (default: .env)
    -h, --help          Show this help message

EXAMPLES:
    $0                              # Standard quick start
    $0 --dry-run                    # Preview actions without executing
    $0 --skip-tests --env-file .env.test  # Use custom env file, skip tests

EOF
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --env-file)
                ENV_FILE="$2"
                shift 2
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
}

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    print_step "1: Checking Prerequisites"

    local missing_deps=()

    # Check Node.js
    if command -v node &> /dev/null; then
        local node_version=$(node -v)
        log_success "Node.js found: ${node_version}"
    else
        log_error "Node.js not found"
        missing_deps+=("node")
    fi

    # Check npm
    if command -v npm &> /dev/null; then
        local npm_version=$(npm -v)
        log_success "npm found: ${npm_version}"
    else
        log_error "npm not found"
        missing_deps+=("npm")
    fi

    # Check Docker
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | cut -d' ' -f3 | tr -d ',')
        log_success "Docker found: ${docker_version}"
    else
        log_error "Docker not found"
        missing_deps+=("docker")
    fi

    # Check Docker Compose
    if docker compose version &> /dev/null; then
        local compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
        log_success "Docker Compose found: ${compose_version}"
    else
        log_error "Docker Compose not found"
        missing_deps+=("docker-compose")
    fi

    # Check Python (optional but recommended)
    if command -v python3 &> /dev/null || command -v python &> /dev/null; then
        local python_cmd=$(command -v python3 || command -v python)
        local python_version=$($python_cmd --version 2>&1)
        log_success "Python found: ${python_version}"
    else
        log_warning "Python not found (optional but recommended)"
    fi

    # Check git
    if command -v git &> /dev/null; then
        local git_version=$(git --version | cut -d' ' -f3)
        log_success "Git found: ${git_version}"
    else
        log_warning "Git not found (optional)"
    fi

    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and try again"
        return 1
    fi

    log_success "All required prerequisites are installed"
    return 0
}

# =============================================================================
# Environment Validation
# =============================================================================

validate_environment() {
    print_step "2: Validating Environment"

    # Check if .env file exists
    if [ ! -f "${ENV_FILE}" ]; then
        log_warning "Environment file not found: ${ENV_FILE}"

        if [ -f "${PROJECT_ROOT}/.env.example" ]; then
            log_info "Creating .env from .env.example..."
            if [ "${DRY_RUN}" = false ]; then
                cp "${PROJECT_ROOT}/.env.example" "${ENV_FILE}"
                log_warning "Please edit ${ENV_FILE} and set appropriate values before continuing"
                log_warning "Especially set SECRET_KEY to a secure random value!"
            fi
        else
            log_error ".env.example not found. Cannot create environment file"
            return 1
        fi
    fi

    # Load and validate critical environment variables
    log_info "Validating environment variables..."

    # Source the .env file safely
    if [ -f "${ENV_FILE}" ]; then
        set -a  # Automatically export all variables
        source "${ENV_FILE}"
        set +a
    fi

    local missing_vars=()

    # Check critical variables
    local required_vars=(
        "SECRET_KEY"
    )

    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done

    # Check if SECRET_KEY is still the default
    if [ "${SECRET_KEY:-}" = "changeme-in-production-generate-random-string" ] || \
       [ "${SECRET_KEY:-}" = "changeme-in-production" ]; then
        log_warning "SECRET_KEY is set to default value. This is unsafe for production!"
        read -p "Generate a new SECRET_KEY? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [ "${DRY_RUN}" = false ]; then
                local new_key=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo "fallback-key-$(date +%s)")
                sed -i.bak "s/^SECRET_KEY=.*/SECRET_KEY=${new_key}/" "${ENV_FILE}"
                rm -f "${ENV_FILE}.bak"
                log_success "Generated new SECRET_KEY"
            fi
        fi
    fi

    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_warning "Missing optional environment variables: ${missing_vars[*]}"
    fi

    log_success "Environment validation complete"
    return 0
}

# =============================================================================
# Install Dependencies
# =============================================================================

install_dependencies() {
    print_step "3: Installing Dependencies"

    cd "${PROJECT_ROOT}"

    if [ "${DRY_RUN}" = true ]; then
        log_info "[DRY-RUN] Would install npm dependencies"
        return 0
    fi

    log_info "Installing npm dependencies..."
    if npm install --legacy-peer-deps; then
        log_success "Dependencies installed successfully"
    else
        log_error "Failed to install dependencies"
        return 1
    fi
}

# =============================================================================
# Run Tests
# =============================================================================

run_tests() {
    if [ "${SKIP_TESTS}" = true ]; then
        log_warning "Skipping tests as requested"
        return 0
    fi

    print_step "4: Running Tests"

    cd "${PROJECT_ROOT}"

    if [ "${DRY_RUN}" = true ]; then
        log_info "[DRY-RUN] Would run test suite"
        return 0
    fi

    log_info "Running TypeScript type checking..."
    if npm run typecheck 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Type checking passed"
    else
        log_warning "Type checking failed (continuing anyway)"
    fi

    log_info "Running lint checks..."
    if npm run lint 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Lint checks passed"
    else
        log_warning "Lint checks failed (continuing anyway)"
    fi

    log_info "Running unit tests..."
    if npm run test 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Unit tests passed"
    else
        log_warning "Unit tests failed (continuing anyway)"
    fi

    return 0
}

# =============================================================================
# Build Everything
# =============================================================================

build_everything() {
    print_step "5: Building Everything"

    cd "${PROJECT_ROOT}"

    if [ "${DRY_RUN}" = true ]; then
        log_info "[DRY-RUN] Would build TypeScript code and Docker images"
        return 0
    fi

    log_info "Building TypeScript code..."
    if npm run build 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "TypeScript build successful"
    else
        log_warning "TypeScript build failed or no build script found"
    fi

    log_info "Building Docker images..."
    if docker compose build 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Docker images built successfully"
    else
        log_error "Failed to build Docker images"
        return 1
    fi

    return 0
}

# =============================================================================
# Start Services
# =============================================================================

start_services() {
    print_step "6: Starting Services"

    cd "${PROJECT_ROOT}"

    if [ "${DRY_RUN}" = true ]; then
        log_info "[DRY-RUN] Would start Docker Compose services"
        return 0
    fi

    log_info "Starting Docker Compose services..."
    if docker compose up -d 2>&1 | tee -a "${LOG_FILE}"; then
        log_success "Services started successfully"
    else
        log_error "Failed to start services"
        return 1
    fi

    return 0
}

# =============================================================================
# Verify Health
# =============================================================================

verify_health() {
    print_step "7: Verifying Service Health"

    if [ "${DRY_RUN}" = true ]; then
        log_info "[DRY-RUN] Would verify service health"
        return 0
    fi

    log_info "Waiting for services to become healthy..."
    sleep 10

    local unhealthy_services=()

    for service in "${!HEALTH_ENDPOINTS[@]}"; do
        local endpoint="${HEALTH_ENDPOINTS[$service]}"
        log_info "Checking ${service} at ${endpoint}..."

        # Try up to 5 times with 5 second intervals
        local attempt=0
        local max_attempts=5
        local healthy=false

        while [ $attempt -lt $max_attempts ]; do
            if curl -f -s -o /dev/null --max-time 5 "${endpoint}" 2>&1; then
                log_success "${service} is healthy"
                healthy=true
                break
            else
                attempt=$((attempt + 1))
                if [ $attempt -lt $max_attempts ]; then
                    sleep 5
                fi
            fi
        done

        if [ "${healthy}" = false ]; then
            log_error "${service} is not responding"
            unhealthy_services+=("$service")
        fi
    done

    if [ ${#unhealthy_services[@]} -gt 0 ]; then
        log_warning "Some services are not healthy: ${unhealthy_services[*]}"
        log_info "Check logs with: docker compose logs -f"
        return 1
    fi

    log_success "All services are healthy"
    return 0
}

# =============================================================================
# Show Next Steps
# =============================================================================

show_next_steps() {
    print_step "8: Next Steps"

    cat << EOF
${GREEN}✓ OpenEvolve Frontend is now running!${NC}

${BLUE}Service URLs:${NC}
  • OpenEvolve API:        http://localhost:8000
  • GraphQL API:           http://localhost:8001
  • Orchestrator/Gateway:  http://localhost:8080
  • BubbleLab Dashboard:   http://localhost:8501
  • Jaeger Tracing:        http://localhost:16686
  • Prometheus Metrics:    http://localhost:9090
  • Grafana Dashboard:     http://localhost:3000 (admin/admin)

${BLUE}Useful Commands:${NC}
  • View logs:             docker compose logs -f
  • Stop services:         docker compose down
  • Restart services:      docker compose restart
  • Check health:          ./scripts/health-check.sh
  • Run smoke tests:       ./scripts/smoke-test.sh
  • Validate deployment:   ./scripts/validate.sh

${BLUE}Documentation:${NC}
  • Project README:        ${PROJECT_ROOT}/README.md
  • Scripts README:        ${PROJECT_ROOT}/scripts/README.md
  • Architecture:          ${PROJECT_ROOT}/docs/ARCHITECTURE.md

${YELLOW}Important:${NC}
  • Check logs for any warnings or errors
  • Update SECRET_KEY in .env for production use
  • Review and configure environment variables
  • Set up authentication for external access

${BLUE}Quick Actions:${NC}
  • Run smoke tests:       ./scripts/smoke-test.sh
  • View all containers:   docker ps
  • Access shell:          docker compose exec openevolve-app bash

EOF

    log_info "Full log saved to: ${LOG_FILE}"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     OpenEvolve Frontend - Quick Start Script                 ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Create logs directory
    mkdir -p "$(dirname "${LOG_FILE}")"

    # Parse arguments
    parse_args "$@"

    # Execute steps
    check_prerequisites || exit 1
    validate_environment || exit 1
    install_dependencies || exit 1
    run_tests || exit 1
    build_everything || exit 1
    start_services || exit 1
    verify_health || exit 1
    show_next_steps

    echo ""
    log_success "Quick start completed successfully!"
    exit 0
}

# Run main function
main "$@"
