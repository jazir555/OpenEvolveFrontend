#!/bin/bash
# =============================================================================
# OpenEvolve Infrastructure Verification Script
# =============================================================================
# This script verifies that all infrastructure services are running and
# accessible. Run this after starting services to confirm everything works.
#
# Usage:
#   ./scripts/verify-infrastructure.sh
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================================"
    echo " $1"
    echo "============================================================================"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Main verification
main() {
    print_header "OpenEvolve Infrastructure Verification"

    # 1. Check Docker
    log_info "Checking Docker installation..."
    if command_exists docker; then
        log_success "Docker is installed"
        docker --version
    else
        log_error "Docker is not installed"
    fi

    echo ""

    # 2. Check Docker Compose
    log_info "Checking Docker Compose..."
    if command_exists docker-compose || docker compose version &> /dev/null; then
        log_success "Docker Compose is installed"
        docker compose version 2>/dev/null || docker-compose --version
    else
        log_error "Docker Compose is not installed"
    fi

    echo ""
    print_header "Checking Service Containers"

    # 3. Check PostgreSQL container
    log_info "Checking PostgreSQL container..."
    if docker ps | grep -q openevolve-postgres; then
        log_success "PostgreSQL container is running"

        # Test connection
        if docker exec openevolve-postgres pg_isready -U openevolve -d openevolve &> /dev/null; then
            log_success "PostgreSQL is accepting connections"

            # Test query
            if docker exec openevolve-postgres psql -U openevolve -d openevolve -c "SELECT 1;" &> /dev/null; then
                log_success "PostgreSQL queries working"
            else
                log_error "PostgreSQL queries failing"
            fi
        else
            log_error "PostgreSQL is not accepting connections"
        fi
    else
        log_error "PostgreSQL container is not running"
    fi

    echo ""

    # 4. Check Qdrant container
    log_info "Checking Qdrant container..."
    if docker ps | grep -q openevolve-qdrant; then
        log_success "Qdrant container is running"

        # Test HTTP endpoint
        if command_exists curl; then
            if curl -sf http://localhost:6333/health &> /dev/null; then
                log_success "Qdrant HTTP API is accessible"

                # Check collections endpoint
                if curl -sf http://localhost:6333/collections &> /dev/null; then
                    log_success "Qdrant collections API is accessible"
                else
                    log_error "Qdrant collections API not accessible"
                fi
            else
                log_error "Qdrant HTTP API is not accessible"
            fi
        else
            log_warning "curl not installed, skipping HTTP checks"
        fi
    else
        log_error "Qdrant container is not running"
    fi

    echo ""

    # 5. Check Redis container
    log_info "Checking Redis container..."
    if docker ps | grep -q openevolve-redis; then
        log_success "Redis container is running"

        # Test connection
        if docker exec openevolve-redis redis-cli ping | grep -q PONG; then
            log_success "Redis is responding to PING"

            # Test info command
            if docker exec openevolve-redis redis-cli INFO server &> /dev/null; then
                log_success "Redis INFO command working"
            else
                log_error "Redis INFO command failing"
            fi
        else
            log_error "Redis is not responding"
        fi
    else
        log_error "Redis container is not running"
    fi

    echo ""
    print_header "Port Accessibility Checks"

    # 6. Check port accessibility
    if command_exists nc || command_exists netcat; then
        log_info "Checking if ports are accessible..."

        # PostgreSQL
        if nc -z localhost 5432 2>/dev/null || netcat -z localhost 5432 2>/dev/null; then
            log_success "Port 5432 (PostgreSQL) is accessible"
        else
            log_error "Port 5432 (PostgreSQL) is not accessible"
        fi

        # Qdrant
        if nc -z localhost 6333 2>/dev/null || netcat -z localhost 6333 2>/dev/null; then
            log_success "Port 6333 (Qdrant) is accessible"
        else
            log_error "Port 6333 (Qdrant) is not accessible"
        fi

        # Redis
        if nc -z localhost 6379 2>/dev/null || netcat -z localhost 6379 2>/dev/null; then
            log_success "Port 6379 (Redis) is accessible"
        else
            log_error "Port 6379 (Redis) is not accessible"
        fi
    else
        log_warning "nc/netcat not installed, skipping port checks"
    fi

    echo ""
    print_header "Summary"

    echo "Total checks: $TOTAL_CHECKS"
    echo -e "${GREEN}Passed:${NC} $PASSED_CHECKS"
    echo -e "${RED}Failed:${NC} $FAILED_CHECKS"
    echo ""

    if [ $FAILED_CHECKS -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! Infrastructure is ready.${NC}"
        echo ""
        echo "Next steps:"
        echo "  1. Configure your application to use the services"
        echo "  2. See docs/INFRASTRUCTURE_SETUP.md for connection details"
        echo ""
        return 0
    else
        echo -e "${RED}✗ Some checks failed. Please review the errors above.${NC}"
        echo ""
        echo "Troubleshooting:"
        echo "  1. Check container logs: docker logs <container-name>"
        echo "  2. Restart services: ./scripts/dev-stop.sh && ./scripts/dev-start.sh"
        echo "  3. See docs/INFRASTRUCTURE_SETUP.md for troubleshooting guide"
        echo ""
        return 1
    fi
}

# Run main function
main "$@"
