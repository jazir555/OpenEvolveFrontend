#!/bin/bash
# ROMA API Health Check Probe
# Verifies that the ROMA API server is responsive and healthy
# Follows "Law of Runtime Truth" - validates actual API behavior

set -euo pipefail

# Configuration
ROMA_SERVER_URL="${ROMA_SERVER_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-10}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-2}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Health check function
check_health() {
    local attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        log_info "Health check attempt $attempt/$MAX_RETRIES..."

        # Check health endpoint
        response=$(curl -s -w "\n%{http_code}" \
            --max-time $TIMEOUT \
            "${ROMA_SERVER_URL}/health" 2>&1) || true

        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | head -n-1)

        if [ "$http_code" = "000" ]; then
            log_error "Connection refused - ROMA server may not be running"
        elif [ "$http_code" = "200" ]; then
            log_info "Health endpoint returned 200 OK"

            # Validate response body contains status
            if echo "$body" | grep -q "status"; then
                log_info "Health check response valid"
                echo "{\"status\":\"healthy\",\"server\":\"${ROMA_SERVER_URL}\",\"response\":${body}}"
                return 0
            else
                log_warn "Health check response missing 'status' field"
            fi
        else
            log_error "Health endpoint returned HTTP $http_code"
        fi

        if [ $attempt -lt $MAX_RETRIES ]; then
            log_info "Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi

        attempt=$((attempt + 1))
    done

    log_error "Health check failed after $MAX_RETRIES attempts"
    echo "{\"status\":\"unhealthy\",\"server\":\"${ROMA_SERVER_URL}\",\"error\":\"Health check failed\"}"
    return 1
}

# Check API availability
check_api_available() {
    log_info "Checking API availability..."

    response=$(curl -s -w "\n%{http_code}" \
        --max-time $TIMEOUT \
        "${ROMA_SERVER_URL}/api/v1/profiles" 2>&1) || true

    http_code=$(echo "$response" | tail -n1)

    if [ "$http_code" = "200" ]; then
        log_info "API endpoint accessible"
        return 0
    else
        log_error "API endpoint returned HTTP $http_code"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting ROMA API health check..."
    log_info "Target server: ${ROMA_SERVER_URL}"

    # Run health check
    if check_health; then
        # Additional API availability check
        if check_api_available; then
            log_info "ROMA API is healthy and ready"
            exit 0
        else
            log_warn "Health check passed but API availability check failed"
            exit 1
        fi
    else
        log_error "ROMA API health check failed"
        exit 1
    fi
}

# Run main
main "$@"
