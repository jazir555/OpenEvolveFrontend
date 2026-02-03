#!/bin/bash
###############################################################################
# LeanAide Adapter - Contract Test Runner
#
# This script runs the contract tests before starting the adapter.
# Following the Federation Constitution's "Fail Fast" doctrine:
#   - If tests fail, container startup is aborted
#   - All contracts must be validated before accepting traffic
#
# Usage:
#   ./run-contract-tests.sh [optional additional npm test args]
#
# Environment Variables:
#   LEANAIDE_API_URL - LeanAide server URL (REQUIRED)
#   LEANAIDE_TIMEOUT_MS - Request timeout in milliseconds (optional)
#   FAIL_FAST - Exit on first test failure (default: true)
#
# Exit Codes:
#   0 - All tests passed
#   1 - Configuration error
#   2 - One or more tests failed
#   3 - LeanAide server not reachable
#
###############################################################################

set -e  # Exit on error (but we'll handle test failures gracefully)

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Configuration validation
validate_config() {
    log_info "Validating configuration..."

    # Check for LEANAIDE_API_URL (Law of Configuration Explicitness)
    if [ -z "$LEANAIDE_API_URL" ]; then
        log_error "LEANAIDE_API_URL is not configured."
        log_error "The adapter requires LEANAIDE_API_URL to be set (no magic defaults)."
        log_error "Example: export LEANAIDE_API_URL=http://localhost:7654"
        exit 1
    fi

    log_success "Configuration validated: LEANAIDE_API_URL=$LEANAIDE_API_URL"
}

# Check if LeanAide server is reachable
check_server() {
    log_info "Checking LeanAide server availability..."

    local max_attempts=30
    local attempt=1
    local wait_time=2

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s -o /dev/null "$LEANAIDE_API_URL" 2>/dev/null; then
            log_success "LeanAide server is reachable at $LEANAIDE_API_URL"
            return 0
        fi

        log_warning "Server not reachable (attempt $attempt/$max_attempts). Retrying in ${wait_time}s..."
        sleep $wait_time
        attempt=$((attempt + 1))
    done

    log_error "LeanAide server is not reachable at $LEANAIDE_API_URL after $max_attempts attempts"
    log_error "Please ensure the LeanAide server is running:"
    log_error "  - Check if LEANAIDE_API_URL is correct"
    log_error "  - Verify the LeanAide container is running"
    log_error "  - Check network connectivity"

    # If LEANAIDE_API_URL is localhost, show helpful commands
    if [[ "$LEANAIDE_API_URL" == *"localhost"* ]] || [[ "$LEANAIDE_API_URL" == *"127.0.0.1"* ]]; then
        log_error "To start LeanAide locally:"
        log_error "  cd /path/to/LeanAide"
        log_error "  lake exe leanaide_process"
    fi

    exit 3
}

# Run contract tests
run_tests() {
    log_info "Running LeanAide contract tests..."
    echo ""

    # Set default timeout if not provided
    export JEST_TIMEOUT_MS=${JEST_TIMEOUT_MS:-$LEANAIDE_TIMEOUT_MS}
    export JEST_TIMEOUT_MS=${JEST_TIMEOUT_MS:-30000}

    # Determine test command based on FAIL_FAST setting
    if [ "$FAIL_FAST" = "false" ]; then
        # Run all tests even if some fail
        log_info "Running tests with FAIL_FAST=false (will run all tests)"
        npm test -- "$@" --no-bail
    else
        # Default: run tests with jest (configurable via jest.config.js)
        log_info "Running tests with FAIL_FAST=true (will stop on first failure)"
        npm test -- "$@"
    fi

    local test_exit_code=$?

    echo ""

    if [ $test_exit_code -eq 0 ]; then
        log_success "All contract tests passed!"
        return 0
    else
        log_error "Contract tests failed with exit code $test_exit_code"
        return 2
    fi
}

# Main execution
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}LeanAide Contract Test Runner${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Print configuration
    log_info "Configuration:"
    echo "  LEANAIDE_API_URL: $LEANAIDE_API_URL"
    echo "  LEANAIDE_TIMEOUT_MS: ${LEANAIDE_TIMEOUT_MS:-30000}"
    echo "  JEST_TIMEOUT_MS: ${JEST_TIMEOUT_MS}"
    echo "  FAIL_FAST: ${FAIL_FAST:-true}"
    echo ""

    # Validate configuration
    validate_config

    # Check server availability
    check_server

    # Run tests
    if run_tests "$@"; then
        echo ""
        log_success "Contract validation complete. Adapter may now start."
        exit 0
    else
        echo ""
        log_error "Contract validation failed. Adapter will NOT start."
        log_error "Please fix the failing tests before starting the adapter."
        exit 2
    fi
}

# Run main function
main "$@"
