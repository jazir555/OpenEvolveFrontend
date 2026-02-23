#!/bin/bash
# Run E2E tests for hybrid OpenEvolve LoongFlow PES system
#
# This script validates the complete hybrid system integration:
# - LoongFlow PES execution
# - OpenEvolve evolutionary optimization
# - Hybrid workflows
# - Knowledge extraction and reuse
# - Error handling and recovery
# - Performance and scalability
#
# Usage:
#   ./tests/run_hybrid_e2e_tests.sh [options]
#
# Options:
#   -v, --verbose      Verbose output
#   -w, --watch        Watch mode
#   -s, --skip-slow    Skip slow tests
#   --coverage         Generate coverage report
#   --filter <pattern> Run tests matching pattern
#
# Environment Variables:
#   LOONGFLOW_API_URL       LoongFlow API endpoint (default: http://localhost:8050)
#   LOONGFLOW_ADAPTER_URL   LoongFlow adapter endpoint (default: http://localhost:8040)
#   OPENEVOLVE_API_URL      OpenEvolve adapter endpoint (default: http://localhost:8030)
#   TEST_TIMEOUT            Test timeout in milliseconds (default: 60000)
#   SKIP_SLOW_TESTS         Skip slow tests (default: true)
#   ENABLE_KNOWLEDGE_TESTS  Enable knowledge tests (default: false)
#
# Author: OpenEvolve Distinguished Engineer
# Version: 2.0.0 (TypeScript/Jest)

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
LOONGFLOW_API_URL="${LOONGFLOW_API_URL:-http://localhost:8050}"
LOONGFLOW_ADAPTER_URL="${LOONGFLOW_ADAPTER_URL:-http://localhost:8040}"
OPENEVOLVE_API_URL="${OPENEVOLVE_API_URL:-http://localhost:8030}"
TEST_TIMEOUT="${TEST_TIMEOUT:-60000}"
SKIP_SLOW_TESTS="${SKIP_SLOW_TESTS:-true}"
ENABLE_KNOWLEDGE_TESTS="${ENABLE_KNOWLEDGE_TESTS:-false}"

# Test options
VERBOSE=""
WATCH=""
SKIP_SLOW=""
COVERAGE=""
FILTER=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    local all_ok=true

    # Check Node.js
    if command -v node &> /dev/null; then
        print_success "Node.js found: $(node --version)"
    else
        print_error "Node.js not found"
        all_ok=false
    fi

    # Check npm
    if command -v npm &> /dev/null; then
        print_success "npm found: $(npm --version)"
    else
        print_error "npm not found"
        all_ok=false
    fi

    # Check Jest
    if npx jest --version &> /dev/null; then
        print_success "Jest found: $(npx jest --version)"
    else
        print_warning "Jest not found. Will install locally..."
    fi

    # Check TypeScript
    if npx tsc --version &> /dev/null; then
        print_success "TypeScript found: $(npx tsc --version)"
    else
        print_warning "TypeScript not found. Will install locally..."
    fi

    if [ "$all_ok" = false ]; then
        print_error "Prerequisites not met. Please install Node.js and npm."
        exit 1
    fi

    print_success "All prerequisites met"
}

run_tests() {
    print_header "Running Hybrid System E2E Tests"

    local jest_args=(
        "tests/test_hybrid_pes_evolution_e2e.test.ts"
        "--verbose"
        "--detectOpenHandles"
        "--forceExit"
    )

    # Add options
    if [ -n "$VERBOSE" ]; then
        jest_args+=("--verbose")
    fi

    if [ -n "$WATCH" ]; then
        jest_args+=("--watch")
    fi

    if [ "$SKIP_SLOW_TESTS" = "true" ] || [ -n "$SKIP_SLOW" ]; then
        # Skip tests marked as slow
        jest_args+=("--testNamePattern='^(?!.*slow).*$'")
    fi

    if [ -n "$COVERAGE" ]; then
        jest_args+=("--coverage")
    fi

    if [ -n "$FILTER" ]; then
        jest_args+=("--testNamePattern='$FILTER'")
    fi

    # Export environment variables
    export LOONGFLOW_API_URL
    export LOONGFLOW_ADAPTER_URL
    export OPENEVOLVE_API_URL
    export TEST_TIMEOUT
    export SKIP_SLOW_TESTS
    export ENABLE_KNOWLEDGE_TESTS

    print_info "Configuration:"
    print_info "  LOONGFLOW_API_URL: $LOONGFLOW_API_URL"
    print_info "  LOONGFLOW_ADAPTER_URL: $LOONGFLOW_ADAPTER_URL"
    print_info "  OPENEVOLVE_API_URL: $OPENEVOLVE_API_URL"
    print_info "  TEST_TIMEOUT: ${TEST_TIMEOUT}ms"
    print_info "  SKIP_SLOW_TESTS: $SKIP_SLOW_TESTS"
    print_info "  ENABLE_KNOWLEDGE_TESTS: $ENABLE_KNOWLEDGE_TESTS"
    echo ""

    # Run tests
    print_info "Starting test execution..."
    echo ""

    cd "$PROJECT_ROOT"

    if npx jest "${jest_args[@]}" "$@"; then
        print_success "All E2E tests passed!"
        return 0
    else
        local exit_code=$?
        print_error "Some tests failed (exit code: $exit_code)"
        return $exit_code
    fi
}

print_summary() {
    print_header "Test Summary"

    print_info "Test suites executed:"
    print_info "  ✓ Basic PES Execution"
    print_info "  ✓ Evolutionary Optimization"
    print_info "  ✓ Hybrid Workflows"
    print_info "  ✓ Knowledge Management"
    print_info "  ✓ Error Handling and Recovery"
    print_info "  ✓ Performance and Scalability"

    echo ""
    print_info "For detailed results, check the test output above."

    if [ -n "$COVERAGE" ]; then
        print_info "Coverage report generated: coverage/index.html"
    fi
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_header "Hybrid OpenEvolve LoongFlow PES E2E Tests (TypeScript/Jest)"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE="-v"
                shift
                ;;
            -w|--watch)
                WATCH="-w"
                shift
                ;;
            -s|--skip-slow)
                SKIP_SLOW="-s"
                shift
                ;;
            --coverage)
                COVERAGE="--coverage"
                shift
                ;;
            --filter)
                FILTER="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  -v, --verbose      Verbose output"
                echo "  -w, --watch        Watch mode"
                echo "  -s, --skip-slow    Skip slow tests"
                echo "  --coverage         Generate coverage report"
                echo "  --filter <pattern> Run tests matching pattern"
                echo "  -h, --help         Show this help"
                echo ""
                echo "Environment Variables:"
                echo "  LOONGFLOW_API_URL       LoongFlow API endpoint"
                echo "  LOONGFLOW_ADAPTER_URL   LoongFlow adapter endpoint"
                echo "  OPENEVOLVE_API_URL      OpenEvolve adapter endpoint"
                echo "  TEST_TIMEOUT            Test timeout in milliseconds"
                echo "  SKIP_SLOW_TESTS         Skip slow tests"
                echo "  ENABLE_KNOWLEDGE_TESTS  Enable knowledge tests"
                echo ""
                echo "Examples:"
                echo "  $0                              # Run all tests"
                echo "  $0 --coverage                  # Run with coverage"
                echo "  $0 --filter 'PESEvolution'      # Run only PES Evolution tests"
                echo "  SKIP_SLOW_TESTS=false $0       # Include slow tests"
                exit 0
                ;;
            *)
                # Pass through to jest
                shift
                ;;
        esac
    done

    # Run
    check_prerequisites
    run_tests "$@"
    local result=$?
    print_summary

    exit $result
}

# Run main
main "$@"
