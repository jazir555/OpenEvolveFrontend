#!/bin/bash

################################################################################
# Gauntlet Performance Benchmark Runner
#
# This script runs comprehensive performance benchmarks for the gauntlet system
# and outputs results in JSON format for CI/CD integration.
#
# Features:
# - Runs all gauntlet benchmarks with configurable parameters
# - Outputs results to JSON with comparison against baseline metrics
# - Generates human-readable summary report
# - Calculates statistical significance
# - Returns appropriate exit codes for CI/CD
#
# Usage:
#   ./run_benchmarks.sh [options]
#
# Options:
#   -o, --output FILE     Output JSON file (default: benchmark_results.json)
#   -n, --runs NUM        Number of runs per benchmark (default: 10)
#   -v, --verbose         Enable verbose output
#   -h, --help            Show this help message
#
# Author: OpenEvolve Gauntlet System
# Date: 2026-02-03
################################################################################

set -e  # Exit on error

# Default values
OUTPUT_FILE="benchmark_results.json"
NUM_RUNS=10
VERBOSE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/gauntlet_benchmarks.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

################################################################################
# Parse Command Line Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -n|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -o, --output FILE     Output JSON file (default: benchmark_results.json)"
            echo "  -n, --runs NUM        Number of runs per benchmark (default: 10)"
            echo "  -v, --verbose         Enable verbose output"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 -o results.json -n 20              # Custom output and runs"
            echo "  $0 --verbose                         # Verbose output"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Pre-flight Checks
################################################################################

print_header "GAUNTLET BENCHMARK SUITE"

# Check if Python is available
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

print_success "Python found: $(python --version)"

# Check if benchmark script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Benchmark script not found: $PYTHON_SCRIPT"
    exit 1
fi

print_success "Benchmark script found"

# Check required Python packages
print_info "Checking required packages..."

REQUIRED_PACKAGES=("numpy" "scipy")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! python -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    print_error "Missing required packages: ${MISSING_PACKAGES[*]}"
    print_info "Install with: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

print_success "All required packages installed"

# Check if scipy is available (for statistical tests)
if python -c "import scipy" 2>/dev/null; then
    print_success "Statistical testing available (scipy)"
else
    print_warning "scipy not available - statistical significance tests will be limited"
    print_info "Install with: pip install scipy"
fi

echo ""

################################################################################
# Run Benchmarks
################################################################################

print_header "RUNNING BENCHMARKS"

print_info "Configuration:"
echo "  Output file: $OUTPUT_FILE"
echo "  Number of runs: $NUM_RUNS"
echo "  Verbose mode: $([ -n "$VERBOSE" ] && echo "enabled" || echo "disabled")"
echo ""

# Create timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
START_TIME=$(date +%s)

# Run the benchmark suite
print_info "Starting benchmark execution..."

if python "$PYTHON_SCRIPT" --output "$OUTPUT_FILE" --runs "$NUM_RUNS" $VERBOSE; then
    BENCHMARK_EXIT_CODE=0
    BENCHMARK_STATUS="SUCCESS"
else
    BENCHMARK_EXIT_CODE=$?
    BENCHMARK_STATUS="FAILED"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""

################################################################################
# Process Results
################################################################################

print_header "BENCHMARK RESULTS"

if [ -f "$OUTPUT_FILE" ]; then
    print_success "Results saved to: $OUTPUT_FILE"

    # Parse and display summary from JSON
    print_info "Summary:"

    if command -v jq &> /dev/null; then
        # Use jq for pretty JSON parsing
        echo ""
        jq -r '
            "Total Tests: \(.total_tests)",
            "Passed: \(.passed)",
            "Failed: \(.failed)",
            "Warnings: \(.warnings)",
            "Pass Rate: \(.summary.pass_rate)",
            "Grade: \(.summary.performance_grade)",
            "Duration: \(.duration_seconds)s"
        ' "$OUTPUT_FILE"
        echo ""

        # Show failed tests if any
        FAILED_COUNT=$(jq '.failed' "$OUTPUT_FILE")
        if [ "$FAILED_COUNT" -gt 0 ]; then
            print_warning "Failed benchmarks:"
            jq -r '.results[] | select(.status == "FAIL") | "  - \(.name): \(.value) \(.unit) (baseline: \(.baseline) \(.unit))"' "$OUTPUT_FILE"
            echo ""
        fi

        # Show warnings if any
        WARNING_COUNT=$(jq '.warnings' "$OUTPUT_FILE")
        if [ "$WARNING_COUNT" -gt 0 ]; then
            print_warning "Warnings:"
            jq -r '.results[] | select(.status == "WARNING") | "  - \(.name): \(.value) \(.unit) (baseline: \(.baseline) \(.unit))"' "$OUTPUT_FILE"
            echo ""
        fi

    else
        # Fallback to grep/awk if jq not available
        echo "  Install jq for better JSON formatting: https://stedolan.github.io/jq/"
        echo ""
        grep -E '"total_tests"|"passed"|"failed"|"warnings"|"pass_rate"|"performance_grade"' "$OUTPUT_FILE" | sed 's/,//' | sed 's/"//g' | sed 's/^/  /'
        echo ""
    fi

    # Calculate file size
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    print_info "Output file size: $FILE_SIZE"

else
    print_error "Results file not created: $OUTPUT_FILE"
    exit 1
fi

################################################################################
# Performance Summary
################################################################################

print_header "PERFORMANCE SUMMARY"

if command -v jq &> /dev/null; then
    # Overall status
    OVERALL_STATUS=$(jq -r '.summary.overall_status' "$OUTPUT_FILE")

    if [ "$OVERALL_STATUS" = "PASS" ]; then
        print_success "Overall Status: PASS"
    else
        print_error "Overall Status: FAIL"
    fi

    # Performance grade
    GRADE=$(jq -r '.summary.performance_grade' "$OUTPUT_FILE")
    print_info "Performance Grade: $GRADE"

    # Component breakdown
    echo ""
    print_info "Component Breakdown:"

    # ML Optimizer
    ML_OPT_RESULTS=$(jq '[.results[] | select(.component == "ml_optimizer")] | length' "$OUTPUT_FILE")
    ML_OPT_PASSED=$(jq '[.results[] | select(.component == "ml_optimizer" and .status == "PASS")] | length' "$OUTPUT_FILE")
    echo "  ML Optimizer: $ML_OPT_PASSED/$ML_OPT_RESULTS passed"

    # Predictive Executor
    PRED_EXEC_RESULTS=$(jq '[.results[] | select(.component == "predictive_executor")] | length' "$OUTPUT_FILE")
    PRED_EXEC_PASSED=$(jq '[.results[] | select(.component == "predictive_executor" and .status == "PASS")] | length' "$OUTPUT_FILE")
    echo "  Predictive Executor: $PRED_EXEC_PASSED/$PRED_EXEC_RESULTS passed"

    # Adaptive Learner
    ADAPT_LEARN_RESULTS=$(jq '[.results[] | select(.component == "adaptive_learner")] | length' "$OUTPUT_FILE")
    ADAPT_LEARN_PASSED=$(jq '[.results[] | select(.component == "adaptive_learner" and .status == "PASS")] | length' "$OUTPUT_FILE")
    echo "  Adaptive Learner: $ADAPT_LEARN_PASSED/$ADAPT_LEARN_RESULTS passed"

    # Intelligent Orchestrator
    ORCH_RESULTS=$(jq '[.results[] | select(.component == "intelligent_orchestrator")] | length' "$OUTPUT_FILE")
    ORCH_PASSED=$(jq '[.results[] | select(.component == "intelligent_orchestrator" and .status == "PASS")] | length' "$OUTPUT_FILE")
    echo "  Intelligent Orchestrator: $ORCH_PASSED/$ORCH_RESULTS passed"

else
    print_warning "Install jq for detailed breakdown"
fi

echo ""

################################################################################
# Timing Information
################################################################################

print_info "Execution completed in ${DURATION}s"

################################################################################
# Final Status
################################################################################

print_header "FINAL STATUS"

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    print_success "All benchmarks completed successfully"
    echo ""
    print_info "Next steps:"
    echo "  1. Review results in: $OUTPUT_FILE"
    echo "  2. Compare against baseline metrics"
    echo "  3. Investigate any failures or warnings"
    echo "  4. Consider updating baselines if improvements made"
    exit 0
else
    print_error "Some benchmarks failed"
    echo ""
    print_info "Next steps:"
    echo "  1. Review failed tests above"
    echo "  2. Check logs for detailed error messages"
    echo "  3. Run with --verbose for more information"
    echo "  4. Fix issues and re-run benchmarks"
    exit $BENCHMARK_EXIT_CODE
fi
