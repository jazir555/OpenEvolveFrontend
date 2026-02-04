#!/bin/bash

###############################################################################
# Gauntlet System Master Probe Runner
#
# Executes all gauntlet system probes to validate complete functionality.
#
# This script runs all component probes in sequence and provides a summary
# of the overall system health.
#
# Returns: 0 if all probes pass, non-zero if any fail
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test tracking
TOTAL_PROBES=0
PASSED_PROBES=0
FAILED_PROBES=0
FAILED_PROBE_NAMES=()

# Helper functions
log_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Print banner
log_header "Gauntlet System Probe Suite"
echo ""
log_info "Running comprehensive probe suite for Gauntlet system components"
log_info "Per CLAUDE.md Law 2: Runtime Truth"
echo ""

# Array of probe scripts
PROBES=(
    "check_ml_optimizer.sh"
    "check_predictive_executor.sh"
    "check_adaptive_learner.sh"
    "check_intelligent_orchestrator.sh"
    "check_websocket.sh"
)

# Run each probe
for probe in "${PROBES[@]}"; do
    TOTAL_PROBES=$((TOTAL_PROBES + 1))

    log_header "Running: $probe"

    if [ -f "$SCRIPT_DIR/$probe" ]; then
        if [ -x "$SCRIPT_DIR/$probe" ]; then
            # Run the probe
            if "$SCRIPT_DIR/$probe"; then
                PASSED_PROBES=$((PASSED_PROBES + 1))
                log_info "✓ $probe PASSED"
            else
                FAILED_PROBES=$((FAILED_PROBES + 1))
                FAILED_PROBE_NAMES+=("$probe")
                log_error "✗ $probe FAILED"
            fi
        else
            FAILED_PROBES=$((FAILED_PROBES + 1))
            FAILED_PROBE_NAMES+=("$probe (not executable)")
            log_error "✗ $probe is not executable"
            log_info "  Run: chmod +x $SCRIPT_DIR/$probe"
        fi
    else
        FAILED_PROBES=$((FAILED_PROBES + 1))
        FAILED_PROBE_NAMES+=("$probe (not found)")
        log_error "✗ $probe not found"
    fi

    echo ""
done

# Print summary
log_header "Probe Suite Summary"

echo ""
echo -e "Total probes:  ${TOTAL_PROBES}"
echo -e "${GREEN}Passed:        ${PASSED_PROBES}${NC}"

if [ $FAILED_PROBES -gt 0 ]; then
    echo -e "${RED}Failed:        ${FAILED_PROBES}${NC}"
    echo ""
    log_error "Failed probes:"
    for name in "${FAILED_PROBE_NAMES[@]}"; do
        echo -e "${RED}  ✗ ${name}${NC}"
    done
    echo ""
    log_error "System probe suite FAILED"
    log_info "Please fix the failed probes before proceeding"
    exit 1
else
    echo -e "${GREEN}Failed:        ${FAILED_PROBES}${NC}"
    echo ""
    log_info "✓ All gauntlet system probes PASSED!"
    echo ""
    log_info "System is ready for operation"
    log_info "All API contracts validated per CLAUDE.md Law 2"
    exit 0
fi
