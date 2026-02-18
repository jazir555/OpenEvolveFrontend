#!/bin/bash
###############################################################################
# PROBE: Complete Adapter API Verification
#
# This master probe runs all individual probes to verify the complete
# Adaptive MDAP/MAKER adapter integration.
#
# Usage: ./probes/check_api.sh
###############################################################################

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "PROBE: Complete Adapter Verification"
echo "=========================================="
echo ""

# Track results
TOTAL_PROBES=0
PASSED_PROBES=0
FAILED_PROBES=0

# Run each probe
probes=(
    "check_adaptive_mdap_api.sh"
    "check_maker_api.sh"
    "check_integration.sh"
)

for probe in "${probes[@]}"; do
    ((TOTAL_PROBES++))
    echo ""
    echo "──────────────────────────────────────────"
    echo "Running: $probe"
    echo "──────────────────────────────────────────"

    if bash "$SCRIPT_DIR/$probe"; then
        ((PASSED_PROBES++))
        log_info "✓ PASSED: $probe"
    else
        ((FAILED_PROBES++))
        log_error "✗ FAILED: $probe"
    fi
done

# Summary
echo ""
echo "=========================================="
echo "PROBE SUMMARY"
echo "=========================================="
echo "Total Probes: $TOTAL_PROBES"
echo "Passed: $PASSED_PROBES"
echo "Failed: $FAILED_PROBES"
echo ""

if [ $FAILED_PROBES -eq 0 ]; then
    log_info "✓ ALL PROBES PASSED - Adapter is fully functional"
    echo ""
    echo "Probe Result: SUCCESS (200 OK)"
    exit 0
else
    log_error "✗ SOME PROBES FAILED - Adapter has issues"
    echo ""
    echo "Probe Result: FAILURE (503 Service Unavailable)"
    exit 1
fi
