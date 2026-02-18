#!/bin/bash
###############################################################################
# Master Probe: V2.0 Features Verification
#
# This script runs all v2.0 feature probes to verify advanced functionality.
# Part of Law 2: Runtime Truth - verify actual behavior, not documentation.
#
# Usage: ./probes/check_v2_features.sh
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
TOTAL_PASSED=0
TOTAL_FAILED=0
PROBE_RESULTS=()

echo "========================================================================"
echo "  MASTER PROBE: V2.0 Features Verification"
echo "========================================================================"
echo ""
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-test}"

echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo -e "${BLUE}This will run all v2.0 feature probes:${NC}"
echo "  1. Async Features"
echo "  2. Cache Features"
echo "  3. Advanced OpenEvolve"
echo "  4. Additional Systems"
echo "  5. UI Features"
echo ""
echo "========================================================================"
echo ""

# Function to run a probe
run_probe() {
    local probe_name=$1
    local probe_script=$2

    echo -e "${BLUE}Running: ${probe_name}${NC}"
    echo "========================================================================"

    if bash "$probe_script"; then
        PROBE_RESULTS+=("${GREEN}[PASS]${NC} ${probe_name}")
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
    else
        PROBE_RESULTS+=("${RED}[FAIL]${NC} ${probe_name}")
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi

    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to probes directory for script execution
cd "$SCRIPT_DIR"

###############################################################################
# Run all v2.0 probes
###############################################################################

run_probe "Async Features" "./check_async_features.sh"
run_probe "Cache Features" "./check_cache_features.sh"
run_probe "Advanced OpenEvolve" "./check_advanced_openevolve.sh"
run_probe "Additional Systems" "./check_additional_systems.sh"
run_probe "UI Features" "./check_ui_features.sh"

###############################################################################
# Summary
###############################################################################
echo "========================================================================"
echo "  MASTER PROBE SUMMARY"
echo "========================================================================"
echo ""
echo "Total Probes: $TOTAL_TESTS"
echo "Passed: $TOTAL_PASSED"
echo "Failed: $TOTAL_FAILED"
echo ""

echo "Individual Results:"
echo "------------------------------------------------------------------------"
for result in "${PROBE_RESULTS[@]}"; do
    echo -e "  $result"
done

echo ""
echo "========================================================================"

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}SUCCESS: All v2.0 feature probes passed${NC}"
    echo ""
    echo "All v2.0 features are verified and operational:"
    echo "  [OK] Async/await concurrent processing"
    echo "  [OK] Response caching with LRU eviction"
    echo "  [OK] Advanced OpenEvolve integration"
    echo "  [OK] Additional systems (CrewAI, MCP, RAGBits, LeanAide, Z3)"
    echo "  [OK] UI dashboard generation"
    echo ""
    exit 0
else
    echo -e "${RED}FAILURE: $TOTAL_FAILED probe(s) failed${NC}"
    echo ""
    echo "Some v2.0 features are not working correctly."
    echo "Review the failed probe output above for details."
    echo ""
    exit 1
fi
