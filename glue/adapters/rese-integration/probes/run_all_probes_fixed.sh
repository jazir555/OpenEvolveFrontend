#!/bin/bash
###############################################################################
# RESE All Probes Runner - Fixed Version
#
# This script runs all RESE probe scripts in sequence and provides a
# comprehensive report of their status.
#
# Following CLAUDE.md principles:
# - Law of Runtime Truth: We verify by execution
# - Law of Configuration Explicitness: All paths configurable via env vars
# - Structured Logging: JSON output for all probes
#
# Usage: ./run_all_probes_fixed.sh
#
# Exit codes:
# 0: All probes passed
# 1: One or more probes failed
# 2: Configuration error
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration (Law of Configuration Explicitness)
FRONTEND_ROOT="${FRONTEND_ROOT:-/c/Users/mmeadow/Documents/OpenEvolve/Frontend}"
ADAPTERS_ROOT="${ADAPTERS_ROOT:-${FRONTEND_ROOT}/glue/adapters}"
PYTHON_CMD="${PYTHON_CMD:-/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBES_ROOT="$SCRIPT_DIR"

# All probe definitions
declare -a PROBE_NAMES=(
    "Phase I: Epistemic Audit"
    "Phase II: Isomorphic Mapping"
    "Phase III: MCTS Search"
    "Phase IV: Architecture Assembly"
    "Full Pipeline Integration"
    "Symbolic Constraint Engine (SCE)"
    "Deep Exploration Engine (DEE)"
    "Logic-to-Loss Translation (LLTL)"
)

declare -a PROBE_PATHS=(
    "${ADAPTERS_ROOT}/rese-phase1/probes/check_phase1.sh"
    "${ADAPTERS_ROOT}/rese-phase2/probes/check_phase2.sh"
    "${ADAPTERS_ROOT}/rese-phase3/probes/check_phase3.sh"
    "${ADAPTERS_ROOT}/rese-phase4/probes/check_phase4.sh"
    "${PROBES_ROOT}/check_full_pipeline.sh"
    "${ADAPTERS_ROOT}/rese-sce/probes/check-sce.sh"
    "${ADAPTERS_ROOT}/rese-dee/probes/check_dee.sh"
    "${ADAPTERS_ROOT}/rese-lltl/probes/check_lltl.sh"
)

declare -a PROBE_KEYS=(
    "phase1"
    "phase2"
    "phase3"
    "phase4"
    "full_pipeline"
    "sce"
    "dee"
    "lltl"
)

# Results storage
declare -a PROBE_STATUS
declare -a PROBE_EXIT_CODES
declare -a PROBE_OUTPUTS

# Counters
TOTAL_PROBES=${#PROBE_NAMES[@]}
PASSED_PROBES=0
FAILED_PROBES=0
SKIPPED_PROBES=0

# Timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CORRELATION_ID=$($PYTHON_CMD -c "import uuid; print(str(uuid.uuid4()))" 2>/dev/null || echo "unknown")

###############################################################################
# Helper Functions
###############################################################################

log_header() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║ $1${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
}

log_section() {
    echo ""
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}─────────────────────────────────────────────────────────────${NC}"
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

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_failure() {
    echo -e "${RED}✗${NC} $1"
}

run_probe() {
    local probe_name="$1"
    local probe_path="$2"
    local probe_key="$3"
    local probe_index=$4

    log_section "Probe $((probe_index + 1))/$TOTAL_PROBES: $probe_name"

    # Check if probe exists
    if [ ! -f "$probe_path" ]; then
        log_error "Probe file not found: $probe_path"
        PROBE_STATUS[$probe_index]="not_found"
        PROBE_EXIT_CODES[$probe_index]=1
        PROBE_OUTPUTS[$probe_index]="Probe file not found"
        FAILED_PROBES=$((FAILED_PROBES + 1))
        return 1
    fi

    # Make executable
    chmod +x "$probe_path" 2>/dev/null || true

    # Run probe from correct directory
    local output_file="/tmp/rese_probe_${probe_key}_$$.json"
    local exit_code=0

    # Change to FRONTEND_ROOT for execution (many probes expect this)
    cd "$FRONTEND_ROOT"

    if bash "$probe_path" > "$output_file" 2>&1; then
        exit_code=0
        log_success "$probe_name PASSED"
        PROBE_STATUS[$probe_index]="pass"
        PASSED_PROBES=$((PASSED_PROBES + 1))
    else
        exit_code=$?
        log_failure "$probe_name FAILED (exit code: $exit_code)"
        PROBE_STATUS[$probe_index]="fail"
        FAILED_PROBES=$((FAILED_PROBES + 1))
    fi

    PROBE_EXIT_CODES[$probe_index]=$exit_code

    # Show output (last 30 lines)
    if [ -f "$output_file" ]; then
        echo ""
        echo "Output (last 30 lines):"
        echo "─────────────────────────────────────────────────────────────"
        tail -30 "$output_file"
        echo "─────────────────────────────────────────────────────────────"
        PROBE_OUTPUTS[$probe_index]=$(cat "$output_file")
        rm -f "$output_file"
    else
        PROBE_OUTPUTS[$probe_index]="No output captured"
    fi

    return $exit_code
}

###############################################################################
# Main Execution
###############################################################################

clear

log_header "RESE PROBE SUITE - COMPREHENSIVE VERIFICATION"
log_info "Timestamp: $TIMESTAMP"
log_info "Correlation ID: $CORRELATION_ID"
log_info "Frontend Root: $FRONTEND_ROOT"
log_info "Adapters Root: $ADAPTERS_ROOT"
log_info "Python: $PYTHON_CMD"
echo ""

# Verify Python is available
if [ ! -f "$PYTHON_CMD" ] && ! command -v python3 &> /dev/null; then
    log_error "Python not found at $PYTHON_CMD"
    log_error "Please install Python 3.9+ or set PYTHON_CMD environment variable"
    exit 2
fi

log_success "Python verified"

# Run all probes
for i in "${!PROBE_NAMES[@]}"; do
    run_probe "${PROBE_NAMES[$i]}" "${PROBE_PATHS[$i]}" "${PROBE_KEYS[$i]}" $i
done

###############################################################################
# Generate Summary Report
###############################################################################

log_header "PROBE SUMMARY REPORT"

echo ""
echo "Total Probes:  $TOTAL_PROBES"
echo -e "Passed:        ${GREEN}$PASSED_PROBES${NC} ✓"
echo -e "Failed:        ${RED}$FAILED_PROBES${NC} ✗"
echo -e "Skipped:       ${YELLOW}$SKIPPED_PROBES${NC} ○"
echo ""

# Generate JSON report
REPORT_FILE="/tmp/rese_probe_report_$$.json"

cat > "$REPORT_FILE" << EOF
{
  "probe_name": "rese_all_probes",
  "probe_type": "comprehensive_verification",
  "correlation_id": "$CORRELATION_ID",
  "timestamp": "$TIMESTAMP",
  "source_service": "rese_integration_probe",
  "target_service": "rese_all_components",
  "configuration": {
    "frontend_root": "$FRONTEND_ROOT",
    "adapters_root": "$ADAPTERS_ROOT",
    "python_cmd": "$PYTHON_CMD"
  },
  "probes": [
EOF

# Add probe results to JSON
for i in "${!PROBE_NAMES[@]}"; do
    comma=$([ $i -lt $((${#PROBE_NAMES[@]} - 1)) ] && echo "," || echo "")

    # Escape output for JSON
    output_escaped=$(echo "${PROBE_OUTPUTS[$i]}" | sed 's/"/\\"/g' | tr -d '\n' | head -c 500)

    cat >> "$REPORT_FILE" << EOF
    {
      "name": "${PROBE_NAMES[$i]}",
      "key": "${PROBE_KEYS[$i]}",
      "path": "${PROBE_PATHS[$i]}",
      "status": "${PROBE_STATUS[$i]}",
      "exit_code": ${PROBE_EXIT_CODES[$i]},
      "output": "$output_escaped"
    }$comma
EOF
done

cat >> "$REPORT_FILE" << EOF
  ],
  "summary": {
    "total_probes": $TOTAL_PROBES,
    "passed": $PASSED_PROBES,
    "failed": $FAILED_PROBES,
    "skipped": $SKIPPED_PROBES,
    "success_rate": $(awk "BEGIN {printf \"%.1f\", ($PASSED_PROBES/$TOTAL_PROBES)*100}")%
  }
}
EOF

# Show report file location
echo ""
log_info "Detailed JSON report saved to: $REPORT_FILE"
echo ""

# Pretty print report if jq available
if command -v jq &> /dev/null; then
    echo "JSON Report (pretty-printed):"
    echo "─────────────────────────────────────────────────────────────"
    jq '.' "$REPORT_FILE" 2>/dev/null || cat "$REPORT_FILE"
    echo "─────────────────────────────────────────────────────────────"
fi

###############################################################################
# Final Status
###############################################################################

echo ""
log_header "FINAL STATUS"

if [ $FAILED_PROBES -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    🎉 ALL PROBES PASSED 🎉                   ║${NC}"
    echo -e "${GREEN}║                                                              ║${NC}"
    echo -e "${GREEN}║  All RESE components are verified and operational.           ║${NC}"
    echo -e "${GREEN}║  The system is ready for integration and use.                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    log_info "Success Rate: 100% ($PASSED_PROBES/$TOTAL_PROBES)"
    echo ""

    # List all passed components
    echo "Verified Components:"
    for i in "${!PROBE_NAMES[@]}"; do
        if [ "${PROBE_STATUS[$i]}" = "pass" ]; then
            echo -e "  ${GREEN}✓${NC} ${PROBE_NAMES[$i]}"
        fi
    done
    echo ""

    exit 0
else
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                 ⚠️  SOME PROBES FAILED ⚠️                     ║${NC}"
    echo -e "${RED}║                                                              ║${NC}"
    echo -e "${RED}║  Please review the errors above and fix the issues.           ║${NC}"
    echo -e "${RED}║  Success Rate: $(awk "BEGIN {printf \"%.1f\", ($PASSED_PROBES/$TOTAL_PROBES)*100}")% ($PASSED_PROBES/$TOTAL_PROBES)                          ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # List failed probes
    echo "Failed Probes:"
    for i in "${!PROBE_NAMES[@]}"; do
        if [ "${PROBE_STATUS[$i]}" = "fail" ]; then
            echo -e "  ${RED}✗${NC} ${PROBE_NAMES[$i]} (exit: ${PROBE_EXIT_CODES[$i]})"
        fi
    done
    echo ""

    # List passed probes
    echo "Passed Probes:"
    for i in "${!PROBE_NAMES[@]}"; do
        if [ "${PROBE_STATUS[$i]}" = "pass" ]; then
            echo -e "  ${GREEN}✓${NC} ${PROBE_NAMES[$i]}"
        fi
    done
    echo ""

    # Common fixes
    echo "Common Fixes:"
    echo "  1. Python path issues → Set PYTHON_CMD or install Python 3.9+"
    echo "  2. Import errors → Check PYTHONPATH includes glue/lib and glue/schemas"
    echo "  3. Missing dependencies → pip install numpy pydantic fastapi uvicorn"
    echo "  4. Working directory → Run from FRONTEND_ROOT: $FRONTEND_ROOT"
    echo "  5. TypeScript/Node.js issues → npm install && npm run build"
    echo ""

    exit 1
fi
