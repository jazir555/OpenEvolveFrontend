#!/bin/bash
###############################################################################
# RESE Full Pipeline Probe Script
#
# Following CLAUDE.md "Law of Runtime Truth":
# - We trust EXECUTION, not documentation
# - This probe verifies the entire RESE pipeline is functional
# - All phases must pass for the system to be considered healthy
#
# This script runs all phase probes in sequence and reports overall health.
#
# Usage: ./check_full_pipeline.sh
#
# Exit codes:
# 0: All phases passed
# 1: One or more phases failed
# 2: Probe execution error
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
ADAPTERS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROBES_ROOT="$SCRIPT_DIR"

# Phase probe paths (using absolute paths from adapters root)
PHASE1_PROBE="$ADAPTERS_ROOT/rese-phase1/probes/check_phase1.sh"
PHASE2_PROBE="$ADAPTERS_ROOT/rese-phase2/probes/check_phase2.sh"
PHASE3_PROBE="$ADAPTERS_ROOT/rese-phase3/probes/check_phase3.sh"
PHASE4_PROBE="$ADAPTERS_ROOT/rese-phase4/probes/check_phase4.sh"

# Results tracking
PHASE1_STATUS="unknown"
PHASE2_STATUS="unknown"
PHASE3_STATUS="unknown"
PHASE4_STATUS="unknown"
PHASE1_OUTPUT=""
PHASE2_OUTPUT=""
PHASE3_OUTPUT=""
PHASE4_OUTPUT=""

# Timestamp
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CORRELATION_ID=$(python3 -c "import uuid; print(str(uuid.uuid4()))" 2>/dev/null || echo "unknown")

###############################################################################
# Helper Functions
###############################################################################

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

run_phase_probe() {
    local phase_name="$1"
    local probe_path="$2"
    local output_var="$3"

    log_info "Running $phase_name probe..."

    if [ ! -f "$probe_path" ]; then
        log_error "Probe not found: $probe_path"
        eval "$output_var='probe_not_found'"
        return 1
    fi

    if [ ! -x "$probe_path" ]; then
        log_warn "Probe not executable, attempting to run with bash..."
        if bash "$probe_path" > /tmp/phase_output.txt 2>&1; then
            eval "$output_var='pass'"
            log_info "$phase_name: PASSED"
            cat /tmp/phase_output.txt | tail -20
            return 0
        else
            eval "$output_var='fail'"
            log_error "$phase_name: FAILED"
            cat /tmp/phase_output.txt | tail -20
            return 1
        fi
    fi

    # Run the probe
    if "$probe_path" > /tmp/phase_output.txt 2>&1; then
        eval "$output_var='pass'"
        log_info "$phase_name: PASSED"
        cat /tmp/phase_output.txt | tail -20
        return 0
    else
        eval "$output_var='fail'"
        log_error "$phase_name: FAILED"
        cat /tmp/phase_output.txt | tail -20
        return 1
    fi
}

###############################################################################
# Main Execution
###############################################################################

# Print header
clear
log_header "RESE Full Pipeline Probe"
log_info "Timestamp: $TIMESTAMP"
log_info "Correlation ID: $CORRELATION_ID"
log_info "Probe Root: $PROBES_ROOT"
echo ""

# Initialize counters
TOTAL_PHASES=4
PASSED_PHASES=0
FAILED_PHASES=0

###############################################################################
# Phase I Probe
###############################################################################

log_header "Phase I: Epistemic Audit"
if run_phase_probe "Phase I" "$PHASE1_PROBE" "PHASE1_STATUS"; then
    PASSED_PHASES=$((PASSED_PHASES + 1))
else
    FAILED_PHASES=$((FAILED_PHASES + 1))
fi

###############################################################################
# Phase II Probe
###############################################################################

log_header "Phase II: Isomorphic Mapping"
if run_phase_probe "Phase II" "$PHASE2_PROBE" "PHASE2_STATUS"; then
    PASSED_PHASES=$((PASSED_PHASES + 1))
else
    FAILED_PHASES=$((FAILED_PHASES + 1))
fi

###############################################################################
# Phase III Probe
###############################################################################

log_header "Phase III: MCTS Search"
if run_phase_probe "Phase III" "$PHASE3_PROBE" "PHASE3_STATUS"; then
    PASSED_PHASES=$((PASSED_PHASES + 1))
else
    FAILED_PHASES=$((FAILED_PHASES + 1))
fi

###############################################################################
# Phase IV Probe
###############################################################################

log_header "Phase IV: Architecture Assembly"
if run_phase_probe "Phase IV" "$PHASE4_PROBE" "PHASE4_STATUS"; then
    PASSED_PHASES=$((PASSED_PHASES + 1))
else
    FAILED_PHASES=$((FAILED_PHASES + 1))
fi

###############################################################################
# Generate Report
###############################################################################

log_header "Pipeline Health Report"

# Generate JSON report
cat << EOF
{
  "probe_name": "check_full_pipeline",
  "probe_type": "full_pipeline_verification",
  "correlation_id": "$CORRELATION_ID",
  "timestamp": "$TIMESTAMP",
  "source_service": "rese_integration_probe",
  "target_service": "rese_full_pipeline",
  "phases": {
    "phase1": {
      "name": "Epistemic Audit",
      "status": "$PHASE1_STATUS"
    },
    "phase2": {
      "name": "Isomorphic Mapping",
      "status": "$PHASE2_STATUS"
    },
    "phase3": {
      "name": "MCTS Search",
      "status": "$PHASE3_STATUS"
    },
    "phase4": {
      "name": "Architecture Assembly",
      "status": "$PHASE4_STATUS"
    }
  },
  "summary": {
    "total_phases": $TOTAL_PHASES,
    "passed_phases": $PASSED_PHASES,
    "failed_phases": $FAILED_PHASES
  }
}
EOF

echo ""

# Print summary
log_info "Summary:"
echo "  Total Phases: $TOTAL_PHASES"
echo "  Passed: $PASSED_PHASES"
echo "  Failed: $FAILED_PHASES"
echo ""

# Determine overall status
if [ $FAILED_PHASES -eq 0 ]; then
    log_info "✓ ALL PHASES PASSED"
    log_info "The RESE pipeline is fully functional and ready for use."
    echo ""
    log_info "Phase I (Epistemic Audit): ${GREEN}OPERATIONAL${NC}"
    log_info "Phase II (Isomorphic Mapping): ${GREEN}OPERATIONAL${NC}"
    log_info "Phase III (MCTS Search): ${GREEN}OPERATIONAL${NC}"
    log_info "Phase IV (Architecture Assembly): ${GREEN}OPERATIONAL${NC}"
    echo ""
    exit 0
else
    log_error "✗ PIPELINE INCOMPLETE"
    log_warn "$FAILED_PHASES phase(s) failed. Please review the errors above."
    echo ""

    # Print status for each phase
    if [ "$PHASE1_STATUS" = "pass" ]; then
        log_info "Phase I (Epistemic Audit): ${GREEN}OPERATIONAL${NC}"
    else
        log_error "Phase I (Epistemic Audit): ${RED}FAILED${NC}"
    fi

    if [ "$PHASE2_STATUS" = "pass" ]; then
        log_info "Phase II (Isomorphic Mapping): ${GREEN}OPERATIONAL${NC}"
    else
        log_error "Phase II (Isomorphic Mapping): ${RED}FAILED${NC}"
    fi

    if [ "$PHASE3_STATUS" = "pass" ]; then
        log_info "Phase III (MCTS Search): ${GREEN}OPERATIONAL${NC}"
    else
        log_error "Phase III (MCTS Search): ${RED}FAILED${NC}"
    fi

    if [ "$PHASE4_STATUS" = "pass" ]; then
        log_info "Phase IV (Architecture Assembly): ${GREEN}OPERATIONAL${NC}"
    else
        log_error "Phase IV (Architecture Assembly): ${RED}FAILED${NC}"
    fi

    echo ""
    log_warn "Please fix the failed phases before using the RESE pipeline."
    echo ""
    exit 1
fi
