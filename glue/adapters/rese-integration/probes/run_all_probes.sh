#!/bin/bash
# run_all_probes.sh
# Master probe runner - executes all RESE probes and generates summary report
# Following "Law of Runtime Truth" - we trust execution, not documentation

set -e

PROBE_DIR="$(dirname "$0")"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
TOTAL_PROBES=3
PASSED_PROBES=0
FAILED_PROBES=0

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║     RESE PROBE SUITE - Runtime Verification                      ║"
echo "║     Following Law of Runtime Truth: Trust Execution              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Started at: $TIMESTAMP"
echo "Probe directory: $PROBE_DIR"
echo ""

# Array to store results
declare -a PROBE_RESULTS
declare -a PROBE_NAMES=("check_rese_dependencies" "check_rese_api" "check_rese_phases")
declare -a PROBE_FILES=("check_rese_dependencies.sh" "check_rese_api.sh" "check_rese_phases.sh")

# Run each probe
for i in "${!PROBE_NAMES[@]}"; do
    PROBE_NAME="${PROBE_NAMES[$i]}"
    PROBE_FILE="${PROBE_FILES[$i]}"
    PROBE_PATH="$PROBE_DIR/$PROBE_FILE"

    echo "─────────────────────────────────────────────────────────────────"
    echo "Running: $PROBE_NAME"
    echo "─────────────────────────────────────────────────────────────────"

    if [ ! -f "$PROBE_PATH" ]; then
        echo "❌ ERROR: Probe file not found: $PROBE_PATH"
        FAILED_PROBES=$((FAILED_PROBES + 1))
        PROBE_RESULTS[$i]="{\"probe\": \"$PROBE_NAME\", \"status\": \"ERROR\", \"message\": \"Probe file not found\"}"
        echo ""
        continue
    fi

    # Make sure probe is executable
    chmod +x "$PROBE_PATH"

    # Run probe and capture exit code
    if bash "$PROBE_PATH" > /tmp/probe_output_$$.json 2>&1; then
        EXIT_CODE=0
        STATUS="✅ PASS"
        PASSED_PROBES=$((PASSED_PROBES + 1))
    else
        EXIT_CODE=$?
        STATUS="❌ FAIL"
        FAILED_PROBES=$((FAILED_PROBES + 1))
    fi

    # Display output
    if [ -f /tmp/probe_output_$$.json ]; then
        # Pretty print with jq if available
        if command -v jq &> /dev/null; then
            jq '.' /tmp/probe_output_$$.json 2>/dev/null || cat /tmp/probe_output_$$.json
        else
            cat /tmp/probe_output_$$.json
        fi
        rm -f /tmp/probe_output_$$.json
    fi

    echo ""
    echo "Exit Code: $EXIT_CODE | Status: $STATUS"
    echo ""
done

# Generate summary
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                        SUMMARY                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Total Probes:  $TOTAL_PROBES"
echo "Passed:        $PASSED_PROBES ✅"
echo "Failed:        $FAILED_PROBES ❌"
echo ""

if [ $FAILED_PROBES -eq 0 ]; then
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    🎉 ALL PROBES PASSED 🎉                       ║"
    echo "║                                                                  ║"
    echo "║  RESE is ready for use.                                          ║"
    echo "║                                                                  ║"
    echo "║  Note: RESE appears to be in bytecode format. For full          ║"
    echo "║  functionality, restore source code (see Task #1).              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    exit 0
else
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    ⚠️  SOME PROBES FAILED ⚠️                     ║"
    echo "║                                                                  ║"
    echo "║  Review the failed probe outputs above for details.             ║"
    echo "║                                                                  ║"
    echo "║  Common issues:                                                  ║"
    echo "║  - Missing dependencies → Run: pip install <package>            ║"
    echo "║  - API not running → Start RESE API: python -m rese.api         ║"
    echo "║  - Path issues → Set RESE_ROOT_DIR environment variable         ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    exit 1
fi
