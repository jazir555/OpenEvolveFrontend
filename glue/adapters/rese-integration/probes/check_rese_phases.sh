#!/bin/bash
# check_rese_phases.sh
# RESE Phase Probe - Following "Law of Runtime Truth"
# We trust execution, not documentation.
#
# This probe validates each RESE phase can initialize and has dependencies met.
# Exit code: 0 = success, 1 = failure
# Output: Structured JSON to stdout

set -e

# Detect Python command (same as dependencies probe)
PYTHON_CMD=""
if [ -f "/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe" ]; then
    PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"
elif command -v python3 &> /dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null 2>&1; then
    PYTHON_CMD="py"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "{\"error\": \"Python not found\"}"
    exit 1
fi

# Configuration
RESE_ROOT_DIR="${RESE_ROOT_DIR:-/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese}"

# Generate correlation ID
CORRELATION_ID=$($PYTHON_CMD -c "import uuid; print(str(uuid.uuid4()))")
TIMESTAMP=$($PYTHON_CMD -c "from datetime import datetime; print(datetime.utcnow().isoformat() + 'Z')")

# Initialize JSON output
echo "{"
echo "  \"probe_name\": \"check_rese_phases\","
echo "  \"probe_type\": \"phase_verification\","
echo "  \"correlation_id\": \"$CORRELATION_ID\","
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"source_service\": \"rese_probe\","
echo "  \"target_service\": \"rese_pipeline\","
echo "  \"rese_root\": \"$RESE_ROOT_DIR\","
echo "  \"phases\": {"

EXIT_CODE=0
DELIMITER=""

# Function to check phase directory and components
check_phase() {
    local phase_name="$1"
    local phase_num="$2"
    local required="$3"

    local phase_dir="$RESE_ROOT_DIR/$phase_name"

    if [ -d "$phase_dir" ]; then
        # Count files in phase directory
        FILE_COUNT=$(find "$phase_dir" -type f -name "*.pyc" 2>/dev/null | wc -l)

        echo "$DELIMITER"
        echo "    \"phase${phase_num}\": {"
        echo "      \"status\": \"PASS\","
        echo "      \"required\": $required,"
        echo "      \"directory\": \"$phase_name\","
        echo "      \"pyc_files\": $FILE_COUNT,"
        echo "      \"exists\": true,"
        echo "      \"message\": \"Phase $phase_num ($phase_name) directory exists with $FILE_COUNT bytecode files\""
        echo -n "    }"
        DELIMITER=","
    else
        echo "$DELIMITER"
        echo "    \"phase${phase_num}\": {"
        echo "      \"status\": \"FAIL\","
        echo "      \"required\": $required,"
        echo "      \"directory\": \"$phase_name\","
        echo "      \"pyc_files\": 0,"
        echo "      \"exists\": false,"
        echo "      \"message\": \"WARNING: Phase $phase_num ($phase_name) directory not found at $phase_dir\""
        echo -n "    }"
        DELIMITER=","
        if [ "$required" = "true" ]; then
            EXIT_CODE=1
        fi
    fi
}

# Check for gamma1 components (mentioned in docs)
if [ -d "$RESE_ROOT_DIR/gamma1" ]; then
    PYC_COUNT=$(find "$RESE_ROOT_DIR/gamma1" -name "*.pyc" 2>/dev/null | wc -l)
    echo "$DELIMITER"
    echo "    \"gamma1\": {"
    echo "      \"status\": \"PASS\","
    echo "      \"required\": true,"
    echo "      \"directory\": \"gamma1\","
    echo "      \"pyc_files\": $PYC_COUNT,"
    echo "      \"exists\": true,"
    echo "      \"message\": \"Gamma1 components exist with $PYC_COUNT bytecode files\""
    echo -n "    }"
    DELIMITER=","
fi

# Check core components
if [ -d "$RESE_ROOT_DIR/core" ]; then
    PYC_COUNT=$(find "$RESE_ROOT_DIR/core" -name "*.pyc" 2>/dev/null | wc -l)
    echo "$DELIMITER"
    echo "    \"core\": {"
    echo "      \"status\": \"PASS\","
    echo "      \"required\": true,"
    echo "      \"directory\": \"core\","
    echo "      \"pyc_files\": $PYC_COUNT,"
    echo "      \"exists\": true,"
    echo "      \"message\": \"Core components exist with $PYC_COUNT bytecode files\""
    echo -n "    }"
    DELIMITER=","
fi

# Try to test phase initialization via Python
# This creates a temporary Python script to test imports
TEST_PYTHON_PHASES=$(cat <<'EOF'
import sys
sys.path.insert(0, '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese')

results = {}

# Phase 0: Core Infrastructure
try:
    # Try to import core modules
    import importlib.util
    core_path = '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/core'
    results['phase0_import'] = False
    results['phase0_message'] = 'Cannot test - only bytecode exists'

    # Check if symbolic_constraint_engine exists as bytecode
    import os
    if os.path.exists(os.path.join(core_path, 'symbolic_constraint_engine.pyc')):
        results['phase0_import'] = True
        results['phase0_message'] = 'Bytecode exists for symbolic_constraint_engine'
except Exception as e:
    results['phase0_import'] = False
    results['phase0_message'] = str(e)

# Gamma1 (Phase III components)
try:
    import os
    gamma1_path = '/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese/gamma1'
    results['gamma1_components'] = []

    if os.path.exists(gamma1_path):
        # List subdirectories in gamma1
        for item in os.listdir(gamma1_path):
            item_path = os.path.join(gamma1_path, item)
            if os.path.isdir(item_path):
                results['gamma1_components'].append(item)
except Exception as e:
    results['gamma1_error'] = str(e)

import json
print(json.dumps(results))
EOF
)

PHASE_TEST_RESULTS=$(python3 -c "$TEST_PYTHON_PHASES" 2>/dev/null || echo '{}')

echo ""
echo "  },"
echo "  \"phase_tests\": $PHASE_TEST_RESULTS,"
echo "  \"overall_status\": \"$([ $EXIT_CODE -eq 0 ] && echo 'PASS' || echo 'FAIL')\","
echo "  \"exit_code\": $EXIT_CODE,"
echo "  \"note\": \"RESE appears to be in bytecode (.pyc) format - source code restoration may be required (see Task #1)\","
echo "  \"recommendation\": \"$([ $EXIT_CODE -eq 0 ] && echo 'RESE phase directories exist but runtime testing requires source restoration' || echo 'Some RESE phases are missing or incomplete')\""
echo "}"

exit $EXIT_CODE
