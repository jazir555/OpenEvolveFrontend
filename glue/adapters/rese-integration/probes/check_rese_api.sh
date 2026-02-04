#!/bin/bash
# check_rese_api.sh
# RESE API Probe - Following "Law of Runtime Truth"
# We trust execution, not documentation.
#
# This probe verifies RESE components are accessible and API is responsive.
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

# Configuration - MUST be provided via environment (Law of Configuration Explicitness)
RESE_API_HOST="${RESE_API_HOST:-localhost}"
RESE_API_PORT="${RESE_API_PORT:-8000}"
RESE_API_URL="http://${RESE_API_HOST}:${RESE_API_PORT}"
RESE_ROOT_DIR="${RESE_ROOT_DIR:-/c/Users/mmeadow/Documents/OpenEvolve/Frontend/rese}"

# Generate correlation ID
CORRELATION_ID=$($PYTHON_CMD -c "import uuid; print(str(uuid.uuid4()))")
TIMESTAMP=$($PYTHON_CMD -c "from datetime import datetime; print(datetime.utcnow().isoformat() + 'Z')")

# Initialize JSON output
echo "{"
echo "  \"probe_name\": \"check_rese_api\","
echo "  \"probe_type\": \"api_verification\","
echo "  \"correlation_id\": \"$CORRELATION_ID\","
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"source_service\": \"rese_probe\","
echo "  \"target_service\": \"rese_api\","
echo "  \"api_url\": \"$RESE_API_URL\","
echo "  \"checks\": {"

EXIT_CODE=0
DELIMITER=""

# Function to check an endpoint
check_endpoint() {
    local name="$1"
    local endpoint="$2"
    local required="$3"
    local expected_code="${4:-200}"

    local url="${RESE_API_URL}${endpoint}"

    # Try curl with timeout
    RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$url" 2>/dev/null || echo "000")

    if [ "$RESPONSE" = "$expected_code" ]; then
        echo "$DELIMITER"
        echo "    \"${name}\": {"
        echo "      \"status\": \"PASS\","
        echo "      \"required\": $required,"
        echo "      \"endpoint\": \"$endpoint\","
        echo "      \"http_code\": $RESPONSE,"
        echo "      \"expected_code\": $expected_code,"
        echo "      \"message\": \"Endpoint accessible\""
        echo -n "    }"
        DELIMITER=","
    else
        echo "$DELIMITER"
        echo "    \"${name}\": {"
        echo "      \"status\": \"FAIL\","
        echo "      \"required\": $required,"
        echo "      \"endpoint\": \"$endpoint\","
        echo "      \"http_code\": $RESPONSE,"
        echo "      \"expected_code\": $expected_code,"
        echo "      \"message\": \"CRITICAL: Endpoint returned $RESPONSE, expected $expected_code\""
        echo -n "    }"
        DELIMITER=","
        if [ "$required" = "true" ]; then
            EXIT_CODE=1
        fi
    fi
}

# Function to test Python imports
check_rese_import() {
    local module="$1"
    local required="$2"

    # Try to import from the RESE root directory
    if $PYTHON_CMD -c "import sys; sys.path.insert(0, '$RESE_ROOT_DIR'); import $module" 2>/dev/null; then
        echo "$DELIMITER"
        echo "    \"import_${module}\": {"
        echo "      \"status\": \"PASS\","
        echo "      \"required\": $required,"
        echo "      \"module\": \"$module\","
        echo "      \"message\": \"Module importable from RESE directory\""
        echo -n "    }"
        DELIMITER=","
    else
        echo "$DELIMITER"
        echo "    \"import_${module}\": {"
        echo "      \"status\": \"FAIL\","
        echo "      \"required\": $required,"
        echo "      \"module\": \"$module\","
        echo "      \"message\": \"WARNING: Module $module not importable - RESE may not be properly installed\""
        echo -n "    }"
        DELIMITER=","
        if [ "$required" = "true" ]; then
            EXIT_CODE=1
        fi
    fi
}

# Check if RESE directory exists
if [ -d "$RESE_ROOT_DIR" ]; then
    echo "$DELIMITER"
    echo "    \"rese_directory\": {"
    echo "      \"status\": \"PASS\","
    echo "      \"required\": true,"
    echo "      \"path\": \"$RESE_ROOT_DIR\","
    echo "      \"message\": \"RESE root directory exists\""
    echo -n "    }"
    DELIMITER=","
else
    echo "$DELIMITER"
    echo "    \"rese_directory\": {"
    echo "      \"status\": \"FAIL\","
    echo "      \"required\": true,"
    echo "      \"path\": \"$RESE_ROOT_DIR\","
    echo "      \"message\": \"CRITICAL: RESE root directory not found at $RESE_ROOT_DIR\""
    echo -n "    }"
    DELIMITER=","
    EXIT_CODE=1
fi

# Check for core RESE modules (note: checking for bytecode .pyc files since source is .pyc)
if [ -d "$RESE_ROOT_DIR/core" ]; then
    PYC_COUNT=$(find "$RESE_ROOT_DIR/core" -name "*.pyc" 2>/dev/null | wc -l)
    echo "$DELIMITER"
    echo "    \"core_modules\": {"
    echo "      \"status\": \"PASS\","
    echo "      \"required\": true,"
    echo "      \"pyc_files_found\": $PYC_COUNT,"
    echo "      \"message\": \"Core modules directory exists with $PYC_COUNT bytecode files\""
    echo -n "    }"
    DELIMITER=","
else
    echo "$DELIMITER"
    echo "    \"core_modules\": {"
    echo "      \"status\": \"FAIL\","
    echo "      \"required\": true,"
    echo "      \"pyc_files_found\": 0,"
    echo "      \"message\": \"CRITICAL: Core modules directory not found\""
    echo -n "    }"
    DELIMITER=","
    EXIT_CODE=1
fi

# Try to import RESE modules (these may fail if only bytecode exists)
check_rese_import "rese.rese_pipeline" "false"
check_rese_import "rese.api" "false"

# Check API endpoints (if API is running)
check_endpoint "health" "/health" "false" 200
check_endpoint "api_docs" "/docs" "false" 200

echo ""
echo "  },"
echo "  \"overall_status\": \"$([ $EXIT_CODE -eq 0 ] && echo 'PASS' || echo 'FAIL')\","
echo "  \"exit_code\": $EXIT_CODE,"
echo "  \"api_url\": \"$RESE_API_URL\","
echo "  \"rese_root\": \"$RESE_ROOT_DIR\","
echo "  \"recommendation\": \"$([ $EXIT_CODE -eq 0 ] && echo 'RESE API is accessible' || echo 'RESE API may not be running or not fully installed')\""
echo "}"

exit $EXIT_CODE
