#!/bin/bash
# check_rese_dependencies.sh
# RESE Dependency Probe - Following "Law of Runtime Truth"
# We trust execution, not documentation.
#
# This probe verifies all required dependencies are present and functional.
# Exit code: 0 = success, 1 = failure
# Output: Structured JSON to stdout

set -e

# Detect Python command (Windows vs Unix)
PYTHON_CMD=""

# Try direct Windows path first (Git Bash compatibility) - most reliable
if [ -f "/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe" ]; then
    PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python.exe"
elif [ -f "/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python" ]; then
    PYTHON_CMD="/c/Users/mmeadow/AppData/Local/Programs/Python/Python311/python"
# Try different Python commands in order
elif command -v python3 &> /dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v py &> /dev/null 2>&1; then
    PYTHON_CMD="py"
fi

# Verify Python is available
if [ -z "$PYTHON_CMD" ]; then
    # Output error in JSON format
    echo "{"
    echo "  \"probe_name\": \"check_rese_dependencies\","
    echo "  \"probe_type\": \"dependency_verification\","
    echo "  \"correlation_id\": \"error-no-python\","
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"overall_status\": \"FAIL\","
    echo "  \"exit_code\": 1,"
    echo "  \"error\": \"Python not found in PATH. Please install Python 3.9+ and add it to PATH, or disable Windows Store execution aliases in Settings > Apps > Advanced settings > App execution aliases\""
    echo "}"
    exit 1
fi

# Generate correlation ID for this probe run
CORRELATION_ID=$($PYTHON_CMD -c "import uuid; print(str(uuid.uuid4()))")
TIMESTAMP=$($PYTHON_CMD -c "from datetime import datetime; print(datetime.utcnow().isoformat() + 'Z')")

# Initialize JSON output
echo "{"
echo "  \"probe_name\": \"check_rese_dependencies\","
echo "  \"probe_type\": \"dependency_verification\","
echo "  \"correlation_id\": \"$CORRELATION_ID\","
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"source_service\": \"rese_probe\","
echo "  \"target_service\": \"rese_core\","
echo "  \"python_command\": \"$PYTHON_CMD\","
echo "  \"checks\": {"

EXIT_CODE=0
DELIMITER=""

# Function to check and output dependency status
check_dependency() {
    local name="$1"
    local command="$2"
    local required="$3"
    local version="$4"

    # Execute version check
    if eval "$command" > /dev/null 2>&1; then
        VERSION_OUTPUT=$(eval "$version" 2>/dev/null || echo "unknown")
        echo "$DELIMITER"
        echo "    \"${name}\": {"
        echo "      \"status\": \"PASS\","
        echo "      \"required\": $required,"
        echo "      \"version\": \"$VERSION_OUTPUT\","
        echo "      \"message\": \"$name is available\""
        echo -n "    }"
        DELIMITER=","
    else
        echo "$DELIMITER"
        echo "    \"${name}\": {"
        echo "      \"status\": \"FAIL\","
        echo "      \"required\": $required,"
        echo "      \"version\": null,"
        echo "      \"message\": \"CRITICAL: $name is NOT installed or not in PATH\""
        echo -n "    }"
        DELIMITER=","
        if [ "$required" = "true" ]; then
            EXIT_CODE=1
        fi
    fi
}

# Check Python version (3.9+ required)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -gt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 9 ]); then
    echo "$DELIMITER"
    echo "    \"python\": {"
    echo "      \"status\": \"PASS\","
    echo "      \"required\": true,"
    echo "      \"version\": \"$PYTHON_VERSION\","
    echo "      \"message\": \"Python version meets requirement (>=3.9)\""
    echo -n "    }"
    DELIMITER=","
else
    echo "$DELIMITER"
    echo "    \"python\": {"
    echo "      \"status\": \"FAIL\","
    echo "      \"required\": true,"
    echo "      \"version\": \"$PYTHON_VERSION\","
    echo "      \"message\": \"CRITICAL: Python 3.9+ required, found $PYTHON_VERSION\""
    echo -n "    }"
    DELIMITER=","
    EXIT_CODE=1
fi

# Check Lean 4 (optional but recommended)
check_dependency "lean4" "which lean" "false" "lean --version 2>&1 | head -1"

# Check required Python packages via import test
check_python_package() {
    local package="$1"
    local import_name="$2"
    local required="$3"

    if $PYTHON_CMD -c "import $import_name" 2>/dev/null; then
        VERSION=$($PYTHON_CMD -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        echo "$DELIMITER"
        echo "    \"${package}\": {"
        echo "      \"status\": \"PASS\","
        echo "      \"required\": $required,"
        echo "      \"version\": \"$VERSION\","
        echo "      \"message\": \"$package is importable\""
        echo -n "    }"
        DELIMITER=","
    else
        echo "$DELIMITER"
        echo "    \"${package}\": {"
        echo "      \"status\": \"FAIL\","
        echo "      \"required\": $required,"
        echo "      \"version\": null,"
        echo "      \"message\": \"CRITICAL: $package cannot be imported - install with: pip install $package\""
        echo -n "    }"
        DELIMITER=","
        if [ "$required" = "true" ]; then
            EXIT_CODE=1
        fi
    fi
}

# Required Python packages for RESE
check_python_package "numpy" "numpy" "true"
check_python_package "pydantic" "pydantic" "true"
check_python_package "fastapi" "fastapi" "true"
check_python_package "uvicorn" "uvicorn" "true"

# Optional but recommended packages
check_python_package "scipy" "scipy" "false"
check_python_package "networkx" "networkx" "false"
check_python_package "psutil" "psutil" "false"
check_python_package "pytest" "pytest" "false"

echo ""
echo "  },"
echo "  \"overall_status\": \"$([ $EXIT_CODE -eq 0 ] && echo 'PASS' || echo 'FAIL')\","
echo "  \"exit_code\": $EXIT_CODE,"
echo "  \"recommendation\": \"$([ $EXIT_CODE -eq 0 ] && echo 'All required dependencies present' || echo 'Install missing required dependencies before proceeding')\""
echo "}"

exit $EXIT_CODE
