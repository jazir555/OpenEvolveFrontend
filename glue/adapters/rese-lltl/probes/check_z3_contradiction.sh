#!/bin/bash
##############################################################################
# Z3 Contradiction Detection Probe Script for LLTL
#
# Purpose: Verify Z3 can detect contradictions in formal commitments
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   Z3_TIMEOUT         - Z3 solver timeout in ms (default: 5000)
#   RESE_SIGNIFICANCE_LEVEL - Statistical significance level (default: 0.05)
#
# Exit Codes:
#   0 - All probes passed
#   1 - Z3 not available
#   2 - SAT case failed
#   3 - UNSAT case failed
#   4 - Unsatisfiable core extraction failed
#   5 - Python not available
#
# Author: RESE Team
# Created: 2026-02-04
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

Z3_TIMEOUT="${Z3_TIMEOUT:-5000}"
RESE_SIGNIFICANCE_LEVEL="${RESE_SIGNIFICANCE_LEVEL:-0.05}"

# =============================================================================
# Utility Functions
# =============================================================================

# Log JSON Lines output
log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_z3_contradiction.sh\"}"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        log_json "error" "python3 is not installed or not in PATH"
        exit 5
    fi
}

# Check if Z3 Python bindings are available
check_z3_python() {
    if ! python3 -c "import z3" 2>/dev/null; then
        log_json "error" "Z3 Python bindings not available (import z3 failed)"
        return 1
    fi
    log_json "info" "Z3 Python bindings available"
    return 0
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Check Z3 is available
probe_z3_available() {
    log_json "info" "Checking Z3 availability"

    if ! check_z3_python; then
        return 1
    fi

    # Get Z3 version
    local version
    version=$(python3 -c "import z3; print(z3.get_version())" 2>/dev/null || echo "unknown")
    log_json "info" "Z3 version: $version"

    return 0
}

# Probe 2: Test SAT (satisfiable) case - no contradictions
probe_sat_case() {
    log_json "info" "Testing SAT case (no contradictions)"

    # Create Python script for SAT test
    local test_script=$(cat <<'EOF'
import z3
import json

# Create solver
solver = z3.Solver()
solver.set("timeout", 5000)

# Create variables
confidence = z3.Real('confidence')
p_value = z3.Real('p_value')

# Add non-contradictory constraints
solver.add(confidence >= 0.90)
solver.add(confidence <= 0.95)
solver.add(p_value <= 0.05)

# Check satisfiability
result = solver.check()

if result == z3.sat:
    model = solver.model()
    output = {
        "status": "sat",
        "confidence": str(model.eval(confidence)),
        "p_value": str(model.eval(p_value))
    }
    print(json.dumps(output))
elif result == z3.unsat:
    output = {"status": "unsat", "error": "Unexpectedly unsatisfiable"}
    print(json.dumps(output))
    exit(1)
else:
    output = {"status": "unknown", "error": "Solver returned unknown"}
    print(json.dumps(output))
    exit(1)
EOF
)

    local result
    result=$(python3 -c "$test_script" 2>&1)

    # Check if result contains valid JSON
    if ! echo "$result" | jq -e '.' &> /dev/null; then
        log_json "error" "SAT test returned invalid JSON: $result"
        return 1
    fi

    # Check status
    local status
    status=$(echo "$result" | jq -r '.status')

    if [ "$status" != "sat" ]; then
        log_json "error" "SAT test failed with status: $status"
        return 1
    fi

    log_json "info" "SAT test passed: $result"
    return 0
}

# Probe 3: Test UNSAT case - with contradictions
probe_unsat_case() {
    log_json "info" "Testing UNSAT case (with contradictions)"

    # Create Python script for UNSAT test
    local test_script=$(cat <<'EOF'
import z3
import json

# Create solver with unsat core tracking
solver = z3.Solver()
solver.set("timeout", 5000)

# Create variables
x = z3.Real('x')

# Add contradictory constraints
c1 = z3.Assert(x > 10)
c2 = z3.Assert(x < 5)

# Name assertions for unsat core
c1_named = z3.Bool("c1")
c2_named = z3.Bool("c2")

solver.assert_and_track(x > 10, "c1")
solver.assert_and_track(x < 5, "c2")

# Check satisfiability
result = solver.check()

if result == z3.unsat:
    # Extract unsat core
    core = solver.unsat_core()
    output = {
        "status": "unsat",
        "contradictions_detected": True,
        "unsat_core": [str(c) for c in core]
    }
    print(json.dumps(output))
elif result == z3.sat:
    output = {"status": "sat", "error": "Unexpectedly satisfiable"}
    print(json.dumps(output))
    exit(1)
else:
    output = {"status": "unknown", "error": "Solver returned unknown"}
    print(json.dumps(output))
    exit(1)
EOF
)

    local result
    result=$(python3 -c "$test_script" 2>&1)

    # Check if result contains valid JSON
    if ! echo "$result" | jq -e '.' &> /dev/null; then
        log_json "error" "UNSAT test returned invalid JSON: $result"
        return 1
    fi

    # Check status
    local status
    status=$(echo "$result" | jq -r '.status')

    if [ "$status" != "unsat" ]; then
        log_json "error" "UNSAT test failed with status: $status"
        return 1
    fi

    # Check contradictions detected
    local contradictions
    contradictions=$(echo "$result" | jq -r '.contradictions_detected')

    if [ "$contradictions" != "True" ]; then
        log_json "error" "UNSAT test did not detect contradictions"
        return 1
    fi

    # Check unsat core
    local core_size
    core_size=$(echo "$result" | jq -r '.unsat_core | length')

    if [ "$core_size" -lt 1 ]; then
        log_json "error" "UNSAT core extraction failed (empty core)"
        return 1
    fi

    log_json "info" "UNSAT test passed with $core_size contradictions: $result"
    return 0
}

# Probe 4: Test formal commitment encoding
probe_formal_commitment_encoding() {
    log_json "info" "Testing formal commitment to Z3 formula encoding"

    # Create Python script for encoding test
    local test_script=$(cat <<'EOF'
import z3
import json
import re

def encode_statement_to_z3(statement):
    """Encode statement as Z3 formula"""
    # Check for inequality
    if '<' in statement and '>' not in statement:
        match = re.search(r'(\w+)\s*<\s*([0-9.]+)', statement)
        if match:
            var, val = match.groups()
            return f"({var} < {val})", var, float(val)
    elif '>' in statement:
        match = re.search(r'(\w+)\s*>\s*([0-9.]+)', statement)
        if match:
            var, val = match.groups()
            return f"({var} > {val})", var, float(val)

    # Check for equality
    elif '=' in statement:
        match = re.search(r'(\w+)\s*=\s*([0-9.]+)', statement)
        if match:
            var, val = match.groups()
            return f"({var} == {val})", var, float(val)

    return None, None, None

# Test encoding
test_cases = [
    ("x > 5", True),
    ("x < 10", True),
    ("confidence >= 0.95", True),
]

results = []
for statement, should_succeed in test_cases:
    formula, var, val = encode_statement_to_z3(statement)
    success = formula is not None
    results.append({
        "statement": statement,
        "encoded": formula,
        "success": success,
        "expected": should_succeed
    })

# Check all encodings succeeded
all_success = all(r["success"] for r in results)

output = {
    "status": "success" if all_success else "failure",
    "encodings": results
}

print(json.dumps(output))

if not all_success:
    exit(1)
EOF
)

    local result
    result=$(python3 -c "$test_script" 2>&1)

    # Check if result contains valid JSON
    if ! echo "$result" | jq -e '.' &> /dev/null; then
        log_json "error" "Encoding test returned invalid JSON: $result"
        return 1
    fi

    # Check status
    local status
    status=$(echo "$result" | jq -r '.status')

    if [ "$status" != "success" ]; then
        log_json "error" "Encoding test failed: $result"
        return 1
    fi

    log_json "info" "Formal commitment encoding test passed"
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting Z3 contradiction detection probe"
    log_json "info" "Z3 timeout: ${Z3_TIMEOUT}ms"
    log_json "info" "Significance level: ${RESE_SIGNIFICANCE_LEVEL}"

    # Check prerequisites
    check_python

    # Run probes sequentially (fail fast on first error)
    if ! probe_z3_available; then
        log_json "error" "Z3 availability probe failed"
        exit 1
    fi

    if ! probe_sat_case; then
        log_json "error" "SAT case probe failed"
        exit 2
    fi

    if ! probe_unsat_case; then
        log_json "error" "UNSAT case probe failed"
        exit 3
    fi

    if ! probe_formal_commitment_encoding; then
        log_json "error" "Formal commitment encoding probe failed"
        exit 4
    fi

    # All probes passed
    log_json "info" "All Z3 contradiction detection probes passed successfully"
    exit 0
}

# Run main function
main "$@"
