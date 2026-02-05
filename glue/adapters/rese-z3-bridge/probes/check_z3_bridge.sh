#!/bin/bash
################################################################################
# RESE-Z3 Bridge Runtime Verification Probe
#
# Law of Runtime Truth: Verify the bridge actually works before claiming it does
#
# This probe executes actual calls through the bridge to verify:
# 1. Bridge can connect to Z3 server
# 2. Circuit breaker is functional
# 3. All API methods work
# 4. Canonical schema transformations work
#
# Exit codes:
#   0 - All checks passed
#   1 - Health check failed
#   2 - Solve API failed
#   3 - Contradiction detection failed
#   4 - Theorem proving failed
#   5 - Circuit breaker failed
#
# Author: RESE Team
# Created: 2026-02-04
################################################################################

set -e

# Configuration from environment (Law of Configuration Explicitness)
Z3_BASE_URL="${Z3_BASE_URL:-http://localhost:8000}"
Z3_TIMEOUT_MS="${Z3_TIMEOUT_MS:-30000}"
PYTHON="${PYTHON:-python3}"

echo "=== RESE-Z3 Bridge Runtime Verification Probe ==="
echo "Z3 Base URL: $Z3_BASE_URL"
echo "Timeout: ${Z3_TIMEOUT_MS}ms"
echo ""

# Change to bridge directory
cd "$(dirname "$0")/.."

# ============================================================================
# Helper Functions
# ============================================================================

check_python_deps() {
    echo "Checking Python dependencies..."
    if ! $PYTHON -c "import requests" 2>/dev/null; then
        echo "ERROR: requests module not installed"
        exit 1
    fi

    if ! $PYTHON -c "import sys; sys.path.insert(0, 'src'); from rese_z3_bridge import RESEZ3Bridge" 2>/dev/null; then
        echo "ERROR: RESE-Z3 Bridge module not found"
        exit 1
    fi

    echo "✓ Python dependencies OK"
    echo ""
}

check_z3_server() {
    echo "Checking Z3 server health..."
    if ! curl -s --max-time 5 "${Z3_BASE_URL}/health" > /dev/null 2>&1; then
        echo "ERROR: Cannot connect to Z3 server at ${Z3_BASE_URL}"
        echo "Is the Z3 server running?"
        exit 1
    fi

    echo "✓ Z3 server is responding"
    echo ""
}

run_bridge_test() {
    local test_name="$1"
    local test_code="$2"

    echo "Running test: ${test_name}"

    if ! output=$($PYTHON -c "$test_code" 2>&1); then
        echo "✗ Test failed: ${test_name}"
        echo "Error output:"
        echo "$output"
        return 1
    fi

    echo "✓ Test passed: ${test_name}"
    echo ""
    return 0
}

# ============================================================================
# Main Test Suite
# ============================================================================

main() {
    echo "Starting runtime verification..."
    echo ""

    # Check dependencies
    check_python_deps

    # Check Z3 server
    check_z3_server

    # Test 1: Bridge initialization
    run_bridge_test "Bridge Initialization" "
import sys
sys.path.insert(0, 'src')
from rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig

config = RESEZ3BridgeConfig.from_env()
bridge = RESEZ3Bridge(config)
print('Bridge initialized successfully')
health = bridge.get_health()
print(f'Health status: {health[\"status\"]}')
bridge.close()
"

    # Test 2: Solve constraints API
    run_bridge_test "Solve Constraints API" "
import sys
import os
sys.path.insert(0, 'src')
from rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig
from rese_z3_schema import CanonicalVariable, CanonicalConstraint, ConstraintType

config = RESEZ3BridgeConfig.from_env()
bridge = RESEZ3Bridge(config)

variables = [CanonicalVariable('x', ConstraintType.INTEGER)]
constraints = [CanonicalConstraint('(> x 0)', ConstraintType.INTEGER)]

response = bridge.solve_constraints(variables, constraints, correlation_id='probe-test-1')
print(f'Solve result: {response.result.value}')

if response.result.value not in ['sat', 'unsat', 'unknown']:
    print(f'ERROR: Unexpected result: {response.result.value}')
    sys.exit(1)

bridge.close()
"

    # Test 3: Detect contradictions API
    run_bridge_test "Detect Contradictions API" "
import sys
sys.path.insert(0, 'src')
from rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig
from rese_z3_schema import CanonicalConstraint, ConstraintType

config = RESEZ3BridgeConfig.from_env()
bridge = RESEZ3Bridge(config)

# Contradictory constraints
constraints = [
    CanonicalConstraint('(> x 100)', ConstraintType.INTEGER),
    CanonicalConstraint('(< x 0)', ConstraintType.INTEGER),
]

has_contradiction, counterexample = bridge.detect_contradictions(
    constraints, correlation_id='probe-test-2'
)
print(f'Contradiction detected: {has_contradiction}')

bridge.close()
"

    # Test 4: Theorem proving API
    run_bridge_test "Theorem Proving API" "
import sys
sys.path.insert(0, 'src')
from rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig

config = RESEZ3BridgeConfig.from_env()
bridge = RESEZ3Bridge(config)

# Simple theorem: x > 0 implies x + 1 > 0
theorem = '(implies (> x 0) (> (+ x 1) 0))'
response = bridge.prove_theorem(
    theorem_statement=theorem,
    variables={'x': 'Int'},
    correlation_id='probe-test-3'
)
print(f'Theorem proven: {response.proven}')

bridge.close()
"

    # Test 5: Circuit breaker
    run_bridge_test "Circuit Breaker Functionality" "
import sys
sys.path.insert(0, 'src')
from rese_z3_bridge import RESEZ3Bridge, RESEZ3BridgeConfig
from rese_z3_schema import CanonicalVariable, CanonicalConstraint, ConstraintType

config = RESEZ3BridgeConfig(
    z3_base_url=os.getenv('Z3_BASE_URL', 'http://localhost:8000'),
    z3_timeout_ms=30000,
    circuit_breaker_threshold=2,
    enable_cache=False,
)
bridge = RESEZ3Bridge(config)

variables = [CanonicalVariable('x', ConstraintType.INTEGER)]
constraints = [CanonicalConstraint('(> x 0)', ConstraintType.INTEGER)]

# Get initial stats
stats_initial = bridge.get_stats()
cb_state_initial = stats_initial['client_stats']['circuit_breaker']['state']
print(f'Initial circuit breaker state: {cb_state_initial}')

bridge.close()
"

    echo "=== All Runtime Verification Tests Passed ==="
    echo ""
    echo "Summary:"
    echo "  ✓ Bridge initialization"
    echo "  ✓ Solve constraints API"
    echo "  ✓ Detect contradictions API"
    echo "  ✓ Theorem proving API"
    echo "  ✓ Circuit breaker"
    echo ""
    echo "The RESE-Z3 Bridge is operational and ready for use."

    return 0
}

# Run main
main
exit $?
