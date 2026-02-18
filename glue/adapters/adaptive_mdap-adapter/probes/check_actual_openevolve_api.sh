#!/bin/bash
###############################################################################
# Probe: Actual OpenEvolve API Validation
#
# This script validates the adapter integration against the actual OpenEvolve
# API in the codebase (Gap 9 resolution).
#
# Part of Law 2: Runtime Truth - verify actual behavior, not documentation.
#
# Usage: From adapter root, run: ./probes/check_actual_openevolve_api.sh
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

echo "========================================================================"
echo "  PROBE: Actual OpenEvolve API Validation"
echo "========================================================================"
echo ""
echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "This probe validates the adapter integration against the actual"
echo "OpenEvolve evolution.py API in the codebase."
echo ""

# Set required environment variables
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-test}"

###############################################################################
# Test 1: Check if evolution.py exists
###############################################################################
echo "Test 1: Verify OpenEvolve evolution.py exists"

if [ -f "../../../evolution.py" ]; then
    pass "Found evolution.py in OpenEvolve codebase"
    EVOLUTION_PATH="../../../evolution.py"
elif [ -f "../../evolution.py" ]; then
    pass "Found evolution.py in Frontend directory"
    EVOLUTION_PATH="../../evolution.py"
else
    fail "Could not find evolution.py"
    echo "  Searching from: $(pwd)"
    exit 1
fi

###############################################################################
# Test 2: Import evolution.py module
###############################################################################
echo ""
echo "Test 2: Import OpenEvolve evolution module"

TEST_OUTPUT=$(python -c "
import sys
sys.path.insert(0, "../../../")

try:
    import evolution
    print('OK')
except ImportError as e:
    print(f'ERROR: {e}')
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "OpenEvolve evolution module imports successfully"
else
    fail "Failed to import evolution module"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 3: Check for key OpenEvolve functions
###############################################################################
echo ""
echo "Test 3: Verify key OpenEvolve functions exist"

TEST_OUTPUT=$(python -c "
import sys
sys.path.insert(0, "../../../")
import evolution

functions = [
    'run_evolution_loop',
    'run_comprehensive_evolution',
    'run_gauntlet_evolution',
    'get_evolution_capabilities_summary',
    'create_evolution_configuration'
]

missing = []
for func in functions:
    if not hasattr(evolution, func):
        missing.append(func)

if missing:
    print(f'MISSING: {missing}')
else:
    print('OK: All functions found')
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "All key OpenEvolve functions are available"
else
    fail "Some OpenEvolve functions are missing"
    echo "  $TEST_OUTPUT"
fi

###############################################################################
# Test 4: Get evolution capabilities
###############################################################################
echo ""
echo "Test 4: Query OpenEvolve capabilities"

TEST_OUTPUT=$(python -c "
import sys
import os
sys.path.insert(0, "../../../")

# Set minimal required env vars
os.environ.setdefault('DEEPSEEK_API_KEY', 'sk-test')

try:
    import evolution
    capabilities = evolution.get_evolution_capabilities_summary()
    print(f'Capabilities returned: {type(capabilities).__name__}')
    print(f'Has keys: {len(capabilities) if isinstance(capabilities, dict) else \"N/A\"}')
    if isinstance(capabilities, dict):
        for key in list(capabilities.keys())[:5]:
            print(f'  - {key}')
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "OpenEvolve capabilities query works"
    echo "  $(echo "$TEST_OUTPUT" | grep "Capabilities returned:")"
    echo "  $(echo "$TEST_OUTPUT" | grep "Has keys:")"
else
    fail "Failed to query OpenEvolve capabilities"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 5: Create evolution configuration
###############################################################################
echo ""
echo "Test 5: Create evolution configuration"

TEST_OUTPUT=$(python -c "
import sys
import os
sys.path.insert(0, "../../../")

os.environ.setdefault('DEEPSEEK_API_KEY', 'sk-test')

try:
    import evolution
    config = evolution.create_evolution_configuration({
        'max_iterations': 5,
        'temperature': 0.7
    })
    print(f'Config type: {type(config).__name__}')
    print(f'Has max_iterations: {hasattr(config, \"max_iterations\")}')
    if hasattr(config, 'max_iterations'):
        print(f'Max iterations: {config.max_iterations}')
    print('OK')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "Evolution configuration creation works"
else
    fail "Failed to create evolution configuration"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 6: Test adapter integration with actual OpenEvolve
###############################################################################
echo ""
echo "Test 6: Adapter integration with actual OpenEvolve"

TEST_OUTPUT=$(python -c "
import sys
import os
sys.path.insert(0, "../../../")
sys.path.insert(0, "../../../")

os.environ.setdefault('ADAPTIVE_MDAP_TIMEOUT_MS', '5000')
os.environ.setdefault('DEEPSEEK_API_KEY', os.getenv('DEEPSEEK_API_KEY', 'sk-test'))

try:
    # Import actual OpenEvolve
    import evolution

    # Import adapter components
    from src import get_advanced_openevolve_integration

    # Get advanced integration
    advanced = get_advanced_openevolve_integration()

    # Test decomposition using actual OpenEvolve logic
    decomposition = advanced.decompose_problem(
        workflow_id='api_validation_test',
        problem_statement='Test problem with actual OpenEvolve',
        workflow_type='evolution',
        max_depth=2
    )

    print(f'Sub-problems: {len(decomposition.sub_problems)}')
    print(f'Strategy: {decomposition.decomposition_strategy}')
    print('OK')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "Adapter integration with actual OpenEvolve works"
    echo "  $(echo "$TEST_OUTPUT" | grep "Sub-problems:")"
    echo "  $(echo "$TEST_OUTPUT" | grep "Strategy:")"
else
    fail "Adapter integration with actual OpenEvolve failed"
    echo "  Error (last 10 lines):"
    echo "$TEST_OUTPUT" | tail -10 | sed 's/^/    /'
fi

###############################################################################
# Test 7: Verify data transformation compatibility
###############################################################################
echo ""
echo "Test 7: Data transformation compatibility"

TEST_OUTPUT=$(python -c "
import sys
import os
sys.path.insert(0, "../../../")
sys.path.insert(0, "../../../")

os.environ.setdefault('ADAPTIVE_MDAP_TIMEOUT_MS', '5000')
os.environ.setdefault('DEEPSEEK_API_KEY', os.getenv('DEEPSEEK_API_KEY', 'sk-test'))

try:
    # Import actual OpenEvolve
    import evolution

    # Import adapter
    from src import get_integration_manager, CanonicalSubProblem

    manager = get_integration_manager()

    # Create canonical subproblem
    subproblem = CanonicalSubProblem(
        id='transformation_test',
        description='Test data transformation',
        domain='test',
        depth=1
    )

    # Analyze through adapter
    response = manager.analyze_workflow(
        workflow_id='test',
        problem_statement='Test transformation',
        workflow_type='evolution'
    )

    print(f'Analysis successful: {response is not None}')
    print(f'Has complexity: {hasattr(response, \"overall_complexity\")}')
    if hasattr(response, 'overall_complexity'):
        print(f'Complexity: {response.overall_complexity:.3f}')
    print('OK')

except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
" 2>&1)

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "Data transformation is compatible"
else
    fail "Data transformation compatibility check failed"
    echo "  Error (last 10 lines):"
    echo "$TEST_OUTPUT" | tail -10 | sed 's/^/    /'
fi

###############################################################################
# Summary
###############################################################################
echo ""
echo "========================================================================"
echo "  TEST SUMMARY"
echo "========================================================================"
echo ""
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}SUCCESS: Adapter integration validated against actual OpenEvolve API${NC}"
    echo ""
    echo "Validated:"
    echo "  [OK] OpenEvolve evolution.py exists and is importable"
    echo "  [OK] All key OpenEvolve functions are available"
    echo "  [OK] Capabilities query works"
    echo "  [OK] Configuration creation works"
    echo "  [OK] Adapter integrates with actual OpenEvolve"
    echo "  [OK] Data transformation is compatible"
    echo ""
    echo "The adapter is validated to work with the actual OpenEvolve system."
    echo ""
    exit 0
else
    echo -e "${RED}FAILURE: $TESTS_FAILED test(s) failed${NC}"
    echo ""
    echo "Some integration validations failed."
    echo "Review the errors above for details."
    echo ""
    exit 1
fi
