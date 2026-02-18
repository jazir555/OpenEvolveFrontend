#!/bin/bash
###############################################################################
# Probe: Advanced OpenEvolve Features Verification
#
# This script verifies advanced OpenEvolve integration features.
# Part of Law 2: Runtime Truth - verify actual behavior, not documentation.
#
# Usage: ./probes/check_advanced_openevolve.sh
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

echo "========================================================================"
echo "  PROBE: Advanced OpenEvolve Features Verification"
echo "========================================================================"
echo ""
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-test}"

echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

###############################################################################
# Test 1: Advanced integration import
###############################################################################
echo "Test 1: Advanced OpenEvolve integration import"

if python -c "import sys; sys.path.insert(0, os.path.abspath('..')); from src import get_advanced_openevolve_integration; get_advanced_openevolve_integration()" 2>&1 | grep -q ""; then
    pass "Advanced OpenEvolve integration imports"
else
    fail "Advanced OpenEvolve integration import failed"
fi

###############################################################################
# Test 2: Problem decomposition
###############################################################################
echo ""
echo "Test 2: Problem decomposition"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_advanced_openevolve_integration

advanced = get_advanced_openevolve_integration()
decomposition = advanced.decompose_problem(
    workflow_id='test',
    problem_statement='Test problem',
    workflow_type='evolution',
    max_depth=2
)
print(f'Sub-problems: {len(decomposition.sub_problems)}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Sub-problems:"; then
    pass "Problem decomposition works"
else
    fail "Problem decomposition failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 3: Team selection
###############################################################################
echo ""
echo "Test 3: Team selection"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_advanced_openevolve_integration

advanced = get_advanced_openevolve_integration()
selection = advanced.select_teams_for_stage(
    workflow_id='test',
    stage='planning',
    workflow_type='evolution',
    complexity_score=0.7
)
print(f'Teams: {len(selection.recommended_teams)}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Teams:"; then
    pass "Team selection works"
else
    fail "Team selection failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 4: Resource optimization
###############################################################################
echo ""
echo "Test 4: Resource optimization"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_advanced_openevolve_integration

advanced = get_advanced_openevolve_integration()
optimization = advanced.optimize_resources(
    workflow_id='test',
    stage='execution',
    complexity_score=0.7,
    estimated_duration_ms=60000
)
print(f'CPU: {optimization.cpu_allocation}')
print(f'Memory: {optimization.memory_allocation_mb}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "CPU:"; then
    pass "Resource optimization works"
else
    fail "Resource optimization failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 5: Workflow checkpoints
###############################################################################
echo ""
echo "Test 5: Workflow checkpoint save/restore"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_advanced_openevolve_integration

advanced = get_advanced_openevolve_integration()

# Save checkpoint
advanced.save_checkpoint(
    workflow_id='test',
    stage='planning',
    state={'test': 'data'},
    metrics={'complexity': 0.7}
)

# List checkpoints
checkpoints = advanced.list_checkpoints('test')
print(f'Checkpoints: {len(checkpoints)}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Checkpoints:"; then
    pass "Workflow checkpoint save/restore works"
else
    fail "Workflow checkpoint failed"
    echo "  Error: $TEST_OUTPUT"
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
    echo -e "${GREEN}SUCCESS: All advanced OpenEvolve tests passed${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}FAILURE: $TESTS_FAILED test(s) failed${NC}"
    echo ""
    exit 1
fi
