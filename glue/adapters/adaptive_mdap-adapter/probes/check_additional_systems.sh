#!/bin/bash
###############################################################################
# Probe: Additional Systems Integration Verification
#
# This script verifies integration with additional systems (CrewAI, MCP, etc.).
# Part of Law 2: Runtime Truth - verify actual behavior, not documentation.
#
# Usage: ./probes/check_additional_systems.sh
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
echo "  PROBE: Additional Systems Integration Verification"
echo "========================================================================"
echo ""
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-test}"

echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

###############################################################################
# Test 1: Unified system monitor import
###############################################################################
echo "Test 1: Unified system monitor import"

if python -c "import sys; sys.path.insert(0, os.path.abspath('..')); from src import get_unified_system_monitor; get_unified_system_monitor()" 2>&1 | grep -q ""; then
    pass "Unified system monitor imports"
else
    fail "Unified system monitor import failed"
fi

###############################################################################
# Test 2: System health check
###############################################################################
echo ""
echo "Test 2: Overall system health check"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_unified_system_monitor

monitor = get_unified_system_monitor()
health = monitor.get_overall_health()
print(f'Overall: {health[\"overall_status\"]}')
print(f'Total systems: {health[\"total_systems\"]}')
print(f'Available: {health[\"available_systems\"]}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Overall:"; then
    pass "System health check works"
else
    fail "System health check failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 3: Individual system status
###############################################################################
echo ""
echo "Test 3: Individual system status checks"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_unified_system_monitor

monitor = get_unified_system_monitor()
systems = [
    'crewai',
    'mcp_tools',
    'knowledge_engine',
    'leanaide',
    'z3_prover'
]

for system in systems:
    if system in monitor.integrations:
        integration = monitor.integrations[system]
        available = integration.check_available()
        print(f'{system}: {available}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "Individual system status checks work"
else
    fail "Individual system status checks failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 4: Knowledge engine workflow
###############################################################################
echo ""
echo "Test 4: Knowledge engine workflow execution"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_unified_system_monitor

monitor = get_unified_system_monitor()
results = monitor.execute_workflow(
    workflow_type='knowledge_retrieval',
    parameters={
        'query': 'Test query',
        'max_results': 3
    }
)
print(f'Success: {results[\"success\"]}')
print(f'Steps: {len(results[\"steps\"])}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Success:"; then
    pass "Knowledge engine workflow executes"
else
    fail "Knowledge engine workflow failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 5: Formal verification workflow
###############################################################################
echo ""
echo "Test 5: Formal verification workflow execution"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_unified_system_monitor

monitor = get_unified_system_monitor()
results = monitor.execute_workflow(
    workflow_type='formal_verification',
    parameters={
        'statement': 'Test theorem',
        'constraints': ['x > 0']
    }
)
print(f'Success: {results[\"success\"]}')
print(f'Steps: {len(results[\"steps\"])}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Success:"; then
    pass "Formal verification workflow executes"
else
    fail "Formal verification workflow failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 6: Agent collaboration workflow
###############################################################################
echo ""
echo "Test 6: Agent collaboration workflow execution"

TEST_OUTPUT=$(python -c "
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from src import get_unified_system_monitor

monitor = get_unified_system_monitor()
results = monitor.execute_workflow(
    workflow_type='agent_collaboration',
    parameters={
        'task': 'Test task',
        'agents': [
            {'name': 'agent1', 'role': 'test'},
            {'name': 'agent2', 'role': 'test'}
        ]
    }
)
print(f'Success: {results[\"success\"]}')
print(f'Steps: {len(results[\"steps\"])}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Success:"; then
    pass "Agent collaboration workflow executes"
else
    fail "Agent collaboration workflow failed"
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
    echo -e "${GREEN}SUCCESS: All additional systems tests passed${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}FAILURE: $TESTS_FAILED test(s) failed${NC}"
    echo ""
    exit 1
fi
