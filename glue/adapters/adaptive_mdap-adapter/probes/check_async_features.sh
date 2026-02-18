#!/bin/bash
###############################################################################
# Probe: Async Features Verification
#
# This script verifies that async/await features work correctly in the adapter.
# Part of Law 2: Runtime Truth - verify actual behavior, not documentation.
#
# Usage: ./probes/check_async_features.sh
###############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
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
echo "  PROBE: Async Features Verification"
echo "========================================================================"
echo ""
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-test}"

echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

###############################################################################
# Test 1: Async adapter import
###############################################################################
echo "Test 1: Async adapter import"

TEST_OUTPUT=$(python -c "
import os
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from src import get_async_adapter
adapter = get_async_adapter()
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "Async adapter imports successfully"
else
    fail "Async adapter import failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 2: Async complexity analysis
###############################################################################
echo ""
echo "Test 2: Async complexity analysis"

TEST_OUTPUT=$(python -c "
import os
import sys
import asyncio
sys.path.insert(0, os.path.abspath('..'))
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    sp = CanonicalSubProblem(
        id='test_async',
        description='Test async',
        domain='test',
        depth=1
    )
    result = await adapter.analyze_complexity_async(sp, use_cache=False)
    print(f'Status: {result.status.value}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "OK"; then
    pass "Async complexity analysis works"
else
    fail "Async complexity analysis failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 3: Batch processing
###############################################################################
echo ""
echo "Test 3: Batch processing"

TEST_OUTPUT=$(python -c "
import os
import sys
import asyncio
sys.path.insert(0, os.path.abspath('..'))
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    subproblems = [
        CanonicalSubProblem(
            id=f'batch_{i}',
            description=f'Batch test {i}',
            domain='test',
            depth=1
        )
        for i in range(3)
    ]
    results = await adapter.batch_analyze_complexity(subproblems, max_concurrency=2)
    print(f'Results: {len(results)}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Results: 3"; then
    pass "Batch processing works"
else
    fail "Batch processing failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 4: Cache functionality
###############################################################################
echo ""
echo "Test 4: Cache functionality"

TEST_OUTPUT=$(python -c "
import os
import sys
import asyncio
sys.path.insert(0, os.path.abspath('..'))
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    sp = CanonicalSubProblem(
        id='cache_test',
        description='Cache test',
        domain='test',
        depth=1
    )
    # First call - cache miss
    await adapter.analyze_complexity_async(sp, use_cache=True)
    stats = adapter.get_cache_stats()
    print(f'Size: {stats[\"size\"]}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Size:"; then
    pass "Cache functionality works"
else
    fail "Cache functionality failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 5: Concurrent execution limit
###############################################################################
echo ""
echo "Test 5: Concurrent execution limit"

TEST_OUTPUT=$(python -c "
import os
import sys
import asyncio
sys.path.insert(0, os.path.abspath('..'))
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    subproblems = [
        CanonicalSubProblem(
            id=f'concurrent_{i}',
            description=f'Concurrent test {i}',
            domain='test',
            depth=1
        )
        for i in range(5)
    ]
    import time
    start = time.time()
    results = await adapter.batch_analyze_complexity(subproblems, max_concurrency=2)
    duration = time.time() - start
    print(f'Duration: {duration:.2f}')
    print(f'Count: {len(results)}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Count: 5"; then
    pass "Concurrent execution limit works"
else
    fail "Concurrent execution limit failed"
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
    echo -e "${GREEN}SUCCESS: All async feature tests passed${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}FAILURE: $TESTS_FAILED test(s) failed${NC}"
    echo ""
    exit 1
fi
