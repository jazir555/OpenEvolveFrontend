#!/bin/bash
###############################################################################
# Probe: Cache Features Verification
#
# This script verifies that caching features work correctly.
# Part of Law 2: Runtime Truth - verify actual behavior, not documentation.
#
# Usage: ./probes/check_cache_features.sh
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
echo "  PROBE: Cache Features Verification"
echo "========================================================================"
echo ""
export ADAPTIVE_MDAP_TIMEOUT_MS=5000
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-test}"

echo "Start Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

###############################################################################
# Test 1: Cache initialization
###############################################################################
echo "Test 1: Cache initialization"

TEST_OUTPUT=$(python -c "
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
from src import get_async_adapter

adapter = get_async_adapter()
stats = adapter.get_cache_stats()
print(f'Max Size: {stats[\"max_size\"]}')
print(f'TTL: {stats[\"ttl\"]}')
print('OK')
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Max Size:"; then
    pass "Cache initializes correctly"
else
    fail "Cache initialization failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 2: Cache miss on first call
###############################################################################
echo ""
echo "Test 2: Cache miss on first call"

TEST_OUTPUT=$(python -c "
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    sp = CanonicalSubProblem(
        id='miss_test',
        description='Cache miss test',
        domain='test',
        depth=1
    )
    # Clear cache first
    adapter.cache.clear()
    stats_before = adapter.get_cache_stats()
    misses_before = stats_before['total_misses']

    await adapter.analyze_complexity_async(sp, use_cache=True)

    stats_after = adapter.get_cache_stats()
    misses_after = stats_after['total_misses']

    print(f'Misses increased: {misses_after > misses_before}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Misses increased: True"; then
    pass "Cache miss tracked correctly"
else
    fail "Cache miss tracking failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 3: Cache hit on second call
###############################################################################
echo ""
echo "Test 3: Cache hit on second call"

TEST_OUTPUT=$(python -c "
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    sp = CanonicalSubProblem(
        id='hit_test',
        description='Cache hit test',
        domain='test',
        depth=1
    )
    # First call
    await adapter.analyze_complexity_async(sp, use_cache=True)
    stats_before = adapter.get_cache_stats()
    hits_before = stats_before['total_hits']

    # Second call (should hit cache)
    await adapter.analyze_complexity_async(sp, use_cache=True)

    stats_after = adapter.get_cache_stats()
    hits_after = stats_after['total_hits']

    print(f'Hits increased: {hits_after > hits_before}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Hits increased: True"; then
    pass "Cache hit tracked correctly"
else
    fail "Cache hit tracking failed"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 4: Cache size limit
###############################################################################
echo ""
echo "Test 4: Cache size limit enforcement"

TEST_OUTPUT=$(python -c "
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    max_size = adapter.get_cache_stats()['max_size']

    # Try to add more items than max size
    for i in range(max_size + 10):
        sp = CanonicalSubProblem(
            id=f'size_test_{i}',
            description=f'Size test {i}',
            domain='test',
            depth=1
        )
        await adapter.analyze_complexity_async(sp, use_cache=True)

    stats = adapter.get_cache_stats()
    size = stats['size']
    print(f'Size within limit: {size <= max_size}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Size within limit: True"; then
    pass "Cache size limit enforced"
else
    fail "Cache size limit not enforced"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 5: Cache hit rate calculation
###############################################################################
echo ""
echo "Test 5: Cache hit rate calculation"

TEST_OUTPUT=$(python -c "
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()
    sp = CanonicalSubProblem(
        id='rate_test',
        description='Rate test',
        domain='test',
        depth=1
    )

    # First call (miss)
    await adapter.analyze_complexity_async(sp, use_cache=True)
    # Second call (hit)
    await adapter.analyze_complexity_async(sp, use_cache=True)
    # Third call (hit)
    await adapter.analyze_complexity_async(sp, use_cache=True)

    stats = adapter.get_cache_stats()
    hit_rate = stats['hit_rate']
    total = stats['total_hits'] + stats['total_misses']

    print(f'Total requests: {total}')
    print(f'Hit rate: {hit_rate:.2f}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Hit rate: 0.67"; then
    pass "Cache hit rate calculated correctly (2/3 = 66.7%)"
else
    fail "Cache hit rate calculation incorrect"
    echo "  Error: $TEST_OUTPUT"
fi

###############################################################################
# Test 6: Cache key generation
###############################################################################
echo ""
echo "Test 6: Cache key generation"

TEST_OUTPUT=$(python -c "
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')) if os.path.exists('../src') else 'src')
from src import get_async_adapter, CanonicalSubProblem

async def test():
    adapter = get_async_adapter()

    # Two identical subproblems should generate same cache key
    sp1 = CanonicalSubProblem(
        id='key_test',
        description='Key test',
        domain='test',
        depth=1
    )
    sp2 = CanonicalSubProblem(
        id='key_test',
        description='Key test',
        domain='test',
        depth=1
    )

    # First call
    await adapter.analyze_complexity_async(sp1, use_cache=True)
    stats_before = adapter.get_cache_stats()
    misses_before = stats_before['total_misses']
    hits_before = stats_before['total_hits']

    # Second identical call should hit cache
    await adapter.analyze_complexity_async(sp2, use_cache=True)

    stats_after = adapter.get_cache_stats()
    misses_after = stats_after['total_misses']
    hits_after = stats_after['total_hits']

    # Should have 1 miss, 1 hit (not 2 misses)
    print(f'Correct behavior: {misses_after == misses_before and hits_after > hits_before}')
    print('OK')

asyncio.run(test())
" 2>&1 || echo "ERROR")

if echo "$TEST_OUTPUT" | grep -q "Correct behavior: True"; then
    pass "Cache key generation is consistent"
else
    fail "Cache key generation is inconsistent"
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
    echo -e "${GREEN}SUCCESS: All cache feature tests passed${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}FAILURE: $TESTS_FAILED test(s) failed${NC}"
    echo ""
    exit 1
fi
