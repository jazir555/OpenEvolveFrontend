#!/bin/bash
# Test script for LoongFlow API
# Following CLAUDE.md: Law of Runtime Truth - verify API actually works

set -e

API_URL="${LOONGFLOW_API_URL:-http://localhost:8000}"

echo "========================================="
echo "LoongFlow API Test Suite"
echo "========================================="
echo "API URL: $API_URL"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
test_endpoint() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"

    echo -n "Testing: $test_name ... "

    if [ -z "$data" ]; then
        response=$(curl -s -X "$method" "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -w "\n%{http_code}")
    else
        response=$(curl -s -X "$method" "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data" \
            -w "\n%{http_code}")
    fi

    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    if [ "$status_code" -ge 200 ] && [ "$status_code" -lt 300 ]; then
        echo -e "${GREEN}PASSED${NC} (HTTP $status_code)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        echo "Response: $body" | head -c 200
        echo ""
    else
        echo -e "${RED}FAILED${NC} (HTTP $status_code)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo "Response: $body"
    fi
    echo ""
}

# Test 1: Health Check
test_endpoint \
    "Health Check" \
    "GET" \
    "/health"

# Test 2: Start Evolution
echo -n "Starting evolution (this may take a moment) ... "
evolution_response=$(curl -s -X POST "$API_URL/api/v1/evolve" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "test-evolution",
        "task": "Test task for API validation",
        "max_generations": 2
    }')

evolution_id=$(echo "$evolution_response" | grep -o '"evolution_id":"[^"]*"' | cut -d'"' -f4)

if [ -n "$evolution_id" ]; then
    echo -e "${GREEN}PASSED${NC}"
    echo "Evolution ID: $evolution_id"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}FAILED${NC}"
    echo "Response: $evolution_response"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 3: Get Status
if [ -n "$evolution_id" ]; then
    test_endpoint \
        "Get Evolution Status" \
        "GET" \
        "/api/v1/status/$evolution_id"

    # Wait a bit for evolution to progress
    echo "Waiting 2 seconds for evolution to progress..."
    sleep 2

    # Test 4: Check Status Again
    test_endpoint \
        "Get Evolution Status (After Progress)" \
        "GET" \
        "/api/v1/status/$evolution_id"

    # Test 5: List Evolutions
    test_endpoint \
        "List Evolutions" \
        "GET" \
        "/api/v1/evolutions?limit=10"

    # Test 6: Wait for completion and get solution
    echo "Waiting for evolution to complete..."
    for i in {1..10}; do
        sleep 1
        status_response=$(curl -s "$API_URL/api/v1/status/$evolution_id")
        status=$(echo "$status_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)

        if [ "$status" = "COMPLETED" ] || [ "$status" = "FAILED" ]; then
            echo "Evolution completed with status: $status"
            break
        fi
        echo -n "."
    done
    echo ""

    # Test 7: Get Solution
    test_endpoint \
        "Get Solution" \
        "GET" \
        "/api/v1/solutions/$evolution_id"

    # Test 8: Delete Evolution
    test_endpoint \
        "Delete Evolution" \
        "DELETE" \
        "/api/v1/evolutions/$evolution_id"

    # Test 9: Verify Deletion
    echo -n "Verifying deletion (should fail with 404) ... "
    delete_check=$(curl -s -X GET "$API_URL/api/v1/status/$evolution_id" -w "\n%{http_code}")
    delete_status=$(echo "$delete_check" | tail -n1)

    if [ "$delete_status" = "404" ]; then
        echo -e "${GREEN}PASSED${NC} (Correctly returns 404)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}FAILED${NC} (Expected 404, got $delete_status)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
fi

# Test 10: Error Handling - Get Non-existent Evolution
test_endpoint \
    "Error Handling (Non-existent Evolution)" \
    "GET" \
    "/api/v1/status/invalid_evolution_id"

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo "Total: $((TESTS_PASSED + TESTS_FAILED))"
echo "========================================="

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
