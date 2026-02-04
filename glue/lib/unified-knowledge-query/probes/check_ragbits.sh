#!/bin/bash

###############################################################################
# RAGBits API Probe Script
#
# Federation Constitution - Law of Runtime Truth:
# "Before implementing a feature, you must write a probe script that executes
#  the call against the live container. If the probe fails, the feature does
#  not exist."
#
# This script verifies that RAGBits API is accessible and returns expected data
###############################################################################

set -e  # Fail on error
set -u  # Fail on undefined variables

# Configuration from environment (Law of Configuration Explicitness)
RAGBITS_URL="${RAGBITS_URL:-http://localhost:8000}"
TIMEOUT="${RAGBITS_TIMEOUT:-5}"
EXPECTED_STATUS=200

echo "=== RAGBits API Probe ==="
echo "Target: $RAGBITS_URL"
echo "Timeout: ${TIMEOUT}s"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Test 1: Health Check
echo "Test 1: Health Check Endpoint"
echo "GET $RAGBITS_URL/health"

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time "$TIMEOUT" \
    "$RAGBITS_URL/health" 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" = "$EXPECTED_STATUS" ]; then
    print_success "Health check passed (HTTP $HEALTH_RESPONSE)"
else
    print_error "Health check failed (HTTP $HEALTH_RESPONSE, expected $EXPECTED_STATUS)"
    exit 1
fi

echo ""

# Test 2: Search API with test query
echo "Test 2: Search API Test"
echo "POST $RAGBITS_URL/api/search"

SEARCH_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "test query", "top_k": 5}' \
    "$RAGBITS_URL/api/search" 2>/dev/null)

HTTP_CODE=$(echo "$SEARCH_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$SEARCH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Search API accessible (HTTP $HTTP_CODE)"

    # Validate response structure
    if echo "$RESPONSE_BODY" | jq -e '.documents' > /dev/null 2>&1; then
        print_success "Response has 'documents' field"

        DOC_COUNT=$(echo "$RESPONSE_BODY" | jq -r '.documents | length' 2>/dev/null || echo "0")
        echo "  └─ Document count: $DOC_COUNT"
    else
        print_warning "Response structure unexpected. Expected 'documents' field."
        echo "  └─ Response: $RESPONSE_BODY"
    fi
else
    print_error "Search API failed (HTTP $HTTP_CODE, expected $EXPECTED_STATUS)"
    echo "  └─ Response: $RESPONSE_BODY"
    exit 1
fi

echo ""

# Test 3: Stats API
echo "Test 3: Statistics API"
echo "GET $RAGBITS_URL/api/stats"

STATS_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    "$RAGBITS_URL/api/stats" 2>/dev/null)

HTTP_CODE=$(echo "$STATS_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$STATS_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Stats API accessible (HTTP $HTTP_CODE)"
    echo "  └─ Response: $RESPONSE_BODY"
else
    print_warning "Stats API not available (HTTP $HTTP_CODE)"
fi

echo ""
echo "=== RAGBits Probe Complete ==="
echo "All critical endpoints are accessible."
echo "RAGBits system is PROBED and VERIFIED."
echo ""
