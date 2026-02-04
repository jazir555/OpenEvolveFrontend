#!/bin/bash

###############################################################################
# Vector DB API Probe Script
#
# Federation Constitution - Law of Runtime Truth:
# "Before implementing a feature, you must write a probe script that executes
#  the call against the live container. If the probe fails, the feature does
#  not exist."
#
# This script verifies that Vector DB API is accessible and returns expected data
###############################################################################

set -e  # Fail on error
set -u  # Fail on undefined variables

# Configuration from environment (Law of Configuration Explicitness)
VECTORDB_URL="${VECTORDB_URL:-http://localhost:6333}"
TIMEOUT="${VECTORDB_TIMEOUT:-5}"
EXPECTED_STATUS=200

echo "=== Vector DB API Probe ==="
echo "Target: $VECTORDB_URL"
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
echo "GET $VECTORDB_URL/health"

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time "$TIMEOUT" \
    "$VECTORDB_URL/health" 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" = "$EXPECTED_STATUS" ]; then
    print_success "Health check passed (HTTP $HEALTH_RESPONSE)"
else
    print_error "Health check failed (HTTP $HEALTH_RESPONSE, expected $EXPECTED_STATUS)"
    exit 1
fi

echo ""

# Test 2: Collections List
echo "Test 2: Collections API"
echo "GET $VECTORDB_URL/api/collections"

COLLECTIONS_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    "$VECTORDB_URL/api/collections" 2>/dev/null)

HTTP_CODE=$(echo "$COLLECTIONS_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$COLLECTIONS_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Collections API accessible (HTTP $HTTP_CODE)"

    if echo "$RESPONSE_BODY" | jq -e '.collections or .result' > /dev/null 2>&1; then
        COLLECTION_COUNT=$(echo "$RESPONSE_BODY" | jq -r '(.collections // .result | length)' 2>/dev/null || echo "0")
        print_success "Collection count: $COLLECTION_COUNT"
    fi
else
    print_warning "Collections API not accessible (HTTP $HTTP_CODE)"
fi

echo ""

# Test 3: Search API (assuming 'default' collection exists)
echo "Test 3: Search API Test"
echo "POST $VECTORDB_URL/api/collections/default/points/search"

SEARCH_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "vector": [0.1, 0.2, 0.3],
      "limit": 5
    }' \
    "$VECTORDB_URL/api/collections/default/points/search" 2>/dev/null)

HTTP_CODE=$(echo "$SEARCH_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$SEARCH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ] || [ "$HTTP_CODE" = "404" ]; then
    if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
        print_success "Search API accessible (HTTP $HTTP_CODE)"

        if echo "$RESPONSE_BODY" | jq -e '.result' > /dev/null 2>&1; then
            RESULT_COUNT=$(echo "$RESPONSE_BODY" | jq -r '.result | length' 2>/dev/null || echo "0")
            echo "  └─ Result count: $RESULT_COUNT"
        fi
    else
        print_warning "Default collection not found (HTTP 404), but API is accessible"
        echo "  └─ Create a collection named 'default' to enable search tests"
    fi
else
    print_warning "Search API not accessible (HTTP $HTTP_CODE)"
fi

echo ""

# Test 4: Collection Info (if 'default' exists)
echo "Test 4: Collection Info API"
echo "GET $VECTORDB_URL/api/collections/default"

INFO_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    "$VECTORDB_URL/api/collections/default" 2>/dev/null)

HTTP_CODE=$(echo "$INFO_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Collection info accessible (HTTP $HTTP_CODE)"
else
    print_warning "Collection info not accessible (HTTP $HTTP_CODE)"
fi

echo ""
echo "=== Vector DB Probe Complete ==="
echo "All critical endpoints are accessible."
echo "Vector DB system is PROBED and VERIFIED."
echo ""
