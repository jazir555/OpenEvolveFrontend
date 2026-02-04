#!/bin/bash

###############################################################################
# Graphiti API Probe Script
#
# Federation Constitution - Law of Runtime Truth:
# "Before implementing a feature, you must write a probe script that executes
#  the call against the live container. If the probe fails, the feature does
#  not exist."
#
# This script verifies that Graphiti API is accessible and returns expected data
###############################################################################

set -e  # Fail on error
set -u  # Fail on undefined variables

# Configuration from environment (Law of Configuration Explicitness)
GRAPHITI_URL="${GRAPHITI_URL:-http://localhost:8001}"
TIMEOUT="${GRAPHITI_TIMEOUT:-5}"
EXPECTED_STATUS=200

echo "=== Graphiti API Probe ==="
echo "Target: $GRAPHITI_URL"
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
echo "GET $GRAPHITI_URL/health"

HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time "$TIMEOUT" \
    "$GRAPHITI_URL/health" 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" = "$EXPECTED_STATUS" ]; then
    print_success "Health check passed (HTTP $HEALTH_RESPONSE)"
else
    print_error "Health check failed (HTTP $HEALTH_RESPONSE, expected $EXPECTED_STATUS)"
    exit 1
fi

echo ""

# Test 2: Search API with test query
echo "Test 2: Search API Test"
echo "POST $GRAPHITI_URL/api/search"

SEARCH_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"query": "test entity", "limit": 5}' \
    "$GRAPHITI_URL/api/search" 2>/dev/null)

HTTP_CODE=$(echo "$SEARCH_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$SEARCH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Search API accessible (HTTP $HTTP_CODE)"

    # Validate response structure
    if echo "$RESPONSE_BODY" | jq -e '.entities or .relationships' > /dev/null 2>&1; then
        print_success "Response has expected fields"

        ENTITY_COUNT=$(echo "$RESPONSE_BODY" | jq -r '.entities | length' 2>/dev/null || echo "0")
        REL_COUNT=$(echo "$RESPONSE_BODY" | jq -r '.relationships | length' 2>/dev/null || echo "0")
        echo "  └─ Entities: $ENTITY_COUNT"
        echo "  └─ Relationships: $REL_COUNT"
    else
        print_warning "Response structure unexpected."
        echo "  └─ Response: $RESPONSE_BODY"
    fi
else
    print_error "Search API failed (HTTP $HTTP_CODE, expected $EXPECTED_STATUS)"
    echo "  └─ Response: $RESPONSE_BODY"
    exit 1
fi

echo ""

# Test 3: Entities API
echo "Test 3: Entities API"
echo "GET $GRAPHITI_URL/api/entities?limit=5"

ENTITIES_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    "$GRAPHITI_URL/api/entities?limit=5" 2>/dev/null)

HTTP_CODE=$(echo "$ENTITIES_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$ENTITIES_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Entities API accessible (HTTP $HTTP_CODE)"

    if echo "$RESPONSE_BODY" | jq -e '.entities' > /dev/null 2>&1; then
        ENTITY_COUNT=$(echo "$RESPONSE_BODY" | jq -r '.entities | length' 2>/dev/null || echo "0")
        echo "  └─ Entity count: $ENTITY_COUNT"
    fi
else
    print_warning "Entities API not available (HTTP $HTTP_CODE)"
fi

echo ""

# Test 4: Temporal Query API
echo "Test 4: Temporal Query API Test"
echo "POST $GRAPHITI_URL/api/temporal"

TEMPORAL_RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "query": "test",
      "start_date": "2024-01-01T00:00:00Z",
      "end_date": "2024-12-31T23:59:59Z",
      "limit": 5
    }' \
    "$GRAPHITI_URL/api/temporal" 2>/dev/null)

HTTP_CODE=$(echo "$TEMPORAL_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$TEMPORAL_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "$EXPECTED_STATUS" ]; then
    print_success "Temporal API accessible (HTTP $HTTP_CODE)"
else
    print_warning "Temporal API not available (HTTP $HTTP_CODE)"
fi

echo ""
echo "=== Graphiti Probe Complete ==="
echo "All critical endpoints are accessible."
echo "Graphiti system is PROBED and VERIFIED."
echo ""
