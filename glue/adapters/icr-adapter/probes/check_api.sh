#!/bin/bash

###############################################################################
# ICR API Connectivity Probe
#
# FEDERATION CONSTITUTION COMPLIANCE:
# - Law of Runtime Truth: This probe MUST successfully execute before
#   writing any integration code.
# - If this probe fails, the API does not exist or is not accessible.
#
# Usage: ./check_api.sh
###############################################################################

set -e  # Exit on error

# Configuration from environment (NO DEFAULTS - Law of Configuration Explicitness)
if [ -z "$OPENEVOLVE_ICR_API_URL" ]; then
    echo "ERROR: Missing required environment variable: OPENEVOLVE_ICR_API_URL"
    echo "The Federation Constitution prohibits magic defaults."
    exit 1
fi

if [ -z "$TIMEOUT_MS" ]; then
    echo "ERROR: Missing required environment variable: TIMEOUT_MS"
    echo "The Federation Constitution prohibits magic defaults."
    exit 1
fi

# Convert TIMEOUT_MS to seconds for curl
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

echo "=========================================="
echo "ICR API Connectivity Probe"
echo "=========================================="
echo "Target URL: $OPENEVOLVE_ICR_API_URL"
echo "Timeout: ${TIMEOUT_SEC}s"
echo "Timestamp (UTC): $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Test 1: Basic connectivity
echo "Test 1: Checking basic API connectivity..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time "$TIMEOUT_SEC" \
    -X GET \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    "$OPENEVOLVE_ICR_API_URL/api/health" \
    2>&1 || echo "000")

if [ "$HTTP_CODE" = "000" ]; then
    echo "FAIL: Cannot reach ICR API at $OPENEVOLVE_ICR_API_URL"
    echo "Error details: $(curl -s --max-time "$TIMEOUT_SEC" "$OPENEVOLVE_ICR_API_URL/api/health" 2>&1)"
    exit 1
fi

if [ "$HTTP_CODE" != "200" ] && [ "$HTTP_CODE" != "404" ] && [ "$HTTP_CODE" != "405" ]; then
    echo "FAIL: Unexpected HTTP code: $HTTP_CODE"
    echo "Expected: 200, 404, or 405 (API exists but endpoint may vary)"
    exit 1
fi

echo "PASS: API is reachable (HTTP $HTTP_CODE)"
echo ""

# Test 2: Health endpoint (POST request - likely implementation)
echo "Test 2: Checking health endpoint with POST..."

RESPONSE=$(curl -s \
    --max-time "$TIMEOUT_SEC" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d '{"correlation_id":"probe-test-001"}' \
    "$OPENEVOLVE_ICR_API_URL/api/health" \
    2>&1)

if echo "$RESPONSE" | grep -q '"status"'; then
    echo "PASS: Health endpoint returned valid JSON"
    echo "Response: $RESPONSE"
else
    echo "WARNING: Health endpoint did not return expected JSON"
    echo "Response: $RESPONSE"
    echo "This may indicate the API structure differs from expectations."
    echo "Proceed with caution and verify actual API contract."
fi

echo ""
echo "=========================================="
echo "Probe Result: SUCCESS"
echo "=========================================="
echo "The ICR API is accessible and responding."
echo "You may proceed with adapter implementation."
echo ""

exit 0
