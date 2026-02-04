#!/bin/bash

###############################################################################
# ICR Refinement Operation Probe
#
# FEDERATION CONSTITUTION COMPLIANCE:
# - Law of Runtime Truth: Test an actual refinement operation
# - Verify the full request/response cycle works end-to-end
#
# Usage: ./check_refinement.sh
###############################################################################

set -e  # Exit on error

# Configuration from environment (NO DEFAULTS)
if [ -z "$OPENEVOLVE_ICR_API_URL" ]; then
    echo "ERROR: Missing required environment variable: OPENEVOLVE_ICR_API_URL"
    exit 1
fi

if [ -z "$TIMEOUT_MS" ]; then
    echo "ERROR: Missing required environment variable: TIMEOUT_MS"
    exit 1
fi

TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

echo "=========================================="
echo "ICR Refinement Operation Probe"
echo "=========================================="
echo "Target URL: $OPENEVOLVE_ICR_API_URL"
echo "Timestamp (UTC): $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo ""

# Generate test correlation ID
CORRELATION_ID="probe-refinement-$(date +%s)"

echo "Testing Refine mode with a simple request..."
echo ""

# Prepare request payload
REQUEST=$(cat <<EOF
{
  "mode": "refine",
  "prompt": "Create a simple hello world function in TypeScript",
  "options": {
    "temperature": 0.7,
    "evolution_mode": "quality",
    "refinement_stages": 1
  },
  "metadata": {
    "correlation_id": "$CORRELATION_ID",
    "timestamp_utc": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "source_service": "probe-script"
  }
}
EOF
)

echo "Request payload:"
echo "$REQUEST" | jq '.' 2>/dev/null || echo "$REQUEST"
echo ""

# Execute request
echo "Executing refinement request..."
RESPONSE=$(curl -s \
    --max-time "$TIMEOUT_SEC" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -H "X-Correlation-ID: $CORRELATION_ID" \
    -d "$REQUEST" \
    "$OPENEVOLVE_ICR_API_URL/api/modes/execute" \
    2>&1)

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time "$TIMEOUT_SEC" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-Correlation-ID: $CORRELATION_ID" \
    -d "$REQUEST" \
    "$OPENEVOLVE_ICR_API_URL/api/modes/execute" \
    2>&1 || echo "000")

echo "HTTP Response Code: $HTTP_CODE"
echo ""

if [ "$HTTP_CODE" = "000" ]; then
    echo "FAIL: Request failed or timed out"
    echo "Error: $RESPONSE"
    exit 1
elif [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    echo "PASS: Request succeeded"

    # Validate response structure
    if echo "$RESPONSE" | jq -e '.mode' >/dev/null 2>&1; then
        echo "PASS: Response contains 'mode' field"
    else
        echo "WARNING: Response missing 'mode' field"
    fi

    if echo "$RESPONSE" | jq -e '.result' >/dev/null 2>&1; then
        echo "PASS: Response contains 'result' field"
    else
        echo "WARNING: Response missing 'result' field"
    fi

    if echo "$RESPONSE" | jq -e '.metadata' >/dev/null 2>&1; then
        echo "PASS: Response contains 'metadata' field"
    else
        echo "WARNING: Response missing 'metadata' field"
    fi

    echo ""
    echo "Response preview:"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"

elif [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "422" ]; then
    echo "PARTIAL: Endpoint exists but request validation failed"
    echo "This is expected if the API structure differs from our schema."
    echo ""
    echo "Error response:"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    echo ""
    echo "ACTION REQUIRED: Update canonical schema to match actual API"
    exit 1
else
    echo "FAIL: Unexpected HTTP code: $HTTP_CODE"
    echo "Response: $RESPONSE"
    exit 1
fi

echo ""
echo "=========================================="
echo "Probe Result: SUCCESS"
echo "=========================================="
echo "The ICR Refine mode is operational."
echo "Correlation ID: $CORRELATION_ID"
echo ""

exit 0
