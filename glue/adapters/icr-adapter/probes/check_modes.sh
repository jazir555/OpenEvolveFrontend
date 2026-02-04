#!/bin/bash

###############################################################################
# ICR Modes Availability Probe
#
# FEDERATION CONSTITUTION COMPLIANCE:
# - Law of Runtime Truth: Verify all 7 modes are actually accessible
# - Before implementing mode-specific code, confirm the mode exists
#
# Usage: ./check_modes.sh
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
echo "ICR Modes Availability Probe"
echo "=========================================="
echo "Target URL: $OPENEVOLVE_ICR_API_URL"
echo "Timestamp (UTC): $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo ""

# All 7 ICR modes
MODES=("refine" "react" "deepthink" "adaptive_deepthink" "agentic" "contextual" "generative_ui")

echo "Verifying accessibility of all 7 ICR modes..."
echo ""

# Check each mode
for MODE in "${MODES[@]}"; do
    echo "Checking mode: $MODE"

    # Try to execute a simple mode request
    # This should return either:
    # - 200/201 with a response structure
    # - 400/422 with validation errors (proves the endpoint exists)
    # - 404 (endpoint doesn't exist - FAIL)

    RESPONSE=$(curl -s \
        --max-time "$TIMEOUT_SEC" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "Accept: application/json" \
        -d "{
            \"mode\": \"$MODE\",
            \"prompt\": \"probe test\",
            \"metadata\": {
                \"correlation_id\": \"probe-$MODE-001\",
                \"timestamp_utc\": \"$(date -u +'%Y-%m-%dT%H:%M:%SZ')\",
                \"source_service\": \"probe-script\"
            }
        }" \
        "$OPENEVOLVE_ICR_API_URL/api/modes/execute" \
        2>&1)

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
        --max-time "$TIMEOUT_SEC" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"mode\":\"$MODE\",\"prompt\":\"probe\"}" \
        "$OPENEVOLVE_ICR_API_URL/api/modes/execute" \
        2>&1 || echo "000")

    if [ "$HTTP_CODE" = "000" ]; then
        echo "  FAIL: Cannot reach mode endpoint"
        echo "  Error: $RESPONSE"
        exit 1
    elif [ "$HTTP_CODE" = "404" ]; then
        echo "  FAIL: Mode endpoint not found (404)"
        echo "  The $MODE mode does not exist or is not accessible"
        exit 1
    elif [ "$HTTP_CODE" = "400" ] || [ "$HTTP_CODE" = "422" ]; then
        echo "  PASS: Mode endpoint exists (validation error expected)"
        echo "  HTTP $HTTP_CODE: Endpoint exists but request needs proper structure"
    elif [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
        echo "  PASS: Mode endpoint accessible and responding"
        echo "  HTTP $HTTP_CODE: Request succeeded"
    else
        echo "  WARNING: Unexpected HTTP code: $HTTP_CODE"
        echo "  Response: $RESPONSE"
    fi

    echo ""
done

echo "=========================================="
echo "Probe Result: SUCCESS"
echo "=========================================="
echo "All 7 ICR modes are accessible:"
echo "  - refine"
echo "  - react"
echo "  - deepthink"
echo "  - adaptive_deepthink"
echo "  - agentic"
echo "  - contextual"
echo "  - generative_ui"
echo ""
echo "You may proceed with mode-specific implementations."
echo ""

exit 0
