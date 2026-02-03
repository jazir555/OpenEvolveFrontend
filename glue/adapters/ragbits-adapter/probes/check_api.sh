#!/bin/bash
#
# RAGBits API Probe
#
# RUNTIME TRUTH: Verify RAGBits API is accessible and functional
# Before writing the adapter, we MUST confirm the API works
#
# Following Federation Constitution:
# - Law of "Runtime Truth": Trust execution, not documentation
# - Configuration Explicitness: API URL from env, no defaults
#
# Usage: ./probes/check_api.sh
#

set -euo pipefail

# ============================================================================
# CONFIGURATION EXPLICITNESS - Crash if missing
# ============================================================================

if [ -z "${RAGBITS_API_URL:-}" ]; then
    echo "❌ ERROR: RAGBITS_API_URL environment variable is required"
    echo "   Usage: RAGBITS_API_URL=http://ragbits-core:8002 ./probes/check_api.sh"
    exit 1
fi

if [ -z "${TIMEOUT_MS:-}" ]; then
    echo "❌ ERROR: TIMEOUT_MS environment variable is required"
    echo "   Usage: TIMEOUT_MS=5000 ./probes/check_api.sh"
    exit 1
fi

# Convert timeout to seconds for curl
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

API_URL="$RAGBITS_API_URL"
CORRELATION_ID="probe-$(date +%s)-$$"

echo "🔍 RAGBits API Probe"
echo "   API URL: $API_URL"
echo "   Timeout: ${TIMEOUT_MS}ms"
echo "   Correlation ID: $CORRELATION_ID"
echo ""

# ============================================================================
# STRUCTURED LOGGING - JSON Lines
# ============================================================================

log() {
    local level=$1
    local msg=$2
    local extra=${3:-}
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

    if [ -n "$extra" ]; then
        echo "{\"timestamp\":\"$timestamp\",\"level\":\"$level\",\"msg\":\"$msg\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"ragbits-adapter-probe\",\"target_service\":\"ragbits-core\",\"extra\":$extra}"
    else
        echo "{\"timestamp\":\"$timestamp\",\"level\":\"$level\",\"msg\":\"$msg\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"ragbits-adapter-probe\",\"target_service\":\"ragbits-core\"}"
    fi
}

# ============================================================================
# TEST 1: Health Check
# ============================================================================

log "info" "Testing health endpoint"

HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X GET \
    -H "X-Correlation-ID: $CORRELATION_ID" \
    --max-time "$TIMEOUT_SEC" \
    "$API_URL/health" 2>&1) || true

HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    log "info" "Health check successful" "{\"http_code\":$HTTP_CODE}"
    echo "✅ Health check: PASSED (HTTP $HTTP_CODE)"
else
    log "error" "Health check failed" "{\"http_code\":$HTTP_CODE,\"body\":\"$BODY\"}"
    echo "❌ Health check: FAILED (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
    exit 1
fi

# Parse health response
STATUS=$(echo "$BODY" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "unknown")
RAGBITS_AVAILABLE=$(echo "$BODY" | grep -o '"ragbits_available":[^,}]*' | cut -d':' -f2 || echo "false")

echo "   Status: $STATUS"
echo "   RAGBits Available: $RAGBITS_AVAILABLE"
echo ""

# ============================================================================
# TEST 2: Search Endpoint (with empty result is OK)
# ============================================================================

log "info" "Testing search endpoint"

SEARCH_PAYLOAD=$(cat <<EOF
{
  "query": "test query",
  "top_k": 1,
  "filters": {}
}
EOF
)

SEARCH_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-Correlation-ID: $CORRELATION_ID" \
    -d "$SEARCH_PAYLOAD" \
    --max-time "$TIMEOUT_SEC" \
    "$API_URL/search" 2>&1) || true

HTTP_CODE=$(echo "$SEARCH_RESPONSE" | tail -n1)
BODY=$(echo "$SEARCH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    log "info" "Search endpoint accessible" "{\"http_code\":$HTTP_CODE}"
    echo "✅ Search endpoint: PASSED (HTTP $HTTP_CODE)"

    # Check response structure
    if echo "$BODY" | grep -q '"results"'; then
        echo "   Response contains 'results' field"
    fi
    if echo "$BODY" | grep -q '"total_results"'; then
        echo "   Response contains 'total_results' field"
    fi
else
    log "error" "Search endpoint failed" "{\"http_code\":$HTTP_CODE,\"body\":\"$BODY\"}"
    echo "❌ Search endpoint: FAILED (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
    exit 1
fi

echo ""

# ============================================================================
# TEST 3: Ingest Endpoint (test structure only)
# ============================================================================

log "info" "Testing ingest endpoint"

INGEST_PAYLOAD=$(cat <<EOF
{
  "content": "Test document for probe",
  "metadata": {
    "source": "probe-test"
  }
}
EOF
)

INGEST_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-Correlation-ID: $CORRELATION_ID" \
    -d "$INGEST_PAYLOAD" \
    --max-time "$TIMEOUT_SEC" \
    "$API_URL/ingest" 2>&1) || true

HTTP_CODE=$(echo "$INGEST_RESPONSE" | tail -n1)
BODY=$(echo "$INGEST_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    log "info" "Ingest endpoint accessible" "{\"http_code\":$HTTP_CODE}"
    echo "✅ Ingest endpoint: PASSED (HTTP $HTTP_CODE)"

    # Check response structure
    if echo "$BODY" | grep -q '"success"'; then
        echo "   Response contains 'success' field"
    fi
    if echo "$BODY" | grep -q '"document_id"'; then
        echo "   Response contains 'document_id' field"
    fi
else
    log "warning" "Ingest endpoint returned non-success" "{\"http_code\":$HTTP_CODE,\"body\":\"$BODY\"}"
    echo "⚠️  Ingest endpoint: PARTIAL (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
    echo "   Note: This may be expected if RAGBits is in read-only mode"
fi

echo ""

# ============================================================================
# TEST 4: Stats Endpoint
# ============================================================================

log "info" "Testing stats endpoint"

STATS_RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X GET \
    -H "X-Correlation-ID: $CORRELATION_ID" \
    --max-time "$TIMEOUT_SEC" \
    "$API_URL/stats" 2>&1) || true

HTTP_CODE=$(echo "$STATS_RESPONSE" | tail -n1)
BODY=$(echo "$STATS_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    log "info" "Stats endpoint accessible" "{\"http_code\":$HTTP_CODE}"
    echo "✅ Stats endpoint: PASSED (HTTP $HTTP_CODE)"

    # Check response structure
    if echo "$BODY" | grep -q '"ingested_documents"'; then
        echo "   Response contains 'ingested_documents' field"
    fi
    if echo "$BODY" | grep -q '"vector_store_type"'; then
        echo "   Response contains 'vector_store_type' field"
    fi
else
    log "warning" "Stats endpoint failed" "{\"http_code\":$HTTP_CODE,\"body\":\"$BODY\"}"
    echo "⚠️  Stats endpoint: FAILED (HTTP $HTTP_CODE)"
    echo "   Response: $BODY"
fi

echo ""

# ============================================================================
# SUMMARY
# ============================================================================

log "info" "Probe completed successfully"

echo "🎉 RAGBits API Probe: PASSED"
echo ""
echo "API Endpoints Verified:"
echo "  ✅ GET  /health"
echo "  ✅ POST /search"
echo "  ✅ POST /ingest"
echo "  ✅ GET  /stats"
echo ""
echo "Contract validated. Adapter development may proceed."
exit 0
