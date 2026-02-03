#!/bin/bash

###############################################################################
# Vector DB API Probe - Test Vector Database API Connectivity
#
# Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
# - Phase 1: The Probe (Discovery)
#
# This script tests the Vector DB API endpoint to verify:
# 1. The backend is accessible
# 2. The API responds correctly
# 3. Response times are acceptable
#
# Usage:
#   ./probes/check_api.sh
#
# Environment Variables:
#   VECTORDB_TYPE      - Backend type (qdrant|pinecone|chroma|pgvector)
#   VECTORDB_URL       - API URL (for Qdrant, Chroma)
#   VECTORDB_API_KEY   - API key (for Pinecone)
#   PINECONE_ENVIRONMENT - Pinecone environment
#   TIMEOUT_MS         - Request timeout in milliseconds
###############################################################################

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BACKEND_TYPE="${VECTORDB_TYPE:-qdrant}"
VECTORDB_URL="${VECTORDB_URL:-http://localhost:6333}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Test result functions
test_passed() {
    log_info "✓ PASSED: $1"
}

test_failed() {
    log_error "✗ FAILED: $1"
    exit 1
}

# Determine API endpoint based on backend type
case "$BACKEND_TYPE" in
    qdrant)
        API_ENDPOINT="${VECTORDB_URL}/"
        EXPECTED_STATUS=200
        ;;
    chroma)
        API_ENDPOINT="${VECTORDB_URL}/api/v1/heartbeat"
        EXPECTED_STATUS=200
        ;;
    pinecone)
        if [ -z "${VECTORDB_API_KEY:-}" ]; then
            test_failed "Pinecone requires VECTORDB_API_KEY"
        fi
        PINECONE_ENV="${PINECONE_ENVIRONMENT:-us-east1-aws}"
        API_ENDPOINT="https://controller.${PINECONE_ENV}.pinecone.io/databases"
        EXPECTED_STATUS=200
        ;;
    pgvector)
        log_warn "pgvector requires database connection testing - skip HTTP probe"
        log_info "Use check_collections.sh to test pgvector"
        exit 0
        ;;
    *)
        test_failed "Unknown backend type: $BACKEND_TYPE"
        ;;
esac

log_info "Vector DB API Probe"
log_info "===================="
log_info "Backend Type: $BACKEND_TYPE"
log_info "API Endpoint: $API_ENDPOINT"
log_info "Timeout: ${TIMEOUT_MS}ms"
log_info ""

# Test 1: Health Check
log_info "Test 1: Health Check"
log_info "Calling: $API_ENDPOINT"

RESPONSE=$(curl -s -w "\n%{http_code}" \
    --max-time "$TIMEOUT_SEC" \
    "$API_ENDPOINT" 2>&1) || {
    test_failed "Health check request failed (curl error)"
}

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

log_info "HTTP Status Code: $HTTP_CODE"
log_info "Response Body: $BODY"

if [ "$HTTP_CODE" -eq "$EXPECTED_STATUS" ]; then
    test_passed "Health check returned $HTTP_OK"
else
    test_failed "Health check returned $HTTP_CODE, expected $EXPECTED_STATUS"
fi

# Test 2: Response Time
log_info ""
log_info "Test 2: Response Time"

START_TIME=$(date +%s%3N)
curl -s -o /dev/null \
    --max-time "$TIMEOUT_SEC" \
    "$API_ENDPOINT" || {
    test_failed "Response time test failed (curl error)"
}
END_TIME=$(date +%s%3N)

RESPONSE_TIME=$((END_TIME - START_TIME))
log_info "Response Time: ${RESPONSE_TIME}ms"

if [ "$RESPONSE_TIME" -lt "$TIMEOUT_MS" ]; then
    test_passed "Response time ${RESPONSE_TIME}ms is within ${TIMEOUT_MS}ms threshold"
else
    test_failed "Response time ${RESPONSE_TIME}ms exceeds ${TIMEOUT_MS}ms threshold"
fi

# Test 3: Response Structure
log_info ""
log_info "Test 3: Response Structure Validation"

case "$BACKEND_TYPE" in
    qdrant)
        # Qdrant should return version information
        if echo "$BODY" | grep -q '"version"' || echo "$BODY" | grep -q '"title"'; then
            test_passed "Qdrant response structure valid"
        else
            test_failed "Qdrant response structure invalid"
        fi
        ;;
    chroma)
        # Chroma heartbeat should return simple response
        if [ -n "$BODY" ]; then
            test_passed "Chroma response structure valid"
        else
            test_failed "Chroma response structure invalid"
        fi
        ;;
    pinecone)
        # Pinecone should return list of databases
        if echo "$BODY" | grep -q '"databases"' || echo "$BODY" | grep -q '[]'; then
            test_passed "Pinecone response structure valid"
        else
            test_failed "Pinecone response structure invalid"
        fi
        ;;
esac

# Summary
log_info ""
log_info "===================="
log_info "All tests PASSED ✓"
log_info "Vector DB API is accessible and responding correctly"
log_info ""
log_info "Probe completed successfully at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

exit 0
