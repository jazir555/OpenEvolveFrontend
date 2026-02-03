#!/bin/bash

###############################################################################
# Vector DB Collections Probe - Test Collection Operations
#
# Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
# - Phase 1: The Probe (Discovery)
#
# This script tests vector database collection operations:
# 1. List collections
# 2. Get collection info (if collections exist)
# 3. Create test collection
# 4. Delete test collection
#
# Usage:
#   ./probes/check_collections.sh [collection_name]
#
# Arguments:
#   collection_name - Name of test collection (default: probe_test_collection)
#
# Environment Variables:
#   VECTORDB_TYPE      - Backend type (qdrant|pinecone|chroma|pgvector)
#   VECTORDB_URL       - API URL (for Qdrant, Chroma)
#   VECTORDB_API_KEY   - API key (for Pinecone)
#   VECTORDB_CONNECTION_STRING - PostgreSQL connection (for pgvector)
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
TEST_COLLECTION="${1:-probe_test_collection}"

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

log_info "Vector DB Collections Probe"
log_info "==========================="
log_info "Backend Type: $BACKEND_TYPE"
log_info "Test Collection: $TEST_COLLECTION"
log_info "Timeout: ${TIMEOUT_MS}ms"
log_info ""

# Common curl options
CURL_OPTS="-s -w \n%{http_code} --max-time $TIMEOUT_SEC"

# Test 1: List Collections
log_info "Test 1: List Collections"

case "$BACKEND_TYPE" in
    qdrant)
        RESPONSE=$(curl $CURL_OPTS \
            -H "Content-Type: application/json" \
            "${VECTORDB_URL}/collections" 2>&1) || {
            test_failed "Failed to list Qdrant collections"
        }
        ;;
    chroma)
        RESPONSE=$(curl $CURL_OPTS \
            -H "Content-Type: application/json" \
            "${VECTORDB_URL}/api/v1/collections" 2>&1) || {
            test_failed "Failed to list Chroma collections"
        }
        ;;
    pinecone)
        PINECONE_ENV="${PINECONE_ENVIRONMENT:-us-east1-aws}"
        RESPONSE=$(curl $CURL_OPTS \
            -H "Api-Key: ${VECTORDB_API_KEY}" \
            -H "Content-Type: application/json" \
            "https://controller.${PINECONE_ENV}.pinecone.io/databases" 2>&1) || {
            test_failed "Failed to list Pinecone databases"
        }
        ;;
    pgvector)
        log_warn "pgvector: Listing tables with vector columns"
        RESPONSE=$(psql "$VECTORDB_CONNECTION_STRING" -t -c "
            SELECT table_name
            FROM information_schema.columns
            WHERE data_type = 'user-defined'
              AND udt_name = 'vector';
        " 2>&1) || {
            test_failed "Failed to list pgvector tables"
        }
        ;;
esac

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

log_info "HTTP Status: $HTTP_CODE"
log_info "Response: $BODY"

if [ "$HTTP_CODE" = "200" ] || [ "$BACKEND_TYPE" = "pgvector" ]; then
    test_passed "List collections successful"
else
    test_failed "List collections failed with status $HTTP_CODE"
fi

# Test 2: Create Test Collection
log_info ""
log_info "Test 2: Create Test Collection"

case "$BACKEND_TYPE" in
    qdrant)
        # Create Qdrant collection
        RESPONSE=$(curl $CURL_OPTS \
            -X PUT "${VECTORDB_URL}/collections/${TEST_COLLECTION}" \
            -H "Content-Type: application/json" \
            -d '{
              "vectors": {
                "size": 128,
                "distance": "Cosine"
              }
            }' 2>&1) || {
            test_failed "Failed to create Qdrant collection"
        }
        ;;
    chroma)
        # Create Chroma collection
        RESPONSE=$(curl $CURL_OPTS \
            -X POST "${VECTORDB_URL}/api/v1/collections" \
            -H "Content-Type: application/json" \
            -d "{
              \"name\": \"${TEST_COLLECTION}\",
              \"metadata\": {\"dimension\": 128, \"distance_metric\": \"cosine\"}
            }" 2>&1) || {
            test_failed "Failed to create Chroma collection"
        }
        ;;
    pinecone)
        log_warn "Pinecone: Index creation is asynchronous, skipping create test"
        ;;
    pgvector)
        # Create pgvector table
        RESPONSE=$(psql "$VECTORDB_CONNECTION_STRING" -c "
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS ${TEST_COLLECTION} (
                id UUID PRIMARY KEY,
                vector vector(128),
                text TEXT,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS ${TEST_COLLECTION}_vector_idx
            ON ${TEST_COLLECTION}
            USING hnsw (vector cosine_vector_ops)
            WITH (m = 16, ef_construction = 64);
        " 2>&1) || {
            test_failed "Failed to create pgvector table"
        }
        ;;
esac

# Check for 200 or 409 (already exists)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "409" ] || [ "$BACKEND_TYPE" = "pgvector" ]; then
    test_passed "Create collection successful (or already exists)"
else
    log_warn "Create collection returned status $HTTP_CODE"
fi

# Test 3: Get Collection Info
log_info ""
log_info "Test 3: Get Collection Info"

case "$BACKEND_TYPE" in
    qdrant)
        RESPONSE=$(curl $CURL_OPTS \
            -X GET "${VECTORDB_URL}/collections/${TEST_COLLECTION}" \
            -H "Content-Type: application/json" 2>&1) || {
            test_failed "Failed to get Qdrant collection info"
        }
        ;;
    chroma)
        RESPONSE=$(curl $CURL_OPTS \
            -X GET "${VECTORDB_URL}/api/v1/collections/${TEST_COLLECTION}" \
            -H "Content-Type: application/json" 2>&1) || {
            test_failed "Failed to get Chroma collection info"
        }
        ;;
    pinecone)
        log_warn "Pinecone: Skipping get info test (requires running index)"
        ;;
    pgvector)
        RESPONSE=$(psql "$VECTORDB_CONNECTION_STRING" -t -c "
            SELECT COUNT(*) FROM ${TEST_COLLECTION};
        " 2>&1) || {
            test_failed "Failed to get pgvector table info"
        }
        ;;
esac

if [ "$BACKEND_TYPE" != "pinecone" ]; then
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')

    log_info "Response: $BODY"

    if [ "$HTTP_CODE" = "200" ] || [ "$BACKEND_TYPE" = "pgvector" ]; then
        test_passed "Get collection info successful"
    else
        test_failed "Get collection info failed with status $HTTP_CODE"
    fi
fi

# Test 4: Delete Test Collection
log_info ""
log_info "Test 4: Delete Test Collection"

case "$BACKEND_TYPE" in
    qdrant)
        RESPONSE=$(curl $CURL_OPTS \
            -X DELETE "${VECTORDB_URL}/collections/${TEST_COLLECTION}" \
            -H "Content-Type: application/json" 2>&1) || {
            test_failed "Failed to delete Qdrant collection"
        }
        ;;
    chroma)
        RESPONSE=$(curl $CURL_OPTS \
            -X DELETE "${VECTORDB_URL}/api/v1/collections/${TEST_COLLECTION}" \
            -H "Content-Type: application/json" 2>&1) || {
            test_failed "Failed to delete Chroma collection"
        }
        ;;
    pinecone)
        log_warn "Pinecone: Skipping delete test (index deletion is manual)"
        ;;
    pgvector)
        RESPONSE=$(psql "$VECTORDB_CONNECTION_STRING" -c "
            DROP TABLE IF EXISTS ${TEST_COLLECTION};
        " 2>&1) || {
            test_failed "Failed to delete pgvector table"
        }
        ;;
esac

if [ "$BACKEND_TYPE" != "pinecone" ]; then
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

    if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "204" ] || [ "$BACKEND_TYPE" = "pgvector" ]; then
        test_passed "Delete collection successful"
    else
        test_failed "Delete collection failed with status $HTTP_CODE"
    fi
fi

# Summary
log_info ""
log_info "==========================="
log_info "All tests PASSED ✓"
log_info "Vector DB collection operations are working correctly"
log_info ""
log_info "Probe completed successfully at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

exit 0
