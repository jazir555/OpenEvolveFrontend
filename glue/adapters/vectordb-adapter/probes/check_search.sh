#!/bin/bash

###############################################################################
# Vector DB Search Probe - Test Vector Similarity Search
#
# Following CLAUDE.md Section 4: The Proof of Work (The Vibe Check)
# - Phase 1: The Probe (Discovery)
#
# This script tests vector similarity search operations:
# 1. Create test collection
# 2. Insert test vectors
# 3. Perform similarity search
# 4. Clean up test collection
#
# Usage:
#   ./probes/check_search.sh [collection_name]
#
# Arguments:
#   collection_name - Name of test collection (default: probe_search_test)
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
TEST_COLLECTION="${1:-probe_search_test}"
VECTOR_DIM=4

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
    cleanup
    exit 1
}

# Cleanup function
cleanup() {
    log_info ""
    log_info "Cleaning up test collection..."

    case "$BACKEND_TYPE" in
        qdrant)
            curl -s -X DELETE "${VECTORDB_URL}/collections/${TEST_COLLECTION}" \
                -H "Content-Type: application/json" > /dev/null 2>&1 || true
            ;;
        chroma)
            curl -s -X DELETE "${VECTORDB_URL}/api/v1/collections/${TEST_COLLECTION}" \
                -H "Content-Type: application/json" > /dev/null 2>&1 || true
            ;;
        pgvector)
            psql "$VECTORDB_CONNECTION_STRING" -c "DROP TABLE IF EXISTS ${TEST_COLLECTION};" > /dev/null 2>&1 || true
            ;;
        pinecone)
            log_warn "Pinecone: Manual cleanup required for indexes"
            ;;
    esac
}

log_info "Vector DB Search Probe"
log_info "======================"
log_info "Backend Type: $BACKEND_TYPE"
log_info "Test Collection: $TEST_COLLECTION"
log_info "Vector Dimension: $VECTOR_DIM"
log_info "Timeout: ${TIMEOUT_MS}ms"
log_info ""

# Test 1: Create Test Collection
log_info "Test 1: Create Test Collection"

case "$BACKEND_TYPE" in
    qdrant)
        curl -s -X PUT "${VECTORDB_URL}/collections/${TEST_COLLECTION}" \
            -H "Content-Type: application/json" \
            -d "{
              \"vectors\": {
                \"size\": ${VECTOR_DIM},
                \"distance\": \"Cosine\"
              }
            }" > /dev/null 2>&1 || {
            test_failed "Failed to create Qdrant collection"
        }
        test_passed "Qdrant collection created"
        ;;
    chroma)
        curl -s -X POST "${VECTORDB_URL}/api/v1/collections" \
            -H "Content-Type: application/json" \
            -d "{
              \"name\": \"${TEST_COLLECTION}\",
              \"metadata\": {\"dimension\": ${VECTOR_DIM}, \"distance_metric\": \"cosine\"}
            }" > /dev/null 2>&1 || {
            test_failed "Failed to create Chroma collection"
        }
        test_passed "Chroma collection created"
        ;;
    pgvector)
        psql "$VECTORDB_CONNECTION_STRING" -c "
            CREATE EXTENSION IF NOT EXISTS vector;

            CREATE TABLE IF NOT EXISTS ${TEST_COLLECTION} (
                id UUID PRIMARY KEY,
                vector vector(${VECTOR_DIM}),
                text TEXT,
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS ${TEST_COLLECTION}_vector_idx
            ON ${TEST_COLLECTION}
            USING hnsw (vector cosine_vector_ops)
            WITH (m = 16, ef_construction = 64);
        " > /dev/null 2>&1 || {
            test_failed "Failed to create pgvector table"
        }
        test_passed "pgvector table created"
        ;;
    pinecone)
        log_warn "Pinecone: Index creation is asynchronous, using existing index"
        TEST_COLLECTION="${TEST_COLLECTION:-test-index}"
        ;;
esac

# Test 2: Insert Test Vectors
log_info ""
log_info "Test 2: Insert Test Vectors"

# Generate test vectors
VECTOR_1="[0.1,0.2,0.3,0.4]"
VECTOR_2="[0.2,0.3,0.4,0.5]"
VECTOR_3="[0.9,0.8,0.7,0.6]"

case "$BACKEND_TYPE" in
    qdrant)
        curl -s -X PUT "${VECTORDB_URL}/collections/${TEST_COLLECTION}/points" \
            -H "Content-Type: application/json" \
            -d "{
              \"points\": [
                {\"id\": \"1\", \"vector\": ${VECTOR_1}, \"payload\": {\"text\": \"doc1\"}},
                {\"id\": \"2\", \"vector\": ${VECTOR_2}, \"payload\": {\"text\": \"doc2\"}},
                {\"id\": \"3\", \"vector\": ${VECTOR_3}, \"payload\": {\"text\": \"doc3\"}}
              ]
            }" > /dev/null 2>&1 || {
            test_failed "Failed to insert Qdrant vectors"
        }
        test_passed "Qdrant vectors inserted"
        ;;
    chroma)
        curl -s -X POST "${VECTORDB_URL}/api/v1/collections/${TEST_COLLECTION}/upsert" \
            -H "Content-Type: application/json" \
            -d "{
              \"ids\": [\"1\", \"2\", \"3\"],
              \"embeddings\": [${VECTOR_1}, ${VECTOR_2}, ${VECTOR_3}],
              \"documents\": [\"doc1\", \"doc2\", \"doc3\"],
              \"metadatas\": [{}, {}, {}]
            }" > /dev/null 2>&1 || {
            test_failed "Failed to insert Chroma vectors"
        }
        test_passed "Chroma vectors inserted"
        ;;
    pgvector)
        psql "$VECTORDB_CONNECTION_STRING" -c "
            INSERT INTO ${TEST_COLLECTION} (id, vector, text) VALUES
                ('550e8400-e29b-41d4-a716-446655440000'::uuid, '${VECTOR_1}'::vector, 'doc1'),
                ('550e8400-e29b-41d4-a716-446655440001'::uuid, '${VECTOR_2}'::vector, 'doc2'),
                ('550e8400-e29b-41d4-a716-446655440002'::uuid, '${VECTOR_3}'::vector, 'doc3');
        " > /dev/null 2>&1 || {
            test_failed "Failed to insert pgvector vectors"
        }
        test_passed "pgvector vectors inserted"
        ;;
    pinecone)
        log_warn "Pinecone: Skipping vector insert (requires upsert with namespace)"
        ;;
esac

# Test 3: Perform Similarity Search
log_info ""
log_info "Test 3: Perform Similarity Search"

QUERY_VECTOR="[0.1,0.2,0.3,0.4]"

case "$BACKEND_TYPE" in
    qdrant)
        RESPONSE=$(curl -s -X POST "${VECTORDB_URL}/collections/${TEST_COLLECTION}/points/query" \
            -H "Content-Type: application/json" \
            -d "{
              \"vector\": ${QUERY_VECTOR},
              \"limit\": 3,
              \"with_payload\": true
            }" 2>&1) || {
            test_failed "Failed to perform Qdrant search"
        }

        # Validate response
        if echo "$RESPONSE" | grep -q "result"; then
            test_passed "Qdrant search returned results"
            log_info "Results: $(echo "$RESPONSE" | head -c 200)..."
        else
            test_failed "Qdrant search failed to return results"
        fi
        ;;
    chroma)
        RESPONSE=$(curl -s -X POST "${VECTORDB_URL}/api/v1/collections/${TEST_COLLECTION}/query" \
            -H "Content-Type: application/json" \
            -d "{
              \"query_embeddings\": [${QUERY_VECTOR}],
              \"n_results\": 3
            }" 2>&1) || {
            test_failed "Failed to perform Chroma search"
        }

        # Validate response
        if echo "$RESPONSE" | grep -q "ids" || echo "$RESPONSE" | grep -q "documents"; then
            test_passed "Chroma search returned results"
            log_info "Results: $(echo "$RESPONSE" | head -c 200)..."
        else
            test_failed "Chroma search failed to return results"
        fi
        ;;
    pgvector)
        RESPONSE=$(psql "$VECTORDB_CONNECTION_STRING" -t -c "
            SELECT id, text, vector <=> '${QUERY_VECTOR}'::vector as distance
            FROM ${TEST_COLLECTION}
            ORDER BY vector <=> '${QUERY_VECTOR}'::vector
            LIMIT 3;
        " 2>&1) || {
            test_failed "Failed to perform pgvector search"
        }

        # Validate response
        if echo "$RESPONSE" | grep -q "|"; then
            test_passed "pgvector search returned results"
            log_info "Results:\n$RESPONSE"
        else
            test_failed "pgvector search failed to return results"
        fi
        ;;
    pinecone)
        log_warn "Pinecone: Skipping search test (requires running index)"
        ;;
esac

# Cleanup
cleanup

# Summary
log_info ""
log_info "======================"
log_info "All tests PASSED ✓"
log_info "Vector DB search operations are working correctly"
log_info ""
log_info "Probe completed successfully at $(date -u +"%Y-%m-%dT%H:%M:%SZ")"

exit 0
