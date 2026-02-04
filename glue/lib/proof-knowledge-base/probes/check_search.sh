#!/bin/bash
#
# Proof Knowledge Base - Search Probe
#
# Tests semantic search capabilities for finding similar proofs.
# Follows the Federation Constitution: Law of Runtime Truth
#
# Usage: ./check_search.sh
#

set -e

PROOF_KB_SERVICE="${PROOF_KB_SERVICE:-http://localhost:3000}"
CORRELATION_ID="$(uuidgen)"

echo "=== Proof Knowledge Base Search Probe ==="
echo "Service: $PROOF_KB_SERVICE"
echo "Correlation ID: $CORRELATION_ID"
echo ""

# First, store some test proofs if they don't exist
echo "Setup: Storing test proofs..."
curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "search-test-proof-1",
    "theorem_id": "search-test-theorem-1",
    "theorem": "Addition is commutative for natural numbers",
    "proof": "theorem add_comm (m n : Nat) : m + n = n + m := by ...",
    "system": "leanaide",
    "status": "valid",
    "tactics": ["induction", "rw"],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }' > /dev/null

curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "search-test-proof-2",
    "theorem_id": "search-test-theorem-2",
    "theorem": "Multiplication distributes over addition",
    "proof": "theorem mul_add (a b c : Nat) : a * (b + c) = a * b + a * c := by ...",
    "system": "z3",
    "status": "valid",
    "tactics": ["simplify", "arith"],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }' > /dev/null

sleep 2 # Allow time for indexing
echo "✓ Test proofs stored"
echo ""

# Test 1: Search by theorem similarity
echo "Test 1: Searching by theorem similarity..."
SEARCH_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/search/similar" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "theorem": {
      "id": "query-theorem-1",
      "statement": "Prove that addition is commutative",
      "type": "theorem",
      "constraints": [],
      "dependencies": [],
      "created_at": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
    },
    "maxResults": 5
  }')

RESULT_COUNT=$(echo "$SEARCH_RESPONSE" | jq 'length // 0')

if [ "$RESULT_COUNT" -lt 1 ]; then
  echo "❌ No search results found"
  echo "Response: $SEARCH_RESPONSE"
  exit 1
fi

echo "✓ Found $RESULT_COUNT similar proofs"
echo ""

# Test 2: Search by natural language content
echo "Test 2: Searching by natural language content..."
CONTENT_SEARCH_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/search/content" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "query": "commutative property of addition",
    "maxResults": 5
  }')

CONTENT_RESULT_COUNT=$(echo "$CONTENT_SEARCH_RESPONSE" | jq 'length // 0')

if [ "$CONTENT_RESULT_COUNT" -lt 1 ]; then
  echo "❌ No content search results found"
  echo "Response: $CONTENT_SEARCH_RESPONSE"
  exit 1
fi

echo "✓ Found $CONTENT_RESULT_COUNT proofs matching content query"
echo ""

# Test 3: Check similarity scores
echo "Test 3: Checking similarity scores..."
FIRST_SCORE=$(echo "$SEARCH_RESPONSE" | jq -r '.[0].similarity_score // empty')

if [ -z "$FIRST_SCORE" ]; then
  echo "❌ No similarity score in response"
  echo "Response: $SEARCH_RESPONSE"
  exit 1
fi

# Check if score is between 0 and 1
if (( $(echo "$FIRST_SCORE < 0" | bc -l) )) || (( $(echo "$FIRST_SCORE > 1" | bc -l) )); then
  echo "❌ Similarity score out of range: $FIRST_SCORE"
  exit 1
fi

echo "✓ Similarity scores valid (first result: $FIRST_SCORE)"
echo ""

# Test 4: Filter by system
echo "Test 4: Filtering by proof system..."
FILTERED_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/search/similar" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "theorem": {
      "id": "query-theorem-2",
      "statement": "Arithmetic properties",
      "type": "theorem",
      "constraints": [],
      "dependencies": [],
      "created_at": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
    },
    "maxResults": 10,
    "filter": {
      "system": "leanaide"
    }
  }')

FILTERED_COUNT=$(echo "$FILTERED_RESPONSE" | jq 'length // 0')

# Verify all results are from leanaide
LEAN_COUNT=$(echo "$FILTERED_RESPONSE" | jq '[.[] | select(.proof.system == "leanaide")] | length')

if [ "$FILTERED_COUNT" != "$LEAN_COUNT" ]; then
  echo "❌ Filter not working correctly"
  echo "Expected $FILTERED_COUNT results from leanaide, got $LEAN_COUNT"
  exit 1
fi

echo "✓ Filtering working correctly ($LEAN_COUNT leanaide proofs)"
echo ""

# Test 5: Search with minimum score threshold
echo "Test 5: Testing minimum score threshold..."
HIGH_SCORE_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/search/similar" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "theorem": {
      "id": "query-theorem-3",
      "statement": "Natural number arithmetic",
      "type": "theorem",
      "constraints": [],
      "dependencies": [],
      "created_at": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
    },
    "maxResults": 10,
    "minScore": 0.5
  }')

# Check if all results meet the threshold
BELOW_THRESHOLD=$(echo "$HIGH_SCORE_RESPONSE" | jq '[.[] | select(.similarity_score < 0.5)] | length')

if [ "$BELOW_THRESHOLD" -gt 0 ]; then
  echo "❌ Found $BELOW_THRESHOLD results below threshold"
  echo "Response: $HIGH_SCORE_RESPONSE"
  exit 1
fi

echo "✓ Minimum score threshold working"
echo ""

echo "=== All Search Tests Passed ✓ ==="
echo ""
echo "Summary:"
echo "  - Theorem similarity search: ✓"
echo "  - Natural language content search: ✓"
echo "  - Similarity scores: ✓"
echo "  - System filtering: ✓"
echo "  - Score threshold filtering: ✓"
echo ""
echo "Correlation ID: $CORRELATION_ID"
