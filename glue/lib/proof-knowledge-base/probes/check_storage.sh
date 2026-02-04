#!/bin/bash
#
# Proof Knowledge Base - Storage Probe
#
# Tests the ability to store proofs in the knowledge base.
# Follows the Federation Constitution: Law of Runtime Truth
#
# Usage: ./check_storage.sh
#

set -e

PROOF_KB_SERVICE="${PROOF_KB_SERVICE:-http://localhost:3000}"
CORRELATION_ID="$(uuidgen)"

echo "=== Proof Knowledge Base Storage Probe ==="
echo "Service: $PROOF_KB_SERVICE"
echo "Correlation ID: $CORRELATION_ID"
echo ""

# Test 1: Store a theorem
echo "Test 1: Storing a theorem..."
THEOREM_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/theorems" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "test-theorem-1",
    "statement": "For all natural numbers n, n + 0 = n",
    "type": "theorem",
    "constraints": ["n ∈ Nat"],
    "dependencies": [],
    "created_at": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }')

THEOREM_ID=$(echo "$THEOREM_RESPONSE" | jq -r '.proof_id // .theorem_id // .id // empty')

if [ -z "$THEOREM_ID" ]; then
  echo "❌ Failed to store theorem"
  echo "Response: $THEOREM_RESPONSE"
  exit 1
fi

echo "✓ Theorem stored with ID: $THEOREM_ID"
echo ""

# Test 2: Store a proof
echo "Test 2: Storing a proof..."
PROOF_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "test-proof-1",
    "theorem_id": "test-theorem-1",
    "theorem": "For all natural numbers n, n + 0 = n",
    "proof": "theorem add_zero (n : Nat) : n + 0 = n := by { induction n with | zero => rfl | succ n ih => rw [add_succ, ih] }",
    "system": "leanaide",
    "status": "valid",
    "confidence": 1.0,
    "tactics": ["induction", "rfl", "rw"],
    "dependencies": [],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'",
    "correlation_id": "'$CORRELATION_ID'"
  }')

PROOF_ID=$(echo "$PROOF_RESPONSE" | jq -r '.proof_id // .id // empty')

if [ -z "$PROOF_ID" ]; then
  echo "❌ Failed to store proof"
  echo "Response: $PROOF_RESPONSE"
  exit 1
fi

echo "✓ Proof stored with ID: $PROOF_ID"
echo ""

# Test 3: Retrieve the proof
echo "Test 3: Retrieving the proof..."
RETRIEVE_RESPONSE=$(curl -s -X GET \
  "$PROOF_KB_SERVICE/proofs/$PROOF_ID" \
  -H "X-Correlation-ID: $CORRELATION_ID")

RETRIEVED_ID=$(echo "$RETRIEVE_RESPONSE" | jq -r '.id // empty')

if [ "$RETRIEVED_ID" != "$PROOF_ID" ]; then
  echo "❌ Failed to retrieve proof"
  echo "Response: $RETRIEVE_RESPONSE"
  exit 1
fi

echo "✓ Proof retrieved successfully"
echo ""

# Test 4: Check vector indexing
echo "Test 4: Checking vector indexing..."
METRICS_RESPONSE=$(curl -s -X GET \
  "$PROOF_KB_SERVICE/metrics" \
  -H "X-Correlation-ID: $CORRELATION_ID")

INDEXED_COUNT=$(echo "$METRICS_RESPONSE" | jq -r '.indexed_proofs // 0')

if [ "$INDEXED_COUNT" -lt 1 ]; then
  echo "❌ No proofs indexed in vector database"
  echo "Response: $METRICS_RESPONSE"
  exit 1
fi

echo "✓ Vector indexing working ($INDEXED_COUNT proofs indexed)"
echo ""

# Test 5: Idempotency check (Law of Idempotency)
echo "Test 5: Testing idempotency (store same proof twice)..."
DUPLICATE_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "test-proof-1",
    "theorem_id": "test-theorem-1",
    "theorem": "For all natural numbers n, n + 0 = n",
    "proof": "theorem add_zero (n : Nat) : n + 0 = n := by { induction n with | zero => rfl | succ n ih => rw [add_succ, ih] }",
    "system": "leanaide",
    "status": "valid",
    "confidence": 1.0,
    "tactics": ["induction", "rfl", "rw"],
    "dependencies": [],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'",
    "correlation_id": "'$CORRELATION_ID'"
  }')

DUPLICATE_SUCCESS=$(echo "$DUPLICATE_RESPONSE" | jq -r '.success // false')

if [ "$DUPLICATE_SUCCESS" != "true" ]; then
  echo "❌ Idempotency check failed"
  echo "Response: $DUPLICATE_RESPONSE"
  exit 1
fi

echo "✓ Idempotency working (can store same proof twice)"
echo ""

echo "=== All Storage Tests Passed ✓ ==="
echo ""
echo "Summary:"
echo "  - Theorem storage: ✓"
echo "  - Proof storage: ✓"
echo "  - Proof retrieval: ✓"
echo "  - Vector indexing: ✓"
echo "  - Idempotency: ✓"
echo ""
echo "Correlation ID: $CORRELATION_ID"
