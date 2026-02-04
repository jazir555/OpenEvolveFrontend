#!/bin/bash
#
# Proof Knowledge Base - Validation Probe
#
# Tests proof validation and dependency checking.
# Follows the Federation Constitution: Law of Runtime Truth
#
# Usage: ./check_validation.sh
#

set -e

PROOF_KB_SERVICE="${PROOF_KB_SERVICE:-http://localhost:3000}"
CORRELATION_ID="$(uuidgen)"

echo "=== Proof Knowledge Base Validation Probe ==="
echo "Service: $PROOF_KB_SERVICE"
echo "Correlation ID: $CORRELATION_ID"
echo ""

# Setup: Store a valid proof and an invalid proof
echo "Setup: Storing test proofs..."

# Valid proof
curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "validation-test-valid",
    "theorem_id": "validation-theorem-1",
    "theorem": "Simple arithmetic identity",
    "proof": "theorem simple : 1 + 1 = 2 := by rfl",
    "system": "leanaide",
    "status": "valid",
    "confidence": 1.0,
    "tactics": ["rfl"],
    "dependencies": [],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }' > /dev/null

# Invalid proof (has a known error)
curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "validation-test-invalid",
    "theorem_id": "validation-theorem-2",
    "theorem": "False statement",
    "proof": "theorem false_proof : 1 = 0 := by rfl",
    "system": "leanaide",
    "status": "invalid",
    "confidence": 0.0,
    "tactics": ["rfl"],
    "dependencies": [],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }' > /dev/null

echo "✓ Test proofs stored"
echo ""

# Test 1: Validate a single proof
echo "Test 1: Validating a single proof..."
VALIDATION_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/validate" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "proofId": "validation-test-valid"
  }')

IS_VALID=$(echo "$VALIDATION_RESPONSE" | jq -r '.is_valid // false')

if [ "$IS_VALID" != "true" ]; then
  echo "❌ Expected valid proof to be validated as valid"
  echo "Response: $VALIDATION_RESPONSE"
  exit 1
fi

echo "✓ Valid proof correctly validated"
echo ""

# Test 2: Validate invalid proof
echo "Test 2: Validating an invalid proof..."
INVALID_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/validate" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "proofId": "validation-test-invalid"
  }')

IS_INVALID=$(echo "$INVALID_RESPONSE" | jq -r '.is_valid // true')

if [ "$IS_INVALID" != "false" ]; then
  echo "❌ Expected invalid proof to be validated as invalid"
  echo "Response: $INVALID_RESPONSE"
  exit 1
fi

ERROR_COUNT=$(echo "$INVALID_RESPONSE" | jq -r '.errors | length // 0')

if [ "$ERROR_COUNT" -lt 1 ]; then
  echo "❌ Expected validation errors for invalid proof"
  echo "Response: $INVALID_RESPONSE"
  exit 1
fi

echo "✓ Invalid proof correctly rejected with $ERROR_COUNT error(s)"
echo ""

# Test 3: Batch validation
echo "Test 3: Batch validating multiple proofs..."
BATCH_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/validate/batch" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "proofIds": ["validation-test-valid", "validation-test-invalid"]
  }')

BATCH_COUNT=$(echo "$BATCH_RESPONSE" | jq 'length // 0')

if [ "$BATCH_COUNT" -ne 2 ]; then
  echo "❌ Expected 2 validation results in batch"
  echo "Response: $BATCH_RESPONSE"
  exit 1
fi

echo "✓ Batch validation completed ($BATCH_COUNT proofs validated)"
echo ""

# Test 4: Check proof dependencies
echo "Test 4: Checking proof dependencies..."

# Store a proof with dependencies
curl -s -X POST \
  "$PROOF_KB_SERVICE/proofs" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "id": "validation-test-with-deps",
    "theorem_id": "validation-theorem-3",
    "theorem": "Theorem with dependencies",
    "proof": "theorem with_deps : 2 + 2 = 4 := by { rw [add_comm] }",
    "system": "leanaide",
    "status": "valid",
    "confidence": 1.0,
    "tactics": ["rw"],
    "dependencies": ["validation-test-valid"],
    "timestamp_utc": "'$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)'"
  }' > /dev/null

DEPS_RESPONSE=$(curl -s -X GET \
  "$PROOF_KB_SERVICE/proofs/validation-test-with-deps/dependencies" \
  -H "X-Correlation-ID: $CORRELATION_ID")

DEPS_COUNT=$(echo "$DEPS_RESPONSE" | jq 'length // 0')

if [ "$DEPS_COUNT" -lt 1 ]; then
  echo "❌ Expected to find dependencies"
  echo "Response: $DEPS_RESPONSE"
  exit 1
fi

echo "✓ Dependencies found ($DEPS_COUNT dependency(s))"
echo ""

# Test 5: Validate dependencies
echo "Test 5: Validating proof dependencies..."
DEPS_VALID_RESPONSE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/validate/dependencies" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "proofId": "validation-test-with-deps"
  }')

DEPS_VALID=$(echo "$DEPS_VALID_RESPONSE" | jq -r '.valid // false')

if [ "$DEPS_VALID" != "true" ]; then
  echo "❌ Expected dependencies to be valid"
  echo "Response: $DEPS_VALID_RESPONSE"
  exit 1
fi

echo "✓ Dependencies validated successfully"
echo ""

# Test 6: Get proof lineage
echo "Test 6: Getting proof lineage..."
LINEAGE_RESPONSE=$(curl -s -X GET \
  "$PROOF_KB_SERVICE/proofs/validation-test-with-deps/lineage?depth=2" \
  -H "X-Correlation-ID: $CORRELATION_ID")

ANCESTOR_COUNT=$(echo "$LINEAGE_RESPONSE" | jq -r '.ancestors | length // 0')

if [ "$ANCESTOR_COUNT" -lt 1 ]; then
  echo "❌ Expected to find ancestors in lineage"
  echo "Response: $LINEAGE_RESPONSE"
  exit 1
fi

echo "✓ Lineage retrieved ($ANCESTOR_COUNT ancestor(s))"
echo ""

# Test 7: Idempotency - validate same proof twice
echo "Test 7: Testing validation idempotency..."
FIRST_VALIDATE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/validate" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "proofId": "validation-test-valid"
  }')

SECOND_VALIDATE=$(curl -s -X POST \
  "$PROOF_KB_SERVICE/validate" \
  -H "Content-Type: application/json" \
  -H "X-Correlation-ID: $CORRELATION_ID" \
  -d '{
    "proofId": "validation-test-valid"
  }')

FIRST_RESULT=$(echo "$FIRST_VALIDATE" | jq -r '.is_valid')
SECOND_RESULT=$(echo "$SECOND_VALIDATE" | jq -r '.is_valid')

if [ "$FIRST_RESULT" != "$SECOND_RESULT" ]; then
  echo "❌ Validation not idempotent"
  echo "First: $FIRST_RESULT, Second: $SECOND_RESULT"
  exit 1
fi

echo "✓ Validation is idempotent"
echo ""

echo "=== All Validation Tests Passed ✓ ==="
echo ""
echo "Summary:"
echo "  - Valid proof validation: ✓"
echo "  - Invalid proof validation: ✓"
echo "  - Batch validation: ✓"
echo "  - Dependency tracking: ✓"
echo "  - Dependency validation: ✓"
echo "  - Lineage retrieval: ✓"
echo "  - Idempotency: ✓"
echo ""
echo "Correlation ID: $CORRELATION_ID"
