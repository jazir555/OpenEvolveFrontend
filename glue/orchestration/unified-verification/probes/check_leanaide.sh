#!/bin/bash

###############################################################################
# PROBE: check_leanaide.sh
#
# Purpose: Verify LeanAide theorem prover is accessible and responding
# Following: Law of "Runtime Truth" - verify the API works before using it
#
# Usage: ./probes/check_leanaide.sh
#
# Exit codes:
#   0 - Probe successful (LeanAide is accessible)
#   1 - Probe failed (LeanAide is not accessible)
###############################################################################

set -e  # Fail fast

# Configuration from environment
LEANAIDE_URL="${LEANAIDE_URL:-http://localhost:8081}"
LEANAIDE_HEALTH_PATH="${LEANAIDE_HEALTH_PATH:-/health}"
TIMEOUT="${TIMEOUT:-5}"

echo "[PROBE] Checking LeanAide Theorem Prover at ${LEANAIDE_URL}"

# Test 1: Health check endpoint
echo "[PROBE] Test 1: Health check endpoint"
HEALTH_URL="${LEANAIDE_URL}${LEANAIDE_HEALTH_PATH}"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time "${TIMEOUT}" \
  "${HEALTH_URL}" || echo "000")

if [ "${HTTP_CODE}" = "200" ]; then
  echo "[PROBE] ✓ Health check passed (HTTP ${HTTP_CODE})"
else
  echo "[PROBE] ✗ Health check failed (HTTP ${HTTP_CODE})"
  exit 1
fi

# Test 2: Simple theorem proving
echo "[PROBE] Test 2: Simple theorem proving"

# Simple theorem: ∀ n : Nat, n + 0 = n
PAYLOAD='{
  "problem": {
    "type": "THEOREM_PROVING",
    "statement": "theorem add_zero (n : Nat) : n + 0 = n := by simp",
    "description": "Simple theorem: n + 0 = n"
  },
  "constraints": {
    "timeout": 5000,
    "precision": "medium"
  }
}'

RESPONSE=$(curl -s -X POST \
  --max-time "${TIMEOUT}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  "${LEANAIDE_URL}/verify" || echo "")

# Check if we got a valid response
if echo "${RESPONSE}" | jq -e '.verified != null' > /dev/null 2>&1; then
  VERIFIED=$(echo "${RESPONSE}" | jq -r '.verified')
  echo "[PROBE] ✓ Theorem proving query successful (verified=${VERIFIED})"
else
  echo "[PROBE] ✗ Theorem proving query failed - invalid response"
  echo "[PROBE] Response: ${RESPONSE}"
  exit 1
fi

# Test 3: Check for required fields in response
echo "[PROBE] Test 3: Response validation"

REQUIRED_FIELDS=("verified" "confidence" "output")
for field in "${REQUIRED_FIELDS[@]}"; do
  if echo "${RESPONSE}" | jq -e ".${field}" > /dev/null 2>&1; then
    echo "[PROBE] ✓ Field '${field}' present"
  else
    echo "[PROBE] ✗ Field '${field}' missing"
    exit 1
  fi
done

# Test 4: Check for proof generation
echo "[PROBE] Test 4: Proof generation"

if echo "${RESPONSE}" | jq -e '.proof' > /dev/null 2>&1; then
  PROOF_LENGTH=$(echo "${RESPONSE}" | jq -r '.proof | length')
  echo "[PROBE] ✓ Proof generated (length=${PROOF_LENGTH})"
else
  echo "[PROBE] ⚠ Proof field not present (may be optional)"
fi

echo "[PROBE] ========================================"
echo "[PROBE] ✓ All LeanAide probe tests passed"
echo "[PROBE] LeanAide is ready for integration"
echo "[PROBE] ========================================"

exit 0
