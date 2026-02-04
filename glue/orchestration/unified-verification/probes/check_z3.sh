#!/bin/bash

###############################################################################
# PROBE: check_z3.sh
#
# Purpose: Verify Z3 SMT solver is accessible and responding
# Following: Law of "Runtime Truth" - verify the API works before using it
#
# Usage: ./probes/check_z3.sh
#
# Exit codes:
#   0 - Probe successful (Z3 is accessible)
#   1 - Probe failed (Z3 is not accessible)
###############################################################################

set -e  # Fail fast

# Configuration from environment
Z3_URL="${Z3_URL:-http://localhost:8080}"
Z3_HEALTH_PATH="${Z3_HEALTH_PATH:-/health}"
TIMEOUT="${TIMEOUT:-5}"

echo "[PROBE] Checking Z3 SMT Solver at ${Z3_URL}"

# Test 1: Health check endpoint
echo "[PROBE] Test 1: Health check endpoint"
HEALTH_URL="${Z3_URL}${Z3_HEALTH_PATH}"

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time "${TIMEOUT}" \
  "${HEALTH_URL}" || echo "000")

if [ "${HTTP_CODE}" = "200" ]; then
  echo "[PROBE] ✓ Health check passed (HTTP ${HTTP_CODE})"
else
  echo "[PROBE] ✗ Health check failed (HTTP ${HTTP_CODE})"
  exit 1
fi

# Test 2: Simple SMT query
echo "[PROBE] Test 2: Simple SMT constraint solving"

# Simple SMT-LIB problem: (declare-const x Int) (assert (> x 0)) (check-sat)
PAYLOAD='{
  "problem": {
    "type": "SMT_CONSTRAINTS",
    "statement": "(declare-const x Int) (assert (> x 0)) (check-sat)",
    "description": "Simple constraint: x > 0"
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
  "${Z3_URL}/verify" || echo "")

# Check if we got a valid response
if echo "${RESPONSE}" | jq -e '.verified != null' > /dev/null 2>&1; then
  VERIFIED=$(echo "${RESPONSE}" | jq -r '.verified')
  echo "[PROBE] ✓ SMT query successful (verified=${VERIFIED})"
else
  echo "[PROBE] ✗ SMT query failed - invalid response"
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

echo "[PROBE] ========================================"
echo "[PROBE] ✓ All Z3 probe tests passed"
echo "[PROBE] Z3 is ready for integration"
echo "[PROBE] ========================================"

exit 0
