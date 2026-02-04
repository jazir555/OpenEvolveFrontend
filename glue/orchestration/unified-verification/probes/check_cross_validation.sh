#!/bin/bash

###############################################################################
# PROBE: check_cross_validation.sh
#
# Purpose: Verify cross-validation workflow between Z3 and LeanAide
# Following: Law of "Runtime Truth" - verify integration works end-to-end
#
# Usage: ./probes/check_cross_validation.sh
#
# Exit codes:
#   0 - Probe successful (Cross-validation works)
#   1 - Probe failed (Integration issues detected)
###############################################################################

set -e  # Fail fast

# Configuration from environment
Z3_URL="${Z3_URL:-http://localhost:8080}"
LEANAIDE_URL="${LEANAIDE_URL:-http://localhost:8081}"
TIMEOUT="${TIMEOUT:-10}"

echo "[PROBE] Checking cross-validation workflow"
echo "[PROBE] Z3: ${Z3_URL}"
echo "[PROBE] LeanAide: ${LEANAIDE_URL}"

# Test 1: Both systems are healthy
echo "[PROBE] Test 1: System health checks"

Z3_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time "${TIMEOUT}" \
  "${Z3_URL}/health" || echo "000")

LEANAIDE_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" \
  --max-time "${TIMEOUT}" \
  "${LEANAIDE_URL}/health" || echo "000")

if [ "${Z3_HEALTH}" = "200" ]; then
  echo "[PROBE] ✓ Z3 is healthy"
else
  echo "[PROBE] ✗ Z3 is unhealthy (HTTP ${Z3_HEALTH})"
  exit 1
fi

if [ "${LEANAIDE_HEALTH}" = "200" ]; then
  echo "[PROBE] ✓ LeanAide is healthy"
else
  echo "[PROBE] ✗ LeanAide is unhealthy (HTTP ${LEANAIDE_HEALTH})"
  exit 1
fi

# Test 2: Parallel verification request
echo "[PROBE] Test 2: Parallel verification (same problem to both systems)"

# Problem suitable for both systems: simple arithmetic verification
PAYLOAD='{
  "problem": {
    "type": "FORMAL_VERIFICATION",
    "statement": "forall x: int, x + 0 = x",
    "description": "Additive identity verification"
  },
  "constraints": {
    "timeout": 5000,
    "precision": "medium",
    "allowedSystems": ["both"]
  },
  "strategy": "parallel"
}'

# Query Z3
echo "[PROBE] Querying Z3..."
Z3_RESPONSE=$(curl -s -X POST \
  --max-time "${TIMEOUT}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  "${Z3_URL}/verify" || echo "")

if echo "${Z3_RESPONSE}" | jq -e '.verified != null' > /dev/null 2>&1; then
  Z3_VERIFIED=$(echo "${Z3_RESPONSE}" | jq -r '.verified')
  Z3_CONFIDENCE=$(echo "${Z3_RESPONSE}" | jq -r '.confidence')
  echo "[PROBE] ✓ Z3 responded (verified=${Z3_VERIFIED}, confidence=${Z3_CONFIDENCE})"
else
  echo "[PROBE] ✗ Z3 response invalid"
  exit 1
fi

# Query LeanAide
echo "[PROBE] Querying LeanAide..."
LEANAIDE_RESPONSE=$(curl -s -X POST \
  --max-time "${TIMEOUT}" \
  -H "Content-Type: application/json" \
  -d "${PAYLOAD}" \
  "${LEANAIDE_URL}/verify" || echo "")

if echo "${LEANAIDE_RESPONSE}" | jq -e '.verified != null' > /dev/null 2>&1; then
  LEANAIDE_VERIFIED=$(echo "${LEANAIDE_RESPONSE}" | jq -r '.verified')
  LEANAIDE_CONFIDENCE=$(echo "${LEANAIDE_RESPONSE}" | jq -r '.confidence')
  echo "[PROBE] ✓ LeanAide responded (verified=${LEANAIDE_VERIFIED}, confidence=${LEANAIDE_CONFIDENCE})"
else
  echo "[PROBE] ✗ LeanAide response invalid"
  exit 1
fi

# Test 3: Cross-validation logic
echo "[PROBE] Test 3: Cross-validation agreement check"

# Check verification outcome agreement
if [ "${Z3_VERIFIED}" = "${LEANAIDE_VERIFIED}" ]; then
  echo "[PROBE] ✓ Verification outcome: AGREEMENT (both say ${Z3_VERIFIED})"
  AGREEMENT=true
else
  echo "[PROBE] ⚠ Verification outcome: DISAGREEMENT (Z3=${Z3_VERIFIED}, LeanAide=${LEANAIDE_VERIFIED})"
  AGREEMENT=false
fi

# Check confidence alignment
CONFIDENCE_DIFF=$(echo "${Z3_CONFIDENCE} ${LEANAIDE_CONFIDENCE}" | awk '{print $1 - $2}' | tr -d '-')
CONFIDENCE_DIFF_THRESHOLD=0.3

if (( $(echo "${CONFIDENCE_DIFF} < ${CONFIDENCE_DIFF_THRESHOLD}" | bc -l) )); then
  echo "[PROBE] ✓ Confidence alignment: GOOD (diff=${CONFIDENCE_DIFF})"
else
  echo "[PROBE] ⚠ Confidence alignment: POOR (diff=${CONFIDENCE_DIFF})"
fi

# Test 4: Combined confidence calculation
echo "[PROBE] Test 4: Combined confidence calculation"

# Simple weighted average (equal weights)
COMBINED_CONFIDENCE=$(echo "${Z3_CONFIDENCE} ${LEANAIDE_CONFIDENCE}" | awk '{print ($1 + $2) / 2}')
echo "[PROBE] Combined confidence: ${COMBINED_CONFIDENCE}"

if (( $(echo "${COMBINED_CONFIDENCE} >= 0.8" | bc -l) )); then
  echo "[PROBE] ✓ High confidence verification achieved"
else
  echo "[PROBE] ⚠ Low confidence - may require manual review"
fi

# Test 5: Response time check
echo "[PROBE] Test 5: Response time validation"

# Extract execution times if present
if echo "${Z3_RESPONSE}" | jq -e '.metadata.executionTime' > /dev/null 2>&1; then
  Z3_TIME=$(echo "${Z3_RESPONSE}" | jq -r '.metadata.executionTime')
  echo "[PROBE] Z3 execution time: ${Z3_TIME}ms"

  if [ "${Z3_TIME}" -lt 10000 ]; then
    echo "[PROBE] ✓ Z3 response time acceptable"
  else
    echo "[PROBE] ⚠ Z3 response time high"
  fi
fi

if echo "${LEANAIDE_RESPONSE}" | jq -e '.metadata.executionTime' > /dev/null 2>&1; then
  LEANAIDE_TIME=$(echo "${LEANAIDE_RESPONSE}" | jq -r '.metadata.executionTime')
  echo "[PROBE] LeanAide execution time: ${LEANAIDE_TIME}ms"

  if [ "${LEANAIDE_TIME}" -lt 15000 ]; then
    echo "[PROBE] ✓ LeanAide response time acceptable"
  else
    echo "[PROBE] ⚠ LeanAide response time high"
  fi
fi

echo "[PROBE] ========================================"
echo "[PROBE] ✓ Cross-validation probe tests passed"
echo "[PROBE] Integration is ready"
echo "[PROBE] ========================================"

exit 0
