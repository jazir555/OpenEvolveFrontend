#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENVIRONMENT=$1
BASE_URL=$2

echo "💨 Running Smoke Tests for $ENVIRONMENT"
echo "📍 URL: $BASE_URL"
echo ""

# Array to track failed tests
failed_tests=0

# Test 1: API Health Endpoint
echo -n "Test 1: API Health Endpoint... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$STATUS" -eq 200 ]; then
    echo -e "${GREEN}✅ PASS${NC} (HTTP $STATUS)"
else
    echo -e "${RED}❌ FAIL${NC} (HTTP $STATUS)"
    ((failed_tests++))
fi

# Test 2: Authentication (should reject bad credentials)
echo -n "Test 2: Authentication (bad credentials)... "
RESPONSE=$(curl -s -X POST "$BASE_URL/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"wrongpassword"}')
if echo "$RESPONSE" | grep -q -E "(401|403|Unauthorized|Invalid credentials)"; then
    echo -e "${GREEN}✅ PASS${NC} (correctly rejects bad credentials)"
else
    echo -e "${RED}❌ FAIL${NC} (unexpected response: $RESPONSE)"
    ((failed_tests++))
fi

# Test 3: API Version
echo -n "Test 3: API Version Endpoint... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/version")
if [ "$STATUS" -eq 200 ]; then
    VERSION=$(curl -s "$BASE_URL/api/version")
    echo -e "${GREEN}✅ PASS${NC} (Version: $VERSION)"
else
    echo -e "${RED}❌ FAIL${NC} (HTTP $STATUS)"
    ((failed_tests++))
fi

# Test 4: Database Health
echo -n "Test 4: Database Connectivity... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/health/database")
if [ "$STATUS" -eq 200 ]; then
    echo -e "${GREEN}✅ PASS${NC} (database accessible)"
else
    echo -e "${RED}❌ FAIL${NC} (HTTP $STATUS)"
    ((failed_tests++))
fi

# Test 5: Redis Health
echo -n "Test 5: Redis Connectivity... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/health/redis")
if [ "$STATUS" -eq 200 ]; then
    echo -e "${GREEN}✅ PASS${NC} (Redis accessible)"
else
    echo -e "${RED}❌ FAIL${NC} (HTTP $STATUS)"
    ((failed_tests++))
fi

# Test 6: CORS Headers
echo -n "Test 6: CORS Configuration... "
CORS_HEADERS=$(curl -s -I -H "Origin: https://example.com" "$BASE_URL/health" | grep -i "access-control-allow-origin")
if [ -n "$CORS_HEADERS" ]; then
    echo -e "${GREEN}✅ PASS${NC} (CORS headers present)"
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (CORS headers not found)"
fi

# Test 7: Response Time
echo -n "Test 7: API Response Time... "
START_TIME=$(date +%s%N)
curl -s "$BASE_URL/health" > /dev/null
END_TIME=$(date +%s%N)
RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))  # Convert to milliseconds

if [ $RESPONSE_TIME -lt 500 ]; then
    echo -e "${GREEN}✅ PASS${NC} (${RESPONSE_TIME}ms < 500ms)"
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (${RESPONSE_TIME}ms >= 500ms)"
fi

# Test 8: WebSocket Handshake
echo -n "Test 8: WebSocket Handshake... "
if curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  "$BASE_URL/ws/monitoring" 2>&1 | grep -q "101 Switching Protocols"; then
    echo -e "${GREEN}✅ PASS${NC} (WebSocket handshake successful)"
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (WebSocket handshake verification inconclusive)"
fi

# Test 9: API Documentation
echo -n "Test 9: API Documentation... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
if [ "$STATUS" -eq 200 ] || [ "$STATUS" -eq 404 ]; then
    echo -e "${GREEN}✅ PASS${NC} (docs endpoint accessible or intentionally disabled)"
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (unexpected status: $STATUS)"
fi

# Test 10: Error Handling
echo -n "Test 10: Error Handling (404)... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/api/nonexistent")
if [ "$STATUS" -eq 404 ]; then
    echo -e "${GREEN}✅ PASS${NC} (returns 404 for non-existent routes)"
else
    echo -e "${RED}❌ FAIL${NC} (expected 404, got $STATUS)"
    ((failed_tests++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $failed_tests -eq 0 ]; then
    echo -e "${GREEN}✅ ALL SMOKE TESTS PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed_tests SMOKE TEST(S) FAILED${NC}"
    exit 1
fi
