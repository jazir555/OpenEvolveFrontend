#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ENVIRONMENT=$1
BASE_URL=${2:-http://localhost:8000}

echo "🏥 Health Check for $ENVIRONMENT"
echo "📍 URL: $BASE_URL"
echo ""

# Array to track failed checks
failed_checks=0

# Check 1: API Gateway Health
echo -n "📡 Checking API Gateway... "
if curl -f -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ HEALTHY${NC}"

    # Get detailed health info
    response=$(curl -s "$BASE_URL/health")
    echo "   Response: $response"
else
    echo -e "${RED}❌ UNHEALTHY${NC}"
    ((failed_checks++))
fi

# Check 2: WebSocket
echo -n "🔌 Checking WebSocket... "
if curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  "$BASE_URL/ws/monitoring" 2>&1 | grep -q "101 Switching Protocols"; then
    echo -e "${GREEN}✅ HEALTHY${NC}"
else
    echo -e "${YELLOW}⚠️  CANNOT VERIFY (may require WebSocket client)${NC}"
fi

# Check 3: Database
echo -n "🗄️  Checking Database connectivity... "
if curl -f -s "$BASE_URL/api/health/database" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ HEALTHY${NC}"

    # Get database stats
    db_response=$(curl -s "$BASE_URL/api/health/database")
    echo "   Response: $db_response"
else
    echo -e "${RED}❌ UNHEALTHY${NC}"
    ((failed_checks++))
fi

# Check 4: Redis
echo -n "🔴 Checking Redis connectivity... "
if curl -f -s "$BASE_URL/api/health/redis" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ HEALTHY${NC}"

    # Get Redis stats
    redis_response=$(curl -s "$BASE_URL/api/health/redis")
    echo "   Response: $redis_response"
else
    echo -e "${RED}❌ UNHEALTHY${NC}"
    ((failed_checks++))
fi

# Check 5: API Version
echo -n "📦 Checking API version... "
if curl -f -s "$BASE_URL/api/version" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ AVAILABLE${NC}"

    # Get version info
    version=$(curl -s "$BASE_URL/api/version")
    echo "   Version: $version"
else
    echo -e "${YELLOW}⚠️  NOT AVAILABLE${NC}"
fi

# Check 6: Container Health
echo -n "🐳 Checking container health... "
if command -v docker &> /dev/null; then
    container_name="openevolve-api-${ENVIRONMENT}"
    if docker ps | grep -q "$container_name"; then
        echo -e "${GREEN}✅ RUNNING${NC}"

        # Get container stats
        echo "   Container: $container_name"
        docker ps --filter "name=$container_name" --format "   Status: {{.Status}}"
    else
        echo -e "${RED}❌ NOT RUNNING${NC}"
        ((failed_checks++))
    fi
else
    echo -e "${YELLOW}⚠️  Docker not available${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}✅ ALL HEALTH CHECKS PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed_checks HEALTH CHECK(S) FAILED${NC}"
    exit 1
fi
