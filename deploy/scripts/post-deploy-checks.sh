#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Running Post-Deployment Checks..."
echo ""

# Array to track failed checks
failed_checks=0

# Check 1: All containers running
echo -n "Check 1: Container Status... "
if command -v docker &> /dev/null; then
    running_containers=$(docker ps --filter "name=openevolve" --format "{{.Names}}" | wc -l)
    if [ "$running_containers" -ge 5 ]; then
        echo -e "${GREEN}✅ PASS${NC} ($running_containers containers running)"
    else
        echo -e "${RED}❌ FAIL${NC} (only $running_containers containers running)"
        ((failed_checks++))
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Docker not available)"
fi

# Check 2: Container health
echo -n "Check 2: Container Health... "
if command -v docker &> /dev/null; then
    unhealthy_containers=$(docker ps --filter "name=openevolve" --format "{{.Names}}\t{{.Status}}" | grep -v "healthy\|running" | wc -l)
    if [ "$unhealthy_containers" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} (all containers healthy)"
    else
        echo -e "${RED}❌ FAIL${NC} ($unhealthy_containers containers unhealthy)"
        ((failed_checks++))
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Docker not available)"
fi

# Check 3: API Gateway logs (no errors)
echo -n "Check 3: API Gateway Logs (errors)... "
if command -v docker &> /dev/null; then
    error_count=$(docker logs openevolve-api-prod 2>&1 | grep -i "error\|exception\|traceback" | wc -l)
    if [ "$error_count" -lt 10 ]; then
        echo -e "${GREEN}✅ PASS${NC} ($error_count errors found)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} ($error_count errors found)"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Docker not available)"
fi

# Check 4: Memory usage
echo -n "Check 4: Container Memory Usage... "
if command -v docker &> /dev/null; then
    high_memory_containers=$(docker stats --no-stream --format "{{.MemUsage}}" | awk '{split($0,a,"/"); if (a[2] ~ /GB/ && a[1] > 1.5) print}' | wc -l)
    if [ "$high_memory_containers" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} (memory usage acceptable)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} ($high_memory_containers containers with high memory)"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Docker not available)"
fi

# Check 5: CPU usage
echo -n "Check 5: Container CPU Usage... "
if command -v docker &> /dev/null; then
    high_cpu_containers=$(docker stats --no-stream --format "{{.CPUPerc}}" | awk '{if ($1+0 > 80) print}' | wc -l)
    if [ "$high_cpu_containers" -eq 0 ]; then
        echo -e "${GREEN}✅ PASS${NC} (CPU usage acceptable)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} ($high_cpu_containers containers with high CPU)"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Docker not available)"
fi

# Check 6: API response time
echo -n "Check 6: API Response Time... "
START_TIME=$(date +%s%N)
if curl -f -s https://openevolve.ai/health > /dev/null 2>&1; then
    END_TIME=$(date +%s%N)
    RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))
    if [ $RESPONSE_TIME -lt 1000 ]; then
        echo -e "${GREEN}✅ PASS${NC} (${RESPONSE_TIME}ms)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} (${RESPONSE_TIME}ms >= 1000ms)"
    fi
else
    echo -e "${RED}❌ FAIL${NC} (API not responding)"
    ((failed_checks++))
fi

# Check 7: Database connectivity
echo -n "Check 7: Database Connectivity... "
if curl -f -s https://openevolve.ai/api/health/database > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (database not accessible)"
    ((failed_checks++))
fi

# Check 8: Redis connectivity
echo -n "Check 8: Redis Connectivity... "
if curl -f -s https://openevolve.ai/api/health/redis > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (Redis not accessible)"
    ((failed_checks++))
fi

# Check 9: Frontend loading
echo -n "Check 9: Frontend Loading... "
STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://openevolve.ai/)
if [ "$STATUS" -eq 200 ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (HTTP $STATUS)"
    ((failed_checks++))
fi

# Check 10: WebSocket connectivity
echo -n "Check 10: WebSocket Connectivity... "
if curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: test" \
  https://openevolve.ai/ws/monitoring 2>&1 | grep -q "101 Switching Protocols"; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (WebSocket handshake verification inconclusive)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}✅ ALL POST-DEPLOYMENT CHECKS PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed_checks POST-DEPLOYMENT CHECK(S) FAILED${NC}"
    exit 1
fi
