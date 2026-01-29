#!/bin/bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 Running Pre-Deployment Checks..."
echo ""

# Array to track failed checks
failed_checks=0

# Check 1: Docker availability
echo -n "Check 1: Docker Installation... "
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (Docker not installed)"
    ((failed_checks++))
fi

# Check 2: Docker Compose availability
echo -n "Check 2: Docker Compose Installation... "
if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (Docker Compose not installed)"
    ((failed_checks++))
fi

# Check 3: Environment file
echo -n "Check 3: Environment Configuration... "
if [ -f .env.production ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (.env.production not found)"
    ((failed_checks++))
fi

# Check 4: Docker disk space
echo -n "Check 4: Docker Disk Space... "
if command -v docker &> /dev/null; then
    available_space=$(docker system df --format "{{.Size}}" | head -1 | sed 's/GB//g' | sed 's/MB//g' | awk '{print $1*1024}')
    if [ -n "$available_space" ] && [ "$available_space" -gt 5120 ]; then
        echo -e "${GREEN}✅ PASS${NC} (>5GB available)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} (Low disk space)"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Docker not available)"
fi

# Check 5: Git status (ensure working directory is clean)
echo -n "Check 5: Git Working Directory... "
if command -v git &> /dev/null; then
    if [ -z "$(git status --porcelain)" ]; then
        echo -e "${GREEN}✅ PASS${NC} (clean working directory)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} (uncommitted changes)"
    fi
else
    echo -e "${YELLOW}⚠️  SKIP${NC} (Git not available)"
fi

# Check 6: API Gateway Dockerfile
echo -n "Check 6: API Gateway Dockerfile... "
if [ -f "../../api/gateway/Dockerfile" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (Dockerfile not found)"
    ((failed_checks++))
fi

# Check 7: Frontend Dockerfile
echo -n "Check 7: Frontend Dockerfile... "
if [ -f "../../BubbleLab/apps/bubble-studio/Dockerfile" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (Dockerfile not found)"
    ((failed_checks++))
fi

# Check 8: Test suite
echo -n "Check 8: Test Suite Availability... "
if [ -d "../../api/gateway/tests" ]; then
    test_count=$(find ../../api/gateway/tests -name "test_*.py" | wc -l)
    echo -e "${GREEN}✅ PASS${NC} ($test_count tests found)"
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (tests directory not found)"
fi

# Check 9: SSL certificates (production only)
echo -n "Check 9: SSL Certificates... "
if [ -d "production/ssl" ]; then
    cert_count=$(find production/ssl -name "*.pem" | wc -l)
    if [ "$cert_count" -ge 2 ]; then
        echo -e "${GREEN}✅ PASS${NC} (SSL certificates present)"
    else
        echo -e "${YELLOW}⚠️  WARNING${NC} (SSL certificates incomplete)"
    fi
else
    echo -e "${YELLOW}⚠️  WARNING${NC} (SSL directory not found)"
fi

# Check 10: Backup availability
echo -n "Check 10: Backup Script... "
if [ -f "backup-production.sh" ]; then
    echo -e "${GREEN}✅ PASS${NC}"
else
    echo -e "${RED}❌ FAIL${NC} (backup script not found)"
    ((failed_checks++))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $failed_checks -eq 0 ]; then
    echo -e "${GREEN}✅ ALL PRE-DEPLOYMENT CHECKS PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ $failed_checks PRE-DEPLOYMENT CHECK(S) FAILED${NC}"
    exit 1
fi
