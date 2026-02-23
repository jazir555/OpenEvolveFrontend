#!/bin/bash
# Quick verification script for deployment configuration fixes
#
# This script validates all fixes from Task #13

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS_COUNT=0
FAIL_COUNT=0

check_pass() {
  echo -e "${GREEN}✅ $1${NC}"
  ((PASS_COUNT++))
}

check_fail() {
  echo -e "${RED}❌ $1${NC}"
  ((FAIL_COUNT++))
}

check_warn() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

echo "=========================================="
echo "Deployment Fixes Verification"
echo "=========================================="
echo ""

# 1. Check Docker Compose files exist
echo "Checking Docker Compose files..."
if [ -f "docker-compose.loongflow-core.yml" ]; then
  check_pass "docker-compose.loongflow-core.yml exists"
else
  check_fail "docker-compose.loongflow-core.yml missing"
fi

if [ -f "infra/docker-compose-all-adapters.yml" ]; then
  check_pass "docker-compose-all-adapters.yml exists"
else
  check_fail "docker-compose-all-adapters.yml missing"
fi

echo ""

# 2. Check Kubernetes manifests exist
echo "Checking Kubernetes manifests..."
if [ -f "infra/k8s-loongflow-core.yaml" ]; then
  check_pass "k8s-loongflow-core.yaml exists"
else
  check_fail "k8s-loongflow-core.yaml missing"
fi

if [ -f "infra/k8s-loongflow-deployment.yaml" ]; then
  check_pass "k8s-loongflow-deployment.yaml exists"
else
  check_fail "k8s-loongflow-deployment.yaml missing"
fi

echo ""

# 3. Check validation script exists and is executable
echo "Checking validation script..."
if [ -f "infra/scripts/validate-env.sh" ]; then
  check_pass "validate-env.sh exists"
  if [ -x "infra/scripts/validate-env.sh" ]; then
    check_pass "validate-env.sh is executable"
  else
    check_warn "validate-env.sh is not executable (run: chmod +x infra/scripts/validate-env.sh)"
  fi
else
  check_fail "validate-env.sh missing"
fi

echo ""

# 4. Check Docker Compose syntax
echo "Checking Docker Compose syntax..."
if command -v docker-compose &> /dev/null; then
  if docker-compose -f docker-compose.loongflow-core.yml config &> /dev/null; then
    check_pass "docker-compose.loongflow-core.yml syntax valid"
  else
    check_fail "docker-compose.loongflow-core.yml syntax error"
  fi

  if docker-compose -f infra/docker-compose-all-adapters.yml config &> /dev/null; then
    check_pass "docker-compose-all-adapters.yml syntax valid"
  else
    check_fail "docker-compose-all-adapters.yml syntax error"
  fi
else
  check_warn "docker-compose not found, skipping syntax check"
fi

echo ""

# 5. Check for hardcoded API keys in Kubernetes manifests
echo "Checking for hardcoded secrets..."
if grep -q "your-openai-api-key-here" infra/k8s-loongflow-core.yaml 2>/dev/null; then
  check_fail "Hardcoded API key found in k8s-loongflow-core.yaml"
else
  check_pass "No hardcoded API keys in k8s-loongflow-core.yaml"
fi

echo ""

# 6. Check for Redis service in core compose
echo "Checking Redis dependency..."
if grep -q "loongflow-redis:" docker-compose.loongflow-core.yml 2>/dev/null; then
  check_pass "Redis service present in docker-compose.loongflow-core.yml"
else
  check_fail "Redis service missing in docker-compose.loongflow-core.yml"
fi

echo ""

# 7. Check for LoongFlow core in all-adapters compose
echo "Checking LoongFlow core in all-adapters..."
if grep -q "loongflow-core:" infra/docker-compose-all-adapters.yml 2>/dev/null; then
  check_pass "LoongFlow core service present in docker-compose-all-adapters.yml"
else
  check_fail "LoongFlow core service missing in docker-compose-all-adapters.yml"
fi

echo ""

# 8. Check for restart policies
echo "Checking restart policies..."
if grep -q "restart: unless-stopped" docker-compose.loongflow-core.yml 2>/dev/null; then
  check_pass "Restart policy present in docker-compose.loongflow-core.yml"
else
  check_fail "Restart policy missing in docker-compose.loongflow-core.yml"
fi

echo ""

# 9. Check for health checks
echo "Checking health checks..."
if grep -q "healthcheck:" docker-compose.loongflow-core.yml 2>/dev/null; then
  check_pass "Health check present in docker-compose.loongflow-core.yml"
else
  check_fail "Health check missing in docker-compose.loongflow-core.yml"
fi

echo ""

# 10. Check documentation
echo "Checking documentation..."
if [ -f "infra/LOONGFLOW_DEPLOYMENT.md" ]; then
  check_pass "LOONGFLOW_DEPLOYMENT.md exists"
  if grep -q "validate-env.sh" infra/LOONGFLOW_DEPLOYMENT.md 2>/dev/null; then
    check_pass "Documentation references validate-env.sh"
  else
    check_warn "Documentation doesn't reference validate-env.sh"
  fi
  if grep -q "Secrets Management" infra/LOONGFLOW_DEPLOYMENT.md 2>/dev/null; then
    check_pass "Documentation includes secrets management warnings"
  else
    check_warn "Documentation missing secrets management warnings"
  fi
else
  check_fail "LOONGFLOW_DEPLOYMENT.md missing"
fi

echo ""

# 11. Check summary document
if [ -f "infra/DEPLOYMENT_FIXES_SUMMARY.md" ]; then
  check_pass "DEPLOYMENT_FIXES_SUMMARY.md exists"
else
  check_warn "DEPLOYMENT_FIXES_SUMMARY.md missing"
fi

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
  echo -e "${GREEN}✅ All critical checks passed!${NC}"
  echo "Deployment configurations are ready."
  exit 0
else
  echo -e "${RED}❌ Some checks failed. Please review the issues above.${NC}"
  exit 1
fi
