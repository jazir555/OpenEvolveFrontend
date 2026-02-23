#!/bin/bash
# LoongFlow Deployment Validation Script
#
# Following CLAUDE.md principles:
# - Law of Runtime Truth: Verify actual deployment, not assumptions
# - Health checks validate real API availability
#
# Usage:
#   ./scripts/validate-loongflow-deployment.sh [environment]
#   ./scripts/validate-loongflow-deployment.sh local
#   ./scripts/validate-loongflow-deployment.sh kubernetes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ENVIRONMENT=${1:-local}
ADAPTER_PORT=${LOONGFLOW_PORT:-8040}
CORE_PORT=${LOONGFLOW_CORE_PORT:-8050}
TIMEOUT=${VALIDATION_TIMEOUT:-5}

# Counters
PASSED=0
FAILED=0
WARNINGS=0

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  LoongFlow Deployment Validation${NC}"
echo -e "${BLUE}  Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Function to print test result
print_result() {
    local test_name="$1"
    local status="$2"
    local message="${3:-}"

    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✅ PASS${NC} - $test_name"
        if [ -n "$message" ]; then
            echo -e "   ${GREEN}$message${NC}"
        fi
        ((PASSED++))
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}❌ FAIL${NC} - $test_name"
        if [ -n "$message" ]; then
            echo -e "   ${RED}$message${NC}"
        fi
        ((FAILED++))
    elif [ "$status" = "WARN" ]; then
        echo -e "${YELLOW}⚠️  WARN${NC} - $test_name"
        if [ -n "$message" ]; then
            echo -e "   ${YELLOW}$message${NC}"
        fi
        ((WARNINGS++))
    fi
}

# Function to check HTTP endpoint
check_http() {
    local url="$1"
    local expected_status="${2:-200}"
    local description="$3"

    local response
    local http_status

    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>&1) || {
        print_result "$description" "FAIL" "Cannot connect to $url"
        return 1
    }

    if [ "$response" = "$expected_status" ]; then
        print_result "$description" "PASS" "HTTP $response"
        return 0
    else
        print_result "$description" "FAIL" "Expected HTTP $expected_status, got HTTP $response"
        return 1
    fi
}

# Function to check JSON response
check_json() {
    local url="$1"
    local field="$2"
    local description="$3"

    local response
    response=$(curl -s --max-time "$TIMEOUT" "$url" 2>&1) || {
        print_result "$description" "FAIL" "Cannot fetch JSON from $url"
        return 1
    }

    if echo "$response" | jq -e ".${field}" > /dev/null 2>&1; then
        local value
        value=$(echo "$response" | jq -r ".${field}")
        print_result "$description" "PASS" "$field = $value"
        return 0
    else
        print_result "$description" "FAIL" "Field '$field' not found in response"
        return 1
    fi
}

# ============================================================================
# Validation Tests
# ============================================================================

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  1. Infrastructure Checks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Docker (local environment)
if [ "$ENVIRONMENT" = "local" ]; then
    if command -v docker &> /dev/null; then
        print_result "Docker installed" "PASS" "$(docker --version | head -1)"
    else
        print_result "Docker installed" "FAIL" "Docker not found in PATH"
    fi

    # Check Docker Compose
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_result "Docker Compose installed" "PASS"
    else
        print_result "Docker Compose installed" "FAIL"
    fi

    # Check federation network
    if docker network ls | grep -q "federation-network"; then
        print_result "federation-network exists" "PASS"
    else
        print_result "federation-network exists" "FAIL" "Create with: docker network create federation-network"
    fi
fi

# Check kubectl (kubernetes environment)
if [ "$ENVIRONMENT" = "kubernetes" ]; then
    if command -v kubectl &> /dev/null; then
        print_result "kubectl installed" "PASS" "$(kubectl version --client 2>&1 | head -1)"
    else
        print_result "kubectl installed" "FAIL"
    fi

    # Check cluster connection
    if kubectl cluster-info &> /dev/null; then
        print_result "Kubernetes cluster accessible" "PASS"
    else
        print_result "Kubernetes cluster accessible" "FAIL"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  2. Service Health Checks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check LoongFlow Core Health
check_http "http://localhost:${CORE_PORT}/health" "200" "LoongFlow Core health endpoint"

# Check Core health response
check_json "http://localhost:${CORE_PORT}/health" "status" "LoongFlow Core status field"

# Check LoongFlow Adapter Health
check_http "http://localhost:${ADAPTER_PORT}/health" "200" "LoongFlow Adapter health endpoint"

# Check Adapter health response
check_json "http://localhost:${ADAPTER_PORT}/health" "status" "LoongFlow Adapter status field"

# Check core connection status
check_json "http://localhost:${ADAPTER_PORT}/health" "core_connection" "Adapter to Core connection"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  3. API Endpoint Checks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check Core API endpoints
check_http "http://localhost:${CORE_PORT}/api/v1/workflows" "405" "Core workflows endpoint (405 Method Not Allowed is expected for GET)"

# Check Adapter API endpoints
check_http "http://localhost:${ADAPTER_PORT}/api/v1/workflows/execute" "405" "Adapter execute endpoint (405 expected for GET)"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  4. Container/Pod Checks${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$ENVIRONMENT" = "local" ]; then
    # Check if core container is running
    if docker ps | grep -q "loongflow-core"; then
        print_result "LoongFlow Core container running" "PASS"
    else
        print_result "LoongFlow Core container running" "FAIL"
    fi

    # Check if adapter container is running
    if docker ps | grep -q "loongflow-adapter"; then
        print_result "LoongFlow Adapter container running" "PASS"
    else
        print_result "LoongFlow Adapter container running" "FAIL"
    fi

    # Check container restarts
    CORE_RESTARTS=$(docker ps -a --filter "name=loongflow-core" --format "{{.RestartCount}}" 2>/dev/null || echo "0")
    if [ "$CORE_RESTARTS" -lt 3 ]; then
        print_result "Core container restart count" "PASS" "$CORE_RESTARTS restarts"
    else
        print_result "Core container restart count" "WARN" "$CORE_RESTARTS restarts (may indicate issues)"
    fi

    ADAPTER_RESTARTS=$(docker ps -a --filter "name=loongflow-adapter" --format "{{.RestartCount}}" 2>/dev/null || echo "0")
    if [ "$ADAPTER_RESTARTS" -lt 3 ]; then
        print_result "Adapter container restart count" "PASS" "$ADAPTER_RESTARTS restarts"
    else
        print_result "Adapter container restart count" "WARN" "$ADAPTER_RESTARTS restarts (may indicate issues)"
    fi
fi

if [ "$ENVIRONMENT" = "kubernetes" ]; then
    # Check namespace
    if kubectl get namespace loongflow-system &> /dev/null; then
        print_result "loongflow-system namespace exists" "PASS"
    else
        print_result "loongflow-system namespace exists" "FAIL"
    fi

    # Check core pods
    CORE_PODS=$(kubectl get pods -n loongflow-system -l app=loongflow-core --no-headers 2>/dev/null | wc -l)
    if [ "$CORE_PODS" -gt 0 ]; then
        READY_CORE=$(kubectl get pods -n loongflow-system -l app=loongflow-core --no-headers 2>/dev/null | grep -c "Running" || echo "0")
        print_result "LoongFlow Core pods" "PASS" "$READY_CORE/$CORE_PODS pods running"
    else
        print_result "LoongFlow Core pods" "FAIL" "No pods found"
    fi

    # Check adapter pods
    ADAPTER_PODS=$(kubectl get pods -n loongflow-system -l app=loongflow-adapter --no-headers 2>/dev/null | wc -l)
    if [ "$ADAPTER_PODS" -gt 0 ]; then
        READY_ADAPTER=$(kubectl get pods -n loongflow-system -l app=loongflow-adapter --no-headers 2>/dev/null | grep -c "Running" || echo "0")
        print_result "LoongFlow Adapter pods" "PASS" "$READY_ADAPTER/$ADAPTER_PODS pods running"
    else
        print_result "LoongFlow Adapter pods" "FAIL" "No pods found"
    fi

    # Check services
    if kubectl get svc -n loongflow-system loongflow-core-service &> /dev/null; then
        print_result "LoongFlow Core service exists" "PASS"
    else
        print_result "LoongFlow Core service exists" "FAIL"
    fi

    if kubectl get svc -n loongflow-system loongflow-adapter-service &> /dev/null; then
        print_result "LoongFlow Adapter service exists" "PASS"
    else
        print_result "LoongFlow Adapter service exists" "FAIL"
    fi
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  5. Functional Tests${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Test workflow execution (optional, requires valid API key)
read -p "Run functional workflow test? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing workflow execution..."

    # Create a simple test workflow
    TEST_RESPONSE=$(curl -s -X POST "http://localhost:${ADAPTER_PORT}/api/v1/workflows/execute" \
        -H "Content-Type: application/json" \
        -d '{
            "goal": "Test workflow",
            "context": {
                "test": true
            }
        }' 2>&1) || true

    if echo "$TEST_RESPONSE" | jq -e '.workflow_id' > /dev/null 2>&1; then
        WORKFLOW_ID=$(echo "$TEST_RESPONSE" | jq -r '.workflow_id')
        print_result "Workflow execution" "PASS" "Workflow ID: $WORKFLOW_ID"
    else
        print_result "Workflow execution" "WARN" "Test failed (may require valid API key)"
    fi
else
    print_result "Workflow execution" "WARN" "Skipped by user"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  6. Metrics and Monitoring${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check metrics endpoints
check_http "http://localhost:${CORE_PORT}/metrics" "200" "Core metrics endpoint"

check_http "http://localhost:${ADAPTER_PORT}/metrics" "200" "Adapter metrics endpoint"

# Check for key metrics
CORE_METRICS=$(curl -s "http://localhost:${CORE_PORT}/metrics" 2>&1) || true
if echo "$CORE_METRICS" | grep -q "loongflow_workflows_total"; then
    print_result "Core workflow metrics exposed" "PASS"
else
    print_result "Core workflow metrics exposed" "WARN" "Metrics may not be populated yet"
fi

ADAPTER_METRICS=$(curl -s "http://localhost:${ADAPTER_PORT}/metrics" 2>&1) || true
if echo "$ADAPTER_METRICS" | grep -q "http_requests_total"; then
    print_result "Adapter HTTP metrics exposed" "PASS"
else
    print_result "Adapter HTTP metrics exposed" "WARN" "Metrics may not be populated yet"
fi

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

TOTAL=$((PASSED + FAILED + WARNINGS))
echo -e "Total tests: ${BLUE}${TOTAL}${NC}"
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${YELLOW}Warnings: ${WARNINGS}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ Deployment validation PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ Deployment validation FAILED${NC}"
    echo ""
    echo "Please check the failed tests above and fix the issues."
    exit 1
fi
