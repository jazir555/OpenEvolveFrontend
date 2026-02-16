#!/bin/bash
# Probe script for Gauntlet and Decomposition API endpoints
#
# Following CLAUDE.md Law of Runtime Truth:
# This script verifies that the API endpoints actually exist and respond correctly.
#
# Usage: ./check-gauntlet-decomposition-api.sh [base_url] [api_key]
#
# Example: ./check-gauntlet-decomposition-api.sh http://localhost:8000 test-key

set -e

# Configuration
BASE_URL="${1:-http://localhost:8000}"
API_KEY="${2:-test-api-key}"

echo "========================================="
echo "Gauntlet & Decomposition API Probe"
echo "========================================="
echo "Base URL: $BASE_URL"
echo "API Key: ${API_KEY:0:8}..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counter for results
PASSED=0
FAILED=0

# Function to check endpoint
check_endpoint() {
    local name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"

    echo -n "Checking $name... "

    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" \
            -X "$method" \
            "$BASE_URL$endpoint" \
            -H "X-API-Key: $API_KEY" \
            -H "Content-Type: application/json" \
            -d "$data" \
            2>&1 || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" \
            -X "$method" \
            "$BASE_URL$endpoint" \
            -H "X-API-Key: $API_KEY" \
            2>&1 || echo "000")
    fi

    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)

    case "$http_code" in
        200|201|202)
            echo -e "${GREEN}✓ PASS${NC} (HTTP $http_code)"
            ((PASSED++))
            return 0
            ;;
        404)
            echo -e "${YELLOW}⚠ SKIP${NC} (HTTP 404 - endpoint not found)"
            ((FAILED++))
            return 1
            ;;
        405)
            echo -e "${RED}✗ FAIL${NC} (HTTP 405 - method not allowed)"
            ((FAILED++))
            return 1
            ;;
        000)
            echo -e "${RED}✗ FAIL${NC} (Connection failed)"
            ((FAILED++))
            return 1
            ;;
        *)
            echo -e "${YELLOW}⚠ WARN${NC} (HTTP $http_code)"
            ((FAILED++))
            return 1
            ;;
    esac
}

echo "========================================="
echo "1. Gauntlet Execution Endpoints"
echo "========================================="

check_endpoint \
    "Execute Gauntlet (POST)" \
    "POST" \
    "/gauntlets/test-gauntlet/execute" \
    '{"content": "test content", "content_type": "text_general", "evolution_mode": "standard"}'

check_endpoint \
    "Get Gauntlet Execution Status (GET)" \
    "GET" \
    "/gauntlets/executions/test-exec/status"

check_endpoint \
    "List Gauntlet Executions (GET)" \
    "GET" \
    "/gauntlets/executions"

echo ""
echo "========================================="
echo "2. Decomposition Execution Endpoints"
echo "========================================="

check_endpoint \
    "Execute Decomposition (POST)" \
    "POST" \
    "/workflows/test-workflow/execute-decomposition" \
    '{"problem_statement": "test problem", "decomposition_method": "hierarchical"}'

check_endpoint \
    "Get Decomposition Execution Status (GET)" \
    "GET" \
    "/decomposition/executions/test-exec/status"

check_endpoint \
    "List Decomposition Executions (GET)" \
    "GET" \
    "/decomposition/executions"

echo ""
echo "========================================="
echo "3. Workflow Template Execution Endpoints"
echo "========================================="

check_endpoint \
    "Execute Workflow Template (POST)" \
    "POST" \
    "/workflow-templates/gauntlet-execution/execute" \
    '{"parameters": {"gauntlet_name": "test", "content_value": "test"}}'

check_endpoint \
    "Get Workflow Template Execution Status (GET)" \
    "GET" \
    "/workflow-templates/executions/test-exec/status"

check_endpoint \
    "Stop Workflow Template Execution (POST)" \
    "POST" \
    "/workflow-templates/executions/test-exec/stop"

echo ""
echo "========================================="
echo "4. Existing Gauntlet CRUD Endpoints"
echo "========================================="

check_endpoint \
    "List Gauntlets (GET)" \
    "GET" \
    "/gauntlets"

check_endpoint \
    "Get Gauntlet (GET)" \
    "GET" \
    "/gauntlets/test-gauntlet"

echo ""
echo "========================================="
echo "5. Existing Workflow Endpoints"
echo "========================================="

check_endpoint \
    "List Workflows (GET)" \
    "GET" \
    "/workflows"

check_endpoint \
    "Get Workflow Plan (GET)" \
    "GET" \
    "/workflows/test-workflow/decomposition-plan"

echo ""
echo "========================================="
echo "Probe Results Summary"
echo "========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo "Total: $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All endpoints passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some endpoints failed!${NC}"
    exit 1
fi
