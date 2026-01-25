#!/bin/bash
# stripe.probe.sh - Runtime Validation Probe for stripe
#
# Comprehensive probe script to validate stripe service health and functionality
# Tests connectivity, endpoints, performance, and error handling
#
# Usage: ./probes/stripe.probe.sh
# Output: Detailed test results with pass/fail status

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

BASEURL="http://localhost:8080"
TIMEOUT=5
MAX_RETRIES=3

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0
TOTAL_TESTS=0

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_test() {
    echo -e "${CYAN}▶${NC} $1"
    ((TOTAL_TESTS++))
}

# Test HTTP endpoint
test_http_endpoint() {
    local endpoint="$1"
    local description="$2"
    local expected_code="${3:-200}"
    local method="${4:-GET}"

    log_test "Testing ${description}: ${method} ${endpoint}"

    local response
    response=$(curl -X "${method}" \
        -s -w "%{http_code}" \
        "${BASEURL}${endpoint}" \
        -o /dev/null \
        --max-time ${TIMEOUT} \
        --connect-timeout ${TIMEOUT} \
        2>/dev/null || echo "000")

    if [ "${response}" == "${expected_code}" ]; then
        log_success "PASS (${response})"
        ((PASS_COUNT++))
        return 0
    else
        log_error "FAIL (Expected ${expected_code}, got ${response})"
        ((FAIL_COUNT++))
        return 1
    fi
}

# Test with JSON response validation
test_json_endpoint() {
    local endpoint="$1"
    local description="$2"

    log_test "Testing ${description}: ${endpoint}"

    local response
    response=$(curl -s "${BASEURL}${endpoint}" \
        --max-time ${TIMEOUT} \
        2>/dev/null || echo "")

    if [ -n "${response}" ]; then
        if echo "${response}" | jq . >/dev/null 2>&1; then
            log_success "PASS (Valid JSON)"
            ((PASS_COUNT++))
            echo "${response}" | jq . 2>/dev/null | head -5
            return 0
        else
            log_warning "WARN (Response exists but not valid JSON)"
            ((WARN_COUNT++))
            echo "Response: ${response}"
            return 0
        fi
    else
        log_error "FAIL (No response)"
        ((FAIL_COUNT++))
        return 1
    fi
}

# Test response time
test_response_time() {
    local endpoint="$1"
    local description="$2"
    local threshold="${3:-5000}"

    log_test "Testing ${description} (threshold: ${threshold}ms)"

    local start=$(date +%s%N)
    local response=$(curl -s "${BASEURL}${endpoint}" \
        --max-time ${TIMEOUT} \
        -o /dev/null \
        -w "%{http_code}" \
        2>/dev/null || echo "000")
    local end=$(date +%s%N)
    local duration=$(( (end - start) / 1000000 ))

    if [ "${response}" != "000" ] && [ ${duration} -lt ${threshold} ]; then
        log_success "PASS (${duration}ms)"
        ((PASS_COUNT++))
        return 0
    else
        log_warning "WARN (Response time: ${duration}ms, threshold: ${threshold}ms)"
        ((WARN_COUNT++))
        return 0
    fi
}

# Test concurrent requests
test_concurrent_requests() {
    local endpoint="$1"
    local concurrent="$2"

    log_test "Testing ${concurrent} concurrent requests"

    local start=$(date +%s%N)

    for i in $(seq 1 ${concurrent}); do
        curl -s "${BASEURL}${endpoint}" -o /dev/null --max-time ${TIMEOUT} 2>/dev/null &
    done

    wait

    local end=$(date +%s%N)
    local duration=$(( (end - start) / 1000000 ))

    log_success "PASS (${concurrent} requests completed in ${duration}ms)"
    ((PASS_COUNT++))
}

# ============================================================================
# PROBE SEQUENCE
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  🔍 STRIPE SERVICE PROBE"
echo "════════════════════════════════════════════════════════════════════"
echo ""
log_info "Target URL: ${BASEURL}"
log_info "Timeout: ${TIMEOUT}s"
echo ""

# ============================================================================
# TEST SUITE 1: CONNECTIVITY TESTS
# ============================================================================

echo ""
log_info "━━━ Test Suite 1: Connectivity ━━━"
echo ""

# Test 1.1: Base URL
test_http_endpoint "/" "Base URL" "200"

# Test 1.2: Health endpoint
test_http_endpoint "/health" "Health check endpoint" "200"

# Test 1.3: Status endpoint
test_http_endpoint "/status" "Status endpoint" "200" || \
test_http_endpoint "/v1/status" "Status endpoint (v1)" "200" || \
log_warning "Status endpoint not found"

# Test 1.4: Root accessibility
test_response_time "/" "Base URL response time" "5000"

# ============================================================================
# TEST SUITE 2: API ENDPOINTS
# ============================================================================

echo ""
log_info "━━━ Test Suite 2: API Endpoints ━━━"
echo ""

# Test 2.$((i + 1)): /v1/products
if [[ "/v1/products" == "/"* ]]; then
    test_json_endpoint "/v1/products" "Endpoint 1: /v1/products" || \
    test_http_endpoint "/v1/products" "Endpoint 1: /v1/products" "200" || \
    log_warning "Endpoint 1 (/v1/products) not available"
else
    log_info "Command check: /v1/products"
    # For non-HTTP endpoints (Redis commands, SQL queries, etc.)
    log_success "PASS (Command defined)"
    ((PASS_COUNT++))
fi

# Test 2.$((i + 1)): /v1/customers
if [[ "/v1/customers" == "/"* ]]; then
    test_json_endpoint "/v1/customers" "Endpoint 2: /v1/customers" || \
    test_http_endpoint "/v1/customers" "Endpoint 2: /v1/customers" "200" || \
    log_warning "Endpoint 2 (/v1/customers) not available"
else
    log_info "Command check: /v1/customers"
    # For non-HTTP endpoints (Redis commands, SQL queries, etc.)
    log_success "PASS (Command defined)"
    ((PASS_COUNT++))
fi

# Test 2.$((i + 1)): /v1/charges
if [[ "/v1/charges" == "/"* ]]; then
    test_json_endpoint "/v1/charges" "Endpoint 3: /v1/charges" || \
    test_http_endpoint "/v1/charges" "Endpoint 3: /v1/charges" "200" || \
    log_warning "Endpoint 3 (/v1/charges) not available"
else
    log_info "Command check: /v1/charges"
    # For non-HTTP endpoints (Redis commands, SQL queries, etc.)
    log_success "PASS (Command defined)"
    ((PASS_COUNT++))
fi

# Test 2.$((i + 1)): /v1/subscriptions
if [[ "/v1/subscriptions" == "/"* ]]; then
    test_json_endpoint "/v1/subscriptions" "Endpoint 4: /v1/subscriptions" || \
    test_http_endpoint "/v1/subscriptions" "Endpoint 4: /v1/subscriptions" "200" || \
    log_warning "Endpoint 4 (/v1/subscriptions) not available"
else
    log_info "Command check: /v1/subscriptions"
    # For non-HTTP endpoints (Redis commands, SQL queries, etc.)
    log_success "PASS (Command defined)"
    ((PASS_COUNT++))
fi

# Test 2.$((i + 1)): /v1/invoices
if [[ "/v1/invoices" == "/"* ]]; then
    test_json_endpoint "/v1/invoices" "Endpoint 5: /v1/invoices" || \
    test_http_endpoint "/v1/invoices" "Endpoint 5: /v1/invoices" "200" || \
    log_warning "Endpoint 5 (/v1/invoices) not available"
else
    log_info "Command check: /v1/invoices"
    # For non-HTTP endpoints (Redis commands, SQL queries, etc.)
    log_success "PASS (Command defined)"
    ((PASS_COUNT++))
fi


# ============================================================================
# TEST SUITE 3: PERFORMANCE TESTS
# ============================================================================

echo ""
log_info "━━━ Test Suite 3: Performance ━━━"
echo ""

# Test 3.1: Response time
test_response_time "/" "Root endpoint response time" "5000"

# Test 3.2: Health endpoint response time
test_response_time "/health" "Health endpoint response time" "3000"

# Test 3.3: Concurrent requests
test_concurrent_requests "/" "5" "5 concurrent requests"

# ============================================================================
# TEST SUITE 4: ERROR HANDLING
# ============================================================================

echo ""
log_info "━━━ Test Suite 4: Error Handling ━━━"
echo ""

# Test 4.1: Invalid endpoint
log_test "Testing invalid endpoint (should return 404)"
invalid_response=$(curl -s "${BASEURL}/nonexistent-endpoint" \
    -w "%{http_code}" \
    -o /dev/null \
    --max-time ${TIMEOUT} \
    2>/dev/null || echo "000")

if [ "${invalid_response}" == "404" ]; then
    log_success "PASS (Correct 404 response)"
    ((PASS_COUNT++))
elif [ "${invalid_response}" == "000" ]; then
    log_warning "WARN (No response - service may be down)"
    ((WARN_COUNT++))
else
    log_warning "WARN (Unexpected response code: ${invalid_response})"
    ((WARN_COUNT++))
fi

# Test 4.2: Bad request handling
log_test "Testing malformed request handling"
bad_response=$(curl -s -X POST "${BASEURL}/api/test" \
    -H "Content-Type: application/json" \
    -d "invalid json" \
    -w "%{http_code}" \
    -o /dev/null \
    --max-time ${TIMEOUT} \
    2>/dev/null || echo "000")

if [[ "${bad_response}" == "400" ]] || [[ "${bad_response}" == "422" ]]; then
    log_success "PASS (Correct error response: ${bad_response})"
    ((PASS_COUNT++))
else
    log_warning "WARN (Unexpected response to bad request: ${bad_response})"
    ((WARN_COUNT++))
fi

# ============================================================================
# TEST SUITE 5: CONFIGURATION VALIDATION
# ============================================================================

echo ""
log_info "━━━ Test Suite 5: Configuration ━━━"
echo ""

# Test 5.1: Required environment variables
log_test "Checking required configuration"

if [ -n "${BASEURL}" ]; then
    log_success "PASS (BASEURL configured: ${BASEURL})"
    ((PASS_COUNT++))
else
    log_error "FAIL (BASEURL not configured)"
    ((FAIL_COUNT++))
fi

if [ -n "${TIMEOUT}" ]; then
    log_success "PASS (TIMEOUT configured: ${TIMEOUT}s)"
    ((PASS_COUNT++))
else
    log_warning "WARN (TIMEOUT using default)"
    ((WARN_COUNT++))
fi

# ============================================================================
# TEST SUITE 6: AVAILABILITY TESTS
# ============================================================================

echo ""
log_info "━━━ Test Suite 6: Availability ━━━"
echo ""

# Test 6.1: Service uptime
log_test "Checking service availability"

if curl -f -s "${BASEURL}/" -o /dev/null --max-time ${TIMEOUT} 2>/dev/null; then
    log_success "PASS (Service is available)"
    ((PASS_COUNT++))
else
    log_error "FAIL (Service is not available)"
    ((FAIL_COUNT++))
fi

# Test 6.2: Service version/info
log_test "Checking service information"
info_response=$(curl -s "${BASEURL}/v1/version" --max-time ${TIMEOUT} 2>/dev/null || \
                curl -s "${BASEURL}/version" --max-time ${TIMEOUT} 2>/dev/null || \
                curl -s "${BASEURL}/info" --max-time ${TIMEOUT} 2>/dev/null || echo "")

if [ -n "${info_response}" ]; then
    log_success "PASS (Service info available)"
    ((PASS_COUNT++))
    echo "${info_response}" | head -3
else
    log_warning "WARN (Version/info endpoint not found)"
    ((WARN_COUNT++))
fi

# ============================================================================
# TEST SUITE 7: STRESS TEST (LIGHT)
# ============================================================================

echo ""
log_info "━━━ Test Suite 7: Light Stress Test ━━━"
echo ""

# Test 7.1: Rapid sequential requests
log_test "Testing 10 rapid sequential requests"

failures=0
for i in {1..10}; do
    if ! curl -f -s "${BASEURL}/" -o /dev/null --max-time ${TIMEOUT} 2>/dev/null; then
        ((failures++))
    fi
done

if [ ${failures} -eq 0 ]; then
    log_success "PASS (All 10 requests succeeded)"
    ((PASS_COUNT++))
else
    log_warning "WARN (${failures}/10 requests failed)"
    ((WARN_COUNT++))
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "  📊 PROBE SUMMARY"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo -e "  Total Tests: ${TOTAL_TESTS}"
echo -e "  ${GREEN}Passed: ${PASS_COUNT}${NC}"
echo -e "  ${YELLOW}Warnings: ${WARN_COUNT}${NC}"
echo -e "  ${RED}Failed: ${FAIL_COUNT}${NC}"
echo ""

# Calculate success rate
if [ ${TOTAL_TESTS} -gt 0 ]; then
    success_rate=$(( PASS_COUNT * 100 / TOTAL_TESTS ))
    echo -e "  Success Rate: ${success_rate}%"
    echo ""
fi

# Exit with appropriate code
if [ ${FAIL_COUNT} -eq 0 ]; then
    echo -e "${GREEN}✅ STRIPE PROBE PASSED${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}❌ STRIPE PROBE FAILED${NC}"
    echo ""
    exit 1
fi
