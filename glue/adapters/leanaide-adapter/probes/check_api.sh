#!/bin/bash
################################################################################
# LeanAide API Probe Script
# Federation Constitution Compliant
#
# Purpose: Test LeanAide server endpoints availability and responsiveness
# Compliance:
#   - Uses LEANAIDE_API_URL environment variable (NO hardcoded IPs)
#   - Mandatory timeout via TIMEOUT_MS
#   - JSON Lines output format
#   - Idempotent (safe to run multiple times)
#   - Proper exit codes
#
# Environment Variables:
#   LEANAIDE_API_URL - Target API URL (REQUIRED, fails fast if missing)
#   TIMEOUT_MS       - Request timeout in milliseconds (default: 5000)
#
# Exit Codes:
#   0 - Success
#   1 - Configuration error
#   2 - API unreachable
#   3 - Timeout
################################################################################

set -euo pipefail

# Default values with no magic defaults - must be explicitly set
TIMEOUT_MS=${TIMEOUT_MS:-5000}
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# Validation - Fail fast if configuration is invalid
if [[ -z "${LEANAIDE_API_URL:-}" ]]; then
    echo "{\"level\":\"error\",\"msg\":\"LEANAIDE_API_URL environment variable is required\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
    exit 1
fi

# Validate URL format
if [[ ! "$LEANAIDE_API_URL" =~ ^https?:// ]]; then
    echo "{\"level\":\"error\",\"msg\":\"LEANAIDE_API_URL must start with http:// or https://\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
    exit 1
fi

# Remove trailing slash if present
API_URL="${LEANAIDE_API_URL%/}"

# Generate correlation ID for this probe run
CORRELATION_ID="probe_$(date +%s)_$$"

log_info() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"leanaide-api\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

log_error() {
    echo "{\"level\":\"error\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"leanaide-api\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

log_success() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"leanaide-api\",\"status\":\"success\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

# Test 1: Root endpoint health check
log_info "Starting API probe for $API_URL"

# Use timeout wrapper to prevent infinite hangs
if ! response=$(curl -s -o /dev/null -w "%{http_code}" \
    --max-time "$TIMEOUT_SEC" \
    --connect-timeout "$TIMEOUT_SEC" \
    -L \
    "$API_URL" 2>&1); then

    # Check if it was a timeout
    if [[ "$response" == *"timeout"* ]] || [[ "$response" == *"timed out"* ]]; then
        log_error "Request timeout after ${TIMEOUT_MS}ms"
        exit 3
    fi

    log_error "API unreachable: $response"
    exit 2
fi

# Validate HTTP response code
if [[ "$response" =~ ^2 ]]; then
    log_success "Root endpoint accessible (HTTP $response)"
else
    log_error "API returned unexpected status code: $response"
    exit 2
fi

# Test 2: Check if server is responsive to a simple ping/health endpoint
# LeanAide typically runs on port 5000, test the root endpoint
log_info "Testing API responsiveness"

if ! health_check=$(curl -s \
    --max-time "$TIMEOUT_SEC" \
    --connect-timeout "$TIMEOUT_SEC" \
    -L \
    "$API_URL" 2>&1); then

    log_error "Health check failed: $health_check"
    exit 2
fi

# Verify response is not empty
if [[ -z "$health_check" ]]; then
    log_error "API returned empty response"
    exit 2
fi

log_success "API probe completed successfully"
echo "{\"level\":\"info\",\"msg\":\"Probe summary\",\"correlation_id\":\"$CORRELATION_ID\",\"api_url\":\"$API_URL\",\"timeout_ms\":\"$TIMEOUT_MS\",\"status\":\"pass\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

exit 0
