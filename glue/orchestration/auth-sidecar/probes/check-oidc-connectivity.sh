#!/bin/bash
# OIDC Connectivity Probe
#
# Following CLAUDE.md Federation Constitution - Law of Runtime Truth
# This probe validates that OIDC provider endpoints are accessible
# Must be run BEFORE deploying OAuth2-Proxy sidecar

set -euo pipefail

# Configuration from environment (Law of Configuration Explicitness)
OIDC_ISSUER="${OIDC_ISSUER:?FATAL: OIDC_ISSUER not set}"
OIDC_AUTHORIZATION_ENDPOINT="${OIDC_AUTHORIZATION_ENDPOINT:?FATAL: OIDC_AUTHORIZATION_ENDPOINT not set}"
OIDC_TOKEN_ENDPOINT="${OIDC_TOKEN_ENDPOINT:?FATAL: OIDC_TOKEN_ENDPOINT not set}"
OIDC_JWKS_URI="${OIDC_JWKS_URI:?FATAL: OIDC_JWKS_URI not set}"
OIDC_USERINFO_ENDPOINT="${OIDC_USERINFO_ENDPOINT:?FATAL: OIDC_USERINFO_ENDPOINT not set}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-10}"

# Function to log messages (structured logging)
log_info() {
    local correlation_id="probe-$(date +%s)-$RANDOM"
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"component\":\"oidc-connectivity-probe\",\"correlation_id\":\"${correlation_id}\"}"
}

log_error() {
    local correlation_id="probe-$(date +%s)-$RANDOM"
    echo "{\"level\":\"error\",\"msg\":\"$1\",\"component\":\"oidc-connectivity-probe\",\"correlation_id\":\"${correlation_id}\"}" >&2
}

# Function to check endpoint accessibility
check_endpoint() {
    local endpoint_name="$1"
    local endpoint_url="$2"
    local expected_status="${3:-200}"

    log_info "Checking ${endpoint_name}: ${endpoint_url}"

    if curl -sf -o /dev/null -w "%{http_code}" --max-time "${HEALTH_CHECK_TIMEOUT}" "${endpoint_url}" | grep -q "${expected_status}"; then
        log_info "✓ ${endpoint_name} is accessible"
        return 0
    else
        log_error "✗ ${endpoint_name} is NOT accessible or returned unexpected status"
        return 1
    fi
}

# Function to validate OIDC issuer URL
check_issuer() {
    log_info "Checking OIDC issuer: ${OIDC_ISSUER}"

    # Try to fetch .well-known/openid-configuration
    local well_known_url="${OIDC_ISSUER}/.well-known/openid-configuration"

    if curl -sf --max-time "${HEALTH_CHECK_TIMEOUT}" "${well_known_url}" > /dev/null 2>&1; then
        log_info "✓ OIDC issuer discovery endpoint is accessible"

        # Parse and validate endpoints
        local config
        config=$(curl -s --max-time "${HEALTH_CHECK_TIMEOUT}" "${well_known_url}")

        # Check required fields
        if echo "${config}" | jq -e '.issuer' > /dev/null 2>&1; then
            log_info "✓ Issuer claim present"
        else
            log_error "✗ Issuer claim missing from discovery document"
            return 1
        fi

        if echo "${config}" | jq -e '.authorization_endpoint' > /dev/null 2>&1; then
            log_info "✓ Authorization endpoint present"
        else
            log_error "✗ Authorization endpoint missing from discovery document"
            return 1
        fi

        if echo "${config}" | jq -e '.token_endpoint' > /dev/null 2>&1; then
            log_info "✓ Token endpoint present"
        else
            log_error "✗ Token endpoint missing from discovery document"
            return 1
        fi

        if echo "${config}" | jq -e '.jwks_uri' > /dev/null 2>&1; then
            log_info "✓ JWKS URI present"
        else
            log_error "✗ JWKS URI missing from discovery document"
            return 1
        fi

        return 0
    else
        log_error "✗ OIDC discovery endpoint NOT accessible"
        return 1
    fi
}

# Main execution
main() {
    log_info "Starting OIDC connectivity probe"

    # Check if required tools are available
    if ! command -v curl >/dev/null 2>&1; then
        log_error "FATAL: curl is required but not installed"
        exit 1
    fi

    if ! command -v jq >/dev/null 2>&1; then
        log_error "FATAL: jq is required but not installed"
        exit 1
    fi

    # Validate endpoints
    local failed=0

    # 1. Check OIDC issuer
    if ! check_issuer; then
        failed=1
    fi

    # 2. Check authorization endpoint (should be accessible without auth for discovery)
    if ! check_endpoint "Authorization endpoint" "${OIDC_AUTHORIZATION_ENDPOINT}" "200|302|405"; then
        # 405 Method Not Allowed is acceptable for GET on auth endpoint
        failed=1
    fi

    # 3. Check token endpoint (will fail without auth, but should be reachable)
    # We expect 401 Unauthorized or 400 Bad Request, which means endpoint exists
    if ! curl -sf --max-time "${HEALTH_CHECK_TIMEOUT}" "${OIDC_TOKEN_ENDPOINT}" > /dev/null 2>&1; then
        # Endpoint returned error, which is expected without credentials
        # Check if we get 400 or 401
        local status
        status=$(curl -s -o /dev/null -w "%{http_code}" --max-time "${HEALTH_CHECK_TIMEOUT}" "${OIDC_TOKEN_ENDPOINT}" 2>/dev/null || echo "000")
        if [[ "${status}" == "400" ]] || [[ "${status}" == "401" ]] || [[ "${status}" == "405" ]]; then
            log_info "✓ Token endpoint is reachable (status ${status})"
        else
            log_error "✗ Token endpoint is NOT reachable (status ${status})"
            failed=1
        fi
    else
        log_info "✓ Token endpoint is reachable"
    fi

    # 4. Check JWKS URI (should be publicly accessible)
    if ! check_endpoint "JWKS URI" "${OIDC_JWKS_URI}" "200"; then
        failed=1
    fi

    # 5. Check userinfo endpoint (will fail without auth, but should be reachable)
    local userinfo_status
    userinfo_status=$(curl -s -o /dev/null -w "%{http_code}" --max-time "${HEALTH_CHECK_TIMEOUT}" "${OIDC_USERINFO_ENDPOINT}" 2>/dev/null || echo "000")
    if [[ "${userinfo_status}" == "401" ]] || [[ "${userinfo_status}" == "400" ]] || [[ "${userinfo_status}" == "403" ]]; then
        log_info "✓ UserInfo endpoint is reachable (status ${userinfo_status})"
    else
        log_error "✗ UserInfo endpoint is NOT reachable (status ${userinfo_status})"
        failed=1
    fi

    # Final result
    if [[ ${failed} -eq 0 ]]; then
        log_info "✓ All OIDC connectivity checks PASSED"
        exit 0
    else
        log_error "✗ Some OIDC connectivity checks FAILED"
        exit 1
    fi
}

# Run main
main
