#!/bin/bash
# OAuth2-Proxy Health Check Probe
#
# Following CLAUDE.md Federation Constitution - Law of Runtime Truth
# This probe validates that OAuth2-Proxy is running and accessible

set -euo pipefail

# Configuration from environment
OAUTH2_PROXY_PORT="${OAUTH2_PROXY_PORT:-4180}"
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-5}"
HEALTH_CHECK_URL="http://localhost:${OAUTH2_PROXY_PORT}/ping"

# Function to log messages (structured logging)
log_info() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"component\":\"oauth2-proxy-health-check\"}"
}

log_error() {
    echo "{\"level\":\"error\",\"msg\":\"$1\",\"component\":\"oauth2-proxy-health-check\"}" >&2
}

# Health check with timeout
if curl -sf --max-time "${HEALTH_CHECK_TIMEOUT}" "${HEALTH_CHECK_URL}" > /dev/null 2>&1; then
    log_info "Health check passed"
    exit 0
else
    log_error "Health check failed: Cannot reach ${HEALTH_CHECK_URL}"
    exit 1
fi
