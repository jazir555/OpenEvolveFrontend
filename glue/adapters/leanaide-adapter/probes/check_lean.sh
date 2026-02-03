#!/bin/bash
################################################################################
# Lean 4 Compiler Probe Script
# Federation Constitution Compliant
#
# Purpose: Verify Lean 4 compiler installation and Lake package manager
# Compliance:
#   - Uses LEAN_PATH and LAKE_PATH environment variables
#   - Mandatory timeout via TIMEOUT_MS
#   - JSON Lines output format
#   - Idempotent (safe to run multiple times)
#   - Proper exit codes
#
# Environment Variables:
#   LEAN_PATH - Path to lean executable (default: lean)
#   LAKE_PATH - Path to lake executable (default: lake)
#   TIMEOUT_MS - Command timeout in milliseconds (default: 10000)
#
# Exit Codes:
#   0 - Success
#   1 - Configuration error
#   2 - Lean compiler not found or invalid
#   3 - Lake not found or invalid
#   4 - Version check failed
################################################################################

set -euo pipefail

# Default values
TIMEOUT_MS=${TIMEOUT_MS:-10000}
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))
LEAN_PATH=${LEAN_PATH:-lean}
LAKE_PATH=${LAKE_PATH:-lake}

# Generate correlation ID
CORRELATION_ID="probe_$(date +%s)_$$"

log_info() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"lean-compiler\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

log_error() {
    echo "{\"level\":\"error\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"lean-compiler\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

log_success() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"lean-compiler\",\"status\":\"success\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

# Test 1: Check if lean executable exists
log_info "Verifying Lean 4 compiler at: $LEAN_PATH"

if ! command -v "$LEAN_PATH" &> /dev/null; then
    log_error "Lean compiler not found at: $LEAN_PATH"
    exit 2
fi

log_success "Lean compiler found"

# Test 2: Get Lean version with timeout
log_info "Checking Lean 4 version"

if ! lean_version=$(timeout "$TIMEOUT_SEC" "$LEAN_PATH" --version 2>&1); then
    log_error "Lean version check failed or timed out after ${TIMEOUT_MS}ms"
    exit 4
fi

# Verify it's Lean 4 (not Lean 3)
if [[ ! "$lean_version" =~ "Lean 4" ]] && [[ ! "$lean_version" =~ "version 4" ]]; then
    log_error "Invalid Lean version detected (expected Lean 4): $lean_version"
    exit 4
fi

log_success "Lean 4 compiler verified: $lean_version"

# Test 3: Check if lake executable exists
log_info "Verifying Lake package manager at: $LAKE_PATH"

if ! command -v "$LAKE_PATH" &> /dev/null; then
    log_error "Lake package manager not found at: $LAKE_PATH"
    exit 3
fi

log_success "Lake package manager found"

# Test 4: Get Lake version with timeout
log_info "Checking Lake version"

if ! lake_version=$(timeout "$TIMEOUT_SEC" "$LAKE_PATH" --version 2>&1); then
    log_error "Lake version check failed or timed out after ${TIMEOUT_MS}ms"
    exit 4
fi

# Extract version number for logging
clean_lake_version=$(echo "$lake_version" | head -n 1)
log_success "Lake package manager verified: $clean_lake_version"

# Test 5: Verify Lake can access Lakefile (idempotent check)
# This checks if lake can list available packages
log_info "Testing Lake functionality"

if ! lake_list=$(timeout "$TIMEOUT_SEC" "$LAKE_PATH" list 2>&1); then
    # Lake list may fail if not in a project directory, which is acceptable
    # We only care that lake can execute
    log_info "Lake executable functional (no project context required)"
else
    log_success "Lake package manager functional"
fi

# Summary
log_success "Lean 4 and Lake probe completed successfully"
echo "{\"level\":\"info\",\"msg\":\"Probe summary\",\"correlation_id\":\"$CORRELATION_ID\",\"lean_path\":\"$LEAN_PATH\",\"lake_path\":\"$LAKE_PATH\",\"lean_version\":\"$lean_version\",\"lake_version\":\"$clean_lake_version\",\"timeout_ms\":\"$TIMEOUT_MS\",\"status\":\"pass\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

exit 0
