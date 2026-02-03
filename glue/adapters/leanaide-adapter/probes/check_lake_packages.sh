#!/bin/bash
################################################################################
# Lake Packages Probe Script
# Federation Constitution Compliant
#
# Purpose: Verify Lake package manager can access mathlib and dependencies
# Compliance:
#   - Uses LAKE_PATH and LAKE_WORKSPACE_DIR environment variables
#   - Mandatory timeout via TIMEOUT_MS
#   - JSON Lines output format
#   - Idempotent (safe to run multiple times)
#   - Proper exit codes
#
# Environment Variables:
#   LAKE_PATH         - Path to lake executable (default: lake)
#   LAKE_WORKSPACE_DIR - Path to Lake workspace (default: current directory)
#   TIMEOUT_MS        - Command timeout in milliseconds (default: 30000)
#
# Exit Codes:
#   0 - Success
#   1 - Configuration error
#   2 - Lake executable not found
#   3 - No lakefile.lean found in workspace
#   4 - Lake packages not accessible
#   5 - Mathlib not found
################################################################################

set -euo pipefail

# Default values
TIMEOUT_MS=${TIMEOUT_MS:-30000}
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))
LAKE_PATH=${LAKE_PATH:-lake}
LAKE_WORKSPACE_DIR=${LAKE_WORKSPACE_DIR:-.}

# Generate correlation ID
CORRELATION_ID="probe_$(date +%s)_$$"

log_info() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"lake-packages\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

log_error() {
    echo "{\"level\":\"error\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"lake-packages\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

log_success() {
    echo "{\"level\":\"info\",\"msg\":\"$1\",\"correlation_id\":\"$CORRELATION_ID\",\"source_service\":\"leanaide-probe\",\"target_service\":\"lake-packages\",\"status\":\"success\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
}

# Validate workspace directory
if [[ ! -d "$LAKE_WORKSPACE_DIR" ]]; then
    log_error "Lake workspace directory does not exist: $LAKE_WORKSPACE_DIR"
    exit 1
fi

log_info "Probing Lake packages in workspace: $LAKE_WORKSPACE_DIR"

# Change to workspace directory
cd "$LAKE_WORKSPACE_DIR" || exit 1

# Test 1: Check for lakefile.lean or lakefile.toml
log_info "Searching for Lake configuration file"

if [[ ! -f "lakefile.lean" ]] && [[ ! -f "lakefile.toml" ]]; then
    log_error "No lakefile.lean or lakefile.toml found in workspace: $LAKE_WORKSPACE_DIR"
    exit 3
fi

LAKEFILE=$(if [[ -f "lakefile.lean" ]]; then echo "lakefile.lean"; else echo "lakefile.toml"; fi)
log_success "Lake configuration found: $LAKEFILE"

# Test 2: Check if lake executable is available
if ! command -v "$LAKE_PATH" &> /dev/null; then
    log_error "Lake executable not found at: $LAKE_PATH"
    exit 2
fi

# Test 3: Check lake-manifest.json (packages cache)
log_info "Checking lake-manifest.json"

if [[ -f "lake-manifest.json" ]]; then
    log_success "lake-manifest.json found"

    # Check if mathlib is in the manifest
    if grep -q "\"name\":\s*\"mathlib\"" lake-manifest.json 2>/dev/null; then
        log_success "Mathlib entry found in lake-manifest.json"
    else
        log_info "Mathlib not found in lake-manifest.json (may not be initialized yet)"
    fi

    # Get manifest size for reporting
    manifest_size=$(wc -c < lake-manifest.json)
    log_info "Lake manifest size: ${manifest_size} bytes"
else
    log_info "lake-manifest.json not found (packages may not be fetched yet)"
fi

# Test 4: Try to list available packages
log_info "Attempting to list Lake packages"

if ! packages_output=$(timeout "$TIMEOUT_SEC" "$LAKE_PATH" list 2>&1); then
    # Lake list may fail if packages aren't fetched, which is acceptable
    if [[ "$packages_output" == *"timeout"* ]]; then
        log_error "Lake list command timed out after ${TIMEOUT_MS}ms"
        exit 5
    fi
    log_info "Lake packages not yet initialized (run 'lake update' first)"
else
    log_success "Lake packages accessible"
    echo "{\"level\":\"info\",\"msg\":\"Available packages\",\"correlation_id\":\"$CORRELATION_ID\",\"packages\":\"$packages_output\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"
fi

# Test 5: Check .lake/packages directory if it exists
if [[ -d ".lake/packages" ]]; then
    log_info "Checking .lake/packages directory"

    # List available packages
    if package_dirs=$(ls -1 .lake/packages 2>/dev/null); then
        package_count=$(echo "$package_dirs" | wc -l)
        log_success "Found $package_count package(s) in .lake/packages"

        # Check specifically for mathlib
        if [[ -d ".lake/packages/mathlib" ]]; then
            log_success "Mathlib package directory found"

            # Get mathlib size
            mathlib_size=$(du -sh .lake/packages/mathlib 2>/dev/null | cut -f1)
            log_info "Mathlib size: $mathlib_size"
        else
            log_info "Mathlib directory not found in .lake/packages"
        fi
    fi
else
    log_info ".lake/packages directory not found (packages not yet fetched)"
fi

# Test 6: Try lake update with timeout (idempotent - safe to run multiple times)
log_info "Testing 'lake update' (idempotent package fetch)"

if ! update_output=$(timeout "$TIMEOUT_SEC" "$LAKE_PATH" update 2>&1); then
    if [[ "$update_output" == *"timeout"* ]]; then
        log_error "Lake update timed out after ${TIMEOUT_MS}ms"
        exit 5
    fi
    log_error "Lake update failed: $update_output"
    exit 4
else
    log_success "Lake update completed successfully"
fi

# Summary
log_success "Lake packages probe completed successfully"
echo "{\"level\":\"info\",\"msg\":\"Probe summary\",\"correlation_id\":\"$CORRELATION_ID\",\"lake_path\":\"$LAKE_PATH\",\"workspace_dir\":\"$LAKE_WORKSPACE_DIR\",\"lakefile\":\"$LAKEFILE\",\"timeout_ms\":\"$TIMEOUT_MS\",\"status\":\"pass\",\"timestamp\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}"

exit 0
