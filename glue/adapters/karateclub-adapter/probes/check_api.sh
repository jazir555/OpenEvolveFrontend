#!/bin/bash
##############################################################################
# KarateClub API Probe Script
#
# Purpose: Verify KarateClub is installed and accessible
# Compliance: Law of Runtime Truth - verify before implementation
#
# Environment Variables Required:
#   PYTHON_PATH          - Python executable path (default: python3)
#   TIMEOUT_MS           - Timeout in milliseconds (default: 10000)
#
# Exit Codes:
#   0 - All probes passed
#   1 - Required environment variable missing
#   2 - KarateClub not installed
#   3 - KarateClub import failed
#   4 - Algorithm availability check failed
#   5 - Python not available
#
# Author: OpenEvolve Federation
# Created: 2026-02-03
##############################################################################

set -euo pipefail

# =============================================================================
# Configuration (from environment variables)
# =============================================================================

PYTHON_PATH="${PYTHON_PATH:-python3}"
TIMEOUT_MS="${TIMEOUT_MS:-10000}"
TIMEOUT_SEC=$((TIMEOUT_MS / 1000))

# =============================================================================
# Utility Functions
# =============================================================================

# Log JSON Lines output
log_json() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "{\"level\":\"$level\",\"msg\":\"$msg\",\"timestamp\":\"$timestamp\",\"probe\":\"check_api.sh\"}"
}

# Check if Python is available
check_python() {
    if ! command -v "$PYTHON_PATH" &> /dev/null; then
        log_json "error" "Python not found at: $PYTHON_PATH"
        exit 5
    fi
}

# Execute Python with timeout
execute_python() {
    local code="$1"

    timeout "$TIMEOUT_SEC" "$PYTHON_PATH" -c "$code" 2>&1 || true
}

# =============================================================================
# Probe Functions
# =============================================================================

# Probe 1: Check Python availability
probe_python() {
    log_json "info" "Checking Python availability: $PYTHON_PATH"

    local version
    version=$($PYTHON_PATH --version 2>&1)

    if [ $? -ne 0 ]; then
        log_json "error" "Python version check failed"
        return 1
    fi

    log_json "info" "Python version: $version"
    return 0
}

# Probe 2: Check KarateClub installation
probe_karateclub_installed() {
    log_json "info" "Checking KarateClub installation"

    local code="
import sys
try:
    import karateclub
    print('OK')
    print(f'Version: {karateclub.__version__}')
except ImportError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "KarateClub not installed: $output"
        return 1
    fi

    local version
    version=$(echo "$output" | grep "Version:" | cut -d' ' -f2)

    log_json "info" "KarateClub installed: $version"
    return 0
}

# Probe 3: Check core modules
probe_core_modules() {
    log_json "info" "Checking KarateClub core modules"

    local code="
import sys
import karateclub

modules_to_check = {
    'node_embedding': ['DeepWalk', 'Node2Vec'],
    'community_detection': ['LabelPropagation'],
    'graph_embedding': ['Graph2Vec'],
}

all_ok = True
for category, modules in modules_to_check.items():
    for module_name in modules:
        try:
            # Try importing
            exec(f'from karateclub import {module_name}')
            print(f'OK: {module_name}')
        except ImportError:
            try:
                # Try submodules
                if category == 'community_detection':
                    exec(f'from karateclub.community_detection.non_overlapping import {module_name}')
                elif category == 'node_embedding':
                    exec(f'from karateclub.node_embedding.neighbourhood import {module_name}')
                print(f'OK: {module_name} (submodule)')
            except ImportError:
                print(f'MISSING: {module_name}')
                all_ok = False

if all_ok:
    print('ALL_OK')
else:
    print('ERROR: Some modules missing')
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "ALL_OK"; then
        log_json "error" "Core modules check failed"
        echo "$output" | while read -r line; do
            log_json "info" "$line"
        done
        return 1
    fi

    echo "$output" | grep "OK:" | while read -r line; do
        log_json "info" "$line"
    done

    return 0
}

# Probe 4: Check NetworkX dependency
probe_networkx() {
    log_json "info" "Checking NetworkX dependency"

    local code="
import sys
try:
    import networkx as nx
    print('OK')
    print(f'NetworkX version: {nx.__version__}')
except ImportError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "NetworkX not installed: $output"
        return 1
    fi

    log_json "info" "NetworkX available"
    return 0
}

# Probe 5: Check NumPy dependency
probe_numpy() {
    log_json "info" "Checking NumPy dependency"

    local code="
import sys
try:
    import numpy as np
    print('OK')
    print(f'NumPy version: {np.__version__}')
except ImportError as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"

    local output
    output=$(execute_python "$code")

    if ! echo "$output" | grep -q "OK"; then
        log_json "error" "NumPy not installed: $output"
        return 1
    fi

    log_json "info" "NumPy available"
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    log_json "info" "Starting KarateClub API probe"
    log_json "info" "Python path: $PYTHON_PATH"
    log_json "info" "Timeout: ${TIMEOUT_MS}ms"

    # Check prerequisites
    check_python

    # Run probes sequentially (fail fast on first error)
    if ! probe_python; then
        log_json "error" "Python check failed"
        exit 2
    fi

    if ! probe_karateclub_installed; then
        log_json "error" "KarateClub installation check failed"
        exit 3
    fi

    if ! probe_networkx; then
        log_json "warn" "NetworkX check failed (optional)"
    fi

    if ! probe_numpy; then
        log_json "warn" "NumPy check failed (optional)"
    fi

    if ! probe_core_modules; then
        log_json "error" "Core modules check failed"
        exit 4
    fi

    # All probes passed
    log_json "info" "All KarateClub API probes passed successfully"
    exit 0
}

# Run main function
main "$@"
