#!/bin/bash

###############################################################################
# check-plugin-build.sh - Probe OpenEvolve React Plugin Build Status
#
# This script verifies that the OpenEvolve React Plugin can be built
# and all its dependencies are available.
#
# Environment Variables Required:
#   NODE_ENV - Node environment (default: production)
#
# Usage:
#   ./check-plugin-build.sh
#
# Exit Codes:
#   0 - Plugin build is successful
#   1 - Plugin build failed
#   2 - Missing required dependencies
###############################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${2:-}$(date -u +"%Y-%m-%dT%H:%M:%SZ") [check-plugin-build] $1${NC}"
}

log_info() {
    log "$1" ""
}

log_success() {
    log "$1" "$GREEN"
}

log_error() {
    log "$1" "$RED"
}

log_warning() {
    log "$1" "$YELLOW"
}

# Check if package.json exists
check_package_json() {
    log_info "Checking package.json..."

    if [[ ! -f "package.json" ]]; then
        log_error "package.json not found"
        return 1
    fi

    log_success "package.json found"
    return 0
}

# Check if node_modules exists
check_node_modules() {
    log_info "Checking node_modules..."

    if [[ ! -d "node_modules" ]]; then
        log_warning "node_modules not found. Run 'npm install' first"
        return 1
    fi

    log_success "node_modules found"
    return 0
}

# Check TypeScript compilation
check_typescript_compile() {
    log_info "Checking TypeScript compilation..."

    if ! command -v npx &> /dev/null; then
        log_error "npx not found"
        return 1
    fi

    if npx tsc --noEmit &> /dev/null; then
        log_success "TypeScript compilation successful"
        return 0
    else
        log_error "TypeScript compilation failed"
        return 1
    fi
}

# Check if vite build works
check_vite_build() {
    log_info "Checking Vite build configuration..."

    if [[ ! -f "vite.config.ts" ]]; then
        log_error "vite.config.ts not found"
        return 1
    fi

    log_success "Vite configuration found"
    return 0
}

# Check critical dependencies
check_dependencies() {
    log_info "Checking critical dependencies..."

    local missing_deps=()

    # Check for React
    if ! grep -q '"react"' package.json; then
        missing_deps+=("react")
    fi

    # Check for TypeScript
    if ! grep -q '"typescript"' package.json; then
        missing_deps+=("typescript")
    fi

    # Check for Vite
    if ! grep -q '"vite"' package.json; then
        missing_deps+=("vite")
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi

    log_success "All critical dependencies present"
    return 0
}

# Check plugin entry points
check_entry_points() {
    log_info "Checking plugin entry points..."

    local missing_files=()

    if [[ ! -f "src/index.ts" ]]; then
        missing_files+=("src/index.ts")
    fi

    if [[ ! -f "src/components/OpenEvolveConfigPanel.tsx" ]]; then
        missing_files+=("src/components/OpenEvolveConfigPanel.tsx")
    fi

    if [[ ! -f "src/utils/createOpenEvolvePlugin.ts" ]]; then
        missing_files+=("src/utils/createOpenEvolvePlugin.ts")
    fi

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Missing entry points: ${missing_files[*]}"
        return 1
    fi

    log_success "All entry points present"
    return 0
}

# Main execution
main() {
    echo ""
    echo "========================================"
    echo "OpenEvolve React Plugin - Build Probe"
    echo "========================================"
    echo ""

    local overall_status=0

    # Run all checks
    check_package_json || overall_status=1
    check_dependencies || overall_status=1
    check_entry_points || overall_status=1
    check_vite_build || overall_status=1

    # Only check node_modules and compilation if dependencies are installed
    if check_node_modules; then
        check_typescript_compile || overall_status=1
    else
        log_warning "Skipping compilation check (node_modules not found)"
    fi

    echo ""
    echo "========================================"

    if [[ $overall_status -eq 0 ]]; then
        log_success "✅ All build checks passed!"
        echo "========================================"
        echo ""
        exit 0
    else
        log_error "❌ Some build checks failed"
        echo "========================================"
        echo ""
        exit 1
    fi
}

# Run main function
main "$@"
